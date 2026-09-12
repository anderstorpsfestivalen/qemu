/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Native offscreen OpenGL contexts and GPU-only presentation export.
 */
#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/bswap.h"
#include "standard-headers/dreamgpu/gl-funcs.h"
#include "standard-headers/dreamgpu/gl.h"
#include "standard-headers/dreamgpu/transport.h"
#include "dreamgpu-gl-platform.h"

#ifdef CONFIG_DARWIN
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <OpenGL/CGLIOSurface.h>
#include <IOSurface/IOSurface.h>
#define dgGenFramebuffers glGenFramebuffersEXT
#define dgDeleteFramebuffers glDeleteFramebuffersEXT
#define dgBindFramebuffer glBindFramebufferEXT
#define dgFramebufferTexture2D glFramebufferTexture2DEXT
#define dgCheckFramebufferStatus glCheckFramebufferStatusEXT
#define dgBlitFramebuffer glBlitFramebufferEXT
#define dgGenRenderbuffers glGenRenderbuffersEXT
#define dgDeleteRenderbuffers glDeleteRenderbuffersEXT
#define dgBindRenderbuffer glBindRenderbufferEXT
#define dgRenderbufferStorage glRenderbufferStorageEXT
#define dgFramebufferRenderbuffer glFramebufferRenderbufferEXT
#else
#include <epoxy/gl.h>
#include <epoxy/egl.h>
#include <gbm.h>
#include <drm_fourcc.h>
#ifndef EGL_DRM_RENDER_NODE_FILE_EXT
#define EGL_DRM_RENDER_NODE_FILE_EXT 0x3377
#endif
#define dgGenFramebuffers glGenFramebuffers
#define dgDeleteFramebuffers glDeleteFramebuffers
#define dgBindFramebuffer glBindFramebuffer
#define dgFramebufferTexture2D glFramebufferTexture2D
#define dgCheckFramebufferStatus glCheckFramebufferStatus
#define dgBlitFramebuffer glBlitFramebuffer
#define dgGenRenderbuffers glGenRenderbuffers
#define dgDeleteRenderbuffers glDeleteRenderbuffers
#define dgBindRenderbuffer glBindRenderbuffer
#define dgRenderbufferStorage glRenderbufferStorage
#define dgFramebufferRenderbuffer glFramebufferRenderbuffer
#endif

#include "dreamgpu-host.h"
#include "gl-api.h"

typedef DreamGpuTexture DgTexture;

typedef DreamGpuTextureNamespace DgTextureNamespace;

struct DgGLPlatform {
    DreamGpuGlApi gl_api;
    uint64_t next_context_serial;
    uint64_t texture_bytes;
    uint32_t texture_count;
    /* One bounded CPU cache for genuine texture reads, never presentation. */
    DgTexture *read_texture;
    uint64_t read_version;
    uint32_t read_level, read_bytes;
    uint8_t *read_pixels;
#ifdef CONFIG_DARWIN
    CGLPixelFormatObj format;
    CGLContextObj root;
#else
    EGLDisplay display;
    EGLConfig config;
    EGLContext root;
    PFNEGLDESTROYIMAGEKHRPROC destroy_image;
    struct gbm_device *gbm;
    int render_fd;
#endif
};

#define DG_ATTRIB_STACK 16
typedef struct {
    GLbitfield mask;
    GLboolean color_sum;
    GLenum draw_buffer, read_buffer;
    DgTexture *texture, *texture_1d;
} DgAttrib;

struct DgGLContext {
    DgGLPlatform *platform;
    uint64_t serial;
#ifdef CONFIG_DARWIN
    CGLContextObj render, completion;
#else
    EGLContext render;
#endif
    gatomicrefcount refs;
    GLuint framebuffer;
    bool initialized;
    uint32_t in_begin;
    DgGLDrawable *drawable;
    DgTextureNamespace *textures;
    DgTexture *default_texture, *bound_texture;
    DgTexture *default_texture_1d, *bound_texture_1d;
    uint32_t guest_errors;
    GLenum draw_buffer, read_buffer;
    unsigned attrib_depth;
    DgAttrib attrib[DG_ATTRIB_STACK];
};

struct DgGLDrawable {
    uint32_t width, height;
    GLuint color, front, depth;
    GLsync last_write;
};

struct DgGLImage {
    uint32_t width, height, stride, offset;
    uint64_t modifier;
#ifdef CONFIG_DARWIN
    IOSurfaceRef surface;
    GLsync fence;
    DgGLContext *context;
#else
    struct gbm_bo *bo;
    EGLImageKHR image;
    int fd, fence_fd;
#endif
};

static void texture_unref(DgGLPlatform *p, DgTexture *texture);
static void texture_namespace_unref(DgGLPlatform *p, DgTextureNamespace *ns);
/* Pure command vocabulary and validation are owned by the Rust core. */
static unsigned texture_params(uint32_t target, uint32_t pname)
{ return dreamgpu_gl_texture_params(target, pname); }
static unsigned texture_env_params(uint32_t target, uint32_t pname)
{ return dreamgpu_gl_texture_env_params(target, pname); }
static unsigned vector_bytes(uint32_t fn, const uint32_t *args)
{ return dreamgpu_gl_vector_bytes(fn, args); }
static bool vector_function(uint32_t fn)
{ return dreamgpu_gl_vector_function(fn); }
static unsigned index_bytes(uint32_t type)
{ return dreamgpu_gl_index_bytes(type); }
static bool buffer_selection(uint32_t mode, bool draw)
{ return dreamgpu_gl_buffer_selection(mode, draw); }
static const char *query_string(uint32_t name)
{ return dreamgpu_gl_query_string(name); }
static unsigned query_shape(uint32_t fn, const uint8_t *args, uint32_t *type)
{ return dreamgpu_gl_query_shape(fn, args, type); }
uint32_t dg_gl_call_validate(uint32_t fn, const uint8_t *args)
{ return dreamgpu_gl_call_validate(fn, args); }
uint32_t dg_gl_data_validate(uint32_t fn, const uint8_t *args,
                              const uint8_t *data, uint32_t bytes)
{ return dreamgpu_gl_data_validate(fn, args, data, bytes); }
uint32_t dg_gl_query_validate(uint32_t fn, const uint8_t *args)
{ return dreamgpu_gl_query_validate(fn, args); }
uint32_t dg_gl_query_result_bytes(uint32_t fn, const uint8_t *args)
{ return dreamgpu_gl_query_result_bytes(fn, args); }
uint32_t dg_gl_function_words(uint32_t fn)
{ return dreamgpu_gl_function_words(fn); }

static GLfloat gl_arg_float(const uint8_t *p);
static GLdouble gl_arg_double(const uint8_t *p);

static void vector_call(uint32_t fn, const uint32_t *a, const uint8_t *data)
{
    GLfloat f[4];
    GLdouble d[4];
    GLint integers[4];
    bool wide = fn == FEnum_glTexGendv || fn == FEnum_glClipPlane;
    unsigned count = vector_bytes(fn, a) / (wide ? 8 : 4);

    for (unsigned i = 0; i < count; i++) {
        integers[i] = ldl_le_p(data + i * 4);
        if (wide) {
            d[i] = gl_arg_double(data + i * 8);
        } else {
            f[i] = gl_arg_float(data + i * 4);
        }
    }
    switch (fn) {
    case FEnum_glTexParameterfv:
        glTexParameterfv(a[0], a[1], f);
        break;
    case FEnum_glTexParameteriv:
        glTexParameteriv(a[0], a[1], integers);
        break;
    case FEnum_glTexEnvfv:
        glTexEnvfv(a[0], a[1], f);
        break;
    case FEnum_glTexEnviv:
        glTexEnviv(a[0], a[1], integers);
        break;
    case FEnum_glLightfv:
        glLightfv(a[0], a[1], f);
        break;
    case FEnum_glMaterialfv:
        glMaterialfv(a[0], a[1], f);
        break;
    case FEnum_glFogfv:
        glFogfv(a[0], f);
        break;
    case FEnum_glLightModelfv:
        glLightModelfv(a[0], f);
        break;
    case FEnum_glTexGenfv:
        glTexGenfv(a[0], a[1], f);
        break;
    case FEnum_glTexGendv:
        glTexGendv(a[0], a[1], d);
        break;
    case FEnum_glClipPlane:
        glClipPlane(a[0], d);
        break;
    }
}

DgGLPlatform *dg_gl_platform_new(const char *render_node, Error **errp)
{
    DgGLPlatform *p = g_new0(DgGLPlatform, 1);
#ifdef CONFIG_DARWIN
    CGLPixelFormatAttribute attrs[] = {
        kCGLPFAAccelerated, kCGLPFAColorSize, 24, kCGLPFAAlphaSize, 8,
        kCGLPFADepthSize, 24, kCGLPFAStencilSize, 8, 0,
    };
    GLint count;
    CGLError err = CGLChoosePixelFormat(attrs, &p->format, &count);

    if (err != kCGLNoError || !p->format) {
        error_setg(errp, "CGL pixel format: %s", CGLErrorString(err));
        g_free(p);
        return NULL;
    }
    err = CGLCreateContext(p->format, NULL, &p->root);
    if (err != kCGLNoError) {
        error_setg(errp, "CGL internal share group: %s", CGLErrorString(err));
        dg_gl_platform_free(p);
        return NULL;
    }
#else
    EGLint count;
    const EGLint attrs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8, EGL_NONE,
    };

    p->render_fd = open(render_node, O_RDWR | O_CLOEXEC);
    if (p->render_fd >= 0) {
        p->gbm = gbm_create_device(p->render_fd);
    }
    if (p->render_fd < 0 || !p->gbm) {
        error_setg_errno(errp, errno, "Opening GPU render node %s",
                         render_node);
        dg_gl_platform_free(p);
        return NULL;
    }
    EGLDeviceEXT devices[32];
    EGLint device_count = 0;

    if (!eglQueryDevicesEXT(G_N_ELEMENTS(devices), devices, &device_count)) {
        error_setg(errp, "EGL device enumeration failed");
        dg_gl_platform_free(p);
        return NULL;
    }
    for (int i = 0; i < device_count; i++) {
        const char *node =
            eglQueryDeviceStringEXT(devices[i], EGL_DRM_RENDER_NODE_FILE_EXT);
        if (node && !strcmp(node, render_node)) {
            p->display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                                   devices[i], NULL);
            break;
        }
    }
    if (p->display == EGL_NO_DISPLAY ||
        !eglInitialize(p->display, NULL, NULL) ||
        !eglBindAPI(EGL_OPENGL_API) ||
        !eglChooseConfig(p->display, attrs, &p->config, 1, &count) || !count) {
        error_setg(errp, "EGL offscreen initialization failed: 0x%x",
                    eglGetError());
        dg_gl_platform_free(p);
        return NULL;
    }
    if (!epoxy_has_egl_extension(p->display, "EGL_EXT_image_dma_buf_import") ||
        !epoxy_has_egl_extension(p->display, "EGL_ANDROID_native_fence_sync")) {
        error_setg(errp, "EGL DMA-BUF import and native fences are required");
        dg_gl_platform_free(p);
        return NULL;
    }
    /*
     * Image lifetime is display-scoped and can outlive all guest contexts.
     * libepoxy's first-call resolver uses the current EGL display, which is
     * absent when a client destroys its context before its exported slots.
     */
    p->destroy_image = (PFNEGLDESTROYIMAGEKHRPROC)
        eglGetProcAddress("eglDestroyImageKHR");
    if (!p->destroy_image) {
        error_setg(errp, "EGL image destruction is required");
        dg_gl_platform_free(p);
        return NULL;
    }
    p->root = eglCreateContext(p->display, p->config, EGL_NO_CONTEXT, NULL);
    if (p->root == EGL_NO_CONTEXT) {
        error_setg(errp, "EGL internal share group: 0x%x", eglGetError());
        dg_gl_platform_free(p);
        return NULL;
    }
#endif
    p->gl_api = (DreamGpuGlApi) { DREAMGPU_GL_API_INIT };
    return p;
}

void dg_gl_platform_free(DgGLPlatform *p)
{
    if (!p) {
        return;
    }
    g_free(p->read_pixels);
#ifdef CONFIG_DARWIN
    if (p->root) {
        CGLDestroyContext(p->root);
    }
    if (p->format) {
        CGLDestroyPixelFormat(p->format);
    }
#else
    if (p->root != EGL_NO_CONTEXT) {
        eglDestroyContext(p->display, p->root);
    }
    if (p->display != EGL_NO_DISPLAY) {
        eglTerminate(p->display);
    }
    if (p->gbm) {
        gbm_device_destroy(p->gbm);
    }
    if (p->render_fd >= 0) {
        close(p->render_fd);
    }
#endif
    g_free(p);
}

void dg_gl_clear_current(DgGLPlatform *p)
{
    if (!p) {
        return;
    }
#ifdef CONFIG_DARWIN
    CGLSetCurrentContext(NULL);
#else
    eglMakeCurrent(p->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
#endif
}

DgGLContext *dg_gl_context_new(DgGLPlatform *p, DgGLContext *share,
                                Error **errp)
{
    DgGLContext *c = g_new0(DgGLContext, 1);

    if (p->next_context_serial == UINT64_MAX) {
        error_setg(errp, "Native context identity exhausted");
        g_free(c);
        return NULL;
    }
    c->serial = ++p->next_context_serial;
    c->platform = p;
    c->draw_buffer = c->read_buffer = GL_BACK;
    if (p->texture_count > DG_GL_MAX_TEXTURES - 2) {
        error_setg(errp, "Native texture object limit reached");
        g_free(c);
        return NULL;
    }
    g_atomic_ref_count_init(&c->refs);
#ifdef CONFIG_DARWIN
    /*
     * Internal attachments share storage; guest object namespaces are mapped
     * by the command layer before exposing any object-taking GL entrypoint.
     */
    CGLError err = CGLCreateContext(p->format, p->root, &c->render);
    if (err == kCGLNoError) {
        err = CGLCreateContext(p->format, c->render, &c->completion);
    }
    if (err != kCGLNoError) {
        error_setg(errp, "CGL context: %s", CGLErrorString(err));
        dg_gl_context_free(c);
        return NULL;
    }
#else
    c->render = eglCreateContext(p->display, p->config, p->root, NULL);
    if (c->render == EGL_NO_CONTEXT) {
        error_setg(errp, "EGL context: 0x%x", eglGetError());
        dg_gl_context_free(c);
        return NULL;
    }
#endif
    c->textures = share ? share->textures : g_new0(DgTextureNamespace, 1);
    c->textures->refs++;
    c->default_texture = g_new0(DgTexture, 1);
    c->default_texture->refs = 2; /* context ownership plus current binding */
    c->default_texture->target = GL_TEXTURE_2D;
    c->bound_texture = c->default_texture;
    c->default_texture_1d = g_new0(DgTexture, 1);
    c->default_texture_1d->refs = 2;
    c->default_texture_1d->target = GL_TEXTURE_1D;
    c->bound_texture_1d = c->default_texture_1d;
    p->texture_count += 2;
    return c;
}

void dg_gl_context_free(DgGLContext *c)
{
    if (!c || !g_atomic_ref_count_dec(&c->refs)) {
        return;
    }
#ifdef CONFIG_DARWIN
    CGLContextObj previous = CGLGetCurrentContext();
    CGLSetCurrentContext(c->render ? c->render : c->platform->root);
    if (c->in_begin) {
        glEnd();
    }
    for (unsigned i = 0; i < c->attrib_depth; i++) {
        texture_unref(c->platform, c->attrib[i].texture);
        texture_unref(c->platform, c->attrib[i].texture_1d);
    }
    texture_unref(c->platform, c->bound_texture);
    texture_unref(c->platform, c->default_texture);
    texture_unref(c->platform, c->bound_texture_1d);
    texture_unref(c->platform, c->default_texture_1d);
    texture_namespace_unref(c->platform, c->textures);
    if (c->framebuffer) {
        CGLSetCurrentContext(c->render);
        dgDeleteFramebuffers(1, &c->framebuffer);
    }
    CGLSetCurrentContext(previous == c->render || previous == c->completion ?
                         NULL : previous);
    if (c->completion) {
        CGLDestroyContext(c->completion);
    }
    if (c->render) {
        CGLDestroyContext(c->render);
    }
#else
    EGLContext previous = eglGetCurrentContext();
    eglMakeCurrent(c->platform->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                    c->render != EGL_NO_CONTEXT ?
                    c->render : c->platform->root);
    if (c->in_begin) {
        glEnd();
    }
    for (unsigned i = 0; i < c->attrib_depth; i++) {
        texture_unref(c->platform, c->attrib[i].texture);
        texture_unref(c->platform, c->attrib[i].texture_1d);
    }
    texture_unref(c->platform, c->bound_texture);
    texture_unref(c->platform, c->default_texture);
    texture_unref(c->platform, c->bound_texture_1d);
    texture_unref(c->platform, c->default_texture_1d);
    texture_namespace_unref(c->platform, c->textures);
    if (c->framebuffer) {
        eglMakeCurrent(c->platform->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                        c->render);
        dgDeleteFramebuffers(1, &c->framebuffer);
    }
    eglMakeCurrent(c->platform->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                    previous == c->render ? EGL_NO_CONTEXT : previous);
    if (c->render != EGL_NO_CONTEXT) {
        eglDestroyContext(c->platform->display, c->render);
    }
#endif
    g_free(c);
}

DgGLDrawable *dg_gl_drawable_new(DgGLPlatform *p, uint32_t width,
                                  uint32_t height, Error **errp)
{
    DgGLDrawable *d = g_new0(DgGLDrawable, 1);

    d->width = width;
    d->height = height;
#ifdef CONFIG_DARWIN
    CGLSetCurrentContext(p->root);
#else
    if (!eglMakeCurrent(p->display, EGL_NO_SURFACE, EGL_NO_SURFACE, p->root)) {
        error_setg(errp, "EGL internal context: 0x%x", eglGetError());
        g_free(d);
        return NULL;
    }
#endif
    GLuint colors[2];
    glGenTextures(2, colors);
    d->color = colors[0];
    d->front = colors[1];
    for (unsigned i = 0; i < G_N_ELEMENTS(colors); i++) {
        glBindTexture(GL_TEXTURE_2D, colors[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, NULL);
    }
    dgGenRenderbuffers(1, &d->depth);
    dgBindRenderbuffer(GL_RENDERBUFFER, d->depth);
    dgRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    if (glGetError() != GL_NO_ERROR) {
        error_setg(errp, "Allocating internal drawable color/depth failed");
        dg_gl_drawable_free(p, d);
        return NULL;
    }
    GLuint framebuffer;
    dgGenFramebuffers(1, &framebuffer);
    dgBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    dgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, d->color, 0);
    dgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                            GL_TEXTURE_2D, d->front, 0);
    dgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    dgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    if (dgCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        dgDeleteFramebuffers(1, &framebuffer);
        error_setg(errp, "Initializing drawable framebuffer failed");
        dg_gl_drawable_free(p, d);
        return NULL;
    }
    glClearColor(0, 0, 0, 0);
    glClearDepth(1);
    glClearStencil(0);
    const GLenum buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, buffers);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    dgBindFramebuffer(GL_FRAMEBUFFER, 0);
    dgDeleteFramebuffers(1, &framebuffer);
    dg_gl_flush_drawable(d);
    return d;
}

void dg_gl_flush_drawable(DgGLDrawable *d)
{
    if (d->last_write) {
        glDeleteSync(d->last_write);
    }
    d->last_write = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
}

void dg_gl_drawable_free(DgGLPlatform *p, DgGLDrawable *d)
{
    if (!d) {
        return;
    }
#ifdef CONFIG_DARWIN
    CGLSetCurrentContext(p->root);
#else
    eglMakeCurrent(p->display, EGL_NO_SURFACE, EGL_NO_SURFACE, p->root);
#endif
    if (d->depth) {
        dgDeleteRenderbuffers(1, &d->depth);
    }
    if (d->last_write) {
        glDeleteSync(d->last_write);
    }
    if (d->color) {
        glDeleteTextures(1, &d->color);
    }
    if (d->front) {
        glDeleteTextures(1, &d->front);
    }
    dg_gl_clear_current(p);
    g_free(d);
}

static GLenum read_attachment(GLenum buffer)
{
    return buffer == GL_BACK || buffer == GL_BACK_LEFT ?
           GL_COLOR_ATTACHMENT0 : GL_COLOR_ATTACHMENT1;
}

static void select_buffers(DgGLContext *c)
{
    if (c->draw_buffer == GL_FRONT_AND_BACK || c->draw_buffer == GL_LEFT) {
        const GLenum buffers[2] = {
            GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
        };
        glDrawBuffers(2, buffers);
    } else {
        glDrawBuffer(c->draw_buffer == GL_NONE ? GL_NONE :
                     read_attachment(c->draw_buffer));
    }
    glReadBuffer(read_attachment(c->read_buffer));
}

void dg_gl_exchange(DgGLContext *c, DgGLDrawable *d)
{
    GLuint back = d->color;
    d->color = d->front;
    d->front = back;
    /*
     * Logical exchange changes names, never pixel storage. Other contexts
     * reattach the current pair when the worker next makes them current.
     */
    dgBindFramebuffer(GL_FRAMEBUFFER, c->framebuffer);
    dgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, d->color, 0);
    dgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                            GL_TEXTURE_2D, d->front, 0);
    select_buffers(c);
}

bool dg_gl_make_current(DgGLContext *c, DgGLDrawable *d, Error **errp)
{
#ifdef CONFIG_DARWIN
    CGLError err = CGLSetCurrentContext(c->render);
    if (err != kCGLNoError) {
        error_setg(errp, "CGL make current: %s", CGLErrorString(err));
        return false;
    }
#else
    if (!eglMakeCurrent(c->platform->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                         c->render)) {
        error_setg(errp, "EGL make current: 0x%x", eglGetError());
        return false;
    }
#endif
    if (c->in_begin) {
        if (c->drawable != d) {
            error_setg(errp, "Cannot change drawable inside an open primitive");
            return false;
        }
        return true;
    }
    if (d->last_write) {
        glWaitSync(d->last_write, 0, GL_TIMEOUT_IGNORED);
        glDeleteSync(d->last_write);
        d->last_write = NULL;
    }
    if (!c->framebuffer) {
        dgGenFramebuffers(1, &c->framebuffer);
    }
    dgBindFramebuffer(GL_FRAMEBUFFER, c->framebuffer);
    dgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, d->color, 0);
    dgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                            GL_TEXTURE_2D, d->front, 0);
    dgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    dgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    select_buffers(c);
    if (dgCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        error_setg(errp, "Internal drawable framebuffer is incomplete");
        return false;
    }
    c->drawable = d;
    if (!c->initialized) {
        glViewport(0, 0, d->width, d->height);
        c->initialized = true;
    }
    return true;
}

static void *host_texture_allocate(void *opaque, size_t bytes)
{
    return g_try_malloc0(bytes);
}

static void host_texture_free(void *opaque, void *object)
{
    g_free(object);
}

static void host_texture_forget_read(void *opaque, DreamGpuTexture *texture)
{
    DgGLPlatform *p = opaque;
    if (p->read_texture == texture) {
        g_clear_pointer(&p->read_pixels, g_free);
        p->read_texture = NULL;
        p->read_bytes = 0;
    }
}

static DreamGpuTextureMemory texture_memory(DgGLPlatform *p)
{
    return (DreamGpuTextureMemory) {
        .api = &p->gl_api, .bytes = &p->texture_bytes, .count = &p->texture_count,
        .opaque = p, .allocate = host_texture_allocate, .free = host_texture_free,
        .forget_read = host_texture_forget_read,
    };
}

static void texture_unref(DgGLPlatform *p, DgTexture *texture)
{
    DreamGpuTextureMemory memory = texture_memory(p);
    dreamgpu_texture_unref(&memory, texture);
}

static void texture_namespace_unref(DgGLPlatform *p, DgTextureNamespace *ns)
{
    DreamGpuTextureMemory memory = texture_memory(p);
    dreamgpu_texture_namespace_unref(&memory, ns);
}

static void store_guest_error(DgGLContext *c, GLenum error)
{
    if (error != GL_NO_ERROR) {
        unsigned bit = error - GL_INVALID_ENUM;
        c->guest_errors |= 1U << (bit < 8 ? bit : 2);
    }
}

static void remember_guest_error(DgGLContext *c)
{
    /* GL defines separate error flags; preserve each while doing host work. */
    for (unsigned i = 0; i < 8; i++) {
        GLenum error = glGetError();
        if (error == GL_NO_ERROR) {
            break;
        }
        store_guest_error(c, error);
    }
}

static void texture_wait(DgGLContext *c, DgTexture *texture)
{
    dreamgpu_texture_wait(&c->platform->gl_api, texture, c->serial);
}

static void texture_written(DgGLContext *c, DgTexture *texture)
{
    dreamgpu_texture_written(&c->platform->gl_api, texture, c->serial);
}

static DgTexture *bound_texture(DgGLContext *c, GLenum target)
{
    return target == GL_TEXTURE_1D ? c->bound_texture_1d : c->bound_texture;
}

static uint32_t bind_texture(DgGLContext *c, GLenum target,
                             uint32_t guest_name)
{
    DreamGpuTextureMemory memory = texture_memory(c->platform);
    return dreamgpu_texture_bind(&memory, c->textures, target, guest_name,
                                 target == GL_TEXTURE_1D ? c->default_texture_1d : c->default_texture,
                                 target == GL_TEXTURE_1D ? &c->bound_texture_1d : &c->bound_texture,
                                 c->serial, &c->guest_errors);
}

static void delete_textures(DgGLContext *c, const uint8_t *data,
                             uint32_t count)
{
    DreamGpuTextureMemory memory = texture_memory(c->platform);
    dreamgpu_texture_delete(&memory, c->textures, data, count,
                             c->default_texture, c->default_texture_1d,
                             &c->bound_texture, &c->bound_texture_1d);
}

static uint32_t copy_texture(DgGLContext *c, uint32_t fn, const uint8_t *args)
{
    DgTexture *t = c->bound_texture;
    DgGLDrawable *d = c->drawable;
    bool image = fn == FEnum_glCopyTexImage2D;
    uint32_t a[8];
    uint32_t error = dg_gl_call_validate(fn, args);

    if (error) {
        return error;
    }
    for (unsigned i = 0; i < G_N_ELEMENTS(a); i++) {
        a[i] = ldl_le_p(args + i * 4);
    }
    uint32_t level = a[1];
    int32_t x = (int32_t)a[image ? 3 : 4];
    int32_t y = (int32_t)a[image ? 4 : 5];
    uint32_t w = a[image ? 5 : 6], h = a[image ? 6 : 7];
    uint64_t allocation = (uint64_t)w * h * 4;

    /*
     * Source x/y are signed GL coordinates. Pixels outside the read drawable
     * have undefined values, not a GL error; Wine copies padded texture extents
     * this way. Native GL handles the source intersection. Keep destination
     * bounds and the independent width/height/allocation work limits strict.
     */
    if (!d ||
        (!image && (a[2] > t->widths[level] || a[3] > t->heights[level] ||
                    w > t->widths[level] - a[2] ||
                    h > t->heights[level] - a[3]))) {
        return DG_GL_ERROR_TEXTURE;
    }
    if (image && c->platform->texture_bytes - t->levels[level] + allocation >
                 DG_GL_MAX_TEXTURE_BYTES) {
        return DG_GL_ERROR_LIMIT;
    }
    texture_wait(c, t);
    remember_guest_error(c);
    if (image) {
        glCopyTexImage2D(a[0], level, a[2], x, y, w, h, 0);
    } else {
        glCopyTexSubImage2D(a[0], level, a[2], a[3], x, y, w, h);
    }
    GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR) {
        store_guest_error(c, gl_error);
        return DG_GL_ERROR_TEXTURE;
    }
    texture_written(c, t);
    if (image) {
        c->platform->texture_bytes = c->platform->texture_bytes -
                                     t->levels[level] + allocation;
        t->levels[level] = allocation;
        t->widths[level] = w;
        t->heights[level] = h;
        t->undefined_levels &= ~(1U << level);
    }
    return 0;
}

static uint32_t draw_arrays(DgGLContext *c, uint32_t fn, const uint32_t *a,
                             const uint8_t *data, uint32_t bytes)
{
    bool elements = fn == FEnum_glDrawElements;
    uint32_t vertices = a[elements ? 3 : 2];
    uint32_t attributes = a[elements ? 4 : 3];
    unsigned stride = DG_GL_VERTEX_SIZE(attributes);
    const uint8_t *indices = data + vertices * stride;
    g_autofree uint8_t *native = NULL;

    if (!vertices || (elements && !a[1])) {
        return 0;
    }
    if ((c->bound_texture->undefined_levels && glIsEnabled(GL_TEXTURE_2D)) ||
        (c->bound_texture_1d->undefined_levels && glIsEnabled(GL_TEXTURE_1D))) {
        return DG_GL_ERROR_TEXTURE;
    }
    if (HOST_BIG_ENDIAN) {
        native = g_memdup2(data, bytes);
        for (unsigned v = 0; v < vertices; v++) {
            for (unsigned word = 0; word < stride / 4; word++) {
                unsigned offset = v * stride + word * 4;
                stl_he_p(native + offset, ldl_le_p(data + offset));
            }
        }
        if (elements && index_bytes(a[2]) > 1) {
            unsigned size = index_bytes(a[2]);
            uint8_t *out = native + vertices * stride;
            for (unsigned i = 0; i < a[1]; i++) {
                if (size == 2) {
                    stw_he_p(out + i * size, lduw_le_p(indices + i * size));
                } else {
                    stl_he_p(out + i * size, ldl_le_p(indices + i * size));
                }
            }
        }
        data = native;
        indices = data + vertices * stride;
    }
    texture_wait(c, c->bound_texture);
    texture_wait(c, c->bound_texture_1d);
    remember_guest_error(c);
    uint32_t gl_error = GL_NO_ERROR;
    uint32_t error = dreamgpu_gl_arrays(&c->platform->gl_api, fn, a,
                                        data, bytes, &gl_error);
    if (gl_error != GL_NO_ERROR) {
        store_guest_error(c, gl_error);
    }
    return error;
}

/* Initialize allocation-only images without exposing recycled texture bytes. */
static GLenum zero_texture(DgTexture *t, uint32_t level, uint32_t w,
                           uint32_t h)
{
    GLenum error = GL_NO_ERROR;
    if (t->target == GL_TEXTURE_1D) {
        g_autofree uint8_t *zero = g_malloc0(w * 4);
        glTexSubImage1D(GL_TEXTURE_1D, level, 0, w, GL_RGBA, GL_UNSIGNED_BYTE, zero);
        return glGetError();
    }
    bool cleared = false;

    /*
     * A deleted-but-bound object has no attachable name; rebinding its old
     * number would select a new object and violate the guest namespace.
     */
    if (t->name && !t->deleted) {
        GLuint framebuffer = 0;
        GLint read_fb, draw_fb;
        GLfloat color[4];
        GLboolean mask[4], scissor = glIsEnabled(GL_SCISSOR_TEST);

        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fb);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fb);
        glGetFloatv(GL_COLOR_CLEAR_VALUE, color);
        glGetBooleanv(GL_COLOR_WRITEMASK, mask);
        dgGenFramebuffers(1, &framebuffer);
        if (!framebuffer) {
            error = glGetError();
            return error ? error : GL_OUT_OF_MEMORY;
        }
        dgBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        dgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                GL_TEXTURE_2D, t->name, level);
        if (dgCheckFramebufferStatus(GL_FRAMEBUFFER) ==
            GL_FRAMEBUFFER_COMPLETE) {
            glDisable(GL_SCISSOR_TEST);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            cleared = true;
        }
        error = glGetError();
        dgBindFramebuffer(GL_READ_FRAMEBUFFER, read_fb);
        dgBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fb);
        glClearColor(color[0], color[1], color[2], color[3]);
        glColorMask(mask[0], mask[1], mask[2], mask[3]);
        if (scissor) {
            glEnable(GL_SCISSOR_TEST);
        }
        if (framebuffer) {
            dgDeleteFramebuffers(1, &framebuffer);
        }
    }
    if (error || cleared) {
        return error;
    }
    /*
     * Non-renderable legacy formats and unnamed bindings still use bounded
     * storage: one zero tile, never a full-size CPU image.
     */
    /*
     * Allocation has no source pixels: clear internal components independently
     * of the caller's otherwise-unused external format (RGB implies alpha1).
     */
    uint32_t row_bytes = w * 4;
    uint32_t rows = MIN(h, 65536 / row_bytes);
    g_autofree uint8_t *zero = g_malloc0(rows * row_bytes);
    for (uint32_t y = 0; y < h; y += rows) {
        glTexSubImage2D(GL_TEXTURE_2D, level, 0, y, w, MIN(rows, h - y),
                         GL_RGBA, GL_UNSIGNED_BYTE, zero);
        error = glGetError();
        if (error) {
            break;
        }
    }
    return error;
}

static uint32_t host_zero_texture(void *opaque, DreamGpuTexture *texture,
                                   uint32_t level, uint32_t width,
                                   uint32_t height)
{
    return zero_texture(texture, level, width, height);
}

uint32_t dg_gl_data_call(DgGLContext *c, uint32_t fn, const uint8_t *args,
                          const uint8_t *data, uint32_t bytes)
{
    uint32_t a[8] = { 0 };
    uint32_t words = dg_gl_function_words(fn) & ~DG_GL_FUNCTION_INLINE_DATA;
    DgTexture *texture = c->bound_texture;

    if (c->in_begin && fn != FEnum_glMaterialfv) {
        return DG_GL_ERROR_CONTEXT;
    }
    for (unsigned i = 0; i < words; i++) {
        a[i] = ldl_le_p(args + i * 4);
    }
    texture = bound_texture(c, a[0]);
    if (vector_function(fn)) {
        bool parameter = fn == FEnum_glTexParameterfv ||
                         fn == FEnum_glTexParameteriv;
        if (parameter) {
            texture_wait(c, texture);
            remember_guest_error(c);
        }
        vector_call(fn, a, data);
        if (parameter) {
            GLenum error = glGetError();
            if (error != GL_NO_ERROR) {
                store_guest_error(c, error);
                return DG_GL_ERROR_TEXTURE;
            }
            texture_written(c, texture);
        }
        return 0;
    }
    if (fn == FEnum_glDrawArrays || fn == FEnum_glDrawElements) {
        return draw_arrays(c, fn, a, data, bytes);
    }
    if (fn == FEnum_glDeleteTextures) {
        delete_textures(c, data, a[0]);
        return 0;
    }
    texture = bound_texture(c, a[0]);
    texture_wait(c, texture);
    remember_guest_error(c);
    uint32_t gl_error = GL_NO_ERROR;
    uint32_t error = dreamgpu_texture_upload(&c->platform->gl_api, texture,
                                             &c->platform->texture_bytes,
                                             c->serial, fn, a, data, bytes,
                                             host_zero_texture, c, &gl_error);
    if (gl_error != GL_NO_ERROR) {
        store_guest_error(c, gl_error);
    }
    return error;
}

bool dg_gl_context_in_begin(DgGLContext *c)
{
    return c->in_begin;
}

/* Read one bounded canonical tile (128 pixels for legacy requests). The first tile
 * takes the intentional GPU readback once; subsequent tiles reuse one cache
 * bounded by the device's maximum texture size. A write invalidates it by
 * version, and freeing an object clears the weak cache identity. */
static uint32_t texture_read(DgGLContext *c, GLenum target, uint32_t level,
                             uint32_t first, uint32_t capacity, uint8_t *result)
{
    DgGLPlatform *p = c->platform;
    DgTexture *texture = bound_texture(c, target);
    uint64_t bytes = (uint64_t)texture->widths[level] * texture->heights[level] * 4;
    if (!bytes || (uint64_t)first * 4 >= bytes || texture->undefined_levels & (1U << level)) {
        return DG_GL_ERROR_TEXTURE;
    }
    if (!first || p->read_texture != texture || p->read_version != texture->version ||
        p->read_level != level || p->read_bytes != bytes) {
        GLint alignment, row_length, skip_rows, skip_pixels, swap_bytes;
        g_clear_pointer(&p->read_pixels, g_free);
        p->read_texture = NULL;
        p->read_bytes = 0;
        if (bytes > (uint64_t)DG_GL_MAX_TEXTURE_DIMENSION * DG_GL_MAX_TEXTURE_DIMENSION * 4) {
            return DG_GL_ERROR_LIMIT;
        }
        p->read_pixels = g_try_malloc0(bytes);
        if (!p->read_pixels) {
            return DG_GL_ERROR_LIMIT;
        }
        texture_wait(c, texture);
        remember_guest_error(c);
        glGetIntegerv(GL_PACK_ALIGNMENT, &alignment);
        glGetIntegerv(GL_PACK_ROW_LENGTH, &row_length);
        glGetIntegerv(GL_PACK_SKIP_ROWS, &skip_rows);
        glGetIntegerv(GL_PACK_SKIP_PIXELS, &skip_pixels);
        glGetIntegerv(GL_PACK_SWAP_BYTES, &swap_bytes);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        glPixelStorei(GL_PACK_SKIP_ROWS, 0);
        glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_PACK_SWAP_BYTES, 0);
        glGetTexImage(target, level, GL_RGBA, GL_UNSIGNED_BYTE, p->read_pixels);
        GLenum error = glGetError();
        glPixelStorei(GL_PACK_ALIGNMENT, alignment);
        glPixelStorei(GL_PACK_ROW_LENGTH, row_length);
        glPixelStorei(GL_PACK_SKIP_ROWS, skip_rows);
        glPixelStorei(GL_PACK_SKIP_PIXELS, skip_pixels);
        glPixelStorei(GL_PACK_SWAP_BYTES, swap_bytes);
        if (error) {
            store_guest_error(c, error);
            g_clear_pointer(&p->read_pixels, g_free);
            return DG_GL_ERROR_HOST;
        }
        p->read_texture = texture;
        p->read_version = texture->version;
        p->read_level = level;
        p->read_bytes = bytes;
    }
    uint32_t remaining = p->read_bytes - first * 4;
    memset(result, 0, capacity);
    memcpy(result, p->read_pixels + first * 4, MIN(remaining, capacity));
    if (remaining <= capacity) {
        g_clear_pointer(&p->read_pixels, g_free);
        p->read_texture = NULL;
        p->read_bytes = 0;
    }
    return 0;
}

uint32_t dg_gl_query(DgGLContext *c, uint32_t fn, const uint8_t *args,
                       uint8_t *result, uint32_t *bytes, uint32_t *type)
{
    uint32_t a = ldl_le_p(args), b = ldl_le_p(args + 4);
    uint32_t d = ldl_le_p(args + 8);
    unsigned count = query_shape(fn, args, type);
    GLint integers[16] = { 0 };
    GLfloat floats[16] = { 0 };
    GLdouble doubles[16] = { 0 };
    GLboolean booleans[16] = { 0 };
    bool logical = false;

    if (!count || c->in_begin) {
        return DG_GL_ERROR_CONTEXT;
    }
    if (fn == FEnum_glGetTexImage) {
        uint32_t error = texture_read(c, a, b & 0xffff, d, count * 4, result);
        if (!error) {
            *bytes = count * 4;
        }
        return error;
    }
    if (fn == FEnum_glReadPixels) {
        unsigned width = d & 0xffff, height = d >> 16;
        GLint alignment, row_length, skip_rows, skip_pixels, swap_bytes;
        if (!c->drawable || a >= c->drawable->width ||
            b >= c->drawable->height || width > c->drawable->width - a ||
            height > c->drawable->height - b) {
            return DG_GL_ERROR_DRAWABLE;
        }
        remember_guest_error(c);
        glGetIntegerv(GL_PACK_ALIGNMENT, &alignment);
        glGetIntegerv(GL_PACK_ROW_LENGTH, &row_length);
        glGetIntegerv(GL_PACK_SKIP_ROWS, &skip_rows);
        glGetIntegerv(GL_PACK_SKIP_PIXELS, &skip_pixels);
        glGetIntegerv(GL_PACK_SWAP_BYTES, &swap_bytes);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        glPixelStorei(GL_PACK_SKIP_ROWS, 0);
        glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_PACK_SWAP_BYTES, 0);
        glReadPixels(a, b, width, height, GL_RGBA, GL_UNSIGNED_BYTE, result);
        GLenum error = glGetError();
        glPixelStorei(GL_PACK_ALIGNMENT, alignment);
        glPixelStorei(GL_PACK_ROW_LENGTH, row_length);
        glPixelStorei(GL_PACK_SKIP_ROWS, skip_rows);
        glPixelStorei(GL_PACK_SKIP_PIXELS, skip_pixels);
        glPixelStorei(GL_PACK_SWAP_BYTES, swap_bytes);
        if (error) {
            store_guest_error(c, error);
            return DG_GL_ERROR_HOST;
        }
        *bytes = count * 4;
        return 0;
    }
    if (fn == FEnum_glGetError) {
        remember_guest_error(c);
        for (unsigned i = 0; i < 8; i++) {
            if (c->guest_errors & (1U << i)) {
                integers[0] = GL_INVALID_ENUM + i;
                c->guest_errors &= ~(1U << i);
                break;
            }
        }
    } else if (fn == FEnum_glGetString) {
        memcpy(result, query_string(a), count);
    } else if (fn == FEnum_glIsTexture) {
        booleans[0] = a && (dreamgpu_texture_lookup(c->textures, a) != NULL);
    } else if (fn == FEnum_glIsEnabled) {
        booleans[0] = glIsEnabled(a);
    } else if (fn == FEnum_glGetTexParameteriv) {
        glGetTexParameteriv(a, b, integers);
    } else if (fn == FEnum_glGetTexParameterfv) {
        glGetTexParameterfv(a, b, floats);
    } else if (fn == FEnum_glGetTexLevelParameteriv) {
        glGetTexLevelParameteriv(a, b, d, integers);
    } else if (fn == FEnum_glGetTexLevelParameterfv) {
        glGetTexLevelParameterfv(a, b, d, floats);
    } else if (fn == FEnum_glGetTexEnviv) {
        glGetTexEnviv(a, b, integers);
    } else if (fn == FEnum_glGetTexEnvfv) {
        glGetTexEnvfv(a, b, floats);
    } else if (fn == FEnum_glGetLightfv) {
        glGetLightfv(a, b, floats);
    } else if (fn == FEnum_glGetLightiv) {
        glGetLightiv(a, b, integers);
    } else if (fn == FEnum_glGetMaterialfv) {
        glGetMaterialfv(a, b, floats);
    } else if (fn == FEnum_glGetMaterialiv) {
        glGetMaterialiv(a, b, integers);
    } else if (fn == FEnum_glGetTexGenfv) {
        glGetTexGenfv(a, b, floats);
    } else if (fn == FEnum_glGetTexGeniv) {
        glGetTexGeniv(a, b, integers);
    } else if (fn == FEnum_glGetTexGendv) {
        glGetTexGendv(a, b, doubles);
    } else if (fn == FEnum_glGetClipPlane) {
        glGetClipPlane(a, doubles);
    } else {
        /* Internal FBO limits and names are not the guest device contract. */
        switch (a) {
        case GL_DRAW_BUFFER:
            integers[0] = c->draw_buffer;
            logical = true;
            break;
        case GL_READ_BUFFER:
            integers[0] = c->read_buffer;
            logical = true;
            break;
        case GL_DOUBLEBUFFER:
            integers[0] = 1;
            logical = true;
            break;
        case GL_STEREO:
        case GL_AUX_BUFFERS:
            logical = true;
            break;
        case GL_TEXTURE_BINDING_1D:
            integers[0] = c->bound_texture_1d->guest_name;
            logical = true;
            break;
        case GL_TEXTURE_BINDING_2D:
            integers[0] = c->bound_texture->guest_name;
            logical = true;
            break;
        case GL_ATTRIB_STACK_DEPTH:
            integers[0] = c->attrib_depth;
            logical = true;
            break;
        case GL_MAX_ATTRIB_STACK_DEPTH:
            integers[0] = DG_ATTRIB_STACK;
            logical = true;
            break;
        case GL_MAX_TEXTURE_SIZE:
            integers[0] = DG_GL_MAX_TEXTURE_DIMENSION;
            logical = true;
            break;
        case GL_MAX_VIEWPORT_DIMS:
            integers[0] = integers[1] = DG_GL_MAX_DIMENSION;
            logical = true;
            break;
        case GL_MAX_LIGHTS:
            integers[0] = 8;
            logical = true;
            break;
        case GL_MAX_CLIP_PLANES:
            integers[0] = 6;
            logical = true;
            break;
        }
        if (!logical) {
            switch (*type) {
            case DG_GL_RESULT_INT:
                glGetIntegerv(a, integers);
                break;
            case DG_GL_RESULT_FLOAT:
                glGetFloatv(a, floats);
                break;
            case DG_GL_RESULT_DOUBLE:
                glGetDoublev(a, doubles);
                break;
            case DG_GL_RESULT_BOOL:
                glGetBooleanv(a, booleans);
                break;
            }
            /* EXT_secondary_color defines secondary alpha as zero. Apple's
             * legacy driver reports one, although color-sum ignores it. Keep
             * the public four-component query faithful to the GL contract. */
            if (a == GL_CURRENT_SECONDARY_COLOR) {
                integers[3] = 0;
                floats[3] = 0;
                doubles[3] = 0;
                booleans[3] = GL_FALSE;
            }
        }
    }
    if (logical) {
        for (unsigned i = 0; i < count; i++) {
            doubles[i] = a == GL_TEXTURE_BINDING_2D ?
                         (uint32_t)integers[i] : integers[i];
            floats[i] = doubles[i];
            booleans[i] = !!integers[i];
        }
    }
    for (unsigned i = 0; i < count; i++) {
        switch (*type) {
        case DG_GL_RESULT_BOOL:
            result[i] = !!booleans[i];
            break;
        case DG_GL_RESULT_INT:
            stl_le_p(result + i * 4, integers[i]);
            break;
        case DG_GL_RESULT_FLOAT: {
            uint32_t bits;
            memcpy(&bits, &floats[i], 4);
            stl_le_p(result + i * 4, bits);
            break;
        }
        case DG_GL_RESULT_DOUBLE: {
            uint64_t bits;
            memcpy(&bits, &doubles[i], 8);
            stq_le_p(result + i * 8, bits);
            break;
        }
        }
    }
    *bytes = dg_gl_query_result_bytes(fn, args);
    return 0;
}

static GLfloat gl_arg_float(const uint8_t *p)
{
    union { uint32_t bits; float f; } v = { .bits = ldl_le_p(p) };
    return v.f;
}

static GLdouble gl_arg_double(const uint8_t *p)
{
    union { uint64_t bits; double f; } v = { .bits = ldq_le_p(p) };
    return v.f;
}

#define U(n) ldl_le_p(args + (n) * 4)
#define F(n) gl_arg_float(args + (n) * 4)
#define D(n) gl_arg_double(args + (n) * 4)
static uint32_t attrib_push(DgGLContext *c, GLbitfield mask)
{
    DgAttrib *saved;
    if (mask & ~GL_ALL_ATTRIB_BITS) {
        store_guest_error(c, GL_INVALID_VALUE);
        return 0;
    }
    if (c->attrib_depth == DG_ATTRIB_STACK) {
        store_guest_error(c, GL_STACK_OVERFLOW);
        return 0;
    }
    remember_guest_error(c);
    glPushAttrib(mask);
    GLenum error = glGetError();
    if (error) {
        store_guest_error(c, error);
        return 0;
    }
    saved = &c->attrib[c->attrib_depth++];
    *saved = (DgAttrib){ .mask = mask, .draw_buffer = c->draw_buffer,
                         .read_buffer = c->read_buffer };
    if (mask & (GL_FOG_BIT | GL_ENABLE_BIT)) {
        saved->color_sum = glIsEnabled(GL_COLOR_SUM);
    }
    if (mask & GL_TEXTURE_BIT) {
        saved->texture = c->bound_texture;
        saved->texture_1d = c->bound_texture_1d;
        saved->texture->refs++;
        saved->texture_1d->refs++;
    }
    return 0;
}

static uint32_t attrib_pop(DgGLContext *c)
{
    DgAttrib *saved;
    if (!c->attrib_depth) {
        store_guest_error(c, GL_STACK_UNDERFLOW);
        return 0;
    }
    saved = &c->attrib[c->attrib_depth - 1];
    if (saved->texture) {
        texture_wait(c, saved->texture);
        texture_wait(c, saved->texture_1d);
    }
    remember_guest_error(c);
    glPopAttrib();
    GLenum error = glGetError();
    if (error) {
        store_guest_error(c, error);
        return 0;
    }
    /* COLOR_SUM belongs to both fog and enable attribute groups. Mesa's
     * legacy stack does not consistently restore it for ENABLE_BIT alone. */
    if (saved->mask & (GL_FOG_BIT | GL_ENABLE_BIT)) {
        if (saved->color_sum) {
            glEnable(GL_COLOR_SUM);
        } else {
            glDisable(GL_COLOR_SUM);
        }
    }
    if (saved->mask & GL_COLOR_BUFFER_BIT) {
        c->draw_buffer = saved->draw_buffer;
    }
    if (saved->mask & GL_PIXEL_MODE_BIT) {
        c->read_buffer = saved->read_buffer;
    }
    if (saved->texture) {
        texture_unref(c->platform, c->bound_texture);
        texture_unref(c->platform, c->bound_texture_1d);
        c->bound_texture = saved->texture;
        c->bound_texture_1d = saved->texture_1d;
        texture_written(c, c->bound_texture);
        texture_written(c, c->bound_texture_1d);
    }
    select_buffers(c);
    c->attrib_depth--;
    return 0;
}

/* Remaining resource-side operations; scalar GL execution is Rust-owned. */
static uint32_t scalar_resource(void *opaque, uint32_t fn, const uint8_t *args)
{
    DgGLContext *c = opaque;
    switch (fn) {
    case FEnum_glPushAttrib:
        return attrib_push(c, U(0));
    case FEnum_glPopAttrib:
        return attrib_pop(c);
    case FEnum_glDrawBuffer:
    case FEnum_glReadBuffer:
        if (!buffer_selection(U(0), fn == FEnum_glDrawBuffer)) {
            return DG_GL_ERROR_UNSUPPORTED;
        }
        if (fn == FEnum_glDrawBuffer) {
            c->draw_buffer = U(0);
        } else {
            c->read_buffer = U(0);
        }
        select_buffers(c);
        break;
    case FEnum_glHint:
        if (dg_gl_call_validate(fn, args)) {
            return DG_GL_ERROR_UNSUPPORTED;
        }
        glHint(U(0), U(1));
        break;
    case FEnum_glBindTexture:
        return bind_texture(c, U(0), U(1));
    case FEnum_glCopyTexImage2D:
    case FEnum_glCopyTexSubImage2D:
        return copy_texture(c, fn, args);
    case FEnum_glTexParameteri:
    case FEnum_glTexParameterf:
        if (texture_params(U(0), U(1)) != 1) {
            return DG_GL_ERROR_TEXTURE;
        }
        texture_wait(c, bound_texture(c, U(0)));
        if (fn == FEnum_glTexParameteri) {
            glTexParameteri(U(0), U(1), U(2));
        } else {
            glTexParameterf(U(0), U(1), F(2));
        }
        texture_written(c, bound_texture(c, U(0)));
        break;
    case FEnum_glTexEnvi:
    case FEnum_glTexEnvf:
        if (texture_env_params(U(0), U(1)) != 1) {
            return DG_GL_ERROR_TEXTURE;
        }
        if (fn == FEnum_glTexEnvi) {
            glTexEnvi(U(0), U(1), U(2));
        } else {
            glTexEnvf(U(0), U(1), F(2));
        }
        break;
    case FEnum_glBegin:
        if (U(0) > GL_POLYGON) {
            return DG_GL_ERROR_CONTEXT;
        }
        if ((c->bound_texture->undefined_levels && glIsEnabled(GL_TEXTURE_2D)) ||
        (c->bound_texture_1d->undefined_levels && glIsEnabled(GL_TEXTURE_1D))) {
            return DG_GL_ERROR_TEXTURE;
        }
        texture_wait(c, c->bound_texture);
        texture_wait(c, c->bound_texture_1d);
        break;
    default:
        return DG_GL_ERROR_UNSUPPORTED;
    }
    return 0;
}

uint32_t dg_gl_call(DgGLContext *c, uint32_t fn, const uint8_t *args)
{
    uint32_t words = dg_gl_function_words(fn);
    if (words > 32) {
        return DG_GL_ERROR_UNSUPPORTED;
    }
    return dreamgpu_gl_scalar(&c->platform->gl_api, &c->in_begin,
                              scalar_resource, c, fn, args, words * 4);
}
#undef U
#undef F
#undef D

DgGLImage *dg_gl_image_new(DgGLPlatform *p, uint32_t width, uint32_t height,
                            Error **errp)
{
    DgGLImage *image = g_new0(DgGLImage, 1);

    image->width = width;
    image->height = height;
#ifdef CONFIG_DARWIN
    int64_t w = width, h = height, bpp = 4, format = 0x42475241;
    CFNumberRef wn = CFNumberCreate(NULL, kCFNumberSInt64Type, &w);
    CFNumberRef hn = CFNumberCreate(NULL, kCFNumberSInt64Type, &h);
    CFNumberRef bn = CFNumberCreate(NULL, kCFNumberSInt64Type, &bpp);
    CFNumberRef fn = CFNumberCreate(NULL, kCFNumberSInt64Type, &format);
    const void *keys[] = { kIOSurfaceWidth, kIOSurfaceHeight,
                           kIOSurfaceBytesPerElement, kIOSurfacePixelFormat };
    const void *values[] = { wn, hn, bn, fn };
    CFDictionaryRef dict = CFDictionaryCreate(NULL, keys, values, 4,
                                              &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);

    image->surface = IOSurfaceCreate(dict);
    CFRelease(dict);
    CFRelease(wn);
    CFRelease(hn);
    CFRelease(bn);
    CFRelease(fn);
    if (!image->surface) {
        error_setg(errp, "Creating IOSurface export slot failed");
        g_free(image);
        return NULL;
    }
    image->stride = IOSurfaceGetBytesPerRow(image->surface);
#else
    image->fd = image->fence_fd = -1;
    const uint64_t modifiers[] = { DRM_FORMAT_MOD_LINEAR };

    image->bo = gbm_bo_create_with_modifiers2(p->gbm, width, height,
                                              GBM_FORMAT_ARGB8888, modifiers, 1,
                                              GBM_BO_USE_RENDERING);
    if (!image->bo || gbm_bo_get_plane_count(image->bo) != 1) {
        error_setg(errp, "A single-plane BGRA GBM export buffer is required");
        dg_gl_image_free(p, image);
        return NULL;
    }
    image->fd = gbm_bo_get_fd(image->bo);
    image->stride = gbm_bo_get_stride(image->bo);
    image->offset = gbm_bo_get_offset(image->bo, 0);
    image->modifier = gbm_bo_get_modifier(image->bo);
    if (image->fd < 0 || image->modifier != DRM_FORMAT_MOD_LINEAR) {
        error_setg(errp, "Exporting linear DMA-BUF failed");
        dg_gl_image_free(p, image);
        return NULL;
    }
    const EGLint attrs[] = {
        EGL_WIDTH, width, EGL_HEIGHT, height,
        EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
        EGL_DMA_BUF_PLANE0_FD_EXT, image->fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, image->offset,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, image->stride,
        EGL_NONE,
    };
    image->image = eglCreateImageKHR(p->display, EGL_NO_CONTEXT,
                                    EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
    if (image->image == EGL_NO_IMAGE_KHR) {
        error_setg(errp, "Importing DMA-BUF to EGL failed: 0x%x",
                    eglGetError());
        dg_gl_image_free(p, image);
        return NULL;
    }
#endif
    return image;
}

void dg_gl_image_free(DgGLPlatform *p, DgGLImage *image)
{
    if (!image) {
        return;
    }
#ifdef CONFIG_DARWIN
    if (image->context) {
        CGLSetCurrentContext(image->context->completion);
        if (image->fence) {
            glDeleteSync(image->fence);
        }
        CGLSetCurrentContext(NULL);
        dg_gl_context_free(image->context);
    }
    if (image->surface) {
        CFRelease(image->surface);
    }
#else
    if (image->image != EGL_NO_IMAGE_KHR) {
        p->destroy_image(p->display, image->image);
    }
    if (image->fence_fd >= 0) {
        close(image->fence_fd);
    }
    if (image->fd >= 0) {
        close(image->fd);
    }
    if (image->bo) {
        gbm_bo_destroy(image->bo);
    }
#endif
    g_free(image);
}

bool dg_gl_export(DgGLContext *c, DgGLDrawable *d, DgGLImage *image,
                    bool exchange, Error **errp)
{
    GLuint attachment = 0, framebuffer = 0;
    GLint read_fb, draw_fb, read_buffer, attachment_binding;
    GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
    bool ok = false;

    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fb);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fb);
    glGetIntegerv(GL_READ_BUFFER, &read_buffer);
    if (exchange) {
        dg_gl_exchange(c, d);
    }
#ifdef CONFIG_DARWIN
    GLenum target = GL_TEXTURE_RECTANGLE_ARB;
    glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE_ARB, &attachment_binding);
    glGenTextures(1, &attachment);
    glBindTexture(target, attachment);
#else
    /*
     * Do not temporarily unbind a guest texture here: another context may
     * have deleted its name while this context retains the original object.
     * Rebinding that numeric name would create a different texture.
     */
    glGetIntegerv(GL_RENDERBUFFER_BINDING, &attachment_binding);
    dgGenRenderbuffers(1, &attachment);
    dgBindRenderbuffer(GL_RENDERBUFFER, attachment);
#endif
#ifdef CONFIG_DARWIN
    CGLError err = CGLTexImageIOSurface2D(c->render, target, GL_RGBA8,
                                         d->width, d->height, GL_BGRA,
                                         GL_UNSIGNED_INT_8_8_8_8_REV,
                                         image->surface, 0);
    if (err != kCGLNoError) {
        error_setg(errp, "Binding IOSurface to GL: %s", CGLErrorString(err));
        goto out;
    }
#else
    glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, image->image);
#endif
    dgGenFramebuffers(1, &framebuffer);
    dgBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
#ifdef CONFIG_DARWIN
    dgFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            target, attachment, 0);
#else
    dgFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_RENDERBUFFER, attachment);
#endif
    if (dgCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_COMPLETE) {
        error_setg(errp, "Export framebuffer is incomplete");
        goto out;
    }
    dgBindFramebuffer(GL_READ_FRAMEBUFFER, c->framebuffer);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glDisable(GL_SCISSOR_TEST);
    /* Guest GL is bottom-left; every exported image is top-left. */
    dgBlitFramebuffer(0, d->height, d->width, 0,
                       0, 0, d->width, d->height,
                       GL_COLOR_BUFFER_BIT, GL_NEAREST);
#ifdef CONFIG_DARWIN
    if (image->context) {
        dg_gl_context_free(image->context);
    }
    image->context = c;
    g_atomic_ref_count_inc(&c->refs);
    image->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!image->fence) {
        error_setg(errp, "Creating GL completion fence failed");
        goto out;
    }
    glFlush();
#else
    EGLSyncKHR fence = eglCreateSyncKHR(c->platform->display,
                                       EGL_SYNC_NATIVE_FENCE_ANDROID, NULL);
    glFlush();
    if (image->fence_fd >= 0) {
        close(image->fence_fd);
    }
    image->fence_fd = eglDupNativeFenceFDANDROID(c->platform->display, fence);
    eglDestroySyncKHR(c->platform->display, fence);
    if (image->fence_fd < 0) {
        error_setg(errp, "Exporting EGL completion fence failed");
        goto out;
    }
#endif
    ok = true;
out:
    dgBindFramebuffer(GL_READ_FRAMEBUFFER, read_fb);
    glReadBuffer(read_buffer);
    dgBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fb);
    if (scissor) {
        glEnable(GL_SCISSOR_TEST);
    }
#ifdef CONFIG_DARWIN
    glBindTexture(target, attachment_binding);
#else
    dgBindRenderbuffer(GL_RENDERBUFFER, attachment_binding);
#endif
    if (framebuffer) {
        dgDeleteFramebuffers(1, &framebuffer);
    }
    if (attachment) {
#ifdef CONFIG_DARWIN
        glDeleteTextures(1, &attachment);
#else
        dgDeleteRenderbuffers(1, &attachment);
#endif
    }
    return ok;
}

bool dg_gl_image_ready(DgGLImage *image, Error **errp)
{
#ifdef CONFIG_DARWIN
    int64_t deadline = g_get_monotonic_time() + 5 * G_USEC_PER_SEC;
    GLenum result = GL_TIMEOUT_EXPIRED;

    CGLSetCurrentContext(image->context->completion);
    for (;;) {
        /*
         * Apple's positive-timeout ClientWaitSync spins in the GL driver
         * (including repeated mach_absolute_time calls). Test immediately,
         * then sleep on this dedicated completion thread while the GPU works.
         * Already completed exports incur no sleep, and the render/input
         * threads remain independent. Keep the existing five-second bound.
         */
        result = glClientWaitSync(image->fence, 0, 0);
        if (result != GL_TIMEOUT_EXPIRED ||
            g_get_monotonic_time() >= deadline) {
            break;
        }
        g_usleep(100);
    }
    glDeleteSync(image->fence);
    image->fence = NULL;
    CGLSetCurrentContext(NULL);
    if (result != GL_ALREADY_SIGNALED && result != GL_CONDITION_SATISFIED) {
        error_setg(errp, "Waiting for GL export completion failed: 0x%x",
                    result);
        return false;
    }
#endif
    return true;
}

void dg_gl_image_metadata(DgGLImage *image, uint32_t *stride,
                            uint32_t *offset, uint64_t *modifier)
{
    *stride = image->stride;
    *offset = image->offset;
    *modifier = image->modifier;
}

uint32_t dg_gl_image_port(DgGLImage *image)
{
#ifdef CONFIG_DARWIN
    return IOSurfaceCreateMachPort(image->surface);
#else
    return 0;
#endif
}

int dg_gl_image_fd(DgGLImage *image)
{
#ifdef CONFIG_DARWIN
    return -1;
#else
    return image->fd;
#endif
}

int dg_gl_image_fence_fd(DgGLImage *image)
{
#ifdef CONFIG_DARWIN
    return -1;
#else
    return image->fence_fd;
#endif
}
