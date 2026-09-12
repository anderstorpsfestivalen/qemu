/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Native offscreen OpenGL contexts and GPU-only presentation export.
 */
#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/bswap.h"
#include "standard-headers/juke/retro-gl-funcs.h"
#include "standard-headers/juke/retro-gl.h"
#include "standard-headers/juke/gpu-transport.h"
#include "juke-retro-gl-platform.h"

#ifdef CONFIG_DARWIN
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <OpenGL/CGLIOSurface.h>
#include <IOSurface/IOSurface.h>
#define jrgGenFramebuffers glGenFramebuffersEXT
#define jrgDeleteFramebuffers glDeleteFramebuffersEXT
#define jrgBindFramebuffer glBindFramebufferEXT
#define jrgFramebufferTexture2D glFramebufferTexture2DEXT
#define jrgCheckFramebufferStatus glCheckFramebufferStatusEXT
#define jrgBlitFramebuffer glBlitFramebufferEXT
#define jrgGenRenderbuffers glGenRenderbuffersEXT
#define jrgDeleteRenderbuffers glDeleteRenderbuffersEXT
#define jrgBindRenderbuffer glBindRenderbufferEXT
#define jrgRenderbufferStorage glRenderbufferStorageEXT
#define jrgFramebufferRenderbuffer glFramebufferRenderbufferEXT
#else
#include <epoxy/gl.h>
#include <epoxy/egl.h>
#include <gbm.h>
#include <drm_fourcc.h>
#ifndef EGL_DRM_RENDER_NODE_FILE_EXT
#define EGL_DRM_RENDER_NODE_FILE_EXT 0x3377
#endif
#define jrgGenFramebuffers glGenFramebuffers
#define jrgDeleteFramebuffers glDeleteFramebuffers
#define jrgBindFramebuffer glBindFramebuffer
#define jrgFramebufferTexture2D glFramebufferTexture2D
#define jrgCheckFramebufferStatus glCheckFramebufferStatus
#define jrgBlitFramebuffer glBlitFramebuffer
#define jrgGenRenderbuffers glGenRenderbuffers
#define jrgDeleteRenderbuffers glDeleteRenderbuffers
#define jrgBindRenderbuffer glBindRenderbuffer
#define jrgRenderbufferStorage glRenderbufferStorage
#define jrgFramebufferRenderbuffer glFramebufferRenderbuffer
#endif

typedef struct JrgTexture {
    GLuint name;
    GLenum target;
    uint64_t version;
    uint32_t guest_name;
    uint32_t refs;
    bool deleted;
    uint32_t undefined_levels;
    GLsync last_write;
    uint64_t writer_serial, waiter_serial;
    uint32_t widths[JRG_GL_MAX_TEXTURE_LEVEL + 1];
    uint32_t heights[JRG_GL_MAX_TEXTURE_LEVEL + 1];
    uint64_t levels[JRG_GL_MAX_TEXTURE_LEVEL + 1];
} JrgTexture;

typedef struct JrgTextureNamespace {
    uint32_t refs;
    GHashTable *names;
} JrgTextureNamespace;

struct JrgGLPlatform {
    uint64_t next_context_serial;
    uint64_t texture_bytes;
    uint32_t texture_count;
    /* One bounded CPU cache for genuine texture reads, never presentation. */
    JrgTexture *read_texture;
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

#define JRG_ATTRIB_STACK 16
typedef struct {
    GLbitfield mask;
    GLboolean color_sum;
    GLenum draw_buffer, read_buffer;
    JrgTexture *texture, *texture_1d;
} JrgAttrib;

struct JrgGLContext {
    JrgGLPlatform *platform;
    uint64_t serial;
#ifdef CONFIG_DARWIN
    CGLContextObj render, completion;
#else
    EGLContext render;
#endif
    gatomicrefcount refs;
    GLuint framebuffer;
    bool initialized, in_begin;
    JrgGLDrawable *drawable;
    JrgTextureNamespace *textures;
    JrgTexture *default_texture, *bound_texture;
    JrgTexture *default_texture_1d, *bound_texture_1d;
    uint32_t guest_errors;
    GLenum draw_buffer, read_buffer;
    unsigned attrib_depth;
    JrgAttrib attrib[JRG_ATTRIB_STACK];
};

struct JrgGLDrawable {
    uint32_t width, height;
    GLuint color, front, depth;
    GLsync last_write;
};

struct JrgGLImage {
    uint32_t width, height, stride, offset;
    uint64_t modifier;
#ifdef CONFIG_DARWIN
    IOSurfaceRef surface;
    GLsync fence;
    JrgGLContext *context;
#else
    struct gbm_bo *bo;
    EGLImageKHR image;
    int fd, fence_fd;
#endif
};

static void texture_unref(JrgGLPlatform *p, JrgTexture *texture);
static void texture_namespace_unref(JrgGLPlatform *p, JrgTextureNamespace *ns);
static GLfloat gl_arg_float(const uint8_t *p);
static GLdouble gl_arg_double(const uint8_t *p);

static unsigned light_params(uint32_t light, uint32_t pname)
{
    if (light < GL_LIGHT0 || light > GL_LIGHT7) {
        return 0;
    }
    switch (pname) {
    case GL_AMBIENT:
    case GL_DIFFUSE:
    case GL_SPECULAR:
    case GL_POSITION:
        return 4;
    case GL_SPOT_DIRECTION:
        return 3;
    case GL_SPOT_EXPONENT:
    case GL_SPOT_CUTOFF:
    case GL_CONSTANT_ATTENUATION:
    case GL_LINEAR_ATTENUATION:
    case GL_QUADRATIC_ATTENUATION:
        return 1;
    default:
        return 0;
    }
}

static unsigned material_params(uint32_t face, uint32_t pname)
{
    if (face != GL_FRONT && face != GL_BACK && face != GL_FRONT_AND_BACK) {
        return 0;
    }
    switch (pname) {
    case GL_AMBIENT:
    case GL_DIFFUSE:
    case GL_SPECULAR:
    case GL_EMISSION:
    case GL_AMBIENT_AND_DIFFUSE:
        return 4;
    case GL_COLOR_INDEXES:
        return 3;
    case GL_SHININESS:
        return 1;
    default:
        return 0;
    }
}

static unsigned fog_params(uint32_t pname)
{
    switch (pname) {
    case GL_FOG_COLOR:
        return 4;
    case GL_FOG_MODE:
    case GL_FOG_DENSITY:
    case GL_FOG_START:
    case GL_FOG_END:
    case GL_FOG_INDEX:
        return 1;
    default:
        return 0;
    }
}

static unsigned light_model_params(uint32_t pname)
{
    switch (pname) {
    case GL_LIGHT_MODEL_AMBIENT:
        return 4;
    case GL_LIGHT_MODEL_LOCAL_VIEWER:
    case GL_LIGHT_MODEL_TWO_SIDE:
    case GL_LIGHT_MODEL_COLOR_CONTROL:
        return 1;
    default:
        return 0;
    }
}

static unsigned texgen_params(uint32_t coord, uint32_t pname)
{
    if (coord != GL_S && coord != GL_T && coord != GL_R && coord != GL_Q) {
        return 0;
    }
    switch (pname) {
    case GL_TEXTURE_GEN_MODE:
        return 1;
    case GL_OBJECT_PLANE:
    case GL_EYE_PLANE:
        return 4;
    default:
        return 0;
    }
}

static unsigned texture_params(uint32_t target, uint32_t pname)
{
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) {
        return 0;
    }
    switch (pname) {
    case GL_TEXTURE_BORDER_COLOR:
        return 4;
    case GL_TEXTURE_MIN_FILTER:
    case GL_TEXTURE_MAG_FILTER:
    case GL_TEXTURE_WRAP_S:
    case GL_TEXTURE_WRAP_T:
    case GL_TEXTURE_BASE_LEVEL:
    case GL_TEXTURE_MAX_LEVEL:
        return 1;
    default:
        return 0;
    }
}

static unsigned texture_env_params(uint32_t target, uint32_t pname)
{
    if (target != GL_TEXTURE_ENV) {
        return 0;
    }
    switch (pname) {
    case GL_TEXTURE_ENV_COLOR:
        return 4;
    case GL_TEXTURE_ENV_MODE:
    case GL_COMBINE_RGB: case GL_COMBINE_ALPHA:
    case GL_RGB_SCALE: case GL_ALPHA_SCALE:
    case GL_SOURCE0_RGB: case GL_SOURCE1_RGB: case GL_SOURCE2_RGB:
    case GL_SOURCE0_ALPHA: case GL_SOURCE1_ALPHA: case GL_SOURCE2_ALPHA:
    case GL_OPERAND0_RGB: case GL_OPERAND1_RGB: case GL_OPERAND2_RGB:
    case GL_OPERAND0_ALPHA: case GL_OPERAND1_ALPHA: case GL_OPERAND2_ALPHA:
        return 1;
    default:
        return 0;
    }
}

static unsigned vector_bytes(uint32_t fn, const uint32_t *a)
{
    switch (fn) {
    case FEnum_glTexParameterfv:
    case FEnum_glTexParameteriv:
        return texture_params(a[0], a[1]) * 4;
    case FEnum_glTexEnvfv:
    case FEnum_glTexEnviv:
        return texture_env_params(a[0], a[1]) * 4;
    case FEnum_glLightfv:
        return light_params(a[0], a[1]) * 4;
    case FEnum_glMaterialfv:
        return material_params(a[0], a[1]) * 4;
    case FEnum_glFogfv:
        return fog_params(a[0]) * 4;
    case FEnum_glLightModelfv:
        return light_model_params(a[0]) * 4;
    case FEnum_glTexGenfv:
        return texgen_params(a[0], a[1]) * 4;
    case FEnum_glTexGendv:
        return texgen_params(a[0], a[1]) * 8;
    case FEnum_glClipPlane:
        return a[0] >= GL_CLIP_PLANE0 && a[0] <= GL_CLIP_PLANE5 ? 32 : 0;
    default:
        return 0;
    }
}

static bool vector_function(uint32_t fn)
{
    return fn == FEnum_glTexParameterfv || fn == FEnum_glTexParameteriv ||
           fn == FEnum_glTexEnvfv || fn == FEnum_glTexEnviv ||
           fn == FEnum_glLightfv || fn == FEnum_glMaterialfv ||
           fn == FEnum_glFogfv || fn == FEnum_glLightModelfv ||
           fn == FEnum_glTexGenfv || fn == FEnum_glTexGendv ||
           fn == FEnum_glClipPlane;
}

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

JrgGLPlatform *jrg_gl_platform_new(const char *render_node, Error **errp)
{
    JrgGLPlatform *p = g_new0(JrgGLPlatform, 1);
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
        jrg_gl_platform_free(p);
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
        jrg_gl_platform_free(p);
        return NULL;
    }
    EGLDeviceEXT devices[32];
    EGLint device_count = 0;

    if (!eglQueryDevicesEXT(G_N_ELEMENTS(devices), devices, &device_count)) {
        error_setg(errp, "EGL device enumeration failed");
        jrg_gl_platform_free(p);
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
        jrg_gl_platform_free(p);
        return NULL;
    }
    if (!epoxy_has_egl_extension(p->display, "EGL_EXT_image_dma_buf_import") ||
        !epoxy_has_egl_extension(p->display, "EGL_ANDROID_native_fence_sync")) {
        error_setg(errp, "EGL DMA-BUF import and native fences are required");
        jrg_gl_platform_free(p);
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
        jrg_gl_platform_free(p);
        return NULL;
    }
    p->root = eglCreateContext(p->display, p->config, EGL_NO_CONTEXT, NULL);
    if (p->root == EGL_NO_CONTEXT) {
        error_setg(errp, "EGL internal share group: 0x%x", eglGetError());
        jrg_gl_platform_free(p);
        return NULL;
    }
#endif
    return p;
}

void jrg_gl_platform_free(JrgGLPlatform *p)
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

void jrg_gl_clear_current(JrgGLPlatform *p)
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

JrgGLContext *jrg_gl_context_new(JrgGLPlatform *p, JrgGLContext *share,
                                Error **errp)
{
    JrgGLContext *c = g_new0(JrgGLContext, 1);

    if (p->next_context_serial == UINT64_MAX) {
        error_setg(errp, "Native context identity exhausted");
        g_free(c);
        return NULL;
    }
    c->serial = ++p->next_context_serial;
    c->platform = p;
    c->draw_buffer = c->read_buffer = GL_BACK;
    if (p->texture_count > JRG_GL_MAX_TEXTURES - 2) {
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
        jrg_gl_context_free(c);
        return NULL;
    }
#else
    c->render = eglCreateContext(p->display, p->config, p->root, NULL);
    if (c->render == EGL_NO_CONTEXT) {
        error_setg(errp, "EGL context: 0x%x", eglGetError());
        jrg_gl_context_free(c);
        return NULL;
    }
#endif
    c->textures = share ? share->textures : g_new0(JrgTextureNamespace, 1);
    if (!share) {
        c->textures->names = g_hash_table_new(g_direct_hash, g_direct_equal);
    }
    c->textures->refs++;
    c->default_texture = g_new0(JrgTexture, 1);
    c->default_texture->refs = 2; /* context ownership plus current binding */
    c->default_texture->target = GL_TEXTURE_2D;
    c->bound_texture = c->default_texture;
    c->default_texture_1d = g_new0(JrgTexture, 1);
    c->default_texture_1d->refs = 2;
    c->default_texture_1d->target = GL_TEXTURE_1D;
    c->bound_texture_1d = c->default_texture_1d;
    p->texture_count += 2;
    return c;
}

void jrg_gl_context_free(JrgGLContext *c)
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
        jrgDeleteFramebuffers(1, &c->framebuffer);
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
        jrgDeleteFramebuffers(1, &c->framebuffer);
    }
    eglMakeCurrent(c->platform->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                    previous == c->render ? EGL_NO_CONTEXT : previous);
    if (c->render != EGL_NO_CONTEXT) {
        eglDestroyContext(c->platform->display, c->render);
    }
#endif
    g_free(c);
}

JrgGLDrawable *jrg_gl_drawable_new(JrgGLPlatform *p, uint32_t width,
                                  uint32_t height, Error **errp)
{
    JrgGLDrawable *d = g_new0(JrgGLDrawable, 1);

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
    jrgGenRenderbuffers(1, &d->depth);
    jrgBindRenderbuffer(GL_RENDERBUFFER, d->depth);
    jrgRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    if (glGetError() != GL_NO_ERROR) {
        error_setg(errp, "Allocating internal drawable color/depth failed");
        jrg_gl_drawable_free(p, d);
        return NULL;
    }
    GLuint framebuffer;
    jrgGenFramebuffers(1, &framebuffer);
    jrgBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    jrgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, d->color, 0);
    jrgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                            GL_TEXTURE_2D, d->front, 0);
    jrgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    jrgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    if (jrgCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        jrgDeleteFramebuffers(1, &framebuffer);
        error_setg(errp, "Initializing drawable framebuffer failed");
        jrg_gl_drawable_free(p, d);
        return NULL;
    }
    glClearColor(0, 0, 0, 0);
    glClearDepth(1);
    glClearStencil(0);
    const GLenum buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, buffers);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    jrgBindFramebuffer(GL_FRAMEBUFFER, 0);
    jrgDeleteFramebuffers(1, &framebuffer);
    jrg_gl_flush_drawable(d);
    return d;
}

void jrg_gl_flush_drawable(JrgGLDrawable *d)
{
    if (d->last_write) {
        glDeleteSync(d->last_write);
    }
    d->last_write = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
}

void jrg_gl_drawable_free(JrgGLPlatform *p, JrgGLDrawable *d)
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
        jrgDeleteRenderbuffers(1, &d->depth);
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
    jrg_gl_clear_current(p);
    g_free(d);
}

static GLenum read_attachment(GLenum buffer)
{
    return buffer == GL_BACK || buffer == GL_BACK_LEFT ?
           GL_COLOR_ATTACHMENT0 : GL_COLOR_ATTACHMENT1;
}

static void select_buffers(JrgGLContext *c)
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

void jrg_gl_exchange(JrgGLContext *c, JrgGLDrawable *d)
{
    GLuint back = d->color;
    d->color = d->front;
    d->front = back;
    /*
     * Logical exchange changes names, never pixel storage. Other contexts
     * reattach the current pair when the worker next makes them current.
     */
    jrgBindFramebuffer(GL_FRAMEBUFFER, c->framebuffer);
    jrgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, d->color, 0);
    jrgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                            GL_TEXTURE_2D, d->front, 0);
    select_buffers(c);
}

bool jrg_gl_make_current(JrgGLContext *c, JrgGLDrawable *d, Error **errp)
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
        jrgGenFramebuffers(1, &c->framebuffer);
    }
    jrgBindFramebuffer(GL_FRAMEBUFFER, c->framebuffer);
    jrgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, d->color, 0);
    jrgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                            GL_TEXTURE_2D, d->front, 0);
    jrgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    jrgFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                               GL_RENDERBUFFER, d->depth);
    select_buffers(c);
    if (jrgCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
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

static void texture_unref(JrgGLPlatform *p, JrgTexture *texture)
{
    if (!texture || --texture->refs) {
        return;
    }
    if (p->read_texture == texture) {
        g_clear_pointer(&p->read_pixels, g_free);
        p->read_texture = NULL;
        p->read_bytes = 0;
    }
    if (texture->last_write) {
        glDeleteSync(texture->last_write);
    }
    if (texture->name && !texture->deleted) {
        glDeleteTextures(1, &texture->name);
    }
    for (unsigned level = 0; level <= JRG_GL_MAX_TEXTURE_LEVEL; level++) {
        p->texture_bytes -= texture->levels[level];
    }
    p->texture_count--;
    g_free(texture);
}

static void texture_namespace_unref(JrgGLPlatform *p, JrgTextureNamespace *ns)
{
    GHashTableIter iter;
    gpointer value;

    if (!ns || --ns->refs) {
        return;
    }
    g_hash_table_iter_init(&iter, ns->names);
    while (g_hash_table_iter_next(&iter, NULL, &value)) {
        texture_unref(p, value);
    }
    g_hash_table_destroy(ns->names);
    g_free(ns);
}

static void store_guest_error(JrgGLContext *c, GLenum error)
{
    if (error != GL_NO_ERROR) {
        unsigned bit = error - GL_INVALID_ENUM;
        c->guest_errors |= 1U << (bit < 8 ? bit : 2);
    }
}

static void remember_guest_error(JrgGLContext *c)
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

static void texture_wait(JrgGLContext *c, JrgTexture *texture)
{
    /* One context already orders its uploads and draws. A foreign context
     * needs one server dependency per write, not another wait on every draw.
     * Serials are never reused even when a context allocation address is. */
    if (texture->last_write && texture->writer_serial != c->serial &&
        texture->waiter_serial != c->serial) {
        glWaitSync(texture->last_write, 0, GL_TIMEOUT_IGNORED);
        texture->waiter_serial = c->serial;
    }
}

static void texture_written(JrgGLContext *c, JrgTexture *texture)
{
    texture->version++;
    texture->writer_serial = c->serial;
    texture->waiter_serial = 0;
    if (texture->last_write) {
        glDeleteSync(texture->last_write);
    }
    texture->last_write = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    /* The normal context switch / present flush submits this fence too. */
}

static JrgTexture *bound_texture(JrgGLContext *c, GLenum target)
{
    return target == GL_TEXTURE_1D ? c->bound_texture_1d : c->bound_texture;
}

static uint32_t bind_texture(JrgGLContext *c, GLenum target,
                             uint32_t guest_name)
{
    JrgTexture *texture, **binding;
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D) {
        return JRG_GL_ERROR_TEXTURE;
    }
    binding = target == GL_TEXTURE_1D ? &c->bound_texture_1d : &c->bound_texture;
    texture = guest_name ?
        g_hash_table_lookup(c->textures->names, GUINT_TO_POINTER(guest_name)) :
        (target == GL_TEXTURE_1D ? c->default_texture_1d : c->default_texture);
    if (texture && texture->target != target) {
        store_guest_error(c, GL_INVALID_OPERATION);
        return 0;
    }
    if (!texture) {
        if (c->platform->texture_count == JRG_GL_MAX_TEXTURES) {
            return JRG_GL_ERROR_LIMIT;
        }
        texture = g_new0(JrgTexture, 1);
        texture->guest_name = guest_name;
        texture->target = target;
        remember_guest_error(c);
        glGenTextures(1, &texture->name);
        if (!texture->name) {
            g_free(texture);
            return JRG_GL_ERROR_HOST;
        }
        texture->refs = 1;
        c->platform->texture_count++;
        g_hash_table_insert(c->textures->names, GUINT_TO_POINTER(guest_name), texture);
    }
    texture_wait(c, texture);
    glBindTexture(target, texture->name);
    if (*binding != texture) {
        texture->refs++;
        texture_unref(c->platform, *binding);
        *binding = texture;
    }
    return 0;
}

static void delete_textures(JrgGLContext *c, const uint8_t *data,
                              uint32_t count)
{
    for (unsigned i = 0; i < count; i++) {
        uint32_t name = ldl_le_p(data + i * 4);
        JrgTexture *texture = g_hash_table_lookup(c->textures->names,
                                                  GUINT_TO_POINTER(name));
        if (!texture) {
            continue;
        }
        glDeleteTextures(1, &texture->name);
        texture->deleted = true;
        g_hash_table_remove(c->textures->names, GUINT_TO_POINTER(name));
        if (c->bound_texture == texture) {
            c->default_texture->refs++;
            c->bound_texture = c->default_texture;
            texture_unref(c->platform, texture);
        }
        if (c->bound_texture_1d == texture) {
            c->default_texture_1d->refs++;
            c->bound_texture_1d = c->default_texture_1d;
            texture_unref(c->platform, texture);
        }
        texture_unref(c->platform, texture);
    }
}

static unsigned texture_components(uint32_t format)
{
    switch (format) {
    case GL_ALPHA:
    case GL_LUMINANCE:
        return 1;
    case GL_LUMINANCE_ALPHA:
        return 2;
    case GL_RGB:
    case GL_BGR:
        return 3;
    case GL_RGBA:
    case GL_BGRA:
        return 4;
    default:
        return 0;
    }
}

static bool texture_internal_format(uint32_t format)
{
    switch (format) {
    case 1:
    case 2:
    case 3:
    case 4:
    case GL_ALPHA:
    case GL_LUMINANCE:
    case GL_LUMINANCE_ALPHA:
    case GL_RGB:
    case GL_RGBA:
    case GL_R3_G3_B2:
    case GL_LUMINANCE4_ALPHA4:
    case GL_RGB4:
    case GL_RGB5:
    case GL_RGBA4:
    case GL_RGB5_A1:
    case GL_ALPHA8:
    case GL_LUMINANCE8:
    case GL_LUMINANCE8_ALPHA8:
    case GL_RGB8:
    case GL_RGBA8:
        return true;
    default:
        return false;
    }
}

static bool buffer_selection(uint32_t mode, bool draw)
{
    switch (mode) {
    case GL_FRONT: case GL_FRONT_LEFT: case GL_BACK: case GL_BACK_LEFT:
    case GL_LEFT:
        return true;
    case GL_NONE: case GL_FRONT_AND_BACK:
        return draw;
    default:
        /* No right-eye or auxiliary storage exists in this pixel format. */
        return false;
    }
}

static bool hint_target(uint32_t target)
{
    return target == GL_PERSPECTIVE_CORRECTION_HINT ||
           target == GL_POINT_SMOOTH_HINT || target == GL_LINE_SMOOTH_HINT ||
           target == GL_POLYGON_SMOOTH_HINT || target == GL_FOG_HINT;
}

uint32_t jrg_gl_call_validate(uint32_t fn, const uint8_t *args)
{
    bool image = fn == FEnum_glCopyTexImage2D;
    uint32_t a[8];

    if (fn == FEnum_glDrawBuffer || fn == FEnum_glReadBuffer) {
        return buffer_selection(ldl_le_p(args), fn == FEnum_glDrawBuffer) ?
               0 : JRG_GL_ERROR_UNSUPPORTED;
    }
    if (fn == FEnum_glHint) {
        uint32_t mode = ldl_le_p(args + 4);
        return hint_target(ldl_le_p(args)) &&
               (mode == GL_DONT_CARE || mode == GL_FASTEST ||
                mode == GL_NICEST) ? 0 : JRG_GL_ERROR_UNSUPPORTED;
    }
    if (!image && fn != FEnum_glCopyTexSubImage2D) {
        return 0;
    }
    for (unsigned i = 0; i < G_N_ELEMENTS(a); i++) {
        a[i] = ldl_le_p(args + i * 4);
    }
    unsigned wi = image ? 5 : 6;
    unsigned hi = image ? 6 : 7;
    if (a[0] != GL_TEXTURE_2D || a[1] > JRG_GL_MAX_TEXTURE_LEVEL ||
        a[wi] > JRG_GL_MAX_TEXTURE_DIMENSION ||
        a[hi] > JRG_GL_MAX_TEXTURE_DIMENSION ||
        (image && (a[7] || a[2] <= 4 || !texture_internal_format(a[2]) ||
                   a[wi] > (JRG_GL_MAX_TEXTURE_DIMENSION >> a[1]) ||
                   a[hi] > (JRG_GL_MAX_TEXTURE_DIMENSION >> a[1])))) {
        return JRG_GL_ERROR_TEXTURE;
    }
    return 0;
}

static uint32_t copy_texture(JrgGLContext *c, uint32_t fn, const uint8_t *args)
{
    JrgTexture *t = c->bound_texture;
    JrgGLDrawable *d = c->drawable;
    bool image = fn == FEnum_glCopyTexImage2D;
    uint32_t a[8];
    uint32_t error = jrg_gl_call_validate(fn, args);

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
        return JRG_GL_ERROR_TEXTURE;
    }
    if (image && c->platform->texture_bytes - t->levels[level] + allocation >
                 JRG_GL_MAX_TEXTURE_BYTES) {
        return JRG_GL_ERROR_LIMIT;
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
        return JRG_GL_ERROR_TEXTURE;
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

static unsigned index_bytes(uint32_t type)
{
    switch (type) {
    case GL_UNSIGNED_BYTE:
        return 1;
    case GL_UNSIGNED_SHORT:
        return 2;
    case GL_UNSIGNED_INT:
        return 4;
    default:
        return 0;
    }
}

static uint32_t array_index(const uint8_t *p, unsigned size)
{
    return size == 1 ? *p : size == 2 ? lduw_le_p(p) : ldl_le_p(p);
}

static uint32_t validate_arrays(uint32_t fn, const uint32_t *a,
                                const uint8_t *data, uint32_t bytes)
{
    bool elements = fn == FEnum_glDrawElements;
    uint32_t vertices = a[elements ? 3 : 2];
    uint32_t attributes = a[elements ? 4 : 3];
    uint32_t indices = elements ? a[1] : 0;
    unsigned index_size = elements ? index_bytes(a[2]) : 0;
    unsigned stride = JRG_GL_VERTEX_SIZE(attributes);
    uint64_t vertex_bytes = (uint64_t)vertices * stride;

    if (a[0] > GL_POLYGON || vertices > JRG_GL_MAX_VERTICES ||
        indices > JRG_GL_MAX_INDICES || (elements && !index_size) ||
        (!elements && a[1]) || (attributes & ~JRG_GL_ARRAY_MASK) ||
        !(attributes & JRG_GL_ARRAY_POSITION) ||
        vertex_bytes + (uint64_t)indices * index_size != bytes) {
        return JRG_GL_ERROR_BATCH;
    }
    for (unsigned i = 0; i < vertices; i++) {
        if (ldl_le_p(data + i * stride +
                      JRG_GL_VERTEX_RESERVED)) {
            return JRG_GL_ERROR_BATCH;
        }
        if ((attributes & JRG_GL_ARRAY_SECONDARY) &&
            ldl_le_p(data + i * stride + JRG_GL_VERTEX_SECONDARY_PAD)) {
            return JRG_GL_ERROR_BATCH;
        }
    }
    const uint8_t *index_data = data + vertex_bytes;
    for (unsigned i = 0; i < indices; i++) {
        if (array_index(index_data + i * index_size, index_size) >= vertices) {
            return JRG_GL_ERROR_BATCH;
        }
    }
    return 0;
}

static uint32_t draw_arrays(JrgGLContext *c, uint32_t fn, const uint32_t *a,
                             const uint8_t *data, uint32_t bytes)
{
    bool elements = fn == FEnum_glDrawElements;
    uint32_t vertices = a[elements ? 3 : 2];
    uint32_t attributes = a[elements ? 4 : 3];
    unsigned stride = JRG_GL_VERTEX_SIZE(attributes);
    const uint8_t *indices = data + vertices * stride;
    g_autofree uint8_t *native = NULL;

    if (!vertices || (elements && !a[1])) {
        return 0;
    }
    if ((c->bound_texture->undefined_levels && glIsEnabled(GL_TEXTURE_2D)) ||
        (c->bound_texture_1d->undefined_levels && glIsEnabled(GL_TEXTURE_1D))) {
        return JRG_GL_ERROR_TEXTURE;
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
    /* Preserve current attributes and leave no borrowed CPU pointer in GL. */
    GLfloat current_color[4], current_normal[3], current_texcoord[4];
    GLfloat current_secondary[4];
    glGetFloatv(GL_CURRENT_COLOR, current_color);
    glGetFloatv(GL_CURRENT_SECONDARY_COLOR, current_secondary);
    glGetFloatv(GL_CURRENT_NORMAL, current_normal);
    glGetFloatv(GL_CURRENT_TEXTURE_COORDS, current_texcoord);
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, stride,
                     data + JRG_GL_VERTEX_POSITION);
#define ARRAY(enabled, cap) do { \
    if (attributes & (enabled)) { \
        glEnableClientState(cap); \
    } else { \
        glDisableClientState(cap); \
    } \
} while (0)
    ARRAY(JRG_GL_ARRAY_COLOR, GL_COLOR_ARRAY);
    ARRAY(JRG_GL_ARRAY_SECONDARY, GL_SECONDARY_COLOR_ARRAY);
    ARRAY(JRG_GL_ARRAY_NORMAL, GL_NORMAL_ARRAY);
    ARRAY(JRG_GL_ARRAY_TEXCOORD, GL_TEXTURE_COORD_ARRAY);
#undef ARRAY
    glColorPointer(4, GL_FLOAT, stride,
                    data + JRG_GL_VERTEX_COLOR);
    glNormalPointer(GL_FLOAT, stride,
                     data + JRG_GL_VERTEX_NORMAL);
    glTexCoordPointer(4, GL_FLOAT, stride,
                       data + JRG_GL_VERTEX_TEXCOORD);
    if (attributes & JRG_GL_ARRAY_SECONDARY) {
        glSecondaryColorPointer(3, GL_FLOAT, stride, data + JRG_GL_VERTEX_SECONDARY);
    }
    if (elements) {
        glDrawElements(a[0], a[1], a[2], indices);
    } else {
        glDrawArrays(a[0], 0, vertices);
    }
    GLenum error = glGetError();
    glPopClientAttrib();
    glColor4fv(current_color);
    glSecondaryColor3fv(current_secondary);
    glNormal3fv(current_normal);
    glTexCoord4fv(current_texcoord);
    if (error != GL_NO_ERROR) {
        store_guest_error(c, error);
        return JRG_GL_ERROR_HOST;
    }
    return 0;
}

/* Initialize allocation-only images without exposing recycled texture bytes. */
static GLenum zero_texture(JrgTexture *t, uint32_t level, uint32_t w,
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
        jrgGenFramebuffers(1, &framebuffer);
        if (!framebuffer) {
            error = glGetError();
            return error ? error : GL_OUT_OF_MEMORY;
        }
        jrgBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        jrgFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                GL_TEXTURE_2D, t->name, level);
        if (jrgCheckFramebufferStatus(GL_FRAMEBUFFER) ==
            GL_FRAMEBUFFER_COMPLETE) {
            glDisable(GL_SCISSOR_TEST);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            cleared = true;
        }
        error = glGetError();
        jrgBindFramebuffer(GL_READ_FRAMEBUFFER, read_fb);
        jrgBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fb);
        glClearColor(color[0], color[1], color[2], color[3]);
        glColorMask(mask[0], mask[1], mask[2], mask[3]);
        if (scissor) {
            glEnable(GL_SCISSOR_TEST);
        }
        if (framebuffer) {
            jrgDeleteFramebuffers(1, &framebuffer);
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

uint32_t jrg_gl_data_validate(uint32_t fn, const uint8_t *args,
                               const uint8_t *data, uint32_t bytes)
{
    uint32_t a[8] = { 0 };
    uint32_t words = jrg_gl_function_words(fn) & ~JRG_GL_FUNCTION_INLINE_DATA;

    for (unsigned i = 0; i < words; i++) {
        a[i] = ldl_le_p(args + i * 4);
    }
    if (vector_function(fn)) {
        unsigned required = vector_bytes(fn, a);
        return required && bytes == required ? 0 : JRG_GL_ERROR_BATCH;
    }
    if (fn == FEnum_glDrawArrays || fn == FEnum_glDrawElements) {
        return validate_arrays(fn, a, data, bytes);
    }
    if (fn == FEnum_glDeleteTextures) {
        return a[0] <= JRG_GL_MAX_TEXTURES && bytes == a[0] * 4 ?
               0 : JRG_GL_ERROR_BATCH;
    }
    bool image = fn == FEnum_glTexImage2D || fn == FEnum_glTexImage1D;
    bool one = fn == FEnum_glTexImage1D || fn == FEnum_glTexSubImage1D;
    uint32_t w = a[image ? 3 : 4];
    uint32_t h = a[image ? 4 : 5];
    unsigned components = texture_components(a[6]);
    bool allocate_only = image && !bytes;
    if (a[0] != (one ? GL_TEXTURE_1D : GL_TEXTURE_2D) || (one && (h != 1 || (!image && a[3]))) ||
        a[1] > JRG_GL_MAX_TEXTURE_LEVEL ||
        !w || !h || w > JRG_GL_MAX_TEXTURE_DIMENSION ||
        h > JRG_GL_MAX_TEXTURE_DIMENSION || !components ||
        a[7] != GL_UNSIGNED_BYTE ||
        (!allocate_only && (uint64_t)w * h * components != bytes) ||
        (image &&
         (a[5] || !texture_internal_format(a[2]) ||
          w > (JRG_GL_MAX_TEXTURE_DIMENSION >> a[1]) ||
          h > (JRG_GL_MAX_TEXTURE_DIMENSION >> a[1])))) {
        return JRG_GL_ERROR_TEXTURE;
    }
    return 0;
}

uint32_t jrg_gl_data_call(JrgGLContext *c, uint32_t fn, const uint8_t *args,
                          const uint8_t *data, uint32_t bytes)
{
    uint32_t a[8] = { 0 };
    uint32_t words = jrg_gl_function_words(fn) & ~JRG_GL_FUNCTION_INLINE_DATA;
    GLint unpack_alignment = 4, unpack_row_length = 0;
    GLint unpack_skip_rows = 0, unpack_skip_pixels = 0, unpack_swap_bytes = 0;
    JrgTexture *texture = c->bound_texture;

    if (c->in_begin && fn != FEnum_glMaterialfv) {
        return JRG_GL_ERROR_CONTEXT;
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
                return JRG_GL_ERROR_TEXTURE;
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
    uint32_t level = a[1];
    bool image = fn == FEnum_glTexImage2D || fn == FEnum_glTexImage1D;
    bool one = fn == FEnum_glTexImage1D || fn == FEnum_glTexSubImage1D;
    uint32_t w = a[image ? 3 : 4];
    uint32_t h = a[image ? 4 : 5];
    uint64_t allocation = (uint64_t)w * h * 4;
    texture = bound_texture(c, a[0]);
    if (image) {
        if (c->platform->texture_bytes - texture->levels[level] + allocation >
            JRG_GL_MAX_TEXTURE_BYTES) {
            return JRG_GL_ERROR_LIMIT;
        }
    } else if (a[2] > texture->widths[level] ||
               a[3] > texture->heights[level] ||
               w > texture->widths[level] - a[2] ||
               h > texture->heights[level] - a[3]) {
        return JRG_GL_ERROR_TEXTURE;
    }
    texture_wait(c, texture);
    remember_guest_error(c);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_alignment);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &unpack_row_length);
    glGetIntegerv(GL_UNPACK_SKIP_ROWS, &unpack_skip_rows);
    glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &unpack_skip_pixels);
    glGetIntegerv(GL_UNPACK_SWAP_BYTES, &unpack_swap_bytes);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SWAP_BYTES, 0);
    if (one && image) {
        glTexImage1D(a[0], level, a[2], w, 0, a[6], a[7], bytes ? data : NULL);
    } else if (one) {
        glTexSubImage1D(a[0], level, a[2], w, a[6], a[7], data);
    } else if (image) {
        glTexImage2D(a[0], level, a[2], w, h, 0, a[6], a[7],
                     bytes ? data : NULL);
    } else {
        glTexSubImage2D(a[0], level, a[2], a[3], w, h, a[6], a[7], data);
    }
    GLenum error = glGetError();
    bool image_written = !error && image;
    if (!error && image && !bytes) {
        error = zero_texture(texture, level, w, h);
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, unpack_alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, unpack_row_length);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, unpack_skip_rows);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, unpack_skip_pixels);
    glPixelStorei(GL_UNPACK_SWAP_BYTES, unpack_swap_bytes);
    if (image_written) {
        c->platform->texture_bytes = c->platform->texture_bytes -
                                     texture->levels[level] + allocation;
        texture->levels[level] = allocation;
        texture->widths[level] = w;
        texture->heights[level] = h;
        /*
         * Even a failed zero clear allocated storage: account it, and forbid
         * sampling until a later complete upload initializes this level.
         */
        if (error) {
            texture->undefined_levels |= 1U << level;
        } else {
            texture->undefined_levels &= ~(1U << level);
        }
    }
    if (image_written || !error) {
        texture_written(c, texture);
    }
    if (error != GL_NO_ERROR) {
        store_guest_error(c, error);
        return JRG_GL_ERROR_TEXTURE;
    }
    if (!image && !a[2] && !a[3] &&
        w == texture->widths[level] && h == texture->heights[level]) {
        texture->undefined_levels &= ~(1U << level);
    }
    return 0;
}

bool jrg_gl_context_in_begin(JrgGLContext *c)
{
    return c->in_begin;
}

static bool query_cap(uint32_t pname)
{
    if ((pname >= GL_LIGHT0 && pname <= GL_LIGHT7) ||
        (pname >= GL_CLIP_PLANE0 && pname <= GL_CLIP_PLANE5)) {
        return true;
    }
    switch (pname) {
    case GL_BLEND:
    case GL_DEPTH_TEST:
    case GL_STENCIL_TEST:
    case GL_SCISSOR_TEST:
    case GL_TEXTURE_1D:
    case GL_TEXTURE_2D:
    case GL_CULL_FACE:
    case GL_LIGHTING:
    case GL_FOG:
    case GL_ALPHA_TEST:
    case GL_DITHER:
    case GL_NORMALIZE:
    case GL_POLYGON_OFFSET_FILL:
    case GL_POLYGON_OFFSET_LINE:
    case GL_POLYGON_OFFSET_POINT:
    case GL_POLYGON_SMOOTH:
    case GL_POLYGON_STIPPLE:
    case GL_LINE_SMOOTH:
    case GL_LINE_STIPPLE:
    case GL_POINT_SMOOTH:
    case GL_COLOR_MATERIAL:
    case GL_COLOR_SUM:
    case GL_TEXTURE_GEN_S:
    case GL_TEXTURE_GEN_T:
    case GL_TEXTURE_GEN_R:
    case GL_TEXTURE_GEN_Q:
        return true;
    default:
        return false;
    }
}

static unsigned query_state_count(uint32_t pname)
{
    if (query_cap(pname)) {
        return 1;
    }
    switch (pname) {
    case GL_MODELVIEW_MATRIX:
    case GL_PROJECTION_MATRIX:
    case GL_TEXTURE_MATRIX:
        return 16;
    case GL_CURRENT_COLOR:
    case GL_CURRENT_SECONDARY_COLOR:
    case GL_CURRENT_TEXTURE_COORDS:
    case GL_VIEWPORT:
    case GL_SCISSOR_BOX:
    case GL_COLOR_CLEAR_VALUE:
    case GL_COLOR_WRITEMASK:
    case GL_FOG_COLOR:
    case GL_LIGHT_MODEL_AMBIENT:
        return 4;
    case GL_CURRENT_NORMAL:
        return 3;
    case GL_DEPTH_RANGE:
    case GL_MAX_VIEWPORT_DIMS:
    case GL_POLYGON_MODE:
    case GL_POINT_SIZE_RANGE:
    case GL_ALIASED_POINT_SIZE_RANGE:
    case GL_ALIASED_LINE_WIDTH_RANGE:
    case GL_LINE_WIDTH_RANGE:
        return 2;
    case GL_PACK_SWAP_BYTES:
    case GL_PACK_LSB_FIRST:
    case GL_PACK_ROW_LENGTH:
    case GL_PACK_SKIP_ROWS:
    case GL_PACK_SKIP_PIXELS:
    case GL_PACK_ALIGNMENT:
    case GL_UNPACK_SWAP_BYTES:
    case GL_UNPACK_LSB_FIRST:
    case GL_UNPACK_ROW_LENGTH:
    case GL_UNPACK_SKIP_ROWS:
    case GL_UNPACK_SKIP_PIXELS:
    case GL_UNPACK_ALIGNMENT:
    case GL_TEXTURE_BINDING_1D:
    case GL_TEXTURE_BINDING_2D:
    case GL_ATTRIB_STACK_DEPTH:
    case GL_MAX_ATTRIB_STACK_DEPTH:
    case GL_MATRIX_MODE:
    case GL_MODELVIEW_STACK_DEPTH:
    case GL_PROJECTION_STACK_DEPTH:
    case GL_TEXTURE_STACK_DEPTH:
    case GL_MAX_MODELVIEW_STACK_DEPTH:
    case GL_MAX_PROJECTION_STACK_DEPTH:
    case GL_MAX_TEXTURE_STACK_DEPTH:
    case GL_MAX_TEXTURE_SIZE:
    case GL_BLEND_SRC:
    case GL_BLEND_DST:
    case GL_DEPTH_FUNC:
    case GL_DEPTH_WRITEMASK:
    case GL_DEPTH_CLEAR_VALUE:
    case GL_STENCIL_CLEAR_VALUE:
    case GL_STENCIL_FUNC:
    case GL_STENCIL_VALUE_MASK:
    case GL_STENCIL_REF:
    case GL_STENCIL_FAIL:
    case GL_STENCIL_PASS_DEPTH_FAIL:
    case GL_STENCIL_PASS_DEPTH_PASS:
    case GL_STENCIL_WRITEMASK:
    case GL_FOG_MODE:
    case GL_FOG_DENSITY:
    case GL_FOG_START:
    case GL_FOG_END:
    case GL_FOG_INDEX:
    case GL_LIGHT_MODEL_LOCAL_VIEWER:
    case GL_LIGHT_MODEL_TWO_SIDE:
    case GL_LIGHT_MODEL_COLOR_CONTROL:
    case GL_COLOR_MATERIAL_FACE:
    case GL_COLOR_MATERIAL_PARAMETER:
    case GL_MAX_LIGHTS:
    case GL_MAX_CLIP_PLANES:
    case GL_POLYGON_OFFSET_FACTOR:
    case GL_POLYGON_OFFSET_UNITS:
    case GL_POINT_SIZE:
    case GL_POINT_SIZE_GRANULARITY:
    case GL_LINE_WIDTH:
    case GL_LINE_WIDTH_GRANULARITY:
    case GL_LINE_STIPPLE_PATTERN:
    case GL_LINE_STIPPLE_REPEAT:
    case GL_ALPHA_TEST_FUNC:
    case GL_ALPHA_TEST_REF:
    case GL_SHADE_MODEL:
    case GL_FRONT_FACE:
    case GL_CULL_FACE_MODE:
    case GL_RED_BITS:
    case GL_GREEN_BITS:
    case GL_BLUE_BITS:
    case GL_ALPHA_BITS:
    case GL_DEPTH_BITS:
    case GL_STENCIL_BITS:
    case GL_RGBA_MODE:
    case GL_INDEX_MODE:
    case GL_DOUBLEBUFFER:
    case GL_DRAW_BUFFER:
    case GL_READ_BUFFER:
    case GL_PERSPECTIVE_CORRECTION_HINT:
    case GL_POINT_SMOOTH_HINT:
    case GL_LINE_SMOOTH_HINT:
    case GL_POLYGON_SMOOTH_HINT:
    case GL_FOG_HINT:
    case GL_STEREO:
    case GL_AUX_BUFFERS:
        return 1;
    default:
        return 0;
    }
}

static const char *query_string(uint32_t name)
{
    switch (name) {
    case GL_VENDOR:
        return "Juke";
    case GL_RENDERER:
        return "Juke retro GPU (native host OpenGL)";
    case GL_EXTENSIONS:
        return "";
    default:
        /* No complete OpenGL version is implemented or advertised yet. */
        return NULL;
    }
}

/* Read one bounded canonical tile (128 pixels for legacy requests). The first tile
 * takes the intentional GPU readback once; subsequent tiles reuse one cache
 * bounded by the device's maximum texture size. A write invalidates it by
 * version, and freeing an object clears the weak cache identity. */
static uint32_t texture_read(JrgGLContext *c, GLenum target, uint32_t level,
                             uint32_t first, uint32_t capacity, uint8_t *result)
{
    JrgGLPlatform *p = c->platform;
    JrgTexture *texture = bound_texture(c, target);
    uint64_t bytes = (uint64_t)texture->widths[level] * texture->heights[level] * 4;
    if (!bytes || (uint64_t)first * 4 >= bytes || texture->undefined_levels & (1U << level)) {
        return JRG_GL_ERROR_TEXTURE;
    }
    if (!first || p->read_texture != texture || p->read_version != texture->version ||
        p->read_level != level || p->read_bytes != bytes) {
        GLint alignment, row_length, skip_rows, skip_pixels, swap_bytes;
        g_clear_pointer(&p->read_pixels, g_free);
        p->read_texture = NULL;
        p->read_bytes = 0;
        if (bytes > (uint64_t)JRG_GL_MAX_TEXTURE_DIMENSION * JRG_GL_MAX_TEXTURE_DIMENSION * 4) {
            return JRG_GL_ERROR_LIMIT;
        }
        p->read_pixels = g_try_malloc0(bytes);
        if (!p->read_pixels) {
            return JRG_GL_ERROR_LIMIT;
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
            return JRG_GL_ERROR_HOST;
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

static unsigned query_shape(uint32_t fn, const uint8_t *args, uint32_t *type)
{
    uint32_t a = ldl_le_p(args), b = ldl_le_p(args + 4);
    uint32_t d = ldl_le_p(args + 8);
    unsigned count = 0;

    *type = JRG_GL_RESULT_INT;
    switch (fn) {
    case FEnum_glGetTexImage:
        return (a == GL_TEXTURE_1D || a == GL_TEXTURE_2D) &&
               (b & 0xffff) <= JRG_GL_MAX_TEXTURE_LEVEL &&
               (b >> JRG_GL_TEXTURE_READ_COUNT_SHIFT) <= JRG_GL_MAX_READBACK_BYTES / 4 &&
               d < JRG_GL_MAX_TEXTURE_DIMENSION * JRG_GL_MAX_TEXTURE_DIMENSION ?
               (b >> JRG_GL_TEXTURE_READ_COUNT_SHIFT ? b >> JRG_GL_TEXTURE_READ_COUNT_SHIFT : JRG_GL_MAX_RESULT_BYTES / 4) : 0;
    case FEnum_glReadPixels: {
        unsigned width = d & 0xffff, height = d >> 16;
        return a < JRG_GL_MAX_DIMENSION && b < JRG_GL_MAX_DIMENSION &&
               width && height && width <= JRG_GL_READ_PIXELS_MAX / height ?
               width * height : 0;
    }
    case FEnum_glGetError:
        return !a && !b && !d;
    case FEnum_glIsTexture:
        *type = JRG_GL_RESULT_BOOL;
        return !b && !d;
    case FEnum_glIsEnabled:
        *type = JRG_GL_RESULT_BOOL;
        return !b && !d && query_cap(a);
    case FEnum_glGetString: {
        const char *s = query_string(a);
        *type = JRG_GL_RESULT_STRING;
        return !b && !d && s ? strlen(s) + 1 : 0;
    }
    case FEnum_glGetBooleanv:
        *type = JRG_GL_RESULT_BOOL;
        return !b && !d ? query_state_count(a) : 0;
    case FEnum_glGetFloatv:
        *type = JRG_GL_RESULT_FLOAT;
        return !b && !d ? query_state_count(a) : 0;
    case FEnum_glGetDoublev:
        *type = JRG_GL_RESULT_DOUBLE;
        return !b && !d ? query_state_count(a) : 0;
    case FEnum_glGetIntegerv:
        return !b && !d ? query_state_count(a) : 0;
    case FEnum_glGetTexParameterfv:
    case FEnum_glGetTexParameteriv:
        count = !d ? texture_params(a, b) : 0;
        *type = fn == FEnum_glGetTexParameterfv ? JRG_GL_RESULT_FLOAT :
                                                JRG_GL_RESULT_INT;
        return count;
    case FEnum_glGetTexLevelParameterfv:
    case FEnum_glGetTexLevelParameteriv:
        if ((a == GL_TEXTURE_1D || a == GL_TEXTURE_2D) && b <= JRG_GL_MAX_TEXTURE_LEVEL) {
            switch (d) {
            case GL_TEXTURE_WIDTH:
            case GL_TEXTURE_HEIGHT:
            case GL_TEXTURE_INTERNAL_FORMAT:
            case GL_TEXTURE_BORDER:
            case GL_TEXTURE_RED_SIZE:
            case GL_TEXTURE_GREEN_SIZE:
            case GL_TEXTURE_BLUE_SIZE:
            case GL_TEXTURE_ALPHA_SIZE:
            case GL_TEXTURE_LUMINANCE_SIZE:
            case GL_TEXTURE_INTENSITY_SIZE:
                count = 1;
                break;
            }
        }
        *type = fn == FEnum_glGetTexLevelParameterfv ? JRG_GL_RESULT_FLOAT :
                                                     JRG_GL_RESULT_INT;
        return count;
    case FEnum_glGetTexEnvfv:
    case FEnum_glGetTexEnviv:
        count = !d ? texture_env_params(a, b) : 0;
        *type = fn == FEnum_glGetTexEnvfv ? JRG_GL_RESULT_FLOAT :
                                          JRG_GL_RESULT_INT;
        return count;
    case FEnum_glGetLightfv:
    case FEnum_glGetLightiv:
        *type = fn == FEnum_glGetLightfv ? JRG_GL_RESULT_FLOAT :
                                          JRG_GL_RESULT_INT;
        return !d ? light_params(a, b) : 0;
    case FEnum_glGetMaterialfv:
    case FEnum_glGetMaterialiv:
        *type = fn == FEnum_glGetMaterialfv ? JRG_GL_RESULT_FLOAT :
                                             JRG_GL_RESULT_INT;
        return !d && a != GL_FRONT_AND_BACK && b != GL_AMBIENT_AND_DIFFUSE ?
               material_params(a, b) : 0;
    case FEnum_glGetTexGenfv:
    case FEnum_glGetTexGeniv:
    case FEnum_glGetTexGendv:
        *type = fn == FEnum_glGetTexGendv ? JRG_GL_RESULT_DOUBLE :
                fn == FEnum_glGetTexGenfv ? JRG_GL_RESULT_FLOAT :
                                           JRG_GL_RESULT_INT;
        return !d ? texgen_params(a, b) : 0;
    case FEnum_glGetClipPlane:
        *type = JRG_GL_RESULT_DOUBLE;
        return !b && !d && a >= GL_CLIP_PLANE0 && a <= GL_CLIP_PLANE5 ? 4 : 0;
    default:
        return 0;
    }
}

uint32_t jrg_gl_query_result_bytes(uint32_t fn, const uint8_t *args)
{
    uint32_t type;
    unsigned count = query_shape(fn, args, &type);
    unsigned size = type == JRG_GL_RESULT_DOUBLE ? 8 :
                    type == JRG_GL_RESULT_INT || type == JRG_GL_RESULT_FLOAT ?
                    4 : 1;
    return count * size;
}

uint32_t jrg_gl_query_validate(uint32_t fn, const uint8_t *args)
{
    uint32_t words = jrg_gl_function_words(fn);
    if (words == UINT32_MAX ||
        (words & JRG_GL_FUNCTION_KIND_MASK) != JRG_GL_FUNCTION_QUERY) {
        return JRG_GL_ERROR_UNSUPPORTED;
    }
    words &= ~JRG_GL_FUNCTION_KIND_MASK;
    for (unsigned i = words; i < 3; i++) {
        if (ldl_le_p(args + i * 4)) {
            return JRG_GL_ERROR_BATCH;
        }
    }
    return jrg_gl_query_result_bytes(fn, args) ? 0 : JRG_GL_ERROR_UNSUPPORTED;
}

uint32_t jrg_gl_query(JrgGLContext *c, uint32_t fn, const uint8_t *args,
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
        return JRG_GL_ERROR_CONTEXT;
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
            return JRG_GL_ERROR_DRAWABLE;
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
            return JRG_GL_ERROR_HOST;
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
        booleans[0] = a && g_hash_table_contains(c->textures->names,
                                                GUINT_TO_POINTER(a));
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
            integers[0] = JRG_ATTRIB_STACK;
            logical = true;
            break;
        case GL_MAX_TEXTURE_SIZE:
            integers[0] = JRG_GL_MAX_TEXTURE_DIMENSION;
            logical = true;
            break;
        case GL_MAX_VIEWPORT_DIMS:
            integers[0] = integers[1] = JRG_GL_MAX_DIMENSION;
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
            case JRG_GL_RESULT_INT:
                glGetIntegerv(a, integers);
                break;
            case JRG_GL_RESULT_FLOAT:
                glGetFloatv(a, floats);
                break;
            case JRG_GL_RESULT_DOUBLE:
                glGetDoublev(a, doubles);
                break;
            case JRG_GL_RESULT_BOOL:
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
        case JRG_GL_RESULT_BOOL:
            result[i] = !!booleans[i];
            break;
        case JRG_GL_RESULT_INT:
            stl_le_p(result + i * 4, integers[i]);
            break;
        case JRG_GL_RESULT_FLOAT: {
            uint32_t bits;
            memcpy(&bits, &floats[i], 4);
            stl_le_p(result + i * 4, bits);
            break;
        }
        case JRG_GL_RESULT_DOUBLE: {
            uint64_t bits;
            memcpy(&bits, &doubles[i], 8);
            stq_le_p(result + i * 8, bits);
            break;
        }
        }
    }
    *bytes = jrg_gl_query_result_bytes(fn, args);
    return 0;
}

uint32_t jrg_gl_function_words(uint32_t fn)
{
    switch (fn) {
    case FEnum_glGetError:
        return JRG_GL_FUNCTION_QUERY;
    case FEnum_glGetBooleanv:
    case FEnum_glGetIntegerv:
    case FEnum_glGetFloatv:
    case FEnum_glGetDoublev:
    case FEnum_glGetString:
    case FEnum_glIsEnabled:
    case FEnum_glIsTexture:
    case FEnum_glGetClipPlane:
        return JRG_GL_FUNCTION_QUERY | 1;
    case FEnum_glGetTexParameterfv:
    case FEnum_glGetTexParameteriv:
    case FEnum_glGetTexEnvfv:
    case FEnum_glGetTexEnviv:
    case FEnum_glGetLightfv:
    case FEnum_glGetLightiv:
    case FEnum_glGetMaterialfv:
    case FEnum_glGetMaterialiv:
    case FEnum_glGetTexGenfv:
    case FEnum_glGetTexGeniv:
    case FEnum_glGetTexGendv:
        return JRG_GL_FUNCTION_QUERY | 2;
    case FEnum_glGetTexLevelParameterfv:
    case FEnum_glGetTexLevelParameteriv:
    case FEnum_glGetTexImage:
    case FEnum_glReadPixels:
        return JRG_GL_FUNCTION_QUERY | 3;
    case FEnum_glDrawArrays:
        return JRG_GL_FUNCTION_INLINE_DATA | 4;
    case FEnum_glDrawElements:
        return JRG_GL_FUNCTION_INLINE_DATA | 5;
    case FEnum_glTexImage1D:
    case FEnum_glTexSubImage1D:
    case FEnum_glTexImage2D:
    case FEnum_glTexSubImage2D:
        return JRG_GL_FUNCTION_INLINE_DATA | 8;
    case FEnum_glDeleteTextures:
    case FEnum_glFogfv:
    case FEnum_glLightModelfv:
    case FEnum_glClipPlane:
        return JRG_GL_FUNCTION_INLINE_DATA | 1;
    case FEnum_glLightfv:
    case FEnum_glMaterialfv:
    case FEnum_glTexGenfv:
    case FEnum_glTexGendv:
    case FEnum_glTexParameterfv:
    case FEnum_glTexParameteriv:
    case FEnum_glTexEnvfv:
    case FEnum_glTexEnviv:
        return JRG_GL_FUNCTION_INLINE_DATA | 2;
    case FEnum_glBindTexture:
        return 2;
    case FEnum_glTexParameteri:
    case FEnum_glTexParameterf:
    case FEnum_glTexEnvi:
    case FEnum_glTexEnvf:
        return 3;
    case FEnum_glEnd:
    case FEnum_glFlush:
    case FEnum_glFinish:
    case FEnum_glLoadIdentity:
    case FEnum_glPushMatrix:
    case FEnum_glPopAttrib:
    case FEnum_glPopMatrix:
        return 0;
    case FEnum_glBegin:
    case FEnum_glPushAttrib:
    case FEnum_glClear:
    case FEnum_glEnable:
    case FEnum_glDisable:
    case FEnum_glMatrixMode:
    case FEnum_glDepthFunc:
    case FEnum_glDepthMask:
    case FEnum_glClearStencil:
    case FEnum_glStencilMask:
    case FEnum_glCullFace:
    case FEnum_glFrontFace:
    case FEnum_glLineWidth:
    case FEnum_glPointSize:
    case FEnum_glShadeModel:
    case FEnum_glDrawBuffer:
    case FEnum_glReadBuffer:
        return 1;
    case FEnum_glBlendFunc:
    case FEnum_glVertex2f:
    case FEnum_glTexCoord2f:
    case FEnum_glClearDepth:
    case FEnum_glAlphaFunc:
    case FEnum_glPolygonMode:
    case FEnum_glPolygonOffset:
    case FEnum_glLineStipple:
    case FEnum_glColorMaterial:
    case FEnum_glFogf:
    case FEnum_glLightModelf:
    case FEnum_glHint:
        return 2;
    case FEnum_glVertex3f:
    case FEnum_glColor3f:
    case FEnum_glSecondaryColor3f:
    case FEnum_glTranslatef:
    case FEnum_glScalef:
    case FEnum_glStencilFunc:
    case FEnum_glStencilOp:
    case FEnum_glNormal3f:
    case FEnum_glLightf:
    case FEnum_glMaterialf:
    case FEnum_glTexGenf:
        return 3;
    case FEnum_glClearColor:
    case FEnum_glColor4f:
    case FEnum_glViewport:
    case FEnum_glScissor:
    case FEnum_glRotatef:
    case FEnum_glDepthRange:
    case FEnum_glColorMask:
    case FEnum_glVertex4f:
    case FEnum_glTexCoord4f:
        return 4;
    case FEnum_glCopyTexImage2D:
    case FEnum_glCopyTexSubImage2D:
        return 8;
    case FEnum_glOrtho:
    case FEnum_glFrustum:
        return 12;
    case FEnum_glLoadMatrixf:
    case FEnum_glMultMatrixf:
        return 16;
    case FEnum_glLoadMatrixd:
    case FEnum_glMultMatrixd:
        return 32;
    default:
        return UINT32_MAX;
    }
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
static uint32_t attrib_push(JrgGLContext *c, GLbitfield mask)
{
    JrgAttrib *saved;
    if (mask & ~GL_ALL_ATTRIB_BITS) {
        store_guest_error(c, GL_INVALID_VALUE);
        return 0;
    }
    if (c->attrib_depth == JRG_ATTRIB_STACK) {
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
    *saved = (JrgAttrib){ .mask = mask, .draw_buffer = c->draw_buffer,
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

static uint32_t attrib_pop(JrgGLContext *c)
{
    JrgAttrib *saved;
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

uint32_t jrg_gl_call(JrgGLContext *c, uint32_t fn, const uint8_t *args)
{
    GLfloat matrix[16];
    GLdouble wide_matrix[16];
    int i;

    if (c->in_begin && fn != FEnum_glEnd && fn != FEnum_glVertex2f &&
        fn != FEnum_glVertex3f && fn != FEnum_glVertex4f &&
        fn != FEnum_glTexCoord4f && fn != FEnum_glColor3f &&
        fn != FEnum_glColor4f && fn != FEnum_glSecondaryColor3f &&
        fn != FEnum_glTexCoord2f &&
        fn != FEnum_glNormal3f && fn != FEnum_glMaterialf) {
        return JRG_GL_ERROR_CONTEXT;
    }
    switch (fn) {
    case FEnum_glPushAttrib:
        return attrib_push(c, U(0));
    case FEnum_glPopAttrib:
        return attrib_pop(c);
    case FEnum_glDrawBuffer:
    case FEnum_glReadBuffer:
        if (!buffer_selection(U(0), fn == FEnum_glDrawBuffer)) {
            return JRG_GL_ERROR_UNSUPPORTED;
        }
        if (fn == FEnum_glDrawBuffer) {
            c->draw_buffer = U(0);
        } else {
            c->read_buffer = U(0);
        }
        select_buffers(c);
        break;
    case FEnum_glHint:
        if (jrg_gl_call_validate(fn, args)) {
            return JRG_GL_ERROR_UNSUPPORTED;
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
            return JRG_GL_ERROR_TEXTURE;
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
            return JRG_GL_ERROR_TEXTURE;
        }
        if (fn == FEnum_glTexEnvi) {
            glTexEnvi(U(0), U(1), U(2));
        } else {
            glTexEnvf(U(0), U(1), F(2));
        }
        break;
    case FEnum_glBegin:
        if (U(0) > GL_POLYGON) {
            return JRG_GL_ERROR_CONTEXT;
        }
        if ((c->bound_texture->undefined_levels && glIsEnabled(GL_TEXTURE_2D)) ||
        (c->bound_texture_1d->undefined_levels && glIsEnabled(GL_TEXTURE_1D))) {
            return JRG_GL_ERROR_TEXTURE;
        }
        texture_wait(c, c->bound_texture);
        texture_wait(c, c->bound_texture_1d);
        c->in_begin = true;
        glBegin(U(0));
        break;
    case FEnum_glEnd:
        if (!c->in_begin) {
            return JRG_GL_ERROR_CONTEXT;
        }
        glEnd();
        c->in_begin = false;
        break;
    case FEnum_glClear:
        glClear(U(0));
        break;
    case FEnum_glClearColor:
        glClearColor(F(0), F(1), F(2), F(3));
        break;
    case FEnum_glClearDepth:
        glClearDepth(D(0));
        break;
    case FEnum_glViewport:
        glViewport(U(0), U(1), U(2), U(3));
        break;
    case FEnum_glScissor:
        glScissor(U(0), U(1), U(2), U(3));
        break;
    case FEnum_glFlush:
        glFlush();
        break;
    case FEnum_glFinish:
        glFinish();
        break;
    case FEnum_glEnable:
        glEnable(U(0));
        break;
    case FEnum_glDisable:
        glDisable(U(0));
        break;
    case FEnum_glMatrixMode:
        glMatrixMode(U(0));
        break;
    case FEnum_glLoadIdentity:
        glLoadIdentity();
        break;
    case FEnum_glPushMatrix:
        glPushMatrix();
        break;
    case FEnum_glPopMatrix:
        glPopMatrix();
        break;
    case FEnum_glDepthFunc:
        glDepthFunc(U(0));
        break;
    case FEnum_glDepthMask:
        glDepthMask(!!U(0));
        break;
    case FEnum_glDepthRange:
        glDepthRange(D(0), D(2));
        break;
    case FEnum_glColorMask:
        glColorMask(!!U(0), !!U(1), !!U(2), !!U(3));
        break;
    case FEnum_glAlphaFunc:
        glAlphaFunc(U(0), F(1));
        break;
    case FEnum_glClearStencil:
        glClearStencil(U(0));
        break;
    case FEnum_glStencilMask:
        glStencilMask(U(0));
        break;
    case FEnum_glStencilFunc:
        glStencilFunc(U(0), U(1), U(2));
        break;
    case FEnum_glStencilOp:
        glStencilOp(U(0), U(1), U(2));
        break;
    case FEnum_glCullFace:
        glCullFace(U(0));
        break;
    case FEnum_glFrontFace:
        glFrontFace(U(0));
        break;
    case FEnum_glPolygonMode:
        glPolygonMode(U(0), U(1));
        break;
    case FEnum_glPolygonOffset:
        glPolygonOffset(F(0), F(1));
        break;
    case FEnum_glLineWidth:
        glLineWidth(F(0));
        break;
    case FEnum_glLineStipple:
        glLineStipple(U(0), U(1));
        break;
    case FEnum_glPointSize:
        glPointSize(F(0));
        break;
    case FEnum_glShadeModel:
        glShadeModel(U(0));
        break;
    case FEnum_glNormal3f:
        glNormal3f(F(0), F(1), F(2));
        break;
    case FEnum_glColorMaterial:
        glColorMaterial(U(0), U(1));
        break;
    case FEnum_glFogf:
        glFogf(U(0), F(1));
        break;
    case FEnum_glLightModelf:
        glLightModelf(U(0), F(1));
        break;
    case FEnum_glLightf:
        glLightf(U(0), U(1), F(2));
        break;
    case FEnum_glMaterialf:
        glMaterialf(U(0), U(1), F(2));
        break;
    case FEnum_glTexGenf:
        glTexGenf(U(0), U(1), F(2));
        break;
    case FEnum_glBlendFunc:
        glBlendFunc(U(0), U(1));
        break;
    case FEnum_glVertex2f:
        glVertex2f(F(0), F(1));
        break;
    case FEnum_glTexCoord2f:
        glTexCoord2f(F(0), F(1));
        break;
    case FEnum_glTexCoord4f:
        glTexCoord4f(F(0), F(1), F(2), F(3));
        break;
    case FEnum_glVertex3f:
        glVertex3f(F(0), F(1), F(2));
        break;
    case FEnum_glVertex4f:
        glVertex4f(F(0), F(1), F(2), F(3));
        break;
    case FEnum_glSecondaryColor3f:
        glSecondaryColor3f(F(0), F(1), F(2));
        break;
    case FEnum_glColor3f:
        glColor3f(F(0), F(1), F(2));
        break;
    case FEnum_glColor4f:
        glColor4f(F(0), F(1), F(2), F(3));
        break;
    case FEnum_glTranslatef:
        glTranslatef(F(0), F(1), F(2));
        break;
    case FEnum_glScalef:
        glScalef(F(0), F(1), F(2));
        break;
    case FEnum_glRotatef:
        glRotatef(F(0), F(1), F(2), F(3));
        break;
    case FEnum_glOrtho:
        glOrtho(D(0), D(2), D(4), D(6), D(8), D(10));
        break;
    case FEnum_glFrustum:
        glFrustum(D(0), D(2), D(4), D(6), D(8), D(10));
        break;
    case FEnum_glLoadMatrixd:
    case FEnum_glMultMatrixd:
        for (i = 0; i < 16; i++) {
            wide_matrix[i] = D(i * 2);
        }
        if (fn == FEnum_glLoadMatrixd) {
            glLoadMatrixd(wide_matrix);
        } else {
            glMultMatrixd(wide_matrix);
        }
        break;
    case FEnum_glLoadMatrixf:
    case FEnum_glMultMatrixf:
        for (i = 0; i < 16; i++) {
            matrix[i] = F(i);
        }
        if (fn == FEnum_glLoadMatrixf) {
            glLoadMatrixf(matrix);
        } else {
            glMultMatrixf(matrix);
        }
        break;
    }
    return 0;
}
#undef U
#undef F
#undef D

JrgGLImage *jrg_gl_image_new(JrgGLPlatform *p, uint32_t width, uint32_t height,
                            Error **errp)
{
    JrgGLImage *image = g_new0(JrgGLImage, 1);

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
        jrg_gl_image_free(p, image);
        return NULL;
    }
    image->fd = gbm_bo_get_fd(image->bo);
    image->stride = gbm_bo_get_stride(image->bo);
    image->offset = gbm_bo_get_offset(image->bo, 0);
    image->modifier = gbm_bo_get_modifier(image->bo);
    if (image->fd < 0 || image->modifier != DRM_FORMAT_MOD_LINEAR) {
        error_setg(errp, "Exporting linear DMA-BUF failed");
        jrg_gl_image_free(p, image);
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
        jrg_gl_image_free(p, image);
        return NULL;
    }
#endif
    return image;
}

void jrg_gl_image_free(JrgGLPlatform *p, JrgGLImage *image)
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
        jrg_gl_context_free(image->context);
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

bool jrg_gl_export(JrgGLContext *c, JrgGLDrawable *d, JrgGLImage *image,
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
        jrg_gl_exchange(c, d);
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
    jrgGenRenderbuffers(1, &attachment);
    jrgBindRenderbuffer(GL_RENDERBUFFER, attachment);
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
    jrgGenFramebuffers(1, &framebuffer);
    jrgBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
#ifdef CONFIG_DARWIN
    jrgFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            target, attachment, 0);
#else
    jrgFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_RENDERBUFFER, attachment);
#endif
    if (jrgCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_COMPLETE) {
        error_setg(errp, "Export framebuffer is incomplete");
        goto out;
    }
    jrgBindFramebuffer(GL_READ_FRAMEBUFFER, c->framebuffer);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glDisable(GL_SCISSOR_TEST);
    /* Guest GL is bottom-left; every exported image is top-left. */
    jrgBlitFramebuffer(0, d->height, d->width, 0,
                       0, 0, d->width, d->height,
                       GL_COLOR_BUFFER_BIT, GL_NEAREST);
#ifdef CONFIG_DARWIN
    if (image->context) {
        jrg_gl_context_free(image->context);
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
    jrgBindFramebuffer(GL_READ_FRAMEBUFFER, read_fb);
    glReadBuffer(read_buffer);
    jrgBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fb);
    if (scissor) {
        glEnable(GL_SCISSOR_TEST);
    }
#ifdef CONFIG_DARWIN
    glBindTexture(target, attachment_binding);
#else
    jrgBindRenderbuffer(GL_RENDERBUFFER, attachment_binding);
#endif
    if (framebuffer) {
        jrgDeleteFramebuffers(1, &framebuffer);
    }
    if (attachment) {
#ifdef CONFIG_DARWIN
        glDeleteTextures(1, &attachment);
#else
        jrgDeleteRenderbuffers(1, &attachment);
#endif
    }
    return ok;
}

bool jrg_gl_image_ready(JrgGLImage *image, Error **errp)
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

void jrg_gl_image_metadata(JrgGLImage *image, uint32_t *stride,
                            uint32_t *offset, uint64_t *modifier)
{
    *stride = image->stride;
    *offset = image->offset;
    *modifier = image->modifier;
}

uint32_t jrg_gl_image_port(JrgGLImage *image)
{
#ifdef CONFIG_DARWIN
    return IOSurfaceCreateMachPort(image->surface);
#else
    return 0;
#endif
}

int jrg_gl_image_fd(JrgGLImage *image)
{
#ifdef CONFIG_DARWIN
    return -1;
#else
    return image->fd;
#endif
}

int jrg_gl_image_fence_fd(JrgGLImage *image)
{
#ifdef CONFIG_DARWIN
    return -1;
#else
    return image->fence_fd;
#endif
}
