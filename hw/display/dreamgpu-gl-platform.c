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


struct DgGLPlatform {
    DreamGpuGlApi gl_api;
    uint64_t next_context_serial;
    uint64_t texture_bytes;
    uint32_t texture_count;
    /* One bounded CPU cache for genuine texture reads, never presentation. */
    DreamGpuReadCache read_cache;
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
    uint32_t initialized;
    uint32_t in_begin;
    DgGLDrawable *drawable;
    DreamGpuContextState state;
};

struct DgGLDrawable {
    DreamGpuNativeDrawable gpu;
    DgGLPlatform *platform;
};



static DreamGpuTextureMemory texture_memory(DgGLPlatform *p);
/* Pure command vocabulary and validation are owned by the Rust core. */

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
    DreamGpuTextureMemory memory = texture_memory(p);
    dreamgpu_read_cache_release(&memory, &p->read_cache);
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
    c->state.draw_buffer = c->state.read_buffer = GL_BACK;
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
    DreamGpuTextureMemory memory = texture_memory(p);
    if (dreamgpu_context_state_init(&memory, &c->state,
                                   share ? share->state.textures : NULL)) {
        error_setg(errp, "Allocating bounded context texture state failed");
        dg_gl_context_free(c);
        return NULL;
    }
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
    DreamGpuTextureMemory memory = texture_memory(c->platform);
    dreamgpu_context_state_release(&memory, &c->state);
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
    DreamGpuTextureMemory memory = texture_memory(c->platform);
    dreamgpu_context_state_release(&memory, &c->state);
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

    d->platform = p;
#ifdef CONFIG_DARWIN
    CGLSetCurrentContext(p->root);
#else
    if (!eglMakeCurrent(p->display, EGL_NO_SURFACE, EGL_NO_SURFACE, p->root)) {
        error_setg(errp, "EGL internal context: 0x%x", eglGetError());
        g_free(d);
        return NULL;
    }
#endif
    if (dreamgpu_drawable_init(&p->gl_api, &d->gpu, width, height)) {
        error_setg(errp, "Allocating internal drawable resources failed");
        dg_gl_drawable_free(p, d);
        return NULL;
    }
    return d;
}

void dg_gl_flush_drawable(DgGLDrawable *d)
{
    dreamgpu_drawable_flush(&d->platform->gl_api, &d->gpu);
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
    dreamgpu_drawable_release(&p->gl_api, &d->gpu);
    dg_gl_clear_current(p);
    g_free(d);
}

void dg_gl_exchange(DgGLContext *c, DgGLDrawable *d)
{
    dreamgpu_drawable_exchange(&c->platform->gl_api, &d->gpu,
                               c->framebuffer, &c->state);
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
    if (dreamgpu_drawable_bind(&c->platform->gl_api, &d->gpu,
                               &c->framebuffer, &c->initialized, &c->state)) {
        error_setg(errp, "Internal drawable framebuffer is incomplete");
        return false;
    }
    c->drawable = d;
    return true;
}

static void *host_texture_allocate(void *opaque, size_t bytes)
{
    /* Rust initializes each object or copies the entire immutable payload. */
    return g_try_malloc(bytes);
}

static void host_texture_free(void *opaque, void *object)
{
    g_free(object);
}

static void host_texture_forget_read(void *opaque, DreamGpuTexture *texture)
{
    DgGLPlatform *p = opaque;
    DreamGpuTextureMemory memory = texture_memory(p);
    dreamgpu_read_cache_forget(&memory, &p->read_cache, texture);
}

static DreamGpuTextureMemory texture_memory(DgGLPlatform *p)
{
    return (DreamGpuTextureMemory) {
        .api = &p->gl_api, .bytes = &p->texture_bytes, .count = &p->texture_count,
        .opaque = p, .allocate = host_texture_allocate, .free = host_texture_free,
        .forget_read = host_texture_forget_read,
    };
}

static DgTexture *bound_texture(DgGLContext *c, GLenum target)
{
    return target == GL_TEXTURE_1D ? c->state.bound_texture_1d : c->state.bound_texture;
}

static uint32_t copy_texture(DgGLContext *c, uint32_t fn, const uint8_t *args)
{
    return dreamgpu_texture_copy(&c->platform->gl_api, c->state.bound_texture,
                                  &c->platform->texture_bytes,
                                  &c->state.guest_errors, c->serial,
                                  c->drawable != NULL, fn, args);
}

uint32_t dg_gl_data_call(DgGLContext *c, uint32_t fn, const uint8_t *args,
                          const uint8_t *data, uint32_t bytes)
{
    DreamGpuTextureMemory memory = texture_memory(c->platform);
    return dreamgpu_gl_data(&memory, &c->state, c->in_begin, c->serial,
                             fn, args, data, bytes);
}

bool dg_gl_context_in_begin(DgGLContext *c)
{
    return c->in_begin;
}

static uint32_t host_texture_read(void *opaque, uint32_t target,
                                  uint32_t level, uint32_t first,
                                  uint32_t capacity, uint8_t *result)
{
    DgGLContext *c = opaque;
    DreamGpuTextureMemory memory = texture_memory(c->platform);
    return dreamgpu_texture_read(&memory, &c->platform->read_cache,
                                  bound_texture(c, target),
                                  &c->state.guest_errors, c->serial,
                                  target, level, first, capacity, result);
}

uint32_t dg_gl_query(DgGLContext *c, uint32_t fn, const uint8_t *args,
                     uint8_t *result, uint32_t capacity, uint32_t *bytes,
                     uint32_t *type)
{
    /* Stack snapshot is disjoint from callback-owned context/cache storage. */
    const DreamGpuQueryState state = {
        .in_begin = c->in_begin, .has_drawable = c->drawable != NULL,
        .width = c->drawable ? c->drawable->gpu.width : 0,
        .height = c->drawable ? c->drawable->gpu.height : 0,
        .draw_buffer = c->state.draw_buffer, .read_buffer = c->state.read_buffer,
        .binding_1d = c->state.bound_texture_1d->guest_name,
        .binding_2d = c->state.bound_texture->guest_name,
        .attrib_depth = c->state.attrib_depth, .textures = c->state.textures,
    };
    return dreamgpu_gl_query(&c->platform->gl_api, &state, &c->state.guest_errors,
                              fn, args, result, capacity, host_texture_read, c,
                              bytes, type);
}

/* Remaining resource-side operations; scalar GL execution is Rust-owned. */
static uint32_t host_copy_texture(void *opaque, uint32_t fn, const uint8_t *args)
{
    return copy_texture(opaque, fn, args);
}

static uint32_t scalar_resource(void *opaque, uint32_t fn, const uint8_t *args)
{
    DgGLContext *c = opaque;
    DreamGpuTextureMemory memory = texture_memory(c->platform);
    return dreamgpu_context_resource(&memory, &c->state, c->serial, fn, args,
                                     dg_gl_function_words(fn) * 4,
                                     host_copy_texture, c);
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

typedef struct DgImageCall {
    DgGLPlatform *platform;
    Error **errp;
} DgImageCall;

static uint32_t image_create(void *opaque, DreamGpuNativeImage *image)
{
    DgImageCall *call = opaque;
    DgGLPlatform *p G_GNUC_UNUSED = call->platform;
    Error **errp = call->errp;
    uint32_t width = image->width, height = image->height;
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
        return DG_GL_ERROR_HOST;
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
        return DG_GL_ERROR_HOST;
    }
    image->fd = gbm_bo_get_fd(image->bo);
    image->stride = gbm_bo_get_stride(image->bo);
    image->offset = gbm_bo_get_offset(image->bo, 0);
    image->modifier = gbm_bo_get_modifier(image->bo);
    if (image->fd < 0 || image->modifier != DRM_FORMAT_MOD_LINEAR) {
        error_setg(errp, "Exporting linear DMA-BUF failed");
        return DG_GL_ERROR_HOST;
    }
    const EGLint attrs[] = {
        EGL_WIDTH, width, EGL_HEIGHT, height,
        EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
        EGL_DMA_BUF_PLANE0_FD_EXT, image->fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, image->offset,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, image->stride,
        EGL_NONE,
    };
    image->native_image = eglCreateImageKHR(p->display, EGL_NO_CONTEXT,
                                    EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
    if (image->native_image == EGL_NO_IMAGE_KHR) {
        error_setg(errp, "Importing DMA-BUF to EGL failed: 0x%x",
                    eglGetError());
        return DG_GL_ERROR_HOST;
    }
#endif
    return 0;
}

static void image_destroy(void *opaque, DreamGpuNativeImage *image)
{
    DgImageCall *call = opaque;
    DgGLPlatform *p G_GNUC_UNUSED = call->platform;
#ifdef CONFIG_DARWIN
    if (image->context) {
        DgGLContext *context = image->context;
        CGLSetCurrentContext(context->completion);
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
    if (image->native_image != EGL_NO_IMAGE_KHR) {
        p->destroy_image(p->display, image->native_image);
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
}

DgGLImage *dg_gl_image_new(DgGLPlatform *p, uint32_t width, uint32_t height,
                            Error **errp)
{
    DreamGpuTextureMemory memory = texture_memory(p);
    DgImageCall call = { .platform = p, .errp = errp };
    uint32_t error;
    DgGLImage *image = dreamgpu_image_new(&memory, width, height, image_create,
                                          image_destroy, &call, &error);
    if (error && errp && !*errp) {
        error_setg(errp, "Allocating native image owner failed: %u", error);
    }
    return image;
}

void dg_gl_image_free(DgGLPlatform *p, DgGLImage *image)
{
    DreamGpuTextureMemory memory = texture_memory(p);
    DgImageCall call = { .platform = p };
    dreamgpu_image_free(&memory, image, image_destroy, &call);
}

typedef struct DgExportCall {
    DgGLContext *context;
    DgGLDrawable *drawable;
    DgGLImage *image;
    Error **errp;
} DgExportCall;

static uint32_t export_bind(void *opaque)
{
    DgExportCall *call = opaque;
#ifdef CONFIG_DARWIN
    CGLError err = CGLTexImageIOSurface2D(call->context->render,
                                         GL_TEXTURE_RECTANGLE_ARB, GL_RGBA8,
                                         call->drawable->gpu.width,
                                         call->drawable->gpu.height, GL_BGRA,
                                         GL_UNSIGNED_INT_8_8_8_8_REV,
                                         call->image->surface, 0);
    if (err != kCGLNoError) {
        error_setg(call->errp, "Binding IOSurface to GL: %s", CGLErrorString(err));
        return DG_GL_ERROR_HOST;
    }
#else
    glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, call->image->native_image);
#endif
    return 0;
}

static uint32_t export_fence(void *opaque)
{
    DgExportCall *call = opaque;
    DgGLContext *c = call->context;
    DgGLImage *image = call->image;
#ifdef CONFIG_DARWIN
    if (image->context) {
        dg_gl_context_free(image->context);
    }
    image->context = c;
    g_atomic_ref_count_inc(&c->refs);
    image->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!image->fence) {
        error_setg(call->errp, "Creating GL completion fence failed");
        return DG_GL_ERROR_HOST;
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
        error_setg(call->errp, "Exporting EGL completion fence failed");
        return DG_GL_ERROR_HOST;
    }
#endif
    return 0;
}

bool dg_gl_export(DgGLContext *c, DgGLDrawable *d, DgGLImage *image,
                    bool exchange, Error **errp)
{
    DgExportCall call = { .context = c, .drawable = d, .image = image,
                          .errp = errp };
    uint32_t rectangle = 0;
#ifdef CONFIG_DARWIN
    rectangle = 1;
#endif
    uint32_t error = dreamgpu_export(&c->platform->gl_api, &d->gpu, &c->state,
                                      c->framebuffer, exchange, rectangle,
                                      export_bind, export_fence, &call);
    if (error && errp && !*errp) {
        error_setg(errp, "Export framebuffer transaction failed: %u", error);
    }
    return error == 0;
}

#ifdef CONFIG_DARWIN
static int64_t export_clock(void *opaque)
{
    return g_get_monotonic_time();
}

static void export_sleep(void *opaque, uint64_t microseconds)
{
    g_usleep(microseconds);
}
#endif

bool dg_gl_image_ready(DgGLImage *image, Error **errp)
{
#ifdef CONFIG_DARWIN
    DgGLContext *context = image->context;
    CGLSetCurrentContext(context->completion);
    uint32_t result = dreamgpu_export_wait(&context->platform->gl_api,
                                            image->fence, export_clock,
                                            export_sleep, NULL);
    image->fence = NULL;
    CGLSetCurrentContext(NULL);
    if (result != GL_ALREADY_SIGNALED && result != GL_CONDITION_SATISFIED) {
        error_setg(errp, "Waiting for GL export completion failed: 0x%x", result);
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
