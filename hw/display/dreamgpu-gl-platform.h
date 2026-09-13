/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef DREAMGPU_GL_PLATFORM_H
#define DREAMGPU_GL_PLATFORM_H

typedef struct DgGLPlatform DgGLPlatform;
typedef struct DgGLContext DgGLContext;
typedef struct DgGLDrawable DgGLDrawable;
typedef struct DreamGpuNativeImage DgGLImage;

DgGLPlatform *dg_gl_platform_new(const char *render_node, Error **errp);
void dg_gl_platform_free(DgGLPlatform *p);
void dg_gl_clear_current(DgGLPlatform *p);
DgGLContext *dg_gl_context_new(DgGLPlatform *p, DgGLContext *share, Error **errp);
void dg_gl_context_free(DgGLContext *c);
DgGLDrawable *dg_gl_drawable_new(DgGLPlatform *p, uint32_t width, uint32_t height, Error **errp);
void dg_gl_drawable_free(DgGLPlatform *p, DgGLDrawable *d);
void dg_gl_flush_drawable(DgGLDrawable *d);
void dg_gl_exchange(DgGLContext *c, DgGLDrawable *d);
bool dg_gl_make_current(DgGLContext *c, DgGLDrawable *d, Error **errp);
uint32_t dg_gl_function_words(uint32_t function);
uint32_t dg_gl_call_validate(uint32_t function, const uint8_t *args);
uint32_t dg_gl_call(DgGLContext *c, uint32_t function, const uint8_t *args);
uint32_t dg_gl_data_validate(uint32_t function, const uint8_t *args, const uint8_t *data,
                             uint32_t bytes);
uint32_t dg_gl_data_call(DgGLContext *c, uint32_t function, const uint8_t *args,
                         const uint8_t *data, uint32_t bytes);
uint32_t dg_gl_query(DgGLContext *c, uint32_t function, const uint8_t *args, uint8_t *result,
                     uint32_t capacity, uint32_t *bytes, uint32_t *type);
bool dg_gl_context_in_begin(DgGLContext *c);

DgGLImage *dg_gl_image_new(DgGLPlatform *p, uint32_t width, uint32_t height, Error **errp);
void dg_gl_image_free(DgGLPlatform *p, DgGLImage *image);
bool dg_gl_export(DgGLContext *c, DgGLDrawable *d, DgGLImage *image, bool exchange, Error **errp);
bool dg_gl_image_ready(DgGLImage *image, Error **errp);
void dg_gl_image_metadata(DgGLImage *image, uint32_t *stride, uint32_t *offset, uint64_t *modifier);
/* Mac returns an owned Mach send right; Linux returns a borrowed DMA-BUF fd. */
uint32_t dg_gl_image_port(DgGLImage *image);
int dg_gl_image_fd(DgGLImage *image);
int dg_gl_image_fence_fd(DgGLImage *image);

uint32_t dg_gl_query_validate(uint32_t function, const uint8_t *args);
uint32_t dg_gl_query_result_bytes(uint32_t function, const uint8_t *args);

#endif
