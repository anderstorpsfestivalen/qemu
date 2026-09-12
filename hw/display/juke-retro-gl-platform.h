/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef JUKE_RETRO_GL_PLATFORM_H
#define JUKE_RETRO_GL_PLATFORM_H

typedef struct JrgGLPlatform JrgGLPlatform;
typedef struct JrgGLContext JrgGLContext;
typedef struct JrgGLDrawable JrgGLDrawable;
typedef struct JrgGLImage JrgGLImage;

JrgGLPlatform *jrg_gl_platform_new(const char *render_node, Error **errp);
void jrg_gl_platform_free(JrgGLPlatform *p);
void jrg_gl_clear_current(JrgGLPlatform *p);
JrgGLContext *jrg_gl_context_new(JrgGLPlatform *p, JrgGLContext *share,
                                Error **errp);
void jrg_gl_context_free(JrgGLContext *c);
JrgGLDrawable *jrg_gl_drawable_new(JrgGLPlatform *p, uint32_t width,
                                  uint32_t height, Error **errp);
void jrg_gl_drawable_free(JrgGLPlatform *p, JrgGLDrawable *d);
void jrg_gl_flush_drawable(JrgGLDrawable *d);
void jrg_gl_exchange(JrgGLContext *c, JrgGLDrawable *d);
bool jrg_gl_make_current(JrgGLContext *c, JrgGLDrawable *d, Error **errp);
uint32_t jrg_gl_function_words(uint32_t function);
uint32_t jrg_gl_call_validate(uint32_t function, const uint8_t *args);
uint32_t jrg_gl_call(JrgGLContext *c, uint32_t function, const uint8_t *args);
uint32_t jrg_gl_data_validate(uint32_t function, const uint8_t *args,
                               const uint8_t *data, uint32_t bytes);
uint32_t jrg_gl_data_call(JrgGLContext *c, uint32_t function,
                          const uint8_t *args, const uint8_t *data,
                          uint32_t bytes);
uint32_t jrg_gl_query(JrgGLContext *c, uint32_t function, const uint8_t *args,
                       uint8_t *result, uint32_t *bytes, uint32_t *type);
bool jrg_gl_context_in_begin(JrgGLContext *c);

JrgGLImage *jrg_gl_image_new(JrgGLPlatform *p, uint32_t width, uint32_t height,
                            Error **errp);
void jrg_gl_image_free(JrgGLPlatform *p, JrgGLImage *image);
bool jrg_gl_export(JrgGLContext *c, JrgGLDrawable *d, JrgGLImage *image,
                    bool exchange, Error **errp);
bool jrg_gl_image_ready(JrgGLImage *image, Error **errp);
void jrg_gl_image_metadata(JrgGLImage *image, uint32_t *stride,
                            uint32_t *offset, uint64_t *modifier);
/* Mac returns an owned Mach send right; Linux returns a borrowed DMA-BUF fd. */
uint32_t jrg_gl_image_port(JrgGLImage *image);
int jrg_gl_image_fd(JrgGLImage *image);
int jrg_gl_image_fence_fd(JrgGLImage *image);

uint32_t jrg_gl_query_validate(uint32_t function, const uint8_t *args);
uint32_t jrg_gl_query_result_bytes(uint32_t function, const uint8_t *args);

#endif
