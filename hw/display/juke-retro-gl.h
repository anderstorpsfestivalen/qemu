/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef JUKE_RETRO_GL_ENGINE_H
#define JUKE_RETRO_GL_ENGINE_H

#include "standard-headers/juke/retro-gl.h"

typedef struct JrgGLEngine JrgGLEngine;
typedef void (*JrgGLNotify)(void *opaque);
typedef struct JrgGLFrameRef {
    uint32_t slot, client, drawable;
    uint64_t epoch, generation;
} JrgGLFrameRef;

typedef struct JrgGLTransfer {
    uint8_t *pixels;
    uint32_t width, height, stride, offset, vram_stride, generation;
    bool writeback, return_cpu;
} JrgGLTransfer;

typedef struct JrgGLCompletion {
    uint32_t sequence, generation, error;
    bool resources_live, reset;
    JrgGLFrameRef present;
    uint32_t result_bytes, result_type;
    uint8_t result[JRG_GL_MAX_RESULT_BYTES];
    uint8_t *bulk_result; /* Lazy bounded allocation; ownership follows completion. */
} JrgGLCompletion;

uint32_t jrg_gl_validate(const uint8_t *data, size_t bytes, uint32_t generation,
                         uint32_t primary_width, uint32_t primary_height,
                         uint32_t vram_size, uint32_t *records);
JrgGLEngine *jrg_gl_engine_new(const char *socket_path, JrgGLNotify notify,
                              void *opaque);
void jrg_gl_engine_free(JrgGLEngine *engine);
bool jrg_gl_engine_submit(JrgGLEngine *engine, uint8_t *data, size_t bytes,
                           uint32_t sequence, uint32_t generation,
                           uint32_t primary_width, uint32_t primary_height,
                           uint32_t records);
void jrg_gl_engine_reset(JrgGLEngine *engine, uint32_t generation,
                          uint64_t cpu_epoch, uint64_t cpu_generation);
bool jrg_gl_engine_completion(JrgGLEngine *engine, JrgGLCompletion *completion);
uint32_t jrg_gl_function_words(uint32_t function);
bool jrg_gl_engine_transfer(JrgGLEngine *engine, JrgGLTransfer *transfer);
void jrg_gl_engine_transfer_done(JrgGLEngine *engine, uint32_t error,
                                 uint64_t cpu_epoch, uint64_t cpu_generation);


#endif
