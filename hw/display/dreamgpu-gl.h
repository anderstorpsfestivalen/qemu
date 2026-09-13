/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef DREAMGPU_GL_ENGINE_H
#define DREAMGPU_GL_ENGINE_H

#include "standard-headers/dreamgpu/gl.h"
#include "dreamgpu-host.h"

typedef struct DgGLEngine DgGLEngine;
typedef void (*DgGLNotify)(void *opaque);
typedef DreamGpuFrame DgGLFrameRef;

typedef struct DgGLTransfer {
    uint8_t *pixels;
    uint32_t width, height, stride, offset, vram_stride, generation;
    bool writeback, return_cpu;
} DgGLTransfer;

typedef DreamGpuCompletion DgGLCompletion;

uint32_t dg_gl_validate(const uint8_t *data, size_t bytes, uint32_t generation,
                        uint32_t primary_width, uint32_t primary_height, uint32_t vram_size,
                        uint32_t *records);
DgGLEngine *dg_gl_engine_new(const char *socket_path, DgGLNotify notify, void *opaque);
void dg_gl_engine_free(DgGLEngine *engine);
bool dg_gl_engine_submit(DgGLEngine *engine, uint8_t *data, size_t bytes, uint32_t sequence,
                         uint32_t generation, uint32_t primary_width, uint32_t primary_height,
                         uint32_t records);
void dg_gl_engine_reset(DgGLEngine *engine, uint32_t generation, uint64_t cpu_epoch,
                        uint64_t cpu_generation);
bool dg_gl_engine_completion(DgGLEngine *engine, DgGLCompletion *completion);
uint32_t dg_gl_function_words(uint32_t function);
bool dg_gl_engine_transfer(DgGLEngine *engine, DgGLTransfer *transfer);
void dg_gl_engine_transfer_done(DgGLEngine *engine, uint32_t error, uint64_t cpu_epoch,
                                uint64_t cpu_generation);

#endif
