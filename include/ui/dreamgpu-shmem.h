/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef UI_DREAMGPU_SHMEM_H
#define UI_DREAMGPU_SHMEM_H

#include "standard-headers/dreamgpu/cursor.h"

typedef struct DreamGpuNativeCursor {
    uint32_t width, height, format;
    int32_t hot_x, hot_y, x, y;
    uint32_t flags;
    uint8_t pixels[DG_CURSOR_MAX_BYTES];
} DreamGpuNativeCursor;

/* BQL only. shape=false updates position/flags without copying pixel data. */
void dreamgpu_shmem_native_cursor(QemuConsole *con, const DreamGpuNativeCursor *cursor,
                              bool shape);

/* Called under BQL. Returns current immutable mapping epoch and the minimum
 * future CPU publication that can supersede an ordered ReturnCpu image. */
bool dreamgpu_shmem_cpu_anchor(QemuConsole *con, uint64_t *epoch,
                           uint64_t *generation);
#endif
