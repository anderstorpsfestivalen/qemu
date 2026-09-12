/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef UI_JUKE_SHMEM_H
#define UI_JUKE_SHMEM_H

#include "standard-headers/juke/retro-cursor.h"

typedef struct JukeNativeCursor {
    uint32_t width, height, format;
    int32_t hot_x, hot_y, x, y;
    uint32_t flags;
    uint8_t pixels[JRG_CURSOR_MAX_BYTES];
} JukeNativeCursor;

/* BQL only. shape=false updates position/flags without copying pixel data. */
void juke_shmem_native_cursor(QemuConsole *con, const JukeNativeCursor *cursor,
                              bool shape);

/* Called under BQL. Returns current immutable mapping epoch and the minimum
 * future CPU publication that can supersede an ordered ReturnCpu image. */
bool juke_shmem_cpu_anchor(QemuConsole *con, uint64_t *epoch,
                           uint64_t *generation);
#endif
