/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef DREAMGPU_REFRESH_H
#define DREAMGPU_REFRESH_H
#include <stdint.h>

/* Absolute microsecond deadlines retain the fractional part of a period.
 * Display listeners accept milliseconds, so round each remaining delay up,
 * never the period itself. Missed frames are skipped, not caught up in a loop. */
typedef struct DreamGpuRefresh {
    uint64_t next_us;
    uint32_t rate_millihz;
    uint32_t remainder;
} DreamGpuRefresh;

static inline void dreamgpu_refresh_set(DreamGpuRefresh *clock, uint32_t rate, uint64_t now_us) {
    clock->rate_millihz = rate;
    clock->next_us = now_us;
    clock->remainder = 0;
}

static inline int dreamgpu_refresh_delay(DreamGpuRefresh *clock, uint64_t now_us) {
    uint64_t deadline = clock->next_us + (clock->remainder != 0);
    if (deadline <= now_us) {
        uint64_t elapsed = (now_us - clock->next_us) * clock->rate_millihz - clock->remainder;
        uint64_t periods = elapsed / UINT64_C(1000000000) + 1;
        uint64_t ticks = periods * UINT64_C(1000000000) + clock->remainder;
        clock->next_us += ticks / clock->rate_millihz;
        clock->remainder = ticks % clock->rate_millihz;
    }
    deadline = clock->next_us + (clock->remainder != 0);
    return (int)((deadline - now_us + 999) / 1000);
}
#endif
