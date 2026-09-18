/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef DREAMGPU_TIMING_H
#define DREAMGPU_TIMING_H
/* Integer Hz are rational periods, never a rounded millisecond interval.
 * 20 blank lines describe this virtual scanout, not the physical monitor. */
typedef struct DgTimingSample {
    uint32_t line, height, phase, begin_ns, end_ns;
} DgTimingSample;
static inline bool dg_timing_rate_valid(uint32_t rate) {
    return rate == 60 || rate == 75 || rate == 85 || rate == 100 || rate == 120;
}
static inline DgTimingSample dg_timing_sample(uint64_t now, uint32_t rate, uint32_t height) {
    const uint64_t second = 1000000000;
    const uint64_t phase = (now % second * rate) % second;
    const uint64_t lines = height + DG_TIMING_BLANK_LINES;
    const uint64_t begin = (height * second + lines - 1) / lines;
    const uint64_t delta = begin > phase ? begin - phase : second - phase + begin;
    return (DgTimingSample){
        .line = phase * lines / second,
        .height = height,
        .phase = (rate << 16) | (phase >= begin),
        .begin_ns = (delta + rate - 1) / rate,
        .end_ns = (second - phase + rate - 1) / rate,
    };
}
#endif
