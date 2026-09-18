/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef I386_TCG_FPU_SINGLE_H
#define I386_TCG_FPU_SINGLE_H

#include "fpu-pc24.h"

/* Loads are exact at every x87 precision. Keep normal values and signed zero
 * out of the canonical conversion, without treating denormals or NaNs as
 * ordinary values. The caller retains architectural exception save/merge. */
static inline floatx80 fpu_load_single(float32 value, float_status *status)
{
    unsigned exponent = (value >> 23) & 255;

    if (exponent - 1 < 254) {
        return packFloatx80(value >> 31, exponent + 16383 - 127,
                            ((uint64_t)(value & 0x7fffff) | 0x800000) << 40);
    }
    if ((value & 0x7fffffff) == 0) {
        return packFloatx80(value >> 31, 0, 0);
    }
    return float32_to_floatx80(value, status);
}

/* This is a representability test, not a precision change: any rounding mode
 * gives the same bits for these exact normal binary32 values and signed zero.
 * Wider, inexact, denormal and exceptional values use the original converter. */
static inline float32 fpu_store_single(floatx80 value, float_status *status)
{
    uint32_t bits;

    if (fpu_pc24_operand(value, &bits)) {
        return bits;
    }
    if ((value.high & 0x7fff) == 0 && value.low == 0) {
        return (uint32_t)(value.high & 0x8000) << 16;
    }
    return floatx80_to_float32(value, status);
}

#endif
