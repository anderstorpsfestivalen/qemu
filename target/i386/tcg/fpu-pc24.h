/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef I386_TCG_FPU_PC24_H
#define I386_TCG_FPU_PC24_H

#include <float.h>
#include "fpu/softfloat.h"
#include "fpu-exact.h"

typedef enum FpuPc24Op {
    FPU_PC24_ADD,
    FPU_PC24_SUB,
    FPU_PC24_MUL,
    FPU_PC24_DIV,
} FpuPc24Op;

/* Pack only exact, normal binary32 operands; never round an x87 load. */
static inline bool fpu_pc24_operand(floatx80 a, uint32_t *bits) {
    unsigned exp = a.high & 0x7fff;

    if (exp < 16383 - 126 || exp > 16383 + 127 || !(a.low & (UINT64_C(1) << 63)) ||
        (a.low & ((UINT64_C(1) << 40) - 1))) {
        return false;
    }
    *bits = ((uint32_t)(a.high & 0x8000) << 16) | ((exp - 16383 + 127) << 23) |
            ((a.low >> 40) & 0x7fffff);
    return true;
}

/* Exact identities do not depend on the precision control. Restrict identities
 * returning an operand to binary32-representable values so even PC24 cannot
 * round away significand bits. Zero/cancellation results need no conversion. */
static inline bool fpu_exact_binary(floatx80 a, floatx80 b, float_status *s,
                                    FpuPc24Op op, floatx80 *out)
{
    bool az = fpu_exact_zero(a), bz = fpu_exact_zero(b);
    uint32_t bits;
    unsigned sign = (a.high ^ b.high) & 0x8000;

    if (op == FPU_PC24_MUL || op == FPU_PC24_DIV) {
        if ((az && fpu_exact_normal(b)) ||
            (op == FPU_PC24_MUL && bz && (az || fpu_exact_normal(a)))) {
            *out = packFloatx80(sign >> 15, 0, 0);
            return true;
        }
        if ((b.high & 0x7fff) == 0x3fff && b.low == (UINT64_C(1) << 63) &&
            fpu_pc24_operand(a, &bits)) {
            *out = a;
            out->high = (out->high & 0x7fff) | sign;
            return true;
        }
        if (op == FPU_PC24_MUL && (a.high & 0x7fff) == 0x3fff &&
            a.low == (UINT64_C(1) << 63) && fpu_pc24_operand(b, &bits)) {
            *out = b;
            out->high = (out->high & 0x7fff) | sign;
            return true;
        }
        return false;
    }
    if (op != FPU_PC24_ADD && op != FPU_PC24_SUB) {
        return false;
    }
    if (op == FPU_PC24_SUB) {
        b.high ^= 0x8000;
    }
    if (az && bz) {
        sign = a.high == b.high ? a.high >> 15
                               : get_float_rounding_mode(s) == float_round_down;
        *out = packFloatx80(sign, 0, 0);
        return true;
    }
    if (az && fpu_pc24_operand(b, &bits)) {
        *out = b;
        return true;
    }
    if (bz && fpu_pc24_operand(a, &bits)) {
        *out = a;
        return true;
    }
    if ((a.high ^ b.high) == 0x8000 && a.low == b.low && fpu_exact_normal(a)) {
        *out = packFloatx80(get_float_rounding_mode(s) == float_round_down, 0, 0);
        return true;
    }
    return false;
}

/*
 * Basic x87 arithmetic only. Like QEMU hardfloat, this requires strict IEEE
 * host arithmetic in its default nearest-even environment. Unlike hardfloat's
 * sticky-inexact shortcut, calculate the current operation's inexact flag so
 * save_exception_flags(), FCLEX and unmasked exceptions remain unchanged.
 */
static inline bool fpu_pc24_try(floatx80 a, floatx80 b, float_status *s, FpuPc24Op op,
                                floatx80 *out) {
#if defined(__FAST_MATH__) || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53 ||        \
    FLT_EVAL_METHOD != 0
    return false;
#else
    uint32_t ab, bb, rb;
    float af, bf, rf;
    bool inexact;

    if (get_floatx80_rounding_precision(s) != floatx80_precision_s ||
        get_float_rounding_mode(s) != float_round_nearest_even || !fpu_pc24_operand(a, &ab) ||
        !fpu_pc24_operand(b, &bb)) {
        return false;
    }
    memcpy(&af, &ab, sizeof(af));
    memcpy(&bf, &bb, sizeof(bf));
    switch (op) {
        case FPU_PC24_ADD:
            rf = af + bf;
            break;
        case FPU_PC24_SUB:
            rf = af - bf;
            break;
        case FPU_PC24_MUL:
            rf = af * bf;
            break;
        case FPU_PC24_DIV:
            rf = af / bf;
            break;
        default:
            return false;
    }
    memcpy(&rb, &rf, sizeof(rb));
    /*
     * Include FLT_MIN itself in the fallback: binary32 can round up to it
     * while PC24 retains a value below it using x87's wider exponent range.
     * Cancellation, underflow and overflow retain the original exact path.
     */
    if ((rb & 0x7fffffff) <= 0x00800000 || (rb & 0x7f800000) == 0x7f800000) {
        return false;
    }
    if (op == FPU_PC24_MUL) {
        /* The product of two 24-bit significands is exact in binary64. */
        inexact = (double)af * (double)bf != (double)rf;
    } else if (op == FPU_PC24_DIV) {
        /* The quotient is exact iff its exact product equals the dividend. */
        inexact = (double)rf * (double)bf != (double)af;
    } else {
        int gap = (int)((ab >> 23) & 255) - (int)((bb >> 23) & 255);

        if (gap > 29 || gap < -29) {
            inexact = true; /* The nonzero smaller operand loses bits. */
        } else {
            /* At most 53 significant bits, including any possible carry. */
            double exact = op == FPU_PC24_ADD ? (double)af + (double)bf : (double)af - (double)bf;
            inexact = exact != (double)rf;
        }
    }
    if (inexact) {
        float_raise(float_flag_inexact, s);
    }
    out->high = (((rb >> 23) & 255) + 16383 - 127) | ((rb >> 16) & 0x8000);
    out->low = (uint64_t)((rb & 0x7fffff) | 0x800000) << 40;
    return true;
#endif
}

static inline floatx80 fpu_pc24_binary(floatx80 a, floatx80 b, float_status *s, FpuPc24Op op) {
    floatx80 result;

    if (fpu_exact_binary(a, b, s, op, &result) || fpu_pc24_try(a, b, s, op, &result)) {
        return result;
    }
    switch (op) {
        case FPU_PC24_ADD:
            return floatx80_add(a, b, s);
        case FPU_PC24_SUB:
            return floatx80_sub(a, b, s);
        case FPU_PC24_MUL:
            return floatx80_mul(a, b, s);
        case FPU_PC24_DIV:
            return floatx80_div(a, b, s);
        default:
            g_assert_not_reached();
    }
}
#endif
