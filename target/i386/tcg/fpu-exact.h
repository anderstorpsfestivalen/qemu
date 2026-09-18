/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef I386_TCG_FPU_EXACT_H
#define I386_TCG_FPU_EXACT_H

#include "fpu/softfloat.h"

static inline bool fpu_exact_zero(floatx80 a)
{
    return !(a.high & 0x7fff) && !a.low;
}

static inline bool fpu_exact_normal(floatx80 a)
{
    unsigned exponent = a.high & 0x7fff;
    return exponent - 1 < 0x7ffe && (a.low >> 63);
}

/* No rounding or exception occurs when comparing canonical finite normals
 * and signed zeros. All other encodings retain the original NaN/denormal
 * handling, including the distinction between quiet and signaling compare. */
static inline FloatRelation fpu_exact_compare(floatx80 a, floatx80 b,
                                              float_status *status, bool quiet)
{
    bool az = fpu_exact_zero(a), bz = fpu_exact_zero(b);
    if ((az || fpu_exact_normal(a)) && (bz || fpu_exact_normal(b))) {
        unsigned ae = a.high & 0x7fff, be = b.high & 0x7fff;
        bool as = a.high >> 15, bs = b.high >> 15;
        bool less;
        if ((az && bz) || (a.high == b.high && a.low == b.low)) {
            return float_relation_equal;
        }
        if (as != bs) {
            return as ? float_relation_less : float_relation_greater;
        }
        less = ae != be ? ae < be : a.low < b.low;
        return less != as ? float_relation_less : float_relation_greater;
    }
    return quiet ? floatx80_compare_quiet(a, b, status)
                 : floatx80_compare(a, b, status);
}

#endif
