/* SPDX-License-Identifier: LGPL-2.1-or-later
 * Conservative code coverage: one bit per 1/64th of a target page.
 * A zero intersection proves that a write cannot touch translated code.
 * Positive intersections still require the ordinary exact TB range check.
 */
#ifndef TCG_TB_CODE_BITMAP_H
#define TCG_TB_CODE_BITMAP_H

static inline uint64_t tb_code_range_mask(unsigned page_bits,
                                          uint64_t start, uint64_t last)
{
    unsigned first, final;

    /* Unexpected/cross-page ranges must never produce a false negative. */
    if (page_bits < 6 || page_bits > 63 || last < start ||
        (start >> page_bits) != (last >> page_bits)) {
        return UINT64_MAX;
    }
    first = (start >> (page_bits - 6)) & 63;
    final = (last >> (page_bits - 6)) & 63;
    return (UINT64_MAX << first) & (UINT64_MAX >> (63 - final));
}
#endif
