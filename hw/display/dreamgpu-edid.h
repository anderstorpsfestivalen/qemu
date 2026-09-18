/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef DREAMGPU_EDID_H
#define DREAMGPU_EDID_H

#include "hw/display/edid.h"

/* The virtual scanout has twenty blank lines; these timings describe its
 * sixty-Hz default. The range descriptor permits the other selected rates.
 * Keep both dimension extrema of the maintained NT mode table explicit. */
static void dg_edid_detailed(uint8_t *desc, unsigned width, unsigned height) {
    unsigned clock = ((width + 160) * (height + 20) * 60 + 5000) / 10000;
    memset(desc, 0, 18);
    desc[0] = clock;
    desc[1] = clock >> 8;
    desc[2] = width;
    desc[3] = 160;
    desc[4] = (width >> 8) << 4;
    desc[5] = height;
    desc[6] = 20;
    desc[7] = (height >> 8) << 4;
    desc[8] = 48;
    desc[9] = 32;
    desc[10] = (3 << 4) | 5;
    desc[12] = 340 & 255;
    desc[13] = 270 & 255;
    desc[14] = 0x11;
    desc[17] = 0x1e; /* separate positive horizontal/vertical sync */
}

static void dg_edid_generate(uint8_t edid[256]) {
    qemu_edid_info info = {
        .vendor = "RHT", /* QEMU's established virtual-monitor vendor */
        .name = "DreamGPU",
        .width_mm = 340,
        .height_mm = 270,
        .prefx = 1024,
        .prefy = 768,
        .maxx = 3840,
        .maxy = 2400,
        .refresh_rate = 60000,
    };
    uint8_t name[18];
    memset(edid, 0, 256);
    qemu_edid_generate(edid, 256, &info);
    memcpy(name, edid + 108, sizeof(name));
    edid[10] = 0x13;
    edid[11] = 0x11; /* distinct virtual product RHT1113 */
    edid[20] = 0x08; /* virtual VGA, separate sync; no physical DP claim */
    dg_edid_detailed(edid + 54, 1024, 768);
    dg_edid_detailed(edid + 72, 3840, 2160);
    dg_edid_detailed(edid + 108, 3200, 2400);
    /* EDID 1.4 adds 255 to the maximum horizontal frequency with bit 3.
     * Older readers see 255 kHz, sufficient for every Win98 INF mode.
     * The full 510-kHz range also covers all NT modes at 120 Hz. */
    edid[94] = 0x08;
    edid[95] = 60;
    edid[96] = 120;
    edid[97] = 30;
    edid[98] = 255;
    edid[99] = 255; /* 2550-MHz pixel-clock envelope */
    /* Retain the name and a 4K sixty-Hz CTA mode. Do not inherit the common
     * generator's fifty-Hz modes outside our selected-rate contract. */
    memset(edid + 128, 0, 128);
    edid[128] = 2;
    edid[129] = 3;
    edid[130] = 6;
    edid[132] = 0x41;
    edid[133] = 97;
    memcpy(edid + 134, name, sizeof(name));
    edid[126] = 1;
    for (unsigned block = 0; block < 2; ++block) {
        unsigned sum = 0;
        for (unsigned i = 0; i < 127; ++i)
            sum += edid[block * 128 + i];
        edid[block * 128 + 127] = -sum;
    }
}

#endif
