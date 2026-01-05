/*
 * QEMU PowerMac Awacs Screamer device support
 *
 * Copyright (c) 2016 Mark Cave-Ayland
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HW_AUDIO_SCREAMER_H
#define HW_AUDIO_SCREAMER_H

#include "qemu/osdep.h"
#include "hw/sysbus.h"
#include "hw/ppc/mac_dbdma.h"
#include "audio/audio.h"

#define TYPE_SCREAMER "screamer"
OBJECT_DECLARE_SIMPLE_TYPE(ScreamerState, SCREAMER)

#define SCREAMER_BUFFER_SIZE 0x4000

struct ScreamerState {
    /*< private >*/
    SysBusDevice parent_obj;

    /*< public >*/
    MemoryRegion mem;
    qemu_irq irq;
    void *dbdma;
    qemu_irq dma_tx_irq;
    qemu_irq dma_rx_irq;

    QEMUSoundCard card;
    SWVoiceOut *voice;
    uint8_t buf[SCREAMER_BUFFER_SIZE];
    uint32_t bpos;
    uint32_t ppos;
    uint32_t rate;
    DBDMA_io io;

    uint32_t regs[6];
    uint32_t codec_ctrl_regs[8];
};

void macio_screamer_register_dma(ScreamerState *s, void *dbdma,
                                  int txchannel, int rxchannel);

#endif /* HW_AUDIO_SCREAMER_H */
