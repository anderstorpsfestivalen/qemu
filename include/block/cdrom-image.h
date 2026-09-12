/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef BLOCK_CDROM_IMAGE_H
#define BLOCK_CDROM_IMAGE_H

/* Internal block ioctls, never guest or host-kernel ioctl structures. */
#define QEMU_CDROM_GET_INFO 0x51434401
#define QEMU_CDROM_READ_RAW 0x51434402
#define CDROM_IMAGE_MAX_TRACKS 99
#define CDROM_IMAGE_MAX_READ 256
#define CDROM_IMAGE_RAW_BYTES 2352

typedef struct CdromImageTrack {
    uint32_t start;       /* INDEX 01, disc LBA (without the 150-frame lead-in) */
    uint32_t index0;      /* pregap start, disc LBA */
    uint32_t file_start;  /* INDEX 01, file sector */
    uint32_t file_index0; /* INDEX 00, or INDEX 01 when absent */
    uint32_t pregap;      /* generated silence, absent from the file */
    uint8_t control;     /* Q-channel control, data=4, audio=0 */
} CdromImageTrack;

typedef struct CdromImageInfo {
    uint32_t sectors;     /* lead-out disc LBA */
    uint16_t sector_bytes;
    uint8_t tracks;
    CdromImageTrack track[CDROM_IMAGE_MAX_TRACKS];
} CdromImageInfo;

typedef struct CdromImageRead {
    uint32_t lba;
    uint32_t count;
    uint8_t *buffer;      /* host-owned, count * 2352 bytes */
} CdromImageRead;

/* Immutable metadata only: caller holds the block graph read lock. This must
 * also work while migration has drained block requests. */
bool bdrv_cdrom_get_info(BlockDriverState *bs, CdromImageInfo *info);

#endif
