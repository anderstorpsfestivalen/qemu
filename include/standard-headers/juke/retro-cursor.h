/* SPDX-License-Identifier: MIT
 * Native Juke cursor ABI. All multibyte fields are little-endian.
 * No host pointers. Kept usable by Windows 9x/NT C toolchains.
 */
#ifndef JUKE_RETRO_CURSOR_H
#define JUKE_RETRO_CURSOR_H

#define JRG_CURSOR_ABI_VERSION       0x00010000
#define JRG_CURSOR_REG_VERSION       0x1080
#define JRG_CURSOR_REG_ADDR_LO       0x1084
#define JRG_CURSOR_REG_ADDR_HI       0x1088
#define JRG_CURSOR_REG_BYTES         0x108c
#define JRG_CURSOR_REG_WIDTH         0x1090
#define JRG_CURSOR_REG_HEIGHT        0x1094
#define JRG_CURSOR_REG_HOT_X         0x1098
#define JRG_CURSOR_REG_HOT_Y         0x109c
#define JRG_CURSOR_REG_FORMAT        0x10a0
#define JRG_CURSOR_REG_X             0x10a4
#define JRG_CURSOR_REG_Y             0x10a8
#define JRG_CURSOR_REG_FLAGS         0x10ac
#define JRG_CURSOR_REG_SEQUENCE      0x10b0
#define JRG_CURSOR_REG_SUBMIT        0x10b4
#define JRG_CURSOR_REG_STATUS        0x10b8
#define JRG_CURSOR_REG_COMPLETED     0x10bc
#define JRG_CURSOR_REG_ERROR         0x10c0
#define JRG_CURSOR_REG_MAX_DIMENSION 0x10c4

#define JRG_CURSOR_SHAPE             1
#define JRG_CURSOR_MOVE              2
#define JRG_CURSOR_VISIBLE           1
#define JRG_CURSOR_NATIVE_ENABLED    2
#define JRG_CURSOR_FLAGS_MASK        3
#define JRG_CURSOR_ARGB_PREMULTIPLIED 1
#define JRG_CURSOR_AND_XOR           2
#define JRG_CURSOR_MAX_DIMENSION     64
#define JRG_CURSOR_MAX_PIXELS        4096
#define JRG_CURSOR_PIXEL_BYTES       8
#define JRG_CURSOR_MAX_BYTES         32768
#define JRG_CURSOR_ERROR_NONE        0
#define JRG_CURSOR_ERROR_SHAPE       1
#define JRG_CURSOR_ERROR_DMA         2
#define JRG_CURSOR_ERROR_FLAGS       3

/*
 * SHAPE snapshots all registers and exactly width*height*8 DMA bytes; MOVE
 * snapshots X/Y/FLAGS only. Both complete synchronously, without an IRQ or
 * host loop, publishing STATUS=DONE|(ERROR if any) and COMPLETED=SEQUENCE.
 * A failure changes neither the accepted shape nor position/visibility.
 * SHAPE also atomically commits its position/flags. X/Y are signed position
 * of the hotspot. Hotspots must lie inside the nonzero <=64x64 shape.
 * NATIVE_ENABLED selects a guest-positioned cursor even while uncaptured;
 * VISIBLE independently shows/hides it. MOVE FLAGS=0 relinquishes that mode.
 * Each pixel is two packed u32 words, top-left rows, with no row padding:
 * ARGB_PREMULTIPLIED: [0xAARRGGBB, 0], color components <= alpha.
 * AND_XOR: [0x00RRGGBB and-mask, 0x00RRGGBB xor-color]. Output RGB is exactly
 * (desktopRGB & and-mask) ^ xor-color; desktop alpha is preserved.
 * Reset disables visibility/native mode. The normal engine reset also
 * clears cursor state. Cursor submission is independent of the GL channel.
 */

/* Independent cursor mapping on the existing v5 display Unix socket.
 * Every notification remains 24 bytes. SCM_RIGHTS belongs to byte0 of C.
 * C: kind='C', bytes1..7 zero, u64 epoch@8, u64 host_timestamp_us@16.
 * S: kind='S', bytes1..7 zero, u64 shape_generation@8, timestamp_us@16.
 * P: kind='P', flags@1, bytes2..7 zero, u64 position_sequence@8,
 *    signed i32 x@16 and y@20 (hotspot position).
 * C precedes S/P for that mapping. Position sequence and shape generation
 * increase independently within its epoch. C resets both consumer watermarks.
 * Each immutable shape includes its position sequence/X/Y/FLAGS snapshot;
 * publish them atomically on S, then ignore P sequences <= that snapshot.
 * This remains coherent if a consumer claims a newer slot than an older S.
 * Pending shape publication precedes its P, including under backpressure.
 * Reconnection and device reset create a new mapping/epoch; C also discards
 * the retained consumer shape. Old leases are never reclaimed.
 * Producer coalesces unsent shape/position updates to the latest state.
 * Cursor-only changes do not publish or dirty a desktop framebuffer.
 */
#define JCUR_MSG_MAPPING             'C'
#define JCUR_MSG_SHAPE               'S'
#define JCUR_MSG_POSITION            'P'
#define JCUR_PACKET_BYTES            24
#define JCUR_PACKET_FLAGS            1
#define JCUR_PACKET_SEQUENCE         8
#define JCUR_PACKET_X                16
#define JCUR_PACKET_Y                20

#define JCUR_MAGIC                   0x5255434a /* JCUR */
#define JCUR_VERSION                 1
#define JCUR_HEADER_BYTES            64
#define JCUR_SLOT_COUNT              3
#define JCUR_SLOT_BYTES              32832
#define JCUR_MAPPING_BYTES           98560
#define JCUR_SLOT_META_BYTES         64
#define JCUR_FREE                    0
#define JCUR_WRITING                 1
#define JCUR_READY                   2
#define JCUR_READING                 3

/* Header offsets. All reserved bytes must be zero. */
#define JCUR_HDR_MAGIC               0
#define JCUR_HDR_VERSION             4
#define JCUR_HDR_MAX_DIMENSION       8
#define JCUR_HDR_SLOT_COUNT          12
#define JCUR_HDR_EPOCH               16 /* immutable u64 */
#define JCUR_HDR_GENERATION          24 /* atomic u64, release published */
#define JCUR_HDR_SLOTS               32 /* three atomic u32 slot states */
#define JCUR_HDR_RESERVED            44 /* zero through byte63 */

/* Slot offsets: HEADER_BYTES + slot*SLOT_BYTES. Producer claims FREE or READY
 * with CAS -> WRITING, writes metadata/pixels, then release-stores READY and
 * header generation. Consumer CAS READY->READING before reading any metadata,
 * copies a complete shape, then release-stores FREE. Never hold a slot for
 * cursor movement. Unused pixel and reserved metadata bytes are zero.
 */
#define JCUR_SLOT_GENERATION         0 /* u64 */
#define JCUR_SLOT_WIDTH              8
#define JCUR_SLOT_HEIGHT             12
#define JCUR_SLOT_HOT_X              16
#define JCUR_SLOT_HOT_Y              20
#define JCUR_SLOT_FORMAT             24
#define JCUR_SLOT_RESERVED           28
#define JCUR_SLOT_POSITION_SEQUENCE  32 /* u64 */
#define JCUR_SLOT_X                  40
#define JCUR_SLOT_Y                  44
#define JCUR_SLOT_FLAGS              48
#define JCUR_SLOT_RESERVED_END       52 /* zero through byte63 */
#define JCUR_SLOT_PIXELS             64 /* canonical packed pixel pairs */

#endif
