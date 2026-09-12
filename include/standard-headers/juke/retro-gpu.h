/* SPDX-License-Identifier: MIT
 * Juke retro GPU guest/host ABI. All fields are little endian uint32_t.
 * Keep this header usable by old Windows C compilers (no QEMU dependencies).
 */
#ifndef JUKE_RETRO_GPU_H
#define JUKE_RETRO_GPU_H

/* Experimental Juke identity; not an upstream allocated PCI device ID. */
#define JRG_PCI_VENDOR_ID          0x1234
#define JRG_PCI_DEVICE_ID          0x1113
#define JRG_ABI_VERSION            0x00010000
#define JRG_MAGIC                  0x47524a51 /* QJRG */
#define JRG_VRAM_BAR               0
#define JRG_MMIO_BAR               2
#define JRG_MMIO_SIZE              0x2000
#define JRG_REG_MAGIC              0x1000
#define JRG_REG_VERSION            0x1004
#define JRG_REG_CAPS               0x1008
#define JRG_REG_VRAM_SIZE          0x100c
#define JRG_REG_BATCH_ADDR_LO      0x1010
#define JRG_REG_BATCH_ADDR_HI      0x1014
#define JRG_REG_BATCH_COUNT        0x1018
#define JRG_REG_SUBMIT_SEQUENCE    0x101c
#define JRG_REG_SUBMIT             0x1020
#define JRG_REG_STATUS             0x1024
#define JRG_REG_COMPLETED_SEQUENCE 0x1028
#define JRG_REG_ERROR              0x102c
#define JRG_REG_IRQ_ENABLE         0x1030
#define JRG_REG_IRQ_STATUS         0x1034
#define JRG_REG_RESET              0x1038
#define JRG_REG_GENERATION         0x103c
#define JRG_REG_MAX_COMMANDS       0x1040
#define JRG_REG_MAX_WORK_BYTES     0x1044

#define JRG_CAP_FILL               0x01
#define JRG_CAP_COPY               0x02
#define JRG_CAP_DAMAGE             0x04
#define JRG_CAP_COMPLETION_IRQ     0x08
#define JRG_CAP_INLINE_NO_IRQ      0x10
#define JRG_CAP_CURSOR             0x20
#define JRG_SUBMIT_START           0x01
#define JRG_SUBMIT_INLINE_NO_IRQ   0x02
#define JRG_STATUS_BUSY            0x01
#define JRG_STATUS_DONE            0x02
#define JRG_STATUS_ERROR           0x04
#define JRG_IRQ_COMPLETION         0x01
#define JRG_ERROR_NONE             0
#define JRG_ERROR_BATCH_COUNT      1
#define JRG_ERROR_DMA              2
#define JRG_ERROR_COMMAND          3
#define JRG_ERROR_BOUNDS           4
#define JRG_ERROR_WORK_LIMIT       5
#define JRG_MAX_COMMANDS           64
#define JRG_MAX_WORK_BYTES         0x04000000
#define JRG_COMMAND_BYTES          40
#define JRG_CMD_FILL               1
#define JRG_CMD_COPY               2
#define JRG_CMD_DAMAGE             3

/* Ten uint32_t words, without padding, in this order. No host pointer fields.
 * Use these offsets with a guest/toolchain-specific unsigned 32-bit type.
 * Offsets and strides are bytes; width/height are pixels; bpp is 1, 2 or 4.
 * COPY requires equal source/destination stride and supports overlapping areas.
 * Unused source fields, unused color and reserved must be zero.
 */
#define JRG_CMD_OPCODE             0
#define JRG_CMD_BPP                4
#define JRG_CMD_SRC_OFFSET         8
#define JRG_CMD_DST_OFFSET         12
#define JRG_CMD_SRC_STRIDE         16
#define JRG_CMD_DST_STRIDE         20
#define JRG_CMD_WIDTH              24
#define JRG_CMD_HEIGHT             28
#define JRG_CMD_COLOR              32
#define JRG_CMD_RESERVED           36

#endif
