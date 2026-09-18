/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "libqos/libqos-pc.h"
#include "hw/display/bochs-vbe.h"
#include "standard-headers/dreamgpu/gpu.h"
#include "standard-headers/dreamgpu/gl.h"
#include "standard-headers/dreamgpu/gl-funcs.h"
#include "standard-headers/dreamgpu/cursor.h"
#include "standard-headers/linux/pci_regs.h"

typedef struct Fixture {
    QOSState *qs;
    QPCIDevice *dev;
    QPCIBar vram, regs;
    uint64_t batch;
    uint32_t sequence;
} Fixture;

static void found_device(QPCIDevice *dev, int devfn, void *opaque) {
    *(QPCIDevice **)opaque = dev;
}

static uint32_t reg_read(Fixture *f, unsigned reg) {
    return qpci_io_readl(f->dev, f->regs, reg);
}

static void reg_write(Fixture *f, unsigned reg, uint32_t value) {
    qpci_io_writel(f->dev, f->regs, reg, value);
}

static void setup(Fixture *f, gconstpointer unused) {
    uint64_t size;

    f->qs = qtest_pc_boot("-machine pc -m 64 -vga none "
                          "-device dreamgpu,id=retro");
    qpci_device_foreach(f->qs->pcibus, DG_PCI_VENDOR_ID, DG_PCI_DEVICE_ID, found_device, &f->dev);
    g_assert_nonnull(f->dev);
    f->vram = qpci_iomap(f->dev, DG_VRAM_BAR, &size);
    g_assert_cmpuint(size, ==, 16 * 1024 * 1024);
    f->regs = qpci_iomap(f->dev, DG_MMIO_BAR, &size);
    g_assert_cmpuint(size, ==, DG_MMIO_SIZE);
    qpci_device_enable(f->dev);
    f->batch =
        qmalloc(f->qs, MAX(DG_MAX_COMMANDS * DG_COMMAND_BYTES, (DG_GL_MAX_RECORDS + 1) * 52));
    f->sequence = 0;
}

static void teardown(Fixture *f, gconstpointer unused) {
    qfree(f->qs, f->batch);
    g_free(f->dev);
    qtest_shutdown(f->qs);
}

static void make_cmd(uint8_t *c, uint32_t op, uint32_t bpp, uint32_t src, uint32_t dst,
                     uint32_t stride, uint32_t width, uint32_t height, uint32_t color) {
    memset(c, 0, DG_COMMAND_BYTES);
    stl_le_p(c + DG_CMD_OPCODE, op);
    stl_le_p(c + DG_CMD_BPP, bpp);
    stl_le_p(c + DG_CMD_SRC_OFFSET, src);
    stl_le_p(c + DG_CMD_DST_OFFSET, dst);
    stl_le_p(c + DG_CMD_SRC_STRIDE, op == DG_CMD_COPY ? stride : 0);
    stl_le_p(c + DG_CMD_DST_STRIDE, stride);
    stl_le_p(c + DG_CMD_WIDTH, width);
    stl_le_p(c + DG_CMD_HEIGHT, height);
    stl_le_p(c + DG_CMD_COLOR, color);
}

static void submit_flags(Fixture *f, const uint8_t *cmds, uint32_t count, uint32_t flags) {
    if (cmds && count <= DG_MAX_COMMANDS) {
        qtest_memwrite(f->qs->qts, f->batch, cmds, count * DG_COMMAND_BYTES);
    }
    reg_write(f, DG_REG_BATCH_ADDR_LO, f->batch);
    reg_write(f, DG_REG_BATCH_ADDR_HI, f->batch >> 32);
    reg_write(f, DG_REG_BATCH_COUNT, count);
    reg_write(f, DG_REG_SUBMIT_SEQUENCE, ++f->sequence);
    reg_write(f, DG_REG_SUBMIT, flags);
}

static void submit(Fixture *f, const uint8_t *cmds, uint32_t count) {
    submit_flags(f, cmds, count, DG_SUBMIT_START);
}

static void await_completion(Fixture *f, uint32_t error) {
    int64_t deadline = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;
    uint32_t status;

    do {
        status = reg_read(f, DG_REG_STATUS);
        g_assert_cmpint(g_get_monotonic_time(), <, deadline);
        if (status & DG_STATUS_BUSY) {
            g_usleep(100);
        }
    } while (status & DG_STATUS_BUSY);
    g_assert_cmpuint(status, ==, DG_STATUS_DONE | (error ? DG_STATUS_ERROR : 0));
    g_assert_cmpuint(reg_read(f, DG_REG_ERROR), ==, error);
    g_assert_cmpuint(reg_read(f, DG_REG_COMPLETED_SEQUENCE), ==, f->sequence);
}

static void test_identity(Fixture *f, gconstpointer unused) {
    g_assert_cmphex(reg_read(f, DG_REG_MAGIC), ==, DG_MAGIC);
    g_assert_cmphex(reg_read(f, DG_REG_VERSION), ==, DG_ABI_VERSION);
    g_assert_cmphex(reg_read(f, DG_REG_CAPS), ==,
                    DG_CAP_FILL | DG_CAP_COPY | DG_CAP_DAMAGE | DG_CAP_COMPLETION_IRQ |
                        DG_CAP_INLINE_NO_IRQ | DG_CAP_CURSOR);
    g_assert_cmpuint(reg_read(f, DG_REG_STATUS), ==, 0);
    g_assert_cmpuint(qpci_io_readw(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET), >=, VBE_DISPI_ID0);
}

static void test_fill(Fixture *f, gconstpointer unused) {
    uint8_t cmd[DG_COMMAND_BYTES];
    uint8_t actual[4096], expected[4096];
    const unsigned widths[] = {1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129};
    unsigned bpp, n, x, y;

    /* Exercise vector-sized fills, short tails and starts not aligned to a
     * vector. Guard bytes include row padding and both ends of the allocation. */
    for (bpp = 1; bpp <= 4; bpp *= 2) {
        for (n = 0; n < ARRAY_SIZE(widths); n++) {
            memset(expected, 0x55, sizeof(expected));
            qpci_memwrite(f->dev, f->vram, 0, expected, sizeof(expected));
            make_cmd(cmd, DG_CMD_FILL, bpp, 0, 68, 544, widths[n], 3, 0x12345678);
            submit(f, cmd, 1);
            await_completion(f, 0);
            for (y = 0; y < 3; y++) {
                for (x = 0; x < widths[n] * bpp; x++) {
                    expected[68 + y * 544 + x] = 0x12345678 >> (8 * (x % bpp));
                }
            }
            qpci_memread(f->dev, f->vram, 0, actual, sizeof(actual));
            g_assert_cmpmem(actual, sizeof(actual), expected, sizeof(expected));
        }
    }
}

static void test_copy_overlap(Fixture *f, gconstpointer unused) {
    uint8_t cmd[DG_COMMAND_BYTES];
    uint8_t actual[1024], expected[1024], source[1024];
    const uint32_t offsets[][2] = {{0, 4}, {4, 0}, {0, 68}, {68, 0}};
    unsigned n, i, y;

    for (i = 0; i < sizeof(source); i++) {
        source[i] = i * 17 + i / 64;
    }
    for (n = 0; n < ARRAY_SIZE(offsets); n++) {
        uint32_t src = offsets[n][0], dst = offsets[n][1];

        memcpy(expected, source, sizeof(expected));
        for (y = 0; y < 6; y++) {
            memcpy(expected + dst + y * 64, source + src + y * 64, 48);
        }
        qpci_memwrite(f->dev, f->vram, 0, source, sizeof(source));
        make_cmd(cmd, DG_CMD_COPY, 4, src, dst, 64, 12, 6, 0);
        submit(f, cmd, 1);
        await_completion(f, 0);
        qpci_memread(f->dev, f->vram, 0, actual, sizeof(actual));
        g_assert_cmpmem(actual, sizeof(actual), expected, sizeof(expected));
    }
}

static void test_invalid(Fixture *f, gconstpointer unused) {
    uint8_t cmds[DG_COMMAND_BYTES * 5], actual[64], initial[64];
    unsigned i;

    memset(initial, 0x97, sizeof(initial));
    qpci_memwrite(f->dev, f->vram, 0, initial, sizeof(initial));
    make_cmd(cmds, DG_CMD_FILL, 4, 0, 0, 64, 16, 1, 0xffabcdef);
    make_cmd(cmds + DG_COMMAND_BYTES, 99, 4, 0, 0, 64, 16, 1, 0);
    submit(f, cmds, 2);
    await_completion(f, DG_ERROR_COMMAND);
    qpci_memread(f->dev, f->vram, 0, actual, sizeof(actual));
    g_assert_cmpmem(actual, sizeof(actual), initial, sizeof(initial));

    submit(f, NULL, 0);
    await_completion(f, DG_ERROR_BATCH_COUNT);
    submit(f, NULL, DG_MAX_COMMANDS + 1);
    await_completion(f, DG_ERROR_BATCH_COUNT);

    make_cmd(cmds, DG_CMD_FILL, 4, 0, UINT32_MAX - 3, 64, 16, 1, 0);
    submit(f, cmds, 1);
    await_completion(f, DG_ERROR_BOUNDS);
    make_cmd(cmds, DG_CMD_FILL, 4, 0, 0, UINT32_MAX, UINT32_MAX, UINT32_MAX, 0);
    submit(f, cmds, 1);
    await_completion(f, DG_ERROR_BOUNDS);
    make_cmd(cmds, DG_CMD_FILL, 0, 0, 0, 64, 16, 1, 0);
    submit(f, cmds, 1);
    await_completion(f, DG_ERROR_COMMAND);
    make_cmd(cmds, DG_CMD_FILL, 4, 0, 0, 64, 0, 1, 0);
    submit(f, cmds, 1);
    await_completion(f, DG_ERROR_BOUNDS);

    for (i = 0; i < 5; i++) {
        make_cmd(cmds + i * DG_COMMAND_BYTES, DG_CMD_FILL, 4, 0, 0, 4096 * 4, 4096, 1024, 0);
    }
    submit(f, cmds, 5);
    await_completion(f, DG_ERROR_WORK_LIMIT);

    reg_write(f, DG_REG_BATCH_COUNT, 1);
    reg_write(f, DG_REG_BATCH_ADDR_LO, UINT32_MAX - 15);
    reg_write(f, DG_REG_BATCH_ADDR_HI, UINT32_MAX);
    reg_write(f, DG_REG_SUBMIT_SEQUENCE, ++f->sequence);
    reg_write(f, DG_REG_SUBMIT, 1);
    await_completion(f, DG_ERROR_DMA);
}

static void test_irq_reset(Fixture *f, gconstpointer unused) {
    uint8_t cmd[DG_COMMAND_BYTES];
    uint32_t generation = reg_read(f, DG_REG_GENERATION);

    make_cmd(cmd, DG_CMD_DAMAGE, 4, 0, 0, 64, 16, 1, 0);
    submit(f, cmd, 1);
    await_completion(f, 0);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, DG_IRQ_COMPLETION);
    g_assert_false(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    reg_write(f, DG_REG_IRQ_ENABLE, DG_IRQ_COMPLETION);
    g_assert_true(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    reg_write(f, DG_REG_IRQ_STATUS, DG_IRQ_COMPLETION);
    g_assert_false(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    submit(f, cmd, 1);
    await_completion(f, 0);
    reg_write(f, DG_REG_RESET, 1);
    g_assert_cmpuint(reg_read(f, DG_REG_GENERATION), ==, generation + 1);
    g_assert_cmpuint(reg_read(f, DG_REG_STATUS), ==, 0);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, 0);
    g_assert_false(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    submit(f, cmd, 1);
    await_completion(f, 0);
}

static void test_inline_irq(Fixture *f, gconstpointer unused) {
    uint8_t commands[4 * DG_COMMAND_BYTES];
    const uint32_t flags = DG_SUBMIT_START | DG_SUBMIT_INLINE_NO_IRQ;

    reg_write(f, DG_REG_IRQ_ENABLE, DG_IRQ_COMPLETION);
    make_cmd(commands, DG_CMD_FILL, 4, 0, 0, 64, 16, 1, 0x12345678);
    submit_flags(f, commands, 1, flags);
    await_completion(f, 0);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, 0);
    g_assert_false(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    g_assert_cmphex(qpci_io_readl(f->dev, f->vram, 0), ==, 0x12345678);

    /* Immediate validation failures also have a final result on return. */
    submit_flags(f, NULL, 0, flags);
    await_completion(f, DG_ERROR_BATCH_COUNT);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, 0);

    /* Negotiation never acknowledges an older, still-pending interrupt. */
    submit(f, commands, 1);
    await_completion(f, 0);
    submit_flags(f, commands, 1, flags);
    await_completion(f, 0);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, DG_IRQ_COMPLETION);
    g_assert_true(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    reg_write(f, DG_REG_IRQ_STATUS, DG_IRQ_COMPLETION);

    /*
     * Work beyond the inline budget must retain its wakeup, including when
     * completion wins the race with the driver's first status read. Masking
     * the line here also proves that pending completion survives rearming.
     */
    reg_write(f, DG_REG_IRQ_ENABLE, 0);
    for (unsigned i = 0; i < 4; i++) {
        make_cmd(commands + i * DG_COMMAND_BYTES, DG_CMD_FILL, 4, 0, 0, 4096 * 4, 4096, 1024,
                 0x11223344);
    }
    submit_flags(f, commands, 4, flags);
    await_completion(f, 0);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, DG_IRQ_COMPLETION);
    g_assert_false(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    reg_write(f, DG_REG_IRQ_ENABLE, DG_IRQ_COMPLETION);
    g_assert_true(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    reg_write(f, DG_REG_IRQ_STATUS, DG_IRQ_COMPLETION);

    /* A subsequent ordinary submit still requests its historical IRQ. */
    make_cmd(commands, DG_CMD_FILL, 4, 0, 0, 64, 16, 1, 0);
    submit(f, commands, 1);
    await_completion(f, 0);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, DG_IRQ_COMPLETION);
    reg_write(f, DG_REG_RESET, 1);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, 0);
}

static void test_scanout(Fixture *f, gconstpointer unused) {
    uint8_t cmd[DG_COMMAND_BYTES];
    g_autofree char *name = NULL;
    g_autofree char *contents = NULL;
    gsize size;
    int fd = g_file_open_tmp("dreamgpu-XXXXXX.ppm", &name, NULL);
    const uint8_t color[] = {0x12, 0x34, 0x56};

    g_assert_cmpint(fd, >=, 0);
    close(fd);
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_XRES, 640);
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_YRES, 480);
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_BPP, 32);
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_ENABLE,
                   VBE_DISPI_ENABLED | VBE_DISPI_LFB_ENABLED);
    qtest_outb(f->qs->qts, 0x3c0, 0x20);
    make_cmd(cmd, DG_CMD_FILL, 4, 0, 0, 640 * 4, 640, 480, 0x00123456);
    submit(f, cmd, 1);
    await_completion(f, 0);
    qtest_qmp_assert_success(f->qs->qts, "{'execute':'screendump','arguments':{'filename':%s}}",
                             name);
    g_assert_true(g_file_get_contents(name, &contents, &size, NULL));
    g_assert_cmpuint(size, >=, 640 * 480 * 3);
    g_assert_cmpmem(contents + size - sizeof(color), sizeof(color), color, sizeof(color));
    unlink(name);
}

static void cursor_submit(Fixture *f, uint32_t operation, uint32_t error) {
    reg_write(f, DG_CURSOR_REG_SEQUENCE, ++f->sequence);
    reg_write(f, DG_CURSOR_REG_SUBMIT, operation);
    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_STATUS), ==,
                     DG_STATUS_DONE | (error ? DG_STATUS_ERROR : 0));
    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_ERROR), ==, error);
    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_COMPLETED), ==, f->sequence);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, 0);
    g_assert_false(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
}

static void test_cursor(Fixture *f, gconstpointer unused) {
    uint8_t pixels[16] = {0};

    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_VERSION), ==, DG_CURSOR_ABI_VERSION);
    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_MAX_DIMENSION), ==, 64);
    reg_write(f, DG_REG_IRQ_ENABLE, DG_IRQ_COMPLETION | DG_IRQ_GL_COMPLETION);
    reg_write(f, DG_CURSOR_REG_X, (uint32_t)-123);
    reg_write(f, DG_CURSOR_REG_Y, INT32_MIN);
    reg_write(f, DG_CURSOR_REG_FLAGS, DG_CURSOR_NATIVE_ENABLED);
    for (unsigned i = 0; i < 100; i++) {
        cursor_submit(f, DG_CURSOR_MOVE, 0);
    }
    reg_write(f, DG_CURSOR_REG_FLAGS, 4);
    cursor_submit(f, DG_CURSOR_MOVE, DG_CURSOR_ERROR_FLAGS);
    reg_write(f, DG_CURSOR_REG_FLAGS, DG_CURSOR_NATIVE_ENABLED | DG_CURSOR_VISIBLE);
    reg_write(f, DG_CURSOR_REG_ADDR_LO, f->batch);
    reg_write(f, DG_CURSOR_REG_ADDR_HI, f->batch >> 32);
    reg_write(f, DG_CURSOR_REG_WIDTH, 2);
    reg_write(f, DG_CURSOR_REG_HEIGHT, 1);
    reg_write(f, DG_CURSOR_REG_HOT_X, 1);
    reg_write(f, DG_CURSOR_REG_HOT_Y, 0);
    reg_write(f, DG_CURSOR_REG_BYTES, sizeof(pixels));
    reg_write(f, DG_CURSOR_REG_FORMAT, DG_CURSOR_ARGB_PREMULTIPLIED);
    stl_le_p(pixels, 0x80402010);
    stl_le_p(pixels + 8, 0xffffffff);
    qtest_memwrite(f->qs->qts, f->batch, pixels, sizeof(pixels));
    cursor_submit(f, DG_CURSOR_SHAPE, 0);
    reg_write(f, DG_CURSOR_REG_HOT_X, 2);
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_SHAPE);
    reg_write(f, DG_CURSOR_REG_HOT_X, 0);
    reg_write(f, DG_CURSOR_REG_WIDTH, 65);
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_SHAPE);
    reg_write(f, DG_CURSOR_REG_WIDTH, 0);
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_SHAPE);
    reg_write(f, DG_CURSOR_REG_WIDTH, 2);
    reg_write(f, DG_CURSOR_REG_BYTES, sizeof(pixels) - 1);
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_SHAPE);
    reg_write(f, DG_CURSOR_REG_BYTES, sizeof(pixels));
    stl_le_p(pixels, 0x00808080); /* not premultiplied */
    qtest_memwrite(f->qs->qts, f->batch, pixels, sizeof(pixels));
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_SHAPE);
    stl_le_p(pixels, 0xffffffff);
    stl_le_p(pixels + 4, 1); /* reserved second ARGB word */
    qtest_memwrite(f->qs->qts, f->batch, pixels, sizeof(pixels));
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_SHAPE);
    reg_write(f, DG_CURSOR_REG_FORMAT, DG_CURSOR_AND_XOR);
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_SHAPE);
    stl_le_p(pixels, 0x00ffffff);
    stl_le_p(pixels + 4, 0x00ffffff); /* exact invert */
    stl_le_p(pixels + 8, 0);          /* opaque white */
    stl_le_p(pixels + 12, 0x00ffffff);
    qtest_memwrite(f->qs->qts, f->batch, pixels, sizeof(pixels));
    cursor_submit(f, DG_CURSOR_SHAPE, 0);
    reg_write(f, DG_CURSOR_REG_ADDR_LO, f->regs.addr);
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_DMA);
    reg_write(f, DG_CURSOR_REG_ADDR_LO, UINT32_MAX - 7);
    reg_write(f, DG_CURSOR_REG_ADDR_HI, UINT32_MAX);
    cursor_submit(f, DG_CURSOR_SHAPE, DG_CURSOR_ERROR_DMA);
    reg_write(f, DG_REG_RESET, 1);
    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_FLAGS), ==, 0);
    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_STATUS), ==, 0);
    g_assert_cmpuint(reg_read(f, DG_CURSOR_REG_WIDTH), ==, 0);
}

static void test_batch_snapshot(Fixture *f, gconstpointer unused) {
    uint8_t cmds[2 * DG_COMMAND_BYTES];
    g_autofree uint8_t *actual = g_malloc(8 * 1024 * 1024);
    unsigned i;

    /* Mix bpp at a work-quantum boundary, then mutate the guest batch. */
    make_cmd(cmds, DG_CMD_FILL, 1, 0, 8 * 1024 * 1024, 1, 1, 1, 0x42);
    make_cmd(cmds + DG_COMMAND_BYTES, DG_CMD_FILL, 4, 0, 0, 8 * 1024 * 1024, 2 * 1024 * 1024, 1,
             0x12345678);
    submit(f, cmds, 2);
    memset(cmds, 0, sizeof(cmds));
    qtest_memwrite(f->qs->qts, f->batch, cmds, sizeof(cmds));
    reg_write(f, DG_REG_SUBMIT_SEQUENCE, 0xdeadbeef);
    await_completion(f, 0);
    qpci_memread(f->dev, f->vram, 0, actual, 8 * 1024 * 1024);
    for (i = 0; i < 8 * 1024 * 1024; i += 4) {
        g_assert_cmphex(ldl_le_p(actual + i), ==, 0x12345678);
    }
    g_assert_cmphex(qpci_io_readb(f->dev, f->vram, 8 * 1024 * 1024), ==, 0x42);
}

static void test_migration(Fixture *f, gconstpointer unused) {
    g_autofree char *dir = g_dir_make_tmp("dreamgpu-migrate-XXXXXX", NULL);
    g_autofree char *socket = g_build_filename(dir, "migration.sock", NULL);
    g_autofree char *uri = g_strdup_printf("unix:%s", socket);
    QOSState *to;
    QPCIDevice *dev = NULL;
    QPCIBar regs, vram;
    uint8_t cmd[DG_COMMAND_BYTES];
    uint64_t size;
    uint32_t generation;
    uint8_t cursor[32] = {0};
    uint8_t gl_record[40] = {0};
    bool native_gl;

    g_assert_nonnull(dir);
    for (unsigned i = 0; i < 4; i++) {
        stl_le_p(cursor + i * 8, 0xff112233);
    }
    qtest_memwrite(f->qs->qts, f->batch, cursor, sizeof(cursor));
    reg_write(f, DG_CURSOR_REG_ADDR_LO, f->batch);
    reg_write(f, DG_CURSOR_REG_ADDR_HI, f->batch >> 32);
    reg_write(f, DG_CURSOR_REG_BYTES, sizeof(cursor));
    reg_write(f, DG_CURSOR_REG_WIDTH, 2);
    reg_write(f, DG_CURSOR_REG_HEIGHT, 2);
    reg_write(f, DG_CURSOR_REG_HOT_X, 1);
    reg_write(f, DG_CURSOR_REG_HOT_Y, 1);
    reg_write(f, DG_CURSOR_REG_FORMAT, DG_CURSOR_ARGB_PREMULTIPLIED);
    reg_write(f, DG_CURSOR_REG_FLAGS, 3);
    reg_write(f, DG_CURSOR_REG_X, (uint32_t)-11);
    reg_write(f, DG_CURSOR_REG_Y, 29);
    cursor_submit(f, DG_CURSOR_SHAPE, 0);
    make_cmd(cmd, DG_CMD_FILL, 4, 0, 0, 64, 16, 1, 0x31415926);
    reg_write(f, DG_REG_IRQ_ENABLE, DG_IRQ_COMPLETION);
    submit(f, cmd, 1);
    await_completion(f, 0);
    generation = reg_read(f, DG_REG_GENERATION);
    native_gl = reg_read(f, DG_GL_REG_VERSION) != 0;
    if (native_gl) {
        stl_le_p(gl_record + DG_GL_OFF_OP, DG_GL_CALL);
        stl_le_p(gl_record + DG_GL_OFF_SIZE, sizeof(gl_record));
        stl_le_p(gl_record + DG_GL_OFF_CLIENT, 1);
        stl_le_p(gl_record + DG_GL_OFF_CONTEXT, 1);
        stl_le_p(gl_record + DG_GL_OFF_GENERATION, generation);
        stl_le_p(gl_record + 32, FEnum_glClear);
        stl_le_p(gl_record + 36, 0x4000);
        qtest_memwrite(f->qs->qts, f->batch + 128, gl_record, sizeof(gl_record));
        reg_write(f, DG_GL_REG_ADDR_LO, f->batch + 128);
        reg_write(f, DG_GL_REG_ADDR_HI, (f->batch + 128) >> 32);
        reg_write(f, DG_GL_REG_BYTES, sizeof(gl_record));
        reg_write(f, DG_GL_REG_GENERATION, generation);
        reg_write(f, DG_GL_REG_RESULT_ADDR_LO, f->batch + 512);
        reg_write(f, DG_GL_REG_RESULT_ADDR_HI, (f->batch + 512) >> 32);
        reg_write(f, DG_GL_REG_RESULT_CAPACITY, 512);
    }
    to = qtest_pc_boot("-machine pc -m 64 -vga none "
                       "-device dreamgpu,id=retro -incoming %s",
                       uri);
    migrate(f->qs, to, uri);
    qpci_device_foreach(to->pcibus, DG_PCI_VENDOR_ID, DG_PCI_DEVICE_ID, found_device, &dev);
    g_assert_nonnull(dev);
    vram = qpci_iomap(dev, DG_VRAM_BAR, &size);
    regs = qpci_iomap(dev, DG_MMIO_BAR, &size);
    qpci_device_enable(dev);
    g_assert_cmphex(qpci_io_readl(dev, vram, 0), ==, 0x31415926);
    g_assert_cmpuint(qpci_io_readl(dev, regs, DG_REG_COMPLETED_SEQUENCE), ==, f->sequence);
    g_assert_cmpuint(qpci_io_readl(dev, regs, DG_REG_GENERATION), ==, generation);
    g_assert_cmpuint(qpci_io_readl(dev, regs, DG_CURSOR_REG_COMPLETED), ==, 1);
    g_assert_cmpuint(qpci_io_readl(dev, regs, DG_CURSOR_REG_FLAGS), ==, 3);
    g_assert_cmpuint(qpci_io_readl(dev, regs, DG_CURSOR_REG_WIDTH), ==, 2);
    g_assert_cmphex(qpci_io_readl(dev, regs, DG_CURSOR_REG_X), ==, (uint32_t)-11);
    g_assert_true(qpci_config_readw(dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    qpci_io_writel(dev, regs, DG_REG_IRQ_STATUS, DG_IRQ_COMPLETION);
    g_assert_false(qpci_config_readw(dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    if (native_gl) {
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_ADDR_LO), ==,
                         (uint32_t)(f->batch + 128));
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_ADDR_HI), ==, (f->batch + 128) >> 32);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_RESULT_ADDR_LO), ==,
                         (uint32_t)(f->batch + 512));
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_RESULT_ADDR_HI), ==,
                         (f->batch + 512) >> 32);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_RESULT_CAPACITY), ==, 512);
        /* Only sequence/doorbell change: retained byte count, GPA and the
         * programmed generation must still validate on the destination. */
        for (unsigned seq = 71; seq < 73; seq++) {
            qpci_io_writel(dev, regs, DG_GL_REG_SEQUENCE, seq);
            qpci_io_writel(dev, regs, DG_GL_REG_SUBMIT, 1);
            g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_ERROR), ==, DG_GL_ERROR_TRANSPORT);
            g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_COMPLETED), ==, seq);
        }
        qpci_io_writel(dev, regs, DG_REG_RESET, 1);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_REG_GENERATION), !=, generation);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_ADDR_LO), ==, 0);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_ADDR_HI), ==, 0);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_RESULT_ADDR_LO), ==, 0);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_RESULT_CAPACITY), ==, 0);
        qpci_io_writel(dev, regs, DG_GL_REG_BYTES, sizeof(gl_record));
        qpci_io_writel(dev, regs, DG_GL_REG_SEQUENCE, 73);
        qpci_io_writel(dev, regs, DG_GL_REG_SUBMIT, 1);
        g_assert_cmpuint(qpci_io_readl(dev, regs, DG_GL_REG_ERROR), ==, DG_GL_ERROR_GENERATION);
    }
    qfree(to, f->batch);
    f->batch = 0;
    g_free(dev);
    qtest_shutdown(to);
    unlink(socket);
    rmdir(dir);
}

static void test_gl_validation(Fixture *f, gconstpointer unused) {
    uint8_t record[40] = {0};
    uint32_t generation = reg_read(f, DG_REG_GENERATION);

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    reg_write(f, DG_GL_REG_QUERY_FUNCTION, FEnum_glClearColor);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, 4);
    reg_write(f, DG_GL_REG_QUERY_FUNCTION, UINT32_MAX);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, UINT32_MAX);

    reg_write(f, DG_GL_REG_GENERATION, generation);
    reg_write(f, DG_GL_REG_ADDR_LO, f->batch);
    reg_write(f, DG_GL_REG_BYTES, sizeof(record));
    reg_write(f, DG_GL_REG_SEQUENCE, 41);
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_CALL);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record));
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, generation);
    stl_le_p(record + 32, UINT32_MAX);
    qtest_memwrite(f->qs->qts, f->batch, record, sizeof(record));
    reg_write(f, DG_GL_REG_SUBMIT, 1);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_ERROR), ==, DG_GL_ERROR_UNSUPPORTED);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_COMPLETED), ==, 41);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, DG_IRQ_GL_COMPLETION);
    reg_write(f, DG_REG_IRQ_ENABLE, DG_IRQ_GL_COMPLETION);
    g_assert_true(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);
    reg_write(f, DG_REG_IRQ_STATUS, DG_IRQ_GL_COMPLETION);
    g_assert_false(qpci_config_readw(f->dev, PCI_STATUS) & PCI_STATUS_INTERRUPT);

    stl_le_p(record + 32, FEnum_glClear);
    stl_le_p(record + 36, 0x4000);
    qtest_memwrite(f->qs->qts, f->batch, record, sizeof(record));
    reg_write(f, DG_GL_REG_SUBMIT, 1);
    /* Valid syntax does not create a worker without an export endpoint. */
    g_assert_cmpuint(reg_read(f, DG_GL_REG_ERROR), ==, DG_GL_ERROR_TRANSPORT);

    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record) + 4);
    qtest_memwrite(f->qs->qts, f->batch, record, sizeof(record));
    reg_write(f, DG_GL_REG_SUBMIT, 1);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_ERROR), ==, DG_GL_ERROR_BATCH);
    reg_write(f, DG_GL_REG_GENERATION, generation - 1);
    reg_write(f, DG_GL_REG_SUBMIT, 1);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_ERROR), ==, DG_GL_ERROR_GENERATION);
    reg_write(f, DG_REG_RESET, 1);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_STATUS), ==, 0);
    g_assert_cmpuint(reg_read(f, DG_REG_IRQ_STATUS), ==, 0);
}

static void gl_submit_error(Fixture *f, uint8_t *records, size_t bytes, uint32_t error) {
    qtest_memwrite(f->qs->qts, f->batch, records, bytes);
    reg_write(f, DG_GL_REG_ADDR_LO, f->batch);
    reg_write(f, DG_GL_REG_BYTES, bytes);
    reg_write(f, DG_GL_REG_GENERATION, reg_read(f, DG_REG_GENERATION));
    reg_write(f, DG_GL_REG_SEQUENCE, ++f->sequence);
    reg_write(f, DG_GL_REG_SUBMIT, 1);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_ERROR), ==, error);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_COMPLETED), ==, f->sequence);
}

static void test_gl_record_limit(Fixture *f, gconstpointer unused) {
    const unsigned bytes = 52;
    uint32_t generation = reg_read(f, DG_REG_GENERATION);

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    g_autofree uint8_t *records = g_malloc0((DG_GL_MAX_RECORDS + 1) * bytes);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_MAX_RECORDS), ==, 1024);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_MAX_BYTES), ==, DG_GL_MAX_BYTES);
    for (unsigned i = 0; i < DG_GL_MAX_RECORDS + 1; i++) {
        uint8_t *r = records + i * bytes;
        stl_le_p(r + DG_GL_OFF_OP, DG_GL_CALL);
        stl_le_p(r + DG_GL_OFF_SIZE, bytes);
        stl_le_p(r + DG_GL_OFF_CLIENT, 1);
        stl_le_p(r + DG_GL_OFF_CONTEXT, 1);
        stl_le_p(r + DG_GL_OFF_GENERATION, generation);
        stl_le_p(r + 32, FEnum_glColor4f);
    }
    /* A valid large scalar batch reaches the endpoint gate; no worker or
     * partial mutation can occur for a rejected overflow or invalid tail. */
    gl_submit_error(f, records, 256 * bytes, DG_GL_ERROR_TRANSPORT);
    gl_submit_error(f, records, 1024 * bytes, DG_GL_ERROR_TRANSPORT);
    gl_submit_error(f, records, 1025 * bytes, DG_GL_ERROR_BATCH);
    stl_le_p(records + 1023 * bytes + DG_GL_OFF_RESERVED, 1);
    gl_submit_error(f, records, 1024 * bytes, DG_GL_ERROR_BATCH);
}

static void test_desktop_validation(Fixture *f, gconstpointer unused) {
    enum { BYTES = DG_GL_HEADER_BYTES + DG_DESKTOP_BYTES };
    uint8_t records[17 * BYTES] = {0};
    uint8_t *r = records + DG_GL_HEADER_BYTES;

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_XRES, 1024);
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_YRES, 1024);
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_BPP, 32);
    qpci_io_writew(f->dev, f->regs, PCI_VGA_BOCHS_OFFSET + 2 * VBE_DISPI_INDEX_ENABLE,
                   VBE_DISPI_ENABLED | VBE_DISPI_LFB_ENABLED);
    stl_le_p(records + DG_GL_OFF_OP, DG_GL_DESKTOP);
    stl_le_p(records + DG_GL_OFF_SIZE, BYTES);
    stl_le_p(records + DG_GL_OFF_CLIENT, 1);
    stl_le_p(records + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    stl_le_p(r + DG_DESKTOP_OP, DG_DESKTOP_SEED);
    stl_le_p(r + DG_DESKTOP_WIDTH, 1024);
    stl_le_p(r + DG_DESKTOP_HEIGHT, 1024);
    stl_le_p(r + DG_DESKTOP_VRAM_STRIDE, 4096);
    /* Valid capture requires an endpoint; validation never copies pixels. */
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_TRANSPORT);
    stl_le_p(r + DG_DESKTOP_RESERVED0, 1);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_BATCH);
    stl_le_p(r + DG_DESKTOP_RESERVED0, 0);
    stl_le_p(r + DG_DESKTOP_VRAM_STRIDE, 4000);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_DESKTOP);
    stl_le_p(r + DG_DESKTOP_VRAM_STRIDE, 4096);
    stl_le_p(r + DG_DESKTOP_SLOT_OR_OFFSET, UINT32_MAX - 3);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_DESKTOP);
    stl_le_p(r + DG_DESKTOP_SLOT_OR_OFFSET, 0);
    stl_le_p(r + DG_DESKTOP_DST_X, 1);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_DESKTOP);
    stl_le_p(r + DG_DESKTOP_DST_X, 0);
    for (unsigned i = 1; i < 17; i++) {
        memcpy(records + i * BYTES, records, BYTES);
    }
    gl_submit_error(f, records, sizeof(records), DG_GL_ERROR_LIMIT);
    stl_le_p(r + DG_DESKTOP_OP, DG_DESKTOP_COPY);
    stl_le_p(r + DG_DESKTOP_VRAM_STRIDE, 0);
    stl_le_p(r + DG_DESKTOP_SRC_X, 1);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_DESKTOP);
    stl_le_p(r + DG_DESKTOP_OP, DG_DESKTOP_BLIT);
    stl_le_p(r + DG_DESKTOP_SRC_X, 0);
    stq_le_p(r + DG_DESKTOP_IMAGE_EPOCH, 1);
    stq_le_p(r + DG_DESKTOP_IMAGE_FRAME, 1);
    stl_le_p(r + DG_DESKTOP_SLOT_OR_OFFSET, 96);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_DESKTOP);
    stl_le_p(r + DG_DESKTOP_SLOT_OR_OFFSET, 0);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_TRANSPORT);
    stq_le_p(r + DG_DESKTOP_IMAGE_FRAME, 0);
    gl_submit_error(f, records, BYTES, DG_GL_ERROR_DESKTOP);
    reg_write(f, DG_REG_RESET, 1);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_STATUS), ==, 0);
}

static void test_texture_validation(Fixture *f, gconstpointer unused) {
    uint8_t record[88] = {0};
    uint8_t *args = record + DG_GL_DATA_ARGS;

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    reg_write(f, DG_GL_REG_QUERY_FUNCTION, FEnum_glTexImage2D);
    g_assert_cmphex(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, DG_GL_FUNCTION_INLINE_DATA | 8);
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_DATA_CALL);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record));
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    stl_le_p(record + DG_GL_DATA_FUNCTION, FEnum_glTexImage2D);
    stl_le_p(record + DG_GL_DATA_BYTES, 16);
    stl_le_p(args, 0x0de1);     /* GL_TEXTURE_2D */
    stl_le_p(args + 8, 0x1908); /* GL_RGBA */
    stl_le_p(args + 12, 2);
    stl_le_p(args + 16, 2);
    stl_le_p(args + 24, 0x1908);
    stl_le_p(args + 28, 0x1401); /* GL_UNSIGNED_BYTE */
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TRANSPORT);
    stl_le_p(record + DG_GL_DATA_BYTES, 0);
    stl_le_p(record + DG_GL_OFF_SIZE, 72);
    gl_submit_error(f, record, 72, DG_GL_ERROR_TRANSPORT);
    stl_le_p(args + 12, DG_GL_MAX_TEXTURE_DIMENSION);
    stl_le_p(args + 16, DG_GL_MAX_TEXTURE_DIMENSION);
    gl_submit_error(f, record, 72, DG_GL_ERROR_TRANSPORT);
    stl_le_p(args + 12, DG_GL_MAX_TEXTURE_DIMENSION + 1);
    gl_submit_error(f, record, 72, DG_GL_ERROR_TEXTURE);
    stl_le_p(args + 12, 2);
    stl_le_p(args + 16, 2);
    stl_le_p(record + DG_GL_DATA_FUNCTION, FEnum_glTexSubImage2D);
    gl_submit_error(f, record, 72, DG_GL_ERROR_TEXTURE);
    stl_le_p(record + DG_GL_DATA_FUNCTION, FEnum_glTexImage2D);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record));
    stl_le_p(record + DG_GL_DATA_BYTES, 15);
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
    stl_le_p(record + DG_GL_DATA_BYTES, 16);
    stl_le_p(args + 12, UINT32_MAX);
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
    stl_le_p(args + 12, 2);
    stl_le_p(args + 20, 1); /* borders are not represented by the bounded ABI */
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
    stl_le_p(args + 20, 0);
    stl_le_p(args + 4, 11); /* 2048 >> 11 permits only one texel per axis. */
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
    stl_le_p(args + 4, 12); /* Outside the advertised mip-level range. */
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
    stl_le_p(args + 4, 0);
    stl_le_p(args + 24, UINT32_MAX);
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
    stl_le_p(args + 24, 0x1908);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record) - 4);
    gl_submit_error(f, record, sizeof(record) - 4, DG_GL_ERROR_BATCH);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record));
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_CALL);
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_UNSUPPORTED);
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_DATA_CALL);
    stl_le_p(record + DG_GL_DATA_FUNCTION, FEnum_glDeleteTextures);
    stl_le_p(record + DG_GL_DATA_BYTES, 8);
    stl_le_p(args, 2);
    stl_le_p(record + DG_GL_OFF_SIZE, 52);
    gl_submit_error(f, record, 52, DG_GL_ERROR_TRANSPORT);
    stl_le_p(args, DG_GL_MAX_TEXTURES + 1);
    gl_submit_error(f, record, 52, DG_GL_ERROR_BATCH);
}

static void test_array_validation(Fixture *f, gconstpointer unused) {
    uint8_t record[40 + 20 + 3 * DG_GL_VERTEX_BYTES + 12] = {0};
    uint8_t *args = record + DG_GL_DATA_ARGS;
    uint8_t *vertices = args + 20;
    uint8_t *indices = vertices + 3 * DG_GL_VERTEX_BYTES;

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    reg_write(f, DG_GL_REG_QUERY_FUNCTION, FEnum_glDrawArrays);
    g_assert_cmphex(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, DG_GL_FUNCTION_INLINE_DATA | 4);
    reg_write(f, DG_GL_REG_QUERY_FUNCTION, FEnum_glDrawElements);
    g_assert_cmphex(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, DG_GL_FUNCTION_INLINE_DATA | 5);
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_DATA_CALL);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record));
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    stl_le_p(record + DG_GL_DATA_FUNCTION, FEnum_glDrawElements);
    stl_le_p(args, 4); /* GL_TRIANGLES */
    stl_le_p(args + 4, 3);
    stl_le_p(args + 12, 3);
    stl_le_p(args + 16, DG_GL_ARRAY_MASK);
    for (unsigned size = 1; size <= 4; size *= 2) {
        unsigned bytes = 3 * DG_GL_VERTEX_BYTES + 3 * size;
        unsigned total = 60 + QEMU_ALIGN_UP(bytes, 4);
        stl_le_p(record + DG_GL_OFF_SIZE, total);
        stl_le_p(record + DG_GL_DATA_BYTES, bytes);
        stl_le_p(args + 8, size == 1 ? 0x1401 : size == 2 ? 0x1403 : 0x1405);
        memset(indices, 0, 12);
        indices[size] = 1;
        indices[2 * size] = 2;
        gl_submit_error(f, record, total, DG_GL_ERROR_TRANSPORT);
        if (size < 4) {
            indices[3 * size] = 1;
            gl_submit_error(f, record, total, DG_GL_ERROR_BATCH);
            indices[3 * size] = 0;
        }
        indices[2 * size] = 3; /* an index equal to the vertex count is OOB */
        gl_submit_error(f, record, total, DG_GL_ERROR_BATCH);
        indices[2 * size] = 2;
        stl_le_p(vertices + DG_GL_VERTEX_RESERVED, 1);
        gl_submit_error(f, record, total, DG_GL_ERROR_BATCH);
        stl_le_p(vertices + DG_GL_VERTEX_RESERVED, 0);
        stl_le_p(args + 16, DG_GL_ARRAY_MASK & ~DG_GL_ARRAY_POSITION);
        gl_submit_error(f, record, total, DG_GL_ERROR_BATCH);
        stl_le_p(args + 16, DG_GL_ARRAY_MASK);
        stl_le_p(args + 12, DG_GL_MAX_VERTICES + 1);
        gl_submit_error(f, record, total, DG_GL_ERROR_BATCH);
        stl_le_p(args + 12, 3);
        stl_le_p(args + 4, DG_GL_MAX_INDICES + 1);
        gl_submit_error(f, record, total, DG_GL_ERROR_BATCH);
        stl_le_p(args + 4, 3);
    }
    memset(args, 0, sizeof(record) - DG_GL_DATA_ARGS);
    stl_le_p(record + DG_GL_DATA_FUNCTION, FEnum_glDrawArrays);
    stl_le_p(record + DG_GL_DATA_BYTES, 3 * DG_GL_VERTEX_BYTES);
    stl_le_p(record + DG_GL_OFF_SIZE, 56 + 3 * DG_GL_VERTEX_BYTES);
    stl_le_p(args, 4);
    stl_le_p(args + 8, 3);
    stl_le_p(args + 12, DG_GL_ARRAY_POSITION);
    gl_submit_error(f, record, 248, DG_GL_ERROR_TRANSPORT);
    stl_le_p(args + 4, UINT32_MAX); /* first must be normalized to zero */
    gl_submit_error(f, record, 248, DG_GL_ERROR_BATCH);
    stl_le_p(args + 4, 0);
    stl_le_p(args + 8, 0);
    stl_le_p(record + DG_GL_DATA_BYTES, 0);
    stl_le_p(record + DG_GL_OFF_SIZE, 56);
    gl_submit_error(f, record, 56, DG_GL_ERROR_TRANSPORT);
}

static void test_copy_texture_validation(Fixture *f, gconstpointer unused) {
    uint8_t record[68] = {0};
    uint8_t *args = record + 36;

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_CALL);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record));
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    for (unsigned sub = 0; sub < 2; sub++) {
        uint32_t fn = sub ? FEnum_glCopyTexSubImage2D : FEnum_glCopyTexImage2D;
        unsigned wi = sub ? 6 : 5, hi = sub ? 7 : 6;

        memset(args, 0, 32);
        stl_le_p(record + 32, fn);
        stl_le_p(args, 0x0de1);
        stl_le_p(args + 8, sub ? 0 : 0x1908);
        stl_le_p(args + wi * 4, 2);
        stl_le_p(args + hi * 4, 2);
        reg_write(f, DG_GL_REG_QUERY_FUNCTION, fn);
        g_assert_cmpuint(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, 8);
        gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TRANSPORT);
        stl_le_p(args + wi * 4, UINT32_MAX);
        gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
        stl_le_p(args + wi * 4, 2);
        stl_le_p(args + hi * 4, DG_GL_MAX_TEXTURE_DIMENSION + 1);
        gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
        stl_le_p(args + hi * 4, 2);
        stl_le_p(args, 0x8513); /* Cube maps have no guest namespace. */
        gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
        stl_le_p(args, 0x0de1);
        stl_le_p(args + 4, DG_GL_MAX_TEXTURE_LEVEL + 1);
        gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
        stl_le_p(args + 4, 0);
        if (!sub) {
            stl_le_p(args + 8, 4); /* Unsized component counts are not legal. */
            gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
            stl_le_p(args + 8, 0x1908);
            stl_le_p(args + 28, 1);
            gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
            stl_le_p(args + 28, 0);
            stl_le_p(args + 4, DG_GL_MAX_TEXTURE_LEVEL);
            gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TEXTURE);
            stl_le_p(args + 4, 0);
        }
        stl_le_p(args + wi * 4, 0);
        stl_le_p(args + hi * 4, 0);
        gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_TRANSPORT);
    }
}

static void test_query_validation(Fixture *f, gconstpointer unused) {
    uint8_t record[2 * DG_GL_QUERY_BYTES] = {0};
    uint64_t result_address = f->batch + 1024;

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    reg_write(f, DG_GL_REG_QUERY_FUNCTION, FEnum_glGetDoublev);
    g_assert_cmphex(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, DG_GL_FUNCTION_QUERY | 1);
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_QUERY);
    stl_le_p(record + DG_GL_OFF_SIZE, DG_GL_QUERY_BYTES);
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    stl_le_p(record + 32, FEnum_glGetDoublev);
    stl_le_p(record + 36, 0x0ba6); /* GL_MODELVIEW_MATRIX */
    reg_write(f, DG_GL_REG_RESULT_ADDR_LO, result_address);
    reg_write(f, DG_GL_REG_RESULT_CAPACITY, DG_GL_MAX_RESULT_BYTES);
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_TRANSPORT);
    reg_write(f, DG_GL_REG_RESULT_CAPACITY, 127);
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_BATCH);
    reg_write(f, DG_GL_REG_RESULT_CAPACITY, DG_GL_MAX_RESULT_BYTES + 1);
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_BATCH);
    reg_write(f, DG_GL_REG_RESULT_CAPACITY, DG_GL_MAX_RESULT_BYTES);
    reg_write(f, DG_GL_REG_RESULT_ADDR_LO, UINT32_MAX - 127);
    reg_write(f, DG_GL_REG_RESULT_ADDR_HI, UINT32_MAX);
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_DMA);
    reg_write(f, DG_GL_REG_RESULT_ADDR_HI, 0);
    reg_write(f, DG_GL_REG_RESULT_ADDR_LO, qpci_config_readl(f->dev, PCI_BASE_ADDRESS_2) & ~0xf);
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_DMA);
    reg_write(f, DG_GL_REG_RESULT_ADDR_LO, result_address);
    stl_le_p(record + 40, 1);
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_BATCH);
    stl_le_p(record + 40, 0);
    stl_le_p(record + 36, 0x8ca6); /* host framebuffer bindings stay private */
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_UNSUPPORTED);
    stl_le_p(record + 32, FEnum_glGetString);
    stl_le_p(record + 36, 0x1f02); /* no complete GL_VERSION advertised yet */
    gl_submit_error(f, record, DG_GL_QUERY_BYTES, DG_GL_ERROR_UNSUPPORTED);
    stl_le_p(record + 36, 0x1f00); /* GL_VENDOR */
    memcpy(record + DG_GL_QUERY_BYTES, record, DG_GL_QUERY_BYTES);
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_BATCH);
    reg_write(f, DG_REG_RESET, 1);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_RESULT_ADDR_LO), ==, 0);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_RESULT_CAPACITY), ==, 0);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_RESULT_BYTES), ==, 0);
}

static void test_buffer_validation(Fixture *f, gconstpointer unused) {
    uint8_t r[DG_GL_QUERY_BYTES] = {0};
    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    stl_le_p(r + DG_GL_OFF_OP, DG_GL_CALL);
    stl_le_p(r + DG_GL_OFF_SIZE, 40);
    stl_le_p(r + DG_GL_OFF_CLIENT, 1);
    stl_le_p(r + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(r + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    for (unsigned read = 0; read < 2; read++) {
        uint32_t fn = read ? FEnum_glReadBuffer : FEnum_glDrawBuffer;
        reg_write(f, DG_GL_REG_QUERY_FUNCTION, fn);
        g_assert_cmpuint(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, 1);
        stl_le_p(r + 32, fn);
        stl_le_p(r + 36, 0x0405); /* BACK */
        gl_submit_error(f, r, 40, DG_GL_ERROR_TRANSPORT);
        stl_le_p(r + 36, 0x0401); /* unsupported FRONT_RIGHT */
        gl_submit_error(f, r, 40, DG_GL_ERROR_UNSUPPORTED);
        stl_le_p(r + 36, 0); /* NONE is only a draw selection */
        gl_submit_error(f, r, 40, read ? DG_GL_ERROR_UNSUPPORTED : DG_GL_ERROR_TRANSPORT);
    }
    stl_le_p(r + DG_GL_OFF_SIZE, 44);
    stl_le_p(r + 32, FEnum_glHint);
    stl_le_p(r + 36, 0x0c50); /* PERSPECTIVE_CORRECTION_HINT */
    stl_le_p(r + 40, 0x1102); /* NICEST */
    gl_submit_error(f, r, 44, DG_GL_ERROR_TRANSPORT);
    stl_le_p(r + 40, UINT32_MAX);
    gl_submit_error(f, r, 44, DG_GL_ERROR_UNSUPPORTED);

    stl_le_p(r + DG_GL_OFF_OP, DG_GL_PRESENT);
    stl_le_p(r + DG_GL_OFF_SIZE, 32);
    stl_le_p(r + DG_GL_OFF_FLAGS, DG_GL_PRESENT_NO_EXPORT);
    gl_submit_error(f, r, 32, DG_GL_ERROR_TRANSPORT);
    stl_le_p(r + DG_GL_OFF_FLAGS, DG_GL_PRESENT_NO_EXPORT | DG_GL_PRESENT_FRONT_ONLY);
    gl_submit_error(f, r, 32, DG_GL_ERROR_BATCH);
    stl_le_p(r + DG_GL_OFF_FLAGS, DG_GL_PRESENT_RETAIN | DG_GL_PRESENT_FRONT_ONLY);
    gl_submit_error(f, r, 32, DG_GL_ERROR_TRANSPORT);

    stl_le_p(r + DG_GL_OFF_FLAGS, DG_GL_PRESENT_BOUNDED);
    gl_submit_error(f, r, 32, DG_GL_ERROR_BATCH);
    stl_le_p(r + DG_GL_OFF_SIZE, 40);
    stl_le_p(r + 32, 16);
    stl_le_p(r + 36, 8);
    gl_submit_error(f, r, 40, DG_GL_ERROR_TRANSPORT);
    stl_le_p(r + DG_GL_OFF_FLAGS, DG_GL_PRESENT_BOUNDED | DG_GL_PRESENT_NO_EXPORT);
    gl_submit_error(f, r, 40, DG_GL_ERROR_TRANSPORT);
    stl_le_p(r + 32, 0);
    gl_submit_error(f, r, 40, DG_GL_ERROR_DRAWABLE);
    stl_le_p(r + 32, DG_GL_MAX_DIMENSION + 1);
    gl_submit_error(f, r, 40, DG_GL_ERROR_DRAWABLE);

    stl_le_p(r + DG_GL_OFF_FLAGS, 0);
    stl_le_p(r + DG_GL_OFF_OP, DG_GL_QUERY);
    stl_le_p(r + DG_GL_OFF_SIZE, sizeof(r));
    stl_le_p(r + 32, FEnum_glReadPixels);
    stl_le_p(r + 36, 0);
    stl_le_p(r + 40, 0);
    stl_le_p(r + 44, 16 | (8 << 16));
    reg_write(f, DG_GL_REG_QUERY_FUNCTION, FEnum_glReadPixels);
    g_assert_cmphex(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, DG_GL_FUNCTION_QUERY | 3);
    reg_write(f, DG_GL_REG_RESULT_ADDR_LO, f->batch + 1024);
    reg_write(f, DG_GL_REG_RESULT_CAPACITY, DG_GL_MAX_RESULT_BYTES);
    gl_submit_error(f, r, sizeof(r), DG_GL_ERROR_TRANSPORT);
    reg_write(f, DG_GL_REG_RESULT_CAPACITY, 511);
    gl_submit_error(f, r, sizeof(r), DG_GL_ERROR_BATCH);
    reg_write(f, DG_GL_REG_RESULT_CAPACITY, DG_GL_MAX_RESULT_BYTES);
    for (unsigned i = 0; i < 4; i++) {
        const uint32_t sizes[] = {129 | (1 << 16), 1, 1 << 16, UINT32_MAX};
        stl_le_p(r + 44, sizes[i]);
        gl_submit_error(f, r, sizeof(r), DG_GL_ERROR_UNSUPPORTED);
    }
    stl_le_p(r + 44, 1 | (1 << 16));
    stl_le_p(r + 36, UINT32_MAX);
    gl_submit_error(f, r, sizeof(r), DG_GL_ERROR_UNSUPPORTED);
}

static void test_raster_validation(Fixture *f, gconstpointer unused) {
    const struct {
        uint32_t function, words;
    } functions[] = {
        {FEnum_glAlphaFunc, 2},   {FEnum_glColorMask, 4},    {FEnum_glDepthMask, 1},
        {FEnum_glDepthRange, 4},  {FEnum_glClearStencil, 1}, {FEnum_glStencilFunc, 3},
        {FEnum_glStencilMask, 1}, {FEnum_glStencilOp, 3},    {FEnum_glCullFace, 1},
        {FEnum_glFrontFace, 1},   {FEnum_glPolygonMode, 2},  {FEnum_glPolygonOffset, 2},
        {FEnum_glLineWidth, 1},   {FEnum_glLineStipple, 2},  {FEnum_glPointSize, 1},
        {FEnum_glShadeModel, 1},
    };
    uint8_t record[DG_GL_HEADER_BYTES + 24] = {0};

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_CALL);
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    for (unsigned i = 0; i < G_N_ELEMENTS(functions); i++) {
        uint32_t bytes = DG_GL_HEADER_BYTES + 4 + functions[i].words * 4;

        reg_write(f, DG_GL_REG_QUERY_FUNCTION, functions[i].function);
        g_assert_cmpuint(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==, functions[i].words);
        stl_le_p(record + DG_GL_HEADER_BYTES, functions[i].function);
        stl_le_p(record + DG_GL_OFF_SIZE, bytes);
        gl_submit_error(f, record, bytes, DG_GL_ERROR_TRANSPORT);
        stl_le_p(record + DG_GL_OFF_SIZE, bytes - 4);
        gl_submit_error(f, record, bytes - 4, DG_GL_ERROR_BATCH);
    }
}

static void test_vector_validation(Fixture *f, gconstpointer unused) {
    const struct {
        uint32_t fn, words, a, b, bytes;
    } vectors[] = {
        {FEnum_glLightfv, 2, 0x4000, 0x1203, 16},
        {FEnum_glMaterialfv, 2, 0x408, 0x1201, 16},
        {FEnum_glFogfv, 1, 0x0b66, 0, 16},
        {FEnum_glLightModelfv, 1, 0x0b53, 0, 16},
        {FEnum_glTexGenfv, 2, 0x2000, 0x2501, 16},
        {FEnum_glTexGendv, 2, 0x2000, 0x2501, 32},
        {FEnum_glClipPlane, 1, 0x3000, 0, 32},
        {FEnum_glTexParameterfv, 2, 0x0de1, 0x1004, 16},
        {FEnum_glTexParameteriv, 2, 0x0de1, 0x2801, 4},
        {FEnum_glTexEnvfv, 2, 0x2300, 0x2201, 16},
        {FEnum_glTexEnviv, 2, 0x2300, 0x2200, 4},
    };
    uint8_t record[80] = {0};

    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_DATA_CALL);
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_CONTEXT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    for (unsigned i = 0; i < G_N_ELEMENTS(vectors); i++) {
        uint32_t bytes = DG_GL_DATA_ARGS + vectors[i].words * 4 + vectors[i].bytes;

        reg_write(f, DG_GL_REG_QUERY_FUNCTION, vectors[i].fn);
        g_assert_cmphex(reg_read(f, DG_GL_REG_FUNCTION_WORDS), ==,
                        DG_GL_FUNCTION_INLINE_DATA | vectors[i].words);
        stl_le_p(record + DG_GL_DATA_FUNCTION, vectors[i].fn);
        stl_le_p(record + DG_GL_DATA_BYTES, vectors[i].bytes);
        stl_le_p(record + DG_GL_DATA_ARGS, vectors[i].a);
        stl_le_p(record + DG_GL_DATA_ARGS + 4, vectors[i].b);
        stl_le_p(record + DG_GL_OFF_SIZE, bytes);
        gl_submit_error(f, record, bytes, DG_GL_ERROR_TRANSPORT);
        stl_le_p(record + DG_GL_DATA_BYTES, vectors[i].bytes - 4);
        stl_le_p(record + DG_GL_OFF_SIZE, bytes - 4);
        gl_submit_error(f, record, bytes - 4, DG_GL_ERROR_BATCH);
        stl_le_p(record + DG_GL_DATA_BYTES, vectors[i].bytes);
        stl_le_p(record + DG_GL_OFF_SIZE, bytes);
        stl_le_p(record + DG_GL_DATA_ARGS, UINT32_MAX);
        gl_submit_error(f, record, bytes, DG_GL_ERROR_BATCH);
    }
}

static void assert_fault_stopped(Fixture *f, uint32_t reason, uint32_t op) {
    QTestState *qts = f->qs->qts;
    QDict *event = qtest_qmp_eventwait_ref(qts, "DREAMGPU_FAULT");
    QDict *data = qdict_get_qdict(event, "data");
    g_assert_cmpuint(qdict_get_int(data, "reason"), ==, reason);
    g_assert_cmpuint(qdict_get_int(data, "operation"), ==, op);
    qobject_unref(event);
    QDict *status = qtest_qmp_assert_success_ref(qts, "{'execute':'query-status'}");
    g_assert_cmpstr(qdict_get_str(status, "status"), ==, "internal-error");
    qobject_unref(status);
    QDict *error = qtest_qmp_assert_failure_ref(qts, "{'execute':'cont'}");
    g_assert_nonnull(strstr(qdict_get_str(error, "desc"), "Resetting"));
    qobject_unref(error);
    reg_write(f, DG_REG_RESET, 1);
    reg_write(f, DG_GL_REG_FAULT_STOP, 0);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_FAULT_STOP), ==, reason);
}

static void test_fault_stop(Fixture *f, gconstpointer unused) {
    if (!reg_read(f, DG_GL_REG_VERSION)) {
        g_test_skip("Native GL transport is not built on this host");
        return;
    }
    reg_write(f, DG_GL_REG_SEQUENCE, 73);
    reg_write(f, DG_GL_REG_FAULT_STOP, DG_GL_FAULT_DRIVER_TIMEOUT);
    assert_fault_stopped(f, DG_GL_FAULT_DRIVER_TIMEOUT, 0);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_FAULT_SEQUENCE), ==, 73);
    qtest_qmp_assert_success(f->qs->qts, "{'execute':'system_reset'}");
    qtest_qmp_eventwait(f->qs->qts, "RESET");
    qpci_config_writel(f->dev, PCI_BASE_ADDRESS_0, f->vram.addr);
    qpci_config_writel(f->dev, PCI_BASE_ADDRESS_2, f->regs.addr);
    qpci_device_enable(f->dev);
    g_assert_cmphex(reg_read(f, DG_REG_MAGIC), ==, DG_MAGIC);
    g_assert_cmpuint(reg_read(f, DG_GL_REG_FAULT_STOP), ==, 0);
    qtest_qmp_assert_success(f->qs->qts, "{'execute':'cont'}");

    uint8_t record[DG_GL_HEADER_BYTES + DG_DESKTOP_BYTES] = {0};
    stl_le_p(record + DG_GL_OFF_OP, DG_GL_DESKTOP);
    stl_le_p(record + DG_GL_OFF_SIZE, sizeof(record));
    stl_le_p(record + DG_GL_OFF_CLIENT, 1);
    stl_le_p(record + DG_GL_OFF_GENERATION, reg_read(f, DG_REG_GENERATION));
    stl_le_p(record + DG_GL_HEADER_BYTES + DG_DESKTOP_OP, DG_DESKTOP_READBACK);
    gl_submit_error(f, record, sizeof(record), DG_GL_ERROR_DESKTOP);
    assert_fault_stopped(f, DG_GL_FAULT_HOST_COHERENCE, DG_DESKTOP_READBACK);
}

int main(int argc, char **argv) {
    g_test_init(&argc, &argv, NULL);
    g_test_add("/dreamgpu/identity", Fixture, NULL, setup, test_identity, teardown);
    g_test_add("/dreamgpu/fill", Fixture, NULL, setup, test_fill, teardown);
    g_test_add("/dreamgpu/copy-overlap", Fixture, NULL, setup, test_copy_overlap, teardown);
    g_test_add("/dreamgpu/invalid-batches", Fixture, NULL, setup, test_invalid, teardown);
    g_test_add("/dreamgpu/irq-reset", Fixture, NULL, setup, test_irq_reset, teardown);
    g_test_add("/dreamgpu/inline-irq", Fixture, NULL, setup, test_inline_irq, teardown);
    g_test_add("/dreamgpu/scanout", Fixture, NULL, setup, test_scanout, teardown);
    g_test_add("/dreamgpu/cursor", Fixture, NULL, setup, test_cursor, teardown);
    g_test_add("/dreamgpu/batch-snapshot", Fixture, NULL, setup, test_batch_snapshot, teardown);
    g_test_add("/dreamgpu/migration", Fixture, NULL, setup, test_migration, teardown);
    g_test_add("/dreamgpu/gl-validation", Fixture, NULL, setup, test_gl_validation, teardown);
    g_test_add("/dreamgpu/gl-record-limit", Fixture, NULL, setup, test_gl_record_limit, teardown);
    g_test_add("/dreamgpu/desktop-validation", Fixture, NULL, setup, test_desktop_validation,
               teardown);
    g_test_add("/dreamgpu/texture-validation", Fixture, NULL, setup, test_texture_validation,
               teardown);
    g_test_add("/dreamgpu/copy-texture-validation", Fixture, NULL, setup,
               test_copy_texture_validation, teardown);
    g_test_add("/dreamgpu/array-validation", Fixture, NULL, setup, test_array_validation, teardown);
    g_test_add("/dreamgpu/query-validation", Fixture, NULL, setup, test_query_validation, teardown);
    g_test_add("/dreamgpu/raster-validation", Fixture, NULL, setup, test_raster_validation,
               teardown);
    g_test_add("/dreamgpu/buffer-validation", Fixture, NULL, setup, test_buffer_validation,
               teardown);
    g_test_add("/dreamgpu/vector-validation", Fixture, NULL, setup, test_vector_validation,
               teardown);
    g_test_add("/dreamgpu/fault-stop", Fixture, NULL, setup, test_fault_stop, teardown);
    return g_test_run();
}
