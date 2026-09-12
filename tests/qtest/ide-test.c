/*
 * IDE test cases
 *
 * Copyright (c) 2013 Kevin Wolf <kwolf@redhat.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "qemu/osdep.h"


#include "libqtest.h"
#include "libqos/libqos.h"
#include "libqos/pci-pc.h"
#include "libqos/malloc-pc.h"
#include "qobject/qdict.h"
#include "qemu/bswap.h"
#include "hw/pci/pci_ids.h"
#include "hw/pci/pci_regs.h"

/* Specified by ATA (physical) CHS geometry for ~64 MiB device.  */
#define TEST_IMAGE_SIZE ((130 * 16 * 63) * 512)

#define IDE_PCI_DEV     1
#define IDE_PCI_FUNC    1

#define IDE_BASE 0x1f0
#define IDE_BASE2 0x3f6
#define IDE_PRIMARY_IRQ 14

#define IDE_CTRL_RESET 0x04

#define ATAPI_BLOCK_SIZE 2048

/* Raw READ CD sector: 12 sync + 4 header + 2048 data + 288 EDC/ECC. */
#define ATAPI_RAW_SIZE   2352
#define ATAPI_RAW_DATA   16

/* How many bytes to receive via ATAPI PIO at one time.
 * Must be less than 0xFFFF. */
#define BYTE_COUNT_LIMIT 5120

enum {
    reg_data        = 0x0,
    reg_feature     = 0x1,
    reg_error       = 0x1,
    reg_nsectors    = 0x2,
    reg_lba_low     = 0x3,
    reg_lba_middle  = 0x4,
    reg_lba_high    = 0x5,
    reg_device      = 0x6,
    reg_status      = 0x7,
    reg_command     = 0x7,
};

enum {
    BSY     = 0x80,
    DRDY    = 0x40,
    DF      = 0x20,
    DRQ     = 0x08,
    ERR     = 0x01,
};

/* Error field */
enum {
    ABRT    = 0x04,
};

enum {
    DEV     = 0x10,
    LBA     = 0x40,
};

enum {
    bmreg_cmd       = 0x0,
    bmreg_status    = 0x2,
    bmreg_prdt      = 0x4,
};

enum {
    CMD_DSM         = 0x06,
    CMD_DIAGNOSE    = 0x90,
    CMD_INIT_DP     = 0x91,  /* INITIALIZE DEVICE PARAMETERS */
    CMD_READ_DMA    = 0xc8,
    CMD_WRITE_DMA   = 0xca,
    CMD_FLUSH_CACHE = 0xe7,
    CMD_IDENTIFY    = 0xec,
    CMD_PACKET      = 0xa0,
    CMD_READ_NATIVE = 0xf8,  /* READ NATIVE MAX ADDRESS */

    CMDF_ABORT      = 0x100,
    CMDF_NO_BM      = 0x200,
    CMDF_NO_WAIT    = 0x400,
};

enum {
    BM_CMD_START    =  0x1,
    BM_CMD_WRITE    =  0x8, /* write = from device to memory */
};

enum {
    BM_STS_ACTIVE   =  0x1,
    BM_STS_ERROR    =  0x2,
    BM_STS_INTR     =  0x4,
};

enum {
    PRDT_EOT        = 0x80000000,
};

#define assert_bit_set(data, mask) g_assert_cmphex((data) & (mask), ==, (mask))
#define assert_bit_clear(data, mask) g_assert_cmphex((data) & (mask), ==, 0)

static QPCIBus *pcibus = NULL;
static QGuestAllocator guest_malloc;

static char *tmp_path[2];
static char *debug_path;

G_GNUC_PRINTF(1, 2)
static QTestState *ide_test_start(const char *cmdline_fmt, ...)
{
    QTestState *qts;
    g_autofree char *full_fmt = g_strdup_printf("-machine pc %s", cmdline_fmt);
    va_list ap;

    va_start(ap, cmdline_fmt);
    qts = qtest_vinitf(full_fmt, ap);
    va_end(ap);

    pc_alloc_init(&guest_malloc, qts, 0);

    return qts;
}

static void ide_test_quit(QTestState *qts)
{
    if (pcibus) {
        qpci_free_pc(pcibus);
        pcibus = NULL;
    }
    alloc_destroy(&guest_malloc);
    qtest_quit(qts);
}

static QPCIDevice *get_pci_device(QTestState *qts, QPCIBar *bmdma_bar,
                                  QPCIBar *ide_bar)
{
    QPCIDevice *dev;
    uint16_t vendor_id, device_id;

    if (!pcibus) {
        pcibus = qpci_new_pc(qts, NULL);
    }

    /* Find PCI device and verify it's the right one */
    dev = qpci_device_find(pcibus, QPCI_DEVFN(IDE_PCI_DEV, IDE_PCI_FUNC));
    g_assert(dev != NULL);

    vendor_id = qpci_config_readw(dev, PCI_VENDOR_ID);
    device_id = qpci_config_readw(dev, PCI_DEVICE_ID);
    g_assert(vendor_id == PCI_VENDOR_ID_INTEL);
    g_assert(device_id == PCI_DEVICE_ID_INTEL_82371SB_1);

    /* Map bmdma BAR */
    *bmdma_bar = qpci_iomap(dev, 4, NULL);

    *ide_bar = qpci_legacy_iomap(dev, IDE_BASE);

    qpci_device_enable(dev);

    return dev;
}

static void free_pci_device(QPCIDevice *dev)
{
    /* libqos doesn't have a function for this, so free it manually */
    g_free(dev);
}

typedef struct PrdtEntry {
    uint32_t addr;
    uint32_t size;
} QEMU_PACKED PrdtEntry;

#define assert_bit_set(data, mask) g_assert_cmphex((data) & (mask), ==, (mask))
#define assert_bit_clear(data, mask) g_assert_cmphex((data) & (mask), ==, 0)

static uint64_t trim_range_le(uint64_t sector, uint16_t count)
{
    /* 2-byte range, 6-byte LBA */
    return cpu_to_le64(((uint64_t)count << 48) + sector);
}

static uint8_t wait_dma_completion(QTestState *qts, QPCIDevice *dev,
                                   QPCIBar bmdma_bar, QPCIBar ide_bar)
{
    uint8_t status;

    /* Wait for the DMA transfer to complete */
    do {
        status = qpci_io_readb(dev, bmdma_bar, bmreg_status);
    } while ((status & (BM_STS_ACTIVE | BM_STS_INTR)) == BM_STS_ACTIVE);

    g_assert_cmpint(qtest_get_irq(qts, IDE_PRIMARY_IRQ), ==,
                    !!(status & BM_STS_INTR));

    /* Check IDE status code */
    assert_bit_set(qpci_io_readb(dev, ide_bar, reg_status), DRDY);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), BSY | DRQ);

    /* Reading the status register clears the IRQ */
    g_assert(!qtest_get_irq(qts, IDE_PRIMARY_IRQ));

    /* Stop DMA transfer if still active */
    if (status & BM_STS_ACTIVE) {
        qpci_io_writeb(dev, bmdma_bar, bmreg_cmd, 0);
    }

    return status;
}

static int send_dma_request_dev(QTestState *qts, QPCIDevice *dev,
                                QPCIBar bmdma_bar, QPCIBar ide_bar, int cmd,
                                uint64_t sector, int nb_sectors,
                                PrdtEntry *prdt, int prdt_entries,
                                void(*post_exec)(QPCIDevice *dev,
                                                 QPCIBar ide_bar,
                                                 uint64_t sector,
                                                 int nb_sectors))
{
    uintptr_t guest_prdt;
    size_t len;
    bool from_dev;
    uint8_t status;
    int flags;

    flags = cmd & ~0xff;
    cmd &= 0xff;

    switch (cmd) {
    case CMD_READ_DMA:
    case CMD_PACKET:
        /* Assuming we only test data reads w/ ATAPI, otherwise we need to know
         * the SCSI command being sent in the packet, too. */
        from_dev = true;
        break;
    case CMD_DSM:
    case CMD_WRITE_DMA:
        from_dev = false;
        break;
    default:
        g_assert_not_reached();
    }

    if (flags & CMDF_NO_BM) {
        qpci_config_writew(dev, PCI_COMMAND,
                           PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
    }

    /* Select device 0 */
    qpci_io_writeb(dev, ide_bar, reg_device, 0 | LBA);

    /* Stop any running transfer, clear any pending interrupt */
    qpci_io_writeb(dev, bmdma_bar, bmreg_cmd, 0);
    qpci_io_writeb(dev, bmdma_bar, bmreg_status, BM_STS_INTR);

    /* Setup PRDT */
    len = sizeof(*prdt) * prdt_entries;
    guest_prdt = guest_alloc(&guest_malloc, len);
    qtest_memwrite(qts, guest_prdt, prdt, len);
    qpci_io_writel(dev, bmdma_bar, bmreg_prdt, guest_prdt);

    /* ATA DMA command */
    if (cmd == CMD_PACKET) {
        /* Enables ATAPI DMA; otherwise PIO is attempted */
        qpci_io_writeb(dev, ide_bar, reg_feature, 0x01);
    } else {
        if (cmd == CMD_DSM) {
            /* trim bit */
            qpci_io_writeb(dev, ide_bar, reg_feature, 0x01);
        }
        qpci_io_writeb(dev, ide_bar, reg_nsectors, nb_sectors);
        qpci_io_writeb(dev, ide_bar, reg_lba_low,    sector & 0xff);
        qpci_io_writeb(dev, ide_bar, reg_lba_middle, (sector >> 8) & 0xff);
        qpci_io_writeb(dev, ide_bar, reg_lba_high,   (sector >> 16) & 0xff);
    }

    qpci_io_writeb(dev, ide_bar, reg_command, cmd);

    if (post_exec) {
        post_exec(dev, ide_bar, sector, nb_sectors);
    }

    /* Start DMA transfer */
    qpci_io_writeb(dev, bmdma_bar, bmreg_cmd,
                   BM_CMD_START | (from_dev ? BM_CMD_WRITE : 0));

    if (flags & CMDF_ABORT) {
        qpci_io_writeb(dev, bmdma_bar, bmreg_cmd, 0);
    }

    if (flags & CMDF_NO_WAIT) {
        return 0;
    }

    status = wait_dma_completion(qts, dev, bmdma_bar, ide_bar);

    return status;
}

static int send_dma_request(QTestState *qts, int cmd, uint64_t sector,
                            int nb_sectors, PrdtEntry *prdt, int prdt_entries,
                            void(*post_exec)(QPCIDevice *dev, QPCIBar ide_bar,
                                             uint64_t sector, int nb_sectors))
{
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t status;

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);
    status = send_dma_request_dev(qts, dev, bmdma_bar, ide_bar,
                                  cmd, sector, nb_sectors, prdt, prdt_entries,
                                  post_exec);
    free_pci_device(dev);

    return status;
}

static QTestState *test_bmdma_setup(void)
{
    QTestState *qts;

    qts = ide_test_start(
        "-drive file=%s,if=ide,cache=writeback,format=raw "
        "-global ide-hd.serial=%s -global ide-hd.ver=%s",
        tmp_path[0], "testdisk", "version");
    qtest_irq_intercept_in(qts, "ioapic");

    return qts;
}

static void test_bmdma_teardown(QTestState *qts)
{
    ide_test_quit(qts);
}

static void test_bmdma_simple_rw(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t status;
    uint8_t *buf;
    uint8_t *cmpbuf;
    size_t len = 512;
    uintptr_t guest_buf;
    PrdtEntry prdt[1];

    qts = test_bmdma_setup();

    guest_buf  = guest_alloc(&guest_malloc, len);
    prdt[0].addr = cpu_to_le32(guest_buf);
    prdt[0].size = cpu_to_le32(len | PRDT_EOT);

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    buf = g_malloc(len);
    cmpbuf = g_malloc(len);

    /* Write 0x55 pattern to sector 0 */
    memset(buf, 0x55, len);
    qtest_memwrite(qts, guest_buf, buf, len);

    status = send_dma_request(qts, CMD_WRITE_DMA, 0, 1, prdt,
                              ARRAY_SIZE(prdt), NULL);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

    /* Write 0xaa pattern to sector 1 */
    memset(buf, 0xaa, len);
    qtest_memwrite(qts, guest_buf, buf, len);

    status = send_dma_request(qts, CMD_WRITE_DMA, 1, 1, prdt,
                              ARRAY_SIZE(prdt), NULL);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

    /* Read and verify 0x55 pattern in sector 0 */
    memset(cmpbuf, 0x55, len);

    status = send_dma_request(qts, CMD_READ_DMA, 0, 1, prdt, ARRAY_SIZE(prdt),
                              NULL);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

    qtest_memread(qts, guest_buf, buf, len);
    g_assert(memcmp(buf, cmpbuf, len) == 0);

    /* Read and verify 0xaa pattern in sector 1 */
    memset(cmpbuf, 0xaa, len);

    status = send_dma_request(qts, CMD_READ_DMA, 1, 1, prdt, ARRAY_SIZE(prdt),
                              NULL);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

    qtest_memread(qts, guest_buf, buf, len);
    g_assert(memcmp(buf, cmpbuf, len) == 0);

    free_pci_device(dev);
    g_free(buf);
    g_free(cmpbuf);

    test_bmdma_teardown(qts);
}

static void test_bmdma_trim(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t status;
    const uint64_t trim_range[] = { trim_range_le(0, 2),
                                    trim_range_le(6, 8),
                                    trim_range_le(10, 1),
                                  };
    const uint64_t bad_range = trim_range_le(TEST_IMAGE_SIZE / 512 - 1, 2);
    size_t len = 512;
    uint8_t *buf;
    uintptr_t guest_buf;
    PrdtEntry prdt[1];

    qts = test_bmdma_setup();

    guest_buf = guest_alloc(&guest_malloc, len);
    prdt[0].addr = cpu_to_le32(guest_buf),
    prdt[0].size = cpu_to_le32(len | PRDT_EOT),

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    buf = g_malloc(len);

    /* Normal request */
    *((uint64_t *)buf) = trim_range[0];
    *((uint64_t *)buf + 1) = trim_range[1];

    qtest_memwrite(qts, guest_buf, buf, 2 * sizeof(uint64_t));

    status = send_dma_request(qts, CMD_DSM, 0, 1, prdt,
                              ARRAY_SIZE(prdt), NULL);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

    /* Request contains invalid range */
    *((uint64_t *)buf) = trim_range[2];
    *((uint64_t *)buf + 1) = bad_range;

    qtest_memwrite(qts, guest_buf, buf, 2 * sizeof(uint64_t));

    status = send_dma_request(qts, CMD_DSM, 0, 1, prdt,
                              ARRAY_SIZE(prdt), NULL);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_set(qpci_io_readb(dev, ide_bar, reg_status), ERR);
    assert_bit_set(qpci_io_readb(dev, ide_bar, reg_error), ABRT);

    free_pci_device(dev);
    g_free(buf);
    test_bmdma_teardown(qts);
}

static void test_bmdma_trim_reset(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar, ide_bar2;
    uint8_t status;
    const uint64_t trim_range[] = {
        trim_range_le(0, 2),
        trim_range_le(6, 8),
    };
    size_t len = 512;
    uint8_t *buf;
    uintptr_t guest_buf;
    PrdtEntry prdt[1];

    qts = ide_test_start(
        "-blockdev file,filename=%s,node-name=img "
        "-blockdev blkdebug,image=img,node-name=dbg,discard=unmap,"
        "inject-error.0.event=none,inject-error.0.iotype=discard,"
        "inject-error.0.errno=0,inject-error.0.delay-ns=1000000 "
        "-device ide-hd,drive=dbg,bus=ide.0",
        tmp_path[0]);
    qtest_irq_intercept_in(qts, "ioapic");

    guest_buf = guest_alloc(&guest_malloc, len);
    prdt[0].addr = cpu_to_le32(guest_buf),
    prdt[0].size = cpu_to_le32(len | PRDT_EOT),

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);
    ide_bar2 = qpci_legacy_iomap(dev, IDE_BASE2);

    buf = g_malloc(len);

    /* TRIM request with two segments */
    *((uint64_t *)buf) = trim_range[0];
    *((uint64_t *)buf + 1) = trim_range[1];

    qtest_memwrite(qts, guest_buf, buf, 2 * sizeof(uint64_t));

    send_dma_request_dev(qts, dev, bmdma_bar, ide_bar, CMD_DSM | CMDF_NO_WAIT, 0, 1, prdt,
                     ARRAY_SIZE(prdt), NULL);

    /* Reset the device while the first segment is in flight */
    qpci_io_writeb(dev, ide_bar2, 0, IDE_CTRL_RESET);

    status = wait_dma_completion(qts, dev, bmdma_bar, ide_bar);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

    free_pci_device(dev);
    g_free(buf);
    test_bmdma_teardown(qts);
}

/*
 * This test is developed according to the Programming Interface for
 * Bus Master IDE Controller (Revision 1.0 5/16/94)
 */
static void test_bmdma_various_prdts(void)
{
    int sectors = 0;
    uint32_t size = 0;

    for (sectors = 1; sectors <= 256; sectors *= 2) {
        QTestState *qts = NULL;
        QPCIDevice *dev = NULL;
        QPCIBar bmdma_bar, ide_bar;

        qts = test_bmdma_setup();
        dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

        for (size = 0; size < 65536; size += 256) {
            uint32_t req_size = sectors * 512;
            uint32_t prd_size = size & 0xfffe; /* bit 0 is always set to 0 */
            uint8_t ret = 0;
            uint8_t req_status = 0;
            uint8_t abort_req_status = 0;
            PrdtEntry prdt[] = {
                {
                    .addr = 0,
                    .size = cpu_to_le32(size | PRDT_EOT),
                },
            };

            /* A value of zero in PRD size indicates 64K */
            if (prd_size == 0) {
                prd_size = 65536;
            }

            /*
             * 1. If PRDs specified a smaller size than the IDE transfer
             * size, then the Interrupt and Active bits in the Controller
             * status register are not set (Error Condition).
             *
             * 2. If the size of the physical memory regions was equal to
             * the IDE device transfer size, the Interrupt bit in the
             * Controller status register is set to 1, Active bit is set to 0.
             *
             * 3. If PRDs specified a larger size than the IDE transfer size,
             * the Interrupt and Active bits in the Controller status register
             * are both set to 1.
             */
            if (prd_size < req_size) {
                req_status = 0;
                abort_req_status = 0;
            } else if (prd_size == req_size) {
                req_status = BM_STS_INTR;
                abort_req_status = BM_STS_INTR;
            } else {
                req_status = BM_STS_ACTIVE | BM_STS_INTR;
                abort_req_status = BM_STS_INTR;
            }

            /* Test the request */
            ret = send_dma_request(qts, CMD_READ_DMA, 0, sectors,
                                   prdt, ARRAY_SIZE(prdt), NULL);
            g_assert_cmphex(ret, ==, req_status);
            assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

            /* Now test aborting the same request */
            ret = send_dma_request(qts, CMD_READ_DMA | CMDF_ABORT, 0,
                                   sectors, prdt, ARRAY_SIZE(prdt), NULL);
            g_assert_cmphex(ret, ==, abort_req_status);
            assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);
        }

        free_pci_device(dev);
        test_bmdma_teardown(qts);
    }
}

static void test_bmdma_no_busmaster(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t status;

    qts = test_bmdma_setup();

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    /* No PRDT_EOT, each entry addr 0/size 64k, and in theory qemu shouldn't be
     * able to access it anyway because the Bus Master bit in the PCI command
     * register isn't set. This is complete nonsense, but it used to be pretty
     * good at confusing and occasionally crashing qemu. */
    PrdtEntry prdt[4096] = { };

    status = send_dma_request(qts, CMD_READ_DMA | CMDF_NO_BM, 0, 512,
                              prdt, ARRAY_SIZE(prdt), NULL);

    /* Not entirely clear what the expected result is, but this is what we get
     * in practice. At least we want to be aware of any changes. */
    g_assert_cmphex(status, ==, BM_STS_ACTIVE | BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);
    free_pci_device(dev);
    test_bmdma_teardown(qts);
}

static void string_cpu_to_be16(uint16_t *s, size_t bytes)
{
    g_assert((bytes & 1) == 0);
    bytes /= 2;

    while (bytes--) {
        *s = cpu_to_be16(*s);
        s++;
    }
}

static void test_specify(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint16_t cyls;
    uint8_t heads, spt;

    qts = ide_test_start(
        "-blockdev driver=file,node-name=hda,filename=%s "
        "-device ide-hd,drive=hda,bus=ide.0,unit=0 ",
        tmp_path[0]);

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    /* Initialize drive with zero sectors per track and one head.  */
    qpci_io_writeb(dev, ide_bar, reg_nsectors, 0);
    qpci_io_writeb(dev, ide_bar, reg_device, 0);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_INIT_DP);

    /* READ NATIVE MAX ADDRESS (CHS mode).  */
    qpci_io_writeb(dev, ide_bar, reg_device, 0xa0);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_READ_NATIVE);

    heads = qpci_io_readb(dev, ide_bar, reg_device) & 0xf;
    ++heads;
    g_assert_cmpint(heads, ==, 16);

    cyls = qpci_io_readb(dev, ide_bar, reg_lba_high) << 8;
    cyls |= qpci_io_readb(dev, ide_bar, reg_lba_middle);
    ++cyls;
    g_assert_cmpint(cyls, ==, 130);

    spt = qpci_io_readb(dev, ide_bar, reg_lba_low);
    g_assert_cmpint(spt, ==, 63);

    ide_test_quit(qts);
    free_pci_device(dev);
}

static void test_identify(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t data;
    uint16_t buf[256];
    int i;
    int ret;

    qts = ide_test_start(
        "-drive file=%s,if=ide,cache=writeback,format=raw "
        "-global ide-hd.serial=%s -global ide-hd.ver=%s",
        tmp_path[0], "testdisk", "version");

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    /* IDENTIFY command on device 0*/
    qpci_io_writeb(dev, ide_bar, reg_device, 0);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_IDENTIFY);

    /* Read in the IDENTIFY buffer and check registers */
    data = qpci_io_readb(dev, ide_bar, reg_device);
    g_assert_cmpint(data & DEV, ==, 0);

    for (i = 0; i < 256; i++) {
        data = qpci_io_readb(dev, ide_bar, reg_status);
        assert_bit_set(data, DRDY | DRQ);
        assert_bit_clear(data, BSY | DF | ERR);

        buf[i] = qpci_io_readw(dev, ide_bar, reg_data);
    }

    data = qpci_io_readb(dev, ide_bar, reg_status);
    assert_bit_set(data, DRDY);
    assert_bit_clear(data, BSY | DF | ERR | DRQ);

    /* Check serial number/version in the buffer */
    string_cpu_to_be16(&buf[10], 20);
    ret = memcmp(&buf[10], "testdisk            ", 20);
    g_assert(ret == 0);

    string_cpu_to_be16(&buf[23], 8);
    ret = memcmp(&buf[23], "version ", 8);
    g_assert(ret == 0);

    /* Write cache enabled bit */
    assert_bit_set(buf[85], 0x20);

    ide_test_quit(qts);
    free_pci_device(dev);
}

static void test_diagnostic(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t data;

    qts = ide_test_start(
        "-blockdev driver=file,node-name=hda,filename=%s "
        "-blockdev driver=file,node-name=hdb,filename=%s "
        "-device ide-hd,drive=hda,bus=ide.0,unit=0 "
        "-device ide-hd,drive=hdb,bus=ide.0,unit=1 ",
        tmp_path[0], tmp_path[1]);

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    /* DIAGNOSE command on device 1 */
    qpci_io_writeb(dev, ide_bar, reg_device, DEV);
    data = qpci_io_readb(dev, ide_bar, reg_device);
    g_assert_cmphex(data & DEV, ==, DEV);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_DIAGNOSE);

    /* Verify that DEVICE is now 0 */
    data = qpci_io_readb(dev, ide_bar, reg_device);
    g_assert_cmphex(data & DEV, ==, 0);

    ide_test_quit(qts);
    free_pci_device(dev);
}

/*
 * Write sector 1 with random data to make IDE storage dirty
 * Needed for flush tests so that flushes actually go though the block layer
 */
static void make_dirty(QTestState *qts, uint8_t device)
{
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t status;
    size_t len = 512;
    uintptr_t guest_buf;
    void* buf;

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    guest_buf = guest_alloc(&guest_malloc, len);
    buf = g_malloc(len);
    memset(buf, rand() % 255 + 1, len);
    g_assert(guest_buf);
    g_assert(buf);

    qtest_memwrite(qts, guest_buf, buf, len);

    PrdtEntry prdt[] = {
        {
            .addr = cpu_to_le32(guest_buf),
            .size = cpu_to_le32(len | PRDT_EOT),
        },
    };

    status = send_dma_request(qts, CMD_WRITE_DMA, 1, 1, prdt,
                              ARRAY_SIZE(prdt), NULL);
    g_assert_cmphex(status, ==, BM_STS_INTR);
    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);

    g_free(buf);
    free_pci_device(dev);
}

static void test_flush(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t data;

    qts = ide_test_start(
        "-drive file=blkdebug::%s,if=ide,cache=writeback,format=raw",
        tmp_path[0]);

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    qtest_irq_intercept_in(qts, "ioapic");

    /* Dirty media so that CMD_FLUSH_CACHE will actually go to disk */
    make_dirty(qts, 0);

    /* Delay the completion of the flush request until we explicitly do it */
    g_free(qtest_hmp(qts, "qemu-io ide0-hd0 \"break flush_to_os A\""));

    /* FLUSH CACHE command on device 0*/
    qpci_io_writeb(dev, ide_bar, reg_device, 0);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_FLUSH_CACHE);

    /* Check status while request is in flight*/
    data = qpci_io_readb(dev, ide_bar, reg_status);
    assert_bit_set(data, BSY | DRDY);
    assert_bit_clear(data, DF | ERR | DRQ);

    /* Complete the command */
    g_free(qtest_hmp(qts, "qemu-io ide0-hd0 \"resume A\""));

    /* Check registers */
    data = qpci_io_readb(dev, ide_bar, reg_device);
    g_assert_cmpint(data & DEV, ==, 0);

    do {
        data = qpci_io_readb(dev, ide_bar, reg_status);
    } while (data & BSY);

    assert_bit_set(data, DRDY);
    assert_bit_clear(data, BSY | DF | ERR | DRQ);

    ide_test_quit(qts);
    free_pci_device(dev);
}

static void test_pci_retry_flush(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t data;

    prepare_blkdebug_script(debug_path, "flush_to_disk");

    qts = ide_test_start(
        "-drive file=blkdebug:%s:%s,if=ide,cache=writeback,format=raw,"
        "rerror=stop,werror=stop",
        debug_path, tmp_path[0]);

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    qtest_irq_intercept_in(qts, "ioapic");

    /* Dirty media so that CMD_FLUSH_CACHE will actually go to disk */
    make_dirty(qts, 0);

    /* FLUSH CACHE command on device 0*/
    qpci_io_writeb(dev, ide_bar, reg_device, 0);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_FLUSH_CACHE);

    /* Check status while request is in flight*/
    data = qpci_io_readb(dev, ide_bar, reg_status);
    assert_bit_set(data, BSY | DRDY);
    assert_bit_clear(data, DF | ERR | DRQ);

    qtest_qmp_eventwait(qts, "STOP");

    /* Complete the command */
    qtest_qmp_assert_success(qts, "{'execute':'cont' }");

    /* Check registers */
    data = qpci_io_readb(dev, ide_bar, reg_device);
    g_assert_cmpint(data & DEV, ==, 0);

    do {
        data = qpci_io_readb(dev, ide_bar, reg_status);
    } while (data & BSY);

    assert_bit_set(data, DRDY);
    assert_bit_clear(data, BSY | DF | ERR | DRQ);

    ide_test_quit(qts);
    free_pci_device(dev);
}

static void test_flush_nodev(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;

    qts = ide_test_start("%s", "");

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    /* FLUSH CACHE command on device 0*/
    qpci_io_writeb(dev, ide_bar, reg_device, 0);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_FLUSH_CACHE);

    /* Just testing that qemu doesn't crash... */

    free_pci_device(dev);
    ide_test_quit(qts);
}

static void test_flush_empty_drive(void)
{
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;

    qts = ide_test_start("-device ide-cd,bus=ide.0");
    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    /* FLUSH CACHE command on device 0 */
    qpci_io_writeb(dev, ide_bar, reg_device, 0);
    qpci_io_writeb(dev, ide_bar, reg_command, CMD_FLUSH_CACHE);

    /* Just testing that qemu doesn't crash... */

    free_pci_device(dev);
    ide_test_quit(qts);
}

typedef struct Read10CDB {
    uint8_t opcode;
    uint8_t flags;
    uint32_t lba;
    uint8_t reserved;
    uint16_t nblocks;
    uint8_t control;
    uint16_t padding;
} __attribute__((__packed__)) Read10CDB;

static void send_scsi_cdb_read10(QPCIDevice *dev, QPCIBar ide_bar,
                                 uint64_t lba, int nblocks)
{
    Read10CDB pkt = { .padding = 0 };
    int i;

    g_assert_cmpint(lba, <=, UINT32_MAX);
    g_assert_cmpint(nblocks, <=, UINT16_MAX);
    g_assert_cmpint(nblocks, >=, 0);

    /* Construct SCSI CDB packet */
    pkt.opcode = 0x28;
    pkt.lba = cpu_to_be32(lba);
    pkt.nblocks = cpu_to_be16(nblocks);

    /* Send Packet */
    for (i = 0; i < sizeof(Read10CDB)/2; i++) {
        qpci_io_writew(dev, ide_bar, reg_data,
                       le16_to_cpu(((uint16_t *)&pkt)[i]));
    }
}

typedef struct ReadCDCDB {
    uint8_t opcode;
    uint8_t sector_type;
    uint32_t lba;
    uint8_t length[3];
    uint8_t main_channel;
    uint8_t sub_channel;
    uint8_t control;
} __attribute__((__packed__)) ReadCDCDB;

static void send_scsi_cdb_read_cd(QPCIDevice *dev, QPCIBar ide_bar,
                                  uint64_t lba, int nblocks)
{
    ReadCDCDB pkt = { };
    int i;

    g_assert_cmpint(lba, <=, UINT32_MAX);
    g_assert_cmpint(nblocks, >=, 0);
    g_assert_cmpint(nblocks, <=, 0xffffff);

    /* Construct SCSI CDB packet */
    pkt.opcode = 0xbe;
    pkt.lba = cpu_to_be32(lba);
    pkt.length[0] = (nblocks >> 16) & 0xff;
    pkt.length[1] = (nblocks >> 8) & 0xff;
    pkt.length[2] = nblocks & 0xff;
    pkt.main_channel = 0xf8; /* sync + headers + user data + EDC/ECC: 2352 */

    /* Send Packet */
    for (i = 0; i < sizeof(ReadCDCDB) / 2; i++) {
        qpci_io_writew(dev, ide_bar, reg_data,
                       le16_to_cpu(((uint16_t *)&pkt)[i]));
    }
}

static void nsleep(QTestState *qts, int64_t nsecs)
{
    const struct timespec val = { .tv_nsec = nsecs };
    nanosleep(&val, NULL);
    qtest_clock_set(qts, nsecs);
}

static uint8_t ide_wait_clear(QTestState *qts, uint8_t flag)
{
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    uint8_t data;
    time_t st;

    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);

    /* Wait with a 5 second timeout */
    time(&st);
    while (true) {
        data = qpci_io_readb(dev, ide_bar, reg_status);
        if (!(data & flag)) {
            free_pci_device(dev);
            return data;
        }
        if (difftime(time(NULL), st) > 5.0) {
            break;
        }
        nsleep(qts, 400);
    }
    g_assert_not_reached();
}

static void ide_wait_intr(QTestState *qts, int irq)
{
    time_t st;
    bool intr;

    time(&st);
    while (true) {
        intr = qtest_get_irq(qts, irq);
        if (intr) {
            return;
        }
        if (difftime(time(NULL), st) > 5.0) {
            break;
        }
        nsleep(qts, 400);
    }

    g_assert_not_reached();
}

#define CDROM_PIO 0
#define CDROM_DMA (1 << 0)
#define CDROM_RAW (1 << 1)
#define CDROM_CUE (1 << 2)
#define CDROM_CUE_2048 (1 << 3)

static void cdrom_read_impl(int nblocks, unsigned flags)
{
    bool dma = flags & CDROM_DMA;
    bool raw = flags & CDROM_RAW;
    bool cue = flags & CDROM_CUE;
    bool cue_2048 = flags & CDROM_CUE_2048;
    g_autofree char *cue_path = g_strconcat(tmp_path[0], ".cue", NULL);
    QTestState *qts;
    QPCIDevice *dev;
    QPCIBar bmdma_bar, ide_bar;
    FILE *fh;
    int patt_blocks = MAX(16, nblocks);
    size_t patt_len = ATAPI_BLOCK_SIZE * patt_blocks;
    char *pattern = g_malloc(patt_len);
    unsigned xfer = raw ? ATAPI_RAW_SIZE : ATAPI_BLOCK_SIZE;
    size_t rxsize = xfer * nblocks;
    uint16_t *rx = g_malloc0(rxsize);
    void (*send_cdb)(QPCIDevice *, QPCIBar, uint64_t, int) =
        raw ? send_scsi_cdb_read_cd : send_scsi_cdb_read10;
    int i, j;
    uint8_t data;
    uint16_t limit;
    size_t ret;

    /* Prepopulate the CDROM with an interesting pattern */
    generate_pattern(pattern, patt_len, ATAPI_BLOCK_SIZE);
    fh = fopen(tmp_path[0], "wb+");
    if (cue && !cue_2048) {
        uint8_t sector[2352] = { 0 };
        memset(sector + 1, 0xff, 10);
        sector[15] = 1;
        memset(sector + 2064, 0x93, 288);
        for (i = 0; i < patt_blocks; i++) {
            memcpy(sector + 16, pattern + i * 2048, 2048);
            g_assert_cmpint(fwrite(sector, 1, sizeof(sector), fh), ==, 2352);
        }
    } else {
        ret = fwrite(pattern, ATAPI_BLOCK_SIZE, patt_blocks, fh);
        g_assert_cmpint(ret, ==, patt_blocks);
    }
    fclose(fh);
    if (cue) {
        g_autofree char *contents = g_strdup_printf(
            "FILE \"%s\" BINARY\nTRACK 01 MODE1/%u\nINDEX 01 00:00:00\n",
            tmp_path[0], cue_2048 ? 2048 : 2352);
        g_assert_true(g_file_set_contents(cue_path, contents, -1, NULL));
    }

    qts = ide_test_start(
            "-drive if=none,file=%s,media=cdrom,format=%s,id=sr0,index=0 "
            "-device ide-cd,drive=sr0,bus=ide.0", cue ? cue_path : tmp_path[0],
            cue ? "cue" : "raw");
    dev = get_pci_device(qts, &bmdma_bar, &ide_bar);
    qtest_irq_intercept_in(qts, "ioapic");

    if (dma) {
        uintptr_t guest_buf = guest_alloc(&guest_malloc, rxsize);
        PrdtEntry prdt[1];

        prdt[0].addr = cpu_to_le32(guest_buf);
        prdt[0].size = cpu_to_le32(rxsize | PRDT_EOT);

        send_dma_request_dev(qts, dev, bmdma_bar, ide_bar, CMD_PACKET, 0,
                             nblocks, prdt, ARRAY_SIZE(prdt), send_cdb);

        qtest_memread(qts, guest_buf, rx, rxsize);
    } else {
        /* PACKET command on device 0 */
        qpci_io_writeb(dev, ide_bar, reg_device, 0);
        qpci_io_writeb(dev, ide_bar, reg_lba_middle, BYTE_COUNT_LIMIT & 0xFF);
        qpci_io_writeb(dev, ide_bar, reg_lba_high,
                       (BYTE_COUNT_LIMIT >> 8 & 0xFF));
        qpci_io_writeb(dev, ide_bar, reg_command, CMD_PACKET);
        /* HP0: Check_Status_A State */
        nsleep(qts, 400);
        data = ide_wait_clear(qts, BSY);
        /* HP1: Send_Packet State */
        assert_bit_set(data, DRQ | DRDY);
        assert_bit_clear(data, ERR | DF | BSY);

        send_cdb(dev, ide_bar, 0, nblocks);

        /*
         * Read data back: occurs in bursts of 'BYTE_COUNT_LIMIT' bytes.
         * If BYTE_COUNT_LIMIT is odd, we transfer BYTE_COUNT_LIMIT - 1 bytes.
         * We allow an odd limit only when the remaining transfer size is
         * less than BYTE_COUNT_LIMIT. However, SCSI's read10 command can only
         * request n blocks, so our request size is always even.
         * For this reason, we assume there is never a hanging byte to fetch.
         */
        g_assert(!(rxsize & 1));
        limit = BYTE_COUNT_LIMIT & ~1;
        for (i = 0; i < DIV_ROUND_UP(rxsize, limit); i++) {
            size_t offset = i * (limit / 2);
            size_t rem = (rxsize / 2) - offset;

            /* HP3: INTRQ_Wait */
            ide_wait_intr(qts, IDE_PRIMARY_IRQ);

            /* HP2: Check_Status_B (and clear IRQ) */
            data = ide_wait_clear(qts, BSY);
            assert_bit_set(data, DRQ | DRDY);
            assert_bit_clear(data, ERR | DF | BSY);

            /* HP4: Transfer_Data */
            for (j = 0; j < MIN((limit / 2), rem); j++) {
                rx[offset + j] = cpu_to_le16(qpci_io_readw(dev, ide_bar,
                                                           reg_data));
            }
        }

        /* Check for final completion IRQ */
        ide_wait_intr(qts, IDE_PRIMARY_IRQ);

        /* Sanity check final state */
        data = ide_wait_clear(qts, DRQ);
        assert_bit_set(data, DRDY);
        assert_bit_clear(data, DRQ | ERR | DF | BSY);
    }

    if (raw) {
        /* The 2048-byte payload of each raw sector sits past its header. */
        for (i = 0; i < nblocks; i++) {
            uint8_t *sec = (uint8_t *)rx + i * ATAPI_RAW_SIZE + ATAPI_RAW_DATA;

            g_assert_cmpint(memcmp(sec, pattern + i * ATAPI_BLOCK_SIZE,
                                   ATAPI_BLOCK_SIZE), ==, 0);
            if (cue && !cue_2048) {
                /* Raw BIN EDC/ECC bytes must survive; do not synthesize ISO. */
                for (j = 2048; j < 2336; j++) {
                    g_assert_cmphex(sec[j], ==, 0x93);
                }
            }
        }
    } else {
        g_assert_cmpint(memcmp(pattern, rx, rxsize), ==, 0);
    }

    g_free(pattern);
    g_free(rx);
    test_bmdma_teardown(qts);
    free_pci_device(dev);
    unlink(cue_path);
}

static void test_cdrom_cue_pio(void)
{
    cdrom_read_impl(16, CDROM_CUE);
}

static void test_cdrom_cue_dma(void)
{
    cdrom_read_impl(16, CDROM_CUE | CDROM_DMA);
}

static void test_cdrom_cue_pio_raw(void)
{
    cdrom_read_impl(16, CDROM_CUE | CDROM_RAW);
}

static void test_cdrom_cue_dma_raw(void)
{
    cdrom_read_impl(16, CDROM_CUE | CDROM_DMA | CDROM_RAW);
}

static void test_cdrom_cue_2048(void)
{
    cdrom_read_impl(16, CDROM_CUE | CDROM_CUE_2048 | CDROM_PIO);
    cdrom_read_impl(16, CDROM_CUE | CDROM_CUE_2048 | CDROM_DMA | CDROM_RAW);
}

static int cdrom_packet(QTestState *qts, QPCIDevice *dev, QPCIBar bar,
                         const uint8_t packet[12], uint8_t *out, unsigned capacity)
{
    unsigned done = 0;
    qpci_io_writeb(dev, bar, reg_device, 0);
    qpci_io_writeb(dev, bar, reg_feature, 0);
    qpci_io_writeb(dev, bar, reg_lba_middle, 0);
    qpci_io_writeb(dev, bar, reg_lba_high, 4);
    qpci_io_writeb(dev, bar, reg_command, CMD_PACKET);
    assert_bit_set(ide_wait_clear(qts, BSY), DRQ);
    for (unsigned i = 0; i < 12; i += 2) {
        qpci_io_writew(dev, bar, reg_data, lduw_le_p(packet + i));
    }
    for (;;) {
        uint8_t status = ide_wait_clear(qts, BSY);
        if (status & ERR) {
            return -1;
        }
        if (!(status & DRQ)) {
            return done;
        }
        unsigned n = qpci_io_readb(dev, bar, reg_lba_middle) |
                     qpci_io_readb(dev, bar, reg_lba_high) << 8;
        g_assert_cmpuint(n, >, 0);
        g_assert_cmpuint(n, <=, capacity - done);
        for (unsigned i = 0; i < n; i += 2) {
            uint16_t value = qpci_io_readw(dev, bar, reg_data);
            out[done++] = value;
            if (i + 1 < n) {
                out[done++] = value >> 8;
            }
        }
    }
}

static int cdrom_set_audio_page(QTestState *qts, QPCIDevice *dev, QPCIBar bar,
                                uint8_t left, uint8_t right)
{
    uint8_t packet[12] = { 0x55, 0x10, 0, 0, 0, 0, 0, 0, 24 };
    uint8_t page[24] = { 0 };
    page[8] = 0x0e; page[9] = 14; page[10] = 4;
    page[16] = 1; page[17] = left; page[18] = 2; page[19] = right;
    qpci_io_writeb(dev, bar, reg_device, 0);
    qpci_io_writeb(dev, bar, reg_feature, 0);
    qpci_io_writeb(dev, bar, reg_lba_middle, 24);
    qpci_io_writeb(dev, bar, reg_lba_high, 0);
    qpci_io_writeb(dev, bar, reg_command, CMD_PACKET);
    assert_bit_set(ide_wait_clear(qts, BSY), DRQ);
    for (unsigned i = 0; i < sizeof(packet); i += 2) {
        qpci_io_writew(dev, bar, reg_data, lduw_le_p(packet + i));
    }
    assert_bit_set(ide_wait_clear(qts, BSY), DRQ);
    g_assert_cmpint(qpci_io_readb(dev, bar, reg_nsectors) & 3, ==, 0);
    for (unsigned i = 0; i < sizeof(page); i += 2) {
        qpci_io_writew(dev, bar, reg_data, lduw_le_p(page + i));
    }
    return ide_wait_clear(qts, BSY) & ERR ? -1 : 0;
}

static void test_cdrom_cue_audio(void)
{
    bool snapshot = have_qemu_img();
    g_autofree char *state = g_strconcat(tmp_path[0], ".qcow2", NULL);
    g_autofree char *state_args = snapshot ?
        g_strdup_printf("-drive file=%s,format=qcow2,if=ide,index=1", state) :
        g_strdup("");
    g_autofree char *cue = g_strconcat(tmp_path[0], ".cue", NULL);
    g_autofree char *contents = g_strdup_printf(
        "FILE \"%s\" BINARY\nTRACK 01 MODE1/2352\nINDEX 01 00:00:00\n"
        "TRACK 02 AUDIO\nINDEX 01 00:00:01\n", tmp_path[0]);
    g_autofree uint8_t *bin = g_malloc0(65 * 2352);
    QPCIBar bm, bar;
    QTestState *qts;
    QPCIDevice *dev;
    uint8_t out[128], packet[12] = { 0x4b, 0, 0, 0, 0, 0, 0, 0, 1 };
    uint8_t sub[12] = { 0x42, 0, 0x40, 1, 0, 0, 0, 0, 16 };
    uint8_t play[12] = { 0x45, 0, 0, 0, 0, 1, 0, 0, 64 };
    uint8_t page[12] = { 0x5a, 0, 0x0e, 0, 0, 0, 0, 0, 24 };
    bin[15] = 1;
    g_assert_true(g_file_set_contents(tmp_path[0], (char *)bin, 65 * 2352, NULL));
    g_assert_true(g_file_set_contents(cue, contents, -1, NULL));
    if (snapshot) {
        g_assert_true(mkimg(state, "qcow2", 1));
    }
    qts = ide_test_start("-audiodev none,id=cdsound "
        "-global ide-cd.audiodev=cdsound -drive if=ide,index=0,media=cdrom,"
        "file=%s,format=cue,readonly=on %s", cue, state_args);
    dev = get_pci_device(qts, &bm, &bar);
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, sizeof(out)), ==, -1);
    g_assert_cmpint(cdrom_packet(qts, dev, bar, page, out, sizeof(out)), ==, 24);
    g_assert_cmpint(out[10], ==, 4);
    g_assert_cmpint(out[17], ==, 255);
    g_assert_cmpint(cdrom_set_audio_page(qts, dev, bar, 0, 128), ==, 0);
    g_assert_cmpint(cdrom_packet(qts, dev, bar, page, out, sizeof(out)), ==, 24);
    g_assert_cmpint(out[17], ==, 0);
    g_assert_cmpint(out[19], ==, 128);
    play[5] = 0; /* A data track cannot become speaker noise. */
    g_assert_cmpint(cdrom_packet(qts, dev, bar, play, out, sizeof(out)), ==, -1);
    play[5] = 1;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, play, out, sizeof(out)), ==, 0);
    packet[8] = 0;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, sizeof(out)), ==, 0);
    qtest_clock_step(qts, 100000000);
    g_assert_cmpint(cdrom_packet(qts, dev, bar, sub, out, sizeof(out)), ==, 16);
    g_assert_cmpint(out[1], ==, 0x12);
    g_assert_cmpuint(ldl_be_p(out + 8), ==, 1);
    if (snapshot) {
        g_autofree char *reply = qtest_hmp(qts, "savevm cd-paused");
        g_assert_cmpstr(reply, ==, "");
    }
    packet[8] = 1;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, sizeof(out)), ==, 0);
    for (unsigned i = 0; i < 200; i++) {
        qtest_clock_step(qts, 10000000);
        g_assert_cmpint(cdrom_packet(qts, dev, bar, sub, out, sizeof(out)), ==, 16);
        if (out[1] == 0x13) {
            break;
        }
    }
    g_assert_cmpint(out[1], ==, 0x13);
    g_assert_cmpuint(ldl_be_p(out + 8), ==, 65);
    if (snapshot) {
        g_autofree char *reply = qtest_hmp(qts, "loadvm cd-paused");
        g_assert_cmpstr(reply, ==, "");
        qtest_clock_step(qts, 100000000);
        g_assert_cmpint(cdrom_packet(qts, dev, bar, sub, out, sizeof(out)), ==, 16);
        g_assert_cmpint(out[1], ==, 0x12);
        g_assert_cmpuint(ldl_be_p(out + 8), ==, 1);
        g_assert_cmpint(cdrom_packet(qts, dev, bar, page, out, sizeof(out)), ==, 24);
        g_assert_cmpint(out[17], ==, 0);
        g_assert_cmpint(out[19], ==, 128);
        packet[8] = 1;
        g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, sizeof(out)), ==, 0);
        for (unsigned i = 0; i < 200; i++) {
            qtest_clock_step(qts, 10000000);
            g_assert_cmpint(cdrom_packet(qts, dev, bar, sub, out, sizeof(out)), ==, 16);
            if (out[1] == 0x13) {
                break;
            }
        }
        g_assert_cmpint(out[1], ==, 0x13);
        g_assert_cmpuint(ldl_be_p(out + 8), ==, 65);
    }
    memset(play, 0, sizeof(play));
    play[0] = 0x47; play[4] = 2; play[5] = 1; play[7] = 2; play[8] = 65;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, play, out, sizeof(out)), ==, 0);
    /* Reset cancels any outstanding prefetch before its storage is reused. */
    qtest_outb(qts, IDE_BASE2, IDE_CTRL_RESET);
    qtest_outb(qts, IDE_BASE2, 0);
    qtest_clock_step(qts, 100000000);
    g_assert_cmpint(cdrom_packet(qts, dev, bar, sub, out, sizeof(out)), ==, 16);
    g_assert_cmpint(out[1], ==, 0x15);
    g_assert_cmpuint(ldl_be_p(out + 8), ==, 0);
    free_pci_device(dev);
    ide_test_quit(qts);
    unlink(cue);
    if (snapshot) {
        unlink(state);
    }
}

static void cdrom_cue_audio_pcm(bool mute_left, bool mixing)
{
    g_autofree char *cue = g_strconcat(tmp_path[0], ".cue", NULL);
    g_autofree char *wav = g_strconcat(tmp_path[0], ".wav", NULL);
    g_autofree char *contents = g_strdup_printf(
        "FILE \"%s\" BINARY\nTRACK 01 AUDIO\nINDEX 01 00:00:00\n", tmp_path[0]);
    g_autofree uint8_t *bin = g_malloc(32 * 2352);
    g_autofree char *pcm = NULL;
    gsize length;
    QPCIBar bm, bar;
    QTestState *qts;
    QPCIDevice *dev;
    uint8_t out[16], play[12] = { 0x45, 0, 0, 0, 0, 0, 0, 0, 32 };
    uint8_t sub[12] = { 0x42, 0, 0x40, 1, 0, 0, 0, 0, 16 };
    for (unsigned i = 0; i < 32 * 2352; i += 4) {
        stw_le_p(bin + i, 1234);
        stw_le_p(bin + i + 2, (uint16_t)-2345);
    }
    g_assert_true(g_file_set_contents(tmp_path[0], (char *)bin, 32 * 2352, NULL));
    g_assert_true(g_file_set_contents(cue, contents, -1, NULL));
    qts = ide_test_start("-audiodev wav,id=cdsound,path=%s,"
        "%s "
        "-global ide-cd.audiodev=cdsound -drive if=ide,index=0,media=cdrom,"
        "file=%s,format=cue,readonly=on", wav,
        mixing ? "out.frequency=44100,out.channels=2,out.format=s16" :
                 "out.mixing-engine=off", cue);
    dev = get_pci_device(qts, &bm, &bar);
    if (mute_left) {
        g_assert_cmpint(cdrom_set_audio_page(qts, dev, bar, 0, 255), ==, 0);
    }
    g_assert_cmpint(cdrom_packet(qts, dev, bar, play, out, sizeof(out)), ==, 0);
    for (unsigned i = 0; i < 100; i++) {
        qtest_clock_step(qts, 10000000);
        g_assert_cmpint(cdrom_packet(qts, dev, bar, sub, out, sizeof(out)), ==, 16);
    }
    g_assert_cmpint(out[1], ==, 0x13);
    free_pci_device(dev);
    ide_test_quit(qts);
    g_assert_true(g_file_get_contents(wav, &pcm, &length, NULL));
    g_assert_cmpuint(length, >=, 44 + 32 * 2352);
    unsigned audible = 0;
    for (unsigned i = 44; i + 4 <= length; i += 4) {
        int16_t left = lduw_le_p(pcm + i), right = lduw_le_p(pcm + i + 2);
        if (left || right) {
            g_assert_cmpint(left, ==, mute_left ? 0 : 1234);
            g_assert_cmpint(right, ==, -2345);
            audible++;
        }
    }
    g_assert_cmpuint(audible, ==, 32 * 588);
    unlink(wav);
    unlink(cue);
}

static void test_cdrom_cue_audio_pcm(void)
{
    cdrom_cue_audio_pcm(false, true);
}

static void test_cdrom_cue_audio_pcm_mute(void)
{
    cdrom_cue_audio_pcm(true, true);
}

static void test_cdrom_cue_audio_pcm_direct(void)
{
    cdrom_cue_audio_pcm(false, false);
}

static void test_cdrom_cue_toc(void)
{
    g_autofree char *cue = g_strconcat(tmp_path[0], ".cue", NULL);
    g_autofree char *contents = g_strdup_printf(
        "FILE \"%s\" BINARY\r\nTRACK 01 MODE1/2352\r\n"
        "INDEX 01 00:00:00\r\nTRACK 02 AUDIO\r\nPREGAP 00:00:03\r\n"
        "INDEX 01 00:00:40\r\nTRACK 03 AUDIO\r\n"
        "INDEX 00 00:00:48\r\nINDEX 01 00:00:50\r\n", tmp_path[0]);
    g_autofree uint8_t *bin = g_malloc(60 * 2352);
    g_autofree uint8_t *out = g_malloc(16 * 2352);
    QPCIBar bm, bar;
    QTestState *qts;
    QPCIDevice *dev;
    uint8_t packet[12] = { 0x43, 0, 0, 0, 0, 0, 0, 0, 128 };

    for (unsigned i = 0; i < 60; i++) {
        memset(bin + i * 2352, i + 1, 2352);
        if (i < 40) {
            bin[i * 2352 + 15] = 1;
        }
    }
    g_assert_true(g_file_set_contents(tmp_path[0], (char *)bin, 60 * 2352, NULL));
    g_assert_true(g_file_set_contents(cue, contents, -1, NULL));
    qts = ide_test_start("-drive if=none,id=disc,file=%s,format=cue,readonly=on "
                         "-device ide-cd,drive=disc,bus=ide.0", cue);
    dev = get_pci_device(qts, &bm, &bar);
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 128), ==, 36);
    g_assert_cmpint(lduw_be_p(out), ==, 34);
    g_assert_cmpint(out[2], ==, 1);
    g_assert_cmpint(out[3], ==, 3);
    for (unsigned i = 0; i < 4; i++) {
        const uint32_t starts[] = { 0, 43, 53, 63 };
        g_assert_cmpint(out[5 + i * 8], ==, i ? 0x10 : 0x14);
        g_assert_cmpint(out[6 + i * 8], ==, i == 3 ? 0xaa : i + 1);
        g_assert_cmpuint(ldl_be_p(out + 8 + i * 8), ==, starts[i]);
    }
    packet[1] = 2;
    packet[6] = 2;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 128), ==, 28);
    g_assert_cmpint(out[6], ==, 2);
    g_assert_cmpint(out[9], ==, 0);
    g_assert_cmpint(out[10], ==, 2);
    g_assert_cmpint(out[11], ==, 43);
    packet[6] = 4;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 128), ==, -1);
    packet[6] = 0;
    packet[2] = 2;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 128), ==, 70);
    g_assert_cmpint(out[7], ==, 0xa0);
    g_assert_cmpint(out[18], ==, 0xa1);
    g_assert_cmpint(out[23], ==, 3);

    /* Exact MCI open request observed from the unmodified Win98 driver. */
    memset(packet, 0, sizeof(packet));
    packet[0] = 0x42; packet[1] = 2; packet[2] = 0x40;
    packet[3] = 1; packet[8] = 16;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 128), ==, 16);
    g_assert_cmpint(out[1], ==, 0x15);
    g_assert_cmpint(out[5], ==, 0x14);
    g_assert_cmpint(out[6], ==, 1);
    g_assert_cmpint(out[7], ==, 1);
    g_assert_cmpint(out[10], ==, 2);
    g_assert_cmpuint(ldl_be_p(out + 12), ==, 0);

    /* Read across stored data, generated silence and original CDDA bytes. */
    memset(packet, 0, sizeof(packet));
    packet[0] = 0xbe; packet[5] = 39; packet[8] = 6; packet[9] = 0xf8;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 16 * 2352), ==,
                    6 * 2352);
    g_assert_cmpmem(out, 2352, bin + 39 * 2352, 2352);
    for (unsigned i = 2352; i < 4 * 2352; i++) {
        g_assert_cmpint(out[i], ==, 0);
    }
    g_assert_cmpmem(out + 4 * 2352, 2 * 2352, bin + 40 * 2352, 2 * 2352);
    packet[1] = 4; packet[5] = 43; packet[8] = 3; packet[9] = 0x10;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 16 * 2352), ==,
                    3 * 2352);
    g_assert_cmpmem(out, 3 * 2352, bin + 40 * 2352, 3 * 2352);
    packet[1] = 8;
    g_assert_cmpint(cdrom_packet(qts, dev, bar, packet, out, 16 * 2352), ==, -1);
    free_pci_device(dev);
    ide_test_quit(qts);
    unlink(cue);
}

/* Parsing must reject inconsistent descriptors before exposing any medium. */
static void test_cdrom_cue_invalid(void)
{
    static const char *invalid[] = {
        "TRACK 01 MODE1/2352\n", /* Missing INDEX. */
        "TRACK 01 MODE1/2352\nINDEX 01 00:00:75\n",
        "TRACK 02 MODE1/2352\nINDEX 01 00:00:00\n",
        "TRACK 01 MODE2/2352\nINDEX 01 00:00:00\n",
        "TRACK 01 MODE1/2352\nINDEX 01 00:00:00\n"
        "TRACK 02 AUDIO\nINDEX 01 00:00:02\n", /* Past BIN end. */
        "TRACK 01 MODE1/2352\nINDEX 01 00:00:00\n"
        "TRACK 02 AUDIO\nINDEX 01 00:00:00\n", /* Empty track. */
        "TRACK 01 MODE1/2352\nINDEX 01 00:00:00\n"
        "TRACK 02 MODE1/2352\nPREGAP 00:02:00\nINDEX 01 00:00:01\n",
        "FILE \"second.bin\" BINARY\n",
    };
    g_autofree char *cue = g_strconcat(tmp_path[0], ".cue", NULL);
    uint8_t bin[2352 * 2] = { 0 };
    QTestState *qts = ide_test_start("");

    g_assert_true(g_file_set_contents(tmp_path[0], (char *)bin, sizeof(bin), NULL));
    for (unsigned i = 0; i < ARRAY_SIZE(invalid) + 2; i++) {
        g_autofree char *contents = g_strdup_printf("FILE \"%s\" BINARY\n%s",
            tmp_path[0], i < ARRAY_SIZE(invalid) ? invalid[i] :
            "TRACK 01 MODE1/2352\nINDEX 01 00:00:00\n");
        size_t length = strlen(contents);
        if (i == ARRAY_SIZE(invalid)) {
            contents[length / 2] = 0; /* Embedded NUL cannot hide directives. */
        } else if (i == ARRAY_SIZE(invalid) + 1) {
            g_assert_true(g_file_set_contents(tmp_path[0], (char *)bin,
                                              sizeof(bin) - 1, NULL));
        }
        g_assert_true(g_file_set_contents(cue, contents, length, NULL));
        QDict *error = qtest_qmp_assert_failure_ref(qts,
            "{'execute':'blockdev-add','arguments':{'driver':'cue',"
            "'node-name':'bad','read-only':true,'file':{'driver':'file',"
            "'filename':%s}}}", cue);
        qobject_unref(error);
    }
    ide_test_quit(qts);
    unlink(cue);
}

static void test_cdrom_pio(void)
{
    cdrom_read_impl(1, CDROM_PIO);
}

static void test_cdrom_pio_large(void)
{
    /* Test a few loops of the PIO DRQ mechanism. */
    cdrom_read_impl(BYTE_COUNT_LIMIT * 4 / ATAPI_BLOCK_SIZE, CDROM_PIO);
}

static void test_cdrom_dma(void)
{
    cdrom_read_impl(1, CDROM_DMA);
}

static void test_cdrom_dma_large(void)
{
    cdrom_read_impl(BYTE_COUNT_LIMIT * 4 / ATAPI_BLOCK_SIZE, CDROM_DMA);
}

static void test_cdrom_pio_raw(void)
{
    cdrom_read_impl(4, CDROM_RAW);
}

static void test_cdrom_dma_raw(void)
{
    cdrom_read_impl(4, CDROM_DMA | CDROM_RAW);
}

int main(int argc, char **argv)
{
    const char *base;
    int i;
    int fd;
    int ret;

    /*
     * "base" stores the starting point where we create temporary files.
     *
     * On Windows, this is set to the relative path of current working
     * directory, because the absolute path causes the blkdebug filename
     * parser fail to parse "blkdebug:path/to/config:path/to/image".
     */
#ifndef _WIN32
    base = g_get_tmp_dir();
#else
    base = ".";
#endif

    /* Create temporary blkdebug instructions */
    debug_path = g_strdup_printf("%s/qtest-blkdebug.XXXXXX", base);
    fd = g_mkstemp(debug_path);
    g_assert(fd >= 0);
    close(fd);

    /* Create a temporary raw image */
    for (i = 0; i < 2; ++i) {
        tmp_path[i] = g_strdup_printf("%s/qtest.XXXXXX", base);
        fd = g_mkstemp(tmp_path[i]);
        g_assert(fd >= 0);
        ret = ftruncate(fd, TEST_IMAGE_SIZE);
        g_assert(ret == 0);
        close(fd);
    }

    /* Run the tests */
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("/ide/read_native", test_specify);

    qtest_add_func("/ide/identify", test_identify);

    qtest_add_func("/ide/diagnostic", test_diagnostic);

    qtest_add_func("/ide/bmdma/simple_rw", test_bmdma_simple_rw);
    qtest_add_func("/ide/bmdma/trim", test_bmdma_trim);
    qtest_add_func("/ide/bmdma/trim_reset", test_bmdma_trim_reset);
    qtest_add_func("/ide/bmdma/various_prdts", test_bmdma_various_prdts);
    qtest_add_func("/ide/bmdma/no_busmaster", test_bmdma_no_busmaster);

    qtest_add_func("/ide/flush", test_flush);
    qtest_add_func("/ide/flush/nodev", test_flush_nodev);
    qtest_add_func("/ide/flush/empty_drive", test_flush_empty_drive);
    qtest_add_func("/ide/flush/retry_pci", test_pci_retry_flush);

    qtest_add_func("/ide/cdrom/pio", test_cdrom_pio);
    qtest_add_func("/ide/cdrom/pio_large", test_cdrom_pio_large);
    qtest_add_func("/ide/cdrom/dma", test_cdrom_dma);
    qtest_add_func("/ide/cdrom/dma_large", test_cdrom_dma_large);
    qtest_add_func("/ide/cdrom/pio_raw", test_cdrom_pio_raw);
    qtest_add_func("/ide/cdrom/dma_raw", test_cdrom_dma_raw);
    qtest_add_func("/ide/cdrom/cue/pio", test_cdrom_cue_pio);
    qtest_add_func("/ide/cdrom/cue/dma", test_cdrom_cue_dma);
    qtest_add_func("/ide/cdrom/cue/pio_raw", test_cdrom_cue_pio_raw);
    qtest_add_func("/ide/cdrom/cue/dma_raw", test_cdrom_cue_dma_raw);
    qtest_add_func("/ide/cdrom/cue/toc", test_cdrom_cue_toc);
    qtest_add_func("/ide/cdrom/cue/mode1_2048", test_cdrom_cue_2048);
    qtest_add_func("/ide/cdrom/cue/invalid", test_cdrom_cue_invalid);
    qtest_add_func("/ide/cdrom/cue/audio", test_cdrom_cue_audio);
    qtest_add_func("/ide/cdrom/cue/audio_pcm", test_cdrom_cue_audio_pcm);
    qtest_add_func("/ide/cdrom/cue/audio_pcm_mute", test_cdrom_cue_audio_pcm_mute);
    qtest_add_func("/ide/cdrom/cue/audio_pcm_direct", test_cdrom_cue_audio_pcm_direct);

    ret = g_test_run();

    /* Cleanup */
    for (i = 0; i < 2; ++i) {
        unlink(tmp_path[i]);
        g_free(tmp_path[i]);
    }
    unlink(debug_path);
    g_free(debug_path);

    return ret;
}
