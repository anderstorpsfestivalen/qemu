/*
 * Exercise physical-memory invalidation with real translated x86 code.
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "qemu/osdep.h"
#include "libqtest.h"

#define REQUEST 0x800
#define ACK 0x802
#define ENTRY 0x804
#define RESULT 0x806

typedef struct Fixture {
    QTestState *qts;
    char *bios;
    uint16_t sequence;
} Fixture;

static void setup(Fixture *f, gconstpointer threading)
{
    /*
     * All data and the stack occupy pages separate from the code under test.
     * This small dispatcher runs only until the test observes its reply and
     * stops the VM; each request has a bounded host deadline.
     */
    static const uint8_t dispatch[] = {
        0xfa,                         /* cli */
        0x31, 0xc0,                   /* xor ax, ax */
        0x8e, 0xd8,                   /* mov ds, ax */
        0x8e, 0xd0,                   /* mov ss, ax */
        0xbc, 0x00, 0x80,             /* mov sp, 0x8000 */
        0x8b, 0x1e, 0x00, 0x08,       /* loop: mov bx, [REQUEST] */
        0x3b, 0x1e, 0x02, 0x08,       /* cmp bx, [ACK] */
        0x74, 0xf6,                   /* je loop */
        0xff, 0x16, 0x04, 0x08,       /* call [ENTRY] */
        0xa3, 0x06, 0x08,             /* mov [RESULT], ax */
        0x89, 0x1e, 0x02, 0x08,       /* mov [ACK], bx */
        0xeb, 0xe9,                   /* jmp loop */
    };
    g_autofree uint8_t *rom = g_malloc0(65536);
    int fd = g_file_open_tmp("qemu-tcg-invalidate-XXXXXX", &f->bios, NULL);

    g_assert_cmpint(fd, >=, 0);
    close(fd);
    memcpy(rom + 0xfff0, "\xea\x00\x10\x00\x00", 5);
    g_assert_true(g_file_set_contents(f->bios, (char *)rom, 65536, NULL));
    f->qts = qtest_initf("-machine pc -nodefaults -accel tcg,thread=%s "
                         "-S -bios %s", (const char *)threading, f->bios);
    qtest_memwrite(f->qts, 0x1000, dispatch, sizeof(dispatch));
    f->sequence = 0;
}

static void teardown(Fixture *f, gconstpointer unused)
{
    qtest_quit(f->qts);
    unlink(f->bios);
    g_free(f->bios);
}

static void function(Fixture *f, uint16_t address, uint16_t result)
{
    const uint8_t code[] = { 0xb8, result, result >> 8, 0xc3 };

    qtest_memwrite(f->qts, address, code, sizeof(code));
}

static void run_function(Fixture *f, uint16_t address, uint16_t expected)
{
    int64_t deadline = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;

    qtest_writew(f->qts, ENTRY, address);
    qtest_writew(f->qts, REQUEST, ++f->sequence);
    qtest_qmp_assert_success(f->qts, "{'execute':'cont'}");
    while (qtest_readw(f->qts, ACK) != f->sequence) {
        g_assert_cmpint(g_get_monotonic_time(), <, deadline);
        g_usleep(1000);
    }
    qtest_qmp_assert_success(f->qts, "{'execute':'stop'}");
    qtest_qmp_eventwait(f->qts, "STOP");
    g_assert_cmphex(qtest_readw(f->qts, RESULT), ==, expected);
}

static unsigned invalidations(Fixture *f)
{
    g_autofree char *info = qtest_hmp(f->qts, "info jit");
    const char *line = strstr(info, "TB invalidate count ");
    unsigned count;

    g_assert_nonnull(line);
    g_assert_cmpint(sscanf(line, "TB invalidate count %u", &count), ==, 1);
    return count;
}

static void same_page(Fixture *f, gconstpointer unused)
{
    unsigned before;

    function(f, 0x2100, 0x1234);
    run_function(f, 0x2100, 0x1234);
    before = invalidations(f);

    /*
     * VAPIC's data and handlers share a page: updating later data must not
     * evict earlier translated instructions. This fails before the fix.
     */
    qtest_writeb(f->qts, 0x2800, 1);
    g_assert_cmpuint(invalidations(f), ==, before);
    run_function(f, 0x2100, 0x1234);
    g_assert_cmpuint(invalidations(f), ==, before);

    /* A write that really overlaps the immediate still invalidates it. */
    qtest_writew(f->qts, 0x2101, 0x5678);
    g_assert_cmpuint(invalidations(f), >, before);
    run_function(f, 0x2100, 0x5678);
}

static void multiple_pages(Fixture *f, gconstpointer unused)
{
    uint8_t replacement[0x100a];
    unsigned before;

    function(f, 0x2ff8, 0x1111);
    function(f, 0x3100, 0x2222);
    function(f, 0x4000, 0x3333);
    run_function(f, 0x2ff8, 0x1111);
    run_function(f, 0x3100, 0x2222);
    run_function(f, 0x4000, 0x3333);
    before = invalidations(f);

    /*
     * One DMA-style write overlaps the tail of page 2, all of page 3 and
     * the head of page 4. Later pages must start at their own boundary.
     */
    qtest_memread(f->qts, 0x2ff9, replacement, sizeof(replacement));
    replacement[0] = replacement[1] = 0xaa;
    replacement[0x108] = replacement[0x109] = 0xbb;
    replacement[0x1008] = replacement[0x1009] = 0xcc;
    qtest_memwrite(f->qts, 0x2ff9, replacement, sizeof(replacement));
    g_assert_cmpuint(invalidations(f), >=, before + 3);
    run_function(f, 0x2ff8, 0xaaaa);
    run_function(f, 0x3100, 0xbbbb);
    run_function(f, 0x4000, 0xcccc);
}

static void spanning_instruction(Fixture *f, gconstpointer unused)
{
    unsigned before;

    /* The MOV immediate itself crosses the page boundary. */
    function(f, 0x2ffe, 0x1234);
    run_function(f, 0x2ffe, 0x1234);
    before = invalidations(f);
    qtest_writeb(f->qts, 0x3008, 1);
    g_assert_cmpuint(invalidations(f), ==, before);
    qtest_writeb(f->qts, 0x3000, 0x56);
    g_assert_cmpuint(invalidations(f), >, before);
    run_function(f, 0x2ffe, 0x5634);
}

static void self_modifying(Fixture *f, gconstpointer unused)
{
    static const uint8_t code[] = {
        0xc6, 0x06, 0x07, 0x21, 0x56, /* mov byte [0x2107], 0x56 */
        0xb8, 0x34, 0x12,             /* mov ax, 0x1234 (patched to 0x5634) */
        0xc3,                         /* ret */
    };

    qtest_memwrite(f->qts, 0x2100, code, sizeof(code));
    run_function(f, 0x2100, 0x5634);
    g_assert_cmpuint(invalidations(f), >, 0);
    run_function(f, 0x2100, 0x5634);
}

static void guest_data_write(Fixture *f, gconstpointer unused)
{
    static const uint8_t code[] = {
        0xc6, 0x06, 0x00, 0x28, 0xa5, /* mov byte [0x2800], 0xa5 */
        0xb8, 0x34, 0x12,             /* mov ax, 0x1234 */
        0xc3,
    };
    unsigned before;

    qtest_memwrite(f->qts, 0x2100, code, sizeof(code));
    run_function(f, 0x2100, 0x1234);
    before = invalidations(f);
    for (unsigned i = 0; i < 32; i++) {
        run_function(f, 0x2100, 0x1234);
    }
    g_assert_cmphex(qtest_readb(f->qts, 0x2800), ==, 0xa5);
    g_assert_cmpuint(invalidations(f), ==, before);
}

static void guest_spanning_write(Fixture *f, gconstpointer unused)
{
    static const uint8_t patch[] = {
        0xc6, 0x06, 0x00, 0x30, 0x56, /* mov byte [0x3000], 0x56 */
        0xb8, 0x00, 0x00,
        0xc3,
    };
    unsigned before;

    function(f, 0x2ffe, 0x1234);
    qtest_memwrite(f->qts, 0x4100, patch, sizeof(patch));
    run_function(f, 0x2ffe, 0x1234);
    before = invalidations(f);
    run_function(f, 0x4100, 0);
    g_assert_cmpuint(invalidations(f), >, before);
    run_function(f, 0x2ffe, 0x5634);
}

/* A page's former data bucket can gain translated code. A cached negative
 * coverage result must not survive that TB insertion. */
static void guest_new_code(Fixture *f, gconstpointer unused)
{
    static const uint8_t patch[] = {
        0xc6, 0x06, 0x01, 0x28, 0x78, /* mov byte [0x2801], 0x78 */
        0xb8, 0x00, 0x00,
        0xc3,
    };
    unsigned before;

    function(f, 0x2800, 0x1234);
    qtest_memwrite(f->qts, 0x2100, patch, sizeof(patch));
    run_function(f, 0x2100, 0); /* The target is still data. */
    run_function(f, 0x2800, 0x1278); /* Now the target has a TB. */
    qtest_writeb(f->qts, 0x2104, 0x56);
    before = invalidations(f);
    run_function(f, 0x2100, 0);
    g_assert_cmpuint(invalidations(f), >, before);
    run_function(f, 0x2800, 0x1256);
}

/* Execute a spanning instruction whose two virtual pages alias one physical
 * page. Both the slow physical writer and the guest NOTDIRTY writer must
 * invalidate its second-page bytes, even though it has only one page-list link. */
static void aliased_pages(Fixture *f, gconstpointer unused)
{
    static const uint8_t enter_paging[] = {
        0xfa, 0x31, 0xc0, 0x8e, 0xd8, /* cli; zero ds */
        0x0f, 0x01, 0x16, 0x00, 0xc0, /* lgdt [0xc000] */
        0x66, 0xb8, 0x00, 0x90, 0x00, 0x00, /* mov eax,0x9000 */
        0x0f, 0x22, 0xd8,             /* mov cr3,eax */
        0x0f, 0x20, 0xc0,             /* mov eax,cr0 */
        0x66, 0x0d, 0x01, 0x00, 0x00, 0x80, /* or eax,PG|PE */
        0x0f, 0x22, 0xc0,             /* mov cr0,eax */
        0xea, 0x00, 0x11, 0x08, 0x00, /* jmp 8:0x1100 */
    };
    static const uint8_t dispatch[] = {
        0xfa, 0xb8, 0x10, 0x00,       /* cli; data selector16 */
        0x8e, 0xd8, 0x8e, 0xd0, 0xbc, 0x00, 0x80,
        0x8b, 0x1e, 0x00, 0x08, 0x3b, 0x1e, 0x02, 0x08, 0x74, 0xf6,
        0xff, 0x16, 0x04, 0x08, 0xa3, 0x06, 0x08,
        0x89, 0x1e, 0x02, 0x08, 0xeb, 0xe9,
    };
    static const uint8_t gdt[] = {
        0,0,0,0,0,0,0,0,
        0xff,0xff,0,0,0,0x9a,0x8f,0, /* 16bit code,4GiB limit */
        0xff,0xff,0,0,0,0x92,0x8f,0,
    };
    static const uint8_t patch[] = {
        0xc6,0x06,0x00,0x30,0x78, /* mov byte [0x3000],0x78 */
        0xb8,0,0,0xc3,
    };
    unsigned before;

    qtest_memset(f->qts, 0x9000, 0, 4096);
    qtest_writel(f->qts, 0x9000, 0xa003);
    for (unsigned i = 0; i < 1024; ++i) {
        qtest_writel(f->qts, 0xa000 + i * 4, (i == 3 ? 0x2000 : i * 4096) | 3);
    }
    qtest_memwrite(f->qts, 0xb000, gdt, sizeof(gdt));
    qtest_writew(f->qts, 0xc000, sizeof(gdt) - 1);
    qtest_writel(f->qts, 0xc002, 0xb000);
    qtest_memwrite(f->qts, 0x1000, enter_paging, sizeof(enter_paging));
    qtest_memwrite(f->qts, 0x1100, dispatch, sizeof(dispatch));
    qtest_memwrite(f->qts, 0x2ffe, "\xb8\x34", 2);
    qtest_memwrite(f->qts, 0x2000, "\x12\xc3", 2);
    qtest_memwrite(f->qts, 0x4100, patch, sizeof(patch));
    run_function(f, 0x2ffe, 0x1234);
    before = invalidations(f);
    qtest_writeb(f->qts, 0x2000, 0x56);
    g_assert_cmpuint(invalidations(f), >, before);
    run_function(f, 0x2ffe, 0x5634);
    before = invalidations(f);
    run_function(f, 0x4100, 0);
    g_assert_cmpuint(invalidations(f), >, before);
    run_function(f, 0x2ffe, 0x7834);
}

static void concurrent_writes(Fixture *f, gconstpointer unused)
{
    static const uint8_t code[] = {
        0xc6, 0x06, 0x00, 0x28, 0xa5, /* loop: mov byte [0x2800], 0xa5 */
        0xb8, 0x34, 0x12,             /* mov ax, 0x1234 (host patches low) */
        0xa3, 0x06, 0x08,             /* mov [RESULT], ax */
        0xff, 0x06, 0x0c, 0x08,       /* inc word [iteration] */
        0x80, 0x3e, 0x0a, 0x08, 0x00, /* cmp byte [stop], 0 */
        0x74, 0xea,                   /* je loop */
        0xc3,
    };
    int64_t deadline = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;

    /*
     * With MTTCG, the vCPU's NOTDIRTY data writes race the main thread's
     * code invalidation and subsequent TB creation. Only the two complete
     * values are legal; no restart may lose the guest's bounded stop request.
     */
    qtest_memwrite(f->qts, 0x2100, code, sizeof(code));
    qtest_writew(f->qts, ENTRY, 0x2100);
    qtest_writew(f->qts, REQUEST, ++f->sequence);
    qtest_qmp_assert_success(f->qts, "{'execute':'cont'}");
    for (unsigned i = 0; i < 64; i++) {
        uint16_t before = qtest_readw(f->qts, 0x80c);
        uint16_t result;

        qtest_writeb(f->qts, 0x2106, i & 1 ? 0x34 : 0x78);
        while (qtest_readw(f->qts, 0x80c) == before) {
            g_assert_cmpint(g_get_monotonic_time(), <, deadline);
            g_usleep(100);
        }
        result = qtest_readw(f->qts, RESULT);
        g_assert_true(result == 0x1234 || result == 0x1278);
    }
    qtest_writeb(f->qts, 0x80a, 1);
    while (qtest_readw(f->qts, ACK) != f->sequence) {
        g_assert_cmpint(g_get_monotonic_time(), <, deadline);
        g_usleep(100);
    }
    qtest_qmp_assert_success(f->qts, "{'execute':'stop'}");
    qtest_qmp_eventwait(f->qts, "STOP");
    g_assert_cmphex(qtest_readb(f->qts, 0x2800), ==, 0xa5);
    g_assert_cmpuint(invalidations(f), >, 0);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    if (qtest_has_accel("tcg")) {
        qtest_add("tcg/invalidate/aliased-pages", Fixture, "single",
                  setup, aliased_pages, teardown);
        qtest_add("tcg/invalidate/aliased-pages-mttcg", Fixture, "multi",
                  setup, aliased_pages, teardown);
        qtest_add("tcg/invalidate/guest-new-code", Fixture, "single",
                  setup, guest_new_code, teardown);
        qtest_add("tcg/invalidate/guest-new-code-mttcg", Fixture, "multi",
                  setup, guest_new_code, teardown);
        qtest_add("tcg/invalidate/same-page", Fixture, "single",
                  setup, same_page, teardown);
        qtest_add("tcg/invalidate/multiple-pages", Fixture, "single",
                  setup, multiple_pages, teardown);
        qtest_add("tcg/invalidate/spanning-instruction", Fixture, "single",
                  setup, spanning_instruction, teardown);
        qtest_add("tcg/invalidate/self-modifying", Fixture, "single",
                  setup, self_modifying, teardown);
        qtest_add("tcg/invalidate/guest-data-write", Fixture, "multi",
                  setup, guest_data_write, teardown);
        qtest_add("tcg/invalidate/guest-spanning-write", Fixture, "multi",
                  setup, guest_spanning_write, teardown);
        qtest_add("tcg/invalidate/mt-self-modifying", Fixture, "multi",
                  setup, self_modifying, teardown);
        qtest_add("tcg/invalidate/concurrent-writes", Fixture, "multi",
                  setup, concurrent_writes, teardown);
    }
    return g_test_run();
}
