/*
 * Juke Shared Memory Display Backend
 *
 * Leased framebuffer snapshots with socket notifications and input.
 * QEMU publishes immutable shared-memory slots; Juke leases them through upload.
 * Input is consumed by the main-loop socket handler independently of refresh.
 *
 * Copyright (c) 2024 Anderstorpsfestivalen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "qemu/main-loop.h"
#include "qemu/error-report.h"
#include "qapi/error.h"
#include "ui/console.h"
#include "ui/juke-shmem.h"
#include "ui/surface.h"
#include "ui/input.h"
#include "qemu/bswap.h"
#include "qemu/memfd.h"
#include "qemu/sockets.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>
#include <CoreGraphics/CoreGraphics.h>
#endif

#ifdef __linux__
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <fcntl.h>
#include <dirent.h>
#endif

/* Input event types - must match Rust side */
#define JUKE_INPUT_MOUSE_REL    1
#define JUKE_INPUT_MOUSE_ABS    2
#define JUKE_INPUT_MOUSE_BTN    3
#define JUKE_INPUT_KEY          4
#define JUKE_INPUT_REFRESH_INTERVAL_MS 1
#define JUKE_INPUT_REFRESH_WINDOW_US 12000

/* Protocol v5. Geometry is immutable for each mmap epoch. Published slots
 * are immutable until the consumer releases its lease. Pixel planes have
 * independently importable, page-aligned storage for no-copy GPU sampling. */
#define JUKE_INPUT_RESET 5
#define JUKE_INPUT_REFRESH 6
#define JUKE_CURSOR_MAX_SIZE 64
#define JUKE_CURSOR_MAX_PIXELS (JUKE_CURSOR_MAX_SIZE * JUKE_CURSOR_MAX_SIZE)
#define JUKE_SHMEM_MAGIC 0x454B554A
#define JUKE_SHMEM_VERSION 5
#define JUKE_ROW_ALIGNMENT 256
#define JUKE_PLANE_ALIGNMENT 65536
#define JUKE_SLOT_COUNT 3
#define JUKE_SLOT_FREE 0
#define JUKE_SLOT_WRITING 1
#define JUKE_SLOT_READY 2
#define JUKE_SLOT_READING 3

typedef struct JukeInputEvent {
    uint8_t type, button, pressed, reserved;
    int32_t x, y;
    uint32_t padding;
    uint64_t id;
} JukeInputEvent;

typedef struct JukeShmemHeader {
    uint32_t magic, version, width, height, stride, format;
    uint64_t frame_counter;
    uint32_t reserved[12]; /* retain geometry prefix for diagnostic clients */
    uint32_t slots[3];
    uint32_t padding;
} JukeShmemHeader;

typedef struct JukeFrameMeta {
    uint64_t generation, published_us, input_id;
    uint32_t cursor_version;
    int32_t cursor_x, cursor_y;
    uint32_t cursor_visible, cursor_width, cursor_height;
    int32_t cursor_hot_x, cursor_hot_y;
    uint32_t cursor[JUKE_CURSOR_MAX_PIXELS];
} JukeFrameMeta;

/* Every socket message is 24 bytes, including SCM_RIGHTS epoch messages. */
typedef struct JukeMessage {
    uint8_t kind, padding[7];
    uint64_t id, timestamp_us;
} JukeMessage;

G_STATIC_ASSERT(sizeof(JukeInputEvent) == 24);
G_STATIC_ASSERT(sizeof(JukeMessage) == 24);
G_STATIC_ASSERT(sizeof(JukeShmemHeader) == 96);
G_STATIC_ASSERT(sizeof(JukeFrameMeta) == 16440);
#define JUKE_PIXEL_BASE QEMU_ALIGN_UP(sizeof(JukeShmemHeader) + \
    JUKE_SLOT_COUNT * sizeof(JukeFrameMeta), JUKE_PLANE_ALIGNMENT)
G_STATIC_ASSERT(JUKE_PIXEL_BASE == 65536);

typedef struct JukeShmemState {
    DisplayChangeListener dcl;
    DisplaySurface *surface;
    JukeShmemHeader *shmem;
    size_t shmem_size;
    size_t plane_size;
    int shmem_fd;
    char *socket_path;
    int client_fd;
    bool fd_sent, dirty, cpu_anchor_pending;
    int32_t mouse_x, mouse_y;
    bool mouse_initialized;
    uint64_t generation, input_id, epoch;
    JukeFrameMeta cursor;
    uint8_t input_bytes[sizeof(JukeInputEvent)];
    size_t input_used;
    bool held_keys[256], held_buttons[INPUT_BUTTON__MAX];
    JukeMessage deferred_ack;
    uint8_t *cursor_shmem;
    int cursor_fd;
    bool cursor_fd_sent, cursor_shape_dirty, cursor_notify_shape;
    bool cursor_position_dirty;
    uint64_t cursor_epoch, cursor_generation, cursor_position_sequence;
    JukeNativeCursor native_cursor;
    int normal_refresh_ms;
    uint64_t input_refresh_until_us;
} JukeShmemState;

/* The configured display has one default console. All access is under BQL. */
static JukeShmemState *juke_active_display;

bool juke_shmem_cpu_anchor(QemuConsole *con, uint64_t *epoch,
                           uint64_t *generation)
{
    JukeShmemState *s = juke_active_display;

    if (!s || s->dcl.con != con || !s->shmem) {
        return false;
    }
    *epoch = s->epoch;
    *generation = s->generation + 1;
    s->dirty = true;
    s->cpu_anchor_pending = true;
    return true;
}

static void juke_shmem_send_fd(JukeShmemState *s);
static int juke_shmem_connect(JukeShmemState *s);
static void juke_shmem_input_ready(void *opaque);
static void juke_shmem_ack_ready(void *opaque);
static void juke_shmem_cursor_publish(JukeShmemState *s);

static uint64_t juke_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static void juke_shmem_release_input(JukeShmemState *s, bool all)
{
    for (int i = 0; i < 256; i++) {
        if (s->held_keys[i] || (all && qemu_input_key_number_to_linux(i))) {
            qemu_input_event_send_key_number(s->dcl.con, i, false);
            s->held_keys[i] = false;
        }
    }
    for (int i = 0; i < INPUT_BUTTON__MAX; i++) {
        if (s->held_buttons[i] || all) {
            qemu_input_queue_btn(NULL, i, false);
            s->held_buttons[i] = false;
        }
    }
    qemu_input_event_sync();
}

static void juke_shmem_disconnect(JukeShmemState *s)
{
    if (s->client_fd >= 0) {
        qemu_set_fd_handler(s->client_fd, NULL, NULL, NULL);
        close(s->client_fd);
        s->client_fd = -1;
    }
    s->fd_sent = false;
    s->cursor_fd_sent = false;
    s->input_used = 0;
    s->deferred_ack.kind = 0;
    juke_shmem_release_input(s, false);
}

static void juke_shmem_notify(JukeShmemState *s, uint8_t kind,
                              uint64_t id, uint64_t timestamp)
{
    JukeMessage msg = { .kind = kind, .id = id, .timestamp_us = timestamp };
    /* Reserved byte carries key edge for causal probe pairing. */
    if (kind == 'A') { msg.padding[0] = s->input_bytes[0] == JUKE_INPUT_KEY ? (s->input_bytes[2] ? 1 : 2) : 0; }
    if (s->client_fd < 0 || !s->fd_sent) {
        return;
    }
    ssize_t n = send(s->client_fd, &msg, sizeof(msg), MSG_DONTWAIT | MSG_NOSIGNAL);
    /* EAGAIN means a prior notification already makes the socket readable.
     * Never block the emulator on diagnostics or a slow display consumer. */
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        if (kind == 'A' && (id & (1ULL << 63))) {
            s->deferred_ack = msg;
            qemu_set_fd_handler(s->client_fd, juke_shmem_input_ready,
                                juke_shmem_ack_ready, s);
        }
        return;
    }
    if (n != sizeof(msg)) {
        juke_shmem_disconnect(s);
    }
}

static void juke_shmem_ack_ready(void *opaque)
{
    JukeShmemState *s = opaque;

    if (s->deferred_ack.kind) {
        ssize_t n = send(s->client_fd, &s->deferred_ack,
                         sizeof(s->deferred_ack), MSG_DONTWAIT | MSG_NOSIGNAL);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ||
                     errno == EINTR)) {
            return;
        }
        if (n != sizeof(s->deferred_ack)) {
            juke_shmem_disconnect(s);
            return;
        }
        s->deferred_ack.kind = 0;
    }
    qemu_set_fd_handler(s->client_fd, juke_shmem_input_ready, NULL, s);
    juke_shmem_cursor_publish(s);
}

/* Return 1 for sent, 0 for backpressure, -1 for a disconnected stream. */
static int juke_shmem_cursor_send(JukeShmemState *s, const uint8_t *packet,
                                   int fd)
{
    union {
        struct cmsghdr align;
        uint8_t bytes[CMSG_SPACE(sizeof(int))];
    } control = { 0 };
    struct iovec iov = { .iov_base = (void *)packet,
                         .iov_len = JCUR_PACKET_BYTES };
    struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1 };

    if (fd >= 0) {
        msg.msg_control = control.bytes;
        msg.msg_controllen = sizeof(control.bytes);
        struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(c), &fd, sizeof(fd));
    }
    ssize_t n = sendmsg(s->client_fd, &msg, MSG_DONTWAIT | MSG_NOSIGNAL);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
        return 0;
    }
    if (n != JCUR_PACKET_BYTES) {
        juke_shmem_disconnect(s);
        return -1;
    }
    return 1;
}

static void juke_shmem_cursor_new_mapping(JukeShmemState *s)
{
    if (s->cursor_shmem) {
        qemu_memfd_free(s->cursor_shmem, JCUR_MAPPING_BYTES, s->cursor_fd);
    }
    s->cursor_shmem = qemu_memfd_alloc("juke-cursor", JCUR_MAPPING_BYTES, 0,
                                       &s->cursor_fd, NULL);
    s->cursor_fd_sent = false;
    if (!s->cursor_shmem) {
        s->cursor_fd = -1;
        error_report("juke-shmem: failed to allocate native cursor mapping");
        return;
    }
    memset(s->cursor_shmem, 0, JCUR_MAPPING_BYTES);
    stl_le_p(s->cursor_shmem + JCUR_HDR_MAGIC, JCUR_MAGIC);
    stl_le_p(s->cursor_shmem + JCUR_HDR_VERSION, JCUR_VERSION);
    stl_le_p(s->cursor_shmem + JCUR_HDR_MAX_DIMENSION,
              JRG_CURSOR_MAX_DIMENSION);
    stl_le_p(s->cursor_shmem + JCUR_HDR_SLOT_COUNT, JCUR_SLOT_COUNT);
    stq_le_p(s->cursor_shmem + JCUR_HDR_EPOCH, ++s->cursor_epoch);
    s->cursor_generation = 0;
    s->cursor_position_sequence = 1;
    s->cursor_shape_dirty = s->native_cursor.width != 0;
    s->cursor_notify_shape = false;
    s->cursor_position_dirty = true;
}

static void juke_shmem_cursor_publish(JukeShmemState *s)
{
    uint8_t packet[JCUR_PACKET_BYTES] = { 0 };
    bool writable = false;
    int sent;

    if (s->client_fd < 0 || !s->fd_sent || !s->cursor_shmem) {
        return;
    }
    if (!s->cursor_fd_sent) {
        packet[0] = JCUR_MSG_MAPPING;
        stq_le_p(packet + 8, s->cursor_epoch);
        stq_le_p(packet + 16, juke_now_us());
        sent = juke_shmem_cursor_send(s, packet, s->cursor_fd);
        if (sent <= 0) {
            writable = sent == 0;
            goto out;
        }
        s->cursor_fd_sent = true;
    }
    if (s->cursor_shape_dirty) {
        bool published = false;
        for (unsigned i = 0; i < JCUR_SLOT_COUNT; i++) {
            uint32_t *state = (uint32_t *)(s->cursor_shmem +
                                          JCUR_HDR_SLOTS + i * 4);
            uint32_t old = qatomic_load_acquire(state);
            if ((old == JCUR_FREE || old == JCUR_READY) &&
                qatomic_cmpxchg(state, old, JCUR_WRITING) == old) {
                const JukeNativeCursor *c = &s->native_cursor;
                uint8_t *slot = s->cursor_shmem + JCUR_HEADER_BYTES +
                                i * JCUR_SLOT_BYTES;

                memset(slot, 0, JCUR_SLOT_BYTES);
                stq_le_p(slot + JCUR_SLOT_GENERATION, ++s->cursor_generation);
                stl_le_p(slot + JCUR_SLOT_WIDTH, c->width);
                stl_le_p(slot + JCUR_SLOT_HEIGHT, c->height);
                stl_le_p(slot + JCUR_SLOT_HOT_X, c->hot_x);
                stl_le_p(slot + JCUR_SLOT_HOT_Y, c->hot_y);
                stl_le_p(slot + JCUR_SLOT_FORMAT, c->format);
                stq_le_p(slot + JCUR_SLOT_POSITION_SEQUENCE,
                          s->cursor_position_sequence);
                stl_le_p(slot + JCUR_SLOT_X, c->x);
                stl_le_p(slot + JCUR_SLOT_Y, c->y);
                stl_le_p(slot + JCUR_SLOT_FLAGS, c->flags);
                memcpy(slot + JCUR_SLOT_PIXELS, c->pixels,
                        c->width * c->height * JRG_CURSOR_PIXEL_BYTES);
                qatomic_store_release(state, JCUR_READY);
                qatomic_store_release((uint64_t *)(s->cursor_shmem +
                                      JCUR_HDR_GENERATION), s->cursor_generation);
                s->cursor_shape_dirty = false;
                s->cursor_notify_shape = true;
                published = true;
                break;
            }
        }
        if (!published) {
            /* Leased slots need a later refresh, never a writable busy loop. */
            goto out;
        }
    }
    if (s->cursor_notify_shape) {
        packet[0] = JCUR_MSG_SHAPE;
        stq_le_p(packet + 8, s->cursor_generation);
        stq_le_p(packet + 16, juke_now_us());
        sent = juke_shmem_cursor_send(s, packet, -1);
        if (sent <= 0) {
            writable = sent == 0;
            goto out;
        }
        s->cursor_notify_shape = false;
    }
    if (s->cursor_position_dirty) {
        packet[0] = JCUR_MSG_POSITION;
        packet[JCUR_PACKET_FLAGS] = s->native_cursor.flags;
        stq_le_p(packet + JCUR_PACKET_SEQUENCE, s->cursor_position_sequence);
        stl_le_p(packet + JCUR_PACKET_X, s->native_cursor.x);
        stl_le_p(packet + JCUR_PACKET_Y, s->native_cursor.y);
        sent = juke_shmem_cursor_send(s, packet, -1);
        if (sent <= 0) {
            writable = sent == 0;
            goto out;
        }
        s->cursor_position_dirty = false;
    }
out:
    if (s->client_fd >= 0) {
        qemu_set_fd_handler(s->client_fd, juke_shmem_input_ready,
            writable || s->deferred_ack.kind ? juke_shmem_ack_ready : NULL, s);
    }
}

void juke_shmem_native_cursor(QemuConsole *con, const JukeNativeCursor *cursor,
                              bool shape)
{
    JukeShmemState *s = juke_active_display;

    if (!s || s->dcl.con != con) {
        return;
    }
    if (shape) {
        s->native_cursor = *cursor;
        s->cursor_shape_dirty = cursor->width != 0;
        if (!cursor->width) {
            s->cursor_notify_shape = false;
            if (s->cursor_shmem) {
                /* A reset retires the old shape, including consumer copies. */
                juke_shmem_cursor_new_mapping(s);
            }
        }
    } else {
        s->native_cursor.x = cursor->x;
        s->native_cursor.y = cursor->y;
        s->native_cursor.flags = cursor->flags;
    }
    s->cursor_position_sequence++;
    s->cursor_position_dirty = true;
    juke_shmem_cursor_publish(s);
}

static JukeFrameMeta *juke_shmem_slot(JukeShmemState *s, int i)
{
    return (JukeFrameMeta *)(s->shmem + 1) + i;
}

static uint8_t *juke_shmem_pixels(JukeShmemState *s, int i)
{
    return (uint8_t *)s->shmem + JUKE_PIXEL_BASE + i * s->plane_size;
}

static void juke_shmem_publish(JukeShmemState *s)
{
    if (s->cpu_anchor_pending) {
        /* Input can publish between refreshes. Refresh the VGA surface first,
         * so a ReturnCpu/reset anchor cannot match stale converted pixels. */
        s->cpu_anchor_pending = false;
        qemu_console_hw_update(s->dcl.con);
    }
    if (!s->shmem || !s->surface || !s->dirty) {
        return;
    }
    for (int i = 0; i < 3; i++) {
        uint32_t state = __atomic_load_n(&s->shmem->slots[i], __ATOMIC_ACQUIRE);
        if ((state == JUKE_SLOT_FREE || state == JUKE_SLOT_READY) &&
            __atomic_compare_exchange_n(&s->shmem->slots[i], &state,
                JUKE_SLOT_WRITING, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            JukeFrameMeta *frame = juke_shmem_slot(s, i);
            *frame = s->cursor;
            frame->generation = ++s->generation;
            frame->input_id = s->input_id;
            uint8_t *pixels = juke_shmem_pixels(s, i);
            const uint8_t *source = surface_data(s->surface);
            size_t row_bytes = (size_t)s->shmem->width * 4;
            size_t source_stride = surface_stride(s->surface);
            if (source_stride == row_bytes && s->shmem->stride == row_bytes) {
                memcpy(pixels, source, row_bytes * s->shmem->height);
            } else {
                for (uint32_t y = 0; y < s->shmem->height; y++) {
                    memcpy(pixels + y * (size_t)s->shmem->stride,
                           source + y * source_stride, row_bytes);
                }
            }
            frame->published_us = juke_now_us();
            __atomic_store_n(&s->shmem->slots[i], JUKE_SLOT_READY, __ATOMIC_RELEASE);
            __atomic_store_n(&s->shmem->frame_counter, s->generation, __ATOMIC_RELEASE);
            s->dirty = false;
            juke_shmem_notify(s, 'F', s->generation, frame->published_us);
            return;
        }
    }
    /* All slots are leased. Keep dirty and retry next display deadline. */
}

static void juke_shmem_process_event(JukeShmemState *s, JukeInputEvent *ev)
{
    bool guest_wants_abs = qemu_input_is_absolute(s->dcl.con);
    if (ev->type == JUKE_INPUT_RESET || ev->type == JUKE_INPUT_REFRESH) {
        if (ev->type == JUKE_INPUT_RESET) {
            /* Include untracked guest keys restored by a VM snapshot. */
            juke_shmem_release_input(s, true);
        }
        s->dirty = true;
        juke_shmem_publish(s);
        juke_shmem_notify(s, 'A', ev->id, juke_now_us());
        return;
    }
    if (!s->shmem) {
        return;
    }
    if (ev->type == JUKE_INPUT_KEY) {
        if (ev->x < 0 || ev->x >= 256) {
            return;
        }
        s->held_keys[ev->x] = ev->pressed;
    }
    if (ev->type == JUKE_INPUT_MOUSE_BTN) {
        if (ev->button >= INPUT_BUTTON__MAX) {
            return;
        }
        s->held_buttons[ev->button] = ev->pressed;
    }
        switch (ev->type) {
        case JUKE_INPUT_MOUSE_REL:
            /* Mark mouse as initialized on first movement - this prevents
             * gfx_switch from resetting position to center */
            s->mouse_initialized = true;

            if (guest_wants_abs) {
                /* Convert relative to absolute */
                s->mouse_x += ev->x;
                s->mouse_y += ev->y;
                if (s->mouse_x < 0) s->mouse_x = 0;
                if (s->mouse_y < 0) s->mouse_y = 0;
                if (s->mouse_x >= (int32_t)s->shmem->width)
                    s->mouse_x = s->shmem->width - 1;
                if (s->mouse_y >= (int32_t)s->shmem->height)
                    s->mouse_y = s->shmem->height - 1;
                qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_X,
                    s->mouse_x, 0, s->shmem->width);
                qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_Y,
                    s->mouse_y, 0, s->shmem->height);
            } else {
                qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_X, ev->x);
                qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_Y, ev->y);
            }
            break;

        case JUKE_INPUT_MOUSE_ABS:
            if (guest_wants_abs) {
                /* Clamp to framebuffer bounds before sending to guest */
                int32_t abs_x = ev->x;
                int32_t abs_y = ev->y;
                if (abs_x < 0) abs_x = 0;
                if (abs_y < 0) abs_y = 0;
                if (abs_x >= (int32_t)s->shmem->width)
                    abs_x = s->shmem->width - 1;
                if (abs_y >= (int32_t)s->shmem->height)
                    abs_y = s->shmem->height - 1;
                qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_X,
                    abs_x, 0, s->shmem->width);
                qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_Y,
                    abs_y, 0, s->shmem->height);
                s->mouse_x = abs_x;
                s->mouse_y = abs_y;
            } else {
                /* Convert absolute to relative */
                if (s->mouse_initialized) {
                    int32_t dx = ev->x - s->mouse_x;
                    int32_t dy = ev->y - s->mouse_y;
                    if (dx != 0 || dy != 0) {
                        qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_X, dx);
                        qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_Y, dy);
                    }
                }
                s->mouse_x = ev->x;
                s->mouse_y = ev->y;
                s->mouse_initialized = true;
            }
            break;

        case JUKE_INPUT_MOUSE_BTN:
            /* Use NULL source like input-linux.c for PS/2 compatibility */
            qemu_input_queue_btn(NULL, ev->button, ev->pressed);
            break;

        case JUKE_INPUT_KEY:
            /* ev->x contains the scancode */
            qemu_input_event_send_key_number(s->dcl.con, ev->x, ev->pressed);
            break;
        }

    qemu_input_event_sync();
    s->input_id = ev->id;
    juke_shmem_notify(s, 'A', ev->id, juke_now_us());
    if (ev->type >= JUKE_INPUT_MOUSE_REL &&
        ev->type <= JUKE_INPUT_KEY) {
        /* Service guest display work promptly after interaction. This is a
         * bounded timer burst, not a polling loop; idle keeps monitor pacing. */
        s->input_refresh_until_us = juke_now_us() + JUKE_INPUT_REFRESH_WINDOW_US;
        qemu_console_listener_set_refresh(&s->dcl, JUKE_INPUT_REFRESH_INTERVAL_MS);
    }
}

/* Main-loop socket readiness wakes input independently of display refresh.
 * A bounded batch gives emulation and other devices a turn during floods. */
static void juke_shmem_input_ready(void *opaque)
{
    JukeShmemState *s = opaque;
    for (int count = 0; count < 256; count++) {
        ssize_t n = recv(s->client_fd, s->input_bytes + s->input_used,
                         sizeof(JukeInputEvent) - s->input_used, MSG_DONTWAIT);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return;
        }
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            juke_shmem_disconnect(s);
            return;
        }
        s->input_used += n;
        if (s->input_used == sizeof(JukeInputEvent)) {
            JukeInputEvent ev;
            memcpy(&ev, s->input_bytes, sizeof(ev));
            s->input_used = 0;
            juke_shmem_process_event(s, &ev);
            if (s->client_fd < 0) {
                return;
            }
        }
    }
}

static void juke_shmem_gfx_update(DisplayChangeListener *dcl,
                                   int x, int y, int w, int h)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);

    if (!s->shmem || !s->surface) {
        return;
    }

    /* Consolidate all device damage into one stable snapshot per refresh. */
    s->dirty = true;
}

static void juke_shmem_gfx_switch(DisplayChangeListener *dcl,
                                   DisplaySurface *new_surface)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);

    s->surface = new_surface;

    if (!new_surface) {
        return;
    }

    int w = surface_width(new_surface);
    int h = surface_height(new_surface);
    int source_stride = surface_stride(new_surface);
    uint32_t format = surface_format(new_surface);
    size_t stride, pixels_size, plane_size, needed;

    /* Reject unsupported surfaces and arithmetic overflow before allocating or
     * copying. Destination row padding is independent of the native surface. */
    if (w <= 0 || h <= 0 || source_stride < 0 ||
        (size_t)w > (UINT32_MAX - (JUKE_ROW_ALIGNMENT - 1)) / 4 ||
        (size_t)source_stride < (size_t)w * 4 ||
        (format != PIXMAN_x8r8g8b8 && format != PIXMAN_a8r8g8b8)) {
        s->surface = NULL;
        error_report("juke-shmem: unsupported display surface geometry/format");
        return;
    }
    stride = QEMU_ALIGN_UP((size_t)w * 4, JUKE_ROW_ALIGNMENT);
    if ((size_t)h > SIZE_MAX / stride ||
        stride * h > SIZE_MAX - (JUKE_PLANE_ALIGNMENT - 1)) {
        s->surface = NULL;
        error_report("juke-shmem: display plane size overflow");
        return;
    }
    pixels_size = stride * h;
    plane_size = QEMU_ALIGN_UP(pixels_size, JUKE_PLANE_ALIGNMENT);
    if (plane_size > (SIZE_MAX - JUKE_PIXEL_BASE) / JUKE_SLOT_COUNT) {
        s->surface = NULL;
        error_report("juke-shmem: display mapping size overflow");
        return;
    }
    needed = JUKE_PIXEL_BASE + JUKE_SLOT_COUNT * plane_size;
    if (needed > PTRDIFF_MAX) {
        s->surface = NULL;
        error_report("juke-shmem: display mapping exceeds addressable range");
        return;
    }

    /* Every geometry/format change creates a new immutable epoch. Old mappings
     * remain valid while the consumer holds frame leases, including on shrink. */
    if (s->shmem) {
        qemu_memfd_free(s->shmem, s->shmem_size, s->shmem_fd);
    }
    s->shmem_size = needed;
    s->plane_size = plane_size;
    s->shmem = qemu_memfd_alloc("juke-fb", needed, 0, &s->shmem_fd, NULL);
    s->fd_sent = false;
    if (!s->shmem) {
        s->shmem_fd = -1;
        error_report("juke-shmem: failed to allocate shared memory");
        return;
    }
    /* Initialize metadata, row padding and page padding before native textures
     * can import any plane. This is paid once per geometry epoch, not per frame. */
    memset(s->shmem, 0, needed);
    s->shmem->magic = JUKE_SHMEM_MAGIC;
    s->shmem->version = JUKE_SHMEM_VERSION;
    s->shmem->width = w;
    s->shmem->height = h;
    s->shmem->stride = stride;
    s->shmem->format = format;
    ++s->epoch;
    s->shmem->reserved[0] = s->epoch;
    s->shmem->reserved[1] = s->epoch >> 32;
    s->dirty = true;

    /* Preserve mouse position across resolution changes, just clamp to new bounds.
     * Only reset to center on first initialization (mouse_initialized = false).
     * This prevents the mouse jumping to center when guest changes resolution. */
    if (!s->mouse_initialized) {
        s->mouse_x = w / 2;
        s->mouse_y = h / 2;
    } else {
        /* Clamp existing position to new resolution bounds */
        if (s->mouse_x >= w) s->mouse_x = w - 1;
        if (s->mouse_y >= h) s->mouse_y = h - 1;
        if (s->mouse_x < 0) s->mouse_x = 0;
        if (s->mouse_y < 0) s->mouse_y = 0;
    }

    if (s->client_fd >= 0) {
        juke_shmem_send_fd(s);
    }
    juke_shmem_publish(s);
}

static void juke_shmem_refresh(DisplayChangeListener *dcl)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);

    if (s->input_refresh_until_us &&
        juke_now_us() >= s->input_refresh_until_us) {
        s->input_refresh_until_us = 0;
        qemu_console_listener_set_refresh(&s->dcl, s->normal_refresh_ms);
    }

    /* Try to (re)connect if not connected */
    if (s->client_fd < 0 && s->socket_path) {
        juke_shmem_connect(s);
    }

    /* Try to send fd if we have connection and shared memory */
    if (s->client_fd >= 0 && s->shmem_fd >= 0 && !s->fd_sent) {
        juke_shmem_send_fd(s);
    }

    qemu_console_hw_update(dcl->con);
    juke_shmem_publish(s);
    juke_shmem_cursor_publish(s);
}

/*
 * Handle cursor shape change from guest
 * Like Cocoa, we read from console cursor storage for reliability
 */
static void juke_shmem_cursor_define(DisplayChangeListener *dcl, QEMUCursor *cursor)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);
    QEMUCursor *con_cursor = qemu_console_get_cursor(dcl->con);
    s->cursor.cursor_width = con_cursor ? MIN(con_cursor->width, JUKE_CURSOR_MAX_SIZE) : 0;
    s->cursor.cursor_height = con_cursor ? MIN(con_cursor->height, JUKE_CURSOR_MAX_SIZE) : 0;
    if (con_cursor) {
        s->cursor.cursor_hot_x = con_cursor->hot_x;
        s->cursor.cursor_hot_y = con_cursor->hot_y;
        for (uint32_t y = 0; y < s->cursor.cursor_height; y++) {
            memcpy(&s->cursor.cursor[y * JUKE_CURSOR_MAX_SIZE],
                   &con_cursor->data[y * con_cursor->width],
                   s->cursor.cursor_width * sizeof(uint32_t));
        }
    }
    s->cursor.cursor_version++;
    s->dirty = true;
}

static void juke_shmem_mouse_set(DisplayChangeListener *dcl, int x, int y, bool on)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);
    s->cursor.cursor_x = x;
    s->cursor.cursor_y = y;
    s->cursor.cursor_visible = on;
    s->dirty = true;
}

static const DisplayChangeListenerOps juke_shmem_ops = {
    .dpy_name          = "juke-shmem",
    .dpy_gfx_update    = juke_shmem_gfx_update,
    .dpy_gfx_switch    = juke_shmem_gfx_switch,
    .dpy_refresh       = juke_shmem_refresh,
    .dpy_cursor_define = juke_shmem_cursor_define,
    .dpy_mouse_set     = juke_shmem_mouse_set,
};

/* Send shared memory fd to client via SCM_RIGHTS */
static void juke_shmem_send_fd(JukeShmemState *s)
{
    if (s->client_fd < 0 || s->shmem_fd < 0 || s->fd_sent) {
        return;
    }

    struct msghdr msg = {0};
    struct iovec iov[1];
    JukeMessage buf = { .kind = 'D' };

    /* Ancillary data buffer for fd */
    char cmsgbuf[CMSG_SPACE(sizeof(int))];

    iov[0].iov_base = &buf;
    iov[0].iov_len = sizeof(buf);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &s->shmem_fd, sizeof(int));

    ssize_t sent = sendmsg(s->client_fd, &msg, MSG_DONTWAIT | MSG_NOSIGNAL);
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return; /* retry on next refresh */
    }
    if (sent != sizeof(buf)) {
        juke_shmem_disconnect(s);
    } else {
        s->fd_sent = true;
    }
}

/* Connect to Juke's socket and send fd (silent on failure for retry) */
static int juke_shmem_connect(JukeShmemState *s)
{
    if (!s->socket_path) {
        return -1;
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        /* Socket creation failure is unusual, worth reporting */
        error_report("juke-shmem: socket failed: %s", strerror(errno));
        return -1;
    }

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        /* Silent failure - will retry in refresh() */
        close(fd);
        return -1;
    }

    /* Connection successful! */
    error_report("juke-shmem: connected to %s", s->socket_path);
    s->client_fd = fd;
    s->fd_sent = false;
    if (!qemu_set_blocking(fd, false, NULL)) {
        juke_shmem_disconnect(s);
        return -1;
    }
#ifdef SO_NOSIGPIPE
    int no_sigpipe = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif
    qemu_set_fd_handler(fd, juke_shmem_input_ready, NULL, s);

    /* A disconnected consumer can leave every slot leased forever. Allocate a
     * fresh epoch instead of resetting slot ownership in memory that surviving
     * render leases may still read. This also publishes the current surface to
     * the new consumer without waiting for guest damage. */
    if (s->surface) {
        juke_shmem_gfx_switch(&s->dcl, s->surface);
    }
    juke_shmem_cursor_new_mapping(s);
    juke_shmem_cursor_publish(s);

    return 0;
}

/*
 * Set up display refresh rate to match monitor (like Cocoa does)
 * This is critical for performance - default is 30ms which limits us to 33fps
 *
 * macOS: Use CVDisplayLink to detect actual monitor refresh rate
 * Linux: Use libdrm to query the active display mode refresh rate
 */
static void juke_shmem_setup_refresh(JukeShmemState *s)
{
    int interval_ms = 0;

#ifdef __APPLE__
    /* Get display refresh rate using CVDisplayLink like Cocoa does */
    CGDirectDisplayID display = CGMainDisplayID();
    CVDisplayLinkRef displayLink;
    if (CVDisplayLinkCreateWithCGDisplay(display, &displayLink) == kCVReturnSuccess) {
        CVTime period = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(displayLink);
        CVDisplayLinkRelease(displayLink);
        if (!(period.flags & kCVTimeIsIndefinite) && period.timeScale > 0) {
            interval_ms = (int)(1000 * period.timeValue / period.timeScale);
        }
    }
#endif

#ifdef __linux__
    /* Get display refresh rate using libdrm */
    DIR *dir = opendir("/dev/dri");
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strncmp(entry->d_name, "card", 4) != 0) {
                continue;
            }
            char path[256];
            snprintf(path, sizeof(path), "/dev/dri/%s", entry->d_name);

            int fd = open(path, O_RDONLY);
            if (fd < 0) {
                continue;
            }

            drmModeRes *res = drmModeGetResources(fd);
            if (res) {
                /* Find active CRTC with highest refresh rate */
                for (int i = 0; i < res->count_crtcs; i++) {
                    drmModeCrtc *crtc = drmModeGetCrtc(fd, res->crtcs[i]);
                    if (crtc && crtc->mode_valid) {
                        /* Calculate refresh rate from mode timing */
                        uint32_t htotal = crtc->mode.htotal;
                        uint32_t vtotal = crtc->mode.vtotal;
                        uint32_t clock = crtc->mode.clock; /* in kHz */
                        if (htotal > 0 && vtotal > 0 && clock > 0) {
                            int refresh_hz = (clock * 1000) / (htotal * vtotal);
                            int this_interval = 1000 / refresh_hz;
                            if (this_interval > 0 && (interval_ms == 0 || this_interval < interval_ms)) {
                                interval_ms = this_interval;
                            }
                        }
                        drmModeFreeCrtc(crtc);
                    }
                }
                drmModeFreeResources(res);
            }
            close(fd);

            if (interval_ms > 0) {
                break; /* Found a valid refresh rate */
            }
        }
        closedir(dir);
    }
#endif

    if (interval_ms > 0 && interval_ms < 100) {
        error_report("juke-shmem: using monitor refresh rate: %dms (~%dHz)",
                    interval_ms, 1000 / interval_ms);
        s->normal_refresh_ms = interval_ms;
    } else {
        /* Fallback: 8ms (~120Hz) - fast enough for any common display */
        error_report("juke-shmem: using fallback refresh rate: 8ms (~120Hz)");
        s->normal_refresh_ms = 8;
    }
    qemu_console_listener_set_refresh(&s->dcl, s->normal_refresh_ms);


}

static void juke_shmem_init(DisplayState *ds, DisplayOptions *opts)
{
    JukeShmemState *s = g_new0(JukeShmemState, 1);

    s->dcl.con = qemu_console_lookup_default();
    juke_active_display = s;
    s->dcl.ops = &juke_shmem_ops;
    s->shmem_fd = -1;
    s->cursor_fd = -1;
    s->client_fd = -1;

    if (opts->u.juke_shmem.socket) {
        s->socket_path = g_strdup(opts->u.juke_shmem.socket);
        /* Connect to Juke's socket - may fail if Juke hasn't created it yet */
        juke_shmem_connect(s);
    }

    qemu_console_register_listener(s->dcl.con, &s->dcl, &juke_shmem_ops);

    /* Set refresh rate to match monitor (critical for performance!) */
    juke_shmem_setup_refresh(s);
}

static QemuDisplay qemu_display_juke_shmem = {
    .type = DISPLAY_TYPE_JUKE_SHMEM,
    .init = juke_shmem_init,
};

static void register_juke_shmem(void)
{
    qemu_display_register(&qemu_display_juke_shmem);
}

type_init(register_juke_shmem);
