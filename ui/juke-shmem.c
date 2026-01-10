/*
 * Juke Shared Memory Display Backend
 *
 * Zero-copy framebuffer sharing between QEMU and Juke via mmap'd memory.
 * QEMU writes framebuffer data directly to shared memory, Juke reads it.
 * Uses atomic frame counter for update notification (no syscalls needed).
 * Also handles input via shared memory ring buffer and hardware cursor support.
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
#include "ui/surface.h"
#include "ui/input.h"
#include "qemu/memfd.h"
#include "qemu/sockets.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

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

/* Single input event (12 bytes) - must match Rust side */
typedef struct JukeInputEvent {
    uint8_t type;           /* JUKE_INPUT_* */
    uint8_t button;         /* For BTN: InputButton enum */
    uint8_t pressed;        /* 1=down, 0=up */
    uint8_t reserved;
    int32_t x;              /* dx for REL, x for ABS, scancode for KEY */
    int32_t y;              /* dy for REL, y for ABS, unused for KEY */
} JukeInputEvent;

#define JUKE_INPUT_RING_SIZE 256

/* Input ring buffer - must match Rust side */
typedef struct JukeInputRing {
    uint32_t write_idx;     /* Written by Juke (atomic) */
    uint32_t read_idx;      /* Written by QEMU (atomic) */
    uint32_t padding[2];    /* Align to 16 bytes */
    JukeInputEvent events[JUKE_INPUT_RING_SIZE];
} JukeInputRing;

/* Cursor constants */
#define JUKE_CURSOR_MAX_SIZE 64
#define JUKE_CURSOR_MAX_PIXELS (JUKE_CURSOR_MAX_SIZE * JUKE_CURSOR_MAX_SIZE)
#define JUKE_CURSOR_DATA_SIZE (JUKE_CURSOR_MAX_PIXELS * 4)  /* RGBA */

/* Header at start of shared memory region - must match Rust side */
typedef struct JukeShmemHeader {
    uint32_t magic;              /* 'JUKE' = 0x454B554A */
    uint32_t version;            /* Protocol version */
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t format;             /* PIXMAN format code */
    uint64_t frame_counter;      /* Incremented on each update */
    uint32_t dirty_x;
    uint32_t dirty_y;
    uint32_t dirty_w;
    uint32_t dirty_h;

    /* Cursor state (v3) */
    uint32_t cursor_version;     /* Incremented when cursor shape changes */
    int32_t cursor_x;            /* Cursor position X */
    int32_t cursor_y;            /* Cursor position Y */
    uint32_t cursor_visible;     /* 0 = hidden, 1 = visible */
    uint32_t cursor_width;       /* Cursor image width (max 64) */
    uint32_t cursor_height;      /* Cursor image height (max 64) */
    int32_t cursor_hot_x;        /* Hotspot X */
    int32_t cursor_hot_y;        /* Hotspot Y */

    /* Cursor RGBA data follows header (JUKE_CURSOR_MAX_PIXELS * 4 bytes) */
    /* JukeInputRing follows after cursor data */
    /* Pixel data follows after input ring */
} JukeShmemHeader;

#define JUKE_SHMEM_MAGIC 0x454B554A  /* 'JUKE' */
#define JUKE_SHMEM_VERSION 3  /* Bumped for cursor support */

typedef struct JukeShmemState {
    DisplayChangeListener dcl;
    DisplaySurface *surface;
    JukeShmemHeader *shmem;
    size_t shmem_size;
    int shmem_fd;
    char *socket_path;
    int client_fd;       /* Connection to Juke */
    bool fd_sent;
} JukeShmemState;

static void juke_shmem_send_fd(JukeShmemState *s);
static int juke_shmem_connect(JukeShmemState *s);

/* Get pointer to cursor pixel data (after header) */
static inline uint32_t *juke_shmem_cursor_data(JukeShmemState *s)
{
    return (uint32_t *)(s->shmem + 1);
}

/* Get pointer to input ring buffer (after header and cursor data) */
static inline JukeInputRing *juke_shmem_input_ring(JukeShmemState *s)
{
    uint8_t *base = (uint8_t *)(s->shmem + 1);
    return (JukeInputRing *)(base + JUKE_CURSOR_DATA_SIZE);
}

/* Get pointer to pixel data (after header, cursor data, and input ring) */
static inline uint8_t *juke_shmem_pixels(JukeShmemState *s)
{
    return (uint8_t *)(juke_shmem_input_ring(s) + 1);
}

/* Process pending input events from Juke */
static void juke_shmem_process_input(JukeShmemState *s)
{
    JukeInputRing *ring = juke_shmem_input_ring(s);
    uint32_t write_idx = __atomic_load_n(&ring->write_idx, __ATOMIC_ACQUIRE);
    uint32_t read_idx = ring->read_idx;

    while (read_idx != write_idx) {
        JukeInputEvent *ev = &ring->events[read_idx % JUKE_INPUT_RING_SIZE];

        switch (ev->type) {
        case JUKE_INPUT_MOUSE_REL:
            qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_X, ev->x);
            qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_Y, ev->y);
            break;
        case JUKE_INPUT_MOUSE_ABS:
            qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_X, ev->x, 0, s->shmem->width);
            qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_Y, ev->y, 0, s->shmem->height);
            break;
        case JUKE_INPUT_MOUSE_BTN:
            qemu_input_queue_btn(s->dcl.con, ev->button, ev->pressed);
            break;
        case JUKE_INPUT_KEY:
            /* ev->x contains the scancode */
            qemu_input_event_send_key_number(s->dcl.con, ev->x, ev->pressed);
            break;
        }

        read_idx++;
    }

    if (read_idx != ring->read_idx) {
        qemu_input_event_sync();
        __atomic_store_n(&ring->read_idx, read_idx, __ATOMIC_RELEASE);
    }
}

static void juke_shmem_gfx_update(DisplayChangeListener *dcl,
                                   int x, int y, int w, int h)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);

    if (!s->shmem || !s->surface) {
        return;
    }

    /* Copy dirty region from QEMU surface to shared memory */
    int stride = surface_stride(s->surface);
    uint8_t *src = surface_data(s->surface);
    uint8_t *dst = juke_shmem_pixels(s);  /* Pixels after header and input ring */

    /* Copy all rows in the dirty region */
    for (int row = y; row < y + h; row++) {
        memcpy(dst + row * stride, src + row * stride, stride);
    }

    /* Update dirty region in header */
    s->shmem->dirty_x = x;
    s->shmem->dirty_y = y;
    s->shmem->dirty_w = w;
    s->shmem->dirty_h = h;

    /* Bump frame counter with release semantics */
    __atomic_fetch_add(&s->shmem->frame_counter, 1, __ATOMIC_RELEASE);
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
    int stride = surface_stride(new_surface);
    size_t pixels_size = stride * h;
    /* Header + cursor data + input ring + pixels */
    size_t needed = sizeof(JukeShmemHeader) + JUKE_CURSOR_DATA_SIZE + sizeof(JukeInputRing) + pixels_size;

    /* Reallocate shared memory if size changed */
    if (needed > s->shmem_size) {
        if (s->shmem) {
            qemu_memfd_free(s->shmem, s->shmem_size, s->shmem_fd);
            s->shmem = NULL;
            s->shmem_fd = -1;
            s->fd_sent = false;
        }

        s->shmem_size = needed;
        s->shmem = qemu_memfd_alloc("juke-fb", needed, 0, &s->shmem_fd, NULL);

        if (!s->shmem) {
            error_report("juke-shmem: failed to allocate shared memory");
            return;
        }
    }

    /* Initialize header */
    s->shmem->magic = JUKE_SHMEM_MAGIC;
    s->shmem->version = JUKE_SHMEM_VERSION;
    s->shmem->width = w;
    s->shmem->height = h;
    s->shmem->stride = stride;
    s->shmem->format = surface_format(new_surface);
    s->shmem->frame_counter = 0;
    s->shmem->dirty_x = 0;
    s->shmem->dirty_y = 0;
    s->shmem->dirty_w = w;
    s->shmem->dirty_h = h;

    /* Initialize cursor state */
    s->shmem->cursor_version = 0;
    s->shmem->cursor_x = 0;
    s->shmem->cursor_y = 0;
    s->shmem->cursor_visible = 0;
    s->shmem->cursor_width = 0;
    s->shmem->cursor_height = 0;
    s->shmem->cursor_hot_x = 0;
    s->shmem->cursor_hot_y = 0;

    /* Initialize input ring buffer */
    JukeInputRing *ring = juke_shmem_input_ring(s);
    ring->write_idx = 0;
    ring->read_idx = 0;

    /* Copy initial surface content */
    uint8_t *src = surface_data(new_surface);
    uint8_t *dst = juke_shmem_pixels(s);
    memcpy(dst, src, pixels_size);

    /* Try to send fd if client connected */
    if (s->client_fd >= 0 && !s->fd_sent) {
        juke_shmem_send_fd(s);
    }
}

static void juke_shmem_refresh(DisplayChangeListener *dcl)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);

    /* Try to (re)connect if not connected */
    if (s->client_fd < 0 && s->socket_path) {
        juke_shmem_connect(s);
    }

    /* Try to send fd if we have connection and shared memory */
    if (s->client_fd >= 0 && s->shmem_fd >= 0 && !s->fd_sent) {
        juke_shmem_send_fd(s);
    }

    /* Process any pending input events from Juke */
    if (s->shmem) {
        juke_shmem_process_input(s);
    }

    graphic_hw_update(dcl->con);
}

/*
 * Handle cursor shape change from guest
 * Like Cocoa, we read from console cursor storage for reliability
 */
static void juke_shmem_cursor_define(DisplayChangeListener *dcl, QEMUCursor *cursor)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);

    if (!s->shmem) {
        return;
    }

    /* Read from console cursor storage like Cocoa does (more reliable than parameter) */
    QEMUCursor *con_cursor = qemu_console_get_cursor(dcl->con);
    if (!con_cursor) {
        /* No cursor - set zero dimensions to signal hidden */
        s->shmem->cursor_width = 0;
        s->shmem->cursor_height = 0;
        __atomic_fetch_add(&s->shmem->cursor_version, 1, __ATOMIC_RELEASE);
        return;
    }

    /* Clamp cursor size to max supported */
    uint32_t w = MIN(con_cursor->width, JUKE_CURSOR_MAX_SIZE);
    uint32_t h = MIN(con_cursor->height, JUKE_CURSOR_MAX_SIZE);

    /* Update cursor metadata */
    s->shmem->cursor_width = w;
    s->shmem->cursor_height = h;
    s->shmem->cursor_hot_x = con_cursor->hot_x;
    s->shmem->cursor_hot_y = con_cursor->hot_y;

    /* Copy cursor pixel data (32-bit RGBA) */
    uint32_t *cursor_pixels = juke_shmem_cursor_data(s);
    for (uint32_t y = 0; y < h; y++) {
        memcpy(&cursor_pixels[y * JUKE_CURSOR_MAX_SIZE],
               &con_cursor->data[y * con_cursor->width],
               w * sizeof(uint32_t));
    }

    /* Bump cursor version with release semantics so reader sees consistent data */
    __atomic_fetch_add(&s->shmem->cursor_version, 1, __ATOMIC_RELEASE);
}

/*
 * Handle cursor position/visibility change from guest
 */
static void juke_shmem_mouse_set(DisplayChangeListener *dcl, int x, int y, bool on)
{
    JukeShmemState *s = container_of(dcl, JukeShmemState, dcl);

    if (!s->shmem) {
        return;
    }

    s->shmem->cursor_x = x;
    s->shmem->cursor_y = y;
    s->shmem->cursor_visible = on ? 1 : 0;

    /* Use release semantics so Juke sees consistent state */
    __atomic_thread_fence(__ATOMIC_RELEASE);
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
    char buf[1] = {0};

    /* Ancillary data buffer for fd */
    char cmsgbuf[CMSG_SPACE(sizeof(int))];

    iov[0].iov_base = buf;
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &s->shmem_fd, sizeof(int));

    ssize_t sent = sendmsg(s->client_fd, &msg, 0);
    if (sent < 0) {
        error_report("juke-shmem: failed to send fd: %s", strerror(errno));
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

    /* Send fd if we already have shared memory */
    if (s->shmem_fd >= 0) {
        juke_shmem_send_fd(s);
    }

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
        update_displaychangelistener(&s->dcl, interval_ms);
    } else {
        /* Fallback: 8ms (~120Hz) - fast enough for any common display */
        error_report("juke-shmem: using fallback refresh rate: 8ms (~120Hz)");
        update_displaychangelistener(&s->dcl, 8);
    }
}

static void juke_shmem_init(DisplayState *ds, DisplayOptions *opts)
{
    JukeShmemState *s = g_new0(JukeShmemState, 1);

    s->dcl.con = qemu_console_lookup_default();
    s->dcl.ops = &juke_shmem_ops;
    s->shmem_fd = -1;
    s->client_fd = -1;

    if (opts->u.juke_shmem.socket) {
        s->socket_path = g_strdup(opts->u.juke_shmem.socket);
        /* Connect to Juke's socket - may fail if Juke hasn't created it yet */
        juke_shmem_connect(s);
    }

    register_displaychangelistener(&s->dcl);

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
