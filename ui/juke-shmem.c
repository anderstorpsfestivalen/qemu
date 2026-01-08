/*
 * Juke Shared Memory Display Backend
 *
 * Zero-copy framebuffer sharing between QEMU and Juke via mmap'd memory.
 * QEMU writes framebuffer data directly to shared memory, Juke reads it.
 * Uses atomic frame counter for update notification (no syscalls needed).
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
#include "qemu/memfd.h"
#include "qemu/sockets.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

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
    /* Pixel data follows immediately after header */
} JukeShmemHeader;

#define JUKE_SHMEM_MAGIC 0x454B554A  /* 'JUKE' */
#define JUKE_SHMEM_VERSION 1

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
    uint8_t *dst = (uint8_t *)(s->shmem + 1);  /* Pixels start after header */

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
    size_t needed = sizeof(JukeShmemHeader) + pixels_size;

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

    /* Copy initial surface content */
    uint8_t *src = surface_data(new_surface);
    uint8_t *dst = (uint8_t *)(s->shmem + 1);
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

    graphic_hw_update(dcl->con);
}

static const DisplayChangeListenerOps juke_shmem_ops = {
    .dpy_name          = "juke-shmem",
    .dpy_gfx_update    = juke_shmem_gfx_update,
    .dpy_gfx_switch    = juke_shmem_gfx_switch,
    .dpy_refresh       = juke_shmem_refresh,
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
