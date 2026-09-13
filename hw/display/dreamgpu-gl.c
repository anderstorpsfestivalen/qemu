/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Bounded process/context-aware GL execution and native image transport.
 */
#include "qemu/osdep.h"
#include "dreamgpu-host.h"
#include "qapi/error.h"
#include "qemu/bswap.h"
#include "qemu/error-report.h"
#include "qemu/sockets.h"
#include "qemu/thread.h"
#include "qemu/cutils.h"
#include "qemu/memfd.h"
#include <sys/mman.h>
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#include "standard-headers/dreamgpu/gl.h"
#include "standard-headers/dreamgpu/transport.h"
#include "dreamgpu-gl.h"
#include "dreamgpu-gl-platform.h"
#include "trace.h"

#ifdef CONFIG_DARWIN
#include <mach/mach.h>
#include <servers/bootstrap.h>
#endif

typedef DreamGpuContext DgContext;
typedef DreamGpuDrawable DgDrawable;

typedef enum SlotState { SLOT_FREE, SLOT_RENDERING, SLOT_PENDING, SLOT_PUBLISHED } SlotState;

typedef DreamGpuExportSlot DgSlot;

/* Frames and desktop commands share one publication order. Only frame jobs
 * wait for a GPU fence; the render thread can continue filling the next frame. */
typedef DreamGpuOutput DgOutput;
typedef DreamGpuCpuSlot DgCpuSlot;
#define DG_MAX_PENDING_DESKTOP 64

typedef DreamGpuBatch DgBatch;
G_STATIC_ASSERT(sizeof(((DgBatch *)0)->result) == DG_GL_MAX_RESULT_BYTES);
G_STATIC_ASSERT(sizeof(((DgGLCompletion *)0)->result) == DG_GL_MAX_RESULT_BYTES);
G_STATIC_ASSERT(G_N_ELEMENTS(((DreamGpuOutputQueue *)0)->items) ==
                DG_GL_MAX_DRAWABLES * DG_GL_EXPORT_SLOTS + DG_MAX_PENDING_DESKTOP);
#if UINTPTR_MAX == UINT64_MAX
G_STATIC_ASSERT(sizeof(DgBatch) == 576);
G_STATIC_ASSERT(sizeof(DgGLCompletion) == 576);
G_STATIC_ASSERT(offsetof(DgGLCompletion, bulk_result) == 568);
#endif

static void *submission_allocate(void *opaque, size_t bytes) {
    return g_try_malloc(bytes);
}

static void submission_free(void *opaque, void *allocation) {
    g_free(allocation);
}

static const DreamGpuSubmissionMemory submission_memory = {
    .allocate = submission_allocate,
    .free = submission_free,
};

struct DgGLEngine {
    QemuMutex lock, send_lock;
    QemuCond cond;
    QemuThread render_thread, completion_thread, release_thread;
    bool stopping, failed, threads_started;
    char *socket_path;
    int socket;
#ifdef CONFIG_DARWIN
    mach_port_t remote;
#endif
    uint8_t uuid[16];
    DgGLPlatform *platform;
    DreamGpuSubmission work;
    DreamGpuOutputQueue output;
    DreamGpuDesktopState desktop;
    DgGLFrameRef last_present;
    DgCpuSlot cpu_slots[DG_TRANSPORT_MAX_CPU_SLOTS];
    DgGLTransfer transfer;
    bool transfer_pending, transfer_claimed;
    uint32_t transfer_error;
    uint64_t cpu_epoch, cpu_generation;
    DreamGpuReply reply;
    DreamGpuResources resources;
    DgSlot slots[DG_GL_MAX_DRAWABLES * DG_GL_EXPORT_SLOTS];
    DgContext *current_context;
    DgDrawable *current_drawable;
    DgGLNotify notify;
    void *opaque;
};

static uint32_t word(const uint8_t *p, unsigned offset) {
    return ldl_le_p(p + offset);
}

/* Failure-only diagnostics carry bounded numeric command headers, never payloads. */
static void trace_rejection(const uint8_t *r, uint32_t size, uint32_t sequence, bool execution,
                            uint32_t error) {
    uint32_t op = word(r, DG_GL_OFF_OP);
    uint32_t args = op == DG_GL_DATA_CALL ? DG_GL_DATA_ARGS : 36;
    char tail[96];
    /* QEMU trace events allow at most ten arguments. Keep the original fields
     * and format only bounded numeric tail words, only on rejected records. */
    snprintf(tail, sizeof(tail), "a4=0x%x a5=0x%x a6=0x%x a7=0x%x",
             size >= args + 20 ? word(r, args + 16) : 0, size >= args + 24 ? word(r, args + 20) : 0,
             size >= args + 28 ? word(r, args + 24) : 0,
             size >= args + 32 ? word(r, args + 28) : 0);
    trace_dreamgpu_gl_reject(
        sequence, execution, op, size >= 36 ? word(r, 32) : 0, size >= args + 4 ? word(r, args) : 0,
        size >= args + 8 ? word(r, args + 4) : 0, size >= args + 12 ? word(r, args + 8) : 0,
        size >= args + 16 ? word(r, args + 12) : 0, error, tail);
}

static void batch_rejection(void *opaque, const uint8_t *record, uint32_t bytes, uint32_t error) {
    trace_rejection(record, bytes, 0, false, error);
}

uint32_t dg_gl_validate(const uint8_t *data, size_t bytes, uint32_t generation,
                        uint32_t primary_width, uint32_t primary_height, uint32_t vram_size,
                        uint32_t *records) {
    return dreamgpu_batch_validate(data, bytes, generation, primary_width, primary_height,
                                   vram_size, records, batch_rejection, NULL);
}

/* Ancillary handles belong to the first byte; never read across records. */
static bool receive_record_fd(int fd, uint8_t *data, int *received_fd) {
    size_t n = 0;
    bool valid = true;

    *received_fd = -1;
    while (n < DG_TRANSPORT_PACKET_BYTES) {
        union {
            struct cmsghdr align;
            char bytes[CMSG_SPACE(4 * sizeof(int))];
        } control;
        struct iovec iov = {data + n, DG_TRANSPORT_PACKET_BYTES - n};
        struct msghdr msg = {
            .msg_iov = &iov,
            .msg_iovlen = 1,
            .msg_control = control.bytes,
            .msg_controllen = sizeof(control),
        };
        ssize_t got = recvmsg(fd, &msg, 0);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            valid = false;
            break;
        }
        for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
            if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS ||
                c->cmsg_len < CMSG_LEN(sizeof(int))) {
                valid = false;
                continue;
            }
            size_t count = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            for (size_t i = 0; i < count; i++) {
                int handle;
                memcpy(&handle, CMSG_DATA(c) + i * sizeof(int), sizeof(handle));
                if (n || *received_fd >= 0) {
                    close(handle);
                    valid = false;
                } else {
                    *received_fd = handle;
                    qemu_set_cloexec(handle);
                }
            }
        }
        if (msg.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) {
            valid = false;
        }
        n += got;
    }
    valid &= n == DG_TRANSPORT_PACKET_BYTES &&
             word(data, DG_TRANSPORT_OFF_MAGIC) == DG_TRANSPORT_MAGIC &&
             word(data, DG_TRANSPORT_OFF_VERSION) == DG_TRANSPORT_VERSION &&
             word(data, DG_TRANSPORT_OFF_SIZE) == DG_TRANSPORT_PACKET_BYTES;
    if (!valid && *received_fd >= 0) {
        close(*received_fd);
        *received_fd = -1;
    }
    return valid;
}

static bool receive_record(int fd, uint8_t *data) {
    int received_fd;
    bool valid = receive_record_fd(fd, data, &received_fd);

    if (received_fd >= 0) {
        close(received_fd);
        return false;
    }
    return valid;
}

static bool send_record(DgGLEngine *e, const uint8_t *packet, const int *fds, unsigned count) {
    union {
        struct cmsghdr align;
        char bytes[CMSG_SPACE(2 * sizeof(int))];
    } control = {0};
    struct iovec iov = {(void *)packet, DG_TRANSPORT_PACKET_BYTES};
    struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1};
    ssize_t sent;
    bool ok = true;

    if (count) {
        msg.msg_control = control.bytes;
        msg.msg_controllen = CMSG_SPACE(count * sizeof(int));
        struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(count * sizeof(int));
        memcpy(CMSG_DATA(c), fds, count * sizeof(int));
    }
    qemu_mutex_lock(&e->send_lock);
    do {
        sent = sendmsg(e->socket, &msg, MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);
    if (sent <= 0) {
        ok = false;
    }
    while (ok && sent < DG_TRANSPORT_PACKET_BYTES) {
        ssize_t n = send(e->socket, packet + sent, DG_TRANSPORT_PACKET_BYTES - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            ok = false;
            break;
        }
        sent += n;
    }
    qemu_mutex_unlock(&e->send_lock);
    return ok;
}

static void packet_init(uint8_t *packet, uint32_t kind) {
    memset(packet, 0, DG_TRANSPORT_PACKET_BYTES);
    stl_le_p(packet + DG_TRANSPORT_OFF_MAGIC, DG_TRANSPORT_MAGIC);
    stl_le_p(packet + DG_TRANSPORT_OFF_VERSION, DG_TRANSPORT_VERSION);
    stl_le_p(packet + DG_TRANSPORT_OFF_KIND, kind);
    stl_le_p(packet + DG_TRANSPORT_OFF_SIZE, DG_TRANSPORT_PACKET_BYTES);
}

static bool connect_consumer(DgGLEngine *e, Error **errp) {
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    struct timeval timeout = {.tv_sec = 2};
    uint8_t hello[DG_TRANSPORT_PACKET_BYTES];

    if (!e->socket_path || strlen(e->socket_path) >= sizeof(addr.sun_path)) {
        error_setg(errp, "A valid gpu-socket path is required for GL export");
        return false;
    }
    pstrcpy(addr.sun_path, sizeof(addr.sun_path), e->socket_path);
    qemu_mutex_lock(&e->lock);
    e->socket = qemu_socket(AF_UNIX, SOCK_STREAM, 0);
    qemu_mutex_unlock(&e->lock);
    if (e->socket < 0 || connect(e->socket, (struct sockaddr *)&addr, sizeof(addr))) {
        error_setg_errno(errp, errno, "Connecting GPU consumer");
        return false;
    }
    setsockopt(e->socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(e->socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef CONFIG_DARWIN
    int yes = 1;
    setsockopt(e->socket, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    if (!receive_record(e->socket, hello) ||
        word(hello, DG_TRANSPORT_OFF_KIND) != DG_TRANSPORT_KIND_HELLO) {
        error_setg(errp, "Invalid GPU consumer handshake");
        return false;
    }
    timeout.tv_sec = 0;
    setsockopt(e->socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#ifdef CONFIG_DARWIN
    const char *service = (const char *)hello + DG_TRANSPORT_HELLO_MACH_SERVICE;
    if (word(hello, DG_TRANSPORT_HELLO_PLATFORM) != DG_TRANSPORT_PLATFORM_MACOS ||
        !memchr(service, 0, DG_TRANSPORT_HELLO_MACH_SERVICE_LEN) ||
        bootstrap_look_up(bootstrap_port, service, &e->remote) != KERN_SUCCESS) {
        error_setg(errp, "Looking up GPU consumer Mach service failed");
        return false;
    }
    e->platform = dg_gl_platform_new(NULL, errp);
#else
    const char *node = (const char *)hello + DG_TRANSPORT_HELLO_RENDER_NODE;
    if (word(hello, DG_TRANSPORT_HELLO_PLATFORM) != DG_TRANSPORT_PLATFORM_LINUX ||
        !memchr(node, 0, DG_TRANSPORT_HELLO_RENDER_NODE_LEN) || !node[0]) {
        error_setg(errp, "GPU consumer must identify its DRM render node");
        return false;
    }
    memcpy(e->uuid, hello + DG_TRANSPORT_OFF_DEVICE_UUID, sizeof(e->uuid));
    e->platform = dg_gl_platform_new(node, errp);
#endif
    return e->platform != NULL;
}

static void *release_worker(void *opaque) {
    DgGLEngine *e = opaque;
    uint8_t packet[DG_TRANSPORT_PACKET_BYTES];
    int received_fd;

    while (receive_record_fd(e->socket, packet, &received_fd)) {
        uint32_t kind = word(packet, DG_TRANSPORT_OFF_KIND);
        uint32_t slot = word(packet, DG_TRANSPORT_OFF_SLOT);
        uint64_t epoch = ldq_le_p(packet + DG_TRANSPORT_OFF_EPOCH);
        uint64_t sequence = ldq_le_p(packet + DG_TRANSPORT_OFF_GENERATION);
        bool valid = true;

        qemu_mutex_lock(&e->lock);
        if (kind == DG_TRANSPORT_KIND_DESKTOP_REPLY) {
            uint32_t result = dreamgpu_reply_receive(&e->reply, packet, received_fd);
            valid = result != 0;
            if (result == 2) {
                received_fd = -1;
            }
        } else if (kind != DG_TRANSPORT_KIND_RELEASE || received_fd >= 0) {
            valid = false;
        } else if (slot >= DG_TRANSPORT_CPU_SLOT_BASE &&
                   slot < DG_TRANSPORT_CPU_SLOT_BASE + DG_TRANSPORT_MAX_CPU_SLOTS) {
            DgCpuSlot *s = &e->cpu_slots[slot - DG_TRANSPORT_CPU_SLOT_BASE];
            dreamgpu_cpu_release(s, epoch, sequence);
        } else if (slot < G_N_ELEMENTS(e->slots)) {
            DgSlot *s = &e->slots[slot];
            dreamgpu_slot_release(s, epoch, sequence);
        } else {
            valid = false;
        }
        qemu_cond_broadcast(&e->cond);
        qemu_mutex_unlock(&e->lock);
        if (received_fd >= 0) {
            close(received_fd);
        }
        if (!valid) {
            break;
        }
    }
    qemu_mutex_lock(&e->lock);
    e->failed = true;
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
    return NULL;
}

static bool send_frame(DgGLEngine *e, DgSlot *s) {
#ifdef CONFIG_DARWIN
    struct {
        mach_msg_header_t header;
        mach_msg_body_t body;
        mach_msg_port_descriptor_t surface;
        uint8_t packet[DG_TRANSPORT_PACKET_BYTES];
    } msg = {0};
    mach_port_t port = dg_gl_image_port(s->image);
    kern_return_t ret;

    if (!MACH_PORT_VALID(port)) {
        return false;
    }
    msg.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) | MACH_MSGH_BITS_COMPLEX;
    msg.header.msgh_size = sizeof(msg);
    msg.header.msgh_remote_port = e->remote;
    msg.header.msgh_id = DG_TRANSPORT_MAGIC;
    msg.body.msgh_descriptor_count = 1;
    msg.surface.name = port;
    msg.surface.disposition = MACH_MSG_TYPE_COPY_SEND;
    msg.surface.type = MACH_MSG_PORT_DESCRIPTOR;
    memcpy(msg.packet, s->packet, sizeof(msg.packet));
    ret = mach_msg(&msg.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT, sizeof(msg), 0, MACH_PORT_NULL,
                   2000, MACH_PORT_NULL);
    mach_port_deallocate(mach_task_self(), port);
    return ret == KERN_SUCCESS;
#else
    int fds[2] = {dg_gl_image_fd(s->image), dg_gl_image_fence_fd(s->image)};
    return send_record(e, s->packet, fds, fds[1] >= 0 ? 2 : 1);
#endif
}

static void *completion_worker(void *opaque) {
    DgGLEngine *e = opaque;

    qemu_mutex_lock(&e->lock);
    for (;;) {
        DgSlot *s;
        Error *err = NULL;

        while (!e->output.len && !e->stopping) {
            qemu_cond_wait(&e->cond, &e->lock);
        }
        if (!e->output.len) {
            break;
        }
        DgOutput output;
        if (!dreamgpu_output_pop(&e->output, &output)) {
            break;
        }
        s = output.slot;
        if (!s) {
            bool publish = !e->failed && !e->stopping && !e->work.reset;
            qemu_mutex_unlock(&e->lock);
            bool sent = publish && send_record(e, output.packet, NULL, 0);
            qemu_mutex_lock(&e->lock);
            if (publish && !sent) {
                e->failed = true;
            }
            dreamgpu_output_done(&e->output, 1);
            qemu_cond_broadcast(&e->cond);
            continue;
        }
        qemu_mutex_unlock(&e->lock);
        bool ready = dg_gl_image_ready(s->image, &err);
        qemu_mutex_lock(&e->lock);
        bool publish = dreamgpu_slot_publish(s, ready, e->failed || e->stopping || e->work.reset);
        qemu_mutex_unlock(&e->lock);
        bool sent = publish && send_frame(e, s);
        if (err) {
            error_report_err(err);
        }
        qemu_mutex_lock(&e->lock);
        unsigned index = (s - e->slots) / DG_GL_EXPORT_SLOTS;
        if (dreamgpu_slot_sent(s, &e->resources.drawables[index], sent, publish, ready)) {
            e->failed = true;
        }
        dreamgpu_output_done(&e->output, 0);
        qemu_cond_broadcast(&e->cond);
    }
    qemu_mutex_unlock(&e->lock);
    return NULL;
}

static bool make_current(DgGLEngine *e, DgContext *c, DgDrawable *d, Error **errp) {
    if (e->current_context == c && e->current_drawable == d) {
        return true;
    }
    if (e->current_context && e->current_drawable &&
        !dg_gl_context_in_begin(e->current_context->native)) {
        dg_gl_flush_drawable(e->current_drawable->native);
    }
    if (!dg_gl_make_current(c->native, d->native, errp)) {
        return false;
    }
    e->current_context = c;
    e->current_drawable = d;
    return true;
}

static void drain_output(DgGLEngine *e) {
    qemu_mutex_lock(&e->lock);
    while (e->output.pending) {
        qemu_cond_wait(&e->cond, &e->lock);
    }
    qemu_mutex_unlock(&e->lock);
}

/* Count queued and in-flight packets so a stalled consumer cannot grow the
 * queue without bound. Blocking here applies only at the fixed backlog limit. */
static uint32_t queue_desktop(DgGLEngine *e, const uint8_t *packet) {
    uint32_t error = 0;

    qemu_mutex_lock(&e->lock);
    while (e->output.desktop == DG_MAX_PENDING_DESKTOP && !e->stopping && !e->work.reset &&
           !e->failed) {
        qemu_cond_wait(&e->cond, &e->lock);
    }
    if (e->work.reset) {
        error = DG_GL_ERROR_GENERATION;
    } else if (e->stopping || e->failed) {
        error = DG_GL_ERROR_TRANSPORT;
    } else if (!dreamgpu_output_push(&e->output, NULL, packet)) {
        error = DG_GL_ERROR_LIMIT;
    } else {
        qemu_cond_broadcast(&e->cond);
    }
    qemu_mutex_unlock(&e->lock);
    return error;
}

static void drop_drawable_resource(DgGLEngine *e, DgDrawable *d) {
    uint8_t packet[DG_TRANSPORT_PACKET_BYTES];

    if (!d->published_generation) {
        return;
    }
    packet_init(packet, DG_TRANSPORT_KIND_RESOURCE_DROP);
    stl_le_p(packet + DG_TRANSPORT_OFF_CLIENT, d->client);
    stl_le_p(packet + DG_TRANSPORT_OFF_DRAWABLE, d->id);
    stq_le_p(packet + DG_TRANSPORT_OFF_EPOCH, d->published_epoch);
    stq_le_p(packet + DG_TRANSPORT_OFF_GENERATION, d->published_generation);
    qemu_mutex_lock(&e->lock);
    bool publish = !e->stopping && !e->failed;
    qemu_mutex_unlock(&e->lock);
    if (publish && !send_record(e, packet, NULL, 0)) {
        qemu_mutex_lock(&e->lock);
        e->failed = true;
        qemu_mutex_unlock(&e->lock);
    }
}

static void delete_drawable(DgGLEngine *e, DgDrawable *d) {
    unsigned index = d - e->resources.drawables;

    drain_output(e);
    qemu_mutex_lock(&e->lock);
    bool resetting = e->work.reset;
    qemu_mutex_unlock(&e->lock);
    if (!resetting) {
        drop_drawable_resource(e, d);
    }
    e->current_context = NULL;
    e->current_drawable = NULL;
    for (unsigned i = 0; i < DG_GL_EXPORT_SLOTS; i++) {
        DgSlot *s = &e->slots[index * DG_GL_EXPORT_SLOTS + i];
        qemu_mutex_lock(&e->lock);
        while (s->state == SLOT_PUBLISHED && !e->stopping && !e->failed) {
            qemu_cond_wait(&e->cond, &e->lock);
        }
        qemu_mutex_unlock(&e->lock);
        /* Exported handles retain backing storage independently of this VM. */
        dg_gl_image_free(e->platform, s->image);
        qemu_mutex_lock(&e->lock);
        s->image = NULL;
        dreamgpu_slot_abort(s);
        qemu_mutex_unlock(&e->lock);
    }
    dg_gl_drawable_free(e->platform, d->native);
}

static void close_client(DgGLEngine *e, uint32_t client);

static uint32_t present(DgGLEngine *e, DgContext *c, DgDrawable *d, uint32_t flags, Error **errp) {
    unsigned first = (d - e->resources.drawables) * DG_GL_EXPORT_SLOTS;
    DgSlot *s = NULL;
    uint32_t stride, offset;
    uint64_t modifier;

    if (flags & DG_GL_PRESENT_NO_EXPORT) {
        dg_gl_exchange(c->native, d->native);
        return 0;
    }
    qemu_mutex_lock(&e->lock);
    while (!e->stopping && !e->work.reset && !e->failed) {
        uint32_t slot = dreamgpu_slot_claim(e->slots, first);
        if (slot != UINT32_MAX) {
            s = &e->slots[slot];
        }
        if (s) {
            break;
        }
        qemu_cond_wait(&e->cond, &e->lock);
    }
    qemu_mutex_unlock(&e->lock);
    if (!s) {
        return DG_GL_ERROR_TRANSPORT;
    }
    if (!s->image) {
        s->image = dg_gl_image_new(e->platform, d->width, d->height, errp);
    }
    if (!s->image ||
        !dg_gl_export(c->native, d->native, s->image, !(flags & DG_GL_PRESENT_FRONT_ONLY), errp)) {
        qemu_mutex_lock(&e->lock);
        dreamgpu_slot_abort(s);
        qemu_mutex_unlock(&e->lock);
        return DG_GL_ERROR_HOST;
    }
    dg_gl_image_metadata(s->image, &stride, &offset, &modifier);
    if (flags & DG_GL_PRESENT_EXCLUSIVE) {
        e->desktop.exclusive = true;
    }
    DreamGpuImageMetadata metadata = {
        .stride = stride,
        .offset = offset,
        .modifier = modifier,
        .ready_fence = dg_gl_image_fence_fd(s->image) >= 0,
    };
    memcpy(metadata.uuid, e->uuid, sizeof(e->uuid));
    uint32_t error = dreamgpu_slot_prepare(s, d, s - e->slots, flags, &metadata);
    if (error) {
        qemu_mutex_lock(&e->lock);
        dreamgpu_slot_abort(s);
        qemu_mutex_unlock(&e->lock);
        return error;
    }
    qemu_mutex_lock(&e->lock);
    e->last_present = (DgGLFrameRef){
        .slot = s - e->slots,
        .client = d->client,
        .drawable = d->id,
        .epoch = d->epoch,
        .generation = s->generation,
    };
    dreamgpu_slot_pending(s);
    if (!dreamgpu_output_push(&e->output, s, NULL)) {
        dreamgpu_slot_abort(s);
        qemu_mutex_unlock(&e->lock);
        return DG_GL_ERROR_LIMIT;
    }
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
    return 0;
}

static uint32_t transfer_pixels(DgGLEngine *e, const uint8_t *r, uint8_t *pixels, uint32_t stride,
                                const DgBatch *batch, bool writeback, bool return_cpu) {
    qemu_mutex_lock(&e->lock);
    e->transfer = (DgGLTransfer){
        .pixels = pixels,
        .stride = stride,
        .width = word(r, DG_DESKTOP_WIDTH),
        .height = word(r, DG_DESKTOP_HEIGHT),
        .offset = word(r, DG_DESKTOP_SLOT_OR_OFFSET),
        .vram_stride = word(r, DG_DESKTOP_VRAM_STRIDE),
        .generation = batch->generation,
        .writeback = writeback,
        .return_cpu = return_cpu,
    };
    e->transfer_pending = true;
    e->transfer_claimed = false;
    e->transfer_error = 0;
    qemu_mutex_unlock(&e->lock);
    e->notify(e->opaque);
    qemu_mutex_lock(&e->lock);
    /*
     * Reset completion is acknowledged by BQL before this allocation can be
     * released. Stop is different: the owner cancels its BH before teardown.
     */
    while (e->transfer_pending && !e->stopping) {
        qemu_cond_wait(&e->cond, &e->lock);
    }
    uint32_t error = e->stopping ? DG_GL_ERROR_GENERATION : e->transfer_error;
    qemu_mutex_unlock(&e->lock);
    return error;
}

static uint8_t *cpu_allocate(void *opaque, size_t bytes, int32_t *fd) {
    return qemu_memfd_alloc("dreamgpu-desktop", bytes, 0, fd, opaque);
}

static void cpu_free(void *opaque, uint8_t *pixels, size_t bytes, int32_t fd) {
    qemu_memfd_free(pixels, bytes, fd);
}

static DgCpuSlot *cpu_slot(DgGLEngine *e, uint32_t width, uint32_t height, Error **errp) {
    DgCpuSlot *slot = NULL;
    DreamGpuCpuMemory memory = {errp, cpu_allocate, cpu_free};

    qemu_mutex_lock(&e->lock);
    while (!e->stopping && !e->work.reset && !e->failed) {
        uint32_t index = dreamgpu_cpu_claim(e->cpu_slots, e->desktop.epoch, e->desktop.sequence);
        if (index != UINT32_MAX) {
            slot = &e->cpu_slots[index];
            break;
        }
        qemu_cond_wait(&e->cond, &e->lock);
    }
    qemu_mutex_unlock(&e->lock);
    if (slot && dreamgpu_cpu_storage(slot, &memory, width, height)) {
        qemu_mutex_lock(&e->lock);
        dreamgpu_cpu_release(slot, slot->epoch, slot->sequence);
        qemu_mutex_unlock(&e->lock);
        return NULL;
    }
    return slot;
}

static uint32_t capture_desktop(DgGLEngine *e, const uint8_t *r, const DgBatch *batch,
                                uint32_t subtype, Error **errp) {
    uint32_t w = word(r, DG_DESKTOP_WIDTH);
    uint32_t h = word(r, DG_DESKTOP_HEIGHT);
    uint8_t packet[DG_TRANSPORT_PACKET_BYTES];
    DgCpuSlot *s;
    uint32_t error;

    s = cpu_slot(e, w, h, errp);
    if (!s) {
        return DG_GL_ERROR_TRANSPORT;
    }
    error = transfer_pixels(e, r, s->pixels, s->stride, batch, false,
                            subtype == DG_TRANSPORT_CPU_RETURN);
    if (!error) {
        dreamgpu_cpu_packet(s, s - e->cpu_slots, subtype, r, e->cpu_epoch, e->cpu_generation,
                            packet);
        if (!send_record(e, packet, &s->fd, 1)) {
            error = DG_GL_ERROR_TRANSPORT;
        }
    }
    if (error) {
        qemu_mutex_lock(&e->lock);
        dreamgpu_cpu_release(s, s->epoch, s->sequence);
        qemu_mutex_unlock(&e->lock);
    }
    return error;
}

static uint32_t readback_desktop(DgGLEngine *e, const uint8_t *r, const DgBatch *batch,
                                 uint8_t *packet) {
    uint32_t error = 0;
    uint32_t w = word(r, DG_DESKTOP_WIDTH);
    uint32_t h = word(r, DG_DESKTOP_HEIGHT);
    DreamGpuReplyResult result;
    int fd = -1;
    struct stat statbuf;
    uint8_t *pixels = MAP_FAILED;
    uint64_t bytes = 0;
    uint32_t stride = 0;

    qemu_mutex_lock(&e->lock);
    dreamgpu_reply_begin(&e->reply, e->desktop.epoch, e->desktop.sequence, packet);
    qemu_mutex_unlock(&e->lock);
    if (!send_record(e, packet, NULL, 0)) {
        error = DG_GL_ERROR_TRANSPORT;
    }
    qemu_mutex_lock(&e->lock);
    int64_t deadline = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;
    while (!error && !e->reply.ready && !e->stopping && !e->work.reset && !e->failed) {
        int64_t remaining = deadline - g_get_monotonic_time();
        if (remaining <= 0) {
            break;
        }
        qemu_cond_timedwait(&e->cond, &e->lock, DIV_ROUND_UP(remaining, G_TIME_SPAN_MILLISECOND));
    }
    dreamgpu_reply_finish(&e->reply, error, e->stopping || e->work.reset, e->failed, &result);
    qemu_mutex_unlock(&e->lock);
    fd = result.fd;
    bool stat_ok = !result.error && fd >= 0 && !fstat(fd, &statbuf);
    error = dreamgpu_reply_layout(&result, w, h, stat_ok, stat_ok ? statbuf.st_size : 0, &bytes,
                                  &stride);
    if (!error) {
        pixels = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (pixels == MAP_FAILED) {
            error = DG_GL_ERROR_HOST;
        } else {
            error = transfer_pixels(e, r, pixels, stride, batch, true, false);
        }
    }
    if (pixels != MAP_FAILED) {
        munmap(pixels, bytes);
    }
    if (fd >= 0) {
        close(fd);
    }
    return error;
}

typedef struct DgDesktopCall {
    DgGLEngine *engine;
    const DgBatch *batch;
    Error **errp;
} DgDesktopCall;

static uint32_t desktop_capture(void *opaque, const uint8_t *r, uint32_t subtype) {
    DgDesktopCall *call = opaque;
    drain_output(call->engine);
    return capture_desktop(call->engine, r, call->batch, subtype, call->errp);
}

static uint32_t desktop_readback(void *opaque, const uint8_t *r, uint8_t *packet) {
    DgDesktopCall *call = opaque;
    drain_output(call->engine);
    return readback_desktop(call->engine, r, call->batch, packet);
}

static uint32_t desktop_queue(void *opaque, const uint8_t *packet) {
    DgDesktopCall *call = opaque;
    return queue_desktop(call->engine, packet);
}

static uint32_t desktop_retained(void *opaque, const uint8_t *record) {
    DgDesktopCall *call = opaque;
    DgGLEngine *e = call->engine;
    uint32_t index = word(record + DG_GL_HEADER_BYTES, DG_DESKTOP_SLOT_OR_OFFSET);
    qemu_mutex_lock(&e->lock);
    uint32_t valid = dreamgpu_desktop_retained(&e->slots[index], record);
    qemu_mutex_unlock(&e->lock);
    return valid;
}

static void desktop_failed(void *opaque) {
    DgDesktopCall *call = opaque;
    DgGLEngine *e = call->engine;
    qemu_mutex_lock(&e->lock);
    e->failed = true;
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
}

static uint32_t execute_desktop(DgGLEngine *e, const uint8_t *record, const DgBatch *batch,
                                Error **errp) {
    DgDesktopCall call = {.engine = e, .batch = batch, .errp = errp};
    const DreamGpuDesktopOps ops = {
        .opaque = &call,
        .capture = desktop_capture,
        .readback = desktop_readback,
        .queue = desktop_queue,
        .retained = desktop_retained,
        .failed = desktop_failed,
    };
    return dreamgpu_desktop_execute(&e->desktop, &ops, record, batch->primary_width,
                                    batch->primary_height);
}

bool dg_gl_engine_transfer(DgGLEngine *e, DgGLTransfer *transfer) {
    bool available;

    qemu_mutex_lock(&e->lock);
    available = e->transfer_pending && !e->transfer_claimed;
    if (available) {
        *transfer = e->transfer;
        e->transfer_claimed = true;
    }
    qemu_mutex_unlock(&e->lock);
    return available;
}

void dg_gl_engine_transfer_done(DgGLEngine *e, uint32_t error, uint64_t cpu_epoch,
                                uint64_t cpu_generation) {
    qemu_mutex_lock(&e->lock);
    e->transfer_error = error;
    e->cpu_epoch = cpu_epoch;
    e->cpu_generation = cpu_generation;
    e->transfer_pending = false;
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
}

/* Native API callbacks deliberately contain no command routing or resource
 * admission policy; the Rust dispatcher owns those decisions and registry. */
typedef struct DgDispatch {
    DgGLEngine *engine;
    DgBatch *batch;
    Error **errp;
} DgDispatch;

static void *host_context_new(void *opaque, void *share) {
    DgDispatch *d = opaque;
    return dg_gl_context_new(d->engine->platform, share, d->errp);
}

static void *host_drawable_new(void *opaque, uint32_t width, uint32_t height) {
    DgDispatch *d = opaque;
    DgGLEngine *e = d->engine;
    drain_output(e);
    if (e->current_context && e->current_drawable &&
        !dg_gl_context_in_begin(e->current_context->native)) {
        dg_gl_flush_drawable(e->current_drawable->native);
    }
    void *native = dg_gl_drawable_new(e->platform, width, height, d->errp);
    e->current_context = NULL;
    e->current_drawable = NULL;
    return native;
}

static void host_context_free(void *opaque, uint32_t index) {
    DgDispatch *d = opaque;
    DgGLEngine *e = d->engine;
    DgContext *c = &e->resources.contexts[index];
    drain_output(e);
    if (e->current_context == c && e->current_drawable && !dg_gl_context_in_begin(c->native)) {
        dg_gl_flush_drawable(e->current_drawable->native);
    }
    dg_gl_context_free(c->native);
    e->current_context = NULL;
}

static void host_drawable_free(void *opaque, uint32_t index) {
    DgDispatch *d = opaque;
    delete_drawable(d->engine, &d->engine->resources.drawables[index]);
}

static void host_close_begin(void *opaque) {
    DgGLEngine *e = ((DgDispatch *)opaque)->engine;
    drain_output(e);
    e->current_context = NULL;
    e->current_drawable = NULL;
}

static uint32_t host_in_begin(void *opaque, uint32_t index) {
    DgGLEngine *e = ((DgDispatch *)opaque)->engine;
    return dg_gl_context_in_begin(e->resources.contexts[index].native);
}

static uint32_t host_make_current(void *opaque, uint32_t ci, uint32_t di) {
    DgDispatch *d = opaque;
    return make_current(d->engine, &d->engine->resources.contexts[ci],
                        &d->engine->resources.drawables[di], d->errp)
               ? 0
               : DG_GL_ERROR_HOST;
}

static uint32_t host_call(void *opaque, uint32_t ci, uint32_t fn, const uint8_t *args) {
    DgGLEngine *e = ((DgDispatch *)opaque)->engine;
    return dg_gl_call(e->resources.contexts[ci].native, fn, args);
}

static uint32_t host_data(void *opaque, uint32_t ci, uint32_t fn, const uint8_t *args,
                          const uint8_t *data, uint32_t bytes) {
    DgGLEngine *e = ((DgDispatch *)opaque)->engine;
    return dg_gl_data_call(e->resources.contexts[ci].native, fn, args, data, bytes);
}

static uint32_t host_words(void *opaque, uint32_t fn) {
    return dg_gl_function_words(fn);
}

static uint32_t host_query(void *opaque, uint32_t ci, uint32_t fn, const uint8_t *args) {
    DgDispatch *d = opaque;
    DgBatch *batch = d->batch;
    uint32_t required = dg_gl_query_result_bytes(fn, args);
    uint8_t *result = dreamgpu_submission_query(batch, &submission_memory, required);
    if (!result) {
        return DG_GL_ERROR_LIMIT;
    }
    return dg_gl_query(d->engine->resources.contexts[ci].native, fn, args, result, required,
                       &batch->result_bytes, &batch->result_type);
}

static uint32_t host_present(void *opaque, uint32_t ci, uint32_t di, uint32_t flags) {
    DgDispatch *d = opaque;
    return present(d->engine, &d->engine->resources.contexts[ci],
                   &d->engine->resources.drawables[di], flags, d->errp);
}

static uint32_t host_desktop(void *opaque, const uint8_t *record) {
    DgDispatch *d = opaque;
    return execute_desktop(d->engine, record, d->batch, d->errp);
}

static DreamGpuPlatform host_platform(DgDispatch *dispatch) {
    return (DreamGpuPlatform){
        .opaque = dispatch,
        .context_new = host_context_new,
        .drawable_new = host_drawable_new,
        .context_free = host_context_free,
        .drawable_free = host_drawable_free,
        .close_begin = host_close_begin,
        .in_begin = host_in_begin,
        .make_current = host_make_current,
        .call = host_call,
        .data = host_data,
        .words = host_words,
        .query = host_query,
        .present = host_present,
        .desktop = host_desktop,
    };
}

static void close_client(DgGLEngine *e, uint32_t client) {
    DgDispatch dispatch = {.engine = e};
    DreamGpuPlatform platform = host_platform(&dispatch);
    dreamgpu_gl_close_client(&e->resources, &platform, client);
}

static uint32_t submission_cancelled(void *opaque) {
    DgGLEngine *e = ((DgDispatch *)opaque)->engine;
    qemu_mutex_lock(&e->lock);
    bool cancelled = e->work.reset || e->stopping;
    qemu_mutex_unlock(&e->lock);
    return cancelled;
}

static void submission_report(void *opaque, const uint8_t *record, uint32_t error) {
    DgDispatch *d = opaque;
    if (error) {
        trace_rejection(record, word(record, DG_GL_OFF_SIZE), d->batch->sequence, true, error);
    }
    if (*d->errp) {
        error_report_err(*d->errp);
        *d->errp = NULL;
    }
}

static void *render_worker(void *opaque) {
    DgGLEngine *e = opaque;
    Error *err = NULL;

    if (!connect_consumer(e, &err)) {
        error_report_err(err);
        qemu_mutex_lock(&e->lock);
        e->failed = true;
        qemu_mutex_unlock(&e->lock);
    } else {
        qemu_mutex_lock(&e->lock);
        if (e->stopping) {
            shutdown(e->socket, SHUT_RDWR);
            qemu_mutex_unlock(&e->lock);
            return NULL;
        }
        qemu_mutex_unlock(&e->lock);
        qemu_thread_create(&e->release_thread, "dreamgpu-gl-release", release_worker, e,
                           QEMU_THREAD_JOINABLE);
        qemu_thread_create(&e->completion_thread, "dreamgpu-gl-fence", completion_worker, e,
                           QEMU_THREAD_JOINABLE);
        e->threads_started = true;
    }
    qemu_mutex_lock(&e->lock);
    while (!e->stopping) {
        DgBatch *batch;
        uint32_t result = 0;

        while (!e->work.pending && !e->work.reset && !e->stopping) {
            qemu_cond_wait(&e->cond, &e->lock);
        }
        if (e->stopping) {
            break;
        }
        if (e->work.reset) {
            DreamGpuResetTicket ticket;
            if (!dreamgpu_submission_reset_snapshot(&e->work, &ticket)) {
                continue;
            }
            qemu_mutex_unlock(&e->lock);
            drain_output(e);
            for (unsigned i = 0; i < G_N_ELEMENTS(e->resources.drawables); i++) {
                if (e->resources.drawables[i].native) {
                    drop_drawable_resource(e, &e->resources.drawables[i]);
                }
            }
            uint8_t reset_packet[DG_TRANSPORT_PACKET_BYTES];
            packet_init(reset_packet, DG_TRANSPORT_KIND_RESET);
            stq_le_p(reset_packet + DG_TRANSPORT_CPU_OFF_LEGACY_EPOCH, ticket.epoch);
            stq_le_p(reset_packet + DG_TRANSPORT_CPU_OFF_LEGACY_FRAME, ticket.frame);
            if (!send_record(e, reset_packet, NULL, 0)) {
                qemu_mutex_lock(&e->lock);
                e->failed = true;
                qemu_mutex_unlock(&e->lock);
            }
            close_client(e, 0);
            qemu_mutex_lock(&e->lock);
            if (!dreamgpu_submission_reset_done(&e->work, &submission_memory, &ticket, &e->desktop,
                                                &e->last_present)) {
                /* A newer reset arrived during native drain/send/close. */
                continue;
            }
            qemu_mutex_unlock(&e->lock);
            e->notify(e->opaque);
            qemu_mutex_lock(&e->lock);
        }
        batch = dreamgpu_submission_take(&e->work);
        if (!batch) {
            continue;
        }
        result = e->failed ? DG_GL_ERROR_TRANSPORT : 0;
        qemu_mutex_unlock(&e->lock);
        uint64_t trace_start_us = batch->trace_queued_us ? g_get_monotonic_time() : 0;
        DgDispatch dispatch = {e, batch, &err};
        DreamGpuPlatform platform = host_platform(&dispatch);
        DreamGpuSubmissionRun run = {
            &dispatch,
            submission_cancelled,
            submission_report,
        };
        result = dreamgpu_submission_run(batch, &e->resources, &platform, &run, result);
        if (trace_start_us) {
            trace_dreamgpu_gl_work(batch->sequence, batch->records, batch->bytes,
                                   trace_start_us - batch->trace_queued_us,
                                   g_get_monotonic_time() - trace_start_us, result);
        }
        qemu_mutex_lock(&e->lock);
        dreamgpu_submission_complete(&e->work, &submission_memory, batch, result, &e->resources,
                                     &e->desktop, &e->last_present);
        qemu_mutex_unlock(&e->lock);
        e->notify(e->opaque);
        qemu_mutex_lock(&e->lock);
    }
    qemu_mutex_unlock(&e->lock);
    dg_gl_clear_current(e->platform);
    return NULL;
}

DgGLEngine *dg_gl_engine_new(const char *socket_path, DgGLNotify notify, void *opaque) {
    DgGLEngine *e = g_new0(DgGLEngine, 1);

    qemu_mutex_init(&e->lock);
    qemu_mutex_init(&e->send_lock);
    qemu_cond_init(&e->cond);
    e->socket_path = g_strdup(socket_path);
    e->socket = -1;
    e->reply.fd = -1;
    for (unsigned i = 0; i < G_N_ELEMENTS(e->cpu_slots); i++) {
        e->cpu_slots[i].fd = -1;
    }
    e->notify = notify;
    e->opaque = opaque;
    qemu_thread_create(&e->render_thread, "dreamgpu-gl", render_worker, e, QEMU_THREAD_JOINABLE);
    return e;
}

void dg_gl_engine_free(DgGLEngine *e) {
    if (!e) {
        return;
    }
    qemu_mutex_lock(&e->lock);
    e->stopping = true;
    if (e->socket >= 0) {
        shutdown(e->socket, SHUT_RDWR);
    }
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
    qemu_thread_join(&e->render_thread);
    if (e->threads_started) {
        qemu_thread_join(&e->completion_thread);
        qemu_thread_join(&e->release_thread);
    }
    close_client(e, 0);
    dg_gl_platform_free(e->platform);
    dreamgpu_submission_free(&e->work, &submission_memory);
#ifdef CONFIG_DARWIN
    if (e->remote) {
        mach_port_deallocate(mach_task_self(), e->remote);
    }
#endif
    if (e->socket >= 0) {
        close(e->socket);
    }
    for (unsigned i = 0; i < G_N_ELEMENTS(e->cpu_slots); i++) {
        DgCpuSlot *s = &e->cpu_slots[i];
        DreamGpuCpuMemory memory = {NULL, cpu_allocate, cpu_free};
        dreamgpu_cpu_storage_free(s, &memory);
    }
    if (e->reply.fd >= 0) {
        close(e->reply.fd);
    }
    g_free(e->socket_path);
    qemu_cond_destroy(&e->cond);
    qemu_mutex_destroy(&e->lock);
    qemu_mutex_destroy(&e->send_lock);
    g_free(e);
}

bool dg_gl_engine_submit(DgGLEngine *e, uint8_t *data, size_t bytes, uint32_t sequence,
                         uint32_t generation, uint32_t primary_width, uint32_t primary_height,
                         uint32_t records) {
    DreamGpuSubmissionRequest request = {
        .data = data,
        .bytes = bytes,
        .sequence = sequence,
        .generation = generation,
        .records = records,
        .trace_queued_us =
            trace_event_get_state_backends(TRACE_DREAMGPU_GL_WORK) ? g_get_monotonic_time() : 0,
        .primary_width = primary_width,
        .primary_height = primary_height,
    };
    qemu_mutex_lock(&e->lock);
    bool accepted = dreamgpu_submission_submit(&e->work, &submission_memory, &request, e->stopping);
    if (accepted) {
        qemu_cond_broadcast(&e->cond);
    }
    qemu_mutex_unlock(&e->lock);
    return accepted;
}

void dg_gl_engine_reset(DgGLEngine *e, uint32_t generation, uint64_t cpu_epoch,
                        uint64_t cpu_generation) {
    qemu_mutex_lock(&e->lock);
    dreamgpu_submission_reset(&e->work, &submission_memory, generation, cpu_epoch, cpu_generation);
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
}

bool dg_gl_engine_completion(DgGLEngine *e, DgGLCompletion *completion) {
    qemu_mutex_lock(&e->lock);
    bool available = dreamgpu_submission_poll(&e->work, completion);
    qemu_mutex_unlock(&e->lock);
    return available;
}
