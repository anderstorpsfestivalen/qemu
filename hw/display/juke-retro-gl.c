/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Bounded process/context-aware GL execution and native image transport.
 */
#include "qemu/osdep.h"
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
#include "standard-headers/juke/retro-gl.h"
#include "standard-headers/juke/gpu-transport.h"
#include "juke-retro-gl.h"
#include "juke-retro-gl-platform.h"
#include "trace.h"

#ifdef CONFIG_DARWIN
#include <mach/mach.h>
#include <servers/bootstrap.h>
#endif

typedef struct JrgContext {
    uint32_t client, id, drawable;
    JrgGLContext *native;
} JrgContext;

typedef struct JrgDrawable {
    uint32_t client, id, width, height;
    uint64_t epoch, generation;
    uint64_t published_epoch, published_generation;
    JrgGLDrawable *native;
} JrgDrawable;

typedef enum SlotState { SLOT_FREE, SLOT_RENDERING, SLOT_PENDING,
                         SLOT_PUBLISHED } SlotState;

typedef struct JrgSlot {
    JrgGLImage *image;
    SlotState state;
    bool sending, release_pending;
    uint64_t generation;
    uint8_t packet[JGPU_PACKET_BYTES];
} JrgSlot;

/* Frames and desktop commands share one publication order. Only frame jobs
 * wait for a GPU fence; the render thread can continue filling the next frame. */
typedef struct JrgOutput {
    JrgSlot *slot;
    uint8_t packet[JGPU_PACKET_BYTES];
} JrgOutput;

#define JRG_MAX_PENDING_DESKTOP 64

typedef struct JrgCpuSlot {
    uint8_t *pixels;
    size_t bytes;
    int fd;
    uint32_t width, height, stride;
    bool published;
    uint64_t epoch, sequence;
} JrgCpuSlot;

typedef struct JrgBatch {
    uint8_t *data;
    size_t bytes;
    uint32_t sequence, generation;
    uint32_t records;
    uint64_t trace_queued_us;
    uint32_t primary_width, primary_height;
    uint32_t result_bytes, result_type;
    uint8_t result[JRG_GL_MAX_RESULT_BYTES];
} JrgBatch;

struct JrgGLEngine {
    QemuMutex lock, send_lock;
    QemuCond cond;
    QemuThread render_thread, completion_thread, release_thread;
    bool stopping, failed, reset, threads_started;
    bool exclusive;
    uint32_t reset_generation;
    uint64_t reset_cpu_epoch, reset_cpu_generation;
    char *socket_path;
    int socket;
#ifdef CONFIG_DARWIN
    mach_port_t remote;
#endif
    uint8_t uuid[16];
    JrgGLPlatform *platform;
    JrgBatch *batch;
    GQueue completions;
    GQueue pending_output;
    uint32_t pending_count, pending_desktop;
    uint64_t next_epoch;
    uint64_t desktop_epoch, desktop_sequence;
    uint32_t desktop_width, desktop_height;
    bool desktop_active, desktop_coherent;
    JrgGLFrameRef last_present;
    JrgCpuSlot cpu_slots[JGPU_MAX_CPU_SLOTS];
    JrgGLTransfer transfer;
    bool transfer_pending, transfer_claimed;
    uint32_t transfer_error;
    uint64_t cpu_epoch, cpu_generation;
    bool reply_pending, reply_ready;
    uint64_t reply_epoch, reply_sequence, reply_token;
    uint8_t reply_packet[JGPU_PACKET_BYTES];
    int reply_fd;
    JrgContext contexts[JRG_GL_MAX_CONTEXTS];
    JrgDrawable drawables[JRG_GL_MAX_DRAWABLES];
    JrgSlot slots[JRG_GL_MAX_DRAWABLES * JRG_GL_EXPORT_SLOTS];
    JrgContext *current_context;
    JrgDrawable *current_drawable;
    JrgGLNotify notify;
    void *opaque;
};

static uint32_t word(const uint8_t *p, unsigned offset)
{
    return ldl_le_p(p + offset);
}

static bool desktop_rect(uint32_t x, uint32_t y, uint32_t width,
                          uint32_t height, uint32_t limit_w, uint32_t limit_h)
{
    return width && height && x <= limit_w && y <= limit_h &&
           width <= limit_w - x && height <= limit_h - y;
}

static uint32_t validate_desktop(const uint8_t *r, uint32_t primary_width,
                                  uint32_t primary_height, uint32_t vram_size,
                                  uint64_t *work)
{
    uint32_t op = word(r, JRG_DESKTOP_OP);
    uint32_t flags = word(r, JRG_DESKTOP_FLAGS);
    uint32_t x = word(r, JRG_DESKTOP_DST_X);
    uint32_t y = word(r, JRG_DESKTOP_DST_Y);
    uint32_t w = word(r, JRG_DESKTOP_WIDTH);
    uint32_t h = word(r, JRG_DESKTOP_HEIGHT);
    uint32_t sx = word(r, JRG_DESKTOP_SRC_X);
    uint32_t sy = word(r, JRG_DESKTOP_SRC_Y);
    uint32_t offset = word(r, JRG_DESKTOP_SLOT_OR_OFFSET);
    uint32_t stride = word(r, JRG_DESKTOP_VRAM_STRIDE);
    uint64_t epoch = ldq_le_p(r + JRG_DESKTOP_IMAGE_EPOCH);
    uint64_t frame = ldq_le_p(r + JRG_DESKTOP_IMAGE_FRAME);
    bool image = op == JRG_DESKTOP_BLIT || op == JRG_DESKTOP_DISCARD;
    bool cpu = op == JRG_DESKTOP_SEED || op == JRG_DESKTOP_PATCH ||
               op == JRG_DESKTOP_RETURN || op == JRG_DESKTOP_READBACK;

    if (op < JRG_DESKTOP_SEED || op > JRG_DESKTOP_DISCARD ||
        word(r, JRG_DESKTOP_RESERVED0) || word(r, JRG_DESKTOP_RESERVED1) ||
        (flags && (op != JRG_DESKTOP_BLIT ||
                   flags != JRG_DESKTOP_BLIT_FINAL))) {
        return JRG_GL_ERROR_BATCH;
    }
    if (op == JRG_DESKTOP_DISCARD) {
        if (x || y || w || h || sx || sy) {
            return JRG_GL_ERROR_BATCH;
        }
    } else if (!desktop_rect(x, y, w, h, primary_width, primary_height)) {
        return JRG_GL_ERROR_DESKTOP;
    }
    if (op == JRG_DESKTOP_SEED || op == JRG_DESKTOP_RETURN) {
        if (x || y || w != primary_width || h != primary_height) {
            return JRG_GL_ERROR_DESKTOP;
        }
    }
    if (op == JRG_DESKTOP_COPY &&
        !desktop_rect(sx, sy, w, h, primary_width, primary_height)) {
        return JRG_GL_ERROR_DESKTOP;
    }
    if (image) {
        if (offset >= JGPU_MAX_EXPORT_SLOTS || !epoch || !frame || stride) {
            return JRG_GL_ERROR_DESKTOP;
        }
    } else if (epoch || frame || (!cpu && (offset || stride))) {
        return JRG_GL_ERROR_BATCH;
    }
    if (op != JRG_DESKTOP_COPY && op != JRG_DESKTOP_BLIT &&
        (sy || (sx && op != JRG_DESKTOP_FILL))) {
        return JRG_GL_ERROR_BATCH;
    }
    if (cpu) {
        uint64_t end = (uint64_t)offset + (uint64_t)(h - 1) * stride + w * 4;
        uint64_t export_bytes = QEMU_ALIGN_UP(
            (uint64_t)QEMU_ALIGN_UP(w * 4, 256) * h, 65536);
        if ((offset & 3) || (stride & 3) || stride < w * 4 ||
            end > vram_size || export_bytes > JGPU_MAX_CPU_BYTES) {
            return JRG_GL_ERROR_DESKTOP;
        }
    }
    *work += (uint64_t)w * h * 4;
    return *work > JRG_DESKTOP_MAX_BYTES ? JRG_GL_ERROR_LIMIT : 0;
}

uint32_t jrg_gl_validate(const uint8_t *data, size_t bytes, uint32_t generation,
                         uint32_t primary_width, uint32_t primary_height,
                         uint32_t vram_size, uint32_t *records)
{
    size_t offset = 0;
    unsigned count = 0, desktop_count = 0;
    uint64_t desktop_work = 0;

    *records = 0;
    if (!bytes || bytes > JRG_GL_MAX_BYTES || (bytes & 3)) {
        return JRG_GL_ERROR_BATCH;
    }
    while (offset < bytes) {
        const uint8_t *r = data + offset;
        uint32_t op, size, flags, expected = JRG_GL_HEADER_BYTES;

        if (bytes - offset < JRG_GL_HEADER_BYTES ||
            ++count > JRG_GL_MAX_RECORDS) {
            return JRG_GL_ERROR_BATCH;
        }
        op = word(r, JRG_GL_OFF_OP);
        size = word(r, JRG_GL_OFF_SIZE);
        flags = word(r, JRG_GL_OFF_FLAGS);
        if (size < JRG_GL_HEADER_BYTES || size > bytes - offset || (size & 3) ||
            !word(r, JRG_GL_OFF_CLIENT) || word(r, JRG_GL_OFF_RESERVED) ||
            (flags && (op != JRG_GL_PRESENT ||
                       (flags & ~(JRG_GL_PRESENT_EXCLUSIVE |
                                  JRG_GL_PRESENT_RETAIN |
                                  JRG_GL_PRESENT_FRONT_ONLY |
                                  JRG_GL_PRESENT_NO_EXPORT |
                                  JRG_GL_PRESENT_BOUNDED)) ||
                       ((flags & JRG_GL_PRESENT_NO_EXPORT) &&
                        (flags & ~JRG_GL_PRESENT_BOUNDED) !=
                         JRG_GL_PRESENT_NO_EXPORT) ||
                       ((flags & JRG_GL_PRESENT_EXCLUSIVE) &&
                        (flags & JRG_GL_PRESENT_RETAIN))))) {
            return JRG_GL_ERROR_BATCH;
        }
        if (word(r, JRG_GL_OFF_GENERATION) != generation) {
            return JRG_GL_ERROR_GENERATION;
        }
        switch (op) {
        case JRG_GL_CREATE_CONTEXT:
            expected += 4;
            break;
        case JRG_GL_CREATE_DRAWABLE:
            expected += 8;
            break;
        case JRG_GL_CALL:
            if (size < expected + 4) {
                return JRG_GL_ERROR_BATCH;
            }
            uint32_t words = jrg_gl_function_words(word(r, expected));
            if (words == UINT32_MAX || (words & JRG_GL_FUNCTION_KIND_MASK)) {
                return JRG_GL_ERROR_UNSUPPORTED;
            }
            expected += 4 + words * 4;
            break;
        case JRG_GL_DATA_CALL: {
            if (size < JRG_GL_DATA_ARGS) {
                return JRG_GL_ERROR_BATCH;
            }
            uint32_t fn = word(r, JRG_GL_DATA_FUNCTION);
            uint32_t count = jrg_gl_function_words(fn);
            uint32_t bytes = word(r, JRG_GL_DATA_BYTES);
            if (count == UINT32_MAX ||
                (count & JRG_GL_FUNCTION_KIND_MASK) !=
                JRG_GL_FUNCTION_INLINE_DATA) {
                return JRG_GL_ERROR_UNSUPPORTED;
            }
            count &= ~JRG_GL_FUNCTION_INLINE_DATA;
            if (bytes > JRG_GL_MAX_BYTES ||
                size != JRG_GL_DATA_ARGS + count * 4 +
                        QEMU_ALIGN_UP(bytes, 4)) {
                return JRG_GL_ERROR_BATCH;
            }
            uint32_t error = jrg_gl_data_validate(fn, r + JRG_GL_DATA_ARGS,
                                                  r + JRG_GL_DATA_ARGS +
                                                  count * 4,
                                                  bytes);
            if (error) {
                return error;
            }
            for (unsigned i = bytes; i < QEMU_ALIGN_UP(bytes, 4); i++) {
                if (r[JRG_GL_DATA_ARGS + count * 4 + i]) {
                    return JRG_GL_ERROR_BATCH;
                }
            }
            expected = size;
            break;
        }
        case JRG_GL_QUERY:
            if (size != JRG_GL_QUERY_BYTES || bytes != JRG_GL_QUERY_BYTES) {
                return JRG_GL_ERROR_BATCH;
            }
            uint32_t query_error = jrg_gl_query_validate(word(r, 32), r + 36);
            if (query_error) {
                return query_error;
            }
            expected = JRG_GL_QUERY_BYTES;
            break;
        case JRG_GL_DESKTOP:
            expected += JRG_DESKTOP_BYTES;
            if (size != expected || ++desktop_count > JRG_DESKTOP_MAX_RECORDS) {
                return JRG_GL_ERROR_BATCH;
            }
            uint32_t error = validate_desktop(r + JRG_GL_HEADER_BYTES,
                                             primary_width, primary_height,
                                             vram_size, &desktop_work);
            if (error) {
                return error;
            }
            break;
        case JRG_GL_DESTROY_CONTEXT:
        case JRG_GL_DESTROY_DRAWABLE:
        case JRG_GL_MAKE_CURRENT:
        case JRG_GL_CLOSE_CLIENT:
            break;
        case JRG_GL_PRESENT:
            if (flags & JRG_GL_PRESENT_BOUNDED) {
                expected += 8;
                if (size != expected) {
                    return JRG_GL_ERROR_BATCH;
                }
                if (!word(r, 32) || !word(r, 36) ||
                    word(r, 32) > JRG_GL_MAX_DIMENSION ||
                    word(r, 36) > JRG_GL_MAX_DIMENSION) {
                    return JRG_GL_ERROR_DRAWABLE;
                }
            }
            break;
        default:
            return JRG_GL_ERROR_UNSUPPORTED;
        }
        if (size != expected) {
            return JRG_GL_ERROR_BATCH;
        }
        if (op == JRG_GL_CALL) {
            uint32_t error = jrg_gl_call_validate(word(r, 32), r + 36);
            if (error) {
                return error;
            }
        }
        if (op == JRG_GL_CREATE_DRAWABLE &&
            (!word(r, 32) || !word(r, 36) ||
             word(r, 32) > JRG_GL_MAX_DIMENSION ||
             word(r, 36) > JRG_GL_MAX_DIMENSION)) {
            return JRG_GL_ERROR_DRAWABLE;
        }
        if ((flags & JRG_GL_PRESENT_EXCLUSIVE) &&
            (!primary_width || !primary_height)) {
            return JRG_GL_ERROR_DRAWABLE;
        }
        offset += size;
    }
    *records = count;
    return 0;
}

/* Ancillary handles belong to the first byte; never read across records. */
static bool receive_record_fd(int fd, uint8_t *data, int *received_fd)
{
    size_t n = 0;
    bool valid = true;

    *received_fd = -1;
    while (n < JGPU_PACKET_BYTES) {
        union {
            struct cmsghdr align;
            char bytes[CMSG_SPACE(4 * sizeof(int))];
        } control;
        struct iovec iov = { data + n, JGPU_PACKET_BYTES - n };
        struct msghdr msg = {
            .msg_iov = &iov, .msg_iovlen = 1,
            .msg_control = control.bytes, .msg_controllen = sizeof(control),
        };
        ssize_t got = recvmsg(fd, &msg, 0);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            valid = false;
            break;
        }
        for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c;
             c = CMSG_NXTHDR(&msg, c)) {
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
    valid &= n == JGPU_PACKET_BYTES &&
             word(data, JGPU_OFF_MAGIC) == JGPU_MAGIC &&
             word(data, JGPU_OFF_VERSION) == JGPU_VERSION &&
             word(data, JGPU_OFF_SIZE) == JGPU_PACKET_BYTES;
    if (!valid && *received_fd >= 0) {
        close(*received_fd);
        *received_fd = -1;
    }
    return valid;
}

static bool receive_record(int fd, uint8_t *data)
{
    int received_fd;
    bool valid = receive_record_fd(fd, data, &received_fd);

    if (received_fd >= 0) {
        close(received_fd);
        return false;
    }
    return valid;
}

static bool send_record(JrgGLEngine *e, const uint8_t *packet,
                         const int *fds, unsigned count)
{
    union {
        struct cmsghdr align;
        char bytes[CMSG_SPACE(2 * sizeof(int))];
    } control = { 0 };
    struct iovec iov = { (void *)packet, JGPU_PACKET_BYTES };
    struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1 };
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
    while (ok && sent < JGPU_PACKET_BYTES) {
        ssize_t n = send(e->socket, packet + sent, JGPU_PACKET_BYTES - sent,
                          MSG_NOSIGNAL);
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

static void packet_init(uint8_t *packet, uint32_t kind)
{
    memset(packet, 0, JGPU_PACKET_BYTES);
    stl_le_p(packet + JGPU_OFF_MAGIC, JGPU_MAGIC);
    stl_le_p(packet + JGPU_OFF_VERSION, JGPU_VERSION);
    stl_le_p(packet + JGPU_OFF_KIND, kind);
    stl_le_p(packet + JGPU_OFF_SIZE, JGPU_PACKET_BYTES);
}

static bool connect_consumer(JrgGLEngine *e, Error **errp)
{
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    struct timeval timeout = { .tv_sec = 2 };
    uint8_t hello[JGPU_PACKET_BYTES];

    if (!e->socket_path || strlen(e->socket_path) >= sizeof(addr.sun_path)) {
        error_setg(errp, "A valid gpu-socket path is required for GL export");
        return false;
    }
    pstrcpy(addr.sun_path, sizeof(addr.sun_path), e->socket_path);
    qemu_mutex_lock(&e->lock);
    e->socket = qemu_socket(AF_UNIX, SOCK_STREAM, 0);
    qemu_mutex_unlock(&e->lock);
    if (e->socket < 0 ||
        connect(e->socket, (struct sockaddr *)&addr, sizeof(addr))) {
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
        word(hello, JGPU_OFF_KIND) != JGPU_KIND_HELLO) {
        error_setg(errp, "Invalid GPU consumer handshake");
        return false;
    }
    timeout.tv_sec = 0;
    setsockopt(e->socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#ifdef CONFIG_DARWIN
    const char *service = (const char *)hello + JGPU_HELLO_MACH_SERVICE;
    if (word(hello, JGPU_HELLO_PLATFORM) != JGPU_PLATFORM_MACOS ||
        !memchr(service, 0, JGPU_HELLO_MACH_SERVICE_LEN) ||
        bootstrap_look_up(bootstrap_port, service, &e->remote) !=
        KERN_SUCCESS) {
        error_setg(errp, "Looking up GPU consumer Mach service failed");
        return false;
    }
    e->platform = jrg_gl_platform_new(NULL, errp);
#else
    const char *node = (const char *)hello + JGPU_HELLO_RENDER_NODE;
    if (word(hello, JGPU_HELLO_PLATFORM) != JGPU_PLATFORM_LINUX ||
        !memchr(node, 0, JGPU_HELLO_RENDER_NODE_LEN) || !node[0]) {
        error_setg(errp, "GPU consumer must identify its DRM render node");
        return false;
    }
    memcpy(e->uuid, hello + JGPU_OFF_DEVICE_UUID, sizeof(e->uuid));
    e->platform = jrg_gl_platform_new(node, errp);
#endif
    return e->platform != NULL;
}

static void *release_worker(void *opaque)
{
    JrgGLEngine *e = opaque;
    uint8_t packet[JGPU_PACKET_BYTES];
    int received_fd;

    while (receive_record_fd(e->socket, packet, &received_fd)) {
        uint32_t kind = word(packet, JGPU_OFF_KIND);
        uint32_t slot = word(packet, JGPU_OFF_SLOT);
        uint64_t epoch = ldq_le_p(packet + JGPU_OFF_EPOCH);
        uint64_t sequence = ldq_le_p(packet + JGPU_OFF_GENERATION);
        bool valid = true;

        qemu_mutex_lock(&e->lock);
        if (kind == JGPU_KIND_DESKTOP_REPLY) {
            uint64_t token = ldq_le_p(packet + JGPU_DESKTOP_OFF_TOKEN);
            if (!e->reply_pending && epoch == e->reply_epoch &&
                sequence == e->reply_sequence && token == e->reply_token) {
                /* A reset can cancel a readback already executing remotely. */
            } else if (!e->reply_pending || e->reply_ready ||
                       epoch != e->reply_epoch ||
                       sequence != e->reply_sequence ||
                       token != e->reply_token) {
                valid = false;
            } else {
                memcpy(e->reply_packet, packet, sizeof(packet));
                e->reply_fd = received_fd;
                received_fd = -1;
                e->reply_ready = true;
            }
        } else if (kind != JGPU_KIND_RELEASE || received_fd >= 0) {
            valid = false;
        } else if (slot >= JGPU_CPU_SLOT_BASE &&
                   slot < JGPU_CPU_SLOT_BASE + JGPU_MAX_CPU_SLOTS) {
            JrgCpuSlot *s = &e->cpu_slots[slot - JGPU_CPU_SLOT_BASE];
            if (s->published && s->epoch == epoch && s->sequence == sequence) {
                s->published = false;
            }
        } else if (slot < G_N_ELEMENTS(e->slots)) {
            JrgSlot *s = &e->slots[slot];
            if (s->state == SLOT_PUBLISHED &&
                epoch == ldq_le_p(s->packet + JGPU_OFF_EPOCH) &&
                sequence == s->generation) {
                if (s->sending) {
                    s->release_pending = true;
                } else {
                    s->state = SLOT_FREE;
                }
            }
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

static bool send_frame(JrgGLEngine *e, JrgSlot *s)
{
#ifdef CONFIG_DARWIN
    struct {
        mach_msg_header_t header;
        mach_msg_body_t body;
        mach_msg_port_descriptor_t surface;
        uint8_t packet[JGPU_PACKET_BYTES];
    } msg = { 0 };
    mach_port_t port = jrg_gl_image_port(s->image);
    kern_return_t ret;

    if (!MACH_PORT_VALID(port)) {
        return false;
    }
    msg.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) |
                           MACH_MSGH_BITS_COMPLEX;
    msg.header.msgh_size = sizeof(msg);
    msg.header.msgh_remote_port = e->remote;
    msg.header.msgh_id = JGPU_MAGIC;
    msg.body.msgh_descriptor_count = 1;
    msg.surface.name = port;
    msg.surface.disposition = MACH_MSG_TYPE_COPY_SEND;
    msg.surface.type = MACH_MSG_PORT_DESCRIPTOR;
    memcpy(msg.packet, s->packet, sizeof(msg.packet));
    ret = mach_msg(&msg.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT,
                    sizeof(msg), 0, MACH_PORT_NULL, 2000, MACH_PORT_NULL);
    mach_port_deallocate(mach_task_self(), port);
    return ret == KERN_SUCCESS;
#else
    int fds[2] = { jrg_gl_image_fd(s->image),
                   jrg_gl_image_fence_fd(s->image) };
    return send_record(e, s->packet, fds, fds[1] >= 0 ? 2 : 1);
#endif
}

static void *completion_worker(void *opaque)
{
    JrgGLEngine *e = opaque;

    qemu_mutex_lock(&e->lock);
    for (;;) {
        JrgSlot *s;
        Error *err = NULL;

        while (g_queue_is_empty(&e->pending_output) && !e->stopping) {
            qemu_cond_wait(&e->cond, &e->lock);
        }
        if (g_queue_is_empty(&e->pending_output)) {
            break;
        }
        JrgOutput *output = g_queue_pop_head(&e->pending_output);
        s = output->slot;
        if (!s) {
            bool publish = !e->failed && !e->stopping && !e->reset;
            qemu_mutex_unlock(&e->lock);
            bool sent = publish && send_record(e, output->packet, NULL, 0);
            qemu_mutex_lock(&e->lock);
            if (publish && !sent) {
                e->failed = true;
            }
            g_free(output);
            e->pending_desktop--;
            e->pending_count--;
            qemu_cond_broadcast(&e->cond);
            continue;
        }
        qemu_mutex_unlock(&e->lock);
        bool ready = jrg_gl_image_ready(s->image, &err);
        qemu_mutex_lock(&e->lock);
        /* A fast consumer may release during send_frame; publish first. */
        s->state = SLOT_PUBLISHED;
        s->sending = true;
        bool publish = ready && !e->failed && !e->stopping && !e->reset;
        qemu_mutex_unlock(&e->lock);
        bool sent = publish && send_frame(e, s);
        if (err) {
            error_report_err(err);
        }
        qemu_mutex_lock(&e->lock);
        s->sending = false;
        if (sent && s->release_pending) {
            s->state = SLOT_FREE;
        }
        if (!sent) {
            s->state = SLOT_FREE;
            if (publish || !ready) {
                e->failed = true;
            }
        }
        if (sent) {
            unsigned index = (s - e->slots) / JRG_GL_EXPORT_SLOTS;
            JrgDrawable *d = &e->drawables[index];
            d->published_epoch = ldq_le_p(s->packet + JGPU_OFF_EPOCH);
            d->published_generation = s->generation;
        }
        g_free(output);
        e->pending_count--;
        qemu_cond_broadcast(&e->cond);
    }
    qemu_mutex_unlock(&e->lock);
    return NULL;
}

static JrgContext *find_context(JrgGLEngine *e, uint32_t client, uint32_t id)
{
    for (unsigned i = 0; i < G_N_ELEMENTS(e->contexts); i++) {
        if (e->contexts[i].native && e->contexts[i].client == client &&
            e->contexts[i].id == id) {
            return &e->contexts[i];
        }
    }
    return NULL;
}

static JrgDrawable *find_drawable(JrgGLEngine *e, uint32_t client, uint32_t id)
{
    for (unsigned i = 0; i < G_N_ELEMENTS(e->drawables); i++) {
        if (e->drawables[i].native && e->drawables[i].client == client &&
            e->drawables[i].id == id) {
            return &e->drawables[i];
        }
    }
    return NULL;
}

static bool make_current(JrgGLEngine *e, JrgContext *c, JrgDrawable *d,
                          Error **errp)
{
    if (e->current_context == c && e->current_drawable == d) {
        return true;
    }
    if (e->current_context && e->current_drawable &&
        !jrg_gl_context_in_begin(e->current_context->native)) {
        jrg_gl_flush_drawable(e->current_drawable->native);
    }
    if (!jrg_gl_make_current(c->native, d->native, errp)) {
        return false;
    }
    e->current_context = c;
    e->current_drawable = d;
    return true;
}

static void drain_output(JrgGLEngine *e)
{
    qemu_mutex_lock(&e->lock);
    while (e->pending_count) {
        qemu_cond_wait(&e->cond, &e->lock);
    }
    qemu_mutex_unlock(&e->lock);
}

/* Count queued and in-flight packets so a stalled consumer cannot grow the
 * queue without bound. Blocking here applies only at the fixed backlog limit. */
static uint32_t queue_desktop(JrgGLEngine *e, const uint8_t *packet)
{
    JrgOutput *output = g_new0(JrgOutput, 1);
    uint32_t error = 0;

    memcpy(output->packet, packet, JGPU_PACKET_BYTES);
    qemu_mutex_lock(&e->lock);
    while (e->pending_desktop == JRG_MAX_PENDING_DESKTOP &&
           !e->stopping && !e->reset && !e->failed) {
        qemu_cond_wait(&e->cond, &e->lock);
    }
    if (e->reset) {
        error = JRG_GL_ERROR_GENERATION;
    } else if (e->stopping || e->failed) {
        error = JRG_GL_ERROR_TRANSPORT;
    } else {
        g_queue_push_tail(&e->pending_output, output);
        e->pending_count++;
        e->pending_desktop++;
        qemu_cond_broadcast(&e->cond);
    }
    qemu_mutex_unlock(&e->lock);
    if (error) {
        g_free(output);
    }
    return error;
}

static void drop_drawable_resource(JrgGLEngine *e, JrgDrawable *d)
{
    uint8_t packet[JGPU_PACKET_BYTES];

    if (!d->published_generation) {
        return;
    }
    packet_init(packet, JGPU_KIND_RESOURCE_DROP);
    stl_le_p(packet + JGPU_OFF_CLIENT, d->client);
    stl_le_p(packet + JGPU_OFF_DRAWABLE, d->id);
    stq_le_p(packet + JGPU_OFF_EPOCH, d->published_epoch);
    stq_le_p(packet + JGPU_OFF_GENERATION, d->published_generation);
    qemu_mutex_lock(&e->lock);
    bool publish = !e->stopping && !e->failed;
    qemu_mutex_unlock(&e->lock);
    if (publish && !send_record(e, packet, NULL, 0)) {
        qemu_mutex_lock(&e->lock);
        e->failed = true;
        qemu_mutex_unlock(&e->lock);
    }
}

static void delete_drawable(JrgGLEngine *e, JrgDrawable *d)
{
    unsigned index = d - e->drawables;

    drain_output(e);
    if (!e->reset) {
        drop_drawable_resource(e, d);
    }
    e->current_context = NULL;
    e->current_drawable = NULL;
    for (unsigned i = 0; i < JRG_GL_EXPORT_SLOTS; i++) {
        JrgSlot *s = &e->slots[index * JRG_GL_EXPORT_SLOTS + i];
        qemu_mutex_lock(&e->lock);
        while (s->state == SLOT_PUBLISHED && !e->stopping && !e->failed) {
            qemu_cond_wait(&e->cond, &e->lock);
        }
        qemu_mutex_unlock(&e->lock);
        /* Exported handles retain backing storage independently of this VM. */
        jrg_gl_image_free(e->platform, s->image);
        qemu_mutex_lock(&e->lock);
        s->image = NULL;
        s->state = SLOT_FREE;
        qemu_mutex_unlock(&e->lock);
    }
    jrg_gl_drawable_free(e->platform, d->native);
    memset(d, 0, sizeof(*d));
}

static void close_client(JrgGLEngine *e, uint32_t client)
{
    drain_output(e);
    e->current_context = NULL;
    e->current_drawable = NULL;
    for (unsigned i = 0; i < G_N_ELEMENTS(e->drawables); i++) {
        if (e->drawables[i].native && (!client ||
                                     e->drawables[i].client == client)) {
            delete_drawable(e, &e->drawables[i]);
        }
    }
    for (unsigned i = 0; i < G_N_ELEMENTS(e->contexts); i++) {
        if (e->contexts[i].native && (!client ||
                                    e->contexts[i].client == client)) {
            jrg_gl_context_free(e->contexts[i].native);
            memset(&e->contexts[i], 0, sizeof(e->contexts[i]));
        }
    }
}

static uint32_t present(JrgGLEngine *e, JrgContext *c, JrgDrawable *d,
                         uint32_t flags, Error **errp)
{
    unsigned first = (d - e->drawables) * JRG_GL_EXPORT_SLOTS;
    JrgSlot *s = NULL;
    uint32_t stride, offset;
    uint64_t modifier;

    if (flags & JRG_GL_PRESENT_NO_EXPORT) {
        jrg_gl_exchange(c->native, d->native);
        return 0;
    }
    qemu_mutex_lock(&e->lock);
    while (!e->stopping && !e->reset && !e->failed) {
        for (unsigned i = 0; i < JRG_GL_EXPORT_SLOTS; i++) {
            if (e->slots[first + i].state == SLOT_FREE) {
                s = &e->slots[first + i];
                s->state = SLOT_RENDERING;
                s->release_pending = false;
                break;
            }
        }
        if (s) {
            break;
        }
        qemu_cond_wait(&e->cond, &e->lock);
    }
    qemu_mutex_unlock(&e->lock);
    if (!s) {
        return JRG_GL_ERROR_TRANSPORT;
    }
    if (!s->image) {
        s->image = jrg_gl_image_new(e->platform, d->width, d->height, errp);
    }
    if (!s->image || !jrg_gl_export(c->native, d->native, s->image,
                                    !(flags & JRG_GL_PRESENT_FRONT_ONLY),
                                    errp)) {
        qemu_mutex_lock(&e->lock);
        s->state = SLOT_FREE;
        qemu_mutex_unlock(&e->lock);
        return JRG_GL_ERROR_HOST;
    }
    jrg_gl_image_metadata(s->image, &stride, &offset, &modifier);
    if (flags & JRG_GL_PRESENT_EXCLUSIVE) {
        e->exclusive = true;
    }
    packet_init(s->packet, flags & JRG_GL_PRESENT_EXCLUSIVE ?
                 JGPU_KIND_FRAME : JGPU_KIND_DRAWABLE);
    uint32_t wire_flags = JGPU_FLAG_TOP_LEFT;
    if (flags & JRG_GL_PRESENT_RETAIN) {
        wire_flags |= JGPU_FLAG_RETAIN_FOR_DESKTOP;
    }
    if (jrg_gl_image_fence_fd(s->image) >= 0) {
        wire_flags |= JGPU_FLAG_READY_FENCE;
    }
    stl_le_p(s->packet + JGPU_OFF_FLAGS, wire_flags);
    stl_le_p(s->packet + JGPU_OFF_WIDTH, d->width);
    stl_le_p(s->packet + JGPU_OFF_HEIGHT, d->height);
    stl_le_p(s->packet + JGPU_OFF_STRIDE, stride);
    stl_le_p(s->packet + JGPU_OFF_OFFSET, offset);
    stl_le_p(s->packet + JGPU_OFF_FOURCC, JGPU_FORMAT_ARGB8888);
    stl_le_p(s->packet + JGPU_OFF_SLOT, s - e->slots);
    stl_le_p(s->packet + JGPU_OFF_CLIENT, d->client);
    stl_le_p(s->packet + JGPU_OFF_DRAWABLE, d->id);
    stq_le_p(s->packet + JGPU_OFF_EPOCH, d->epoch);
    s->generation = ++d->generation;
    stq_le_p(s->packet + JGPU_OFF_GENERATION, s->generation);
    stq_le_p(s->packet + JGPU_OFF_MODIFIER, modifier);
    memcpy(s->packet + JGPU_OFF_DEVICE_UUID, e->uuid, sizeof(e->uuid));
    qemu_mutex_lock(&e->lock);
    e->last_present = (JrgGLFrameRef) {
        .slot = s - e->slots, .client = d->client, .drawable = d->id,
        .epoch = d->epoch, .generation = s->generation,
    };
    s->state = SLOT_PENDING;
    JrgOutput *output = g_new0(JrgOutput, 1);
    output->slot = s;
    g_queue_push_tail(&e->pending_output, output);
    e->pending_count++;
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
    return 0;
}

static uint32_t transfer_pixels(JrgGLEngine *e, const uint8_t *r,
                                 uint8_t *pixels, uint32_t stride,
                                 const JrgBatch *batch, bool writeback,
                                 bool return_cpu)
{
    qemu_mutex_lock(&e->lock);
    e->transfer = (JrgGLTransfer) {
        .pixels = pixels, .stride = stride,
        .width = word(r, JRG_DESKTOP_WIDTH),
        .height = word(r, JRG_DESKTOP_HEIGHT),
        .offset = word(r, JRG_DESKTOP_SLOT_OR_OFFSET),
        .vram_stride = word(r, JRG_DESKTOP_VRAM_STRIDE),
        .generation = batch->generation,
        .writeback = writeback, .return_cpu = return_cpu,
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
    uint32_t error = e->stopping ? JRG_GL_ERROR_GENERATION : e->transfer_error;
    qemu_mutex_unlock(&e->lock);
    return error;
}

static JrgCpuSlot *cpu_slot(JrgGLEngine *e, size_t bytes, Error **errp)
{
    JrgCpuSlot *slot = NULL;

    qemu_mutex_lock(&e->lock);
    while (!e->stopping && !e->reset && !e->failed) {
        for (unsigned i = 0; i < G_N_ELEMENTS(e->cpu_slots); i++) {
            if (!e->cpu_slots[i].published) {
                slot = &e->cpu_slots[i];
                slot->published = true;
                slot->epoch = e->desktop_epoch;
                slot->sequence = e->desktop_sequence;
                break;
            }
        }
        if (slot) {
            break;
        }
        qemu_cond_wait(&e->cond, &e->lock);
    }
    qemu_mutex_unlock(&e->lock);
    if (!slot) {
        return NULL;
    }
    if (!slot->pixels || slot->bytes != bytes) {
        if (slot->pixels) {
            qemu_memfd_free(slot->pixels, slot->bytes, slot->fd);
        }
        slot->bytes = bytes;
        slot->fd = -1;
        slot->pixels = qemu_memfd_alloc("juke-desktop", bytes, 0, &slot->fd,
                                        errp);
        if (slot->pixels) {
            memset(slot->pixels, 0, bytes);
        }
    }
    if (!slot->pixels) {
        qemu_mutex_lock(&e->lock);
        slot->published = false;
        qemu_mutex_unlock(&e->lock);
        return NULL;
    }
    return slot;
}

static uint32_t capture_desktop(JrgGLEngine *e, const uint8_t *r,
                                 const JrgBatch *batch, uint32_t subtype,
                                 Error **errp)
{
    uint32_t w = word(r, JRG_DESKTOP_WIDTH);
    uint32_t h = word(r, JRG_DESKTOP_HEIGHT);
    uint32_t stride = QEMU_ALIGN_UP(w * 4, 256);
    size_t bytes = QEMU_ALIGN_UP((size_t)stride * h, 65536);
    uint8_t packet[JGPU_PACKET_BYTES];
    JrgCpuSlot *s;
    uint32_t error;

    if (bytes > JGPU_MAX_CPU_BYTES) {
        return JRG_GL_ERROR_LIMIT;
    }
    s = cpu_slot(e, bytes, errp);
    if (!s) {
        return JRG_GL_ERROR_TRANSPORT;
    }
    if (s->width != w || s->height != h || s->stride != stride) {
        memset(s->pixels, 0, bytes);
        s->width = w;
        s->height = h;
        s->stride = stride;
    }
    error = transfer_pixels(e, r, s->pixels, stride, batch, false,
                              subtype == JGPU_CPU_RETURN);
    if (!error) {
        packet_init(packet, JGPU_KIND_CPU_DESKTOP);
        stl_le_p(packet + JGPU_CPU_OFF_SUBTYPE, subtype);
        stl_le_p(packet + JGPU_OFF_WIDTH, w);
        stl_le_p(packet + JGPU_OFF_HEIGHT, h);
        stl_le_p(packet + JGPU_OFF_STRIDE, stride);
        stl_le_p(packet + JGPU_OFF_FOURCC, JGPU_FORMAT_ARGB8888);
        stl_le_p(packet + JGPU_OFF_SLOT,
                 JGPU_CPU_SLOT_BASE + (s - e->cpu_slots));
        stq_le_p(packet + JGPU_OFF_EPOCH, s->epoch);
        stq_le_p(packet + JGPU_OFF_GENERATION, s->sequence);
        stq_le_p(packet + JGPU_CPU_OFF_ALLOCATION, bytes);
        stl_le_p(packet + JGPU_CPU_OFF_DST_X, word(r, JRG_DESKTOP_DST_X));
        stl_le_p(packet + JGPU_CPU_OFF_DST_Y, word(r, JRG_DESKTOP_DST_Y));
        if (subtype == JGPU_CPU_RETURN) {
            stq_le_p(packet + JGPU_CPU_OFF_LEGACY_EPOCH, e->cpu_epoch);
            stq_le_p(packet + JGPU_CPU_OFF_LEGACY_FRAME, e->cpu_generation);
        }
        if (!send_record(e, packet, &s->fd, 1)) {
            error = JRG_GL_ERROR_TRANSPORT;
        }
    }
    if (error) {
        qemu_mutex_lock(&e->lock);
        s->published = false;
        qemu_mutex_unlock(&e->lock);
    }
    return error;
}

static uint32_t readback_desktop(JrgGLEngine *e, const uint8_t *r,
                                  const JrgBatch *batch, uint8_t *packet)
{
    uint32_t error = 0;
    uint32_t w = word(r, JRG_DESKTOP_WIDTH);
    uint32_t h = word(r, JRG_DESKTOP_HEIGHT);
    uint64_t token = e->desktop_sequence;
    int fd = -1;
    struct stat statbuf;
    uint8_t *pixels = MAP_FAILED;
    uint64_t bytes = 0;
    uint32_t stride = 0;

    stq_le_p(packet + JGPU_DESKTOP_OFF_TOKEN, token);
    qemu_mutex_lock(&e->lock);
    e->reply_pending = true;
    e->reply_ready = false;
    e->reply_epoch = e->desktop_epoch;
    e->reply_sequence = e->desktop_sequence;
    e->reply_token = token;
    qemu_mutex_unlock(&e->lock);
    if (!send_record(e, packet, NULL, 0)) {
        error = JRG_GL_ERROR_TRANSPORT;
    }
    qemu_mutex_lock(&e->lock);
    int64_t deadline = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;
    while (!error && !e->reply_ready && !e->stopping && !e->reset &&
           !e->failed) {
        int64_t remaining = deadline - g_get_monotonic_time();
        if (remaining <= 0) {
            break;
        }
        qemu_cond_timedwait(&e->cond, &e->lock,
                            DIV_ROUND_UP(remaining, G_TIME_SPAN_MILLISECOND));
    }
    if (e->stopping || e->reset) {
        error = JRG_GL_ERROR_GENERATION;
    } else if (!error && (!e->reply_ready || e->failed)) {
        error = JRG_GL_ERROR_TRANSPORT;
    }
    fd = e->reply_fd;
    e->reply_fd = -1;
    if (!error) {
        bytes = ldq_le_p(e->reply_packet + JGPU_CPU_OFF_ALLOCATION);
        stride = word(e->reply_packet, JGPU_OFF_STRIDE);
        if (word(e->reply_packet, 16) || fd < 0 ||
            word(e->reply_packet, JGPU_OFF_WIDTH) != w ||
            word(e->reply_packet, JGPU_OFF_HEIGHT) != h ||
            stride != w * 4 || bytes != (uint64_t)stride * h ||
            bytes > JGPU_MAX_CPU_BYTES || fstat(fd, &statbuf) ||
            statbuf.st_size < 0 || (uint64_t)statbuf.st_size < bytes) {
            error = JRG_GL_ERROR_DESKTOP;
        }
    }
    e->reply_pending = false;
    e->reply_ready = false;
    qemu_mutex_unlock(&e->lock);
    if (!error) {
        pixels = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (pixels == MAP_FAILED) {
            error = JRG_GL_ERROR_HOST;
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

static uint32_t execute_desktop(JrgGLEngine *e, const uint8_t *record,
                                  const JrgBatch *batch, Error **errp)
{
    const uint8_t *r = record + JRG_GL_HEADER_BYTES;
    uint32_t op = word(r, JRG_DESKTOP_OP);
    uint32_t client = word(record, JRG_GL_OFF_CLIENT);
    uint32_t drawable = word(record, JRG_GL_OFF_DRAWABLE);
    uint32_t w = word(r, JRG_DESKTOP_WIDTH);
    uint32_t h = word(r, JRG_DESKTOP_HEIGHT);
    uint32_t wire_op = 0, error = 0;
    uint8_t packet[JGPU_PACKET_BYTES];

    if (op == JRG_DESKTOP_SEED) {
        if (e->desktop_active) {
            return JRG_GL_ERROR_DESKTOP;
        }
        ++e->desktop_epoch;
        e->desktop_sequence = 0;
        e->desktop_width = batch->primary_width;
        e->desktop_height = batch->primary_height;
    } else if (!e->desktop_active ||
               batch->primary_width != e->desktop_width ||
               batch->primary_height != e->desktop_height) {
        return JRG_GL_ERROR_DESKTOP;
    }
    ++e->desktop_sequence;
    switch (op) {
    case JRG_DESKTOP_SEED:
    case JRG_DESKTOP_PATCH:
    case JRG_DESKTOP_RETURN:
        if (op == JRG_DESKTOP_RETURN && !e->desktop_coherent) {
            --e->desktop_sequence;
            return JRG_GL_ERROR_DESKTOP;
        }
        /* CPU patches and returns must follow every queued GPU operation. */
        drain_output(e);
        error = capture_desktop(e, r, batch,
                                  op == JRG_DESKTOP_SEED ? JGPU_CPU_SEED :
                                  op == JRG_DESKTOP_PATCH ? JGPU_CPU_PATCH :
                                  JGPU_CPU_RETURN, errp);
        if (!error && op == JRG_DESKTOP_SEED) {
            e->desktop_active = e->desktop_coherent = true;
        }
        if (!error && op == JRG_DESKTOP_RETURN) {
            e->exclusive = false;
            e->desktop_active = e->desktop_coherent = false;
        }
        if (error) {
            --e->desktop_sequence;
        }
        return error;
    case JRG_DESKTOP_FILL:
        wire_op = JGPU_DESKTOP_FILL;
        break;
    case JRG_DESKTOP_COPY:
        wire_op = JGPU_DESKTOP_COPY;
        break;
    case JRG_DESKTOP_READBACK:
        wire_op = JGPU_DESKTOP_READBACK;
        break;
    case JRG_DESKTOP_BLIT:
    case JRG_DESKTOP_DISCARD: {
        uint32_t index = word(r, JRG_DESKTOP_SLOT_OR_OFFSET);
        uint64_t epoch = ldq_le_p(r + JRG_DESKTOP_IMAGE_EPOCH);
        uint64_t frame = ldq_le_p(r + JRG_DESKTOP_IMAGE_FRAME);
        JrgSlot *s = &e->slots[index];
        qemu_mutex_lock(&e->lock);
        /* The matching frame is already ahead of this command in the output
         * FIFO. Retained storage stays alive until the desktop consumer's
         * final blit/discard; no GPU-fence wait belongs on this thread. */
        bool valid = (s->state == SLOT_PENDING || s->state == SLOT_PUBLISHED) &&
            word(s->packet, JGPU_OFF_CLIENT) == client &&
            word(s->packet, JGPU_OFF_DRAWABLE) == drawable &&
            ldq_le_p(s->packet + JGPU_OFF_EPOCH) == epoch &&
            s->generation == frame &&
            (word(s->packet, JGPU_OFF_FLAGS) & JGPU_FLAG_RETAIN_FOR_DESKTOP);
        if (op == JRG_DESKTOP_BLIT) {
            valid &= desktop_rect(word(r, JRG_DESKTOP_SRC_X),
                                    word(r, JRG_DESKTOP_SRC_Y), w, h,
                                    word(s->packet, JGPU_OFF_WIDTH),
                                    word(s->packet, JGPU_OFF_HEIGHT));
        }
        qemu_mutex_unlock(&e->lock);
        if (!valid) {
            --e->desktop_sequence;
            return JRG_GL_ERROR_DESKTOP;
        }
        wire_op = op == JRG_DESKTOP_BLIT ? JGPU_DESKTOP_BLIT :
                                          JGPU_DESKTOP_DISCARD;
        break;
    }
    }
    packet_init(packet, JGPU_KIND_DESKTOP_OP);
    stl_le_p(packet + JGPU_DESKTOP_OFF_OPCODE, wire_op);
    stl_le_p(packet + JGPU_DESKTOP_OFF_DST_X, word(r, JRG_DESKTOP_DST_X));
    stl_le_p(packet + JGPU_DESKTOP_OFF_DST_Y, word(r, JRG_DESKTOP_DST_Y));
    stl_le_p(packet + JGPU_DESKTOP_OFF_WIDTH, w);
    stl_le_p(packet + JGPU_DESKTOP_OFF_HEIGHT, h);
    stq_le_p(packet + JGPU_OFF_EPOCH, e->desktop_epoch);
    stq_le_p(packet + JGPU_OFF_GENERATION, e->desktop_sequence);
    if (op == JRG_DESKTOP_FILL) {
        stl_le_p(packet + JGPU_DESKTOP_OFF_COLOR, word(r, JRG_DESKTOP_SRC_X));
    } else if (op == JRG_DESKTOP_COPY || op == JRG_DESKTOP_BLIT ||
               op == JRG_DESKTOP_DISCARD) {
        stl_le_p(packet + JGPU_DESKTOP_OFF_SRC_X, word(r, JRG_DESKTOP_SRC_X));
        stl_le_p(packet + JGPU_DESKTOP_OFF_SRC_Y, word(r, JRG_DESKTOP_SRC_Y));
    }
    if (op == JRG_DESKTOP_BLIT || op == JRG_DESKTOP_DISCARD) {
        stl_le_p(packet + JGPU_OFF_CLIENT, client);
        stl_le_p(packet + JGPU_OFF_DRAWABLE, drawable);
        stl_le_p(packet + JGPU_DESKTOP_OFF_GPU_SLOT,
                 word(r, JRG_DESKTOP_SLOT_OR_OFFSET));
        stq_le_p(packet + JGPU_DESKTOP_OFF_GPU_EPOCH,
                 ldq_le_p(r + JRG_DESKTOP_IMAGE_EPOCH));
        stq_le_p(packet + JGPU_DESKTOP_OFF_GPU_FRAME,
                 ldq_le_p(r + JRG_DESKTOP_IMAGE_FRAME));
        stl_le_p(packet + JGPU_DESKTOP_OFF_FLAGS, word(r, JRG_DESKTOP_FLAGS));
    }
    if (op == JRG_DESKTOP_READBACK) {
        drain_output(e);
        error = readback_desktop(e, r, batch, packet);
        if (!error && w == e->desktop_width && h == e->desktop_height) {
            e->desktop_coherent = true;
        }
    } else {
        error = queue_desktop(e, packet);
        if (op != JRG_DESKTOP_DISCARD) {
            e->desktop_coherent = false;
        }
    }
    if (error && error != JRG_GL_ERROR_GENERATION) {
        qemu_mutex_lock(&e->lock);
        e->failed = true;
        qemu_cond_broadcast(&e->cond);
        qemu_mutex_unlock(&e->lock);
    }
    return error;
}

bool jrg_gl_engine_transfer(JrgGLEngine *e, JrgGLTransfer *transfer)
{
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

void jrg_gl_engine_transfer_done(JrgGLEngine *e, uint32_t error,
                                 uint64_t cpu_epoch, uint64_t cpu_generation)
{
    qemu_mutex_lock(&e->lock);
    e->transfer_error = error;
    e->cpu_epoch = cpu_epoch;
    e->cpu_generation = cpu_generation;
    e->transfer_pending = false;
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
}



static uint32_t execute_record(JrgGLEngine *e, const uint8_t *r,
                                JrgBatch *batch, Error **errp)
{
    uint32_t op = word(r, JRG_GL_OFF_OP);
    uint32_t client = word(r, JRG_GL_OFF_CLIENT);
    uint32_t context = word(r, JRG_GL_OFF_CONTEXT);
    uint32_t drawable = word(r, JRG_GL_OFF_DRAWABLE);
    JrgContext *c = find_context(e, client, context);
    JrgDrawable *d = find_drawable(e, client, drawable);

    switch (op) {
    case JRG_GL_DESKTOP:
        return execute_desktop(e, r, batch, errp);
    case JRG_GL_CREATE_CONTEXT: {
        JrgContext *share = find_context(e, client, word(r, 32));
        if (!context || c || (word(r, 32) && !share)) {
            return JRG_GL_ERROR_CONTEXT;
        }
        for (unsigned i = 0; i < G_N_ELEMENTS(e->contexts); i++) {
            if (!e->contexts[i].native) {
                c = &e->contexts[i];
                c->native = jrg_gl_context_new(e->platform,
                                               share ? share->native : NULL,
                                               errp);
                if (!c->native) {
                    return JRG_GL_ERROR_HOST;
                }
                c->id = context;
                c->client = client;
                return 0;
            }
        }
        return JRG_GL_ERROR_LIMIT;
    }
    case JRG_GL_CREATE_DRAWABLE:
        if (!drawable || d) {
            return JRG_GL_ERROR_DRAWABLE;
        }
        /* Front + back + depth/stencil + three immutable export slots. */
        uint64_t allocation = (uint64_t)word(r, 32) * word(r, 36) * 24;
        for (unsigned i = 0; i < G_N_ELEMENTS(e->drawables); i++) {
            allocation += (uint64_t)e->drawables[i].width *
                          e->drawables[i].height * 24;
        }
        if (allocation > JRG_GL_MAX_IMAGE_BYTES) {
            return JRG_GL_ERROR_LIMIT;
        }
        for (unsigned i = 0; i < G_N_ELEMENTS(e->drawables); i++) {
            if (!e->drawables[i].native) {
                drain_output(e);
                if (e->current_context && e->current_drawable &&
                    !jrg_gl_context_in_begin(e->current_context->native)) {
                    jrg_gl_flush_drawable(e->current_drawable->native);
                }
                d = &e->drawables[i];
                d->native = jrg_gl_drawable_new(e->platform, word(r, 32),
                                                word(r, 36), errp);
                e->current_context = NULL;
                e->current_drawable = NULL;
                if (!d->native) {
                    return JRG_GL_ERROR_HOST;
                }
                d->client = client;
                d->id = drawable;
                d->width = word(r, 32);
                d->height = word(r, 36);
                ++e->next_epoch;
                for (unsigned j = 0; j < G_N_ELEMENTS(e->drawables); j++) {
                    e->drawables[j].epoch = e->next_epoch;
                }
                return 0;
            }
        }
        return JRG_GL_ERROR_LIMIT;
    case JRG_GL_CLOSE_CLIENT:
        close_client(e, client);
        return 0;
    case JRG_GL_DESTROY_CONTEXT:
        if (!c) {
            return JRG_GL_ERROR_CONTEXT;
        }
        drain_output(e);
        if (e->current_context == c && e->current_drawable &&
            !jrg_gl_context_in_begin(c->native)) {
            jrg_gl_flush_drawable(e->current_drawable->native);
        }
        jrg_gl_context_free(c->native);
        memset(c, 0, sizeof(*c));
        e->current_context = NULL;
        return 0;
    case JRG_GL_DESTROY_DRAWABLE:
        if (!d) {
            return JRG_GL_ERROR_DRAWABLE;
        }
        delete_drawable(e, d);
        return 0;
    case JRG_GL_MAKE_CURRENT:
        if (!c || !d) {
            return JRG_GL_ERROR_CONTEXT;
        }
        if (!make_current(e, c, d, errp)) {
            return JRG_GL_ERROR_HOST;
        }
        c->drawable = drawable;
        return 0;
    case JRG_GL_CALL:
    case JRG_GL_DATA_CALL:
    case JRG_GL_QUERY:
    case JRG_GL_PRESENT:
        if (!c) {
            return JRG_GL_ERROR_CONTEXT;
        }
        d = find_drawable(e, client, c->drawable);
        if (!d || (drawable && d->id != drawable)) {
            return JRG_GL_ERROR_DRAWABLE;
        }
        if (op == JRG_GL_PRESENT &&
            (word(r, JRG_GL_OFF_FLAGS) & JRG_GL_PRESENT_BOUNDED) &&
            (word(r, 32) != d->width || word(r, 36) != d->height)) {
            /*
             * The window may have resized since the frontend submitted its
             * drawing. Reject its locked WNDOBJ size before any exchange or
             * export can feed stale image dimensions to the compositor.
             */
            return JRG_GL_ERROR_DRAWABLE;
        }
        if (op != JRG_GL_CALL && jrg_gl_context_in_begin(c->native)) {
            return JRG_GL_ERROR_CONTEXT;
        }
        if (!make_current(e, c, d, errp)) {
            return JRG_GL_ERROR_HOST;
        }
        if (op == JRG_GL_CALL) {
            return jrg_gl_call(c->native, word(r, 32), r + 36);
        }
        if (op == JRG_GL_DATA_CALL) {
            uint32_t fn = word(r, JRG_GL_DATA_FUNCTION);
            uint32_t count = jrg_gl_function_words(fn) &
                             ~JRG_GL_FUNCTION_INLINE_DATA;
            return jrg_gl_data_call(c->native, fn, r + JRG_GL_DATA_ARGS,
                                    r + JRG_GL_DATA_ARGS + count * 4,
                                    word(r, JRG_GL_DATA_BYTES));
        }
        if (op == JRG_GL_QUERY) {
            return jrg_gl_query(c->native, word(r, 32), r + 36,
                                batch->result, &batch->result_bytes,
                                &batch->result_type);
        }
        if ((word(r, JRG_GL_OFF_FLAGS) & JRG_GL_PRESENT_EXCLUSIVE) &&
            (d->width != batch->primary_width ||
             d->height != batch->primary_height)) {
            return JRG_GL_ERROR_DRAWABLE;
        }
        return present(e, c, d, word(r, JRG_GL_OFF_FLAGS), errp);
    default:
        return JRG_GL_ERROR_UNSUPPORTED;
    }
}

static void *render_worker(void *opaque)
{
    JrgGLEngine *e = opaque;
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
        qemu_thread_create(&e->release_thread, "juke-gl-release",
                            release_worker, e, QEMU_THREAD_JOINABLE);
        qemu_thread_create(&e->completion_thread, "juke-gl-fence",
                            completion_worker, e, QEMU_THREAD_JOINABLE);
        e->threads_started = true;
    }
    qemu_mutex_lock(&e->lock);
    while (!e->stopping) {
        JrgBatch *batch;
        uint32_t result = 0;

        while (!e->batch && !e->reset && !e->stopping) {
            qemu_cond_wait(&e->cond, &e->lock);
        }
        if (e->stopping) {
            break;
        }
        if (e->reset) {
            qemu_mutex_unlock(&e->lock);
            drain_output(e);
            for (unsigned i = 0; i < G_N_ELEMENTS(e->drawables); i++) {
                if (e->drawables[i].native) {
                    drop_drawable_resource(e, &e->drawables[i]);
                }
            }
            uint8_t reset_packet[JGPU_PACKET_BYTES];
            packet_init(reset_packet, JGPU_KIND_RESET);
            stq_le_p(reset_packet + JGPU_CPU_OFF_LEGACY_EPOCH,
                     e->reset_cpu_epoch);
            stq_le_p(reset_packet + JGPU_CPU_OFF_LEGACY_FRAME,
                     e->reset_cpu_generation);
            if (!send_record(e, reset_packet, NULL, 0)) {
                qemu_mutex_lock(&e->lock);
                e->failed = true;
                qemu_mutex_unlock(&e->lock);
            }
            close_client(e, 0);
            qemu_mutex_lock(&e->lock);
            e->reset = false;
            e->exclusive = false;
            e->desktop_active = e->desktop_coherent = false;
            memset(&e->last_present, 0, sizeof(e->last_present));
            JrgGLCompletion *reset_done = g_new0(JrgGLCompletion, 1);
            reset_done->generation = e->reset_generation;
            reset_done->reset = true;
            g_queue_push_tail(&e->completions, reset_done);
            qemu_mutex_unlock(&e->lock);
            e->notify(e->opaque);
            qemu_mutex_lock(&e->lock);
        }
        batch = e->batch;
        e->batch = NULL;
        if (!batch) {
            continue;
        }
        result = e->failed ? JRG_GL_ERROR_TRANSPORT : 0;
        qemu_mutex_unlock(&e->lock);
        uint64_t trace_start_us = batch->trace_queued_us ?
                                 g_get_monotonic_time() : 0;
        for (size_t offset = 0; !result && offset < batch->bytes;) {
            qemu_mutex_lock(&e->lock);
            bool cancelled = e->reset || e->stopping;
            qemu_mutex_unlock(&e->lock);
            if (cancelled) {
                result = JRG_GL_ERROR_GENERATION;
                break;
            }
            err = NULL;
            result = execute_record(e, batch->data + offset, batch, &err);
            if (err) {
                error_report_err(err);
            }
            offset += word(batch->data + offset, JRG_GL_OFF_SIZE);
        }
        if (trace_start_us) {
            trace_juke_retro_gl_work(batch->sequence, batch->records,
                                    batch->bytes,
                                    trace_start_us - batch->trace_queued_us,
                                    g_get_monotonic_time() - trace_start_us,
                                    result);
        }
        JrgGLCompletion *done = g_new(JrgGLCompletion, 1);
        *done = (JrgGLCompletion) { .sequence = batch->sequence,
                                    .generation = batch->generation,
                                    .error = result,
                                    .resources_live = e->exclusive ||
                                                      e->desktop_active,
                                    .present = e->last_present };
        if (!result) {
            done->result_bytes = batch->result_bytes;
            done->result_type = batch->result_type;
            memcpy(done->result, batch->result, batch->result_bytes);
        }
        for (unsigned i = 0; i < G_N_ELEMENTS(e->contexts); i++) {
            done->resources_live |= e->contexts[i].native != NULL;
        }
        for (unsigned i = 0; i < G_N_ELEMENTS(e->drawables); i++) {
            done->resources_live |= e->drawables[i].native != NULL;
        }
        g_free(batch->data);
        g_free(batch);
        qemu_mutex_lock(&e->lock);
        if (g_queue_get_length(&e->completions) == 16) {
            g_free(g_queue_pop_head(&e->completions));
        }
        g_queue_push_tail(&e->completions, done);
        qemu_mutex_unlock(&e->lock);
        e->notify(e->opaque);
        qemu_mutex_lock(&e->lock);
    }
    qemu_mutex_unlock(&e->lock);
    jrg_gl_clear_current(e->platform);
    return NULL;
}

JrgGLEngine *jrg_gl_engine_new(const char *socket_path, JrgGLNotify notify,
                              void *opaque)
{
    JrgGLEngine *e = g_new0(JrgGLEngine, 1);

    qemu_mutex_init(&e->lock);
    qemu_mutex_init(&e->send_lock);
    qemu_cond_init(&e->cond);
    g_queue_init(&e->completions);
    g_queue_init(&e->pending_output);
    e->socket_path = g_strdup(socket_path);
    e->socket = -1;
    e->reply_fd = -1;
    for (unsigned i = 0; i < G_N_ELEMENTS(e->cpu_slots); i++) {
        e->cpu_slots[i].fd = -1;
    }
    e->notify = notify;
    e->opaque = opaque;
    qemu_thread_create(&e->render_thread, "juke-gl", render_worker, e,
                        QEMU_THREAD_JOINABLE);
    return e;
}

void jrg_gl_engine_free(JrgGLEngine *e)
{
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
    jrg_gl_platform_free(e->platform);
    if (e->batch) {
        g_free(e->batch->data);
        g_free(e->batch);
    }
    g_queue_clear_full(&e->completions, g_free);
#ifdef CONFIG_DARWIN
    if (e->remote) {
        mach_port_deallocate(mach_task_self(), e->remote);
    }
#endif
    if (e->socket >= 0) {
        close(e->socket);
    }
    for (unsigned i = 0; i < G_N_ELEMENTS(e->cpu_slots); i++) {
        JrgCpuSlot *s = &e->cpu_slots[i];
        if (s->pixels) {
            qemu_memfd_free(s->pixels, s->bytes, s->fd);
        }
    }
    if (e->reply_fd >= 0) {
        close(e->reply_fd);
    }
    g_free(e->socket_path);
    qemu_cond_destroy(&e->cond);
    qemu_mutex_destroy(&e->lock);
    qemu_mutex_destroy(&e->send_lock);
    g_free(e);
}

bool jrg_gl_engine_submit(JrgGLEngine *e, uint8_t *data, size_t bytes,
                           uint32_t sequence, uint32_t generation,
                           uint32_t primary_width, uint32_t primary_height,
                           uint32_t records)
{
    qemu_mutex_lock(&e->lock);
    if (e->batch || e->stopping) {
        qemu_mutex_unlock(&e->lock);
        return false;
    }
    e->batch = g_new0(JrgBatch, 1);
    *e->batch = (JrgBatch) { .data = data, .bytes = bytes,
                            .sequence = sequence, .generation = generation,
                            .records = records,
                            .trace_queued_us =
                                trace_event_get_state_backends(
                                    TRACE_JUKE_RETRO_GL_WORK) ?
                                    g_get_monotonic_time() : 0,
                            .primary_width = primary_width,
                            .primary_height = primary_height };
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
    return true;
}

void jrg_gl_engine_reset(JrgGLEngine *e, uint32_t generation,
                          uint64_t cpu_epoch, uint64_t cpu_generation)
{
    qemu_mutex_lock(&e->lock);
    e->reset = true;
    e->reset_generation = generation;
    e->reset_cpu_epoch = cpu_epoch;
    e->reset_cpu_generation = cpu_generation;
    if (e->batch) {
        g_free(e->batch->data);
        g_free(e->batch);
        e->batch = NULL;
    }
    qemu_cond_broadcast(&e->cond);
    qemu_mutex_unlock(&e->lock);
}

bool jrg_gl_engine_completion(JrgGLEngine *e, JrgGLCompletion *completion)
{
    JrgGLCompletion *done;

    qemu_mutex_lock(&e->lock);
    done = g_queue_pop_head(&e->completions);
    qemu_mutex_unlock(&e->lock);
    if (!done) {
        return false;
    }
    *completion = *done;
    g_free(done);
    return true;
}
