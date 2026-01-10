/*
 * Juke Shared Memory Audio Backend
 *
 * Zero-copy audio output to Juke via mmap'd shared memory ring buffer.
 * QEMU writes audio samples directly to shared memory, Juke reads them.
 *
 * Copyright (c) 2024 Anderstorpsfestivalen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "qemu/audio.h"
#include "qemu/memfd.h"
#include "qemu/error-report.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

#define AUDIO_CAP "juke"
#include "audio_int.h"

/* Magic number: 'JAUD' = 0x4455414A */
#define JUKE_AUDIO_MAGIC 0x4455414A
/* Version 2: Added muted, volume_left, volume_right fields for guest volume control */
#define JUKE_AUDIO_VERSION 2

/* Ring buffer size in frames (must be power of 2) */
#define JUKE_AUDIO_RING_FRAMES 8192

/* Audio format codes (match Rust side) */
#define JUKE_AUDIO_FMT_S16LE 1
#define JUKE_AUDIO_FMT_F32LE 2

/*
 * Audio shared memory header - must match Rust JukeAudioHeader
 */
typedef struct JukeAudioHeader {
    uint32_t magic;           /* JUKE_AUDIO_MAGIC */
    uint32_t version;         /* Protocol version */
    uint32_t sample_rate;     /* e.g., 48000 */
    uint32_t channels;        /* 1 or 2 */
    uint32_t format;          /* JUKE_AUDIO_FMT_* */
    uint32_t ring_frames;     /* Buffer size in frames */
    uint32_t write_idx;       /* Written by QEMU (frame index) */
    uint32_t read_idx;        /* Written by Juke (frame index) */
    uint32_t enabled;         /* 1 = playing, 0 = paused */
    uint32_t muted;           /* 1 = muted by guest, 0 = not muted (v2) */
    uint32_t volume_left;     /* Left channel volume 0-255 (v2) */
    uint32_t volume_right;    /* Right channel volume 0-255 (v2) */
    uint32_t padding[4];      /* Pad to 64 bytes */
    /* Audio samples follow (ring_frames * channels * bytes_per_sample) */
} JukeAudioHeader;

typedef struct JukeAudioState {
    JukeAudioHeader *shmem;
    size_t shmem_size;
    int shmem_fd;
    char *socket_path;
    int client_fd;
    bool fd_sent;
} JukeAudioState;

typedef struct JukeVoiceOut {
    HWVoiceOut hw;
    JukeAudioState *state;
    RateCtl rate;
} JukeVoiceOut;

/*
 * Send shared memory fd to Juke via SCM_RIGHTS
 */
static void juke_audio_send_fd(JukeAudioState *s)
{
    if (s->client_fd < 0 || s->shmem_fd < 0 || s->fd_sent) {
        return;
    }

    struct msghdr msg = {0};
    struct iovec iov[1];
    char buf[1] = {0};
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
        error_report("juke-audio: failed to send fd: %s", strerror(errno));
    } else {
        s->fd_sent = true;
        error_report("juke-audio: sent shmem fd to Juke");
    }
}

/*
 * Try to connect to Juke's socket (Juke is the server)
 */
static int juke_audio_connect(JukeAudioState *s)
{
    if (!s->socket_path || s->client_fd >= 0) {
        return s->client_fd >= 0 ? 0 : -1;
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    error_report("juke-audio: connected to %s", s->socket_path);
    s->client_fd = fd;
    s->fd_sent = false;

    /* Send fd if we have shmem ready */
    if (s->shmem_fd >= 0) {
        juke_audio_send_fd(s);
    }

    return 0;
}

/*
 * Write audio samples to ring buffer
 */
static size_t juke_write(HWVoiceOut *hw, void *buf, size_t len)
{
    JukeVoiceOut *juke = (JukeVoiceOut *)hw;
    JukeAudioState *s = juke->state;

    if (!s || !s->shmem) {
        /* Not ready - use rate control to throttle */
        return audio_rate_get_bytes(&juke->rate, &hw->info, len);
    }

    /* Try to connect if not connected */
    if (s->client_fd < 0) {
        juke_audio_connect(s);
    }

    /* Try to send fd if connected but not sent */
    if (s->client_fd >= 0 && !s->fd_sent) {
        juke_audio_send_fd(s);
    }

    JukeAudioHeader *hdr = s->shmem;

    /* Check if Juke has enabled playback */
    if (!__atomic_load_n(&hdr->enabled, __ATOMIC_ACQUIRE)) {
        /* Disabled - consume audio at rate but don't store */
        return audio_rate_get_bytes(&juke->rate, &hw->info, len);
    }

    uint32_t ring_frames = hdr->ring_frames;
    uint32_t write_idx = hdr->write_idx;
    uint32_t read_idx = __atomic_load_n(&hdr->read_idx, __ATOMIC_ACQUIRE);

    /* Calculate available space in frames */
    uint32_t used = (write_idx - read_idx) & (ring_frames - 1);
    uint32_t free_frames = ring_frames - used - 1;  /* -1 to distinguish full from empty */

    /* Calculate frame size and convert len to frames */
    size_t frame_size = hw->info.bytes_per_frame;
    size_t frames_to_write = len / frame_size;

    if (frames_to_write > free_frames) {
        frames_to_write = free_frames;
    }

    if (frames_to_write == 0) {
        /* Buffer full - use rate control */
        return audio_rate_get_bytes(&juke->rate, &hw->info, len);
    }

    /* Get pointer to audio data (after header) */
    uint8_t *ring_data = (uint8_t *)(hdr + 1);
    size_t ring_bytes = ring_frames * frame_size;

    /* Write samples to ring buffer, handling wrap-around */
    size_t write_offset = (write_idx & (ring_frames - 1)) * frame_size;
    size_t bytes_to_write = frames_to_write * frame_size;
    size_t first_chunk = ring_bytes - write_offset;

    if (first_chunk >= bytes_to_write) {
        memcpy(ring_data + write_offset, buf, bytes_to_write);
    } else {
        memcpy(ring_data + write_offset, buf, first_chunk);
        memcpy(ring_data, (uint8_t *)buf + first_chunk, bytes_to_write - first_chunk);
    }

    /* Update write index with release semantics */
    __atomic_store_n(&hdr->write_idx, write_idx + frames_to_write, __ATOMIC_RELEASE);

    return bytes_to_write;
}

static int juke_init_out(HWVoiceOut *hw, struct audsettings *as, void *drv_opaque)
{
    JukeVoiceOut *juke = (JukeVoiceOut *)hw;
    JukeAudioState *s = (JukeAudioState *)drv_opaque;

    juke->state = s;

    audio_pcm_init_info(&hw->info, as);
    hw->samples = JUKE_AUDIO_RING_FRAMES;
    audio_rate_start(&juke->rate);

    /* Allocate shared memory for audio ring buffer */
    if (!s->shmem) {
        size_t ring_bytes = JUKE_AUDIO_RING_FRAMES * hw->info.bytes_per_frame;
        s->shmem_size = sizeof(JukeAudioHeader) + ring_bytes;

        s->shmem = qemu_memfd_alloc("juke-audio", s->shmem_size, 0, &s->shmem_fd, NULL);
        if (!s->shmem) {
            error_report("juke-audio: failed to allocate shared memory");
            return -1;
        }

        /* Initialize header */
        s->shmem->magic = JUKE_AUDIO_MAGIC;
        s->shmem->version = JUKE_AUDIO_VERSION;
        s->shmem->sample_rate = as->freq;
        s->shmem->channels = as->nchannels;
        s->shmem->format = (as->fmt == AUDIO_FORMAT_F32) ? JUKE_AUDIO_FMT_F32LE : JUKE_AUDIO_FMT_S16LE;
        s->shmem->ring_frames = JUKE_AUDIO_RING_FRAMES;
        s->shmem->write_idx = 0;
        s->shmem->read_idx = 0;
        s->shmem->enabled = 0;  /* Juke will enable when ready */
        s->shmem->muted = 0;
        s->shmem->volume_left = 255;   /* Full volume */
        s->shmem->volume_right = 255;

        error_report("juke-audio: initialized %uHz %uch format=%u ring=%u frames",
                     s->shmem->sample_rate, s->shmem->channels,
                     s->shmem->format, s->shmem->ring_frames);

        /* Try to connect and send fd */
        juke_audio_connect(s);
        if (s->client_fd >= 0 && !s->fd_sent) {
            juke_audio_send_fd(s);
        }
    }

    return 0;
}

static void juke_fini_out(HWVoiceOut *hw)
{
    /* Cleanup handled in juke_audio_fini */
}

/*
 * Handle volume changes from guest OS
 * This is called when the guest's mixer settings change
 */
static void juke_volume_out(HWVoiceOut *hw, Volume *vol)
{
    JukeVoiceOut *juke = (JukeVoiceOut *)hw;
    JukeAudioState *s = juke->state;

    if (!s || !s->shmem) {
        return;
    }

    JukeAudioHeader *hdr = s->shmem;

    /* Update mute state with release semantics for cross-process visibility */
    __atomic_store_n(&hdr->muted, vol->mute ? 1 : 0, __ATOMIC_RELEASE);

    /* Update volume (QEMU uses 0-255 range, same as our protocol) */
    uint32_t vol_l = vol->vol[0];
    uint32_t vol_r = (vol->channels > 1) ? vol->vol[1] : vol->vol[0];

    __atomic_store_n(&hdr->volume_left, vol_l, __ATOMIC_RELEASE);
    __atomic_store_n(&hdr->volume_right, vol_r, __ATOMIC_RELEASE);
}

static void juke_enable_out(HWVoiceOut *hw, bool enable)
{
    JukeVoiceOut *juke = (JukeVoiceOut *)hw;

    if (enable) {
        audio_rate_start(&juke->rate);
    }

    /* Note: s->shmem->enabled is controlled by Juke (reader), not QEMU */
}

static void *juke_audio_init(Audiodev *dev, Error **errp)
{
    JukeAudioState *s;

    /* Get socket path from audiodev options (required) */
    const char *socket_path = dev->u.juke.path;

    s = g_new0(JukeAudioState, 1);
    s->socket_path = g_strdup(socket_path);
    s->shmem_fd = -1;
    s->client_fd = -1;

    error_report("juke-audio: initialized, will connect to %s", socket_path);

    return s;
}

static void juke_audio_fini(void *opaque)
{
    JukeAudioState *s = (JukeAudioState *)opaque;

    if (s->client_fd >= 0) {
        close(s->client_fd);
    }

    if (s->shmem) {
        qemu_memfd_free(s->shmem, s->shmem_size, s->shmem_fd);
    }

    g_free(s->socket_path);
    g_free(s);
}

static struct audio_pcm_ops juke_pcm_ops = {
    .init_out = juke_init_out,
    .fini_out = juke_fini_out,
    .write    = juke_write,
    .buffer_get_free = audio_generic_buffer_get_free,
    .run_buffer_out = audio_generic_run_buffer_out,
    .enable_out = juke_enable_out,
    .volume_out = juke_volume_out,
    /* No input support for now */
};

static struct audio_driver juke_audio_driver = {
    .name           = "juke",
    .init           = juke_audio_init,
    .fini           = juke_audio_fini,
    .pcm_ops        = &juke_pcm_ops,
    .max_voices_out = 1,
    .max_voices_in  = 0,
    .voice_size_out = sizeof(JukeVoiceOut),
    .voice_size_in  = 0,
};

static void register_audio_juke(void)
{
    audio_driver_register(&juke_audio_driver);
}
type_init(register_audio_juke);
