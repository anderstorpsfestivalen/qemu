/*
 * Event-driven Red Book audio for IDE optical images.
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * CUE FILE BINARY is Intel (little-endian) PCM, 44100 Hz, stereo, signed16.
 * Reference: GNU ccd2cue FILE documentation; MMC-1 sections 5.1 and 5.2.3.1.
 * All state is under the BQL. Two bounded buffers decouple asynchronous media
 * reads from the existing QEMU audio clock. No timer or I/O runs while idle.
 */
#include "qemu/osdep.h"
#include "qemu/audio.h"
#include "qemu/error-report.h"
#include "qapi/error.h"
#include "system/block-backend.h"
#include "ide-internal.h"

#define CDDA_FRAMES 16
#define CDDA_BYTES (CDDA_FRAMES * CDROM_IMAGE_RAW_BYTES)
#define CDDA_PLAYING 0x11
#define CDDA_PAUSED 0x12
#define CDDA_COMPLETED 0x13
#define CDDA_ERROR 0x14
#define CDDA_STOPPED 0x15

typedef struct CDAudioBuffer {
    uint8_t data[CDDA_BYTES];
    unsigned length;
    unsigned used;
} CDAudioBuffer;

struct IDECDAudio {
    IDEState *drive;
    SWVoiceOut *voice;
    BlockAIOCB *aiocb;
    CdromImageRead request;
    CDAudioBuffer buffer[2];
    unsigned read_index;
    unsigned write_index;
    uint32_t next_lba;
    bool stopping;
};

static void cd_audio_prefetch(IDECDAudio *a);

static void cd_audio_read_done(void *opaque, int ret)
{
    IDECDAudio *a = opaque;
    IDEState *s = a->drive;
    CDAudioBuffer *b = &a->buffer[a->write_index];

    a->aiocb = NULL;
    if (a->stopping) {
        return;
    }
    if (ret < 0) {
        s->cdrom_audio_status = CDDA_ERROR;
        audio_be_set_active_out(s->cdrom_audio_be, a->voice, false);
        return;
    }
    b->length = a->request.count * CDROM_IMAGE_RAW_BYTES;
    b->used = a->next_lba == s->cdrom_position ? s->cdrom_audio_offset : 0;
    a->next_lba += a->request.count;
    a->write_index ^= 1;
    if (s->cdrom_audio_status == CDDA_PLAYING) {
        audio_be_set_active_out(s->cdrom_audio_be, a->voice, true);
    }
    cd_audio_prefetch(a);
}

static void cd_audio_prefetch(IDECDAudio *a)
{
    IDEState *s = a->drive;
    CDAudioBuffer *b = &a->buffer[a->write_index];

    if (a->stopping || a->aiocb || b->length ||
        (s->cdrom_audio_status != CDDA_PLAYING &&
         s->cdrom_audio_status != CDDA_PAUSED) ||
        a->next_lba >= s->cdrom_audio_end) {
        return;
    }
    a->request = (CdromImageRead) {
        .lba = a->next_lba,
        .count = MIN(CDDA_FRAMES, s->cdrom_audio_end - a->next_lba),
        .buffer = b->data,
    };
    a->aiocb = blk_aio_ioctl(s->blk, QEMU_CDROM_READ_RAW, &a->request,
                             cd_audio_read_done, a);
}

static void cd_audio_callback(void *opaque, int available)
{
    IDECDAudio *a = opaque;
    IDEState *s = a->drive;

    if (s->cdrom_audio_status != CDDA_PLAYING) {
        return;
    }
    /* At most the two prefetched buffers; one callback cannot loop on I/O. */
    for (unsigned i = 0; i < 2 && available >= 4; i++) {
        CDAudioBuffer *b = &a->buffer[a->read_index];
        unsigned wanted, written;
        if (!b->length) {
            break;
        }
        wanted = MIN((unsigned)available & ~3u, b->length - b->used);
        written = audio_be_write(s->cdrom_audio_be, a->voice,
                                  b->data + b->used, wanted);
        assert(!(written & 3));
        b->used += written;
        available -= written;
        unsigned consumed = s->cdrom_audio_offset + written;
        s->cdrom_position += consumed / CDROM_IMAGE_RAW_BYTES;
        s->cdrom_audio_offset = consumed % CDROM_IMAGE_RAW_BYTES;
        if (b->used != b->length) {
            break;
        }
        b->used = b->length = 0;
        a->read_index ^= 1;
        cd_audio_prefetch(a);
    }
    if (s->cdrom_position == s->cdrom_audio_end) {
        s->cdrom_audio_status = CDDA_COMPLETED;
        audio_be_set_active_out(s->cdrom_audio_be, a->voice, false);
    }
}

void ide_cd_audio_volume(IDEState *s)
{
    if (s->cdrom_audio && s->cdrom_audio->voice) {
        audio_be_set_volume_out_lr(s->cdrom_audio_be, s->cdrom_audio->voice,
                                   false,
                                   s->cdrom_audio_channel[0] ?
                                   s->cdrom_audio_volume[0] : 0,
                                   s->cdrom_audio_channel[1] ?
                                   s->cdrom_audio_volume[1] : 0);
    }
}

void ide_cd_audio_stop(IDEState *s, uint8_t status)
{
    IDECDAudio *a = s->cdrom_audio;
    s->cdrom_audio_status = status;
    if (!a) {
        return;
    }
    a->stopping = true;
    if (a->aiocb) {
        blk_aio_cancel(a->aiocb);
        a->aiocb = NULL;
    }
    if (a->voice) {
        audio_be_set_active_out(s->cdrom_audio_be, a->voice, false);
        audio_be_close_out(s->cdrom_audio_be, a->voice);
        a->voice = NULL;
    }
    memset(a->buffer, 0, sizeof(a->buffer));
    a->read_index = a->write_index = 0;
    a->stopping = false;
}

bool ide_cd_audio_play(IDEState *s, uint32_t start, uint32_t end,
                        unsigned offset, bool paused)
{
    struct audsettings settings = {
        .freq = 44100, .nchannels = 2, .fmt = AUDIO_FORMAT_S16,
        .big_endian = false,
    };
    IDECDAudio *a;
    Error *error = NULL;

    ide_cd_audio_stop(s, CDDA_STOPPED);
    if (!s->cdrom_audio_be && !audio_be_check(&s->cdrom_audio_be, &error)) {
        error_report_err(error);
        return false;
    }
    if (!s->cdrom_audio) {
        s->cdrom_audio = g_new0(IDECDAudio, 1);
        s->cdrom_audio->drive = s;
    }
    a = s->cdrom_audio;
    a->voice = audio_be_open_out(s->cdrom_audio_be, NULL, "ide-cd-audio", a,
                                  cd_audio_callback, &settings);
    if (!a->voice) {
        return false;
    }
    s->cdrom_position = a->next_lba = start;
    s->cdrom_audio_end = end;
    s->cdrom_audio_offset = offset;
    s->cdrom_audio_status = paused ? CDDA_PAUSED : CDDA_PLAYING;
    ide_cd_audio_volume(s);
    cd_audio_prefetch(a);
    return true;
}

void ide_cd_audio_pause(IDEState *s, bool resume)
{
    s->cdrom_audio_status = resume ? CDDA_PLAYING : CDDA_PAUSED;
    if (s->cdrom_audio && s->cdrom_audio->voice) {
        audio_be_set_active_out(s->cdrom_audio_be, s->cdrom_audio->voice,
                                resume);
    }
}

void ide_cd_audio_reset(IDEState *s)
{
    ide_cd_audio_stop(s, CDDA_STOPPED);
    s->cdrom_position = s->cdrom_audio_end = s->cdrom_audio_offset = 0;
    s->cdrom_audio_volume[0] = s->cdrom_audio_volume[1] = 255;
    s->cdrom_audio_channel[0] = 1;
    s->cdrom_audio_channel[1] = 2;
}

void ide_cd_audio_exit(IDEState *s)
{
    ide_cd_audio_stop(s, CDDA_STOPPED);
    g_clear_pointer(&s->cdrom_audio, g_free);
}
