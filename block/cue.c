/*
 * Read-only CDRWIN CUE/BIN optical media.
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * FILE indexes address stored sectors; PREGAP inserts sectors not present in
 * the BIN. This distinction is also implemented by 86Box cdrom_image.c and
 * cdrdao. Only the explicitly supported grammar is accepted here.
 */
#include "qemu/osdep.h"
#include "block/block-io.h"
#include "block/block_int.h"
#include "block/cdrom-image.h"
#include "qapi/error.h"
#include "qobject/qdict.h"
#include "qemu/cutils.h"
#include "qemu/module.h"

#define CUE_MAX_TEXT 65536
#define CUE_MAX_SECTORS (100 * 60 * 75)
#define CUE_READ_SECTORS 32

typedef struct BDRVCueState {
    BdrvChild *data;
    CdromImageInfo info;
} BDRVCueState;

bool bdrv_cdrom_get_info(BlockDriverState *bs, CdromImageInfo *info)
{
    if (!bs || !bs->drv || !bs->drv->format_name ||
        strcmp(bs->drv->format_name, "cue")) {
        return false;
    }
    BDRVCueState *s = bs->opaque;
    *info = s->info;
    return true;
}

static bool cue_number(const char *s, unsigned max, uint32_t *value)
{
    unsigned n;
    if (!*s || !g_ascii_isdigit(*s) || qemu_strtoui(s, NULL, 10, &n) ||
        n > max) {
        return false;
    }
    *value = n;
    return true;
}

static bool cue_time(const char *s, uint32_t *frames)
{
    g_auto(GStrv) part = g_strsplit(s, ":", -1);
    uint32_t m, sec, f;
    if (g_strv_length(part) != 3 || !cue_number(part[0], 99, &m) ||
        !cue_number(part[1], 59, &sec) || !cue_number(part[2], 74, &f)) {
        return false;
    }
    *frames = (m * 60 + sec) * 75 + f;
    return true;
}

static int cue_parse(char *text, CdromImageInfo *info, char **filename,
                      Error **errp)
{
    g_auto(GStrv) lines = g_strsplit(text, "\n", -1);
    bool have_index = false, have_zero = false, have_pregap = false;
    CdromImageTrack *track = NULL;

    for (unsigned line = 0; lines[line]; line++) {
        g_autoptr(GPtrArray) words = g_ptr_array_new_with_free_func(g_free);
        char *p = g_strstrip(lines[line]);
        while (*p) {
            char *start;
            bool quoted = *p == '"';
            start = p + quoted;
            p = start;
            while (*p && (quoted ? *p != '"' : !g_ascii_isspace(*p))) {
                p++;
            }
            if (quoted && !*p) {
                goto invalid;
            }
            g_ptr_array_add(words, g_strndup(start, p - start));
            if (quoted) {
                p++;
                if (*p && !g_ascii_isspace(*p)) {
                    goto invalid;
                }
            }
            while (g_ascii_isspace(*p)) {
                p++;
            }
        }
        if (!words->len) {
            continue;
        }
        char **w = (char **)words->pdata;
        if (!strcmp(w[0], "REM")) {
            continue;
        }
        if (!strcmp(w[0], "FILE")) {
            if (words->len != 3 || *filename || track || !*w[1] ||
                strcmp(w[2], "BINARY") || path_has_protocol(w[1])) {
                goto invalid;
            }
            *filename = g_strdup(w[1]);
        } else if (!strcmp(w[0], "TRACK")) {
            uint32_t number;
            uint16_t bytes;
            uint8_t control;
            if (words->len != 3 || !*filename || (track && !have_index) ||
                !cue_number(w[1], CDROM_IMAGE_MAX_TRACKS, &number) ||
                number != info->tracks + 1) {
                goto invalid;
            }
            if (!strcmp(w[2], "AUDIO")) {
                bytes = 2352;
                control = 0;
            } else if (!strcmp(w[2], "MODE1/2352")) {
                bytes = 2352;
                control = 4;
            } else if (!strcmp(w[2], "MODE1/2048")) {
                bytes = 2048;
                control = 4;
            } else {
                goto invalid;
            }
            if (info->tracks && info->sector_bytes != bytes) {
                goto invalid;
            }
            info->sector_bytes = bytes;
            track = &info->track[info->tracks++];
            track->control = control;
            have_index = have_zero = have_pregap = false;
        } else if (!strcmp(w[0], "PREGAP")) {
            if (words->len != 2 || !track || have_index || have_zero ||
                have_pregap || info->tracks == 1 || track->control ||
                !cue_time(w[1], &track->pregap)) {
                goto invalid;
            }
            have_pregap = true;
        } else if (!strcmp(w[0], "INDEX")) {
            uint32_t index, frame;
            if (words->len != 3 || !track || have_index ||
                !cue_number(w[1], 1, &index) || !cue_time(w[2], &frame)) {
                goto invalid;
            }
            if (!index) {
                if (have_zero || have_pregap || info->tracks == 1) {
                    goto invalid;
                }
                track->file_index0 = frame;
                have_zero = true;
            } else {
                if ((info->tracks == 1 && frame) ||
                    (have_zero && frame < track->file_index0)) {
                    goto invalid;
                }
                track->file_start = frame;
                if (!have_zero) {
                    track->file_index0 = frame;
                }
                have_index = true;
            }
        } else {
            goto invalid;
        }
        continue;
invalid:
        error_setg(errp, "Unsupported or invalid CUE directive on line %u: %s",
                   line + 1, lines[line]);
        return -EINVAL;
    }
    if (!track || !have_index) {
        error_setg(errp, "CUE requires a FILE and complete sequential tracks");
        return -EINVAL;
    }
    return 0;
}

typedef struct CueLength {
    BlockDriverState *bs;
    int64_t result;
    bool done;
} CueLength;

static void coroutine_fn cue_length_entry(void *opaque)
{
    CueLength *p = opaque;
    GRAPH_RDLOCK_GUARD();
    /* bdrv_getlength rounds to 512 bytes; CD frames require exact length. */
    p->result = p->bs->drv->bdrv_co_getlength(p->bs);
    p->done = true;
}

static int64_t cue_file_length(BlockDriverState *bs, Error **errp)
{
    CueLength length = { .bs = bs };
    if (!bs->drv->protocol_name || strcmp(bs->drv->protocol_name, "file") ||
        !bs->drv->bdrv_co_getlength) {
        error_setg(errp, "CUE sources currently require local files");
        return -ENOTSUP;
    }
    qemu_coroutine_enter(qemu_coroutine_create(cue_length_entry, &length));
    BDRV_POLL_WHILE(bs, !length.done);
    return length.result;
}

static int cue_open(BlockDriverState *bs, QDict *options, int flags,
                     Error **errp)
{
    BDRVCueState *s = bs->opaque;
    g_autofree char *text = NULL, *name = NULL, *directory = NULL, *path = NULL;
    int64_t length;
    uint32_t added = 0;
    int ret;

    bdrv_graph_rdlock_main_loop();
    ret = bdrv_apply_auto_read_only(bs, NULL, errp);
    bdrv_graph_rdunlock_main_loop();
    if (ret < 0) {
        return ret;
    }
    ret = bdrv_open_file_child(NULL, options, "file", bs, errp);
    if (ret < 0) {
        return ret;
    }
    length = cue_file_length(bs->file->bs, errp);
    if (length <= 0 || length > CUE_MAX_TEXT) {
        if (length >= 0) {
            error_setg(errp, "CUE text must contain 1..65536 bytes");
        }
        return -EINVAL;
    }
    text = g_malloc0(length + 1);
    bdrv_graph_rdlock_main_loop();
    ret = bdrv_pread(bs->file, 0, length, text, 0);
    bdrv_graph_rdunlock_main_loop();
    if (ret < 0) {
        error_setg_errno(errp, -ret, "Cannot read CUE text");
        return ret;
    }
    if (memchr(text, 0, length)) {
        error_setg(errp, "CUE text contains an embedded NUL");
        return -EINVAL;
    }
    ret = cue_parse(text, &s->info, &name, errp);
    if (ret < 0) {
        return ret;
    }
    bdrv_graph_rdlock_main_loop();
    directory = bdrv_dirname(bs->file->bs, errp);
    bdrv_graph_rdunlock_main_loop();
    if (!directory) {
        return -EINVAL;
    }
    path = path_is_absolute(name) ? g_strdup(name) :
                                   g_strconcat(directory, name, NULL);
    /* Never recursively probe a format from the raw BIN bytes. */
    if (!qdict_haskey(options, "data-file")) {
        qdict_put_str(options, "data-file.driver", "file");
    }
    s->data = bdrv_open_child(qdict_haskey(options, "data-file") ? NULL : path,
                              options, "data-file", bs, &child_of_bds,
                              BDRV_CHILD_DATA, false, errp);
    if (!s->data) {
        return -EINVAL;
    }
    length = cue_file_length(s->data->bs, errp);
    if (length <= 0 || length % s->info.sector_bytes ||
        length / s->info.sector_bytes > CUE_MAX_SECTORS) {
        if (length >= 0) {
            error_setg(errp, "BIN requires complete sectors, max 100 minutes");
        }
        return -EINVAL;
    }
    uint32_t file_sectors = length / s->info.sector_bytes;
    for (unsigned i = 0; i < s->info.tracks; i++) {
        CdromImageTrack *t = &s->info.track[i];
        if (t->file_start >= file_sectors ||
            (i && t->file_index0 <= s->info.track[i - 1].file_start)) {
            error_setg(errp, "CUE track %u is empty, overlaps, or exceeds BIN",
                       i + 1);
            return -EINVAL;
        }
        t->index0 = t->file_index0 + added;
        added += t->pregap;
        t->start = t->file_start + added;
    }
    if (file_sectors + added > CUE_MAX_SECTORS) {
        error_setg(errp, "CUE including generated gaps exceeds 100 minutes");
        return -EINVAL;
    }
    s->info.sectors = file_sectors + added;
    bs->total_sectors = (uint64_t)s->info.sectors * 4;
    return 0;
}

static const CdromImageTrack *cue_track(BDRVCueState *s, uint32_t lba)
{
    for (unsigned i = s->info.tracks; i > 0; i--) {
        if (lba >= s->info.track[i - 1].index0) {
            return &s->info.track[i - 1];
        }
    }
    return NULL;
}

static int coroutine_fn GRAPH_RDLOCK
cue_read_run(BlockDriverState *bs, uint32_t lba, unsigned maximum,
              uint8_t *out, bool raw, unsigned *count)
{
    BDRVCueState *s = bs->opaque;
    const CdromImageTrack *t = cue_track(s, lba);
    uint32_t end;
    int ret;

    if (!t || lba >= s->info.sectors || (raw && s->info.sector_bytes != 2352) ||
        (!raw && (!t->control || lba < t->start))) {
        return -EIO;
    }
    unsigned index = t - s->info.track;
    end = index + 1 < s->info.tracks ? s->info.track[index + 1].index0 :
                                      s->info.sectors;
    *count = MIN(maximum, MIN(end - lba, CUE_READ_SECTORS));
    if (lba < t->start && t->pregap) {
        *count = MIN(*count, t->start - lba);
        memset(out, 0, *count * 2352);
        return 0;
    }
    int64_t file_sector = (int64_t)t->file_start + lba - t->start;
    ret = bdrv_co_pread(s->data, file_sector * s->info.sector_bytes,
                         *count * s->info.sector_bytes, out, 0);
    if (ret < 0) {
        return ret;
    }
    if (!raw && s->info.sector_bytes == 2352) {
        for (unsigned i = 0; i < *count; i++) {
            if (out[i * 2352 + 15] != 1) {
                return -EIO;
            }
        }
    }
    return 0;
}

static int coroutine_fn GRAPH_RDLOCK
cue_co_preadv(BlockDriverState *bs, int64_t offset, int64_t bytes,
               QEMUIOVector *qiov, BdrvRequestFlags flags)
{
    BDRVCueState *s = bs->opaque;
    g_autofree uint8_t *sectors = g_malloc(CUE_READ_SECTORS * 2352);
    int64_t done = 0;
    unsigned payload = s->info.sector_bytes == 2352 ? 16 : 0;

    /* One bounded contiguous file read serves up to 32 logical sectors. */
    while (done < bytes) {
        uint32_t lba = offset / 2048;
        unsigned skip = offset % 2048;
        unsigned count, maximum = MIN(CUE_READ_SECTORS,
                                      DIV_ROUND_UP(bytes - done + skip, 2048));
        int ret = cue_read_run(bs, lba, maximum, sectors, false, &count);
        if (ret < 0) {
            return ret;
        }
        for (unsigned i = 0; i < count; i++) {
            unsigned n = MIN(bytes - done, 2048 - skip);
            const uint8_t *p = sectors + i * s->info.sector_bytes;
            qemu_iovec_from_buf(qiov, done, p + payload + skip, n);
            offset += n;
            done += n;
            skip = 0;
        }
    }
    return 0;
}

static int coroutine_fn GRAPH_RDLOCK
cue_co_ioctl(BlockDriverState *bs, unsigned long int req, void *buf)
{
    BDRVCueState *s = bs->opaque;
    if (req == QEMU_CDROM_GET_INFO) {
        memcpy(buf, &s->info, sizeof(s->info));
        return 0;
    }
    if (req == QEMU_CDROM_READ_RAW) {
        CdromImageRead *r = buf;
        if (!r->buffer || r->count > CDROM_IMAGE_MAX_READ ||
            r->lba > s->info.sectors || r->count > s->info.sectors - r->lba) {
            return -EINVAL;
        }
        for (unsigned i = 0; i < r->count; ) {
            unsigned count;
            int ret = cue_read_run(bs, r->lba + i, r->count - i,
                                    r->buffer + i * 2352, true, &count);
            if (ret < 0) {
                return ret;
            }
            i += count;
        }
        return 0;
    }
    return -ENOTSUP;
}

static BlockDriver bdrv_cue = {
    .format_name = "cue",
    .instance_size = sizeof(BDRVCueState),
    .bdrv_open = cue_open,
    .bdrv_child_perm = bdrv_default_perms,
    .bdrv_co_preadv = cue_co_preadv,
    .bdrv_co_ioctl = cue_co_ioctl,
    .is_format = true,
};

static void bdrv_cue_init(void)
{
    bdrv_register(&bdrv_cue);
}
block_init(bdrv_cue_init);
