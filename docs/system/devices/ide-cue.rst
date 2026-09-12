CUE/BIN optical media
====================

The read-only ``cue`` block format preserves a disc's data and audio tracks,
INDEX 00/01 boundaries and absent PREGAP sectors. It is selected explicitly;
the original BIN is opened as a raw local file and is never modified.

Use the ordinary IDE CD-ROM device and audio backend::

  -audiodev juke,id=audio0,path=/tmp/juke-audio.sock \
  -global ide-cd.audiodev=audio0 \
  -drive file=disc.cue,format=cue,readonly=on,media=cdrom,if=ide,index=2

An explicit ``-drive if=none,id=disc,... -device ide-cd,drive=disc,
audiodev=audio0`` works as well. QMP ``blockdev-change-medium`` accepts
``format: "cue"`` with the existing IDE device ID. An ISO remains ``raw``.

Supported representation
------------------------

* One local ``FILE "name.bin" BINARY``; quoted paths, CRLF and REM comments.
* Sequential tracks 1 through 99: AUDIO, MODE1/2352, or MODE1/2048. All tracks
  in a single BIN must have the same stored sector size.
* INDEX 01 is required, with optional INDEX 00 on subsequent tracks. Audio
  PREGAP inserts absent zero-filled sectors and cannot accompany INDEX 00.
* The first track starts at file frame zero. Descriptors are limited to
  64 KiB, and the validated complete disc is limited to 100 minutes.

Other directives, multiple files, partial stored sectors, overlapping or
empty tracks, and ambiguous layouts fail at open with a descriptive error.
FILE indexes address stored sectors; PREGAP changes subsequent disc LBAs.
Reported MSF additionally includes the standard 150-frame lead-in.

The IDE frontend exposes real TOC formats 0, 1 and 2, current-position
subchannel data, and normal or raw PIO/DMA sector reads. Raw reads preserve
stored mode headers and EDC/ECC. Audio reads return original 2352-byte frames;
normal 2048-byte data reads reject audio tracks.

Audio playback
--------------

PLAY AUDIO (10/12/MSF), PAUSE/RESUME and STOP PLAY consume actual CDDA samples.
The audio-control mode page supports current, default and changeable values,
and PIO MODE SELECT (10) sets independent stereo volume/mute. Stereo output
routing is left-to-left and right-to-right (or muted); unsupported routing,
SOTC and DMA MODE SELECT return a command error. Data tracks cannot be played.

CUE BINARY samples are 44.1 kHz stereo signed 16-bit little-endian PCM. Two
16-sector buffers are filled asynchronously through the block layer; the
existing audio backend clock consumes them. Idle media starts no audio voice
or playback timer. Reset, ejection and data reads stop playback and cancel
prefetch. Snapshot state preserves the playback range, position, paused state
and volume, and validates it against the restored medium.

Verification
------------

With the existing QEMU build, run::

  QTEST_QEMU_BINARY=./qemu-system-i386 QTEST_QEMU_IMG=./qemu-img \
    tests/qtest/ide-test -p /i386/ide/cdrom/cue

Tests cover PIO/DMA reads, original raw bytes, pregaps and TOC addresses,
malformed descriptors, the Windows MCI subchannel request, playback/paused
state, snapshot restoration, reset cancellation, exact full-length stereo
PCM output and independent mute. Existing ISO tests remain under
``/i386/ide/cdrom``.
