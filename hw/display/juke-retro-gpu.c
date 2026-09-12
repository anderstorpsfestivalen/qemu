/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Juke VGA-compatible display with bounded native 2D command execution.
 */
#include "qemu/osdep.h"
#include "qemu/main-loop.h"
#include "qemu/module.h"
#include "qemu/bswap.h"
#include "hw/pci/pci_device.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"
#include "migration/blocker.h"
#include "qapi/error.h"
#include "qemu/error-report.h"
#include "qapi/qapi-events-ui.h"
#include "system/runstate.h"
#include "standard-headers/juke/retro-gpu.h"
#include "standard-headers/juke/retro-gl.h"
#ifdef CONFIG_JUKE_RETRO_GL
#include "juke-retro-gl.h"
#include "juke-retro-gl-platform.h"
#endif
#include "vga_int.h"
#include "ui/juke-shmem.h"
#include "trace.h"
#include "juke-retro-diagnostics.h"

#define TYPE_JUKE_RETRO_GPU "qemu-retro-gpu"
OBJECT_DECLARE_SIMPLE_TYPE(JukeRetroGPU, JUKE_RETRO_GPU)

/* Bound work per main-loop visit so large batches do not monopolize input. */
#define JRG_WORK_QUANTUM (256 * 1024)
#define JRG_INLINE_QUANTUM (4 * 1024 * 1024)
#define JRG_ROW_QUANTUM 2048

struct JukeRetroGPU {
    PCIDevice parent_obj;
    VGACommonState vga;
    MemoryRegion mmio, regs, vga_regs[4];
    QEMUBH *work_bh;
    uint32_t addr_lo, addr_hi, count, sequence;
    uint32_t status, completed, error, irq_enable, irq_status, generation;
    uint32_t active_count, active_sequence, command_index, row, column;
    uint64_t trace_start_us, validated_bytes;
    uint32_t trace_chunks;
    bool inline_no_irq; /* Transient, only true inside a negotiated SUBMIT. */
    JukeNativeCursor cursor;
    uint32_t cursor_addr_lo, cursor_addr_hi, cursor_bytes;
    uint32_t cursor_width, cursor_height, cursor_hot_x, cursor_hot_y;
    uint32_t cursor_format, cursor_x, cursor_y, cursor_flags, cursor_sequence;
    uint32_t cursor_status, cursor_completed, cursor_error;
    uint8_t commands[JRG_MAX_COMMANDS * JRG_COMMAND_BYTES];
    char *gpu_socket;
    bool diagnostics_enabled;
    JrgDiagnostics *diagnostics;
#ifdef CONFIG_JUKE_RETRO_GL
    JrgGLEngine *gl;
    QEMUBH *gl_bh;
    JrgGLFrameRef gl_present;
    JrgGLTransfer gl_transfer;
    bool gl_transfer_active;
    uint32_t gl_transfer_row, gl_transfer_column;
    Error *gl_blocker;
    uint32_t gl_addr_lo, gl_addr_hi, gl_bytes, gl_sequence, gl_generation;
    uint32_t gl_status, gl_completed, gl_error, gl_query_function;
    uint32_t gl_result_addr_lo, gl_result_addr_hi, gl_result_capacity;
    uint32_t gl_result_bytes, gl_result_type, gl_active_result_capacity;
    uint64_t gl_active_result_address;
    uint32_t gl_sensitive_op, gl_fault, gl_fault_operation, gl_fault_sequence;
    uint64_t gl_trace_start_us;
#endif
};

static uint32_t cmd_word(const uint8_t *cmd, unsigned offset)
{
    return ldl_le_p(cmd + offset);
}

static void jrg_update_irq(JukeRetroGPU *s)
{
    pci_set_irq(&s->parent_obj, !!(s->irq_enable & s->irq_status));
}

static void jrg_complete(JukeRetroGPU *s, uint32_t error)
{
    if (s->trace_start_us) {
        uint64_t elapsed = g_get_monotonic_time() - s->trace_start_us;
        trace_juke_retro_gpu_complete(s->active_sequence, error,
                                     s->trace_chunks, elapsed);
    }
    s->error = error;
    s->status = JRG_STATUS_DONE | (error ? JRG_STATUS_ERROR : 0);
    s->completed = s->active_sequence;
    s->active_count = s->command_index = s->row = s->column = 0;
    if (!s->inline_no_irq) {
        s->irq_status |= JRG_IRQ_COMPLETION;
    }
    jrg_update_irq(s);
}

static bool jrg_cursor_pixels_valid(const uint8_t *data, uint32_t count,
                                    uint32_t format)
{
    for (uint32_t i = 0; i < count; i++) {
        uint32_t a = ldl_le_p(data + i * 8);
        uint32_t b = ldl_le_p(data + i * 8 + 4);

        if (format == JRG_CURSOR_AND_XOR) {
            if ((a | b) & 0xff000000) {
                return false;
            }
        } else if (b || (a & 255) > (a >> 24) ||
                   ((a >> 8) & 255) > (a >> 24) ||
                   ((a >> 16) & 255) > (a >> 24)) {
            return false;
        }
    }
    return true;
}

static bool jrg_cursor_source_ram(JukeRetroGPU *s, uint64_t address,
                                  uint32_t bytes)
{
    hwaddr translated, length = bytes;
    MemoryRegion *mr;

    if (address > UINT64_MAX - bytes) {
        return false;
    }
    RCU_READ_LOCK_GUARD();
    mr = address_space_translate(pci_get_address_space(&s->parent_obj),
                                  address, &translated, &length, false,
                                  MEMTXATTRS_UNSPECIFIED);
    return length >= bytes && memory_region_is_ram(mr) &&
           !memory_region_is_rom(mr);
}

static void jrg_cursor_submit(JukeRetroGPU *s, uint32_t operation)
{
    g_autofree uint8_t *data = NULL;
    uint32_t error = 0;
    uint64_t address = ((uint64_t)s->cursor_addr_hi << 32) | s->cursor_addr_lo;

    if (s->cursor_flags & ~JRG_CURSOR_FLAGS_MASK) {
        error = JRG_CURSOR_ERROR_FLAGS;
    } else if (operation == JRG_CURSOR_SHAPE) {
        if (!s->cursor_width || s->cursor_width > JRG_CURSOR_MAX_DIMENSION ||
            !s->cursor_height || s->cursor_height > JRG_CURSOR_MAX_DIMENSION ||
            s->cursor_hot_x >= s->cursor_width ||
            s->cursor_hot_y >= s->cursor_height ||
            (s->cursor_format != JRG_CURSOR_ARGB_PREMULTIPLIED &&
             s->cursor_format != JRG_CURSOR_AND_XOR) ||
            s->cursor_bytes != s->cursor_width * s->cursor_height * 8) {
            error = JRG_CURSOR_ERROR_SHAPE;
        } else if (!jrg_cursor_source_ram(s, address, s->cursor_bytes)) {
            error = JRG_CURSOR_ERROR_DMA;
        } else {
            data = g_malloc(s->cursor_bytes);
            if (pci_dma_read(&s->parent_obj, address, data,
                              s->cursor_bytes) != MEMTX_OK) {
                error = JRG_CURSOR_ERROR_DMA;
            } else if (!jrg_cursor_pixels_valid(data,
                        s->cursor_width * s->cursor_height, s->cursor_format)) {
                error = JRG_CURSOR_ERROR_SHAPE;
            }
        }
    } else if (operation != JRG_CURSOR_MOVE) {
        error = JRG_CURSOR_ERROR_SHAPE;
    }
    if (!error) {
        if (operation == JRG_CURSOR_SHAPE) {
            s->cursor.width = s->cursor_width;
            s->cursor.height = s->cursor_height;
            s->cursor.hot_x = s->cursor_hot_x;
            s->cursor.hot_y = s->cursor_hot_y;
            s->cursor.format = s->cursor_format;
            memset(s->cursor.pixels, 0, sizeof(s->cursor.pixels));
            memcpy(s->cursor.pixels, data, s->cursor_bytes);
        }
        s->cursor.x = s->cursor_x;
        s->cursor.y = s->cursor_y;
        s->cursor.flags = s->cursor_flags;
        juke_shmem_native_cursor(s->vga.con, &s->cursor,
                                  operation == JRG_CURSOR_SHAPE);
    }
    s->cursor_status = JRG_STATUS_DONE | (error ? JRG_STATUS_ERROR : 0);
    s->cursor_error = error;
    s->cursor_completed = s->cursor_sequence;
}

static uint32_t jrg_cursor_read(JukeRetroGPU *s, uint32_t reg)
{
    switch (reg) {
    case JRG_CURSOR_REG_VERSION: return JRG_CURSOR_ABI_VERSION;
    case JRG_CURSOR_REG_ADDR_LO: return s->cursor_addr_lo;
    case JRG_CURSOR_REG_ADDR_HI: return s->cursor_addr_hi;
    case JRG_CURSOR_REG_BYTES: return s->cursor_bytes;
    case JRG_CURSOR_REG_WIDTH: return s->cursor_width;
    case JRG_CURSOR_REG_HEIGHT: return s->cursor_height;
    case JRG_CURSOR_REG_HOT_X: return s->cursor_hot_x;
    case JRG_CURSOR_REG_HOT_Y: return s->cursor_hot_y;
    case JRG_CURSOR_REG_FORMAT: return s->cursor_format;
    case JRG_CURSOR_REG_X: return s->cursor_x;
    case JRG_CURSOR_REG_Y: return s->cursor_y;
    case JRG_CURSOR_REG_FLAGS: return s->cursor_flags;
    case JRG_CURSOR_REG_SEQUENCE: return s->cursor_sequence;
    case JRG_CURSOR_REG_STATUS: return s->cursor_status;
    case JRG_CURSOR_REG_COMPLETED: return s->cursor_completed;
    case JRG_CURSOR_REG_ERROR: return s->cursor_error;
    case JRG_CURSOR_REG_MAX_DIMENSION: return JRG_CURSOR_MAX_DIMENSION;
    default: return 0;
    }
}

static void jrg_cursor_write(JukeRetroGPU *s, uint32_t reg, uint32_t value)
{
    switch (reg) {
    case JRG_CURSOR_REG_ADDR_LO:
        s->cursor_addr_lo = value;
        break;
    case JRG_CURSOR_REG_ADDR_HI:
        s->cursor_addr_hi = value;
        break;
    case JRG_CURSOR_REG_BYTES:
        s->cursor_bytes = value;
        break;
    case JRG_CURSOR_REG_WIDTH:
        s->cursor_width = value;
        break;
    case JRG_CURSOR_REG_HEIGHT:
        s->cursor_height = value;
        break;
    case JRG_CURSOR_REG_HOT_X:
        s->cursor_hot_x = value;
        break;
    case JRG_CURSOR_REG_HOT_Y:
        s->cursor_hot_y = value;
        break;
    case JRG_CURSOR_REG_FORMAT:
        s->cursor_format = value;
        break;
    case JRG_CURSOR_REG_X:
        s->cursor_x = value;
        break;
    case JRG_CURSOR_REG_Y:
        s->cursor_y = value;
        break;
    case JRG_CURSOR_REG_FLAGS:
        s->cursor_flags = value;
        break;
    case JRG_CURSOR_REG_SEQUENCE:
        s->cursor_sequence = value;
        break;
    case JRG_CURSOR_REG_SUBMIT:
        jrg_cursor_submit(s, value);
        break;
    }
}

static void jrg_cursor_reset(JukeRetroGPU *s)
{
    s->cursor_addr_lo = s->cursor_addr_hi = s->cursor_bytes = 0;
    s->cursor_width = s->cursor_height = s->cursor_hot_x = s->cursor_hot_y = 0;
    s->cursor_format = s->cursor_x = s->cursor_y = s->cursor_flags = 0;
    s->cursor_sequence = s->cursor_status = s->cursor_completed = 0;
    s->cursor_error = 0;
    memset(&s->cursor, 0, sizeof(s->cursor));
    juke_shmem_native_cursor(s->vga.con, &s->cursor, true);
}

#ifdef CONFIG_JUKE_RETRO_GL
static void jrg_fault_stop(JukeRetroGPU *s, uint32_t reason, uint32_t sequence)
{
    if (!reason || reason > JRG_GL_FAULT_DRIVER_INTERNAL || s->gl_fault) {
        return;
    }
    s->gl_fault = reason;
    s->gl_fault_operation = s->gl_sensitive_op;
    s->gl_fault_sequence = sequence;
    g_autofree char *path = object_get_canonical_path(OBJECT(s));
    error_report("Juke graphics coherence fault %u "
                 "(operation %u, sequence %u); "
                 "guest stopped, system reset required", reason,
                 s->gl_fault_operation, s->gl_fault_sequence);
    qapi_event_send_juke_retro_gpu_fault(path, reason, s->gl_fault_operation,
                                        s->gl_fault_sequence);
    vm_stop(RUN_STATE_INTERNAL_ERROR);
}

static bool jrg_result_ram(JukeRetroGPU *s, uint64_t address, uint32_t bytes)
{
    hwaddr translated, length = bytes;
    MemoryRegion *mr;

    if (!bytes || bytes > JRG_GL_MAX_READBACK_BYTES ||
        address > UINT64_MAX - bytes) {
        return false;
    }
    RCU_READ_LOCK_GUARD();
    mr = address_space_translate(pci_get_address_space(&s->parent_obj),
                                  address, &translated, &length, true,
                                  MEMTXATTRS_UNSPECIFIED);
    return length >= bytes && memory_region_is_ram(mr) &&
           !memory_region_is_rom(mr);
}

static void jrg_gl_complete(JukeRetroGPU *s, uint32_t sequence, uint32_t error)
{
    if (s->gl_trace_start_us) {
        trace_juke_retro_gl_complete(sequence, error,
                                    g_get_monotonic_time() -
                                    s->gl_trace_start_us);
        s->gl_trace_start_us = 0;
    }
    if (error && s->gl_sensitive_op) {
        jrg_fault_stop(s, JRG_GL_FAULT_HOST_COHERENCE, sequence);
    }
    s->gl_status = JRG_STATUS_DONE | (error ? JRG_STATUS_ERROR : 0);
    s->gl_completed = sequence;
    s->gl_error = error;
    s->irq_status |= JRG_IRQ_GL_COMPLETION;
    jrg_update_irq(s);
}

static void jrg_gl_transfer_work(JukeRetroGPU *s)
{
    JrgGLTransfer *t = &s->gl_transfer;
    uint32_t budget = JRG_WORK_QUANTUM;
    uint32_t chunks = 0, error = 0;
    uint64_t cpu_epoch = 0, cpu_generation = 0;

    if (!s->gl_transfer_active && jrg_gl_engine_transfer(s->gl, t)) {
        s->gl_transfer_active = true;
        s->gl_transfer_row = s->gl_transfer_column = 0;
    }
    if (!s->gl_transfer_active) {
        return;
    }
    if (t->generation != s->generation) {
        error = JRG_GL_ERROR_GENERATION;
    }
    while (!error && budget && s->gl_transfer_row < t->height) {
        uint32_t row = s->gl_transfer_row;
        uint32_t column = s->gl_transfer_column;
        uint32_t count = MIN(budget, t->width * 4 - column);
        uint64_t offset = (uint64_t)t->offset +
                          (uint64_t)row * t->vram_stride + column;
        uint8_t *pixels = t->pixels + (size_t)row * t->stride + column;
        if (offset + count > s->vga.vram_size) {
            error = JRG_GL_ERROR_DESKTOP;
            break;
        }
        if (t->writeback) {
            memcpy(s->vga.vram_ptr + offset, pixels, count);
            memory_region_set_dirty(&s->vga.vram, offset, count);
        } else {
            memcpy(pixels, s->vga.vram_ptr + offset, count);
        }
        budget -= count;
        if (++chunks == JRG_ROW_QUANTUM) {
            budget = 0;
        }
        s->gl_transfer_column += count;
        if (s->gl_transfer_column == t->width * 4) {
            s->gl_transfer_column = 0;
            s->gl_transfer_row++;
        }
    }
    if (!error && s->gl_transfer_row < t->height) {
        qemu_bh_schedule(s->gl_bh);
        return;
    }
    if (!error && t->return_cpu &&
        !juke_shmem_cpu_anchor(s->vga.con, &cpu_epoch, &cpu_generation)) {
        error = JRG_GL_ERROR_DESKTOP;
    }
    s->gl_transfer_active = false;
    jrg_gl_engine_transfer_done(s->gl, error, cpu_epoch, cpu_generation);
}

static void jrg_gl_completed_bh(void *opaque)
{
    JukeRetroGPU *s = opaque;
    JrgGLCompletion done;

    if (s->gl) {
        jrg_gl_transfer_work(s);
    }
    while (s->gl && jrg_gl_engine_completion(s->gl, &done)) {
        if (done.generation != s->generation) {
            g_free(done.bulk_result);
            continue;
        }
        if (!done.reset) {
            s->gl_present = done.present;
            if (!done.error && done.result_bytes) {
                if (done.result_bytes > s->gl_active_result_capacity ||
                    !jrg_result_ram(s, s->gl_active_result_address,
                                     done.result_bytes) ||
                    pci_dma_write(&s->parent_obj, s->gl_active_result_address,
                                   done.bulk_result ? done.bulk_result : done.result, done.result_bytes) !=
                    MEMTX_OK) {
                    done.error = JRG_GL_ERROR_DMA;
                } else {
                    s->gl_result_bytes = done.result_bytes;
                    s->gl_result_type = done.result_type;
                }
            }
            jrg_gl_complete(s, done.sequence, done.error);
        }
        if (!done.resources_live && !(s->gl_status & JRG_STATUS_BUSY)) {
            migrate_del_blocker(&s->gl_blocker);
        }
        g_free(done.bulk_result);
    }
}

static void jrg_gl_notify(void *opaque)
{
    JukeRetroGPU *s = opaque;

    qemu_bh_schedule(s->gl_bh);
}

static void jrg_gl_submit(JukeRetroGPU *s)
{
    uint64_t address = ((uint64_t)s->gl_addr_hi << 32) | s->gl_addr_lo;
    uint32_t width = 0, height = 0, error, records;
    g_autofree uint8_t *data = NULL;
    Error *err = NULL;

    if ((s->gl_status & JRG_STATUS_BUSY) || s->gl_fault) {
        return;
    }
    s->gl_trace_start_us =
        trace_event_get_state_backends(TRACE_JUKE_RETRO_GL_SUBMIT) ||
        trace_event_get_state_backends(TRACE_JUKE_RETRO_GL_COMPLETE) ?
        g_get_monotonic_time() : 0;
    s->gl_sensitive_op = 0;
    s->gl_result_bytes = s->gl_result_type = 0;
    s->gl_active_result_capacity = 0;
    if (!s->gl_bytes || s->gl_bytes > JRG_GL_MAX_BYTES || (s->gl_bytes & 3)) {
        jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_BATCH);
        return;
    }
    if (s->gl_generation != s->generation) {
        jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_GENERATION);
        return;
    }
    data = g_malloc(s->gl_bytes);
    if (address > UINT64_MAX - s->gl_bytes ||
        pci_dma_read(&s->parent_obj, address, data, s->gl_bytes) != MEMTX_OK) {
        jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_DMA);
        return;
    }
    /* Recognize even an invalid coherence request before signaling failure. */
    for (size_t offset = 0; offset + JRG_GL_HEADER_BYTES <= s->gl_bytes;) {
        const uint8_t *r = data + offset;
        uint32_t size = cmd_word(r, JRG_GL_OFF_SIZE);
        if (size < JRG_GL_HEADER_BYTES || size > s->gl_bytes - offset) {
            break;
        }
        if (cmd_word(r, JRG_GL_OFF_OP) == JRG_GL_DESKTOP && size >= 36) {
            uint32_t op = cmd_word(r, JRG_GL_HEADER_BYTES);
            if (op == JRG_DESKTOP_READBACK || op == JRG_DESKTOP_RETURN) {
                s->gl_sensitive_op = op;
            }
        }
        offset += size;
    }
    if (s->vga.vbe_regs[VBE_DISPI_INDEX_ENABLE] & VBE_DISPI_ENABLED) {
        width = s->vga.vbe_regs[VBE_DISPI_INDEX_XRES];
        height = s->vga.vbe_regs[VBE_DISPI_INDEX_YRES];
    }
    error = jrg_gl_validate(data, s->gl_bytes, s->generation, width, height,
                              s->vga.vram_size, &records);
    if (error) {
        jrg_gl_complete(s, s->gl_sequence, error);
        return;
    }
    if (cmd_word(data, JRG_GL_OFF_OP) == JRG_GL_QUERY) {
        uint32_t required = jrg_gl_query_result_bytes(cmd_word(data, 32),
                                                       data + 36);
        uint64_t result_address = ((uint64_t)s->gl_result_addr_hi << 32) |
                                 s->gl_result_addr_lo;
        if (s->gl_result_capacity < required ||
            s->gl_result_capacity > JRG_GL_MAX_READBACK_BYTES) {
            jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_BATCH);
            return;
        }
        if (!jrg_result_ram(s, result_address, s->gl_result_capacity)) {
            jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_DMA);
            return;
        }
        s->gl_active_result_address = result_address;
        s->gl_active_result_capacity = s->gl_result_capacity;
    }
    if (s->diagnostics_enabled) {
        s->diagnostics->batches++;
        s->diagnostics->bytes += s->gl_bytes;
    }
    for (size_t offset = 0; offset < s->gl_bytes;) {
        const uint8_t *r = data + offset;
        if (s->diagnostics_enabled) {
            jrg_diagnostic_record(s->diagnostics, r);
        }
        if (cmd_word(r, JRG_GL_OFF_OP) == JRG_GL_DESKTOP &&
            (s->vga.vbe_regs[VBE_DISPI_INDEX_BPP] != 32 ||
             (s->status & JRG_STATUS_BUSY))) {
            jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_DESKTOP);
            return;
        }
        offset += cmd_word(r, JRG_GL_OFF_SIZE);
    }
    if (!s->gpu_socket || !*s->gpu_socket) {
        jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_TRANSPORT);
        return;
    }
    if (!s->gl_blocker) {
        error_setg(&s->gl_blocker,
                   "Juke active OpenGL resources cannot be migrated or saved");
        if (migrate_add_blocker(&s->gl_blocker, &err) < 0) {
            error_report_err(err);
            jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_HOST);
            return;
        }
    }
    if (!s->gl) {
        s->gl = jrg_gl_engine_new(s->gpu_socket, jrg_gl_notify, s);
    }
    if (!jrg_gl_engine_submit(s->gl, data, s->gl_bytes, s->gl_sequence,
                              s->generation, width, height, records)) {
        jrg_gl_complete(s, s->gl_sequence, JRG_GL_ERROR_LIMIT);
        return;
    }
    data = NULL;
    s->gl_status = JRG_STATUS_BUSY;
    s->gl_error = 0;
    if (s->gl_trace_start_us) {
        trace_juke_retro_gl_submit(s->gl_sequence, records, s->gl_bytes,
                                  g_get_monotonic_time() -
                                  s->gl_trace_start_us);
    }
}

static uint64_t jrg_gl_read(JukeRetroGPU *s, hwaddr addr)
{
    switch (addr) {
    case JRG_GL_REG_FAULT_STOP: return s->gl_fault;
    case JRG_GL_REG_FAULT_OPERATION: return s->gl_fault_operation;
    case JRG_GL_REG_FAULT_SEQUENCE: return s->gl_fault_sequence;
    case JRG_GL_REG_RESULT_ADDR_LO: return s->gl_result_addr_lo;
    case JRG_GL_REG_RESULT_ADDR_HI: return s->gl_result_addr_hi;
    case JRG_GL_REG_RESULT_CAPACITY: return s->gl_result_capacity;
    case JRG_GL_REG_RESULT_BYTES: return s->gl_result_bytes;
    case JRG_GL_REG_RESULT_TYPE: return s->gl_result_type;
    case JRG_GL_REG_PRESENT_SLOT: return s->gl_present.slot;
    case JRG_GL_REG_PRESENT_EPOCH_LO: return s->gl_present.epoch;
    case JRG_GL_REG_PRESENT_EPOCH_HI: return s->gl_present.epoch >> 32;
    case JRG_GL_REG_PRESENT_FRAME_LO: return s->gl_present.generation;
    case JRG_GL_REG_PRESENT_FRAME_HI: return s->gl_present.generation >> 32;
    case JRG_GL_REG_PRESENT_CLIENT: return s->gl_present.client;
    case JRG_GL_REG_PRESENT_DRAWABLE: return s->gl_present.drawable;
    case JRG_GL_REG_VERSION: return JRG_GL_VERSION;
    case JRG_GL_REG_ADDR_LO: return s->gl_addr_lo;
    case JRG_GL_REG_ADDR_HI: return s->gl_addr_hi;
    case JRG_GL_REG_BYTES: return s->gl_bytes;
    case JRG_GL_REG_SEQUENCE: return s->gl_sequence;
    case JRG_GL_REG_GENERATION: return s->generation;
    case JRG_GL_REG_STATUS: return s->gl_status;
    case JRG_GL_REG_COMPLETED: return s->gl_completed;
    case JRG_GL_REG_ERROR: return s->gl_error;
    case JRG_GL_REG_MAX_BYTES: return JRG_GL_MAX_BYTES;
    case JRG_GL_REG_MAX_RECORDS: return JRG_GL_MAX_RECORDS;
    case JRG_GL_REG_QUERY_FUNCTION: return s->gl_query_function;
    case JRG_GL_REG_FUNCTION_WORDS:
        return jrg_gl_function_words(s->gl_query_function);
    default: return 0;
    }
}

static void jrg_gl_write(JukeRetroGPU *s, hwaddr addr, uint32_t value)
{
    switch (addr) {
    case JRG_GL_REG_FAULT_STOP:
        jrg_fault_stop(s, value, s->gl_sequence);
        break;
    case JRG_GL_REG_RESULT_ADDR_LO:
        s->gl_result_addr_lo = value;
        break;
    case JRG_GL_REG_RESULT_ADDR_HI:
        s->gl_result_addr_hi = value;
        break;
    case JRG_GL_REG_RESULT_CAPACITY:
        s->gl_result_capacity = value;
        break;
    case JRG_GL_REG_ADDR_LO:
        s->gl_addr_lo = value;
        break;
    case JRG_GL_REG_ADDR_HI:
        s->gl_addr_hi = value;
        break;
    case JRG_GL_REG_BYTES:
        s->gl_bytes = value;
        break;
    case JRG_GL_REG_SEQUENCE:
        s->gl_sequence = value;
        break;
    case JRG_GL_REG_GENERATION:
        s->gl_generation = value;
        break;
    case JRG_GL_REG_QUERY_FUNCTION:
        s->gl_query_function = value;
        break;
    case JRG_GL_REG_SUBMIT:
        if (value == 1) {
            jrg_gl_submit(s);
        }
        break;
    }
}
#endif

static bool jrg_rect_valid(JukeRetroGPU *s, uint32_t offset,
                           uint32_t stride, uint32_t width, uint32_t height,
                           uint32_t bpp)
{
    uint64_t row_bytes = (uint64_t)width * bpp;
    uint64_t end = (uint64_t)offset + (uint64_t)(height - 1) * stride +
                   row_bytes;

    return width && height && !(offset % bpp) && !(stride % bpp) &&
           row_bytes <= stride && end <= s->vga.vram_size;
}

/* Validate the entire immutable batch before any framebuffer changes. */
static uint32_t jrg_validate(JukeRetroGPU *s)
{
    uint64_t work = 0;
    unsigned i;

    if (!s->active_count || s->active_count > JRG_MAX_COMMANDS) {
        return JRG_ERROR_BATCH_COUNT;
    }
    for (i = 0; i < s->active_count; i++) {
        const uint8_t *c = s->commands + i * JRG_COMMAND_BYTES;
        uint32_t op = cmd_word(c, JRG_CMD_OPCODE);
        uint32_t bpp = cmd_word(c, JRG_CMD_BPP);
        uint32_t src = cmd_word(c, JRG_CMD_SRC_OFFSET);
        uint32_t dst = cmd_word(c, JRG_CMD_DST_OFFSET);
        uint32_t ss = cmd_word(c, JRG_CMD_SRC_STRIDE);
        uint32_t ds = cmd_word(c, JRG_CMD_DST_STRIDE);
        uint32_t width = cmd_word(c, JRG_CMD_WIDTH);
        uint32_t height = cmd_word(c, JRG_CMD_HEIGHT);

        if ((op != JRG_CMD_FILL && op != JRG_CMD_COPY &&
             op != JRG_CMD_DAMAGE) || (bpp != 1 && bpp != 2 && bpp != 4) ||
            cmd_word(c, JRG_CMD_RESERVED) ||
            (op != JRG_CMD_COPY && (src || ss)) ||
            (op != JRG_CMD_FILL && cmd_word(c, JRG_CMD_COLOR))) {
            return JRG_ERROR_COMMAND;
        }
        if (!jrg_rect_valid(s, dst, ds, width, height, bpp) ||
            (op == JRG_CMD_COPY &&
             (ss != ds || !jrg_rect_valid(s, src, ss, width, height, bpp)))) {
            return JRG_ERROR_BOUNDS;
        }
        work += (uint64_t)width * height * bpp;
        if (work > JRG_MAX_WORK_BYTES) {
            return JRG_ERROR_WORK_LIMIT;
        }
    }
    s->validated_bytes = work;
    return JRG_ERROR_NONE;
}

static void jrg_run(JukeRetroGPU *s, uint32_t budget)
{
    uint64_t work = 0;
    uint32_t rows = 0;
    bool tracing = trace_event_get_state_backends(TRACE_JUKE_RETRO_GPU_WORK);
    uint64_t start_us = tracing ? g_get_monotonic_time() : 0;

    if (s->trace_start_us) {
        s->trace_chunks++;
    }
    while ((s->status & JRG_STATUS_BUSY) &&
           s->command_index < s->active_count) {
        const uint8_t *c = s->commands +
                           s->command_index * JRG_COMMAND_BYTES;
        uint32_t op = cmd_word(c, JRG_CMD_OPCODE);
        uint32_t bpp = cmd_word(c, JRG_CMD_BPP);
        uint32_t src = cmd_word(c, JRG_CMD_SRC_OFFSET);
        uint32_t dst = cmd_word(c, JRG_CMD_DST_OFFSET);
        uint32_t stride = cmd_word(c, JRG_CMD_DST_STRIDE);
        uint32_t width = cmd_word(c, JRG_CMD_WIDTH);
        uint32_t height = cmd_word(c, JRG_CMD_HEIGHT);
        uint32_t bytes = width * bpp;
        bool reverse = op == JRG_CMD_COPY && dst > src;
        uint32_t y = reverse ? height - 1 - s->row : s->row;
        uint32_t n = MIN(bytes - s->column, budget - work) &
                     ~(bpp - 1);
        uint32_t xoffset = reverse ? bytes - s->column - n : s->column;
        uint32_t offset = dst + y * stride + xoffset;
        uint8_t *out = s->vga.vram_ptr + offset;

        if (!n) {
            break;
        }
        if (op == JRG_CMD_COPY) {
            memmove(out, s->vga.vram_ptr + src + y * stride + xoffset, n);
        } else if (op == JRG_CMD_FILL) {
            uint32_t color = cmd_word(c, JRG_CMD_COLOR);
            uint32_t x;

            if (bpp == 1) {
                memset(out, color, n);
            } else {
                for (x = 0; x < n; x += bpp) {
                    if (bpp == 2) {
                        stw_le_p(out + x, color);
                    } else {
                        stl_le_p(out + x, color);
                    }
                }
            }
        }
        memory_region_set_dirty(&s->vga.vram, offset, n);
        work += n;
        s->column += n;
        if (s->column == bytes) {
            s->column = 0;
            if (++s->row == height) {
                s->row = 0;
                s->command_index++;
            }
        }
        ++rows;
        if (work >= budget || rows == JRG_ROW_QUANTUM) {
            break;
        }
    }
    if (tracing) {
        trace_juke_retro_gpu_work(s->active_sequence, work, rows,
                                 g_get_monotonic_time() - start_us,
                                 budget == JRG_INLINE_QUANTUM);
    }
    if (s->command_index == s->active_count) {
        jrg_complete(s, JRG_ERROR_NONE);
    } else {
        qemu_bh_schedule(s->work_bh);
    }
}

static void jrg_work(void *opaque)
{
    jrg_run(opaque, JRG_WORK_QUANTUM);
}

static void jrg_submit(JukeRetroGPU *s)
{
    uint64_t addr = ((uint64_t)s->addr_hi << 32) | s->addr_lo;
    uint32_t error;

    /* A producer owns the channel until completion; never replace its work. */
    if (s->status & JRG_STATUS_BUSY) {
        return;
    }
    s->trace_start_us =
        trace_event_get_state_backends(TRACE_JUKE_RETRO_GPU_COMPLETE) ?
        g_get_monotonic_time() : 0;
    s->trace_chunks = 0;
    s->validated_bytes = 0;
    s->active_sequence = s->sequence;
    s->active_count = s->count;
    s->command_index = s->row = s->column = 0;
    s->error = JRG_ERROR_NONE;
    if (!s->count || s->count > JRG_MAX_COMMANDS) {
        jrg_complete(s, JRG_ERROR_BATCH_COUNT);
        return;
    }
    if (addr > UINT64_MAX - s->count * JRG_COMMAND_BYTES ||
        pci_dma_read(&s->parent_obj, addr, s->commands,
                     s->count * JRG_COMMAND_BYTES) != MEMTX_OK) {
        jrg_complete(s, JRG_ERROR_DMA);
        return;
    }
    error = jrg_validate(s);
    if (error) {
        jrg_complete(s, error);
        return;
    }
    trace_juke_retro_gpu_submit(s->active_sequence, s->active_count,
                               s->validated_bytes, JRG_INLINE_QUANTUM);
    s->status = JRG_STATUS_BUSY;
    /*
     * A full 1024x768x32 GDI blit fits in one bounded MMIO exit. Larger
     * operations continue in small BH quanta; narrow/tall rectangles are
     * separately bounded by the row limit to cap dirty-logging overhead.
     */
    jrg_run(s, JRG_INLINE_QUANTUM);
}

static void jrg_engine_reset(JukeRetroGPU *s)
{
    qemu_bh_cancel(s->work_bh);
    s->trace_start_us = 0;
    s->addr_lo = s->addr_hi = s->count = s->sequence = 0;
    s->status = s->completed = s->error = 0;
    s->irq_enable = s->irq_status = 0;
    s->active_count = s->active_sequence = s->command_index = 0;
    s->row = s->column = 0;
    memset(s->commands, 0, sizeof(s->commands));
    s->generation++;
    jrg_cursor_reset(s);
#ifdef CONFIG_JUKE_RETRO_GL
    s->gl_addr_lo = s->gl_addr_hi = s->gl_bytes = s->gl_sequence = 0;
    s->gl_generation = s->gl_status = s->gl_completed = s->gl_error = 0;
    s->gl_query_function = 0;
    s->gl_sensitive_op = 0;
    s->gl_result_addr_lo = s->gl_result_addr_hi = s->gl_result_capacity = 0;
    s->gl_trace_start_us = 0;
    s->gl_result_bytes = s->gl_result_type = s->gl_active_result_capacity = 0;
    s->gl_active_result_address = 0;
    memset(&s->gl_present, 0, sizeof(s->gl_present));
    if (s->gl) {
        uint64_t epoch = 0, generation = 0;
        juke_shmem_cpu_anchor(s->vga.con, &epoch, &generation);
        jrg_gl_engine_reset(s->gl, s->generation, epoch, generation);
    }
#endif
    jrg_update_irq(s);
}

static uint64_t jrg_read(void *opaque, hwaddr addr, unsigned size)
{
    JukeRetroGPU *s = opaque;
    if (s->diagnostics_enabled && addr + JRG_REG_MAGIC < JRG_MMIO_SIZE) {
        s->diagnostics->reads[(addr + JRG_REG_MAGIC) / 4]++;
    }

    if (addr + JRG_REG_MAGIC >= JRG_CURSOR_REG_VERSION &&
        addr + JRG_REG_MAGIC <= JRG_CURSOR_REG_MAX_DIMENSION) {
        return jrg_cursor_read(s, addr + JRG_REG_MAGIC);
    }
#ifdef CONFIG_JUKE_RETRO_GL
    if (addr + JRG_REG_MAGIC >= JRG_GL_REG_VERSION) {
        return jrg_gl_read(s, addr + JRG_REG_MAGIC);
    }
#endif

    switch (addr + JRG_REG_MAGIC) {
    case JRG_REG_MAGIC: return JRG_MAGIC;
    case JRG_REG_VERSION: return JRG_ABI_VERSION;
    case JRG_REG_CAPS:
        return JRG_CAP_FILL | JRG_CAP_COPY | JRG_CAP_DAMAGE |
               JRG_CAP_COMPLETION_IRQ | JRG_CAP_INLINE_NO_IRQ | JRG_CAP_CURSOR
#ifdef CONFIG_JUKE_RETRO_GL
               | (s->gpu_socket && *s->gpu_socket ?
                  JRG_CAP_GL_TRANSPORT | JRG_CAP_GL_FRONT_BUFFERS |
                  JRG_CAP_GL_PRESENT_BOUNDS | JRG_CAP_GL_BULK_READBACK : 0)
#endif
               ;
    case JRG_REG_VRAM_SIZE: return s->vga.vram_size;
    case JRG_REG_BATCH_ADDR_LO: return s->addr_lo;
    case JRG_REG_BATCH_ADDR_HI: return s->addr_hi;
    case JRG_REG_BATCH_COUNT: return s->count;
    case JRG_REG_SUBMIT_SEQUENCE: return s->sequence;
    case JRG_REG_STATUS: return s->status;
    case JRG_REG_COMPLETED_SEQUENCE: return s->completed;
    case JRG_REG_ERROR: return s->error;
    case JRG_REG_IRQ_ENABLE: return s->irq_enable;
    case JRG_REG_IRQ_STATUS: return s->irq_status;
    case JRG_REG_GENERATION: return s->generation;
    case JRG_REG_MAX_COMMANDS: return JRG_MAX_COMMANDS;
    case JRG_REG_MAX_WORK_BYTES: return JRG_MAX_WORK_BYTES;
    default: return 0;
    }
}

static void jrg_write(void *opaque, hwaddr addr, uint64_t value, unsigned size)
{
    JukeRetroGPU *s = opaque;
    if (s->diagnostics_enabled && addr + JRG_REG_MAGIC < JRG_MMIO_SIZE) {
        s->diagnostics->writes[(addr + JRG_REG_MAGIC) / 4]++;
    }

    if (addr + JRG_REG_MAGIC >= JRG_CURSOR_REG_VERSION &&
        addr + JRG_REG_MAGIC <= JRG_CURSOR_REG_MAX_DIMENSION) {
        jrg_cursor_write(s, addr + JRG_REG_MAGIC, value);
        return;
    }
#ifdef CONFIG_JUKE_RETRO_GL
    if (addr + JRG_REG_MAGIC >= JRG_GL_REG_VERSION) {
        jrg_gl_write(s, addr + JRG_REG_MAGIC, value);
        return;
    }
#endif

    switch (addr + JRG_REG_MAGIC) {
    case JRG_REG_BATCH_ADDR_LO:
        s->addr_lo = value;
        break;
    case JRG_REG_BATCH_ADDR_HI:
        s->addr_hi = value;
        break;
    case JRG_REG_BATCH_COUNT:
        s->count = value;
        break;
    case JRG_REG_SUBMIT_SEQUENCE:
        s->sequence = value;
        break;
    case JRG_REG_SUBMIT:
        if (value == JRG_SUBMIT_START ||
            value == (JRG_SUBMIT_START | JRG_SUBMIT_INLINE_NO_IRQ)) {
            /* Only this MMIO callback can suppress its own completion IRQ. */
            s->inline_no_irq = value & JRG_SUBMIT_INLINE_NO_IRQ;
            jrg_submit(s);
            s->inline_no_irq = false;
        }
        break;
    case JRG_REG_IRQ_ENABLE:
        s->irq_enable = value & (JRG_IRQ_COMPLETION | JRG_IRQ_GL_COMPLETION);
        jrg_update_irq(s);
        break;
    case JRG_REG_IRQ_STATUS:
        s->irq_status &= ~(value &
                           (JRG_IRQ_COMPLETION | JRG_IRQ_GL_COMPLETION));
        jrg_update_irq(s);
        break;
    case JRG_REG_RESET:
        if (value == 1) {
#ifdef CONFIG_JUKE_RETRO_GL
            if (s->gl_fault) {
                break;
            }
#endif
            jrg_engine_reset(s);
        }
        break;
    }
}

static const MemoryRegionOps jrg_ops = {
    .read = jrg_read,
    .write = jrg_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = { .min_access_size = 4, .max_access_size = 4 },
    .impl = { .min_access_size = 4, .max_access_size = 4 },
};

static int jrg_post_load(void *opaque, int version_id)
{
    JukeRetroGPU *s = opaque;

    if ((s->cursor.flags & ~JRG_CURSOR_FLAGS_MASK) ||
        (s->cursor_status & JRG_STATUS_BUSY) ||
        s->cursor.width > JRG_CURSOR_MAX_DIMENSION ||
        s->cursor.height > JRG_CURSOR_MAX_DIMENSION ||
        (!!s->cursor.width != !!s->cursor.height) ||
        (s->cursor.width &&
         (s->cursor.hot_x < 0 || s->cursor.hot_y < 0 ||
          s->cursor.hot_x >= s->cursor.width ||
          s->cursor.hot_y >= s->cursor.height ||
          (s->cursor.format != JRG_CURSOR_ARGB_PREMULTIPLIED &&
           s->cursor.format != JRG_CURSOR_AND_XOR) ||
          !jrg_cursor_pixels_valid(s->cursor.pixels,
                    s->cursor.width * s->cursor.height, s->cursor.format)))) {
        return -EINVAL;
    }
#ifdef CONFIG_JUKE_RETRO_GL
    /* Active GL jobs/resources are never part of a valid saved state. */
    if (s->gl_status & JRG_STATUS_BUSY) {
        return -EINVAL;
    }
#endif
    if (s->status & JRG_STATUS_BUSY) {
        const uint8_t *c;

        if (jrg_validate(s) || s->command_index >= s->active_count ||
            s->row >= cmd_word(s->commands +
                              s->command_index * JRG_COMMAND_BYTES,
                              JRG_CMD_HEIGHT)) {
            return -EINVAL;
        }
        c = s->commands + s->command_index * JRG_COMMAND_BYTES;
        if (s->column >= cmd_word(c, JRG_CMD_WIDTH) *
                         cmd_word(c, JRG_CMD_BPP) ||
            s->column % cmd_word(c, JRG_CMD_BPP)) {
            return -EINVAL;
        }
        qemu_bh_schedule(s->work_bh);
    }
    jrg_update_irq(s);
    juke_shmem_native_cursor(s->vga.con, &s->cursor, true);
    return 0;
}

static int jrg_pre_load(void *opaque)
{
    JukeRetroGPU *s = opaque;

    qemu_bh_cancel(s->work_bh);
    s->trace_start_us = 0;
    jrg_cursor_reset(s);
#ifdef CONFIG_JUKE_RETRO_GL
    /*
     * Restoring a pre-3D checkpoint discards the current graphics session.
     * This explicit VM restore may wait for teardown; normal MMIO never does.
     */
    qemu_bh_cancel(s->gl_bh);
    s->gl_transfer_active = false;
    jrg_gl_engine_free(s->gl);
    s->gl = NULL;
    memset(&s->gl_present, 0, sizeof(s->gl_present));
    s->gl_result_addr_lo = s->gl_result_addr_hi = s->gl_result_capacity = 0;
    s->gl_result_bytes = s->gl_result_type = s->gl_active_result_capacity = 0;
    s->gl_active_result_address = 0;
    s->gl_fault = s->gl_fault_operation = s->gl_fault_sequence = 0;
    migrate_del_blocker(&s->gl_blocker);
#endif
    return 0;
}

static const VMStateDescription vmstate_jrg_cursor = {
    .name = "juke-retro-cursor",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(width, JukeNativeCursor),
        VMSTATE_UINT32(height, JukeNativeCursor),
        VMSTATE_UINT32(format, JukeNativeCursor),
        VMSTATE_INT32(hot_x, JukeNativeCursor),
        VMSTATE_INT32(hot_y, JukeNativeCursor),
        VMSTATE_INT32(x, JukeNativeCursor),
        VMSTATE_INT32(y, JukeNativeCursor),
        VMSTATE_UINT32(flags, JukeNativeCursor),
        VMSTATE_UINT8_ARRAY(pixels, JukeNativeCursor, JRG_CURSOR_MAX_BYTES),
        VMSTATE_END_OF_LIST()
    },
};

static const VMStateDescription vmstate_jrg = {
    .name = TYPE_JUKE_RETRO_GPU,
    .version_id = 3,
    .minimum_version_id = 1,
    .pre_load = jrg_pre_load,
    .post_load = jrg_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, JukeRetroGPU),
        VMSTATE_STRUCT(vga, JukeRetroGPU, 0, vmstate_vga_common,
                       VGACommonState),
        VMSTATE_UINT32(addr_lo, JukeRetroGPU),
        VMSTATE_UINT32(addr_hi, JukeRetroGPU),
        VMSTATE_UINT32(count, JukeRetroGPU),
        VMSTATE_UINT32(sequence, JukeRetroGPU),
        VMSTATE_UINT32(status, JukeRetroGPU),
        VMSTATE_UINT32(completed, JukeRetroGPU),
        VMSTATE_UINT32(error, JukeRetroGPU),
        VMSTATE_UINT32(irq_enable, JukeRetroGPU),
        VMSTATE_UINT32(irq_status, JukeRetroGPU),
        VMSTATE_UINT32(generation, JukeRetroGPU),
        VMSTATE_UINT32(active_count, JukeRetroGPU),
        VMSTATE_UINT32(active_sequence, JukeRetroGPU),
        VMSTATE_UINT32(command_index, JukeRetroGPU),
        VMSTATE_UINT32(row, JukeRetroGPU),
        VMSTATE_UINT32(column, JukeRetroGPU),
        VMSTATE_UINT8_ARRAY(commands, JukeRetroGPU,
                           JRG_MAX_COMMANDS * JRG_COMMAND_BYTES),
        VMSTATE_STRUCT(cursor, JukeRetroGPU, 3, vmstate_jrg_cursor,
                        JukeNativeCursor),
        VMSTATE_UINT32_V(cursor_addr_lo, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_addr_hi, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_bytes, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_width, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_height, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_hot_x, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_hot_y, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_format, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_x, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_y, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_flags, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_sequence, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_status, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_completed, JukeRetroGPU, 3),
        VMSTATE_UINT32_V(cursor_error, JukeRetroGPU, 3),
#ifdef CONFIG_JUKE_RETRO_GL
        VMSTATE_UINT32(gl_addr_lo, JukeRetroGPU),
        VMSTATE_UINT32(gl_addr_hi, JukeRetroGPU),
        VMSTATE_UINT32(gl_bytes, JukeRetroGPU),
        VMSTATE_UINT32(gl_sequence, JukeRetroGPU),
        VMSTATE_UINT32(gl_generation, JukeRetroGPU),
        VMSTATE_UINT32(gl_status, JukeRetroGPU),
        VMSTATE_UINT32(gl_completed, JukeRetroGPU),
        VMSTATE_UINT32(gl_error, JukeRetroGPU),
        VMSTATE_UINT32(gl_query_function, JukeRetroGPU),
        VMSTATE_UINT32_V(gl_result_addr_lo, JukeRetroGPU, 2),
        VMSTATE_UINT32_V(gl_result_addr_hi, JukeRetroGPU, 2),
        VMSTATE_UINT32_V(gl_result_capacity, JukeRetroGPU, 2),
        VMSTATE_UINT32_V(gl_result_bytes, JukeRetroGPU, 2),
        VMSTATE_UINT32_V(gl_result_type, JukeRetroGPU, 2),
#endif
        VMSTATE_END_OF_LIST()
    },
};

static void jrg_reset(DeviceState *dev)
{
    JukeRetroGPU *s = JUKE_RETRO_GPU(dev);

#ifdef CONFIG_JUKE_RETRO_GL
    s->gl_fault = s->gl_fault_operation = s->gl_fault_sequence = 0;
#endif
    jrg_engine_reset(s);
    vga_common_reset(&s->vga);
}

static void jrg_realize(PCIDevice *dev, Error **errp)
{
    JukeRetroGPU *s = JUKE_RETRO_GPU(dev);

    if (!vga_common_init(&s->vga, OBJECT(dev), errp)) {
        return;
    }
    vga_init(&s->vga, OBJECT(dev), pci_address_space(dev),
             pci_address_space_io(dev), true);
    s->vga.con = qemu_graphic_console_create(DEVICE(dev), 0,
                                            s->vga.hw_ops, &s->vga);
    s->work_bh = qemu_bh_new(jrg_work, s);
#ifdef CONFIG_JUKE_RETRO_GL
    s->gl_bh = qemu_bh_new(jrg_gl_completed_bh, s);
#endif
    pci_register_bar(dev, JRG_VRAM_BAR, PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vga.vram);
    memory_region_init(&s->mmio, OBJECT(dev), "juke-retro-gpu.mmio",
                       JRG_MMIO_SIZE);
    pci_std_vga_mmio_region_init(&s->vga, OBJECT(dev), &s->mmio,
                                 s->vga_regs, true, false);
    memory_region_init_io(&s->regs, OBJECT(dev), &jrg_ops, s,
                          "juke-retro-gpu.commands", 0x1000);
    memory_region_add_subregion(&s->mmio, JRG_REG_MAGIC, &s->regs);
    pci_register_bar(dev, JRG_MMIO_BAR, PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->mmio);
    pci_set_byte(dev->config + PCI_INTERRUPT_PIN, 1);
}

/* Diagnostics are host control state, deliberately not part of VM migration.
 * Enabling starts a fresh bounded window; disabling freezes it for inspection. */
static bool jrg_diagnostics_get(Object *obj, Error **errp)
{
    return JUKE_RETRO_GPU(obj)->diagnostics_enabled;
}

static void jrg_diagnostics_set(Object *obj, bool enabled, Error **errp)
{
    JukeRetroGPU *s = JUKE_RETRO_GPU(obj);
    if (enabled) {
        if (!s->diagnostics) {
            s->diagnostics = g_new0(JrgDiagnostics, 1);
        } else {
            memset(s->diagnostics, 0, sizeof(*s->diagnostics));
        }
        s->diagnostics->start_us = g_get_monotonic_time();
    } else if (s->diagnostics_enabled) {
        s->diagnostics->elapsed_us = g_get_monotonic_time() - s->diagnostics->start_us;
    }
    s->diagnostics_enabled = enabled;
}

static char *jrg_diagnostics_stats(Object *obj, Error **errp)
{
    JukeRetroGPU *s = JUKE_RETRO_GPU(obj);
    const JrgDiagnostics *d = s->diagnostics;
    GString *out = g_string_new(NULL);
    bool comma = false;
    g_string_append_printf(out, "{\"schema\":1,\"enabled\":%s,\"elapsed_us\":%" PRIu64 ",\"mmio\":[",
        s->diagnostics_enabled ? "true" : "false", d ? (s->diagnostics_enabled ?
            (uint64_t)g_get_monotonic_time() - d->start_us : d->elapsed_us) : 0);
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->reads); i++) {
            if (!d->reads[i] && !d->writes[i]) {
                continue;
            }
            g_string_append_printf(out, "%s{\"offset\":%u,\"reads\":%" PRIu64 ",\"writes\":%" PRIu64 "}",
                comma ? "," : "", i * 4, d->reads[i], d->writes[i]);
            comma = true;
        }
    }
    g_string_append_printf(out, "],\"gl\":{\"batches\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"records\":%" PRIu64 ",\"operations\":[",
        d ? d->batches : 0, d ? d->bytes : 0, d ? d->records : 0);
    comma = false;
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->operations); i++) {
            if (d->operations[i]) {
                g_string_append_printf(out, "%s{\"op\":%u,\"count\":%" PRIu64 "}", comma ? "," : "", i, d->operations[i]);
                comma = true;
            }
        }
    }
    g_string_append(out, "],\"functions\":[");
    comma = false;
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->functions); i++) {
            if (d->functions[i]) {
                g_string_append_printf(out, "%s{\"function\":%u,\"count\":%" PRIu64 "}", comma ? "," : "", i, d->functions[i]);
                comma = true;
            }
        }
    }
    g_string_append(out, "],\"desktop\":[");
    comma = false;
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->desktop); i++) {
            if (d->desktop[i]) {
                g_string_append_printf(out, "%s{\"op\":%u,\"count\":%" PRIu64 ",\"rectangle_bytes\":%" PRIu64 "}",
                    comma ? "," : "", i, d->desktop[i], d->desktop_bytes[i]);
                comma = true;
            }
        }
    }
    g_string_append(out, "],\"queries\":[");
    if (d) {
        for (unsigned i = 0; i < d->query_count; i++) {
            const JrgDiagnosticQuery *q = &d->queries[i];
            g_string_append_printf(out, "%s{\"function\":%u,\"arg0\":%u,\"arg1\":%u,\"count\":%" PRIu64 "}",
                i ? "," : "", q->function, q->arg0, q->arg1, q->count);
        }
    }
    g_string_append_printf(out, "],\"query_overflow\":%" PRIu64 ",\"function_overflow\":%" PRIu64 "}}",
        d ? d->query_overflow : 0, d ? d->function_overflow : 0);
    return g_string_free(out, false);
}

static void jrg_instance_init(Object *obj)
{
    object_property_add_bool(obj, "diagnostic-counters", jrg_diagnostics_get, jrg_diagnostics_set);
    object_property_add_str(obj, "diagnostic-stats", jrg_diagnostics_stats, NULL);
}

static const Property jrg_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", JukeRetroGPU, vga.vram_size_mb, 16),
    DEFINE_PROP_STRING("gpu-socket", JukeRetroGPU, gpu_socket),
};

static void jrg_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);
    AcpiDevAmlIfClass *ac = ACPI_DEV_AML_IF_CLASS(klass);

    pc->realize = jrg_realize;
    pc->vendor_id = JRG_PCI_VENDOR_ID;
    pc->device_id = JRG_PCI_DEVICE_ID;
    pc->revision = 1;
    pc->class_id = PCI_CLASS_DISPLAY_VGA;
    pc->romfile = "vgabios-stdvga.bin";
    dc->desc = "Juke VGA/VBE display with native 2D acceleration";
    dc->hotpluggable = false;
    dc->vmsd = &vmstate_jrg;
    device_class_set_props(dc, jrg_properties);
    device_class_set_legacy_reset(dc, jrg_reset);
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    ac->build_dev_aml = build_vga_aml;
}

static void jrg_finalize(Object *obj)
{
    JukeRetroGPU *s = JUKE_RETRO_GPU(obj);

#ifdef CONFIG_JUKE_RETRO_GL
    if (s->gl_bh) {
        qemu_bh_cancel(s->gl_bh);
    }
    s->gl_transfer_active = false;
    jrg_gl_engine_free(s->gl);
    if (s->gl_bh) {
        qemu_bh_delete(s->gl_bh);
    }
    migrate_del_blocker(&s->gl_blocker);
#endif
    if (s->work_bh) {
        qemu_bh_delete(s->work_bh);
    }
    g_free(s->diagnostics);
}

static const TypeInfo jrg_info = {
    .name = TYPE_JUKE_RETRO_GPU,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(JukeRetroGPU),
    .instance_init = jrg_instance_init,
    .instance_finalize = jrg_finalize,
    .class_init = jrg_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { TYPE_ACPI_DEV_AML_IF },
        { },
    },
};

static void jrg_register_types(void)
{
    type_register_static(&jrg_info);
}
type_init(jrg_register_types)
