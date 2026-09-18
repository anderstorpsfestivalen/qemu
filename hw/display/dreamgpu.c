/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * DreamGPU VGA-compatible display with bounded native 2D command execution.
 */
#include "qemu/osdep.h"
#include "qemu/main-loop.h"
#include "qemu/timer.h"
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
#include "standard-headers/dreamgpu/gpu.h"
#include "standard-headers/dreamgpu/gl.h"
#ifdef CONFIG_DREAMGPU_GL
#include "dreamgpu-gl.h"
#include "dreamgpu-gl-platform.h"
#endif
#include "vga_int.h"
#include "ui/dreamgpu-shmem.h"
#include "trace.h"
#include "dreamgpu-diagnostics.h"
#include "dreamgpu-host.h"
#include "dreamgpu-timing.h"
#include "dreamgpu-edid.h"

#define TYPE_DREAMGPU "dreamgpu"
OBJECT_DECLARE_SIMPLE_TYPE(DreamGpu, DREAMGPU)

/* Bound work per main-loop visit so large batches do not monopolize input. */
#define DG_WORK_QUANTUM (256 * 1024)
#define DG_INLINE_QUANTUM (4 * 1024 * 1024)
#define DG_ROW_QUANTUM 2048

struct DreamGpu {
    PCIDevice parent_obj;
    VGACommonState vga;
    MemoryRegion mmio, regs, vga_regs[4];
    uint8_t edid[256];
    QEMUBH *work_bh;
    QEMUTimer *timing_timer;
    uint32_t timing_rate, timing_serial, timing_sequence, timing_completed, timing_status;
    DgTimingSample timing_sample;
    uint32_t addr_lo, addr_hi, count, sequence;
    uint32_t status, completed, error, irq_enable, irq_status, generation;
    uint32_t active_count, active_sequence, command_index, row, column;
    uint64_t trace_start_us, validated_bytes;
    uint32_t trace_chunks;
    bool inline_no_irq; /* Transient, only true inside a negotiated SUBMIT. */
    DreamGpuNativeCursor cursor;
    uint32_t cursor_addr_lo, cursor_addr_hi, cursor_bytes;
    uint32_t cursor_width, cursor_height, cursor_hot_x, cursor_hot_y;
    uint32_t cursor_format, cursor_x, cursor_y, cursor_flags, cursor_sequence;
    uint32_t cursor_status, cursor_completed, cursor_error;
    uint8_t commands[DG_MAX_COMMANDS * DG_COMMAND_BYTES];
    char *gpu_socket;
    bool diagnostics_enabled;
    DgDiagnostics *diagnostics;
#ifdef CONFIG_DREAMGPU_GL
    DgGLEngine *gl;
    QEMUBH *gl_bh;
    DgGLFrameRef gl_present;
    DgGLTransfer gl_transfer;
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

static void dg_update_irq(DreamGpu *s) {
    pci_set_irq(&s->parent_obj, !!(s->irq_enable & s->irq_status));
}

static uint32_t dg_timing_height(DreamGpu *s) {
    uint32_t height = s->vga.vbe_regs[VBE_DISPI_INDEX_ENABLE] & VBE_DISPI_ENABLED
                          ? s->vga.vbe_regs[VBE_DISPI_INDEX_YRES]
                          : qemu_console_get_height(s->vga.con, 480);
    return height && height <= 16384 ? height : 480;
}

static void dg_timing_finish(DreamGpu *s, uint32_t status) {
    timer_del(s->timing_timer);
    if (s->timing_status != DG_TIMING_PENDING) {
        return;
    }
    s->timing_status = status;
    s->timing_completed = s->timing_sequence;
    s->irq_status |= DG_IRQ_DISPLAY_TIMING;
    dg_update_irq(s);
}

static void dg_timing_expired(void *opaque) {
    dg_timing_finish(opaque, DG_TIMING_DONE);
}

static uint32_t dg_timing_read(DreamGpu *s, uint32_t reg) {
    switch (reg) {
        case DG_TIMING_REG_VERSION:
            return DG_TIMING_VERSION;
        case DG_TIMING_REG_RATES:
            return DG_TIMING_RATES;
        case DG_TIMING_REG_RATE:
            return s->timing_rate;
        case DG_TIMING_REG_SNAPSHOT:
            s->timing_sample = dg_timing_sample(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL),
                                                s->timing_rate, dg_timing_height(s));
            if (!++s->timing_serial)
                ++s->timing_serial;
            return s->timing_serial;
        case DG_TIMING_REG_SERIAL:
            return s->timing_serial;
        case DG_TIMING_REG_SCANLINE:
            return s->timing_sample.line;
        case DG_TIMING_REG_HEIGHT:
            return s->timing_sample.height;
        case DG_TIMING_REG_PHASE:
            return s->timing_sample.phase;
        case DG_TIMING_REG_BEGIN_NS:
            return s->timing_sample.begin_ns;
        case DG_TIMING_REG_END_NS:
            return s->timing_sample.end_ns;
        case DG_TIMING_REG_SEQUENCE:
            return s->timing_sequence;
        case DG_TIMING_REG_COMPLETED:
            return s->timing_completed;
        case DG_TIMING_REG_STATUS:
            return s->timing_status;
        default:
            return 0;
    }
}

static void dg_timing_write(DreamGpu *s, uint32_t reg, uint32_t value) {
    switch (reg) {
        case DG_TIMING_REG_RATE:
            if (dg_timing_rate_valid(value)) {
                /* Mode commits also cancel a waiter when only dimensions changed. */
                dg_timing_finish(s, DG_TIMING_CANCELLED);
                s->timing_rate = value;
            }
            break;
        case DG_TIMING_REG_SEQUENCE:
            if (s->timing_status != DG_TIMING_PENDING)
                s->timing_sequence = value;
            break;
        case DG_TIMING_REG_COMMAND:
            if (value == DG_TIMING_CANCEL) {
                dg_timing_finish(s, DG_TIMING_CANCELLED);
            } else if ((value == DG_TIMING_WAIT_BEGIN || value == DG_TIMING_WAIT_END) &&
                       s->timing_status != DG_TIMING_PENDING && s->timing_sequence) {
                uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
                DgTimingSample sample = dg_timing_sample(now, s->timing_rate, dg_timing_height(s));
                s->timing_status = DG_TIMING_PENDING;
                s->irq_status &= ~DG_IRQ_DISPLAY_TIMING;
                dg_update_irq(s);
                timer_mod(s->timing_timer,
                          now + (value == DG_TIMING_WAIT_BEGIN ? sample.begin_ns : sample.end_ns));
            }
            break;
    }
}

static void dg_complete(DreamGpu *s, uint32_t error) {
    if (s->trace_start_us) {
        uint64_t elapsed = g_get_monotonic_time() - s->trace_start_us;
        trace_dreamgpu_gpu_complete(s->active_sequence, error, s->trace_chunks, elapsed);
    }
    s->error = error;
    s->status = DG_STATUS_DONE | (error ? DG_STATUS_ERROR : 0);
    s->completed = s->active_sequence;
    s->active_count = s->command_index = s->row = s->column = 0;
    if (!s->inline_no_irq) {
        s->irq_status |= DG_IRQ_COMPLETION;
    }
    dg_update_irq(s);
}

/* QEMU owns RAM translation and DMA; Rust owns snapshot transactions. */
static uint8_t *dg_snapshot_allocate(void *opaque, uint32_t bytes) {
    return g_try_malloc(bytes);
}
static void dg_snapshot_free(void *opaque, uint8_t *data) {
    g_free(data);
}
static uint32_t dg_snapshot_read(void *opaque, uint64_t address, uint8_t *data, uint32_t bytes) {
    DreamGpu *s = opaque;
    return pci_dma_read(&s->parent_obj, address, data, bytes) == MEMTX_OK;
}
static uint32_t dg_snapshot_ram(void *opaque, uint64_t address, uint32_t bytes, uint32_t write) {
    DreamGpu *s = opaque;
    hwaddr translated, length = bytes;
    MemoryRegion *mr;
    RCU_READ_LOCK_GUARD();
    mr = address_space_translate(pci_get_address_space(&s->parent_obj), address, &translated,
                                 &length, write, MEMTXATTRS_UNSPECIFIED);
    return length >= bytes && memory_region_is_ram(mr) && !memory_region_is_rom(mr);
}
static DreamGpuDeviceMemory dg_snapshot_memory(DreamGpu *s) {
    return (DreamGpuDeviceMemory){
        s, dg_snapshot_allocate, dg_snapshot_free, dg_snapshot_read, dg_snapshot_ram,
    };
}
static void dg_cursor_submit(DreamGpu *s, uint32_t operation) {
    DreamGpuDeviceMemory memory = dg_snapshot_memory(s);
    DreamGpuCursorRequest request = {
        .address = ((uint64_t)s->cursor_addr_hi << 32) | s->cursor_addr_lo,
        .operation = operation,
        .flags = s->cursor_flags,
        .bytes = s->cursor_bytes,
        .width = s->cursor_width,
        .height = s->cursor_height,
        .hot_x = s->cursor_hot_x,
        .hot_y = s->cursor_hot_y,
        .format = s->cursor_format,
    };
    uint32_t error = dreamgpu_device_cursor(&memory, &request, s->cursor.pixels);
    if (!error) {
        if (operation == DG_CURSOR_SHAPE) {
            s->cursor.width = s->cursor_width;
            s->cursor.height = s->cursor_height;
            s->cursor.hot_x = s->cursor_hot_x;
            s->cursor.hot_y = s->cursor_hot_y;
            s->cursor.format = s->cursor_format;
        }
        s->cursor.x = s->cursor_x;
        s->cursor.y = s->cursor_y;
        s->cursor.flags = s->cursor_flags;
        dreamgpu_shmem_native_cursor(s->vga.con, &s->cursor, operation == DG_CURSOR_SHAPE);
    }
    s->cursor_status = DG_STATUS_DONE | (error ? DG_STATUS_ERROR : 0);
    s->cursor_error = error;
    s->cursor_completed = s->cursor_sequence;
}

static uint32_t dg_cursor_read(DreamGpu *s, uint32_t reg) {
    switch (reg) {
        case DG_CURSOR_REG_VERSION:
            return DG_CURSOR_ABI_VERSION;
        case DG_CURSOR_REG_ADDR_LO:
            return s->cursor_addr_lo;
        case DG_CURSOR_REG_ADDR_HI:
            return s->cursor_addr_hi;
        case DG_CURSOR_REG_BYTES:
            return s->cursor_bytes;
        case DG_CURSOR_REG_WIDTH:
            return s->cursor_width;
        case DG_CURSOR_REG_HEIGHT:
            return s->cursor_height;
        case DG_CURSOR_REG_HOT_X:
            return s->cursor_hot_x;
        case DG_CURSOR_REG_HOT_Y:
            return s->cursor_hot_y;
        case DG_CURSOR_REG_FORMAT:
            return s->cursor_format;
        case DG_CURSOR_REG_X:
            return s->cursor_x;
        case DG_CURSOR_REG_Y:
            return s->cursor_y;
        case DG_CURSOR_REG_FLAGS:
            return s->cursor_flags;
        case DG_CURSOR_REG_SEQUENCE:
            return s->cursor_sequence;
        case DG_CURSOR_REG_STATUS:
            return s->cursor_status;
        case DG_CURSOR_REG_COMPLETED:
            return s->cursor_completed;
        case DG_CURSOR_REG_ERROR:
            return s->cursor_error;
        case DG_CURSOR_REG_MAX_DIMENSION:
            return DG_CURSOR_MAX_DIMENSION;
        default:
            return 0;
    }
}

static void dg_cursor_write(DreamGpu *s, uint32_t reg, uint32_t value) {
    switch (reg) {
        case DG_CURSOR_REG_ADDR_LO:
            s->cursor_addr_lo = value;
            break;
        case DG_CURSOR_REG_ADDR_HI:
            s->cursor_addr_hi = value;
            break;
        case DG_CURSOR_REG_BYTES:
            s->cursor_bytes = value;
            break;
        case DG_CURSOR_REG_WIDTH:
            s->cursor_width = value;
            break;
        case DG_CURSOR_REG_HEIGHT:
            s->cursor_height = value;
            break;
        case DG_CURSOR_REG_HOT_X:
            s->cursor_hot_x = value;
            break;
        case DG_CURSOR_REG_HOT_Y:
            s->cursor_hot_y = value;
            break;
        case DG_CURSOR_REG_FORMAT:
            s->cursor_format = value;
            break;
        case DG_CURSOR_REG_X:
            s->cursor_x = value;
            break;
        case DG_CURSOR_REG_Y:
            s->cursor_y = value;
            break;
        case DG_CURSOR_REG_FLAGS:
            s->cursor_flags = value;
            break;
        case DG_CURSOR_REG_SEQUENCE:
            s->cursor_sequence = value;
            break;
        case DG_CURSOR_REG_SUBMIT:
            dg_cursor_submit(s, value);
            break;
    }
}

static void dg_cursor_reset(DreamGpu *s) {
    s->cursor_addr_lo = s->cursor_addr_hi = s->cursor_bytes = 0;
    s->cursor_width = s->cursor_height = s->cursor_hot_x = s->cursor_hot_y = 0;
    s->cursor_format = s->cursor_x = s->cursor_y = s->cursor_flags = 0;
    s->cursor_sequence = s->cursor_status = s->cursor_completed = 0;
    s->cursor_error = 0;
    memset(&s->cursor, 0, sizeof(s->cursor));
    dreamgpu_shmem_native_cursor(s->vga.con, &s->cursor, true);
}

#ifdef CONFIG_DREAMGPU_GL
static void dg_fault_stop(DreamGpu *s, uint32_t reason, uint32_t sequence) {
    if (!reason || reason > DG_GL_FAULT_DRIVER_INTERNAL || s->gl_fault) {
        return;
    }
    s->gl_fault = reason;
    s->gl_fault_operation = s->gl_sensitive_op;
    s->gl_fault_sequence = sequence;
    g_autofree char *path = object_get_canonical_path(OBJECT(s));
    error_report("DreamGPU graphics coherence fault %u "
                 "(operation %u, sequence %u); "
                 "guest stopped, system reset required",
                 reason, s->gl_fault_operation, s->gl_fault_sequence);
    qapi_event_send_dreamgpu_fault(path, reason, s->gl_fault_operation, s->gl_fault_sequence);
    vm_stop(RUN_STATE_INTERNAL_ERROR);
}

static bool dg_result_ram(DreamGpu *s, uint64_t address, uint32_t bytes) {
    hwaddr translated, length = bytes;
    MemoryRegion *mr;

    if (!bytes || bytes > DG_GL_MAX_READBACK_BYTES || address > UINT64_MAX - bytes) {
        return false;
    }
    RCU_READ_LOCK_GUARD();
    mr = address_space_translate(pci_get_address_space(&s->parent_obj), address, &translated,
                                 &length, true, MEMTXATTRS_UNSPECIFIED);
    return length >= bytes && memory_region_is_ram(mr) && !memory_region_is_rom(mr);
}

static void dg_gl_complete(DreamGpu *s, uint32_t sequence, uint32_t error) {
    if (s->gl_trace_start_us) {
        trace_dreamgpu_gl_complete(sequence, error, g_get_monotonic_time() - s->gl_trace_start_us);
        s->gl_trace_start_us = 0;
    }
    if (error && s->gl_sensitive_op) {
        dg_fault_stop(s, DG_GL_FAULT_HOST_COHERENCE, sequence);
    }
    s->gl_status = DG_STATUS_DONE | (error ? DG_STATUS_ERROR : 0);
    s->gl_completed = sequence;
    s->gl_error = error;
    s->irq_status |= DG_IRQ_GL_COMPLETION;
    dg_update_irq(s);
}

static void dg_gl_transfer_work(DreamGpu *s) {
    DgGLTransfer *t = &s->gl_transfer;
    uint32_t budget = DG_WORK_QUANTUM;
    uint32_t chunks = 0, error = 0;
    uint64_t cpu_epoch = 0, cpu_generation = 0;

    if (!s->gl_transfer_active && dg_gl_engine_transfer(s->gl, t)) {
        s->gl_transfer_active = true;
        s->gl_transfer_row = s->gl_transfer_column = 0;
    }
    if (!s->gl_transfer_active) {
        return;
    }
    if (t->generation != s->generation) {
        error = DG_GL_ERROR_GENERATION;
    }
    if (t->primary_bpp != s->vga.vbe_regs[VBE_DISPI_INDEX_BPP]) {
        error = DG_GL_ERROR_DESKTOP;
    }
    while (!error && budget && s->gl_transfer_row < t->height) {
        uint32_t row = s->gl_transfer_row;
        uint32_t column = s->gl_transfer_column;
        uint32_t count = MIN(budget, t->width * 4 - column);
        bool packed16 = t->primary_bpp == 16;
        uint32_t primary_count = packed16 ? count / 2 : count;
        uint64_t offset =
            (uint64_t)t->offset + (uint64_t)row * t->vram_stride + (packed16 ? column / 2 : column);
        uint8_t *pixels = t->pixels + (size_t)row * t->stride + column;
        if (offset + primary_count > s->vga.vram_size) {
            error = DG_GL_ERROR_DESKTOP;
            break;
        }
        if (packed16) {
            dreamgpu_primary16_transfer(pixels, s->vga.vram_ptr + offset, count / 4, t->writeback);
            if (t->writeback) {
                memory_region_set_dirty(&s->vga.vram, offset, primary_count);
            }
        } else if (t->writeback) {
            memcpy(s->vga.vram_ptr + offset, pixels, count);
            memory_region_set_dirty(&s->vga.vram, offset, count);
        } else {
            memcpy(pixels, s->vga.vram_ptr + offset, count);
        }
        budget -= count;
        if (++chunks == DG_ROW_QUANTUM) {
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
        !dreamgpu_shmem_cpu_anchor(s->vga.con, &cpu_epoch, &cpu_generation)) {
        error = DG_GL_ERROR_DESKTOP;
    }
    s->gl_transfer_active = false;
    dg_gl_engine_transfer_done(s->gl, error, cpu_epoch, cpu_generation);
}

static void dg_gl_completed_bh(void *opaque) {
    DreamGpu *s = opaque;
    DgGLCompletion done;

    if (s->gl) {
        dg_gl_transfer_work(s);
    }
    while (s->gl && dg_gl_engine_completion(s->gl, &done)) {
        if (done.generation != s->generation) {
            g_free(done.bulk_result);
            continue;
        }
        if (!done.reset) {
            s->gl_present = done.present;
            if (!done.error && done.result_bytes) {
                if (done.result_bytes > s->gl_active_result_capacity ||
                    !dg_result_ram(s, s->gl_active_result_address, done.result_bytes) ||
                    pci_dma_write(&s->parent_obj, s->gl_active_result_address,
                                  done.bulk_result ? done.bulk_result : done.result,
                                  done.result_bytes) != MEMTX_OK) {
                    done.error = DG_GL_ERROR_DMA;
                } else {
                    s->gl_result_bytes = done.result_bytes;
                    s->gl_result_type = done.result_type;
                }
            }
            dg_gl_complete(s, done.sequence, done.error);
        }
        if (!done.resources_live && !(s->gl_status & DG_STATUS_BUSY)) {
            migrate_del_blocker(&s->gl_blocker);
        }
        g_free(done.bulk_result);
    }
}

static void dg_gl_notify(void *opaque) {
    DreamGpu *s = opaque;

    qemu_bh_schedule(s->gl_bh);
}

static void dg_gl_dimensions(DreamGpu *s, uint32_t *width, uint32_t *height) {
    *width = *height = 0;
    if (s->vga.vbe_regs[VBE_DISPI_INDEX_ENABLE] & VBE_DISPI_ENABLED) {
        *width = s->vga.vbe_regs[VBE_DISPI_INDEX_XRES];
        *height = s->vga.vbe_regs[VBE_DISPI_INDEX_YRES];
    }
}
static uint32_t dg_snapshot_gl_validate(void *opaque, const uint8_t *data, uint32_t bytes,
                                        uint32_t *records) {
    DreamGpu *s = opaque;
    uint32_t width, height;
    dg_gl_dimensions(s, &width, &height);
    return dg_gl_validate(data, bytes, s->generation, width, height, s->vga.vram_size, records);
}
static void dg_snapshot_diagnostics_begin(void *opaque, uint32_t bytes) {
    DreamGpu *s = opaque;
    if (s->diagnostics_enabled) {
        s->diagnostics->batches++;
        s->diagnostics->bytes += bytes;
    }
}
static void dg_snapshot_diagnostic_record(void *opaque, const uint8_t *record) {
    DreamGpu *s = opaque;
    if (s->diagnostics_enabled) {
        dg_diagnostic_record(s->diagnostics, record);
    }
}
static uint32_t dg_snapshot_enqueue(void *opaque, uint8_t *data, uint32_t bytes, uint32_t records) {
    DreamGpu *s = opaque;
    uint32_t width, height;
    Error *err = NULL;
    if (!s->gpu_socket || !*s->gpu_socket) {
        return DG_GL_ERROR_TRANSPORT;
    }
    if (!s->gl_blocker) {
        error_setg(&s->gl_blocker, "DreamGPU active OpenGL resources cannot be migrated or saved");
        if (migrate_add_blocker(&s->gl_blocker, &err) < 0) {
            error_report_err(err);
            return DG_GL_ERROR_HOST;
        }
    }
    if (!s->gl) {
        s->gl = dg_gl_engine_new(s->gpu_socket, dg_gl_notify, s);
    }
    dg_gl_dimensions(s, &width, &height);
    return dg_gl_engine_submit(s->gl, data, bytes, s->gl_sequence, s->generation, width, height,
                               records)
               ? 0
               : DG_GL_ERROR_LIMIT;
}
static void dg_gl_submit(DreamGpu *s) {
    DreamGpuDeviceGlCallbacks callbacks = {
        .memory = dg_snapshot_memory(s),
        .validate = dg_snapshot_gl_validate,
        .diagnostics_begin = dg_snapshot_diagnostics_begin,
        .diagnostic_record = dg_snapshot_diagnostic_record,
        .enqueue = dg_snapshot_enqueue,
    };
    DreamGpuDeviceGlRequest request = {
        .address = ((uint64_t)s->gl_addr_hi << 32) | s->gl_addr_lo,
        .result_address = ((uint64_t)s->gl_result_addr_hi << 32) | s->gl_result_addr_lo,
        .bytes = s->gl_bytes,
        .generation = s->gl_generation,
        .current_generation = s->generation,
        .bpp = s->vga.vbe_regs[VBE_DISPI_INDEX_BPP],
        .busy_2d = s->status & DG_STATUS_BUSY,
        .result_capacity = s->gl_result_capacity,
    };
    DreamGpuDeviceGlResult result;
    uint32_t error;
    if ((s->gl_status & DG_STATUS_BUSY) || s->gl_fault) {
        return;
    }
    s->gl_trace_start_us = trace_event_get_state_backends(TRACE_DREAMGPU_GL_SUBMIT) ||
                                   trace_event_get_state_backends(TRACE_DREAMGPU_GL_COMPLETE)
                               ? g_get_monotonic_time()
                               : 0;
    s->gl_sensitive_op = 0;
    s->gl_result_bytes = s->gl_result_type = 0;
    s->gl_active_result_capacity = 0;
    error = dreamgpu_device_gl_submit(&callbacks, &request, &result);
    s->gl_sensitive_op = result.sensitive;
    s->gl_active_result_capacity = result.result_capacity;
    if (result.result_capacity) {
        s->gl_active_result_address = request.result_address;
    }
    if (error) {
        dg_gl_complete(s, s->gl_sequence, error);
        return;
    }
    s->gl_status = DG_STATUS_BUSY;
    s->gl_error = 0;
    if (s->gl_trace_start_us) {
        trace_dreamgpu_gl_submit(s->gl_sequence, result.records, s->gl_bytes,
                                 g_get_monotonic_time() - s->gl_trace_start_us);
    }
}

static uint64_t dg_gl_read(DreamGpu *s, hwaddr addr) {
    switch (addr) {
        case DG_GL_REG_FAULT_STOP:
            return s->gl_fault;
        case DG_GL_REG_FAULT_OPERATION:
            return s->gl_fault_operation;
        case DG_GL_REG_FAULT_SEQUENCE:
            return s->gl_fault_sequence;
        case DG_GL_REG_RESULT_ADDR_LO:
            return s->gl_result_addr_lo;
        case DG_GL_REG_RESULT_ADDR_HI:
            return s->gl_result_addr_hi;
        case DG_GL_REG_RESULT_CAPACITY:
            return s->gl_result_capacity;
        case DG_GL_REG_RESULT_BYTES:
            return s->gl_result_bytes;
        case DG_GL_REG_RESULT_TYPE:
            return s->gl_result_type;
        case DG_GL_REG_PRESENT_SLOT:
            return s->gl_present.slot;
        case DG_GL_REG_PRESENT_EPOCH_LO:
            return s->gl_present.epoch;
        case DG_GL_REG_PRESENT_EPOCH_HI:
            return s->gl_present.epoch >> 32;
        case DG_GL_REG_PRESENT_FRAME_LO:
            return s->gl_present.generation;
        case DG_GL_REG_PRESENT_FRAME_HI:
            return s->gl_present.generation >> 32;
        case DG_GL_REG_PRESENT_CLIENT:
            return s->gl_present.client;
        case DG_GL_REG_PRESENT_DRAWABLE:
            return s->gl_present.drawable;
        case DG_GL_REG_VERSION:
            return DG_GL_VERSION;
        case DG_GL_REG_ADDR_LO:
            return s->gl_addr_lo;
        case DG_GL_REG_ADDR_HI:
            return s->gl_addr_hi;
        case DG_GL_REG_BYTES:
            return s->gl_bytes;
        case DG_GL_REG_SEQUENCE:
            return s->gl_sequence;
        case DG_GL_REG_GENERATION:
            return s->generation;
        case DG_GL_REG_STATUS:
            return s->gl_status;
        case DG_GL_REG_COMPLETED:
            return s->gl_completed;
        case DG_GL_REG_ERROR:
            return s->gl_error;
        case DG_GL_REG_MAX_BYTES:
            return DG_GL_MAX_BYTES;
        case DG_GL_REG_MAX_RECORDS:
            return DG_GL_MAX_RECORDS;
        case DG_GL_REG_QUERY_FUNCTION:
            return s->gl_query_function;
        case DG_GL_REG_FUNCTION_WORDS:
            return dg_gl_function_words(s->gl_query_function);
        default:
            return 0;
    }
}

static void dg_gl_write(DreamGpu *s, hwaddr addr, uint32_t value) {
    switch (addr) {
        case DG_GL_REG_FAULT_STOP:
            dg_fault_stop(s, value, s->gl_sequence);
            break;
        case DG_GL_REG_RESULT_ADDR_LO:
            s->gl_result_addr_lo = value;
            break;
        case DG_GL_REG_RESULT_ADDR_HI:
            s->gl_result_addr_hi = value;
            break;
        case DG_GL_REG_RESULT_CAPACITY:
            s->gl_result_capacity = value;
            break;
        case DG_GL_REG_ADDR_LO:
            s->gl_addr_lo = value;
            break;
        case DG_GL_REG_ADDR_HI:
            s->gl_addr_hi = value;
            break;
        case DG_GL_REG_BYTES:
            s->gl_bytes = value;
            break;
        case DG_GL_REG_SEQUENCE:
            s->gl_sequence = value;
            break;
        case DG_GL_REG_GENERATION:
            s->gl_generation = value;
            break;
        case DG_GL_REG_QUERY_FUNCTION:
            s->gl_query_function = value;
            break;
        case DG_GL_REG_SUBMIT:
            if (value == 1) {
                dg_gl_submit(s);
            }
            break;
    }
}
#endif

/* The host Rust engine validates the complete immutable DMA snapshot. */

static void dg_transfer(void *opaque, uint32_t op, uint32_t bpp, uint64_t src, uint64_t dst,
                        uint32_t bytes, uint32_t color) {
    DreamGpu *s = opaque;
    uint8_t *out = s->vga.vram_ptr + dst;

    /* Guest CPU threads can access VRAM outside BQL. Keep RAM access in QEMU:
     * Rust receives integer bounds only and never creates guest RAM references. */
    if (op == DG_CMD_COPY) {
        memmove(out, s->vga.vram_ptr + src, bytes);
    } else if (op == DG_CMD_FILL) {
        if (bpp == 1) {
            memset(out, color, bytes);
        } else if (bpp == 2) {
            /* Constant strides let the compiler vectorize validated fills.
             * Keep unaligned little-endian stores and the original work bound. */
            for (uint32_t x = 0; x < bytes; x += 2) {
                stw_le_p(out + x, color);
            }
        } else {
            /* The immutable batch validator accepts only 1, 2 or 4-byte pixels. */
            for (uint32_t x = 0; x < bytes; x += 4) {
                stl_le_p(out + x, color);
            }
        }
    }
    memory_region_set_dirty(&s->vga.vram, dst, bytes);
}

static void dg_run(DreamGpu *s, uint32_t budget) {
    DreamGpuProgress progress = {
        s->command_index,
        s->row,
        s->column,
    };
    DreamGpuWork work;
    uint32_t error;
    bool tracing = trace_event_get_state_backends(TRACE_DREAMGPU_GPU_WORK);
    uint64_t start_us = tracing ? g_get_monotonic_time() : 0;

    if (s->trace_start_us) {
        s->trace_chunks++;
    }
    /* The BQL remains held; Rust never stores VRAM pointers or invokes QEMU
     * asynchronously. Preserve the existing migratable progress fields. */
    error = dreamgpu_2d_execute(s->commands, s->active_count, s->vga.vram_size, &progress, budget,
                                dg_transfer, s, &work);
    if (error) {
        dg_complete(s, error);
        return;
    }
    s->command_index = progress.command;
    s->row = progress.row;
    s->column = progress.column;
    if (tracing) {
        trace_dreamgpu_gpu_work(s->active_sequence, work.bytes, work.chunks,
                                g_get_monotonic_time() - start_us, budget == DG_INLINE_QUANTUM);
    }
    if (s->command_index == s->active_count) {
        dg_complete(s, DG_ERROR_NONE);
    } else {
        qemu_bh_schedule(s->work_bh);
    }
}

static void dg_work(void *opaque) {
    dg_run(opaque, DG_WORK_QUANTUM);
}

static void dg_submit(DreamGpu *s) {
    uint64_t addr = ((uint64_t)s->addr_hi << 32) | s->addr_lo;
    DreamGpuDeviceMemory memory = dg_snapshot_memory(s);
    uint32_t error;

    /* A producer owns the channel until completion; never replace its work. */
    if (s->status & DG_STATUS_BUSY) {
        return;
    }
    s->trace_start_us =
        trace_event_get_state_backends(TRACE_DREAMGPU_GPU_COMPLETE) ? g_get_monotonic_time() : 0;
    s->trace_chunks = 0;
    s->validated_bytes = 0;
    s->active_sequence = s->sequence;
    s->active_count = s->count;
    s->command_index = s->row = s->column = 0;
    s->error = DG_ERROR_NONE;
    error = dreamgpu_device_2d_capture(&memory, addr, s->count, s->commands, s->vga.vram_size,
                                       &s->validated_bytes);
    if (error) {
        dg_complete(s, error);
        return;
    }
    trace_dreamgpu_gpu_submit(s->active_sequence, s->active_count, s->validated_bytes,
                              DG_INLINE_QUANTUM);
    s->status = DG_STATUS_BUSY;
    /*
     * A full 1024x768x32 GDI blit fits in one bounded MMIO exit. Larger
     * operations continue in small BH quanta; narrow/tall rectangles are
     * separately bounded by the row limit to cap dirty-logging overhead.
     */
    dg_run(s, DG_INLINE_QUANTUM);
}

static void dg_engine_reset(DreamGpu *s) {
    qemu_bh_cancel(s->work_bh);
    timer_del(s->timing_timer);
    s->timing_rate = DG_TIMING_DEFAULT_HZ;
    s->timing_status = DG_TIMING_IDLE;
    s->timing_sequence = s->timing_completed = s->timing_serial = 0;
    s->trace_start_us = 0;
    s->addr_lo = s->addr_hi = s->count = s->sequence = 0;
    s->status = s->completed = s->error = 0;
    s->irq_enable = s->irq_status = 0;
    s->active_count = s->active_sequence = s->command_index = 0;
    s->row = s->column = 0;
    memset(s->commands, 0, sizeof(s->commands));
    s->generation++;
    dg_cursor_reset(s);
#ifdef CONFIG_DREAMGPU_GL
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
        dreamgpu_shmem_cpu_anchor(s->vga.con, &epoch, &generation);
        dg_gl_engine_reset(s->gl, s->generation, epoch, generation);
    }
#endif
    dg_update_irq(s);
}

static uint64_t dg_read(void *opaque, hwaddr addr, unsigned size) {
    DreamGpu *s = opaque;
    if (s->diagnostics_enabled && addr + DG_REG_MAGIC < DG_MMIO_SIZE) {
        s->diagnostics->reads[(addr + DG_REG_MAGIC) / 4]++;
    }

    if (addr + DG_REG_MAGIC >= DG_TIMING_REG_VERSION &&
        addr + DG_REG_MAGIC <= DG_TIMING_REG_STATUS) {
        return dg_timing_read(s, addr + DG_REG_MAGIC);
    }
    if (addr + DG_REG_MAGIC >= DG_CURSOR_REG_VERSION &&
        addr + DG_REG_MAGIC <= DG_CURSOR_REG_MAX_DIMENSION) {
        return dg_cursor_read(s, addr + DG_REG_MAGIC);
    }
#ifdef CONFIG_DREAMGPU_GL
    if (addr + DG_REG_MAGIC >= DG_GL_REG_VERSION) {
        return dg_gl_read(s, addr + DG_REG_MAGIC);
    }
#endif

    switch (addr + DG_REG_MAGIC) {
        case DG_REG_MAGIC:
            return DG_MAGIC;
        case DG_REG_VERSION:
            return DG_ABI_VERSION;
        case DG_REG_CAPS:
            return DG_CAP_FILL | DG_CAP_COPY | DG_CAP_DAMAGE | DG_CAP_COMPLETION_IRQ |
                   DG_CAP_INLINE_NO_IRQ | DG_CAP_CURSOR | DG_CAP_DISPLAY_TIMING
#ifdef CONFIG_DREAMGPU_GL
                   | (s->gpu_socket && *s->gpu_socket
                          ? DG_CAP_GL_TRANSPORT | DG_CAP_GL_FRONT_BUFFERS |
                                DG_CAP_GL_PRESENT_BOUNDS | DG_CAP_GL_BULK_READBACK
                          : 0)
#endif
                ;
        case DG_REG_VRAM_SIZE:
            return s->vga.vram_size;
        case DG_REG_BATCH_ADDR_LO:
            return s->addr_lo;
        case DG_REG_BATCH_ADDR_HI:
            return s->addr_hi;
        case DG_REG_BATCH_COUNT:
            return s->count;
        case DG_REG_SUBMIT_SEQUENCE:
            return s->sequence;
        case DG_REG_STATUS:
            return s->status;
        case DG_REG_COMPLETED_SEQUENCE:
            return s->completed;
        case DG_REG_ERROR:
            return s->error;
        case DG_REG_IRQ_ENABLE:
            return s->irq_enable;
        case DG_REG_IRQ_STATUS:
            return s->irq_status;
        case DG_REG_GENERATION:
            return s->generation;
        case DG_REG_MAX_COMMANDS:
            return DG_MAX_COMMANDS;
        case DG_REG_MAX_WORK_BYTES:
            return DG_MAX_WORK_BYTES;
        default:
            return 0;
    }
}

static void dg_write(void *opaque, hwaddr addr, uint64_t value, unsigned size) {
    DreamGpu *s = opaque;
    if (s->diagnostics_enabled && addr + DG_REG_MAGIC < DG_MMIO_SIZE) {
        s->diagnostics->writes[(addr + DG_REG_MAGIC) / 4]++;
    }

    if (addr + DG_REG_MAGIC >= DG_TIMING_REG_VERSION &&
        addr + DG_REG_MAGIC <= DG_TIMING_REG_STATUS) {
        dg_timing_write(s, addr + DG_REG_MAGIC, value);
        return;
    }
    if (addr + DG_REG_MAGIC >= DG_CURSOR_REG_VERSION &&
        addr + DG_REG_MAGIC <= DG_CURSOR_REG_MAX_DIMENSION) {
        dg_cursor_write(s, addr + DG_REG_MAGIC, value);
        return;
    }
#ifdef CONFIG_DREAMGPU_GL
    if (addr + DG_REG_MAGIC >= DG_GL_REG_VERSION) {
        dg_gl_write(s, addr + DG_REG_MAGIC, value);
        return;
    }
#endif

    switch (addr + DG_REG_MAGIC) {
        case DG_REG_BATCH_ADDR_LO:
            s->addr_lo = value;
            break;
        case DG_REG_BATCH_ADDR_HI:
            s->addr_hi = value;
            break;
        case DG_REG_BATCH_COUNT:
            s->count = value;
            break;
        case DG_REG_SUBMIT_SEQUENCE:
            s->sequence = value;
            break;
        case DG_REG_SUBMIT:
            if (value == DG_SUBMIT_START || value == (DG_SUBMIT_START | DG_SUBMIT_INLINE_NO_IRQ)) {
                /* Only this MMIO callback can suppress its own completion IRQ. */
                s->inline_no_irq = value & DG_SUBMIT_INLINE_NO_IRQ;
                dg_submit(s);
                s->inline_no_irq = false;
            }
            break;
        case DG_REG_IRQ_ENABLE:
            s->irq_enable =
                value & (DG_IRQ_COMPLETION | DG_IRQ_GL_COMPLETION | DG_IRQ_DISPLAY_TIMING);
            dg_update_irq(s);
            break;
        case DG_REG_IRQ_STATUS:
            s->irq_status &=
                ~(value & (DG_IRQ_COMPLETION | DG_IRQ_GL_COMPLETION | DG_IRQ_DISPLAY_TIMING));
            dg_update_irq(s);
            break;
        case DG_REG_RESET:
            if (value == 1) {
#ifdef CONFIG_DREAMGPU_GL
                if (s->gl_fault) {
                    break;
                }
#endif
                dg_engine_reset(s);
            }
            break;
    }
}

static const MemoryRegionOps dg_ops = {
    .read = dg_read,
    .write = dg_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {.min_access_size = 4, .max_access_size = 4},
    .impl = {.min_access_size = 4, .max_access_size = 4},
};

static int dg_post_load(void *opaque, int version_id) {
    DreamGpu *s = opaque;

    if (!dg_timing_rate_valid(s->timing_rate))
        return -EINVAL;
    /* An in-flight wait is cancelled on migration, never silently lost. */
    if (s->timing_status == DG_TIMING_PENDING)
        dg_timing_finish(s, DG_TIMING_CANCELLED);
    DreamGpuDeviceRestore restored = {
        .cursor =
            {
                .flags = s->cursor.flags,
                .width = s->cursor.width,
                .height = s->cursor.height,
                .hot_x = s->cursor.hot_x,
                .hot_y = s->cursor.hot_y,
                .format = s->cursor.format,
            },
        .cursor_status = s->cursor_status,
#ifdef CONFIG_DREAMGPU_GL
        .gl_status = s->gl_status,
#endif
        .status = s->status,
        .count = s->active_count,
        .command = s->command_index,
        .row = s->row,
        .column = s->column,
        .vram = s->vga.vram_size,
    };
    if (!dreamgpu_device_restore(&restored, s->cursor.pixels, s->commands, &s->validated_bytes)) {
        return -EINVAL;
    }
    if (s->status & DG_STATUS_BUSY) {
        qemu_bh_schedule(s->work_bh);
    }
    dg_update_irq(s);
    dreamgpu_shmem_native_cursor(s->vga.con, &s->cursor, true);
    return 0;
}

static int dg_pre_load(void *opaque) {
    DreamGpu *s = opaque;

    qemu_bh_cancel(s->work_bh);
    s->trace_start_us = 0;
    dg_cursor_reset(s);
#ifdef CONFIG_DREAMGPU_GL
    /*
     * Restoring a pre-3D checkpoint discards the current graphics session.
     * This explicit VM restore may wait for teardown; normal MMIO never does.
     */
    qemu_bh_cancel(s->gl_bh);
    s->gl_transfer_active = false;
    dg_gl_engine_free(s->gl);
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

static const VMStateDescription vmstate_dg_cursor = {
    .name = "dreamgpu-cursor",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields =
        (const VMStateField[]){
            VMSTATE_UINT32(width, DreamGpuNativeCursor),
            VMSTATE_UINT32(height, DreamGpuNativeCursor),
            VMSTATE_UINT32(format, DreamGpuNativeCursor),
            VMSTATE_INT32(hot_x, DreamGpuNativeCursor), VMSTATE_INT32(hot_y, DreamGpuNativeCursor),
            VMSTATE_INT32(x, DreamGpuNativeCursor), VMSTATE_INT32(y, DreamGpuNativeCursor),
            VMSTATE_UINT32(flags, DreamGpuNativeCursor),
            VMSTATE_UINT8_ARRAY(pixels, DreamGpuNativeCursor, DG_CURSOR_MAX_BYTES),
            VMSTATE_END_OF_LIST()},
};

static const VMStateDescription vmstate_dg = {
    .name = TYPE_DREAMGPU,
    .version_id = 4,
    .minimum_version_id = 1,
    .pre_load = dg_pre_load,
    .post_load = dg_post_load,
    .fields =
        (const VMStateField[]){
            VMSTATE_PCI_DEVICE(parent_obj, DreamGpu),
            VMSTATE_STRUCT(vga, DreamGpu, 0, vmstate_vga_common, VGACommonState),
            VMSTATE_UINT32(addr_lo, DreamGpu),
            VMSTATE_UINT32(addr_hi, DreamGpu),
            VMSTATE_UINT32(count, DreamGpu),
            VMSTATE_UINT32(sequence, DreamGpu),
            VMSTATE_UINT32(status, DreamGpu),
            VMSTATE_UINT32(completed, DreamGpu),
            VMSTATE_UINT32(error, DreamGpu),
            VMSTATE_UINT32(irq_enable, DreamGpu),
            VMSTATE_UINT32(irq_status, DreamGpu),
            VMSTATE_UINT32(generation, DreamGpu),
            VMSTATE_UINT32_V(timing_rate, DreamGpu, 4),
            VMSTATE_UINT32_V(timing_sequence, DreamGpu, 4),
            VMSTATE_UINT32_V(timing_completed, DreamGpu, 4),
            VMSTATE_UINT32_V(timing_status, DreamGpu, 4),
            VMSTATE_UINT32(active_count, DreamGpu),
            VMSTATE_UINT32(active_sequence, DreamGpu),
            VMSTATE_UINT32(command_index, DreamGpu),
            VMSTATE_UINT32(row, DreamGpu),
            VMSTATE_UINT32(column, DreamGpu),
            VMSTATE_UINT8_ARRAY(commands, DreamGpu, DG_MAX_COMMANDS *DG_COMMAND_BYTES),
            VMSTATE_STRUCT(cursor, DreamGpu, 3, vmstate_dg_cursor, DreamGpuNativeCursor),
            VMSTATE_UINT32_V(cursor_addr_lo, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_addr_hi, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_bytes, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_width, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_height, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_hot_x, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_hot_y, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_format, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_x, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_y, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_flags, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_sequence, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_status, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_completed, DreamGpu, 3),
            VMSTATE_UINT32_V(cursor_error, DreamGpu, 3),
#ifdef CONFIG_DREAMGPU_GL
            VMSTATE_UINT32(gl_addr_lo, DreamGpu),
            VMSTATE_UINT32(gl_addr_hi, DreamGpu),
            VMSTATE_UINT32(gl_bytes, DreamGpu),
            VMSTATE_UINT32(gl_sequence, DreamGpu),
            VMSTATE_UINT32(gl_generation, DreamGpu),
            VMSTATE_UINT32(gl_status, DreamGpu),
            VMSTATE_UINT32(gl_completed, DreamGpu),
            VMSTATE_UINT32(gl_error, DreamGpu),
            VMSTATE_UINT32(gl_query_function, DreamGpu),
            VMSTATE_UINT32_V(gl_result_addr_lo, DreamGpu, 2),
            VMSTATE_UINT32_V(gl_result_addr_hi, DreamGpu, 2),
            VMSTATE_UINT32_V(gl_result_capacity, DreamGpu, 2),
            VMSTATE_UINT32_V(gl_result_bytes, DreamGpu, 2),
            VMSTATE_UINT32_V(gl_result_type, DreamGpu, 2),
#endif
            VMSTATE_END_OF_LIST()},
};

static void dg_reset(DeviceState *dev) {
    DreamGpu *s = DREAMGPU(dev);

#ifdef CONFIG_DREAMGPU_GL
    s->gl_fault = s->gl_fault_operation = s->gl_fault_sequence = 0;
#endif
    dg_engine_reset(s);
    vga_common_reset(&s->vga);
    timer_del(s->timing_timer);
    s->timing_rate = DG_TIMING_DEFAULT_HZ;
    s->timing_status = DG_TIMING_IDLE;
    s->timing_sequence = s->timing_completed = s->timing_serial = 0;
}

static void dg_realize(PCIDevice *dev, Error **errp) {
    DreamGpu *s = DREAMGPU(dev);

    if (!vga_common_init(&s->vga, OBJECT(dev), errp)) {
        return;
    }
    vga_init(&s->vga, OBJECT(dev), pci_address_space(dev), pci_address_space_io(dev), true);
    s->vga.con = qemu_graphic_console_create(DEVICE(dev), 0, s->vga.hw_ops, &s->vga);
    s->work_bh = qemu_bh_new(dg_work, s);
    s->timing_rate = DG_TIMING_DEFAULT_HZ;
    s->timing_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, dg_timing_expired, s);
#ifdef CONFIG_DREAMGPU_GL
    s->gl_bh = qemu_bh_new(dg_gl_completed_bh, s);
#endif
    pci_register_bar(dev, DG_VRAM_BAR, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vga.vram);
    memory_region_init(&s->mmio, OBJECT(dev), "dreamgpu.mmio", DG_MMIO_SIZE);
    pci_std_vga_mmio_region_init(&s->vga, OBJECT(dev), &s->mmio, s->vga_regs, true, false);
    /* The common helper's EDID option assumes PCIVGAState, not DreamGpu.
     * Own this region explicitly; SeaVGABIOS reads DDC data from BAR2+0. */
    dg_edid_generate(s->edid);
    qemu_edid_region_io(&s->vga_regs[3], OBJECT(dev), s->edid, sizeof(s->edid));
    memory_region_add_subregion(&s->mmio, 0, &s->vga_regs[3]);
    memory_region_init_io(&s->regs, OBJECT(dev), &dg_ops, s, "dreamgpu.commands", 0x1000);
    memory_region_add_subregion(&s->mmio, DG_REG_MAGIC, &s->regs);
    pci_register_bar(dev, DG_MMIO_BAR, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    pci_set_byte(dev->config + PCI_INTERRUPT_PIN, 1);
}

/* Diagnostics are host control state, deliberately not part of VM migration.
 * Enabling starts a fresh bounded window; disabling freezes it for inspection. */
static bool dg_diagnostics_get(Object *obj, Error **errp) {
    return DREAMGPU(obj)->diagnostics_enabled;
}

static void dg_diagnostics_set(Object *obj, bool enabled, Error **errp) {
    DreamGpu *s = DREAMGPU(obj);
    if (enabled) {
        if (!s->diagnostics) {
            s->diagnostics = g_new0(DgDiagnostics, 1);
        } else {
            memset(s->diagnostics, 0, sizeof(*s->diagnostics));
        }
        s->diagnostics->start_us = g_get_monotonic_time();
    } else if (s->diagnostics_enabled) {
        s->diagnostics->elapsed_us = g_get_monotonic_time() - s->diagnostics->start_us;
    }
    s->diagnostics_enabled = enabled;
}

static char *dg_diagnostics_stats(Object *obj, Error **errp) {
    DreamGpu *s = DREAMGPU(obj);
    const DgDiagnostics *d = s->diagnostics;
    GString *out = g_string_new(NULL);
    bool comma = false;
    g_string_append_printf(
        out, "{\"schema\":1,\"enabled\":%s,\"elapsed_us\":%" PRIu64 ",\"mmio\":[",
        s->diagnostics_enabled ? "true" : "false",
        d ? (s->diagnostics_enabled ? (uint64_t)g_get_monotonic_time() - d->start_us
                                    : d->elapsed_us)
          : 0);
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->reads); i++) {
            if (!d->reads[i] && !d->writes[i]) {
                continue;
            }
            g_string_append_printf(out,
                                   "%s{\"offset\":%u,\"reads\":%" PRIu64 ",\"writes\":%" PRIu64 "}",
                                   comma ? "," : "", i * 4, d->reads[i], d->writes[i]);
            comma = true;
        }
    }
    g_string_append_printf(out,
                           "],\"gl\":{\"batches\":%" PRIu64 ",\"bytes\":%" PRIu64
                           ",\"records\":%" PRIu64 ",\"operations\":[",
                           d ? d->batches : 0, d ? d->bytes : 0, d ? d->records : 0);
    comma = false;
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->operations); i++) {
            if (d->operations[i]) {
                g_string_append_printf(out, "%s{\"op\":%u,\"count\":%" PRIu64 "}", comma ? "," : "",
                                       i, d->operations[i]);
                comma = true;
            }
        }
    }
    g_string_append(out, "],\"functions\":[");
    comma = false;
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->functions); i++) {
            if (d->functions[i]) {
                g_string_append_printf(out, "%s{\"function\":%u,\"count\":%" PRIu64 "}",
                                       comma ? "," : "", i, d->functions[i]);
                comma = true;
            }
        }
    }
    g_string_append(out, "],\"desktop\":[");
    comma = false;
    if (d) {
        for (unsigned i = 0; i < G_N_ELEMENTS(d->desktop); i++) {
            if (d->desktop[i]) {
                g_string_append_printf(
                    out, "%s{\"op\":%u,\"count\":%" PRIu64 ",\"rectangle_bytes\":%" PRIu64 "}",
                    comma ? "," : "", i, d->desktop[i], d->desktop_bytes[i]);
                comma = true;
            }
        }
    }
    g_string_append(out, "],\"queries\":[");
    if (d) {
        for (unsigned i = 0; i < d->query_count; i++) {
            const DgDiagnosticQuery *q = &d->queries[i];
            g_string_append_printf(
                out, "%s{\"function\":%u,\"arg0\":%u,\"arg1\":%u,\"count\":%" PRIu64 "}",
                i ? "," : "", q->function, q->arg0, q->arg1, q->count);
        }
    }
    g_string_append_printf(out,
                           "],\"query_overflow\":%" PRIu64 ",\"function_overflow\":%" PRIu64 "}}",
                           d ? d->query_overflow : 0, d ? d->function_overflow : 0);
    return g_string_free(out, false);
}

static void dg_instance_init(Object *obj) {
    object_property_add_bool(obj, "diagnostic-counters", dg_diagnostics_get, dg_diagnostics_set);
    object_property_add_str(obj, "diagnostic-stats", dg_diagnostics_stats, NULL);
}

static const Property dg_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", DreamGpu, vga.vram_size_mb, 256),
    DEFINE_PROP_STRING("gpu-socket", DreamGpu, gpu_socket),
};

static void dg_class_init(ObjectClass *klass, const void *data) {
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);
    AcpiDevAmlIfClass *ac = ACPI_DEV_AML_IF_CLASS(klass);

    pc->realize = dg_realize;
    pc->vendor_id = DG_PCI_VENDOR_ID;
    pc->device_id = DG_PCI_DEVICE_ID;
    pc->revision = 1;
    pc->class_id = PCI_CLASS_DISPLAY_VGA;
    pc->romfile = "vgabios-stdvga.bin";
    dc->desc = "DreamGPU VGA/VBE display with native 2D acceleration";
    dc->hotpluggable = false;
    dc->vmsd = &vmstate_dg;
    device_class_set_props(dc, dg_properties);
    device_class_set_legacy_reset(dc, dg_reset);
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    ac->build_dev_aml = build_vga_aml;
}

static void dg_finalize(Object *obj) {
    DreamGpu *s = DREAMGPU(obj);

#ifdef CONFIG_DREAMGPU_GL
    if (s->gl_bh) {
        qemu_bh_cancel(s->gl_bh);
    }
    s->gl_transfer_active = false;
    dg_gl_engine_free(s->gl);
    if (s->gl_bh) {
        qemu_bh_delete(s->gl_bh);
    }
    migrate_del_blocker(&s->gl_blocker);
#endif
    if (s->work_bh) {
        qemu_bh_delete(s->work_bh);
    }
    timer_free(s->timing_timer);
    g_free(s->diagnostics);
}

static const TypeInfo dg_info = {
    .name = TYPE_DREAMGPU,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(DreamGpu),
    .instance_init = dg_instance_init,
    .instance_finalize = dg_finalize,
    .class_init = dg_class_init,
    .interfaces =
        (const InterfaceInfo[]){
            {INTERFACE_CONVENTIONAL_PCI_DEVICE},
            {TYPE_ACPI_DEV_AML_IF},
            {},
        },
};

static void dg_register_types(void) {
    type_register_static(&dg_info);
}
type_init(dg_register_types)
