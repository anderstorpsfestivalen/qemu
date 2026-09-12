/* SPDX-License-Identifier: GPL-2.0-or-later
 * Optional host-only profiling. BQL serializes recording and QOM snapshots;
 * no guest pointers, payloads, per-command allocation or log writes are kept.
 */
#ifndef DREAMGPU_DIAGNOSTICS_H
#define DREAMGPU_DIAGNOSTICS_H
#define DG_DIAGNOSTIC_FUNCTIONS 4096
#define DG_DIAGNOSTIC_QUERIES 128

typedef struct {
    uint32_t function, arg0, arg1;
    uint64_t count;
} DgDiagnosticQuery;

typedef struct {
    uint64_t start_us, elapsed_us;
    uint64_t reads[DG_MMIO_SIZE / 4], writes[DG_MMIO_SIZE / 4];
    uint64_t batches, bytes, records, operations[32], desktop[16], desktop_bytes[16];
    uint64_t functions[DG_DIAGNOSTIC_FUNCTIONS];
    uint64_t function_overflow, query_overflow;
    uint32_t query_count;
    DgDiagnosticQuery queries[DG_DIAGNOSTIC_QUERIES];
} DgDiagnostics;

static void dg_diagnostic_record(DgDiagnostics *d, const uint8_t *r)
{
    uint32_t op = ldl_le_p(r + DG_GL_OFF_OP);
    d->records++;
    if (op < G_N_ELEMENTS(d->operations)) {
        d->operations[op]++;
    }
    if (op == DG_GL_DESKTOP) {
        uint32_t operation = ldl_le_p(r + DG_GL_HEADER_BYTES);
        if (operation < G_N_ELEMENTS(d->desktop)) {
            d->desktop[operation]++;
            d->desktop_bytes[operation] += (uint64_t)ldl_le_p(r + DG_GL_HEADER_BYTES + DG_DESKTOP_WIDTH) *
                ldl_le_p(r + DG_GL_HEADER_BYTES + DG_DESKTOP_HEIGHT) * 4;
        }
    }
    if (op == DG_GL_CALL || op == DG_GL_DATA_CALL || op == DG_GL_QUERY) {
        uint32_t function = ldl_le_p(r + DG_GL_HEADER_BYTES);
        if (function < G_N_ELEMENTS(d->functions)) {
            d->functions[function]++;
        } else {
            d->function_overflow++;
        }
        if (op == DG_GL_QUERY) {
            uint32_t a0 = ldl_le_p(r + 36), a1 = ldl_le_p(r + 40);
            for (unsigned i = 0; i < d->query_count; i++) {
                DgDiagnosticQuery *q = &d->queries[i];
                if (q->function == function && q->arg0 == a0 && q->arg1 == a1) {
                    q->count++;
                    return;
                }
            }
            if (d->query_count < G_N_ELEMENTS(d->queries)) {
                d->queries[d->query_count++] = (DgDiagnosticQuery) {
                    .function = function, .arg0 = a0, .arg1 = a1, .count = 1,
                };
            } else {
                d->query_overflow++;
            }
        }
    }
}
#endif
