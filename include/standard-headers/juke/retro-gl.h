/* SPDX-License-Identifier: MIT */
#ifndef JUKE_RETRO_GL_H
#define JUKE_RETRO_GL_H

#define JRG_CAP_GL_TRANSPORT       0x100
#define JRG_CAP_GL_FRONT_BUFFERS   0x200
#define JRG_CAP_GL_PRESENT_BOUNDS  0x400
#define JRG_IRQ_GL_COMPLETION      0x02
#define JRG_GL_REG_VERSION         0x1100
#define JRG_GL_REG_ADDR_LO         0x1104
#define JRG_GL_REG_ADDR_HI         0x1108
#define JRG_GL_REG_BYTES           0x110c
#define JRG_GL_REG_SEQUENCE        0x1110
#define JRG_GL_REG_GENERATION      0x1114
#define JRG_GL_REG_SUBMIT          0x1118
#define JRG_GL_REG_STATUS          0x111c
#define JRG_GL_REG_COMPLETED       0x1120
#define JRG_GL_REG_ERROR           0x1124
#define JRG_GL_REG_MAX_BYTES       0x1128
#define JRG_GL_REG_MAX_RECORDS     0x112c
#define JRG_GL_REG_QUERY_FUNCTION  0x1130
#define JRG_GL_REG_FUNCTION_WORDS  0x1134
#define JRG_GL_REG_PRESENT_SLOT     0x1140
#define JRG_GL_REG_PRESENT_EPOCH_LO 0x1144
#define JRG_GL_REG_PRESENT_EPOCH_HI 0x1148
#define JRG_GL_REG_PRESENT_FRAME_LO 0x114c
#define JRG_GL_REG_PRESENT_FRAME_HI 0x1150
#define JRG_GL_REG_PRESENT_CLIENT   0x1154
#define JRG_GL_REG_PRESENT_DRAWABLE 0x1158
#define JRG_GL_REG_RESULT_ADDR_LO   0x115c
#define JRG_GL_REG_RESULT_ADDR_HI   0x1160
#define JRG_GL_REG_RESULT_CAPACITY  0x1164
#define JRG_GL_REG_RESULT_BYTES     0x1168
#define JRG_GL_REG_RESULT_TYPE      0x116c
#define JRG_GL_REG_FAULT_STOP       0x1170
#define JRG_GL_REG_FAULT_OPERATION  0x1174
#define JRG_GL_REG_FAULT_SEQUENCE   0x1178
#define JRG_GL_VERSION             0x00010000
#define JRG_GL_MAX_BYTES           0x01010000
#define JRG_GL_MAX_RECORDS         1024
#define JRG_GL_MAX_CONTEXTS        32
#define JRG_GL_MAX_DRAWABLES       32
#define JRG_GL_MAX_DIMENSION       4096
#define JRG_GL_MAX_IMAGE_BYTES     0x20000000
#define JRG_GL_EXPORT_SLOTS        3

/* ADDR_LO/HI, BYTES, RESULT_ADDR_LO/HI, RESULT_CAPACITY and the programmed
 * GENERATION persist across submit and
 * completion, and are included in inactive-GL migration. Engine reset clears
 * them and increments REG_GENERATION. A driver can cache DMA programming by
 * that generation; each submit still snapshots the complete command bytes.
 * CAPS and VERSION are invariant for a realized device. STATUS/COMPLETED refer
 * to the accepted operation; DONE without ERROR guarantees REG_ERROR is zero.
 * A pending submission must still check completion before reusing its RAM.
 */

/* Variable records: 32-byte header followed by opcode arguments/data.
 * All integers/IEEE754 float argument words are little endian. SIZE includes
 * header, must be >=32 and multiple of4; concatenated SIZEs equal batch BYTES.
 * CLIENT is a nonzero driver-allocated process token, not the process PID.
 * CONTEXT/DRAWABLE are nonzero per-client IDs when required by the opcode.
 * GENERATION must match the engine generation read before submission.
 * Drivers negotiate min(local limit, REG_MAX_BYTES/REG_MAX_RECORDS) when
 * opening a channel and after a generation change. Older hosts advertise
 * 256 records; record layout and ordering are unchanged by a larger limit.
 */
#define JRG_GL_OFF_OP              0
#define JRG_GL_OFF_SIZE            4
#define JRG_GL_OFF_CLIENT          8
#define JRG_GL_OFF_CONTEXT         12
#define JRG_GL_OFF_DRAWABLE        16
#define JRG_GL_OFF_FLAGS           20
#define JRG_GL_OFF_RESERVED        24
#define JRG_GL_OFF_GENERATION      28
#define JRG_GL_HEADER_BYTES        32

#define JRG_GL_CREATE_CONTEXT      1 /* one word: share-context ID, zero=none */
#define JRG_GL_DESTROY_CONTEXT     2
#define JRG_GL_CREATE_DRAWABLE     3 /* two words: width, height */
#define JRG_GL_DESTROY_DRAWABLE    4
#define JRG_GL_MAKE_CURRENT        5
#define JRG_GL_CALL                6 /* function enum, exact argument words */
#define JRG_GL_PRESENT             7
#define JRG_GL_CLOSE_CLIENT        8
#define JRG_GL_DESKTOP             9 /* fixed 64-byte descriptor below */
#define JRG_GL_DATA_CALL           10
#define JRG_GL_QUERY               11

/* Normal PRESENT exports DRAWABLE, never a complete desktop. EXCLUSIVE must
 * be set explicitly by the privileged display driver and dimensions must
 * match the active VBE primary. No implicit full-desktop takeover occurs.
 */
#define JRG_GL_PRESENT_EXCLUSIVE   1
#define JRG_GL_PRESENT_RETAIN      2
/* Ordinary PRESENT exchanges logical FRONT/BACK, then exports FRONT.
 * FRONT_ONLY exports the current FRONT without exchange, for explicit flush
 * of front-buffer rendering. Resource identity/retention otherwise match.
 * The kernel window path owns this flag; ordinary user PRESENT stays flags0.
 */
#define JRG_GL_PRESENT_FRONT_ONLY  4
/* Occluded normal swap: exchange buffers, publish no image/result identity.
 * May combine only BOUNDED. The window driver retains no export slot. */
#define JRG_GL_PRESENT_NO_EXPORT   8
/* Append expected drawable width,height to PRESENT (40 bytes total). The
 * privileged window driver snapshots these while GDI owns the window lock.
 * Mismatch rejects before exchange/export, never enters desktop composition.
 * Combine with normal, FRONT_ONLY or NO_EXPORT presentation semantics. */
#define JRG_GL_PRESENT_BOUNDED     16
#define JRG_GL_READ_PIXELS_MAX     128
/* QUERY glReadPixels has three normalized words: nonnegative x, y, and
 * width | (height << 16). Nonempty rectangles contain <=128 pixels and must
 * fit the selected drawable. Result INT words contain canonical RGBA8 bytes
 * in GL bottom-left row order (R in the low byte), <=512 bytes total. Guest
 * frontends tile/convert the public format/type and honor their PACK state.
 */

#define JRG_GL_ERROR_NONE          0
#define JRG_GL_ERROR_BATCH         1
#define JRG_GL_ERROR_DMA           2
#define JRG_GL_ERROR_UNSUPPORTED   3
#define JRG_GL_ERROR_CONTEXT       4
#define JRG_GL_ERROR_DRAWABLE      5
#define JRG_GL_ERROR_HOST          6
#define JRG_GL_ERROR_TRANSPORT     7
#define JRG_GL_ERROR_GENERATION    8
#define JRG_GL_ERROR_LIMIT         9
#define JRG_GL_ERROR_DESKTOP       10
#define JRG_GL_ERROR_TEXTURE       11
/* DESKTOP descriptors follow the usual 32-byte GL record header. The kernel
 * display driver serializes all GDI primary accesses until batch completion.
 * CPU captures/readbacks are serviced in bounded main-loop quanta; no guest
 * writes may race a pending capture. CPU operations always use BGRA32 pixels.
 * Seed/Return cover the active VBE primary; Patch can update a subrectangle.
 * Return requires a full-primary Readback after the last GPU modification.
 * Fill color occupies SRC_X; all unused words must be zero. BLIT/DISCARD use
 * CLIENT/DRAWABLE from the enclosing header and exact PRESENT result IDs.
 * BLIT_FINAL consumes the receiver's retained reference after the final clip.
 */
#define JRG_DESKTOP_BYTES          64
#define JRG_DESKTOP_MAX_RECORDS    64
#define JRG_DESKTOP_MAX_BYTES      0x04000000
#define JRG_DESKTOP_SEED           1
#define JRG_DESKTOP_PATCH          2
#define JRG_DESKTOP_RETURN         3
#define JRG_DESKTOP_FILL           4
#define JRG_DESKTOP_COPY           5
#define JRG_DESKTOP_BLIT           6
#define JRG_DESKTOP_READBACK       7
#define JRG_DESKTOP_DISCARD        8
#define JRG_DESKTOP_BLIT_FINAL     1
#define JRG_DESKTOP_OP             0
#define JRG_DESKTOP_FLAGS          4
#define JRG_DESKTOP_DST_X          8
#define JRG_DESKTOP_DST_Y          12
#define JRG_DESKTOP_WIDTH          16
#define JRG_DESKTOP_HEIGHT         20
#define JRG_DESKTOP_SRC_X          24
#define JRG_DESKTOP_SRC_Y          28
#define JRG_DESKTOP_RESERVED0      32
#define JRG_DESKTOP_RESERVED1      36
#define JRG_DESKTOP_SLOT_OR_OFFSET 40
#define JRG_DESKTOP_VRAM_STRIDE    44
#define JRG_DESKTOP_IMAGE_EPOCH    48
#define JRG_DESKTOP_IMAGE_FRAME    56

/* DATA_CALL has function32, byte_count36, fixed scalar arguments40, then an
 * immutable tightly packed byte payload and zero padding to a four-byte size.
 * Guest pointers are never transmitted. The wrapper resolves guest pixel-unpack
 * stride/skips before submission; host uploads use canonical unpack state.
 * QUERY_FUNCTION sets INLINE_DATA in the word count for these functions.
 * TexImage2D/TexSubImage2D omit their pointer argument (eight scalar words).
 * TexImage2D byte_count=0 allocates zero-initialized storage with the same
 * format/dimension/mip bounds; follow it with bounded TexSubImage2D tiles.
 * An empty TexSubImage2D payload is not an allocation request.
 * DeleteTextures has one count word, then that many little-endian guest names.
 */
#define JRG_GL_FUNCTION_INLINE_DATA 0x80000000
#define JRG_GL_FUNCTION_QUERY       0x40000000
#define JRG_GL_FUNCTION_KIND_MASK   0xc0000000
#define JRG_GL_DATA_FUNCTION        32
#define JRG_GL_DATA_BYTES           36
#define JRG_GL_DATA_ARGS            40
#define JRG_GL_MAX_TEXTURES         4096
#define JRG_GL_MAX_TEXTURE_DIMENSION 2048
#define JRG_GL_MAX_TEXTURE_BYTES    0x10000000
#define JRG_GL_MAX_TEXTURE_LEVEL    11

/* Client arrays are copied at draw time, never retained as guest pointers.
 * DrawArrays scalar words: mode, first(0), vertex_count, enabled attributes.
 * DrawElements: mode, index_count, index_type, vertex_count, attributes.
 * Vertex records precede tightly packed U8/U16/U32 indices in Elements.
 * The wrapper normalizes positions/colors/normals/coordinates to float32;
 * all components and indices use little endian. Position is always enabled.
 * DrawArrays snapshots only the requested range, translating first to zero.
 */
#define JRG_GL_VERTEX_BYTES          64
#define JRG_GL_VERTEX_POSITION       0  /* four floats */
#define JRG_GL_VERTEX_COLOR          16 /* four floats */
#define JRG_GL_VERTEX_NORMAL         32 /* three floats */
#define JRG_GL_VERTEX_TEXCOORD       44 /* four floats */
#define JRG_GL_VERTEX_RESERVED       60 /* zero */
#define JRG_GL_ARRAY_POSITION        1
#define JRG_GL_ARRAY_COLOR           2
#define JRG_GL_ARRAY_NORMAL          4
#define JRG_GL_ARRAY_TEXCOORD        8
#define JRG_GL_ARRAY_MASK            15
#define JRG_GL_MAX_VERTICES          65536
#define JRG_GL_MAX_INDICES           262144

/* QUERY must be the sole record in its batch. Payload is function,arg0,arg1,
 * arg2 (16 bytes), with unused arguments zero. Get* uses pname; GetError has
 * no arguments; GetTexParameter/GetTexEnv use target,pname; GetTexLevelParameter
 * uses target,level,pname. QUERY_FUNCTION returns QUERY | scalar argument count.
 * The privileged driver owns a writable RAM DMA result buffer and programs
 * RESULT_ADDR/CAPACITY before submission. These registers are snapshotted;
 * bytes/type become valid only after successful completion of that sequence.
 * Results are packed little-endian values or a NUL-terminated ASCII string.
 */
#define JRG_GL_QUERY_BYTES           48
#define JRG_GL_MAX_RESULT_BYTES      512
#define JRG_GL_RESULT_BOOL           1
#define JRG_GL_RESULT_INT            2
#define JRG_GL_RESULT_FLOAT          3
#define JRG_GL_RESULT_DOUBLE         4
#define JRG_GL_RESULT_STRING         5

/* A coherence failure cannot return through a void display synchronization
 * callback. Write a reason here, then wait on a nonsignaled kernel event
 * without returning to GDI. Host enters internal-error (QMP cont rejects until
 * system reset), preserves exported canvas resources and emits a fault event.
 * READBACK/RETURN errors also trigger this stop before their completion IRQ.
 * Engine RESET cannot clear the fault; a system reset or checkpoint restore
 * is required. Zero/unknown writes do not clear or replace the first reason.
 */
#define JRG_GL_FAULT_HOST_COHERENCE   1
#define JRG_GL_FAULT_DRIVER_TIMEOUT  2
#define JRG_GL_FAULT_DRIVER_VALIDATE 3
#define JRG_GL_FAULT_DRIVER_INTERNAL 4

/* QUERY_FUNCTION selects donor function enum. FUNCTION_WORDS returns exact
 * argument word count (possibly INLINE_DATA) or UINT32_MAX when unsupported.
 * No complete OpenGL version is advertised merely for having a transport.
 */
#endif
