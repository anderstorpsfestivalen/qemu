DreamGPU retro GPU, ABI 1.0
======================

``-vga none -device dreamgpu`` provides standard VGA and Bochs VBE
scanout together with native host-side 2D operations. Adding
``gpu-socket=/path/to/socket`` enables a bounded experimental host OpenGL
transport and native shared-image export on macOS and Linux. This is not yet
a complete OpenGL implementation, a Windows ICD, or Glide/Direct3D support.

PCI vendor/device ``1234:1113`` is an experimental DreamGPU identity, not an
upstream allocated ID. The device uses the standard VGA option ROM, whose
PCI identity QEMU patches to match the device. BAR0 is prefetchable video
memory (16 MiB by default, configurable with ``vgamem_mb``). BAR2 is an
8 KiB MMIO aperture. Standard VGA, VBE and framebuffer byte-order registers
occupy their usual offsets 0x400, 0x500 and 0x600. DreamGPU registers begin at
0x1000. The public constants and command offsets are in
``include/standard-headers/dreamgpu/gpu.h``.

Submission and completion
-------------------------

The ABI uses aligned 32-bit little-endian MMIO accesses and physical DMA
addresses. A guest driver must enable PCI memory decoding and bus mastering,
allocate a contiguous DMA command buffer, and serialize access to the single
device submission channel. User processes must not access the channel directly.

Read MAGIC, VERSION, CAPS, GENERATION and the limits before submitting work.
VERSION has the major version in the high 16 bits. The base 2D ABI supports FILL, COPY, DAMAGE and completion interrupts.
The separate GL transport capability is present only when built with its native
host dependencies and configured with a GPU socket.

With STATUS.BUSY clear, write BATCH_ADDR_LO/HI, BATCH_COUNT and a driver-chosen
SUBMIT_SEQUENCE. Publish all command-buffer writes with the guest platform's
DMA write barrier, then write 1 to SUBMIT. QEMU synchronously copies the entire
bounded buffer into device-owned memory and validates every command before
changing video memory. The guest can reuse its command memory after SUBMIT
returns; it must wait for completion before accessing affected framebuffer
pixels or issuing dependent work.

A valid batch executes up to 4 MiB of pixel work inline in the submit MMIO
exit, enough for a complete 1024x768x32 desktop copy. Work is additionally
capped at 2,048 row chunks to bound overhead on narrow/tall rectangles. Small
GDI commands therefore complete before SUBMIT returns. Remaining work continues
asynchronously in 256-KiB main-loop work units with the same row limit. The device
remains BUSY until the entire batch completes. Submission while
BUSY is ignored and cannot replace active work. The guest must not rely on
partially executed framebuffer contents. Progress does not require guest
polling or a continuously running host timer.

Completion sets DONE, updates COMPLETED_SEQUENCE and latches IRQ_STATUS bit
0. Errors additionally set ERROR in STATUS and provide an ERROR register code;
validation and DMA errors do not modify the framebuffer. IRQ_ENABLE bit 0
enables a level-triggered PCI INTA interrupt for the latched completion.
IRQ_STATUS is write-one-to-clear. Masking an interrupt preserves its pending
status. Acknowledge an earlier completion before starting the next batch so
each enabled interrupt identifies the corresponding completion.

When CAPS includes INLINE_NO_IRQ, the driver normally writes
``START | INLINE_NO_IRQ`` (3) to SUBMIT. Completion wholly inside that MMIO
write still publishes DONE, ERROR and COMPLETED_SEQUENCE, but does not latch
a new completion interrupt. This includes immediate validation failures.
Remaining asynchronous work always latches its completion interrupt, even
when it finishes before the guest's first status read. Submission never clears
an earlier pending interrupt, and ordinary SUBMIT=1 retains its original
interrupt behavior. The suppression flag exists only during the MMIO callback;
it cannot leak into a continued or subsequently restored asynchronous batch.

Before submitting, the driver clears its completion event and acknowledges
the preceding interrupt. After submission it checks status and sequence once,
then sleeps on its event only if work remains. The interrupt handler signals
that event; a completion between the status check and wait remains signaled.
The driver must not clear the event between checking status and waiting.
This avoids unnecessary ISR/DPC work without polling or a lost wakeup.

Commands
--------

Each command is exactly 40 bytes: ten unsigned little-endian 32-bit words,
with the offsets defined in the public header. All addresses within commands
are BAR0-relative byte offsets. Width and height are pixels; BPP is bytes per
pixel and must be 1, 2 or 4. Strides must accommodate the requested row width.
Offsets and strides must be aligned to BPP, and every addressed row must fit
within video memory. Zero-sized rectangles are invalid.

* FILL writes the low BPP bytes of COLOR to each destination pixel in
  little-endian order. SRC_OFFSET and SRC_STRIDE must be zero.
* COPY copies source pixels to the destination with SRCCOPY semantics.
  Source and destination strides must match. Horizontal and vertical overlap
  is supported, with the result equivalent to copying from the original source
  rectangle. COLOR must be zero.
* DAMAGE marks a guest-written destination rectangle dirty for the standard
  VGA scanout path. It does not copy or change pixels. Source fields and COLOR
  must be zero.

RESERVED must always be zero. Unknown opcodes or invalid fields reject the
entire batch. At most 64 commands and 64 MiB of destination pixel work are
allowed per batch. These limits are also available through MMIO. Split larger
workloads into ordered batches. Normal guest framebuffer writes continue to
use QEMU's existing dirty logging without a required DAMAGE command.

Independent native cursor
-------------------------

CURSOR capability adds the synchronous register channel documented in
``include/standard-headers/dreamgpu/cursor.h``. SHAPE captures at most
64x64 immutable pixel pairs from guest RAM, validates geometry/hotspot and
premultiplied-alpha or exact RGB AND/XOR semantics, then atomically commits
shape, position and visibility. MOVE changes signed hotspot position and
flags without allocation, GPU work, interrupts or framebuffer copies.
NATIVE_ENABLED and VISIBLE are independent: a guest cursor can remain visible
at its guest position while host input is uncaptured. Invalid requests preserve
the accepted cursor, and only expose an immediate status/sequence error.

The existing display socket carries a separate cursor FD (C), shape updates
(S) and coalesced position updates (P). Three immutable slots hold complete
shapes with their exact position snapshots; move traffic is only 24 bytes.
Consumers claim before inspecting a slot, copy it on shape change, then release
it immediately. Epoch replacement preserves old mappings across disconnect.
Socket backpressure uses readiness callbacks, and leased-slot backpressure
retries at the existing display refresh instead of busy looping. Cursor-only
updates never mark the normal CPU framebuffer dirty or allocate GL resources.
Cursor state is reset and migrated with the device, including immutable pixels.

Reset, display modes and migration
---------------------------------

Writing 1 to RESET cancels remaining work, clears command and interrupt state,
and increments GENERATION. Completed portions of an interrupted batch remain
in video memory; the driver must repaint after abandoning an operation.
This engine reset leaves the display mode and framebuffer contents intact.
A system reset additionally resets standard VGA/VBE state.

The driver must drain or reset the engine before changing display mode or
reusing any framebuffer area. Commands operate on byte-addressed video memory,
not on implicit current-mode coordinates. There is no per-mode host polling.

Migration includes VGA state, video memory, registers and the immutable active
batch with its row and byte cursor. Loading cancels any old local callback and resumes
the loaded work, retaining ordering and overlap-copy direction. Resetting
does not leave a callback capable of signaling an obsolete completion.

Validation
----------

``tests/qtest/dreamgpu-test.c`` exercises PCI/VBE discovery, pixel fills,
horizontal/vertical overlapping copies, whole-batch validation, malformed
counts and addresses, bounded work, interrupt masking/acknowledgement/reset,
immutable submissions, migration and actual VGA scanout through a screendump.
These tests establish device
behavior; Windows driver performance and compatibility require guest tests.

Native GL transport and image lifetime
--------------------------------------

``include/standard-headers/dreamgpu/gl.h`` defines the separate 0x1100 MMIO
channel and versioned variable-length records. The driver allocates process
client tokens and serializes access. Each client owns separately identified
contexts and drawables. Commands carry the device generation; stale commands,
unsupported functions, malformed lengths and resource/work limits are rejected.
The entire DMA record batch is copied and syntactically validated before the
worker sees it. Dynamic failures may follow earlier successfully executed
records, so the driver must not replay a failed batch as software operations.

Only the functions returned by QUERY_FUNCTION/FUNCTION_WORDS are implemented.
The current subset includes fixed-function matrix/state, immediate vertices,
clear, viewport/scissor, synchronization, bounded 2D texture objects and
client-array draws.
Donor function numbers come from
qemu-3dfx's generated vocabulary; this does not imply that its entire dispatch
implementation is supported. Bounded state queries are implemented; other
object classes are not advertised yet. Internal host objects share storage
across native contexts, while guest texture names map to separate namespaces.
Only explicitly shared guest contexts share a texture namespace; client tokens
cannot share objects. Deleting a texture name does not invalidate an object
still bound by another sharing context. Reusing the name creates a new object.

DATA_CALL records contain a function number, byte count, fixed scalar arguments
and an immutable inline byte payload, with zero padding to four-byte alignment.
QUERY_FUNCTION returns ``0x80000000 | scalar_word_count`` for this form.
TexImage2D and TexSubImage2D take eight scalar words followed by tightly packed
unsigned-byte pixels; DeleteTextures takes a count and little-endian guest
names. The guest wrapper must resolve pixel-unpack row lengths and skips before
submission. No guest pointers reach host GL. Accepted image formats are alpha,
luminance, luminance-alpha, RGB/BGR and RGBA/BGRA. Images have nonzero dimensions
of at most 2048, zero border, and at most 12 explicitly supplied mip levels.
Each level's dimensions obey the maximum texture size shifted by its level.
There are at most 4096 live texture objects and 256 MiB of texture storage,
accounted conservatively at four bytes per pixel. Texture filters, wrapping
and base/max levels are supported; automatic mipmap generation is not exposed.
The maximum immutable batch is 16 MiB plus 64 KiB for framing.

An empty TexImage2D payload allocates zero-initialized storage with the same
format, dimension, mip and memory limits. The frontend can follow it with row
tiles in TexSubImage2D records smaller than its fixed 64 KiB DMA buffer.
Renderable named images clear on the GPU; other formats and unnamed or
deleted-but-bound objects use a bounded zero tile. Allocation never changes
the guest's framebuffer binding, clear color, write mask or scissor state.
An initialization failure accounts allocated memory and prevents sampling
undefined contents until a complete upload initializes that level.

CopyTexImage2D and CopyTexSubImage2D take eight scalar words. Their source must
lie within the current canonical drawable; copies remain on the GPU and use
the same guest namespace, storage and mip limits as uploads. Immutable
TexParameterfv/iv and TexEnvfv/iv vectors take one scalar or four color
components, with corresponding bounded getters. Vertex4f and TexCoord4f
preserve homogeneous position and projected texture coordinates.

DrawArrays and DrawElements use DATA_CALL with immutable normalized vertex
records. Each 64-byte record contains position4f, color4f, normal3f,
texture-coordinate4f and a zero reserved word. An attribute mask controls
optional color, normal and texture-coordinate arrays. DrawArrays snapshots
only the requested source range and normalizes ``first`` to zero; DrawElements
appends tightly packed unsigned-byte, unsigned-short or unsigned-int indices.
There are at most 65536 vertices and 262144 indices per draw. Every index is
validated against the captured vertex count before the batch executes.
Host GL reads the immutable batch directly on little-endian hosts, without
another vertex staging copy. Client pointers and current attributes are
restored after the draw; no GL pointer outlives the captured batch.

Begin/End primitives may cross batch boundaries. Internal presentation or
texture updates while a primitive is open fail without exporting an image.
The guest can close the primitive with End before submitting later work.

Raster state includes color/depth/stencil write masks, depth range, alpha test,
stencil functions and operations, culling/front-face selection, polygon mode
and offset, line/point size, line stipple and shading mode. State remains local
to its guest context. Presentation exports ignore guest write masks while
preserving their values. Native pixel tests cover masked clears, depth writes,
stencil-limited drawing, alpha rejection and front-face culling; the bounded
query vocabulary includes the corresponding state.

Fixed-function lighting supports eight lights, front/back material state,
normals and color-material tracking. Fog, generated texture coordinates and
six transformed clip planes use bounded vectors of at most four floats or
doubles, validated before execution. Corresponding getters return their exact
scalar/vector shape. Frustum supplements the matrix vocabulary. Native tests
verify directional lighting, emission, color tracking, fog endpoints, generated
texture coordinates and modelview-transformed clipping against exact pixels.

QUERY is a single-record batch with a function number and up to three scalar
arguments. QUERY_FUNCTION returns ``0x40000000 | scalar_argument_count`` for
this form. The driver programs a writable RAM result address and capacity of
at most 512 bytes; these registers are snapshotted when the batch is accepted.
The GL worker collects the result and the device completion callback copies
it to guest RAM before signaling the completion IRQ. Result size and type are
valid only for that successfully completed sequence. ROM, MMIO, unmapped or
undersized result destinations are rejected before the query executes.

GetError preserves native error flags consumed by internal host operations.
GetBooleanv/GetIntegerv/GetFloatv/GetDoublev support a bounded whitelist of
scalar, vector and matrix state. Texture binding queries return guest names;
size limits describe this device. IsEnabled, IsTexture, texture parameters,
texture-level dimensions/formats and texture environment queries are also
available. GetString returns DreamGPU vendor/renderer strings and no extensions.
GL_VERSION remains unsupported because the current subset does not implement
a complete OpenGL version. Host framebuffer IDs, native object pointers and
host GL version/extension claims are never returned.

ReadPixels is a normalized three-argument QUERY: x, y, and packed
``width | (height << 16)``. It reads the context's logical read buffer into
at most 128 tightly packed RGBA8 pixels, bottom row first. Each INT result word
contains red in its low byte, then green, blue and alpha. The complete tile
must lie inside the drawable; empty/oversized/negative tiles and insufficient
result capacity are rejected before native access. Guest frontends clip public
rectangles, choose deterministic values for undefined outside-window pixels,
tile larger requests and implement public format/type and PACK state locally.
Readback occurs only for explicit requests, on the GL worker.

DrawBuffer and ReadBuffer select distinct logical FRONT/BACK textures, with
BACK initially selected for both. Left-buffer aliases are supported; drawing
NONE disables color output and FRONT_AND_BACK writes both attachments. Stereo
and auxiliary selections are rejected. Queries return the logical selection
and DOUBLEBUFFER=true, never private FBO attachment identifiers. Hint accepts
the five legacy targets and DONT_CARE/FASTEST/NICEST modes.

Native contexts and export resources are created only on the first explicit
GL/desktop request. A condition-variable worker executes GL commands, and a
separate completion worker handles ready image publication. No GL calls, GPU
fence waits, socket writes or release-credit waits run under the device BQL.
All workers sleep when idle. Render targets are offscreen FBOs; QEMU opens no
second SDL/native presentation window. Device CPU-only use does not allocate
GL resources or change the normal VGA display path.

Each drawable has at most three immutable exported slots. The producer reuses
a slot only after its exact epoch/slot/generation release. Mac exports use
IOSurface BGRA images and transferable Mach send rights, published after the
GL completion fence is observed. Linux exports use an EGL device matching the
consumer's DRM render node, linear GBM DMA-BUF storage and native sync-file
fences that the consumer waits on its GPU. The export blit flips GL's origin to
top-left in GPU work. Neither host reads presentation pixels through CPU RAM.

Normal PRESENT publishes a DRAWABLE resource, which never replaces the whole
VM desktop. An explicit privileged EXCLUSIVE present is allowed only when the
drawable exactly matches the active VBE primary. PRESENT_RETAIN holds an exact
image for ordered window clipping/composition; last-present result registers
identify it without guessing the latest frame. RESOURCE_DROP cutoff messages
retire resources before numeric slot reuse, including late Mach arrivals.
Reset similarly retires resources and gates CPU output at a reset-coherent
shared-display epoch/generation. Active native graphics resources block
migration/snapshot save. Restoring an earlier CPU-only snapshot tears down the
old worker/IPC session before loading device state.

With FRONT_BUFFERS capability, ordinary PRESENT exchanges the logical texture
names and exports the new FRONT. There is no additional color copy for the
exchange. PRESENT_FRONT_ONLY exports the current FRONT without exchanging it;
the kernel window path combines it with RETAIN for explicit front flushes.
PRESENT_NO_EXPORT exchanges an occluded drawable without taking an
export slot or changing the last-present result identity. Each drawable's
memory budget includes both color textures, depth/stencil and three exports
(24 bytes per pixel). This storage is allocated only after a 3D request.

With PRESENT_BOUNDS capability, the privileged window path adds
PRESENT_BOUNDED and appends its locked window width and height to the record
(40 bytes total). The native worker checks these against the actual drawable
before exchanging buffers or exporting an image. A resize mismatch returns
DRAWABLE without entering desktop composition; the frontend can recreate the
drawable on its next binding. This guard combines with normal, FRONT_ONLY,
RETAIN and NO_EXPORT semantics. NO_EXPORT accepts only BOUNDED alongside it.
It adds no GPU work or CPU pixel copies to ordinary presentation.

Native diskless acceptance tests in DreamGPU verify exact texture pixels and
orientation through QEMU to Metal/Vulkan, namespace and explicit-sharing
behavior, subimage updates, deletion with a surviving binding, slot recycling,
native array/indexed draws and ordered desktop composition. This establishes
the transport and native renderer; it does not claim complete OpenGL, Glide, or Direct3D game support.

Ordered desktop composition
---------------------------

The GL record opcode DESKTOP contains the fixed 64-byte descriptor in
``gl.h``. The kernel display driver holds its primary/GDI serialization
barrier until completion, including across CPU accesses. The host first
validates bounds for every descriptor; captured or written VRAM is then copied
in 256-KiB main-loop quanta while the GL worker sleeps. Captures go directly
into separately leased shared-memory FDs, with 256-byte rows and 64-KiB rounded
allocations. Padding is initialized and never contains stale pixel data from
another geometry. Eight CPU slots and 64 MiB per allocation bound retention.

SEED captures the exact complete CPU primary at the transition into GPU desktop
composition. PATCH captures a precise CPU-written occlusion region. FILL and
overlap-safe COPY execute on the consumer's canonical desktop canvas; BLIT
references one exact retained GL image with source rectangle and destination
coordinates. Each clipped BLIT preserves ordering, and the final one consumes
the retained registry reference. DISCARD releases unused retained images.
The resource identity and the desktop epoch/sequence are independent.

READBACK waits asynchronously for a bounded immutable FD reply and writes the
requested pixels into guest VRAM. A full-primary readback establishes coherent
VRAM after GPU modifications. RETURN requires that coherence, captures the
exact complete returned CPU image and supplies a future legacy display anchor;
normal CPU frames cannot supersede it until they reach that anchor. Existing
CPU-only operation needs none of this traffic. The fixed 128-byte host protocol
and all handle/lifetime details are in
``include/standard-headers/dreamgpu/transport.h``.

Native integration tests in DreamGPU verify actual QEMU command submission through
CGL/IOSurface/Metal and EGL/DMA-BUF/Vulkan, slot releases, vertical orientation,
window clipping, CPU occlusion and readback pixels written back into guest VRAM.
Guest game support still requires the source-built Windows transport/wrappers
and the remaining API vocabulary; transport tests alone do not establish it.

Coherence failures stop the guest
--------------------------------

An unsuccessful desktop READBACK or RETURN stops the VM in ``internal-error``
before signaling completion. A privileged driver detecting its own validation
failure or timeout writes FAULT_STOP and then waits indefinitely on a
nonsignaled kernel event. That path must not return through a void display
synchronization callback: GDI must never continue writing a stale mapped
primary. The host's requested vCPU stop and the driver's nonreturning wait
together cover even the final instructions of a translated CPU block.

``DREAMGPU_FAULT`` reports device path, reason, operation and submission
sequence over QMP. The last canvas and its resources remain available for
display and diagnostics; failure does not grant CPU ownership. QMP ``cont``
rejects ``internal-error`` until a system reset. Guest engine RESET and a zero
fault-register write cannot clear this condition; a system reset or a healthy
checkpoint restore clears the device fault state.

The fault qtest verifies event fields, stopped state, rejected resume, and
system reset recovery. A native integration test injects a failed renderer
readback, verifies that no CPU ownership transition occurs, and checks the
retained canvas pixels through an independent diagnostic readback.

Native 2D diagnostics
---------------------

Enable ``-trace enable=dreamgpu_gpu_*`` only during a diagnostic capture.
The submit event records sequence, command count and validated pixel bytes;
work events record each inline/BH quantum's bytes, row chunks and native elapsed
microseconds. Completion records total elapsed time and number of quanta.
Clock sampling is conditional on the corresponding trace event being enabled.
This separates native copy execution from scheduling and guest driver latency.
Disable tracing for comparative performance captures.

Namespace and saved state
-------------------------

The native device is ``dreamgpu`` and the display backend is
``dreamgpu-shmem``. The QMP fault event is
``DREAMGPU_FAULT``. Guest command identifiers, register offsets, PCI identity,
shared-memory layouts and fixed wire signatures are unchanged by this naming
migration. Protocol headers live under ``standard-headers/dreamgpu``.

VMState section names now use ``dreamgpu`` and ``dreamgpu-cursor``. Older
snapshots containing the previous device section names are not an activation
path for this namespace change: cleanly stop the old guest and cold boot its
disk using the matched DreamGPU native runtime, guest driver and SDK package.
