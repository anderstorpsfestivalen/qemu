DreamGPU native GL implementation inventory
======================================

The device currently accepts 103 donor function IDs. This is an implementation
inventory for the guest frontend, not an OpenGL version claim. The authoritative
allowlist is ``dg_gl_function_words`` in
``hw/display/dreamgpu-gl-platform.c``. QUERY_FUNCTION reports the exact
argument count and whether a function uses scalar CALL, immutable DATA_CALL,
or bounded QUERY. Unlisted functions are rejected.

Implemented execution forms
---------------------------

* Scalar drawing/state: Begin, End, Vertex2f/3f/4f, Color3f, Color4f,
  TexCoord2f/4f, Clear, ClearColor, ClearDepth, Viewport, Scissor, Enable,
  Disable, DepthFunc, BlendFunc, Flush and Finish.
* Raster state: AlphaFunc, ColorMask, DepthMask/Range, ClearStencil,
  StencilFunc/Mask/Op, CullFace, FrontFace, PolygonMode/Offset, LineWidth,
  LineStipple, PointSize and ShadeModel, with bounded state queries.
* Logical double buffers: DrawBuffer and ReadBuffer select distinct FRONT/BACK
  storage (including left aliases, draw NONE and FRONT_AND_BACK). Both initially
  select BACK. Normal presentation exchanges texture names without copying the
  image; explicit front flush publishes without exchange, and occluded swaps
  exchange without acquiring an export slot. Hint accepts the five OpenGL 1.1
  targets with DONT_CARE, FASTEST or NICEST, with state queries.
* Lighting and generated coordinates: Normal3f, ColorMaterial, Lightf/fv,
  LightModelf/fv, Materialf/fv, Fogf/fv, TexGenf/fv/dv and ClipPlane. Bounded
  vectors contain at most four floats/doubles; light/material/texgen/clip
  queries return exact scalar/vector shapes. Eight lights and six clip planes
  are exposed.
* Matrices: MatrixMode, LoadIdentity, PushMatrix, PopMatrix, Translatef,
  Scalef, Rotatef, Ortho, Frustum, LoadMatrixf and MultMatrixf.
* Virtualized 2D textures: BindTexture, TexParameteri/f/iv/fv, TexEnvi/f/iv/fv,
  TexImage2D, TexSubImage2D, CopyTexImage2D, CopyTexSubImage2D and DeleteTextures.
  Parameter vectors contain one scalar or four border/environment components.
  Inline images are tightly packed unsigned-byte alpha/luminance/LA/RGB/BGR/
  RGBA/BGRA. Copies stay on the GPU, read only inside the current canonical
  drawable, preserve guest texture namespaces, and obey the same object,
  dimension, mip-level and storage bounds as uploads. Borders remain zero;
  copies outside the drawable and depth formats are explicitly rejected.
  TexImage2D with an empty payload allocates zero-initialized storage. The
  frontend can then upload row tiles through a small fixed DMA buffer; no
  full-size guest physical allocation is required. Named renderable textures
  clear on the GPU; legacy formats and unnamed/deleted bindings use one bounded
  64 KiB zero tile while preserving the same texture object.
* Captured client arrays: DrawArrays and DrawElements, with normalized
  position/color/normal/texture-coordinate vertices and validated U8/U16/U32
  indices. Host pointers are temporary and never refer to live guest memory.
* Queries: GetError, GetBooleanv, GetIntegerv, GetFloatv, GetDoublev,
  GetString, IsEnabled, IsTexture, GetTexParameteriv/fv,
  GetTexLevelParameteriv/fv and GetTexEnviv/fv. State selectors are explicitly
  bounded. Strings identify DreamGPU and expose no GL version or extensions.
  ReadPixels uses normalized rectangles of at most 128 pixels, returning packed
  RGBA8 words through the same 512-byte typed result buffer. The frontend clips
  the public rectangle, tiles it and applies format conversion and PACK state.

Guest frontend responsibilities
------------------------------

The kernel owns client tokens, DMA command/result buffers, generation checks,
interrupt completion, and nonreturning coherence-fault handling. User API
arguments must never supply physical addresses or another process's token.

The frontend can normalize scalar type/vector aliases into the implemented
forms, maintain client-array pointers and pixel-unpack state locally, allocate
guest texture names for GenTextures, and snapshot array/image inputs at the
appropriate API call. Sharing, deletion and process exit must preserve the
same guest object namespace semantics as the host transport. Pointer getters
return frontend-owned guest pointers, never host addresses.

WGL still needs pixel-format selection/description, thread current-context
ownership, context sharing, window lifecycle/clipping, and SwapBuffers through
the ordered desktop protocol. A drawable export alone is not a desktop frame.

Remaining functionality before claiming complete OpenGL 1.1
----------------------------------------------------------

The following are missing families, grouped to guide implementation rather
than suggesting that an unimplemented function should silently succeed:

* Polygon stipple input/output and any raster state beyond the implemented
  whitelist still need bounded payloads and explicit acceptance tests.
* Additional texture operations: 1D images/subimages, GetTexImage,
  residency/priority semantics,
  and any advertised packed or palette formats. Hidden native object names
  must remain virtualized through every new operation.
* Pixel/raster operations: RasterPos, Bitmap, DrawPixels, CopyPixels,
  PixelZoom/Transfer/Map and their queries. ReadPixels beyond the canonical
  RGBA8 form needs explicit conversion semantics and acceptance tests; large
  public reads must be tiled, never exceed the bounded query-result storage.
* Attribute stacks and display lists: these need guest-owned namespaces and
  state snapshots, including texture bindings/deletion. Forwarding native
  PushAttrib directly would conflict with private host state and stack use.
* Remaining client-array operations: ArrayElement, InterleavedArrays,
  edge/index arrays and relevant local pointer/state queries.
* Evaluators, selection/feedback, accumulation and color-index API semantics
  where required by the exposed context mode; no acceptance coverage exists.

Glide and Direct3D/WineD3D compatibility remain separate integration work.
Current tests establish native transport, object isolation, exact rendered
pixels and lifecycle behavior; they do not establish game compatibility.
