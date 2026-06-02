# External Tensor References

Status: design note for Plan 075 Phase 7.1c, with initial file-backed
`SourceExternalTensor` / FFI builder support added in Plan 076 Phase 1.

## Problem

Inline tensor literals are useful for tests, small examples, and hand-written
graphs. They are the wrong default for model-scale weights. A host-language
frontend that requires `values=[...]` for large tensors forces bytes through
source text, parser allocation, source IR storage, graph constants, archive
packing, and backend weight packing. That path is easy to understand but it is
not the zero-copy-oriented path Hologram should expose for production weights.

The source frontend architecture therefore needs a common external tensor
contract before large Python, TypeScript, Rust, or SDK examples start inlining
weights by convention.

## Contract

External tensor references belong at the shared source/SDK boundary, not in a
specific parser:

```text
Python / TypeScript / Rust parser -> SourceProgram -> Graph -> Compiler
Python / TypeScript / Rust SDK    -> SourceProgram -> Graph -> Compiler
FFI builder API                   -> SourceProgram -> Graph -> Compiler
```

The source IR now has a source-level constant form alongside inline
`SourceTensorLiteral`:

```rust
SourceExternalTensor {
    name: SourceSymbol,
    dtype: DTypeId,
    shape: ShapeDescriptor,
    location: SourceExternalTensorLocation,
    byte_offset: u64,
    byte_len: u64,
    content_hash: [u8; 32],
}

SourceExternalTensorLocation =
    File(path)
```

Archive-section and address-reference locations remain planned extensions; the
initial implementation supports file-backed references in host (`std`) compiler
builds.

## Invariants

- The frontend never executes host-language code to obtain tensor bytes.
- The parser or SDK records a typed reference: dtype, shape, byte range, and
  content hash are known before lowering.
- The compiler validates `byte_len == element_count(shape) * dtype_width`.
- The compiler validates the content hash before bytes enter the archive.
- Relative file paths resolve from `HOLOGRAM_EXTERNAL_TENSOR_ROOT` when it is
  set, otherwise from the compiler process' current directory.
- When `HOLOGRAM_EXTERNAL_TENSOR_ROOT` is set, every resolved external tensor
  path, relative or absolute, must canonicalize under that root.
- The compiler streams or maps referenced bytes into the archive/packing path
  once. It must not require a `Vec<f32> -> Vec<u8>` double materialization.
- The graph/runtime boundary remains source-agnostic. By the time backend
  lowering emits `KernelCall`s, external references have become ordinary
  constants, packed weights, or archive sections.
- Runtime execution never opens host paths, follows source URIs, interprets
  parser spans, or performs source-language dispatch.

## Frontend Syntax Sketch

Native Hologram text source can grow an explicit reference form:

```text
const w: f32[4096, 4096] <- file("weights/w.bin", offset=0, len=67108864, blake3="...")
```

Python source extraction can recognize a builder call without executing Python:

```python
w = h.const_ref(
    "w",
    dtype="f32",
    shape=[4096, 4096],
    file="weights/w.bin",
    offset=0,
    len=67108864,
    blake3="...",
)
```

TypeScript and Rust use the same fields in their own restricted builder
surface:

```ts
const w = h.constRef("w", {
    dtype: "f32",
    shape: [4096, 4096],
    file: "weights/w.bin",
    offset: 0,
    len: 67108864,
    blake3: "...",
});
```

```rust
let w = h.const_ref(
    "w",
    dtype("f32"),
    shape([4096, 4096]),
    file("weights/w.bin"),
    offset(0),
    len(67108864),
    blake3("..."),
);
```

SDKs and the FFI builder should construct the same source-level reference
directly instead of round-tripping through parser syntax.

## FFI and SDK Boundary

The SDK path should be a first-class graph authoring path:

```text
hologram_source_builder_new()
hologram_source_builder_input(...)
hologram_source_builder_const_ref(...)
hologram_source_builder_op(...)
hologram_source_builder_output(...)
hologram_source_builder_compile(...)
```

That FFI surface should build `SourceProgram` or `Graph` directly. It should
not expose parser internals, source-language spans, or host-language AST nodes.
Language SDKs can wrap the FFI builder idiomatically while still compiling to
the same IR as parsed source files.

This split keeps the responsibilities clear:

- Parsers extract graph regions from existing source files.
- SDKs build graph/source IR directly.
- The compiler validates and resolves external tensors.
- The runtime executes source-agnostic archives.

## Open Decisions

- Whether hash-only `AddressRef` constants may be satisfied from a substrate
  store without a local file path.
- Whether external tensors should continue lowering first to `ConstantStore`
  entries or stream directly to archive sections for large weights. The initial
  implementation lowers file-backed refs to ordinary graph constants.
- Whether packed backend-specific weight layouts are cached by `(content_hash,
  dtype, shape, backend)`.

Until archive-section streaming is implemented, docs and examples should keep
inline constants small and use file-backed refs only in host compiler flows
where the referenced path, byte range, and BLAKE3 digest are explicit.
