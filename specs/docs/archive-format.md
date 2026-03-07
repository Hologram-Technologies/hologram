# hologram: Archive Format (`.holo`)

---

## Overview

The `.holo` format is a binary archive for storing compiled hologram graphs,
weights, execution schedules, and metadata. It is designed for zero-copy loading
via rkyv serialization and mmap-compatible page alignment.

The header and section table together specify **how** a `.holo` file should be
loaded and executed — the header provides structural layout, while well-known
sections (layer headers, pipelines, weight metadata) define execution behavior.

---

## File Layout

```
┌─────────────────────────┐
│ Header (80 bytes)       │  HOLO magic, version, CRC checksums, offsets
│ (padded to PAGE_SIZE)   │
├─────────────────────────┤
│ Graph Section           │  rkyv-serialized SerializedGraph
│ (page-aligned)          │
├─────────────────────────┤
│ Custom Sections         │  LayerHeader, WeightIndex, Pipeline, etc.
│ (each page-aligned)     │
├─────────────────────────┤
│ Section Table           │  rkyv-serialized SectionTable
│ (page-aligned)          │
├─────────────────────────┤
│ Weights Section         │  raw quantized weights / constants
│ (page-aligned)          │
└─────────────────────────┘
```

---

## Header

Fixed 80-byte `repr(C)` header at file offset 0. Zero-copy deserialized via
`bytemuck::pod_read_unaligned`. All multi-byte integers in native byte order.

```rust
pub struct HoloHeader {
    pub magic: [u8; 4],              // b"HOLO" (0x484F4C4F)
    pub version: u32,                // FORMAT_VERSION = 1
    pub graph_offset: u64,           // byte offset to graph section
    pub graph_size: u64,             // graph section size in bytes
    pub weights_offset: u64,         // byte offset to weights section
    pub weights_size: u64,           // weights section size in bytes
    pub section_table_offset: u64,   // byte offset to section table
    pub section_table_size: u64,     // section table size in bytes
    pub total_size: u64,             // total archive size in bytes
    pub graph_checksum: u32,         // CRC32 of graph section
    pub weights_checksum: u32,       // CRC32 of weights section
    pub section_count: u32,          // number of section table entries
    pub flags: u32,                  // reserved for future use (currently 0)
}
```

### Magic and Version

- Magic bytes: `b"HOLO"` (ASCII, 4 bytes)
- Version: currently `1` (FORMAT_VERSION)
- Readers MUST reject archives with unrecognized versions
- The `flags` field is reserved — currently always `0`

### Version Compatibility

The `version` field is a monotonic `u32`. Compatibility rules:

1. **No forward compatibility:** readers MUST reject archives with `version > FORMAT_VERSION`. Unknown future versions are never loaded speculatively.
2. **Backward compatibility is opt-in:** readers MAY accept `version < FORMAT_VERSION` if the runtime explicitly supports backward-compatible loading for that version range. By default, only exact version matches are accepted.
3. **Breaking change indicator:** bit 31 of the `flags` field is reserved as a breaking-change flag. When set, readers that do not recognize the archive's version MUST reject it without attempting any fallback parsing. This enables future format revisions to signal incompatibility unambiguously.
4. **No in-place migration:** archives cannot be upgraded in place. To move an archive to a new format version, recompile from the source graph. There is no migration tool.
5. **Version increment policy:**
   - Adding new optional section kinds (kind values) does NOT bump `version` — readers that don't recognize a section kind simply skip it.
   - Changes to the header layout, graph serialization format, section table structure, or checksum algorithm MUST bump `version`.
   - Removing or redefining a well-known section kind MUST bump `version`.

### Header Validation

```
1. Read 80 bytes at offset 0
2. Check magic == b"HOLO"
3. Check version == FORMAT_VERSION (currently 1)
4. Validate: graph_offset + graph_size <= total_size
5. Validate: weights_offset + weights_size <= total_size
6. Validate: section_table_offset + section_table_size <= total_size
7. Verify CRC32(graph_bytes) == graph_checksum
8. Verify CRC32(weights_bytes) == weights_checksum
```

Archives that fail any validation step are rejected entirely — partial loading
is not supported.

---

## Section Table

rkyv-serialized index of all sections in the archive:

```rust
pub struct SectionTable {
    pub entries: Vec<SectionEntry>,
}

pub struct SectionEntry {
    pub kind: u32,       // section kind identifier
    pub offset: u64,     // byte offset in archive
    pub size: u64,       // byte size
    pub checksum: u32,   // CRC32 of section data
}
```

Section checksums are validated on load, same as graph and weights checksums.

### Well-Known Section Kinds

| Kind | Constant | Purpose |
|------|----------|---------|
| 1 | `SECTION_WEIGHT_INDEX` | Weight tensor metadata (shapes, dtypes, offsets) |
| 2 | `SECTION_LAYER_HEADER` | Layer descriptors and execution schedule |
| 3 | `SECTION_PIPELINE` | Multi-model pipeline configuration |
| ≥ 0x1000 | `SECTION_CUSTOM_BASE` | Consumer-defined custom sections |

### Custom Section Kind Allocation

Section kind values are partitioned into namespaces to prevent collisions
between independent tools:

| Range | Owner | Purpose |
|-------|-------|---------|
| 1–0xFFF | hologram | Reserved for well-known section kinds |
| 0x1000–0x1FFF | hologram-ai | AI-specific sections (weight metadata, pipeline config) |
| 0x2000–0x2FFF | hologram-sandbox | Sandbox-specific sections |
| 0x3000–0x7FFF | Reserved | Future Hologram ecosystem projects |
| 0x8000–0xFFFFFFFF | Third-party | Available for external consumers |

**Validation rules:**

- `HoloWriter::add_section()` MUST reject kind values below `SECTION_CUSTOM_BASE`
  (0x1000) that are not well-known. This prevents accidental overwriting of
  reserved section kinds.
- Within a single archive, duplicate section kinds are forbidden.
  `HoloWriter::add_section()` MUST return an error if the same kind is added
  twice.
- Readers MUST skip section kinds they do not recognize rather than failing.
  Unknown section kinds are not an error — they enable forward-compatible
  extension.

---

## Execution Specification: Layer Headers (Kind 2)

The `LayerHeader` section is the primary mechanism for specifying **how** a
`.holo` file should be executed. It defines layers, their entrypoints, and
the parallel execution schedule.

```rust
pub struct LayerHeader {
    pub layers: Vec<LayerDescriptor>,
    pub schedule: Vec<Vec<LayerId>>,
}
```

### LayerDescriptor

Each layer describes a unit of execution:

```rust
pub struct LayerDescriptor {
    pub id: LayerId,
    pub name: String,
    pub entrypoint: LayerEntrypoint,
    pub inputs: Vec<TensorPort>,
    pub outputs: Vec<TensorPort>,
    pub group: u32,
    pub plan_offset: u64,
    pub plan_size: u64,
}
```

### LayerEntrypoint

Specifies what a layer executes:

```rust
pub enum LayerEntrypoint {
    Graph,              // execute the embedded graph
    Subgraph(u32),      // execute subgraph by ID
    External(String),   // network/external reference
}
```

### Execution Schedule

The `schedule` field is a `Vec<Vec<LayerId>>`:

- Each outer element is a **parallel level**
- All `LayerId`s within a level can execute concurrently
- Levels execute sequentially — level N must complete before level N+1 starts

This schedule drives the `KvExecutor`'s level-by-level dispatch.

### TensorPort

Input/output specification for each layer:

```rust
pub struct TensorPort {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: WeightDType,
}
```

---

## Weight Metadata (Kind 1)

Optional per-tensor metadata for the weights section:

```rust
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: WeightDType,
    pub offset: u64,                         // offset within weights section
    pub size: u64,                           // byte size
    pub quantization: Option<QuantizationParams>,
    pub checksum: u32,                       // CRC32 of tensor data
}
```

### WeightDType

```
F32, F64, F16, BF16, I8, U8, I16, I32, I64, I4
```

Quantization parameters are attached per-tensor, preserving quantization
metadata through the archive.

---

## Pipeline Archives (Kind 3)

A `.holo` file can contain multiple models as a pipeline:

```rust
pub struct PipelineHeader {
    pub models: Vec<PipelineEntry>,
}

pub struct PipelineEntry {
    pub name: String,     // model name
    pub offset: u64,      // offset in wrapper's weights section
    pub size: u64,        // size of sub-archive
    pub checksum: u32,    // CRC32 of sub-archive
}
```

Each sub-archive is a complete `.holo` file embedded in the wrapper's weights
section. Sub-archives have their own headers, graphs, weights, and sections.
Models are indexed by name or position.

---

## Page Alignment

All section boundaries are aligned to `PAGE_SIZE = 4096` bytes. This enables:

- **mmap loading** — sections mapped directly into virtual memory
- **Zero-copy access** — no buffer copying after loading
- **Lazy page faulting** — weights pages loaded on first access

Padding bytes between sections are zero-filled. The header itself is padded
from 80 bytes to `PAGE_SIZE` before the graph section begins.

---

## Checksums

CRC32 checksums protect data integrity at three levels:

1. **Graph checksum** — header field, validated on load
2. **Weights checksum** — header field, validated on load
3. **Section checksums** — per-entry in `SectionTable`, validated on load
4. **Tensor checksums** — per-tensor in `TensorMetadata` (optional, for weight index)

Checksum mismatches produce errors — archives are never loaded partially or
with unverified data.

---

## Writing

### HoloWriter

Builder pattern for constructing archives:

```rust
let archive = HoloWriter::new()
    .set_graph(&serialized_graph)
    .set_weights(weights_bytes)
    .add_section(SECTION_LAYER_HEADER, &layer_header_bytes)
    .add_section(SECTION_WEIGHT_INDEX, &weight_index_bytes)
    .build()?;
```

The writer:

1. Serializes the graph via rkyv
2. Computes page-aligned offsets for each section
3. Inserts zero-fill padding between sections
4. Computes CRC32 checksums for all sections
5. Builds the section table
6. Writes the 80-byte header (padded to PAGE_SIZE)
7. Returns the complete archive as `Vec<u8>`

---

## Loading

### HoloLoader

```rust
pub fn load_from_bytes(bytes: &[u8]) -> Result<LoadedPlan>

impl LoadedPlan {
    pub fn graph(&self) -> &SerializedGraph
    pub fn weights(&self) -> &[u8]
    pub fn section(&self, kind: u32) -> Option<&[u8]>
    pub fn layer_header(&self) -> Option<LayerHeader>
}
```

Loading steps:

1. Read and validate the 80-byte header (magic, version, bounds)
2. Validate CRC32 checksums for graph and weights sections
3. Deserialize the section table
4. Validate CRC32 checksums for each section
5. Return `LoadedPlan` with references into the buffer

For mmap-backed loading, `LoadedPlan` keeps the mapping alive for its lifetime.
`ConstantData::Deferred` entries are resolved lazily as pages are faulted in.

---

## Serialization

All serialization uses **rkyv** exclusively. No serde dependency. The header
uses **bytemuck** for POD (Plain Old Data) zero-copy access. rkyv provides:

- **Zero-copy deserialization** — data structures used directly from the buffer
- **No allocation on load** — critical for large model archives
- **Deterministic output** — same graph produces same bytes

---

## How the Header Drives Execution

The header and sections together form a complete execution specification:

1. **Header** tells the loader where each section lives and validates integrity
2. **Graph section** contains the compiled computation graph
3. **Layer header (kind 2)** tells the executor:
   - What layers exist and their entrypoints
   - Input/output tensor shapes and dtypes for each layer
   - The parallel execution schedule (which layers run concurrently)
4. **Weight index (kind 1)** tells the loader:
   - Where each weight tensor lives in the weights section
   - Its shape, dtype, and quantization parameters
5. **Pipeline header (kind 3)** enables multi-model archives

The runtime reads the header, validates checksums, deserializes the layer
header, and uses the embedded schedule to drive `KvExecutor` dispatch.

---

## Size Considerations

| Component | Typical size |
|-----------|-------------|
| Header + padding | 4 KB (80 bytes padded to PAGE_SIZE) |
| Graph section | 1 KB – 100 KB (depends on node count) |
| Layer header section | < 1 KB typically |
| Weight index section | < 1 KB typically |
| Section table | < 1 KB typically |
| Weights section | Varies (bytes to gigabytes for large models) |
| Page alignment padding | Up to 4 KB per section boundary |
