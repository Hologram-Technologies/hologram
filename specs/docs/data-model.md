# Data Model — hologram

## Core Types

| Type | Description | Where defined |
|------|-------------|---------------|
| `ElementWiseView` | 256-byte cache-aligned lookup table for O(1) unary functions | `hologram-core/src/view/` |
| `LutOp` | Enum of 21+ precomputed activation functions (Sigmoid, Relu, Gelu, etc.) | `hologram-core/src/op/` |
| `PrimOp` | Enum of 10 primitive operations (Add, Sub, Mul, Xor, etc.) | `hologram-core/src/op/` |
| `ByteRing` | Ring algebra on Z/256Z for modular arithmetic | `hologram-core/src/ring/` |
| `Graph` | Arena-based expression graph with generation versioning | `hologram-graph/src/graph/` |
| `GraphOp` | Union type for graph operations (Lut, Prim, FusedView, MatMulLut, Custom) | `hologram-graph/src/graph/` |
| `NodeId` | (index: u32, generation: u32) handle for graph nodes | `hologram-graph/src/node/` |
| `ExecutionSchedule` | Ordered parallel levels for execution | `hologram-graph/src/schedule/` |
| `SubgraphDef` | Template for reusable subgraph patterns | `hologram-graph/src/subgraph/` |
| `HoloHeader` | Archive header (magic, version, metadata) | `hologram-archive/src/format/` |
| `LoadedPlan` | Deserialized archive view (graph + weights) | `hologram-archive/src/loader/` |
| `KvExecutor` | Level-by-level graph executor | `hologram-exec/src/eval/` |
| `KvStore` | Operation dispatch via precomputed tables | `hologram-exec/src/kv/` |
| `BufferArena` | Workspace memory with liveness-based reuse | `hologram-exec/src/buffer/` |
| `CustomOpRegistry` | User-defined operation registration | `hologram-exec/src/kv/` |
| `GraphInputs` | Named input map (index → bytes) | `hologram-exec/src/eval/` |
| `GraphOutputs` | Named output list (name → bytes) | `hologram-exec/src/eval/` |

---

## Relationships

### Graph Structure

```
Graph
 ├── nodes: Arena<Node>           # slot + generation storage
 │    └── Node
 │         ├── op: GraphOp        # operation type
 │         ├── inputs: SmallVec   # predecessor NodeIds
 │         └── generation: u32    # validity token
 ├── inputs: Vec<NodeId>          # input port nodes
 ├── outputs: Vec<NodeId>         # output port nodes
 └── subgraphs: Vec<SubgraphDef>  # callable templates
```

### Execution Structure

```
ExecutionSchedule
 └── levels: Vec<ParallelLevel>
      └── ParallelLevel
           └── nodes: Vec<NodeId>  # nodes with satisfied deps
```

### Archive Structure

```
HoloArchive (.holo file)
 ├── HoloHeader
 │    ├── magic: [u8; 4]           # "HOLO"
 │    ├── version: u32             # format version
 │    └── metadata: Metadata
 ├── GraphSection
 │    └── rkyv-serialized Graph
 ├── WeightsSection
 │    └── quantized constant data
 ├── LayerHeader
 │    ├── inputs: Vec<TensorPort>
 │    └── outputs: Vec<TensorPort>
 └── SectionTable
      └── entries: Vec<SectionEntry>
           ├── kind: SectionKind
           ├── offset: u64
           ├── size: u64
           └── crc32: u32
```

---

## Invariants

### Graph Invariants

1. **Acyclic**: Graph has no cycles; topological sort always succeeds
2. **Generation validity**: NodeId.generation matches Node.generation or node is stale
3. **Input connectivity**: Every non-input node has at least one predecessor
4. **Output reachability**: Every node is reachable from outputs (no dead nodes after fusion)

### Execution Invariants

1. **Level ordering**: Nodes in level N have all dependencies in levels < N
2. **Disjoint writes**: Nodes in same level write to disjoint arena slots
3. **Arena bounds**: All slot accesses are within allocated arena

### Archive Invariants

1. **Magic match**: First 4 bytes are `"HOLO"`
2. **Version match**: Format version matches loader version
3. **CRC validity**: Section CRC32 matches computed checksum
4. **Alignment**: Section offsets are 4 KB aligned (for mmap)

---

## Serialization

All persistent types use **rkyv** for zero-copy serialization:

```rust
#[derive(Archive, Serialize, Deserialize)]
pub struct Graph { ... }
```

### Serialization Properties

| Property | Guarantee |
|----------|-----------|
| Zero-copy | Deserialization casts bytes to typed reference |
| Deterministic | Same input produces identical bytes |
| Backward compat | None; single format version |
| Validation | `bytecheck` validates pointer integrity |

### Weight Encoding

| WeightDType | Storage | Description |
|-------------|---------|-------------|
| `F32` | 4 bytes | IEEE 754 single |
| `F64` | 8 bytes | IEEE 754 double |
| `Q4` | 4 bits | Index into 16-entry codebook |
| `Q8` | 8 bits | Index into 256-entry codebook |

---

## Versioning / Migrations

### Format Versioning

- `.holo` format has a version number in header
- **No backwards compatibility**: Only current version is supported
- Format changes require MAJOR version bump of hologram

### Migration Strategy

When format changes:

1. Increment format version in `HoloHeader`
2. Update `HoloWriter` to emit new format
3. Update `HoloLoader` to read new format
4. Remove old format support (no migration path)
5. Re-compile all archives from source graphs

### Graph Versioning

- Graphs are not versioned separately from archives
- Graph changes that affect serialization require format version bump
- Adding new `GraphOp` variants is a breaking change