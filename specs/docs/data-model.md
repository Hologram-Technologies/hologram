# Data Model — hologram

## Core Types

| Type | Description | Where defined |
|------|-------------|---------------|
| `ElementWiseView` | 256-byte LUT for O(1) byte-to-byte function application | `hologram-core::view` |
| `LutOp` | Enum of precomputed unary operations (Relu, Sigmoid, etc.) | `hologram-core::op::lut_op` |
| `PrimOp` | Primitive binary/unary ops (Add, Sub, Xor, Neg, etc.) | `hologram-core::op::prim` |
| `Graph` | DAG of `GraphOp` nodes with edges and constants | `hologram-graph::graph` |
| `GraphOp` | Node operation variant (Input, Lut, Prim, Output, FusedView, etc.) | `hologram-graph::graph` |
| `NodeId` | Unique identifier for a graph node | `hologram-graph::graph::node` |
| `ExecutionSchedule` | Ordered levels of node IDs for parallel execution | `hologram-graph::schedule` |
| `ConstantStore` | Map from `ConstantId` to `ConstantData` | `hologram-graph::constant` |
| `HoloHeader` | Fixed-size binary header for `.holo` archives | `hologram-archive::format::header` |
| `LoadedPlan` | Deserialized graph + schedule + weights from `.holo` | `hologram-archive::loader::plan` |
| `BufferArena` | HashMap-based storage for execution intermediates | `hologram-exec::buffer::arena` |
| `KvStore` | Stateless dispatch table for graph operations | `hologram-exec::kv::store` |
| `KvExecutor` | Stateful executor that runs graphs level-by-level | `hologram-exec::eval::executor` |

---

## Relationships

```
Graph
├── nodes: Vec<GraphOp>
├── edges: adjacency (inputs per node)
└── constants: ConstantStore
        └── ConstantId → ConstantData (Bytes | Deferred)

ExecutionSchedule
└── levels: Vec<Vec<NodeId>>  (nodes per level, dependencies satisfied)

LoadedPlan
├── graph: Graph (deserialized)
├── schedule: ExecutionSchedule
├── weights: &[u8]
└── header: HoloHeader

BufferArena
└── buffers: HashMap<NodeId, Vec<u8>>

KvExecutor
├── arena: BufferArena
└── registry: Option<CustomOpRegistry>
```

---

## Invariants

| Type | Invariants |
|------|------------|
| `ElementWiseView` | Exactly 256 entries; immutable after construction |
| `Graph` | Acyclic; all node inputs reference existing nodes with lower indices |
| `ExecutionSchedule` | All nodes in level N have dependencies only in levels < N |
| `NodeId` | Unique within a graph; index + generation for validity checking |
| `ConstantStore` | IDs are monotonically increasing; no gaps |
| `BufferArena` | Each `NodeId` has at most one buffer |
| `HoloHeader` | Magic bytes = `HOLO`; version must match `FORMAT_VERSION` |

---

## Serialization

All persistent types use **rkyv 0.8** for zero-copy serialization:

- `Graph`, `ExecutionSchedule`, `ConstantStore` derive `Archive`, `Serialize`, `Deserialize`
- `HoloHeader` uses `bytemuck` for fixed-layout binary (no rkyv overhead)
- Weights are stored as raw `&[u8]` sections
- `.holo` format: `Header (64 bytes) | Graph (rkyv) | Weights | Sections`

---

## Versioning / Migrations

Single format version policy: no backwards compatibility. The `.holo` format version is stored in `HoloHeader::version`. If the version doesn't match `FORMAT_VERSION`, loading fails.

When format changes are needed:
1. Bump `FORMAT_VERSION`
2. Update writer and loader
3. Recompile all `.holo` archives