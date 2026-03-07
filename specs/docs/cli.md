# hologram: CLI Specification

---

## Overview

The hologram CLI is the primary tool for compiling graphs into `.holo` archives,
executing archives, and inspecting their contents. It ships as part of the
`hologram-cli` crate and is built when the `cli` feature is enabled.

```sh
hologram <command> [options]
```

---

## Global Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--help` | — | Print help |
| `--version` | — | Print version |
| `-v`, `--verbose` | false | Enable verbose output |
| `-q`, `--quiet` | false | Suppress non-essential output |

---

## Commands

### `hologram compile`

Compiles a serialized graph into an optimized `.holo` archive.

```sh
hologram compile --input <graph.bin> --output <dir/> [--no-fuse]
```

**Arguments:**

| Flag | Required | Description |
|------|----------|-------------|
| `--input <path>` | Yes | Path to the serialized graph (rkyv format) |
| `--output <dir>` | Yes | Output directory for the `.holo` archive |
| `--no-fuse` | No | Disable the fusion/optimization pass |

**Process:**

1. Read and deserialize the graph from rkyv bytes
2. Run the compiler pipeline:
   - Parse: validate graph structure
   - Fuse: constant folding + view fusion + CSE (unless `--no-fuse`)
   - Plan & Emit: liveness analysis → workspace layout → scheduling → archive
3. Create the output directory if it does not exist
4. Write the `.holo` archive into the output directory
5. Print compilation statistics

**Output directory:** The `--output` flag specifies a directory, not a filename.
The compiler derives the archive filename from the input filename (replacing the
extension with `.holo`). The compile command MUST create the output directory
(and any missing parents) if it does not exist. For example:

```sh
hologram compile --input model/graph.bin --output build/out/
# writes build/out/graph.holo
```

The current implementation uses `std::fs::write()` to a file path — this needs
to be changed to accept a directory, derive the filename, and call
`std::fs::create_dir_all()` before writing.

**Output (stdout):**

```
Compiled graph.bin → model.holo
  Nodes:             42
  Levels:            8
  Workspace slots:   12
  Constants folded:  3
  Views fused:       7
  CSE eliminated:    2
  Archive size:      4.2 KB
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid graph (parse failure) |
| 2 | I/O error (file not found, write failure) |

---

### `hologram run`

Loads and executes a `.holo` archive with provided inputs.

```sh
hologram run <model.holo> --input <INDEX:HEX> [--input <INDEX:HEX> ...]
```

**Arguments:**

| Flag | Required | Description |
|------|----------|-------------|
| `<model.holo>` | Yes | Path to the `.holo` archive |
| `--input <INDEX:HEX>` | Yes (1+) | Input data as index:hex pairs |

**Input format:**

Inputs are specified as `INDEX:HEX` pairs where:

- `INDEX` is the zero-based input index (matching the graph's input order)
- `HEX` is the hex-encoded byte data

Examples:

```sh
hologram run model.holo --input 0:deadbeef
hologram run model.holo --input 0:ff --input 1:00ff00
```

**Process:**

1. Load the `.holo` archive via `load_from_bytes()`
2. Parse hex inputs into `GraphInputs`
3. Build the `ExecutionSchedule`
4. Execute via `KvExecutor::execute()`
5. Print named outputs as hex strings

**Output (stdout):**

```
output_0: a3b2c1d0
output_1: ff00
```

Additional output includes:
- Layer info (inputs, outputs)
- Execution time (milliseconds)

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Archive load failure (corrupt, bad checksum) |
| 2 | Input parse error (bad hex, wrong index) |
| 3 | Execution error |

---

### `hologram inspect`

Inspects a `.holo` archive at varying levels of detail without executing it.

```sh
hologram inspect <model.holo> [--detail <LEVEL>]
```

**Arguments:**

| Flag | Required | Description |
|------|----------|-------------|
| `<model.holo>` | Yes | Path to the `.holo` archive |
| `--detail <LEVEL>` | No | Inspection depth (default: `summary`) |

**Detail levels:**

| Level | Description |
|-------|-------------|
| `summary` | File size, format version, node count, I/O names, schedule overview (default) |
| `graph` | All nodes with operations, edges, and constant references |
| `schedule` | Detailed execution schedule — parallel levels, nodes per level, critical path |
| `sections` | Section table — kind, size, offset, and checksum for each section |
| `weights` | Weight/constant metadata — tensor names, shapes, dtypes, quantization, offsets |
| `layout` | Visual byte-map of archive layout |
| `full` | Everything combined (summary + graph + schedule + sections + weights + layout) |
| `json` | Machine-readable JSON of all information |

Multiple detail levels can be combined:

```bash
hologram inspect model.holo --detail graph --detail schedule
```

**Process:**

1. Load the archive via `HoloLoader`
2. Build the execution schedule
3. Read section table and weight index (if present)
4. Print information for the requested detail level

#### `--detail summary` (default)

```
Archive:       model.holo
File size:     4.2 KB
Format:        HOLO v1

Graph:
  Nodes:       42
  Inputs:      x (index 0)
  Outputs:     y (index 0)

Schedule:
  Levels:      8
  Critical path: 8
  Parallelism:   5.25x
```

#### `--detail graph`

```
Graph (42 nodes):
  [0] Input "x"
  [1] Lut(Relu) ← [0]
  [2] Lut(Sigmoid) ← [1]
  [3] FusedView ← [0]    (256-byte table)
  [4] Constant(id=0)     (1024 bytes)
  [5] MatMulLut4(id=1) ← [3, 4]
  ...
  [41] Output "y" ← [40]
```

Lists every node in the graph with its index, operation type, input edges, and
relevant metadata (table size for fused views, byte size for constants, constant
ID for quantized matmul).

#### `--detail schedule`

```
Execution Schedule (8 levels, 42 nodes):
  Level 0:  [0]                          (1 node)
  Level 1:  [1, 3, 4]                    (3 nodes)
  Level 2:  [2, 5, 6, 7, 8]             (5 nodes)
  ...
  Level 7:  [41]                         (1 node)

  Critical path:    8 levels
  Max parallelism:  5 nodes (level 2)
  Avg parallelism:  5.25x
```

Shows the full parallel execution schedule — every level and its constituent
nodes, plus critical path and parallelism statistics.

#### `--detail sections`

```
Section Table (3 entries):
  Kind 1 (WEIGHT_INDEX)   offset=8192    size=512     checksum=0xA3B2C1D0
  Kind 2 (LAYER_HEADER)   offset=12288   size=256     checksum=0xF1E2D3C4
  Kind 3 (PIPELINE)       offset=16384   size=128     checksum=0x12345678
```

Dumps the section table with kind (resolved to name for well-known kinds),
byte offset, byte size, and CRC32 checksum for each entry.

#### `--detail weights`

```
Weights (2.1 KB total, 4 tensors):
  "fc1.weight"  [768, 256]  Q4_0   offset=0      size=768    checksum=0xAABBCCDD
  "fc1.bias"    [256]       F32    offset=768    size=1024   checksum=0x11223344
  "fc2.weight"  [256, 10]   Q4_0   offset=1792   size=256    checksum=0x55667788
  "fc2.bias"    [10]        F32    offset=2048   size=40     checksum=0x99AABBCC
```

Reads the weight index section (kind 1) and lists each tensor with name, shape,
dtype, offset within the weights section, byte size, and checksum. Requires the
archive to contain a `SECTION_WEIGHT_INDEX` section; prints "No weight index
section" otherwise.

#### `--detail full`

Concatenation of all sections above: summary, graph, schedule, sections, weights,
and layout — separated by blank lines.

#### `--detail json`

Machine-readable JSON combining all information. Suitable for tooling, CI
pipelines, and scripting. Structure:

```json
{
  "archive": { "name": "model.holo", "size": 4301, "format_version": 1 },
  "graph": { "node_count": 42, "inputs": [...], "outputs": [...], "nodes": [...] },
  "schedule": { "levels": [...], "critical_path": 8, "parallelism": 5.25 },
  "sections": [...],
  "weights": { "total_size": 2150, "tensors": [...] }
}
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Archive load failure |

---

## Configuration

The CLI does not use a configuration file. All options are passed as command-line arguments.

Environment variables:
- `RAYON_NUM_THREADS`: Control parallel execution thread count
- `RUST_LOG`: Enable debug logging (e.g., `RUST_LOG=hologram=debug`)

---

## Usage Examples

### Compile and run a simple graph

```sh
# Compile (writes relu_chain.holo into build/)
hologram compile --input relu_chain.bin --output build/

# Quick summary
hologram inspect build/relu_chain.holo

# Detailed graph structure
hologram inspect build/relu_chain.holo --detail graph

# Full inspection (everything)
hologram inspect build/relu_chain.holo --detail full

# Machine-readable output for tooling
hologram inspect build/relu_chain.holo --detail json

# Run with a single byte input
hologram run build/relu_chain.holo --input 0:80

# Run with multiple inputs
hologram run build/relu_chain.holo --input 0:deadbeef --input 1:cafebabe
```

### Compile without fusion (debugging)

```sh
hologram compile --input graph.bin --output build/ --no-fuse
hologram inspect build/graph.holo --detail schedule
```

### Debugging

```bash
# Verbose compilation
hologram compile --input model.graph --output build/ -v

# Full archive inspection
hologram inspect model.holo --detail full

# JSON output for tooling
hologram inspect model.holo --detail json > metadata.json
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error / Invalid graph / Archive load failure |
| 2 | I/O error / Invalid arguments / Input parse error |
| 3 | Execution error |

---

## Integration with hologram-ai

The `hologram-ai` CLI delegates compilation to the hologram CLI. When
`hologram-ai compile` is invoked, it:

1. Imports the model and lowers to `hologram::Graph`
2. Serializes the graph
3. Invokes `hologram compile` (or calls `hologram::compile()` as a library)
4. Produces the final `.holo` archive

See ADR-0009 for the delegation design.
