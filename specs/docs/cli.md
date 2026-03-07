# CLI Reference — hologram

## Commands

| Command | Description |
|---------|-------------|
| `hologram compile` | Compile a graph file into a `.holo` archive |
| `hologram run` | Execute a `.holo` archive with provided inputs |
| `hologram inspect` | Print archive metadata without executing |

---

## Global Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--help` | — | Print help |
| `--version` | — | Print version |
| `-v`, `--verbose` | false | Enable verbose output |
| `-q`, `--quiet` | false | Suppress non-essential output |

---

## compile

Compile a serialized graph into a `.holo` archive.

```bash
hologram compile <INPUT> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<INPUT>` | Yes | Path to rkyv-serialized graph file |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `<INPUT>.holo` | Output archive path |
| `--no-fuse` | false | Disable fusion optimization |

### Output

Prints compilation statistics:
- Node count
- Schedule levels
- Workspace slots
- Fusion results (folded, fused, CSE)

### Example

```bash
hologram compile model.graph --output model.holo
# Compiled "model.graph" -> "model.holo"
#   nodes: 42
#   levels: 7
#   workspace slots: 5
#   fusion: 3 folded, 2 fused, 1 CSE
```

---

## run

Execute a `.holo` archive with provided inputs.

```bash
hologram run <FILE> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<FILE>` | Yes | Path to `.holo` archive |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | — | Input in format `INDEX:HEX` (repeatable) |

### Input Format

Inputs are specified as `INDEX:HEX` pairs:
- `INDEX`: Input port number (0, 1, 2, ...)
- `HEX`: Hex-encoded byte string

### Output

Prints:
- Layer info (inputs, outputs)
- Output values (hex-encoded)
- Execution time (milliseconds)

### Example

```bash
hologram run model.holo --input 0:deadbeef --input 1:cafebabe
# Layer: main
#   inputs: 2
#   outputs: 1
# Output 0: 42beef...
# Execution time: 0.5 ms
```

---

## inspect

Print archive metadata without executing.

```bash
hologram inspect <FILE> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<FILE>` | Yes | Path to `.holo` archive |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-d`, `--detail` | `summary` | Detail level (repeatable) |

### Detail Levels

| Level | Description |
|-------|-------------|
| `summary` | File size, format version, node count, I/O names |
| `graph` | All nodes with ops, edges, constants |
| `schedule` | Parallel levels, nodes per level, critical path |
| `sections` | Section table (kind, size, offset, CRC32) |
| `weights` | Weight tensor metadata |
| `layout` | Visual byte-map of archive layout |
| `full` | All of the above |
| `json` | Machine-readable JSON output |

Multiple detail levels can be combined:

```bash
hologram inspect model.holo --detail graph --detail schedule
```

### Example

```bash
hologram inspect model.holo --detail full
# hologram archive: model.holo
# Format version: 1
# File size: 4096 bytes
#
# Graph:
#   nodes: 42
#   inputs: ["x", "y"]
#   outputs: ["z"]
# ...
```

---

## Configuration

The CLI does not use a configuration file. All options are passed as command-line arguments.

Environment variables:
- `RAYON_NUM_THREADS`: Control parallel execution thread count
- `RUST_LOG`: Enable debug logging (e.g., `RUST_LOG=hologram=debug`)

---

## Examples

### Full Pipeline

```bash
# Compile a graph
hologram compile my_model.graph --output my_model.holo

# Inspect the archive
hologram inspect my_model.holo --detail summary

# Run with inputs
hologram run my_model.holo --input 0:00112233

# Run with multiple inputs
hologram run my_model.holo --input 0:deadbeef --input 1:cafebabe
```

### Debugging

```bash
# Verbose compilation
hologram compile model.graph -v

# Full archive inspection
hologram inspect model.holo --detail full

# JSON output for tooling
hologram inspect model.holo --detail json > metadata.json
```

---

## Exit Codes

| Code | Meaning |
|------|--------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Invalid archive format |
| 5 | Execution error |