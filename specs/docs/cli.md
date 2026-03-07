# CLI Reference — hologram

## Commands

| Command | Description |
|---------|-------------|
| `hologram compile <source> -o <output>` | Compile a source file to `.holo` archive format |
| `hologram run <archive> [inputs...]` | Execute a `.holo` file with provided inputs |
| `hologram inspect <archive>` | Print metadata without executing |

---

## Global Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--help` | — | Print help |
| `--version` | — | Print version |

---

## Subcommand Details

### compile

```bash
hologram compile <source> -o <output.holo>
```

| Flag | Description |
|------|-------------|
| `-o, --output <path>` | Output `.holo` file path (required) |
| `--no-fuse` | Disable fusion optimization pass |

### run

```bash
hologram run <archive.holo> [--input <file>]...
```

| Flag | Description |
|------|-------------|
| `--input <file>` | Input data file (can be repeated) |
| `--output <file>` | Write output to file instead of stdout |

### inspect

```bash
hologram inspect <archive.holo>
```

Prints:
- Header: magic, version, node count, weight size
- Graph structure summary
- Section table

---

## Configuration

No configuration file. All options are passed via command-line flags.

---

## Examples

```bash
# Compile a graph definition
hologram compile model.graph -o model.holo

# Run with input files
hologram run model.holo --input data.bin

# Inspect archive metadata
hologram inspect model.holo
```

---

## Exit Codes

| Code | Meaning |
|------|--------|
| 0 | Success |
| 1 | General error (invalid args, file not found, etc.) |
| 2 | Compilation error |
| 3 | Execution error |