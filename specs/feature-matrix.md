# Feature Matrix — Hologram Target Compatibility

## holo-core feature flags

| Feature      | Description                                        | Default |
|--------------|----------------------------------------------------|---------|
| `std`        | Enables std + serialize + rkyv validation          | ✓       |
| `serialize`  | rkyv Archive/Serialize/Deserialize derives + alloc | off     |
| `simd`       | SIMD-accelerated LUT paths (x86 AVX2/SSE4.2)      | off     |
| `no_alloc`   | Marker for StaticBuf-only usage (no heap)          | off     |

## Cross-compilation targets

| Feature        | x86_64-linux | wasm32 (no_std) | thumbv7em (ARM) | esp32 |
|----------------|:------------:|:---------------:|:---------------:|:-----:|
| `std`          | ✓            | ✗               | ✗               | ✗     |
| `serialize`    | ✓            | ✗               | ✗               | ✗     |
| `simd`         | ✓            | partial¹        | ✗               | ✗     |
| `no_alloc`     | ✓            | ✓               | ✓               | ✓     |
| StaticBuf      | ✓            | ✓               | ✓               | ✓     |
| LUT tables     | ✓            | ✓               | ✓               | ✓²    |
| rkyv 0.8       | ✓            | ✗³              | ✗³              | ✗³    |
| rayon parallel | ✓            | ✗               | ✗               | ✗     |

¹ WASM SIMD requires `target-feature=+simd128`; not enabled by default.
² ESP32 (Xtensa) requires nightly `xtensa-esp32-none-elf` target.
³ rkyv is optional (`serialize` feature); excluded automatically for no_std builds.

## Build recipes

```bash
# Standard build (std + serialize + simd)
just build

# WASM no_std (no rkyv, no std, LUTs + StaticBuf only)
just wasm-nostd

# ARM bare-metal (Cortex-M4F, no_std)
just embedded

# Full CI (fmt + clippy + tests)
just ci
```

## Binary size estimates (holo-core, release, no_std)

| Target           | Approx .text size |
|------------------|-------------------|
| wasm32-unknown   | ~40 KB            |
| thumbv7em-none   | ~35 KB            |

These are below the 100 KB Sprint 8 target. The dominant sections are
the precomputed LUT tables in `.rodata` (~64 KB for full Q0 tables).
