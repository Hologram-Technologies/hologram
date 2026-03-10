# Sprint: UOR-Based Lossless Compression

**Sprint Goal**: Implement lossless compression using UOR ring algebra observables, integrate into hologram-archive, and build a site visualization demo.

**Plan**: [specs/plans/uor-compression-implementation.md](plans/uor-compression-implementation.md)

---

## Tasks

### Phase 1: Bootstrap hologram-compression
- [x] Fix crate structure (lib.rs, Cargo.toml with hologram-core dep)
- [x] Create module skeleton (codec, stratum, ring_diff, torus_block, entropy, float_plane, permute, pipeline, header)

### Phase 2: Core compression algorithms
- [x] Codec types (CompressedBlock, CompressionMode, CompressionStats)
- [x] Header format (HLZC magic, mode, permute_id, original_len)
- [x] Stratum partition tables + intra-stratum rank codec (SPEC)
- [x] Ring-differential coding (RDC) with order-0 and order-1 predictors
- [x] Orbit-torus blocked coding (page/offset split)
- [x] rANS entropy backend (encoder + decoder)
- [x] Frequency counting + normalization
- [x] Float byte-plane transposition (f32/f64)
- [ ] Bijective pre-transforms (ElementWiseView permutations)
- [ ] Pipeline orchestration + mode selection
- [ ] Full end-to-end compress/decompress with all 4 modes

### Phase 3: Archive integration (hologram repo)
- [ ] Add hologram-compression as dependency to hologram-archive
- [ ] CompressionScheme in TensorMetadata
- [ ] Compression flag bits in HoloHeader
- [ ] Default-on compression for weight sections
- [ ] Transparent decompression on load
- [ ] Graph section compression (Mode 0)

### Phase 4: WASM FFI + Site demo
- [ ] New WASM functions (compress, decompress, stats, histogram, ring_algebra, float_plane_transpose)
- [ ] Site demo page (compression.astro)
- [ ] Register in site config sidebar

---

## Status
- **Started**: 2026-03-09
- **Current phase**: Phase 2 (Core algorithms)
