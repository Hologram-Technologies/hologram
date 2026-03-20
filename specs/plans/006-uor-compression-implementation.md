# UOR-Based Lossless Compression: Implementation + Site Visualization

## Context

**Theoretical foundation**: The Landauer limit establishes that erasing one bit of information costs a minimum of kT·ln(2) energy. This defines the irreducible information content of any data — the bits that cannot be removed without thermodynamic cost. Our compression aims to find this minimal representation.

**Approach**: Use UOR's algebraic observables (stratum, spectrum, torus position, ring relationships) to decompose data and factor out structural redundancy, arriving at the irreducible information content. Each observable reveals a different kind of redundancy that purely statistical methods might miss. The Shannon entropy is the theoretical floor; UOR algebra is the lens for reaching it.

The `hologram-compression` sibling repo exists but is empty. We'll implement compression there using hologram-core's algebraic primitives (`ByteDatum`, `ByteRing`, stratum tables, `ElementWiseView`), integrate it into hologram-archive (default on), and visualize it on the hologram site.

**Key constraint**: All compression is bit-perfect lossless. `decompress(compress(x)) == x` exactly.

---

## Phase 1: Bootstrap hologram-compression

### 1a. Fix crate structure
- Replace `src/main.rs` with `src/lib.rs`
- Add dependencies to `Cargo.toml`:
  - `hologram-core` (path dep — provides `ByteDatum`, `ByteRing`, stratum/torus tables, `ElementWiseView`)
  - `uor-foundation` (from UOR-Framework)
  - No external compression libraries

### 1b. Module structure
```
src/
  lib.rs              -- public API: compress(), decompress(), CompressionMode
  codec.rs            -- Compressor trait, CompressedBlock, CompressionScheme
  stratum.rs          -- Stratum partition tables + intra-stratum rank codec
  ring_diff.rs        -- Ring-differential predictor (delta coding in Z/256Z)
  torus_block.rs      -- Orbit-torus blocked coding (page/offset split)
  entropy/
    mod.rs            -- Entropy coding trait
    ans.rs            -- rANS (range asymmetric numeral systems) backend
    histogram.rs      -- Frequency counting + normalization
  float_plane.rs      -- IEEE 754 byte-plane transposition for f32/f64
  permute.rs          -- Bijective ElementWiseView pre-transforms
  pipeline.rs         -- Full transform pipeline orchestration + mode selection
  header.rs           -- Compressed block header format
```

---

## Phase 2: Core compression algorithms

Each algorithm uses a different UOR observable to expose redundancy — structural information that exists in the data but is invisible in the raw byte encoding. By factoring out this redundancy, we approach the irreducible representation.

### 2a. Stratum-Partitioned Entropy Coding (SPEC) — `stratum.rs`

**Observable**: `ByteDatum::stratum()` (popcount) — reveals how "heavy" each byte is in terms of set bits.

Uses stratum to partition Z/256Z into 9 equivalence classes:

| Stratum | Values | Bits needed per value |
|---------|--------|----------------------|
| 0       | 1      | 0                    |
| 1       | 8      | 3                    |
| 2       | 28     | 5                    |
| 3       | 56     | 6                    |
| 4       | 70     | ~6.13 (log2(70))     |
| 5-8     | mirror | mirror               |

**How it works:**
1. Map each byte to `(stratum, intra-stratum-rank)` — compile-time 256-entry lookup tables
2. Entropy-code the stratum stream (9 symbols — highly compressible when data clusters)
3. Encode intra-stratum ranks with per-stratum ANS models
4. **Lossless**: inverse table `(stratum, rank) → byte` is bijective

**Why it helps**: Weight tensors concentrate in specific strata (near-zero = low stratum, near-max = high stratum). The stratum stream becomes very low entropy.

**Stratum symmetry**: `stratum(bnot(x)) == 8 - stratum(x)` — automatically handles complement-symmetric distributions.

### 2b. Ring-Differential Coding (RDC) — `ring_diff.rs`

**Observable**: `ByteRing::sub` — reveals sequential correlation (how much each value changes from its predecessor in the ring).

Delta coding using ring arithmetic instead of integer subtraction:

```
residual[i] = ByteRing::sub(data[i], predictor[i])   // forward
data[i]     = ByteRing::add(residual[i], predictor[i]) // inverse (exact)
```

Predictors (all in Z/256Z):
- **Order-0**: `pred[i] = data[i-1]`
- **Order-1**: `pred[i] = add(data[i-1], sub(data[i-1], data[i-2]))` (linear extrapolation in ring)

Residuals cluster near 0 in Z/256Z for correlated data → low entropy → compresses well.

### 2c. Orbit-Torus Blocked Coding — `torus_block.rs`

Uses hologram-core's existing `torus_page_q0()` function to split each byte into two parts:
- **Page** = `x / 8` (which block of 8 values: 0-31, needs 5 bits raw)
- **Offset** = `x % 8` (position within that block: 0-7, needs 3 bits raw)

Encoding:
- Separate the page stream and offset stream
- Entropy-code the page stream (when data clusters in a few pages, this compresses heavily)
- Offset stream stays at 3 bits/symbol
- Net: fewer than 8 bits/byte when page distribution is skewed

Effective for quantized weights clustered near a zero-point (one torus page dominates).

### 2d. rANS Entropy Backend — `entropy/ans.rs`

rANS (range Asymmetric Numeral Systems) is the final step that turns transformed data into a compressed bitstream. It encodes symbols based on their frequency — common symbols get fewer bits, rare ones get more — achieving near-optimal compression (close to the Shannon entropy limit). It's a modern alternative to Huffman coding with better compression and simpler streaming.

- Pure `no_std` implementation, no external dependencies (fits hologram's zero-dependency approach)
- Frequency table normalization to power-of-2 denominators
- Per-stratum frequency tables for SPEC mode

### 2e. Float Byte-Plane Transposition — `float_plane.rs`

**Observable**: IEEE 754 structure — reveals that the information content is highly non-uniform across byte positions (exponent bytes carry far less entropy than mantissa bytes).

For f32/f64 tensors, transpose into separate byte planes before compression:
- **f32** → 4 planes of N bytes each
- Plane 3 (sign+exponent): extremely low entropy → SPEC excels
- Plane 2 (exponent+mantissa_hi): medium entropy → RDC + SPEC
- Plane 1 (mantissa_mid): moderate → RDC
- Plane 0 (mantissa_lo): near-random → minimal compression
- Each plane compressed independently

### 2f. Bijective Pre-Transforms — `permute.rs`

`ElementWiseView` permutations applied before compression to reduce entropy:
- Gray code reorder
- Stratum-sort permutation
- Neg-complement reorder
- Auto-select best by trial-compressing a sample
- 1-byte permutation ID stored in header

### 2g. Pipeline Orchestration — `pipeline.rs`

```
Raw bytes → [Mode Select] → [Pre-Transform] → [Ring-Diff] → [SPEC/Torus] → [ANS] → compressed
```

Mode selection heuristic: analyze first 1024 bytes, compute stratum histogram + entropy estimate, pick best mode.

Modes:
- **Mode 0 (Generic)**: RDC + ANS
- **Mode 1 (Stratum)**: SPEC
- **Mode 2 (Float)**: Byte-plane transpose + per-plane SPEC/RDC
- **Mode 3 (Quantized)**: Torus-blocked coding

---

## Phase 3: Archive integration (hologram repo)

### 3a. Add hologram-compression as dependency
- `crates/hologram-archive/Cargo.toml` — add path dependency

### 3b. Wire into archive format
- Add `CompressionScheme` to weight `TensorMetadata` alongside existing `QuantizationParams`
  - File: `crates/hologram-archive/src/weight/mod.rs`
- Add compression flag bits to `HoloHeader.flags`
  - File: `crates/hologram-archive/src/format/header.rs`
- Compress weight sections on write (default on)
- Transparent decompression on load in `loader/bytes.rs`

### 3c. Graph section compression
- Apply generic mode (Mode 0) to rkyv-serialized graph data
- Transparent: reader checks flag, decompresses if set

---

## Phase 4: WASM FFI + Site demo

### 4a. New WASM functions (`crates/hologram-ffi/src/wasm/mod.rs`)

```rust
compress(data: &[u8], mode: i32) -> Vec<u8>
decompress(compressed: &[u8]) -> Vec<u8>
compression_stats(data: &[u8], mode: i32) -> Vec<f64>
stratum_histogram(data: &[u8]) -> Vec<u32>
ring_algebra(a: u8, b: u8) -> Vec<u8>
float_plane_transpose(f32_bytes: &[u8]) -> Vec<u8>
```

### 4b. Site demo page (`site/src/pages/demo/compression.astro`)

- Section 1: "From Landauer to Minimal Representation"
- Section 2: "Ring Algebra Explorer"
- Section 3: "Stratum Analysis"
- Section 4: "Compression Ratios"
- Section 5: "Float Byte-Plane Visualization"

---

## Expected compression ratios

| Data Type | Best Mode | Ratio |
|-----------|-----------|-------|
| Random bytes | Any | ~1.00x |
| f32 Gaussian weights | Float (Mode 2) | 1.3-1.8x |
| f32 sparse weights | Float (Mode 2) | 2.0-4.0x |
| Q0 symmetric quantized | Stratum (Mode 1) | 1.2-1.5x |
| Q0 clustered (k=16) | Torus (Mode 3) | ~2.0x |
| rkyv graph data | Generic (Mode 0) | 1.5-3.0x |

## Algebraic invariants to test

1. `add(sub(x, pred), pred) == x` — ring-diff round-trip
2. `from_stratum_rank(to_stratum_rank(x)) == x` — stratum partition bijectivity
3. `stratum(bnot(x)) == 8 - stratum(x)` — complement symmetry
4. `bnot(neg(x)) == succ(x)` — critical identity
5. `inverse(permute(x)) == x` — pre-transform bijectivity
