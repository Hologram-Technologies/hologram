# Plan 017: Zero-Copy Pipeline Weights

## Context

Pipeline archives (prefill + decode) currently embed full weight copies in each
sub-archive. For TinyLlama-1.1B (3.85GB weights), this means ~7.7GB on disk and
two 3.85GB copies in memory at load time. With compression disabled for zero-copy
mmap, this doubles again.

The weight dedup infrastructure (`WeightStore`, `WeightDedupIndex`,
`SECTION_WEIGHT_DEDUP`) already exists (Sprint 16 Phase 10) and the pipeline
loader already resolves dedup entries. The missing piece: the compiler doesn't
use it — each sub-archive embeds its own full weight copy.

## Goal

Pipeline archives store weights **once** in the wrapper. Sub-archives contain
only graph + sections (~250MB), no embedded weights. Loading is zero-copy via
mmap — the executor resolves `ConstantData::Deferred { source_id }` directly
into the wrapper's weight region.

Expected results:
- Archive size: 7.7GB → 4.1GB (one copy + two graphs)
- Load time: 20s+ → <1s (mmap, no decompression, no weight copy)
- RSS: ~4GB → ~200MB (weights accessed on-demand via page faults)

## Architecture

```
Pipeline Archive (.holo)
┌──────────────────────────────┐
│  HoloHeader                  │
├──────────────────────────────┤
│  Graph: (empty)              │
├──────────────────────────────┤
│  Sections:                   │
│  ├─ PipelineHeader           │  models: [{name, offset, size}]
│  ├─ WeightDedupIndex         │  entries: [{component, offset, size}]
│  ├─ ModelMeta                │
│  └─ Tokenizer                │
├──────────────────────────────┤
│  Weights Region:             │
│  ├─ Sub-archive: lm.prefill  │  graph + sections only (no weights)
│  ├─ Sub-archive: lm.decode   │  graph + sections only (no weights)
│  └─ Shared weights blob      │  all model weights, deduplicated
└──────────────────────────────┘
```

Sub-archives have `weights_size=0` in their headers. Their graphs contain
`ConstantData::Deferred { source_id }` where `source_id` points into the
**shared weights blob** (not the sub-archive's own weight region).

At load time, `LoadedPipeline::from_bytes` detects empty weights, looks up
the component in `WeightDedupIndex`, and grafts the shared weight slice onto
the `LoadedPlan` via `set_weights()` (or `new_borrowed()` for zero-copy).

## Changes

### Phase 1: hologram (this repo) — Archive format + loader

**1.1: `LoadedPipeline` zero-copy weight resolution**

File: `crates/hologram-archive/src/loader/pipeline.rs`

Currently `set_weights(weights[start..end].to_vec())` copies the shared weights.
Change to borrow directly from the wrapper's mmap:

```rust
// Instead of: model_plan.set_weights(weights[w_start..w_end].to_vec());
// Use:        unsafe { model_plan.set_weights_borrowed(&wrapper_bytes[abs_start..abs_end]); }
```

Add `LoadedPlan::set_weights_borrowed(&[u8])` that sets `Cow::Borrowed` with
lifetime-extended slice (same pattern as `new_borrowed`).

**1.2: `PipelineWriter` with shared weight blob**

File: `crates/hologram-archive/src/writer/pipeline_writer.rs`

Add `PipelineWriter::build_with_shared_weights()`:
1. Accept sub-archives (graph-only, no weights) + a `WeightDedupIndex` + shared blob
2. Layout: sub-archives first, shared blob last (page-aligned)
3. `PipelineHeader` entries point to sub-archives
4. `WeightDedupIndex` entries point into the shared blob region

**1.3: `load_from_bytes_zero_copy` for pipeline**

File: `crates/hologram-archive/src/loader/pipeline.rs`

Add zero-copy pipeline loading that borrows sub-archive graph bytes and
shared weight bytes directly from the mmap without any allocation.

### Phase 2: hologram-ai — Compiler changes

**2.1: Shared weight extraction during compilation**

File: `crates/hologram-ai/src/compiler.rs`

When building a pipeline archive:
1. Compile prefill and decode graphs separately
2. Extract their weight blobs
3. Feed both into `WeightStore` for deduplication (content-addressed by BLAKE3)
4. Build shared blob + `WeightDedupIndex`
5. Rewrite sub-archive graphs: update `ConstantData::Deferred { source_id }`
   to point into the shared blob instead of per-archive weight regions
6. Build sub-archives with empty weights
7. Build wrapper via `PipelineWriter::build_with_shared_weights()`

**2.2: `HoloRunner` zero-copy pipeline loading**

File: `crates/hologram-ai/src/compiler.rs`

`from_storage` for pipeline archives:
1. Mmap the wrapper
2. Parse `PipelineHeader` and `WeightDedupIndex` from sections
3. For each sub-archive: `load_from_bytes_zero_copy` on the sub-archive slice
4. Resolve weights via dedup index → borrow from wrapper mmap
5. No copies at any point

### Phase 3: Tests

- **test_pipeline_shared_weights**: Build pipeline with shared weights,
  verify both sub-archives resolve constants correctly
- **test_pipeline_zero_copy_load**: Verify mmap pipeline loading produces
  identical results to the copy-based loader
- **test_weight_dedup_cross_model**: Two models sharing identical weight tensors
  → shared blob stores them once
- **test_pipeline_constant_resolution**: Execute a simple pipeline graph,
  verify constants resolve from shared blob offsets

## Reuse

| Component | Status | File |
|-----------|--------|------|
| `WeightStore` | Exists | `hologram-archive/src/weight/dedup.rs` |
| `WeightDedupIndex` | Exists | `hologram-archive/src/weight/dedup.rs` |
| `SECTION_WEIGHT_DEDUP` | Exists | `hologram-archive/src/section/mod.rs` |
| `LoadedPipeline` dedup resolution | Exists | `hologram-archive/src/loader/pipeline.rs` |
| `LoadedPlan::set_weights` | Exists | `hologram-archive/src/loader/plan.rs` |
| `Cow<'static, [u8]>` weights | New (this sprint) | `hologram-archive/src/loader/plan.rs` |
| `load_from_bytes_zero_copy` | New (this sprint) | `hologram-archive/src/loader/bytes.rs` |

## Verification

```bash
# hologram
cargo test -p hologram-archive -- pipeline
cargo test -p hologram-exec -- tape

# hologram-ai (after Phase 2)
hologram-ai compile -m model_causal.onnx -t tokenizer.json -o /tmp/test
hologram-ai run /tmp/test/model_causal.holo --prompt "Hi" --max-tokens 5
# Expected: loads in <1s, RSS <500MB, produces tokens
```
