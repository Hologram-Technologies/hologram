# The invariance ladder, and the structure/content κ law

Two design laws the substrate now applies uniformly. Both are informed by the
exact-formalization discipline (the UOR matmul and driftless-torus proofs):
state each guarantee exactly where it is unconditionally true, derive every
constant from declared structure, and let no tolerance decide semantics.
Everything asserted here is pinned by a witness in-tree; nothing is a Lean
theorem and nothing below claims to be.

## The invariance ladder

What "bit-identical" may be claimed, by rung. A claim from a higher rung must
never be cited for a lower one.

1. **Integer paths** (i8 / i4 / E8CB matmul): **codec-invariant,
   schedule-independent, and machine-invariant.** The accumulation is an exact
   `i32` sum of alphabet-bounded products under the derived ceiling
   `K_MAX = ⌊cap / B²⌋` (`kernel_call.rs`, derived from
   `ALPHABET_BOUND`/`ACCUM_CAPACITY`, pinned by
   `k_max_is_derived_from_the_alphabet_bound`). Integer addition is
   associative and commutative, so any tiling, threading, or completion order
   yields the same integer, on every lane. Witnesses:
   `integer_gemv_bits_are_the_cross_lane_golden`,
   `tiers_that_decode_to_the_same_weights_agree_bit_for_bit`,
   `batched_integer_gemv_equals_row_by_row_bit_for_bit`.

2. **f32 paths, within one lane**: **structure-pinned bit-identity.** Every
   output cell is computed whole, by one participant, running one pinned
   serial order — so pooled == serial, split-KV == precatenated, padded ==
   tight, moved == copied, and chunked == sequential are all *exact* on any
   single target. Witnesses: `parallel_gemv_matches_serial_bitwise` (attention
   fixture), `decode_attention.rs` (three bitwise witnesses),
   `kv_cache_write.rs` (move == copy), `flow_law.rs` (chunk == steps).

3. **f32 across lanes**: **bit-identical for the dot and sum reductions only;
   a non-claim for everything else.**

   - *Identical on every lane:* `simd_f32_dot` and `simd_f32_sum`. Both fix
     one association on every target — sixteen lanes over the whole blocks
     of sixteen (lane `j` folds the terms at `i ≡ j (mod 16)` left to right),
     the halving tree `lane[j] += lane[j + w]` for `w = 8, 4, 2, 1`, then the
     `n mod 16` tail left to right, with multiply and add rounded separately.
     `cpu/simd/canonical.rs` holds the scalar definition and vector twins
     (AVX2, SSE2, NEON, SIMD128) that keep the same sixteen lanes in
     registers and walk the same tree without FMA; Rust neither reassociates
     nor contracts float arithmetic, so every twin executes the defined order
     and dispatch changes speed, never bits. Sixteen lanes were chosen over
     thirty-two by measurement: equal on native x86-64 AVX2 (dot, n = 4096:
     134.7 vs 133.9 ns), and on wasm32 SIMD128 dot n = 4096 took 240 ns with
     sixteen and 409 ns with thirty-two. Witnesses:
     `kc12_f32_dot_and_sum_bits_are_the_cross_target_golden` (one pinned
     digest, run on every CI lane), `kc12b_…_equal_the_canonical_reference_bit_for_bit`
     (an independent restatement of the order, every length 0…300), and
     `scripts/wasm-conformance.sh` (the same digest body run natively and on
     the three wasm32 tiers, compared; CI job `wasm-conformance`). Measured
     digest on all of them: `76408ae96e679411`. NaN payload bits are not
     part of the claim (wasm does not fix them).
   - *Where that reaches:* the reduction step of every kernel built on them —
     attention scores (including the wasm worker pool's tiles, which call the
     same row kernel), the LayerNorm / GroupNorm means and variances, the
     RMSNorm and L2-norm sums of squares, ReduceSum, and the portable matmul
     fallback. The kernels as a whole are **not** claimed identical: their
     other steps (for example `simd_f32_exp_inplace` in attention) are not
     witnessed across lanes.
   - *Not identical:* `matmul_f32_blocked`. Its register-tiled leaves fuse
     multiply-add on x86 and NEON and multiply-then-add on wasm SIMD128, and
     wasm32 without SIMD128 takes the portable fallback, so three substrates
     give three results. Measured digests (`matmul_digest`): native x86-64
     `23de98716143b4e2`, wasm32 baseline `7769cf78c61e036a`, wasm32 SIMD128
     and relaxed-SIMD `8bf9265d83fd372e`. Likewise `simd_f32_fmadd` (fused on
     x86 and NEON only). This is the remaining gap: an f32 graph whose path
     includes an f32 matmul does not carry a machine-independent κ.
   - *No claim:* every other f32 kernel. No test or document may assert
     cross-lane f32 bit-identity beyond the identical list above. For full
     inference, the cross-lane golden is still the integer rung.

## Structure vs. content — where a value lives in κ

A kernel call's **signature** (opcode ‖ scalar params) is *structure*: it
names the shape of the computation and is fixed at compile time. Operand
**bytes** are *content*: they ride the operands' κ-labels and may change every
step without recompiling anything. The rule for placing a value:

> If it changes per step, it is content and must ride an operand.
> If it is fixed for the compiled graph, it is structure and lives in the
> signature.

Applied uniformly across the decode path:

| value | placement | vehicle |
|---|---|---|
| bucket geometry (`past_len`, `bucket_rows`, `planes`, `head_dim`) | structure | call signature |
| realized context length | content | the additive mask's bytes |
| KV write position (ring, wraps mod bucket) | content | the 4-byte `pos` operand |
| gather indices | content | the `indices` operand |
| norm epsilon | structure | `epsilon_bits` — **honored exactly**; `0` selects the pinned default; garbage is refused loud |
| softmax/attention visibility | content | the mask; `−∞` erases a key *exactly* |

Two consequences, both pinned:

- **One compiled step-graph serves every step.** The per-step identity of a
  decode step is carried entirely by operand κ-labels; nothing recompiles and
  no signature changes as the context grows.
- **No tolerance decides semantics.** A fully-masked row is a legal, total
  input with pinned exact semantics (attention → the zero vector; softmax →
  zeros; log-softmax → `−∞`), not a NaN accident rescued by a floor
  (`pinned_semantics.rs`). A declared epsilon is used as declared, never
  silently rewritten.

## Residency: ownership over recency

The transient pool's two-generation window is residency by *recency* — right
for intermediates, wrong for state the host still owns. A **κ-lease**
(`retain_label`/`release_label`) is residency by *ownership*: the value
survives every walk until released, and — the load-bearing composition — **a
lease is a borrow, so the `KvCacheWrite` in-place move requires unique
ownership**. A leased cache steps by honest copy (the pre-image survives:
speculative-rollback and branch-decode primitive); releasing the last lease
restores the move. Witnesses: `lease.rs`. The resource side of the same
discipline: a steady-state decode loop holds pool allocation *exactly*
constant (best-fit buffer recycling; `confinement.rs`).
