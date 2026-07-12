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

3. **f32 across lanes**: **a non-claim.** wasm SIMD128 has no FMA; x86/NEON
   contractions differ. An f32 result does not carry a machine-independent κ,
   and no test or document may assert cross-lane f32 bit-identity. The
   cross-lane golden is the integer rung above.

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
