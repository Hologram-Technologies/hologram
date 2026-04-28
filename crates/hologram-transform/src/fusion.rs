//! Planner-side fusion pass over a [`TransformChain`].
//!
//! Fusion rewrites the chain's `nodes` slice to replace specific
//! patterns with their fused-op equivalents. The address table /
//! workspace is built *after* fusion, so any tensor whose only
//! consumer was a fused intermediate becomes dead and is naturally
//! avoided by downstream consumers (the planner still allocates a
//! span for declared tensors — a future "dead tensor sweep" can
//! reclaim those).
//!
//! Fusion is **opt-in**. Callers use `compile_fused(&chain)`
//! (clones + fuses + compiles) or `fuse(&mut chain)` directly when
//! they own the chain. The default `compile(&chain)` path remains
//! unchanged so unfused plans can still be exercised by tests and
//! conformance harnesses.
//!
//! ## Currently recognised patterns
//!
//! ### SwiGlu (`Silu(gate) → Mul(silu_out, up)` → `FusedSwiGlu(gate, up)`)
//!
//! The dominant gating activation in modern LLMs (LLaMA, Mistral,
//! Gemma, …). Two kernel calls collapse into one fused dispatch.
//! Mirrors `hologram_graph::fusion::float_fusion::try_fuse_swiglu`.
//!
//! Conditions checked:
//! - Silu's output tensor is consumed by exactly one node (the Mul).
//! - The Mul's other operand isn't (transitively) the Silu input.
//!   (Defensive — Silu is unary so its input can't equal its output;
//!   this guard catches pathological self-references.)
//!
//! ## Adding a fusion
//!
//! New patterns land here as a private `try_fuse_<name>` function
//! invoked by the public [`fuse`] sweep. Each helper returns `true`
//! when it rewrote the chain (so the outer loop knows to re-scan).
//! Conservative defaults: skip when in doubt; correctness > coverage.

use crate::chain::TransformChain;
use hologram_ops::SemanticOp;

/// Apply every recognised fusion pattern to `chain` repeatedly until
/// no further rewrites are possible. The pass is idempotent and
/// shape-preserving — each rewrite removes some nodes but never
/// introduces a new tensor.
pub fn fuse(chain: &mut TransformChain) {
    // Outer loop handles nested patterns and chains where one
    // fusion exposes another. Each pass scans the (possibly
    // shorter) `nodes` slice and rewrites at most one node before
    // restarting; a clean pass with no rewrites stops the loop.
    loop {
        let progressed = try_fuse_swiglu(chain);
        if !progressed {
            break;
        }
    }
}

/// Convenience: clone the chain, run [`fuse`], compile.
///
/// `compile(&chain)` and `compile_fused(&chain)` are observationally
/// equivalent on chains with no fusable patterns; on chains where a
/// pattern fires, `compile_fused` produces fewer `KernelCall`s.
pub fn compile_fused(
    chain: &crate::chain::TransformChain,
) -> Result<crate::plan::CompiledPlan, crate::error::PlanError> {
    let mut owned = chain.clone();
    fuse(&mut owned);
    crate::planner::compile(&owned)
}

/// Try to fuse a single Silu→Mul pair. Returns `true` if a rewrite
/// happened.
///
/// Pattern (concrete):
/// ```text
///   nodes[i]:   SemanticOp::Unary(_, Silu) inputs=[gate] outputs=[silu_out]
///   nodes[j]:   SemanticOp::Mul             inputs=[silu_out, up] outputs=[c]
/// ```
/// Rewrites `nodes[j]` to `SemanticOp::FusedSwiGlu` with
/// `inputs=[gate, up]` and removes `nodes[i]`.
fn try_fuse_swiglu(chain: &mut TransformChain) -> bool {
    // Find a Silu node whose output is consumed by exactly one Mul.
    let mut victim: Option<(usize, usize, /* silu_input_slot */ usize)> = None;
    'outer: for (i, n) in chain.nodes.iter().enumerate() {
        if !matches!(n.op, SemanticOp::Silu) {
            continue;
        }
        if n.inputs.len() != 1 || n.outputs.len() != 1 {
            continue;
        }
        let silu_out_tensor = n.outputs[0].tensor;

        // Count consumers of `silu_out_tensor`. Silu is anchored to
        // exactly one Mul iff there's a single consumer node *that
        // happens to be a Mul*.
        let mut consumers =
            chain.nodes.iter().enumerate().filter(|(k, m)| {
                *k != i && m.inputs.iter().any(|inp| inp.tensor == silu_out_tensor)
            });
        let Some((j, mul_node)) = consumers.next() else {
            continue;
        };
        if consumers.next().is_some() {
            continue; // multiple consumers — silu_out is shared, can't fuse.
        }
        let SemanticOp::Mul = mul_node.op else {
            continue;
        };
        if mul_node.inputs.len() != 2 {
            continue;
        }
        // Locate which input slot of the Mul is silu_out — the other
        // is `up`.
        let silu_slot = mul_node
            .inputs
            .iter()
            .position(|inp| inp.tensor == silu_out_tensor);
        let Some(slot) = silu_slot else {
            continue;
        };
        // Defensive: ensure the Mul's other operand isn't the same
        // tensor as Silu's input. Doesn't happen in well-formed
        // chains, but bail rather than rewrite something subtle.
        let other = mul_node.inputs[1 - slot].tensor;
        if other == n.inputs[0].tensor {
            continue 'outer;
        }
        victim = Some((i, j, slot));
        break;
    }

    let Some((silu_idx, mul_idx, silu_slot_in_mul)) = victim else {
        return false;
    };

    // Pull the gate and up tensor refs out before mutating `nodes`.
    let gate_ref = chain.nodes[silu_idx].inputs[0];
    let up_ref = chain.nodes[mul_idx].inputs[1 - silu_slot_in_mul];

    // Rewrite the Mul into a FusedSwiGlu (gate first, up second).
    let mul_node = &mut chain.nodes[mul_idx];
    mul_node.op = SemanticOp::FusedSwiGlu;
    mul_node.inputs.clear();
    mul_node.inputs.push(gate_ref);
    mul_node.inputs.push(up_ref);

    // Remove the now-orphaned Silu node.
    chain.nodes.remove(silu_idx);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::AddressRef;
    use crate::chain::{TransformChain, UnaryInputs};

    fn build_swiglu_chain() -> TransformChain {
        let mut b = TransformChain::builder();
        let gate = b.add_tensor(&[4], false);
        let up = b.add_tensor(&[4], false);
        let silu_out = b.add_tensor(&[4], false);
        let prod = b.add_tensor(&[4], false);
        b.push_unary(
            SemanticOp::Silu,
            UnaryInputs {
                input: AddressRef::of(gate),
                output: AddressRef::of(silu_out),
            },
        )
        .expect("silu unary");
        b.push_mul_forward_only(crate::chain::AddInputs {
            a: AddressRef::of(silu_out),
            b: AddressRef::of(up),
            c: AddressRef::of(prod),
        });
        b.build()
    }

    #[test]
    fn swiglu_pattern_fuses_to_one_node() {
        let mut chain = build_swiglu_chain();
        assert_eq!(chain.nodes.len(), 2);
        fuse(&mut chain);
        assert_eq!(chain.nodes.len(), 1);
        assert!(matches!(chain.nodes[0].op, SemanticOp::FusedSwiGlu));
        // FusedSwiGlu carries gate at slot 0 and up at slot 1.
        assert_eq!(chain.nodes[0].inputs[0].tensor.0, 0); // gate
        assert_eq!(chain.nodes[0].inputs[1].tensor.0, 1); // up
        assert_eq!(chain.nodes[0].outputs[0].tensor.0, 3); // prod
    }

    #[test]
    fn fuse_is_idempotent_on_unfusable_chain() {
        // Plain Add → no SwiGlu pattern.
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[3], false);
        let bb = b.add_tensor(&[3], false);
        let c = b.add_tensor(&[3], false);
        b.push_add(crate::chain::AddInputs {
            a: AddressRef::of(a),
            b: AddressRef::of(bb),
            c: AddressRef::of(c),
        });
        let mut chain = b.build();
        let before = chain.nodes.clone();
        fuse(&mut chain);
        assert_eq!(chain.nodes, before);
    }

    #[test]
    fn shared_silu_output_is_not_fused() {
        // If the Silu output feeds two consumers, fusion must skip —
        // otherwise the second consumer reads a stale tensor.
        let mut b = TransformChain::builder();
        let gate = b.add_tensor(&[4], false);
        let up = b.add_tensor(&[4], false);
        let silu_out = b.add_tensor(&[4], false);
        let mul_out = b.add_tensor(&[4], false);
        let add_out = b.add_tensor(&[4], false);
        b.push_unary(
            SemanticOp::Silu,
            UnaryInputs {
                input: AddressRef::of(gate),
                output: AddressRef::of(silu_out),
            },
        )
        .expect("silu unary");
        b.push_mul_forward_only(crate::chain::AddInputs {
            a: AddressRef::of(silu_out),
            b: AddressRef::of(up),
            c: AddressRef::of(mul_out),
        });
        // Second consumer of silu_out:
        b.push_add(crate::chain::AddInputs {
            a: AddressRef::of(silu_out),
            b: AddressRef::of(up),
            c: AddressRef::of(add_out),
        });
        let mut chain = b.build();
        let before_len = chain.nodes.len();
        fuse(&mut chain);
        assert_eq!(chain.nodes.len(), before_len);
    }

    #[test]
    fn compile_fused_produces_single_kernel_call_for_swiglu() {
        let chain = build_swiglu_chain();
        let unfused = crate::planner::compile(&chain).unwrap();
        let fused = compile_fused(&chain).unwrap();
        assert_eq!(unfused.forward.len(), 2); // Silu + Mul
        assert_eq!(fused.forward.len(), 1); // FusedSwiGlu
    }
}
