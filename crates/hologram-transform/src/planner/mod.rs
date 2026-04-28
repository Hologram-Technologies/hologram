//! Compile a `TransformChain` into a `CompiledPlan`.
//!
//! The planner is the *only* place in `hologram-transform` that may make
//! algorithmic decisions: kernel selection, workspace sizing, address
//! resolution, and backward-graph emission. Everything downstream (the
//! executor and kernels) is mechanical.
//!
//! Layout:
//! - `mod.rs`     — orchestration, address tables, public [`compile`].
//! - `forward.rs` — `SemanticOp` → `KernelCall` lowering for forward ops.
//! - `backward.rs` — `BackwardRule` → `KernelCall` emission for grads.

mod backward;
mod forward;

use crate::address::TensorId;
use crate::chain::{Tensor, TransformChain};
use crate::error::PlanError;
use crate::plan::{AddressTable, CompiledPlan, KernelCall, SlotSpan, WorkspaceLayout};

/// Compile a chain into a plan. Allocates only at compile time.
pub fn compile(chain: &TransformChain) -> Result<CompiledPlan, PlanError> {
    let address_table = build_address_table(chain);
    let workspace = WorkspaceLayout {
        total_elements: total_workspace(&address_table),
    };
    let forward = lower_forward(chain, &address_table)?;
    let backward = lower_backward(chain, &address_table)?;
    Ok(CompiledPlan {
        forward,
        backward,
        address_table,
        workspace,
    })
}

fn build_address_table(chain: &TransformChain) -> AddressTable {
    let n = chain.n_tensors();
    let mut spans = Vec::with_capacity(n);
    let mut grads = Vec::with_capacity(n);
    let mut cursor = 0usize;
    for t in &chain.tensors {
        let len = t.total_elements();
        spans.push(SlotSpan {
            offset: cursor,
            len,
        });
        cursor += len;
    }
    for t in &chain.tensors {
        grads.push(grad_slot(t.requires_grad, t.total_elements(), &mut cursor));
    }
    AddressTable {
        spans: spans.into_boxed_slice(),
        grads: grads.into_boxed_slice(),
    }
}

fn grad_slot(requires_grad: bool, len: usize, cursor: &mut usize) -> SlotSpan {
    if !requires_grad {
        return SlotSpan::empty(*cursor);
    }
    let s = SlotSpan {
        offset: *cursor,
        len,
    };
    *cursor += len;
    s
}

fn total_workspace(table: &AddressTable) -> usize {
    let last_v = table.spans.iter().map(|s| s.offset + s.len).max();
    let last_g = table.grads.iter().map(|s| s.offset + s.len).max();
    core::cmp::max(last_v.unwrap_or(0), last_g.unwrap_or(0))
}

fn lower_forward(
    chain: &TransformChain,
    table: &AddressTable,
) -> Result<Box<[KernelCall]>, PlanError> {
    let mut out = Vec::with_capacity(chain.nodes.len());
    for node in &chain.nodes {
        out.push(forward::lower_node(chain, table, node)?);
    }
    Ok(out.into_boxed_slice())
}

fn lower_backward(
    chain: &TransformChain,
    table: &AddressTable,
) -> Result<Box<[KernelCall]>, PlanError> {
    let mut out = Vec::new();
    for node in chain.nodes.iter().rev() {
        let Some(rule) = node.backward else { continue };
        backward::emit(chain, table, node, rule, &mut out)?;
    }
    Ok(out.into_boxed_slice())
}

/// Resolve a tensor by id or surface a typed planner error.
pub(super) fn require_tensor(chain: &TransformChain, id: TensorId) -> Result<&Tensor, PlanError> {
    chain.tensor(id).ok_or(PlanError::UnknownTensor(id.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::AddressRef;
    use crate::chain::{AddInputs, MatMulInputs};

    #[test]
    fn address_table_packs_tensors_then_grads() {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[4], true);
        let bb = b.add_tensor(&[4], false);
        let chain = b.build();
        let table = build_address_table(&chain);
        assert_eq!(table.span(a), SlotSpan { offset: 0, len: 4 });
        assert_eq!(table.span(bb), SlotSpan { offset: 4, len: 4 });
        assert_eq!(table.grad(a), SlotSpan { offset: 8, len: 4 });
        assert_eq!(table.grad(bb).len, 0);
    }

    #[test]
    fn add_chain_compiles_with_forward_and_backward() {
        let plan = compile_simple_add(true).unwrap();
        assert_eq!(plan.forward_len(), 1);
        assert_eq!(plan.backward_len(), 1);
        assert!(matches!(plan.forward[0], KernelCall::Add(_)));
        assert!(matches!(plan.backward[0], KernelCall::AddGrad(_)));
    }

    #[test]
    fn add_chain_without_grad_skips_backward() {
        let plan = compile_simple_add(false).unwrap();
        assert_eq!(plan.forward_len(), 1);
        assert_eq!(plan.backward_len(), 0);
    }

    #[test]
    fn matmul_chain_emits_both_grad_kernels() {
        let plan = compile_simple_matmul().unwrap();
        assert_eq!(plan.forward_len(), 1);
        assert_eq!(plan.backward_len(), 2);
        assert!(matches!(plan.backward[0], KernelCall::MatMulGradA(_)));
        assert!(matches!(plan.backward[1], KernelCall::MatMulGradB(_)));
    }

    #[test]
    fn matmul_shape_mismatch_is_caught_at_build_time() {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[2, 3], true);
        let bb = b.add_tensor(&[5, 4], true);
        let c = b.add_tensor(&[2, 4], true);
        let err = b
            .push_matmul(MatMulInputs {
                a: AddressRef::of(a),
                b: AddressRef::of(bb),
                c: AddressRef::of(c),
            })
            .unwrap_err();
        assert!(matches!(err, PlanError::ShapeMismatch { op: "matmul", .. }));
    }

    fn compile_simple_add(with_grad: bool) -> Result<CompiledPlan, PlanError> {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[4], with_grad);
        let bb = b.add_tensor(&[4], with_grad);
        let c = b.add_tensor(&[4], with_grad);
        let ins = AddInputs {
            a: AddressRef::of(a),
            b: AddressRef::of(bb),
            c: AddressRef::of(c),
        };
        if with_grad {
            b.push_add(ins);
        } else {
            b.push_add_forward_only(ins);
        }
        compile(&b.build())
    }

    fn compile_simple_matmul() -> Result<CompiledPlan, PlanError> {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[2, 3], true);
        let bb = b.add_tensor(&[3, 5], true);
        let c = b.add_tensor(&[2, 5], true);
        b.push_matmul(MatMulInputs {
            a: AddressRef::of(a),
            b: AddressRef::of(bb),
            c: AddressRef::of(c),
        })?;
        compile(&b.build())
    }
}
