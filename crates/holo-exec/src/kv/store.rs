//! KV-lookup dispatch: routes `GraphOp` to the correct O(1) kernel.

use holo_core::op::PrimOp;
use holo_core::view::ElementWiseView;
use holo_graph::graph::GraphOp;

use crate::error::{ExecError, ExecResult};

/// Stateless dispatch table for O(1) graph operations.
///
/// All lookup tables are static in `holo-core`; this type is
/// zero-sized and provides only static methods.
pub struct KvStore;

impl KvStore {
    /// Apply a unary operation via `ElementWiseView`.
    #[must_use]
    pub fn apply_unary(view: &ElementWiseView, input: &[u8]) -> Vec<u8> {
        let mut output = vec![0u8; input.len()];
        view.apply_to(input, &mut output);
        output
    }

    /// Apply a binary `PrimOp` element-wise on two inputs.
    pub fn apply_binary(
        op: PrimOp,
        lhs: &[u8],
        rhs: &[u8],
    ) -> ExecResult<Vec<u8>> {
        if lhs.len() != rhs.len() {
            return Err(ExecError::LengthMismatch {
                expected: lhs.len(),
                actual: rhs.len(),
            });
        }
        let out: Vec<u8> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| op.apply_binary(a, b))
            .collect();
        Ok(out)
    }

    /// Dispatch a `GraphOp` given its input buffers.
    ///
    /// `Input` and `Constant` nodes are handled by the caller
    /// (they inject data into the arena directly).
    pub fn dispatch(
        op: &GraphOp,
        inputs: &[&[u8]],
    ) -> ExecResult<Vec<u8>> {
        match op {
            GraphOp::Output => Ok(inputs[0].to_vec()),
            GraphOp::Lut(_) | GraphOp::FusedView(_) => {
                let view = op.to_view().unwrap();
                Ok(Self::apply_unary(&view, inputs[0]))
            }
            GraphOp::Prim(p) if p.arity() == 1 => {
                let view = op.to_view().unwrap();
                Ok(Self::apply_unary(&view, inputs[0]))
            }
            GraphOp::Prim(p) => Self::apply_binary(*p, inputs[0], inputs[1]),
            GraphOp::Input | GraphOp::Constant(_) => {
                Ok(inputs.first().copied().unwrap_or(&[]).to_vec())
            }
            GraphOp::CallSubgraph(_) => {
                Err(ExecError::UnsupportedOp("CallSubgraph".into()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use holo_core::op::LutOp;

    #[test]
    fn unary_identity() {
        let view = ElementWiseView::identity();
        let input = vec![0, 1, 2, 255];
        assert_eq!(KvStore::apply_unary(&view, &input), input);
    }

    #[test]
    fn unary_increment() {
        let view = ElementWiseView::new(|x| x.wrapping_add(1));
        let input = vec![0, 1, 254, 255];
        assert_eq!(
            KvStore::apply_unary(&view, &input),
            vec![1, 2, 255, 0]
        );
    }

    #[test]
    fn dispatch_sigmoid() {
        let op = GraphOp::Lut(LutOp::Sigmoid);
        let input = vec![0u8, 128, 255];
        let result = KvStore::dispatch(&op, &[&input]).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn dispatch_relu() {
        let op = GraphOp::Lut(LutOp::Relu);
        let input = vec![0u8, 128, 255];
        let result = KvStore::dispatch(&op, &[&input]).unwrap();
        // Relu: values below threshold → 0, above → identity
        assert_eq!(result[0], LutOp::Relu.apply(0));
        assert_eq!(result[1], LutOp::Relu.apply(128));
    }

    #[test]
    fn dispatch_fused_view() {
        let view = ElementWiseView::new(|x| x.wrapping_mul(2));
        let op = GraphOp::FusedView(view);
        let input = vec![1, 2, 3];
        let result = KvStore::dispatch(&op, &[&input]).unwrap();
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn dispatch_prim_neg() {
        let op = GraphOp::Prim(PrimOp::Neg);
        let input = vec![0, 1, 128, 255];
        let result = KvStore::dispatch(&op, &[&input]).unwrap();
        assert_eq!(result[0], 0u8.wrapping_neg());
        assert_eq!(result[1], 1u8.wrapping_neg());
    }

    #[test]
    fn dispatch_prim_bnot() {
        let op = GraphOp::Prim(PrimOp::Bnot);
        let input = vec![0, 255, 0x0F];
        let result = KvStore::dispatch(&op, &[&input]).unwrap();
        assert_eq!(result, vec![255, 0, 0xF0]);
    }

    #[test]
    fn binary_add() {
        let result =
            KvStore::apply_binary(PrimOp::Add, &[10, 200], &[5, 100])
                .unwrap();
        assert_eq!(result, vec![15, 44]); // 200+100=300 mod 256=44
    }

    #[test]
    fn binary_sub() {
        let result =
            KvStore::apply_binary(PrimOp::Sub, &[10, 5], &[3, 10]).unwrap();
        assert_eq!(result[0], 7);
        assert_eq!(result[1], 251); // 5-10 mod 256
    }

    #[test]
    fn binary_xor() {
        let result =
            KvStore::apply_binary(PrimOp::Xor, &[0xFF, 0x0F], &[0x0F, 0xF0])
                .unwrap();
        assert_eq!(result, vec![0xF0, 0xFF]);
    }

    #[test]
    fn binary_length_mismatch() {
        let result =
            KvStore::apply_binary(PrimOp::Add, &[1, 2, 3], &[4, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn dispatch_output_copies() {
        let op = GraphOp::Output;
        let input = vec![42, 43, 44];
        let result = KvStore::dispatch(&op, &[&input]).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn dispatch_call_subgraph_unsupported() {
        use holo_graph::SubgraphId;
        let op = GraphOp::CallSubgraph(SubgraphId::new(0));
        let result = KvStore::dispatch(&op, &[]);
        assert!(result.is_err());
    }
}
