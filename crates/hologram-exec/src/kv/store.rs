//! KV-lookup dispatch: routes `GraphOp` to the correct O(1) kernel.

use hologram_core::op::PrimOp;
use hologram_core::view::ElementWiseView;
use hologram_graph::constant::{ConstantData, ConstantStore};
use hologram_graph::graph::GraphOp;

use crate::error::{ExecError, ExecResult};
use crate::kv::registry::CustomOpRegistry;
use crate::lut_gemm::matmul::{lut_gemm_4bit, lut_gemm_8bit};
use crate::lut_gemm::quantize::{QuantizedWeights4, QuantizedWeights8};

/// Stateless dispatch table for O(1) graph operations.
///
/// All lookup tables are static in `hologram-core`; this type is
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
    pub fn apply_binary(op: PrimOp, lhs: &[u8], rhs: &[u8]) -> ExecResult<Vec<u8>> {
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
        registry: Option<&CustomOpRegistry>,
    ) -> ExecResult<Vec<u8>> {
        Self::dispatch_with_constants(op, inputs, &ConstantStore::new(), registry)
    }

    /// Dispatch with access to the graph's constant store.
    ///
    /// MatMulLut ops resolve their quantized weights from constants.
    /// Pass a `CustomOpRegistry` to enable custom op dispatch.
    pub fn dispatch_with_constants(
        op: &GraphOp,
        inputs: &[&[u8]],
        constants: &ConstantStore,
        registry: Option<&CustomOpRegistry>,
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
            GraphOp::CallSubgraph(_) => Err(ExecError::UnsupportedOp("CallSubgraph".into())),
            GraphOp::MatMulLut4(cid) => dispatch_lut_gemm_4(inputs[0], *cid, constants),
            GraphOp::MatMulLut8(cid) => dispatch_lut_gemm_8(inputs[0], *cid, constants),
            GraphOp::BatchMatMulLut4(cid) => dispatch_lut_gemm_4(inputs[0], *cid, constants),
            GraphOp::BatchMatMulLut8(cid) => dispatch_lut_gemm_8(inputs[0], *cid, constants),
            GraphOp::Custom { id, .. } => registry
                .ok_or_else(|| ExecError::UnsupportedOp(format!("custom op {}", id.raw())))?
                .dispatch(*id, inputs, constants),
        }
    }
}

/// Resolve constant and run 4-bit LUT-GEMM.
fn dispatch_lut_gemm_4(
    activation_bytes: &[u8],
    cid: hologram_graph::constant::ConstantId,
    constants: &ConstantStore,
) -> ExecResult<Vec<u8>> {
    let weight_bytes = resolve_constant_bytes(cid, constants)?;
    let qw = rkyv::from_bytes::<QuantizedWeights4, rkyv::rancor::Error>(weight_bytes)
        .map_err(|e| ExecError::InvalidQuantization(e.to_string()))?;
    let activations = cast_f32(activation_bytes)?;
    let m = activations.len() / qw.rows as usize;
    let n = qw.cols as usize;
    let mut output = vec![0.0f32; m * n];
    lut_gemm_4bit(activations, &qw, &mut output);
    Ok(bytemuck::cast_slice(&output).to_vec())
}

/// Resolve constant and run 8-bit LUT-GEMM.
fn dispatch_lut_gemm_8(
    activation_bytes: &[u8],
    cid: hologram_graph::constant::ConstantId,
    constants: &ConstantStore,
) -> ExecResult<Vec<u8>> {
    let weight_bytes = resolve_constant_bytes(cid, constants)?;
    let qw = rkyv::from_bytes::<QuantizedWeights8, rkyv::rancor::Error>(weight_bytes)
        .map_err(|e| ExecError::InvalidQuantization(e.to_string()))?;
    let activations = cast_f32(activation_bytes)?;
    let m = activations.len() / qw.rows as usize;
    let n = qw.cols as usize;
    let mut output = vec![0.0f32; m * n];
    lut_gemm_8bit(activations, &qw, &mut output);
    Ok(bytemuck::cast_slice(&output).to_vec())
}

/// Resolve a constant ID to its raw bytes.
fn resolve_constant_bytes(
    cid: hologram_graph::constant::ConstantId,
    constants: &ConstantStore,
) -> ExecResult<&[u8]> {
    let data = constants
        .get(cid)
        .ok_or(ExecError::ConstantNotFound(cid.raw()))?;
    match data {
        ConstantData::Bytes(bytes) => Ok(bytes),
        ConstantData::Deferred { .. } => Err(ExecError::UnsupportedOp("deferred constant".into())),
    }
}

/// Cast `&[u8]` to `&[f32]` via bytemuck.
fn cast_f32(bytes: &[u8]) -> ExecResult<&[f32]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| ExecError::ShapeMismatch {
        expected: "f32-aligned bytes".into(),
        actual: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::LutOp;

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
        assert_eq!(KvStore::apply_unary(&view, &input), vec![1, 2, 255, 0]);
    }

    #[test]
    fn dispatch_sigmoid() {
        let op = GraphOp::Lut(LutOp::Sigmoid);
        let input = vec![0u8, 128, 255];
        let result = KvStore::dispatch(&op, &[&input], None).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn dispatch_relu() {
        let op = GraphOp::Lut(LutOp::Relu);
        let input = vec![0u8, 128, 255];
        let result = KvStore::dispatch(&op, &[&input], None).unwrap();
        assert_eq!(result[0], LutOp::Relu.apply(0));
        assert_eq!(result[1], LutOp::Relu.apply(128));
    }

    #[test]
    fn dispatch_fused_view() {
        let view = ElementWiseView::new(|x| x.wrapping_mul(2));
        let op = GraphOp::FusedView(view);
        let input = vec![1, 2, 3];
        let result = KvStore::dispatch(&op, &[&input], None).unwrap();
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn dispatch_prim_neg() {
        let op = GraphOp::Prim(PrimOp::Neg);
        let input = vec![0, 1, 128, 255];
        let result = KvStore::dispatch(&op, &[&input], None).unwrap();
        assert_eq!(result[0], 0u8.wrapping_neg());
        assert_eq!(result[1], 1u8.wrapping_neg());
    }

    #[test]
    fn dispatch_prim_bnot() {
        let op = GraphOp::Prim(PrimOp::Bnot);
        let input = vec![0, 255, 0x0F];
        let result = KvStore::dispatch(&op, &[&input], None).unwrap();
        assert_eq!(result, vec![255, 0, 0xF0]);
    }

    #[test]
    fn binary_add() {
        let result = KvStore::apply_binary(PrimOp::Add, &[10, 200], &[5, 100]).unwrap();
        assert_eq!(result, vec![15, 44]); // 200+100=300 mod 256=44
    }

    #[test]
    fn binary_sub() {
        let result = KvStore::apply_binary(PrimOp::Sub, &[10, 5], &[3, 10]).unwrap();
        assert_eq!(result[0], 7);
        assert_eq!(result[1], 251); // 5-10 mod 256
    }

    #[test]
    fn binary_xor() {
        let result = KvStore::apply_binary(PrimOp::Xor, &[0xFF, 0x0F], &[0x0F, 0xF0]).unwrap();
        assert_eq!(result, vec![0xF0, 0xFF]);
    }

    #[test]
    fn binary_length_mismatch() {
        let result = KvStore::apply_binary(PrimOp::Add, &[1, 2, 3], &[4, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn dispatch_output_copies() {
        let op = GraphOp::Output;
        let input = vec![42, 43, 44];
        let result = KvStore::dispatch(&op, &[&input], None).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn dispatch_call_subgraph_unsupported() {
        use hologram_graph::SubgraphId;
        let op = GraphOp::CallSubgraph(SubgraphId::new(0));
        let result = KvStore::dispatch(&op, &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn dispatch_matmul_lut4() {
        use crate::lut_gemm::quantize::quantize_4bit;

        let k = 4usize;
        let n = 2usize;
        let weights = vec![1.0f32; k * n];
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let qw_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).unwrap().to_vec();

        let mut constants = ConstantStore::new();
        let cid = constants.insert(ConstantData::Bytes(qw_bytes));

        let activations = [1.0f32, 2.0, 3.0, 4.0]; // 1×4
        let act_bytes: &[u8] = bytemuck::cast_slice(&activations);

        let op = GraphOp::MatMulLut4(cid);
        let result = KvStore::dispatch_with_constants(&op, &[act_bytes], &constants, None).unwrap();
        let output: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(output.len(), n);
        // sum(1+2+3+4)=10, all weights=1.0
        for &v in output {
            assert!((v - 10.0).abs() < 0.5, "got {v}");
        }
    }

    #[test]
    fn dispatch_matmul_lut8() {
        use crate::lut_gemm::quantize::quantize_8bit;

        let k = 4usize;
        let n = 2usize;
        let weights = vec![2.0f32; k * n];
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let qw_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).unwrap().to_vec();

        let mut constants = ConstantStore::new();
        let cid = constants.insert(ConstantData::Bytes(qw_bytes));

        let activations = [1.0f32, 1.0, 1.0, 1.0];
        let act_bytes: &[u8] = bytemuck::cast_slice(&activations);

        let op = GraphOp::MatMulLut8(cid);
        let result = KvStore::dispatch_with_constants(&op, &[act_bytes], &constants, None).unwrap();
        let output: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(output.len(), n);
        // sum(1*2 * 4) = 8
        for &v in output {
            assert!((v - 8.0).abs() < 0.1, "got {v}");
        }
    }

    #[test]
    fn dispatch_matmul_lut_missing_constant() {
        use hologram_graph::constant::ConstantId;
        let op = GraphOp::MatMulLut4(ConstantId::new(99));
        let act = [1.0f32; 4];
        let act_bytes: &[u8] = bytemuck::cast_slice(&act);
        let result =
            KvStore::dispatch_with_constants(&op, &[act_bytes], &ConstantStore::new(), None);
        assert!(result.is_err());
    }
}
