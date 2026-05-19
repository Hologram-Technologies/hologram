//! Tape validation for constrained execution.
//!
//! Walks a tape's instructions and rejects kernels not permitted
//! by the constrained profile's allowlist and custom-op policy.

use crate::constrained::profile::ConstrainedProfile;
use crate::error::{ExecError, ExecResult};
use crate::tape::EnumTape;

use super::profile::KernelDiscriminant;

/// Validate that every instruction in the tape is permitted by the profile.
///
/// Returns `Ok(())` if all kernels pass, or `Err(ConstrainedViolation)` with
/// the first rejected kernel's discriminant name.
pub fn validate_constrained_tape(tape: &EnumTape, profile: &ConstrainedProfile) -> ExecResult<()> {
    for (idx, instr) in tape.instructions.iter().enumerate() {
        let disc = KernelDiscriminant::from_kernel(&instr.kernel);

        // Custom ops require explicit opt-in.
        if disc == KernelDiscriminant::Custom && !profile.allow_custom_ops {
            return Err(ExecError::ConstrainedViolation(format!(
                "instruction {idx}: Custom ops not allowed in constrained profile"
            )));
        }

        // Check kernel allowlist (if configured).
        if let Some(ref allowlist) = profile.kernel_allowlist {
            if !allowlist.is_allowed(&instr.kernel) {
                return Err(ExecError::ConstrainedViolation(format!(
                    "instruction {idx}: kernel {disc:?} not in constrained allowlist"
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained::profile::KernelAllowlist;
    use crate::tape::{FastPath, ShapeSource, TapeInstruction, TapeKernel};
    use std::collections::HashSet;

    fn make_tape(kernels: Vec<TapeKernel>) -> EnumTape {
        let instructions: Vec<TapeInstruction> = kernels
            .into_iter()
            .enumerate()
            .map(|(i, kernel)| TapeInstruction {
                kernel,
                input_indices: smallvec::smallvec![],
                output_idx: i as u32,
                output_byte_hint: 0,
                output_elem_size: 4,
                weight_offset_hint: 0,
                passthrough: false,
                can_reuse_input: false,
                output_meta: None,
                fast_path: FastPath::default(),
                shape_source: ShapeSource::default(),
            })
            .collect();
        EnumTape {
            instructions,
            level_offsets: vec![0],
            consumer_counts: vec![],
            level_weight_ranges: vec![],
            checkpoint_map: std::collections::HashMap::new(),
            checkpoint_enabled: false,
            slot_assignments: vec![],
            heap_only_eviction: false,
            n_slots: 0,
        }
    }

    #[test]
    fn no_allowlist_accepts_everything() {
        let tape = make_tape(vec![TapeKernel::InlineAdd, TapeKernel::Output]);
        let profile = ConstrainedProfile::default();
        assert!(validate_constrained_tape(&tape, &profile).is_ok());
    }

    #[test]
    fn allowlist_accepts_permitted_kernels() {
        let tape = make_tape(vec![TapeKernel::InlineAdd, TapeKernel::InlineMul]);
        let profile = ConstrainedProfile {
            kernel_allowlist: Some(KernelAllowlist::compute()),
            ..Default::default()
        };
        assert!(validate_constrained_tape(&tape, &profile).is_ok());
    }

    #[test]
    fn allowlist_rejects_unpermitted_kernels() {
        let tape = make_tape(vec![TapeKernel::KvWrite {
            layer: 0,
            n_kv_heads: 1,
            head_dim: 64,
            is_key: true,
            heads_first: false,
        }]);
        let profile = ConstrainedProfile {
            kernel_allowlist: Some(KernelAllowlist::compute()),
            ..Default::default()
        };
        let err = validate_constrained_tape(&tape, &profile).unwrap_err();
        match err {
            ExecError::ConstrainedViolation(msg) => {
                assert!(msg.contains("KvWrite"));
            }
            other => panic!("expected ConstrainedViolation, got {other:?}"),
        }
    }

    #[test]
    fn custom_ops_rejected_by_default() {
        use hologram_graph::constant::ConstantStore;
        let handler: crate::kv::CustomHandler = std::sync::Arc::new(
            |_inputs: &[&[u8]], _constants: &ConstantStore| -> crate::error::ExecResult<Vec<u8>> {
                Ok(vec![])
            },
        );
        let tape = make_tape(vec![TapeKernel::Custom(handler)]);
        let profile = ConstrainedProfile::default();
        let err = validate_constrained_tape(&tape, &profile).unwrap_err();
        match err {
            ExecError::ConstrainedViolation(msg) => {
                assert!(msg.contains("Custom"));
            }
            other => panic!("expected ConstrainedViolation, got {other:?}"),
        }
    }

    #[test]
    fn inference_preset_includes_ai_kernels() {
        let allowlist = KernelAllowlist::inference();
        let tape = make_tape(vec![
            TapeKernel::InlineRmsNorm {
                size: 128,
                epsilon: f32::to_bits(1e-5),
            },
            TapeKernel::InlineSoftmax { size: 64 },
            TapeKernel::InlineRoPE {
                dim: 64,
                base: f32::to_bits(10000.0),
                n_heads: 8,
            },
            TapeKernel::Output,
        ]);
        let profile = ConstrainedProfile {
            kernel_allowlist: Some(allowlist),
            ..Default::default()
        };
        assert!(validate_constrained_tape(&tape, &profile).is_ok());
    }

    #[test]
    fn compute_preset_rejects_kv_ops() {
        let allowlist = KernelAllowlist::compute();
        assert!(!allowlist.is_allowed(&TapeKernel::KvRead {
            layer: 0,
            n_kv_heads: 1,
            head_dim: 64,
            heads_first: false,
        }));
    }

    #[test]
    fn empty_tape_passes() {
        let tape = make_tape(vec![]);
        let profile = ConstrainedProfile {
            kernel_allowlist: Some(KernelAllowlist::from_discriminants(HashSet::new())),
            ..Default::default()
        };
        assert!(validate_constrained_tape(&tape, &profile).is_ok());
    }
}
