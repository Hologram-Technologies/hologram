//! Persistent constrained runner for bounded-memory tape execution.
//!
//! Analogous to [`InferenceSession`](crate::mmap::InferenceSession) but
//! enforces the constraints defined by a [`ConstrainedProfile`]: kernel
//! allowlist validation, weight window bounds, and memory caps.

use hologram_archive::loader::plan::LoadedPlan;

use crate::constrained::profile::{ConstrainedProfile, WeightPolicy};
use crate::constrained::tape_subset::validate_constrained_tape;
use crate::constrained::weight_window::WeightWindow;
use crate::error::{ExecError, ExecResult};
use crate::eval::executor::GraphInputs;
use crate::eval::executor::GraphOutputs;
use crate::tape::EnumTape;

/// Persistent execution object for constrained tape execution.
///
/// Validates the tape against the profile at construction time. Reuses
/// the weight cache across calls. Enforces memory limits per the profile.
///
/// When `weight_policy` is `BoundedWindow`, the runner uses per-instruction
/// weight window ensure/evict via [`EnumTape::execute_with_weight_window`].
/// Other policies use the standard `execute_direct` path.
pub struct ConstrainedRunner<'a> {
    tape: &'a EnumTape,
    plan: &'a LoadedPlan,
    profile: ConstrainedProfile,
    weight_cache: parking_lot::RwLock<crate::kv::WeightCache>,
    weight_window: WeightWindow,
    step_count: u64,
    peak_activation_bytes: usize,
}

impl<'a> ConstrainedRunner<'a> {
    /// Create a new constrained runner.
    ///
    /// Validates the tape against the profile's kernel allowlist. Returns
    /// `Err(ConstrainedViolation)` if the tape contains disallowed kernels.
    pub fn new(
        tape: &'a EnumTape,
        plan: &'a LoadedPlan,
        profile: ConstrainedProfile,
    ) -> ExecResult<Self> {
        validate_constrained_tape(tape, &profile)?;

        let weight_window = WeightWindow::new(profile.max_weight_bytes);

        Ok(Self {
            tape,
            plan,
            profile,
            weight_cache: parking_lot::RwLock::new(crate::kv::WeightCache::new()),
            weight_window,
            step_count: 0,
            peak_activation_bytes: 0,
        })
    }

    /// Execute the tape with the given inputs.
    ///
    /// For `BoundedWindow` policy, uses per-instruction weight window
    /// ensure/evict and tracks peak activation memory. For other policies,
    /// dispatches via `execute_direct`.
    pub fn run(&mut self, inputs: &GraphInputs) -> ExecResult<GraphOutputs> {
        let sg = self.plan.graph();
        let weights = self.plan.weights();
        let compiled_dtypes = sg.node_dtypes_map();
        let compiled_shapes = sg.node_shapes_map();

        let mut arena = crate::buffer::BufferArena::with_capacity(sg.nodes.len());
        crate::mmap::seed_arena(
            sg,
            weights,
            &compiled_dtypes,
            &compiled_shapes,
            inputs,
            &mut arena,
        )?;

        if self.step_count == 0 {
            self.tape.prewarm_arena(&mut arena);
        }

        let tape_ctx = crate::tape::TapeContext::new(&sg.constants, weights, &self.weight_cache);

        if self.profile.weight_policy == WeightPolicy::BoundedWindow {
            // Per-instruction weight window path.
            let peak = self.tape.execute_with_weight_window(
                &mut arena,
                &tape_ctx,
                &mut self.weight_window,
            )?;

            // Enforce activation memory cap.
            if peak > self.profile.max_activation_bytes {
                return Err(ExecError::ConstrainedViolation(format!(
                    "peak activation memory {} bytes exceeds cap {} bytes",
                    peak, self.profile.max_activation_bytes
                )));
            }
            if peak > self.peak_activation_bytes {
                self.peak_activation_bytes = peak;
            }
        } else {
            // Standard dispatch for other weight policies.
            self.tape.execute_direct(&mut arena, &tape_ctx)?;
        }

        self.step_count += 1;
        crate::mmap::collect_outputs(sg, &mut arena)
    }

    /// Number of inference steps executed.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Current weight window usage in bytes.
    #[must_use]
    pub fn weight_usage(&self) -> usize {
        self.weight_window.current_usage()
    }

    /// Peak activation memory observed across all runs (bytes).
    #[must_use]
    pub fn peak_activation_bytes(&self) -> usize {
        self.peak_activation_bytes
    }
}

/// One-shot convenience wrapper for constrained execution.
pub fn execute_constrained(
    tape: &EnumTape,
    plan: &LoadedPlan,
    inputs: &GraphInputs,
    profile: ConstrainedProfile,
) -> ExecResult<GraphOutputs> {
    let mut runner = ConstrainedRunner::new(tape, plan, profile)?;
    runner.run(inputs)
}
