//! Cross-backend conformance harness.
//!
//! Runs the same [`CompiledPlan`] on two [`CanonicalBackend`]s and
//! compares their workspace bytes within a tolerance. The reference
//! side is typically [`CpuBackend`] (canonical CPU kernels from
//! `hologram-ops`); the candidate side is whatever device backend is
//! being validated (Metal, WebGPU, Atlas, …).
//!
//! The harness is the standard contract Phase 3.5 backends are
//! validated against: a backend is "conformant" iff every
//! [`KernelCall`] in a representative plan produces values within
//! tolerance of the reference. Per ADR-050 the canonical kernels are
//! the *semantic baseline* — backends that diverge are wrong, not
//! "alternative implementations".
//!
//! [`CanonicalBackend`]: crate::backend::CanonicalBackend
//! [`CompiledPlan`]: crate::plan::CompiledPlan
//! [`CpuBackend`]: crate::backend::CpuBackend
//! [`KernelCall`]: crate::plan::KernelCall

use crate::backend::CanonicalBackend;
use crate::buffer::BufferSet;
use crate::error::ExecError;
use crate::plan::CompiledPlan;

/// Tolerances for [`compare`].
///
/// Element-wise: a comparison passes iff
/// `|a - b| <= max(abs, rel * max(|a|, |b|))`.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Absolute tolerance.
    pub abs: f32,
    /// Relative tolerance (against the larger of the two magnitudes).
    pub rel: f32,
}

impl Tolerance {
    /// Tight tolerance suitable for f32 reference vs. f32 candidate
    /// running on the same algorithmic path (≈3 ULPs at unit scale).
    pub const TIGHT: Self = Self {
        abs: 1e-6,
        rel: 1e-6,
    };

    /// Loose tolerance suitable for cross-precision comparisons (e.g.
    /// candidate runs in fp16 internally and accumulates in fp32).
    pub const LOOSE: Self = Self {
        abs: 1e-3,
        rel: 1e-3,
    };

    /// Build a custom tolerance.
    #[must_use]
    pub const fn new(abs: f32, rel: f32) -> Self {
        Self { abs, rel }
    }

    #[inline]
    fn passes(self, a: f32, b: f32) -> bool {
        let diff = (a - b).abs();
        if !diff.is_finite() {
            // NaN vs NaN: treat as equal. Inf vs Inf with same sign:
            // also equal. Anything else with non-finite diff: fail.
            return (a.is_nan() && b.is_nan())
                || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum());
        }
        let scale = a.abs().max(b.abs());
        diff <= self.abs.max(self.rel * scale)
    }
}

/// Detail of the first divergent element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mismatch {
    /// Index in the workspace where the divergence was found.
    pub index: usize,
    /// Reference value at that index.
    pub reference: f32,
    /// Candidate value at that index.
    pub candidate: f32,
    /// Largest absolute element-wise diff seen across the workspace.
    pub max_abs_diff: f32,
}

/// Result of a conformance comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Conformance {
    /// All workspace elements pass within tolerance.
    Match {
        /// Largest absolute diff observed (may be > 0 even when all
        /// pass).
        max_abs_diff: f32,
    },
    /// At least one element diverged. Carries the first failing index.
    Diverged(Mismatch),
}

impl Conformance {
    /// `true` when the candidate matched the reference everywhere.
    #[inline]
    #[must_use]
    pub fn is_match(self) -> bool {
        matches!(self, Conformance::Match { .. })
    }
}

/// Run `plan.forward` on both backends starting from the same
/// pre-seeded `BufferSet` contents, and compare the resulting
/// workspaces.
///
/// `seed` is invoked twice — once per backend — to let callers stamp
/// the initial values (forward inputs, weights, …) into a fresh
/// [`BufferSet`]. The seeded contents must be deterministic; otherwise
/// the comparison is meaningless.
///
/// Returns `Conformance::Match` iff the two backends produced
/// element-wise equivalent workspaces under `tolerance`.
pub fn check_forward<R, C, S>(
    plan: &CompiledPlan,
    reference: &mut R,
    candidate: &mut C,
    mut seed: S,
    tolerance: Tolerance,
) -> Result<Conformance, ExecError>
where
    R: CanonicalBackend,
    C: CanonicalBackend,
    S: FnMut(&mut BufferSet),
{
    let ref_storage = run_one(plan, reference, &mut seed, /* backward = */ false)?;
    let cand_storage = run_one(plan, candidate, &mut seed, false)?;
    Ok(compare(&ref_storage, &cand_storage, tolerance))
}

/// Like [`check_forward`] but also runs `plan.backward` after
/// `plan.forward`. Callers are expected to seed any required
/// upstream gradient slots (`dC`, …) inside `seed` before backward
/// runs.
pub fn check_forward_then_backward<R, C, S>(
    plan: &CompiledPlan,
    reference: &mut R,
    candidate: &mut C,
    mut seed: S,
    tolerance: Tolerance,
) -> Result<Conformance, ExecError>
where
    R: CanonicalBackend,
    C: CanonicalBackend,
    S: FnMut(&mut BufferSet),
{
    let ref_storage = run_one(plan, reference, &mut seed, true)?;
    let cand_storage = run_one(plan, candidate, &mut seed, true)?;
    Ok(compare(&ref_storage, &cand_storage, tolerance))
}

/// Element-wise compare two workspaces under `tolerance`. Public so
/// callers with bespoke run schemes (e.g. step-by-step replay) can
/// reuse it.
#[must_use]
pub fn compare(reference: &[f32], candidate: &[f32], tolerance: Tolerance) -> Conformance {
    debug_assert_eq!(reference.len(), candidate.len());
    let mut max_abs_diff = 0.0_f32;
    let mut first_fail: Option<(usize, f32, f32)> = None;
    for (i, (&r, &c)) in reference.iter().zip(candidate.iter()).enumerate() {
        let d = (r - c).abs();
        if d.is_finite() && d > max_abs_diff {
            max_abs_diff = d;
        }
        if !tolerance.passes(r, c) && first_fail.is_none() {
            first_fail = Some((i, r, c));
        }
    }
    match first_fail {
        Some((index, reference_v, candidate_v)) => Conformance::Diverged(Mismatch {
            index,
            reference: reference_v,
            candidate: candidate_v,
            max_abs_diff,
        }),
        None => Conformance::Match { max_abs_diff },
    }
}

fn run_one<B, S>(
    plan: &CompiledPlan,
    backend: &mut B,
    seed: &mut S,
    backward: bool,
) -> Result<Vec<f32>, ExecError>
where
    B: CanonicalBackend,
    S: FnMut(&mut BufferSet),
{
    let mut buffers = BufferSet::for_plan(plan);
    seed(&mut buffers);
    backend.run(buffers.storage_mut(), &plan.forward)?;
    if backward {
        backend.run(buffers.storage_mut(), &plan.backward)?;
    }
    backend.flush()?;
    Ok(buffers.storage_mut().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::plan::{AddCall, AddressTable, KernelCall, SlotSpan, WorkspaceLayout};

    fn tiny_add_plan() -> CompiledPlan {
        // workspace[0] + workspace[1] -> workspace[2].
        CompiledPlan {
            forward: Box::new([KernelCall::Add(AddCall {
                a: SlotSpan { offset: 0, len: 1 },
                b: SlotSpan { offset: 1, len: 1 },
                c: SlotSpan { offset: 2, len: 1 },
            })]),
            backward: Box::new([]),
            address_table: AddressTable {
                spans: Box::new([]),
                grads: Box::new([]),
            },
            workspace: WorkspaceLayout { total_elements: 3 },
        }
    }

    #[test]
    fn cpu_backend_matches_itself() {
        let plan = tiny_add_plan();
        let mut a = CpuBackend::new();
        let mut b = CpuBackend::new();
        let res = check_forward(
            &plan,
            &mut a,
            &mut b,
            |buf| {
                buf.write_span(SlotSpan { offset: 0, len: 1 }, &[1.5]);
                buf.write_span(SlotSpan { offset: 1, len: 1 }, &[2.5]);
            },
            Tolerance::TIGHT,
        )
        .unwrap();
        assert!(res.is_match(), "{:?}", res);
    }

    /// A backend that deliberately corrupts results — proves the
    /// harness actually catches divergence, not just rubber-stamps.
    struct WrongBackend;
    impl CanonicalBackend for WrongBackend {
        fn dispatch(&mut self, storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError> {
            // Run the real op, then perturb the destination of every Add.
            hologram_ops::dispatch(storage, call);
            if let KernelCall::Add(c) = call {
                storage[c.c.offset] += 0.1;
            }
            Ok(())
        }
        fn name(&self) -> &'static str {
            "wrong"
        }
    }

    #[test]
    fn divergent_backend_is_caught() {
        let plan = tiny_add_plan();
        let mut reference = CpuBackend::new();
        let mut wrong = WrongBackend;
        let res = check_forward(
            &plan,
            &mut reference,
            &mut wrong,
            |buf| {
                buf.write_span(SlotSpan { offset: 0, len: 1 }, &[1.5]);
                buf.write_span(SlotSpan { offset: 1, len: 1 }, &[2.5]);
            },
            Tolerance::TIGHT,
        )
        .unwrap();
        match res {
            Conformance::Diverged(m) => {
                assert_eq!(m.index, 2);
                assert!((m.reference - 4.0).abs() < 1e-6);
                assert!((m.candidate - 4.1).abs() < 1e-6);
                assert!(m.max_abs_diff > 0.05);
            }
            other => panic!("expected divergence, got {:?}", other),
        }
    }

    #[test]
    fn nan_compares_equal_under_tolerance() {
        let r = [f32::NAN, 1.0];
        let c = [f32::NAN, 1.0];
        assert!(matches!(
            compare(&r, &c, Tolerance::TIGHT),
            Conformance::Match { .. }
        ));
    }

    #[test]
    fn relative_tolerance_kicks_in_at_large_magnitudes() {
        // 1e6 vs 1e6 + 0.5: abs diff 0.5 > TIGHT.abs (1e-6) but well
        // within TIGHT.rel (1e-6 * 1e6 = 1.0).
        let r = [1.0e6_f32];
        let c = [1.000_000_5e6_f32];
        assert!(matches!(
            compare(&r, &c, Tolerance::TIGHT),
            Conformance::Match { .. }
        ));
    }
}
