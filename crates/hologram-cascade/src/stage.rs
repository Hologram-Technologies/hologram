//! Cascade stage types and transition logic.
//!
//! The cascade has 7 stages matching the uor-foundation ontology:
//! `stage_initialization` (Ω⁰) through `stage_convergence` (π).
//!
//! Stage dispatch uses a 7-entry function pointer table for O(1) transitions
//! with zero branching in the hot loop.

use hologram_core::op::RingLevel;
use hologram_graph::graph::node::NodeId;
use hologram_graph::fusion::FusionStats;

use crate::liveness::LivenessInterval;
use crate::qedl::{EncodingId, QedlBoundary};
use crate::workspace::WorkspaceLayout;

/// Cascade stage discriminant. `#[repr(u8)]` for O(1) indexed jump table dispatch.
///
/// Maps 1:1 to the cascade stage constants in `uor_foundation::kernel::cascade`.
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum CascadeStage {
    /// Stage 0 — Initialization (Ω⁰): initialize state vector from CompileUnit.
    Init = 0,
    /// Stage 1 — Declare (Ω¹): dispatch resolver selection.
    Declare = 1,
    /// Stage 2 — Factorize (Ω²): ground to valid ring address.
    Factorize = 2,
    /// Stage 3 — Resolve (Ω³): resolve scheduling constraints.
    Resolve = 3,
    /// Stage 4 — Attest (Ω⁴): accumulate without contradiction.
    Attest = 4,
    /// Stage 5 — Extract (Ω⁵): extract coherent output.
    Extract = 5,
    /// Stage 6 — Convergence (π): terminal state.
    Converge = 6,
}

impl CascadeStage {
    /// Total number of cascade stages.
    pub const COUNT: usize = 7;

    /// Advance to the next stage. Wraps at `Converge`.
    #[inline]
    pub const fn next(self) -> Self {
        match self {
            Self::Init => Self::Declare,
            Self::Declare => Self::Factorize,
            Self::Factorize => Self::Resolve,
            Self::Resolve => Self::Attest,
            Self::Attest => Self::Extract,
            Self::Extract => Self::Converge,
            Self::Converge => Self::Converge, // terminal
        }
    }

    /// Human-readable stage name (matches ontology `stageName` constants).
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Init => "Initialization",
            Self::Declare => "Declare",
            Self::Factorize => "Factorize",
            Self::Resolve => "Resolve",
            Self::Attest => "Attest",
            Self::Extract => "Extract",
            Self::Converge => "Convergence",
        }
    }

    /// Expected phase string (matches ontology `expectedPhase` constants).
    #[inline]
    pub const fn expected_phase(self) -> &'static str {
        match self {
            Self::Init => "Ω⁰",
            Self::Declare => "Ω¹",
            Self::Factorize => "Ω²",
            Self::Resolve => "Ω³",
            Self::Attest => "Ω⁴",
            Self::Extract => "Ω⁵",
            Self::Converge => "π",
        }
    }
}

/// Result of executing a single cascade stage handler.
#[derive(Debug, Clone, PartialEq)]
pub enum Transition {
    /// Proceed to the next stage in sequence.
    Advance,
    /// Skip directly to the given stage (e.g., cache hit → Extract).
    Skip(CascadeStage),
    /// Terminal error — cascade halts.
    Halt(HaltReason),
    /// Cascade has converged (stage 6 completed successfully).
    Converged,
}

/// Reason for cascade halt.
#[derive(Debug, Clone, PartialEq)]
pub enum HaltReason {
    /// Budget exhausted during execution.
    BudgetExhausted { consumed: f64, allocated: f64 },
    /// Contradiction detected during attestation.
    Contradiction(String),
    /// Stage-specific failure.
    StageFailure { stage: CascadeStage, message: String },
}

impl core::fmt::Display for HaltReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BudgetExhausted { consumed, allocated } => {
                write!(f, "budget exhausted: consumed {} > allocated {}", consumed, allocated)
            }
            Self::Contradiction(msg) => write!(f, "contradiction: {}", msg),
            Self::StageFailure { stage, message } => {
                write!(f, "stage {} ({}): {}", stage.name(), stage.expected_phase(), message)
            }
        }
    }
}

/// Mutable state vector for one cascade evaluation.
///
/// Initialized once per CompileUnit submission. Intermediate products
/// are populated by each stage and consumed by subsequent stages.
pub struct CascadeState {
    /// Current stage in the cascade.
    pub stage: CascadeStage,
    /// Content-addressed identifier of the root term graph (BLAKE3).
    pub unit_address: [u8; 32],
    /// Quantum level of the computation.
    pub quantum_level: RingLevel,
    /// Thermodynamic budget allocated by the submitter (k_B T units).
    pub budget_allocated: f64,
    /// Cumulative Landauer cost consumed so far.
    pub budget_consumed: f64,

    // ── Intermediate products populated by stages ──
    /// Graph produced by Init (from term lowering). Consumed by Declare, Factorize.
    pub graph: Option<hologram_graph::Graph>,
    /// Execution schedule produced by Resolve.
    pub schedule: Option<hologram_graph::ExecutionSchedule>,
    /// Serialized graph snapshot produced by Extract for tape building.
    pub serialized_graph: Option<hologram_archive::format::graph::SerializedGraph>,
    /// Compiled execution tape produced by Extract.
    pub tape: Option<hologram_exec::tape::EnumTape>,
    /// Archive bytes produced by Converge.
    pub archive_bytes: Option<Vec<u8>>,

    // ── Full pipeline output fields ──
    /// Fusion statistics captured by Factorize (Ω²).
    pub fusion_stats: FusionStats,
    /// Liveness intervals captured by Attest (Ω⁴). None until graph is processed.
    pub liveness_intervals: Option<Vec<LivenessInterval>>,
    /// Workspace layout captured by Attest (Ω⁴).
    pub workspace_layout: Option<WorkspaceLayout>,
    /// QEDL domain-crossing boundaries captured by Attest (Ω⁴). None until graph is processed.
    pub qedl_boundaries: Option<Vec<(NodeId, QedlBoundary, EncodingId)>>,
    /// Whether to skip the fusion pass in Factorize.
    pub skip_fusion: bool,
    /// Number of Prim nodes promoted to ring-level variants by Declare (Ω¹).
    pub ring_prims_promoted: usize,
}

impl core::fmt::Debug for CascadeState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CascadeState")
            .field("stage", &self.stage)
            .field("quantum_level", &self.quantum_level)
            .field("budget_allocated", &self.budget_allocated)
            .field("budget_consumed", &self.budget_consumed)
            .field("has_graph", &self.graph.is_some())
            .field("has_schedule", &self.schedule.is_some())
            .field("has_tape", &self.tape.is_some())
            .field("has_archive", &self.archive_bytes.is_some())
            .field("skip_fusion", &self.skip_fusion)
            .field("qedl_boundaries", &self.qedl_boundaries.as_ref().map_or(0, |v| v.len()))
            .finish()
    }
}

impl CascadeState {
    /// Create initial state from a CompileUnit's parameters.
    pub fn from_unit(
        unit_address: [u8; 32],
        quantum_level: RingLevel,
        budget: f64,
    ) -> Self {
        Self {
            stage: CascadeStage::Init,
            unit_address,
            quantum_level,
            budget_allocated: budget,
            budget_consumed: 0.0,
            graph: None,
            schedule: None,
            serialized_graph: None,
            tape: None,
            archive_bytes: None,
            fusion_stats: FusionStats::default(),
            liveness_intervals: None,
            workspace_layout: None,
            qedl_boundaries: None,
            skip_fusion: false,
            ring_prims_promoted: 0,
        }
    }

    /// Check if the budget has been exceeded.
    #[inline]
    pub fn budget_exceeded(&self) -> bool {
        self.budget_consumed > self.budget_allocated
    }

    /// Remaining budget.
    #[inline]
    pub fn budget_remaining(&self) -> f64 {
        self.budget_allocated - self.budget_consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_count() {
        assert_eq!(CascadeStage::COUNT, 7);
    }

    #[test]
    fn stage_sequence() {
        let mut s = CascadeStage::Init;
        let expected = [
            CascadeStage::Declare,
            CascadeStage::Factorize,
            CascadeStage::Resolve,
            CascadeStage::Attest,
            CascadeStage::Extract,
            CascadeStage::Converge,
            CascadeStage::Converge, // terminal wraps
        ];
        for e in expected {
            s = s.next();
            assert_eq!(s, e);
        }
    }

    #[test]
    fn stage_names() {
        assert_eq!(CascadeStage::Init.name(), "Initialization");
        assert_eq!(CascadeStage::Converge.name(), "Convergence");
    }

    #[test]
    fn stage_phases() {
        assert_eq!(CascadeStage::Init.expected_phase(), "Ω⁰");
        assert_eq!(CascadeStage::Converge.expected_phase(), "π");
    }

    #[test]
    fn state_budget_tracking() {
        let mut state = CascadeState::from_unit([0u8; 32], RingLevel::Q0, 10.0);
        assert!(!state.budget_exceeded());
        assert_eq!(state.budget_remaining(), 10.0);

        state.budget_consumed = 7.5;
        assert!(!state.budget_exceeded());
        assert_eq!(state.budget_remaining(), 2.5);

        state.budget_consumed = 10.5;
        assert!(state.budget_exceeded());
    }

    #[test]
    fn repr_u8_indexed() {
        // Verify the repr(u8) layout enables indexed jump table access.
        assert_eq!(CascadeStage::Init as usize, 0);
        assert_eq!(CascadeStage::Declare as usize, 1);
        assert_eq!(CascadeStage::Factorize as usize, 2);
        assert_eq!(CascadeStage::Resolve as usize, 3);
        assert_eq!(CascadeStage::Attest as usize, 4);
        assert_eq!(CascadeStage::Extract as usize, 5);
        assert_eq!(CascadeStage::Converge as usize, 6);
    }
}
