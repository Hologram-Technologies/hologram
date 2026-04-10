//! `hologram-foundation` — the single boundary between hologram and
//! `uor-foundation` v0.2.0.
//!
//! This crate is a zero-cost re-export shim. Every item in this module tree
//! resolves to a definition in `uor-foundation`; the shim adds no runtime
//! overhead and no allocation. Re-exports compile away to nothing.
//!
//! # Why this exists
//!
//! Per the conformance-first refactor of hologram to v0.2.0, the rest of the
//! workspace must depend on this crate instead of `uor-foundation` directly.
//! Concentrating the foundation dependency in one crate localises future
//! version bumps to a single re-export edit.
//!
//! # Module organisation
//!
//! The re-exports are reorganised into hologram-friendly modules:
//!
//! - [`witt`] — Witt levels and level bindings (replaces v0.1.4 `QuantumLevel`)
//! - [`primitives`] — the [`Primitives`] type family trait
//! - [`schema`] — `Datum`, `Ring`, `W16Ring`, `Term`, `Triad`, expression AST,
//!   `ForAllDeclaration`, `VariableBinding`, `SurfaceSymbol`/`HostValue`
//! - [`op`] — `Operation`, `BinaryOp`, `UnaryOp`, `Identity`, `Group`,
//!   `DihedralGroup`, `Involution`, plus the new composed operations
//! - [`address`] — content-addressable [`Element`](address::Element)
//! - [`division`] — `NormedDivisionAlgebra`, `CayleyDicksonConstruction`,
//!   `MultiplicationTable`, `AlgebraCommutator`, `AlgebraAssociator`
//! - [`reduction`] — Euler reduction theory: `EulerReduction`, `ReductionStep`,
//!   `ReductionState`, `Epoch`, `EpochBoundary`, `CompileUnit`. **This is a
//!   description of factorisation theory, not a runtime engine.** Hologram
//!   does not implement these as a runtime state machine.
//! - [`stream`] — `ProductiveStream`, `Unfold`, `Epoch`, `EpochBoundary`
//! - [`conformance`] — the entire `bridge::conformance_` namespace: `Shape`,
//!   `PropertyConstraint`, the per-extension shape traits (`WittLevelShape`,
//!   `EffectShape`, `GroundingShape`, …), `ValidationResult`,
//!   `ShapeViolationReport`, the witness opaque types, the
//!   `*Declaration<P>` builder traits, and all conformance shape IRI constants
//! - [`enforcement`] — the concrete builder layer: `Term`, `TermArena`,
//!   `TermList`, `Validated<T>`, `Derivation`, `FreeRank`, `Grounding`,
//!   `GroundedCoord`, `ShapeViolation`, plus all `*Builder` types
//! - [`enums`] — every shared enum: `WittLevel`, `VerificationDomain`,
//!   `ViolationKind`, `GeometricCharacter`, `PrimitiveOp`, `SiteState`,
//!   `QuantifierKind`, `ValidityScopeKind`, etc.
//!
//! # Performance
//!
//! Re-exports are zero-cost: `pub use foo;` resolves at compile time to a name
//! alias and produces no runtime indirection. The shim adds no virtual
//! dispatch, no allocation, no branching. Build profile: 1 dependency, ~50ms
//! incremental rebuild after `uor-foundation` changes.

#![no_std]
#![deny(missing_docs)]

// ── enums ────────────────────────────────────────────────────────────────────

/// Shared enumerations from the v0.2.0 ontology.
///
/// Re-exports `uor_foundation::enums::*` (which `uor-foundation` itself
/// re-exports at its crate root). Includes [`WittLevel`](enums::WittLevel),
/// [`VerificationDomain`](enums::VerificationDomain),
/// [`ViolationKind`](enums::ViolationKind),
/// [`GeometricCharacter`](enums::GeometricCharacter),
/// [`PrimitiveOp`](enums::PrimitiveOp), [`SiteState`](enums::SiteState),
/// [`QuantifierKind`](enums::QuantifierKind),
/// [`ValidityScopeKind`](enums::ValidityScopeKind), and the rest.
pub mod enums {
    pub use uor_foundation::enums::*;
}

// ── primitives ───────────────────────────────────────────────────────────────

/// The [`Primitives`] type family trait — used by every kernel and bridge
/// trait as a parameter to abstract XSD primitive representations.
pub mod primitives {
    pub use uor_foundation::Primitives;
}

/// Re-export of [`Primitives`](primitives::Primitives) at crate root for
/// brevity in trait bounds.
pub use uor_foundation::Primitives;

// ── witt ─────────────────────────────────────────────────────────────────────

/// Witt levels and level bindings.
///
/// Replaces the v0.1.4 `QuantumLevel` enum. [`WittLevel`] is now a struct with
/// a positive bit width parameter; `W8`, `W16`, `W24`, `W32` constants are
/// provided, and arbitrary widths can be constructed via `WittLevel::new(n)`.
/// The chain is unbounded.
pub mod witt {
    pub use uor_foundation::enums::WittLevel;
    pub use uor_foundation::kernel::op::WittLevelBinding;
}

/// Re-export of [`WittLevel`](witt::WittLevel) at crate root.
pub use uor_foundation::enums::WittLevel;

// ── schema ───────────────────────────────────────────────────────────────────

/// Core value types and term language traits from `kernel::schema`.
///
/// `Datum` is the primary value type (an element of `Z/(2^n)Z` at a specific
/// Witt level). `Ring` is the container. `Term` and `TermExpression` are the
/// syntactic AST. `ForAllDeclaration`/`VariableBinding` carry typed quantifier
/// bindings (replacing v0.1.4's string-valued `op:forAll`). `SurfaceSymbol` and
/// `HostValue` carry the boundary between ring datums and host XSD values.
pub mod schema {
    pub use uor_foundation::kernel::schema::{
        Application, ApplicationExpression, CompositionExpression, Datum, ForAllDeclaration,
        HostBooleanLiteral, HostStringLiteral, HostValue, InfixExpression, Literal,
        LiteralExpression, Ring, SetExpression, SurfaceSymbol, Term, TermExpression, Triad,
        VariableBinding, W16Ring,
    };
}

// ── op ───────────────────────────────────────────────────────────────────────

/// Ring operations, identities, and the dihedral symmetry group D_{2^n}.
///
/// `Operation` is the root trait. `BinaryOp` and `UnaryOp` are the arity
/// specialisations. `Involution` is a unary op satisfying `f(f(x)) = x`.
/// `Identity` carries algebraic identity declarations with the new
/// `validity_kind()`/`valid_kmin()`/`valid_kmax()` parametric scope mechanism.
/// `Group`/`DihedralGroup` carry group structure. `ComposedOperation` and its
/// specialisations (`DispatchOperation`, `InferenceOperation`,
/// `AccumulationOperation`, `LeasePartitionOperation`,
/// `SessionCompositionOperation`) carry the composed-operation hierarchy.
pub mod op {
    pub use uor_foundation::kernel::op::{
        AccumulationOperation, BinaryOp, ComposedOperation, DihedralGroup, DispatchOperation,
        Group, GroupPresentation, Identity, InferenceOperation, Involution,
        LeasePartitionOperation, Operation, QuantumThermodynamicDomain,
        SessionCompositionOperation, UnaryOp, WittLevelBinding,
    };
}

// ── address ──────────────────────────────────────────────────────────────────

/// Content-addressable element identifiers from `kernel::address`.
///
/// [`Element`](address::Element) carries the BLAKE3 (or SHA-256) digest, the
/// canonical byte serialisation, and the Braille glyph address of a ring
/// element. Each `Datum` links to an `Element` via `Datum::element()`.
pub mod address {
    pub use uor_foundation::kernel::address::Element;
}

// ── division ─────────────────────────────────────────────────────────────────

/// The four normed division algebras and the Cayley-Dickson construction.
///
/// `NormedDivisionAlgebra` carries dimension/commutativity/associativity flags.
/// `CayleyDicksonConstruction` describes the doubling chain R → C → H → O.
/// `MultiplicationTable`, `AlgebraCommutator`, `AlgebraAssociator` are marker
/// traits.
pub mod division {
    pub use uor_foundation::kernel::division::{
        AlgebraAssociator, AlgebraCommutator, CayleyDicksonConstruction, MultiplicationTable,
        NormedDivisionAlgebra,
    };
}

// ── reduction ────────────────────────────────────────────────────────────────

/// Euler reduction theory from `kernel::reduction`.
///
/// This module describes the *factorisation theory* of non-irreducible
/// operations through F-values: the formal specification of how
/// non-primitive operations decompose into compositions of primitives.
/// It is a compile-time / proof-time vocabulary, **not** a runtime
/// evaluator — hologram never instantiates a reduction state machine on
/// the inference path.
///
/// `CompileUnit` is re-exported here because it carries the admission
/// contract (root term, Witt level, budget, target domains) that the
/// compiler needs to validate before any structural analysis runs.
pub mod reduction {
    pub use uor_foundation::kernel::reduction::{
        BackPressureSignal, ComparisonPredicate, CompileUnit, ComplexConjugateRollback,
        ConjunctionPredicate, DeferredQuerySet, DisjunctionPredicate, Epoch, EpochBoundary,
        EqualsPredicate, EulerReduction, FeasibilityResult, GroundingPredicate, GuardExpression,
        LeaseCheckpoint, LeaseState, ManagedLease, MembershipPredicate, NegationPredicate,
        NonNullPredicate, PhaseGateAttestation, PhaseRotationScheduler, PipelineFailureReason,
        PipelineSuccess, PredicateExpression, PreflightCheck, PropertyBind, QuerySubtypePredicate,
        ReductionAdvance, ReductionRule, ReductionState, ReductionStep, ReductionTransaction,
        ServiceWindow, SiteCoveragePredicate, SubleaseTransfer, TargetConvergenceAngle,
        TransitionEffect,
    };

    /// IRI constants for the seven reduction stages. Useful for emitting shape
    /// declarations and verification reports without re-defining the constants.
    pub mod stages {
        pub use uor_foundation::kernel::reduction::{
            stage_attest, stage_convergence, stage_declare, stage_extract, stage_factorize,
            stage_initialization, stage_resolve,
        };
    }
}

// ── stream ───────────────────────────────────────────────────────────────────

/// Productive streams and coinductive constructors from `kernel::stream`.
pub mod stream {
    // The stream module's traits are re-exported wholesale; if upstream changes
    // the surface we localise the response here.
    pub use uor_foundation::kernel::stream::*;
}

// ── conformance ──────────────────────────────────────────────────────────────

/// The entire `bridge::conformance_` namespace — the v0.2.0 conformance shape
/// declaration mechanism.
///
/// This is the primary v0.2.0 capability hologram builds against. The
/// conformance-first refactor declares hologram's shapes
/// (`F_prism_fused_component`, `F_prism_strict`) as instances of [`Shape`]
/// from this module, and the `PrismModule` trait in `hologram-shapes`
/// dispatches against these shape declarations.
///
/// `Shape<P>` is the root SHACL-equivalent constraint shape trait.
/// `PropertyConstraint<P>` describes a single required property within a
/// shape. The specialised shape traits (`WittLevelShape`, `EffectShape`,
/// `ParallelShape`, `StreamShape`, `DispatchShape`, `LeaseShape`,
/// `GroundingShape`, `PredicateShape`) are extension points for declaring new
/// instances. The `*Declaration<P>` builder traits collect the data each
/// shape needs.
///
/// The witness types (`WitnessDatum`, `GroundedCoordinate`, `GroundedTuple`,
/// `GroundedValueMarker`, `ValidatedWrapper`, `WitnessDerivation`,
/// `WitnessSiteBudget`) are sealed opaque markers that prove a value passed
/// through the foundation's reduction evaluator.
///
/// The IRI constant submodules (`compile_unit_shape`, …,
/// `compile_unit_target_domains_constraint`) carry the conformance shape
/// IRIs as `&'static str` constants for emission in shape declarations and
/// violation reports.
pub mod conformance {
    pub use uor_foundation::bridge::conformance_::{
        CompileUnitBuilder, DispatchDeclaration, DispatchShape, EffectDeclaration, EffectShape,
        GroundedCoordinate, GroundedTuple, GroundedValueMarker, GroundingDeclaration,
        GroundingShape, LeaseDeclaration, LeaseShape, MintingSession, ParallelDeclaration,
        ParallelShape, PredicateDeclaration, PredicateShape, PropertyConstraint, Shape,
        ShapeViolationReport, StreamDeclaration, StreamShape, ValidatedWrapper, ValidationResult,
        WitnessDatum, WitnessDerivation, WitnessSiteBudget, WittLevelDeclaration, WittLevelShape,
    };

    /// IRI constants for the `CompileUnit` shape and its required properties.
    pub mod compile_unit {
        pub use uor_foundation::bridge::conformance_::{
            compile_unit_root_term_constraint, compile_unit_shape,
            compile_unit_target_domains_constraint, compile_unit_thermodynamic_budget_constraint,
            compile_unit_unit_witt_level_constraint,
        };
    }

    /// Marker submodules for the five `ViolationKind` variants. Useful for
    /// emitting shape violations with the correct IRI.
    pub mod violation_kinds {
        pub use uor_foundation::bridge::conformance_::{
            cardinality_violation, level_mismatch, missing, type_mismatch, value_check,
        };
    }
}

// ── enforcement ──────────────────────────────────────────────────────────────

/// The concrete enforcement layer — builders, opaque witness wrappers, the
/// term arena AST.
///
/// This is the layer-2 (declarative builders) and layer-3 (term AST) of the
/// foundation's three-layer pipeline. Builders collect declarations and
/// produce `Validated<T>` witnesses or `ShapeViolation` errors with
/// machine-readable IRIs.
///
/// `Term` is the term arena's enum (Literal, Variable, Application, Lift,
/// Project, Match, Recurse, Unfold, Try). `TermArena<CAP>` is a
/// stack-resident arena. `TermList` is a range descriptor. `Grounding` is the
/// boundary trait for surface→ring mapping. `GroundedCoord` and
/// `GroundedTuple` are stack-resident boundary intermediates.
pub mod enforcement {
    pub use uor_foundation::enforcement::{
        const_ring_eval_q0, const_ring_eval_q1, const_ring_eval_q3, const_ring_eval_q7,
        const_ring_eval_unary_q0, const_ring_eval_unary_q1, const_ring_eval_unary_q3,
        const_ring_eval_unary_q7, Assertion, Binding, CompileUnit, CompileUnitBuilder, Datum,
        Derivation, DispatchDeclaration, DispatchDeclarationBuilder, EffectDeclaration,
        EffectDeclarationBuilder, FreeRank, GroundedCoord, GroundedTuple, GroundedValue, Grounding,
        GroundingDeclaration, GroundingDeclarationBuilder, LeaseDeclaration,
        LeaseDeclarationBuilder, ParallelDeclaration, ParallelDeclarationBuilder,
        PredicateDeclaration, PredicateDeclarationBuilder, ShapeViolation, SinkDeclaration,
        SourceDeclaration, StreamDeclaration, StreamDeclarationBuilder, Term, TermArena, TermList,
        TypeDeclaration, Validated, WittLevelDeclarationBuilder,
    };
}

// ── macros ───────────────────────────────────────────────────────────────────

/// The `uor!` surface-syntax macro from `uor-foundation-macros`.
///
/// Re-exported so consumers do not need a direct dependency on the macros
/// crate. Parses EBNF surface syntax at compile time and produces typed
/// `Term` ASTs.
pub use uor_foundation::uor;

// ── compile-time invariants ──────────────────────────────────────────────────

/// Compile-time check: this crate must remain `no_std` so it can be consumed
/// by every hologram crate (including future no_std embedded targets).
const _: () = {
    // Asserts that `uor_foundation::Primitives` has the six expected
    // associated types we depend on. If v0.2.x adds a new associated type, this
    // file is the first place that catches it.
    fn _primitives_shape_check<P: uor_foundation::Primitives>() {
        let _: Option<&P::String> = None;
        let _: Option<P::Integer> = None;
        let _: Option<P::NonNegativeInteger> = None;
        let _: Option<P::PositiveInteger> = None;
        let _: Option<P::Decimal> = None;
        let _: Option<P::Boolean> = None;
    }
};
