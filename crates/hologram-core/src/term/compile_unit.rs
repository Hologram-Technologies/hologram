//! CompileUnit — the typed input to the cascade pipeline.
//!
//! Implements `uor_foundation::kernel::cascade::CompileUnit<HoloPrimitives>`,
//! providing the bridge between hologram's arena-allocated term representation
//! and the foundation's ontology-level cascade admission contract.

extern crate alloc;
use alloc::boxed::Box;

use crate::HoloPrimitives;

use super::{Assertion, Binding, TermArena, TermId, TypeId};

use uor_foundation::enums::VerificationDomain;
use uor_foundation::WittLevel as QuantumLevel;

/// Maximum number of let-bindings per compile unit.
pub const MAX_BINDINGS: usize = 64;
/// Maximum number of assertions per compile unit.
pub const MAX_ASSERTIONS: usize = 32;
/// Maximum number of type declarations per compile unit.
pub const MAX_TYPE_DECLS: usize = 16;
/// Maximum effect declarations per compile unit.
pub const MAX_EFFECT_DECLS: usize = 32;
/// Maximum dispatch declarations per compile unit.
pub const MAX_DISPATCH_DECLS: usize = 16;
/// Maximum number of target verification domains (all 12 defined by the ontology).
pub const MAX_DOMAINS: usize = 12;

/// A declared effect for cascade registration.
#[derive(Clone, Copy, Debug)]
pub struct EffectDecl {
    pub budget_delta: i64,
    pub commutes: bool,
    pub fiber_count: u8,
    pub target_fibers: [u32; 8],
}

/// A declared dispatch rule for cascade registration.
#[derive(Clone, Copy, Debug)]
pub struct DispatchDecl {
    pub resolver_id: u16,
    pub priority: u16,
}

/// The typed input to the cascade pipeline.
///
/// Packages a root [`Term`](super::TermKind), its target quantum level,
/// the verification domains the submitter requires, and a thermodynamic
/// budget authorizing the maximum Landauer cost of resolution.
///
/// Mirrors `cascade:CompileUnit` from uor-foundation v0.1.4.
#[derive(Debug)]
pub struct HoloCompileUnit {
    /// The root term of the computation.
    pub root_term: TermId,

    /// Arena holding all term nodes for this unit.
    pub arena: TermArena,

    /// Let-bindings in declaration order. Boxed to reduce stack footprint.
    pub bindings: Box<[Binding; MAX_BINDINGS]>,
    pub binding_count: u8,

    /// Assertions to verify. Boxed to reduce stack footprint.
    pub assertions: Box<[Assertion; MAX_ASSERTIONS]>,
    pub assertion_count: u8,

    /// Type declarations. Boxed to reduce stack footprint.
    pub type_decls: Box<[super::TypeDecl; MAX_TYPE_DECLS]>,
    pub type_decl_count: u8,

    /// Effect declarations. Boxed to reduce stack footprint.
    pub effect_decls: [EffectDecl; MAX_EFFECT_DECLS],
    pub effect_decl_count: u8,

    /// Dispatch declarations. Boxed to reduce stack footprint.
    pub dispatch_decls: [DispatchDecl; MAX_DISPATCH_DECLS],
    pub dispatch_decl_count: u8,

    /// Quantum level. Maps to `cascade:unitQuantumLevel`.
    /// Supports any level (Q0–Q7+), not limited to Q0–Q3.
    pub quantum_level: QuantumLevel,

    /// Target verification domains. Materialized array for trait compliance
    /// (`CompileUnit::target_domains()` returns `&[VerificationDomain]`).
    pub target_domains_array: [VerificationDomain; MAX_DOMAINS],
    pub target_domain_count: u8,

    /// Thermodynamic budget in k_B T units (`xsd:decimal` mapped to `f64`).
    ///
    /// Minimum viable budget: `bitsWidth(Q_k) * ln(2)`.
    /// - Q0: >= 5.545
    /// - Q1: >= 11.090
    /// - Q2: >= 16.636
    /// - Q3: >= 22.181
    pub thermodynamic_budget: f64,

    /// Content-addressed identifier. Computed by CS_7 during `stage_initialization`,
    /// **not** declared by the submitter.
    ///
    /// BLAKE3 hash of `canonicalBytes(transitiveClosure(rootTerm))`.
    /// Excludes budget, domains, and quantum level to enable memoization.
    pub unit_address: [u8; 32],

    /// Typed address wrapping `unit_address` for the `CompileUnit` trait.
    /// Populated by `preflight::compute_unit_address`.
    pub address: HoloAddress,

    /// Preflight check results.
    pub preflight: PreflightStatus,
}

/// Bitmask tracking preflight check pass/fail. 2 bytes, zero heap.
///
/// Each bit position corresponds to a `preflightOrder` index from the ontology.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreflightStatus {
    /// Bit N set = check N has been run.
    pub checked: u8,
    /// Bit N set = check N passed (only meaningful if `checked` bit is also set).
    pub passed: u8,
}

impl PreflightStatus {
    /// `cascade:BudgetSolvencyCheck` — preflightOrder 0.
    pub const BUDGET_SOLVENCY: u8 = 0;
    /// `cascade:FeasibilityCheck` — preflightOrder 1.
    pub const FEASIBILITY: u8 = 1;
    /// `cascade:DispatchCoverageCheck` — preflightOrder 2.
    pub const DISPATCH_COVERAGE: u8 = 2;
    /// `cascade:PackageCoherenceCheck` — preflightOrder 3.
    pub const PACKAGE_COHERENCE: u8 = 3;
    /// `cascade:PreflightTiming` — preflightOrder 4.
    pub const PREFLIGHT_TIMING: u8 = 4;
    /// `cascade:RuntimeTiming` — preflightOrder 5.
    pub const RUNTIME_TIMING: u8 = 5;
    /// Enforcement-level validation (v0.1.4 builder) — preflightOrder 6.
    pub const ENFORCEMENT_VALIDATE: u8 = 6;

    /// Mark a check as passed.
    #[inline]
    pub fn mark_passed(&mut self, check: u8) {
        self.checked |= 1 << check;
        self.passed |= 1 << check;
    }

    /// Mark a check as failed.
    #[inline]
    pub fn mark_failed(&mut self, check: u8) {
        self.checked |= 1 << check;
        self.passed &= !(1 << check);
    }

    /// Returns `true` if the given check has been run and passed.
    #[inline]
    pub fn is_passed(&self, check: u8) -> bool {
        (self.checked & (1 << check) != 0) && (self.passed & (1 << check) != 0)
    }

    /// Returns `true` if all run checks passed.
    #[inline]
    pub fn all_passed(&self) -> bool {
        self.checked != 0 && self.passed == self.checked
    }
}

// ── Marker type for schema::Term<HoloPrimitives> ─────────────────────────────

/// Marker type satisfying `uor_foundation::kernel::schema::Term<HoloPrimitives>`.
///
/// The actual term data lives in [`TermArena`]; this marker type exists
/// solely to satisfy the associated type bound on `CompileUnit::Term`.
pub struct HoloTerm;

impl uor_foundation::kernel::schema::Term<HoloPrimitives> for HoloTerm {}
impl uor_foundation::kernel::schema::TermExpression<HoloPrimitives> for HoloTerm {}

/// Singleton for `CompileUnit::root_term()` return.
static HOLO_TERM_SINGLETON: HoloTerm = HoloTerm;

// ── HoloAddress: content-addressed identifier ────────────────────────────────

/// Content-addressed identifier wrapping a BLAKE3 hash.
///
/// Implements `uor_foundation::kernel::address::Element<HoloPrimitives>`.
/// The digest is a 64-character lowercase hex string of the 32-byte BLAKE3 hash.
/// The glyph is a 2-character Braille encoding of the first byte.
pub struct HoloAddress {
    /// Raw 32-byte BLAKE3 hash.
    hash: [u8; 32],
    /// 64-character hex digest string + null terminator.
    digest_hex: [u8; 65],
    /// Braille glyph encoding of the first byte (2 Braille characters, 6 bytes UTF-8).
    glyph_buf: [u8; 6],
}

impl HoloAddress {
    /// Zero address (all zeros).
    pub const ZERO: Self = Self {
        hash: [0u8; 32],
        digest_hex: [b'0'; 65],
        glyph_buf: [0xE2, 0xA0, 0x80, 0xE2, 0xA0, 0x80],
    };

    /// Create an address from a 32-byte BLAKE3 hash.
    pub fn from_hash(hash: [u8; 32]) -> Self {
        let mut digest_hex = [0u8; 65];
        for (i, byte) in hash.iter().enumerate() {
            let hi = byte >> 4;
            let lo = byte & 0x0F;
            digest_hex[i * 2] = HEX_CHARS[hi as usize];
            digest_hex[i * 2 + 1] = HEX_CHARS[lo as usize];
        }
        digest_hex[64] = 0; // null terminator for safety

        let first_byte = hash[0];
        let lo6 = first_byte & 0x3F;
        let hi2 = (first_byte >> 6) & 0x03;
        let glyph_buf = [0xE2, 0xA0, 0x80 + lo6, 0xE2, 0xA0, 0x80 + hi2];

        Self {
            hash,
            digest_hex,
            glyph_buf,
        }
    }

    /// Raw 32-byte hash.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.hash
    }
}

const HEX_CHARS: [u8; 16] = *b"0123456789abcdef";

impl uor_foundation::kernel::address::Element<HoloPrimitives> for HoloAddress {
    fn length(&self) -> u64 {
        32
    }

    fn addresses(&self) -> &str {
        // The Braille glyph string of the first byte, kept for display.
        // SAFETY: glyph_buf contains valid UTF-8 Braille characters.
        unsafe { core::str::from_utf8_unchecked(&self.glyph_buf) }
    }

    fn digest(&self) -> &str {
        // SAFETY: digest_hex contains valid ASCII hex characters.
        unsafe { core::str::from_utf8_unchecked(&self.digest_hex[..64]) }
    }

    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    /// Per ADR-052: canonical_bytes are the bytes that hash to digest.
    /// For `HoloAddress` these are the raw 32-byte BLAKE3 hash bytes.
    fn canonical_bytes(&self) -> &[u8] {
        &self.hash
    }

    fn witt_length(&self) -> u64 {
        256 // 8-bit address space
    }
}

impl core::fmt::Debug for HoloAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Use raw hex bytes directly instead of trait method to avoid import requirement.
        let hex = unsafe { core::str::from_utf8_unchecked(&self.digest_hex[..64]) };
        f.debug_struct("HoloAddress").field("digest", &hex).finish()
    }
}

// ── CompileUnit trait implementation ──────────────────────────────────────────

impl uor_foundation::kernel::reduction::CompileUnit<HoloPrimitives> for HoloCompileUnit {
    type TermExpression = HoloTerm;

    #[inline]
    fn root_term(&self) -> &Self::TermExpression {
        &HOLO_TERM_SINGLETON
    }

    #[inline]
    fn unit_witt_level(&self) -> QuantumLevel {
        self.quantum_level
    }

    fn target_domains(&self) -> &[VerificationDomain] {
        &self.target_domains_array[..self.target_domain_count as usize]
    }

    #[inline]
    fn thermodynamic_budget(&self) -> &str {
        // 0.3.0 returns &H::HostString = &str. We store the budget as
        // f64 so format it lazily; this is allocator-bounded but used
        // only at trait-driven code paths.
        // For now, return an empty string sentinel — real consumers
        // read `self.thermodynamic_budget` directly via the inherent
        // field. Future ADR can introduce a stored string buffer if
        // the trait method becomes load-bearing.
        ""
    }

    type Element = HoloAddress;

    fn unit_address(&self) -> &Self::Element {
        &self.address
    }
}

// ── Constructor ──────────────────────────────────────────────────────────────

impl HoloCompileUnit {
    /// Create a new compile unit with the given parameters.
    pub fn new(
        arena: TermArena,
        root_term: TermId,
        quantum_level: QuantumLevel,
        thermodynamic_budget: f64,
        target_domains: &[VerificationDomain],
    ) -> Self {
        let mut target_domains_array = [VerificationDomain::Algebraic; MAX_DOMAINS];
        let count = target_domains.len().min(MAX_DOMAINS);
        target_domains_array[..count].copy_from_slice(&target_domains[..count]);

        Self {
            root_term,
            arena,
            bindings: Box::new(
                [Binding {
                    var: super::VarId(0),
                    ty: TypeId::UNCONSTRAINED,
                    rhs: TermId(0),
                }; MAX_BINDINGS],
            ),
            binding_count: 0,
            assertions: Box::new(
                [Assertion {
                    lhs: TermId(0),
                    rhs: TermId(0),
                    canonical: false,
                }; MAX_ASSERTIONS],
            ),
            assertion_count: 0,
            type_decls: Box::new(
                [super::TypeDecl {
                    name_id: super::VarId(0),
                    constraint: super::ConstraintKind::Residue,
                    value: TermId(0),
                }; MAX_TYPE_DECLS],
            ),
            type_decl_count: 0,
            effect_decls: [EffectDecl {
                budget_delta: 0,
                commutes: true,
                fiber_count: 0,
                target_fibers: [0; 8],
            }; MAX_EFFECT_DECLS],
            effect_decl_count: 0,
            dispatch_decls: [DispatchDecl {
                resolver_id: 0,
                priority: 0,
            }; MAX_DISPATCH_DECLS],
            dispatch_decl_count: 0,
            quantum_level,
            target_domains_array,
            target_domain_count: count as u8,
            thermodynamic_budget,
            unit_address: [0u8; 32],
            address: HoloAddress::ZERO,
            preflight: PreflightStatus::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::PrimOp;
    use crate::term::TermKind;
    use uor_foundation::WittLevel as QuantumLevel;

    #[test]
    fn preflight_status_bitmask() {
        let mut ps = PreflightStatus::default();
        assert!(!ps.is_passed(PreflightStatus::BUDGET_SOLVENCY));
        assert!(!ps.all_passed());

        ps.mark_passed(PreflightStatus::BUDGET_SOLVENCY);
        assert!(ps.is_passed(PreflightStatus::BUDGET_SOLVENCY));
        assert!(ps.all_passed());

        ps.mark_failed(PreflightStatus::FEASIBILITY);
        assert!(ps.is_passed(PreflightStatus::BUDGET_SOLVENCY));
        assert!(!ps.is_passed(PreflightStatus::FEASIBILITY));
        assert!(!ps.all_passed());
    }

    #[test]
    fn compile_unit_construction() {
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(1));
        let b = arena.alloc(TermKind::IntLit(2));
        let root = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: a,
            rhs: b,
        });

        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::W8,
            6.0, // > 5.545 minimum for Q0
            &[VerificationDomain::Algebraic],
        );

        assert_eq!(unit.root_term, root);
        assert_eq!(unit.quantum_level, QuantumLevel::W8);
        assert_eq!(unit.thermodynamic_budget, 6.0);
        assert_eq!(unit.target_domain_count, 1);
        assert_eq!(unit.arena.len(), 3);
    }

    #[test]
    fn compile_unit_trait_impl() {
        use uor_foundation::kernel::reduction::CompileUnit;

        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(42));

        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::W16,
            12.0,
            &[
                VerificationDomain::Algebraic,
                VerificationDomain::Thermodynamic,
            ],
        );

        assert_eq!(unit.unit_witt_level(), QuantumLevel::W16);
        // 0.3.0 returns &str sentinel; the f64 budget is the inherent field.
        assert_eq!(unit.thermodynamic_budget(), "");
        assert_eq!(unit.target_domains().len(), 2);
    }

    #[test]
    fn holo_address_digest_correctness() {
        use uor_foundation::kernel::address::Element;
        let hash = [0xABu8; 32];
        let addr = HoloAddress::from_hash(hash);
        let digest = addr.digest();
        assert_eq!(digest.len(), 64);
        assert_eq!(&digest[..2], "ab"); // first byte 0xAB → "ab"
        assert_eq!(addr.digest_algorithm(), "blake3");
        assert_eq!(addr.length(), 32);
        assert_eq!(addr.witt_length(), 256);
    }

    #[test]
    fn holo_address_glyph_valid_utf8() {
        use uor_foundation::kernel::address::Element;
        for byte in 0..=255u8 {
            let mut hash = [0u8; 32];
            hash[0] = byte;
            let addr = HoloAddress::from_hash(hash);
            // addresses() returns the Braille glyph string — must be valid UTF-8.
            let g = addr.addresses();
            assert_eq!(g.len(), 6); // 2 Braille chars × 3 bytes UTF-8 each
        }
    }

    #[test]
    fn holo_address_zero() {
        use uor_foundation::kernel::address::Element;
        let addr = HoloAddress::ZERO;
        assert_eq!(addr.as_bytes(), &[0u8; 32]);
        // digest should be all '0's
        assert!(addr.digest().chars().all(|c| c == '0'));
    }

    #[test]
    fn term_arena_alloc_performance() {
        // Performance contract: 1M allocs < 50ms (~50ns each)
        let start = std::time::Instant::now();
        let mut arena = TermArena::new();
        for i in 0..1_000_000u32 {
            arena.alloc(TermKind::IntLit(i as i64));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200, // generous CI margin
            "1M arena allocs took {}ms (target < 200ms)",
            elapsed.as_millis()
        );
    }
}
