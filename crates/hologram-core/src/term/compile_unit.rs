//! CompileUnit — the typed input to the finder-pipeline compiler.
//!
//! Implements `hologram_foundation::reduction::CompileUnit<HoloPrimitives>`,
//! providing the bridge between hologram's arena-allocated term
//! representation and the foundation's ontology-level admission contract.

extern crate alloc;
use alloc::boxed::Box;

use crate::HoloPrimitives;

use super::{Assertion, Binding, TermArena, TermId, TypeId};

use hologram_foundation::enums::VerificationDomain;
use hologram_foundation::WittLevel;

/// Maximum number of let-bindings per compile unit.
pub const MAX_BINDINGS: usize = 64;
/// Maximum number of assertions per compile unit.
pub const MAX_ASSERTIONS: usize = 32;
/// Maximum number of type declarations per compile unit.
pub const MAX_TYPE_DECLS: usize = 16;
/// Maximum number of target verification domains (all 12 defined by the ontology).
pub const MAX_DOMAINS: usize = 12;

/// The typed input to the finder-pipeline compiler.
///
/// Packages a root [`Term`](super::TermKind), its target Witt level, the
/// verification domains the submitter requires, and a thermodynamic
/// budget authorising the maximum Landauer cost of resolution.
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

    /// Witt level — the bit width of the ring this unit targets.
    /// Supports any width via `WittLevel::new(n)`, not just the four
    /// spec-named W8/W16/W24/W32 levels.
    pub witt_level: WittLevel,

    /// Target verification domains. Materialized array for trait compliance
    /// (`CompileUnit::target_domains()` returns `&[VerificationDomain]`).
    pub target_domains_array: [VerificationDomain; MAX_DOMAINS],
    pub target_domain_count: u8,

    /// Thermodynamic budget in k_B T units (`xsd:decimal` mapped to `f64`).
    ///
    /// Minimum viable budget: `bitsWidth(W_n) * ln(2)`.
    /// - W8:  >= 5.545
    /// - W16: >= 11.090
    /// - W24: >= 16.636
    /// - W32: >= 22.181
    pub thermodynamic_budget: f64,

    /// Content-addressed identifier — BLAKE3 hash of
    /// `canonicalBytes(transitiveClosure(rootTerm))`. Computed by
    /// `preflight::compute_unit_address`, **not** declared by the
    /// submitter. Excludes budget, domains, and Witt level to enable
    /// memoisation.
    pub unit_address: [u8; 32],

    /// Typed address wrapping `unit_address` for the `CompileUnit` trait.
    /// Populated by `preflight::compute_unit_address`.
    pub address: HoloAddress,

    /// Preflight check results.
    pub preflight: PreflightStatus,
}

/// Bitmask tracking preflight check pass/fail. 2 bytes, zero heap.
///
/// Each bit position corresponds to one preflight phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreflightStatus {
    /// Bit N set = check N has been run.
    pub checked: u8,
    /// Bit N set = check N passed (only meaningful if `checked` bit is also set).
    pub passed: u8,
}

impl PreflightStatus {
    /// Budget solvency check — declared budget ≥ Landauer minimum.
    pub const BUDGET_SOLVENCY: u8 = 0;
    /// Feasibility check.
    pub const FEASIBILITY: u8 = 1;
    /// Dispatch coverage check.
    pub const DISPATCH_COVERAGE: u8 = 2;
    /// Package coherence check.
    pub const PACKAGE_COHERENCE: u8 = 3;
    /// Preflight timing measurement.
    pub const PREFLIGHT_TIMING: u8 = 4;
    /// Runtime timing measurement.
    pub const RUNTIME_TIMING: u8 = 5;
    /// Enforcement-level (declarative property) validation.
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

/// Marker type satisfying `hologram_foundation::schema::Term<HoloPrimitives>`.
///
/// The actual term data lives in [`TermArena`]; this marker type exists
/// solely to satisfy the associated type bound on `CompileUnit::Term`.
pub struct HoloTerm;

impl hologram_foundation::schema::Term<HoloPrimitives> for HoloTerm {}
impl hologram_foundation::schema::TermExpression<HoloPrimitives> for HoloTerm {}

/// Singleton for `CompileUnit::root_term()` return.
static HOLO_TERM_SINGLETON: HoloTerm = HoloTerm;

// ── HoloAddress: content-addressed identifier ────────────────────────────────

/// Content-addressed identifier wrapping a BLAKE3 hash.
///
/// Implements `hologram_foundation::address::Element<HoloPrimitives>`.
///
/// Per Amendment 43 section 2:
/// - `canonical_bytes()` returns the hex-encoded canonical byte serialisation
///   of the addressed datum (the pre-image of the hash).
/// - `digest()` returns hex(BLAKE3(canonical_raw)) — the content hash.
/// - `digest ≠ canonical_bytes` — the digest is the hash OF the canonical
///   bytes, not a copy of them.
///
/// Phase 13 deleted the v0.1.4 Braille glyph encoding in favour of
/// Amendment 43's `header(k) || le_bytes(x, k+1)` canonical form.
pub struct HoloAddress {
    /// Raw 32-byte BLAKE3 hash.
    hash: [u8; 32],
    /// hex(BLAKE3(canonical_raw)) — 64 lowercase hex chars.
    digest_hex: [u8; 65],
    /// hex(canonical_raw) — the pre-image of the hash. Heap-allocated
    /// because the canonical representation of a term graph can be
    /// arbitrarily large. Only used by `Element::canonical_bytes()`.
    canonical_hex: alloc::string::String,
}

impl HoloAddress {
    /// Zero address (all zeros).
    pub fn zero() -> Self {
        Self {
            hash: [0u8; 32],
            digest_hex: [b'0'; 65],
            canonical_hex: alloc::string::String::new(),
        }
    }

    /// Create a content-addressed identifier from a BLAKE3 hash and the
    /// raw canonical bytes that were hashed.
    ///
    /// `hash` = `blake3::hash(&canonical_raw)`.
    /// `canonical_raw` = the canonical byte serialisation of the addressed
    /// datum (e.g., the concatenation of per-node canonical byte encodings
    /// for a term graph).
    pub fn from_hash_with_canonical(hash: [u8; 32], canonical_raw: &[u8]) -> Self {
        let mut digest_hex = [0u8; 65];
        for (i, byte) in hash.iter().enumerate() {
            let hi = byte >> 4;
            let lo = byte & 0x0F;
            digest_hex[i * 2] = HEX_CHARS[hi as usize];
            digest_hex[i * 2 + 1] = HEX_CHARS[lo as usize];
        }
        digest_hex[64] = 0;

        // Hex-encode the canonical raw bytes for the Element trait.
        let mut canonical_hex = alloc::string::String::with_capacity(canonical_raw.len() * 2);
        for &b in canonical_raw {
            canonical_hex.push(HEX_CHARS[(b >> 4) as usize] as char);
            canonical_hex.push(HEX_CHARS[(b & 0x0F) as usize] as char);
        }

        Self {
            hash,
            digest_hex,
            canonical_hex,
        }
    }

    /// Backwards-compatible constructor that takes only the hash.
    /// `canonical_bytes()` returns the digest hex (same as v0.1.4 behaviour)
    /// for call sites that don't have the pre-image available.
    pub fn from_hash(hash: [u8; 32]) -> Self {
        let mut digest_hex = [0u8; 65];
        for (i, byte) in hash.iter().enumerate() {
            let hi = byte >> 4;
            let lo = byte & 0x0F;
            digest_hex[i * 2] = HEX_CHARS[hi as usize];
            digest_hex[i * 2 + 1] = HEX_CHARS[lo as usize];
        }
        digest_hex[64] = 0;
        let hex_str = unsafe { core::str::from_utf8_unchecked(&digest_hex[..64]) };
        Self {
            hash,
            digest_hex,
            canonical_hex: alloc::string::String::from(hex_str),
        }
    }

    /// Raw 32-byte hash.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.hash
    }
}

const HEX_CHARS: [u8; 16] = *b"0123456789abcdef";

impl hologram_foundation::address::Element<HoloPrimitives> for HoloAddress {
    /// Byte length of the canonical encoding.
    fn length(&self) -> u64 {
        (self.canonical_hex.len() / 2) as u64
    }

    fn addresses(&self) -> &str {
        self.digest()
    }

    /// BLAKE3 content hash of the canonical bytes, as 64 lowercase hex chars.
    fn digest(&self) -> &str {
        // SAFETY: digest_hex contains valid ASCII hex characters.
        unsafe { core::str::from_utf8_unchecked(&self.digest_hex[..64]) }
    }

    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    /// Amendment 43 canonical byte serialisation (hex-encoded).
    /// This is the pre-image of the hash: `digest == hex(blake3(unhex(canonical_bytes)))`.
    fn canonical_bytes(&self) -> &str {
        &self.canonical_hex
    }

    fn witt_length(&self) -> u64 {
        256 // 256-bit address space (32-byte BLAKE3)
    }
}

impl core::fmt::Debug for HoloAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let hex = unsafe { core::str::from_utf8_unchecked(&self.digest_hex[..64]) };
        f.debug_struct("HoloAddress").field("digest", &hex).finish()
    }
}

// ── CompileUnit trait implementation ──────────────────────────────────────────

impl hologram_foundation::reduction::CompileUnit<HoloPrimitives> for HoloCompileUnit {
    type TermExpression = HoloTerm;

    #[inline]
    fn root_term(&self) -> &Self::TermExpression {
        &HOLO_TERM_SINGLETON
    }

    #[inline]
    fn unit_witt_level(&self) -> WittLevel {
        self.witt_level
    }

    fn target_domains(&self) -> &[VerificationDomain] {
        &self.target_domains_array[..self.target_domain_count as usize]
    }

    #[inline]
    fn thermodynamic_budget(&self) -> f64 {
        self.thermodynamic_budget
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
        witt_level: WittLevel,
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
            witt_level,
            target_domains_array,
            target_domain_count: count as u8,
            thermodynamic_budget,
            unit_address: [0u8; 32],
            address: HoloAddress::zero(),
            preflight: PreflightStatus::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::PrimOp;
    use crate::term::TermKind;
    use hologram_foundation::WittLevel;

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
            WittLevel::W8,
            6.0, // > 5.545 minimum for Q0
            &[VerificationDomain::Algebraic],
        );

        assert_eq!(unit.root_term, root);
        assert_eq!(unit.witt_level, WittLevel::W8);
        assert_eq!(unit.thermodynamic_budget, 6.0);
        assert_eq!(unit.target_domain_count, 1);
        assert_eq!(unit.arena.len(), 3);
    }

    #[test]
    fn compile_unit_trait_impl() {
        use hologram_foundation::reduction::CompileUnit;

        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(42));

        let unit = HoloCompileUnit::new(
            arena,
            root,
            WittLevel::W16,
            12.0,
            &[
                VerificationDomain::Algebraic,
                VerificationDomain::Thermodynamic,
            ],
        );

        assert_eq!(unit.unit_witt_level(), WittLevel::W16);
        assert_eq!(unit.thermodynamic_budget(), 12.0);
        assert_eq!(unit.target_domains().len(), 2);
    }

    #[test]
    fn holo_address_digest_correctness() {
        use hologram_foundation::address::Element;
        let hash = [0xABu8; 32];
        let addr = HoloAddress::from_hash(hash);
        let digest = addr.digest();
        assert_eq!(digest.len(), 64);
        assert_eq!(&digest[..2], "ab"); // first byte 0xAB → "ab"
        assert_eq!(addr.digest_algorithm(), "blake3");
        assert_eq!(addr.length(), 32);
        // v0.2.0 renamed `quantum()` to `witt_length()`.
        assert_eq!(addr.witt_length(), 256);
    }

    // The v0.1.4 `holo_address_glyph_valid_utf8` test exercised the
    // `glyph()` method on the Element trait. v0.2.0 removed `glyph()` from
    // the trait surface; the Braille glyph buffer is now an inherent
    // implementation detail of `HoloAddress`. The test is removed here
    // because the trait surface no longer exposes the access.

    #[test]
    fn holo_address_zero() {
        use hologram_foundation::address::Element;
        let addr = HoloAddress::zero();
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
