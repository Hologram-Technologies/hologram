//! CompileUnit metadata section for .holo archives.
//!
//! Embeds provenance information from the CompileUnit that produced this archive:
//! unit address, Witt length, budget, domain count, and term count.

use super::{EmbeddableSection, SECTION_COMPILE_UNIT_META};

/// CompileUnit metadata embedded in the archive.
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct CompileUnitMeta {
    /// Content-addressed identifier (BLAKE3 hash).
    pub unit_address: [u8; 32],
    /// Witt length in bits — the bit width of the unit's declared
    /// `WittLevel` (8, 16, 24, 32, or any custom width). Replaces the
    /// v0.1.4 `quantum_level: u8` index field per the Phase 10
    /// no-backwards-compat cleanup.
    pub witt_length: u32,
    /// Thermodynamic budget in k_B T units.
    pub budget: f64,
    /// Number of target verification domains.
    pub domain_count: u8,
    /// Number of term nodes in the arena.
    pub term_count: u32,
    /// Number of let-bindings.
    pub binding_count: u8,
    /// Number of assertions.
    pub assertion_count: u8,
}

impl EmbeddableSection for CompileUnitMeta {
    fn section_kind(&self) -> u32 {
        SECTION_COMPILE_UNIT_META
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("CompileUnitMeta serialization")
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddable_section_kind() {
        let meta = CompileUnitMeta {
            unit_address: [0u8; 32],
            witt_length: 8,
            budget: 100.0,
            domain_count: 1,
            term_count: 10,
            binding_count: 0,
            assertion_count: 0,
        };
        assert_eq!(meta.section_kind(), SECTION_COMPILE_UNIT_META);
    }

    #[test]
    fn rkyv_round_trip() {
        let meta = CompileUnitMeta {
            unit_address: [42u8; 32],
            witt_length: 16,
            budget: 50.5,
            domain_count: 3,
            term_count: 100,
            binding_count: 5,
            assertion_count: 2,
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&meta).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<CompileUnitMeta>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.witt_length, 16);
        assert_eq!(archived.term_count, 100);
        assert_eq!(archived.binding_count, 5);
    }
}
