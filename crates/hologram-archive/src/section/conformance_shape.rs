//! Conformance shape section for `.holo` archives.
//!
//! Embeds the `Shape` declaration that the archive's compiled tape conforms
//! to. Every archive emitted by `hologram-compiler` carries this section,
//! and the loader (or the loading `PrismModule`) validates the declared
//! shape against the loading module's expected shape before any execution
//! proceeds.
//!
//! This is the v0.2.0 conformance-first contract: an archive is not just a
//! blob of compiled bytes — it is a *value* in `Val(F)` for a specific
//! declared `F`, and that `F` is identified at the file level so loaders
//! can refuse to load mismatched archives.
//!
//! # Wire format
//!
//! Stored via `rkyv` like the other section types. The payload is small
//! (~96 bytes including the IRI string) so the section overhead is
//! negligible compared to the rest of the archive.
//!
//! # Performance
//!
//! - **Emission:** one rkyv serialise per compile. **Perf: COMPILE-TIME.**
//! - **Validation:** one byte-array compare (32 bytes) per archive load.
//!   **Perf: NEUTRAL** — adds a single memcmp to the load path.
//! - **Inference:** zero. The section is consulted only at load time.

use super::{EmbeddableSection, SECTION_CONFORMANCE_SHAPE};

/// Conformance shape declaration embedded in the archive header.
///
/// Identifies which shape `F` the compiled tape carries. Two ways to
/// identify the shape are stored:
///
/// 1. **`shape_id`** — the 32-byte content-addressed shape identifier.
///    This is what the loader/Prism module compares for fast equality
///    checking. Computed by the same FNV1a const-fn that
///    `hologram_shapes::Shape` uses, so it round-trips deterministically.
///
/// 2. **`shape_iri`** — the human-readable IRI of the shape's target
///    class. Used for diagnostics and for cross-process introspection
///    (e.g., a tool that lists all archives in a directory and reports
///    which shapes they target without loading the runtime).
///
/// The `primitive_count` field is a sanity check: if the loaded shape's
/// declared `primitives.len()` doesn't match the count stored in the
/// archive, the loader can reject the archive even before doing the
/// shape ID comparison.
///
/// `min_witt_length` and `max_witt_length` document the Witt-level range
/// the archive's compiled tape uses. The loader can refuse archives whose
/// range exceeds the loading module's `substrate_requirements()`.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ConformanceShapeSection {
    /// 32-byte content-addressed identifier of the declared shape.
    pub shape_id: [u8; 32],
    /// IRI of the shape's target class (e.g.,
    /// `"https://hologram.uor.foundation/conformance/F_prism_fused_component"`).
    pub shape_iri: String,
    /// Human-readable name of the shape (e.g., `"F_prism_fused_component"`).
    pub shape_name: String,
    /// Number of primitives declared by the shape's algebra. The loader
    /// validates this against the loaded shape's `primitives.len()`.
    pub primitive_count: u32,
    /// Minimum Witt length (bit width) used by the compiled tape.
    pub min_witt_length: u32,
    /// Maximum Witt length (bit width) used by the compiled tape.
    pub max_witt_length: u32,
}

impl ConformanceShapeSection {
    /// Construct a new section. Accepts `&str` for IRI and name —
    /// conversion to `String` happens here at the rkyv serialization
    /// boundary, not at the call site.
    #[must_use]
    pub fn new(
        shape_id: [u8; 32],
        shape_iri: &str,
        shape_name: &str,
        primitive_count: u32,
        min_witt_length: u32,
        max_witt_length: u32,
    ) -> Self {
        Self {
            shape_id,
            shape_iri: shape_iri.to_owned(),
            shape_name: shape_name.to_owned(),
            primitive_count,
            min_witt_length,
            max_witt_length,
        }
    }
}

impl EmbeddableSection for ConformanceShapeSection {
    fn section_kind(&self) -> u32 {
        SECTION_CONFORMANCE_SHAPE
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("ConformanceShapeSection serialization")
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_section() -> ConformanceShapeSection {
        ConformanceShapeSection::new(
            [0xABu8; 32],
            "https://hologram.uor.foundation/conformance/F_prism_fused_component",
            "F_prism_fused_component",
            60,
            8,
            32,
        )
    }

    #[test]
    fn embeddable_section_kind() {
        let s = sample_section();
        assert_eq!(s.section_kind(), SECTION_CONFORMANCE_SHAPE);
    }

    #[test]
    fn rkyv_round_trip() {
        let s = sample_section();
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&s).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<ConformanceShapeSection>, rkyv::rancor::Error>(&bytes)
                .unwrap();
        assert_eq!(archived.shape_id, [0xABu8; 32]);
        assert_eq!(&*archived.shape_name, "F_prism_fused_component");
        assert_eq!(archived.primitive_count, 60);
        assert_eq!(archived.min_witt_length, 8);
        assert_eq!(archived.max_witt_length, 32);
    }

    #[test]
    fn deserialize_via_to_bytes() {
        let s = sample_section();
        let bytes = s.to_bytes();
        let recovered: ConformanceShapeSection =
            rkyv::from_bytes::<ConformanceShapeSection, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(recovered, s);
    }

    #[test]
    fn round_trip_preserves_iri_with_special_characters() {
        // Non-ASCII and IRI fragment characters should round-trip cleanly
        // because rkyv's String impl is byte-faithful.
        let s = ConformanceShapeSection::new(
            [0u8; 32],
            "https://hologram.uor.foundation/conformance/Fμ_prism#fused-component_v2",
            "F_prism_fused_component_with_unusual_name",
            42,
            8,
            32,
        );
        let bytes = s.to_bytes();
        let recovered: ConformanceShapeSection =
            rkyv::from_bytes::<ConformanceShapeSection, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(recovered, s);
    }

    #[test]
    fn corrupt_payload_fails_to_deserialize() {
        // Garbage bytes must produce an Err rather than panicking or
        // returning a half-built struct.
        let garbage = vec![0xFFu8; 16];
        let result = rkyv::from_bytes::<ConformanceShapeSection, rkyv::rancor::Error>(&garbage);
        assert!(result.is_err());
    }

    #[test]
    fn truncated_payload_fails_to_deserialize() {
        // Take a real serialised payload and slice it down to a single
        // byte — rkyv must reject this rather than panicking.
        let bytes = sample_section().to_bytes();
        let truncated = &bytes[..1];
        let result = rkyv::from_bytes::<ConformanceShapeSection, rkyv::rancor::Error>(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn distinct_shape_ids_round_trip_distinctly() {
        let a = ConformanceShapeSection::new([0xAAu8; 32], "iri-a", "name-a", 5, 8, 8);
        let b = ConformanceShapeSection::new([0xBBu8; 32], "iri-b", "name-b", 5, 8, 8);
        let ra: ConformanceShapeSection =
            rkyv::from_bytes::<_, rkyv::rancor::Error>(&a.to_bytes()).unwrap();
        let rb: ConformanceShapeSection =
            rkyv::from_bytes::<_, rkyv::rancor::Error>(&b.to_bytes()).unwrap();
        assert_ne!(ra.shape_id, rb.shape_id);
        assert_ne!(ra, rb);
    }
}
