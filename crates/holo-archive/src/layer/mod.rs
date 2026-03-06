//! Layer location references for embedded, external, and remote layers.

/// Where a layer's data is located.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub enum LayerLocation {
    /// Embedded within this archive at (offset, size).
    Embedded { offset: u64, size: u64 },
    /// External file path (String, not PathBuf — rkyv compatible).
    External(String),
    /// Network registry reference.
    Registry { url: String, version: String },
}

impl LayerLocation {
    /// Whether this layer is embedded in the archive.
    #[must_use]
    pub fn is_embedded(&self) -> bool {
        matches!(self, Self::Embedded { .. })
    }

    /// Whether this layer requires network access.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Registry { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::Deserialize;

    #[test]
    fn embedded_is_embedded() {
        let loc = LayerLocation::Embedded {
            offset: 0,
            size: 100,
        };
        assert!(loc.is_embedded());
        assert!(!loc.is_remote());
    }

    #[test]
    fn external_not_embedded() {
        let loc = LayerLocation::External("/path/to/model".into());
        assert!(!loc.is_embedded());
        assert!(!loc.is_remote());
    }

    #[test]
    fn registry_is_remote() {
        let loc = LayerLocation::Registry {
            url: "https://registry.example.com".into(),
            version: "1.0.0".into(),
        };
        assert!(!loc.is_embedded());
        assert!(loc.is_remote());
    }

    #[test]
    fn rkyv_round_trip() {
        let loc = LayerLocation::Embedded {
            offset: 4096,
            size: 2048,
        };
        let bytes = rkyv::to_bytes::<_, 128>(&loc).unwrap();
        let archived =
            rkyv::check_archived_root::<LayerLocation>(&bytes).unwrap();
        let deser: LayerLocation =
            archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(deser, loc);
    }
}
