//! Layer location references for embedded, external, and remote layers.

/// Where a layer's data is located.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
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
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&loc).unwrap();
        let deser = rkyv::from_bytes::<LayerLocation, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(deser, loc);
    }
}
