//! Layer execution schedule: layers + parallel level ordering.

use super::{LayerDescriptor, LayerId};
use crate::section::{EmbeddableSection, SECTION_LAYER_HEADER};

/// Execution plan header: layers + execution schedule.
///
/// Embedded as a section in the archive. Contains all layer descriptors
/// and their execution ordering as parallel level groups.
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
pub struct LayerHeader {
    /// All layer descriptors in this archive.
    pub layers: Vec<LayerDescriptor>,
    /// Execution order as parallel level groups.
    pub schedule: Vec<Vec<LayerId>>,
}

impl Default for LayerHeader {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerHeader {
    /// Create an empty layer header.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            schedule: Vec::new(),
        }
    }

    /// Number of layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Number of schedule levels.
    #[must_use]
    pub fn num_levels(&self) -> usize {
        self.schedule.len()
    }

    /// Find a layer by ID.
    #[must_use]
    pub fn find_layer(&self, id: LayerId) -> Option<&LayerDescriptor> {
        self.layers.iter().find(|l| l.id == id)
    }
}

impl EmbeddableSection for LayerHeader {
    fn section_kind(&self) -> u32 {
        SECTION_LAYER_HEADER
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 1024>(self)
            .expect("LayerHeader serialization")
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entrypoint::{LayerEntrypoint, TensorPort};
    use crate::weight::WeightDType;

    fn sample_layer() -> LayerDescriptor {
        LayerDescriptor {
            id: LayerId(0),
            name: "test".into(),
            entrypoint: LayerEntrypoint::Graph,
            inputs: vec![TensorPort {
                name: "x".into(),
                shape: vec![1],
                dtype: WeightDType::F32,
            }],
            outputs: vec![],
            group: 0,
            plan_offset: 0,
            plan_size: 0,
        }
    }

    #[test]
    fn new_is_empty() {
        let h = LayerHeader::new();
        assert_eq!(h.layer_count(), 0);
        assert_eq!(h.num_levels(), 0);
    }

    #[test]
    fn find_layer() {
        let mut h = LayerHeader::new();
        h.layers.push(sample_layer());
        assert!(h.find_layer(LayerId(0)).is_some());
        assert!(h.find_layer(LayerId(99)).is_none());
    }

    #[test]
    fn embeddable_section() {
        let h = LayerHeader::new();
        assert_eq!(h.section_kind(), SECTION_LAYER_HEADER);
        let bytes = h.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn rkyv_round_trip() {
        let mut h = LayerHeader::new();
        h.layers.push(sample_layer());
        h.schedule.push(vec![LayerId(0)]);
        let bytes = rkyv::to_bytes::<_, 1024>(&h).unwrap();
        let archived =
            rkyv::check_archived_root::<LayerHeader>(&bytes).unwrap();
        assert_eq!(archived.layers.len(), 1);
        assert_eq!(archived.schedule.len(), 1);
    }
}
