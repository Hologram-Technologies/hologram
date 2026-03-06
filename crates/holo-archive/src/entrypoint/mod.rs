//! Execution entrypoints: layer descriptors and tensor ports.

pub mod schedule;

use crate::weight::WeightDType;

/// Unique identifier for a layer.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub struct LayerId(pub u32);

/// Describes a single tensor I/O port on a layer.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct TensorPort {
    /// Port name (e.g. "hidden_state").
    pub name: String,
    /// Expected shape dimensions.
    pub shape: Vec<u64>,
    /// Expected data type.
    pub dtype: WeightDType,
}

/// What a layer executes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub enum LayerEntrypoint {
    /// Execute the archive's embedded graph.
    Graph,
    /// Execute a named subgraph by ID.
    Subgraph(u32),
    /// External reference (future: network distribution).
    External(String),
}

/// Descriptor for a single executable layer in the archive.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct LayerDescriptor {
    /// Layer identifier.
    pub id: LayerId,
    /// Human-readable name.
    pub name: String,
    /// What this layer executes.
    pub entrypoint: LayerEntrypoint,
    /// Input tensor ports.
    pub inputs: Vec<TensorPort>,
    /// Output tensor ports.
    pub outputs: Vec<TensorPort>,
    /// Parallel execution group.
    pub group: u32,
    /// Byte offset of this layer's plan in the archive.
    pub plan_offset: u64,
    /// Byte size of this layer's plan.
    pub plan_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_id_equality() {
        assert_eq!(LayerId(0), LayerId(0));
        assert_ne!(LayerId(0), LayerId(1));
    }

    #[test]
    fn tensor_port_construction() {
        let p = TensorPort {
            name: "x".into(),
            shape: vec![1, 3, 224, 224],
            dtype: WeightDType::F32,
        };
        assert_eq!(p.name, "x");
        assert_eq!(p.shape.len(), 4);
    }

    #[test]
    fn layer_entrypoint_variants() {
        let g = LayerEntrypoint::Graph;
        let s = LayerEntrypoint::Subgraph(3);
        let e = LayerEntrypoint::External("http://x".into());
        assert_ne!(g, s);
        assert_ne!(s, e);
    }

    #[test]
    fn rkyv_layer_descriptor() {
        let ld = LayerDescriptor {
            id: LayerId(0),
            name: "encoder".into(),
            entrypoint: LayerEntrypoint::Graph,
            inputs: vec![TensorPort {
                name: "in".into(),
                shape: vec![1, 768],
                dtype: WeightDType::F32,
            }],
            outputs: vec![],
            group: 0,
            plan_offset: 4096,
            plan_size: 512,
        };
        let bytes = rkyv::to_bytes::<_, 512>(&ld).unwrap();
        let archived = rkyv::check_archived_root::<LayerDescriptor>(&bytes).unwrap();
        assert_eq!(archived.name.as_str(), "encoder");
        assert_eq!(archived.plan_offset, 4096);
    }
}
