//! Node, NodeId, InputSlot types for the compute graph.

extern crate alloc;
use tinyvec::TinyVec;

/// Generational node identifier for safe arena access.
///
/// A stale `NodeId` (wrong generation) safely returns `None` on lookup,
/// preventing use-after-free in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct NodeId {
    index: u32,
    generation: u32,
}

impl NodeId {
    /// Create a new NodeId.
    #[inline]
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// A null sentinel (never matches a valid slot).
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self {
            index: u32::MAX,
            generation: u32::MAX,
        }
    }

    /// Whether this is the null sentinel.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.index == u32::MAX && self.generation == u32::MAX
    }

    /// Arena slot index.
    #[inline]
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Generation counter for this slot.
    #[inline]
    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generation
    }
}

impl core::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_null() {
            write!(f, "NodeId(null)")
        } else {
            write!(f, "NodeId({}g{})", self.index, self.generation)
        }
    }
}

/// Where a node input originates.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum InputSource {
    /// From another node's output.
    Node(NodeId),
    /// From a graph-level input port.
    GraphInput { index: u32 },
    /// Not connected.
    None,
}

impl Default for InputSource {
    #[inline]
    fn default() -> Self {
        Self::None
    }
}

/// An input connection to a node.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct InputSlot {
    /// Where this input comes from.
    pub source: InputSource,
    /// Which output port of the source (0 for single-output nodes).
    pub output_port: u32,
}

impl InputSlot {
    /// Input from another node (port 0).
    #[inline]
    #[must_use]
    pub const fn from_node(id: NodeId) -> Self {
        Self {
            source: InputSource::Node(id),
            output_port: 0,
        }
    }

    /// Input from another node at a specific output port.
    #[inline]
    #[must_use]
    pub const fn from_node_port(id: NodeId, port: u32) -> Self {
        Self {
            source: InputSource::Node(id),
            output_port: port,
        }
    }

    /// Input from a graph-level input port.
    #[inline]
    #[must_use]
    pub const fn from_graph_input(index: u32) -> Self {
        Self {
            source: InputSource::GraphInput { index },
            output_port: 0,
        }
    }

    /// Whether this slot has no source connected.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self.source, InputSource::None)
    }
}

use super::GraphOp;

/// A node in the compute graph.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Node {
    /// Unique generational identifier.
    pub id: NodeId,
    /// The operation this node performs.
    pub op: GraphOp,
    /// Input connections. `TinyVec<[InputSlot; 2]>` inlines up to 2 inputs
    /// (covers unary + binary ops — the common case) without heap allocation.
    /// Spills to heap for variadic ops (Concat, etc.).
    pub inputs: TinyVec<[InputSlot; 2]>,
    /// Number of output ports.
    pub num_outputs: u32,
}

impl Node {
    /// Create a new node with the given ID and op.
    #[inline]
    #[must_use]
    pub fn new(id: NodeId, op: GraphOp) -> Self {
        Self {
            id,
            op,
            inputs: TinyVec::new(),
            num_outputs: 1,
        }
    }

    /// Iterator over predecessor NodeIds (skips non-node sources).
    pub fn dependencies(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inputs.iter().filter_map(|slot| match slot.source {
            InputSource::Node(id) => Some(id),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_null() {
        let null = NodeId::null();
        assert!(null.is_null());
        assert!(!NodeId::new(0, 0).is_null());
    }

    #[test]
    fn node_id_eq() {
        let a = NodeId::new(1, 0);
        let b = NodeId::new(1, 0);
        let c = NodeId::new(1, 1); // different generation
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn node_id_debug() {
        let id = NodeId::new(3, 1);
        let s = alloc::format!("{id:?}");
        assert_eq!(s, "NodeId(3g1)");
        let null = NodeId::null();
        assert_eq!(alloc::format!("{null:?}"), "NodeId(null)");
    }

    #[test]
    fn input_slot_from_node() {
        let id = NodeId::new(5, 0);
        let slot = InputSlot::from_node(id);
        assert_eq!(slot.source, InputSource::Node(id));
        assert_eq!(slot.output_port, 0);
        assert!(!slot.is_empty());
    }

    #[test]
    fn input_slot_from_graph_input() {
        let slot = InputSlot::from_graph_input(2);
        assert_eq!(slot.source, InputSource::GraphInput { index: 2 });
        assert!(!slot.is_empty());
    }

    #[test]
    fn input_slot_default_is_empty() {
        let slot = InputSlot::default();
        assert!(slot.is_empty());
    }

    #[test]
    fn node_dependencies() {
        let a = NodeId::new(0, 0);
        let b = NodeId::new(1, 0);
        let mut node = Node::new(NodeId::new(2, 0), GraphOp::Output);
        node.inputs.push(InputSlot::from_node(a));
        node.inputs.push(InputSlot::from_graph_input(0));
        node.inputs.push(InputSlot::from_node(b));
        let deps: alloc::vec::Vec<_> = node.dependencies().collect();
        assert_eq!(deps, alloc::vec![a, b]);
    }

    #[test]
    fn rkyv_node_id_round_trip() {
        let id = NodeId::new(42, 7);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&id).unwrap();
        let archived = rkyv::access::<rkyv::Archived<NodeId>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.index, 42);
        assert_eq!(archived.generation, 7);
    }
}
