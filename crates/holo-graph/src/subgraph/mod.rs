//! Subgraph templates: reusable graph fragments with named I/O.

extern crate alloc;
use alloc::string::String;

pub mod flatten;

use crate::graph::Graph;

/// A reusable graph template with named I/O ports.
#[derive(Debug, Clone)]
pub struct SubgraphDef {
    /// Template name.
    pub name: String,
    /// The template graph.
    pub graph: Graph,
}

impl SubgraphDef {
    /// Create a new subgraph template.
    #[must_use]
    pub fn new(name: impl Into<String>, graph: Graph) -> Self {
        Self {
            name: name.into(),
            graph,
        }
    }

    /// Number of input ports.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.graph.inputs().len()
    }

    /// Number of output ports.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.graph.outputs().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subgraph_def_creation() {
        let g = Graph::new();
        let def = SubgraphDef::new("test", g);
        assert_eq!(def.name, "test");
        assert_eq!(def.num_inputs(), 0);
        assert_eq!(def.num_outputs(), 0);
    }
}
