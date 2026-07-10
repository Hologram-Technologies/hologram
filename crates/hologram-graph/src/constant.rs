//! Inline constant store.

use crate::node::ConstantId;
use crate::registry::{DTypeId, ShapeId};
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct ConstantEntry {
    pub bytes: Vec<u8>,
    pub dtype: DTypeId,
    pub shape: ShapeId,
}

#[derive(Debug, Default, Clone)]
pub struct ConstantStore {
    entries: Vec<ConstantEntry>,
    /// Per-entry **external κ**: the content fingerprint of a body that is not
    /// in the graph and arrives at materialization. `None` for an ordinary
    /// constant, whose bytes are right here.
    external: Vec<Option<[u8; 32]>>,
}

impl ConstantStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entry: ConstantEntry) -> ConstantId {
        let id = ConstantId(self.entries.len() as u32);
        self.entries.push(entry);
        self.external.push(None);
        id
    }

    /// Insert a **weightless** constant: the graph carries the weight's κ, not
    /// its bytes, and the body arrives at materialization through a
    /// `WeightProvider`.
    ///
    /// This is the binding a weight-paging consumer uses. It is what makes
    /// `QuantAttrs::weight_layout = OUTPUT_MAJOR` reachable: the compiler has no
    /// bytes to transpose, so the *binder* promises to materialize `[n, k]`, and
    /// the load-time fusion emits the fused output-major decode call. An
    /// ordinary constant carries its bytes in `[k, n]` and may not make that
    /// promise — the compiler transposes such a weight itself.
    ///
    /// `shape` is still the logical `[k, n]`; only the bound bytes are `[n, k]`.
    /// The archive emits this entry **by reference**, with no body, so a
    /// weightless archive is dedupable across models and carries no weight
    /// bytes at all. Loading it requires `InferenceSession::load_paged` with a
    /// provider that has `kappa`; a fully-resident `load` fails loud.
    pub fn insert_external(
        &mut self,
        dtype: DTypeId,
        shape: ShapeId,
        kappa: [u8; 32],
    ) -> ConstantId {
        let id = ConstantId(self.entries.len() as u32);
        self.entries.push(ConstantEntry {
            bytes: Vec::new(),
            dtype,
            shape,
        });
        self.external.push(Some(kappa));
        id
    }

    pub fn get(&self, id: ConstantId) -> Option<&ConstantEntry> {
        self.entries.get(id.0 as usize)
    }

    /// The external κ of a weightless constant, or `None` if its bytes are in
    /// the graph.
    pub fn external(&self, id: ConstantId) -> Option<[u8; 32]> {
        self.external.get(id.0 as usize).copied().flatten()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
