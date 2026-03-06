//! Constant storage for graph nodes.

extern crate alloc;
use alloc::vec::Vec;

/// Identifier for a constant in the ConstantStore.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub struct ConstantId(u32);

impl ConstantId {
    /// Create a new constant identifier.
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The raw index.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> u32 {
        self.0
    }
}

/// Constant data stored in the graph.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub enum ConstantData {
    /// Inline byte blob.
    Bytes(Vec<u8>),
    /// Deferred: loaded from archive at runtime.
    Deferred { byte_size: u64, source_id: u64 },
}

impl ConstantData {
    /// Byte length of the data.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        match self {
            Self::Bytes(v) => v.len() as u64,
            Self::Deferred { byte_size, .. } => *byte_size,
        }
    }

    /// Whether this is deferred (not yet loaded).
    #[must_use]
    pub const fn is_deferred(&self) -> bool {
        matches!(self, Self::Deferred { .. })
    }
}

/// Store for all constants referenced by graph nodes.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub struct ConstantStore {
    data: Vec<ConstantData>,
}

impl ConstantStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Insert a constant and return its ID.
    pub fn insert(&mut self, constant: ConstantData) -> ConstantId {
        let id = ConstantId(self.data.len() as u32);
        self.data.push(constant);
        id
    }

    /// Look up a constant by ID.
    #[must_use]
    pub fn get(&self, id: ConstantId) -> Option<&ConstantData> {
        self.data.get(id.0 as usize)
    }

    /// Number of stored constants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut store = ConstantStore::new();
        let id = store.insert(ConstantData::Bytes(alloc::vec![42]));
        assert_eq!(id.raw(), 0);
        assert_eq!(store.get(id), Some(&ConstantData::Bytes(alloc::vec![42])));
    }

    #[test]
    fn multiple_inserts() {
        let mut store = ConstantStore::new();
        let a = store.insert(ConstantData::Bytes(alloc::vec![1]));
        let b = store.insert(ConstantData::Bytes(alloc::vec![2]));
        assert_eq!(a.raw(), 0);
        assert_eq!(b.raw(), 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn get_invalid() {
        let store = ConstantStore::new();
        assert!(store.get(ConstantId::new(99)).is_none());
    }

    #[test]
    fn empty_store() {
        let store = ConstantStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn byte_size() {
        let inline = ConstantData::Bytes(alloc::vec![1, 2, 3]);
        assert_eq!(inline.byte_size(), 3);
        let deferred = ConstantData::Deferred {
            byte_size: 1024,
            source_id: 0,
        };
        assert_eq!(deferred.byte_size(), 1024);
        assert!(deferred.is_deferred());
        assert!(!inline.is_deferred());
    }

    #[test]
    fn rkyv_round_trip() {
        let mut store = ConstantStore::new();
        store.insert(ConstantData::Bytes(alloc::vec![10, 20]));
        let bytes = rkyv::to_bytes::<_, 256>(&store).unwrap();
        let archived = rkyv::check_archived_root::<ConstantStore>(&bytes).unwrap();
        assert_eq!(archived.data.len(), 1);
    }
}
