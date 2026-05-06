//! BLAKE3-deduped weight store (spec X.3).

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightFingerprint(pub [u8; 32]);

impl WeightFingerprint {
    pub fn of(bytes: &[u8]) -> Self {
        Self(blake3::hash(bytes).into())
    }
}

#[derive(Debug, Default, Clone)]
pub struct WeightStore {
    /// Body keyed by fingerprint.
    bodies: HashMap<WeightFingerprint, Vec<u8>>,
}

impl WeightStore {
    pub fn new() -> Self { Self::default() }

    /// Insert weight bytes; returns the dedup key. Duplicate bodies share storage.
    pub fn insert(&mut self, bytes: Vec<u8>) -> WeightFingerprint {
        let fp = WeightFingerprint::of(&bytes);
        self.bodies.entry(fp).or_insert(bytes);
        fp
    }

    pub fn get(&self, fp: WeightFingerprint) -> Option<&[u8]> {
        self.bodies.get(&fp).map(|v| v.as_slice())
    }

    pub fn entries(&self) -> impl Iterator<Item = (&WeightFingerprint, &Vec<u8>)> {
        self.bodies.iter()
    }

    pub fn len(&self) -> usize { self.bodies.len() }
    pub fn is_empty(&self) -> bool { self.bodies.is_empty() }
}
