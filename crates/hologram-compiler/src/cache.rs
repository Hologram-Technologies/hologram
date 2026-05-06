//! Per-compile certificate cache (spec VII.4).
//!
//! Lives entirely in compiler memory; no persistence. Keyed by content
//! fingerprint of (op_marker_iri, witt_level, backend_kind).

use hashbrown::HashMap;
use hologram_archive::certificate_codec::CertificateRecord;
use hologram_backend::KernelCall;
use uor_foundation::enforcement::ContentFingerprint;

#[derive(Debug, Clone)]
pub struct CachedCertificate {
    /// Per-node certificate record (witt_bits + width_bytes + fingerprint).
    pub record: CertificateRecord,
    pub kernel_call: KernelCall,
}

#[derive(Default)]
pub struct CertificateCache {
    map: HashMap<[u8; 32], CachedCertificate>,
}

impl CertificateCache {
    pub fn new() -> Self { Self::default() }

    pub fn get(&self, fp: &ContentFingerprint<32>) -> Option<&CachedCertificate> {
        self.map.get(fp.as_bytes())
    }

    pub fn insert(&mut self, fp: ContentFingerprint<32>, cached: CachedCertificate) {
        self.map.insert(*fp.as_bytes(), cached);
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    /// Lookup by raw 32-byte fingerprint key.
    pub fn get_raw(&self, key: &[u8; 32]) -> Option<&CachedCertificate> {
        self.map.get(key)
    }

    /// Insert by raw 32-byte fingerprint key.
    pub fn insert_raw(&mut self, key: [u8; 32], value: CachedCertificate) {
        self.map.insert(key, value);
    }
}
