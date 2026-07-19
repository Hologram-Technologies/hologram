//! `Validated<LiftChainCertificate>` serialization (spec X.1 Certificates section).
//!
//! Each certificate encodes as:
//!   u16 witt_bits
//!   u8 width_bytes (fingerprint active width)
//!   [u8; 32] content_fingerprint bytes
//!
//! The encoding is per-node; the section concatenates count (u32) + entries.

use crate::error::ArchiveError;
use alloc::vec::Vec;
use prism::seal::Validated;
use prism::uor_foundation::enforcement::LiftChainCertificate;

/// One node's certificate in serialized form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CertificateRecord {
    pub witt_bits: u16,
    pub width_bytes: u8,
    pub fingerprint: [u8; 32],
}

impl CertificateRecord {
    pub fn from_validated(v: &Validated<LiftChainCertificate>) -> Self {
        let cert = v.inner();
        let fp = cert.content_fingerprint();
        Self {
            witt_bits: cert.witt_bits(),
            width_bytes: fp.width_bytes(),
            fingerprint: *fp.as_bytes(),
        }
    }
}

/// Encode a sequence of certificate records into a byte blob.
pub fn encode(records: &[CertificateRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + records.len() * 35);
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for r in records {
        out.extend_from_slice(&r.witt_bits.to_le_bytes());
        out.push(r.width_bytes);
        out.extend_from_slice(&r.fingerprint);
    }
    out
}

/// Decode the blob written by `encode`.
pub fn decode(bytes: &[u8]) -> Result<Vec<CertificateRecord>, ArchiveError> {
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated {
            needed: 4,
            actual: bytes.len(),
        });
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count.min(bytes.len())); // cap on untrusted count (DoS)
    let entry_size = 2 + 1 + 32;
    let mut cursor = 4usize;
    for _ in 0..count {
        if cursor + entry_size > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: cursor + entry_size,
                actual: bytes.len(),
            });
        }
        let witt_bits = u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        let width_bytes = bytes[cursor];
        cursor += 1;
        let fingerprint: [u8; 32] = bytes[cursor..cursor + 32].try_into().unwrap();
        cursor += 32;
        out.push(CertificateRecord {
            witt_bits,
            width_bytes,
            fingerprint,
        });
    }
    Ok(out)
}
