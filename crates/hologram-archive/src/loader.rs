//! Archive loader (spec X.2).

use crate::error::ArchiveError;
use crate::format::{SectionKind, SectionRef, FORMAT_VERSION, MAGIC};
use alloc::vec::Vec;

/// Parsed plan view backed by a byte slice. Zero-copy where possible.
pub struct LoadedPlan<'a> {
    bytes: &'a [u8],
    sections: Vec<SectionRef>,
}

impl<'a> LoadedPlan<'a> {
    pub fn section(&self, kind: SectionKind) -> Result<&'a [u8], ArchiveError> {
        for s in &self.sections {
            if s.kind == kind {
                let start = s.offset as usize;
                let end = start + s.length as usize;
                return self.bytes.get(start..end).ok_or(ArchiveError::Truncated {
                    needed: end,
                    actual: self.bytes.len(),
                });
            }
        }
        Err(ArchiveError::SectionMissing(kind))
    }

    pub fn sections(&self) -> &[SectionRef] {
        &self.sections
    }
}

pub struct HoloLoader<'a> {
    bytes: &'a [u8],
    /// 32-byte BLAKE3 footer fingerprint over `bytes[..len-32]`, captured
    /// at verification time. This is the archive's canonical content
    /// fingerprint per spec X.1 — `execute_attested` routes it through
    /// prism's pipeline as the W256 literal anchoring the
    /// `Grounded<Digest<32>>` attestation.
    fingerprint: [u8; 32],
}

impl<'a> HoloLoader<'a> {
    /// Parse + verify the archive header and footer (BLAKE3 fingerprint over
    /// all preceding bytes, per spec X.1).
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ArchiveError> {
        if bytes.len() < 4 + 2 + 2 + 2 + 32 {
            return Err(ArchiveError::Truncated {
                needed: 40,
                actual: bytes.len(),
            });
        }
        if bytes[..4] != MAGIC {
            let mut m = [0u8; 4];
            m.copy_from_slice(&bytes[..4]);
            return Err(ArchiveError::BadMagic(m));
        }
        let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
        if ver != FORMAT_VERSION {
            return Err(ArchiveError::UnsupportedVersion(ver));
        }

        // Footer verification: bytes[len-32..] must equal the 32-byte
        // content fingerprint over bytes[..len-32], computed through
        // hologram's canonical `Hasher<32>` selection
        // (`prism::crypto::Blake3Hasher` per wiki ADR-031).
        use prism::vocabulary::Hasher;
        let footer_start = bytes.len() - 32;
        let expected: [u8; 32] = hologram_host::HologramHasher::initial()
            .fold_bytes(&bytes[..footer_start])
            .finalize();
        let actual: [u8; 32] =
            bytes[footer_start..]
                .try_into()
                .map_err(|_| ArchiveError::Truncated {
                    needed: bytes.len(),
                    actual: bytes.len(),
                })?;
        if expected != actual {
            return Err(ArchiveError::ChecksumMismatch);
        }

        Ok(Self {
            bytes,
            fingerprint: actual,
        })
    }

    /// Construct a loader without footer verification. For tests and tools
    /// that build partial archives manually.
    pub fn from_bytes_unchecked(bytes: &'a [u8]) -> Result<Self, ArchiveError> {
        if bytes.len() < 4 + 2 + 2 + 2 + 32 {
            return Err(ArchiveError::Truncated {
                needed: 40,
                actual: bytes.len(),
            });
        }
        if bytes[..4] != MAGIC {
            let mut m = [0u8; 4];
            m.copy_from_slice(&bytes[..4]);
            return Err(ArchiveError::BadMagic(m));
        }
        let footer_start = bytes.len() - 32;
        let fingerprint: [u8; 32] =
            bytes[footer_start..]
                .try_into()
                .map_err(|_| ArchiveError::Truncated {
                    needed: bytes.len(),
                    actual: bytes.len(),
                })?;
        Ok(Self { bytes, fingerprint })
    }

    /// Return the archive's canonical 32-byte content fingerprint
    /// (the verified BLAKE3 footer, spec X.1). This is the anchor
    /// `execute_attested` routes through prism::pipeline::run.
    #[inline]
    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    pub fn into_plan(self) -> Result<LoadedPlan<'a>, ArchiveError> {
        let _flags = u16::from_le_bytes([self.bytes[6], self.bytes[7]]);
        let count = u16::from_le_bytes([self.bytes[8], self.bytes[9]]) as usize;
        let mut sections = Vec::with_capacity(count);
        let mut cursor = 10usize;
        for _ in 0..count {
            if cursor + 24 > self.bytes.len() {
                return Err(ArchiveError::Truncated {
                    needed: cursor + 24,
                    actual: self.bytes.len(),
                });
            }
            let kind_byte = self.bytes[cursor];
            let kind = match kind_byte {
                1 => SectionKind::KernelCalls,
                2 => SectionKind::Schedule,
                3 => SectionKind::Weights,
                4 => SectionKind::ShapeRegistry,
                5 => SectionKind::DTypeRegistry,
                6 => SectionKind::Certificates,
                7 => SectionKind::Trace,
                8 => SectionKind::Metadata,
                9 => SectionKind::Inputs,
                10 => SectionKind::Outputs,
                11 => SectionKind::Constants,
                12 => SectionKind::ExecPlan,
                13 => SectionKind::WarmStart,
                14 => SectionKind::TierAssignments,
                _ => return Err(ArchiveError::Io("unknown section kind")),
            };
            cursor += 8; // kind + pad(7)
            let off = u64::from_le_bytes(self.bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            let len = u64::from_le_bytes(self.bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            sections.push(SectionRef {
                kind,
                offset: off,
                length: len,
            });
        }
        Ok(LoadedPlan {
            bytes: self.bytes,
            sections,
        })
    }
}
