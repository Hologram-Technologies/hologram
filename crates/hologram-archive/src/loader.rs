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
                // Checked: offset/length are attacker-controlled u64s from the archive header, and
                // `usize` is 32-bit on wasm32 / bare-metal — `start + length` would otherwise
                // overflow on a forged section (parser-hardening, spec 03).
                let end = start
                    .checked_add(s.length as usize)
                    .ok_or(ArchiveError::Truncated {
                        needed: usize::MAX,
                        actual: self.bytes.len(),
                    })?;
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

    /// The `.holo` v3 application manifest section bytes, if present (spec 03) —
    /// the opaque canonical form of an `AppManifest` realization, decoded by the
    /// app-load layer (`hologram-space`). `None` for a bare tensor archive that
    /// carries no manifest (a v2 archive, or a v3 archive read purely as a
    /// tensor container). Zero-copy: the slice borrows the archive.
    pub fn app_manifest(&self) -> Option<&'a [u8]> {
        self.section(SectionKind::AppManifest).ok()
    }

    /// Every `Extension` section, parsed to `(key, bytes)` in archive order
    /// (zero-copy: `bytes` borrows the archive). Open producer metadata
    /// (tokenizer, generation config, …); the runtime carries it opaquely.
    pub fn extensions(&self) -> Result<alloc::vec::Vec<(&'a str, &'a [u8])>, ArchiveError> {
        let mut out = alloc::vec::Vec::new();
        for s in &self.sections {
            if s.kind != SectionKind::Extension {
                continue;
            }
            let start = s.offset as usize;
            // Checked (see `section`): a forged u64 offset/length must not overflow `usize`.
            let end = start
                .checked_add(s.length as usize)
                .ok_or(ArchiveError::Truncated {
                    needed: usize::MAX,
                    actual: self.bytes.len(),
                })?;
            let payload = self.bytes.get(start..end).ok_or(ArchiveError::Truncated {
                needed: end,
                actual: self.bytes.len(),
            })?;
            if payload.len() < 2 {
                return Err(ArchiveError::Truncated {
                    needed: 2,
                    actual: payload.len(),
                });
            }
            let key_len = u16::from_le_bytes(payload[..2].try_into().unwrap()) as usize;
            let key = payload
                .get(2..2 + key_len)
                .ok_or(ArchiveError::Truncated {
                    needed: 2 + key_len,
                    actual: payload.len(),
                })
                .and_then(|b| {
                    core::str::from_utf8(b)
                        .map_err(|_| ArchiveError::Io("extension key is not valid UTF-8"))
                })?;
            out.push((key, &payload[2 + key_len..]));
        }
        Ok(out)
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
        // Read-shim (spec 03 §Compatibility): accept v2 tensor archives through
        // the current version; writers emit v3 only. Below MIN_READ_VERSION or
        // above the current version is rejected.
        if !(crate::format::MIN_READ_VERSION..=FORMAT_VERSION).contains(&ver) {
            return Err(ArchiveError::UnsupportedVersion(ver));
        }

        // Footer verification: bytes[len-32..] must equal the 32-byte
        // content fingerprint over bytes[..len-32], computed through
        // hologram's canonical `Hasher<32>` selection
        // (`prism::crypto::Blake3Hasher` per wiki ADR-031).
        use prism::vocabulary::Hasher;
        let footer_start = bytes.len() - 32;
        let expected: [u8; 32] = hologram_types::HologramHasher::initial()
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
                14 => SectionKind::Extension,
                15 => SectionKind::AppManifest,
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
