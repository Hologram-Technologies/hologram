//! Section table: index of sections within the archive.

use bytemuck::{Pod, Zeroable};

/// An entry in the section table.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SectionEntry {
    /// Section kind identifier.
    pub kind: u32,
    /// Byte offset within the archive.
    pub offset: u64,
    /// Byte size of the section data.
    pub size: u64,
    /// BLAKE3 hash of the section data.
    pub checksum: [u8; 32],
}

/// Fixed-layout binary representation of a section entry.
///
/// Uses `bytemuck::Pod` for zero-copy serialization (same pattern as
/// `HoloHeader`). Fields ordered u64-first to avoid padding under
/// `#[repr(C)]`: 8 + 8 + 4 + 32 = 52 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct SectionEntryRaw {
    /// Byte offset within the archive.
    pub offset: u64,
    /// Byte size of the section data.
    pub size: u64,
    /// Section kind identifier.
    pub kind: u32,
    /// Padding for alignment after kind (4 bytes).
    pub _pad: u32,
    /// BLAKE3 hash of the section data.
    pub checksum: [u8; 32],
}

/// Size in bytes of the entry-count prefix in the raw section table.
pub const RAW_TABLE_HEADER_SIZE: usize = 4;

impl From<&SectionEntry> for SectionEntryRaw {
    #[inline]
    fn from(e: &SectionEntry) -> Self {
        Self {
            offset: e.offset,
            size: e.size,
            kind: e.kind,
            _pad: 0,
            checksum: e.checksum,
        }
    }
}

impl From<SectionEntryRaw> for SectionEntry {
    #[inline]
    fn from(r: SectionEntryRaw) -> Self {
        Self {
            kind: r.kind,
            offset: r.offset,
            size: r.size,
            checksum: r.checksum,
        }
    }
}

/// Table of all sections in an archive.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct SectionTable {
    /// Section entries.
    pub entries: Vec<SectionEntry>,
}

impl SectionTable {
    /// Create an empty section table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a section entry by kind.
    #[must_use]
    pub fn find(&self, kind: u32) -> Option<&SectionEntry> {
        self.entries.iter().find(|e| e.kind == kind)
    }

    /// Add an entry.
    pub fn push(&mut self, entry: SectionEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the table to a fixed-layout byte vector.
    ///
    /// Format: 4-byte LE entry count, then `N × size_of::<SectionEntryRaw>()` bytes of
    /// [`SectionEntryRaw`] entries via `bytemuck::cast_slice`.
    /// Zero-cost: no rkyv overhead, deterministic size.
    #[must_use]
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        let raw_entries: Vec<SectionEntryRaw> =
            self.entries.iter().map(SectionEntryRaw::from).collect();
        let count = raw_entries.len() as u32;
        let entry_bytes = bytemuck::cast_slice::<SectionEntryRaw, u8>(&raw_entries);
        let mut buf = Vec::with_capacity(RAW_TABLE_HEADER_SIZE + entry_bytes.len());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(entry_bytes);
        buf
    }

    /// Deserialize from the fixed-layout format produced by [`to_raw_bytes`].
    ///
    /// Returns `Err` if the bytes don't conform to the expected layout
    /// (wrong length or missing count header).
    pub fn from_raw_bytes(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < RAW_TABLE_HEADER_SIZE {
            return Err("section table too short for header");
        }
        let count = u32::from_le_bytes(
            data[..4]
                .try_into()
                .map_err(|_| "section table header read failed")?,
        ) as usize;
        let entry_data = &data[RAW_TABLE_HEADER_SIZE..];
        let expected = count * core::mem::size_of::<SectionEntryRaw>();
        if entry_data.len() < expected {
            return Err("section table truncated");
        }
        // bytemuck::cast_slice requires alignment; use pod_read_unaligned per entry.
        let entries: Vec<SectionEntry> = (0..count)
            .map(|i| {
                let start = i * core::mem::size_of::<SectionEntryRaw>();
                let raw: SectionEntryRaw = bytemuck::pod_read_unaligned(
                    &entry_data[start..start + core::mem::size_of::<SectionEntryRaw>()],
                );
                SectionEntry::from(raw)
            })
            .collect();
        Ok(Self { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let t = SectionTable::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn push_and_find() {
        let mut t = SectionTable::new();
        t.push(SectionEntry {
            kind: 1,
            offset: 4096,
            size: 256,
            checksum: [0xAB; 32],
        });
        assert_eq!(t.len(), 1);
        let found = t.find(1).unwrap();
        assert_eq!(found.offset, 4096);
        assert_eq!(found.checksum, [0xAB; 32]);
    }

    #[test]
    fn find_missing() {
        let t = SectionTable::new();
        assert!(t.find(99).is_none());
    }

    #[test]
    fn rkyv_section_entry() {
        let e = SectionEntry {
            kind: 2,
            offset: 8192,
            size: 1024,
            checksum: [0xDE; 32],
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&e).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<SectionEntry>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.kind, 2);
        assert_eq!(archived.size, 1024);
    }

    #[test]
    fn raw_roundtrip() {
        let mut t = SectionTable::new();
        t.push(SectionEntry {
            kind: 1,
            offset: 4096,
            size: 256,
            checksum: [0xAB; 32],
        });
        t.push(SectionEntry {
            kind: 2,
            offset: 8192,
            size: 1024,
            checksum: [0xDE; 32],
        });
        let bytes = t.to_raw_bytes();
        assert_eq!(bytes.len(), 4 + 2 * core::mem::size_of::<SectionEntryRaw>());
        let t2 = SectionTable::from_raw_bytes(&bytes).expect("roundtrip should succeed");
        assert_eq!(t, t2);
    }

    #[test]
    fn raw_bytes_bad_data_returns_none() {
        assert!(SectionTable::from_raw_bytes(&[]).is_err());
        assert!(SectionTable::from_raw_bytes(&[1, 0, 0, 0]).is_err()); // count=1 but no entries
    }

    #[test]
    fn rkyv_section_table() {
        let mut t = SectionTable::new();
        t.push(SectionEntry {
            kind: 1,
            offset: 0,
            size: 100,
            checksum: [0u8; 32],
        });
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&t).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<SectionTable>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.entries.len(), 1);
    }
}
