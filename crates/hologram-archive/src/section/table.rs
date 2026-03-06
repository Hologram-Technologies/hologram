//! Section table: index of sections within the archive.

/// An entry in the section table.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SectionEntry {
    /// Section kind identifier.
    pub kind: u32,
    /// Byte offset within the archive.
    pub offset: u64,
    /// Byte size of the section data.
    pub size: u64,
    /// CRC32 of the section data.
    pub checksum: u32,
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
            checksum: 0xABCD,
        });
        assert_eq!(t.len(), 1);
        let found = t.find(1).unwrap();
        assert_eq!(found.offset, 4096);
        assert_eq!(found.checksum, 0xABCD);
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
            checksum: 0xDEAD,
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&e).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<SectionEntry>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.kind, 2);
        assert_eq!(archived.size, 1024);
    }

    #[test]
    fn rkyv_section_table() {
        let mut t = SectionTable::new();
        t.push(SectionEntry {
            kind: 1,
            offset: 0,
            size: 100,
            checksum: 0,
        });
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&t).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<SectionTable>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.entries.len(), 1);
    }
}
