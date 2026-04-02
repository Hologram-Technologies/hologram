//! On-disk header for a .holo archive.

use bytemuck::{Pod, Zeroable};

use super::{FORMAT_VERSION, HOLO_MAGIC};

/// Fixed size of the serialized header in bytes.
pub const HEADER_SIZE: usize = 184;

/// On-disk header for a .holo archive.
///
/// Fixed-layout binary struct using `bytemuck` for zero-copy
/// serialization. Fields are ordered to avoid padding under `repr(C)`:
/// u64 fields grouped together, then u32 fields grouped together.
///
/// All multi-byte integers are stored in native byte order (LE on
/// x86/ARM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct HoloHeader {
    /// Magic bytes (must be HOLO_MAGIC).
    pub magic: [u8; 4],
    /// Format version.
    pub version: u32,
    /// Byte offset of the serialized graph section.
    pub graph_offset: u64,
    /// Byte size of the serialized graph section.
    pub graph_size: u64,
    /// Byte offset of the weights section.
    pub weights_offset: u64,
    /// Byte size of the weights section.
    pub weights_size: u64,
    /// Byte offset of the section table.
    pub section_table_offset: u64,
    /// Byte size of the rkyv-serialized section table.
    pub section_table_size: u64,
    /// Total archive size in bytes.
    pub total_size: u64,
    /// Byte offset of the embedded cascade certificate section.
    /// Zero if no certificate is embedded.
    pub certificate_offset: u64,
    /// Byte size of the embedded cascade certificate section.
    /// Zero if no certificate is embedded.
    pub certificate_size: u64,
    /// BLAKE3 hash of the graph section bytes.
    pub graph_checksum: [u8; 32],
    /// BLAKE3 hash of the weights section bytes.
    pub weights_checksum: [u8; 32],
    /// Content-addressed identifier of the root term graph (CS_7).
    /// BLAKE3 hash of `canonicalBytes(transitiveClosure(rootTerm))`.
    /// All zeros if not computed.
    pub unit_address: [u8; 32],
    /// Number of entries in the section table.
    pub section_count: u32,
    /// Reserved flags for future use.
    pub flags: u32,
}

/// Flag bit: graph section is compressed via hologram-compression.
pub const FLAG_GRAPH_COMPRESSED: u32 = 1 << 0;

/// Flag bit: weights section is compressed via hologram-compression.
pub const FLAG_WEIGHTS_COMPRESSED: u32 = 1 << 1;

/// Flag bit: archive contains compression metadata per tensor.
pub const COMPRESSION_FLAG: u32 = 0x0010;

impl HoloHeader {
    /// Whether the graph section is compressed.
    #[must_use]
    pub fn is_graph_compressed(&self) -> bool {
        self.flags & FLAG_GRAPH_COMPRESSED != 0
    }

    /// Whether the weights section is compressed.
    #[must_use]
    pub fn is_weights_compressed(&self) -> bool {
        self.flags & FLAG_WEIGHTS_COMPRESSED != 0
    }

    /// Whether the magic bytes match HOLO_MAGIC.
    #[must_use]
    pub fn is_valid_magic(&self) -> bool {
        self.magic == HOLO_MAGIC
    }

    /// Whether the format version is supported.
    #[must_use]
    pub fn is_supported_version(&self) -> bool {
        self.version == FORMAT_VERSION
    }

    /// Serialize the header to a byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Deserialize a header from a byte slice.
    ///
    /// Uses `pod_read_unaligned` so the input buffer need not be
    /// aligned to the struct's alignment.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < HEADER_SIZE {
            return None;
        }
        Some(bytemuck::pod_read_unaligned::<Self>(&data[..HEADER_SIZE]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> HoloHeader {
        HoloHeader {
            magic: HOLO_MAGIC,
            version: FORMAT_VERSION,
            graph_offset: 4096,
            graph_size: 512,
            weights_offset: 8192,
            weights_size: 1024,
            graph_checksum: [0xDE; 32],
            weights_checksum: [0xCA; 32],
            unit_address: [0u8; 32],
            certificate_offset: 0,
            certificate_size: 0,
            section_count: 0,
            section_table_offset: 0,
            section_table_size: 0,
            total_size: 12288,
            flags: 0,
        }
    }

    #[test]
    fn valid_magic() {
        assert!(sample_header().is_valid_magic());
    }

    #[test]
    fn invalid_magic() {
        let mut h = sample_header();
        h.magic = *b"NOPE";
        assert!(!h.is_valid_magic());
    }

    #[test]
    fn supported_version() {
        assert!(sample_header().is_supported_version());
    }

    #[test]
    fn unsupported_version() {
        let mut h = sample_header();
        h.version = 999;
        assert!(!h.is_supported_version());
    }

    #[test]
    fn header_size_matches() {
        assert_eq!(std::mem::size_of::<HoloHeader>(), HEADER_SIZE);
    }

    #[test]
    fn bytemuck_round_trip() {
        let h = sample_header();
        let bytes = h.as_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);

        let h2 = HoloHeader::from_bytes(bytes).unwrap();
        assert_eq!(h, h2);
        assert_eq!(h2.magic, HOLO_MAGIC);
        assert_eq!(h2.version, FORMAT_VERSION);
        assert_eq!(h2.graph_offset, 4096);
        assert_eq!(h2.weights_checksum, [0xCA; 32]);
    }

    #[test]
    fn from_bytes_too_short() {
        assert!(HoloHeader::from_bytes(&[0u8; 10]).is_none());
    }
}
