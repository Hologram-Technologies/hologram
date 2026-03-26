# ADR-001: BLAKE3 Archive Checksums (Format v2)

## Status

Accepted

## Context

The `.holo` archive format (v1) uses CRC32 checksums for integrity verification
of graph and weight sections, section table entries, and tensor metadata. CRC32
is a 32-bit non-cryptographic hash optimised for error detection in serial data
streams.

Meanwhile, the hologram ecosystem already uses BLAKE3 for weight deduplication
(`hologram-archive/src/weight/dedup.rs`). This creates an inconsistency: two
different hash algorithms for related integrity purposes within the same crate.

BLAKE3 offers several advantages over CRC32 for this use case:

- **Consistency**: aligns with existing BLAKE3 usage in weight dedup and UOR
  internals.
- **Inherent parallelism**: BLAKE3's Bao tree structure parallelises hashing of
  large buffers automatically, which matters for multi-GB weight blobs.
- **Stronger integrity**: 256-bit output vs 32-bit provides collision resistance
  suitable for content-addressable storage and deduplication.

## Decision

1. **Replace CRC32 with BLAKE3** for all archive checksums: header fields
   (`graph_checksum`, `weights_checksum`), section table entries, and tensor
   metadata.
2. **Expand checksum fields** from `u32` (4 bytes) to `[u8; 32]` (32 bytes).
3. **Bump `FORMAT_VERSION`** from 1 to 2.
4. **Clean break**: no backward compatibility with v1 archives. Existing v1
   archives must be recompiled. The format is pre-1.0 and there are no external
   consumers.
5. **Remove `crc32fast`** dependency from `hologram-archive`.

### Header layout change

The `HoloHeader` (`repr(C)`, `Pod`) grows from 80 to 136 bytes:

```
 0..4    magic: [u8; 4]
 4..8    version: u32
 8..64   (unchanged u64 fields)
64..96   graph_checksum: [u8; 32]    (was u32 at 64..68)
96..128  weights_checksum: [u8; 32]  (was u32 at 68..72)
128..132 section_count: u32          (was at 72..76)
132..136 flags: u32                  (was at 76..80)
```

## Consequences

- All existing `.holo` archives become unloadable (acceptable: pre-1.0 format).
- `HEADER_SIZE` increases from 80 to 136 bytes (negligible impact on file size).
- The `crc32fast` dependency is removed, simplifying the dependency tree.
- Large weight blobs benefit from BLAKE3's internal parallelism during both
  compilation (write) and verified loading (read).
