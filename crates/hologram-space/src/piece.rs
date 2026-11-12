//! # Verified streams and verified byte ranges over BLAKE3's own tree
//!
//! A `blake3:<hex>` κ-label is the root of BLAKE3's binary hash tree over 1 KiB chunks. This module
//! cuts that tree at a fixed height, [`PIECE_LEN`] = 1 MiB (2^10 chunks), and keeps only that one
//! level: the 32-byte chaining value of every 1 MiB piece, a [`PieceTree`]. Because the pieces are
//! real subtrees of the BLAKE3 tree, folding their chaining values with BLAKE3's parent rule yields
//! the blob's ordinary BLAKE3 hash. So the piece table is **self-certifying against the existing
//! κ-label**: no second identity, no signed side manifest. A receiver that knows only the κ can
//!
//! 1. take a piece table from any untrusted source and check it against the κ
//!    ([`PieceTree::verify_root`], 2·n compressions for n pieces), then
//! 2. verify every 1 MiB piece independently, in any order, from any source
//!    ([`PieceTree::verify_piece`]), and so serve or accept any byte range after verifying only
//!    the pieces that cover it ([`PieceTree::pieces_covering`]).
//!
//! This is the Bao construction with a 1 MiB chunk group and only the bottom level of the outboard
//! kept: 32 bytes per MiB (128 KiB for a 4 GiB blob) instead of Bao's 6.25% outboard at 1 KiB
//! granularity. The interior nodes are recomputed from the leaves, which costs microseconds.
//!
//! [`StreamVerifier`] covers the case with no table at all: hash bytes as they arrive and confirm
//! the κ at the end, holding nothing but the hasher state. [`PieceTreeBuilder`] produces the piece
//! table from a stream in one pass, holding at most one piece.
//!
//! The tree arithmetic uses the 1.5 line's semi-public internals: `blake3::guts` (the chunk and
//! parent primitives the `bao` crate uses) and `blake3::platform::hash_many` (the multi-lane SIMD
//! routine `blake3::Hasher` itself runs). The workspace cannot move past blake3 1.5 while
//! `uor-prism-crypto` 0.4 caps it at `<1.6`; the 1.8 `hazmat` module is the supported API for the
//! same chaining values, so moving the pin later is a local rewrite of `subtree_cv` that changes no
//! byte of output (the tests pin every result to the one-shot hash).

use alloc::vec::Vec;
use core::ops::Range;

use alloc::vec;
use blake3::guts::{parent_cv, ChunkState, CHUNK_LEN};
use blake3::platform::Platform;
use blake3::IncrementCounter;
use hologram_types::prism::crypto::Blake3Hasher;
use hologram_types::prism::vocabulary::Hasher;

use crate::{KappaLabel, KappaLabel71, StoreError};

/// Bytes per piece: 1 MiB, the granularity of verified ranges and of parallel fetch.
pub const PIECE_LEN: u64 = 1 << 20;

/// Chunks per piece (2^10). A piece is an aligned, complete subtree of the BLAKE3 tree.
const CHUNKS_PER_PIECE: u64 = PIECE_LEN / CHUNK_LEN as u64;

/// Magic prefix of the encoded piece table ([`PieceTree::encode`]): format name and version 1.
const TREE_MAGIC: &[u8; 8] = b"HPTREE01";

/// Header width of the encoded piece table: magic, blob length (u64 LE), piece length (u64 LE).
const TREE_HEADER: usize = 8 + 8 + 8;

/// Why a verification failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    /// The κ-label is not on the BLAKE3 σ-axis (only BLAKE3 has a tree to verify against).
    NotBlake3,
    /// The bytes (or the piece table) do not hash to the expected value.
    Mismatch,
    /// The byte count differs from what the piece table or the caller declared.
    Length,
    /// A piece index or byte range lies outside the blob.
    OutOfRange,
    /// An encoded piece table is truncated or malformed.
    Malformed,
}

impl From<VerifyError> for StoreError {
    fn from(e: VerifyError) -> Self {
        match e {
            VerifyError::OutOfRange => StoreError::OutOfRange,
            VerifyError::NotBlake3 => StoreError::UnknownAxis,
            _ => StoreError::KappaMismatch,
        }
    }
}

/// The 32-byte BLAKE3 digest a `blake3:<64 hex>` κ-label names.
pub fn blake3_digest(kappa: &KappaLabel71) -> Result<[u8; 32], VerifyError> {
    let bytes = kappa.as_array();
    if &bytes[..7] != b"blake3:" {
        return Err(VerifyError::NotBlake3);
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        let hi = hex_val(bytes[7 + 2 * i]).ok_or(VerifyError::NotBlake3)?;
        let lo = hex_val(bytes[8 + 2 * i]).ok_or(VerifyError::NotBlake3)?;
        *o = (hi << 4) | lo;
    }
    Ok(out)
}

/// The `blake3:<64 hex>` κ-label of a 32-byte BLAKE3 digest.
pub fn blake3_kappa(digest: &[u8; 32]) -> KappaLabel71 {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 71];
    buf[..7].copy_from_slice(b"blake3:");
    for (i, &b) in digest.iter().enumerate() {
        buf[7 + 2 * i] = HEX[(b >> 4) as usize];
        buf[8 + 2 * i] = HEX[(b & 0x0f) as usize];
    }
    KappaLabel::from_bytes(&buf).expect("71-byte ASCII blake3 κ-label by construction")
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

/// Number of pieces a blob of `len` bytes has. An empty blob has one (empty) piece.
pub fn piece_count(len: u64) -> u64 {
    if len == 0 {
        1
    } else {
        len.div_ceil(PIECE_LEN)
    }
}

/// Largest power of two `<= n` (n ≥ 1).
fn pow2_floor(n: u64) -> u64 {
    1u64 << (63 - n.leading_zeros())
}

/// BLAKE3's IV (the SHA-256 IV), as the key words for the unkeyed hash mode.
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
/// BLAKE3 domain flags (spec §2.1).
const CHUNK_START: u8 = 1 << 0;
const CHUNK_END: u8 = 1 << 1;
const PARENT: u8 = 1 << 2;

/// Chaining value of `bytes` as the BLAKE3 subtree whose first chunk has index `chunk_offset`.
/// `is_root` finalizes it as the whole-input root instead (only for an input that is one subtree).
///
/// Full chunks and each level of parents go through `Platform::hash_many`, the same multi-lane
/// SIMD routine `blake3::Hasher` uses (AVX-512 hashes 16 chunks at once); a trailing partial chunk
/// and a root parent use the `guts` primitives. Levels are reduced pairwise with an odd node
/// carried up, which builds exactly BLAKE3's left-balanced tree. The tests check every result
/// against the one-shot hash, so a change in these semi-public internals fails loudly.
fn subtree_cv(bytes: &[u8], chunk_offset: u64, is_root: bool) -> [u8; 32] {
    if bytes.len() <= CHUNK_LEN {
        return *ChunkState::new(chunk_offset)
            .update(bytes)
            .finalize(is_root)
            .as_bytes();
    }
    let platform = Platform::detect();
    let full = bytes.len() / CHUNK_LEN;
    let tail = &bytes[full * CHUNK_LEN..];
    let n = full + usize::from(!tail.is_empty());
    let mut cvs = vec![0u8; n * 32];
    let chunks: Vec<&[u8; CHUNK_LEN]> = bytes[..full * CHUNK_LEN]
        .chunks_exact(CHUNK_LEN)
        .map(|c| c.try_into().expect("exact chunk"))
        .collect();
    platform.hash_many(
        &chunks,
        &IV,
        chunk_offset,
        IncrementCounter::Yes,
        0,
        CHUNK_START,
        CHUNK_END,
        &mut cvs[..full * 32],
    );
    if !tail.is_empty() {
        let cv = ChunkState::new(chunk_offset + full as u64)
            .update(tail)
            .finalize(false);
        cvs[full * 32..].copy_from_slice(cv.as_bytes());
    }
    // Reduce level by level until two nodes remain; the last parent may carry the root flag.
    let mut count = n;
    let mut next = vec![0u8; (n / 2 + 1) * 32];
    while count > 2 {
        let pairs = count / 2;
        let blocks: Vec<&[u8; 64]> = cvs[..pairs * 64]
            .chunks_exact(64)
            .map(|c| c.try_into().expect("exact block"))
            .collect();
        platform.hash_many(
            &blocks,
            &IV,
            0,
            IncrementCounter::No,
            PARENT,
            0,
            0,
            &mut next[..pairs * 32],
        );
        if count % 2 == 1 {
            next[pairs * 32..pairs * 32 + 32].copy_from_slice(&cvs[(count - 1) * 32..count * 32]);
        }
        count = pairs + count % 2;
        core::mem::swap(&mut cvs, &mut next);
    }
    let left: [u8; 32] = cvs[..32].try_into().expect("32 bytes");
    let right: [u8; 32] = cvs[32..64].try_into().expect("32 bytes");
    *parent_cv(&left.into(), &right.into(), is_root).as_bytes()
}

/// Fold piece chaining values into their parent with BLAKE3's split rule (pieces are aligned
/// power-of-two subtrees, so the rule over pieces is the rule over chunks). `cvs.len() ≥ 2` when
/// `is_root`.
fn merge(cvs: &[[u8; 32]], is_root: bool) -> [u8; 32] {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let left = pow2_floor(cvs.len() as u64 - 1) as usize;
    let l = merge(&cvs[..left], false);
    let r = merge(&cvs[left..], false);
    *parent_cv(&l.into(), &r.into(), is_root).as_bytes()
}

/// The chaining value of piece `index` with content `bytes`, for a blob of `piece_total` pieces:
/// the entry [`PieceTree`] holds for that piece. A one-piece blob's only entry is its root hash.
/// Lets a store compute a table's entries in parallel and assemble them with
/// [`PieceTree::from_parts`].
pub fn piece_chaining_value(index: u64, bytes: &[u8], piece_total: u64) -> [u8; 32] {
    subtree_cv(bytes, index * CHUNKS_PER_PIECE, piece_total == 1)
}

/// The bottom level of a blob's BLAKE3 tree at 1 MiB: its length and one chaining value per piece.
/// Self-certifying against the blob's `blake3:` κ-label (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PieceTree {
    len: u64,
    cvs: Vec<[u8; 32]>,
}

impl PieceTree {
    /// Build the table from the whole blob in memory.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut b = PieceTreeBuilder::new();
        b.update(bytes);
        b.finish()
    }

    /// Assemble a table from a declared length and per-piece chaining values (e.g. computed in
    /// parallel by a store). Checks only the shape; call [`verify_root`](Self::verify_root) before
    /// trusting it.
    pub fn from_parts(len: u64, cvs: Vec<[u8; 32]>) -> Result<Self, VerifyError> {
        if cvs.len() as u64 != piece_count(len) {
            return Err(VerifyError::Length);
        }
        Ok(Self { len, cvs })
    }

    /// Blob length in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// True for the empty blob.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of pieces (≥ 1).
    pub fn piece_count(&self) -> u64 {
        self.cvs.len() as u64
    }

    /// Per-piece chaining values, in order.
    pub fn piece_cvs(&self) -> &[[u8; 32]] {
        &self.cvs
    }

    /// Byte range `[start, end)` of piece `index`.
    pub fn piece_range(&self, index: u64) -> Result<Range<u64>, VerifyError> {
        if index >= self.piece_count() {
            return Err(VerifyError::OutOfRange);
        }
        let start = index * PIECE_LEN;
        Ok(start..(start + PIECE_LEN).min(self.len))
    }

    /// The pieces that cover bytes `[offset, offset + len)`. Empty for `len == 0`.
    pub fn pieces_covering(&self, offset: u64, len: u64) -> Result<Range<u64>, VerifyError> {
        let end = offset.checked_add(len).ok_or(VerifyError::OutOfRange)?;
        if end > self.len {
            return Err(VerifyError::OutOfRange);
        }
        if len == 0 {
            return Ok(0..0);
        }
        Ok(offset / PIECE_LEN..(end - 1) / PIECE_LEN + 1)
    }

    /// The blob's BLAKE3 hash, recomputed from the piece chaining values.
    pub fn root(&self) -> [u8; 32] {
        merge(&self.cvs, self.cvs.len() > 1)
    }

    /// The blob's κ-label (`blake3:<root>`).
    pub fn kappa(&self) -> KappaLabel71 {
        blake3_kappa(&self.root())
    }

    /// Check that this table belongs to `kappa`. A table from an untrusted source is usable only
    /// after this returns `Ok`.
    pub fn verify_root(&self, kappa: &KappaLabel71) -> Result<(), VerifyError> {
        if self.root() == blake3_digest(kappa)? {
            Ok(())
        } else {
            Err(VerifyError::Mismatch)
        }
    }

    /// Check that `bytes` is exactly piece `index` of the blob.
    pub fn verify_piece(&self, index: u64, bytes: &[u8]) -> Result<(), VerifyError> {
        let range = self.piece_range(index)?;
        if bytes.len() as u64 != range.end - range.start {
            return Err(VerifyError::Length);
        }
        if piece_chaining_value(index, bytes, self.piece_count()) == self.cvs[index as usize] {
            Ok(())
        } else {
            Err(VerifyError::Mismatch)
        }
    }

    /// Verify the pieces `first..` laid out contiguously in `bytes` (as fetched for a range), and
    /// return the sub-slice that is exactly `[offset, offset + len)`.
    pub fn verify_range<'a>(
        &self,
        offset: u64,
        len: u64,
        covering_bytes: &'a [u8],
    ) -> Result<&'a [u8], VerifyError> {
        let pieces = self.pieces_covering(offset, len)?;
        if pieces.is_empty() {
            return Ok(&[]);
        }
        let start = pieces.start * PIECE_LEN;
        let end = self.piece_range(pieces.end - 1)?.end;
        if covering_bytes.len() as u64 != end - start {
            return Err(VerifyError::Length);
        }
        for i in pieces {
            let r = self.piece_range(i)?;
            let s = (r.start - start) as usize;
            let e = (r.end - start) as usize;
            self.verify_piece(i, &covering_bytes[s..e])?;
        }
        let s = (offset - start) as usize;
        Ok(&covering_bytes[s..s + len as usize])
    }

    /// Wire form: `HPTREE01`, blob length (u64 LE), piece length (u64 LE), then 32 bytes per piece.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TREE_HEADER + 32 * self.cvs.len());
        out.extend_from_slice(TREE_MAGIC);
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&PIECE_LEN.to_le_bytes());
        for cv in &self.cvs {
            out.extend_from_slice(cv);
        }
        out
    }

    /// Parse the wire form. Checks shape only; call [`verify_root`](Self::verify_root) next.
    pub fn decode(bytes: &[u8]) -> Result<Self, VerifyError> {
        if bytes.len() < TREE_HEADER || &bytes[..8] != TREE_MAGIC {
            return Err(VerifyError::Malformed);
        }
        let len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
        let piece_len = u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes"));
        if piece_len != PIECE_LEN {
            return Err(VerifyError::Malformed);
        }
        let body = &bytes[TREE_HEADER..];
        if body.len() % 32 != 0 || (body.len() / 32) as u64 != piece_count(len) {
            return Err(VerifyError::Malformed);
        }
        let cvs = body
            .chunks_exact(32)
            .map(|c| c.try_into().expect("32 bytes"))
            .collect();
        Ok(Self { len, cvs })
    }
}

/// Builds a [`PieceTree`] from bytes arriving in order, holding at most one piece (1 MiB).
#[derive(Debug, Default)]
pub struct PieceTreeBuilder {
    len: u64,
    cvs: Vec<[u8; 32]>,
    buf: Vec<u8>,
}

impl PieceTreeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next bytes of the blob.
    pub fn update(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len() as u64;
        while !bytes.is_empty() {
            // A full buffered piece is finalized only once more bytes prove it is not the last:
            // a one-piece blob's piece is its root and finalizes differently.
            if self.buf.len() as u64 == PIECE_LEN {
                let index = self.cvs.len() as u64;
                self.cvs
                    .push(subtree_cv(&self.buf, index * CHUNKS_PER_PIECE, false));
                self.buf.clear();
            }
            let room = PIECE_LEN as usize - self.buf.len();
            let take = room.min(bytes.len());
            self.buf.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
        }
    }

    /// Bytes seen so far.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// True before any byte arrives.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finalize the table.
    pub fn finish(mut self) -> PieceTree {
        let index = self.cvs.len() as u64;
        let is_root = index == 0;
        self.cvs
            .push(subtree_cv(&self.buf, index * CHUNKS_PER_PIECE, is_root));
        PieceTree {
            len: self.len,
            cvs: self.cvs,
        }
    }
}

/// Confirms a stream against a κ-label without holding it: feed bytes in order, then
/// [`finish`](Self::finish). Uses the workspace σ-axis hasher (`prism::crypto::Blake3Hasher`), so
/// it accepts exactly what [`crate::verify_kappa`] accepts for the same bytes.
pub struct StreamVerifier {
    expected: [u8; 32],
    hasher: Blake3Hasher,
    len: u64,
    expected_len: Option<u64>,
}

impl StreamVerifier {
    /// Expect the stream to hash to `kappa` (a `blake3:` label).
    pub fn new(kappa: &KappaLabel71) -> Result<Self, VerifyError> {
        Ok(Self {
            expected: blake3_digest(kappa)?,
            hasher: Blake3Hasher::initial(),
            len: 0,
            expected_len: None,
        })
    }

    /// Also require exactly `len` bytes; [`update`](Self::update) fails as soon as the stream
    /// overruns it, so a source cannot make the receiver hash an endless body.
    pub fn with_len(mut self, len: u64) -> Self {
        self.expected_len = Some(len);
        self
    }

    /// Feed the next bytes.
    pub fn update(&mut self, bytes: &[u8]) -> Result<(), VerifyError> {
        self.len += bytes.len() as u64;
        if let Some(max) = self.expected_len {
            if self.len > max {
                return Err(VerifyError::Length);
            }
        }
        self.hasher = core::mem::take(&mut self.hasher).fold_bytes(bytes);
        Ok(())
    }

    /// Bytes seen so far.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// True before any byte arrives.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `Ok` iff the whole stream hashed to the expected κ (and had the expected length, if set).
    pub fn finish(self) -> Result<(), VerifyError> {
        if let Some(want) = self.expected_len {
            if self.len != want {
                return Err(VerifyError::Length);
            }
        }
        if self.hasher.finalize() == self.expected {
            Ok(())
        } else {
            Err(VerifyError::Mismatch)
        }
    }
}

/// Destination for verified pieces arriving in any order (a parallel fetch writes here). The
/// caller has already checked each piece against a [`PieceTree`] whose root matched the κ.
pub trait PieceSink: Send + Sync {
    /// Store piece `index`, which begins at byte `index * PIECE_LEN`.
    fn put_piece(&self, index: u64, bytes: &[u8]) -> Result<(), StoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address_bytes;

    fn blob(len: usize, seed: u8) -> Vec<u8> {
        let mut x = (seed as u32).wrapping_mul(0x9E37_79B9) | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect()
    }

    /// Lengths that straddle chunk, piece and power-of-two subtree boundaries.
    const LENS: &[usize] = &[
        0,
        1,
        1023,
        1024,
        1025,
        PIECE_LEN as usize - 1,
        PIECE_LEN as usize,
        PIECE_LEN as usize + 1,
        2 * PIECE_LEN as usize,
        3 * PIECE_LEN as usize + 17,
        4 * PIECE_LEN as usize,
        5 * PIECE_LEN as usize - 1,
    ];

    #[test]
    fn tree_root_is_the_blob_kappa() {
        for &n in LENS {
            let b = blob(n, 7);
            let t = PieceTree::from_bytes(&b);
            assert_eq!(t.kappa(), address_bytes(&b), "len {n}");
            assert_eq!(t.piece_count(), piece_count(n as u64));
            t.verify_root(&address_bytes(&b)).unwrap();
        }
    }

    #[test]
    fn every_piece_verifies_and_a_flipped_byte_fails_exactly_its_piece() {
        let n = 3 * PIECE_LEN as usize + 17;
        let b = blob(n, 3);
        let t = PieceTree::from_bytes(&b);
        for i in 0..t.piece_count() {
            let r = t.piece_range(i).unwrap();
            t.verify_piece(i, &b[r.start as usize..r.end as usize])
                .unwrap();
        }
        // Planted defect: one flipped bit in piece 2.
        let mut bad = b.clone();
        bad[2 * PIECE_LEN as usize + 5] ^= 1;
        for i in 0..t.piece_count() {
            let r = t.piece_range(i).unwrap();
            let got = t.verify_piece(i, &bad[r.start as usize..r.end as usize]);
            assert_eq!(got.is_err(), i == 2, "piece {i}");
        }
        // A correct piece presented at the wrong index fails.
        let r0 = t.piece_range(0).unwrap();
        assert_eq!(
            t.verify_piece(1, &b[r0.start as usize..r0.end as usize]),
            Err(VerifyError::Mismatch)
        );
        // A truncated piece fails on length.
        assert_eq!(t.verify_piece(0, &b[..10]), Err(VerifyError::Length));
    }

    #[test]
    fn single_piece_blob_verifies_as_root() {
        let b = blob(5000, 9);
        let t = PieceTree::from_bytes(&b);
        assert_eq!(t.piece_count(), 1);
        t.verify_piece(0, &b).unwrap();
        let mut bad = b.clone();
        bad[0] ^= 0x80;
        assert_eq!(t.verify_piece(0, &bad), Err(VerifyError::Mismatch));
    }

    #[test]
    fn a_forged_table_is_refused_by_the_kappa() {
        let b = blob(2 * PIECE_LEN as usize + 1, 5);
        let k = address_bytes(&b);
        let t = PieceTree::from_bytes(&b);
        // Planted defect: a source swaps one chaining value for the hash of its own bytes.
        let mut cvs = t.piece_cvs().to_vec();
        cvs[1][0] ^= 1;
        let forged = PieceTree::from_parts(t.len(), cvs).unwrap();
        assert_eq!(forged.verify_root(&k), Err(VerifyError::Mismatch));
        // A table for different content is refused too.
        let other = PieceTree::from_bytes(&blob(2 * PIECE_LEN as usize + 1, 6));
        assert_eq!(other.verify_root(&k), Err(VerifyError::Mismatch));
        // A table claiming a different length cannot be built with the wrong piece count.
        assert!(PieceTree::from_parts(t.len() + PIECE_LEN, t.piece_cvs().to_vec()).is_err());
    }

    #[test]
    fn encode_decode_roundtrip_and_malformed_input() {
        let b = blob(3 * PIECE_LEN as usize, 1);
        let t = PieceTree::from_bytes(&b);
        let wire = t.encode();
        assert_eq!(wire.len(), 24 + 3 * 32);
        assert_eq!(PieceTree::decode(&wire).unwrap(), t);
        assert_eq!(
            PieceTree::decode(&wire[..wire.len() - 1]),
            Err(VerifyError::Malformed)
        );
        let mut bad_magic = wire.clone();
        bad_magic[0] = b'X';
        assert_eq!(PieceTree::decode(&bad_magic), Err(VerifyError::Malformed));
    }

    #[test]
    fn verified_range_returns_exactly_the_requested_bytes() {
        let b = blob(3 * PIECE_LEN as usize + 100, 2);
        let t = PieceTree::from_bytes(&b);
        for &(off, len) in &[
            (0u64, 1u64),
            (PIECE_LEN - 1, 2),
            (PIECE_LEN, PIECE_LEN),
            (5, 3 * PIECE_LEN + 95),
            (3 * PIECE_LEN + 99, 1),
            (7, 0),
        ] {
            let p = t.pieces_covering(off, len).unwrap();
            let (s, e) = if p.is_empty() {
                (0, 0)
            } else {
                (
                    t.piece_range(p.start).unwrap().start,
                    t.piece_range(p.end - 1).unwrap().end,
                )
            };
            let got = t
                .verify_range(off, len, &b[s as usize..e as usize])
                .unwrap();
            assert_eq!(got, &b[off as usize..(off + len) as usize]);
        }
        assert_eq!(
            t.pieces_covering(3 * PIECE_LEN + 100, 1),
            Err(VerifyError::OutOfRange)
        );
        // Planted defect inside a covering piece but outside the requested bytes still fails:
        // a range is only as trustworthy as the whole pieces under it.
        let mut bad = b[..2 * PIECE_LEN as usize].to_vec();
        bad[PIECE_LEN as usize + 9] ^= 1;
        assert_eq!(
            t.verify_range(PIECE_LEN, 4, &bad[PIECE_LEN as usize..]),
            Err(VerifyError::Mismatch)
        );
    }

    #[test]
    fn stream_verifier_accepts_the_blob_and_rejects_a_flip_or_overrun() {
        let b = blob(2 * PIECE_LEN as usize + 3, 4);
        let k = address_bytes(&b);
        let mut v = StreamVerifier::new(&k).unwrap().with_len(b.len() as u64);
        for c in b.chunks(4096 + 7) {
            v.update(c).unwrap();
        }
        v.finish().unwrap();

        let mut bad = b.clone();
        bad[123_456] ^= 4;
        let mut v = StreamVerifier::new(&k).unwrap();
        v.update(&bad).unwrap();
        assert_eq!(v.finish(), Err(VerifyError::Mismatch));

        let mut v = StreamVerifier::new(&k).unwrap().with_len(b.len() as u64);
        v.update(&b).unwrap();
        assert_eq!(v.update(b"x"), Err(VerifyError::Length));

        let mut v = StreamVerifier::new(&k).unwrap().with_len(b.len() as u64);
        v.update(&b[..10]).unwrap();
        assert_eq!(v.finish(), Err(VerifyError::Length));
    }

    #[test]
    fn builder_matches_one_shot_for_any_split() {
        let b = blob(2 * PIECE_LEN as usize + 77, 8);
        let one = PieceTree::from_bytes(&b);
        for step in [
            1usize,
            1000,
            65_536,
            PIECE_LEN as usize,
            3 * PIECE_LEN as usize,
        ] {
            let mut builder = PieceTreeBuilder::new();
            for c in b.chunks(step.max(1)) {
                builder.update(c);
            }
            assert_eq!(builder.finish(), one, "step {step}");
        }
    }

    #[test]
    fn non_blake3_labels_are_refused() {
        let sha = crate::address_bytes_axis("sha256", b"x").unwrap();
        let k = KappaLabel71::from_bytes(&<[u8; 71]>::try_from(sha.as_slice()).unwrap()).unwrap();
        assert_eq!(StreamVerifier::new(&k).err(), Some(VerifyError::NotBlake3));
        assert_eq!(blake3_digest(&k), Err(VerifyError::NotBlake3));
    }
}
