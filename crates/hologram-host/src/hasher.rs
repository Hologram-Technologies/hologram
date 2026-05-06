//! `HologramHasher` — BLAKE3-backed `Hasher<32>` impl (spec III.3).

use uor_foundation::enforcement::Hasher;

/// BLAKE3 at the canonical 32-byte width.
///
/// `OUTPUT_BYTES = 32` always. ADR-001 / ADR-052 anchor BLAKE3 as the
/// canonical hash for hologram artifacts.
#[derive(Clone)]
pub struct HologramHasher {
    inner: blake3::Hasher,
}

impl Hasher<32> for HologramHasher {
    const OUTPUT_BYTES: usize = 32;

    #[inline]
    fn initial() -> Self {
        Self { inner: blake3::Hasher::new() }
    }

    #[inline]
    fn fold_byte(mut self, b: u8) -> Self {
        self.inner.update(&[b]);
        self
    }

    #[inline]
    fn fold_bytes(mut self, bytes: &[u8]) -> Self {
        self.inner.update(bytes);
        self
    }

    #[inline]
    fn finalize(self) -> [u8; 32] {
        self.inner.finalize().into()
    }
}
