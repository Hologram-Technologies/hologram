//! Hardware-abstraction seam (spec 02 §4): `BlockDevice` + `NetworkInterface`.
//!
//! Absorbed from the former `hologram-bare-hal` crate (P1). Above these traits the
//! device-agnostic store/net backends run; below them per-device drivers (NVMe, AHCI,
//! e1000, virtio-net) implement the hardware specifics — adding a device class is
//! implementing one of these traits. `no_std + alloc`. The full UEFI boot path + real
//! drivers live in the bare space's target-only binary; this is the seam, exercised here
//! by an in-memory [`RamBlockDevice`].

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::task::Waker;
use spin::Mutex;
// `async_trait` emits an unqualified `Box`; not in the `no_std` prelude (std provides it).
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// A block device — NVMe namespace, AHCI port, virtio-blk, etc. All I/O is async; the
/// driver wakes the caller's task on completion (bare-metal §3.2.1).
#[async_trait::async_trait]
pub trait BlockDevice: Send + Sync {
    fn sector_size(&self) -> u32;
    fn sector_count(&self) -> u64;
    /// Read `sectors` sectors starting at `lba` into `buffer` (`buffer.len() == sectors * sector_size`).
    async fn read(&self, lba: u64, sectors: u32, buffer: &mut [u8]) -> Result<(), DeviceError>;
    /// Write `sectors` sectors starting at `lba` from `buffer`.
    async fn write(&self, lba: u64, sectors: u32, buffer: &[u8]) -> Result<(), DeviceError>;
    /// Block until all pending writes are durable.
    async fn flush(&self) -> Result<(), DeviceError>;
    /// Device-unique identifier (NVMe NGUID, AHCI WWN, …) — stable across reboots.
    fn device_uuid(&self) -> [u8; 16];
}

/// A network interface — e1000, virtio-net, igb, etc. Transmit/receive are
/// buffer-oriented; smoltcp drives polling (bare-metal §3.2.1).
pub trait NetworkInterface: Send + Sync {
    fn mac_address(&self) -> [u8; 6];
    fn mtu(&self) -> u32;
    /// Transmit one frame; `Err(Backpressure)` if the TX queue is full.
    fn transmit(&self, frame: &[u8]) -> Result<usize, NicError>;
    /// Receive one frame; `Ok(0)` if none available.
    fn receive(&self, buffer: &mut [u8]) -> Result<usize, NicError>;
    /// Register a waker fired when frames arrive.
    fn register_rx_waker(&self, waker: Waker);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    OutOfRange,
    HardwareFault(u32),
    Aborted,
    Backpressure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NicError {
    Backpressure,
    HardwareFault(u32),
    LinkDown,
}

/// An in-memory [`BlockDevice`] — a RAM disk. Proves the seam is implementable and is the
/// fixture a bare `KappaStore` runs against without real hardware. `Clone` shares one
/// backing store (via `Arc`), so two stores opened on a clone observe the same "disk" —
/// the reboot/persistence test.
#[derive(Clone)]
pub struct RamBlockDevice {
    sector_size: u32,
    data: Arc<Mutex<Vec<u8>>>,
    uuid: [u8; 16],
}

impl RamBlockDevice {
    pub fn new(sector_size: u32, sector_count: u64) -> Self {
        Self {
            sector_size,
            data: Arc::new(Mutex::new(vec![
                0u8;
                (sector_size as u64 * sector_count) as usize
            ])),
            uuid: [0xA5; 16],
        }
    }

    fn span(&self, lba: u64, sectors: u32) -> Result<(usize, usize), DeviceError> {
        let start = (lba * self.sector_size as u64) as usize;
        let len = sectors as usize * self.sector_size as usize;
        let end = start.checked_add(len).ok_or(DeviceError::OutOfRange)?;
        if end > self.data.lock().len() {
            return Err(DeviceError::OutOfRange);
        }
        Ok((start, end))
    }
}

#[async_trait::async_trait]
impl BlockDevice for RamBlockDevice {
    fn sector_size(&self) -> u32 {
        self.sector_size
    }
    fn sector_count(&self) -> u64 {
        self.data.lock().len() as u64 / self.sector_size as u64
    }
    async fn read(&self, lba: u64, sectors: u32, buffer: &mut [u8]) -> Result<(), DeviceError> {
        if buffer.len() != sectors as usize * self.sector_size as usize {
            return Err(DeviceError::OutOfRange);
        }
        let (s, e) = self.span(lba, sectors)?;
        buffer.copy_from_slice(&self.data.lock()[s..e]);
        Ok(())
    }
    async fn write(&self, lba: u64, sectors: u32, buffer: &[u8]) -> Result<(), DeviceError> {
        if buffer.len() != sectors as usize * self.sector_size as usize {
            return Err(DeviceError::OutOfRange);
        }
        let (s, e) = self.span(lba, sectors)?;
        self.data.lock()[s..e].copy_from_slice(buffer);
        Ok(())
    }
    async fn flush(&self) -> Result<(), DeviceError> {
        Ok(())
    }
    fn device_uuid(&self) -> [u8; 16] {
        self.uuid
    }
}

/// A source of randomness — the platform's CSPRNG seam (spec 02 §4). A native space fills from
/// the OS (`getrandom`); the browser space from `crypto.getRandomValues`; a bare-metal space
/// from a hardware RNG. Above it, key generation and nonces draw bytes without knowing the source.
pub trait Entropy: Send + Sync {
    /// Fill `buf` with random bytes.
    fn fill(&self, buf: &mut [u8]);
}

/// A monotonic millisecond clock — the platform's time seam (spec 02 §4). A native space reads
/// the OS clock; the browser space `performance.now()`; a bare-metal space the TSC/PIT. Above it,
/// timeouts and fuel budgets measure elapsed time without a platform clock.
pub trait Clock: Send + Sync {
    /// Milliseconds since an arbitrary but fixed epoch — non-decreasing (monotonic).
    fn now_millis(&self) -> u64;
}

/// A deterministic reference [`Entropy`] for hermetic V&V: a seeded SplitMix64 PRNG — **not** a
/// CSPRNG, but reproducible so tests are deterministic. A real space wires a secure source.
pub struct SeededEntropy {
    state: Mutex<u64>,
}

impl SeededEntropy {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            state: Mutex::new(seed),
        }
    }
}

impl Default for SeededEntropy {
    fn default() -> Self {
        Self::new(0x9E37_79B9_7F4A_7C15)
    }
}

impl Entropy for SeededEntropy {
    fn fill(&self, buf: &mut [u8]) {
        let mut s = self.state.lock();
        for chunk in buf.chunks_mut(8) {
            // SplitMix64
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let bytes = z.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

/// A manually-advanced reference [`Clock`] for hermetic V&V: `now_millis()` returns the current
/// value; [`advance`](ManualClock::advance) moves it forward — deterministic time for tests.
pub struct ManualClock {
    millis: Mutex<u64>,
}

impl ManualClock {
    /// A clock starting at `millis`.
    pub fn new(millis: u64) -> Self {
        Self {
            millis: Mutex::new(millis),
        }
    }
    /// Advance the clock by `delta` milliseconds.
    pub fn advance(&self, delta: u64) {
        let mut m = self.millis.lock();
        *m = m.wrapping_add(delta);
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Clock for ManualClock {
    fn now_millis(&self) -> u64 {
        *self.millis.lock()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn ram_block_device_roundtrips_and_bounds_check() {
        let dev = RamBlockDevice::new(512, 8);
        assert_eq!(dev.sector_count(), 8);
        pollster::block_on(async {
            let payload = std::vec![0xCD; 1024]; // 2 sectors
            dev.write(2, 2, &payload).await.unwrap();
            let mut back = std::vec![0u8; 1024];
            dev.read(2, 2, &mut back).await.unwrap();
            assert_eq!(back, payload);
            // Out-of-range read is refused (no silent truncation).
            assert_eq!(
                dev.read(7, 4, &mut std::vec![0u8; 2048]).await,
                Err(DeviceError::OutOfRange)
            );
        });
    }
}
