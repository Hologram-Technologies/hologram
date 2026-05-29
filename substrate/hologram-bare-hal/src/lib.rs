#![no_std]
//! # hologram-bare-hal
//!
//! The bare-metal substrate's hardware-abstraction seam (bare-metal spec §3.2.1). Above these
//! traits, `BareMetalKappaStore`/`BareMetalKappaSync` are device-agnostic; below them, per-device
//! drivers (NVMe, AHCI, e1000, virtio-net) implement the hardware specifics. Adding a device class
//! is implementing one of these traits. `no_std + alloc`.
//!
//! The full UEFI boot path + real drivers live in the bare-metal binary (target-only); this crate
//! is the seam, exercised here by an in-memory [`RamBlockDevice`].

extern crate alloc;

use alloc::boxed::Box; // async-trait emits unqualified `Box` (no_std).
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::task::Waker;
use spin::Mutex;

/// A block device — NVMe namespace, AHCI port, virtio-blk, etc. All I/O is async; the driver wakes
/// the caller's task on completion (bare-metal §3.2.1).
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

/// A network interface — e1000, virtio-net, igb, etc. Transmit/receive are buffer-oriented; smoltcp
/// drives polling (bare-metal §3.2.1).
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

/// An in-memory [`BlockDevice`] — a RAM disk. Proves the seam is implementable and is the test
/// fixture a `BareMetalKappaStore` runs against without real hardware. `Clone` shares one backing
/// store (via `Arc`), so two `BareMetalKappaStore`s opened on a clone observe the same "disk" — the
/// reboot/persistence test.
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
