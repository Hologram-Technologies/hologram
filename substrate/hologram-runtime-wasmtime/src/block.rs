//! [`WasmBlockDevice`] — a [`BlockDevice`] whose sector I/O is executed by an **imported, verified
//! Wasm driver**. The driver module exports `sector_size`/`sector_count`/`read`/`write`/`flush` and
//! holds the "disk" in its own linear memory; the host moves bytes through a fixed scratch region.
//! This is how a running engine *uses* an imported driver: the device the store runs on is the
//! driver's code, not substrate-authored.

use hologram_bare_hal::{BlockDevice, DeviceError};
use hologram_substrate_core::RuntimeError;
use spin::Mutex;
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

/// Host scratch pointer in the driver's linear memory for read/write transfers.
const IO_PTR: i32 = 0x1000;

struct Driver {
    store: Store<()>,
    memory: Memory,
    read: TypedFunc<(i64, i32, i32), i32>,
    write: TypedFunc<(i64, i32, i32), i32>,
    flush: TypedFunc<(), i32>,
}

/// A block device backed by a Wasm driver module.
pub struct WasmBlockDevice {
    inner: Mutex<Driver>,
    sector_size: u32,
    sector_count: u64,
    uuid: [u8; 16],
}

fn ifail(_e: impl core::fmt::Debug) -> RuntimeError {
    RuntimeError::InstantiationFailed("wasm driver")
}

impl WasmBlockDevice {
    /// Instantiate a (verified) driver module's bytes and bind it as a block device.
    pub fn from_code(code: &[u8]) -> Result<Self, RuntimeError> {
        let engine = Engine::default();
        let module = Module::new(&engine, code).map_err(ifail)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).map_err(ifail)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or(RuntimeError::InstantiationFailed("driver exports no memory"))?;
        let ss: TypedFunc<(), i32> = instance.get_typed_func(&mut store, "sector_size").map_err(ifail)?;
        let sc: TypedFunc<(), i64> = instance.get_typed_func(&mut store, "sector_count").map_err(ifail)?;
        let read = instance.get_typed_func(&mut store, "read").map_err(ifail)?;
        let write = instance.get_typed_func(&mut store, "write").map_err(ifail)?;
        let flush = instance.get_typed_func(&mut store, "flush").map_err(ifail)?;
        let sector_size = ss.call(&mut store, ()).map_err(ifail)? as u32;
        let sector_count = sc.call(&mut store, ()).map_err(ifail)? as u64;
        Ok(Self {
            inner: Mutex::new(Driver { store, memory, read, write, flush }),
            sector_size,
            sector_count,
            uuid: [0x77; 16],
        })
    }
}

#[async_trait::async_trait]
impl BlockDevice for WasmBlockDevice {
    fn sector_size(&self) -> u32 {
        self.sector_size
    }
    fn sector_count(&self) -> u64 {
        self.sector_count
    }
    async fn read(&self, lba: u64, sectors: u32, buffer: &mut [u8]) -> Result<(), DeviceError> {
        let mut guard = self.inner.lock();
        let d = &mut *guard; // reborrow so disjoint fields (read/store/memory) can be split-borrowed
        // The driver copies disk[lba..] → its scratch region; the host then reads it out.
        d.read
            .call(&mut d.store, (lba as i64, sectors as i32, IO_PTR))
            .map_err(|_| DeviceError::HardwareFault(1))?;
        let off = IO_PTR as usize;
        let data = d.memory.data(&d.store);
        buffer.copy_from_slice(data.get(off..off + buffer.len()).ok_or(DeviceError::OutOfRange)?);
        Ok(())
    }
    async fn write(&self, lba: u64, sectors: u32, buffer: &[u8]) -> Result<(), DeviceError> {
        let mut guard = self.inner.lock();
        let d = &mut *guard;
        // The host stages bytes in the driver's scratch region; the driver copies them → disk[lba..].
        let off = IO_PTR as usize;
        let dst = d.memory.data_mut(&mut d.store);
        dst.get_mut(off..off + buffer.len()).ok_or(DeviceError::OutOfRange)?.copy_from_slice(buffer);
        d.write
            .call(&mut d.store, (lba as i64, sectors as i32, IO_PTR))
            .map_err(|_| DeviceError::HardwareFault(2))?;
        Ok(())
    }
    async fn flush(&self) -> Result<(), DeviceError> {
        let mut guard = self.inner.lock();
        let d = &mut *guard;
        d.flush.call(&mut d.store, ()).map_err(|_| DeviceError::HardwareFault(3))?;
        Ok(())
    }
    fn device_uuid(&self) -> [u8; 16] {
        self.uuid
    }
}
