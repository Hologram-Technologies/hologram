//! [`WasmNetworkInterface`] — a [`NetworkInterface`] whose packet I/O is executed by an
//! **imported, verified Wasm driver** (arch §11.9). The driver module exports `mac_address`,
//! `mtu`, `transmit`, `receive`, and holds RX queue state in its own linear memory; the host moves
//! frame bytes through a fixed scratch region (the same pattern as [`WasmBlockDevice`]).
//!
//! This closes the HAL driver-import path on the network side: V&V class **NI** asserts a
//! codemodule-κ → live driver-backed device round-trip, symmetric to the **DU** class for block
//! devices (`runtime/tests/driver_import.rs` + `runtime-wasmtime/tests/driver_backed_device.rs`).

use core::task::Waker;
use hologram_bare_hal::{NetworkInterface, NicError};
use hologram_substrate_core::RuntimeError;
use spin::Mutex;
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

/// Host scratch pointer in the driver's linear memory for TX/RX transfers. Symmetric with
/// `WasmBlockDevice::IO_PTR`.
const IO_PTR: i32 = 0x2000;

struct Driver {
    store: Store<()>,
    memory: Memory,
    /// `transmit(ptr, len) -> i32` — bytes already staged at `[ptr..ptr+len]`; returns bytes written.
    transmit: TypedFunc<(i32, i32), i32>,
    /// `receive(ptr, cap) -> i32` — driver writes a frame to `[ptr..ptr+cap]`; returns frame length
    /// (0 = nothing).
    receive: TypedFunc<(i32, i32), i32>,
}

/// A network interface backed by a Wasm driver module.
pub struct WasmNetworkInterface {
    inner: Mutex<Driver>,
    mac: [u8; 6],
    mtu: u32,
}

fn ifail(_e: impl core::fmt::Debug) -> RuntimeError {
    RuntimeError::InstantiationFailed("wasm net driver")
}

impl WasmNetworkInterface {
    /// Instantiate a (verified) driver module's bytes and bind it as a network interface.
    pub fn from_code(code: &[u8]) -> Result<Self, RuntimeError> {
        let engine = Engine::default();
        let module = Module::new(&engine, code).map_err(ifail)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).map_err(ifail)?;
        let memory =
            instance
                .get_memory(&mut store, "memory")
                .ok_or(RuntimeError::InstantiationFailed(
                    "net driver exports no memory",
                ))?;
        // `mac_address(out_ptr)` writes the 6-byte MAC at `out_ptr`.
        let mac_fn: TypedFunc<i32, ()> = instance
            .get_typed_func(&mut store, "mac_address")
            .map_err(ifail)?;
        mac_fn.call(&mut store, IO_PTR).map_err(ifail)?;
        let mut mac = [0u8; 6];
        let data = memory.data(&store);
        mac.copy_from_slice(
            data.get(IO_PTR as usize..IO_PTR as usize + 6)
                .ok_or(RuntimeError::InstantiationFailed("mac scratch oob"))?,
        );
        // `mtu() -> i32`.
        let mtu_fn: TypedFunc<(), i32> =
            instance.get_typed_func(&mut store, "mtu").map_err(ifail)?;
        let mtu = mtu_fn.call(&mut store, ()).map_err(ifail)? as u32;

        let transmit = instance
            .get_typed_func(&mut store, "transmit")
            .map_err(ifail)?;
        let receive = instance
            .get_typed_func(&mut store, "receive")
            .map_err(ifail)?;

        Ok(Self {
            inner: Mutex::new(Driver {
                store,
                memory,
                transmit,
                receive,
            }),
            mac,
            mtu,
        })
    }
}

impl NetworkInterface for WasmNetworkInterface {
    fn mac_address(&self) -> [u8; 6] {
        self.mac
    }
    fn mtu(&self) -> u32 {
        self.mtu
    }
    fn transmit(&self, frame: &[u8]) -> Result<usize, NicError> {
        let mut guard = self.inner.lock();
        let d = &mut *guard;
        let off = IO_PTR as usize;
        // Stage the frame in the driver's scratch region.
        let dst = d.memory.data_mut(&mut d.store);
        let slot = dst
            .get_mut(off..off + frame.len())
            .ok_or(NicError::HardwareFault(1))?;
        slot.copy_from_slice(frame);
        // Driver consumes [scratch..scratch+len], returns bytes written.
        let written = d
            .transmit
            .call(&mut d.store, (IO_PTR, frame.len() as i32))
            .map_err(|_| NicError::HardwareFault(2))?;
        if written < 0 {
            return Err(NicError::Backpressure);
        }
        Ok(written as usize)
    }
    fn receive(&self, buffer: &mut [u8]) -> Result<usize, NicError> {
        let mut guard = self.inner.lock();
        let d = &mut *guard;
        let off = IO_PTR as usize;
        // Driver writes a frame at scratch and returns its length (0 = no frame ready).
        let n = d
            .receive
            .call(&mut d.store, (IO_PTR, buffer.len() as i32))
            .map_err(|_| NicError::HardwareFault(3))?;
        if n < 0 {
            return Err(NicError::HardwareFault(4));
        }
        let n = n as usize;
        if n == 0 {
            return Ok(0);
        }
        let src = d.memory.data(&d.store);
        let frame = src.get(off..off + n).ok_or(NicError::HardwareFault(5))?;
        buffer[..n].copy_from_slice(frame);
        Ok(n)
    }
    fn register_rx_waker(&self, _waker: Waker) {
        // The driver-import test fixture is poll-based; production NICs use IRQ-driven wakers
        // dispatched via the driver's own `register_rx_waker` (a follow-on host import).
    }
}
