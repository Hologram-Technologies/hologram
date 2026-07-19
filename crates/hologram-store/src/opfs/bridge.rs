//! Browser sync/async bridge for `KappaStore` over OPFS (architecture §9 G-C2 → B3).
//!
//! `KappaStore` is **sync** (architecture §3.2 — bounded local work; matches OPFS's
//! `FileSystemSyncAccessHandle`, which is only available inside a Worker). The browser surface
//! that wants a `KappaStore` is the **main thread**, but OPFS access from the main thread is
//! async-only (`FileSystemDirectoryHandle.getFileHandle()` returns a Promise). The architectural
//! gap, G-C2, is the bridge between these two regimes.
//!
//! The pattern implemented here: a **SharedArrayBuffer** of fixed shape, written by the main
//! thread (the request), read by a paired Worker, written back (the response), with
//! `Atomics.wait` / `Atomics.notify` synchronizing the two sides. The main thread spins on
//! `Atomics.wait` (which suspends the main thread; this is the legal way to do *sync* I/O from
//! main on the modern web platform). The Worker drains requests and runs the existing async
//! `opfs_put` / `opfs_get` / `opfs_iterate` / `opfs_delete` against `FileSystemSyncAccessHandle`s
//! it holds.
//!
//! ## SAB wire format
//!
//! All fields are little-endian. Offsets in bytes. Capacity = `SAB_CAPACITY` (default 8 MiB).
//!
//! ```text
//! +0   STATE   i32 — 0 = idle, 1 = request, 2 = response, 3 = error
//! +4   OP      u32 — opcode (see `Op`)
//! +8   LEN_A   u32 — length of payload A (axis / κ / pins-list)
//! +12  LEN_B   u32 — length of payload B (bytes)
//! +16  RESULT  i64 — packed result (e.g. κ-as-71-bytes via LEN_A, or u32 status code)
//! +24  PAYLOAD ...  — request OR response payload bytes
//! ```
//!
//! The main thread writes (OP, LEN_A, LEN_B, PAYLOAD), sets STATE=1, calls `Atomics.notify`.
//! The Worker (waiting on STATE) wakes, executes the op, writes (LEN_A, LEN_B, RESULT, PAYLOAD),
//! sets STATE=2, calls `Atomics.notify`. The main thread (waiting again) reads the response.
//!
//! ## Why this is uor-native
//!
//! The bridge is a *transport*, not a *naming* surface. κ-labels still flow as canonical bytes
//! over the SAB — verify-on-receipt happens in the Worker exactly as the existing OPFS path
//! does. SPINE-1..6 are all preserved across the boundary.

use alloc::string::String;
use alloc::vec::Vec;
use hologram_space::{address_bytes, verify_kappa, KappaLabel};
use js_sys::{Atomics, Int32Array, SharedArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// SAB total capacity (8 MiB). Operators can size up via [`SyncOpfsBridge::new_with_capacity`].
pub const SAB_CAPACITY: usize = 8 * 1024 * 1024;

/// SAB header offsets.
const OFF_STATE: u32 = 0;
const OFF_OP: u32 = 4;
const OFF_LEN_A: u32 = 8;
const OFF_LEN_B: u32 = 12;
const OFF_RESULT: u32 = 16;
const OFF_PAYLOAD: u32 = 24;

/// SAB state values (written to OFF_STATE).
const STATE_IDLE: i32 = 0;
const STATE_REQUEST: i32 = 1;
const STATE_RESPONSE: i32 = 2;
const STATE_ERROR: i32 = 3;

/// Opcodes written to OFF_OP. Append-only (SPINE-5).
#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Op {
    Put = 1,
    Get = 2,
    Contains = 3,
    Delete = 4,
    Iterate = 5,
}

impl Op {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Put),
            2 => Some(Self::Get),
            3 => Some(Self::Contains),
            4 => Some(Self::Delete),
            5 => Some(Self::Iterate),
            _ => None,
        }
    }
}

/// Main-thread sync façade for the OPFS bridge. Wraps a SharedArrayBuffer that a Worker reads
/// + writes through the [`Op`] protocol.
#[wasm_bindgen]
pub struct SyncOpfsBridge {
    sab: SharedArrayBuffer,
    i32: Int32Array,
    u8: Uint8Array,
}

#[wasm_bindgen]
impl SyncOpfsBridge {
    /// New bridge backed by a SharedArrayBuffer of `SAB_CAPACITY` bytes.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::new_with_capacity(SAB_CAPACITY as u32)
    }

    /// New bridge with explicit capacity.
    pub fn new_with_capacity(capacity: u32) -> Self {
        let sab = SharedArrayBuffer::new(capacity);
        let i32 = Int32Array::new(&sab);
        let u8 = Uint8Array::new(&sab);
        // Initialize STATE = IDLE.
        Atomics::store(&i32, OFF_STATE / 4, STATE_IDLE).ok();
        Self { sab, i32, u8 }
    }

    /// Hand the SAB to the Worker (postMessage) — the Worker constructs its own
    /// [`BridgeWorker`] over this buffer.
    pub fn sab(&self) -> SharedArrayBuffer {
        self.sab.clone()
    }

    /// Main-thread `put(axis, bytes)` — synchronous. Blocks via `Atomics.wait` until the paired
    /// Worker has stored the bytes and written the κ-label back to the SAB. Returns the κ as a
    /// 71-byte string.
    pub fn put(&self, axis: &str, bytes: &[u8]) -> Result<String, JsValue> {
        let axis_bytes = axis.as_bytes();
        let payload_len = axis_bytes.len() + bytes.len();
        if OFF_PAYLOAD as usize + payload_len > self.sab.byte_length() as usize {
            return Err(JsValue::from_str("payload exceeds SAB capacity"));
        }
        // Write OP and lengths.
        Atomics::store(&self.i32, OFF_OP / 4, Op::Put as i32)?;
        Atomics::store(&self.i32, OFF_LEN_A / 4, axis_bytes.len() as i32)?;
        Atomics::store(&self.i32, OFF_LEN_B / 4, bytes.len() as i32)?;
        // Write payload: axis then bytes.
        let axis_u8 = Uint8Array::from(axis_bytes);
        self.u8.set(&axis_u8, OFF_PAYLOAD);
        let bytes_u8 = Uint8Array::from(bytes);
        self.u8
            .set(&bytes_u8, OFF_PAYLOAD + axis_bytes.len() as u32);
        // Transition: idle → request, then wait for response.
        self.send_and_wait()?;
        // Response payload: 71-byte κ-label string.
        let kappa_len = Atomics::load(&self.i32, OFF_LEN_A / 4)? as usize;
        if kappa_len != 71 {
            return Err(JsValue::from_str("bridge: malformed κ response"));
        }
        let mut out = [0u8; 71];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.u8.get_index(OFF_PAYLOAD + i as u32);
        }
        String::from_utf8(out.to_vec()).map_err(|_| JsValue::from_str("κ utf8"))
    }

    /// Main-thread `get(kappa)` — synchronous. Returns the bytes, or `null` if absent. Verifies
    /// on receipt (SPINE-4) — the responder may not lie about content.
    pub fn get(&self, kappa: &str) -> Result<JsValue, JsValue> {
        let k = kappa.as_bytes();
        if k.len() != 71 {
            return Err(JsValue::from_str("κ must be 71 bytes"));
        }
        Atomics::store(&self.i32, OFF_OP / 4, Op::Get as i32)?;
        Atomics::store(&self.i32, OFF_LEN_A / 4, 71)?;
        Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
        let k_u8 = Uint8Array::from(k);
        self.u8.set(&k_u8, OFF_PAYLOAD);
        self.send_and_wait()?;
        // Response: LEN_B = bytes length (0 ⇒ absent); RESULT lo32 = ok flag.
        let result = Atomics::load(&self.i32, OFF_RESULT / 4)? as u32;
        if result == 0 {
            return Ok(JsValue::NULL);
        }
        let n = Atomics::load(&self.i32, OFF_LEN_B / 4)? as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.u8.get_index(OFF_PAYLOAD + i as u32));
        }
        // Verify on receipt — the Worker may not lie.
        let kappa_arr: [u8; 71] = kappa.as_bytes().try_into().unwrap();
        let label = KappaLabel::<71>::from_bytes(&kappa_arr)
            .map_err(|_| JsValue::from_str("bridge: malformed κ"))?;
        if !verify_kappa(&out, &label).unwrap_or(false) {
            return Err(JsValue::from_str("bridge: response failed σ-axis verify"));
        }
        Ok(Uint8Array::from(out.as_slice()).into())
    }

    /// Main-thread `contains(kappa)` — synchronous; `true` iff OPFS holds bytes at that κ.
    pub fn contains(&self, kappa: &str) -> Result<bool, JsValue> {
        let k = kappa.as_bytes();
        if k.len() != 71 {
            return Err(JsValue::from_str("κ must be 71 bytes"));
        }
        Atomics::store(&self.i32, OFF_OP / 4, Op::Contains as i32)?;
        Atomics::store(&self.i32, OFF_LEN_A / 4, 71)?;
        Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
        let k_u8 = Uint8Array::from(k);
        self.u8.set(&k_u8, OFF_PAYLOAD);
        self.send_and_wait()?;
        let present = Atomics::load(&self.i32, OFF_RESULT / 4)? != 0;
        Ok(present)
    }

    /// Main-thread `delete(kappa)` — synchronous; `true` iff the file was removed (false ⇒ absent).
    pub fn delete(&self, kappa: &str) -> Result<bool, JsValue> {
        let k = kappa.as_bytes();
        if k.len() != 71 {
            return Err(JsValue::from_str("κ must be 71 bytes"));
        }
        Atomics::store(&self.i32, OFF_OP / 4, Op::Delete as i32)?;
        Atomics::store(&self.i32, OFF_LEN_A / 4, 71)?;
        Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
        let k_u8 = Uint8Array::from(k);
        self.u8.set(&k_u8, OFF_PAYLOAD);
        self.send_and_wait()?;
        let removed = Atomics::load(&self.i32, OFF_RESULT / 4)? != 0;
        Ok(removed)
    }

    /// Main-thread `iterate()` — synchronous; returns a `js_sys::Array` of κ-label strings.
    pub fn iterate(&self) -> Result<js_sys::Array, JsValue> {
        Atomics::store(&self.i32, OFF_OP / 4, Op::Iterate as i32)?;
        Atomics::store(&self.i32, OFF_LEN_A / 4, 0)?;
        Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
        self.send_and_wait()?;
        let n = Atomics::load(&self.i32, OFF_RESULT / 4)? as usize;
        let out = js_sys::Array::new();
        for i in 0..n {
            let mut k = [0u8; 71];
            for j in 0..71 {
                k[j] = self.u8.get_index(OFF_PAYLOAD + 4 + (i * 71 + j) as u32);
            }
            if let Ok(s) = String::from_utf8(k.to_vec()) {
                out.push(&JsValue::from_str(&s));
            }
        }
        Ok(out)
    }

    fn send_and_wait(&self) -> Result<(), JsValue> {
        // Transition STATE → REQUEST, then notify the Worker (which is waiting on STATE_IDLE).
        Atomics::store(&self.i32, OFF_STATE / 4, STATE_REQUEST)?;
        Atomics::notify(&self.i32, OFF_STATE / 4)?;
        // Wait until STATE != REQUEST (Worker has set it to RESPONSE or ERROR).
        loop {
            let _ = Atomics::wait(&self.i32, OFF_STATE / 4, STATE_REQUEST);
            let cur = Atomics::load(&self.i32, OFF_STATE / 4)?;
            if cur != STATE_REQUEST {
                break;
            }
            // Spurious wakeup (or "not-equal" — state was already != REQUEST). Loop and re-check.
        }
        let state = Atomics::load(&self.i32, OFF_STATE / 4)?;
        let _ = Atomics::store(&self.i32, OFF_STATE / 4, STATE_IDLE);
        if state == STATE_ERROR {
            return Err(JsValue::from_str("bridge: Worker reported error"));
        }
        Ok(())
    }
}

impl Default for SyncOpfsBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ── Worker side ─────────────────────────────────────────────────────────────────────────────

/// Worker-side dispatcher. Constructed from the SAB the main thread postMessage'd; `serve_step`
/// blocks until a request arrives, handles it, and writes the response. The Worker's JS glue
/// runs this in a loop (or one-shot per request, per design).
#[wasm_bindgen]
pub struct BridgeWorker {
    sab: SharedArrayBuffer,
    i32: Int32Array,
    u8: Uint8Array,
}

#[wasm_bindgen]
impl BridgeWorker {
    #[wasm_bindgen(constructor)]
    pub fn new(sab: SharedArrayBuffer) -> Self {
        let i32 = Int32Array::new(&sab);
        let u8 = Uint8Array::new(&sab);
        Self { sab, i32, u8 }
    }

    /// Try to dispatch one pending request. If the SAB state is not `REQUEST`, returns
    /// `Ok(NO_WORK)` immediately — the JS Worker loop is expected to suspend on
    /// `Atomics.waitAsync(i32_view, STATE_OFFSET, STATE_IDLE)` between calls (the standard
    /// non-blocking wait primitive on Workers; see `web/bridge-worker.mjs`). Returns the
    /// opcode that was handled on a normal dispatch.
    pub async fn serve_step(&self) -> Result<u32, JsValue> {
        let cur = Atomics::load(&self.i32, OFF_STATE / 4)?;
        if cur != STATE_REQUEST {
            return Ok(u32::MAX);
        }
        let op = Atomics::load(&self.i32, OFF_OP / 4)? as u32;
        let len_a = Atomics::load(&self.i32, OFF_LEN_A / 4)? as usize;
        let len_b = Atomics::load(&self.i32, OFF_LEN_B / 4)? as usize;
        let mut payload_a = Vec::with_capacity(len_a);
        for i in 0..len_a {
            payload_a.push(self.u8.get_index(OFF_PAYLOAD + i as u32));
        }
        let mut payload_b = Vec::with_capacity(len_b);
        for i in 0..len_b {
            payload_b.push(self.u8.get_index(OFF_PAYLOAD + (len_a + i) as u32));
        }
        let handled = match Op::from_u32(op) {
            Some(Op::Put) => {
                let axis = match String::from_utf8(payload_a) {
                    Ok(s) => s,
                    Err(_) => {
                        self.write_error()?;
                        return Ok(op);
                    }
                };
                if axis != "blake3" {
                    self.write_error()?;
                    return Ok(op);
                }
                // Compute κ locally so we can return it without re-hashing on the response side.
                let kappa = address_bytes(&payload_b);
                let put_result = crate::opfs::opfs_put(payload_b).await;
                match put_result {
                    Ok(_) => {
                        let bytes = kappa.as_str().as_bytes();
                        let arr = Uint8Array::from(bytes);
                        self.u8.set(&arr, OFF_PAYLOAD);
                        Atomics::store(&self.i32, OFF_LEN_A / 4, bytes.len() as i32)?;
                        Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
                        Atomics::store(&self.i32, OFF_RESULT / 4, 1)?;
                        self.complete()?;
                    }
                    Err(_) => {
                        self.write_error()?;
                    }
                }
                op
            }
            Some(Op::Get) => {
                let kappa = match String::from_utf8(payload_a) {
                    Ok(s) => s,
                    Err(_) => {
                        self.write_error()?;
                        return Ok(op);
                    }
                };
                let result = crate::opfs::opfs_get(kappa).await;
                match result {
                    Ok(v) if v.is_null() => {
                        Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
                        Atomics::store(&self.i32, OFF_RESULT / 4, 0)?; // absent
                        self.complete()?;
                    }
                    Ok(v) => {
                        let arr = Uint8Array::new(&v);
                        let n = arr.length() as usize;
                        self.u8.set(&arr, OFF_PAYLOAD);
                        Atomics::store(&self.i32, OFF_LEN_B / 4, n as i32)?;
                        Atomics::store(&self.i32, OFF_RESULT / 4, 1)?;
                        self.complete()?;
                    }
                    Err(_) => {
                        self.write_error()?;
                    }
                }
                op
            }
            Some(Op::Contains) => {
                let kappa = match String::from_utf8(payload_a) {
                    Ok(s) => s,
                    Err(_) => {
                        self.write_error()?;
                        return Ok(op);
                    }
                };
                let present = match crate::opfs::opfs_get(kappa).await {
                    Ok(v) if !v.is_null() => 1,
                    Ok(_) => 0,
                    Err(_) => {
                        self.write_error()?;
                        return Ok(op);
                    }
                };
                Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
                Atomics::store(&self.i32, OFF_RESULT / 4, present)?;
                self.complete()?;
                op
            }
            Some(Op::Delete) => {
                let kappa = match String::from_utf8(payload_a) {
                    Ok(s) => s,
                    Err(_) => {
                        self.write_error()?;
                        return Ok(op);
                    }
                };
                let deleted = match crate::opfs::opfs_delete(kappa).await {
                    Ok(b) => {
                        if b {
                            1
                        } else {
                            0
                        }
                    }
                    Err(_) => {
                        self.write_error()?;
                        return Ok(op);
                    }
                };
                Atomics::store(&self.i32, OFF_LEN_B / 4, 0)?;
                Atomics::store(&self.i32, OFF_RESULT / 4, deleted)?;
                self.complete()?;
                op
            }
            Some(Op::Iterate) => {
                // Iterate returns an array of κ-strings. Pack as `u32 LE count | (71 bytes)*`.
                match crate::opfs::opfs_iterate().await {
                    Ok(arr_val) => {
                        let arr: js_sys::Array =
                            arr_val.dyn_into().unwrap_or_else(|_| js_sys::Array::new());
                        let n = arr.length() as usize;
                        // Cap by SAB capacity; the structural cap is the SAB size, not policy.
                        let capacity = self.sab.byte_length() as usize;
                        let max_n = (capacity.saturating_sub(OFF_PAYLOAD as usize + 4)) / 71;
                        let actual = n.min(max_n);
                        // Write count.
                        let count_bytes = (actual as u32).to_le_bytes();
                        for (i, &b) in count_bytes.iter().enumerate() {
                            self.u8.set_index(OFF_PAYLOAD + i as u32, b);
                        }
                        // Write each κ string (71 bytes).
                        for i in 0..actual {
                            if let Some(s) = arr.get(i as u32).as_string() {
                                let bytes = s.as_bytes();
                                if bytes.len() == 71 {
                                    for (j, &b) in bytes.iter().enumerate() {
                                        self.u8.set_index(OFF_PAYLOAD + 4 + (i * 71 + j) as u32, b);
                                    }
                                }
                            }
                        }
                        Atomics::store(&self.i32, OFF_LEN_B / 4, (4 + actual * 71) as i32)?;
                        Atomics::store(&self.i32, OFF_RESULT / 4, actual as i32)?;
                        self.complete()?;
                    }
                    Err(_) => {
                        self.write_error()?;
                    }
                }
                op
            }
            None => {
                self.write_error()?;
                op
            }
        };
        Ok(handled)
    }

    fn complete(&self) -> Result<(), JsValue> {
        Atomics::store(&self.i32, OFF_STATE / 4, STATE_RESPONSE)?;
        Atomics::notify(&self.i32, OFF_STATE / 4)?;
        Ok(())
    }

    fn write_error(&self) -> Result<(), JsValue> {
        Atomics::store(&self.i32, OFF_STATE / 4, STATE_ERROR)?;
        Atomics::notify(&self.i32, OFF_STATE / 4)?;
        Ok(())
    }

    /// Diagnostic: SAB size in bytes.
    pub fn capacity(&self) -> u32 {
        self.sab.byte_length()
    }

    /// Sentinel returned by [`serve_step`] when no request is pending (the JS loop should
    /// `Atomics.waitAsync(i32_view, 0, STATE_IDLE)` until the main thread notifies). `u32::MAX`.
    pub fn no_work() -> u32 {
        u32::MAX
    }

    /// The `Int32Array` view used by `Atomics.waitAsync` in the JS Worker loop. Exposed so the
    /// JS glue can `Atomics.waitAsync(bridge.i32_view, 0, STATE_IDLE)` without re-constructing
    /// the view (and so the offset/state constants stay consistent across Rust + JS).
    pub fn i32_view(&self) -> Int32Array {
        self.i32.clone()
    }
}

/// Constants exposed to the JS glue so the Worker can correctly call `Atomics.waitAsync`.
#[wasm_bindgen]
pub fn bridge_state_offset() -> u32 {
    OFF_STATE / 4
}

#[wasm_bindgen]
pub fn bridge_state_idle() -> i32 {
    STATE_IDLE
}
