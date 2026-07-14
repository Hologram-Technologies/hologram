//! # hologram-store-opfs
//!
//! The **browser OPFS `KappaStore`** (spec §5.4): κ→bytes persisted in the Origin Private File
//! System, keyed by hologram's σ-axis κ-label (`address_bytes`). Persistence is per-origin, so
//! stored κ survive a page reload. Verified in a real browser (Chromium) via Playwright.
//!
//! The κ-addressing is the *same* `hologram-substrate-core` path the other substrates use — so a κ
//! minted in the browser is byte-identical to one minted on native/bare-metal (substrate-tripling).

extern crate alloc;

pub mod bridge;

use hologram_space::{address_bytes, references, verify_kappa, KappaLabel, KappaLabel71};
use js_sys::{Array, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
    FileSystemWritableFileStream,
};

/// Mint the κ-label for `bytes` on the BLAKE3 σ-axis (the same address as every other substrate).
#[wasm_bindgen]
pub fn address(bytes: &[u8]) -> String {
    address_bytes(bytes).as_str().to_string()
}

/// Verify `bytes` re-derive to `kappa` through the σ-axis (SPINE-4).
#[wasm_bindgen]
pub fn verify(bytes: &[u8], kappa: &str) -> bool {
    let arr: [u8; 71] = match kappa.as_bytes().try_into() {
        Ok(a) => a,
        Err(_) => return false,
    };
    match KappaLabel::<71>::from_bytes(&arr) {
        Ok(k) => verify_kappa(bytes, &k).unwrap_or(false),
        Err(_) => false,
    }
}

/// OPFS file name for a κ-label (`:` is not a valid OPFS name char).
fn file_name(kappa: &str) -> String {
    kappa.replace(':', "_")
}

async fn opfs_root() -> Result<FileSystemDirectoryHandle, JsValue> {
    let storage = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?.navigator().storage();
    JsFuture::from(storage.get_directory()).await?.dyn_into::<FileSystemDirectoryHandle>()
}

/// Persist `bytes` under their κ-label in OPFS; returns the κ-label. Idempotent at the content level
/// (same bytes ⇒ same κ ⇒ same file).
#[wasm_bindgen]
pub async fn opfs_put(bytes: Vec<u8>) -> Result<String, JsValue> {
    let kappa = address_bytes(&bytes);
    let dir = opfs_root().await?;
    let opts = FileSystemGetFileOptions::new();
    opts.set_create(true);
    let fh = JsFuture::from(dir.get_file_handle_with_options(&file_name(kappa.as_str()), &opts))
        .await?
        .dyn_into::<FileSystemFileHandle>()?;
    let writable = JsFuture::from(fh.create_writable()).await?.dyn_into::<FileSystemWritableFileStream>()?;
    JsFuture::from(writable.write_with_u8_array(&bytes)?).await?;
    JsFuture::from(writable.close()).await?;
    Ok(kappa.as_str().to_string())
}

/// Read the bytes stored under `kappa` from OPFS; `null` if absent (eviction-tolerant). The bytes
/// are **verified** against `kappa` before returning (SPINE-4) — a tampered OPFS file is rejected.
#[wasm_bindgen]
pub async fn opfs_get(kappa: String) -> Result<JsValue, JsValue> {
    let dir = opfs_root().await?;
    let fh = match JsFuture::from(dir.get_file_handle(&file_name(&kappa))).await {
        Ok(h) => h.dyn_into::<FileSystemFileHandle>()?,
        Err(_) => return Ok(JsValue::NULL), // not present locally
    };
    let file = JsFuture::from(fh.get_file()).await?.dyn_into::<web_sys::File>()?;
    let buf = JsFuture::from(file.array_buffer()).await?;
    let arr = Uint8Array::new(&buf);
    let bytes = arr.to_vec();
    if !verify(&bytes, &kappa) {
        return Err(JsValue::from_str("OPFS content failed σ-axis verification"));
    }
    Ok(Uint8Array::from(bytes.as_slice()).into())
}

// ───────────────────────────── OPFS GC (arch §11.5) ─────────────────────────────

/// Reverse `file_name`: turn a `_`-encoded OPFS file name back into the canonical κ-label. Returns
/// `None` if the name doesn't parse as a κ.
fn name_to_kappa(name: &str) -> Option<KappaLabel71> {
    let original = name.replacen('_', ":", 1);
    let arr: [u8; 71] = original.as_bytes().try_into().ok()?;
    KappaLabel::from_bytes(&arr).ok()
}

/// List every κ-label currently held in OPFS. The browser's `FileSystemDirectoryHandle` exposes an
/// async iterator over `[name, handle]` entries; we walk it via the JS `keys()` async iterator.
#[wasm_bindgen]
pub async fn opfs_iterate() -> Result<JsValue, JsValue> {
    let dir = opfs_root().await?;
    // `dir.keys()` returns an async iterator yielding the file names. Iterating it requires the
    // JS-level `next()` calls (chrome supports this on FileSystemDirectoryHandle).
    let keys_fn = Reflect::get(&dir, &JsValue::from_str("keys"))?;
    let keys_fn = keys_fn
        .dyn_into::<js_sys::Function>()
        .map_err(|_| JsValue::from_str("dir.keys is not a function"))?;
    let iterator = keys_fn.call0(&dir)?;
    let next_fn = Reflect::get(&iterator, &JsValue::from_str("next"))?;
    let next_fn = next_fn
        .dyn_into::<js_sys::Function>()
        .map_err(|_| JsValue::from_str("iterator.next is not a function"))?;
    let out = Array::new();
    loop {
        let result_promise = next_fn.call0(&iterator)?;
        let result = JsFuture::from(js_sys::Promise::from(result_promise)).await?;
        let done = Reflect::get(&result, &JsValue::from_str("done"))?;
        if done.is_truthy() {
            break;
        }
        let name = Reflect::get(&result, &JsValue::from_str("value"))?;
        if let Some(name_str) = name.as_string() {
            if let Some(k) = name_to_kappa(&name_str) {
                out.push(&JsValue::from_str(k.as_str()));
            }
        }
    }
    Ok(out.into())
}

/// Remove the file backing `kappa` from OPFS. Returns `true` if a file was removed, `false` if it
/// wasn't present.
#[wasm_bindgen]
pub async fn opfs_delete(kappa: String) -> Result<bool, JsValue> {
    let dir = opfs_root().await?;
    match JsFuture::from(dir.remove_entry(&file_name(&kappa))).await {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Reachability walk + sweep (arch §11.5). `pins` is the durable root set; the walk follows each
/// pin's `references()` via the realization registry recursively (SPINE-3) and deletes every OPFS
/// file whose κ is not in the resulting reachable set. Returns the count of files deleted.
#[wasm_bindgen]
pub async fn opfs_gc(pins: JsValue) -> Result<u32, JsValue> {
    // Parse the JS array of κ-label strings into KappaLabel71s.
    let pins_arr: Array = pins
        .dyn_into()
        .map_err(|_| JsValue::from_str("opfs_gc: pins must be an array of κ-label strings"))?;
    let mut roots: alloc::vec::Vec<KappaLabel71> = alloc::vec::Vec::new();
    for v in pins_arr.iter() {
        let s = v
            .as_string()
            .ok_or_else(|| JsValue::from_str("opfs_gc: pin entry not a string"))?;
        let arr: [u8; 71] = s
            .as_bytes()
            .try_into()
            .map_err(|_| JsValue::from_str("opfs_gc: pin κ wrong length"))?;
        roots.push(
            KappaLabel::from_bytes(&arr)
                .map_err(|_| JsValue::from_str("opfs_gc: pin κ malformed"))?,
        );
    }

    // Mark: BFS through references() from each pin.
    use alloc::collections::BTreeSet;
    let mut reachable: BTreeSet<[u8; 71]> = BTreeSet::new();
    let mut frontier: alloc::vec::Vec<KappaLabel71> = roots.clone();
    while let Some(k) = frontier.pop() {
        if !reachable.insert(*k.as_array()) {
            continue;
        }
        // Read bytes from OPFS; if absent (e.g. a pin not yet locally cached) skip.
        let bytes = match opfs_get(k.as_str().to_string()).await {
            Ok(v) if !v.is_null() => Uint8Array::new(&v).to_vec(),
            _ => continue,
        };
        if let Ok(refs) = references(&bytes, hologram_realizations::REGISTRY) {
            for r in refs {
                if !reachable.contains(r.as_array()) {
                    frontier.push(r);
                }
            }
        }
    }

    // Sweep: enumerate OPFS files; delete those whose κ isn't reachable.
    let listed = opfs_iterate().await?;
    let listed: Array = listed.dyn_into().unwrap();
    let mut deleted = 0u32;
    for v in listed.iter() {
        if let Some(s) = v.as_string() {
            let arr: [u8; 71] = match s.as_bytes().try_into() {
                Ok(a) => a,
                Err(_) => continue,
            };
            if !reachable.contains(&arr) {
                if opfs_delete(s).await? {
                    deleted += 1;
                }
            }
        }
    }
    Ok(deleted)
}
