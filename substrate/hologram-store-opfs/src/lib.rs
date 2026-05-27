//! # hologram-store-opfs
//!
//! The **browser OPFS `KappaStore`** (spec §5.4): κ→bytes persisted in the Origin Private File
//! System, keyed by hologram's σ-axis κ-label (`address_bytes`). Persistence is per-origin, so
//! stored κ survive a page reload. Verified in a real browser (Chromium) via Playwright.
//!
//! The κ-addressing is the *same* `hologram-substrate-core` path the other substrates use — so a κ
//! minted in the browser is byte-identical to one minted on native/bare-metal (substrate-tripling).

use hologram_substrate_core::{address_bytes, verify_kappa, KappaLabel};
use js_sys::Uint8Array;
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
