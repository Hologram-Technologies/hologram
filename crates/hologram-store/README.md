# hologram-store

> The hologram `KappaStore` backends in one crate, feature-gated per platform.

`hologram-store` consolidates the former `hologram-store-{bare,native,opfs}` sibling crates into a
single crate whose backends are selected by feature. Each backend is disjoint in its dependencies
and its target, and every backend passes the shared `hologram-tck` conformance TCK **identically to
the in-memory reference**. κ is the σ-axis content address throughout (BLAKE3, 71-byte
`blake3:<hex>`), so a κ minted by any backend is byte-identical to one minted anywhere else —
verify-by-re-derivation, SPINE-4.

## What it provides

- `bare` module — a `no_std` + alloc bare-metal `KappaStore` over a raw `BlockDevice`, no filesystem: dual header sectors (alternating writes, higher-`gen` wins, crash-atomic), a chain of κ-addressed copy-on-write leaf pages as the index, a pinned-set chain, and bump-allocated data extents. Device I/O is async, driven by a minimal `no_std` busy-poll `block_on`.
- `native` module — a WASI/std `KappaStore` on a **redb** B-tree index, with content sharding above `SHARD_THRESHOLD` (64 KiB) into content-addressed shards plus a shard manifest, and a size-bounded LRU read-through cache (`CacheConfig`) that honors the SP zero-copy floor. Reachability `gc` walks the realization registry.
- `opfs` module — a browser OPFS `KappaStore`. `OpfsKappaStore` (in `sync_store`) is the in-product synchronous backend: a single append-only OPFS pack file plus an in-RAM offset index, driven through a `FileSystemSyncAccessHandle` inside a Worker. The `js_api` + SAB `bridge` layer adds an async, file-per-κ persistence + GC reference with `#[wasm_bindgen]` exports.

## Features

- `std` — enables `std` in `hologram-space`. Pulled in transitively by `native`, `opfs`, and `js-api`.
- `bare` — the `no_std` bare-metal `BlockDevice` backend (`bare` module); pulls `hologram-types`, `hashbrown`, `spin`.
- `native` — the std redb B-tree backend (`native` module); implies `std`, pulls `redb`.
- `opfs` — the `wasm32` browser OPFS backend (`opfs` module) — the sync `OpfsKappaStore` only (rlib, no `wasm-bindgen`); implies `std`, pulls `web-sys`.
- `js-api` — the async, file-per-κ `#[wasm_bindgen]` JS layer over OPFS (`js_api` + `bridge`); implies `opfs` and pulls the bindgen deps (`wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`).

No feature is enabled by default; pick the backend for your target.

## Targets & build notes

`bare` is `no_std` + `alloc` (bare-metal / embedded, e.g. `thumbv7em`). `native` is std-only.
`opfs` / `js-api` target `wasm32`. A consumer wanting only the sync OPFS backend takes
`default-features = false` + `opfs` and pulls no `wasm-bindgen`. The Playwright browser bundle for
the async JS API is built via `scripts/opfs-browser-test.sh`
(`cargo rustc --crate-type cdylib -p hologram-store --features js-api`).

Part of the [hologram](../../README.md) workspace.
