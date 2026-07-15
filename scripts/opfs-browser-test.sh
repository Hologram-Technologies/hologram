#!/usr/bin/env bash
# End-to-end OPFS KappaStore test in a real browser: build the wasm32 store, generate JS glue, and
# run the Playwright (Chromium) test — put/get round-trip + reload persistence + verify-on-receipt.
# Requires: wasm32-unknown-unknown target, wasm-bindgen, node, and Playwright's Chromium.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/crates/hologram-store-opfs"

if ! command -v wasm-bindgen >/dev/null || ! command -v node >/dev/null; then
  echo "SKIP: wasm-bindgen and/or node not available"; exit 0
fi

echo "==> building OPFS store (wasm32-unknown-unknown)"
cargo build --release --target wasm32-unknown-unknown --manifest-path "$CRATE/Cargo.toml"
wasm-bindgen "$CRATE/target/wasm32-unknown-unknown/release/hologram_store_opfs.wasm" \
  --out-dir "$CRATE/web/pkg" --target web

cd "$CRATE/web"
[ -d node_modules/playwright ] || npm i playwright >/dev/null 2>&1
if [ ! -d "$HOME/.cache/ms-playwright" ]; then
  echo "SKIP: Playwright browser not installed (run: npx playwright install chromium)"; exit 0
fi

echo "==> running OPFS test in Chromium"
node opfs-test.mjs
