#!/usr/bin/env bash
# End-to-end OPFS KappaStore test in a real browser: build the wasm32 store, generate JS glue, and
# run the Playwright (Chromium) test — put/get round-trip + reload persistence + verify-on-receipt.
# Requires: wasm32-unknown-unknown target, wasm-bindgen, node, and Playwright's Chromium.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$ROOT/crates/hologram-store/web"

if ! command -v wasm-bindgen >/dev/null || ! command -v node >/dev/null; then
  echo "SKIP: wasm-bindgen and/or node not available"; exit 0
fi

echo "==> building OPFS store (wasm32-unknown-unknown)"
# hologram-store's opfs js-api layer, built as a cdylib. `--crate-type cdylib` keeps hologram-store a
# plain lib for its normal rlib consumers (same trick as scripts/build-emulator.sh); since the store
# is now a workspace member, the wasm lands in the workspace target dir.
cargo rustc --release --target wasm32-unknown-unknown -p hologram-store --features js-api --crate-type cdylib
wasm-bindgen "$ROOT/target/wasm32-unknown-unknown/release/hologram_store.wasm" \
  --out-dir "$WEB/pkg" --target web

cd "$WEB"
[ -d node_modules/playwright ] || npm i playwright >/dev/null 2>&1
if [ ! -d "$HOME/.cache/ms-playwright" ]; then
  echo "SKIP: Playwright browser not installed (run: npx playwright install chromium)"; exit 0
fi

echo "==> running OPFS test in Chromium"
node opfs-test.mjs
