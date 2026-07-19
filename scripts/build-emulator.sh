#!/usr/bin/env bash
# Build the hologram system-emulator **codemodule** — the emulator core compiled to a hologram Wasm
# container (a `cdylib` exporting the `hg_*` container ABI and importing only the `hologram` host
# ABI; the ADR-009 execution surface). This absorbs the former `hologram-emulator-codemodule` crate:
# the codemodule now lives behind `hologram-emulator`'s `codemodule` feature, built as a cdylib via
# `cargo rustc --crate-type cdylib` (so normal builds of the crate stay plain libs).
#
# Usage: scripts/build-emulator.sh            # release wasm32 codemodule
#        CARGO=/path/to/cargo scripts/build-emulator.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CARGO="${CARGO:-cargo}"

"$CARGO" rustc -p hologram-emulator \
  --target wasm32-unknown-unknown \
  --no-default-features --features codemodule \
  --release --crate-type cdylib

OUT="target/wasm32-unknown-unknown/release/hologram_emulator.wasm"
if [[ -f "$OUT" ]]; then
  echo "built emulator codemodule: $OUT ($(wc -c < "$OUT") bytes)"
else
  echo "error: expected codemodule at $OUT" >&2
  exit 1
fi
