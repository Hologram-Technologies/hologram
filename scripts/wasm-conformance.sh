#!/usr/bin/env bash
# Cross-substrate f32 kernel conformance: run one digest body
# (hologram_compute::cpu::conformance) natively and on wasm32, and compare.
#
#   scripts/wasm-conformance.sh
#
# The wasm half is built for every wasm32 tier the runtime ships — baseline,
# SIMD128 (the deployed browser kernel), and SIMD128 + relaxed-SIMD — and run
# under Node's built-in WebAssembly engine (crates/hologram-compute/conformance-wasm/run.mjs).
#
# Gate: the dot/sum digest must be identical on the native lane and every wasm
# tier (docs/numerics/invariance.md, CONFORMANCE.md KC-7). The matmul digest is
# printed for every substrate but does not gate: matmul_f32_blocked is not
# cross-substrate identical (FMA leaves on x86/NEON, multiply-then-add on wasm).
#
# Needs: the wasm32-unknown-unknown target and Node.js >= 22.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
DRIVER="crates/hologram-compute/conformance-wasm"

native="$(cargo run -q --release -p hologram-compute --example conformance_digest)"
echo "native:              $(echo "$native" | tr '\n' ' ')"
native_dot="$(echo "$native" | awk '$1 == "dot_sum" { print $2 }')"
[ -n "$native_dot" ] || { echo "::error::no native dot_sum digest"; exit 1; }

rc=0
for tier in "baseline:" "simd128:-Ctarget-feature=+simd128" \
            "relaxed-simd:-Ctarget-feature=+simd128,+relaxed-simd"; do
  name="${tier%%:*}"
  flags="${tier#*:}"
  target_dir="$ROOT/target/wasm-conformance/$name"
  RUSTFLAGS="$flags" cargo build -q --release --locked \
    --manifest-path "$DRIVER/Cargo.toml" --target wasm32-unknown-unknown \
    --target-dir "$target_dir"
  wasm="$target_dir/wasm32-unknown-unknown/release/hologram_compute_conformance_wasm.wasm"
  out="$(node "$DRIVER/run.mjs" "$wasm")"
  printf 'wasm32 %-13s %s\n' "$name:" "$(echo "$out" | tr '\n' ' ')"
  dot="$(echo "$out" | awk '$1 == "dot_sum" { print $2 }')"
  if [ "$dot" != "$native_dot" ]; then
    echo "::error::dot/sum digest differs: native $native_dot, wasm32 $name $dot"
    rc=1
  fi
done

[ "$rc" -eq 0 ] && echo "dot/sum: native == wasm32 on every tier ($native_dot)"
exit "$rc"
