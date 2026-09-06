#!/usr/bin/env bash
# CC-47 — suspend → κ snapshot → drop → resume parity on AArch64 and x86-64.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v cargo >/dev/null 2>&1 || {
    echo "cc47-lifecycle-arch-parity: cargo unavailable" >&2
    exit 127
}
cargo test --release --manifest-path "$ROOT/Cargo.toml" -p holospaces \
    --test cc47_lifecycle_parity -- --nocapture
