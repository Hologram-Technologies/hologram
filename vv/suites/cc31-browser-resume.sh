#!/usr/bin/env bash
# CC-31 — persisted browser resume plus a live idle-shell host differential.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cargo test --release --manifest-path "$ROOT/Cargo.toml" -p holospaces \
  --test cc31_resume_terminal -- --ignored --nocapture
HOLOSPACES_BROWSER_ONLY=cc31 "$ROOT/scripts/browser-manager-test.sh"
