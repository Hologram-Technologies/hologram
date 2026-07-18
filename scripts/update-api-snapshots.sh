#!/usr/bin/env bash
# Regenerate the committed public-API snapshots under `api/<crate>.txt` for
# every workspace library crate, using cargo-public-api. Run this whenever you
# change a public API; commit the resulting `api/` diff alongside the code so
# the change is reviewed and the `api-gate.yml` check passes.
#
# Usage:
#   scripts/update-api-snapshots.sh           # regenerate (overwrite) snapshots
#   scripts/update-api-snapshots.sh --check    # exit 1 if any snapshot is stale
set -euo pipefail

MODE="${1:-write}"

if ! command -v cargo-public-api >/dev/null 2>&1; then
  echo "cargo-public-api not found; install with:" >&2
  echo "  cargo install cargo-public-api --locked" >&2
  exit 127
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p api

# Library crates that carry a public-API contract (skip the CLI binary and the
# bench harness, which expose no consumed library surface).
CRATES=(
  uor-hologram
  hologram-types
  hologram-ops
  hologram-graph
  hologram-compute
  hologram-archive
  hologram-exec
  hologram-compiler
  hologram-ffi
)

rc=0
for crate in "${CRATES[@]}"; do
  out="api/${crate}.txt"
  tmp="$(mktemp)"
  echo "==> $crate"
  # `--simplified` keeps the snapshot to the stable item surface (no blanket
  # auto-trait/blanket-impl noise), so diffs reflect intentional API changes.
  cargo public-api --simplified -p "$crate" --target x86_64-unknown-linux-gnu > "$tmp"
  if [ "$MODE" = "--check" ]; then
    if ! diff -u "$out" "$tmp" >/dev/null 2>&1; then
      echo "::error::API snapshot for $crate is stale — run scripts/update-api-snapshots.sh"
      diff -u "$out" "$tmp" || true
      rc=1
    fi
    rm -f "$tmp"
  else
    mv "$tmp" "$out"
  fi
done

[ "$MODE" = "--check" ] || echo "Wrote API snapshots to api/"
exit $rc
