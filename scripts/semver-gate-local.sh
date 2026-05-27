#!/usr/bin/env bash
# Run the semver gate locally: check the workspace's public API against a
# baseline ref and fail if the version bump doesn't cover the API change
# (under 0.x: breaking ⇒ minor bump, additive ⇒ patch bump). Mirrors
# `semver-gate.yml`.
#
# Usage: scripts/semver-gate-local.sh [BASELINE_REF]   (default: origin/main)
set -euo pipefail

BASELINE_REF="${1:-origin/main}"

if ! command -v cargo-semver-checks >/dev/null 2>&1; then
  echo "cargo-semver-checks not found; install with:" >&2
  echo "  cargo install cargo-semver-checks --locked" >&2
  exit 127
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE_SHA="$(git rev-parse "$BASELINE_REF")"
echo "==> Semver-checking workspace public API vs $BASELINE_REF ($BASE_SHA)…"
exec cargo semver-checks check-release \
  --workspace \
  --baseline-rev "$BASE_SHA" \
  --default-features
