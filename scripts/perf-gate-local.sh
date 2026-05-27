#!/usr/bin/env bash
# Run the performance gate locally: benchmark the current working tree against
# a baseline ref (default: origin/main) and report regressions, exactly as the
# `perf-gate.yml` CI job does.
#
# Usage:
#   scripts/perf-gate-local.sh [BASELINE_REF] [THRESHOLD] [NOISE_SIGMAS]
#
#   BASELINE_REF   git ref to compare against        (default: origin/main)
#   THRESHOLD      tolerated relative median slowdown (default: 0.10 = 10%)
#   NOISE_SIGMAS   slowdown must exceed Nσ of jitter  (default: 2.0)
#
# The current tree is benchmarked in place; the baseline is benchmarked in a
# throwaway git worktree so your tree is untouched. Exits non-zero on regression.
set -euo pipefail

BASELINE_REF="${1:-origin/main}"
THRESHOLD="${2:-0.10}"
NOISE_SIGMAS="${3:-2.0}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE_SHA="$(git rev-parse "$BASELINE_REF")"
WORKTREE="$(mktemp -d)"
trap 'git worktree remove --force "$WORKTREE" 2>/dev/null || true; rm -rf "$WORKTREE"' EXIT

echo "==> Benchmarking current tree (PR side)…"
cargo bench --workspace
python3 scripts/aggregate-benchmarks.py target/criterion "$ROOT/pr-bench.json" "$(git rev-parse HEAD)"

echo "==> Benchmarking baseline $BASELINE_REF ($BASE_SHA)…"
git worktree add --detach "$WORKTREE" "$BASE_SHA"
( cd "$WORKTREE" && cargo bench --workspace )
python3 scripts/aggregate-benchmarks.py "$WORKTREE/target/criterion" "$ROOT/base-bench.json" "$BASE_SHA"

echo "==> Comparing…"
python3 scripts/compare-benchmarks.py base-bench.json pr-bench.json \
  --threshold "$THRESHOLD" --noise-sigmas "$NOISE_SIGMAS"
