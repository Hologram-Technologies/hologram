#!/usr/bin/env bash
# Run the performance gate locally: benchmark the current working tree against a
# baseline ref (default: origin/main) and report regressions + benchmark-
# lifecycle violations, exactly as `perf-gate.yml` does (interleaved best-of-N
# min, the cv-floor noise model, and the manifest lifecycle gate).
#
# Usage:
#   scripts/perf-gate-local.sh [BASELINE_REF] [RUNS] [THRESHOLD] [NOISE_SIGMAS] [CV_FLOOR]
#
#   BASELINE_REF   git ref to compare against         (default: origin/main)
#   RUNS           best-of-N benchmark repeats         (default: 3)
#   THRESHOLD      tolerated relative median slowdown  (default: 0.10 = 10%)
#   NOISE_SIGMAS   slowdown must exceed Nσ of jitter   (default: 2.0)
#   CV_FLOOR       noise floor as fraction of baseline (default: 0.07)
#
# The current tree is benchmarked in place; the baseline is benchmarked in a
# throwaway git worktree so your tree is untouched. Exits non-zero on a
# regression or lifecycle violation.
set -euo pipefail

BASELINE_REF="${1:-origin/main}"
RUNS="${2:-3}"
THRESHOLD="${3:-0.10}"
NOISE_SIGMAS="${4:-2.0}"
CV_FLOOR="${5:-0.07}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE_SHA="$(git rev-parse "$BASELINE_REF")"
WORKTREE="$(mktemp -d)"
trap 'git worktree remove --force "$WORKTREE" 2>/dev/null || true; rm -rf "$WORKTREE"' EXIT

echo "==> Benchmarking current tree vs $BASELINE_REF ($BASE_SHA), interleaved best-of-$RUNS…"
git worktree add --detach "$WORKTREE" "$BASE_SHA"
scripts/interleave-bench-gate.sh \
  "$ROOT" "$WORKTREE" "$ROOT/scripts" "$RUNS" \
  "$ROOT/pr-bench.json" "$ROOT/base-bench.json"

BASE_MANIFEST_ARG=()
[ -f "$WORKTREE/benches/manifest.toml" ] && BASE_MANIFEST_ARG=(--baseline-manifest "$WORKTREE/benches/manifest.toml")

echo "==> Comparing…"
python3 scripts/compare-benchmarks.py base-bench.json pr-bench.json \
  --threshold "$THRESHOLD" --noise-sigmas "$NOISE_SIGMAS" --cv-floor "$CV_FLOOR" \
  --manifest benches/manifest.toml "${BASE_MANIFEST_ARG[@]}"
