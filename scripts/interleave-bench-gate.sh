#!/usr/bin/env bash
# Interleaved best-of-N benchmarking of two source trees on one runner.
#
# The perf gate compares a PR tree against a baseline tree. Running them as two
# separate phases (all PR runs, then all baseline runs ~minutes later) lets the
# runner's throughput drift between phases read as a regression — best-of-N only
# de-noises *within* a phase, not across the gap. So instead we **interleave**:
# each round benchmarks the baseline then the PR (seconds apart), so any drift in
# that round hits both sides equally; then each side is reduced to its
# per-benchmark minimum median across rounds (transient spikes removed).
#
# Usage:
#   interleave-bench-gate.sh <pr_tree> <base_tree> <scripts_dir> <runs> \
#       <pr_out.json> <base_out.json>
set -euo pipefail

PR_TREE="$1"
BASE_TREE="$2"
SCRIPTS="$3"
RUNS="$4"
PR_OUT="$5"
BASE_OUT="$6"
# Criterion sampling, env-overridable (no hardcoded ceiling). Defaults trade a
# few minutes per round for stable medians; CI/tests can shorten them.
CRIT=(
  --warm-up-time "${BENCH_WARMUP:-1}"
  --measurement-time "${BENCH_MEASURE:-3}"
  --sample-size "${BENCH_SAMPLES:-30}"
)

# Criterion bench targets of the `hologram-bench` crate in <tree>. Selecting
# them by `--bench <name>` keeps the crate lib's libtest harness (which rejects
# criterion flags) out of the run, independent of that tree's `[lib] bench`.
bench_targets() {
  cargo metadata --manifest-path "$1/Cargo.toml" --no-deps --format-version 1 \
    | python3 -c "import json,sys; m=json.load(sys.stdin); [print(t['name']) for p in m['packages'] if p['name']=='hologram-bench' for t in p['targets'] if 'bench' in t['kind']]"
}

run_one() { # <tree> <out.json>
  local tree="$1" out="$2"
  local flags=()
  while IFS= read -r t; do flags+=(--bench "$t"); done < <(bench_targets "$tree")
  [ "${#flags[@]}" -gt 0 ] || { echo "::error::no criterion bench targets in $tree"; exit 1; }
  ( cd "$tree" && cargo bench -p hologram-bench "${flags[@]}" -- "${CRIT[@]}" )
  python3 "$SCRIPTS/aggregate-benchmarks.py" "$tree/target/criterion" "$out" "round"
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
base_jsons=()
pr_jsons=()
for r in $(seq 1 "$RUNS"); do
  echo "==> round $r/$RUNS — baseline"
  run_one "$BASE_TREE" "$TMP/base_$r.json"
  echo "==> round $r/$RUNS — PR"
  run_one "$PR_TREE" "$TMP/pr_$r.json"
  base_jsons+=("$TMP/base_$r.json")
  pr_jsons+=("$TMP/pr_$r.json")
done

python3 "$SCRIPTS/bench-reduce-min.py" "${base_jsons[@]}" -o "$BASE_OUT"
python3 "$SCRIPTS/bench-reduce-min.py" "${pr_jsons[@]}" -o "$PR_OUT"
echo "Interleaved best-of-$RUNS complete → $PR_OUT (PR), $BASE_OUT (baseline)"
