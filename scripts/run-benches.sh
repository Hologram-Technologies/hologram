#!/usr/bin/env bash
# Run the criterion benchmarks N times and reduce to the per-benchmark minimum
# median (best-of-N), writing one aggregated bench.json. Used by the perf gate
# for both the PR and the baseline tree, and by `perf-gate-local.sh`.
#
# Robust across trees: it enumerates the `hologram-bench` crate's criterion
# bench *targets* from `cargo metadata` and runs each with `--bench <name>`, so
# the libtest harness on the crate's lib never sees (and rejects) criterion's
# timing flags — independent of whether that tree sets `[lib] bench = false`.
#
# Usage:
#   scripts/run-benches.sh <out.json> [runs] [scripts_dir] [-- <criterion args>]
#
#   out.json      aggregated, min-reduced output
#   runs          number of repeats (default: 3) — min across them kills
#                 transient CI contention spikes
#   scripts_dir   dir holding aggregate/reduce scripts (default: this script's
#                 dir; pass the PR's scripts when benching a baseline checkout)
set -euo pipefail

OUT="${1:?usage: run-benches.sh <out.json> [runs] [scripts_dir] [-- criterion-args]}"
shift
RUNS=3
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRIT_ARGS=(--warm-up-time 1 --measurement-time 3 --sample-size 30)

# Optional positional runs / scripts_dir, then optional `-- criterion args`.
[ "${1:-}" != "--" ] && [ -n "${1:-}" ] && { RUNS="$1"; shift; }
[ "${1:-}" != "--" ] && [ -n "${1:-}" ] && { SCRIPTS="$1"; shift; }
if [ "${1:-}" = "--" ]; then shift; CRIT_ARGS=("$@"); fi

# Enumerate this tree's criterion bench targets.
mapfile -t TARGETS < <(
  cargo metadata --no-deps --format-version 1 \
    | python3 -c "import json,sys; m=json.load(sys.stdin); [print(t['name']) for p in m['packages'] if p['name']=='hologram-bench' for t in p['targets'] if 'bench' in t['kind']]"
)
[ "${#TARGETS[@]}" -gt 0 ] || { echo "::error::no criterion bench targets found"; exit 1; }
BENCH_FLAGS=(); for t in "${TARGETS[@]}"; do BENCH_FLAGS+=(--bench "$t"); done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
RUN_JSONS=()
for i in $(seq 1 "$RUNS"); do
  echo "==> benchmark run $i/$RUNS (${#TARGETS[@]} targets)"
  cargo bench -p hologram-bench "${BENCH_FLAGS[@]}" -- "${CRIT_ARGS[@]}"
  python3 "$SCRIPTS/aggregate-benchmarks.py" target/criterion "$TMP/run_$i.json" "run-$i"
  RUN_JSONS+=("$TMP/run_$i.json")
done

python3 "$SCRIPTS/bench-reduce-min.py" "${RUN_JSONS[@]}" -o "$OUT"
