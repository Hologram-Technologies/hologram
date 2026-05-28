#!/usr/bin/env python3
"""Reduce N aggregated benchmark runs to the per-benchmark **minimum** median.

On shared CI runners, contention only ever makes a benchmark *slower* than its
true cost — never faster. So across N independent runs the **minimum** median is
the best estimate of the uncontended time and the most reproducible statistic to
gate on. This removes the transient spikes that make a single run read a
false regression on an unchanged codebase.

For the benchmark kept (the one with the min median) we carry its own std-dev,
so the downstream noise model stays self-consistent.

Usage:
    bench-reduce-min.py run1.json run2.json [run3.json …] -o out.json
"""

import argparse
import json
import sys


def reduce_min(runs):
    """runs: list of aggregated bench.json dicts → one merged dict.

    Per benchmark, `median_ns` is the **minimum** median across rounds (the
    least-contended estimate of the true cost), and `max_ns` is the **maximum**
    median across rounds — the observed run-to-run spread, which the gate uses
    as a data-driven, per-benchmark noise band (so an intrinsically noisy
    microbenchmark widens its own tolerance instead of false-failing)."""
    best = {}  # name -> bench dict carrying the smallest median seen
    hi = {}  # name -> largest median seen across rounds
    for data in runs:
        for b in data.get("benchmarks", []):
            name, median = b.get("name"), b.get("median_ns")
            if name is None or median is None:
                continue
            if name not in best or median < best[name]["median_ns"]:
                best[name] = b
            hi[name] = max(hi.get(name, median), median)
    meta = runs[0] if runs else {}
    out = []
    for n in sorted(best):
        b = dict(best[n])
        b["max_ns"] = hi[n]
        out.append(b)
    return {
        "sha": meta.get("sha", ""),
        "timestamp": meta.get("timestamp", ""),
        "reduction": f"min-of-{len(runs)}",
        "benchmarks": out,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="aggregated bench.json files")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)
    try:
        runs = []
        for p in args.runs:
            with open(p) as f:
                runs.append(json.load(f))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    merged = reduce_min(runs)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Reduced {len(runs)} run(s) → {len(merged['benchmarks'])} benchmarks (min median)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
