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
    """runs: list of aggregated bench.json dicts → one merged dict (min median)."""
    best = {}  # name -> bench dict with the smallest median seen
    for data in runs:
        for b in data.get("benchmarks", []):
            name, median = b.get("name"), b.get("median_ns")
            if name is None or median is None:
                continue
            if name not in best or median < best[name]["median_ns"]:
                best[name] = b
    meta = runs[0] if runs else {}
    return {
        "sha": meta.get("sha", ""),
        "timestamp": meta.get("timestamp", ""),
        "reduction": f"min-of-{len(runs)}",
        "benchmarks": [best[n] for n in sorted(best)],
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
