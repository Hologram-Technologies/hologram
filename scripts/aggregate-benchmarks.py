#!/usr/bin/env python3
"""Aggregate Criterion benchmark results into a single JSON file.

Usage:
    python3 scripts/aggregate-benchmarks.py <criterion_dir> <output.json> <sha>

Arguments:
    criterion_dir  Path to target/criterion/ directory
    output.json    Output file path
    sha            Git commit SHA to embed in output
"""

import json
import os
import sys
from datetime import datetime, timezone


def find_benchmark_pairs(criterion_dir):
    """Walk criterion_dir yielding (benchmark_json_path, estimates_json_path) pairs."""
    for root, dirs, files in os.walk(criterion_dir):
        if os.path.basename(root) == "new" and "estimates.json" in files and "benchmark.json" in files:
            yield (
                os.path.join(root, "benchmark.json"),
                os.path.join(root, "estimates.json"),
            )


def parse_benchmark(benchmark_path, estimates_path):
    with open(benchmark_path) as f:
        bdata = json.load(f)
    with open(estimates_path) as f:
        edata = json.load(f)

    name = bdata.get("full_id") or bdata.get("title") or bdata.get("group_id", "unknown")

    mean_ns = edata.get("mean", {}).get("point_estimate")
    median_ns = edata.get("median", {}).get("point_estimate")
    std_dev_ns = edata.get("std_dev", {}).get("point_estimate")

    return {
        "name": name,
        "mean_ns": round(mean_ns, 3) if mean_ns is not None else None,
        "median_ns": round(median_ns, 3) if median_ns is not None else None,
        "std_dev_ns": round(std_dev_ns, 3) if std_dev_ns is not None else None,
    }


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <criterion_dir> <output.json> <sha>", file=sys.stderr)
        sys.exit(1)

    criterion_dir, output_path, sha = sys.argv[1], sys.argv[2], sys.argv[3]

    if not os.path.isdir(criterion_dir):
        print(f"ERROR: {criterion_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    benchmarks = []
    for bpath, epath in sorted(find_benchmark_pairs(criterion_dir)):
        try:
            benchmarks.append(parse_benchmark(bpath, epath))
        except Exception as e:
            print(f"WARNING: skipping {bpath}: {e}", file=sys.stderr)

    if not benchmarks:
        print("ERROR: no benchmark results found", file=sys.stderr)
        sys.exit(1)

    result = {
        "sha": sha,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmarks": benchmarks,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(benchmarks)} benchmarks to {output_path}")


if __name__ == "__main__":
    main()
