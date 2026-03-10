#!/usr/bin/env python3
"""Inject benchmark JSON into the calculator demo page.

Usage:
    python3 scripts/inject-bench-data.py <bench.json> <calculator.astro>

Replaces the BENCH_DATA variable in the Astro page with fresh benchmark results.
"""

import json
import re
import sys


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <bench.json> <calculator.astro>", file=sys.stderr)
        sys.exit(1)

    bench_path, astro_path = sys.argv[1], sys.argv[2]

    with open(bench_path) as f:
        bench_data = json.load(f)

    # Compact JSON for embedding
    bench_json = json.dumps(bench_data, separators=(",", ":"))

    with open(astro_path) as f:
        content = f.read()

    # Replace the BENCH_DATA assignment
    pattern = r'var BENCH_DATA = \{.*?\};'
    replacement = f'var BENCH_DATA = {bench_json};'

    new_content, count = re.subn(pattern, replacement, content, count=1)

    if count == 0:
        print("ERROR: Could not find BENCH_DATA in the Astro file", file=sys.stderr)
        sys.exit(1)

    with open(astro_path, "w") as f:
        f.write(new_content)

    n = len(bench_data.get("benchmarks", []))
    sha = bench_data.get("sha", "unknown")[:7]
    print(f"Injected {n} benchmarks (sha: {sha}) into {astro_path}")


if __name__ == "__main__":
    main()
