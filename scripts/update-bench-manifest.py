#!/usr/bin/env python3
"""Register the current benchmarks in `benches/manifest.toml`.

The manifest is the committed registry of every tracked benchmark — the
"benchmark surface snapshot", analogous to the API snapshot. The lifecycle gate
(`compare-benchmarks.py --manifest`) uses it to ensure no benchmark silently
appears or disappears: a new benchmark must be registered, and an `active` one
can only fall off after being marked `deprecated` (and versioned out).

This tool ADDS newly-seen benchmarks as `status = "active"` and PRESERVES every
existing entry (including `deprecated` ones) — it never deletes. Removing a
benchmark is a deliberate manual edit (delete its entry) that the lifecycle gate
permits only once the entry is `deprecated`.

Usage:
    update-bench-manifest.py <bench.json> [--manifest benches/manifest.toml]
        [--check]      # exit 1 if any running benchmark is unregistered

The deprecation transition is a manual edit:
    [benchmarks."old/bench"]
    status = "deprecated"
    deprecated_since = "0.6.0"
"""

import argparse
import json
import sys


def parse(path):
    import tomllib

    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get("benchmarks", {})
    except FileNotFoundError:
        return {}


def render(benchmarks):
    out = [
        "# Benchmark registry — the committed snapshot of every tracked benchmark.",
        "# Managed by scripts/update-bench-manifest.py; the lifecycle gate",
        "# (scripts/compare-benchmarks.py) enforces that benchmarks only appear",
        "# once registered and only fall off once deprecated + versioned out.",
        "",
    ]
    for name in sorted(benchmarks):
        entry = benchmarks[name]
        out.append(f'[benchmarks."{name}"]')
        out.append(f'status = "{entry.get("status", "active")}"')
        if "deprecated_since" in entry:
            out.append(f'deprecated_since = "{entry["deprecated_since"]}"')
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_json")
    ap.add_argument("--manifest", default="benches/manifest.toml")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    try:
        with open(args.bench_json) as f:
            running = {b["name"] for b in json.load(f).get("benchmarks", []) if b.get("name")}
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    manifest = parse(args.manifest)
    unregistered = sorted(running - set(manifest))

    if args.check:
        if unregistered:
            print("Unregistered benchmarks (run scripts/update-bench-manifest.py):", file=sys.stderr)
            for n in unregistered:
                print(f"  - {n}", file=sys.stderr)
            return 1
        print("All running benchmarks are registered.")
        return 0

    for n in unregistered:
        manifest[n] = {"status": "active"}
    import os

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w") as f:
        f.write(render(manifest))
    print(f"Registered {len(unregistered)} new benchmark(s); {len(manifest)} total in {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
