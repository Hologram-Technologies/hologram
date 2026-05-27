#!/usr/bin/env python3
"""Write a human-readable markdown summary of hologram benchmark results.

Usage:
    python3 scripts/summarize-benchmarks.py <bench.json> [--output <path>]

Arguments:
    bench.json   Output of aggregate-benchmarks.py
    --output     Write markdown to this path (default: $GITHUB_STEP_SUMMARY)

If neither --output nor $GITHUB_STEP_SUMMARY is set, writes to stdout.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def fmt_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns / 1_000_000:.2f} ms"


# ---------------------------------------------------------------------------
# Benchmark name parsing
# ---------------------------------------------------------------------------

def parse_name(name: str) -> tuple[str, str]:
    """Split 'suite/rest' into (suite, rest). Single-segment names use '' as suite."""
    parts = name.split("/", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])


def suite_label(suite: str) -> str:
    return suite if suite else "(ungrouped)"


# ---------------------------------------------------------------------------
# Comparison detection
# ---------------------------------------------------------------------------

# Each entry: (label_template, native_pattern, hologram_filter)
#   native_pattern: callable(variant_name) -> bool  — marks the baseline
#   hologram_filter: callable(variant_name) -> bool — marks hologram entries
#   label_template: callable(suite, native_variant, holo_variant) -> str

_NATIVE_PATTERNS = [
    # exact variant names
    lambda v: v in {"f64", "native", "naive"},
    # contains these substrings
    lambda v: "naive_matmul" in v,
    lambda v: v.startswith("sync_"),
]

def is_native(variant: str) -> bool:
    return any(p(variant) for p in _NATIVE_PATTERNS)


def is_hologram(variant: str) -> bool:
    return not is_native(variant)


def comparison_label(suite: str, native_v: str, holo_v: str) -> str:
    # strip common bench_ prefix for readability
    def clean(s: str) -> str:
        return s.removeprefix("bench_")

    return f"{clean(suite)}: {clean(holo_v)} vs {clean(native_v)}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_comparison_table(benchmarks: list[dict]) -> list[dict]:
    """Return rows for the hologram-vs-native comparison table."""
    # Group by (suite, parent_group) where parent_group is everything before
    # the final segment. This handles both:
    #   lut_gemm/bench_lut_gemm_q4_64x64  → suite=lut_gemm, variant=bench_lut_gemm_q4_64x64
    #   q1/bench_q1_vs_q0_vs_f64_sigmoid/f64  → suite=q1, rest=bench_.../f64
    #     parent = q1/bench_q1_vs_q0_vs_f64_sigmoid, variant = f64

    groups: dict[str, dict[str, dict]] = defaultdict(dict)

    for b in benchmarks:
        name = b["name"]
        parts = name.rsplit("/", 1)
        if len(parts) == 2:
            group, variant = parts
        else:
            group, variant = "", parts[0]
        groups[group][variant] = b

    rows = []
    for group, variants in sorted(groups.items()):
        native_vs = {v: b for v, b in variants.items() if is_native(v)}
        holo_vs = {v: b for v, b in variants.items() if is_hologram(v)}

        if not native_vs or not holo_vs:
            continue

        # Pair each hologram variant with the best-matching native baseline.
        # If there's only one native baseline, pair all hologram entries with it.
        native_items = list(native_vs.items())

        for hv, hb in sorted(holo_vs.items()):
            # pick closest native by name similarity (longest common prefix)
            nv, nb = max(
                native_items,
                key=lambda kv: len(os.path.commonprefix([hv, kv[0]])),
            )

            h_mean = hb.get("mean_ns")
            n_mean = nb.get("mean_ns")
            if h_mean is None or n_mean is None or h_mean == 0:
                continue

            speedup = n_mean / h_mean
            suite = group.split("/")[0] if group else ""
            rows.append(
                {
                    "label": comparison_label(suite, nv, hv),
                    "holo_ns": h_mean,
                    "native_ns": n_mean,
                    "speedup": speedup,
                }
            )

    return rows


def render_speedup(speedup: float) -> str:
    if speedup >= 1.05:
        return f"**{speedup:.1f}× faster**"
    if speedup <= 0.95:
        return f"{1 / speedup:.1f}× slower"
    return "≈ same"


def build_suite_sections(benchmarks: list[dict]) -> dict[str, list[dict]]:
    suites: dict[str, list[dict]] = defaultdict(list)
    for b in benchmarks:
        suite, _ = parse_name(b["name"])
        suites[suite_label(suite)].append(b)
    return dict(sorted(suites.items()))


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render(bench_json: dict) -> str:
    sha = bench_json.get("sha", "unknown")
    ts = bench_json.get("timestamp", "")
    benchmarks = bench_json.get("benchmarks", [])

    lines: list[str] = []
    lines.append(f"## Benchmark Results — `{sha[:7]}`")
    if ts:
        lines.append(f"*{ts}*")
    lines.append("")

    # --- Comparison table ---
    comparisons = build_comparison_table(benchmarks)
    if comparisons:
        lines.append("### Hologram vs Native")
        lines.append("")
        lines.append("| Comparison | Hologram | Native | Speedup |")
        lines.append("|---|---|---|---|")
        for row in comparisons:
            lines.append(
                f"| {row['label']} | {fmt_ns(row['holo_ns'])} "
                f"| {fmt_ns(row['native_ns'])} | {render_speedup(row['speedup'])} |"
            )
        lines.append("")
    else:
        lines.append("*No hologram-vs-native comparisons detected.*")
        lines.append("")

    # --- Per-suite detail sections ---
    lines.append("### All Results")
    lines.append("")
    suites = build_suite_sections(benchmarks)
    for suite_name, entries in suites.items():
        lines.append(f"<details><summary>{suite_name} ({len(entries)} benchmarks)</summary>")
        lines.append("")
        lines.append("| Benchmark | Mean | Median | ±Std |")
        lines.append("|---|---|---|---|")
        for b in entries:
            _, variant = parse_name(b["name"])
            mean = fmt_ns(b["mean_ns"]) if b.get("mean_ns") is not None else "—"
            median = fmt_ns(b["median_ns"]) if b.get("median_ns") is not None else "—"
            std = fmt_ns(b["std_dev_ns"]) if b.get("std_dev_ns") is not None else "—"
            lines.append(f"| `{variant}` | {mean} | {median} | ±{std} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_json", help="Path to bench.json")
    parser.add_argument("--output", help="Output path (default: $GITHUB_STEP_SUMMARY or stdout)")
    args = parser.parse_args()

    with open(args.bench_json) as f:
        data = json.load(f)

    md = render(data)

    output_path = args.output or os.environ.get("GITHUB_STEP_SUMMARY")
    if output_path:
        with open(output_path, "a") as f:
            f.write(md)
            f.write("\n")
    else:
        sys.stdout.write(md)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
