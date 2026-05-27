#!/usr/bin/env python3
"""Gate a PR on benchmark regressions vs a baseline.

Compares two aggregated benchmark JSON files (see `aggregate-benchmarks.py`)
by per-benchmark **median** and fails (exit 1) if any benchmark regressed
beyond the threshold. A regression must clear *two* bars to count, so CI
runner noise doesn't fail honest PRs:

  1. Relative: pr_median > base_median * (1 + threshold)
  2. Noise:    the slowdown exceeds `noise-sigmas` × the combined std-dev
              (sqrt(base_std² + pr_std²)) — i.e. it's outside the measured
              jitter, not just above the percentage line.

Benchmarks only in the PR (new) are reported but never gate; benchmarks only
in the baseline (removed/renamed) are reported as warnings, not failures.

Usage:
    compare-benchmarks.py <base.json> <pr.json> [--threshold 0.10]
        [--noise-sigmas 2.0] [--output summary.md]

Exit code: 0 = no regression, 1 = at least one regression, 2 = usage/IO error.
"""

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        data = json.load(f)
    by_name = {}
    for b in data.get("benchmarks", []):
        name = b.get("name")
        median = b.get("median_ns")
        if name is None or median is None:
            continue
        by_name[name] = b
    return data, by_name


def fmt_ns(ns):
    if ns is None:
        return "—"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3f} {unit}"
    return f"{ns:.1f} ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="baseline (target branch) bench.json")
    ap.add_argument("pr", help="PR bench.json")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="max tolerated relative median slowdown (0.10 = 10%%)",
    )
    ap.add_argument(
        "--noise-sigmas",
        type=float,
        default=2.0,
        help="slowdown must also exceed this many combined std-devs to count",
    )
    ap.add_argument("--output", help="also write the markdown report here")
    args = ap.parse_args()

    try:
        base_meta, base = load(args.base)
        pr_meta, pr = load(args.pr)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    regressions = []
    rows = []
    new_benches = []

    for name in sorted(pr):
        p = pr[name]
        pr_med = p["median_ns"]
        if name not in base:
            new_benches.append(name)
            continue
        base_med = base[name]["median_ns"]
        if not base_med or base_med <= 0:
            continue
        ratio = pr_med / base_med
        delta_pct = (ratio - 1.0) * 100.0

        base_std = base[name].get("std_dev_ns") or 0.0
        pr_std = p.get("std_dev_ns") or 0.0
        noise = (base_std**2 + pr_std**2) ** 0.5
        slowdown_ns = pr_med - base_med

        over_threshold = ratio > (1.0 + args.threshold)
        over_noise = slowdown_ns > args.noise_sigmas * noise
        regressed = over_threshold and over_noise

        if regressed:
            status = "❌ REGRESSED"
            regressions.append((name, delta_pct))
        elif delta_pct <= -args.threshold * 100.0:
            status = "🚀 faster"
        elif over_threshold:
            status = "⚠️ noisy"  # above % line but within jitter — not gated
        else:
            status = "✅"
        rows.append((name, base_med, pr_med, delta_pct, status))

    removed = sorted(set(base) - set(pr))

    # Build markdown report.
    out = []
    out.append("## Performance gate")
    out.append("")
    out.append(
        f"Baseline `{(base_meta.get('sha') or '?')[:12]}` vs "
        f"PR `{(pr_meta.get('sha') or '?')[:12]}` — "
        f"threshold {args.threshold * 100:.0f}% median slowdown, "
        f"outside {args.noise_sigmas:g}σ noise."
    )
    out.append("")
    out.append("| Benchmark | Baseline | PR | Δ | Status |")
    out.append("|---|--:|--:|--:|---|")
    for name, b, p, d, status in sorted(rows, key=lambda r: -r[3]):
        out.append(f"| `{name}` | {fmt_ns(b)} | {fmt_ns(p)} | {d:+.1f}% | {status} |")
    if new_benches:
        out.append("")
        out.append(f"_New benchmarks (no baseline, not gated): {', '.join('`'+n+'`' for n in new_benches)}._")
    if removed:
        out.append("")
        out.append(f"_⚠️ Benchmarks absent from PR (renamed/removed): {', '.join('`'+n+'`' for n in removed)}._")
    out.append("")
    if regressions:
        out.append(f"### ❌ {len(regressions)} regression(s) — merge blocked")
        for name, d in sorted(regressions, key=lambda r: -r[1]):
            out.append(f"- `{name}`: {d:+.1f}%")
    else:
        out.append("### ✅ No performance regressions")
    report = "\n".join(out) + "\n"

    print(report)
    if args.output:
        with open(args.output, "a") as f:
            f.write(report)

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
