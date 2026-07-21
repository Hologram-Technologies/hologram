#!/usr/bin/env python3
"""Gate a PR on benchmark regressions + benchmark-lifecycle violations.

Compares two aggregated benchmark JSON files (see `aggregate-benchmarks.py`)
by per-benchmark **median** and fails (exit 1) on either:

* a **regression** — a benchmark got slower past the gate, or
* a **lifecycle violation** — the set of benchmarks changed in a way the
  manifest (`benches/manifest.toml`) doesn't sanction.

Regression — must clear *two* bars so CI runner noise doesn't fail honest PRs:

  1. Relative: pr_median > base_median * (1 + threshold).
  2. Noise:    the slowdown exceeds `noise-sigmas` × an effective std-dev of
               max(combined measured std, cv-floor × base_median). The cv-floor
               models cross-run machine variance on shared CI runners, which a
               single run's (tight) std-dev badly underestimates.

  Pair this with best-of-N min reduction (see `bench-reduce-min.py`) upstream to
  remove transient contention spikes; the cv-floor catches the residual.

Lifecycle (when --manifest is given) — a benchmark may only fall off after it
has been *deprecated and versioned out*, never in one step:

  * A benchmark present in the run but absent from the manifest → FAIL
    (register it: `status = "active"`).
  * A manifest-`active` benchmark not present in the run → FAIL (you removed or
    broke an active benchmark; deprecate it first).
  * A benchmark that is `active` in the baseline manifest but gone from the PR
    manifest → FAIL (removed without first being deprecated). Allowed only if it
    was `deprecated` in the baseline.
  * `deprecated` benchmarks are excluded from regression gating and may be
    absent.
  * `reference` benchmarks — comparison baselines (an unfused / computed /
    uncached variant kept only to contrast against the fast path, never shipped)
    — are excluded from the regression gate but remain tracked. Their scalar
    timings are measurement-unstable, so a swing there is reported, not
    release-blocking.

Usage:
    compare-benchmarks.py <base.json> <pr.json> [--threshold 0.10]
        [--noise-sigmas 2.0] [--cv-floor 0.07]
        [--manifest benches/manifest.toml] [--baseline-manifest …]
        [--output summary.md]

Exit code: 0 = clean, 1 = regression or lifecycle violation, 2 = usage/IO error.
"""

import argparse
import json
import sys


# ── pure, testable core ────────────────────────────────────────────────────


def index(data):
    """{name: bench-dict} from an aggregated bench.json dict."""
    out = {}
    for b in data.get("benchmarks", []):
        name, median = b.get("name"), b.get("median_ns")
        if name is not None and median is not None:
            out[name] = b
    return out


def load_manifest(path):
    """Parse a benchmark manifest TOML → {name: {"status", "deprecated_since"}}."""
    import tomllib

    with open(path, "rb") as f:
        doc = tomllib.load(f)
    return doc.get("benchmarks", {})


def deprecated_names(manifest):
    return {n for n, m in manifest.items() if m.get("status") == "deprecated"}


def reference_names(manifest):
    """Comparison-baseline benchmarks (`status = "reference"`): deliberately-slow
    paths kept only to contrast against the fast path — never shipped. Measured
    and reported, but excluded from the regression gate (see module docstring)."""
    return {n for n, m in manifest.items() if m.get("status") == "reference"}


def active_names(manifest):
    # Absent status defaults to active, so a bare entry still counts as tracked.
    return {n for n, m in manifest.items() if m.get("status", "active") == "active"}


def classify(base, pr, *, threshold, noise_sigmas, cv_floor, skip=frozenset(),
             reference=frozenset()):
    """Per-benchmark regression classification. Returns (rows, regressions).

    `rows` is [(name, base_med, pr_med, delta_pct, status)]; `regressions` is
    the subset that gates [(name, delta_pct)]. `skip` names are compared for the
    report but never gate (deprecated or reference benchmarks); `reference` is
    the subset of `skip` labeled as comparison baselines rather than deprecated."""
    rows, regressions, new = [], [], []
    for name in sorted(pr):
        pr_med = pr[name]["median_ns"]
        if name not in base:
            new.append(name)
            continue
        base_med = base[name]["median_ns"]
        if not base_med or base_med <= 0:
            continue
        ratio = pr_med / base_med
        delta_pct = (ratio - 1.0) * 100.0
        combined_std = (
            (base[name].get("std_dev_ns") or 0.0) ** 2
            + (pr[name].get("std_dev_ns") or 0.0) ** 2
        ) ** 0.5
        # Effective noise floors the measured jitter at cv_floor of the baseline
        # so shared-runner variance doesn't read as a regression.
        eff_noise = max(combined_std, cv_floor * base_med)
        slowdown = pr_med - base_med

        over_threshold = ratio > (1.0 + threshold)
        over_noise = slowdown > noise_sigmas * eff_noise
        if name in skip:
            status = "⏸ reference" if name in reference else "⏸ deprecated"
        elif over_threshold and over_noise:
            status = "❌ REGRESSED"
            regressions.append((name, delta_pct))
        elif delta_pct <= -threshold * 100.0:
            status = "🚀 faster"
        elif over_threshold:
            status = "⚠️ noisy"  # above % line but within jitter — not gated
        else:
            status = "✅"
        rows.append((name, base_med, pr_med, delta_pct, status))
    return rows, regressions, new


def lifecycle_violations(base, pr, manifest, baseline_manifest):
    """Manifest/lifecycle errors as a list of human-readable strings."""
    if manifest is None:
        return []
    errs = []
    pr_names = set(pr)
    active = active_names(manifest)
    deprecated = deprecated_names(manifest)
    tracked = set(manifest)

    for name in sorted(pr_names - tracked):
        errs.append(
            f"benchmark `{name}` is not in the manifest — register it "
            f"(`scripts/update-bench-manifest.py`)"
        )
    for name in sorted(active - pr_names):
        errs.append(
            f"manifest lists `{name}` as active but it did not run — deprecate "
            f"it before removing"
        )
    if baseline_manifest is not None:
        base_active = active_names(baseline_manifest)
        for name in sorted(base_active - tracked):
            errs.append(
                f"benchmark `{name}` was active and was removed from the "
                f"manifest without first being deprecated"
            )
    # `deprecated` is referenced for symmetry/readability; deprecated entries are
    # intentionally unconstrained here (they may stay or be versioned out).
    _ = deprecated
    return errs


# ── rendering + CLI ─────────────────────────────────────────────────────────


def fmt_ns(ns):
    if ns is None:
        return "—"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3f} {unit}"
    return f"{ns:.1f} ns"


def render(base_meta, pr_meta, rows, regressions, new, removed, lifecycle, args):
    out = ["## Performance gate", ""]
    out.append(
        f"Baseline `{(base_meta.get('sha') or '?')[:12]}` vs "
        f"PR `{(pr_meta.get('sha') or '?')[:12]}` — threshold "
        f"{args.threshold * 100:.0f}% median slowdown, outside "
        f"{args.noise_sigmas:g}σ of max(measured, {args.cv_floor * 100:.0f}% cv) noise."
    )
    out += ["", "| Benchmark | Baseline | PR | Δ | Status |", "|---|--:|--:|--:|---|"]
    for name, b, p, d, status in sorted(rows, key=lambda r: -r[3]):
        out.append(f"| `{name}` | {fmt_ns(b)} | {fmt_ns(p)} | {d:+.1f}% | {status} |")
    if new:
        out += ["", f"_New benchmarks (no baseline, not gated): {', '.join('`'+n+'`' for n in new)}._"]
    if removed:
        out += ["", f"_Benchmarks absent from PR: {', '.join('`'+n+'`' for n in removed)}._"]
    out.append("")
    if lifecycle:
        out.append(f"### ❌ {len(lifecycle)} benchmark-lifecycle violation(s)")
        out += [f"- {e}" for e in lifecycle]
        out.append("")
    if regressions:
        out.append(f"### ❌ {len(regressions)} regression(s) — merge blocked")
        out += [f"- `{n}`: {d:+.1f}%" for n, d in sorted(regressions, key=lambda r: -r[1])]
    elif not lifecycle:
        out.append("### ✅ No performance regressions")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("pr")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--noise-sigmas", type=float, default=2.0)
    ap.add_argument("--cv-floor", type=float, default=0.07,
                    help="floor for run-to-run noise as a fraction of the baseline median")
    ap.add_argument("--manifest", help="PR benches/manifest.toml (enables lifecycle gate)")
    ap.add_argument("--baseline-manifest", help="target-branch benches/manifest.toml")
    ap.add_argument("--output")
    args = ap.parse_args(argv)

    try:
        with open(args.base) as f:
            base_data = json.load(f)
        with open(args.pr) as f:
            pr_data = json.load(f)
        manifest = load_manifest(args.manifest) if args.manifest else None
        baseline_manifest = (
            load_manifest(args.baseline_manifest) if args.baseline_manifest else None
        )
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    base, pr = index(base_data), index(pr_data)
    # Deprecated and reference benchmarks are measured/reported but never gate.
    reference = reference_names(manifest) if manifest else frozenset()
    skip = (deprecated_names(manifest) | reference) if manifest else frozenset()
    rows, regressions, new = classify(
        base, pr, threshold=args.threshold, noise_sigmas=args.noise_sigmas,
        cv_floor=args.cv_floor, skip=skip, reference=reference,
    )
    removed = sorted(set(base) - set(pr))
    lifecycle = lifecycle_violations(base, pr, manifest, baseline_manifest)

    report = render(
        base_data, pr_data, rows, regressions, new, removed, lifecycle, args
    )
    print(report)
    if args.output:
        with open(args.output, "a") as f:
            f.write(report)
    return 1 if (regressions or lifecycle) else 0


if __name__ == "__main__":
    sys.exit(main())
