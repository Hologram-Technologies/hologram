#!/usr/bin/env python3
"""Tests for the quality-gate tooling (regression compare, best-of-N reduction,
benchmark lifecycle, API changelog categorization).

Pure-Python, no external deps — run with `python3 scripts/test_quality_gates.py`
(exit 0 = all pass) or under pytest. The Rust-tool gates (cargo-semver-checks,
cargo-public-api) are validated by running those tools in CI, not here.
"""

import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cmp = _load("compare_benchmarks", "compare-benchmarks.py")
red = _load("bench_reduce_min", "bench-reduce-min.py")
chg = _load("api_changelog", "api-changelog.py")


def bench(name, median, std=0.0):
    return {"name": name, "median_ns": median, "std_dev_ns": std}


# ── regression classification ───────────────────────────────────────────────


def test_real_regression_gates():
    base = {"m": bench("m", 1_000_000, 10_000)}
    pr = {"m": bench("m", 1_300_000, 10_000)}
    rows, regs, new = cmp.classify(base, pr, threshold=0.10, noise_sigmas=2.0, cv_floor=0.07)
    assert [n for n, _ in regs] == ["m"], regs
    assert abs(regs[0][1] - 30.0) < 1e-6, regs


def test_small_slowdown_within_threshold_passes():
    base = {"m": bench("m", 1_000_000, 1000)}
    pr = {"m": bench("m", 1_050_000, 1000)}  # +5%, under 10%
    _, regs, _ = cmp.classify(base, pr, threshold=0.10, noise_sigmas=2.0, cv_floor=0.07)
    assert regs == []


def test_cv_floor_absorbs_cross_run_variance():
    # +13% with tiny measured std would gate under a std-only model, but the
    # cv-floor (7%) widens the noise band enough (2σ ⇒ ~14%) to suppress it —
    # exactly the false positive seen on the shared CI runner.
    base = {"m": bench("m", 17_800_000, 5_000)}
    pr = {"m": bench("m", 20_200_000, 5_000)}  # +13.5%
    _, regs, _ = cmp.classify(base, pr, threshold=0.10, noise_sigmas=2.0, cv_floor=0.07)
    assert regs == [], regs
    # A genuine, larger regression still gates despite the cv-floor.
    pr_big = {"m": bench("m", 24_000_000, 5_000)}  # +34.8%
    _, regs2, _ = cmp.classify(base, pr_big, threshold=0.10, noise_sigmas=2.0, cv_floor=0.07)
    assert [n for n, _ in regs2] == ["m"]


def test_deprecated_bench_not_gated():
    base = {"m": bench("m", 1_000_000, 1000)}
    pr = {"m": bench("m", 2_000_000, 1000)}  # +100% but deprecated
    rows, regs, _ = cmp.classify(
        base, pr, threshold=0.10, noise_sigmas=2.0, cv_floor=0.07, skip={"m"}
    )
    assert regs == []
    assert any("deprecated" in r[4] for r in rows)


def test_new_benchmark_reported_not_gated():
    base = {}
    pr = {"n": bench("n", 500, 10)}
    _, regs, new = cmp.classify(base, pr, threshold=0.10, noise_sigmas=2.0, cv_floor=0.07)
    assert regs == [] and new == ["n"]


# ── benchmark lifecycle ─────────────────────────────────────────────────────


def test_unregistered_benchmark_fails():
    errs = cmp.lifecycle_violations(
        base={}, pr={"x": bench("x", 1, 0)}, manifest={}, baseline_manifest=None
    )
    assert any("not in the manifest" in e for e in errs)


def test_active_benchmark_missing_from_run_fails():
    manifest = {"x": {"status": "active"}}
    errs = cmp.lifecycle_violations(base={}, pr={}, manifest=manifest, baseline_manifest=None)
    assert any("did not run" in e for e in errs)


def test_active_removed_without_deprecation_fails():
    # `x` active in baseline manifest, gone from PR manifest → removed without
    # first being deprecated.
    base_m = {"x": {"status": "active"}}
    pr_m = {}  # entry deleted
    errs = cmp.lifecycle_violations(base={}, pr={}, manifest=pr_m, baseline_manifest=base_m)
    assert any("without first being deprecated" in e for e in errs)


def test_deprecated_then_removed_is_allowed():
    base_m = {"x": {"status": "deprecated", "deprecated_since": "0.6.0"}}
    pr_m = {}  # now versioned out
    errs = cmp.lifecycle_violations(base={}, pr={}, manifest=pr_m, baseline_manifest=base_m)
    assert errs == [], errs


def test_clean_active_set_no_violations():
    manifest = {"x": {"status": "active"}}
    errs = cmp.lifecycle_violations(
        base={"x": bench("x", 1, 0)}, pr={"x": bench("x", 1, 0)},
        manifest=manifest, baseline_manifest=manifest,
    )
    assert errs == []


# ── best-of-N reduction ─────────────────────────────────────────────────────


def test_reduce_min_takes_fastest_per_bench():
    runs = [
        {"benchmarks": [bench("a", 100), bench("b", 999)]},  # b spiked
        {"benchmarks": [bench("a", 110), bench("b", 200)]},
        {"benchmarks": [bench("a", 95), bench("b", 205)]},
    ]
    merged = red.reduce_min(runs)
    got = {b["name"]: b["median_ns"] for b in merged["benchmarks"]}
    assert got == {"a": 95, "b": 200}
    assert merged["reduction"] == "min-of-3"


# ── API changelog categorization (the four scenarios) ───────────────────────


def test_api_added():
    cats = chg.categorize([], ["pub fn k::foo()"])
    assert cats["added"] == ["pub fn k::foo()"] and not cats["removed"]


def test_api_removed():
    cats = chg.categorize(["pub fn k::foo()"], [])
    assert cats["removed"] == ["pub fn k::foo()"]


def test_api_changed_signature_is_not_add_plus_remove():
    cats = chg.categorize(["pub fn k::foo(x: u8)"], ["pub fn k::foo(x: u8, y: u8)"])
    assert cats["changed"] == ["pub fn k::foo(x: u8, y: u8)"]
    assert not cats["added"] and not cats["removed"]


def test_api_deprecated_detected_separately_from_changed():
    cats = chg.categorize(["pub fn k::foo()"], ["#[deprecated] pub fn k::foo()"])
    assert cats["deprecated"] == ["#[deprecated] pub fn k::foo()"]
    assert not cats["changed"]


def test_api_changelog_renders_all_sections():
    cats = {
        "added": ["pub fn k::a()"],
        "removed": ["pub fn k::r()"],
        "changed": ["pub fn k::c(x: u8)"],
        "deprecated": ["#[deprecated] pub fn k::d()"],
    }
    s = chg.render_section(cats, "0.6.0", crate="k")
    for tok in ("v0.6.0", "Added", "Deprecated", "Removed (breaking)", "Changed (breaking)"):
        assert tok in s, tok


def _run():
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
