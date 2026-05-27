#!/usr/bin/env python3
"""Categorize the public-API diff between two snapshots into the four lifecycle
scenarios, and emit a changelog section for a version.

Consumes `cargo public-api` snapshots (one public item per line). Matching is by
item **key** — the line truncated before its argument list — so a signature
change is recognised as the *same* item *changed*, not an unrelated add+remove:

  * **Added**     — key present in new, absent in old.
  * **Removed**   — key present in old, absent in new.
  * **Deprecated**— key in both; the new line carries a `deprecated` marker the
                    old one didn't (so a `#[deprecated]` addition is tracked
                    even though the signature is unchanged).
  * **Changed**   — key in both; the line text differs otherwise.

Used by the release tooling to maintain `api/CHANGELOG.md` across versions: at
each release the per-crate snapshots are regenerated, diffed against the prior
release's archived snapshots, and a dated/versioned section is written.

Usage:
    api-changelog.py --old <old_snapshot> --new <new_snapshot> --version X.Y.Z
        [--crate NAME] [--output api/CHANGELOG.md]
"""

import argparse
import sys


def item_key(line):
    """Identity of an API item, independent of its argument list / deprecation
    marker, so a changed signature maps to the same item."""
    s = line.strip()
    # Drop a leading deprecation marker for keying (see `is_deprecated`).
    for mark in ("#[deprecated] ", "#[deprecated]"):
        if s.startswith(mark):
            s = s[len(mark):].lstrip()
            break
    return s.split("(", 1)[0].strip()


def is_deprecated(line):
    return "deprecated" in line.lower()


def read_lines(path):
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def categorize(old_lines, new_lines):
    """→ {"added": [...], "removed": [...], "changed": [...], "deprecated": [...]}
    each a sorted list of API lines (new line for changed/deprecated/added)."""
    old = {}
    for ln in old_lines:
        old.setdefault(item_key(ln), ln)
    new = {}
    for ln in new_lines:
        new.setdefault(item_key(ln), ln)

    added, removed, changed, deprecated = [], [], [], []
    for k, ln in new.items():
        if k not in old:
            added.append(ln)
        elif old[k] != ln:
            if is_deprecated(ln) and not is_deprecated(old[k]):
                deprecated.append(ln)
            else:
                changed.append(ln)
    for k, ln in old.items():
        if k not in new:
            removed.append(ln)
    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": sorted(changed),
        "deprecated": sorted(deprecated),
    }


def render_section(cats, version, crate=None):
    scope = f" — `{crate}`" if crate else ""
    out = [f"## v{version}{scope}", ""]
    order = [
        ("Added", "added"),
        ("Changed (breaking)", "changed"),
        ("Deprecated", "deprecated"),
        ("Removed (breaking)", "removed"),
    ]
    any_change = False
    for title, key in order:
        items = cats[key]
        if not items:
            continue
        any_change = True
        out.append(f"### {title}")
        out += [f"- `{ln}`" for ln in items]
        out.append("")
    if not any_change:
        out += ["_No public-API changes._", ""]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--crate")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    try:
        cats = categorize(read_lines(args.old), read_lines(args.new))
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    section = render_section(cats, args.version, args.crate)
    print(section)
    if args.output:
        # Prepend the newest section beneath the title.
        header = "# Public API changelog\n\n"
        try:
            with open(args.output) as f:
                existing = f.read()
        except OSError:
            existing = header
        body = existing[len(header):] if existing.startswith(header) else existing
        with open(args.output, "w") as f:
            f.write(header + section + "\n" + body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
