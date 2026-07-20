#!/usr/bin/env bash
# Publish every publishable workspace crate to crates.io in dependency (topological) order —
# leaves first, the `uor-hologram` facade last. `cargo publish` (>=1.66) waits for each crate to
# land on the index before returning, so the next crate's path->version deps resolve.
#
# ⚠ crates.io versions are PERMANENT (a bad version can only be *yanked*, never deleted), and a
# partial run leaves the crates it already published live. This runs behind the `crates-io`
# environment gate in publish-crates.yml. Mark any crate `publish = false` in its Cargo.toml to skip
# it (the order is discovered from `cargo metadata`, so it drops out automatically).
#
#   CARGO_REGISTRY_TOKEN=… scripts/publish-crates.sh     # publish
#   DRY_RUN=1               scripts/publish-crates.sh     # just print the computed order
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"

order="$(cargo metadata --format-version 1 --no-deps | python3 -c '
import sys, json
m = json.load(sys.stdin)
# Publishable workspace members: `publish` is null (any registry) or a non-empty list; [] = publish=false.
members = {p["name"]: p for p in m["packages"] if p.get("publish") != []}
names = set(members)
deps = {n: {d["name"] for d in members[n].get("dependencies", []) if d["name"] in names} for n in names}
out, seen = [], set()
def visit(n):
    if n in seen: return
    seen.add(n)
    for d in sorted(deps[n]): visit(d)
    out.append(n)
for n in sorted(names): visit(n)
print(" ".join(out))
')"
echo "Publish order (${#order} chars): $order"
[ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN — not publishing."; exit 0; }

if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "CARGO_REGISTRY_TOKEN not set — skipping crates.io publish (nothing published)."
  exit 0
fi

for crate in $order; do
  echo "── cargo publish -p ${crate} ──"
  cargo publish -p "${crate}" \
    || { echo "publish FAILED at ${crate} — any crates published above are already LIVE (permanent)."; exit 1; }
done
echo "All crates published to crates.io."
