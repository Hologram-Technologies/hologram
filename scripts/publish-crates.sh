#!/usr/bin/env bash
# Publish every publishable workspace crate to crates.io in dependency (topological) order —
# leaves first, the `uor-hologram` facade last. After each upload, this script independently polls
# crates.io for that exact crate/version before allowing any dependent crate to publish.
#
# ⚠ crates.io versions are PERMANENT (a bad version can only be *yanked*, never deleted), and a
# partial run leaves the crates it already published live. This runs behind the `crates-io`
# environment gate in publish-crates.yml. Mark any crate `publish = false` in its Cargo.toml to skip
# it (the order is discovered from `cargo metadata`, so it drops out automatically).
#
#   CARGO_REGISTRY_TOKEN=… scripts/publish-crates.sh     # publish
#   DRY_RUN=1               scripts/publish-crates.sh     # package + verify without publishing
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
version="$(scripts/workspace-version.sh)"
echo "Publish order (${#order} chars): $order"

if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "CARGO_REGISTRY_TOKEN not set — refusing to publish." >&2
  exit 1
fi

# Package the COMPLETE graph and compare every already-public version before the first permanent
# upload. A retry therefore skips byte-identical packages, but refuses a same-version/different-byte
# collision. This cannot make independent registry writes transactional; it does make interruption
# recoverable and ensures no package is uploaded before the whole local graph is packageable.
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
declare -A local_checksum remote_checksum
package_args=(--locked --no-verify)
[ "${ALLOW_DIRTY:-0}" = "1" ] && package_args+=(--allow-dirty)
package_selectors=()
for crate in $order; do
  package_selectors+=(-p "$crate")
done

registry_checksum() {
  local crate="$1" out="$2" code
  code="$(curl --silent --show-error --output "$out" --write-out '%{http_code}' \
    --connect-timeout 10 --max-time 30 --retry 2 \
    --user-agent 'hologram-release/0.13 (+https://github.com/Hologram-Technologies/hologram)' \
    "https://crates.io/api/v1/crates/${crate}/${version}")"
  case "$code" in
    200) python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"]["checksum"])' "$out" ;;
    404) printf '%s\n' MISSING ;;
    *) echo "crates.io preflight failed for ${crate}@${version}: HTTP ${code}" >&2; return 1 ;;
  esac
}

# Cargo's multi-package mode rewrites workspace dependency edges as one coherent release set. A
# series of single-package calls cannot package a dependent until its leaf is already on crates.io,
# which would defeat the all-before-first-write preflight.
echo "── package complete workspace release graph ──"
CARGO_TARGET_DIR="$stage/target" cargo package "${package_selectors[@]}" "${package_args[@]}"

for crate in $order; do
  echo "── checksum preflight ${crate}@${version} ──"
  artifact="$stage/target/package/${crate}-${version}.crate"
  [ -f "$artifact" ] || { echo "missing packaged artifact: $artifact" >&2; exit 1; }
  local_checksum["$crate"]="$(sha256sum "$artifact" | cut -d' ' -f1)"
  remote_checksum["$crate"]="$(registry_checksum "$crate" "$stage/${crate}.json")"
  if [ "${remote_checksum[$crate]}" != MISSING ] \
    && [ "${remote_checksum[$crate]}" != "${local_checksum[$crate]}" ]; then
    echo "refusing ${crate}@${version}: public checksum ${remote_checksum[$crate]} differs from packaged ${local_checksum[$crate]}" >&2
    exit 1
  fi
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN — complete package/checksum preflight passed; not publishing."
  exit 0
fi

# Validate that the secret is a live crates.io API credential before the first upload. crates.io
# does not expose a non-mutating endpoint for the token's publish-new scope, so the required account
# permission remains an environment/owner control; a denied first upload still creates no crate.
umask 077
printf 'header = "Authorization: %s"\n' "$CARGO_REGISTRY_TOKEN" > "$stage/curl-auth.conf"
auth_code="$(curl --config "$stage/curl-auth.conf" \
  --silent --show-error --output "$stage/me.json" --write-out '%{http_code}' \
  --connect-timeout 10 --max-time 30 --retry 2 \
  --user-agent 'hologram-release/0.13 (+https://github.com/Hologram-Technologies/hologram)' \
  https://crates.io/api/v1/me)"
if [ "$auth_code" != 200 ]; then
  echo "crates.io credential preflight failed: HTTP ${auth_code}" >&2
  exit 1
fi

for crate in $order; do
  if [ "${remote_checksum[$crate]}" = "${local_checksum[$crate]}" ]; then
    echo "── ${crate}@${version} already public with accepted checksum; skipping ──"
    continue
  fi
  echo "── cargo publish -p ${crate} ──"
  if ! cargo publish --locked -p "${crate}"; then
    # A concurrent/retried publisher may have won the race after preflight. Accept only identical
    # immutable bytes; every other failure remains fatal.
    observed="$(registry_checksum "$crate" "$stage/${crate}-after-failure.json")" || exit 1
    if [ "$observed" = "${local_checksum[$crate]}" ]; then
      echo "${crate}@${version} became public concurrently with the accepted checksum; continuing."
      continue
    fi
    echo "publish FAILED at ${crate}; observed checksum=${observed}, expected=${local_checksum[$crate]}" >&2
    exit 1
  fi

  # crates.io accepts the upload before every index/cache edge necessarily serves it. Do not begin
  # a dependent package until the just-published exact version is downloadable with the exact
  # preflight checksum.
  available=0
  for attempt in $(seq 1 30); do
    observed="$(registry_checksum "$crate" "$stage/${crate}-propagation.json")" || exit 1
    if [ "$observed" = "${local_checksum[$crate]}" ] \
      && cargo info "${crate}@${version}" --registry crates-io >/dev/null 2>&1; then
      available=1
      break
    fi
    if [ "$observed" != MISSING ] && [ "$observed" != "${local_checksum[$crate]}" ]; then
      echo "publish FAILED: ${crate}@${version} propagated with checksum ${observed}, expected ${local_checksum[$crate]}" >&2
      exit 1
    fi
    echo "waiting for ${crate}@${version} registry propagation (${attempt}/30)"
    sleep 10
  done
  if [ "$available" -ne 1 ]; then
    echo "publish FAILED: ${crate}@${version} did not become downloadable within 300 seconds." >&2
    echo "Any crates published above are already LIVE (permanent); refusing dependent publishes." >&2
    exit 1
  fi
done
echo "All crates published to crates.io."
