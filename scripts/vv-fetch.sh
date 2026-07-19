#!/usr/bin/env bash
#
# scripts/vv-fetch.sh — materialize the V&V artifact tree (vv/artifacts/, ~170M) for MG-7.
#
# The artifacts (real Linux kernels riscv64/aarch64/x86-64, ext4 images, OCI layouts, the
# vscode-web executable core, the riscv/aarch64 ISA batteries) are the EXTERNAL authorities the
# holospaces CC suites boot against. They are carried EXTERNALLY — never committed here
# (.gitignore blocks vv/artifacts/) — and materialized on demand from the provenance-pinned
# holospaces revision that committed them (vv/ARTIFACTS_PIN), then verified against the recorded
# sha256 sidecars. Idempotent: a present, verified tree is left untouched.
#
# Every artifact's ultimate provenance (the reproducible mke2fs/BuildKit/kernel-build command or
# the pinned fetch URL + integrity) is recorded in vv/PROVENANCE.md and each ccN/SOURCE.txt; this
# script fetches the pinned committed bytes and checks them, rather than rebuilding kernels each run.
#
# Source, in order of preference:
#   1. $HOLOSPACES_DIR or ../holospaces — a local checkout (the transition default).
#   2. $HOLOSPACES_GIT — the holospaces git remote (CI; archived repo, still fetchable at the pin).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/vv/artifacts"
PIN="$(tr -d '[:space:]' < "$ROOT/vv/ARTIFACTS_PIN")"
SRC_LOCAL="${HOLOSPACES_DIR:-$ROOT/../holospaces}"
SRC_REMOTE="${HOLOSPACES_GIT:-https://github.com/Hologram-Technologies/holospaces.git}"

log() { printf '  %s\n' "$*" >&2; }

# Portable sha256 -c (Linux coreutils `sha256sum` / macOS `shasum -a 256`).
sha_check() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum -c --quiet "$@"
    else shasum -a 256 -c "$@" >/dev/null; fi
}

# Verify one sha256 sidecar. Sidecars use mixed path conventions: some list repo-root-relative
# paths (`vv/artifacts/cc7/rootfs.ext4`), others list paths relative to their own directory
# (`Image.gz`). Try repo-root first, then the sidecar's own dir — whichever resolves.
# Returns: 0 = OK, 1 = present-but-mismatch (corrupt), 2 = target files absent. The absent case is
# a fetched-by-pin artifact (e.g. cc17 vscode-web, intel-sdm) that is NOT git-committed and is
# materialized by its own suite/tooling — not a corruption, so callers skip it.
verify_one() {
    local s="$1"
    ( cd "$ROOT" && sha_check "${s#"$ROOT"/}" ) 2>/dev/null && return 0
    ( cd "$(dirname "$s")" && sha_check "$(basename "$s")" ) 2>/dev/null && return 0
    # Not OK — do any of the sidecar's target files exist? (strip `HASH  ` / `HASH *` prefix)
    local p
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        if [ -e "$ROOT/$p" ] || [ -e "$(dirname "$s")/$p" ]; then return 1; fi
    done < <(sed -E 's/^[0-9a-fA-F]+[ *]+//' "$s")
    return 2
}

# Verify every recorded sha256 sidecar. Non-zero on any real mismatch (corrupt/wrong-pin).
# Fetched-by-pin artifacts (absent from the committed tree) are logged and skipped, not failed.
verify() {
    [ -d "$DEST" ] || return 1
    local sidecars ok=1 s
    sidecars=$(find "$DEST" -name '*.sha256' 2>/dev/null || true)
    [ -n "$sidecars" ] || return 1
    while IFS= read -r s; do
        [ -n "$s" ] || continue
        verify_one "$s"
        case $? in
            0) : ;;
            2) log "fetched-by-pin elsewhere (not committed): ${s#"$ROOT"/} — skipping" ;;
            *) log "sha256 MISMATCH: ${s#"$ROOT"/}"; ok=0 ;;
        esac
    done <<< "$sidecars"
    [ "$ok" -eq 1 ]
}

if verify >/dev/null 2>&1; then
    log "vv/artifacts present and verified (pin ${PIN:0:12}) — nothing to do."
    exit 0
fi

log "materializing vv/artifacts from holospaces@${PIN:0:12} …"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -d "$SRC_LOCAL/.git" ]; then
    log "source: local checkout $SRC_LOCAL"
    git -C "$SRC_LOCAL" archive "$PIN" vv/artifacts | tar -x -C "$tmp"
else
    log "source: remote $SRC_REMOTE"
    git clone --quiet --no-checkout --filter=blob:none "$SRC_REMOTE" "$tmp/hs"
    git -C "$tmp/hs" archive "$PIN" vv/artifacts | tar -x -C "$tmp"
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$tmp/vv/artifacts" "$DEST"

if verify; then
    n=$(find "$DEST" -name '*.sha256' | wc -l | tr -d ' ')
    log "OK — vv/artifacts materialized + verified against $n sha256 sidecars."
else
    log "FAILED — sha256 mismatch after fetch; the artifact tree may be corrupt or the pin is wrong."
    exit 1
fi
