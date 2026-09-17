#!/usr/bin/env bash
# CC-45 DOGFOOD — holospaces builds in its OWN, unmodified devcontainer.
#
# The #13 reference, end to end and for real: this repo's `.devcontainer` (Ubuntu
# 24.04 + the unmodified toolchain + its ghcr features) is built by the **Dev
# Container CLI** exactly as Codespaces/Gitpod would, its rootfs is exported, and the
# `cc45_dogfood` witness boots it on the holospaces **x86-64 core** and has the real
# **gcc 13.3** (cc1 → as → ld over glibc 2.39 + ld.so) compile a C program in-guest —
# then runs the binary it built (DOGFOOD-GCC-BUILT:42).
#
# The rootfs is multi-GiB, so it is NOT a committed fixture — it is built here. Needs
# docker + node + network for the base/features, and a 16 GiB-class release runner.
# The rootfs layers and sparse disk are consumed one at a time, keeping the full
# Ubuntu+gcc boot bounded below that runner class instead of retaining duplicates.
#
# Status: HEAVY, REQUIRED — `vv/suites/cc45-x64-devcontainer.sh` invokes this
#   witness as part of the release V&V closure. It is a REAL, reproducible
#   validation (no fixture, no stub, no skip); it can also be run directly:
#       bash vv/heavy/cc45-dogfood-devcontainer.sh
#   The fast CC-45 witnesses remain useful for diagnosis, but cannot replace this
#   full dogfood: the actual repository devcontainer, 4096 dynamic fork/exec cycles,
#   and the real gcc toolchain compiling and running a program in the guest.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEVCONTAINER_CLI_VERSION="0.87.0"
if [ -n "${CC45_DOGFOOD_ROOTFS:-}" ]; then
    ROOTFS="$CC45_DOGFOOD_ROOTFS"
    mkdir -p "$(dirname "$ROOTFS")"
    KEEP_ROOTFS=1
else
    ROOTFS="$(mktemp -d)/devcontainer-rootfs.tar"
    KEEP_ROOTFS=0
fi

command -v docker >/dev/null 2>&1 || { echo "cc45-dogfood: docker is required to build the devcontainer" >&2; exit 1; }
command -v npx >/dev/null 2>&1 || { echo "cc45-dogfood: node/npx is required for the Dev Container CLI" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "cc45-dogfood: sha256sum is required to bind the rootfs cache" >&2; exit 1; }

INPUT_SHA="$({ printf 'devcontainer-cli=%s\n' "$DEVCONTAINER_CLI_VERSION"; sha256sum "$ROOT/.devcontainer/devcontainer.json"; sha256sum "$ROOT/.devcontainer/devcontainer-lock.json"; } | sha256sum | cut -d' ' -f1)"
INPUT_SHORT="$(printf '%.12s' "$INPUT_SHA")"
IMG="holospaces-dogfood:cc45-$INPUT_SHORT"
CTR="holospaces-dogfood-cc45-$INPUT_SHORT-$$"
STAMP="${ROOTFS}.source.sha256"
ROOTFS_TMP="${ROOTFS}.tmp.$$"
STAMP_TMP="${STAMP}.tmp.$$"
ROOTFS_READY=0
STAMP_INPUT=""
STAMP_ROOTFS=""
if [ -s "$ROOTFS" ] && [ -f "$STAMP" ]; then
    read -r STAMP_INPUT STAMP_ROOTFS < "$STAMP" || true
    if [ "$STAMP_INPUT" = "$INPUT_SHA" ] && [ "$(sha256sum "$ROOTFS" | cut -d' ' -f1)" = "$STAMP_ROOTFS" ]; then
        ROOTFS_READY=1
    fi
fi

cleanup() {
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    docker rmi -f "$IMG" >/dev/null 2>&1 || true
    rm -f "$ROOTFS_TMP" "$STAMP_TMP" 2>/dev/null || true
    [ "$KEEP_ROOTFS" = 1 ] || rm -f "$ROOTFS" 2>/dev/null || true
}
trap cleanup EXIT

# A 4.4 GB devcontainer image + a 4.2 GB rootfs export + a transient ext4 image need
# headroom; on a CI runner reclaim the large preinstalled toolchains we don't use
# (best-effort, only what's present). Harmless locally (the dirs usually don't exist).
if [ -n "${CI:-}" ]; then
    for d in /usr/share/dotnet /usr/local/lib/android /opt/ghc \
             /opt/hostedtoolcache/CodeQL /usr/local/.ghcup; do
        [ -d "$d" ] && sudo rm -rf "$d" 2>/dev/null || true
    done
fi

if [ "$ROOTFS_READY" = 1 ]; then
    echo "== [1/3] exact devcontainer rootfs cache matches $INPUT_SHA =="
    echo "== [2/3] reuse exported rootfs ($(du -h "$ROOTFS" | cut -f1)) =="
else
    rm -f "$ROOTFS" "$STAMP"
    echo "== [1/3] build THIS repo's devcontainer, unmodified (Dev Container CLI) =="
    # Exactly the Codespaces/Gitpod resolution: the image + every digest-locked feature.
    npx --yes "@devcontainers/cli@$DEVCONTAINER_CLI_VERSION" build \
        --workspace-folder "$ROOT" --image-name "$IMG" >/dev/null 2>&1 \
        || { echo "cc45-dogfood: devcontainer build failed" >&2; exit 1; }
    echo "built $IMG ($(docker image inspect "$IMG" --format '{{.Size}}' 2>/dev/null) bytes)"

    echo "== [2/3] export its rootfs =="
    docker create --name "$CTR" "$IMG" true >/dev/null 2>&1 \
        || { echo "cc45-dogfood: docker create failed" >&2; exit 1; }
    docker export "$CTR" -o "$ROOTFS_TMP" \
        || { echo "cc45-dogfood: docker export failed" >&2; exit 1; }
    ROOTFS_SHA="$(sha256sum "$ROOTFS_TMP" | cut -d' ' -f1)"
    mv "$ROOTFS_TMP" "$ROOTFS"
    printf '%s %s\n' "$INPUT_SHA" "$ROOTFS_SHA" > "$STAMP_TMP"
    mv "$STAMP_TMP" "$STAMP"
    echo "exported rootfs: $(du -h "$ROOTFS" | cut -f1)"
fi

echo "== [3/3] boot it on the x86-64 core + compile in-guest =="
CC45_DOGFOOD_ROOTFS="$ROOTFS" cargo test --release --manifest-path "$ROOT/Cargo.toml" \
    -p holospaces --test cc45_dogfood holospaces_builds_in_its_own_real_devcontainer \
    -- --nocapture \
    || { echo "cc45-dogfood: the real devcontainer did not build a program in-guest" >&2; exit 1; }

echo "cc45-dogfood: PASS — holospaces built a program in its own unmodified devcontainer (gcc 13.3, in-guest, on the x86-64 core)"
