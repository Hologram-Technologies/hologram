#!/bin/sh
# Hologram CLI installer.
#
# Downloads a prebuilt `hologram` binary from the GitHub releases — no repository
# clone and no Rust toolchain required. On platforms without a prebuilt binary it
# falls back to building from source with `cargo` (if available).
#
#   curl -fsSL https://raw.githubusercontent.com/Hologram-Technologies/hologram/main/install.sh | sh
#
# Options (flags, or the matching HOLOGRAM_* environment variables):
#   --version <vX.Y.Z>   release tag to install           (default: latest)
#   --bin-dir <dir>      where to install the binary       (default: ~/.local/bin)
#   -h, --help           show this help
set -eu

REPO="Hologram-Technologies/hologram"
BIN="hologram"
VERSION="${HOLOGRAM_VERSION:-latest}"
BIN_DIR="${HOLOGRAM_BIN_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"

info() { printf 'hologram-install: %s\n' "$*"; }
err()  { printf 'hologram-install: error: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,13p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version)   VERSION="${2:-}"; shift 2 ;;
    --version=*) VERSION="${1#*=}"; shift ;;
    --bin-dir)   BIN_DIR="${2:-}"; shift 2 ;;
    --bin-dir=*) BIN_DIR="${1#*=}"; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)           err "unknown argument: $1 (try --help)" ;;
  esac
done

# Pick a downloader.
if command -v curl >/dev/null 2>&1; then
  dl() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  dl() { wget -qO "$2" "$1"; }
else
  err "need either curl or wget on PATH"
fi

# Build from source when there is no matching prebuilt binary.
fallback_cargo() {
  reason="$1"
  if command -v cargo >/dev/null 2>&1; then
    info "$reason — building from source with cargo instead"
    if [ "$VERSION" = latest ]; then
      exec cargo install --git "https://github.com/$REPO" --locked "$BIN-cli"
    fi
    exec cargo install --git "https://github.com/$REPO" --tag "$VERSION" --locked "$BIN-cli"
  fi
  err "$reason, and cargo is not installed.
Install Rust from https://rustup.rs, then run:
  cargo install --git https://github.com/$REPO --locked $BIN-cli"
}

# Map uname to a release-asset suffix.
os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Linux)  os_tag=linux ;;
  Darwin) os_tag=macos ;;
  *)      os_tag="" ;;
esac
case "$arch" in
  x86_64|amd64)  arch_tag=amd64 ;;
  arm64|aarch64) arch_tag=arm64 ;;
  *)             arch_tag="" ;;
esac

asset=""
case "${os_tag}-${arch_tag}" in
  linux-amd64|macos-amd64|macos-arm64) asset="${BIN}-${os_tag}-${arch_tag}.tar.gz" ;;
esac
[ -n "$asset" ] || fallback_cargo "no prebuilt binary for ${os}/${arch}"

if [ "$VERSION" = latest ]; then
  url="https://github.com/$REPO/releases/latest/download/$asset"
else
  url="https://github.com/$REPO/releases/download/$VERSION/$asset"
fi

tmp=$(mktemp -d 2>/dev/null || mktemp -d -t hologram-install)
trap 'rm -rf "$tmp"' EXIT INT TERM

info "downloading $asset ($VERSION)"
dl "$url" "$tmp/$asset" || fallback_cargo "download failed ($url)"
tar -xzf "$tmp/$asset" -C "$tmp" || err "could not extract $asset"

# Locate the binary regardless of the archive's internal layout.
src=$(find "$tmp" -type f -name "$BIN" 2>/dev/null | head -n1)
[ -n "$src" ] || err "'$BIN' not found inside $asset"

mkdir -p "$BIN_DIR"
if install -m 0755 "$src" "$BIN_DIR/$BIN" 2>/dev/null; then :; else
  cp "$src" "$BIN_DIR/$BIN" && chmod 0755 "$BIN_DIR/$BIN"
fi

info "installed to $BIN_DIR/$BIN"
"$BIN_DIR/$BIN" --version 2>/dev/null || true

# Remind the user if the install dir is not on PATH.
case ":${PATH}:" in
  *":$BIN_DIR:"*) ;;
  *)
    info "note: $BIN_DIR is not on your PATH — add it, e.g.:"
    info "  export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac
