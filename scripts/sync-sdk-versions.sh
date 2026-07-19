#!/usr/bin/env bash
# Keep every SDK package version in lockstep with the shared workspace version
# (scripts/workspace-version.sh). The SDK packages (TypeScript native + wasm, Python) and their Rust
# driver crates are workspace-EXCLUDED (wasm `cdylib` / N-API targets), so they CANNOT inherit
# `[workspace.package].version` the way in-tree crates do. Without an explicit sync they drift silently
# — as they did (0.6/0.7 while the workspace was 0.10), which stale-locked the wasm driver a whole
# version behind. `version-bump.yml` runs this in write mode; CI gates it with `--check`.
#
#   scripts/sync-sdk-versions.sh          # rewrite every SDK version to the workspace version
#   scripts/sync-sdk-versions.sh --check  # exit 1 if any SDK version differs (drift gate)
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
WS="$(scripts/workspace-version.sh)"
MODE="${1:-write}"

# Every file that carries an SDK package version. JSON = the npm `"version"`; TOML = the `[package]`
# / `[project]` `version`. The first version line in each of these manifests is the package version.
FILES=(
  "sdk/typescript/package.json"
  "sdk/typescript/native/package.json"
  "sdk/typescript/wasm/package.json"
  "sdk/typescript/native/native/Cargo.toml"
  "sdk/typescript/wasm/driver/Cargo.toml"
  "sdk/python/pyproject.toml"
)

rc=0
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "::error::sync-sdk-versions: missing $f" >&2
    rc=1
    continue
  fi
  case "$f" in
    *.json) cur="$(grep -m1 '"version"' "$f" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" ;;
    *)      cur="$(grep -m1 '^version = ' "$f" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" ;;
  esac
  if [ "$cur" = "$WS" ]; then
    continue
  fi
  if [ "$MODE" = "--check" ]; then
    echo "::error::SDK version drift: $f is $cur, workspace is $WS — run scripts/sync-sdk-versions.sh"
    rc=1
  else
    case "$f" in
      *.json) perl -0pi -e "s/(\"version\":\s*\")[0-9]+\.[0-9]+\.[0-9]+/\${1}${WS}/" "$f" ;;
      *)      perl -0pi -e "s/(^version = \")[0-9]+\.[0-9]+\.[0-9]+/\${1}${WS}/m" "$f" ;;
    esac
    echo "  $f: $cur -> $WS"
  fi
done

if [ "$MODE" = "--check" ] && [ "$rc" -eq 0 ]; then
  echo "all SDK versions match the workspace ($WS)"
fi
exit $rc
