#!/usr/bin/env bash
# Release step: turn the freshly-regenerated `api/<crate>.txt` snapshots into the
# cross-version API history — prepend a categorized section to `api/CHANGELOG.md`
# (diffing against the previous release's archive) and archive the current
# snapshots under `api/history/v<new>/`.
#
# Pure file operations (no cargo), so it is unit-testable in isolation — the
# caller (version-bump.yml / a test) is responsible for having regenerated the
# `api/` snapshots first (scripts/update-api-snapshots.sh).
#
# Usage: release-api-history.sh <new_version> [prev_version]
set -euo pipefail

NEW="${1:?usage: release-api-history.sh <new_version> [prev_version]}"
PREV="${2:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if ! ls api/*.txt >/dev/null 2>&1; then
  echo "::error::no api/*.txt snapshots — run scripts/update-api-snapshots.sh first" >&2
  exit 1
fi

new_combined="$(mktemp)"
prev_combined="$(mktemp)"
trap 'rm -f "$new_combined" "$prev_combined"' EXIT

cat api/*.txt > "$new_combined"
if [ -n "$PREV" ] && ls "api/history/v${PREV}"/*.txt >/dev/null 2>&1; then
  cat "api/history/v${PREV}"/*.txt > "$prev_combined"
else
  : > "$prev_combined"   # no prior archive: everything reads as Added
fi

python3 "$HERE/api-changelog.py" \
  --old "$prev_combined" --new "$new_combined" \
  --version "$NEW" --output api/CHANGELOG.md

mkdir -p "api/history/v${NEW}"
cp api/*.txt "api/history/v${NEW}/"
echo "Updated api/CHANGELOG.md and archived snapshots to api/history/v${NEW}/"
