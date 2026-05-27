#!/usr/bin/env bash
# Tests scripts/release-api-history.sh: the cross-version API history/changelog
# logic the release workflow runs. Simulates two releases with hand-crafted API
# snapshots (no cargo) and asserts the changelog accumulates correctly and the
# per-version snapshots are archived.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() { echo "FAIL: $1" >&2; exit 1; }
assert_contains() { grep -qF "$2" "$1" || fail "expected '$2' in $1"; }

# workspace-version.sh must return the shared version (used by the release
# tooling in place of the non-existent `hologram` package). Run from the repo
# before switching to the throwaway work dir.
WV="$("$SCRIPTS/workspace-version.sh")"
echo "$WV" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+' || fail "workspace-version.sh returned non-semver: '$WV'"
echo "ok   workspace-version: $WV"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

mkdir -p api

# ── Release 1 (v0.5.0): no prior archive ⇒ everything is Added ───────────────
cat > api/demo.txt <<'EOF'
pub fn demo::keep() -> u8
pub fn demo::old_sig(x: u8) -> u8
pub fn demo::soon_gone()
pub fn demo::to_remove()
EOF
"$SCRIPTS/release-api-history.sh" 0.5.0 ""

[ -f api/CHANGELOG.md ] || fail "CHANGELOG.md not created"
[ -d api/history/v0.5.0 ] || fail "history/v0.5.0 not archived"
[ -f api/history/v0.5.0/demo.txt ] || fail "demo.txt not archived in v0.5.0"
assert_contains api/CHANGELOG.md "## v0.5.0"
assert_contains api/CHANGELOG.md "### Added"
assert_contains api/CHANGELOG.md "pub fn demo::keep() -> u8"

# ── Release 2 (v0.6.0): add, change signature, deprecate, remove ─────────────
cat > api/demo.txt <<'EOF'
pub fn demo::keep() -> u8
pub fn demo::old_sig(x: u8, y: u8) -> u8
#[deprecated] pub fn demo::soon_gone()
pub fn demo::brand_new()
EOF
"$SCRIPTS/release-api-history.sh" 0.6.0 0.5.0

[ -d api/history/v0.6.0 ] || fail "history/v0.6.0 not archived"
# Newest section is on top.
head -8 api/CHANGELOG.md | grep -qF "## v0.6.0" || fail "v0.6.0 section not prepended"
# Both versions are retained in the accumulated changelog.
assert_contains api/CHANGELOG.md "## v0.6.0"
assert_contains api/CHANGELOG.md "## v0.5.0"
# Scenario categorization.
assert_contains api/CHANGELOG.md "pub fn demo::brand_new()"          # Added
assert_contains api/CHANGELOG.md "pub fn demo::old_sig(x: u8, y: u8) -> u8"  # Changed
assert_contains api/CHANGELOG.md "#[deprecated] pub fn demo::soon_gone()"    # Deprecated
assert_contains api/CHANGELOG.md "pub fn demo::to_remove()"          # Removed
for sec in "### Added" "### Changed (breaking)" "### Deprecated" "### Removed (breaking)"; do
  assert_contains api/CHANGELOG.md "$sec"
done

echo "ok   release-api-history: two-release flow, categorization, accumulation, archive"
echo "PASS test-release-tooling"
