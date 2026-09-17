#!/usr/bin/env bash
# CC-32 — distinct holospace identities use disjoint durable browser state.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOLOSPACES_BROWSER_ONLY=cc32 "$ROOT/scripts/browser-manager-test.sh"
