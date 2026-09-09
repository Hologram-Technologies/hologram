#!/usr/bin/env bash
# CC-42 — the deployed browser manager pulls, assembles, and boots a real OCI image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOLOSPACES_BROWSER_ONLY=cc42 "$ROOT/scripts/browser-manager-test.sh"
