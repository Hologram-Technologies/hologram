#!/usr/bin/env bash
# CC-34 — the deployed workbench reaches its in-devcontainer capability over the bridge.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOLOSPACES_BROWSER_ONLY=cc34 "$ROOT/scripts/browser-manager-test.sh"
