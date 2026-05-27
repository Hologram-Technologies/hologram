#!/usr/bin/env bash
# Print the shared workspace version — the single source of truth every crate
# inherits via `[workspace.package].version`. Read straight from the root
# Cargo.toml (no cargo invocation, no dependency on a package *named* hologram,
# which does not exist — the `hologram` binary is produced by the `hologram-cli`
# package). Used by the release tooling (version-bump.yml, publish.yml).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 - "$ROOT/Cargo.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    doc = tomllib.load(f)
print(doc["workspace"]["package"]["version"])
PY
