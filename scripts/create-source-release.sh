#!/usr/bin/env bash
# Create the deterministic source archive and canonical commit-binding manifest attested by the
# release workflow. The Git tag locates the release; GitHub OIDC/Sigstore provenance authenticates
# these exact bytes and the workflow/issuer/repository identity that produced them.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 VERSION OUTPUT_DIRECTORY" >&2
  exit 2
fi

version="$1"
output_directory="$2"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta|rc)\.[0-9]+)?$ ]] || {
  echo "invalid release version: $version" >&2
  exit 2
}

commit="$(git rev-parse "v${version}^{commit}")"
test "$commit" = "$(git rev-parse HEAD)" || {
  echo "release tag v${version} does not resolve to checked-out HEAD" >&2
  exit 1
}

mkdir -p "$output_directory"
archive="$output_directory/hologram-source-v${version}.tar.gz"
manifest="$output_directory/hologram-source-v${version}.json"
git archive --format=tar --prefix="hologram-${version}/" "$commit" | gzip -n > "$archive"

SOURCE_ARCHIVE="$archive" SOURCE_COMMIT="$commit" SOURCE_VERSION="$version" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

archive = Path(os.environ["SOURCE_ARCHIVE"])
manifest = {
    "archive": archive.name,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "commit": os.environ["SOURCE_COMMIT"],
    "schema": "hologram/source-release/1",
    "tag": f"v{os.environ['SOURCE_VERSION']}",
}
archive.with_suffix("").with_suffix(".json").write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

test -s "$archive"
test -s "$manifest"
