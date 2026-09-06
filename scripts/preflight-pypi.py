#!/usr/bin/env python3
"""Preflight and verify the immutable PyPI wheel set for the workspace version."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def expected_wheels(dist: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for wheel in sorted(dist.glob("*.whl")):
        rows[wheel.name] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if len(rows) != 4:
        raise RuntimeError(f"expected exactly 4 release wheels, found {len(rows)}")
    return rows


def public_wheels(version: str) -> dict[str, str]:
    request = Request(
        f"https://pypi.org/pypi/uor-hologram/{version}/json",
        headers={"User-Agent": "hologram-release/0.13"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return {}
        raise
    return {row["filename"]: row["digests"]["sha256"] for row in body["urls"]}


def compare(expected: dict[str, str], observed: dict[str, str]) -> list[str]:
    extra = sorted(set(observed) - set(expected))
    if extra:
        raise RuntimeError(f"public PyPI version contains unexpected files: {', '.join(extra)}")
    for name, digest in observed.items():
        if expected[name] != digest:
            raise RuntimeError(
                f"refusing public {name}: checksum {digest} differs from built {expected[name]}"
            )
    return sorted(set(expected) - set(observed))


def workspace_version() -> str:
    with Path("Cargo.toml").open("rb") as source:
        return tomllib.load(source)["workspace"]["package"]["version"]


def emit_output(name: str, value: str) -> None:
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as sink:
            sink.write(f"{name}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--publish-dist", type=Path, default=Path("publish-dist"))
    parser.add_argument("--wait-public", action="store_true")
    args = parser.parse_args()
    version = workspace_version()
    expected = expected_wheels(args.dist)

    attempts = 30 if args.wait_public else 1
    for attempt in range(1, attempts + 1):
        observed = public_wheels(version)
        missing = compare(expected, observed)
        if not missing:
            print(f"PyPI uor-hologram=={version} has the complete accepted wheel closure")
            emit_output("missing", "0")
            return
        if args.wait_public:
            print(f"waiting for {len(missing)} PyPI wheels ({attempt}/{attempts})")
            time.sleep(10)
            continue

        if args.publish_dist.exists():
            shutil.rmtree(args.publish_dist)
        args.publish_dist.mkdir(parents=True)
        for name in missing:
            shutil.copy2(args.dist / name, args.publish_dist / name)
        print(f"PyPI preflight passed; {len(missing)} of 4 wheels require publication")
        emit_output("missing", str(len(missing)))
        return
    raise RuntimeError(f"PyPI uor-hologram=={version} did not reach complete public closure")


if __name__ == "__main__":
    main()
