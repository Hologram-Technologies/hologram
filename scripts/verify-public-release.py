#!/usr/bin/env python3
"""Wait for the complete cross-registry package closure before creating a source release."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import tomllib
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


NPM_PACKAGES = ("@tryhologram/sdk", "@tryhologram/native", "@tryhologram/wasm")
REQUIRED_WORKFLOWS = (
    "release.yml",
    "publish-crates.yml",
    "publish-npm.yml",
    "publish-pypi.yml",
)


def get_json(url: str, token: str | None = None) -> dict | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hologram-release/0.13",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def publishable_crates() -> list[str]:
    metadata = json.loads(
        subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps"])
    )
    return sorted(package["name"] for package in metadata["packages"] if package.get("publish") != [])


def successful_tag_run(rows: list[dict], commit: str, tag: str) -> bool:
    return any(
        row.get("event") == "push"
        and row.get("head_sha") == commit
        and row.get("head_branch") == tag
        and row.get("status") == "completed"
        and row.get("conclusion") == "success"
        for row in rows
    )


def missing_workflow_runs(repository: str, commit: str, version: str, token: str) -> list[str]:
    missing = []
    tag = f"v{version}"
    for workflow in REQUIRED_WORKFLOWS:
        encoded_workflow = quote(workflow, safe="")
        body = get_json(
            f"https://api.github.com/repos/{repository}/actions/workflows/{encoded_workflow}/runs"
            f"?event=push&head_sha={commit}&per_page=100",
            token,
        )
        rows = [] if body is None else body.get("workflow_runs", [])
        if not successful_tag_run(rows, commit, tag):
            missing.append(f"GitHub Actions:{workflow}@{tag} successful exact-commit run")
    return missing


def missing_public(version: str, crates: list[str]) -> list[str]:
    missing = []
    for crate in crates:
        if get_json(f"https://crates.io/api/v1/crates/{crate}/{version}") is None:
            missing.append(f"crates.io:{crate}@{version}")
    for package in NPM_PACKAGES:
        encoded = quote(package, safe="")
        if get_json(f"https://registry.npmjs.org/{encoded}/{version}") is None:
            missing.append(f"npm:{package}@{version}")
    if get_json(f"https://pypi.org/pypi/uor-hologram/{version}/json") is None:
        missing.append(f"PyPI:uor-hologram=={version}")
    return missing


def workspace_version() -> str:
    with Path("Cargo.toml").open("rb") as source:
        return tomllib.load(source)["workspace"]["package"]["version"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=workspace_version())
    parser.add_argument("--attempts", type=int, default=360)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    repository = os.environ.get("GITHUB_REPOSITORY")
    commit = os.environ.get("GITHUB_SHA")
    token = os.environ.get("GITHUB_TOKEN")
    if not repository or not commit or not token:
        raise RuntimeError("GITHUB_REPOSITORY, GITHUB_SHA, and GITHUB_TOKEN are required")
    crates = publishable_crates()
    if len(crates) != 19:
        raise RuntimeError(f"expected 19 publishable crates, found {len(crates)}: {crates}")
    for attempt in range(1, args.attempts + 1):
        missing = missing_workflow_runs(repository, commit, args.version, token)
        missing.extend(missing_public(args.version, crates))
        if not missing:
            print(f"complete public package closure verified for Hologram {args.version}")
            return
        print(f"public closure incomplete ({attempt}/{args.attempts}): {', '.join(missing)}")
        if attempt < args.attempts:
            time.sleep(args.interval)
    raise RuntimeError(f"public package closure remains incomplete for Hologram {args.version}")


if __name__ == "__main__":
    main()
