#!/usr/bin/env python3
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "preflight_pypi", Path(__file__).with_name("preflight-pypi.py")
)
assert spec is not None and spec.loader is not None
preflight_pypi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight_pypi)
compare = preflight_pypi.compare

release_spec = importlib.util.spec_from_file_location(
    "verify_public_release", Path(__file__).with_name("verify-public-release.py")
)
assert release_spec is not None and release_spec.loader is not None
verify_public_release = importlib.util.module_from_spec(release_spec)
release_spec.loader.exec_module(verify_public_release)
successful_tag_run = verify_public_release.successful_tag_run


class PyPiPreflightTests(unittest.TestCase):
    def test_missing_files_are_returned_in_order(self):
        self.assertEqual(compare({"b": "2", "a": "1"}, {"a": "1"}), ["b"])

    def test_checksum_disagreement_fails(self):
        with self.assertRaisesRegex(RuntimeError, "differs from built"):
            compare({"a": "1"}, {"a": "wrong"})

    def test_unexpected_public_file_fails(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected files"):
            compare({"a": "1"}, {"a": "1", "extra": "2"})


class BrowserToolInstallationTests(unittest.TestCase):
    def test_release_browser_tools_replace_cached_binaries_and_verify_exact_pins(self):
        workflow = (Path(__file__).parent.parent / ".github/workflows/release.yml").read_text()
        expected_bindgen = [
            "cargo install wasm-bindgen-cli --version 0.2.122 --locked --force",
            'test "$(wasm-bindgen --version)" = "wasm-bindgen 0.2.122"',
        ]
        for name, expected in (
            ("Install wasm-bindgen-cli (pinned to the crate's wasm-bindgen)", expected_bindgen),
            ("Install wasm-pack + wasm-bindgen-cli (browser CC suites)", [
                "cargo install wasm-pack --version 0.13.1 --locked --force",
                'test "$(wasm-pack --version)" = "wasm-pack 0.13.1"',
                *expected_bindgen,
            ]),
        ):
            with self.subTest(step=name):
                step = workflow.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]
                self.assertEqual([line.strip() for line in step.splitlines() if line.strip()], ["run: |", *expected])


class PublicReleaseTests(unittest.TestCase):
    def test_requested_release_version_must_match_source(self):
        with mock.patch.object(sys, "argv", ["verify-public-release.py", "--version", "0.0.0"]), \
                mock.patch.object(verify_public_release, "workspace_version", return_value="0.13.1"):
            with self.assertRaisesRegex(RuntimeError, "differs from checked-out workspace"):
                verify_public_release.main()

    def test_final_release_requires_independent_crate_bytes_and_propagates_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), \
                    mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "fixture/hologram", "GITHUB_SHA": "a" * 40, "GITHUB_TOKEN": "synthetic"}), \
                    mock.patch.object(sys, "argv", ["verify-public-release.py", "--attempts", "1"]), \
                    mock.patch.object(verify_public_release, "workspace_version", return_value="0.13.1"), \
                    mock.patch.object(verify_public_release, "publishable_crates", return_value=[f"fixture-{index}" for index in range(19)]), \
                    mock.patch.object(verify_public_release, "missing_workflow_runs", return_value=[]), \
                    mock.patch.object(verify_public_release, "missing_public", return_value=[]), \
                    mock.patch.object(verify_public_release.subprocess, "run") as run, \
                    mock.patch("builtins.print") as emit:
                if failure:
                    run.side_effect = subprocess.CalledProcessError(1, "synthetic checker")
                    with self.assertRaises(subprocess.CalledProcessError):
                        verify_public_release.main()
                    emit.assert_not_called()
                else:
                    verify_public_release.main()
                    emit.assert_called_once()
                run.assert_called_once_with(["bash", "scripts/publish-crates.sh", "--verify-published"], check=True)

    def test_only_successful_push_for_exact_tag_and_commit_is_accepted(self):
        expected = {
            "event": "push",
            "head_sha": "abc",
            "head_branch": "v1.2.3",
            "status": "completed",
            "conclusion": "success",
        }
        self.assertTrue(successful_tag_run([expected], "abc", "v1.2.3"))
        for field, wrong in (
            ("event", "workflow_dispatch"),
            ("head_sha", "def"),
            ("head_branch", "main"),
            ("status", "in_progress"),
            ("conclusion", "failure"),
        ):
            row = expected | {field: wrong}
            self.assertFalse(successful_tag_run([row], "abc", "v1.2.3"))


if __name__ == "__main__":
    unittest.main()
