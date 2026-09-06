#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest


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


class PublicReleaseTests(unittest.TestCase):
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
