#!/usr/bin/env python3
"""Exercise the actual release-wait shell with isolated synthetic transports."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = tomllib.loads((ROOT / "Cargo.toml").read_text())
VERSION = WORKSPACE["workspace"]["package"]["version"]
PACKAGES = []
for member in [".", *WORKSPACE["workspace"]["members"]]:
    package = tomllib.loads((ROOT / member / "Cargo.toml").read_text())["package"]
    if package.get("publish") is not False:
        PACKAGES.append({"name": package["name"], "version": VERSION, "publish": None, "dependencies": []})

TRANSPORT = r'''#!/usr/bin/env python3
import hashlib,json,os,sys
from pathlib import Path
tool=Path(sys.argv[0]).name
args=sys.argv[1:]
config=json.loads(Path(os.environ["RELEASE_TEST_CONFIG"]).read_text())
with open(os.environ["RELEASE_TEST_LOG"], "a") as log:
    log.write(json.dumps({"tool":tool,"args":args})+"\n")
if tool in ("cargo","curl") and config.get("expect_no_token",True):
    assert "CARGO_REGISTRY_TOKEN" not in os.environ
    assert "CARGO_REGISTRIES_CRATES_IO_TOKEN" not in os.environ
    assert "CARGO_REGISTRY_GLOBAL_CREDENTIAL_PROVIDERS" not in os.environ
    assert "CARGO_REGISTRIES_CRATES_IO_CREDENTIAL_PROVIDER" not in os.environ
    assert not (Path(os.environ["CARGO_HOME"])/"credentials.toml").exists()
    assert not (Path(os.environ["CARGO_HOME"])/"config.toml").exists()
packages=config["packages"]
def payload(name): return ("synthetic package: "+name+"@"+config["version"]+"\n").encode()
if tool == "gh":
    assert args[0] == "api" and "head_sha="+os.environ["GITHUB_SHA"] in args[1]
    print(json.dumps({"workflow_runs":[{"event":"workflow_dispatch","head_sha":config.get("run_sha",os.environ["GITHUB_SHA"]),"created_at":"2026-09-16T00:00:00Z","status":"completed","conclusion":"success"}]}))
elif tool == "sleep":
    # Wrong-head green results must remain waiting, never enter acceptance. End
    # this synthetic transport immediately instead of sleeping for three hours.
    sys.exit(73)
elif tool == "cargo":
    if args[0] == "metadata": print(json.dumps({"packages":packages}))
    elif args[0] == "package":
        assert "--locked" in args and "--no-verify" in args
        if config.get("package_failure"): sys.exit(12)
        selected=[args[i+1] for i,arg in enumerate(args) if arg == "-p"]
        assert set(selected)=={row["name"] for row in packages}
        target=Path(os.environ["CARGO_TARGET_DIR"])/"package"
        target.mkdir(parents=True)
        for name in selected:
            if name != config.get("missing_artifact"):
                (target/(name+"-"+config["version"]+".crate")).write_bytes(payload(name))
        if config.get("source_race"):
            with open("Cargo.toml","a") as source: source.write("\n# changed during packaging\n")
    else: raise RuntimeError("forbidden cargo command: "+repr(args))
elif tool == "curl":
    assert args[0] == "--disable"
    assert not any("Authorization" in arg or arg == "--config" for arg in args)
    out=Path(args[args.index("--output")+1])
    url=args[-1]
    if url.startswith("https://crates.io/api/v1/crates/"):
        name,version=url.removeprefix("https://crates.io/api/v1/crates/").split("/")
        assert version==config["version"]
        if config.get("network_failure"): sys.exit(7)
        status=config.get("http",200)
        if config.get("missing") in ("all",name): status=404
        row={"crate":name,"num":version,"yanked":False,"checksum":hashlib.sha256(payload(name)).hexdigest()}
        for field,value in config.get("record",{}).items(): row[field]=value
        out.write_text("not JSON" if config.get("malformed") else json.dumps({"version":row}))
        print(status,end="")
    elif url.startswith("https://static.crates.io/crates/"):
        name,filename=url.removeprefix("https://static.crates.io/crates/").split("/")
        assert filename==name+"-"+config["version"]+".crate"
        out.write_bytes(b"stale published bytes" if config.get("wrong_download") in ("all",name) else payload(name))
        if config.get("download_source_race") and name == "uor-hologram":
            with open("Cargo.toml","a") as source: source.write("\n# changed during download\n")
    else: raise RuntimeError("forbidden registry operation: "+url)
else: raise RuntimeError("unexpected fixture tool")
'''


class PublishedCrateBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="hologram-published-crates-test.")
        self.addCleanup(self.scratch.cleanup)
        self.work = Path(self.scratch.name)
        self.repo = self.work / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        for name in ("publish-crates.sh", "wait-release-gate.sh", "workspace-version.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.repo / "scripts" / name)
        (self.repo / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{VERSION}"\n')
        self.git("init", "--quiet")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "test: synthetic release source")
        self.commit = self.git("rev-parse", "HEAD").strip()
        binaries = self.work / "bin"
        binaries.mkdir()
        for name in ("gh", "cargo", "curl", "sleep"):
            executable = binaries / name
            executable.write_text(TRANSPORT)
            executable.chmod(0o755)
        self.config_path = self.work / "config.json"
        self.log_path = self.work / "commands.jsonl"
        self.config = {"packages": PACKAGES, "version": VERSION}
        self.env = os.environ | {
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "GITHUB_REPOSITORY": "fixture/hologram",
            "GITHUB_SHA": self.commit,
            "GH_TOKEN": "synthetic-read-token",
            "CARGO_REGISTRY_TOKEN": "synthetic-do-not-read-or-publish",
            "CARGO_REGISTRIES_CRATES_IO_TOKEN": "synthetic-do-not-read-or-publish",
            "CARGO_REGISTRY_GLOBAL_CREDENTIAL_PROVIDERS": "synthetic-forbidden-provider",
            "CARGO_REGISTRIES_CRATES_IO_CREDENTIAL_PROVIDER": "synthetic-forbidden-provider",
            "RELEASE_TEST_CONFIG": str(self.config_path),
            "RELEASE_TEST_LOG": str(self.log_path),
        }
        ambient = self.work / "ambient-cargo-home"
        ambient.mkdir()
        (ambient / "credentials.toml").write_text('[registry]\ntoken="synthetic-do-not-read-or-publish"\n')
        (ambient / "config.toml").write_text('[registry]\nglobal-credential-providers=["synthetic-forbidden-provider"]\n')
        self.env["CARGO_HOME"] = str(ambient)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def run_gate(self, config=None, env=None, workflow="publish-crates.yml"):
        self.config_path.write_text(json.dumps(self.config | (config or {})))
        result = subprocess.run(
            ["bash", "scripts/wait-release-gate.sh", workflow], cwd=self.repo,
            env=self.env | (env or {}), capture_output=True, text=True, timeout=30,
        )
        self.assertNotIn("synthetic-do-not-read-or-publish", result.stdout + result.stderr)
        return result

    def commands(self):
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def reject(self, config, message, env=None):
        result = self.run_gate(config, env)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)
        self.assertNotIn("accepted exact source", result.stdout)

    def test_complete_exact_public_bytes_succeed_even_with_publish_credentials_in_parent(self):
        self.assertEqual(len(PACKAGES), 19)
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Complete exact published crate closure verified", result.stdout)
        commands = self.commands()
        self.assertEqual(sum(row["tool"] == "curl" and "static.crates.io" in row["args"][-1] for row in commands), 19)
        self.assertFalse(any(row["tool"] == "cargo" and row["args"][0] == "publish" for row in commands))

    def test_green_dry_run_without_publication_fails_even_if_caller_sets_dry_run(self):
        self.reject({"missing": "all"}, "workflow success is not publication", {"DRY_RUN": "1", "ALLOW_DIRTY": "1"})

    def test_missing_final_member_fails_complete_closure(self):
        self.reject({"missing": "uor-hologram"}, "missing uor-hologram")

    def test_wrong_registry_checksum_fails(self):
        self.reject({"record": {"checksum": "0" * 64}}, "differs from packaged")

    def test_stale_download_fails_even_when_registry_checksum_matches(self):
        self.reject({"wrong_download": "uor-hologram"}, "public bytes differ from exact source")

    def test_wrong_version_name_yanked_and_malformed_checksum_fail(self):
        for field, value, message in (
            ("num", "0.0.0", "identity/version/status"),
            ("crate", "wrong-crate", "identity/version/status"),
            ("yanked", True, "identity/version/status"),
            ("checksum", "wrong", "not lowercase SHA-256"),
            ("checksum", None, "not lowercase SHA-256"),
        ):
            with self.subTest(field=field, value=value):
                self.reject({"record": {field: value}}, message)

    def test_registry_transport_or_json_failure_is_not_absence_or_success(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                self.reject({"http": status}, f"HTTP {status}")
        self.reject({"malformed": True}, "JSONDecodeError")
        self.reject({"network_failure": True}, "checksum preflight")

    def test_missing_duplicate_and_wrong_version_metadata_fail(self):
        self.reject({"packages": PACKAGES[:-1]}, "exactly 19 distinct")
        self.reject({"packages": PACKAGES[:-1] + [PACKAGES[0]]}, "exactly 19 distinct")
        self.reject({"packages": [PACKAGES[0] | {"version": "0.0.0"}, *PACKAGES[1:]]}, "identity/version differs")

    def test_missing_artifact_and_packaging_failure_fail(self):
        self.reject({"missing_artifact": "uor-hologram"}, "missing packaged artifact")
        self.reject({"package_failure": True}, "package complete workspace release graph")

    def test_wrong_source_revision_dirty_checkout_and_mid_build_change_fail(self):
        self.reject({}, "clean exact GITHUB_SHA", {"GITHUB_SHA": "0" * 40})
        with (self.repo / "Cargo.toml").open("a") as source:
            source.write("\n# uncommitted\n")
        self.reject({}, "clean exact GITHUB_SHA", {"ALLOW_DIRTY": "1"})
        self.git("restore", "Cargo.toml")
        self.reject({"source_race": True}, "clean exact GITHUB_SHA")

    def test_wrong_head_green_run_cannot_enter_source_or_registry_acceptance(self):
        result = self.run_gate({"run_sha": "0" * 40})
        self.assertEqual(result.returncode, 73)
        self.assertNotIn("accepted exact source", result.stdout)
        self.assertEqual([row["tool"] for row in self.commands()], ["gh", "sleep"])

    def test_inherited_cargo_configuration_is_refused_without_reading(self):
        directory = self.work / ".cargo"
        directory.mkdir()
        config = directory / "config.toml"
        config.write_text("synthetic-unreadable-credential-config")
        config.chmod(0)
        self.reject({}, "refuses inherited Cargo configuration")
        self.assertEqual([row["tool"] for row in self.commands()], ["gh"])

    def test_source_changed_after_downloads_cannot_receive_acceptance(self):
        self.reject({"download_source_race": True}, "clean exact GITHUB_SHA")

    def test_npm_uses_pinned_rust_before_independent_crate_packaging(self):
        workflow = (ROOT / ".github/workflows/publish-npm.yml").read_text().split("\n  publish:\n", 1)[1]
        rust = workflow.index("dtolnay/rust-toolchain@d1031067263f94b142dd6c0ce24c5eb9d02d52a0")
        self.assertLess(rust, workflow.index("bash scripts/wait-release-gate.sh publish-crates.yml"))
        self.assertIn("toolchain: 1.98.1", workflow[rust:])

    def test_ordinary_release_ci_wait_does_not_package_or_use_registry(self):
        result = self.run_gate(workflow="release.yml")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([row["tool"] for row in self.commands()], ["gh"])


if __name__ == "__main__":
    unittest.main()
