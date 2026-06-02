from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class BuildPy(build_py):
    def run(self) -> None:
        super().run()
        self._build_native_library()

    def _build_native_library(self) -> None:
        root = Path(__file__).resolve().parents[2]
        subprocess.check_call(["cargo", "build", "-p", "hologram-ffi", "--release"], cwd=root)
        source = root / "target" / "release" / rust_library_name()
        target = Path(self.build_lib) / "hologram" / python_library_name()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


class BDistWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False


def rust_library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libhologram_ffi.dylib"
    if system == "Windows":
        return "hologram_ffi.dll"
    return "libhologram_ffi.so"


def python_library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "_hologram_ffi.dylib"
    if system == "Windows":
        return "_hologram_ffi.dll"
    return "_hologram_ffi.so"


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": BDistWheel})
