from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
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


class BinaryDistribution(Distribution):
    # Marks the distribution as non-pure so the bundled `_hologram_ffi` shared library lands in the
    # package root (platlib) and the wheel carries a platform tag — without this, setuptools treats
    # the package as pure and shunts every file under `<name>.data/purelib/`.
    def has_ext_modules(self) -> bool:
        return True


class BDistWheel(bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        # The bundled `_hologram_ffi` is a plain shared library loaded via ctypes — it has no CPython
        # ABI, so one platform wheel serves every Python 3.x. Force `py3-none-<platform>` instead of
        # the default `cpXY-cpXY-<platform>` (which would need a separate wheel per interpreter).
        _, _, plat = super().get_tag()
        # GitHub's setup-python ships a `universal2` (fat) macOS interpreter, so get_tag() reports
        # `macosx_..._universal2` on BOTH the arm64 and x86_64 runners — but cargo builds a single-arch
        # dylib. Left alone, the two macOS wheels collide on one filename AND mislabel the arch. Retag
        # to the real build arch. (No-op off macOS or when already single-arch, e.g. a local build.)
        if plat.startswith("macosx") and plat.endswith("_universal2"):
            plat = plat.rsplit("_", 1)[0] + "_" + platform.machine()
        return "py3", "none", plat


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


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": BDistWheel}, distclass=BinaryDistribution)
