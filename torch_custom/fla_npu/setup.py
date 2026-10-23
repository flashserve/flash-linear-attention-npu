# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

SETUP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SETUP_DIR.parents[1]
PACKAGE_DIR = SETUP_DIR / "fla_npu"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fla_npu_artifacts import (  # noqa: E402
    get_distribution_name, get_package_version, get_wheel_build_tag,
    get_wheel_platform_tag,
)
from stable_build_policy import (  # noqa: E402
    LAUNCHER_FILES, RUNTIME_REQUIREMENTS, reject_legacy_artifacts,
    validate_build_options, validate_launcher_payload, write_runtime_meta,
)

TRITON_CORE_PACKAGE = "fla_npu.ops.triton.triton_core"
TRITON_CORE_SOURCE = REPO_ROOT / "fla" / "ops" / "triton" / "triton_core"
OPP_PACKAGE_DATA = ["opp/**/*", "opp/vendors/config.ini"]


class CleanBuildPy(_build_py):
    def run(self):
        validate_build_options()
        reject_legacy_artifacts(PACKAGE_DIR)
        _build_stable_abi_library()
        write_runtime_meta(PACKAGE_DIR)
        built_package_dir = Path(self.build_lib) / "fla_npu"
        if built_package_dir.exists():
            shutil.rmtree(built_package_dir)
        super().run()
        for name in ("fla_npu_opp_env.py", "fla_npu_opp_env.pth", "fla_npu_opp_manage.py"):
            src = SETUP_DIR / name
            if src.exists():
                self.copy_file(str(src), str(Path(self.build_lib) / name))
        validate_launcher_payload(built_package_dir)


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return False


class StableBdistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.build_number = get_wheel_build_tag() or None

    def get_tag(self):
        return "py3", "none", get_wheel_platform_tag()


def _packages():
    packages = find_packages()
    if TRITON_CORE_SOURCE.exists() and TRITON_CORE_PACKAGE not in packages:
        packages.append(TRITON_CORE_PACKAGE)
    return packages


def _package_dir():
    if not TRITON_CORE_SOURCE.exists():
        return {}
    return {TRITON_CORE_PACKAGE: os.path.relpath(TRITON_CORE_SOURCE, SETUP_DIR).replace(os.sep, "/")}


def _build_stable_abi_library():
    """Build the mandatory launcher using vendored headers and torch link libraries."""
    builder = SETUP_DIR / "csrc" / "build_stable.py"
    subprocess.run(
        [sys.executable, str(builder), "--no-debug-probe", "--out",
         str(PACKAGE_DIR / "libfla_npu_stable.so")],
        check=True,
    )
    validate_launcher_payload(PACKAGE_DIR)


validate_build_options()
setup(
    name=get_distribution_name(),
    version=get_package_version(REPO_ROOT),
    description="FLA NPU Stable-ABI runtime",
    packages=_packages(),
    package_dir=_package_dir(),
    package_data={"fla_npu": OPP_PACKAGE_DATA + list(LAUNCHER_FILES)},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=RUNTIME_REQUIREMENTS,
    distclass=BinaryDistribution,
    cmdclass={"build_py": CleanBuildPy, "bdist_wheel": StableBdistWheel},
)
