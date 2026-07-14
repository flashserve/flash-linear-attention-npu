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
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import find_packages, setup


SETUP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "flash-linear-attention-npu"
DEFAULT_VERSION = "1.0.0"
TRITON_CORE_PACKAGE = "fla_npu.ops.triton.triton_core"
TRITON_CORE_SOURCE = REPO_ROOT / "fla" / "ops" / "triton" / "triton_core"
OPP_PACKAGE_DATA = [
    "opp/**/*",
    "opp/vendors/config.ini",
    "opp/vendors/fla_npu_transformer/README.txt",
]


def _env_flag(name):
    return os.getenv(name, "FALSE").upper() in {"1", "TRUE", "YES", "ON"}


def _package_version():
    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return DEFAULT_VERSION

    sys.path.insert(0, str(scripts_dir))
    try:
        from fla_npu_artifacts import get_package_version

        return get_package_version(REPO_ROOT)
    except Exception:
        return DEFAULT_VERSION
    finally:
        try:
            sys.path.remove(str(scripts_dir))
        except ValueError:
            pass


def _packages():
    packages = find_packages()
    if TRITON_CORE_SOURCE.exists() and TRITON_CORE_PACKAGE not in packages:
        packages.append(TRITON_CORE_PACKAGE)
    return packages


def _package_dir():
    if not TRITON_CORE_SOURCE.exists():
        return {}
    # Keep a single Triton source tree in fla/ and map it into the standalone wheel.
    triton_core_dir = os.path.relpath(TRITON_CORE_SOURCE, SETUP_DIR).replace(os.sep, "/")
    return {TRITON_CORE_PACKAGE: triton_core_dir}


def _setup_pure_python():
    setup(
        name=PACKAGE_NAME,
        version=_package_version(),
        description="FLA NPU Python runtime",
        packages=_packages(),
        package_dir=_package_dir(),
        package_data={"fla_npu": OPP_PACKAGE_DATA},
        include_package_data=True,
        zip_safe=False,
    )


def _setup_legacy_extension():
    import torch
    import torch_npu
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    pytorch_version = subprocess.check_output(
        [sys.executable, "-c", 'import torch; print(torch.__version__.split("+")[0])']
    ).decode("utf-8").strip()
    version_parts = pytorch_version.split(".")
    pytorch_version_dir = f"v{version_parts[0]}r{version_parts[1]}"

    os.environ["PYTORCH_VERSION"] = pytorch_version
    os.environ["PYTORCH_CUSTOM_DERIVATIVES_PATH"] = os.path.join(
        os.path.dirname(__file__), f"op-plugin/config/{pytorch_version_dir}/derivatives.yaml"
    )
    os.environ["ACNN_EXTENSION_PATH"] = os.path.dirname(__file__)
    os.environ["ACNN_EXTENSION_SWITCH"] = "TRUE"

    def get_sources():
        sources = []
        aten_dir = os.path.join(os.path.dirname(__file__), "torch_npu/csrc/aten")
        if os.path.exists(aten_dir):
            for root, _, files in os.walk(aten_dir):
                for file in files:
                    if file.endswith((".cpp", ".cc")):
                        sources.append(os.path.join(root, file))
        ops_dir = os.path.join(os.path.dirname(__file__), "op_plugin")
        if os.path.exists(ops_dir):
            for root, _, files in os.walk(ops_dir):
                for file in files:
                    if file.endswith((".cpp", ".cc")):
                        sources.append(os.path.join(root, file))

        excluded = {
            f"{aten_dir}/VariableTypeEverything.cpp",
            f"{aten_dir}/ADInplaceOrViewTypeEverything.cpp",
            f"{aten_dir}/python_functionsEverything.cpp",
            f"{aten_dir}/RegisterFunctionalizationEverything.cpp",
        }
        return [source for source in sources if source not in excluded]

    def get_include_dirs():
        torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
        base_dir = os.path.dirname(__file__)
        include_dirs = []
        for relative in ("torch_npu/csrc/aten", "op_plugin", "."):
            path = os.path.join(base_dir, relative)
            if os.path.exists(path):
                include_dirs.append(path)
        include_dirs.extend(
            [
                os.path.join(torch_npu_path, "include"),
                os.path.join(torch_npu_path, "include", "third_party", "acl", "inc"),
                os.path.join(torch_npu_path, "include", "third_party", "hccl", "inc"),
                os.path.join(torch_npu_path, "include", "third_party", "op-plugin"),
            ]
        )
        return include_dirs

    def get_compile_args():
        compile_args = ["-std=c++17"]
        if sys.platform == "win32":
            compile_args.append("/MD")
        elif sys.platform == "linux":
            compile_args.append("-fPIC")
        return compile_args

    def get_link_args():
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        torch_npu_lib = os.path.join(os.path.dirname(torch_npu.__file__), "lib")
        link_args = [
            "-ltorch_npu",
            "-ltorch",
            "-lc10",
            f"-L{sysconfig.get_config_var('LIBDIR')}",
            f"-L{torch_lib}",
            f"-L{torch_npu_lib}",
        ]
        if sys.platform == "linux":
            link_args.extend(
                [
                    "-Wl,--enable-new-dtags",
                    "-Wl,-rpath,$ORIGIN/../torch/lib",
                    "-Wl,-rpath,$ORIGIN/../torch_npu/lib",
                    "-Wl,-rpath,$ORIGIN/opp/vendors/fla_npu_transformer/op_api/lib",
                ]
            )
        return link_args

    setup(
        name=PACKAGE_NAME,
        version=_package_version(),
        description="FLA NPU legacy PyTorch extension",
        ext_modules=[
            CppExtension(
                "fla_npu.custom_aclnn_extension_lib",
                sources=get_sources(),
                include_dirs=get_include_dirs(),
                extra_compile_args=get_compile_args(),
                extra_link_args=get_link_args(),
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
        packages=_packages(),
        package_dir=_package_dir(),
        package_data={"fla_npu": OPP_PACKAGE_DATA},
        include_package_data=True,
    )


if _env_flag("FLA_NPU_BUILD_LEGACY_EXTENSION"):
    _setup_legacy_extension()
else:
    _setup_pure_python()
