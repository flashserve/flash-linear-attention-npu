from __future__ import annotations

import importlib
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import setuptools


try:
    import packaging.version  # noqa: F401
except ImportError:
    from pip._vendor import packaging as vendored_packaging
    from pip._vendor.packaging import version as vendored_packaging_version

    sys.modules["packaging"] = vendored_packaging
    sys.modules["packaging.version"] = vendored_packaging_version


if not hasattr(importlib, "metadata"):
    # The project requires Python >=3.9. This compatibility shim only lets the
    # source-only packaging tests run on older maintenance hosts.
    metadata_module = types.ModuleType("importlib.metadata")

    def _missing_distribution(_name: str) -> str:
        raise RuntimeError("distribution metadata is unavailable")

    metadata_module.version = _missing_distribution
    importlib.metadata = metadata_module
    sys.modules["importlib.metadata"] = metadata_module


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    REPO_ROOT
    / "torch_custom"
    / "fla_npu"
    / "fla_npu"
    / "__init__.py"
)


def _load_setup() -> tuple[dict[str, object], dict[str, object]]:
    setup_kwargs: dict[str, object] = {}

    def capture_setup(**kwargs) -> None:
        setup_kwargs.update(kwargs)

    with mock.patch.object(setuptools, "setup", side_effect=capture_setup):
        setup_globals = runpy.run_path(str(REPO_ROOT / "setup.py"))
    return setup_globals, setup_kwargs


def _create_minimal_vendor(vendor_dir: Path, *, include_alias: bool = False) -> None:
    required_files = (
        vendor_dir / "op_api" / "lib" / "libcust_opapi.so",
        vendor_dir
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "op_host"
        / "lib"
        / "linux"
        / "x86_64"
        / "libophost.so",
        vendor_dir
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "ascend910b"
        / "sample"
        / "sample.o",
        vendor_dir
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "config"
        / "ascend910b"
        / "binary_info_config.json",
    )
    for path in required_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    if include_alias:
        (vendor_dir / "op_api" / "lib" / "libopapi.so").write_bytes(b"unsafe")


def _create_runtime_package(site_root: Path) -> tuple[Path, Path, Path]:
    package_dir = site_root / "fla_npu"
    package_dir.mkdir(parents=True)
    runtime_path = package_dir / "__init__.py"
    shutil.copy2(RUNTIME_SOURCE, runtime_path)
    vendor_dir = package_dir / "opp" / "vendors" / "fla_npu_transformer"
    packaged_opapi = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    packaged_opapi.parent.mkdir(parents=True)
    packaged_opapi.write_bytes(b"packaged-test")
    return runtime_path, vendor_dir, packaged_opapi


def _load_runtime_without_legacy_extension(runtime_path: Path) -> dict[str, object]:
    source = runtime_path.read_text(encoding="utf-8")
    marker = "\n_load_opextension_so()\n"
    if marker not in source:
        raise AssertionError("Unable to isolate fla_npu runtime initialization")
    source = source.rsplit(marker, 1)[0]
    namespace = {
        "__file__": str(runtime_path),
        "__name__": "fla_npu_runtime_test",
    }
    exec(compile(source, str(runtime_path), "exec"), namespace)
    return namespace


class WheelEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_globals, cls.setup_kwargs = _load_setup()

    def test_staging_removes_conflicting_libopapi_alias(self) -> None:
        stage_run_package = self.setup_globals["_stage_run_package"]

        with tempfile.TemporaryDirectory() as temp_dir:
            opp_root = Path(temp_dir) / "opp"

            def fake_install_run_package(_run_file, target_root) -> None:
                _create_minimal_vendor(
                    Path(target_root) / "vendors" / "fla_npu_transformer",
                    include_alias=True,
                )

            stage_run_package.__globals__["_install_run_package"] = fake_install_run_package
            stage_run_package(Path(temp_dir) / "unused.run", opp_root)

            lib_dir = (
                opp_root
                / "vendors"
                / "fla_npu_transformer"
                / "op_api"
                / "lib"
            )
            self.assertTrue((lib_dir / "libcust_opapi.so").is_file())
            self.assertFalse((lib_dir / "libopapi.so").exists())

            custom_build = (REPO_ROOT / "cmake" / "custom_build.cmake").read_text(
                encoding="utf-8"
            )
            symbol_build = (REPO_ROOT / "cmake" / "symbol.cmake").read_text(
                encoding="utf-8"
            )
            self.assertIn("NO_SONAME ON", custom_build)
            self.assertIn("NO_SONAME ON", symbol_build)

    def test_generated_set_env_is_idempotent(self) -> None:
        rewrite_set_env = self.setup_globals["_rewrite_set_env"]

        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = (
                Path(temp_dir) / "opp" / "vendors" / "fla_npu_transformer"
            )
            rewrite_set_env(vendor_dir)
            set_env = vendor_dir / "bin" / "set_env.bash"
            script = f"""
set -euo pipefail
unset ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH FLA_NPU_OPP_PATH FLA_NPU_OP_API_LIB
source {set_env!s}
source {set_env!s}
[[ "${{ASCEND_CUSTOM_OPP_PATH}}" == "{vendor_dir.parent.parent}:{vendor_dir}" ]]
[[ -z "${{LD_LIBRARY_PATH-}}" ]]
[[ "${{FLA_NPU_OPP_PATH}}" == "{vendor_dir.parent.parent}" ]]
[[ "${{FLA_NPU_OP_API_LIB}}" == "{vendor_dir}/op_api/lib/libcust_opapi.so" ]]
"""
            subprocess.run(["bash", "-c", script], check=True)

    def test_external_opp_install_removes_conflicting_alias(self) -> None:
        install_opp_globals = runpy.run_path(
            str(
                REPO_ROOT
                / "torch_custom"
                / "fla_npu"
                / "fla_npu"
                / "install_opp.py"
            )
        )
        install_opp = install_opp_globals["install_opp"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_opp_root = temp_root / "package-opp"
            vendor_src = (
                package_opp_root
                / "vendors"
                / "fla_npu_transformer"
            )
            _create_minimal_vendor(vendor_src, include_alias=True)
            install_opp.__globals__["PACKAGE_OPP_ROOT"] = package_opp_root

            install_root = temp_root / "external-opp"
            install_opp(install_root)
            installed_vendor = (
                install_root
                / "vendors"
                / "fla_npu_transformer"
            )
            self.assertTrue(
                (installed_vendor / "op_api" / "lib" / "libcust_opapi.so").is_file()
            )
            self.assertFalse(
                (installed_vendor / "op_api" / "lib" / "libopapi.so").exists()
            )

            set_env = installed_vendor / "bin" / "set_env.bash"
            script = f"""
set -euo pipefail
unset ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH FLA_NPU_OPP_PATH FLA_NPU_OP_API_LIB
source {set_env!s}
[[ -z "${{LD_LIBRARY_PATH-}}" ]]
"""
            subprocess.run(["bash", "-c", script], check=True)

    def test_legacy_extension_does_not_publish_packaged_opapi(self) -> None:
        extension_source = (
            REPO_ROOT
            / "torch_custom"
            / "fla_npu"
            / "op_plugin"
            / "ops"
            / "opapi"
            / "FLANpuOpApi.cpp"
        ).read_text(encoding="utf-8")
        fallback_source = (
            REPO_ROOT / "common" / "include" / "fallback" / "fallback.h"
        ).read_text(encoding="utf-8")
        extension_setup = (
            REPO_ROOT / "torch_custom" / "fla_npu" / "setup.py"
        ).read_text(encoding="utf-8")

        self.assertIn("RTLD_LOCAL | RTLD_NOW", extension_source)
        self.assertNotIn("RTLD_LAZY | RTLD_GLOBAL", extension_source)
        self.assertIn("RTLD_LOCAL | RTLD_NOW", fallback_source)
        self.assertNotIn("dlopen(embeddedOpApiLib, RTLD_LAZY | RTLD_GLOBAL)", fallback_source)
        self.assertNotIn("/opp/vendors/{vendor_dir}/op_api/lib", extension_setup)
        self.assertIn("$ORIGIN/../torch/lib", extension_setup)
        self.assertIn("$ORIGIN/../torch_npu/lib", extension_setup)

    def test_runtime_loads_cann_then_packaged_opapi_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, vendor_dir, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )
            external_vendor = Path(temp_dir) / "external" / "vendors" / "other"
            _create_minimal_vendor(external_vendor)

            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_HOME_PATH": "/fake/cann",
                    "FLA_NPU_OPP_PATH": str(external_vendor),
                    "ASCEND_CUSTOM_OPP_PATH": str(external_vendor),
                    "ASCEND_OPP_PATH": str(external_vendor.parent.parent),
                },
                clear=True,
            ):
                def fake_cdll(path, *, mode):
                    if str(path) == "libopapi.so":
                        return mock.sentinel.cann_opapi
                    return mock.sentinel.custom_opapi

                with mock.patch("ctypes.CDLL", side_effect=fake_cdll) as cdll:
                    runtime_globals = _load_runtime_without_legacy_extension(runtime_path)
                    first = runtime_globals["load_ascendc_opapi_libraries"]()
                    second = runtime_globals["load_ascendc_opapi_libraries"]()

                self.assertIs(first, second)
                self.assertEqual(
                    first,
                    [mock.sentinel.custom_opapi, mock.sentinel.cann_opapi],
                )
                self.assertEqual(len(cdll.call_args_list), 2)
                self.assertEqual(cdll.call_args_list[0].args, ("libopapi.so",))
                self.assertEqual(
                    Path(cdll.call_args_list[1].args[0]).resolve(),
                    packaged_opapi.resolve(),
                )
                expected_mode = (
                    getattr(os, "RTLD_LOCAL", 0)
                    | getattr(os, "RTLD_NOW", 0)
                    | getattr(os, "RTLD_NODELETE", 0)
                )
                self.assertEqual(cdll.call_args_list[0].kwargs["mode"], expected_mode)
                self.assertEqual(cdll.call_args_list[1].kwargs["mode"], expected_mode)
                self.assertEqual(
                    expected_mode & getattr(os, "RTLD_GLOBAL", 0),
                    0,
                )
                self.assertNotIn("LD_LIBRARY_PATH", os.environ)
                self.assertEqual(
                    os.environ["FLA_NPU_OP_API_LIB"],
                    str(packaged_opapi.resolve()),
                )
                self.assertEqual(
                    os.environ["ASCEND_CUSTOM_OPP_PATH"],
                    os.pathsep.join(
                        [
                            str(vendor_dir.parent.parent),
                            str(vendor_dir),
                            str(external_vendor),
                        ]
                    ),
                )

    def test_runtime_does_not_fallback_to_external_custom_opp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, _, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )
            packaged_opapi.unlink()
            external_vendor = Path(temp_dir) / "external" / "vendors" / "other"
            _create_minimal_vendor(external_vendor)

            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_HOME_PATH": "/fake/cann",
                    "FLA_NPU_OPP_PATH": str(external_vendor),
                    "ASCEND_CUSTOM_OPP_PATH": str(external_vendor),
                    "ASCEND_OPP_PATH": str(external_vendor.parent.parent),
                },
                clear=True,
            ):
                runtime_globals = _load_runtime_without_legacy_extension(runtime_path)
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    r"packaged FLA NPU op_api library",
                ):
                    runtime_globals["load_ascendc_opapi_libraries"]()

if __name__ == "__main__":
    unittest.main()
