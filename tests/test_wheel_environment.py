from __future__ import annotations

import csv
import importlib
import os
import runpy
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


def _load_setup() -> tuple[dict[str, object], dict[str, object]]:
    setup_kwargs: dict[str, object] = {}

    def capture_setup(**kwargs) -> None:
        setup_kwargs.update(kwargs)

    with mock.patch.object(setuptools, "setup", side_effect=capture_setup):
        setup_globals = runpy.run_path(str(REPO_ROOT / "setup.py"))
    return setup_globals, setup_kwargs


def _load_run_package_finalizer() -> dict[str, object]:
    return runpy.run_path(
        str(
            REPO_ROOT
            / "scripts"
            / "package"
            / "ops_transformer"
            / "scripts"
            / "finalize_wheel_opp.py"
        )
    )


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

    def test_package_build_always_starts_from_clean_outputs(self) -> None:
        setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
        build_source = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")

        self.assertNotIn("FLA_NPU_INCREMENTAL_BUILD", setup_source)
        self.assertNotIn("FLA_NPU_OPS", setup_source)
        self.assertNotIn("FLA_NPU_SKIP_RUN_BUILD", setup_source)
        self.assertNotIn("FLA_NPU_SKIP_RUN_INSTALL", setup_source)
        self.assertNotIn("--incremental", build_source)
        self.assertIn("set_env\n\nclean\nclean_build_out", build_source)

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
[[ "${{LD_LIBRARY_PATH}}" == "{vendor_dir}/op_api/lib" ]]
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

    def test_run_package_overlay_finalization_is_idempotent(self) -> None:
        finalize_wheel_opp = _load_run_package_finalizer()["finalize_wheel_opp"]

        with tempfile.TemporaryDirectory() as temp_dir:
            site_root = Path(temp_dir) / "site-packages"
            package_dir = site_root / "fla_npu"
            vendor_dir = (
                package_dir / "opp" / "vendors" / "fla_npu_transformer"
            )
            _create_minimal_vendor(vendor_dir, include_alias=True)
            config = package_dir / "opp" / "vendors" / "config.ini"
            config.write_text("load_priority=fla_npu_transformer\n", encoding="utf-8")

            dist_info = site_root / "flash_linear_attention_npu-1.0.dist-info"
            dist_info.mkdir(parents=True)
            record = dist_info / "RECORD"
            with record.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(
                    [
                        ["fla_npu/__init__.py", "", ""],
                        ["fla_npu/opp/stale-from-older-overlay.json", "", ""],
                        [f"{dist_info.name}/RECORD", "", ""],
                    ]
                )

            finalize_wheel_opp(package_dir)
            first_record = record.read_bytes()
            first_set_env = (vendor_dir / "bin" / "set_env.bash").read_bytes()
            finalize_wheel_opp(package_dir)

            self.assertEqual(record.read_bytes(), first_record)
            self.assertEqual(
                (vendor_dir / "bin" / "set_env.bash").read_bytes(),
                first_set_env,
            )
            self.assertTrue(
                (vendor_dir / "op_api" / "lib" / "libcust_opapi.so").is_file()
            )
            self.assertFalse(
                (vendor_dir / "op_api" / "lib" / "libopapi.so").exists()
            )

            with record.open("r", encoding="utf-8", newline="") as handle:
                recorded_paths = {row[0] for row in csv.reader(handle) if row}
            current_opp_paths = {
                path.relative_to(site_root).as_posix()
                for path in (package_dir / "opp").rglob("*")
                if path.is_file() or path.is_symlink()
            }
            self.assertTrue(current_opp_paths.issubset(recorded_paths))
            self.assertNotIn(
                "fla_npu/opp/stale-from-older-overlay.json",
                recorded_paths,
            )
            self.assertIn("fla_npu/__init__.py", recorded_paths)

            set_env = vendor_dir / "bin" / "set_env.bash"
            script = f"""
set -euo pipefail
unset ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH FLA_NPU_OPP_PATH FLA_NPU_OP_API_LIB
source {set_env!s}
source {set_env!s}
[[ "${{ASCEND_CUSTOM_OPP_PATH}}" == "{vendor_dir.parent.parent}:{vendor_dir}" ]]
[[ "${{LD_LIBRARY_PATH}}" == "{vendor_dir}/op_api/lib" ]]
[[ "${{FLA_NPU_OPP_PATH}}" == "{vendor_dir.parent.parent}" ]]
[[ "${{FLA_NPU_OP_API_LIB}}" == "{vendor_dir}/op_api/lib/libcust_opapi.so" ]]
"""
            subprocess.run(["bash", "-c", script], check=True)

            dynamic_dir = (
                vendor_dir
                / "op_impl"
                / "ai_core"
                / "tbe"
                / "fla_npu_transformer_impl"
                / "dynamic"
            )
            dynamic_dir.mkdir(parents=True)
            source = dynamic_dir / "sample.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            bytecode = dynamic_dir / "__pycache__" / "sample.cpython-311.pyc"
            bytecode.parent.mkdir()
            bytecode.write_bytes(b"generated-by-pip")
            finalize_wheel_opp(package_dir)

            bytecode_relative = bytecode.relative_to(site_root).as_posix()
            with record.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            for row in rows:
                if row and row[0] == bytecode_relative:
                    row[1:] = ["", ""]
            with record.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(rows)

            checker = runpy.run_path(
                str(REPO_ROOT / "scripts" / "check_install_workflows.py")
            )
            checker["_assert_record_covers_opp"](package_dir)
            self.assertNotIn(
                bytecode_relative,
                checker["_manifest_from_directory"](package_dir),
            )

    def test_both_run_package_installers_finalize_wheel_overlay(self) -> None:
        installers = (
            REPO_ROOT / "cmake" / "scripts" / "custom" / "install.sh",
            REPO_ROOT
            / "scripts"
            / "package"
            / "ops_transformer"
            / "scripts"
            / "install.sh",
        )
        for installer in installers:
            with self.subTest(installer=installer):
                source = installer.read_text(encoding="utf-8")
                self.assertIn('finalize_wheel_opp "${wheel_opp_root}"', source)
                self.assertNotIn(
                    '"${dst_vendor}/op_api/lib/libopapi.so"',
                    source,
                )

        custom_build = (REPO_ROOT / "cmake" / "custom_build.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("finalize_wheel_opp.py", custom_build)

    def test_import_requires_cann_environment(self) -> None:
        runtime_path = (
            REPO_ROOT
            / "torch_custom"
            / "fla_npu"
            / "fla_npu"
            / "__init__.py"
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CANN environment is not initialized"):
                runpy.run_path(str(runtime_path))

    def test_import_loads_runtime_and_removes_stale_libopapi_alias(self) -> None:
        runtime_path = (
            REPO_ROOT
            / "torch_custom"
            / "fla_npu"
            / "fla_npu"
            / "__init__.py"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = (
                Path(temp_dir)
                / "opp"
                / "vendors"
                / "fla_npu_transformer"
            )
            _create_minimal_vendor(vendor_dir, include_alias=True)

            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_HOME_PATH": "/fake/cann",
                    "FLA_NPU_OPP_PATH": str(vendor_dir),
                },
                clear=True,
            ):
                with mock.patch("ctypes.CDLL", return_value=mock.sentinel.custom_opapi):
                    with self.assertWarnsRegex(RuntimeWarning, "Removed a stale"):
                        runtime_globals = runpy.run_path(str(runtime_path))
                    first = runtime_globals["load_ascendc_opapi_libraries"]()
                    second = runtime_globals["load_ascendc_opapi_libraries"]()
                    self.assertIs(first, second)
                    self.assertEqual(first, [mock.sentinel.custom_opapi])
            self.assertFalse(
                (vendor_dir / "op_api" / "lib" / "libopapi.so").exists()
            )

if __name__ == "__main__":
    unittest.main()
