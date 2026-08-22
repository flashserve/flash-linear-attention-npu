from __future__ import annotations

import csv
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
            self.assertFalse((opp_root.parent / "_lib").exists())

            custom_build = (REPO_ROOT / "cmake" / "custom_build.cmake").read_text(
                encoding="utf-8"
            )
            symbol_build = (REPO_ROOT / "cmake" / "symbol.cmake").read_text(
                encoding="utf-8"
            )
            self.assertIn("NO_SONAME ON", custom_build)
            self.assertIn("NO_SONAME ON", symbol_build)

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
            self.assertFalse((package_dir / "_lib").exists())
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
[[ -z "${{LD_LIBRARY_PATH-}}" ]]
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

    def test_legacy_extension_rpath_keeps_standard_vendor_layout(self) -> None:
        setup_source = (
            REPO_ROOT / "torch_custom" / "fla_npu" / "setup.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("-Wl,-rpath,$ORIGIN/_lib", setup_source)
        self.assertIn(
            "-Wl,-rpath,$ORIGIN/opp/vendors/fla_npu_transformer/op_api/lib",
            setup_source,
        )

    def test_public_environment_diagnostic_recognizes_packaged_opapi(self) -> None:
        collector = runpy.run_path(str(REPO_ROOT / "scripts" / "collect_public_env.py"))
        has_op_api = collector["_has_op_api"]

        with tempfile.TemporaryDirectory() as temp_dir:
            packaged_opapi = (
                Path(temp_dir)
                / "fla_npu"
                / "opp"
                / "vendors"
                / "fla_npu_transformer"
                / "op_api"
                / "lib"
                / "libcust_opapi.so"
            )
            packaged_opapi.parent.mkdir(parents=True)
            packaged_opapi.write_bytes(b"packaged-test")
            with mock.patch.dict(
                os.environ,
                {"FLA_NPU_OP_API_LIB": str(packaged_opapi)},
                clear=True,
            ):
                self.assertTrue(has_op_api(Path(temp_dir) / "empty-opp"))

    def test_import_requires_cann_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CANN environment is not initialized"):
                runpy.run_path(str(RUNTIME_SOURCE))

    def test_import_loads_runtime_and_removes_stale_libopapi_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, vendor_dir, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )
            stale_alias = vendor_dir / "op_api" / "lib" / "libopapi.so"
            stale_alias.parent.mkdir(parents=True, exist_ok=True)
            stale_alias.write_bytes(b"unsafe")
            external_vendor = Path(temp_dir) / "cann" / "vendors" / "other"
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
                    del mode
                    if str(path) == "libopapi.so":
                        return mock.sentinel.cann_opapi
                    return mock.sentinel.custom_opapi

                with mock.patch("ctypes.CDLL", side_effect=fake_cdll) as cdll:
                    with self.assertWarnsRegex(RuntimeWarning, "Removed a stale"):
                        runtime_globals = runpy.run_path(str(runtime_path))
                    first = runtime_globals["load_ascendc_opapi_libraries"]()
                    second = runtime_globals["load_ascendc_opapi_libraries"]()
                    self.assertIs(first, second)
                    self.assertEqual(
                        first,
                        [mock.sentinel.custom_opapi, mock.sentinel.cann_opapi],
                    )

                    self.assertEqual(len(cdll.call_args_list), 2)
                    cann_call, custom_call = cdll.call_args_list
                    cann_args, cann_kwargs = cann_call
                    custom_args, custom_kwargs = custom_call
                    self.assertEqual(cann_args, ("libopapi.so",))
                    self.assertEqual(
                        custom_args,
                        (str(packaged_opapi),),
                    )
                    expected_mode = (
                        getattr(os, "RTLD_LOCAL", 0)
                        | getattr(os, "RTLD_NOW", 0)
                        | getattr(os, "RTLD_NODELETE", 0)
                    )
                    self.assertEqual(cann_kwargs["mode"], expected_mode)
                    self.assertEqual(custom_kwargs["mode"], expected_mode)
                    self.assertEqual(
                        expected_mode & getattr(os, "RTLD_GLOBAL", 0),
                        0,
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
                    self.assertEqual(
                        os.environ["ASCEND_OPP_PATH"],
                        str(external_vendor.parent.parent),
                    )
                    self.assertNotIn("LD_LIBRARY_PATH", os.environ)
                    self.assertEqual(
                        os.environ["FLA_NPU_OP_API_LIB"],
                        str(packaged_opapi),
                    )
            self.assertFalse(
                (vendor_dir / "op_api" / "lib" / "libopapi.so").exists()
            )

    def test_import_reports_missing_cann_opapi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, _, _ = _create_runtime_package(Path(temp_dir) / "site-packages")

            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_HOME_PATH": "/fake/cann",
                },
                clear=True,
            ):
                with mock.patch("ctypes.CDLL", side_effect=OSError("not found")):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"Unable to load the CANN op_api library.*source.*CANN set_env\.sh",
                    ):
                        runpy.run_path(str(runtime_path))

    def test_import_accepts_packaged_opapi_in_standard_vendor_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, _, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )

            with mock.patch.dict(
                os.environ,
                {"ASCEND_HOME_PATH": "/fake/cann"},
                clear=True,
            ):
                with mock.patch(
                    "ctypes.CDLL",
                    side_effect=(mock.sentinel.cann_opapi, mock.sentinel.custom_opapi),
                ) as cdll:
                    runtime_globals = runpy.run_path(str(runtime_path))
                    self.assertEqual(
                        runtime_globals["load_ascendc_opapi_libraries"](),
                        [mock.sentinel.custom_opapi, mock.sentinel.cann_opapi],
                    )
                    self.assertEqual(
                        Path(cdll.call_args_list[1][0][0]).resolve(),
                        packaged_opapi.resolve(),
                    )

    def test_import_does_not_fallback_to_external_custom_opp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, _, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )
            packaged_opapi.unlink()
            external_vendor = Path(temp_dir) / "cann" / "vendors" / "external"
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
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    r"packaged FLA NPU op_api library",
                ):
                    runpy.run_path(str(runtime_path))

if __name__ == "__main__":
    unittest.main()
