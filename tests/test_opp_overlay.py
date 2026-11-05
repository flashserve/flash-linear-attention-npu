from __future__ import annotations

import json
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


REPO_ROOT = Path(__file__).resolve().parents[1]
FLA_SETUP = REPO_ROOT / "torch_custom" / "fla_npu"
INSTALL_OPP = FLA_SETUP / "fla_npu" / "install_opp.py"
OPP_ENV = FLA_SETUP / "fla_npu_opp_env.py"
PARITY = FLA_SETUP / "test" / "test_stable_parity.py"
ABI = runpy.run_path(str(FLA_SETUP / "fla_npu" / "_op_abi.py"))


def _minimal_vendor(path: Path) -> None:
    library = path / "op_api" / "lib" / "libcust_opapi.so"
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_bytes(b"test")
    _write_aclnn_header(path, "int64_t")


def _write_aclnn_header(vendor: Path, scalar_type: str) -> None:
    header = vendor / "op_api" / "include" / "aclnnop" / "aclnn_demo.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        "aclnnStatus aclnnDemoGetWorkspaceSize(\n"
        "    const aclTensor *x,\n"
        f"    {scalar_type} mode,\n"
        "    uint64_t *workspaceSize,\n"
        "    aclOpExecutor **executor);\n"
        "aclnnStatus aclnnDemo(void *workspace, uint64_t workspaceSize,\n"
        "    aclOpExecutor *executor, aclrtStream stream);\n",
        encoding="utf-8",
    )
    with mock.patch.dict(ABI["write_manifest"].__globals__, {
        "exported_symbols": lambda _: {"aclnnDemo", "aclnnDemoGetWorkspaceSize"}
    }):
        ABI["write_manifest"](vendor)


class OverlayInstallerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = runpy.run_path(str(INSTALL_OPP))

    def _tree(self, root: Path) -> tuple[Path, Path, Path]:
        opp_root = root / "package" / "opp"
        vendors = opp_root / "vendors"
        base = vendors / "fla_npu_transformer"
        source = root / "source" / "fla_npu_transformer"
        _minimal_vendor(base)
        _minimal_vendor(source)
        self.module["install_overlay"].__globals__["PACKAGE_OPP_ROOT"] = opp_root
        return vendors, base, source

    def test_clean_install_and_remove_use_documented_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            vendors, base, source = self._tree(Path(temp_dir))
            installed = self.module["install_overlay"](source)
            self.assertEqual(installed.name, "fla_npu_overlay_transformer")
            self.assertTrue(base.is_dir())
            self.assertTrue(self.module["remove_overlay"]())
            self.assertTrue(base.is_dir())
            self.assertFalse(installed.exists())
            self.assertEqual(
                (vendors / "config.ini").read_text(encoding="utf-8"),
                "load_priority=fla_npu_transformer\n",
            )

    def test_existing_custom_overlay_does_not_change_default_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            vendors, _, source = self._tree(Path(temp_dir))
            _minimal_vendor(vendors / "fla_npu_overlay_custom")
            installed = self.module["install_overlay"](source)
            self.assertEqual(installed.name, "fla_npu_overlay_transformer")

    def test_rejects_path_and_config_injection_names(self) -> None:
        invalid = (
            "../fla_npu_transformer",
            "fla_npu_transformer",
            "fla_npu_overlay_",
            "fla_npu_overlay_bad/name",
            "fla_npu_overlay_bad,name",
            "fla_npu_overlay_bad\nload_priority=evil",
            ".",
            "..",
        )
        validate = self.module["_validate_overlay_name"]
        for name in invalid:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Refusing"):
                validate(name)

    def test_config_writer_ignores_unsafe_manual_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            vendors, _, _ = self._tree(Path(temp_dir))
            (vendors / "fla_npu_overlay_bad,name").mkdir()
            with self.assertWarnsRegex(RuntimeWarning, "unsafe name"):
                order = self.module["_write_vendors_config"](vendors)
            self.assertEqual(order, ["fla_npu_transformer"])
            self.assertNotIn(",name", (vendors / "config.ini").read_text(encoding="utf-8"))

    def test_matching_public_aclnn_abi_can_be_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            _, base, source = self._tree(Path(temp_dir))
            _write_aclnn_header(base, "int64_t")
            _write_aclnn_header(source, "int64_t")
            installed = self.module["install_overlay"](source)
            self.assertTrue(installed.is_dir())

    def test_public_aclnn_abi_drift_is_rejected_before_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            vendors, base, source = self._tree(Path(temp_dir))
            _write_aclnn_header(base, "int64_t")
            _write_aclnn_header(source, "bool")
            with self.assertRaisesRegex(RuntimeError, "public aclnn ABI differs"):
                self.module["install_overlay"](source)
            self.assertFalse((vendors / "fla_npu_overlay_transformer").exists())

    def test_failed_replacement_preserves_previous_overlay_base_and_config(self) -> None:
        for failure in ("copy", "staged_abi", "set_env", "move_old", "move_new", "config"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp_dir, \
                    mock.patch.dict(os.environ, {}, clear=True):
                vendors, base, source = self._tree(Path(temp_dir))
                install = self.module["install_overlay"]
                target = install(source)
                (target / "revision").write_text("old")
                (source / "revision").write_text("new")
                snapshot = lambda path: {p.relative_to(path): p.read_bytes()
                                         for p in path.rglob("*") if p.is_file()}
                old_overlay, old_base = snapshot(target), snapshot(base)
                config = vendors / "config.ini"
                old_config = config.read_bytes()
                real_rename = Path.rename
                real_validate = install.__globals__["_validate_overlay_abi"]

                def fail_copy(src, dst):
                    Path(dst).mkdir()
                    (Path(dst) / "partial").write_text("incomplete")
                    self.assertEqual(snapshot(target), old_overlay)
                    raise OSError("injected copy failure")

                def validate(path):
                    if path.name == "new":
                        raise OSError("injected staged_abi failure")
                    return real_validate(path)

                def rename(path, destination):
                    if ((failure == "move_old" and path == target)
                            or (failure == "move_new" and path.name == "new")):
                        raise OSError("injected rename failure")
                    return real_rename(path, destination)

                def fail_config(_):
                    config.write_bytes(b"partial config")
                    raise OSError("injected config failure")

                replacements = {
                    "staged_abi": {"_validate_overlay_abi": validate},
                    "set_env": {"_write_set_env": mock.Mock(side_effect=OSError("injected set_env failure"))},
                    "config": {"_write_vendors_config": fail_config},
                }.get(failure, {})
                with mock.patch.dict(install.__globals__, replacements), \
                        mock.patch.object(Path, "rename", rename):
                    if failure == "copy":
                        with mock.patch.object(shutil, "copytree", side_effect=fail_copy), \
                                self.assertRaisesRegex(OSError, "injected"):
                            install(source, force=True)
                    else:
                        with self.assertRaisesRegex(OSError, "injected"):
                            install(source, force=True)
                self.assertEqual(snapshot(target), old_overlay)
                self.assertEqual(snapshot(base), old_base)
                self.assertEqual(config.read_bytes(), old_config)
                self.assertEqual(list(vendors.parent.glob(".overlay-*")), [])

    def test_successful_replacement_discovers_only_final_vendor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            vendors, _, source = self._tree(Path(temp_dir))
            install = self.module["install_overlay"]
            target = install(source)
            (target / "old-only").touch()
            (source / "new-only").touch()
            real_write = install.__globals__["_write_set_env"]

            def observe_staging(path):
                self.assertEqual(self.module["overlay_vendor_names"](), [target.name])
                self.assertNotEqual(path.parent, vendors)
                real_write(path)

            with mock.patch.dict(install.__globals__, {"_write_set_env": observe_staging}):
                install(source, force=True)
            self.assertTrue((target / "new-only").is_file())
            self.assertFalse((target / "old-only").exists())
            self.assertEqual(list(vendors.parent.glob(".overlay-*")), [])

    def test_first_install_failure_restores_absent_config_and_vendor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            vendors, _, source = self._tree(Path(temp_dir))
            install = self.module["install_overlay"]
            config = vendors / "config.ini"

            def fail_config(_):
                config.write_text("partial")
                raise OSError("injected config failure")

            with mock.patch.dict(install.__globals__, {"_write_vendors_config": fail_config}), \
                    self.assertRaisesRegex(OSError, "injected"):
                install(source)
            self.assertFalse((vendors / "fla_npu_overlay_transformer").exists())
            self.assertFalse(config.exists())

    def test_failed_rollback_preserves_recovery_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            vendors, _, source = self._tree(Path(temp_dir))
            install = self.module["install_overlay"]
            target = install(source)
            (target / "old-only").touch()
            old_config = (vendors / "config.ini").read_bytes()
            real_rename = Path.rename

            def rename(path, destination):
                if path.name == "previous":
                    raise OSError("cannot restore backup")
                return real_rename(path, destination)

            with mock.patch.dict(install.__globals__, {
                    "_write_vendors_config": mock.Mock(side_effect=OSError("config failure"))}), \
                    mock.patch.object(Path, "rename", rename), \
                    self.assertRaisesRegex(RuntimeError, "recovery files preserved"):
                install(source, force=True)
            recovery, = vendors.parent.glob(".overlay-*")
            self.assertTrue((recovery / "previous/old-only").is_file())
            self.assertEqual((recovery / "config.ini").read_bytes(), old_config)

    @unittest.skipUnless(os.name == "posix", "Linux interprocess management lock")
    def test_management_lock_rejects_competing_remove(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            vendors, _, source = self._tree(Path(temp_dir))
            target = self.module["install_overlay"](source)
            script = (f"import runpy; from pathlib import Path; m=runpy.run_path({str(INSTALL_OPP)!r}); "
                      f"m['remove_overlay'].__globals__['PACKAGE_OPP_ROOT']=Path({str(vendors.parent)!r}); "
                      "m['remove_overlay']()")
            with self.module["_overlay_update_lock"]():
                result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                        text=True, timeout=10, env=dict(os.environ))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another overlay install/remove is in progress", result.stderr)
            self.assertTrue(target.is_dir())


class EarlyOppEnvironmentTest(unittest.TestCase):
    def test_custom_overlay_is_prepared_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_root = Path(temp_dir)
            module_path = site_root / "fla_npu_opp_env.py"
            shutil.copy2(OPP_ENV, module_path)
            vendors = site_root / "fla_npu" / "opp" / "vendors"
            overlay = vendors / "fla_npu_overlay_custom"
            base = vendors / "fla_npu_transformer"
            overlay.mkdir(parents=True)
            base.mkdir()
            (vendors / "config.ini").write_text(
                "load_priority=fla_npu_overlay_custom,fla_npu_transformer\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                runpy.run_path(str(module_path))
                paths = os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)
            self.assertEqual(
                paths,
                [
                    str(overlay.resolve()),
                    str(base.resolve()),
                ],
            )

    def test_explicit_empty_overlay_uses_only_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_root = Path(temp_dir)
            module_path = site_root / "fla_npu_opp_env.py"
            shutil.copy2(OPP_ENV, module_path)
            vendors = site_root / "fla_npu" / "opp" / "vendors"
            overlay = vendors / "fla_npu_overlay_transformer"
            base = vendors / "fla_npu_transformer"
            overlay.mkdir(parents=True)
            base.mkdir()
            environment = {
                "FLA_NPU_OVERLAY_VENDOR": "",
                "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(
                    [str(overlay.resolve()), str(base.resolve())]
                ),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                with self.assertWarnsRegex(RuntimeWarning, "normalized package-local"):
                    runpy.run_path(str(module_path))
                paths = os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)
            self.assertEqual(paths, [str(base.resolve())])

    def test_existing_package_paths_are_reordered_and_external_order_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_root = Path(temp_dir)
            module_path = site_root / "fla_npu_opp_env.py"
            shutil.copy2(OPP_ENV, module_path)
            opp_root = site_root / "fla_npu" / "opp"
            vendors = opp_root / "vendors"
            overlay = vendors / "fla_npu_overlay_custom"
            base = vendors / "fla_npu_transformer"
            external_a = site_root / "external-a"
            external_b = site_root / "external-b"
            overlay.mkdir(parents=True)
            base.mkdir()
            external_a.mkdir()
            external_b.mkdir()
            (vendors / "config.ini").write_text(
                "load_priority=fla_npu_overlay_custom,fla_npu_transformer\n",
                encoding="utf-8",
            )
            existing = os.pathsep.join(
                [
                    str(base.resolve()),
                    str(external_a.resolve()),
                    str(overlay.resolve()),
                    str(opp_root.resolve()),
                    str(overlay.resolve() / "op_api" / "lib"),
                    str(external_b.resolve()),
                ]
            )
            with mock.patch.dict(
                os.environ, {"ASCEND_CUSTOM_OPP_PATH": existing}, clear=True
            ):
                with self.assertWarnsRegex(RuntimeWarning, "normalized package-local"):
                    runpy.run_path(str(module_path))
                paths = os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)
            self.assertEqual(
                paths,
                [
                    str(overlay.resolve()),
                    str(base.resolve()),
                    str(external_a.resolve()),
                    str(external_b.resolve()),
                ],
            )


class StableParityContractTest(unittest.TestCase):
    def _module(self) -> dict:
        torch = types.ModuleType("torch")
        torch.Tensor = type("Tensor", (), {})
        ascendc = types.ModuleType("fla_npu.ops.ascendc")
        ascendc._ASCENDC_OPS = ("npu_test",)
        ops = types.ModuleType("fla_npu.ops")
        ops.ascendc = ascendc
        package = types.ModuleType("fla_npu")
        package.ops = ops
        with mock.patch.dict(
            sys.modules,
            {
                "torch": torch,
                "fla_npu": package,
                "fla_npu.ops": ops,
                "fla_npu.ops.ascendc": ascendc,
            },
        ):
            return runpy.run_path(str(PARITY))

    @staticmethod
    def _report(backend: str, *, entry: str, status: str = "ok", **output) -> dict:
        record = {"entry": entry, "status": status, "outputs": [output] if output else []}
        if status != "ok":
            record["error"] = "injected"
        return {
            "backend_env": backend,
            "launcher_usable": backend == "stable",
            "published_ops": ["npu_test"],
            "operators": {"npu_test": record},
        }

    def _compare(self, left: dict, right: dict, *, allow_incomplete: bool = False, versions: bool = False) -> int:
        module = self._module()
        with tempfile.TemporaryDirectory() as temp_dir:
            left_path = Path(temp_dir) / "left.json"
            right_path = Path(temp_dir) / "right.json"
            left_path.write_text(json.dumps(left), encoding="utf-8")
            right_path.write_text(json.dumps(right), encoding="utf-8")
            env = {"FLA_NPU_PARITY_ALLOW_INCOMPLETE": "1"} if allow_incomplete else {}
            with mock.patch.dict(os.environ, env, clear=True):
                return module["compare"](str(left_path), str(right_path), versions=versions)

    def test_versions_require_distinct_real_packages_and_stable_results(self) -> None:
        output = dict(shape=[1], dtype="float32", sha256="same-output")
        left = self._report("stable", entry="stable", **output)
        right = self._report("stable", entry="stable", **output)
        left.update(revision="base", package_source_sha256="a" * 64)
        right.update(revision="candidate", package_source_sha256="b" * 64)
        self.assertEqual(self._compare(left, right, versions=True), 0)
        for field in ("revision", "package_source_sha256"):
            invalid = dict(right, **{field: left[field]})
            self.assertEqual(self._compare(left, invalid, versions=True), 1)
        bad = json.loads(json.dumps(right))
        bad["operators"]["npu_test"]["fallbacks"] = 1
        self.assertEqual(self._compare(left, bad, versions=True), 1)
        bad["operators"]["npu_test"]["fallbacks"] = 0
        bad["operators"]["npu_test"]["outputs"][0]["sha256"] = "different"
        self.assertEqual(self._compare(left, bad, versions=True), 1)
        bad["operators"]["npu_test"]["status"] = "error"
        self.assertEqual(self._compare(left, bad, versions=True, allow_incomplete=True), 1)

    def test_default_rejects_same_backend_and_errors(self) -> None:
        output = {"shape": [1], "dtype": "torch.float32", "sha256": "same", "sum": 1.0}
        left = self._report("ctypes", entry="ctypes", **output)
        same_backend = self._report("stable", entry="ctypes", **output)
        errored = self._report("stable", entry="stable", status="error")
        self.assertEqual(self._compare(left, same_backend), 1)
        self.assertEqual(self._compare(left, errored), 1)
        self.assertEqual(self._compare(left, same_backend, allow_incomplete=True), 0)

    def test_compares_shape_dtype_and_values(self) -> None:
        base = {"shape": [1], "dtype": "torch.float32", "sha256": "same", "sum": 1.0}
        left = self._report("ctypes", entry="ctypes", **base)
        self.assertEqual(self._compare(left, self._report("stable", entry="stable", **base)), 0)
        for changed in (
            dict(base, shape=[1, 1]),
            dict(base, dtype="torch.float16"),
            dict(base, sha256="different"),
        ):
            with self.subTest(changed=changed):
                self.assertEqual(
                    self._compare(left, self._report("stable", entry="stable", **changed)),
                    1,
                )

    def test_recipe_selection_rejects_unknown_names(self) -> None:
        module = self._module()
        with mock.patch.dict(
            os.environ, {"FLA_NPU_PARITY_OPS": "npu_chunk_fwd_o,npu_missing"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "npu_missing"):
                module["_selected_recipes"]()


if __name__ == "__main__":
    unittest.main()
