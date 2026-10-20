from __future__ import annotations

import json
import os
import runpy
import shutil
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


def _minimal_vendor(path: Path) -> None:
    library = path / "op_api" / "lib" / "libcust_opapi.so"
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_bytes(b"test")


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

    def _compare(self, left: dict, right: dict, *, allow_incomplete: bool = False) -> int:
        module = self._module()
        with tempfile.TemporaryDirectory() as temp_dir:
            left_path = Path(temp_dir) / "left.json"
            right_path = Path(temp_dir) / "right.json"
            left_path.write_text(json.dumps(left), encoding="utf-8")
            right_path.write_text(json.dumps(right), encoding="utf-8")
            env = {"FLA_NPU_PARITY_ALLOW_INCOMPLETE": "1"} if allow_incomplete else {}
            with mock.patch.dict(os.environ, env, clear=True):
                return module["compare"](str(left_path), str(right_path))

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
