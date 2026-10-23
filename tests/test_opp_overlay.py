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
        return runpy.run_path(str(PARITY))

    @staticmethod
    def _report() -> dict:
        verify = runpy.run_path(str(REPO_ROOT / "scripts/verify_stable_only_results.py"))
        tensor = {"tensor": True, "shape": [1], "dtype": "torch.float32",
                  "sha256": "0" * 64, "finite": True, "requires_grad": False,
                  "version": 0, "device": "npu:0", "stride": [1], "storage_offset": 0}
        inputs = {"kind": "tuple", "items": [tensor]}
        return {
            "format": 2, "script": PARITY.name, "script_sha256": "same-script",
            "torch": "2.7.1", "public_ops": ["npu_chunk_fwd_o"], "exit_code": 0,
            "arguments": ["run", "npu_chunk_fwd_o"],
            "signatures_before": {"npu_chunk_fwd_o": "(q)"},
            "signatures_after": {"npu_chunk_fwd_o": "(q)"},
            "calls": [{"op": "npu_chunk_fwd_o", "inputs": inputs,
                       "input_sha256": verify["digest_description"](inputs),
                       "outputs": tensor.copy(), "inputs_after": inputs, "finite": True}],
        }

    def _compare(self, left: dict, right: dict) -> int:
        module = self._module()
        with tempfile.TemporaryDirectory() as temp_dir:
            left_path, right_path = Path(temp_dir) / "left.json", Path(temp_dir) / "right.json"
            left_path.write_text(json.dumps(left), encoding="utf-8")
            right_path.write_text(json.dumps(right), encoding="utf-8")
            with mock.patch.dict(os.environ, {"FLA_NPU_PARITY_OPS": "npu_chunk_fwd_o"}, clear=True):
                return module["compare"](str(left_path), str(right_path))

    def test_both_stable_runs_compare_signatures_inputs_and_outputs(self) -> None:
        self.assertEqual(self._compare(self._report(), self._report()), 0)
        for field, value in (("shape", [1, 1]), ("dtype", "torch.float16"),
                             ("sha256", "1" * 64), ("finite", False)):
            with self.subTest(field=field):
                right = self._report()
                right["calls"][0]["outputs"][field] = value
                self.assertEqual(self._compare(self._report(), right), 1)

    def test_errors_empty_runs_and_legacy_backend_reports_fail(self) -> None:
        for report in ({"backend_env": "ctypes", "operators": {}},
                       dict(self._report(), calls=[]), dict(self._report(), exit_code=1)):
            with self.subTest(report=report):
                self.assertEqual(self._compare(self._report(), report), 1)
        right = self._report()
        right["calls"][0]["error"] = {"type": "RuntimeError", "message": "injected"}
        self.assertEqual(self._compare(self._report(), right), 1)

    def test_dump_records_exact_recipe_selection(self) -> None:
        module = self._module()
        verifier = types.SimpleNamespace(record=mock.Mock(return_value=0))
        with mock.patch.dict(module["dump"].__globals__, {"_verification_tool": lambda: verifier}), \
                mock.patch.dict(os.environ, {"FLA_NPU_PARITY_OPS": "npu_chunk_fwd_o"}, clear=True):
            self.assertEqual(module["dump"]("report.json"), 0)
        verifier.record.assert_called_once_with(Path("report.json"), PARITY, ["run", "npu_chunk_fwd_o"])

    def test_recipe_selection_rejects_unknown_names(self) -> None:
        module = self._module()
        with mock.patch.dict(
            os.environ, {"FLA_NPU_PARITY_OPS": "npu_chunk_fwd_o,npu_missing"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "npu_missing"):
                module["_selected_recipes"]()


if __name__ == "__main__":
    unittest.main()
