"""Isolation contract tests; no torch, CANN, or NPU execution is required."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


LOADER = Path(__file__).resolve().parents[1] / "fla_npu_experimental" / "_loader.py"


class LoaderTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("_experimental_loader_under_test", LOADER)
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.vendor = self.root / "vendors" / self.loader.VENDOR
        self.custom = self.vendor / "op_api" / "lib" / "libcust_opapi.so"
        self.custom.parent.mkdir(parents=True)
        self.custom.write_bytes(b"test fixture; never loaded")
        self.manifest = {
            "mode": "experimental", "vendor": self.loader.VENDOR,
            "operators": [{"name": "fused_recurrent_rwkv8", "op_type": "FusedRecurrentRwkv8",
                           "aclnn_symbols": ["aclnnFusedRecurrentRwkv8",
                                             "aclnnFusedRecurrentRwkv8GetWorkspaceSize"]}],
        }
        self.write_manifest()
        self.env = mock.patch.dict(os.environ, {
            "FLA_NPU_DISABLE_PTH": "1",
            "FLA_NPU_EXPERIMENTAL_OPP_PATH": str(self.root),
            "ASCEND_CUSTOM_OPP_PATH": str(self.vendor),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.maps = mock.patch.object(self.loader, "_mapped_libraries", return_value=set())
        self.maps.start()
        self.addCleanup(self.maps.stop)
        original_finders = sys.meta_path[:]
        self.addCleanup(lambda: sys.meta_path.__setitem__(slice(None), original_finders))

    def write_manifest(self):
        (self.vendor / self.loader.MANIFEST).write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_configure_is_lazy_and_does_not_import_torch(self):
        before = set(sys.modules)
        with mock.patch.object(self.loader.ctypes, "CDLL", side_effect=AssertionError("must not load")):
            self.loader.configure()
        self.assertEqual(self.loader._CONFIGURATION, (self.root, self.vendor, self.custom))
        self.assertFalse({"torch", "torch_npu", "fla_npu"} & (set(sys.modules) - before))

    def test_requires_pth_disabled(self):
        os.environ.pop("FLA_NPU_DISABLE_PTH")
        with self.assertRaisesRegex(RuntimeError, "FLA_NPU_DISABLE_PTH"):
            self.loader.configure()

    def test_rejects_relative_root(self):
        os.environ["FLA_NPU_EXPERIMENTAL_OPP_PATH"] = "relative/opp"
        with self.assertRaisesRegex(RuntimeError, "absolute"):
            self.loader.configure()

    def test_rejects_mixed_stable_and_experimental_search_path(self):
        stable = self.root / "stable"
        stable.mkdir()
        os.environ["ASCEND_CUSTOM_OPP_PATH"] += os.pathsep + str(stable)
        with self.assertRaisesRegex(RuntimeError, "only the selected experimental"):
            self.loader.configure()

    def test_rejects_missing_search_path(self):
        os.environ.pop("ASCEND_CUSTOM_OPP_PATH")
        with self.assertRaisesRegex(RuntimeError, "ASCEND_CUSTOM_OPP_PATH"):
            self.loader.configure()

    def test_rejects_stable_manifest(self):
        for key, value in (("mode", "stable"), ("vendor", "fla_npu_transformer")):
            with self.subTest(key=key):
                original = self.manifest[key]
                self.manifest[key] = value
                self.write_manifest()
                with self.assertRaisesRegex(RuntimeError, "mode/vendor"):
                    self.loader.configure()
                self.manifest[key] = original

    def test_requires_operator_and_both_aclnn_symbols(self):
        for entries in ([], [{"name": "fused_recurrent_rwkv8", "op_type": "FusedRecurrentRwkv8",
                              "aclnn_symbols": ["aclnnFusedRecurrentRwkv8"]}]):
            with self.subTest(entries=entries):
                self.manifest["operators"] = entries
                self.write_manifest()
                with self.assertRaises(RuntimeError):
                    self.loader.configure()

    def test_requires_opapi_file(self):
        self.custom.unlink()
        with self.assertRaises(FileNotFoundError):
            self.loader.configure()

    def test_rejects_symlink_escape(self):
        outside = self.root / "stable-opapi.so"
        outside.write_bytes(b"outside")
        self.custom.unlink()
        try:
            self.custom.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.loader.configure()

    def test_rejects_stable_loaded_first(self):
        with mock.patch.dict(sys.modules, {"fla_npu": mock.Mock()}):
            with self.assertRaisesRegex(RuntimeError, "separate Python processes"):
                self.loader.configure()

    def test_rejects_npu_runtime_loaded_first(self):
        with mock.patch.dict(sys.modules, {"torch_npu": mock.Mock()}):
            with self.assertRaisesRegex(RuntimeError, "before torch_npu"):
                self.loader.configure()

    def test_rejects_native_runtime_loaded_first(self):
        with mock.patch.object(self.loader, "_mapped_libraries", return_value={Path("/sdk/libascendcl.so")}):
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                self.loader.configure()

    def test_rejects_foreign_custom_library(self):
        with mock.patch.object(self.loader, "_mapped_libraries", return_value={Path("/stable/libcust_opapi.so")}):
            with self.assertRaisesRegex(RuntimeError, "different custom OPP"):
                self.loader.configure()

    def test_rejects_stable_import_after_configuration(self):
        self.loader.configure()
        with self.assertRaisesRegex(ImportError, "separate process"):
            importlib.util.find_spec("fla_npu")

    def test_rechecks_environment_before_each_call(self):
        self.loader.configure()
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(self.root)
        with self.assertRaisesRegex(RuntimeError, "environment changed"):
            self.loader.validate_active_environment()

    def test_rejects_native_stable_launcher(self):
        with mock.patch.object(self.loader, "_mapped_libraries", return_value={Path("/wheel/libfla_npu_stable.so")}):
            with self.assertRaisesRegex(RuntimeError, "Stable launcher"):
                self.loader.configure()


    def test_accepts_another_operator_identity(self):
        self.manifest["operators"] = [{"name": "fixture_probe",
            "op_type": "FixtureProbe", "aclnn_symbols": [
                "aclnnFixtureProbe", "aclnnFixtureProbeGetWorkspaceSize"]}]
        self.write_manifest()
        self.loader.configure()

    def test_rejects_duplicate_operator(self):
        self.manifest["operators"] *= 2
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, "unique"):
            self.loader.configure()


if __name__ == "__main__":
    unittest.main()
