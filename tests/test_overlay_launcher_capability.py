"""Fail closed before resolving an op when its launcher cannot order vendors."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STABLE = ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py"


class OverlayLauncherCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.environ = mock.patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)
        self.events = []
        self.library = types.SimpleNamespace(
            fla_npu_stable_overlay_order_version=mock.Mock(
                side_effect=lambda: self.events.append("probe") or 1))
        self.op = mock.Mock()
        self.registration = mock.Mock()
        self.registration.npu_test = self.op
        self.torch = types.SimpleNamespace(
            __version__="2.7.1",
            ops=types.SimpleNamespace(
                load_library=lambda path: self.events.append(("load", path)),
                fla_npu_stable=self.registration,
            ),
        )
        # Import this module without running the real package initialization or
        # loading torch/CANN; the package policy has separate exhaustive tests.
        spec = importlib.util.spec_from_file_location("fla_npu.ops.ascendc._stable", STABLE)
        self.stable = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.stable)
        self.resolve_path = self.stable._lib_path
        policy = types.ModuleType("fla_npu._backend_policy")
        policy.validate_backend_environment = lambda: None
        patches = [
            mock.patch.dict(sys.modules, {"torch": self.torch, "fla_npu._backend_policy": policy}),
            mock.patch.object(self.stable, "_init_hot"),
            mock.patch.object(self.stable, "_modules", return_value=(self.torch, None)),
            mock.patch.object(self.stable, "_check_build_stamp",
                              side_effect=lambda path: self.events.append(("stamp", path))),
            mock.patch.object(self.stable, "_lib_path", return_value="packaged-launcher.so"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        os.environ["FLA_NPU_OP_API_LIBS"] = os.pathsep.join(["overlay.so", "base.so"])

    def test_checks_actual_launcher_after_stamp_before_op_resolution(self):
        with mock.patch.object(ctypes, "CDLL", return_value=self.library) as cdll:
            self.assertIs(self.stable._op("npu_test"), self.op)
            self.assertIs(self.stable._op("npu_test"), self.op)
        cdll.assert_called_once_with("packaged-launcher.so")
        self.assertEqual(self.events, [("load", "packaged-launcher.so"),
                                      ("stamp", "packaged-launcher.so"), "probe"])
        self.assertEqual(self.library.fla_npu_stable_overlay_order_version.argtypes, [])
        self.assertIs(self.library.fla_npu_stable_overlay_order_version.restype, ctypes.c_int32)
        self.op.assert_not_called()

    def test_override_uses_actual_path_not_packaged_filename(self):
        os.environ["FLA_NPU_STABLE_LIB"] = "override-launcher.so"
        with mock.patch.object(self.stable, "_lib_path", self.resolve_path), \
                mock.patch.object(ctypes, "CDLL", return_value=self.library) as cdll:
            self.stable.load()
        cdll.assert_called_once_with("override-launcher.so")
        self.assertEqual(self.stable._loaded_path, "override-launcher.so")

    def test_missing_old_broken_or_unloadable_capability_refuses_execution(self):
        cases = [
            types.SimpleNamespace(),
            types.SimpleNamespace(fla_npu_stable_overlay_order_version=mock.Mock(return_value=0)),
            types.SimpleNamespace(fla_npu_stable_overlay_order_version=mock.Mock(return_value=-1)),
            types.SimpleNamespace(fla_npu_stable_overlay_order_version=mock.Mock(side_effect=OSError("bad probe"))),
            OSError("cannot open actual launcher"),
        ]
        for library in cases:
            with self.subTest(library=library):
                kwargs = {"side_effect": library} if isinstance(library, Exception) else {"return_value": library}
                with mock.patch.object(ctypes, "CDLL", **kwargs):
                    with self.assertRaisesRegex(RuntimeError, "Refusing to mix base host") as error:
                        self.stable._op("npu_test")
                self.assertIn("fla_npu_opp_manage", str(error.exception))
                self.assertIsNone(self.stable._loaded_path)
                self.assertEqual(self.stable._OP_CACHE, {})
                self.op.assert_not_called()

    def test_base_only_does_not_require_overlay_export(self):
        for libraries in ("", "base.so", os.pathsep + "base.so" + os.pathsep):
            with self.subTest(libraries=libraries):
                self.stable._loaded_path = None
                os.environ["FLA_NPU_OP_API_LIBS"] = libraries
                with mock.patch.object(ctypes, "CDLL", side_effect=AssertionError("unexpected probe")):
                    self.stable.load()
                self.assertEqual(self.stable._loaded_path, "packaged-launcher.so")

    def test_newer_capability_versions_are_accepted(self):
        self.library.fla_npu_stable_overlay_order_version.side_effect = None
        self.library.fla_npu_stable_overlay_order_version.return_value = 2
        with mock.patch.object(ctypes, "CDLL", return_value=self.library):
            self.stable.load()
        self.assertEqual(self.stable._loaded_path, "packaged-launcher.so")

    def test_bad_stamp_stops_before_capability_probe(self):
        with mock.patch.object(self.stable, "_check_build_stamp", side_effect=RuntimeError("stamp mismatch")), \
                mock.patch.object(ctypes, "CDLL") as cdll:
            with self.assertRaisesRegex(RuntimeError, "stamp mismatch"):
                self.stable._op("npu_test")
        cdll.assert_not_called()
        self.assertEqual(self.stable._OP_CACHE, {})
        self.assertIsNone(self.stable._loaded_path)


if __name__ == "__main__":
    unittest.main()
