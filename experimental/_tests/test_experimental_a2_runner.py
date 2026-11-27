"""A bad environment, crash, or missing case must never count as isolation success."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("isolation", ROOT / "experimental/_tests/isolation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
probe = runner


class A2RunnerTest(unittest.TestCase):
    def test_negative_requires_expected_error_at_exercise(self):
        case = {"error": "only the selected experimental"}
        result = {"stage": "exercise", "error": {"type": "RuntimeError", "message": "only the selected experimental vendor"}}
        self.assertEqual(runner.evaluate(case, result, 0), "PASS")
        self.assertEqual(runner.evaluate(case, {**result, "stage": "setup"}, 0), "FAIL")

    def test_crash_is_not_a_successful_rejection(self):
        result = {"stage": "exercise", "error": {"type": "RuntimeError", "message": "reject"}}
        for code in (-11, 1, 139):
            self.assertEqual(runner.evaluate({"error": "reject"}, result, code), "FAIL")

    def test_wrong_reason_and_missing_dependency_fail(self):
        for kind, message in (("ModuleNotFoundError", "reject"), ("RuntimeError", "different failure"),
                              ("AssertionError", "reject")):
            self.assertEqual(runner.evaluate({"error": "reject"}, {
                "stage": "exercise", "error": {"type": kind, "message": message}}, 0), "FAIL")

    def test_successful_negative_is_a_failure(self):
        self.assertEqual(runner.evaluate({"error": "reject"}, {"stage": "exercise"}, 0), "FAIL")

    def test_positive_requires_actual_exercise(self):
        self.assertEqual(runner.evaluate({}, {"stage": "exercise"}, 0), "PASS")
        self.assertEqual(runner.evaluate({}, {}, 0), "FAIL")

    def test_catalog_is_complete_and_not_a_unit_fixture(self):
        cases = runner.CASES
        self.assertGreater(len(cases), 0)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        self.assertTrue(runner.MUTATIONS <= {c["scenario"] for c in cases})
        self.assertFalse({"stable", "overlay", "late_disable_pth"} & {c["scenario"] for c in cases})

    def test_experimental_environment_removes_inherited_stable_selection(self):
        with patch.dict(os.environ, FLA_NPU_OP_API_LIBS="bad", FLA_NPU_STABLE_LIB="bad", PYTHONPATH="bad"):
            env = runner.environment({"mode": "experimental", "scenario": "experimental"},
                                     {"experimental_package": "/private/fla_npu_experimental"})
        self.assertNotIn("FLA_NPU_OP_API_LIBS", env)
        self.assertNotIn("FLA_NPU_STABLE_LIB", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["FLA_NPU_DISABLE_PTH"], "1")

    def test_real_pth_case_does_not_disable_hook(self):
        env = runner.environment({"mode": "stable", "scenario": "stable", "pth": "enabled"}, {})
        self.assertNotIn("FLA_NPU_DISABLE_PTH", env)

    def test_only_toolkit_python_paths_survive_environment_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toolkit = root / "cann"
            allowed = toolkit / "python/site-packages"
            outside = root / "source"
            with patch.dict(os.environ, ASCEND_HOME_PATH=str(toolkit),
                            PYTHONPATH=os.pathsep.join([str(outside), str(allowed)])):
                env = runner.environment({"mode": "stable", "scenario": "stable"}, {})
                self.assertEqual(env["PYTHONPATH"], str(allowed))

    def test_snapshot_detects_protected_payload_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "libcust_opapi.so"
            payload.write_bytes(b"original")
            before = runner.snapshot([root])
            payload.write_bytes(b"changed")
            self.assertNotEqual(before, runner.snapshot([root]))
            payload.unlink()
            self.assertNotEqual(before, runner.snapshot([root]))

    def test_deleted_protected_root_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "protected"
            root.mkdir()
            before = runner.snapshot([root])
            root.rmdir()
            after = runner.snapshot([root])
            self.assertNotEqual(before, after)
            self.assertEqual(after[str(root)]["."], {"missing_or_special": True})

    def test_snapshot_records_links_not_just_target_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first", root / "second"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            link = root / "link"
            try:
                link.symlink_to(first)
            except OSError:
                self.skipTest("Creating symlinks is not available")
            before = runner.snapshot([root])
            link.unlink()
            link.symlink_to(second)
            self.assertNotEqual(before, runner.snapshot([root]))

    def test_missing_profile_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_case({"id": "L01", "profile": "experimental_only"}, {}, Path(tmp))
            self.assertEqual(result["status"], "BLOCKED")

    def test_timed_out_negative_case_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = {"id": "L07", "profile": "combined", "scenario": "missing_disable",
                    "mode": "experimental", "error": "FLA_NPU_DISABLE_PTH"}
            config = {"python": sys.executable, "experimental_package": tmp,
                      "stable_package": tmp, "external_experimental_opp": tmp}
            with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 1)):
                result = runner.run_case(case, config, Path(tmp))
            self.assertEqual(result["status"], "FAIL")

    def test_direct_mutation_probe_without_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "opp/vendors/fla_npu_experimental_transformer/op_api/lib/libcust_opapi.so"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"untouched")
            with self.assertRaises(KeyError):
                probe.exercise("corrupt_host", {"experimental_package": str(root)}, {})
            self.assertEqual(library.read_bytes(), b"untouched")
