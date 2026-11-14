"""Manual validation checks real results and stops on failures."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("experimental_validation", ROOT / "experimental/_tests/validate.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
spec = importlib.util.spec_from_file_location("experimental_incremental", ROOT / "experimental/_tests/incremental.py")
incremental = importlib.util.module_from_spec(spec)
spec.loader.exec_module(incremental)


class ManualValidation(unittest.TestCase):
    def test_formal_gdr_library_paths_do_not_leak_into_experimental_processes(self):
        paths = ["/cann/lib64", "/site/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib",
                 "/site/fla_npu/opp/vendors/fla_npu_overlay_test/op_api/lib", "/driver/lib64"]
        with patch.dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(paths)):
            env = validation.clean_environment({"mode": "stable", "scenario": "stable"}, {})
        self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep), [paths[0], paths[-1]])

    def execute(self, soc="ascend910b", fail=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = validation.Validation(root, soc=soc)
            calls = []

            def action(name):
                calls.append(name)
                if fail == name:
                    raise RuntimeError("injected missing/failed result")
                return {"checked": name}

            with patch.object(runner, "dependencies", lambda: action("dependencies")), \
                    patch.object(runner, "build", lambda: action("build")), \
                    patch.object(runner, "isolation", lambda: action("isolation")), \
                    patch.object(runner, "accuracy", lambda mode: action(mode)), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = runner.execute()
            return code, calls, json.loads((root / "summary.json").read_text())

    def test_a2_runs_both_launch_modes_after_isolation(self):
        code, calls, result = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["dependencies", "build", "isolation", "queue", "inline"])
        self.assertTrue(result["complete"])

    def test_a5_explicitly_does_not_claim_accuracy(self):
        code, calls, result = self.execute("ascend950")
        self.assertEqual((code, calls), (0, ["dependencies", "build"]))
        self.assertEqual(result["phases"]["accuracy-queue"]["status"], "skipped")

    def test_failure_prevents_following_checks_and_is_recorded(self):
        for phase in ("dependencies", "build", "isolation", "queue", "inline"):
            with self.subTest(phase=phase):
                code, calls, result = self.execute(fail=phase)
                self.assertEqual(code, 1)
                self.assertEqual(calls[-1], phase)
                self.assertFalse(result["complete"])
                self.assertEqual(result["status"], "failure")

    def test_unknown_soc_cannot_silently_skip_npu_suite(self):
        code, calls, result = self.execute("unknown")
        self.assertEqual((code, calls, result["status"]), (1, [], "failure"))

    def test_missing_short_or_failed_reference_report_rejected(self):
        valid = {"complete": True, "results": [{"index": i, "status": "PASS"} for i in range(3)],
                 "case_indices": [0, 1, 2], "found": True, "all_pass": True, "total": 3, "check_pass": 3,
                 "exec_fail": 0, "check_fail": 0}
        self.assertEqual(validation.check_accuracy(valid, 3), valid)
        for change in ({"complete": False}, {"results": []}, {"case_indices": [0, 1, 1]},
                       {"results": [{"index": 0, "status": "ERROR"}] * 3},
                       {"found": False}, {"total": 0}, {"total": 2}, {"check_pass": 2},
                       {"exec_fail": 1}, {"check_fail": 1}, {"all_pass": False}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validation.check_accuracy({**valid, **change}, 3)


class IncrementalValidation(unittest.TestCase):
    def test_discovery_does_not_import_the_manager_or_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            package = site / "fla_npu_experimental"
            package.mkdir()
            for path in (site / "fla_npu_experimental_manage.py", package / "__init__.py"):
                path.write_text("raise RuntimeError('discovery must not import this module')\n")
            env = dict(os.environ, PYTHONPATH=str(site), FLA_NPU_DISABLE_PTH="1",
                       TORCH_DEVICE_BACKEND_AUTOLOAD="0")
            self.assertEqual(incremental.installed_base(env), package.resolve())

    def test_all_stages_and_failure_cleanup_use_the_discovered_base(self):
        for fail_install in (False, True):
            with self.subTest(fail_install=fail_install), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = root / "selected/fla_npu_experimental"
                other = root / "other/fla_npu_experimental"
                for path in (package, other):
                    path.mkdir(parents=True)
                    (path / "base.txt").write_text("unchanged")
                output = root / "results"
                management = []
                exercises = []

                def run(command, *, env, **kwargs):
                    self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(package.parent))
                    self.assertEqual(env["FLA_NPU_EXPERIMENTAL_OPP_PATH"], str(package / "opp"))
                    if command[0] == sys.executable:
                        self.assertEqual(command[command.index("--package-dir") + 1], str(package))
                        management.append(command)
                        if fail_install and "--source" in command:
                            raise RuntimeError("injected installation failure")
                        return SimpleNamespace(stdout=json.dumps({"state": "active"}))
                    self.assertEqual(command[4], str(package / "opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash"))
                    report = Path(command[-1])
                    name = report.stem
                    exercises.append(name)
                    report.write_text(json.dumps({"complete": True, "check_pass": 3, "all_pass": True}))
                    trace = Path(env["FLA_A2_TRACE_DIR"])
                    trace.mkdir()
                    overlay = name not in ("disabled", "restored")
                    (trace / "trace.json").write_text(json.dumps({
                        "wrapper_module": "_fla_npu_experimental_layer" if name in ("wheel", "wheel-plus-run") else "fla_npu_experimental",
                        "custom_opapi": "/wheel_layers/probe.so" if overlay else "/base/probe.so"}))

                with patch.object(incremental, "installed_base", return_value=package), \
                     patch.object(incremental.subprocess, "run", side_effect=run):
                    if fail_install:
                        with self.assertRaisesRegex(RuntimeError, "injected installation failure"):
                            incremental.validate(root / "update.whl", root / "update.run", output)
                    else:
                        incremental.validate(root / "update.whl", root / "update.run", output)
                        self.assertEqual(exercises, ["wheel", "wheel-plus-run", "disabled", "restored", "run-only"])
                self.assertIn("--remove-overlay", management[-1])
                self.assertEqual((other / "base.txt").read_text(), "unchanged")

    def test_cli_rejects_an_external_package_target(self):
        result = subprocess.run([sys.executable, str(ROOT / "experimental/_tests/incremental.py"),
                                 "--package", "external", "--wheel", "update.whl", "--run", "update.run",
                                 "--output", "unused"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --package", result.stderr)


if __name__ == "__main__":
    unittest.main()
