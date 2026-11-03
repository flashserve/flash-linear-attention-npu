#!/usr/bin/env python3
"""Exercise ATK's CI entry without importing ATK, Torch, or touching an NPU."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("taskset"), "Linux CPU affinity test")
class AtkCiEntryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.atk_root = self.root / "tests/atk"
        self.op_dir = self.atk_root / "demo"
        self.op_dir.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / "tests/atk/run_test_cpu.sh", self.atk_root / "run_test_cpu.sh")
        (self.op_dir / "executor_demo.py").write_text("# Fixture, ATK is mocked.\n")
        (self.atk_root / "common").mkdir()
        (self.atk_root / "common/check_atk_result.py").write_text(
            "import os, sys\nassert '--require-report' in sys.argv\nsys.exit(int(os.environ.get('ATK_RESULT_EXIT', '0')))\n"
        )
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.log = self.root / "calls.jsonl"
        atk = self.bin_dir / "atk"
        atk.write_text("#!" + sys.executable + "\n" + r'''
import json, os, sys
if sys.argv[1:] == ["--version"]:
    print("ATK 26.8.8")
    raise SystemExit(0)
with open(os.environ["ATK_TEST_LOG"], "a") as stream:
    stream.write(json.dumps({"argv": sys.argv[1:], "cpus": sorted(os.sched_getaffinity(0)),
                             "threads": {key: os.environ.get(key) for key in
                                ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_THREAD_LIMIT"]}}) + "\n")
raise SystemExit(int(os.environ.get("ATK_TEST_EXIT", "0")))
''')
        atk.chmod(0o755)
        self.write_cases()

    def write_cases(self, accuracy=200, perf=1, mss=1):
        for suffix, count in (("", accuracy), ("_perf", perf), ("_mss", mss)):
            path = self.op_dir / f"atk_demo{suffix}.json"
            path.write_text(json.dumps([{"id": index, "name": "demo"} for index in range(count)]))

    def run_entry(self, *extra, **overrides):
        environment = os.environ.copy()
        for key in ("CASE_START", "CASE_END", "ACCURACY_START", "ACCURACY_END", "ATK_ENV", "ATK_BIN", "CANN_ENV", "FLA_NPU_ENV", "FLA_NPU_OPP_ENV"):
            environment.pop(key, None)
        environment.update({"PATH": str(self.bin_dir) + os.pathsep + os.environ["PATH"],
                            "ATK_TEST_LOG": str(self.log), "ATK_OUTPUT_ROOT": str(self.root / "output"),
                            "OMP_NUM_THREADS": "128", "OPENBLAS_NUM_THREADS": "128"})
        environment.update(overrides)
        return subprocess.run(["bash", str(self.atk_root / "run_test_cpu.sh"), "-op=demo",
                               "-scope=accuracy", "-ci=true", "-soc=A2", "-npu_device_id=3", *extra],
                              cwd=self.root, env=environment, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_complete_assets_run_all_accuracy_with_real_cpu_limit(self):
        result = self.run_entry()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("accuracy: expected >= 200, actual 200", result.stdout)
        call, = self.calls()
        argv = call["argv"]
        self.assertEqual(argv[argv.index("--max_task") + 1], "16")
        self.assertEqual(argv[argv.index("--bind_cpu_type") + 1], "0")
        self.assertEqual(argv[argv.index("--devices") + 1], "3")
        self.assertEqual(argv[argv.index("--task") + 1], "accuracy")
        self.assertNotIn("-s", argv)
        self.assertNotIn("-e", argv)
        self.assertGreater(len(call["cpus"]), 0)
        self.assertLessEqual(len(call["cpus"]), 16)
        self.assertEqual(set(call["threads"].values()), {"1"})

    def test_insufficient_accuracy_reports_actual_and_advice_before_atk(self):
        self.write_cases(accuracy=199)
        result = self.run_entry()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("accuracy: expected >= 200, actual 199", result.stderr)
        self.assertIn("修改建议", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_missing_perf_and_empty_mss_report_both_assets(self):
        self.write_cases(mss=0)
        (self.op_dir / "atk_demo_perf.json").unlink()
        result = self.run_entry()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("perf: expected >= 1, actual 0", result.stderr)
        self.assertIn("mss: expected >= 1, actual 0", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_duplicate_ids_or_wrong_json_cannot_satisfy_count(self):
        for payload in ([{"id": 1}] * 200, {"cases": [{"id": n} for n in range(200)]}):
            with self.subTest(payload_type=type(payload).__name__):
                (self.op_dir / "atk_demo.json").write_text(json.dumps(payload))
                result = self.run_entry()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("accuracy: expected >= 200, actual 0", result.stderr)
                self.assertEqual(self.calls(), [])

    def test_ci_rejects_case_ranges_and_other_scopes(self):
        for args, settings in (([], {"CASE_START": "0", "CASE_END": "1"}),
                               ([], {"ACCURACY_START": "0"}), (["-scope=all"], {})):
            with self.subTest(args=args, settings=settings):
                result = self.run_entry(*args, **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("建议", result.stderr)
                self.assertEqual(self.calls(), [])

    def test_atk_command_and_result_failures_include_reproduction(self):
        for setting in ("ATK_TEST_EXIT", "ATK_RESULT_EXIT"):
            with self.subTest(setting=setting):
                result = self.run_entry(**{setting: "7"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("算子 demo ATK 精度", result.stderr)
                self.assertIn("建议", result.stderr)
                self.assertIn("-npu_device_id=3", result.stderr)

    def test_ci_uses_fresh_output_on_each_invocation(self):
        first = self.run_entry()
        second = self.run_entry()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        paths = [call["argv"][call["argv"].index("--output_path") + 1] for call in self.calls()]
        self.assertEqual(len(set(paths)), 2)

    def test_missing_result_checker_fails_closed(self):
        (self.atk_root / "common/check_atk_result.py").unlink()
        result = self.run_entry()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("缺少结果检查器", result.stderr)

    def test_missing_atk_tells_ci_admin_to_upgrade_image(self):
        result = self.run_entry(ATK_BIN=str(self.root / "missing-atk"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CI 镜像中没有 atk", result.stderr)
        self.assertIn("SHA256 重建", result.stderr)

    def test_real_result_checker_rejects_missing_report_in_ci(self):
        # No xlsx is read here; keep this regression independent of openpyxl installation.
        (self.bin_dir / "openpyxl.py").write_text("# Empty fixture, no workbook may be read.\n")
        environment = {**os.environ, "PYTHONPATH": str(self.bin_dir)}
        command = [sys.executable, str(REPO_ROOT / "tests/atk/common/check_atk_result.py"),
                   "--type", "accuracy", "--op", "demo", "--output-root", str(self.root / "absent")]
        optional = subprocess.run(command, env=environment, capture_output=True, text=True)
        required = subprocess.run([*command, "--require-report"], env=environment, capture_output=True, text=True)
        self.assertEqual(optional.returncode, 0, optional.stderr)
        self.assertNotEqual(required.returncode, 0)
        self.assertIn("未找到报告", required.stderr)
        self.assertIn("建议", required.stderr)

    def test_wrapper_only_invokes_public_entry_sequentially(self):
        ci_dir = self.root / "ci"
        ci_dir.mkdir()
        shutil.copyfile(REPO_ROOT / "ci/run_atk_checks.sh", ci_dir / "run_atk_checks.sh")
        (self.atk_root / "run_test_cpu.sh").write_text('printf "%s\\n" "$*" >> "$ATK_TEST_LOG"\n')
        environment = {**os.environ, "CI_ATK_OPS": "first,second", "NPU_DEVICE_ID": "2", "SOC": "ascend950",
                       "ATK_TEST_LOG": str(self.log)}
        result = subprocess.run(["bash", str(ci_dir / "run_atk_checks.sh")], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = self.log.read_text().splitlines()
        self.assertEqual(lines, [f"-op={op} -scope=accuracy -ci=true -npu_device_id=2 -soc=ascend950" for op in ("first", "second")])
        self.log.unlink()
        for invalid in ("first,", "../bad", "first,,second"):
            with self.subTest(invalid=invalid):
                result = subprocess.run(["bash", str(ci_dir / "run_atk_checks.sh")],
                                        env={**environment, "CI_ATK_OPS": invalid}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
