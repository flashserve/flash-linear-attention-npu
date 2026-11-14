"""Run post-failure diagnostics independently of the interrupted build step."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
STEPS = WORKFLOW["jobs"]["ascend-npu"]["steps"]


class LogCollectionTest(unittest.TestCase):
    def test_full_logs_are_archived_before_cleanup_even_if_summary_fails(self):
        summary = next(i for i, step in enumerate(STEPS) if step.get("id") == "npu_ci_summary")
        upload = next(i for i, step in enumerate(STEPS)
                      if step.get("with", {}).get("name", "").startswith("npu-ci-logs-"))
        cleanup = next(i for i, step in enumerate(STEPS) if step["name"].startswith("清理 "))
        self.assertLess(summary, upload)
        self.assertLess(upload, cleanup)
        self.assertEqual(STEPS[summary]["if"], "always() && steps.npu_ci.outcome != 'skipped'")
        self.assertEqual(STEPS[upload]["if"],
                         "always() && needs.prepare.outputs.debug == 'true' && steps.npu_ci.outcome != 'skipped'")
        paths = STEPS[upload]["with"]["path"].splitlines()
        self.assertEqual(paths, ["repo/${{ env.CI_RAW_LOG_FILE }}", "repo/dist/fla-npu-build.log"])
        self.assertTrue(STEPS[upload]["with"]["include-hidden-files"])
        self.assertEqual(STEPS[upload]["with"]["retention-days"], 7)

    def test_debug_is_opt_in(self):
        inputs = WORKFLOW.get("on", WORKFLOW.get(True))["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["debug"]["type"], "boolean")
        self.assertIs(inputs["debug"]["default"], False)
        self.assertEqual(WORKFLOW["jobs"]["prepare"]["outputs"]["debug"],
                         "${{ steps.prepare.outputs.debug }}")

    def test_existing_quiet_pip_log_preserves_early_backend_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text(
                '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n',
                encoding="utf-8",
            )
            (root / "backend.py").write_text(
                "from pathlib import Path\n"
                "def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):\n"
                "    directory = Path(metadata_directory) / 'ci_log_fixture-1.0.dist-info'\n"
                "    directory.mkdir()\n"
                "    (directory / 'METADATA').write_text('Metadata-Version: 2.1\\nName: ci-log-fixture\\nVersion: 1.0\\n')\n"
                "    return directory.name\n"
                "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
                "    print('fixture.cpp:7: error: ORIGINAL_BACKEND_ERROR', flush=True)\n"
                "    for index in range(1000): print('later build output', index, flush=True)\n"
                "    raise RuntimeError('fixture build failed')\n",
                encoding="utf-8",
            )
            log = root / "fla-npu-build.log"
            command = [sys.executable, "-m", "pip", "wheel", "--no-index",
                       "--disable-pip-version-check", "--no-build-isolation", "--no-deps",
                       "--log", str(log), ".", "-w", str(root / "dist")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            raw = log.read_text(encoding="utf-8")
            self.assertIn("ORIGINAL_BACKEND_ERROR", raw)
            self.assertNotIn("ORIGINAL_BACKEND_ERROR", "\n".join(raw.splitlines()[-40:]))
            self.assertTrue(all(f"later build output {i}\n" in raw for i in range(1000)))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell test")
    def test_failure_cancellation_and_missing_outputs_still_produce_diagnostics(self):
        summary = next(step for step in STEPS if step.get("id") == "npu_ci_summary")
        for outcome, exit_code, expected in (
            ("failure", "2", 2), ("failure", "", 1),
            ("cancelled", "", 130), ("cancelled", "0", 130),
            ("success", "0", 0),
        ):
            with self.subTest(outcome=outcome, exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "ci").mkdir()
                shutil.copy(ROOT / "ci/summarize_npu_ci.py", root / "ci/summarize_npu_ci.py")
                (root / ".ci-tmp").mkdir()
                raw = root / ".ci-tmp/raw.log"
                raw.write_text("source.cpp:7: error: original compiler diagnostic\n", encoding="utf-8")
                env = dict(os.environ, NPU_CI_OUTCOME=outcome, NPU_CI_EXIT_CODE=exit_code,
                           CI_RAW_LOG_FILE=".ci-tmp/raw.log", CI_OPS="", CI_MODE="quick",
                           CI_ACCURACY_REPORT_FILE=".ci-tmp/accuracy.json",
                           CI_STAGE_REPORT_FILE=".ci-tmp/stages.json",
                           CI_DIAGNOSTIC_FILE=".ci-tmp/diagnostic.json",
                           CI_SUMMARY_FILE=".ci-tmp/summary.md", CI_ACCURACY_PLATFORM="a5",
                           CI_SOC="ascend950", CI_ACCURACY_HEAD_SHA="a" * 40,
                           CI_ACCURACY_RUN_ID="42", CI_ACCURACY_RUN_ATTEMPT="1",
                           GITHUB_STEP_SUMMARY=str(root / "step-summary.md"))
                result = subprocess.run(["bash", "-c", summary["run"]], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=10)
                # Even a successful command must fail closed if its reports are missing.
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                report = json.loads((root / ".ci-tmp/diagnostic.json").read_text())
                self.assertEqual(report["status"], "failure")
                self.assertEqual(report["execution"]["exit_code"], expected)
                self.assertIn("original compiler diagnostic", json.dumps(report))
                self.assertEqual((root / "step-summary.md").read_text(),
                                 (root / ".ci-tmp/summary.md").read_text())
                self.assertTrue(raw.exists())
                if outcome == "cancelled":
                    self.assertIn("cancelled", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
