#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci" / "manage_npu_ci_stage_report.py"


class ManageNpuCiStageReportTest(unittest.TestCase):
    def _run(self, output: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "CI_ACCURACY_PLATFORM": "a5",
                "CI_SOC": "ascend950",
                "CI_ACCURACY_HEAD_SHA": "a" * 40,
                "CI_ACCURACY_RUN_ID": "42",
                "CI_ACCURACY_RUN_ATTEMPT": "1",
            }
        )
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--output", str(output), *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_records_independent_stage_results_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "stages.json"
            self.assertEqual(self._run(output, "init").returncode, 0)
            self.assertEqual(
                self._run(
                    output,
                    "update",
                    "--stage",
                    "environment-contracts",
                    "--status",
                    "success",
                    "--exit-code",
                    "0",
                ).returncode,
                0,
            )
            self.assertEqual(
                self._run(
                    output,
                    "update",
                    "--stage",
                    "opp-package",
                    "--status",
                    "failure",
                    "--exit-code",
                    "2",
                    "--reason",
                    "compiler failed",
                ).returncode,
                0,
            )
            self.assertEqual(self._run(output, "finalize").returncode, 1)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "npu-ci-stage-report-v1")
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["status"], "failure")
            self.assertEqual(payload["metadata"]["platform"], "a5")
            self.assertEqual(payload["stages"]["environment-contracts"]["status"], "success")
            self.assertEqual(payload["stages"]["opp-package"]["exit_code"], 2)
            self.assertEqual(payload["stages"]["standalone-layout"]["status"], "not_run")

    def test_rejects_inconsistent_success_exit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "stages.json"
            self.assertEqual(self._run(output, "init").returncode, 0)
            completed = self._run(
                output,
                "update",
                "--stage",
                "opp-package",
                "--status",
                "success",
                "--exit-code",
                "1",
            )
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
