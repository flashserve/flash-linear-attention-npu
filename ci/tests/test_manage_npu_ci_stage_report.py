#!/usr/bin/env python3

import csv
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
    def _run(self, output: Path, *args: str, table: Path | None = None,
             run_id: str = "42") -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("CI_STAGE_TIMING_FILE", None)
        if table is not None:
            env["CI_STAGE_TIMING_FILE"] = str(table)
        env.update(
            {
                "CI_ACCURACY_PLATFORM": "a5",
                "CI_SOC": "ascend950",
                "CI_ACCURACY_HEAD_SHA": "a" * 40,
                "CI_ACCURACY_RUN_ID": run_id,
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

    def test_duration_preserves_first_running_timestamp(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "stages.json"
            self.assertEqual(self._run(output, "init").returncode, 0)
            for _ in range(2):
                self.assertEqual(self._run(output, "update", "--stage", "environment-contracts",
                                           "--status", "running").returncode, 0)
                current = json.loads(output.read_text())["stages"]["environment-contracts"]
                if _ == 0:
                    first_start = current["started_at_monotonic_ns"]
                self.assertEqual(current["started_at_monotonic_ns"], first_start)
            self.assertEqual(self._run(output, "update", "--stage", "environment-contracts",
                                       "--status", "failure", "--exit-code", "7").returncode, 0)
            stage = json.loads(output.read_text())["stages"]["environment-contracts"]
            self.assertGreater(stage["duration_seconds"], 0)
            self.assertTrue(stage["finished_at"])

    @unittest.skipUnless(os.name == "posix", "persistent table uses Linux flock")
    def test_table_keeps_history_and_failed_run_without_finalize(self):
        with tempfile.TemporaryDirectory() as temp:
            table = Path(temp) / "stage-timings.csv"
            output = Path(temp) / "stages.json"
            for run_id in ("41", "42"):
                self.assertEqual(self._run(output, "init", table=table, run_id=run_id).returncode, 0)
            for status, extra in (("running", ()), ("failure", ("--exit-code", "3"))):
                self.assertEqual(self._run(output, "update", "--stage", "opp-package",
                                           "--status", status, *extra, table=table).returncode, 0)
            self.assertEqual(self._run(output, "update", "--stage", "gdr-example-st",
                                       "--status", "skipped", table=table).returncode, 0)
            output.unlink()
            with table.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertNotIn("scoped-overlay", {row["stage"] for row in rows})
            failure = next(row for row in rows if row["run_id"] == "42" and row["stage"] == "opp-package")
            self.assertEqual(failure["status"], "failure")
            self.assertGreater(float(failure["duration_seconds"]), 0)
            skipped = next(row for row in rows if row["run_id"] == "42" and row["stage"] == "gdr-example-st")
            self.assertEqual(skipped["duration_seconds"], "")

    @unittest.skipUnless(os.name == "posix", "persistent table uses Linux flock")
    def test_parallel_runs_do_not_overwrite_each_other(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as temp:
            table = Path(temp) / "stage-timings.csv"
            def initialize_run(index):
                return self._run(Path(temp) / f"{index}.json", "init", table=table,
                                 run_id=str(index)).returncode
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(list(pool.map(initialize_run, range(8))), [0] * 8)
            with table.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 40)
            self.assertEqual(len({(row["run_id"], row["stage"]) for row in rows}), 40)


if __name__ == "__main__":
    unittest.main()
