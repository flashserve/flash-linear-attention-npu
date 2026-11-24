#!/usr/bin/env python3
"""Contract test for the documentation-only accuracy report marker."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "summarize_npu_ci", ROOT / "ci" / "summarize_npu_ci.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DocumentationAccuracyReportTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
    def test_documentation_pipeline_preserves_report_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "platform": "a5", "soc": "ascend950", "head_sha": "a" * 40,
                "run_id": "42", "run_attempt": "1",
            }
            env = {
                **os.environ,
                "CI_CHANGED_FILES": (
                    "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd/README.md\n"
                    "tests/atk/chunk_gated_delta_rule_bwd/README.md"
                ),
                "CI_MODE": "quick",
                "CI_OPS": "",
                "CI_ACCURACY_PLATFORM": metadata["platform"],
                "CI_SOC": metadata["soc"],
                "CI_ACCURACY_HEAD_SHA": metadata["head_sha"],
                "CI_ACCURACY_RUN_ID": metadata["run_id"],
                "CI_ACCURACY_RUN_ATTEMPT": metadata["run_attempt"],
                "CI_STAGE_REPORT_FILE": str(root / "stages.json"),
                "CI_ACCURACY_REPORT_FILE": str(root / "accuracy.json"),
                "CI_STAGE_TIMING_FILE": "",
            }
            completed = subprocess.run(
                ["bash", "ci/run_checks.sh"], cwd=ROOT, env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((root / "accuracy.json").read_text(encoding="utf-8"))
            stages = json.loads((root / "stages.json").read_text(encoding="utf-8"))
            self.assertEqual(report["metadata"], metadata)
            self.assertEqual(report["metadata"], stages["metadata"])
            self.assertEqual(report["classification"], "docs-only")
            self.assertEqual(report["status"], "skipped")
            self.assertEqual(stages["status"], "success")
            accuracy, error = MODULE.load_accuracy_report(root / "accuracy.json", True)
            self.assertIsNone(error)
            self.assertEqual(accuracy["status"], "success")

    def test_explicit_documentation_skip_satisfies_required_report(self):
        report = {
            "schema": "gdr-accuracy-report-v1",
            "complete": True,
            "status": "skipped",
            "classification": "docs-only",
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "not_run": 0,
                "accuracy_total": 0,
                "accuracy_passed": 0,
                "accuracy_failed": 0,
                "accuracy_not_run": 0,
            },
            "cases": [],
        }
        path = ROOT / ".ci-tmp" / "test-docs-only-accuracy-report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
        try:
            accuracy, error = MODULE.load_accuracy_report(path, True)
        finally:
            path.unlink(missing_ok=True)
        self.assertIsNone(error)
        self.assertEqual(accuracy["status"], "success")
        self.assertEqual(accuracy["total"], 0)


if __name__ == "__main__":
    unittest.main()
