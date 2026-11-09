#!/usr/bin/env python3
"""Contract test for the documentation-only accuracy report marker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "summarize_npu_ci", ROOT / "ci" / "summarize_npu_ci.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DocumentationAccuracyReportTest(unittest.TestCase):
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
        import json

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
