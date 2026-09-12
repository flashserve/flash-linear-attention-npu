#!/usr/bin/env python3
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_example_st_cases.py"
SPEC = importlib.util.spec_from_file_location("run_example_st_cases", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunExampleStCasesTest(unittest.TestCase):
    def test_failed_process_before_accuracy_stage_is_not_reported_as_accuracy_failure(self):
        process = mock.Mock()
        process.stdout = iter(["kernel.cpp:7: error: undeclared identifier\n"])
        process.wait.return_value = 2
        case = {"name": "compile-failure", "extra_args": ["--accuracy-check"]}

        with mock.patch.object(runner.subprocess, "Popen", return_value=process):
            report = runner._run_case(
                ["python", "example.py", "--accuracy-check"], Path.cwd(), case
            )

        self.assertIs(report["accuracy_check"], True)
        self.assertEqual(report["accuracy_status"], "not_run")
        self.assertEqual(report["metrics"], [])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["return_code"], 2)

    def test_non_finite_metrics_are_serialized_as_strings(self):
        metric = runner._parse_accuracy_metric(
            "o: finite=False allclose=False cosine_ok=False "
            "cosine=nan max_abs=inf mean_abs=-inf"
        )

        self.assertIsNotNone(metric)
        self.assertEqual(metric["cosine"], "nan")
        self.assertEqual(metric["max_abs"], "inf")
        self.assertEqual(metric["mean_abs"], "-inf")
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            case = {
                "name": "non-finite",
                "status": "failed",
                "return_code": 1,
                "accuracy_check": True,
                "accuracy_status": "failed",
                "metrics": [metric],
            }
            runner._write_accuracy_report(report_path, [case], {}, complete=True)
            report_text = report_path.read_text(encoding="utf-8")
            report = json.loads(report_text, parse_constant=lambda value: self.fail(value))
            self.assertIs(report["complete"], True)
            self.assertNotIn("NaN", report_text)
            self.assertNotIn("Infinity", report_text)

    def test_report_writer_rejects_unhandled_non_finite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            case = {
                "name": "invalid",
                "status": "failed",
                "return_code": 1,
                "accuracy_check": True,
                "accuracy_status": "failed",
                "metrics": [{"cosine": math.nan}],
            }

            with self.assertRaises(ValueError):
                runner._write_accuracy_report(report_path, [case], {}, complete=False)
            self.assertFalse(report_path.exists())

    def test_report_writer_marks_intermediate_report_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            case = {
                "name": "partial",
                "status": "passed",
                "return_code": 0,
                "accuracy_check": True,
                "accuracy_status": "passed",
                "metrics": [{"finite": True, "allclose": True, "cosine_ok": True}],
            }

            runner._write_accuracy_report(report_path, [case], {}, complete=False)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIs(report["complete"], False)


if __name__ == "__main__":
    unittest.main()
