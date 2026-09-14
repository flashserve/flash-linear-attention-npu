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
    def _required_case(self, name):
        cases = runner._read_cases(SCRIPT_PATH.parents[1] / "ci" / "example_st_cases.json")
        return next(case for case in cases if case["name"] == name)

    @staticmethod
    def _metric_line(tensor, *, tol, cos_min):
        return (
            f"{tensor}: finite=True allclose=True tol={tol} cosine=1.0 "
            f"cos_min={cos_min} cosine_ok=True max_abs=0.0 bad_frac=0.0\n"
        )

    def test_case_report_records_required_shape_contract(self):
        case = {
            "name": "required-case",
            "B": 1,
            "T": 128,
            "chunk_size": 64,
            "query_head": 2,
            "value_head": 2,
            "Kdim": 128,
            "Vdim": 128,
            "dtype": "FP16",
            "varlen": True,
            "extra_args": ["--cu-seqlens", "0,64,128", "--accuracy-check"],
        }

        report = runner._blank_case_report(case, "not_run")

        self.assertEqual(
            report["contract"],
            {
                "script": "examples/flash_gated_delta_rule.py",
                "batch": 1,
                "tokens": 128,
                "chunk_size": 64,
                "query_heads": 2,
                "value_heads": 2,
                "key_dim": 128,
                "value_dim": 128,
                "dtype": "fp16",
                "varlen": True,
                "cu_seqlens": [0, 64, 128],
                "mean_len": 1024,
                "gate_source": "g",
                "gate_function": "logsigmoid",
                "initial_state": "none",
                "output_final_state": False,
                "qk_l2norm": True,
                "demo_model": False,
                "conv_kernel": 4,
                "seed": 42,
                "scale": None,
                "accuracy_tensors": ["o", "dq", "dk", "dv", "dbeta", "dg"],
                "accuracy_thresholds": {
                    "output_tol": 5e-3,
                    "grad_tol": 8e-3,
                    "beta_grad_tol": 2e-2,
                    "gate_grad_tol": 2e-2,
                    "output_cos_min": 0.999,
                    "grad_cos_min": 0.999,
                    "beta_grad_cos_min": 0.99,
                    "gate_grad_cos_min": 0.99,
                },
            },
        )

    def test_managed_case_arguments_cannot_be_overridden_from_extra_args(self):
        base_case = {
            "name": "case1_current_default",
            "script": "examples/flash_gated_delta_rule.py",
            "B": 1,
            "T": 4087,
            "dtype": "bf16",
        }

        for override in (
            "--tokens",
            "--dtype=fp16",
            "--no-varlen",
            "--device",
            "--legacy-unfused-core",
            "--accuracy-output-tol=1000000",
        ):
            with self.subTest(override=override):
                case = {**base_case, "extra_args": [override, "1"]}
                with self.assertRaisesRegex(ValueError, "runner-managed option"):
                    runner._build_command(Path.cwd(), 0, case)

        for abbreviation in ("--tok", "--no-var", "--cu-s"):
            with self.subTest(abbreviation=abbreviation):
                case = {**base_case, "extra_args": [abbreviation, "1"]}
                with self.assertRaisesRegex(ValueError, "abbreviate protected option"):
                    runner._build_command(Path.cwd(), 0, case)

    def test_accuracy_command_uses_fixed_thresholds_once(self):
        case = self._required_case("case1_current_default")

        command = runner._build_command(SCRIPT_PATH.parents[1], 0, case)

        for option, value in runner.ACCURACY_THRESHOLDS.items():
            with self.subTest(option=option):
                self.assertEqual(command.count(option), 1)
                index = command.index(option)
                self.assertEqual(command[index + 1], str(value))

    def test_required_accuracy_case_must_use_gdr_example(self):
        case = {
            "name": "case1_current_default",
            "script": "examples/other.py",
        }

        with self.assertRaisesRegex(ValueError, "must use"):
            runner._build_command(Path.cwd(), 0, case)

    def test_required_accuracy_case_contract_is_fixed(self):
        case = json.loads(
            json.dumps(self._required_case("case1_current_default"))
        )
        case["T"] = 4086

        with self.assertRaisesRegex(ValueError, "contract mismatch: tokens"):
            runner._build_command(SCRIPT_PATH.parents[1], 0, case)

    def test_required_report_invariants_change_runner_result_to_failure(self):
        case = self._required_case("gdr_accuracy_dense_b2_t128_h2_d128_fp16")
        complete_lines = ["accuracy check:\n"]
        for tensor in runner.DEFAULT_ACCURACY_TENSORS:
            tol_field, cos_min_field = runner.METRIC_THRESHOLD_FIELDS[tensor]
            complete_lines.append(
                self._metric_line(
                    tensor,
                    tol=runner.ACCURACY_THRESHOLD_CONTRACT[tol_field],
                    cos_min=runner.ACCURACY_THRESHOLD_CONTRACT[cos_min_field],
                )
            )
        complete_lines.append("accuracy check passed\n")

        variants = {
            "missing": complete_lines[:6] + [complete_lines[-1]],
            "duplicate": complete_lines[:-1] + [complete_lines[1], complete_lines[-1]],
            "threshold": [
                line.replace("tol=0.005", "tol=1.0") if line.startswith("o:") else line
                for line in complete_lines
            ],
            "failed-check": [
                line.replace("allclose=True", "allclose=False")
                if line.startswith("o:")
                else line
                for line in complete_lines
            ],
            "missing-check": [
                line.replace(" cosine_ok=True", "") if line.startswith("o:") else line
                for line in complete_lines
            ],
        }
        for name, lines in variants.items():
            with self.subTest(name=name):
                process = mock.Mock()
                process.stdout = iter(lines)
                process.wait.return_value = 0
                with mock.patch.object(runner.subprocess, "Popen", return_value=process):
                    with mock.patch("builtins.print"):
                        report = runner._run_case(
                            ["python", "example.py", "--accuracy-check"],
                            SCRIPT_PATH.parents[1],
                            case,
                        )
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["return_code"], 2)
                self.assertEqual(report["accuracy_status"], "failed")
                self.assertTrue(report["validation_errors"])

    def test_default_run_fails_when_required_case_is_missing_or_disabled(self):
        all_cases = runner._read_cases(
            SCRIPT_PATH.parents[1] / "ci" / "example_st_cases.json"
        )
        variants = {
            "missing": all_cases[1:],
            "disabled": [
                {**case, "enabled": False}
                if case["name"] == "case1_current_default"
                else case
                for case in all_cases
            ],
        }
        for name, cases in variants.items():
            with self.subTest(name=name):
                with mock.patch.object(runner, "_read_cases", return_value=cases):
                    with mock.patch(
                        "sys.argv",
                        ["run_example_st_cases.py", "--device", "0", "--dry-run"],
                    ):
                        with mock.patch("builtins.print"):
                            self.assertEqual(runner.main(), 2)

    def test_cu_seqlens_must_be_unique(self):
        case = {
            "name": "custom-case",
            "extra_args": [
                "--cu-seqlens",
                "0,64,128",
                "--cu-seqlens=0,128",
            ],
        }

        with self.assertRaisesRegex(ValueError, "at most once"):
            runner._blank_case_report(case, "not_run")

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

    def test_zero_exit_without_accuracy_pass_marker_fails_runner_case(self):
        process = mock.Mock()
        process.stdout = iter(["accuracy check:\n"])
        process.wait.return_value = 0
        case = {"name": "custom-accuracy", "extra_args": ["--accuracy-check"]}

        with mock.patch.object(runner.subprocess, "Popen", return_value=process):
            with mock.patch("builtins.print"):
                report = runner._run_case(
                    ["python", "example.py", "--accuracy-check"], Path.cwd(), case
                )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["return_code"], 2)
        self.assertEqual(report["accuracy_status"], "failed")
        self.assertIn("accuracy check did not complete", report["validation_errors"][0])

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
