"""Regression checks for skipped sanitizer runs and incomplete pipeline coverage."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import openpyxl


PATH = Path(__file__).with_name("check_pipeline.py")
SPEC = importlib.util.spec_from_file_location("check_pipeline", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PipelineGateTests(unittest.TestCase):
    def test_default_assets_are_a2_only(self):
        self.assertEqual(MODULE.MSS_MANIFEST.name, "atk_solve_tri_a2_mss.json")
        self.assertEqual(MODULE.EXECUTOR.name, "executor_solve_tri_a2.py")
        with self.assertRaisesRegex(ValueError, "both pipeline cases"):
            MODULE.pipeline_coverage(MODULE.OP_DIR / "atk_solve_tri_mss.json", 20)

    def test_a5_executor_is_rejected(self):
        cases = json.loads(MODULE.MSS_MANIFEST.read_text(encoding="utf-8"))
        next(case for case in cases if case["id"] == 206)["api_type"] = "executor_solve_tri"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mss.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent A2 executor and SoC"):
                MODULE.pipeline_coverage(path, 20)

    def test_other_soc_is_rejected(self):
        cases = json.loads(MODULE.MSS_MANIFEST.read_text(encoding="utf-8"))
        case = next(case for case in cases if case["id"] == 206)
        field = next(item for item in case["inputs"] if item["name"] == "case_spec")
        spec = json.loads(field["range_values"])
        spec["soc"] = "ascend950"
        field["range_values"] = json.dumps(spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mss.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent A2 executor and SoC"):
                MODULE.pipeline_coverage(path, 20)

    def test_no_active_tool_is_not_a_pass(self):
        text = "\n".join([
            "[mssanitizer] No active sanitizer tool on kernel SolveTri_abc_64.",
            "[mssanitizer] WARN: racecheck is not supported without --cce-enable-sanitizer.",
            "[mssanitizer] Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
        ])
        with self.assertRaisesRegex(ValueError, "skipped"):
            MODULE.check_sanitizer(text, "racecheck", minimum=1)

    def test_missing_kernel_does_not_pass(self):
        with self.assertRaisesRegex(ValueError, "only 0"):
            MODULE.check_sanitizer("Sanitizer finished on kernel Fill_abc. No error detected.", "racecheck")

    def test_both_launches_required(self):
        text = "\n".join([
            "Start racecheck sanitizer on kernel SolveTri_abc_64.",
            "Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
        ])
        with self.assertRaisesRegex(ValueError, "only 1"):
            MODULE.check_sanitizer(text, "racecheck")

    def test_actual_error_does_not_pass(self):
        with self.assertRaisesRegex(ValueError, "reported an error"):
            MODULE.check_sanitizer("[mssanitizer] ERROR: data race on kernel SolveTri_abc_64", "racecheck")

    def test_unrelated_builtin_skip_is_separate(self):
        text = "\n".join([
            "No active sanitizer tool on kernel Fill_abc.",
            "WARN: racecheck is not supported and temporarily ignored.",
            "Sanitizer finished on kernel Fill_abc. No error detected.",
            "Start racecheck sanitizer on kernel SolveTri_abc_64.",
            "Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
            "Start racecheck sanitizer on kernel SolveTri_abc_64.",
            "Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
        ])
        self.assertEqual(MODULE.check_sanitizer(text, "racecheck")["checked_launches"], 2)

    def test_register_warning_invalidates_detection(self):
        text = "\n".join([
            "Checking kernel SolveTri_abc_64.",
            "WARN: FFTS_BASE_ADDR is not initialized.",
            "Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
        ])
        with self.assertRaisesRegex(ValueError, "skipped"):
            MODULE.check_sanitizer(text, "racecheck", minimum=1)

    def test_register_cleanup_warning_fails_even_with_clean_finish(self):
        text = "\n".join([
            "Start memcheck sanitizer on kernel SolveTri_abc_64.",
            "Warning:Register FFTS_BASE_ADDR was not reset to default in block aic(0) on kernel SolveTri_abc_64.",
            "Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
        ])
        with self.assertRaisesRegex(ValueError, "reported an error.*Register FFTS_BASE_ADDR"):
            MODULE.check_sanitizer(text, "memcheck", minimum=1)

    def test_wrong_sanitizer_tool_does_not_pass(self):
        text = "\n".join([
            "Start memcheck sanitizer on kernel SolveTri_abc_64.",
            "Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
        ] * 2)
        with self.assertRaisesRegex(ValueError, "expected racecheck, observed memcheck"):
            MODULE.check_sanitizer(text, "racecheck")

    def test_different_kernel_finishes_cannot_replace_missing_checks(self):
        text = "\n".join([
            "Start racecheck sanitizer on kernel SolveTri_abc_64.",
            "Sanitizer finished on kernel SolveTri_other_64. No error detected.",
        ] * 2)
        with self.assertRaisesRegex(ValueError, "kernel counts do not match"):
            MODULE.check_sanitizer(text, "racecheck")

    def test_kernel_multiplicity_must_match(self):
        text = "\n".join([
            "Start racecheck sanitizer on kernel SolveTri_a_64.",
            "Start racecheck sanitizer on kernel SolveTri_a_64.",
            "Start racecheck sanitizer on kernel SolveTri_b_64.",
            "Sanitizer finished on kernel SolveTri_a_64. No error detected.",
            "Sanitizer finished on kernel SolveTri_b_64. No error detected.",
            "Sanitizer finished on kernel SolveTri_b_64. No error detected.",
        ])
        with self.assertRaisesRegex(ValueError, "kernel counts do not match"):
            MODULE.check_sanitizer(text, "racecheck")

    def test_oob_warning_is_failure_even_with_clean_finish(self):
        text = "\n".join([
            "Start memcheck sanitizer on kernel SolveTri_abc_64.",
            "====== WARNING: out of bounds of size 128",
            "====== at 0x12c04e1f2200 on GM when writing data in SolveTri_abc_64",
            "Sanitizer finished on kernel SolveTri_abc_64. No error detected.",
        ] * 2)
        with self.assertRaisesRegex(ValueError, "reported an error.*out of bounds"):
            MODULE.check_sanitizer(text, "memcheck")

    def test_detected_errors_summary_alone_is_failure(self):
        text = "\n".join([
            "Start memcheck sanitizer on kernel SolveTri_abc_64.",
            "Sanitizer finished on kernel SolveTri_abc_64. See all detected errors above.",
        ] * 2)
        with self.assertRaisesRegex(ValueError, "reported an error.*See all detected errors above"):
            MODULE.check_sanitizer(text, "memcheck")

    def test_current_long_cases_exercise_every_a2_core(self):
        coverage = MODULE.pipeline_coverage(MODULE.MSS_MANIFEST, 20)
        self.assertEqual({item["id"] for item in coverage}, {206, 207})
        self.assertTrue(all(item["min_batches_per_aic"] > 4 for item in coverage))
        self.assertTrue(all(item["min_tiles_per_aic"] == 195 for item in coverage))
        self.assertTrue(all(item["tiles_per_aic"] == 207 for item in coverage))

    def test_ceil_distribution_tail_can_have_fewer_than_five_batches(self):
        with self.assertRaisesRegex(ValueError, "only 57 tiles.*every AIC"):
            MODULE.pipeline_coverage(MODULE.MSS_MANIFEST, 60)

    def test_insufficient_coverage_does_not_pass(self):
        with self.assertRaisesRegex(ValueError, "every AIC"):
            MODULE.pipeline_coverage(MODULE.MSS_MANIFEST, 64)


class PipelineReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cases = [case for case in json.loads(MODULE.MSS_MANIFEST.read_text(encoding="utf-8"))
                      if case["id"] in MODULE.PIPELINE_IDS]

    def tearDown(self):
        self.temporary.cleanup()

    def write_report(self):
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "statistic"
        sheet.append(["\u7f16\u53f7", "\u8fd0\u884c\u7ed3\u679c", "npu_dut_\u5185\u5b58\u68c0\u6d4b\u901a\u8fc7", "\u7528\u4f8bjson\u4fe1\u606f"])
        for case in self.cases:
            sheet.append([case["id"], "SUCCESS", True, json.dumps(case)])
        book.save(self.root / "report.xlsx")
        book.close()

    def change_spec(self, key, value):
        field = next(item for item in self.cases[0]["inputs"] if item["name"] == "case_spec")
        spec = json.loads(field["range_values"])
        spec[key] = value
        field["range_values"] = json.dumps(spec)

    def test_current_case_contracts_pass(self):
        self.write_report()
        self.assertEqual(len(MODULE.check_atk_reports(self.root)), 1)

    def test_historical_small_case_with_same_id_fails(self):
        self.change_spec("T", 128)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "input contract differs"):
            MODULE.check_atk_reports(self.root)

    def test_other_seed_fails(self):
        self.change_spec("seed", 1)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "input contract differs"):
            MODULE.check_atk_reports(self.root)

    def test_other_executor_fails(self):
        self.cases[0]["api_type"] = "executor_different_operator"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "input contract differs"):
            MODULE.check_atk_reports(self.root)

    def test_varlen_distribution_mismatch_fails(self):
        field = next(item for item in self.cases[1]["inputs"] if item["name"] == "case_spec")
        spec = json.loads(field["range_values"])
        spec["seqlens"] = [4097, 4096]
        field["range_values"] = json.dumps(spec)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "input contract differs"):
            MODULE.check_atk_reports(self.root)


if __name__ == "__main__":
    unittest.main()
