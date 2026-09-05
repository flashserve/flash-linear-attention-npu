#!/usr/bin/env python3
"""Focused tests for ATK runtime report discovery and fail-closed checks."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import openpyxl


_MODULE_PATH = Path(__file__).with_name("check_atk_result.py")
_SPEC = importlib.util.spec_from_file_location("check_atk_result_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_summary(path: Path, header, row, *, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "summary"
    sheet.append(header)
    sheet.append(row)
    workbook.save(path)
    workbook.close()
    os.utime(path, (mtime, mtime))


class CheckAtkResultTest(unittest.TestCase):
    def test_accuracy_finds_nested_report_from_current_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy" / "atk_output" / "atk_demo_1" / "report" / "result.xlsx"
            _write_summary(
                report,
                [
                    "名称", "总用例数", "执行成功用例个数", "执行失败用例个数",
                    "通过用例个数", "错误信息匹配用例个数", "通过率", "精度是否达标",
                ],
                ["cpu_golden", "1", "1", "0", "1", "0", "100.0", "Pass"],
                mtime=200.0,
            )

            current = _MODULE.check_accuracy(str(root), "demo", newer_than=199.0)
            stale = _MODULE.check_accuracy(str(root), "demo", newer_than=201.0)

            self.assertTrue(current["found"])
            self.assertTrue(current["all_pass"])
            self.assertFalse(stale["found"])
            self.assertFalse(stale["all_pass"])

    def test_determinism_requires_all_cases_to_execute_and_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "atk_output" / "atk_demo_mss_1" / "report" / "result.xlsx"
            header = [
                "名称", "总用例数", "执行成功用例个数", "执行失败用例个数",
                "通过用例个数", "错误信息匹配用例个数", "通过率",
                "精度是否达标", "确定性计算是否达标",
            ]
            _write_summary(
                report,
                header,
                ["npu_dut", "4", "4", "0", "4", "0", "100.0", "-", "Pass"],
                mtime=300.0,
            )
            passed = _MODULE.check_determinism(str(root), "demo", newer_than=299.0)
            self.assertTrue(passed["all_pass"])

            _write_summary(
                report,
                header,
                ["npu_dut", "4", "3", "1", "3", "0", "75.0", "-", "Pass"],
                mtime=301.0,
            )
            failed = _MODULE.check_determinism(str(root), "demo", newer_than=299.0)
            self.assertFalse(failed["all_pass"])

    def test_accuracy_requires_explicit_pass_indicator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "accuracy" / "atk_output" / "atk_demo_1" / "report" / "result.xlsx"
            _write_summary(
                report,
                [
                    "名称", "总用例数", "执行成功用例个数", "执行失败用例个数",
                    "通过用例个数", "错误信息匹配用例个数", "通过率", "精度是否达标",
                ],
                ["cpu_golden", "1", "1", "0", "1", "0", "100.0", "-"],
                mtime=400.0,
            )

            result = _MODULE.check_accuracy(str(root), "demo", newer_than=399.0)
            self.assertTrue(result["found"])
            self.assertFalse(result["all_pass"])

    def test_missing_report_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "check_atk_result.py",
                "--type", "accuracy",
                "--output-root", tmp,
                "--op", "demo",
                "--newer-than", "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as raised:
                    _MODULE.main()
            self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
