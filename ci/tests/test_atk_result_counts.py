import ast
import glob
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tests/atk/common/check_atk_result.py'
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_extract_summary_row', '_find_root_reports', '_find_xlsx_files', '_is_mssanitizer_report')]
NAMESPACE = {'glob': glob, 'os': os}
exec(compile(ast.fix_missing_locations(ast.Module(body=FUNCTIONS, type_ignores=[])), str(SOURCE), 'exec'), NAMESPACE)
extract = NAMESPACE['_extract_summary_row']

HEADER = ['节点', '总用例数', '执行成功用例个数', '执行失败用例个数',
          '精度通过用例个数', '错误信息匹配用例个数', '精度通过率', '精度是否达标']


class AtkResultCountsTest(unittest.TestCase):
    def test_matched_errors_count_as_passes(self):
        result = extract(HEADER, [['cpu', '66', '66', '0', '60', '6', '100', 'Pass']])
        self.assertEqual((result['check_pass'], result['check_fail']), (66, 0))
        self.assertTrue(result['all_pass'])

    def test_positive_failures_remain_failures(self):
        result = extract(HEADER, [['cpu', '60', '60', '0', '41', '6', '78.3333', 'Failed']])
        self.assertEqual((result['check_pass'], result['check_fail']), (47, 13))
        self.assertFalse(result['all_pass'])

    def test_unmatched_error_fails(self):
        result = extract(HEADER, [['cpu', 66, 66, 0, 60, 5, 98.48, 'Failed']])
        self.assertFalse(result['all_pass'])
        self.assertEqual(result['check_fail'], 1)

    def test_incomplete_execution_fails(self):
        result = extract(HEADER, [['cpu', 66, 65, 1, 60, 6, 100, 'Pass']])
        self.assertFalse(result['all_pass'])

    def test_all_failed_accuracy_is_not_mssanitizer(self):
        result = extract(HEADER, [['cpu', 4, 4, 0, 0, 0, 0, 'Pass']])
        self.assertEqual(result['check_pass'], 0)
        self.assertFalse(result['all_pass'])

    def test_multiple_nodes(self):
        result = extract(HEADER, [['a', 3, 3, 0, 2, 1, 100, 'Pass'],
                                  ['b', 4, 4, 0, 3, 1, 100, 'Pass']])
        self.assertEqual(result['check_pass'], 7)
        self.assertTrue(result['all_pass'])

    def test_mssanitizer_without_pass_counts(self):
        header = ['节点', '总用例数', '内存检测通过率', '内存检测是否达标']
        for status, passed in [('Pass', True), ('Failed', False)]:
            result = extract(header, [['npu', 4, 100 if passed else 0, status]])
            self.assertEqual(result['all_pass'], passed)
            self.assertEqual(result['check_pass'], 4 if passed else 0)

    def test_empty_report_fails(self):
        self.assertFalse(extract(HEADER, [])['all_pass'])

    def test_mssanitizer_with_execution_counts(self):
        header = ['节点', '总用例数', '执行成功用例个数', '执行失败用例个数', '内存检测是否达标']
        result = extract(header, [['npu', 4, 4, 0, 'Pass']])
        self.assertTrue(result['all_pass'])

    def test_default_and_explicit_output_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for prefix in ('', 'atk_output'):
                path = root / prefix / 'atk_chunk_kda_bwd_mss_1' / 'report' / 'results.xlsx'
                path.parent.mkdir(parents=True)
                path.touch()
                expected.append(str(path))
            self.assertCountEqual(NAMESPACE['_find_root_reports'](directory, 'chunk_kda_bwd'), expected)


if __name__ == '__main__':
    unittest.main()
