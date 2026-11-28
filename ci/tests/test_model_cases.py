"""Acceptance of fresh and restored KDA/GDN per-case numerical evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_cases"))
from verdict import passed


class ModelReportTest(unittest.TestCase):
    def test_platform_evidence(self):
        cases = [
            {"suite": "kda", "tensor_seed": 1},
            {"suite": "gdn", "entry": "decode", "tensor_seed": 2},
            {"suite": "gdn", "entry": "prefill", "tensor_seed": 3},
        ]
        ok = {"passed": True}
        for case in cases:
            row = dict(case, passed=True, output=ok, state=ok, route=ok,
                       metrics=dict.fromkeys(("o", "final_state", "dq", "dk", "dv", "dbeta", "dg"), ok))
            self.assertTrue(passed(case, row, "a2"))
            self.assertTrue(passed(case, row, "a5"))
            self.assertFalse(passed(case, dict(row, tensor_seed=-1), "a5"))
            if case["suite"] == "kda":
                del row["metrics"]["dg"]
            elif case["entry"] == "decode":
                del row["state"]
            else:
                del row["route"]
                self.assertTrue(passed(case, row, "a2"))
            self.assertFalse(passed(case, row, "a5"))
