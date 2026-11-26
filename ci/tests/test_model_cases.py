"""The optional model job must fail on missing, stale or incomplete platform results."""
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_cases"))
from run import plan
from verdict import verify


class ModelReportTest(unittest.TestCase):
    def test_both_platforms_coverage_and_evidence(self):
        day = date(2026, 11, 26)
        cases = plan(day)
        self.assertEqual([sum(c["suite"] == s for c in cases) for s in ("kda", "gdn")], [39, 5])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            for platform in ("a2", "a5"):
                folder = root / f"model-cases-{platform}-42"
                folder.mkdir()
                (folder / "summary.json").write_text(json.dumps(dict(
                    platform=platform, date=day.isoformat(), head_sha="abc", run_id="42",
                    cases={case["id"]: True for case in cases})))
                for case in cases:
                    ok = {"passed": True}
                    row = dict(case, passed=True, output=ok, state=ok, route=ok,
                               metrics=dict.fromkeys(("o", "final_state", "dq", "dk", "dv", "dbeta", "dg"), ok))
                    (folder / (case["id"] + ".json")).write_text(json.dumps(row))
            self.assertTrue(verify(root, day, "abc", "42"))
            self.assertFalse(verify(root, day, "different-head", "42"))
            target = folder / "prefill_t3072_h8_v24.json"
            row = json.loads(target.read_bytes())
            del row["route"]
            target.write_text(json.dumps(row))
            self.assertFalse(verify(root, day, "abc", "42"))
            target.unlink()
            self.assertFalse(verify(root, day, "abc", "42"))
