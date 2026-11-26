"""Cross-PR result reuse, partial ATK hits and invalid cache fallbacks."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI))
from smoke_cache import ResultCache

spec = importlib.util.spec_from_file_location("atk_smoke_run", CI.parent / "tests/atk/smoke/run.py")
atk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atk)


class SmokeCacheTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = patch.dict(os.environ, CI_SMOKE_CACHE_DIR=str(self.root / "cache"),
                         CI_SMOKE_RUNTIME="a" * 64, CI_ACCURACY_RUN_ID="123", CI_ACCURACY_HEAD_SHA="old")
        env.start()
        self.addCleanup(env.stop)

    def test_runtime_namespace_corruption_and_origins(self):
        cache = ResultCache()
        key = cache.key("model", {"seed": 15, "lower_bound": -3})
        cache.put(key, {"passed": True})
        with patch.dict(os.environ, CI_ACCURACY_HEAD_SHA="new", CI_ACCURACY_RUN_ID="456"):
            self.assertEqual(ResultCache().get(key)["origin"]["run_id"], "123")
        with patch.dict(os.environ, CI_SMOKE_RUNTIME="b" * 64):
            self.assertIsNone(ResultCache().get(ResultCache().key("model", {"seed": 15, "lower_bound": -3})))
        self.assertIsNone(cache.get(cache.key("model", {"seed": 15, "lower_bound": -2})))
        path = cache.root / (key + ".json.gz")
        entry = json.loads(gzip.decompress(path.read_bytes()))
        entry["value"]["passed"] = False
        path.write_bytes(gzip.compress(json.dumps(entry).encode()))
        self.assertIsNone(cache.get(key))
        path.write_bytes(b"truncated")
        self.assertIsNone(cache.get(key))

    def test_atk_runs_only_misses_and_keeps_native_evidence(self):
        suite = dict(platform="A2", op="probe", group="probe", count=3)
        standard = {"acc": {atk.STANDARD: {}}}
        cases = [dict(id=i, name=f"case-{i}", default_seed=seed, standard=standard) for i, seed in enumerate([1, 2, None])]
        config = self.root / "suite.json"
        config.write_text(json.dumps(cases))
        runs = []

        def execute(command, **kwargs):
            pending = json.loads(Path(command[command.index("-c") + 1]).read_bytes())
            runs.append([c["id"] for c in pending])
            folder = Path(command[command.index("-o") + 1])
            report = folder / "atk_output/run/report"
            report.mkdir(parents=True)
            for role in ("dut", "low", "high"):
                (folder / ("events." + role)).write_text("\n".join(
                    json.dumps([c["id"], c["name"], [[[1], "bf16"]]]) for c in pending))
            rows = [{"用例json信息": c, "cpu_cpu_golden_精度通过": True, "精度详情": {"outputs": [
                     {"filename": "output_0.pt", "result": True, "new_benchmark_indicate": {"ratio": 1}}]}}
                    for c in pending]
            (report / "native.json").write_text(json.dumps(rows))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(atk, "cases_for", return_value=cases), patch.object(atk, "case_path", return_value=config), \
                patch.object(atk.subprocess, "run", side_effect=execute):
            for attempt in (1, 2):
                folder = self.root / str(attempt)
                folder.mkdir()
                if attempt == 2:
                    cases[1]["default_seed"] = 99
                atk.run_suite(suite, folder, 0)
        self.assertEqual(runs, [[0, 1, 2], [1, 2]])
        reused = json.loads((self.root / "2/probe/cache/report/reused.json").read_bytes())
        self.assertEqual([r["id"] for r in reused], [0])
        self.assertTrue(reused[0]["evidence"]["row"]["cpu_cpu_golden_精度通过"])
