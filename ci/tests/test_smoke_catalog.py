"""Bootstrap the smoke framework with an empty registry, then enroll an operator."""
from contextlib import redirect_stdout
from datetime import date
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


CI = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load("smoke_catalog_runner", CI.parent / "tests/atk/smoke/run.py")
with patch.object(sys, "path", [str(CI / "model_cases"), *sys.path]):
    models = load("smoke_model_runner", CI / "model_cases/run.py")


class SmokeCatalogTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def cli(self, *args):
        out = StringIO()
        with patch.object(smoke, "ROOT", self.root), patch.object(sys, "argv", ["run.py", *args]), redirect_stdout(out):
            smoke.main()
        return out.getvalue().strip()

    def test_empty_registry_and_first_operator(self):
        self.assertEqual(json.loads(self.cli("matrix")), {"include": []})
        self.assertEqual(self.cli("validate"), "0 cases in 0 groups across 0 platform files")
        op = self.root / "cases/probe"
        (op / "A5").mkdir(parents=True)
        catalog = op / "A5/cases.json"
        case = {"id": 1, "name": "probe", "api_type": "fla_smoke", "standard": {"acc": smoke.STANDARD}}
        catalog.write_text(json.dumps([{"group": "smoke", "count": 1, "reference": "executor_probe.py", "cases": [case]}]))
        with self.assertRaisesRegex(ValueError, "one cases.json"):
            self.cli("validate")
        (op / "suite.json").write_text(json.dumps({"source_root": "fla/ops/ascendc/probe", "platforms": {"A5": True}}))
        with self.assertRaisesRegex(ValueError, "missing reference"):
            self.cli("validate")
        (op / "executor_probe.py").touch()
        self.assertEqual(json.loads(self.cli("matrix")), {"include": [{"op": "probe", "platform": "A5"}]})
        self.assertEqual(self.cli("validate"), "1 cases in 1 groups across 1 platform files")

    def test_empty_and_partial_model_catalogs(self):
        with patch.object(models, "HERE", self.root):
            self.assertEqual(models.plan(date(2026, 11, 28)), [])
            (self.root / "gdn_cases.json").write_text("{}")
            with self.assertRaises(FileNotFoundError):
                models.plan(date(2026, 11, 28))
