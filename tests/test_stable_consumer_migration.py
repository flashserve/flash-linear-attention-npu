"""Legacy caller migration preserves every original argument expression."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests/fixtures/stable_migrated_consumers.json"
SPEC = importlib.util.spec_from_file_location(
    "migrated_stable", ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py")
STABLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STABLE)


class MigratedConsumerContractTest(unittest.TestCase):
    def test_original_arguments_and_stable_bindings(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))["calls"]
        self.assertEqual(len(contract), 9)
        for name, expected in contract.items():
            with self.subTest(path=name):
                tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
                calls = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    target = ast.unparse(node.func)
                    self.assertFalse(target.startswith("torch.ops.npu.npu_"), target)
                    if not target.startswith("ascendc.npu_"):
                        continue
                    op = target.rsplit(".", 1)[1]
                    calls.append({"op": op, "args": [ast.unparse(arg) for arg in node.args],
                                  "kwargs": [[kw.arg, ast.unparse(kw.value)] for kw in node.keywords]})
                    self.assertFalse(any(isinstance(arg, ast.Starred) for arg in node.args))
                    self.assertTrue(all(kw.arg is not None for kw in node.keywords))
                    inspect.signature(getattr(STABLE, op)).bind(
                        *(object() for _ in node.args), **{kw.arg: object() for kw in node.keywords})
                self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
