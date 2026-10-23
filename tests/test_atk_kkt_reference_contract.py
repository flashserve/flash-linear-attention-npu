"""CPU-only checks for the KKT ATK beta-row contract; no ATK/NPU import."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import runpy
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = ROOT / "tests/atk/chunk_scaled_dot_kkt/executor_chunk_scaled_dot_kkt.py"
REFERENCE = ROOT / "torch_custom/fla_npu/test/test_npu_chunk_scaled_dot_kkt.py"


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"missing CPU reference functions in {path}")
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)


@unittest.skipIf(torch is None, "CPU PyTorch is required")
class KktReferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared = runpy.run_path(str(ROOT / "tests/atk/common/_ascendc_common_executor.py"))
        cls.executor = {**shared, "torch": torch}
        _load_functions(EXECUTOR, {"build_inputs", "_chunk_scaled_dot_kkt_ref"}, cls.executor)
        cls.independent = {"torch": torch, "math": math}
        _load_functions(REFERENCE, {"chunk_scaled_dot_kkt_reference", "iter_chunk_ranges"}, cls.independent)

        # Retain the pre-fix column-beta implementation as an explicit negative
        # control. It must fail on the very same inputs and unchanged tolerance.
        source = EXECUTOR.read_text(encoding="utf-8")
        if source.count("beta_chunk[:, None]") != 1:
            raise AssertionError("expected one reviewed row-beta expression")
        tree = ast.parse(source.replace("beta_chunk[:, None]", "beta_chunk[None, :]"))
        wrong = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_chunk_scaled_dot_kkt_ref")
        cls.negative = {**shared, "torch": torch}
        exec(compile(ast.Module(body=[wrong], type_ignores=[]), "pre_fix_column_beta", "exec"), cls.negative)

    def test_nonuniform_beta_has_hand_computable_row_scaling(self):
        k = torch.zeros((1, 1, 3, 16), dtype=torch.float64)
        k[0, 0, :, :2] = torch.tensor([[1, 0], [2, 1], [3, 4]])
        inputs = {"k": k, "g": torch.zeros((1, 1, 3), dtype=torch.float64),
                  "beta": torch.tensor([[[0.125, 0.25, 0.5]]], dtype=torch.float64), "chunk_size": 16}
        expected = torch.zeros((1, 1, 3, 16), dtype=torch.float32)
        expected[0, 0, 1, 0] = 0.5   # (2*1 + 1*0) * beta[1]
        expected[0, 0, 2, 0] = 1.5   # (3*1 + 4*0) * beta[2]
        expected[0, 0, 2, 1] = 5.0   # (3*2 + 4*1) * beta[2]
        actual = self.executor["_chunk_scaled_dot_kkt_ref"](inputs)
        independent = self.independent["chunk_scaled_dot_kkt_reference"](
            inputs["k"], inputs["g"], inputs["beta"], inputs["chunk_size"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(independent, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.negative["_chunk_scaled_dot_kkt_ref"](inputs), expected))

    def test_frozen_domain_three_seeds_align_and_reject_old_axis(self):
        cases = json.loads((EXECUTOR.parent / "atk_chunk_scaled_dot_kkt.json").read_text(encoding="utf-8"))
        for case in cases:
            spec = json.loads(next(item["range_values"] for item in case["inputs"] if item["name"] == "case_spec"))
            self.assertEqual(spec["HK"], spec["HV"], "this repair does not extend GVA coverage")
            for seed in (20260817, 20260818, 20260819):
                with self.subTest(dtype=spec["dtype"], seed=seed):
                    inputs = self.executor["build_inputs"]({**spec, "seed": seed}, torch.device("cpu"), high_precision=True)
                    actual = self.executor["_chunk_scaled_dot_kkt_ref"](inputs)
                    independent = self.independent["chunk_scaled_dot_kkt_reference"](
                        inputs["k"], inputs["g"], inputs["beta"], inputs["chunk_size"])
                    wrong = self.negative["_chunk_scaled_dot_kkt_ref"](inputs)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(actual.shape, independent.shape)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    # Existing ATK mixed_tolerance_bm FP32 pointwise constants,
                    # not the input BF16/FP16 tolerance. Require all correct
                    # points to pass; the historical axis must fail >1%.
                    torch.testing.assert_close(actual, independent, rtol=9.77e-4, atol=1.53e-5)
                    wrong_matched = torch.isclose(wrong, independent, rtol=9.77e-4, atol=1.53e-5).float().mean().item()
                    self.assertLess(wrong_matched, 0.99)


if __name__ == "__main__":
    unittest.main()
