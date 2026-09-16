"""Exercise six-output GDN consumers without requiring an NPU runtime."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class GdnForwardConsumerTest(unittest.TestCase):
    def test_example_preserves_its_four_output_contract(self):
        outputs = tuple(object() for _ in range(6))
        op = Mock(return_value=outputs)
        ns = load_functions("examples/flash_gated_delta_rule.py", {"flash_chunk_gated_delta_rule_fwd"},
                            {"ascendc_chunk_gated_delta_rule_fwd": op, "_chunk_list": lambda *args: None})
        for final in (False, True):
            result = ns["flash_chunk_gated_delta_rule_fwd"](
                q=None, k=None, v=None, g=None, beta=None, scale=1.0,
                initial_state=None, output_final_state=final)
            self.assertEqual(result, (outputs[2], outputs[0], outputs[3], outputs[1] if final else None))
            self.assertTrue(op.call_args.kwargs["disable_recompute"])

    def test_atk_dut_and_benchmark_keep_matching_output_order(self):
        outputs = tuple(object() for _ in range(4))
        op = Mock(return_value=(*outputs, None, None))
        package = types.ModuleType("fla_npu")
        ops = types.ModuleType("fla_npu.ops")
        ops.ascendc = types.SimpleNamespace(chunk_gated_delta_rule_fwd=op)
        ns = load_functions("tests/atk/chunk_gated_delta_rule_fwd/executor_chunk_gated_delta_rule_fwd.py",
                            {"run_npu", "_public_outputs"},
                            {"deterministic_initial_state": lambda case: None,
                             "canonical_chunk_indices": lambda *args: None})
        with patch.dict(sys.modules, {"fla_npu": package, "fla_npu.ops": ops}):
            for final in (False, True):
                case = types.SimpleNamespace(output_final_state=final, chunk_size=64, cu_seqlens=None, scale=1.0)
                actual = ns["run_npu"]("dut", [None] * 5, case)
                benchmark = ns["_public_outputs"](outputs, case, "benchmark")
                self.assertEqual(actual, benchmark)
                self.assertEqual(actual, outputs if final else (outputs[0], outputs[2], outputs[3]))


if __name__ == "__main__":
    unittest.main()
