"""CPU-only guards for the ATK-free accuracy entry and dependency preflight."""
import ast
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("accuracy_runner", ROOT / "experimental/_tests/accuracy.py")
preflight = load("preflight", ROOT / "experimental/_tests/check_environment.py")
try:
    import torch
except ImportError:
    torch = None


class AccuracyContract(unittest.TestCase):
    def test_inputs_and_reference_do_not_drift_from_self_contained_atk_executor(self):
        paths = [ROOT / "experimental/_tests/rwkv8_reference.py",
                 ROOT / "experimental/tests/atk/fused_recurrent_rwkv8/executor_fused_recurrent_rwkv8.py"]
        trees = [{node.name: ast.dump(node) for node in ast.parse(path.read_text(encoding="utf-8")).body
                  if isinstance(node, (ast.FunctionDef, ast.ClassDef))} for path in paths]
        for name in ("_rwkv8_inputs", "_Rwkv8Result", "_wkv7_decay"):
            self.assertEqual(trees[0][name], trees[1][name], name)

    def test_quick_and_full_use_the_original_catalog(self):
        self.assertEqual([case["index"] for case in runner.load_cases("quick")], [0, 1, 2])
        full = runner.load_cases("full")
        self.assertEqual(len(full), 209)
        self.assertEqual({case["spec"]["dtype"] for case in full}, {"fp32", "fp16", "bf16"})
        for indices in ([], [0, 0], [-1], [209]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                runner.load_cases("full", indices)

    def test_missing_atk_only_warns_but_required_import_failure_blocks(self):
        for missing in (None, "torch_npu", "wheel"):
            with patch.object(preflight, "probe", lambda name: {"available": name != missing, "error": "missing"}), \
                    patch.object(preflight, "optional_atk", return_value={"ATK": None, "kombu": None}), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = preflight.check()
            self.assertEqual(result["status"], "success" if missing is None else "failure")
            self.assertEqual(len(result["warnings"]), 1)
            self.assertEqual(result["accuracy_backend"], "cpu-reference")


@unittest.skipIf(torch is None, "requires CPU torch")
class AccuracyMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load("reference", ROOT / "experimental/_tests/rwkv8_reference.py")

    def test_shared_reference_preserves_atk_fp32_and_fp64_results(self):
        from typing import Any, NamedTuple, Optional
        source = ROOT / "experimental/tests/atk/fused_recurrent_rwkv8/executor_fused_recurrent_rwkv8.py"
        names = {"_Rwkv8Result", "_wkv7_decay", "fused_recurrent_rwkv8_reference"}
        nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        namespace = dict(torch=torch, Any=Any, NamedTuple=NamedTuple, Optional=Optional)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        for dtype in ("fp16", "bf16", "fp32"):
            for high in (False, True):
                inputs = self.reference._rwkv8_inputs(dict(dtype=dtype, B=1, H=2, T=7, K=8, V=16,
                    initial_state=True, chunk_len=3, output_chunk_state=True, output_sa=True), torch.device("cpu"), high)
                actual = self.reference.fused_recurrent_rwkv8_reference(**inputs)
                expected = namespace["fused_recurrent_rwkv8_reference"](**inputs)
                for a, e in zip(actual, expected):
                    self.assertTrue(torch.equal(a, e), (dtype, high))

    def test_golden_and_dut_receive_identical_quantized_inputs(self):
        for dtype in ("fp16", "bf16", "fp32"):
            spec = dict(dtype=dtype, B=1, H=2, T=3, K=4, V=5, initial_state=True)
            original = self.reference._rwkv8_inputs(spec, torch.device("cpu"), False)
            high = self.reference._rwkv8_inputs(spec, torch.device("cpu"), True)
            for name, value in original.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value.double(), high[name]), (dtype, name))
                else:
                    self.assertEqual(value, high[name])

    def test_matching_fraction_and_outlier_envelope_are_both_required(self):
        golden = torch.ones(1000)
        passing = golden.clone()
        passing[:5] += 0.005
        self.assertTrue(self.reference.compare_output(passing, golden)["pass"])
        ratio_failure = golden.clone()
        ratio_failure[:20] += 0.005
        self.assertFalse(self.reference.compare_output(ratio_failure, golden)["pass"])
        outlier_failure = golden.clone()
        outlier_failure[0] += 0.02
        result = self.reference.compare_output(outlier_failure, golden)
        self.assertFalse(result["pass"])
        self.assertEqual(result["envelope_failures"], 1)

    def test_dtype_thresholds_finite_empty_and_shape_contracts(self):
        for dtype, atol in ((torch.float16, 1.95e-3), (torch.bfloat16, 1.56e-2), (torch.float32, 1.53e-5)):
            self.assertEqual(self.reference.compare_output(torch.ones(2, dtype=dtype), torch.ones(2))["atol"], atol)
        self.assertTrue(self.reference.compare_output(torch.empty(0), torch.empty(0))["pass"])
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(AssertionError):
                self.reference.compare_output(torch.tensor([bad]), torch.ones(1))
            with self.assertRaises(AssertionError):
                self.reference.compare_output(torch.ones(1), torch.tensor([bad]))
        with self.assertRaises(AssertionError):
            self.reference.compare_output(torch.ones(1), torch.ones(2))

    def test_optional_outputs_cannot_be_dropped_or_have_wrong_dtype(self):
        spec = dict(dtype="fp16", B=1, H=1, T=3, K=4, V=5, chunk_len=2,
                    initial_state=True, output_chunk_state=True, output_sa=True)
        inputs = self.reference._rwkv8_inputs(spec, torch.device("cpu"), False)
        golden = self.reference.fused_recurrent_rwkv8_reference(**inputs)
        self.assertTrue(all(x["pass"] for x in self.reference.check_outputs(golden, golden, inputs).values()))
        for outputs in ((golden.o,), (golden.o, None, golden.sa), (golden.o, golden.s.half(), golden.sa)):
            with self.assertRaises(AssertionError):
                self.reference.check_outputs(outputs, golden, inputs)

    @unittest.skipUnless(os.environ.get("FLA_EXPERIMENTAL_TEST_ATK_PARITY") == "1",
                         "ATK comparator parity is an opt-in check")
    def test_comparator_matches_installed_atk_on_boundary_and_random_samples(self):
        from atk.configs.mixed_tolerance_benchmark_config import MixedToleranceBenchmarkConfig
        from atk.tasks.post_process.mixed_tolerance_benchmark_compare import MixedToleranceBenchmarkAccuracyCompare
        comparator = object.__new__(MixedToleranceBenchmarkAccuracyCompare)
        comparator.mixed_tolerance_standard = MixedToleranceBenchmarkConfig()
        comparator.output_dtype_overrides = {}
        generator = torch.Generator().manual_seed(1234)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for scale in (0.0, 1e-4, 0.003, 0.03, 0.3, 3.0):
                golden = torch.randn(1000, generator=generator)
                actual = (golden + scale * torch.randn(1000, generator=generator)).to(dtype)
                expected = comparator.compute_accuracy_result(actual, golden, "output_0.pt").result
                self.assertEqual(self.reference.compare_output(actual, golden)["pass"], bool(expected))
            for count, error in ((5, .005), (10, .005), (20, .005), (1, .02), (1, 10.0)):
                golden = torch.ones(1000)
                actual = golden.clone()
                actual[:count] += error
                actual = actual.to(dtype)
                expected = comparator.compute_accuracy_result(actual, golden, "output_0.pt").result
                self.assertEqual(self.reference.compare_output(actual, golden)["pass"], bool(expected))


if __name__ == "__main__":
    unittest.main()
