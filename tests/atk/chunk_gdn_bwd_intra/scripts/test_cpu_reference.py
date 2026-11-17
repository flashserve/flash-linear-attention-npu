"""CPU-only reference regressions; no ATK installation or NPU is required.

Run with: python tests/atk/chunk_gdn_bwd_intra/scripts/test_cpu_reference.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
except (ImportError, OSError) as error:
    torch = None
    TORCH_ERROR = str(error)
else:
    TORCH_ERROR = ""


@unittest.skipIf(torch is None, f"CPU PyTorch is unavailable: {TORCH_ERROR}")
class CpuReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only the import-time ATK/device interfaces are stubbed; all numerical
        # operations below use real CPU PyTorch and the production executor.
        stubs = {}
        for name in (
            "atk", "atk.configs", "atk.configs.dataset_config",
            "atk.configs.results_config", "atk.tasks", "atk.tasks.api_execute",
            "atk.tasks.api_execute.base_api", "fla_npu", "fla_npu.ops",
        ):
            stubs[name] = types.ModuleType(name)
        stubs["atk.configs.dataset_config"].InputDataset = object
        stubs["atk.configs.results_config"].TaskResult = object
        stubs["atk.tasks.api_execute.base_api"].BaseApi = object
        stubs["atk.tasks.api_execute"].register = lambda name: lambda cls: cls
        stubs["fla_npu.ops"].ascendc = types.SimpleNamespace()
        source = Path(__file__).resolve().parents[1] / "executor_chunk_gdn_bwd_intra.py"
        module_spec = importlib.util.spec_from_file_location("intra_cpu_under_test", source)
        cls.executor = importlib.util.module_from_spec(module_spec)
        with patch.dict(sys.modules, stubs), patch.object(sys, "path", list(sys.path)):
            module_spec.loader.exec_module(cls.executor)

    def test_fp64_benchmark_does_not_round_workspace_or_outputs(self):
        for dtype in ("bf16", "fp16"):
            for use_exp2 in (False, True):
                with self.subTest(dtype=dtype, use_exp2=use_exp2):
                    spec = dict(B=1, HK=1, HV=1, T=1, K=1, V=1,
                                chunk_size=64, dtype=dtype, g_dtype="fp32",
                                beta_dtype="fp32", use_exp2=use_exp2, seed=20260904)
                    inputs = self.executor.build_inputs(spec, torch.device("cpu"), True)
                    gate = torch.exp2 if use_exp2 else torch.exp
                    a = inputs["A"][0, 0, 0, 0]
                    beta = inputs["beta"][0, 0, 0]
                    g = inputs["g"][0, 0, 0]
                    q = inputs["q"][0, 0, 0, 0]
                    k = inputs["k"][0, 0, 0, 0]
                    v = inputs["v"][0, 0, 0, 0]
                    d_o = inputs["d_o"][0, 0, 0, 0]
                    intermediates = (a * beta * gate(g), a * beta,
                                     inputs["scale"] * k * q)
                    right = (k, v, d_o)
                    benchmark = self.executor.run_cpu_benchmark(spec)
                    matched = self.executor._cpu_ref(inputs)
                    for actual, device_ref, intermediate, operand in zip(
                        benchmark, matched, intermediates, right
                    ):
                        self.assertEqual(actual.dtype, torch.float64)
                        torch.testing.assert_close(actual.flatten()[0], intermediate * operand,
                                                   rtol=1e-14, atol=0)
                        rounded = intermediate.to(inputs["main_dtype"]).double() * operand
                        torch.testing.assert_close(device_ref.flatten()[0], rounded,
                                                   rtol=1e-14, atol=0)
                        self.assertNotEqual(actual.item(), device_ref.item())

    def test_same_quantized_inputs_are_promoted_for_benchmark(self):
        for dtype in ("bf16", "fp16"):
            with self.subTest(dtype=dtype):
                spec = dict(B=1, HK=2, HV=10, T=3, K=2, V=2,
                            chunk_size=64, dtype=dtype, cu_seqlens=[0, 1, 3])
                normal = self.executor.build_inputs(spec, torch.device("cpu"))
                precise = self.executor.build_inputs(spec, torch.device("cpu"), True)
                for name in ("q", "k", "v", "g", "beta", "A", "d_o"):
                    torch.testing.assert_close(normal[name].double(), precise[name], rtol=0, atol=0)
                for output in self.executor.run_cpu_benchmark(spec):
                    self.assertEqual(output.dtype, torch.float64)
                    self.assertTrue(torch.isfinite(output).all().item())
                for output in self.executor.run_cpu(spec, high_precision=True):
                    self.assertEqual(output.dtype, normal["main_dtype"])
                for output in self.executor.run_cpu(spec):
                    self.assertEqual(output.dtype, torch.float32)

    def test_atk_entry_dispatch_and_normalization(self):
        api = self.executor.FunctionApi.__new__(self.executor.FunctionApi)
        api.device = "cpu"
        api.task_result = types.SimpleNamespace(is_benchmark_task=False)
        data = types.SimpleNamespace(kwargs={"case_spec": {"dtype": "bf16"}})
        main_output = (torch.ones(1, dtype=torch.bfloat16),) * 3
        fp64_output = (torch.ones(1, dtype=torch.float64),) * 3
        with patch.object(self.executor, "run_cpu", return_value=main_output) as run_cpu, \
             patch.object(self.executor, "run_cpu_benchmark", return_value=fp64_output) as benchmark:
            outputs = api(data)
            self.assertEqual(run_cpu.call_args.kwargs, {"high_precision": True})
            benchmark.assert_not_called()
            self.assertTrue(all(output.dtype == torch.bfloat16 for output in outputs))
            api.task_result.is_benchmark_task = True
            outputs = api(data)
            benchmark.assert_called_once()
            self.assertTrue(all(output.dtype == torch.float32 for output in outputs))
            self.assertEqual(run_cpu.call_count, 1)
            api.cpu(data)
            self.assertEqual(run_cpu.call_args.kwargs, {"high_precision": False})


if __name__ == "__main__":
    unittest.main(verbosity=2)
