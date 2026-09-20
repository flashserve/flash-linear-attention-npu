#!/usr/bin/env python3
"""CPU 输入合同：历史表达式、三角色共享数据和 raw 保存重放，不替代 NPU 验收。"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
from gdn_reference import INPUT_PREPARATION_VERSION, effective_inputs, run_golden_reference


def load_executor():
    """只替换 ATK 框架外壳，执行实际 build_inputs 和 FunctionApi 初始化代码。"""
    names = (
        "atk", "atk.configs", "atk.configs.dataset_config", "atk.configs.results_config",
        "atk.tasks", "atk.tasks.api_execute", "atk.tasks.api_execute.base_api",
    )
    modules = {name: ModuleType(name) for name in names}

    class BaseApi:
        def __init__(self, result):
            self.device = result.device
            self.device_id = result.device_id

    modules["atk.configs.dataset_config"].InputDataset = SimpleNamespace
    modules["atk.configs.results_config"].TaskResult = SimpleNamespace
    modules["atk.tasks.api_execute"].register = lambda _name: lambda cls: cls
    modules["atk.tasks.api_execute.base_api"].BaseApi = BaseApi
    spec = importlib.util.spec_from_file_location(
        "gdn_input_contract_executor", OP_DIR / "executor_chunk_gated_delta_rule_fwd.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


EXECUTOR = load_executor()


def raw_values(dtype):
    # 转置后非连续，覆盖准备阶段的 contiguous；固定数值避免随机性。
    tensor = torch.linspace(-0.05, 0.05, 512).reshape(1, 1, 128, 4).transpose(2, 3)
    gate = torch.tensor([[[-1.0, -0.25, 0.25, 1.0]]]).transpose(1, 2)
    return {
        "q": tensor.to(dtype), "k": tensor.to(dtype), "v": tensor.to(dtype),
        "g": gate.double(), "beta": gate.to(dtype), "scale": 128 ** -0.5,
        "chunk_size": 64, "is_varlen": False, "scenario": "dense",
        "cu_seqlens_spec": "none", "qkv_dtype": "fp16" if dtype == torch.float16 else "bf16",
    }


class InputContractTest(unittest.TestCase):
    def assert_tensors_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for actual, expected in zip(left, right):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(torch.equal(actual, expected))

    def test_historical_expression_and_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                values = raw_values(dtype)
                _, public_dtype, inputs = EXECUTOR.build_inputs(values)
                self.assertEqual(public_dtype, dtype)
                q, k, v, g, beta = inputs
                self.assert_tensors_equal((q, k, v), tuple(values[n] for n in ("q", "k", "v")))
                self.assert_tensors_equal(
                    (g, beta),
                    ((-torch.sigmoid(values["g"].float()) * 0.1).float(),
                     torch.sigmoid(values["beta"].float()).to(dtype)),
                )
                self.assertTrue(all(t.is_contiguous() and t.device.type == "cpu" for t in inputs))
                self.assertTrue(bool(((g < 0) & (g > -0.1)).all()))
                self.assertTrue(bool(((beta > 0) & (beta < 1)).all()))

    def test_preparation_does_not_mutate_raw_tensors(self):
        values = raw_values(torch.float16)
        tensors = tuple(values[n] for n in ("q", "k", "v", "g", "beta"))
        original = tuple(t.clone() for t in tensors)
        first = effective_inputs(*tensors, torch.float16)
        second = effective_inputs(*tensors, torch.float16)
        self.assert_tensors_equal(tensors, original)
        self.assert_tensors_equal(first, second)

    def test_shared_dataset_is_prepared_once_for_each_role(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                values = raw_values(dtype)
                original = {key: value.clone() for key, value in values.items() if isinstance(value, torch.Tensor)}
                dataset = SimpleNamespace(kwargs=values)
                expected = EXECUTOR.build_inputs(values)[2]
                for node, device, is_bm, role in (
                    ("phase6", "npu", False, "dut"),
                    ("gold", "npu", False, "benchmark"),
                    ("cpu", "cpu", True, "golden"),
                ):
                    result = SimpleNamespace(name=node, device=device, device_id=0,
                                             is_benchmark_task=is_bm, case_config=SimpleNamespace(id=0))
                    api = EXECUTOR.FunctionApi(result)
                    # 不调用NPU；只检查各角色进入设备搬运前的准备结果。
                    with patch.object(EXECUTOR, "_npu_device", return_value=torch.device("cpu")):
                        api.init_by_input_data(dataset)
                        api.init_by_input_data(dataset)
                    self.assertEqual(api._role, role)
                    self.assert_tensors_equal(api._inputs, expected)
                    if device == "npu":
                        self.assert_tensors_equal(api._npu_inputs, expected)
                    self.assertEqual(api.export_custom_data()["input_preparation_version"], INPUT_PREPARATION_VERSION)
                    self.assertEqual(api.export_custom_data()["saved_input_semantics"], "raw_before_input_preparation")
                for key, value in original.items():
                    self.assertTrue(torch.equal(values[key], value), key)

    def test_saved_raw_replay_matches_original(self):
        values = raw_values(torch.bfloat16)
        expected = EXECUTOR.build_inputs(values)[2]
        with io.BytesIO() as buffer:
            torch.save(values, buffer)
            buffer.seek(0)
            restored = torch.load(buffer, map_location="cpu", weights_only=True)
        self.assert_tensors_equal(EXECUTOR.build_inputs(restored)[2], expected)

    def test_cpu_golden_small_cases_remain_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            for chunk in (64, 128):
                for varlen in (False, True):
                    with self.subTest(dtype=dtype, chunk=chunk, varlen=varlen):
                        values = raw_values(dtype)
                        values.update(chunk_size=chunk, is_varlen=varlen,
                                      scenario="state_zero_final" if varlen else "dense",
                                      cu_seqlens_spec="0,1,4" if varlen else "none")
                        case, public_dtype, inputs = EXECUTOR.build_inputs(values)
                        outputs = run_golden_reference(*inputs, case, public_dtype)
                        self.assertEqual(len(outputs), 4 if varlen else 3)
                        self.assertTrue(all(bool(torch.isfinite(t).all()) for t in outputs))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
