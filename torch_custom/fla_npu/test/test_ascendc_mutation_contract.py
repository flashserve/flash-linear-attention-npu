# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_INIT_PATH = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc" / "__init__.py"


class FakeTensor:
    def __init__(self, *, requires_grad: bool = False):
        self.requires_grad = requires_grad


def fake_torch(incremented):
    module = types.ModuleType("torch")
    module.Tensor = FakeTensor
    module.autograd = types.SimpleNamespace(
        graph=types.SimpleNamespace(increment_version=lambda tensors: incremented.extend(tensors))
    )
    return module


def load_ascendc_module(raw_calls):
    fake_fla_npu = types.ModuleType("fla_npu")
    fake_fla_npu.__path__ = []
    fake_fla_npu.load_ascendc_opapi_libraries = lambda: None

    fake_ops = types.ModuleType("fla_npu.ops")
    fake_ops.__path__ = []

    ctypes_module = types.ModuleType("fla_npu.ops.ascendc._aclnn_ctypes")

    def npu_causal_conv1d(
        x,
        weight,
        bias=None,
        conv_states=None,
        *,
        query_start_loc=None,
        cache_indices=None,
        initial_state_mode=None,
        num_accepted_tokens=None,
        activation_mode=0,
        pad_slot_id=-1,
        run_mode=0,
        head_num=0,
    ):
        raw_calls.append(conv_states)
        return "output"

    def npu_causal_conv1d_fn(x, weight, bias, conv_states=None, **kwargs):
        del x, weight, bias, kwargs
        raw_calls.append(conv_states)
        return "output"

    def npu_causal_conv1d_update(x, conv_state, weight, **kwargs):
        del x, weight, kwargs
        raw_calls.append(conv_state)
        return "output"

    def npu_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        *,
        cu_seqlens,
        inplace_final_state=True,
        **kwargs,
    ):
        del q, k, v, g, beta, cu_seqlens, inplace_final_state, kwargs
        raw_calls.append(initial_state)
        return "output", initial_state

    ctypes_module.ASCENDC_CTYPES_OPS = {
        "npu_causal_conv1d": npu_causal_conv1d,
        "npu_causal_conv1d_fn": npu_causal_conv1d_fn,
        "npu_causal_conv1d_update": npu_causal_conv1d_update,
        "npu_recurrent_kda": npu_recurrent_kda,
    }
    modules = {
        "fla_npu": fake_fla_npu,
        "fla_npu.ops": fake_ops,
        "fla_npu.ops.ascendc._aclnn_ctypes": ctypes_module,
    }

    spec = importlib.util.spec_from_file_location(
        "fla_npu.ops.ascendc",
        ASCENDC_INIT_PATH,
        submodule_search_locations=[str(ASCENDC_INIT_PATH.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    modules["fla_npu.ops.ascendc"] = module
    return module, spec, modules


class AscendCMutationContractTest(unittest.TestCase):
    def test_legacy_public_signatures_are_preserved(self):
        raw_calls = []
        module, spec, modules = load_ascendc_module(raw_calls)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)

        expected = [
            "x",
            "weight",
            "bias",
            "conv_states",
            "query_start_loc",
            "cache_indices",
            "initial_state_mode",
            "num_accepted_tokens",
            "activation_mode",
            "pad_slot_id",
            "run_mode",
            "head_num",
        ]
        self.assertEqual(
            list(inspect.signature(module.causal_conv1d).parameters), expected
        )
        self.assertEqual(
            list(inspect.signature(module.npu_causal_conv1d).parameters), expected
        )

    def test_mutable_raw_op_increments_state_version_after_launch(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor()
            result = module.npu_causal_conv1d(
                FakeTensor(), FakeTensor(), conv_states=state
            )

        self.assertEqual(result, "output")
        self.assertEqual(raw_calls, [state])
        self.assertEqual(incremented, [state])
        self.assertEqual(module.MUTATED_ARGUMENTS["npu_causal_conv1d"], ("conv_states",))

    def test_mutable_state_requiring_grad_is_rejected_before_launch(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor(requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, r"conv_states.*must not require gradients"
            ):
                module.npu_causal_conv1d(
                    FakeTensor(), FakeTensor(), conv_states=state
                )

        self.assertEqual(raw_calls, [])
        self.assertEqual(incremented, [])

    def test_high_level_fn_owns_mutation_contract(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor()
            result = module.causal_conv1d_fn(
                FakeTensor(),
                FakeTensor(),
                None,
                conv_states=state,
            )

        self.assertEqual(result, "output")
        self.assertEqual(raw_calls, [state])
        self.assertEqual(incremented, [state])
        self.assertEqual(module.MUTATED_ARGUMENTS["causal_conv1d_fn"], ("conv_states",))

    def test_high_level_update_rejects_state_grad(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor(requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, r"conv_state.*must not require gradients"
            ):
                module.causal_conv1d_update(
                    FakeTensor(),
                    state,
                    FakeTensor(),
                )

        self.assertEqual(raw_calls, [])
        self.assertEqual(incremented, [])

    def test_causal_conv1d_compatibility_api_warns_and_is_exported(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)
        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor()
            with self.assertWarnsRegex(FutureWarning, "2027/02"):
                result = module.causal_conv1d(
                    FakeTensor(),
                    FakeTensor(),
                    conv_states=state,
                    query_start_loc=[0, 1],
                    cache_indices=[0],
                    initial_state_mode=[1],
                    num_accepted_tokens=[1],
                    activation_mode=1,
                    pad_slot_id=-7,
                    run_mode=1,
                    head_num=2,
                )

        self.assertEqual(result, "output")
        self.assertEqual(raw_calls, [state])
        self.assertEqual(incremented, [state])
        self.assertIn("causal_conv1d", module.__all__)
        self.assertEqual(module.MUTATED_ARGUMENTS["causal_conv1d"], ("conv_states",))
        self.assertEqual(module.BACKWARD_OPS["causal_conv1d"], "causal_conv1d_bwd")

    def test_causal_conv1d_fn_uses_registered_backward_mapping(self):
        raw_calls = []
        module, spec, modules = load_ascendc_module(raw_calls)
        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)

        self.assertEqual(
            module.BACKWARD_OPS["causal_conv1d_fn"],
            "causal_conv1d_bwd",
        )
        self.assertEqual(
            module.BACKWARD_OPS["npu_causal_conv1d_fn"],
            "npu_causal_conv1d_bwd",
        )

    def test_causal_conv1d_compatibility_api_forwards_original_arguments(self):
        raw_calls = []
        module, spec, modules = load_ascendc_module(raw_calls)
        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)

        direct_call = mock.Mock(return_value="output")
        values = {
            "x": object(),
            "weight": object(),
            "bias": object(),
            "conv_states": object(),
            "query_start_loc": [0, 1],
            "cache_indices": [2],
            "initial_state_mode": [1],
            "num_accepted_tokens": [1],
        }
        with mock.patch.object(
            module,
            "_get_direct_op",
            return_value=direct_call,
        ) as get_direct_op:
            with self.assertWarnsRegex(FutureWarning, "2027/02"):
                result = module.causal_conv1d(
                    values["x"],
                    values["weight"],
                    values["bias"],
                    values["conv_states"],
                    query_start_loc=values["query_start_loc"],
                    cache_indices=values["cache_indices"],
                    initial_state_mode=values["initial_state_mode"],
                    num_accepted_tokens=values["num_accepted_tokens"],
                    activation_mode=1,
                    pad_slot_id=-7,
                    run_mode=1,
                    head_num=2,
                )

        self.assertEqual(result, "output")
        get_direct_op.assert_called_once_with("npu_causal_conv1d")
        direct_call.assert_called_once_with(
            **values,
            activation_mode=1,
            pad_slot_id=-7,
            run_mode=1,
            head_num=2,
        )

    def test_causal_conv1d_compatibility_api_keeps_prefill_autograd_binding(self):
        import torch

        raw_calls = []
        module, spec, modules = load_ascendc_module(raw_calls)
        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)

        forward_call = mock.Mock(side_effect=lambda **kwargs: kwargs["x"].clone())
        backward_call = mock.Mock(
            side_effect=lambda **kwargs: (
                torch.ones_like(kwargs["x"]),
                torch.full_like(kwargs["weight"], 2),
                torch.full((kwargs["weight"].shape[-1],), 3.0),
                torch.empty(0),
            )
        )

        def get_direct_op(name):
            if name == "npu_causal_conv1d":
                return forward_call
            if name == "npu_causal_conv1d_bwd":
                return backward_call
            raise AssertionError(f"unexpected direct op: {name}")

        x = torch.ones((2, 2), requires_grad=True)
        weight = torch.ones((2, 2), requires_grad=True)
        bias = torch.ones((2,), requires_grad=True)
        state = torch.zeros((1, 1, 2))
        with mock.patch.object(module, "_get_direct_op", side_effect=get_direct_op):
            with self.assertWarnsRegex(FutureWarning, "2027/02"):
                output = module.causal_conv1d(x, weight, bias, state)
            output.sum().backward()

        forward_call.assert_called_once()
        backward_call.assert_called_once()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))
        self.assertTrue(torch.equal(weight.grad, torch.full_like(weight, 2)))
        self.assertTrue(torch.equal(bias.grad, torch.full_like(bias, 3)))

    def test_torch_npu_ops_compat_exports_both_legacy_names(self):
        raw_calls = []
        module, spec, modules = load_ascendc_module(raw_calls)
        fake_torch_npu = types.ModuleType("torch_npu")
        fake_torch_npu.ops = types.SimpleNamespace()
        modules["torch_npu"] = fake_torch_npu
        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module.install_torch_npu_ops_compat()

        self.assertIs(fake_torch_npu.ops.causal_conv1d, module.causal_conv1d)
        self.assertIs(
            fake_torch_npu.ops.npu_causal_conv1d,
            module.npu_causal_conv1d,
        )

    def test_recurrent_kda_increments_mutable_state_version(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            inputs = [FakeTensor() for _ in range(5)]
            state = FakeTensor()
            result = module.npu_recurrent_kda(*inputs, state, cu_seqlens=FakeTensor())

        self.assertEqual(result, ("output", state))
        self.assertEqual(raw_calls, [state])
        self.assertEqual(incremented, [state])
        self.assertEqual(module.MUTATED_ARGUMENTS["npu_recurrent_kda"], ("initial_state",))


if __name__ == "__main__":
    unittest.main()
