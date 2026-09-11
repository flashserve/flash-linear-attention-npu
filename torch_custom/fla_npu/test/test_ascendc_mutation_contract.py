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

import inspect
import importlib.util
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

    def npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        *,
        beta,
        scale=1.0,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens=None,
        g=None,
        gk=None,
    ):
        del key, value, beta, scale, actual_seq_lengths, ssm_state_indices
        del num_accepted_tokens, g, gk
        raw_calls.append(state)
        return "output"

    ctypes_module.ASCENDC_CTYPES_OPS = {
        "npu_causal_conv1d": npu_causal_conv1d,
        "npu_recurrent_kda": npu_recurrent_kda,
        "npu_recurrent_gated_delta_rule": npu_recurrent_gated_delta_rule,
    }
    # Hermetic stub: importing the real `_thin` module would pull in the built
    # extension, so make `_get_thin_op` resolve to nothing and keep these tests
    # focused on the mutation wrapper.
    fake_thin = types.ModuleType("fla_npu.ops.ascendc._thin")
    modules = {
        "fla_npu": fake_fla_npu,
        "fla_npu.ops": fake_ops,
        "fla_npu.ops.ascendc._aclnn_ctypes": ctypes_module,
        "fla_npu.ops.ascendc._thin": fake_thin,
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
    def test_mutable_raw_op_increments_state_version_after_launch(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            state = FakeTensor()
            result = module.npu_causal_conv1d(FakeTensor(), FakeTensor(), conv_states=state)

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
            with self.assertRaisesRegex(RuntimeError, r"must not require gradients"):
                module.npu_causal_conv1d(FakeTensor(), FakeTensor(), conv_states=state)

        self.assertEqual(raw_calls, [])
        self.assertEqual(incremented, [])

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

    # ------------------------------------------------------------------
    # Fast-path equivalence with the previous signature.bind() reference.
    # ------------------------------------------------------------------

    @staticmethod
    def _reference_mutation(module, name, signature, args, kwargs):
        """Previous implementation: bind, apply defaults, then predicate."""

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        active = module.MUTATED_ARGUMENTS.get(name, ())
        declared = module.MUTATION_FLAGS.get(name)
        if declared is not None and not bound.arguments[declared[0]]:
            active = ()
        elif declared is None:
            predicate = module.MUTATION_PREDICATES.get(name)
            if predicate is not None and not predicate(bound.arguments):
                active = ()
        return [bound.arguments[argument] for argument in active]

    def _call_forms(self, module):
        """(label, name, args, kwargs) matrix covering both flag values."""

        nk, nv, dim = 8, 16, 128
        q = FakeTensor()
        one = lambda: FakeTensor()  # noqa: E731
        args5 = [one() for _ in range(5)]
        state = FakeTensor()
        kda_kwargs = dict(cu_seqlens=one(), ssm_state_indices=one())
        conv_kwargs = dict(query_start_loc=one())
        gdr_kwargs = dict(beta=one(), scale=0.088, actual_seq_lengths=one(),
                          ssm_state_indices=one())
        del nk, nv, dim
        return [
            # kda: mutated arg positional -> fast path decides from the flag.
            ("kda positional, flag omitted", "npu_recurrent_kda",
             [*args5, state], dict(kda_kwargs)),
            ("kda positional, flag=True", "npu_recurrent_kda",
             [*args5, state], dict(kda_kwargs, inplace_final_state=True)),
            ("kda positional, flag=False", "npu_recurrent_kda",
             [*args5, state], dict(kda_kwargs, inplace_final_state=False)),
            ("kda positional None state, flag=False", "npu_recurrent_kda",
             [*args5, None], dict(kda_kwargs, inplace_final_state=False)),
            # kda: mutated arg keyword -> reference (slow) path, same answer.
            ("kda keyword, flag=True", "npu_recurrent_kda", args5,
             dict(kda_kwargs, initial_state=state, inplace_final_state=True)),
            ("kda keyword, flag=False", "npu_recurrent_kda", args5,
             dict(kda_kwargs, initial_state=state, inplace_final_state=False)),
            ("kda initial_state defaulted", "npu_recurrent_kda", args5,
             dict(kda_kwargs)),
            # conv1d: no declared flag, so the mutated arg decides alone.
            ("conv1d conv_states positional", "npu_causal_conv1d",
             [FakeTensor(), FakeTensor(), None, state], dict(conv_kwargs)),
            ("conv1d conv_states keyword", "npu_causal_conv1d",
             [FakeTensor(), FakeTensor(), None],
             dict(conv_kwargs, conv_states=state)),
            # recurrent GDR: required positional state, no flag.
            ("gdr state positional", "npu_recurrent_gated_delta_rule",
             [FakeTensor(), FakeTensor(), FakeTensor(), state],
             dict(gdr_kwargs)),
            ("gdr state keyword", "npu_recurrent_gated_delta_rule",
             [FakeTensor(), FakeTensor(), FakeTensor()],
             dict(gdr_kwargs, state=state)),
        ]

    def test_fast_path_matches_bind_reference_for_every_call_form(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            for label, name, args, kwargs in self._call_forms(module):
                signature = inspect.signature(
                    module.ASCENDC_CTYPES_OPS[name])
                plan = module._mutation_plan(name, signature)
                fast_mutated, used_fast = module._resolve_mutation(
                    plan, signature,
                    module.MUTATION_PREDICATES.get(name),
                    tuple(args), dict(kwargs))
                expected = self._reference_mutation(
                    module, name, signature, tuple(args), dict(kwargs))
                self.assertEqual(
                    fast_mutated, expected,
                    f"{label}: fast path disagrees with bind reference")
                # The fast path applies exactly when the mutated tensor is
                # passed positionally within the signature's positional part.
                mutated = module.MUTATED_ARGUMENTS[name][0]
                positional = [
                    parameter.name
                    for parameter in signature.parameters.values()
                    if parameter.kind in module._POSITIONAL_KINDS
                ]
                expect_fast = (
                    mutated not in kwargs
                    and mutated in positional
                    and len(args) > positional.index(mutated)
                )
                self.assertEqual(
                    used_fast, expect_fast,
                    f"{label}: unexpected path choice")

    def test_kda_never_binds_when_state_is_positional(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            inputs = [FakeTensor() for _ in range(5)]
            state = FakeTensor()
            kwargs = dict(cu_seqlens=FakeTensor())
            # Fail loudly if the wrapper reaches for signature.bind().
            with mock.patch.object(inspect.Signature, "bind",
                                   side_effect=AssertionError("slow path used")):
                module.npu_recurrent_kda(*inputs, state,
                                         inplace_final_state=True, **kwargs)
                module.npu_recurrent_kda(*inputs, state,
                                         inplace_final_state=False, **kwargs)
                with self.assertRaises(AssertionError):
                    module.npu_recurrent_kda(*inputs, initial_state=state,
                                             inplace_final_state=True, **kwargs)

        # inplace writes back -> bump; scratch state -> no bump.
        self.assertEqual(incremented, [state])

    def test_declared_flag_default_must_match_signature(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            # The signature default is True; declaring False must fail loudly
            # instead of silently skipping a version bump on the hot path.
            module.MUTATION_FLAGS["npu_recurrent_kda"] = (
                "inplace_final_state", False)
            module._get_direct_op.cache_clear()
            with self.assertRaisesRegex(
                    RuntimeError, "disagrees with the operator signature default"):
                module._get_direct_op("npu_recurrent_kda")

    def test_unknown_declared_flag_argument_is_rejected(self):
        raw_calls = []
        incremented = []
        module, spec, modules = load_ascendc_module(raw_calls)
        modules["torch"] = fake_torch(incremented)

        with mock.patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module.MUTATION_FLAGS["npu_recurrent_kda"] = ("no_such_flag", True)
            module._get_direct_op.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "unknown argument"):
                module._get_direct_op("npu_recurrent_kda")


if __name__ == "__main__":
    unittest.main()
