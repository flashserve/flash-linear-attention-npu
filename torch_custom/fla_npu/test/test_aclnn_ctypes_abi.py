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

import ctypes
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_DIR = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc"


def load_aclnn_ctypes_module():
    package_name = "fla_npu_test_aclnn_ctypes"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASCENDC_DIR)]
    sys.modules[package_name] = package

    for module_name in ("_runtime", "_kda_policy", "_aclnn_ctypes"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            ASCENDC_DIR / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}._aclnn_ctypes"]


ACLNN_CTYPES = load_aclnn_ctypes_module()


class FakeTensor:
    def __init__(self, shape, dtype=None, *, device_type="npu", contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device_type)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


class FakeCallContext:
    def __init__(self):
        self.descriptor_names = []
        self.descriptor_metadata = []

    def tensor(
        self,
        tensor,
        name,
        *,
        acl_format_override=None,
        storage_shape_override=None,
    ):
        self.descriptor_names.append(name)
        self.descriptor_metadata.append(
            (name, tensor, acl_format_override, storage_shape_override)
        )
        return ctypes.c_void_p(0x1000 + len(self.descriptor_names))

    def int_array(self, values):
        del values
        self.descriptor_names.append("query_start_loc")
        return ctypes.c_void_p(0x2000)


class AclnnCtypesAbiTest(unittest.TestCase):
    def test_gdn_training_and_inference_output_contract(self):
        import inspect

        function = ACLNN_CTYPES.npu_chunk_gated_delta_rule_fwd
        self.assertIs(inspect.signature(function).parameters["disable_recompute"].default, True)
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.bfloat16 = object()
        q = FakeTensor((1, 2, 65, 128), fake_torch.bfloat16)
        v = FakeTensor((1, 4, 65, 256), fake_torch.bfloat16)
        g = FakeTensor((1, 65, 4), fake_torch.float32)
        beta = FakeTensor(g.shape, fake_torch.bfloat16)
        state = FakeTensor((1, 4, 128, 256), fake_torch.float32)
        captured = {}

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["tensors"] = {row[0]: row[1] for row in context.descriptor_metadata}
            return outputs

        modes = (("default", {}, True), ("none", {"disable_recompute": None}, True),
                 ("training", {"disable_recompute": True}, True),
                 ("inference", {"disable_recompute": False}, False))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty), \
                mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
            for mode, options, training in modes:
                for with_h in (False, True):
                    for with_final_state in (False, True):
                        with self.subTest(mode=mode, h=with_h, final_state=with_final_state):
                            outputs = function(
                                q, q, v, g, beta, initial_state=state,
                                output_final_state=with_final_state,
                                return_intermediate_states=with_h, a_log=None, dt_bias=None, **options,
                            )
                            tensors = captured["tensors"]
                            self.assertEqual(captured["name"], "aclnnChunkGatedDeltaRuleFwd")
                            self.assertEqual(len(captured["args"]), 27)
                            self.assertEqual(len(outputs), 10)
                            self.assertIsNone(outputs[4])
                            self.assertIs(outputs[0], tensors["o"])
                            self.assertEqual(outputs[0].shape, (1, 65, 4, 256))
                            self.assertIs(outputs[1], tensors["final_state"])
                            self.assertEqual(outputs[1] is not None, with_final_state)
                            if training:
                                self.assertIs(outputs[2], tensors["g_cumsum"])
                                self.assertIs(outputs[3], tensors["A"])
                                self.assertEqual(outputs[2].shape, (1, 65, 4))
                                self.assertIs(outputs[2].dtype, fake_torch.float32)
                                self.assertEqual(outputs[3].shape, (1, 4, 65, 64))
                            else:
                                self.assertIsNone(tensors["g_cumsum"])
                                self.assertIsNone(tensors["A"])
                                self.assertIsNone(outputs[2])
                                self.assertIsNone(outputs[3])
                            if with_h:
                                self.assertIs(outputs[5], tensors["h"])
                                self.assertEqual(outputs[5].shape, (1, 4, 2, 128, 256))
                            else:
                                self.assertIsNone(tensors["h"])
                                self.assertIsNone(outputs[5])

    def test_gdn_norm_outputs_and_input_aliases(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.bfloat16 = object()
        captured = {}

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_call(name, build_args, outputs):
            context = FakeCallContext()
            captured["args"] = build_args(context)
            captured["tensors"] = {row[0]: row[1] for row in context.descriptor_metadata}
            return outputs

        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty), \
                mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call):
            for layout in ("BNSD", "BSND", "NTD", "TND"):
                shape = (1, 65, 2, 128) if layout in ("BSND", "TND") else (1, 2, 65, 128)
                q = FakeTensor(shape, fake_torch.bfloat16)
                k = FakeTensor(shape, fake_torch.bfloat16)
                g = FakeTensor((1, 65, 2), fake_torch.float32)
                for enabled in (False, True):
                    for training in (False, True):
                        with self.subTest(layout=layout, norm=enabled, training=training):
                            outputs = ACLNN_CTYPES.npu_chunk_gated_delta_rule_fwd(
                                q, k, q, g, g, layout=layout,
                                use_qk_l2norm_in_kernel=enabled, disable_recompute=training,
                            )
                            self.assertEqual(len(outputs), 10)
                            self.assertEqual(len(captured["args"]), 27)
                            q_hat, k_hat, q_rstd, k_rstd = outputs[6:]
                            if enabled:
                                self.assertIsNot(q_hat, q)
                                self.assertIsNot(k_hat, k)
                                for tensor in (q_hat, k_hat):
                                    self.assertEqual(tensor.shape, shape)
                                    self.assertIs(tensor.dtype, fake_torch.bfloat16)
                                for tensor in (q_rstd, k_rstd):
                                    self.assertEqual(tensor.shape, (1, 2, 65))
                                    self.assertIs(tensor.dtype, fake_torch.float32)
                                for name, tensor in zip(
                                    ("q_hat", "k_hat", "q_rstd", "k_rstd"), outputs[6:]
                                ):
                                    self.assertIs(captured["tensors"][name], tensor)
                            else:
                                self.assertIs(q_hat, q)
                                self.assertIs(k_hat, k)
                                self.assertIsNone(q_rstd)
                                self.assertIsNone(k_rstd)
                                for name in ("q_hat", "k_hat", "q_rstd", "k_rstd"):
                                    self.assertIsNone(captured["tensors"][name])

    def test_gdn_reserved_gate_arguments_rejected_before_launch(self):
        fake_torch = types.ModuleType("torch")
        q = FakeTensor((1, 2, 65, 128))
        g = FakeTensor((1, 65, 2))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(ACLNN_CTYPES, "_call_aclnn") as launch:
            for options in ({"a_log": g}, {"dt_bias": g}, {"use_gate_in_kernel": True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    ACLNN_CTYPES.npu_chunk_gated_delta_rule_fwd(q, q, q, g, g, **options)
            launch.assert_not_called()

    def test_chunk_gdn_bwd_intra_signature_has_no_debug_stage(self):
        import inspect

        expected_argtypes = [
            *([ctypes.c_void_p] * 9),
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_bool,
            *([ctypes.c_void_p] * 3),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkGdnBwdIntra"],
            expected_argtypes,
        )
        signature = inspect.signature(ACLNN_CTYPES.npu_chunk_gdn_bwd_intra)
        self.assertNotIn("stage", signature.parameters)
        self.assertIs(signature.parameters["use_exp2"].default, True)

        captured = {}
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.float16 = object()
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_empty_like(tensor, **kwargs):
            return FakeTensor(tensor.shape, kwargs.get("dtype", tensor.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            return outputs

        dtype = fake_torch.bfloat16
        q = FakeTensor((1, 3, 65, 128), dtype)
        k = FakeTensor(q.shape, dtype)
        v = FakeTensor((1, 6, 65, 128), dtype)
        g = FakeTensor((1, 6, 65), fake_torch.float32)
        beta = FakeTensor((1, 6, 65), fake_torch.bfloat16)
        a = FakeTensor((1, 6, 65, 64), dtype)
        d_o = FakeTensor(v.shape, dtype)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
                with mock.patch.object(
                    ACLNN_CTYPES, "_empty_like", side_effect=fake_empty_like
                ):
                    with mock.patch.object(
                        ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn
                    ):
                        outputs = ACLNN_CTYPES.npu_chunk_gdn_bwd_intra(
                            q, k, v, g, beta, a, d_o, 0.125, 64
                        )

        operator_argtypes = expected_argtypes[:-2]
        self.assertEqual(captured["name"], "aclnnChunkGdnBwdIntra")
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertTrue(captured["args"][11].value)

    def test_chunk_gated_delta_rule_bwd_dhu_signature_and_default_use_exp2(self):
        import inspect

        import torch

        expected_argtypes = [
            *([ctypes.c_void_p] * 11),
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_bool,
            *([ctypes.c_void_p] * 3),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkGatedDeltaRuleBwdDhu"],
            expected_argtypes,
        )
        self.assertIs(
            inspect.signature(ACLNN_CTYPES.npu_chunk_gated_delta_rule_bwd_dhu)
            .parameters["use_exp2"]
            .default,
            False,
        )

        captured = {}

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_empty_like(tensor, **kwargs):
            return FakeTensor(tensor.shape, kwargs.get("dtype", tensor.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            return outputs

        dtype = torch.float16
        q = FakeTensor((1, 2, 64, 128), dtype)
        state = FakeTensor((1, 2, 64, 128), dtype)
        g = FakeTensor((1, 2, 64), torch.float32)
        with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
            with mock.patch.object(ACLNN_CTYPES, "_empty_like", side_effect=fake_empty_like):
                with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                    ACLNN_CTYPES.npu_chunk_gated_delta_rule_bwd_dhu(
                        q, q, q, state, state, scale=0.125, chunk_size=64, g=g
                    )

        operator_argtypes = expected_argtypes[:-2]
        self.assertEqual(captured["name"], "aclnnChunkGatedDeltaRuleBwdDhu")
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertFalse(captured["args"][13].value)
        self.assertFalse(captured["args"][14].value)

    def test_recurrent_gated_delta_rule_requires_at_least_one_gate_before_launch(self):
        with mock.patch.object(ACLNN_CTYPES, "_call_aclnn") as call_aclnn:
            with self.assertRaisesRegex(
                RuntimeError,
                r"^npu_recurrent_gated_delta_rule: either g or gk must be provided\.$",
            ):
                ACLNN_CTYPES.npu_recurrent_gated_delta_rule(
                    None,
                    None,
                    None,
                    None,
                    beta=None,
                    actual_seq_lengths=None,
                    ssm_state_indices=None,
                )

        call_aclnn.assert_not_called()

    def test_recurrent_gated_delta_rule_signature_matches_aclnn_prototype(self):
        expected_argtypes = [
            *([ctypes.c_void_p] * 10),
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnRecurrentGatedDeltaRule"],
            expected_argtypes,
        )

    def test_recurrent_gated_delta_rule_wrapper_uses_nd_descriptors(self):
        captured = {}
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()
        fake_torch.int32 = object()

        inputs = {
            "query": FakeTensor((5, 4, 128), fake_torch.bfloat16),
            "key": FakeTensor((5, 4, 128), fake_torch.bfloat16),
            "value": FakeTensor((5, 8, 128), fake_torch.bfloat16),
            "state": FakeTensor(
                (5, 8, 128, 128),
                fake_torch.float32,
                contiguous=False,
            ),
            "beta": FakeTensor((5, 8), fake_torch.bfloat16),
            "actual_seq_lengths": FakeTensor((3,), fake_torch.int32),
            "ssm_state_indices": FakeTensor((5,), fake_torch.int32),
            "g": FakeTensor((5, 8), fake_torch.float32),
            "gk": FakeTensor((5, 8, 128), fake_torch.float32),
            "num_accepted_tokens": FakeTensor((2,), fake_torch.int32),
        }

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["descriptor_names"] = context.descriptor_names
            captured["descriptor_metadata"] = context.descriptor_metadata
            return outputs

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
                with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                    output = ACLNN_CTYPES.npu_recurrent_gated_delta_rule(
                        inputs["query"],
                        inputs["key"],
                        inputs["value"],
                        inputs["state"],
                        beta=inputs["beta"],
                        scale=0.125,
                        actual_seq_lengths=inputs["actual_seq_lengths"],
                        ssm_state_indices=inputs["ssm_state_indices"],
                        num_accepted_tokens=inputs["num_accepted_tokens"],
                        g=inputs["g"],
                        gk=inputs["gk"],
                    )

        operator_argtypes = ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES[
            "aclnnRecurrentGatedDeltaRule"
        ][:-2]
        self.assertEqual(captured["name"], "aclnnRecurrentGatedDeltaRule")
        self.assertEqual(output.shape, inputs["value"].shape)
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertEqual(
            captured["descriptor_names"],
            [
                "query",
                "key",
                "value",
                "beta",
                "state",
                "actual_seq_lengths",
                "ssm_state_indices",
                "g",
                "gk",
                "num_accepted_tokens",
                "out",
            ],
        )
        for name, tensor, format_override, storage_shape in captured[
            "descriptor_metadata"
        ]:
            self.assertEqual(format_override, ACLNN_CTYPES.ACL_FORMAT_ND, name)
            expected_storage_shape = None if name == "state" else tensor.shape
            self.assertEqual(storage_shape, expected_storage_shape, name)

    def test_causal_conv1d_bwd_signature_matches_aclnn_prototype(self):
        expected_argtypes = [
            *([ctypes.c_void_p] * 7),
            ctypes.c_int64,
            ctypes.c_char_p,
            *([ctypes.c_void_p] * 4),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnCausalConv1dBwd"],
            expected_argtypes,
        )

    def test_causal_conv1d_bwd_wrapper_builds_one_value_per_operator_argtype(self):
        captured = {}

        def fake_empty(shape, like, **kwargs):
            del like, kwargs
            return FakeTensor(shape)

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["descriptor_names"] = context.descriptor_names
            return outputs

        x = FakeTensor((2, 17, 80))
        weight = FakeTensor((4, 80))
        dy = FakeTensor((2, 17, 80))
        with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
            with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                outputs = ACLNN_CTYPES.npu_causal_conv1d_bwd(
                    x=x,
                    y=None,
                    weight=weight,
                    dy=dy,
                    input_layout="BSH",
                )

        operator_argtypes = ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnCausalConv1dBwd"][:-2]
        self.assertEqual(captured["name"], "aclnnCausalConv1dBwd")
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertEqual(
            captured["descriptor_names"],
            [
                "x",
                "y",
                "weight",
                "dy",
                "initial_state",
                "dht",
                "query_start_loc",
                "dx",
                "dw",
                "db",
                "dh0",
            ],
        )


@unittest.skipUnless("--npu" in sys.argv, "pass --npu to run Ascend 950 integration tests")
class NormOutputsTest(unittest.TestCase):
    def test_export_and_backward(self):
        import itertools
        import torch
        import torch_npu  # noqa: F401
        from fla_npu.ops import ascendc

        torch.npu.set_device(0)
        for layout, enabled, varlen in itertools.product(
            ("BNSD", "BSND", "NTD", "TND"), (False, True), (False, True)
        ):
            with self.subTest(layout=layout, norm=enabled, varlen=varlen):
                torch.manual_seed(42)
                # Hk != T != Hv exposes accidental transposition/head expansion.
                b, hk, hv, t, d = 1, 2, 4, 65, 128
                q = torch.randn(b, hk, t, d).to(torch.bfloat16)
                k = torch.randn_like(q)
                if not enabled:
                    q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
                    k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
                v = torch.randn(b, hv, t, d).to(q.dtype)
                g = -torch.rand(b, hv, t) * 0.1
                beta = torch.sigmoid(torch.randn(b, hv, t))
                q, k, v, g, beta = [x.npu() for x in (q, k, v, g, beta)]
                cu = [0, 1, t] if varlen else None
                indices = [0, 0, 1, 0] if varlen else None
                prep = ascendc.npu_chunk_gated_delta_rule_fwd_prepare(
                    q, k, v, g, beta, chunk_size=64,
                    use_qk_l2norm_in_kernel=enabled, use_exp2=True,
                    cu_seqlens=cu, chunk_indices=indices,
                )
                torch.npu.synchronize()
                print(f"PREPARE_DONE {layout=} {enabled=} {varlen=}", flush=True)
                sequence_major = layout in ("BSND", "TND")
                public = [x.transpose(1, 2).contiguous() if sequence_major else x for x in (q, k, v)]
                common = dict(layout=layout, use_exp2=True, cu_seqlens=cu, chunk_indices=indices)
                outputs = ascendc.npu_chunk_gated_delta_rule_fwd(
                    *public, g.transpose(1, 2).contiguous(), beta.transpose(1, 2).contiguous(),
                    use_qk_l2norm_in_kernel=enabled, output_final_state=True, **common,
                )
                torch.npu.synchronize()
                print(f"FWD_DONE {layout=} {enabled=} {varlen=}", flush=True)
                self.assertEqual(len(outputs), 10)
                hats = outputs[6:]
                if enabled:
                    for actual, expected in zip(hats[:2], prep[:2]):
                        if sequence_major:
                            expected = expected.transpose(1, 2)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for actual, expected in zip(hats[2:], prep[2:4]):
                        self.assertEqual(actual.shape, (b, hk, t))
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                else:
                    self.assertIs(hats[0], public[0])
                    self.assertIs(hats[1], public[1])
                    self.assertIsNone(hats[2])
                    self.assertIsNone(hats[3])
                # Exporting intermediates must not change the mathematical forward.
                explicit = ascendc.npu_chunk_gated_delta_rule_fwd(
                    hats[0], hats[1], public[2], g.transpose(1, 2).contiguous(),
                    beta.transpose(1, 2).contiguous(), output_final_state=True,
                    disable_recompute=False, **common,
                )
                torch.npu.synchronize()
                print("EXPLICIT_FWD_DONE", flush=True)
                for actual, expected in zip(outputs[:2], explicit[:2]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertIsNone(explicit[2])
                self.assertIsNone(explicit[3])
                gradients = ascendc.npu_chunk_gated_delta_rule_bwd(
                    hats[0], hats[1], public[2], outputs[2], beta.transpose(1, 2).contiguous(),
                    outputs[3], torch.ones_like(outputs[0]), d ** -0.5,
                    use_qk_l2norm_in_kernel=enabled, q_rstd=hats[2], k_rstd=hats[3], **common,
                )
                torch.npu.synchronize()
                print("BWD_DONE", flush=True)
                for tensor in (*outputs, *gradients):
                    if tensor is not None:
                        self.assertTrue(torch.isfinite(tensor).all().item())
                self.assertEqual(gradients[0].shape, public[0].shape)
                self.assertEqual(gradients[1].shape, public[1].shape)


if __name__ == "__main__":
    if "--npu" in sys.argv:
        sys.argv.remove("--npu")
    unittest.main()
