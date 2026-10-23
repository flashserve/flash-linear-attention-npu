"""Stable parameter, platform and output contract regression.

Historical filename retained for CI. Python forwarding is checked offline;
allocation/dtype/alias contracts are verified on the real C++ Stable adapter
with --npu. ACLNN C prototypes are independently checked by op_abi_validate.
"""
from __future__ import annotations
import importlib.util
import inspect
import itertools
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ASCENDC_DIR = Path(__file__).resolve().parents[1] / "fla_npu/ops/ascendc"
spec = importlib.util.spec_from_file_location("stable_contract", ASCENDC_DIR / "_stable.py")
STABLE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STABLE)


class FakeTensor:
    def __init__(self, shape, index=0):
        self.shape = shape
        self.device = types.SimpleNamespace(type="npu", index=index)
        self.requires_grad = False


class StableAbiContractTest(unittest.TestCase):
    def setUp(self):
        self.launch = mock.Mock()
        self.patches = [mock.patch.object(STABLE, "_op", return_value=self.launch),
                        mock.patch.object(STABLE, "_host_ints", side_effect=lambda x: x),
                        mock.patch.object(STABLE, "_current_stream_ptr", return_value=123)]
        for patch in self.patches:
            patch.start(); self.addCleanup(patch.stop)

    def test_gdn_forward_all_policy_flags_preserve_positions(self):
        q = FakeTensor((1, 2, 65, 128))
        tensors = [q, q, object(), object(), object()]
        outputs = tuple(object() for _ in range(10))
        self.launch.return_value = outputs
        for layout, norm, training, final, intermediate in itertools.product(
                ("BSND", "BNSD", "TND", "NTD"), (False, True), (False, True), (False, True), (False, True)):
            with self.subTest(layout=layout, norm=norm, training=training, final=final, intermediate=intermediate):
                self.launch.reset_mock()
                result = STABLE.npu_chunk_gated_delta_rule_fwd(
                    *tensors, layout=layout, disable_recompute=training,
                    use_qk_l2norm_in_kernel=norm, output_final_state=final,
                    return_intermediate_states=intermediate)
                self.assertIs(result, outputs)
                args = self.launch.call_args.args
                self.assertEqual(args[:5], tuple(tensors))
                self.assertEqual(args[8], ("BSND", "BNSD", "TND", "NTD").index(layout))
                self.assertEqual(args[9], 128 ** -0.5)
                self.assertEqual((args[12], args[15], args[16], args[17]), (norm, training, final, intermediate))
                self.assertEqual(args[-1], 123)
                self.launch.assert_called_once()

    def test_gdn_default_and_explicit_none_match_stable_baseline(self):
        self.assertIs(inspect.signature(STABLE.npu_chunk_gated_delta_rule_fwd).parameters["disable_recompute"].default, True)
        q = FakeTensor((1, 2, 65, 128))
        for options, expected in (({}, True), ({"disable_recompute": None}, False)):
            STABLE.npu_chunk_gated_delta_rule_fwd(q, q, q, q, q, **options)
            self.assertEqual(self.launch.call_args.args[15], expected)

    def test_gdn_reserved_gate_arguments_rejected_before_launch(self):
        for options in ({"a_log": object()}, {"dt_bias": object()}, {"use_gate_in_kernel": True}):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                STABLE.npu_chunk_gated_delta_rule_fwd(*(object() for _ in range(5)), **options)
        self.launch.assert_not_called()

    def test_recurrent_requires_gate_before_launch(self):
        with self.assertRaisesRegex(RuntimeError, "either g or gk"):
            STABLE.npu_recurrent_gated_delta_rule(None, None, None, None, beta=None,
                                                actual_seq_lengths=None, ssm_state_indices=None)
        self.launch.assert_not_called()

    def test_recurrent_tensor_order_and_one_state_version_bump(self):
        values = [FakeTensor((1,)) for _ in range(10)]
        with mock.patch.object(STABLE, "_OP_CACHE", {}), mock.patch.object(STABLE, "_INC_VERSION") as bump:
            STABLE.npu_recurrent_gated_delta_rule(*values[:4], beta=values[4],
                actual_seq_lengths=values[5], ssm_state_indices=values[6],
                num_accepted_tokens=values[7], g=values[8], gk=values[9], scale=0.25)
            self.assertEqual(self.launch.call_args.args, (*values, 0.25, 123))
            bump.assert_called_once_with((values[3],))

    def test_gdn_intra_signature_has_no_debug_stage(self):
        self.assertNotIn("debug_stage", inspect.signature(STABLE.npu_chunk_gdn_bwd_intra).parameters)

    def test_dhu_default_use_exp2(self):
        self.assertIs(inspect.signature(STABLE.npu_chunk_gated_delta_rule_bwd_dhu).parameters["use_exp2"].default, False)

    def test_finalize_platforms_and_chunk_parameters(self):
        for name, chunks in (("Ascend910B3", (64,128)), ("Ascend910C1", (64,128)),
                             ("Ascend910_9391", (64,128)), ("Ascend910_9381", (64,128)), ("Ascend950", (64,))):
            for chunk, index in itertools.product(chunks, (None, 0, 2)):
                q = FakeTensor((1, 1, 65, 128), index)
                npu = types.SimpleNamespace(current_device=mock.Mock(return_value=7), get_device_name=mock.Mock(return_value=name))
                with self.subTest(name=name, chunk=chunk, index=index), mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(npu=npu)}):
                    STABLE.npu_chunk_gated_delta_rule_bwd_finalize(*([q]*11), chunk_size=chunk)
                    self.assertEqual(self.launch.call_args.args[17], chunk)
                    npu.get_device_name.assert_called_once_with(7 if index is None else index)

    def test_finalize_unsupported_platforms_rejected(self):
        for name in ("Ascend910A", "Ascend310P3"):
            q = FakeTensor((1,))
            npu = types.SimpleNamespace(get_device_name=lambda index: name)
            with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(npu=npu)}):
                with self.assertRaisesRegex(RuntimeError, "only supports"):
                    STABLE.npu_chunk_gated_delta_rule_bwd_finalize(*([q]*11))
        self.launch.assert_not_called()

    def test_composite_backward_does_not_query_python_platform(self):
        npu = types.SimpleNamespace(get_device_name=mock.Mock(side_effect=AssertionError("unexpected platform probe")))
        with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(npu=npu)}):
            STABLE.npu_chunk_gated_delta_rule_bwd(*([object()]*7), 0.25)
        npu.get_device_name.assert_not_called()
        self.assertEqual(self.launch.call_args.args[17:19], (0.25, 64))


@unittest.skipUnless("--npu" in sys.argv, "pass --npu to run real allocation contracts")
class GdnAllocationContractTest(unittest.TestCase):
    def test_training_inference_final_state_and_intermediate_slots(self):
        import os
        import torch
        import torch_npu
        from fla_npu.ops import ascendc
        torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
        q = torch.zeros((1, 2, 65, 128), dtype=torch.bfloat16, device="npu")
        k = torch.zeros_like(q)
        v = torch.zeros((1, 4, 65, 128), dtype=q.dtype, device="npu")
        gate = torch.zeros((1, 65, 4), dtype=torch.float32, device="npu")
        beta = torch.zeros_like(gate)
        state = torch.zeros((1, 4, 128, 128), dtype=torch.float32, device="npu")
        for training, final, intermediate in itertools.product((False, True), repeat=3):
            with self.subTest(training=training, final=final, intermediate=intermediate):
                out = ascendc.npu_chunk_gated_delta_rule_fwd(
                    q, k, v, gate, beta, initial_state=state, use_exp2=True,
                    disable_recompute=training, output_final_state=final,
                    return_intermediate_states=intermediate)
                torch.npu.synchronize()
                self.assertEqual(len(out), 10)
                self.assertEqual(tuple(out[0].shape), (1, 65, 4, 128))
                self.assertEqual(out[0].dtype, q.dtype)
                self.assertEqual(out[1] is not None, final)
                if final:
                    self.assertEqual(tuple(out[1].shape), tuple(state.shape))
                    self.assertEqual(out[1].dtype, torch.float32)
                for slot, shape, dtype in ((2, (1, 65, 4), torch.float32),
                                           (3, (1, 4, 65, 64), q.dtype)):
                    self.assertEqual(out[slot] is not None, training)
                    if training:
                        self.assertEqual(tuple(out[slot].shape), shape)
                        self.assertEqual(out[slot].dtype, dtype)
                self.assertIsNone(out[4])
                self.assertEqual(out[5] is not None, intermediate)
                if intermediate:
                    self.assertEqual(tuple(out[5].shape), (1, 2, 4, 128, 128))
                self.assertIs(out[6], q)
                self.assertIs(out[7], k)
                self.assertIsNone(out[8])
                self.assertIsNone(out[9])


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
