"""公开返回值集成回归；使用已安装的完整 wheel，在 Ascend 950 上执行。"""

import itertools
import unittest

import torch
import torch_npu  # noqa: F401
from fla_npu.ops import ascendc


class NormOutputsTest(unittest.TestCase):
    def test_export_and_backward(self):
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
    unittest.main()
