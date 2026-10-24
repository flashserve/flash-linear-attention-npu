"""CP 边界组织逻辑的 CPU 检查；这些测试不验证 NPU 内核。

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest discover -s tests -p test_cp_examples.py
"""
import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "cp"))
import chunk_kda_cp
import chunk_gdn_cp


class BoundaryChecks:
    def test_dht_reaches_dhu_without_padding(self):
        x = self.example.make_inputs(128, 2, 19)
        dht = torch.randn(1, 2, 128, 128).bfloat16()
        dv = torch.zeros_like(x["v"])
        dh = torch.zeros(1, 2, 2, 128, 128).bfloat16()
        fake = types.ModuleType("fla_npu.ops.ascendc")
        calls = []

        def dhu(*args, **kwargs):
            calls.append((args, kwargs))
            return dh, None, dv

        fake.chunk_gated_delta_rule_bwd_dhu = dhu
        with patch.dict(sys.modules, {"fla_npu.ops.ascendc": fake}):
            for terminal in (dht, torch.zeros_like(dht)):
                actual = self.example.dhu_with_boundary(
                    x["q"], x["k"], x["k"], x["do"], dv, x["g"], terminal,
                    scale=128 ** -0.5)
                args, kwargs = calls[-1]
                self.assertIs(kwargs["dht"], terminal)
                self.assertIs(kwargs["gK" if self.kind == "kda" else "g"], x["g"])
                for got, expected in zip(args[:5], (x["q"], x["k"], x["k"], x["do"], dv)):
                    self.assertIs(got, expected)
                self.assertIs(actual[0], dh)
                self.assertIs(actual[1], dv)

    def test_prefix_suffix_and_empty_neighbours(self):
        # 使用不同且不可交换的状态变换，保证颠倒任一合并顺序都会改变结果。
        summaries = []
        for i in range(4):
            m = torch.eye(128).unsqueeze(0)
            m[0, i, (i + 1) % 4] = i + 1
            h = torch.full((1, 128, 128), float(i + 1))
            summaries.append(torch.cat((h, m), -1))
        calls = []

        def gather(outputs, local):
            for dst, src in zip(outputs, summaries):
                dst.copy_(src)

        def merge(h, ag, count, rank, *, forward, state_v_first):
            calls.append((count, rank, forward))
            indices = range(rank - count, rank) if forward else range(rank + count, rank, -1)
            h.zero_()
            for i in indices:
                h.copy_(ag[i, :, :, 128:] @ h + ag[i, :, :, :128])

        fake = types.ModuleType("fla_npu.ops.ascendc")
        fake.merge_fwd_bwd_kernel = merge
        with patch.dict(sys.modules, {"fla_npu.ops.ascendc": fake}), \
                patch.object(self.example.dist, "all_gather", side_effect=gather) as transport:
            for forward in (True, False):
                for rank in range(4):
                    result = self.example.merge_boundary(summaries[rank], forward=forward, rank=rank, world=4)
                    expected = torch.zeros(1, 128, 128)
                    indices = list(range(4))[:rank] if forward else list(range(4))[rank + 1:][::-1]
                    for i in indices:
                        expected = summaries[i][..., 128:] @ expected + summaries[i][..., :128]
                    torch.testing.assert_close(result, expected.unsqueeze(0).bfloat16())
            self.assertEqual(transport.call_count, 8)  # 首尾进程也必须参与通信
        self.assertEqual(calls, [(1, 1, True), (2, 2, True), (3, 3, True),
                                 (3, 0, False), (2, 1, False), (1, 2, False)])

    def test_world_one_does_not_merge_own_summary(self):
        fake = types.ModuleType("fla_npu.ops.ascendc")
        fake.merge_fwd_bwd_kernel = lambda *a, **k: self.fail("world=1 must skip merge")
        with patch.dict(sys.modules, {"fla_npu.ops.ascendc": fake}):
            for forward in (True, False):
                result = self.example.merge_boundary(torch.ones(2, 128, 256), forward=forward, rank=0, world=1)
                self.assertEqual(tuple(result.shape), (1, 2, 128, 128))
                self.assertEqual(result.count_nonzero().item(), 0)

    def test_dht_contains_only_future_loss(self):
        torch.set_num_threads(1)
        for kind in (self.kind,):
            with self.subTest(kind=kind):
                x = self.example.make_inputs(128, 1, 7)
                x["do"][:, :, 64:] = 0
                _, states, dht = self.example.recurrent_reference(x, 128 ** -0.5, 64)
                self.assertGreater(states[1].abs().max().item(), 0)
                self.assertEqual(dht[1].count_nonzero().item(), 0)
                self.assertEqual(dht[2].count_nonzero().item(), 0)

    def test_future_loss_reaches_previous_rank_inputs(self):
        torch.set_num_threads(1)
        for kind in (self.kind,):
            with self.subTest(kind=kind):
                x = self.example.make_inputs(128, 1, 11)
                x["do"][:, :, :64] = 0
                out, _, dht = self.example.recurrent_reference(x, 128 ** -0.5, 64)
                self.assertGreater(dht[1].abs().max().item(), 1e-4)
                self.assertGreater(out["dv"][:, :, :64].abs().max().item(), 1e-4)


class KdaBoundaryTests(BoundaryChecks, unittest.TestCase):
    example = chunk_kda_cp
    kind = "kda"


class GdnBoundaryTests(BoundaryChecks, unittest.TestCase):
    example = chunk_gdn_cp
    kind = "gdn"


class GdnGvaTests(unittest.TestCase):
    def test_arguments(self):
        self.assertEqual(chunk_gdn_cp.parse_args(["--heads", "3"]).value_heads, 3)
        self.assertEqual(chunk_gdn_cp.parse_args(["--heads", "2", "--value-heads", "6"]).value_heads, 6)
        for hv in (0, 3, 10, 130):
            with self.subTest(hv=hv), self.assertRaises(SystemExit):
                chunk_gdn_cp.parse_args(["--heads", "2", "--value-heads", str(hv)])

    def test_grouped_reference(self):
        """逐 value head 独立计算，核对共享 Q/K 的梯度求和及不同 head 的边界。"""
        torch.set_num_threads(1)
        for ratio in (2, 3, 4):
            with self.subTest(ratio=ratio):
                x = chunk_gdn_cp.make_inputs(128, 2, 43, 2 * ratio)
                actual, states, dstates = chunk_gdn_cp.recurrent_reference(x, 128 ** -0.5, 64)
                expected = {n: torch.zeros_like(t) for n, t in actual.items()}
                for hv in range(2 * ratio):
                    hk = hv // ratio
                    one = {n: t[:, hk:hk + 1] if n in ("q", "k") else t[:, hv:hv + 1]
                           for n, t in x.items()}
                    result, hs, dhs = chunk_gdn_cp.recurrent_reference(one, 128 ** -0.5, 64)
                    for name, value in result.items():
                        h = hk if name in ("dq", "dk") else hv
                        expected[name][:, h:h + 1] += value
                    torch.testing.assert_close(states[1][:, hv:hv + 1], hs[1])
                    torch.testing.assert_close(dstates[1][:, hv:hv + 1], dhs[1])
                for name in actual:
                    torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
                # 组内贡献必须求和而不是平均；此错误应被当前精度判据拒绝。
                self.assertFalse(chunk_gdn_cp.metric(actual["dk"] / ratio, expected["dk"], 0.008)[0])


if __name__ == "__main__":
    unittest.main()
