"""CP 边界组织逻辑的 CPU 检查；这些测试不验证 NPU 内核。

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest discover -s tests -p test_cp_examples.py
"""
import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import chunk_kda_cp
import chunk_gdn_cp


class BoundaryChecks:
    def test_synthetic_chunks_inject_terminal_state(self):
        torch.set_num_threads(1)
        gen = torch.Generator().manual_seed(19)
        q = torch.randn(1, 2, 64, 128, generator=gen).bfloat16()
        dht = torch.randn(1, 2, 128, 128, generator=gen).bfloat16()
        for key_gate in (self.kind == "kda",):
            gate = torch.zeros_like(q, dtype=torch.float32) if key_gate else torch.zeros(1, 2, 64)
            extended = self.example.append_terminal_boundary(q, q, q, q, q, gate, dht, 128 ** -0.5)
            qe, ke, we, doe, dve, ge = extended
            for value in extended[:5]:
                torch.testing.assert_close(value[:, :, :64], q)
            self.assertEqual(ke[:, :, 64:].count_nonzero().item(), 0)
            self.assertEqual(we[:, :, 64:].count_nonzero().item(), 0)
            self.assertEqual(dve[:, :, 64:].count_nonzero().item(), 0)
            self.assertEqual(ge[:, :, 64:].count_nonzero().item(), 0)
            state = torch.zeros_like(dht, dtype=torch.float32)
            for begin in (128, 64):  # 仅反向扫描追加的两个虚拟分块
                state += (qe[:, :, begin:begin + 64].float().transpose(-1, -2)
                          @ doe[:, :, begin:begin + 64].float()) * (128 ** -0.5)
            torch.testing.assert_close(state, dht.float(), rtol=0.004, atol=0.0001)

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


if __name__ == "__main__":
    unittest.main()
