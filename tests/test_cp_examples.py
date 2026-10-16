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


class GdnGvaTests(unittest.TestCase):
    def test_arguments(self):
        self.assertEqual(chunk_gdn_cp.parse_args(["--heads", "3"]).value_heads, 3)
        self.assertEqual(chunk_gdn_cp.parse_args(["--heads", "2", "--value-heads", "6"]).value_heads, 6)
        for hv in (0, 3, 10, 130):
            with self.subTest(hv=hv), self.assertRaises(SystemExit):
                chunk_gdn_cp.parse_args(["--heads", "2", "--value-heads", str(hv)])

    def test_grouped_reference_and_terminal_boundary(self):
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
                dht = dstates[1].bfloat16()
                qe, ke, we, doe, dve, ge = chunk_gdn_cp.append_terminal_boundary(
                    x["q"], x["k"], torch.zeros_like(x["v"]), x["do"],
                    torch.zeros_like(x["v"]), x["g"], dht, 128 ** -0.5)
                for hv in range(2 * ratio):
                    injected = qe[:, hv // ratio, -128:].float().transpose(-1, -2) @ doe[:, hv, -128:].float()
                    torch.testing.assert_close(injected * (128 ** -0.5), dht[:, hv].float(),
                                               rtol=0.004, atol=0.0001)
                self.assertEqual(tuple(we.shape), (1, 2 * ratio, 256, 128))
                for tail in (ke, we, dve, ge):
                    self.assertEqual(tail[:, :, -128:].count_nonzero().item(), 0)


class KdaArch22BackwardTests(unittest.TestCase):
    def test_split_backward_matches_recurrence_with_boundary_loss(self):
        """独立逐词元参考验证 A2/A3 拆分反向，包含非零首末状态和共享参数梯度。"""
        torch.set_num_threads(1)
        x = {n: t.float() for n, t in chunk_kda_cp.make_inputs(128, 3, 29).items()}
        x["A_log"] = torch.tensor([-0.2, 0.1, 0.3])
        x["dt_bias"] = torch.linspace(-0.1, 0.1, 384).reshape(3, 128)
        shape = (1, 3, 2, 64, 128)
        gen = torch.Generator().manual_seed(31)
        h = torch.randn(1, 2, 3, 128, 128, generator=gen) * 0.05
        dh = torch.randn(h.shape, generator=gen) * 0.1
        scale = 128 ** -0.5
        leaves = {n: t.clone().requires_grad_(True) for n, t in x.items() if n != "do"}
        raw = leaves["g"] + leaves["dt_bias"][None, :, None, :]
        gate = -5 * torch.sigmoid(leaves["A_log"].exp()[None, :, None, None] * raw)
        loss = torch.zeros(())
        for block in range(2):
            state = h[:, block]
            for t in range(block * 64, (block + 1) * 64):
                state = state * gate[:, :, t].exp().unsqueeze(-1)
                key = leaves["k"][:, :, t]
                delta = (leaves["v"][:, :, t] - (key.unsqueeze(-1) * state).sum(-2))
                delta = delta * leaves["beta"][:, :, t, None]
                state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
                out = (leaves["q"][:, :, t, :, None] * state).sum(-2) * scale
                loss = loss + (out * x["do"][:, :, t]).sum()
            loss = loss + (state * dh[:, block]).sum()
        loss.backward()

        # 构造前向保存量；测试标杆仍是上面的独立递推，不对本段求导。
        gc = gate.detach().reshape(shape).cumsum(-2) / torch.log(torch.tensor(2.0))
        qc, kc, vc = (x[n].reshape(shape) for n in ("q", "k", "v"))
        beta = x["beta"].reshape(1, 3, 2, 64, 1)
        weights = torch.exp2(gc.unsqueeze(-2) - gc.unsqueeze(-3))
        kk = (kc.unsqueeze(-2) * kc.unsqueeze(-3) * weights).sum(-1)
        inverse = torch.linalg.inv(torch.eye(64) + torch.tril(beta * kk, diagonal=-1))
        aqk = torch.tril((qc.unsqueeze(-2) * kc.unsqueeze(-3) * weights).sum(-1)) * scale
        hc = h.permute(0, 2, 1, 3, 4)
        vn = inverse @ (beta * vc - (beta * kc * torch.exp2(gc)) @ hc)
        saved = (gc.reshape_as(x["g"]), aqk.reshape(1, 3, 128, 64),
                 inverse.reshape(1, 3, 128, 64))
        d_aqk, dv0, dq_raw = chunk_kda_cp.backward_prepare_a2_a3(
            saved[1], vn.reshape_as(x["v"]), x["do"], h, scale)
        kg = kc * torch.exp2(gc[..., -1:, :] - gc)
        du = dv0.reshape(shape) + kg @ dh.permute(0, 2, 1, 3, 4)

        def intra(q, k, g, b, daq, dak, dq, dk, db, dg, **kwargs):
            # 以完整成对门控表达式求导，独立检查收尾阶段传给块内核的梯度。
            q, k, g, b = [t.detach().clone().requires_grad_() for t in (q, k, g, b)]
            qr, kr, gr = [t.reshape(shape) for t in (q, k, g)]
            e = torch.exp2(gr.unsqueeze(-2) - gr.unsqueeze(-3))
            qk = (qr.unsqueeze(-2) * kr.unsqueeze(-3) * e).sum(-1)
            kk = (kr.unsqueeze(-2) * kr.unsqueeze(-3) * e).sum(-1)
            objective = (qk * daq.reshape_as(qk)).sum()
            objective += (b.reshape(1, 3, 2, 64, 1) * kk * dak.reshape_as(kk)).sum()
            delta = torch.autograd.grad(objective, (q, k, b, g))
            return (dq + delta[0], dk + delta[1], db + delta[2],
                    dg + delta[3] / torch.log(torch.tensor(2.0)))

        fake = types.ModuleType("fla_npu.ops.ascendc")
        fake.chunk_kda_bwd_intra = intra
        with patch.dict(sys.modules, {"fla_npu.ops.ascendc": fake}):
            actual = chunk_kda_cp.backward_finalize_a2_a3(
                x, saved, h, vn.reshape_as(x["v"]), dh, du.reshape_as(x["v"]),
                d_aqk, dq_raw, scale)
        for name, value in zip(("q", "k", "v", "beta", "g", "A_log", "dt_bias"), actual):
            with self.subTest(gradient=name):
                torch.testing.assert_close(value, leaves[name].grad, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
