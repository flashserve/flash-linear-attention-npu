# SPDX-License-Identifier: BSD-3-Clause
"""Validate the three A2 operator entry points, including storage mutation."""

import os
import unittest

import torch


class TestNativeStages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name != "torch_npu":
                raise
            raise unittest.SkipTest("requires torch_npu") from exc
        if not torch.npu.is_available():
            raise unittest.SkipTest("requires an Ascend NPU")
        from fla_npu.ops import ascendc

        cls.ops = ascendc
        cls.device = torch.device("npu", int(os.getenv("VDN_TEST_DEVICE", "0")))
        if "910B" not in torch.npu.get_device_name(cls.device):
            raise unittest.SkipTest("requires an A2 VDN operator package")
        torch.npu.set_device(cls.device)

    def test_add_diagonal(self):
        base = torch.arange(128 * 128, dtype=torch.float32).view(128, 128) / 1024
        for batch in (1, 17):
            with self.subTest(batch=batch):
                source = base.repeat(batch, 1, 1)
                correction = source.to(self.device)
                two_eye = 2 * torch.eye(128)
                self.ops.npu_vdn_add_diagonal(correction, two_eye.to(self.device))
                torch.testing.assert_close(correction.cpu(), source + two_eye, rtol=0, atol=0)

    def test_seeded_scan(self):
        frames, heads = 5, 2
        eye = torch.eye(128)
        transition = eye.repeat(frames, heads, 1, 1) * 0.5
        injection = eye.repeat(frames, heads, 1, 1)
        injection *= torch.arange(1, frames * heads + 1).view(frames, heads, 1, 1)
        start = eye.repeat(heads, 1, 1)
        prefix = torch.cat((start[None], injection)).to(self.device)
        suffix = torch.cat((injection, -start[None])).to(self.device)
        self.ops.npu_vdn_scan(transition.to(self.device), prefix, suffix)
        left, right = [start], [-start]
        for t in range(frames):
            left.append(left[-1] * 0.5 + injection[t])
            right.append(right[-1] * 0.5 + injection[-1 - t])
        torch.testing.assert_close(prefix.cpu(), torch.stack(left), rtol=0, atol=0)
        torch.testing.assert_close(suffix.cpu(), torch.stack(list(reversed(right))), rtol=0, atol=0)

    def test_bridge_tail_with_nonzero_start(self):
        frames, heads = 5, 2
        rng = torch.Generator().manual_seed(439)
        prefix = torch.randn(frames + 1, heads, 128, 128, generator=rng)
        suffix = torch.randn(prefix.shape, generator=rng)
        bounds = torch.tensor([[0, 2], [0, 2], [2, 4], [2, 3], [3, 4]])
        alpha = torch.rand(frames, heads, 128, generator=rng) * 0.2 + 0.79
        logs = torch.cat((torch.zeros_like(alpha[:1]), alpha.log().cumsum(0)))
        start = 2
        expected = []
        for t in range(start, frames):
            lo, hi = bounds[t].tolist()
            left = alpha[lo:t + 1].prod(0).unsqueeze(1) * prefix[lo]
            right = alpha[t:hi + 1].prod(0).unsqueeze(1) * suffix[hi + 1]
            expected.append(left + right)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                out = torch.empty(frames - start, heads, 128, 128, dtype=dtype, device=self.device)
                self.ops.npu_vdn_bridge(
                    prefix.to(self.device), suffix.to(self.device), bounds.to(self.device),
                    logs.to(self.device), start, out,
                )
                tolerance = 2e-5 if dtype == torch.float32 else 0.008
                torch.testing.assert_close(out.cpu().float(), torch.stack(expected), atol=tolerance, rtol=tolerance)

    def test_rejects_overlapping_storage(self):
        frames = 2
        eye = torch.eye(128, device=self.device)
        transition = eye.repeat(frames, 1, 1, 1)
        bank = eye.repeat(frames + 1, 1, 1, 1)
        suffix = bank.clone()
        bounds = torch.tensor([[0, 1], [1, 1]], device=self.device)
        logs = torch.zeros(frames + 1, 1, 128, device=self.device)
        correction = eye.repeat(2, 1, 1)
        with self.assertRaisesRegex(RuntimeError, r"aclnnVdnScanGetWorkspaceSize failed: 161002\b"):
            self.ops.npu_vdn_scan(transition, bank, bank)
        with self.assertRaisesRegex(RuntimeError, r"aclnnVdnBridgeGetWorkspaceSize failed: 161002\b"):
            self.ops.npu_vdn_bridge(bank, suffix, bounds, logs, 0, bank[:frames])
        with self.assertRaisesRegex(RuntimeError, r"aclnnVdnAddDiagonalGetWorkspaceSize failed: 161002\b"):
            self.ops.npu_vdn_add_diagonal(correction, correction[0])

    def test_accepts_disjoint_views_in_one_storage(self):
        frames = 2
        eye = torch.eye(128, device=self.device)
        storage = torch.zeros(2 * (frames + 1), 1, 128, 128, device=self.device)
        prefix, suffix = storage.split(frames + 1)
        prefix[0].copy_(eye)
        suffix[-1].copy_(-eye)
        self.ops.npu_vdn_scan(eye.repeat(frames, 1, 1, 1), prefix, suffix)
        actual = storage.cpu()
        expected = torch.eye(128).expand(frames + 1, 1, 128, 128)
        torch.testing.assert_close(actual[:frames + 1], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[frames + 1:], -expected, rtol=0, atol=0)

    def test_bridge_accepts_disjoint_source_and_output_views(self):
        frames, heads, dim = 3, 1, 128
        rng = torch.Generator().manual_seed(431)
        prefix_host = torch.randn(frames + 1, heads, dim, dim, generator=rng)
        suffix_host = torch.randn(prefix_host.shape, generator=rng)
        suffix = suffix_host.to(self.device)
        bounds = torch.tensor([[0, 1], [1, 1], [1, 2]], device=self.device)
        logs = torch.zeros(frames + 1, heads, dim, device=self.device)
        expected = torch.stack((prefix_host[1] + suffix_host[2], prefix_host[1] + suffix_host[3]))
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                storage = torch.zeros(frames + 3, heads, dim, dim, device=self.device)
                prefix = storage[:frames + 1].copy_(prefix_host)
                out = storage[frames + 1:].view(dtype).flatten()[:expected.numel()].view_as(expected)
                self.ops.npu_vdn_bridge(prefix, suffix, bounds, logs, 1, out)
                torch.testing.assert_close(prefix.cpu(), prefix_host, rtol=0, atol=0)
                torch.testing.assert_close(out.cpu(), expected.to(dtype), rtol=0, atol=0)
