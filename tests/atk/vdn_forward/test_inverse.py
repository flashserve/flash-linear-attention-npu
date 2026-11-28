# SPDX-License-Identifier: BSD-3-Clause
"""Exercise both FP32 inverse strategies against I and the spectral boundary."""
import os
import unittest

import torch


class TestInverse(unittest.TestCase):
    def test_normalized_residual(self):
        try:
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name != "torch_npu":
                raise
            self.skipTest("requires torch_npu")
        if not torch.npu.is_available():
            self.skipTest("requires an Ascend NPU")
        from fla_npu.ops.triton.triton_core.vdn_cholesky import solve
        from fla_npu.ops.vdn._newton import inverse_sub

        device = torch.device("npu", int(os.getenv("VDN_TEST_DEVICE", "0")))
        torch.npu.set_device(device)
        self.addCleanup(setattr, torch.npu.matmul, "allow_hf32", torch.npu.matmul.allow_hf32)
        torch.npu.matmul.allow_hf32 = False
        rng = torch.Generator().manual_seed(405)
        rotation, _ = torch.linalg.qr(torch.randn(128, 128, generator=rng, dtype=torch.float64))
        identity = torch.eye(128)
        rank_one = torch.zeros(128)
        rank_one[0] = 405
        spectra = [torch.zeros(128), rank_one, torch.linspace(0, 405, 128)]
        host = torch.stack([identity + (rotation * eig) @ rotation.mT for eig in spectra]).float()
        m = host.to(device)
        # The Cholesky panel update owns and overwrites its input workspace.
        factor = solve(m.clone())
        for rule, x in (("vdn_solve", factor.mT @ factor),
                        ("newton_schulz", inverse_sub(m, identity.to(device)))):
            with self.subTest(rule=rule):
                residual = (host.double() @ x.cpu().double() - identity).norm(dim=(-2, -1))
                self.assertLess((residual / identity.norm()).max().item(), 2e-5)
