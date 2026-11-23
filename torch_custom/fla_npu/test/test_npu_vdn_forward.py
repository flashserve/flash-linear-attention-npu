"""Public wheel smoke test for the video delta-network forward."""
import os
import unittest

import torch


class TestVDNForward(unittest.TestCase):
    def test_full_window_text_state(self):
        import torch_npu
        from fla_npu.ops.vdn import vdn_forward

        device = torch.device("npu", int(os.environ.get("TEST_DEVICE_ID", 0)))
        if "910B" not in torch.npu.get_device_name(device):
            self.skipTest("validated on Ascend 910B")
        with torch.npu.device(device):
            q = torch.ones(1, 1, 1, 128, device=device)
            seed = torch.eye(128, device=device).unsqueeze(0)
            beta = torch.zeros(1, 1, 1, device=device)
            alpha = torch.full((1, 1, 128), 0.25, device=device)
            bounds = torch.tensor([[0, 0]])
            for rule in ("vdn_solve", "newton_schulz"):
                with self.subTest(rule=rule):
                    y = vdn_forward(q, torch.zeros_like(q), q, beta, alpha, bounds,
                                    seed, delta_rule=rule)
                    torch.testing.assert_close(y.cpu(), torch.full(q.shape, 0.5))


if __name__ == "__main__":
    unittest.main()

