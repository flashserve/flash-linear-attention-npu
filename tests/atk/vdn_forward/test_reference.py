# SPDX-License-Identifier: BSD-3-Clause
import unittest

import torch

from scripts.reference import reference


class TestWindowAndState(unittest.TestCase):
    def test_scalar_hand_calculation(self):
        # Exterior states at t=1 come from frames 0 and 2.
        q = torch.ones(3, 1, 1, 1)
        v = torch.tensor([2., 4., 6.]).view_as(q)
        w = torch.ones(3, 1, 1)
        a = torch.full_like(w, .5)
        bounds = torch.arange(3).expand(2, 3).T
        for seed in (0., 2.):
            out = reference(q, q, v, w, a, bounds, torch.tensor([[[seed]]]))
            expected = .5 * ((seed*.5+2.)/2. + (seed*.5+6.)/2.)
            self.assertEqual(out[1].item(), expected)

    def test_full_window(self):
        q = torch.tensor([[[[1., 2.]]], [[[0., -2.]]]])
        beta = torch.ones(2, 1, 1)
        alpha = torch.tensor([[[.5, .5]], [[.25, 1.]]])
        bounds = torch.tensor([[0, 1], [0, 1]])
        self.assertEqual(reference(q,q,q,beta,alpha,bounds).count_nonzero().item(), 0)
        seed = torch.eye(2).unsqueeze(0)
        result = reference(q,q,q,beta,alpha,bounds,seed)
        torch.testing.assert_close(result[0,0,0], torch.tensor([.625, 2.]))
        torch.testing.assert_close(result[1,0,0], torch.tensor([0., -3.]))

    def test_heads_are_independent_and_layouts_agree(self):
        rng = torch.Generator().manual_seed(43)
        for dtype in (torch.float32, torch.bfloat16):
            args = [torch.randn(5, 3, 2, 4, generator=rng).transpose(1,2).to(dtype) for _ in range(3)]
            beta = torch.rand(5, 2, 3, generator=rng).to(dtype)
            alpha = torch.rand(5, 2, 4, generator=rng)
            bounds = torch.tensor([[0,0],[0,2],[1,3],[3,4],[1,4]])
            for silu in (True, False):
                for norm in (True, False):
                    flags = dict(silu=silu, qk_l2norm=norm)
                    together = reference(*args,beta,alpha,bounds,**flags)
                    slices = [reference(*(a[:,h:h+1].contiguous() for a in args),beta[:,h:h+1],
                                        alpha[:,h:h+1],bounds,**flags) for h in range(2)]
                    torch.testing.assert_close(together,torch.cat(slices,1))
                    self.assertEqual(together.dtype,dtype)

