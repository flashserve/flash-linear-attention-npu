"""Check the bounded-memory golden against the direct, token-by-token delta rule."""
import unittest

import torch

from reference import reference


class GoldenTest(unittest.TestCase):
    def test_reference(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)
        for batch, tokens, cu, bound in ((2, 17, None, -5.0), (1, 273, [0, 9, 9, 273], -0.01)):
            with self.subTest(batch=batch, cu=cu):
                heads, dim = 2, 4
                x = {name: torch.randn(batch, tokens, heads, dim) for name in ("q", "k", "v", "g")}
                x["beta"] = torch.rand(batch, tokens, heads)
                x["A_log"] = torch.rand(heads)
                x["dt_bias"] = torch.randn(heads * dim)
                dout = torch.randn_like(x["v"])
                actual = reference(x, dout, cu, bound)
                leaves = [x[name].requires_grad_() for name in ("q", "k", "v", "g", "beta")]
                q, k, v, raw, beta = [leaf.flatten(0, 1) for leaf in leaves]
                q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
                k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
                decay = (bound * ((raw + x["dt_bias"].view(heads, dim))
                                 * x["A_log"].exp()[:, None]).sigmoid()).exp()
                bounds = cu if cu else range(0, (batch + 1) * tokens, tokens)
                outputs, finals = [], []
                for start, end in zip(bounds, bounds[1:]):
                    state = torch.zeros(heads, dim, dim)
                    for t in range(start, end):
                        state = state * decay[t, :, :, None]
                        delta = (v[t] - (state * k[t, :, :, None]).sum(1)) * beta[t, :, None]
                        state = state + k[t, :, :, None] * delta[:, None]
                        outputs.append((state * q[t, :, :, None]).sum(1) * dim ** -0.5)
                    finals.append(state)
                out = torch.stack(outputs).reshape_as(dout)
                gradients = torch.autograd.grad(out, leaves, dout)
                for name, expected in zip(("o", "final_state", "dq", "dk", "dv", "dg", "dbeta"),
                                          (out, torch.stack(finals), *gradients)):
                    torch.testing.assert_close(actual[name], expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
