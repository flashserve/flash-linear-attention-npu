"""Small independent closed-form checks of the preserved CPU golden."""

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "_tests"))
from rwkv8_reference import fused_recurrent_rwkv8_golden


class CpuGoldenTest(unittest.TestCase):
    def check_case(self, *, reverse=False, initial=0.0):
        # z=b=0 removes the rank-one correction. With all-one k/v, each
        # state element follows s' = exp(-1)*s + 1, and o = 8*s'.
        q = torch.ones(1, 1, 2, 8)
        v = torch.ones(1, 1, 2, 16)
        zero = torch.zeros_like(q)
        init = torch.full((1, 1, 8, 16), initial) if initial else None
        result = fused_recurrent_rwkv8_golden(
            q, zero, q, v, zero, zero,
            initial_state=init, reverse=reverse,
            output_chunk_state=True, output_sa=True, chunk_len=1,
        )
        decay = math.exp(-1.0)
        first = decay * initial + 1.0
        second = decay * first + 1.0
        expected_states = (second, first) if reverse else (first, second)
        for token, state in enumerate(expected_states):
            torch.testing.assert_close(result.o[:, :, token], torch.full((1, 1, 16), 8.0 * state))
            torch.testing.assert_close(result.s[:, :, token], torch.full((1, 1, 8, 16), state))
        torch.testing.assert_close(result.sa, torch.zeros_like(result.sa))

    def test_zero_initial_state(self):
        self.check_case()

    def test_nonzero_initial_state(self):
        self.check_case(initial=2.0)

    def test_reverse(self):
        self.check_case(reverse=True)


if __name__ == "__main__":
    unittest.main()
