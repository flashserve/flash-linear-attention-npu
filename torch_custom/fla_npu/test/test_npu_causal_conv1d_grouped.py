"""Exercise grouped causal convolution training with an independent CPU reference."""
import os
import unittest
from itertools import accumulate

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from fla_npu.ops.ascendc import causal_conv1d_grouped


def reference(inputs, weights, offsets, seeds):
    outputs = []
    input_grads = []
    weight_grads = []
    for features, weight, seed in zip(inputs, weights, seeds):
        x = features.detach().float()
        w = weight.detach().float()
        y = torch.empty_like(features)
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        for begin, end in zip(offsets, offsets[1:]):
            if begin == end:
                continue
            doc = x[begin:end]
            conv = torch.zeros_like(doc)
            for tap in range(len(w)):
                delay = len(w) - 1 - tap
                rows = len(doc) - delay
                if rows > 0:
                    conv[delay:] += doc[:rows] * w[tap]
            preactivation = conv.to(features.dtype).float()
            y[begin:end] = F.silu(preactivation).to(features.dtype)
            if seed is None:
                continue
            sigmoid = torch.sigmoid(preactivation)
            dy = (
                seed[begin:end].float()
                * sigmoid
                * (1.0 + preactivation * (1.0 - sigmoid))
            )
            for tap in range(len(w)):
                delay = len(w) - 1 - tap
                rows = len(doc) - delay
                if rows > 0:
                    dx[begin : begin + rows] += dy[delay:] * w[tap]
                    dw[tap] += (doc[:rows] * dy[delay:]).sum(0)
        outputs.append(y)
        input_grads.append(None if seed is None else dx.to(features.dtype))
        weight_grads.append(None if seed is None else dw.to(weight.dtype))
    return (*outputs, *input_grads, *weight_grads)


class TestGroupedCausalConv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("requires Ascend 950")
        torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
        if "Ascend950" not in torch.npu.get_device_name():
            raise unittest.SkipTest("requires Ascend 950")

    def run_case(
        self,
        lengths,
        *,
        single_output=False,
        activation_checkpoint=False,
        shapes=((80, 4),) * 3,
        dtype=torch.bfloat16
    ):
        generator = torch.Generator().manual_seed(193)
        offsets = [0, *accumulate(lengths)]
        tokens = sum(lengths)
        cpu_inputs = tuple(
            torch.randn(tokens, channels * 2, dtype=dtype, generator=generator)[
                :, ::2
            ].requires_grad_()
            for channels, width in shapes
        )
        cpu_weights = tuple(
            (
                torch.randn(channels, width, dtype=dtype, generator=generator).T / 16
            ).requires_grad_()
            for channels, width in shapes
        )
        # The same noncontiguous logical tensors reach the device operator.
        inputs = tuple(
            torch.stack((t.detach(), t.detach()), dim=-1).npu()[..., 0].requires_grad_()
            for t in cpu_inputs
        )
        weights = tuple(
            t.detach().T.contiguous().npu().T.requires_grad_() for t in cpu_weights
        )
        boundaries = torch.tensor(offsets, device="npu", dtype=torch.int32)

        def f(*args):
            return causal_conv1d_grouped(args[:3], args[3:], boundaries)

        out = (
            checkpoint(f, *inputs, *weights, use_reentrant=False)
            if activation_checkpoint
            else f(*inputs, *weights)
        )

        selected = (2,) if single_output else (0, 1, 2)
        dy = tuple(
            torch.randn(tokens, shapes[i][0], dtype=dtype, generator=generator)
            for i in selected
        )
        seeds = [None, None, None]
        for i, grad_output in zip(selected, dy):
            seeds[i] = grad_output
        expected_values = reference(cpu_inputs, cpu_weights, offsets, seeds)
        grad = torch.autograd.grad(
            tuple(out[i] for i in selected),
            (*inputs, *weights),
            tuple(d.npu() for d in dy),
            allow_unused=True,
        )
        for expected, actual in zip(expected_values, (*out, *grad)):
            if expected is None:
                self.assertIsNone(actual)
            else:
                torch.testing.assert_close(actual.cpu(), expected, atol=0.01, rtol=0.01)

    def test_differing_lengths_empty_documents_and_channel_tail(self):
        self.run_case([0, 1, 2, 3, 4, 0, 7, 13, 65])

    def test_many_documents(self):
        self.run_case([i % 5 for i in range(267)])

    def test_independent_widths_and_channel_counts(self):
        self.run_case(
            [1, 2, 4, 35], shapes=((32, 2), (48, 3), (64, 4)), dtype=torch.float16
        )

    def test_unused_branches_and_activation_checkpoint(self):
        self.run_case([2, 11, 23], single_output=True, activation_checkpoint=True)

    def test_invalid_offsets(self):
        features = tuple(
            torch.zeros(8, 16, dtype=torch.bfloat16, device="npu") for _ in range(3)
        )
        weights = tuple(
            torch.ones(4, 16, dtype=torch.bfloat16, device="npu") for _ in range(3)
        )
        for values in ([1, 8], [0, 4, 3, 8], [0, 7]):
            with self.subTest(offsets=values):
                with self.assertRaisesRegex(ValueError, "Offsets"):
                    causal_conv1d_grouped(
                        features, weights, torch.tensor(values, device="npu")
                    )
