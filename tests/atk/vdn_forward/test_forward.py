# SPDX-License-Identifier: BSD-3-Clause
"""Run with unittest discovery. VDN_TEST_DEVICE selects the logical device."""
import os
import unittest

import torch
import torch.nn.functional as F

from scripts.cases import apply_short_conv, check, expected, make_case, on_device, short_conv_weights


class TestForward(unittest.TestCase):
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
        from fla_npu.ops.vdn import vdn_forward

        cls.forward = staticmethod(vdn_forward)
        cls.device = torch.device("npu", int(os.getenv("VDN_TEST_DEVICE", "0")))
        torch.npu.set_device(cls.device)
        matmul, conv = torch.npu.matmul, torch.npu.conv
        cls.addClassCleanup(setattr, matmul, "allow_hf32", matmul.allow_hf32)
        cls.addClassCleanup(setattr, conv, "allow_hf32", conv.allow_hf32)
        matmul.allow_hf32 = conv.allow_hf32 = False

    def test_solvers_and_features(self):
        for frames, heads, tokens, dtype in [(1, 1, 1, torch.float32),
                                            (5, 2, 13, torch.float32),
                                            (9, 3, 33, torch.bfloat16)]:
            case = make_case(frames, heads, tokens, dtype, text=frames != 5, strided=frames != 9)
            args = on_device(case, self.device)
            for silu, norm in [(False, False), (True, False), (False, True), (True, True)]:
                gold = expected(case, silu, norm)
                for rule in ("vdn_solve", "newton_schulz"):
                    with self.subTest(shape=(frames, heads, tokens), rule=rule, silu=silu, norm=norm):
                        out = self.forward(*args, delta_rule=rule, frame_tile=4,
                                           value_first=True, fuse_silu=silu, fuse_qk_l2norm=norm)
                        check(out, gold)

    def test_short_convolution_across_tile(self):
        case = make_case(7, 2, 15, torch.float32)
        channels = 2 * 128
        rng = torch.Generator().manual_seed(91)
        weights = {}
        prepared = list(case)
        for i, name in enumerate(("k", "v"), 1):
            spatial = torch.randn(channels, 1, 5, 5, generator=rng) * 0.2
            temporal = torch.randn(channels, 1, 5, generator=rng) * 0.4
            weights[name] = (spatial.to(self.device), temporal.to(self.device))
            volume = case[i].transpose(2, 3).reshape(7, channels, 3, 5)
            volume = F.conv2d(volume, spatial, padding=2, groups=channels)
            lines = volume.flatten(2).permute(2, 1, 0)
            lines = F.conv1d(lines, temporal, padding=2, groups=channels)
            prepared[i] = lines.permute(2, 1, 0).reshape(7, 2, 128, 15).transpose(2, 3)
        args = on_device(case, self.device)
        for rule in ("newton_schulz", "vdn_solve"):
            with self.subTest(rule=rule):
                out = self.forward(*args, short_conv=weights, frame_size=(3, 5),
                                   frame_tile=4, delta_rule=rule, fuse_silu=True, fuse_qk_l2norm=True)
                check(out, expected(prepared))

    def test_bf16_short_conv_weighted_value(self):
        for frames, grid, strided_beta in [(1, (1, 1), False), (7, (3, 5), True)]:
            heads = 2
            case = list(make_case(frames, heads, grid[0] * grid[1], torch.bfloat16))
            if strided_beta:
                rng = torch.Generator().manual_seed(77)
                case[3] = torch.rand(case[3].shape, generator=rng)
            weights = short_conv_weights(heads, torch.bfloat16)
            on_npu = {key: tuple(w.to(self.device) for w in pair) for key, pair in weights.items()}
            args = list(on_device(case, self.device))
            if strided_beta:
                padded = torch.empty((*args[3].shape, 2), device=self.device)
                padded[..., 0].copy_(args[3])
                args[3] = padded[..., 0]
            gold = expected(apply_short_conv(case, grid, weights))
            for rule in ("newton_schulz", "vdn_solve"):
                with self.subTest(frames=frames, rule=rule):
                    out = self.forward(*args, delta_rule=rule, frame_tile=4, short_conv=on_npu,
                                       frame_size=grid, fuse_silu=True, fuse_qk_l2norm=True)
                    check(out, gold)

    def test_temporal_bf16_rounding(self):
        from fla_npu.ops.triton.triton_core.vdn_short_conv import feature_preprocessor

        case = make_case(7, 1, 3, torch.bfloat16)
        spatial, temporal = short_conv_weights(1, torch.bfloat16)["v"]
        spatial.zero_()
        spatial[:, 0, 2, 2] = 1
        weights = {"v": (spatial, temporal)}
        expected_v = apply_short_conv(case, (1, 3), weights)[2]
        q, _, v = on_device(case, self.device)[:3]
        npu_weights = {"v": (spatial.to(self.device), temporal.to(self.device))}
        pre = feature_preprocessor(q, silu=False, qk_l2norm=False, short_conv=npu_weights,
                                   frame_size=(1, 3))
        actual = torch.cat([pre(v, rows, "v").cpu() for rows in (slice(0, 4), slice(4, 7))])
        torch.testing.assert_close(actual, expected_v, rtol=0, atol=0)

    def test_invalid_inputs(self):
        args = on_device(make_case(3, 1, 3, torch.float32), self.device)
        for kwargs in ({"frame_tile": 3}, {"frame_tile": 0}, {"delta_rule": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.forward(*args, **kwargs)
        invalid = list(args)
        invalid[5] = torch.tensor([[0, 0], [2, 2], [2, 2]])
        with self.assertRaises(ValueError):
            self.forward(*invalid)
