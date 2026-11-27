# SPDX-License-Identifier: BSD-3-Clause
"""Regression coverage for the 910B whole-forward dispatch and its fallback."""
import importlib
import os
import struct
import unittest
from unittest.mock import patch

import torch

from scripts.cases import apply_short_conv, check, expected, make_case, on_device, short_conv_weights


class TestVdnPipeline(unittest.TestCase):
    def test_core_count_tiling(self):
        from fla_npu.ops.vdn import _manual
        if not _manual._HAS_KERNEL:
            self.skipTest("requires the CANN 9.2.1 pipeline payload")
        for frames, heads in ((85, 7), (87, 7), (87, 14)):
            for cores in (20, 24):
                with self.subTest(frames=frames, heads=heads, cores=cores):
                    conv, *grams = [table.numpy().tobytes() for table in
                                    _manual._tiling("cpu", frames, heads, cores)]
                    self.assertEqual(struct.unpack_from("<4i", conv, 92), (cores // 4, 1, 1, 4))
                    batch = frames * heads
                    per_core = (batch + cores - 1) // cores
                    for table in grams:
                        for offset, expected in ((24, batch), (76, per_core), (88, cores), (156, per_core)):
                            self.assertEqual(struct.unpack_from("<i", table, offset)[0], expected)

    def test_dispatch_and_equivalence(self):
        try:
            import torch_npu
        except ModuleNotFoundError:
            self.skipTest("requires torch_npu")
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        from fla_npu.ops.vdn import _manual
        entry = importlib.import_module("fla_npu.ops.vdn.forward")
        device = torch.device("npu", int(os.getenv("VDN_TEST_DEVICE", "0")))
        if not _manual._HAS_KERNEL or "910B" not in torch.npu.get_device_name(device):
            self.skipTest("requires the CANN 9.2.1 pipeline payload and 910B")
        props = torch.npu.get_device_properties(device)
        if (props.cube_core_num, props.vector_core_num) not in ((20, 40), (24, 48)):
            self.skipTest("requires a supported AIC/AIV configuration")
        with torch.npu.device(device), torch.inference_mode():
            old_matmul = torch.npu.matmul.allow_hf32
            old_conv = torch.npu.conv.allow_hf32
            previous = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                torch.npu.matmul.allow_hf32 = False
                torch.npu.conv.allow_hf32 = False
                torch.use_deterministic_algorithms(True)
                for f, h, tokens, conv, active in [
                    (85, 7, 405, True, True), (87, 7, 405, False, True),
                    (87, 14, 405, True, True), (86, 7, 405, False, False),
                    (85, 7, 404, False, False), (3, 2, 17, False, False),
                ]:
                    with self.subTest(frames=f, heads=h, tokens=tokens, conv=conv):
                        case = make_case(f, h, tokens, torch.bfloat16, text=False, seed=1503)
                        weights = short_conv_weights(h, torch.bfloat16) if conv else {}
                        args = on_device(case, device)
                        options = dict(delta_rule="newton_schulz", fuse_silu=True, fuse_qk_l2norm=True,
                                       short_conv={k: tuple(t.to(device) for t in w) for k, w in weights.items()},
                                       frame_size=(15, 27) if conv else None)
                        with patch.object(entry, "_try_manual", return_value=None):
                            baseline = entry.vdn_forward(*args, **options)
                        with patch.object(_manual, "_launch", wraps=_manual._launch) as launch:
                            output = entry.vdn_forward(*args, **options)
                        self.assertEqual(launch.call_count, int(active))
                        self.assertTrue(torch.equal(output.view(torch.int16), baseline.view(torch.int16)))
                        oracle = expected(apply_short_conv(case, (15, 27), weights) if conv else case)
                        check(output, oracle)
            finally:
                torch.use_deterministic_algorithms(previous, warn_only=warn_only)
                torch.npu.matmul.allow_hf32 = old_matmul
                torch.npu.conv.allow_hf32 = old_conv
