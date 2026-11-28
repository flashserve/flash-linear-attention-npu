# SPDX-License-Identifier: BSD-3-Clause
"""The full VDN frontend consumes frame means and projected text features."""
import importlib
import math
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scripts.cases import apply_short_conv, check, expected, make_case, on_device, short_conv_weights


def text_initial_state(key, value, beta):
    length, heads, dim = key.shape
    beta = beta.reshape(length, heads).T.unsqueeze(-1)
    key = F.normalize(F.silu(key), dim=-1, eps=1e-6).transpose(0, 1).contiguous()
    value = (F.silu(value).transpose(0, 1) * beta).contiguous()
    key32 = key.float()
    scaled = (key32 * beta.float()).unsqueeze(0)
    gram = scaled.mT @ key32.unsqueeze(0)
    b = (value.unsqueeze(0).mT @ key.unsqueeze(0)).float()
    identity = torch.eye(dim, device=key.device)
    matrix = (gram + gram.mT) * 0.5 + identity
    scale = 2.0 / (length + 2)
    steps = math.ceil(math.log2(math.log(1e-7) / math.log1p(-scale)))
    x = (scale * identity).expand_as(matrix)
    for _ in range(steps):
        x = x @ (2 * identity - matrix @ x)
    return 0.5 * (b @ x)[0]


class TestFrontend(unittest.TestCase):
    def test_projected_inputs(self):
        try:
            import torch_npu
        except ModuleNotFoundError:
            self.skipTest("requires torch_npu")
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        mod = importlib.import_module("fla_npu.ops.vdn")
        stable = importlib.import_module("fla_npu.ops.ascendc._stable")
        device = torch.device("npu", int(os.getenv("VDN_TEST_DEVICE", "0")))
        old_flags = (torch.npu.matmul.allow_hf32, torch.npu.conv.allow_hf32,
                     torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
        with torch.inference_mode(), torch.npu.device(device):
            try:
                torch.npu.matmul.allow_hf32 = False
                torch.npu.conv.allow_hf32 = False
                torch.use_deterministic_algorithms(True)
                for f,h,length in ((85,7,3797),(87,14,257),(85,7,405),(85,7,0)):
                    with self.subTest(frames=f,heads=h,length=length):
                        self.compare_case(mod,stable,device,f,h,length)
            finally:
                torch.npu.matmul.allow_hf32, torch.npu.conv.allow_hf32 = old_flags[:2]
                torch.use_deterministic_algorithms(old_flags[2],warn_only=old_flags[3])

    def compare_case(self, mod, stable, device, f, h, length):
        rng = torch.Generator().manual_seed(714)
        def sample(*shape):
            return torch.randn(shape,generator=rng).to(device)
        case = list(make_case(f,h,405,torch.bfloat16,seed=2203))
        mean = sample(f,5376)
        down = (sample(128,5376) * 0.015).bfloat16()
        up = (sample((h+2)*128,128) * 0.08).bfloat16()[128:(h+1)*128]
        dt_bias = sample(h*128) - 4
        a_log = sample(h+2)[1:1+h]
        delta = F.linear(F.linear(mean,down.float()),up.float()) + dt_bias
        case[4] = torch.exp(-a_log.exp()[None,:,None] * F.softplus(delta.reshape(f,h,128))).cpu()
        tk = tv = beta = None
        if length:
            packed = sample(length,h,3,128).bfloat16()
            tk, tv = packed[:,:,0],packed[:,:,2]
            beta = sample(length,h+2).sigmoid().bfloat16()[:,1:h+1]
            if length == 405:
                tk[:,0].zero_()
                beta[:,0].zero_()
        case[6] = text_initial_state(tk,tv,beta).cpu() if length else None
        args = on_device(case,device)
        weights = short_conv_weights(h,torch.bfloat16)
        conv = {k:tuple(x.to(device) for x in pair) for k,pair in weights.items()}
        options = dict(short_conv=conv,frame_size=(27,15))
        old = mod.vdn_forward(*args,delta_rule="newton_schulz",fuse_silu=True,fuse_qk_l2norm=True,**options)
        observed = []
        original = stable._op
        def lookup(name):
            fn = original(name)
            def invoke(*values):
                fn(*values)
                if name=="npu_vdn_pipeline_from_features":
                    observed.extend((values[4],values[6]))
            return invoke
        with patch.object(stable,"_op",lookup):
            out = mod.vdn_forward_from_features(*args[:4],args[5],frame_mean=mean,down_weight=down,up_weight=up,
                    dt_bias=dt_bias,a_log=a_log,text_k=tk,text_v=tv,text_beta=beta,**options)
        frontend = importlib.import_module("fla_npu.ops.vdn.frontend")
        soc = torch.npu.get_device_name(device)
        if (frontend._manual._HAS_KERNEL and (frontend._manual._PACKAGE/"manual_tiling"/(soc+"-frontend.json.z")).is_file()):
            self.assertTrue(observed)
        if observed:
            self.assertTrue(torch.equal(observed[0].view(torch.int32),args[4].view(torch.int32)))
            if length:
                self.assertTrue(torch.equal(observed[1].view(torch.int32),args[6].view(torch.int32)))
        self.assertTrue(torch.equal(out.view(torch.int16),old.view(torch.int16)))
        check(out,expected(apply_short_conv(case,(27,15),weights)))
