# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

ascendc_ops = None
if torch.npu.is_available():
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


# CPU golden reference adapted from:
# vllm-project/vllm/tests/kernels/mamba/test_causal_conv1d.py
def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_update_spec_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    activation: str | None = None,
):
    """CPU golden for vLLM Triton _causal_conv1d_update_kernel IS_SPEC_DECODING path.

    x: (batch, dim, seqlen)
    conv_state: (batch, dim, state_len)
    weight: (dim, width)
    num_accepted_tokens: (batch,)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if num_accepted_tokens is None:
        raise ValueError("num_accepted_tokens must be provided for spec decode golden")

    dtype_in = x.dtype
    x = x.to(weight.dtype)
    conv_state = conv_state.to(weight.dtype)
    if not isinstance(num_accepted_tokens, torch.Tensor):
        num_accepted_tokens = torch.tensor(
            num_accepted_tokens, dtype=torch.long, device=x.device
        )
    else:
        num_accepted_tokens = num_accepted_tokens.to(device=x.device, dtype=torch.long)

    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    keep = width - 2
    required_state_len = (width - 1) + (seqlen - 1)
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    assert state_len >= required_state_len

    out = torch.empty_like(x)
    for seq_idx in range(batch):
        offset = int(num_accepted_tokens[seq_idx].item()) - 1
        assert 0 <= offset <= seqlen - 1
        hist = conv_state[seq_idx : seq_idx + 1, :, offset : offset + width - 1]
        x_cat = torch.cat([hist, x[seq_idx : seq_idx + 1]], dim=-1)
        out_seq = F.conv1d(
            x_cat, weight.unsqueeze(1), bias, padding=0, groups=dim
        )[..., :seqlen]
        if activation is not None:
            out_seq = F.silu(out_seq)
        out[seq_idx : seq_idx + 1] = out_seq

        if keep > 0:
            conv_state[seq_idx, :, :keep] = conv_state[
                seq_idx, :, offset + 1 : offset + 1 + keep
            ]
        conv_state[seq_idx, :, keep : keep + seqlen] = x[seq_idx]
    return out.to(dtype=dtype_in)


def make_tensor(shape, *, dtype=torch.float16, start=1.0, device="npu"):
    numel = 1
    for dim in shape:
        numel *= dim
    data = torch.arange(start, start + numel, dtype=torch.float32).reshape(shape)
    data = data / 128.0
    return data.to(dtype=dtype, device=device)


def activation_from_mode(mode: int):
    return None if mode == 0 else "silu"


def op_weight_to_ref(weight_op: torch.Tensor) -> torch.Tensor:
    return weight_op.detach().cpu().float().transpose(0, 1).contiguous()


def op_conv_states_to_ref(conv_states_op: torch.Tensor) -> torch.Tensor:
    return conv_states_op.detach().cpu().float().permute(0, 2, 1).contiguous()


def ref_conv_states_to_op(conv_states_ref: torch.Tensor) -> torch.Tensor:
    return conv_states_ref.permute(0, 2, 1).contiguous()


def op_batch_x_to_ref(x_op: torch.Tensor) -> torch.Tensor:
    return x_op.detach().cpu().float().permute(0, 2, 1).contiguous()


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestCausalConv1d(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    @classmethod
    def setUpClass(cls):
        global ascendc_ops
        if ascendc_ops is None:
            from fla_npu.ops import ascendc as loaded_ascendc_ops

            ascendc_ops = loaded_ascendc_ops

    def call_op(self, **kwargs):
        return ascendc_ops.npu_causal_conv1d(**kwargs)

    def test_legacy_python_apis_match_on_npu_and_warn(self):
        x = make_tensor((1, 4, 16), start=1.0)
        weight = make_tensor((4, 16), start=101.0)
        bias = make_tensor((16,), start=201.0)
        npu_state = make_tensor((1, 3, 16), start=301.0)
        causal_state = npu_state.clone()

        with self.assertWarnsRegex(FutureWarning, "2027/02"):
            npu_y = ascendc_ops.npu_causal_conv1d(
                x,
                weight,
                bias,
                npu_state,
                activation_mode=1,
            )
        with self.assertWarnsRegex(FutureWarning, "2027/02"):
            causal_y = ascendc_ops.causal_conv1d(
                x,
                weight,
                bias,
                causal_state,
                activation_mode=1,
            )

        self.assertTensorClose(causal_y, npu_y)
        self.assertTensorClose(causal_state, npu_state)

    def assertTensorClose(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertTrue(
            torch.allclose(
                actual.detach().cpu().float(),
                expected.detach().cpu().float(),
                rtol=rtol,
                atol=atol,
            ),
            msg=(
                f"max_abs_diff="
                f"{(actual.detach().cpu().float() - expected.detach().cpu().float()).abs().max().item():.6f}"
            ),
        )

    def test_npu_causal_conv1d_prefill_batch_matches_cpu_golden(self):
        x = make_tensor((2, 4, 16), start=1.0)
        weight_op = make_tensor((4, 16), start=101.0)
        bias = make_tensor((16,), start=201.0)
        conv_states = make_tensor((2, 3, 16), start=301.0)
        conv_states_ref = op_conv_states_to_ref(conv_states)

        y = self.call_op(
            x=x,
            weight=weight_op,
            bias=bias,
            conv_states=conv_states,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )

        x_ref = op_batch_x_to_ref(x)
        weight_ref = op_weight_to_ref(weight_op)
        bias_ref = bias.detach().cpu().float()
        y_ref, final_states_ref = causal_conv1d_ref(
            x_ref,
            weight_ref,
            bias=bias_ref,
            initial_states=None,
            return_final_states=True,
            final_states_out=conv_states_ref,
            activation=activation_from_mode(1),
        )

        self.assertTensorClose(y, y_ref.permute(0, 2, 1).contiguous(), rtol=1e-1, atol=2e-1)
        self.assertTensorClose(
            conv_states,
            final_states_ref.permute(0, 2, 1).contiguous(),
            rtol=1e-1,
            atol=2e-1,
        )

    def test_npu_causal_conv1d_prefill_batch_output_reshape_matches_cpu_golden(self):
        x = make_tensor((2, 4, 32), start=1.0)
        weight_op = make_tensor((4, 32), start=101.0)
        bias = make_tensor((32,), start=201.0)
        conv_states = make_tensor((2, 3, 32), start=301.0)
        conv_states_ref = op_conv_states_to_ref(conv_states)
        head_num = 2

        y = self.call_op(
            x=x,
            weight=weight_op,
            bias=bias,
            conv_states=conv_states,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
            head_num=head_num,
        )

        x_ref = op_batch_x_to_ref(x)
        weight_ref = op_weight_to_ref(weight_op)
        bias_ref = bias.detach().cpu().float()
        y_ref, final_states_ref = causal_conv1d_ref(
            x_ref,
            weight_ref,
            bias=bias_ref,
            initial_states=None,
            return_final_states=True,
            final_states_out=conv_states_ref,
            activation=activation_from_mode(1),
        )
        batch, seqlen, dim = x.shape
        expected = (
            y_ref.permute(0, 2, 1)
            .reshape(batch, seqlen, head_num, dim // head_num)
            .transpose(1, 2)
            .contiguous()
        )
        self.assertTensorClose(y, expected, rtol=1e-1, atol=2e-1)
        self.assertTensorClose(
            conv_states,
            final_states_ref.permute(0, 2, 1).contiguous(),
            rtol=1e-1,
            atol=2e-1,
        )

    def test_npu_causal_conv1d_varlen_initial_state_matches_cpu_golden(self):
        x = make_tensor((5, 16), start=1.0)
        weight_op = make_tensor((4, 16), start=101.0)
        conv_states = make_tensor((2, 3, 16), start=301.0)

        query_start_loc = [0, 2, 5]
        cache_indices = [0, 1]
        initial_state_mode = [1, 0]
        conv_states_ref = op_conv_states_to_ref(conv_states)

        y = self.call_op(
            x=x,
            weight=weight_op,
            bias=None,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=initial_state_mode,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
        )

        weight_ref = op_weight_to_ref(weight_op)
        x_ref = x.detach().cpu().float()
        outputs = []
        for seq_idx in range(len(query_start_loc) - 1):
            start = query_start_loc[seq_idx]
            end = query_start_loc[seq_idx + 1]
            x_seq = x_ref[start:end].transpose(0, 1).unsqueeze(0).contiguous()
            initial_state = None
            if initial_state_mode[seq_idx]:
                initial_state = conv_states_ref[cache_indices[seq_idx]].unsqueeze(0)
            y_seq, _ = causal_conv1d_ref(
                x_seq,
                weight_ref,
                bias=None,
                initial_states=initial_state,
                return_final_states=True,
                final_states_out=conv_states_ref[cache_indices[seq_idx]].unsqueeze(0),
                activation=activation_from_mode(0),
            )
            outputs.append(y_seq.squeeze(0).transpose(0, 1).contiguous())

        y_ref = torch.cat(outputs, dim=0)
        conv_states_expected = ref_conv_states_to_op(conv_states_ref)

        self.assertTensorClose(y, y_ref, rtol=1e-1, atol=2e-1)
        self.assertTensorClose(conv_states, conv_states_expected, rtol=1e-1, atol=2e-1)

    def test_causal_conv1d_fn_varlen_device_head_num_output_reshape_matches_cpu_golden(self):
        x = make_tensor((5, 32), start=1.0)
        weight_op = make_tensor((4, 32), start=101.0)
        conv_states = make_tensor((2, 3, 32), start=301.0)

        query_start_loc = [0, 2, 5]
        cache_indices = [0, 1]
        initial_state_mode = [1, 0]
        conv_states_ref = op_conv_states_to_ref(conv_states)
        head_num = 2

        y = ascendc_ops.causal_conv1d_fn(
            x,
            weight_op,
            None,
            conv_states,
            activation=None,
            query_start_loc=torch.tensor(
                query_start_loc, dtype=torch.int32, device=x.device
            ),
            cache_indices=torch.tensor(
                cache_indices, dtype=torch.int32, device=x.device
            ),
            has_initial_state=torch.tensor(
                initial_state_mode, dtype=torch.bool, device=x.device
            ),
            null_block_id=None,
            head_num=head_num,
        )

        weight_ref = op_weight_to_ref(weight_op)
        x_ref = x.detach().cpu().float()
        outputs = []
        for seq_idx in range(len(query_start_loc) - 1):
            start = query_start_loc[seq_idx]
            end = query_start_loc[seq_idx + 1]
            x_seq = x_ref[start:end].transpose(0, 1).unsqueeze(0).contiguous()
            initial_state = None
            if initial_state_mode[seq_idx]:
                initial_state = conv_states_ref[cache_indices[seq_idx]].unsqueeze(0)
            y_seq, _ = causal_conv1d_ref(
                x_seq,
                weight_ref,
                bias=None,
                initial_states=initial_state,
                return_final_states=True,
                final_states_out=conv_states_ref[cache_indices[seq_idx]].unsqueeze(0),
                activation=activation_from_mode(0),
            )
            outputs.append(y_seq.squeeze(0).transpose(0, 1).contiguous())

        y_ref = torch.cat(outputs, dim=0)
        conv_states_expected = ref_conv_states_to_op(conv_states_ref)

        seqlen, dim = x.shape
        expected = (
            y_ref.reshape(seqlen, head_num, dim // head_num)
            .transpose(0, 1)
            .contiguous()
        )
        self.assertTensorClose(y, expected, rtol=1e-1, atol=2e-1)
        self.assertTensorClose(conv_states, conv_states_expected, rtol=1e-1, atol=2e-1)

    def test_npu_causal_conv1d_update_matches_cpu_golden(self):
        x = make_tensor((2, 16), start=1.0)
        weight_op = make_tensor((4, 16), start=101.0)
        bias = make_tensor((16,), start=201.0)
        conv_states = make_tensor((2, 3, 16), start=301.0)
        conv_states_ref = op_conv_states_to_ref(conv_states)

        y = self.call_op(
            x=x,
            weight=weight_op,
            bias=bias,
            conv_states=conv_states,
            cache_indices=[0, 1],
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

        x_ref = x.detach().cpu().float()
        weight_ref = op_weight_to_ref(weight_op)
        bias_ref = bias.detach().cpu().float()

        y_ref = causal_conv1d_update_ref(
            x_ref,
            conv_states_ref,
            weight_ref,
            bias=bias_ref,
            activation=activation_from_mode(1),
            cache_seqlens=None,
        )

        self.assertTensorClose(y, y_ref)
        self.assertTensorClose(conv_states, ref_conv_states_to_op(conv_states_ref))

    def test_vllm_style_varlen_update_uses_max_query_len(self):
        dim, width, total_tokens = 16, 3, 5
        x = make_tensor((total_tokens, dim), start=1.0)
        x_before = x.clone()
        weight = make_tensor((width, dim), start=101.0)
        bias = make_tensor((dim,), start=201.0)
        conv_state = make_tensor((3, width - 1, dim), start=301.0)
        conv_state_ref = op_conv_states_to_ref(conv_state)
        query_values = [0, 2, total_tokens]
        query_start_loc = torch.tensor(
            query_values, dtype=torch.int32, device=x.device
        )
        state_indices = torch.tensor([1, 2], dtype=torch.int32, device=x.device)

        with self.assertRaisesRegex(ValueError, "observed segment length 3"):
            ascendc_ops.causal_conv1d_update(
                x,
                conv_state,
                weight,
                bias,
                activation="silu",
                conv_state_indices=state_indices,
                query_start_loc=query_start_loc,
                max_query_len=2,
                validate_data=True,
            )

        returned = ascendc_ops.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation="silu",
            conv_state_indices=state_indices,
            query_start_loc=query_start_loc,
            max_query_len=3,
            validate_data=True,
        )
        torch.npu.synchronize()

        expected_parts = []
        for seq_idx, cache_idx in enumerate((1, 2)):
            start, end = query_values[seq_idx : seq_idx + 2]
            selected_state = conv_state_ref[cache_idx : cache_idx + 1]
            expected_seq = causal_conv1d_update_ref(
                x_before[start:end]
                .detach()
                .cpu()
                .float()
                .transpose(0, 1)
                .unsqueeze(0),
                selected_state,
                op_weight_to_ref(weight),
                bias=bias.detach().cpu().float(),
                activation="silu",
            )
            expected_parts.append(
                expected_seq.squeeze(0).transpose(0, 1).contiguous()
            )

        self.assertEqual(returned.data_ptr(), x.data_ptr())
        self.assertTensorClose(
            x, torch.cat(expected_parts, dim=0), rtol=1e-1, atol=2e-1
        )
        self.assertTensorClose(
            conv_state,
            ref_conv_states_to_op(conv_state_ref),
            rtol=1e-1,
            atol=2e-1,
        )

    def test_npu_causal_conv1d_spec_decode_matches_cpu_golden(self):
        x = make_tensor((2, 4, 16), start=1.0)
        weight = make_tensor((4, 16), start=101.0)
        bias = make_tensor((16,), start=201.0)
        conv_states = make_tensor((2, 6, 16), start=301.0)
        conv_states_ref = op_conv_states_to_ref(conv_states)
        weight_ref = op_weight_to_ref(weight)
        bias_ref = bias.detach().cpu().float()
        x_ref = op_batch_x_to_ref(x)
        num_accepted_tokens = [2, 4]

        y = self.call_op(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            cache_indices=[0, 1],
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=1,
        )

        y_ref = causal_conv1d_update_spec_ref(
            x_ref,
            conv_states_ref,
            weight_ref,
            bias=bias_ref,
            num_accepted_tokens=num_accepted_tokens,
            activation=activation_from_mode(0),
        )
        self.assertTensorClose(y, y_ref.permute(0, 2, 1).contiguous())
        self.assertTensorClose(conv_states, ref_conv_states_to_op(conv_states_ref))

    def test_npu_causal_conv1d_update_width3_no_bias_no_activation(self):
        x = make_tensor((3, 3, 16), start=1.0)
        weight = make_tensor((3, 16), start=101.0)
        conv_states = make_tensor((3, 2, 16), start=301.0)
        conv_states_ref = op_conv_states_to_ref(conv_states)

        y = self.call_op(
            x=x,
            weight=weight,
            bias=None,
            conv_states=conv_states,
            cache_indices=[0, 1, 2],
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=1,
        )

        y_ref = causal_conv1d_update_ref(
            op_batch_x_to_ref(x),
            conv_states_ref,
            op_weight_to_ref(weight),
            bias=None,
            activation=None,
            cache_seqlens=None,
        )
        self.assertTensorClose(y, y_ref.permute(0, 2, 1).contiguous())
        self.assertTensorClose(conv_states, ref_conv_states_to_op(conv_states_ref))

    def test_npu_causal_conv1d_update_with_batch_gather_padding_matches_valid_rows(self):
        pad_slot_id = -1
        x = make_tensor((5, 3, 16), start=1.0)
        weight = make_tensor((4, 16), start=101.0)
        bias = make_tensor((16,), start=201.0)
        conv_states = make_tensor((7, 3, 16), start=301.0)
        conv_states_before = conv_states.clone()
        conv_states_ref = op_conv_states_to_ref(conv_states_before)
        valid_cache_indices = [1, 3, 5]
        cache_indices = valid_cache_indices + [pad_slot_id, pad_slot_id]

        y = self.call_op(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            cache_indices=cache_indices,
            activation_mode=1,
            pad_slot_id=pad_slot_id,
            run_mode=1,
        )

        x_ref = op_batch_x_to_ref(x)
        weight_ref = op_weight_to_ref(weight)
        bias_ref = bias.detach().cpu().float()
        for seq_idx, cache_idx in enumerate(cache_indices):
            if cache_idx == pad_slot_id:
                continue
            y_ref = causal_conv1d_update_ref(
                x_ref[seq_idx : seq_idx + 1],
                conv_states_ref[cache_idx : cache_idx + 1],
                weight_ref,
                bias=bias_ref,
                activation=activation_from_mode(1),
                cache_seqlens=None,
            )
            self.assertTensorClose(
                y[seq_idx : seq_idx + 1], y_ref.permute(0, 2, 1).contiguous()
            )

        conv_states_expected = ref_conv_states_to_op(conv_states_ref)
        self.assertTensorClose(
            conv_states[valid_cache_indices], conv_states_expected[valid_cache_indices]
        )
        unused_indices = [idx for idx in range(conv_states.shape[0]) if idx not in valid_cache_indices]
        self.assertTensorClose(
            conv_states[unused_indices], conv_states_before[unused_indices]
        )

    def test_npu_causal_conv1d_varlen_pad_slot_matches_valid_segments(self):
        pad_slot_id = -1
        x = make_tensor((7, 16), start=1.0)
        weight = make_tensor((4, 16), start=101.0)
        bias = make_tensor((16,), start=201.0)
        conv_states = make_tensor((2, 3, 16), start=301.0)
        conv_states_before = conv_states.clone()
        conv_states_ref = op_conv_states_to_ref(conv_states_before)

        query_start_loc = [0, 2, 4, 7]
        cache_indices = [0, pad_slot_id, 1]
        initial_state_mode = [1, 0, 1]

        y = self.call_op(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=initial_state_mode,
            activation_mode=1,
            pad_slot_id=pad_slot_id,
            run_mode=0,
        )

        x_ref = x.detach().cpu().float()
        weight_ref = op_weight_to_ref(weight)
        bias_ref = bias.detach().cpu().float()
        valid_token_ranges = []
        expected_outputs = []
        for seq_idx, cache_idx in enumerate(cache_indices):
            start = query_start_loc[seq_idx]
            end = query_start_loc[seq_idx + 1]
            if cache_idx == pad_slot_id:
                continue
            valid_token_ranges.append((start, end))
            x_seq = x_ref[start:end].transpose(0, 1).unsqueeze(0).contiguous()
            initial_state = None
            if initial_state_mode[seq_idx]:
                initial_state = conv_states_ref[cache_idx].unsqueeze(0)
            y_seq, _ = causal_conv1d_ref(
                x_seq,
                weight_ref,
                bias=bias_ref,
                initial_states=initial_state,
                return_final_states=True,
                final_states_out=conv_states_ref[cache_idx].unsqueeze(0),
                activation=activation_from_mode(1),
            )
            expected_outputs.append(y_seq.squeeze(0).transpose(0, 1).contiguous())

        for (start, end), expected in zip(valid_token_ranges, expected_outputs):
            self.assertTensorClose(y[start:end], expected, rtol=1e-1, atol=2e-1)

        self.assertTensorClose(conv_states, ref_conv_states_to_op(conv_states_ref), rtol=1e-1, atol=2e-1)

    def test_causal_conv1d_fn_device_metadata_matches_cpu_golden(self):
        total_tokens, dim, width = 5, 16, 4
        x = make_tensor((total_tokens, dim), start=1.0)
        weight = make_tensor((width, dim), start=101.0)
        bias = make_tensor((dim,), start=201.0)
        conv_states = make_tensor((3, width - 1, dim), start=301.0)
        conv_states_ref = op_conv_states_to_ref(conv_states)
        query_start_loc = torch.tensor([0, 2, total_tokens], dtype=torch.int32, device=x.device)
        cache_indices = torch.tensor([1, 2], dtype=torch.int32, device=x.device)
        has_initial_state = torch.tensor([False, True], dtype=torch.bool, device=x.device)

        y = ascendc_ops.causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            activation="silu",
            validate_data=True,
        )

        expected = []
        x_ref = x.detach().cpu().float()
        for seq_idx, cache_idx in enumerate([1, 2]):
            start, end = [0, 2, total_tokens][seq_idx : seq_idx + 2]
            initial_state = conv_states_ref[cache_idx].unsqueeze(0) if seq_idx == 1 else None
            y_seq, _ = causal_conv1d_ref(
                x_ref[start:end].transpose(0, 1).unsqueeze(0),
                op_weight_to_ref(weight),
                bias=bias.detach().cpu().float(),
                initial_states=initial_state,
                return_final_states=True,
                final_states_out=conv_states_ref[cache_idx].unsqueeze(0),
                activation="silu",
            )
            expected.append(y_seq.squeeze(0).transpose(0, 1).contiguous())

        self.assertTensorClose(y, torch.cat(expected, dim=0), rtol=1e-1, atol=2e-1)
        self.assertTensorClose(
            conv_states,
            ref_conv_states_to_op(conv_states_ref),
            rtol=1e-1,
            atol=2e-1,
        )

    def test_causal_conv1d_fn_host_metadata_matches_device_metadata_without_warning(self):
        total_tokens, dim, width = 5, 16, 4
        x = make_tensor((total_tokens, dim), start=1.0)
        weight = make_tensor((width, dim), start=101.0)
        bias = make_tensor((dim,), start=201.0)
        host_state = make_tensor((3, width - 1, dim), start=301.0)
        device_state = host_state.clone()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            host_y = ascendc_ops.causal_conv1d_fn(
                x,
                weight,
                bias,
                host_state,
                query_start_loc_cpu=[0, 2, total_tokens],
                cache_indices_cpu=[1, 2],
                has_initial_state_cpu=[0, 1],
                activation="silu",
                validate_data=True,
            )

        device_y = ascendc_ops.causal_conv1d_fn(
            x,
            weight,
            bias,
            device_state,
            query_start_loc=torch.tensor(
                [0, 2, total_tokens], dtype=torch.int32, device=x.device
            ),
            cache_indices=torch.tensor([1, 2], dtype=torch.int32, device=x.device),
            has_initial_state=torch.tensor([False, True], dtype=torch.bool, device=x.device),
            activation="silu",
            validate_data=True,
        )

        self.assertFalse(any(item.category is FutureWarning for item in caught))
        self.assertTensorClose(host_y, device_y, rtol=1e-1, atol=2e-1)
        self.assertTensorClose(host_state, device_state, rtol=1e-1, atol=2e-1)

    def test_causal_conv1d_fn_batch_matches_cpu_golden(self):
        batch, seqlen, dim, width = 2, 4, 16, 4
        x = make_tensor((batch, seqlen, dim), start=1.0)
        weight = make_tensor((width, dim), start=101.0)
        bias = make_tensor((dim,), start=201.0)

        y = ascendc_ops.causal_conv1d_fn(
            x,
            weight,
            bias,
            activation="silu",
        )

        y_ref, _ = causal_conv1d_ref(
            op_batch_x_to_ref(x),
            op_weight_to_ref(weight),
            bias=bias.detach().cpu().float(),
            activation="silu",
        )
        self.assertTensorClose(
            y,
            y_ref.permute(0, 2, 1).contiguous(),
            rtol=1e-1,
            atol=2e-1,
        )

    def test_causal_conv1d_fn_omitted_state_uses_zero_history(self):
        total_tokens, dim, width = 5, 16, 4
        x = make_tensor((total_tokens, dim), start=1.0)
        weight = make_tensor((width, dim), start=101.0)
        query_start_loc = torch.tensor([0, 2, total_tokens], dtype=torch.int32, device=x.device)

        y = ascendc_ops.causal_conv1d_fn(
            x,
            weight,
            None,
            query_start_loc=query_start_loc,
            activation=None,
        )

        expected = []
        x_ref = x.detach().cpu().float()
        for start, end in ((0, 2), (2, total_tokens)):
            y_seq, _ = causal_conv1d_ref(
                x_ref[start:end].transpose(0, 1).unsqueeze(0),
                op_weight_to_ref(weight),
                initial_states=None,
                activation=None,
            )
            expected.append(y_seq.squeeze(0).transpose(0, 1).contiguous())
        self.assertTensorClose(y, torch.cat(expected, dim=0), rtol=1e-1, atol=2e-1)

    def test_causal_conv1d_update_preserves_out_and_state_semantics(self):
        batch, seqlen, dim, width = 2, 3, 16, 3
        x = make_tensor((batch, seqlen, dim), start=1.0)
        x_before = x.clone()
        weight = make_tensor((width, dim), start=101.0)
        bias = make_tensor((dim,), start=201.0)
        conv_state = make_tensor((3, width - 1, dim), start=301.0)
        conv_state_ref = op_conv_states_to_ref(conv_state)
        indices = torch.tensor([1, 2], dtype=torch.int32, device=x.device)
        out = torch.empty_like(x)

        returned = ascendc_ops.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation="silu",
            conv_state_indices=indices,
            out=out,
        )

        selected_states = conv_state_ref[[1, 2]].clone()
        expected_ref = causal_conv1d_update_ref(
            op_batch_x_to_ref(x_before),
            selected_states,
            op_weight_to_ref(weight),
            bias=bias.detach().cpu().float(),
            activation="silu",
        )
        conv_state_ref[1].copy_(selected_states[0])
        conv_state_ref[2].copy_(selected_states[1])

        self.assertEqual(returned.data_ptr(), out.data_ptr())
        self.assertTensorClose(x, x_before)
        self.assertTensorClose(out, expected_ref.permute(0, 2, 1).contiguous(), rtol=1e-1, atol=2e-1)
        self.assertTensorClose(
            conv_state,
            ref_conv_states_to_op(conv_state_ref),
            rtol=1e-1,
            atol=2e-1,
        )


if __name__ == "__main__":
    unittest.main()
