import unittest

import os
import torch
import torch_npu
import torch.nn.functional as F

import fla_npu

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

    def call_op(self, **kwargs):
        self.assertTrue(hasattr(torch.ops.npu, "npu_causal_conv1d"))
        return torch.ops.npu.npu_causal_conv1d(**kwargs)

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


if __name__ == "__main__":
    unittest.main()
