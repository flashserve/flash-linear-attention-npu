import os
import unittest

import torch
import torch_npu

import fla_npu


torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def make_tensor(shape, *, dtype=torch.float16, offset=0, device="cpu"):
    numel = 1
    for dim in shape:
        numel *= dim
    values = torch.arange(offset, offset + numel, dtype=torch.float32)
    values = (values.remainder(37) - 18.0) / 64.0
    return values.reshape(shape).to(dtype=dtype, device=device)


def causal_conv1d_preactivation_ref(x, weight, initial_state=None):
    """FP32 forward preactivation for logical [B, T, D] input."""
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    initial_fp32 = None if initial_state is None else initial_state.float()
    batch, seqlen, dim = x_fp32.shape
    width = weight_fp32.shape[0]
    out = torch.zeros((batch, seqlen, dim), dtype=torch.float32)

    for offset in range(width):
        rows = seqlen - offset
        if rows <= 0:
            break
        kernel = width - 1 - offset
        out[:, offset:, :] += x_fp32[:, :rows, :] * weight_fp32[kernel]

    if initial_fp32 is not None:
        for offset in range(1, width):
            kernel = width - 1 - offset
            for row in range(min(seqlen, offset)):
                state_slot = width - offset + row
                out[:, row, :] += (
                    initial_fp32[:, state_slot, :] * weight_fp32[kernel]
                )
    return out


def causal_conv1d_bwd_ref(
    x,
    y,
    weight,
    dy,
    initial_state=None,
    dht=None,
    activation=0,
    cast_reduction_outputs=True,
):
    """CPU golden. FP16/BF16 inputs are promoted to FP32 before calculation."""
    if activation not in (0, 1, 2):
        raise ValueError("activation must be 0, 1, or 2")
    if activation != 0 and y is None:
        raise ValueError("y is required when activation is enabled")

    dtype_in = x.dtype
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    grad = dy.float()
    initial_fp32 = None if initial_state is None else initial_state.float()
    dht_fp32 = None if dht is None else dht.float()

    if activation != 0:
        y_fp32 = y.float()
        sigmoid = torch.sigmoid(y_fp32)
        grad = grad * sigmoid * (1.0 + y_fp32 * (1.0 - sigmoid))

    batch, seqlen, dim = x_fp32.shape
    width = weight_fp32.shape[0]
    dx = torch.zeros((batch, seqlen, dim), dtype=torch.float32)
    dw = torch.zeros((width, dim), dtype=torch.float32)
    db = grad.sum(dim=(0, 1))
    dh0 = torch.zeros((batch, width, dim), dtype=torch.float32)

    for offset in range(width):
        rows = seqlen - offset
        if rows <= 0:
            break
        kernel = width - 1 - offset
        grad_slice = grad[:, offset:, :]
        dx[:, :rows, :] += grad_slice * weight_fp32[kernel]
        dw[kernel] += (grad_slice * x_fp32[:, :rows, :]).sum(dim=(0, 1))

    if initial_fp32 is not None:
        for offset in range(1, width):
            kernel = width - 1 - offset
            for row in range(min(seqlen, offset)):
                state_slot = width - offset + row
                dw[kernel] += (
                    grad[:, row, :] * initial_fp32[:, state_slot, :]
                ).sum(dim=0)

        for state_slot in range(1, width):
            for row in range(min(seqlen, state_slot)):
                kernel = state_slot - 1 - row
                dh0[:, state_slot, :] += (
                    grad[:, row, :] * weight_fp32[kernel]
                )

    if dht_fp32 is not None:
        tail_start = max(0, seqlen - (width - 1))
        tail_rows = seqlen - tail_start
        dx[:, tail_start:, :] += dht_fp32[:, 1 : 1 + tail_rows, :]

    if cast_reduction_outputs:
        dw = dw.to(dtype_in)
        db = db.to(dtype_in)
    return dx.to(dtype_in), dw, db, dh0.to(dtype_in)


def varlen_preactivation_ref(x, weight, initial_state, query_start_loc):
    total_tokens, dim = x.shape
    y = torch.empty((total_tokens, dim), dtype=torch.float32)
    for batch_idx, (start, end) in enumerate(
        zip(query_start_loc[:-1], query_start_loc[1:])
    ):
        initial = None
        if initial_state is not None:
            initial = initial_state[batch_idx : batch_idx + 1]
        y[start:end] = causal_conv1d_preactivation_ref(
            x[start:end].unsqueeze(0), weight, initial
        ).squeeze(0)
    return y


def varlen_bwd_ref(
    x,
    y,
    weight,
    dy,
    initial_state,
    dht,
    query_start_loc,
    activation,
):
    batch = len(query_start_loc) - 1
    width, dim = weight.shape
    dx = torch.empty_like(x)
    dw = torch.zeros((width, dim), dtype=torch.float32)
    db = torch.zeros((dim,), dtype=torch.float32)
    dh0 = torch.empty((batch, width, dim), dtype=x.dtype)

    for batch_idx, (start, end) in enumerate(
        zip(query_start_loc[:-1], query_start_loc[1:])
    ):
        y_seq = None if y is None else y[start:end].unsqueeze(0)
        initial_seq = initial_state[batch_idx : batch_idx + 1]
        dht_seq = dht[batch_idx : batch_idx + 1]
        dx_seq, dw_seq, db_seq, dh0_seq = causal_conv1d_bwd_ref(
            x[start:end].unsqueeze(0),
            y_seq,
            weight,
            dy[start:end].unsqueeze(0),
            initial_seq,
            dht_seq,
            activation,
            cast_reduction_outputs=False,
        )
        dx[start:end] = dx_seq.squeeze(0)
        dw += dw_seq
        db += db_seq
        dh0[batch_idx] = dh0_seq.squeeze(0)
    return dx, dw.to(x.dtype), db.to(x.dtype), dh0


def logical_to_bnsd(tensor, num_heads):
    batch, seqlen, dim = tensor.shape
    head_dim = dim // num_heads
    return (
        tensor.reshape(batch, seqlen, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def logical_to_ntd(tensor, num_heads):
    total_tokens, dim = tensor.shape
    head_dim = dim // num_heads
    return (
        tensor.reshape(total_tokens, num_heads, head_dim)
        .permute(1, 0, 2)
        .contiguous()
    )


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestCausalConv1dBwd(unittest.TestCase):
    rtol = 8e-2
    atol = 8e-2

    def call_op(self, **kwargs):
        self.assertTrue(hasattr(torch.ops.npu, "npu_causal_conv1d_bwd"))
        return torch.ops.npu.npu_causal_conv1d_bwd(**kwargs)

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(tuple(actual_cpu.shape), tuple(expected_cpu.shape))
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=(
                f"shape={tuple(actual_cpu.shape)}, "
                f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}"
            ),
        )

    def assertOutputsClose(self, actual, expected, *, rtol=None, atol=None):
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertTensorClose(
                actual_tensor, expected_tensor, rtol=rtol, atol=atol
            )

    def test_fixed_bsh_all_activations_matches_cpu_golden(self):
        batch, seqlen, dim, width = 2, 17, 80, 4
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = make_tensor((batch, seqlen, dim), dtype=dtype, offset=1)
            weight = make_tensor((width, dim), dtype=dtype, offset=101)
            dy = make_tensor((batch, seqlen, dim), dtype=dtype, offset=201)
            initial_state = make_tensor(
                (batch, width, dim), dtype=dtype, offset=301
            )
            dht = make_tensor((batch, width, dim), dtype=dtype, offset=401)
            preactivation = causal_conv1d_preactivation_ref(
                x, weight, initial_state
            ).to(dtype)

            for activation in (0, 1, 2):
                with self.subTest(dtype=dtype, activation=activation):
                    y = None if activation == 0 else preactivation
                    expected = causal_conv1d_bwd_ref(
                        x,
                        y,
                        weight,
                        dy,
                        initial_state,
                        dht,
                        activation,
                    )
                    actual = self.call_op(
                        x=x.npu(),
                        y=None if y is None else y.npu(),
                        weight=weight.npu(),
                        dy=dy.npu(),
                        initial_state=initial_state.npu(),
                        dht=dht.npu(),
                        activation=activation,
                        input_layout="BSH",
                    )
                    tolerance = 2e-4 if dtype == torch.float32 else None
                    self.assertOutputsClose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )

    def test_bnsd_matches_cpu_golden(self):
        batch, num_heads, seqlen, head_dim, width = 2, 2, 9, 16, 2
        dim = num_heads * head_dim
        dtype = torch.bfloat16
        x = make_tensor((batch, seqlen, dim), dtype=dtype, offset=11)
        weight = make_tensor((width, dim), dtype=dtype, offset=111)
        dy = make_tensor((batch, seqlen, dim), dtype=dtype, offset=211)
        initial_state = make_tensor(
            (batch, width, dim), dtype=dtype, offset=311
        )
        dht = make_tensor((batch, width, dim), dtype=dtype, offset=411)
        y = causal_conv1d_preactivation_ref(x, weight, initial_state).to(dtype)

        expected = causal_conv1d_bwd_ref(
            x, y, weight, dy, initial_state, dht, activation=2
        )
        actual = self.call_op(
            x=x.npu(),
            y=logical_to_bnsd(y, num_heads).npu(),
            weight=weight.npu(),
            dy=logical_to_bnsd(dy, num_heads).npu(),
            initial_state=initial_state.npu(),
            dht=dht.npu(),
            activation=2,
            input_layout="BNSD",
        )
        self.assertOutputsClose(actual, expected)

    def test_varlen_tnd_and_ntd_match_cpu_golden(self):
        query_start_loc = [0, 5, 13]
        batch, total_tokens, dim, width = 2, query_start_loc[-1], 32, 4
        num_heads = 2
        dtype = torch.float16
        x = make_tensor((total_tokens, dim), dtype=dtype, offset=21)
        weight = make_tensor((width, dim), dtype=dtype, offset=121)
        dy = make_tensor((total_tokens, dim), dtype=dtype, offset=221)
        initial_state = make_tensor(
            (batch, width, dim), dtype=dtype, offset=321
        )
        dht = make_tensor((batch, width, dim), dtype=dtype, offset=421)
        y = varlen_preactivation_ref(
            x, weight, initial_state, query_start_loc
        ).to(dtype)
        expected = varlen_bwd_ref(
            x,
            y,
            weight,
            dy,
            initial_state,
            dht,
            query_start_loc,
            activation=1,
        )

        for input_layout in ("TND", "NTD"):
            with self.subTest(input_layout=input_layout):
                if input_layout == "TND":
                    x_input, y_input, dy_input = x, y, dy
                else:
                    x_input = x
                    y_input = logical_to_ntd(y, num_heads)
                    dy_input = logical_to_ntd(dy, num_heads)
                actual = self.call_op(
                    x=x_input.npu(),
                    y=y_input.npu(),
                    weight=weight.npu(),
                    dy=dy_input.npu(),
                    initial_state=initial_state.npu(),
                    dht=dht.npu(),
                    query_start_loc=query_start_loc,
                    activation=1,
                    input_layout=input_layout,
                )
                self.assertOutputsClose(actual, expected)


if __name__ == "__main__":
    unittest.main()
