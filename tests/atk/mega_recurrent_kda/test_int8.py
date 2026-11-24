"""Validate fused A2 INT8 QKV against the native quantized projection."""
import ctypes

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from executor_mega_recurrent_kda import build_inputs, device_inputs
from mega_recurrent_kda_reference import mega_recurrent_kda_golden

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or "910B" not in torch.npu.get_device_name(), reason="Ascend A2 INT8 QKV"
)
MUTABLE = ("conv_state", "replay_u", "replay_k", "replay_decay")


def quantized_weights(host, args, explicit):
    w = host["qkv_projection_weight"].float().T.contiguous()
    sw = (w.abs().amax(dim=0) / 127).bfloat16()
    w = (w / sw).round().clamp(-127, 127).to(torch.int8)
    device = args["hidden_states"].device
    cols = w.shape[1]
    packed = w.reshape(448, 16, cols // 32, 32).permute(2, 0, 1, 3).contiguous().to(device)
    if explicit:
        args["qkv_projection_weight"] = packed
    else:
        previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        try:
            torch.npu.config.allow_internal_format = True
            args["qkv_projection_weight"] = torch_npu.npu_format_cast(w.to(device), 29)
        finally:
            torch.npu.config.allow_internal_format = previous in (b"enable", "enable")
    args["qkv_projection_weight_scale"] = sw.to(device)
    return packed


def model_qkv(hidden, packed, channel_scale):
    from aclnn_error_probe import _CallContext, runtime

    codes, sx = torch_npu.npu_dynamic_quant(hidden.reshape(-1, 7168))
    out = torch.empty((len(codes), channel_scale.numel()), device=codes.device, dtype=torch.bfloat16)
    args = _CallContext(runtime())
    api = ctypes.CDLL("libopapi.so")
    ptr, u64 = ctypes.c_void_p, ctypes.c_uint64
    ws, executor = u64(), ptr()
    stream = torch.npu.current_stream(hidden.device)
    try:
        query = api.aclnnQuantMatmulWeightNzGetWorkspaceSize
        query.argtypes = [ptr] * 9 + [ctypes.c_bool, ctypes.c_bool, ctypes.c_int64, ptr,
                                      ctypes.POINTER(u64), ctypes.POINTER(ptr)]

        def nd(t, name):
            return args.tensor(t, name, acl_format_override=2, storage_shape_override=t.shape)

        weight = args.tensor(packed.view(7168, -1), "weight", acl_format_override=29,
                             storage_shape_override=packed.shape)
        descriptors = [nd(codes, "x"), weight, nd(sx, "token_scale"), nd(channel_scale, "channel_scale"), nd(out, "out")]
        status = query(*descriptors[:4], None, None, None, None, None, False, False, 0, descriptors[-1],
                       ctypes.byref(ws), ctypes.byref(executor))
        assert status == 0, args.api._recent_error_message()
        buffer = torch.empty(ws.value, dtype=torch.uint8, device=hidden.device)
        execute = api.aclnnQuantMatmulWeightNz
        execute.argtypes = [ptr, u64, ptr, ptr]
        status = execute(buffer.data_ptr(), ws, executor, stream.npu_stream)
        assert status == 0, args.api._recent_error_message()
        return out.cpu()
    finally:
        args.destroy()


@pytest.mark.parametrize("lengths,heads,layout,dtype,explicit", [
    ([4] * 8, 6, "TH", torch.int32, True),
    ([4] * 8, 6, "BSH", torch.int32, False),
    ([63, 2], 1, "TH", torch.int64, True),
    ([1, 65, 7], 2, "TH", torch.int32, False),
    ([3] * 23, 3, "BSH", torch.int32, True),
    ([1], 1, "BSH", torch.int32, True),
    ([1] * 65, 2, "TH", torch.int64, True),
    ([193], 8, "TH", torch.int64, False),
    ([4] * 17, 6, "BSH", torch.int32, True),
    ([63, 2], 6, "TH", torch.int64, True),
])
def test_int8_qkv(lengths, heads, layout, dtype, explicit):
    check_int8_qkv(lengths, heads, layout, dtype, explicit)


def test_int8_long_sequence_scale_rounding():
    check_int8_qkv([193], 6, "TH", torch.int64, False, check_scale_order=True)


def check_int8_qkv(lengths, heads, layout, dtype, explicit, check_scale_order=False):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda, npu_mega_recurrent_kda_metadata

    model = (len(lengths), heads) == (8, 6) and layout == "TH"
    host = build_inputs(dict(name="int8", lengths=lengths, heads=heads, layout=layout, conv="flat",
                             slots=567 if model else len(lengths) + 1, seed=43))
    if model:
        host["state_indices"] = torch.linspace(0, 566, 8).round().int().flip([0])
    if layout == "TH":
        host["cu_seqlens"] = host["cu_seqlens"].to(dtype)
    dev = torch.device("npu", torch.npu.current_device())
    args = device_inputs(host, dev)
    packed = quantized_weights(host, args, explicit)
    if layout == "TH" and explicit:
        args["metadata"] = npu_mega_recurrent_kda_metadata(args["cu_seqlens"], sum(lengths))
    for step in range(2 if model else 1):
        x = host["hidden_states"].reshape(-1, 7168)
        if step:
            x.neg_()
            host["conv_num_accepted_tokens"] = torch.tensor(lengths, dtype=torch.int32)
            args["conv_num_accepted_tokens"] = host["conv_num_accepted_tokens"].to(dev)
        elif len(x) > 1:
            x[0].zero_()
        args["hidden_states"] = host["hidden_states"].to(dev)
        projected = model_qkv(args["hidden_states"], packed, args["qkv_projection_weight_scale"])
        initial = {key: args[key].clone() for key in MUTABLE}
        versions = [args[key]._version for key in MUTABLE]
        got = npu_mega_recurrent_kda(**args)
        assert [args[key]._version for key in MUTABLE] == [version + 1 for version in versions]
        if heads == 1:
            repeat_args = dict(args, **{key: initial[key].clone() for key in MUTABLE})
            repeat = npu_mega_recurrent_kda(**repeat_args)
            assert all(torch.equal(left, right) for left, right in zip(got, repeat))
        assert all(t.data_ptr() == args[k].data_ptr() for t, k in zip(got[1:], MUTABLE))
        assert torch.equal(args["checkpoint_state"].cpu(), host["checkpoint_state"])
        results = tuple(t.cpu() for t in got)
        slots = host["state_indices"].long()
        stored = torch.cat([results[1][slot, 2:2+n] for slot, n in zip(slots, lengths)])
        if check_scale_order:
            # FP32 scale multiplication can change the final BF16 rounding at a tie.
            # Verify both sides from the same integer dot product at every differing element.
            rows, cols = torch.where(stored != projected)
            assert rows.numel() <= 1, f"unexpected number of QKV rounding differences: {rows.numel()}"
            if len(rows):
                codes, token_scale = torch_npu.npu_dynamic_quant(args["hidden_states"].reshape(-1, 7168))
                weights = packed.cpu().permute(1, 2, 0, 3).reshape(7168, -1)
                dots = (codes.cpu()[rows].int() * weights[:, cols].T.int()).sum(dim=1, dtype=torch.int32).float()
                sx = token_scale.cpu()[rows].float()
                sw = args["qkv_projection_weight_scale"].cpu()[cols].float()
                torch.testing.assert_close(stored[rows, cols], ((dots * sw) * sx).bfloat16(), atol=0, rtol=0)
                torch.testing.assert_close(projected[rows, cols], (dots * (sw * sx)).bfloat16(), atol=0, rtol=0)
        else:
            torch.testing.assert_close(stored, projected, atol=0, rtol=0)
        expected = mega_recurrent_kda_golden(**host, projected_qkv=stored)
        for i, (actual, want, (atol, rtol)) in enumerate(zip(results, expected, [
            (0.0156, 0.0156), (0, 0), (1.53e-5, 9.77e-4), (1.95e-3, 1.95e-3), (1.53e-5, 9.77e-4)
        ])):
            if i:
                if i >= 2:
                    actual, want = actual.transpose(1, 2), want.transpose(1, 2)
                active = torch.zeros(actual.shape[:2], dtype=torch.bool)
                for slot, n in zip(slots, lengths):
                    active[slot, :n + (2 if i == 1 else 0)] = True
                assert torch.equal(actual[~active], want[~active]), i
                actual, want = actual[active], want[active]
            err = (actual.float() - want.float()).abs()
            assert torch.isfinite(err).all()
            failed = (err > atol + rtol * want.float().abs()).sum().item()
            if i == 4 and failed > err.numel() * 0.001:
                # This output depends only on the two BF16 decay projections. Check the
                # complete pool against the previously validated, independent BF16 call.
                baseline = dict(args, **initial)
                baseline.pop("qkv_projection_weight_scale")
                baseline["qkv_projection_weight"] = host["qkv_projection_weight"].to(dev)
                old_decay = npu_mega_recurrent_kda(**baseline)[4].cpu()
                torch.testing.assert_close(results[4], old_decay, atol=0, rtol=0)
            else:
                assert failed <= err.numel() * 0.001, (i, failed, err.max())


@pytest.mark.parametrize("invalid", ["missing_scale", "scale_dtype", "scale_length", "algorithm", "plain_weight",
                                     "wrong_nz_shape", "host_prefix"])
def test_invalid_int8(invalid):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    host = build_inputs(dict(name="invalid", lengths=[2], heads=1, layout="TH", conv="flat", seed=9))
    args = device_inputs(host, torch.device("npu", torch.npu.current_device()))
    w = host["qkv_projection_weight"]
    args["qkv_projection_weight"] = torch.zeros((12, 448, 16, 32), dtype=torch.int8, device="npu")
    args["qkv_projection_weight_scale"] = torch.ones(384, dtype=torch.bfloat16, device="npu")
    if invalid == "missing_scale":
        args.pop("qkv_projection_weight_scale")
    elif invalid == "scale_dtype":
        args["qkv_projection_weight_scale"] = args["qkv_projection_weight_scale"].float()
    elif invalid == "scale_length":
        args["qkv_projection_weight_scale"] = args["qkv_projection_weight_scale"][:-1]
    elif invalid == "algorithm":
        args["qkv_scale_alg"] = 1
    elif invalid == "plain_weight":
        args["qkv_projection_weight"] = w.T.contiguous().to("npu", dtype=torch.int8)
    elif invalid == "wrong_nz_shape":
        args["qkv_projection_weight"] = args["qkv_projection_weight"].reshape(12, 448, 32, 16)
    else:
        args["cu_seqlens"] = host["cu_seqlens"]
    before = [args[k].cpu() for k in MUTABLE]
    with pytest.raises((ValueError, RuntimeError)):
        npu_mega_recurrent_kda(**args)
        torch.npu.synchronize()
    assert all(torch.equal(v, args[k].cpu()) for k, v in zip(MUTABLE, before))
