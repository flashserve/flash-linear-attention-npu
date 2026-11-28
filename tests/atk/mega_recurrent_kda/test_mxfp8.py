"""Compare the fused A5 MX projection to the vendor subgraph and the five-output CPU reference."""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from executor_mega_recurrent_kda import build_inputs, device_inputs, expected_outputs, to_device

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or "950" not in torch.npu.get_device_name(), reason="Ascend 950 MXFP8"
)


MUTABLE = ("conv_state", "replay_u", "replay_k", "replay_decay")


def official_qkv(args, algorithm):
    codes, sx = torch_npu.npu_dynamic_mx_quant(
        args["hidden_states"].flatten(0, -2), axis=-1, dst_type=torch.float8_e4m3fn, round_mode="rint",
        block_size=32, scale_alg=algorithm,
    )
    return torch_npu.npu_quant_matmul(
        codes, args["qkv_projection_weight"], args["qkv_projection_weight_scale"].view(torch.uint8),
        pertoken_scale=sx, scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[0, 0, 32], output_dtype=torch.bfloat16,
    )[0].cpu()


def check_step(host, args, lengths):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    projected = official_qkv(args, host["qkv_scale_alg"])
    actual = npu_mega_recurrent_kda(**args)
    assert all(t.data_ptr() == args[k].data_ptr() for t, k in zip(actual[1:], MUTABLE))
    assert torch.equal(args["checkpoint_state"].cpu(), host["checkpoint_state"])
    actual = tuple(t.cpu() for t in actual)
    slots = host["state_indices"].long()
    stored = torch.cat([actual[1][slot, 2:2+n] for slot, n in zip(slots, lengths)])
    torch.testing.assert_close(stored, projected, rtol=0.00782, atol=0.0001)
    expected = expected_outputs(host)
    tolerances = [(0.0156, 0.0156), (0.0156, 0.0156), (1.53e-5, 9.77e-4),
                  (1.95e-3, 1.95e-3), (1.53e-5, 9.77e-4)]
    for i, (got, want) in enumerate(zip(actual, expected)):
        if i:
            if i >= 2:
                got, want = got.transpose(1, 2), want.transpose(1, 2)
            active = torch.zeros(got.shape[:2], dtype=torch.bool)
            for slot, n in zip(slots, lengths):
                active[slot, :n + (2 if i == 1 else 0)] = True
            assert torch.equal(got[~active], want[~active]), (i, "inactive state modified")
            got, want = got[active], want[active]
        atol, rtol = tolerances[i]
        assert torch.isfinite(got).all()
        error = (got.float() - want.float()).abs()
        failed = (error > atol + rtol * want.float().abs()).sum().item()
        assert failed <= got.numel() * 0.001, (i, failed, got.numel(), error.max().item())


@pytest.mark.parametrize("lengths,heads,layout", [
    ([4], 1, "BSH"), ([63, 2], 2, "TH"), ([4] * 30, 12, "TH"),
    ([4] * 33, 3, "TH"), ([4] * 65, 3, "BSH"),
    ([4] * 15 + [5, 3] + [4] * 13, 12, "TH"),
    ([1, 64, 2], 5, "TH"), ([255], 3, "TH"), ([257], 3, "TH"),
    ([3] * 40, 8, "BSH"),
    ([4] * 65, 6, "TH"), ([4] * 96, 6, "BSH"),
    ([17] * 4, 8, "TH"),
])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_mx_qkv(lengths, heads, layout, algorithm):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda_metadata

    large_pool = heads == 12
    spec = dict(name="mx", lengths=lengths, heads=heads, layout=layout, conv="flat", seed=17, qkv_scale_alg=algorithm)
    if large_pool:
        spec["slots"] = 492
    host = build_inputs(spec)
    if heads == 8 and lengths == [17] * 4:
        # Isolate the cross-tile schedule. One product per decay-A output makes its
        # BF16 rounding independent of the CPU and Cube accumulation orders.
        weight = host["decay_projection_a_weight"]
        columns = weight.abs().argmax(dim=1, keepdim=True)
        values = weight.gather(1, columns)
        weight.zero_().scatter_(1, columns, values)
    if large_pool:
        host["state_indices"] = torch.linspace(0, 491, len(lengths)).round().int().flip([0])
    if layout == "TH":
        host["cu_seqlens"] = host["cu_seqlens"].to(torch.int64 if algorithm else torch.int32)
    device = torch.device("npu", torch.npu.current_device())
    args = device_inputs(host, device)
    if layout == "TH" and algorithm == 0:
        args["metadata"] = npu_mega_recurrent_kda_metadata(args["cu_seqlens"], sum(lengths))
    if algorithm == 1:
        args["qkv_projection_weight_scale"] = args["qkv_projection_weight_scale"].view(torch.float8_e8m0fnu)
    check_step(host, args, lengths)
    if large_pool:
        # A second batch reuses the device's updated convolution history with new tokens and acceptance counts.
        host["hidden_states"] = -host["hidden_states"].roll(1, dims=0)
        host["conv_num_accepted_tokens"] = torch.tensor(lengths, dtype=torch.int32)
        for name in ("hidden_states", "conv_num_accepted_tokens"):
            args[name] = host[name].to(device)
        check_step(host, args, lengths)


@pytest.mark.parametrize("hidden,lengths,algorithm", [(4096, [4, 4], 0), (4160, [63, 2], 1), (7168, [1], 0)])
def test_hidden_width(hidden, lengths, algorithm):
    host = build_inputs(dict(name="width", lengths=lengths, heads=2, layout="TH", conv="flat", seed=17,
                             hidden=hidden, qkv_scale_alg=algorithm))
    check_step(host, device_inputs(host, torch.device("npu", torch.npu.current_device())), lengths)


def test_cross_tile_carry_is_deterministic():
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda, npu_mega_recurrent_kda_metadata

    # Rows 128..191 consume one request's carry and publish a different request's carry.
    lengths = [3, 187, 7, 1, 66, 129, 3]
    host = build_inputs(dict(name="carry", lengths=lengths, heads=12, layout="TH", conv="flat",
                             slots=8, seed=17, qkv_scale_alg=0))
    device = torch.device("npu", torch.npu.current_device())
    args = device_inputs(host, device)
    args["metadata"] = npu_mega_recurrent_kda_metadata(args["cu_seqlens"], sum(lengths))
    initial = {key: args[key].clone() for key in MUTABLE}
    check_step(host, args, lengths)
    expected = None
    for _ in range(12):
        for key, value in initial.items():
            args[key].copy_(value)
        outputs = tuple(t.cpu().contiguous().view(torch.uint8) for t in npu_mega_recurrent_kda(**args))
        if expected is None:
            expected = outputs
        else:
            assert all(torch.equal(a, b) for a, b in zip(outputs, expected))


@pytest.mark.parametrize("algorithm", [0, 1])
def test_scale_zero_and_bf16_subnormal(algorithm):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    host = build_inputs(dict(name="zero_scale", lengths=[4], heads=1, layout="BSH", conv="flat", seed=23))
    raw = host["hidden_states"].view(torch.int16).reshape(4, 7168)
    raw.zero_()
    raw[1:, :32] = torch.arange(32, dtype=torch.int16) + torch.tensor([[0], [128], [896]])
    dev = torch.device("npu", torch.npu.current_device())
    args = {k: to_device(v, dev) for k, v in host.items()}
    args["a_log"] = args.pop("A_log")
    w = torch.zeros(1, 7168, 384)
    c = torch.arange(384)
    w[0, c % 32, c] = 1
    args["qkv_projection_weight"] = torch_npu.npu_format_cast(w.to(torch.float8_e4m3fn).to(dev), 29)
    sw = torch.full((1, 112, 384, 2), 127, dtype=torch.uint8)
    sw[0, 0, :, 0] = 247
    args.update(qkv_projection_weight_scale=sw.to(dev), qkv_scale_alg=algorithm)
    codes, sx = torch_npu.npu_dynamic_mx_quant(
        args["hidden_states"].reshape(4, 7168), dst_type=torch.float8_e4m3fn, block_size=32, scale_alg=algorithm
    )
    expected = torch_npu.npu_quant_matmul(
        codes, args["qkv_projection_weight"], args["qkv_projection_weight_scale"], pertoken_scale=sx,
        scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[0, 0, 32], output_dtype=torch.bfloat16,
    )[0].cpu()
    actual = npu_mega_recurrent_kda(**args)[1][host["state_indices"][0].item(), 2:6].cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def mx_inputs():
    host = build_inputs(dict(name="invalid_mx", lengths=[2], heads=1, layout="TH", conv="flat", seed=19,
                             qkv_scale_alg=0))
    return device_inputs(host, torch.device("npu", torch.npu.current_device()))


@pytest.mark.parametrize("invalid", ["missing_scale", "bf16_with_scale", "plain_weight", "packed_scale",
                                      "scale_shape", "scale_dtype", "algorithm", "host_prefix"])
def test_invalid_mx_inputs(mx_inputs, invalid):
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda

    args = dict(mx_inputs)
    w = args["qkv_projection_weight"]
    sw = args["qkv_projection_weight_scale"]
    if invalid == "missing_scale":
        args.pop("qkv_projection_weight_scale")
    elif invalid == "bf16_with_scale":
        args["qkv_projection_weight"] = torch.empty((384, 7168), dtype=torch.bfloat16, device=w.device)
    elif invalid == "plain_weight":
        args["qkv_projection_weight"] = torch_npu.npu_format_cast(w, 2)
    elif invalid == "packed_scale":
        args["qkv_projection_weight_scale"] = torch_npu.npu_format_cast(sw, 29)
    elif invalid == "scale_shape":
        args["qkv_projection_weight_scale"] = sw.flatten()
    elif invalid == "scale_dtype":
        args["qkv_projection_weight_scale"] = sw.float()
    elif invalid == "algorithm":
        args["qkv_scale_alg"] = 2
    else:
        args["cu_seqlens"] = args["cu_seqlens"].cpu()
    before = [args[k].cpu() for k in (*MUTABLE, "checkpoint_state")]
    with pytest.raises((ValueError, RuntimeError)):
        npu_mega_recurrent_kda(**args)
        torch.npu.synchronize()
    assert all(torch.equal(args[k].cpu(), old) for k, old in zip((*MUTABLE, "checkpoint_state"), before))
