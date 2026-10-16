"""A5 regression: BF16 raw gate and its exact FP32 expansion agree."""

import math
import ctypes

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import chunk_kda_bwd
from test_npu_chunk_kda_a5_staging import _chunked_reference


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("log_dtype", [torch.bfloat16, torch.float32])
def test_finalize_raw_g_dtype_pairs(beta_dtype, log_dtype):
    """Exercise all eight registered raw_g/beta/A_log combinations directly."""
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("requires Ascend950")
    from fla_npu.ops.ascendc._runtime import call_aclnn, ACL_FORMAT_ND

    torch.manual_seed(47)
    shape, scalar, matrix, state = (1, 2, 65, 128), (1, 2, 65), (1, 2, 65, 64), (1, 2, 2, 128, 128)

    def rand(dims, dtype=torch.bfloat16):
        return (torch.randn(dims) * 0.02).to(dtype).npu()

    raw = rand(shape)
    inputs = [rand(shape) for _ in range(3)] + [rand(shape, torch.float32), raw,
              rand(scalar, beta_dtype), rand((2,), log_dtype), rand((2, 128), torch.float32),
              rand(matrix), rand(shape), rand(state), rand(state), rand(shape),
              rand(matrix, torch.float32), rand(shape, torch.float32), None, None]

    def run(gate):
        tensors = list(inputs)
        tensors[4] = gate
        outputs = [torch.empty(shape, dtype=torch.bfloat16, device="npu") for _ in range(3)]
        outputs += [torch.empty(scalar, dtype=beta_dtype, device="npu"),
                    torch.empty(shape, device="npu"), torch.empty((2,), device="npu"),
                    torch.empty((2, 128), device="npu")]

        def build(ctx):
            def nd(x):
                return ctx.tensor(x, acl_format_override=ACL_FORMAT_ND,
                                  storage_shape_override=tuple(x.shape) if x is not None else None)
            return ([nd(x) for x in tensors] + [ctx.int_array(None), ctx.int_array(None),
                    ctypes.c_double(128 ** -0.5), ctypes.c_double(-5), ctypes.c_int64(64),
                    ctypes.c_bool(True), ctypes.c_bool(True), ctypes.c_bool(True), ctypes.c_bool(False)]
                    + [nd(x) for x in outputs])

        call_aclnn("aclnnChunkKdaBwdFinalize", build, tuple(outputs))
        torch.npu.synchronize()
        return [x.cpu() for x in outputs]

    for got, ref in zip(run(raw), run(raw.float())):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [7, 19, 43])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("packed,norm", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_bf16_raw_g_matches_fp32(seed, saved, packed, norm, beta_dtype):
    check_raw_g_equivalence(seed, saved, packed, norm, beta_dtype)


def check_raw_g_equivalence(seed, saved, packed, norm, beta_dtype, heads=8, tokens=131):
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("requires Ascend950")
    torch.manual_seed(seed)
    dim = 128
    shape = (1, tokens, heads, dim)
    q = torch.nn.functional.normalize(torch.randn(shape), dim=-1).bfloat16().npu()
    k = torch.nn.functional.normalize(torch.randn(shape), dim=-1).bfloat16().npu()
    v = torch.randn(shape).bfloat16().npu()
    raw = (torch.randn(shape) * 0.3 - 2).bfloat16().npu()
    beta = torch.sigmoid(torch.randn(shape[:-1])).to(beta_dtype).npu()
    a_log = torch.zeros(heads).npu()
    bias = torch.randn(heads, dim).mul_(0.1).npu()
    cu = (0, 1, 66, tokens) if packed else None
    indices = tuple(x for i, (a, b) in enumerate(zip(cu, cu[1:]))
                    for c in range(math.ceil((b - a) / 64)) for x in (i, c)) if packed else None
    scale = dim ** -0.5
    # Prepare saved tensors using the existing independent CPU forward
    # reference, so this backward regression needs only the backward OPP.
    spans = cu if packed else (0, tokens)
    gate = -5.0 * torch.sigmoid(raw.cpu().float() + bias.cpu())
    fwd = _chunked_reference(
        q, k, v, gate, beta, torch.zeros(len(spans) - 1, heads, dim, dim), spans,
    )
    _, _, gk, aqk, akk, w, _, qg, kg, v_new, h, _ = fwd
    # The CPU helper exports V-first states; the optimized backward is K-first.
    h = h.transpose(-1, -2).contiguous()
    gk, aqk, akk, w, qg, kg, v_new, h = (
        x.npu() for x in (gk, aqk, akk, w, qg, kg, v_new, h)
    )

    def head_first(x):
        x = x.transpose(1, 2).contiguous()
        return x.squeeze(0) if packed else x

    q, k, v, raw, beta = map(head_first, (q, k, v, raw, beta))
    if packed:
        gk, aqk, akk, w, qg, kg, v_new = (
            x.squeeze(0) if x.ndim == 4 else x
            for x in (gk, aqk, akk, w, qg, kg, v_new)
        )
        h = h.squeeze(0) if h.ndim == 5 else h
    do = torch.randn(tuple(q.shape)).bfloat16().npu()
    rstd = torch.ones(tuple(q.shape[:-1])).npu() if norm else None
    if not saved:
        gk = w = qg = kg = v_new = h = None

    def run(gate):
        outputs = chunk_kda_bwd(
            q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, do, scale,
            raw_g=gate, A_log=a_log, dt_bias=bias, safe_gate=True,
            lower_bound=-5.0, use_gate_in_kernel=True, disable_recompute=saved,
            use_exp2=True, implementation="optimized", cu_seqlens=cu,
            chunk_indices=indices, q_rstd=rstd, k_rstd=rstd,
        )
        torch.npu.synchronize()
        return tuple(x.cpu() if x is not None else None for x in outputs)

    actual, expected = run(raw), run(raw.float())
    for name, got, ref in zip(("dq", "dk", "dv", "db", "dg", "dh0", "dA", "dbias"), actual, expected):
        if ref is None:
            assert got is None, name
        else:
            assert torch.isfinite(got).all(), name
            # Both inputs contain identical values. Conversion location alone
            # must not change any output, including parameter reductions.
            torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=name)
