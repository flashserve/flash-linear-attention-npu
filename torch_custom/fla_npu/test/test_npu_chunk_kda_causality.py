"""Causality regressions for the generic A2/A3 KDA safe-gate implementation."""
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module", autouse=True)
def generic_a2_a3_scope():
    import torch_npu  # noqa: F401
    import fla_npu.ops.ascendc  # noqa: F401 -- register OPP before device query.

    if "950" in torch.npu.get_device_name():
        pytest.skip("This regression covers generic A2/A3; A5 uses the unaffected arch35 implementation.")


def make_case(n, d, dtype, gate_dtype):
    """Normal initialization; fixed full-length draws keep tails independent."""
    rng = torch.Generator(device="cpu").manual_seed(20261103)
    shape = (1, 143, 16, d)

    def normal(size):
        return torch.randn(size, generator=rng, device="cpu", dtype=torch.float32)

    q, k = [F.normalize(normal(shape), dim=-1).to(dtype) for _ in range(2)]
    v = normal(shape).to(dtype)
    g = normal(shape).to(gate_dtype)
    beta = normal(shape[:-1]).sigmoid()  # FP32, already transformed.
    state = normal((1, 16, d, d))  # Nonzero FP32 KV initial state.
    xs = [x[:, :n].contiguous().clone() for x in (q, k, v, g, beta)]
    kw = dict(initial_state=state, A_log=torch.zeros(16),
              dt_bias=torch.zeros(16 * d), layout="BSND", chunk_size=64,
              cu_seqlens=(0, n),
              chunk_indices=tuple(y for c in range((n + 63) // 64) for y in (0, c)),
              safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
              output_final_state=True, state_v_first=False,
              disable_recompute=False, return_intermediate_states=False,
              use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
              allow_neg_eigval=False, use_exp2=True, epsilon=1e-6)
    return xs, d ** -0.5, kw


def run_case(xs, scale, kw):
    """Fresh device copies of ALL tensors, including mutable state, per call."""
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import chunk_kda_fwd

    def copy(x):
        return x.clone().to("npu") if isinstance(x, torch.Tensor) else x

    with torch.no_grad():
        result = chunk_kda_fwd(*[copy(x) for x in xs], scale,
                               **{key: copy(value) for key, value in kw.items()})
    # Snapshot before another invocation can mutate state or reuse storage.
    return tuple(result[i].detach().cpu().clone() for i in (0, 1, 3))


def valid_score(aqk, n):
    """Aqk is B,H,T,64; only each chunk's inclusive lower triangle is valid."""
    mask = torch.arange(64)[None, :] <= (torch.arange(n) % 64)[:, None]
    return aqk[..., :n, :][..., mask]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32], ids=["g-bf16", "g-fp32"])
@pytest.mark.parametrize("d", [64, 128], ids=["D64", "D128"])
@pytest.mark.parametrize("n", [17, 63, 64, 143], ids=["T17", "T63", "T64", "T143"])
def test_future_gate_cannot_change_past(n, d, dtype, gate_dtype):
    xs, scale, kw = make_case(n, d, dtype, gate_dtype)
    changed = [x.clone() for x in xs]
    changed[3][:, 16] += 1  # Zero-based token 16 only; same shape and all other data.
    baseline = run_case(xs, scale, kw)
    perturbed = run_case(changed, scale, kw)
    for out, state, aqk in (baseline, perturbed):
        assert torch.isfinite(out).all(), "nonfinite output"
        assert torch.isfinite(state).all(), "nonfinite final state"
        assert torch.isfinite(valid_score(aqk, n)).all(), "nonfinite valid Aqk"
    assert torch.equal(baseline[0][:, :16], perturbed[0][:, :16]), "past output changed"
    assert torch.equal(valid_score(baseline[2], 16), valid_score(perturbed[2], 16)), "past Aqk changed"


def test_large_amplitude_safe_gate_stays_finite():
    xs, scale, kw = make_case(64, 128, torch.bfloat16, torch.bfloat16)
    for x in xs:
        x.zero_()
    xs[0][..., 0] = 1000
    xs[1][..., 0] = 1000
    xs[3].fill_(100)
    kw["initial_state"].zero_()
    out, state, aqk = run_case(xs, scale, kw)
    assert torch.equal(out, torch.zeros_like(out)), "output must remain zero"
    assert torch.equal(state, torch.zeros_like(state)), "state must remain zero"
    assert torch.isfinite(valid_score(aqk, 64)).all(), "nonfinite valid Aqk"


def test_historical_score_does_not_inflate():
    import math

    xs, scale, kw = make_case(64, 128, torch.bfloat16, torch.bfloat16)
    for x in xs:
        x.zero_()
    xs[0][..., 0] = 1000
    xs[1][..., 0] = 1000
    xs[3][:, :33] = -0.625
    kw["initial_state"].zero_()
    kw["use_gate_in_kernel"] = False
    _, _, aqk = run_case(xs, scale, kw)
    # Decay excludes gate[0]: sum(gate[1:33]) = -20 in natural-log units.
    actual = aqk[..., 32, 0].double()
    expected = torch.full_like(actual, scale * 1e6 * math.exp(-20))
    # Ascend/ATK 45cc4061bc019e73f63a11e96496e97df0d6810c:
    # MixedToleranceBenchmarkAccuracyCompare BF16 defaults rtol=atol=1.56e-2.
    # This rejects artificial inflation; zero is allowed for this tiny term.
    # Apply the budget to one score per head, not a full ATK verdict.
    torch.testing.assert_close(actual, expected, rtol=1.56e-2, atol=1.56e-2)
