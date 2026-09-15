"""Wheel 安装态数值回归：已迁移 stable 算子的 ctypes-vs-stable parity。

用法（221 上，wheel 已 pip install --target envXXX）:
    PYTHONPATH=/path/envXXX python tests/stable_abi/regression_ops.py

每个场景对同一输入分别走 ctypes 与 stable 两条 host 路径，断言每个输出
tensor 的逐元素差为 0（同一 OPP kernel，期望 bitwise 相同）。host P50
仅作参考；合法域说明见 stable-abi-inventory.md。
"""
from __future__ import annotations

import itertools
import json
import os
import re
import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import _stable as _launcher  # noqa: E402


# name -> max |diff| observed for that scenario (0.0 when it matched).  The
# driver snapshots this into tests/stable_abi/stable_scenarios.json, so "which scenarios
# are covered" is a checked-in fact that a later run has to reproduce rather
# than a number printed once in a log.
SCENARIOS: dict[str, float] = {}
# name -> reason, for cases both backends reject (domain limits, missing kernel).
SKIPPED: dict[str, str] = {}
# Named slices of the scenario list.
#
# The whole matrix is the merge gate: every operator at every layout, flag,
# dtype and boundary it declares.  On 910B3 that is ~10 minutes, which is the
# wrong granularity for an edit loop that touched one family, so the scenarios
# are grouped by operator family and a driver can run one group instead.
#
# Membership is by function name, so the two drivers (this one and
# regression_stable_full.py, which carries a few extra conv1d scenarios) share
# the same table.  check_groups() fails if a scenario belongs to no group, so
# adding a scenario cannot silently drop it out of every fast path.
GROUPS: dict[str, tuple[str, ...]] = {
    # What vLLM actually calls, in the order it calls it: the decode path.
    "hot": (
        "scenario_recurrent_gated_delta_rule",
        "scenario_recurrent_kda",
        "scenario_conv1d_new_apis",
        "scenario_conv1d_update",
    ),
    "recurrent": (
        "scenario_recurrent_gated_delta_rule",
        "scenario_recurrent_kda",
    ),
    "conv1d": (
        "scenario_conv1d_new_apis",
        "scenario_conv1d_prefill",
        "scenario_conv1d_update",
        "scenario_conv1d_update_offset_state",
        "scenario_conv1d_update_paged_state",
        "scenario_conv1d_prefill_paged_state",
        "scenario_conv1d_varlen_initial_state",
        "scenario_conv1d_gather_padding",
        "scenario_conv1d_varlen_pad_slot",
        "scenario_conv1d_bwd_bnsd",
    ),
    "kda": (
        "scenario_chunk_kda_fwd",
        "scenario_chunk_kda_fwd_variants",
        "scenario_chunk_kda_bwd_intra",
        "scenario_chunk_kda_bwd",
        "scenario_chunk_kda_bwd_recompute",
        "scenario_kda_gate_cumsum",
    ),
    # Everything in the chunked GDN backward, which is the bulk of the matrix
    # and the slowest group (T=256..1024 shapes).
    "chunk": (
        "scenario_recompute",
        "scenario_pwy_full",
        "scenario_pwy",
        "scenario_dv_local",
        "scenario_pwy_da",
        "scenario_gated_fwd_h",
        "scenario_chunk_fwd_h",
        "scenario_chunk_fwd_o",
        "scenario_bwd_dhu",
        "scenario_dqkwg",
        "scenario_chunk_local_cumsum",
        "scenario_scaled_dot_kkt",
        "scenario_solve_tri_dense",
        "scenario_solve_tri_guards",
        "scenario_chunk_gated_delta_rule_fwd",
    ),
    "smoke": ("scenario_fast_gelu",),
    # Ascend950-only: the fused backward composes the 950-only finalize kernel,
    # so it runs with the A5 driver rather than in the A2 matrix.
    "a5": ("scenario_chunk_gated_delta_rule_bwd",),
}


def check_groups(names: list[str]) -> list[str]:
    """Scenario names that no group selects (must be empty)."""

    covered = {name for members in GROUPS.values() for name in members}
    return [name for name in names if name not in covered]


def select_groups(names: list[str], wanted: list[str]) -> list[str]:
    """Filter *names* by group, keeping the driver's own order."""

    if not wanted:
        return names
    selected: set[str] = set()
    for group in wanted:
        if group not in GROUPS:
            raise SystemExit(f"unknown group {group!r}; known: "
                             f"{', '.join(sorted(GROUPS))}")
        selected.update(GROUPS[group])
    return [name for name in names if name in selected]


def missing_from_groups(names: list[str]) -> None:
    """Fail loudly when a scenario was added without a group."""

    ungrouped = check_groups(names)
    if ungrouped:
        raise SystemExit(
            "these scenarios are in no group, so a --group run would silently "
            f"skip them: {', '.join(ungrouped)}; add them to GROUPS in "
            "regression_ops.py")


def group_cli(parser) -> None:
    """The --group/--list-groups options both drivers share."""

    parser.add_argument("--group", action="append", default=[],
                        help="run only this group (repeatable); see "
                             "--list-groups")
    parser.add_argument("--list-groups", action="store_true",
                        help="print the groups and exit")


def print_groups(names: list[str]) -> None:
    print(f"{'group':<12} {'scenarios':>9}  members")
    for group, members in GROUPS.items():
        present = [name for name in names if name in members]
        print(f"{group:<12} {len(present):>9}  {', '.join(present)}")
    print(f"{'total':<12} {len(names):>9}")


def pct(vals, q):
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def host_p50(fn, n=200):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.npu.synchronize()
    return pct(ts, 0.5)


def _value_diff(left, right) -> float:
    """Max |left - right|, with non-finite entries compared by kind.

    ``(a - b).abs().max()`` is not an equality test once either side is not
    finite: ``inf - inf`` is NaN, so two *identical* infinite tensors were
    reported as a failure.  Measured on Ascend950PR, the T=256 / chunk=128 /
    g=float32 variant of ``chunk_kda_fwd`` produces the same 89 infinities on
    both host paths -- a kernel-side overflow the parity check has to be able
    to express as "equal", otherwise the gate fails for a reason that has
    nothing to do with the host path it exists to guard.

    Equal non-finite entries count as equal (NaN to NaN, inf to inf); a
    finite/infinite or +inf/-inf pair does not.
    """

    same = left == right
    both_nan = torch.isnan(left) & torch.isnan(right)
    if bool((same | both_nan).all()):
        return 0.0
    return float((left - right).abs().max().item())


def assert_parity(name, oc, ot):
    """Bit-exact comparison, recorded in SCENARIOS under *name*.

    A case that only prints "PASS" is invisible to the scenario-set check, so
    every comparison in this file has to come through here or through
    `record_extra`."""

    if not isinstance(oc, tuple):
        oc = (oc,)
        ot = (ot,)
    assert len(oc) == len(ot), f"{name}: output count mismatch"
    for i, (a, b) in enumerate(zip(oc, ot)):
        if a is None or b is None:
            assert a is None and b is None, f"{name}[{i}]: None mismatch"
            continue
        assert tuple(a.shape) == tuple(b.shape), f"{name}[{i}]: shape"
        left, right = a.float(), b.float()
        diff = _value_diff(left, right)
        assert diff == 0.0, f"{name}[{i}]: diff={diff}"
    SCENARIOS[name] = 0.0
    print(f"PASS {name}")


def record_extra(name, left, right):
    """Record an *additional* comparison under *name*.

    Some scenarios check more than the operator's outputs -- an in-place state
    that must match, a pad slot that must come back untouched.  Printing "PASS"
    for those leaves them outside the scenario set, so deleting the check would
    not be noticed; going through here puts them in the baseline with the same
    convention as `assert_parity`.
    """

    diff = _value_diff(left.float(), right.float())
    assert diff == 0.0, f"{name}: diff={diff}"
    SCENARIOS[name] = 0.0
    print(f"PASS {name}")


def scenario_fast_gelu():
    x = torch.randn(4, 128, 256, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    assert_parity("fast_gelu_custom",
                  ct.npu_fast_gelu_custom(x), _launcher.npu_fast_gelu_custom(x))
    grad = torch.randn_like(x)
    assert_parity("fast_gelu_custom_backward",
                  ct.npu_fast_gelu_custom_backward(grad, x),
                  _launcher.npu_fast_gelu_custom_backward(grad, x))


def scenario_recurrent_gated_delta_rule():
    batch = 8
    num_key_heads, num_value_heads, dim = 8, 16, 128
    gap, offset = 16384, 12288
    inner = num_value_heads * dim * dim
    block_stride = inner + gap

    def make_state():
        backing = torch.empty(batch * block_stride * 4, dtype=torch.int8,
                              device="npu")
        typed = backing.view(torch.float32)
        state = torch.as_strided(
            typed,
            size=(batch, num_value_heads, dim, dim),
            stride=(block_stride, dim * dim, dim, 1),
            storage_offset=offset,
        )
        state.zero_()
        return state

    def norm(t):
        return torch.nn.functional.normalize(t, p=2, dim=-1)

    query = norm(torch.randn(batch, num_key_heads, dim, device="npu")).to(
        torch.bfloat16)
    key = norm(torch.randn(batch, num_key_heads, dim, device="npu")).to(
        torch.bfloat16)
    value = torch.randn(batch, num_value_heads, dim, dtype=torch.bfloat16,
                        device="npu")
    beta = torch.rand(batch, num_value_heads, dtype=torch.bfloat16,
                      device="npu")
    g = torch.rand(batch, num_value_heads, dtype=torch.float32, device="npu")
    actual_seq_lengths = torch.tensor([0] + [1] * batch, dtype=torch.int32,
                                      device="npu")
    ssm_state_indices = torch.arange(batch, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    state_c = make_state()
    state_t = make_state()
    kw = dict(beta=beta, g=g, scale=dim ** -0.5,
              actual_seq_lengths=actual_seq_lengths,
              ssm_state_indices=ssm_state_indices,
              num_accepted_tokens=None)
    out_c = ct.npu_recurrent_gated_delta_rule(query, key, value, state_c, **kw)
    out_t = _launcher.npu_recurrent_gated_delta_rule(query, key, value, state_t,
                                                 **kw)
    torch.npu.synchronize()
    assert_parity("recurrent_gated_delta_rule", out_c, out_t)
    record_extra("recurrent_gated_delta_rule(state)", state_c, state_t)


def scenario_recompute():
    B, Hk, Hv, T, K, V, cs = 1, 2, 4, 256, 128, 256, 64
    dt = torch.float16
    k = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    v = torch.randn(B, Hv, T, V, dtype=dt, device="npu")
    beta = torch.randn(B, Hv, T, dtype=dt, device="npu")
    A = torch.randn(B, Hv, T, cs, dtype=dt, device="npu")
    g = torch.randn(B, Hv, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(g=g, gk=None, cu_seqlens=None, chunk_indices=None)
    assert_parity("recompute_w_u_fwd",
                  ct.npu_recompute_w_u_fwd(k, v, beta, A, cs, **kw),
                  _launcher.npu_recompute_w_u_fwd(k, v, beta, A, cs, **kw))


def scenario_pwy_full():
    B, H, T, K, V, cs = 1, 4, 256, 128, 128, 64
    dt = torch.float16
    k = torch.rand(B, H, T, K, dtype=dt, device="npu")
    v = torch.rand(B, H, T, V, dtype=dt, device="npu")
    beta = torch.rand(B, H, T, dtype=dt, device="npu")
    A = torch.rand(B, H, T, cs, dtype=dt, device="npu")
    dA = torch.rand(B, H, T, cs, dtype=dt, device="npu")
    dw = torch.rand(B, H, T, K, dtype=dt, device="npu")
    du = torch.rand(B, H, T, V, dtype=dt, device="npu")
    g = torch.rand(B, H, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(cu_seqlens=None, chunk_indices=None)
    assert_parity("prepare_wy_repr_bwd_full",
                  ct.npu_prepare_wy_repr_bwd_full(
                      k, v, beta, A, dA, dw, du, g, cs, **kw),
                  _launcher.npu_prepare_wy_repr_bwd_full(
                      k, v, beta, A, dA, dw, du, g, cs, **kw))


def scenario_pwy():
    B, HK, HV, T, K, V, cs = 1, 4, 8, 256, 128, 128, 64
    dt = torch.bfloat16
    k = torch.rand(B, HK, T, K, dtype=dt, device="npu")
    v = torch.rand(B, HV, T, V, dtype=dt, device="npu")
    beta = torch.rand(B, HV, T, dtype=torch.float32, device="npu")
    A = torch.rand(B, HV, T, cs, dtype=dt, device="npu")
    dw = torch.rand(B, HV, T, K, dtype=dt, device="npu")
    du = torch.rand(B, HV, T, V, dtype=dt, device="npu")
    g = torch.rand(B, HV, T, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kw = dict(chunk_size=cs, cu_seqlens=None, chunk_indices=None)
    assert_parity("prepare_wy_repr_bwd",
                  ct.npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, **kw),
                  _launcher.npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, **kw))


def scenario_dv_local():
    B, Hqk, Hdo, T, K, V, cs = 1, 2, 4, 256, 128, 128, 64
    dt = torch.float16
    q = torch.randn(B, Hqk, T, K, dtype=dt, device="npu")
    k = torch.randn(B, Hqk, T, K, dtype=dt, device="npu")
    d_o = torch.randn(B, Hdo, T, V, dtype=dt, device="npu")
    g = torch.randn(B, Hdo, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(scale=0.0625, chunk_size=cs, g_gamma=None, A=None,
              cu_seqlens=None, chunk_indices=None)
    assert_parity("chunk_bwd_dv_local",
                  ct.npu_chunk_bwd_dv_local(q, k, d_o, g, **kw),
                  _launcher.npu_chunk_bwd_dv_local(q, k, d_o, g, **kw))


def scenario_pwy_da():
    B, H, T, K, V, cs = 1, 4, 256, 128, 128, 64
    dt = torch.float16
    k = torch.rand(B, H, T, K, dtype=dt, device="npu")
    v = torch.rand(B, H, T, V, dtype=dt, device="npu")
    beta = torch.rand(B, H, T, dtype=dt, device="npu")
    A = torch.rand(B, H, T, cs, dtype=dt, device="npu")
    dw = torch.rand(B, H, T, K, dtype=dt, device="npu")
    du = torch.rand(B, H, T, V, dtype=dt, device="npu")
    g = torch.rand(B, H, T, dtype=dt, device="npu")
    torch.npu.synchronize()
    kw = dict(chunk_size=cs, cu_seqlens=None, chunk_indices=None)
    assert_parity("prepare_wy_repr_bwd_da",
                  ct.npu_prepare_wy_repr_bwd_da(
                      k, v, beta, A, dw, du, g, **kw),
                  _launcher.npu_prepare_wy_repr_bwd_da(
                      k, v, beta, A, dw, du, g, **kw))


def _fwd_h_inputs(B, Hk, Hv, T, K, V, dt=torch.bfloat16):
    k = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    w = torch.randn(B, Hv, T, K, dtype=dt, device="npu")
    u = torch.randn(B, Hv, T, V, dtype=dt, device="npu")
    g = -torch.rand(B, Hv, T, dtype=dt, device="npu") * 5 - 1e-3
    return k, w, u, g


def scenario_gated_fwd_h():
    B, Hk, Hv, T, K, V, cs = 1, 2, 2, 256, 128, 128, 64
    k, w, u, g = _fwd_h_inputs(B, Hk, Hv, T, K, V)
    torch.npu.synchronize()
    assert_parity(
        "chunk_gated_delta_rule_fwd_h(dense)",
        ct.npu_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size=cs),
        _launcher.npu_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size=cs))
    is0 = torch.randn(B, Hv, K, V, dtype=torch.float32, device="npu")
    assert_parity(
        "chunk_gated_delta_rule_fwd_h(final)",
        ct.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=is0, output_final_state=True,
            chunk_size=cs),
        _launcher.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=is0, output_final_state=True,
            chunk_size=cs))
    # state_v_first is the other declared flag of this operator (it swaps the
    # state's last two axes).
    isv = torch.randn(B, Hv, V, K, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    parity_or_domain_skip(
        "chunk_gated_delta_rule_fwd_h(state_v_first)",
        lambda: ct.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=isv, output_final_state=True,
            chunk_size=cs, state_v_first=True),
        lambda: _launcher.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=isv, output_final_state=True,
            chunk_size=cs, state_v_first=True))


def scenario_chunk_fwd_h():
    B, Hk, Hv, T, K, V, cs = 1, 2, 2, 256, 128, 128, 64
    k, w, u, g = _fwd_h_inputs(B, Hk, Hv, T, K, V)
    torch.npu.synchronize()
    assert_parity("chunk_fwd_h",
                  ct.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs),
                  _launcher.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs))
    # The declared flags of this operator: each one switches a different kernel
    # path, so they are covered rather than left to the default.
    for label, kw in (("final_state", dict(output_final_state=True)),
                      ("save_new_value_false", dict(save_new_value=False)),
                      ("state_v_first", dict(state_v_first=True)),
                      ("use_exp2", dict(use_exp2=True))):
        torch.npu.synchronize()
        parity_or_domain_skip(
            f"chunk_fwd_h({label})",
            lambda kw=kw: ct.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs, **kw),
            lambda kw=kw: _launcher.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs,
                                                **kw))


def scenario_chunk_fwd_o():
    B, Hk, Hv, T, K, V, cs = 1, 2, 2, 256, 128, 128, 64
    dt = torch.bfloat16
    q = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    k = torch.randn(B, Hk, T, K, dtype=dt, device="npu")
    h = torch.randn(B, Hv, T // cs, K, V, dtype=dt, device="npu")
    v = torch.randn(B, Hv, T, V, dtype=dt, device="npu")
    g = torch.randn(B, Hv, T, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    scale = 0.08838834764831845
    assert_parity(
        "chunk_fwd_o(BNSD)",
        ct.npu_chunk_fwd_o(q, k, v, h, scale, g=g, chunk_size=cs,
                           output_layout="BNSD"),
        _launcher.npu_chunk_fwd_o(q, k, v, h, scale, g=g, chunk_size=cs,
                              output_layout="BNSD"))
    # NTD is the other output layout this kernel accepts from BNSD inputs, and
    # it is where the spec's per-layout alloc mapping would go wrong silently.
    torch.npu.synchronize()
    assert_parity(
        "chunk_fwd_o(NTD)",
        ct.npu_chunk_fwd_o(q, k, v, h, scale, g=g, chunk_size=cs,
                           output_layout="NTD"),
        _launcher.npu_chunk_fwd_o(q, k, v, h, scale, g=g, chunk_size=cs,
                              output_layout="NTD"))
    # The rest of the declared domain is rejected by this OPP's kernel when
    # driven with BNSD inputs (BSND/TND expect the inputs laid out that way) --
    # recorded as a skip with the status rather than deleted.
    for label, kw in (("BSND", dict(output_layout="BSND")),
                      ("TND", dict(output_layout="TND")),
                      ("use_exp2", dict(use_exp2=True)),
                      ("transpose_state_layout",
                       dict(transpose_state_layout=True))):
        torch.npu.synchronize()
        parity_or_domain_skip(
            f"chunk_fwd_o({label})",
            lambda kw=kw: ct.npu_chunk_fwd_o(q, k, v, h, scale, g=g,
                                             chunk_size=cs, **kw),
            lambda kw=kw: _launcher.npu_chunk_fwd_o(q, k, v, h, scale, g=g,
                                                chunk_size=cs, **kw))


def scenario_bwd_dhu():
    H, T, K, V, cs = 4, 256, 128, 128, 64
    dt = torch.float16
    cu = [0, 128, 256]
    ci = [0, 0, 0, 1, 1, 0, 1, 1]
    q = torch.randn(1, H, T, K, dtype=dt, device="npu")
    k = torch.randn(1, H, T, K, dtype=dt, device="npu")
    w = torch.randn(1, H, T, K, dtype=dt, device="npu")
    do = torch.randn(1, H, T, V, dtype=dt, device="npu")
    dv = torch.randn(1, H, T, V, dtype=dt, device="npu")
    g = (-torch.sort(torch.rand(H * T, device="npu"), descending=True)[0]
         .reshape(1, H, T).to(dt))
    torch.npu.synchronize()
    kw = dict(scale=K ** -0.5, chunk_size=cs, g=g, gK=None, h0=None,
              dht=None, cu_seqlens=cu, chunk_indices=ci)
    assert_parity("chunk_gated_delta_rule_bwd_dhu",
                  ct.npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, do, dv, **kw),
                  _launcher.npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, do, dv, **kw))


def _clone_args(kwargs):
    def clone(value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    return {key: clone(value) for key, value in kwargs.items()}


def _backend_carries(name: str) -> bool:
    """Whether the selected backend exposes *name* at all.

    The OPP in a given environment does not carry every kernel, so an
    operator the backend cannot expose is recorded as a skip rather than
    counted as covered.
    """

    return callable(getattr(_launcher, name, None))


def _aclnn_status(exc):
    match = re.search(r"(?:aclnnStatus|status|failed:)\s*=?\s*(\d{4,6})",
                      str(exc))
    return match.group(1) if match else str(exc)[:120]


def _aclnn_status_or_none(exc):
    """The aclnn status in *exc*, or None when it is a Python-level error.

    The launcher validates only what is free (the dispatcher schema); the
    reference validates more in Python, so for an illegal input the two can
    disagree about *how* they fail.  Kernel-vs-kernel rejections must still
    match exactly, which is why the two cases are told apart rather than lumped
    together.
    """

    match = re.search(r"(?:aclnnStatus|failed:)\s*=?\s*(\d{4,6})", str(exc))
    return match.group(1) if match else None


def parity_or_domain_skip(name, call_ct, call_launcher):
    """Parity, or a recorded skip when the *reference* itself rejects the input.

    Some legal-looking parameter combinations are not implemented by the kernel
    build in this OPP.  When ctypes rejects them too, there is nothing to
    compare and the honest record is "both paths rejected it, with this status"
    rather than a silently deleted case.
    """

    try:
        reference = call_ct()
    except AttributeError as exc:
        # The OPP in this environment does not ship that kernel at all.
        SKIPPED[name] = f"the OPP does not carry the kernel: {exc}"
        print(f"SKIP {name} ({SKIPPED[name]})")
        return
    # The reference validates in Python and raises ValueError for some
    # operators and RuntimeError for others (`use_exp2=False is not supported`
    # is a ValueError), so both count as "the reference refused it".
    except (RuntimeError, ValueError) as exc:
        status = _aclnn_status_or_none(exc)
        try:
            call_launcher()
        except (RuntimeError, ValueError) as launcher_exc:
            launcher_status = _aclnn_status_or_none(launcher_exc)
            if status is None:
                # The reference refused it before reaching aclnn (its Python
                # validation); the launcher passed it on and the kernel refused.
                SKIPPED[name] = (
                    f"reference rejects by validation "
                    f"({str(exc).splitlines()[0][:70]}); stable rejects too "
                    f"({str(launcher_exc).splitlines()[0][:50]})")
            else:
                assert status == launcher_status, (
                    f"{name}: ctypes rejected with {status} but stable with "
                    f"{launcher_status}")
                SKIPPED[name] = f"both backends rejected the inputs: {status}"
            print(f"SKIP {name} ({SKIPPED[name]})")
            return
        raise AssertionError(
            f"{name}: ctypes rejected the inputs ({status}) but the stable "
            f"backend accepted them") from None
    launcher_result = call_launcher()
    assert_parity(name, reference, launcher_result)


def _conv1d_outcome(fn, kwargs):
    args = _clone_args(kwargs)
    try:
        out = fn(**args)
        torch.npu.synchronize()
        return "ok", out, args
    except (RuntimeError, AttributeError) as exc:
        return "err", _aclnn_status(exc), args


def _conv1d_parity(name, kwargs, mutated=("conv_states",), defined_rows=None,
                   frozen_state_slots=None):
    """Compare npu_causal_conv1d through both backends on *separate* copies.

    ``conv_states`` is written in place, so the two backends must not share it:
    each gets a private clone and both the outputs and the mutated state are
    compared.  When the kernel rejects the inputs outright, parity means the
    same rejection on both paths -- recorded as SKIP with the aclnn status
    rather than counted as coverage.

    ``defined_rows`` names the output rows the kernel is required to write.  A
    pad slot (``cache_indices == pad_slot_id``) is skipped by the kernel, so its
    output row is uninitialised memory: it is *not* part of the contract and
    comparing it across two separate allocations would only ever measure the
    allocator's leftovers.  ``frozen_state_slots`` names the state rows that
    must come back untouched (same pad story, on ``conv_states``).
    """

    if not _backend_carries("npu_causal_conv1d"):
        reason = "the selected backend does not carry npu_causal_conv1d"
        SKIPPED[name] = reason
        print(f"SKIP {name} ({reason})")
        return
    kind_ct, out_ct, args_ct = _conv1d_outcome(ct.npu_causal_conv1d, kwargs)
    kind_th, out_th, args_th = _conv1d_outcome(_launcher.npu_causal_conv1d, kwargs)
    assert kind_ct == kind_th, (
        f"{name}: ctypes={kind_ct} but stable={kind_th} "
        f"({out_ct if kind_ct == 'err' else ''})")
    if kind_ct == "err":
        assert out_ct == out_th, (
            f"{name}: aclnn status mismatch ctypes={out_ct} stable={out_th}")
        SKIPPED[name] = f"both backends rejected the inputs: {out_ct}"
        print(f"SKIP {name} (both backends rejected the inputs: {out_ct})")
        return
    if defined_rows is None:
        assert_parity(name, out_ct, out_th)
    else:
        assert_parity(f"{name}[defined rows]", out_ct[defined_rows],
                      out_th[defined_rows])
    for flag in mutated:
        if args_ct.get(flag) is None:
            continue
        if frozen_state_slots is None:
            left, right = args_ct[flag], args_th[flag]
        else:
            keep = [index for index in range(args_ct[flag].shape[0])
                    if index not in frozen_state_slots]
            left, right = args_ct[flag][keep], args_th[flag][keep]
        record_extra(f"{name}({flag})", left, right)
        if frozen_state_slots is not None:
            # Untouched is the contract for a pad slot, not merely "equal on
            # both paths": compare against the value handed in.
            original = kwargs[flag]
            frozen = [index for index in range(original.shape[0])
                      if index in frozen_state_slots]
            record_extra(f"{name}(pad state slots untouched)",
                         args_ct[flag][frozen], original[frozen])


def _seq(n, start):
    return (torch.arange(n).float() + start)


def scenario_conv1d_prefill():
    """Reference: test_npu_causal_conv1d_prefill_* (run_mode=0)."""

    for dim, head_num in ((16, 0), (32, 2)):
        x = (_seq(2 * 4 * dim, 1.0).reshape(2, 4, dim)).to(torch.bfloat16).npu()
        weight = (_seq(4 * dim, 101.0).reshape(4, dim)).to(torch.bfloat16).npu()
        bias = _seq(dim, 201.0).to(torch.bfloat16).npu()
        states = _seq(2 * 3 * dim, 301.0).reshape(2, 3, dim).to(torch.bfloat16).npu()
        _conv1d_parity(f"conv1d_prefill(dim={dim},head_num={head_num})", dict(
            x=x, weight=weight, bias=bias, conv_states=states,
            activation_mode=1, pad_slot_id=-1, run_mode=0, head_num=head_num))


def scenario_conv1d_varlen_initial_state():
    """Reference: test_npu_causal_conv1d_varlen_initial_state_* (run_mode=0).

    A2 rejects the varlen form outright (aclnnCausalConv1dGetWorkspaceSize
    561002) on both backends; the helper records that as SKIP with the status
    instead of pretending the case is covered.

    The cache indices avoid block 0: that is `null_block_id`, and a sequence
    addressing it is skipped by the kernel (see scenario_conv1d_new_apis).
    """

    x = _seq(5 * 16, 1.0).to(torch.bfloat16).npu()
    weight = _seq(4 * 16, 101.0).reshape(4, 16).to(torch.bfloat16).npu()
    states = _seq(3 * 3 * 16, 301.0).reshape(3, 3, 16).to(torch.bfloat16).npu()
    _conv1d_parity("conv1d_varlen_initial_state", dict(
        x=x, weight=weight, bias=None, conv_states=states,
        query_start_loc=[0, 2, 5], cache_indices=[1, 2],
        initial_state_mode=[1, 0], activation_mode=0, pad_slot_id=-1,
        run_mode=0))


def scenario_conv1d_update():
    """Reference: test_npu_causal_conv1d_update_* and _spec_decode_*.

    Every case keeps block 0 out of `cache_indices`: that id is `null_block_id`
    (the padding slot), and a sequence addressing it is skipped by the kernel,
    so comparing that row would compare uninitialised memory.
    """

    cases = [
        ("update", dict(x=_seq(2 * 16, 1.0).reshape(2, 16).to(torch.bfloat16).npu(),
                        weight=_seq(4 * 16, 101.0).reshape(4, 16).to(torch.bfloat16).npu(),
                        bias=_seq(16, 201.0).to(torch.bfloat16).npu(),
                        conv_states=_seq(3 * 3 * 16, 301.0).reshape(3, 3, 16).to(torch.bfloat16).npu(),
                        cache_indices=[1, 2], activation_mode=1,
                        pad_slot_id=-1, run_mode=1)),
        ("spec_decode", dict(x=_seq(2 * 4 * 16, 1.0).reshape(2, 4, 16).to(torch.bfloat16).npu(),
                             weight=_seq(4 * 16, 101.0).reshape(4, 16).to(torch.bfloat16).npu(),
                             bias=_seq(16, 201.0).to(torch.bfloat16).npu(),
                             conv_states=_seq(3 * 6 * 16, 301.0).reshape(3, 6, 16).to(torch.bfloat16).npu(),
                             cache_indices=[1, 2], num_accepted_tokens=[2, 4],
                             activation_mode=0, pad_slot_id=-1, run_mode=1)),
        ("width3_no_bias", dict(x=_seq(3 * 16, 1.0).reshape(3, 16).to(torch.bfloat16).npu(),
                                weight=_seq(3 * 16, 101.0).reshape(3, 16).to(torch.bfloat16).npu(),
                                bias=None,
                                conv_states=_seq(4 * 2 * 16, 301.0).reshape(4, 2, 16).to(torch.bfloat16).npu(),
                                cache_indices=[1, 2, 3], activation_mode=0,
                                pad_slot_id=-1, run_mode=1)),
    ]
    for label, kwargs in cases:
        _conv1d_parity(f"conv1d_{label}", kwargs)


def scenario_conv1d_update_offset_state():
    """Update where ``conv_states`` is a contiguous view with a storage offset.

    The descriptor carries this view's own strides and storage offset, so a
    state handed over as storage-base + offset is read and written at the rows
    the caller keeps; a state that fell back to dense addressing from the
    storage base would land somewhere else entirely
    (``scenario_conv1d_update_paged_state`` is the block-strided variant of the
    same check).  The states must not be cloned on the way in, because a clone
    is dense and would hide the very thing being tested.
    """

    lines, state_len, dim, gap = 4, 3, 16, 96

    def make_case():
        backing = torch.zeros(lines * state_len * dim + gap,
                              dtype=torch.bfloat16, device="npu")
        view = backing[gap:].view(lines, state_len, dim)
        view.copy_(_seq(lines * state_len * dim, 301.0)
                   .to(torch.bfloat16).npu().view(lines, state_len, dim))
        return view

    x = _seq(2 * 16, 1.0).reshape(2, 16).to(torch.bfloat16).npu()
    weight = _seq(4 * 16, 101.0).reshape(4, 16).to(torch.bfloat16).npu()
    bias = _seq(16, 201.0).to(torch.bfloat16).npu()
    kw = dict(weight=weight, bias=bias, cache_indices=[1, 2],
              activation_mode=0, pad_slot_id=-1, run_mode=1)
    state_ct, state_th = make_case(), make_case()
    assert state_ct.is_contiguous() and int(state_ct.storage_offset()) == gap, (
        "the case has to hand over a contiguous view with a storage offset")
    out_ct = ct.npu_causal_conv1d(x.clone(), conv_states=state_ct, **kw)
    out_th = _launcher.npu_causal_conv1d(x.clone(), conv_states=state_th, **kw)
    torch.npu.synchronize()
    assert_parity("conv1d_update_offset_state", out_ct, out_th)
    record_extra("conv1d_update_offset_state(conv_states)", state_ct, state_th)


def _paged_conv_state(lines, state_len, dim, gap, initial):
    """A block-strided view over one flat buffer: what a paged cache looks like.

    Block ``i`` starts at ``i * (state_len * dim + gap)``, so the dense strides
    the operator falls back to when the runtime drops the view description
    address a different row for every block but the first.
    """

    block_stride = state_len * dim + gap
    backing = torch.zeros(lines * block_stride, dtype=torch.bfloat16, device="npu")
    view = torch.as_strided(backing, (lines, state_len, dim),
                            (block_stride, dim, 1), 0)
    view.copy_(initial.view(lines, state_len, dim))
    return view


def _conv1d_paged_state_parity(name, kwargs, state_ct, state_th, dense):
    """Both backends on the paged state, plus a dense reference.

    The dense reference is what a backend-to-backend comparison cannot catch:
    when a runtime drops the view description, both backends fall back to the
    same dense addressing and agree with each other while disagreeing with the
    rows the caller actually keeps in the cache.
    """

    def run(op, state):
        return op(kwargs["x"], conv_states=state,
                  **{k: v for k, v in kwargs.items() if k != "x"})

    out_ref = run(ct.npu_causal_conv1d, dense)
    out_ct = run(ct.npu_causal_conv1d, state_ct)
    torch.npu.synchronize()
    assert_parity(f"{name}[dense reference]", out_ref, out_ct)
    record_extra(f"{name}[dense reference state]", dense, state_ct)
    if not _backend_carries("npu_causal_conv1d"):
        SKIPPED[name] = "the selected backend does not carry npu_causal_conv1d"
        print(f"SKIP {name} (the selected backend does not carry npu_causal_conv1d)")
        return
    out_th = run(_launcher.npu_causal_conv1d, state_th)
    torch.npu.synchronize()
    assert_parity(name, out_ct, out_th)
    record_extra(f"{name}(conv_states)", state_ct, state_th)


def scenario_conv1d_update_paged_state():
    """Update against a paged (block-strided) conv cache.

    A serving cache is one flat buffer with thousands of blocks and only a few
    of them live, so the state is never dense.  This is the shape that used to
    be wrong: the tiling asked an optional input for its strides, was answered
    with nothing on a runtime that drops the view description, and addressed
    dense rows -- the state write-back landed in the gaps and the caller kept
    reading its stale rows.  It is still wrong to hand such a state over on such
    a runtime, which is why the adapter stages it there
    (``_runtime.conv1d_view_state_supported``).
    """

    lines, state_len, dim, gap = 5, 3, 16, 48
    initial = _seq(lines * state_len * dim, 301.0).to(torch.bfloat16).npu()
    state_ct = _paged_conv_state(lines, state_len, dim, gap, initial)
    state_th = _paged_conv_state(lines, state_len, dim, gap, initial)
    dense = _paged_conv_state(lines, state_len, dim, gap, initial).contiguous()
    assert not state_ct.is_contiguous(), "the case has to hand over a paged state"
    kwargs = dict(
        x=_seq(2 * dim, 1.0).reshape(2, dim).to(torch.bfloat16).npu(),
        weight=_seq(4 * dim, 101.0).reshape(4, dim).to(torch.bfloat16).npu(),
        bias=_seq(dim, 201.0).to(torch.bfloat16).npu(),
        cache_indices=[1, 2],
        activation_mode=0,
        pad_slot_id=-1,
        run_mode=1,
    )
    _conv1d_paged_state_parity("conv1d_update_paged_state", kwargs,
                               state_ct, state_th, dense)


def scenario_conv1d_prefill_paged_state():
    """Prefill (``run_mode=0``) against the same paged cache.

    The forward path reads its history out of ``conv_states`` and writes the
    new one back, so a stride mix-up corrupts the cache here too.
    """

    lines, state_len, dim, gap = 5, 3, 16, 48
    initial = _seq(lines * state_len * dim, 401.0).to(torch.bfloat16).npu()
    state_ct = _paged_conv_state(lines, state_len, dim, gap, initial)
    state_th = _paged_conv_state(lines, state_len, dim, gap, initial)
    dense = _paged_conv_state(lines, state_len, dim, gap, initial).contiguous()
    kwargs = dict(
        x=_seq(2 * 2 * dim, 1.0).reshape(2, 2, dim).to(torch.bfloat16).npu(),
        weight=_seq(4 * dim, 101.0).reshape(4, dim).to(torch.bfloat16).npu(),
        bias=None,
        query_start_loc=[0, 2, 4],
        cache_indices=[1, 2],
        initial_state_mode=[1, 1],
        activation_mode=0,
        pad_slot_id=-1,
        run_mode=0,
    )
    _conv1d_paged_state_parity("conv1d_prefill_paged_state", kwargs,
                               state_ct, state_th, dense)


def scenario_conv1d_gather_padding():
    """Reference: test_npu_causal_conv1d_update_with_batch_gather_padding_*.

    ``cache_indices`` carries pad_slot_id for the rows the kernel must skip:
    rows 3-4 of the output are uninitialised by contract, and state slots 0, 2,
    4 and 6 must come back untouched (the unit test checks the same two
    things).  Only the defined part is compared.
    """

    x = _seq(5 * 3 * 16, 1.0).reshape(5, 3, 16).to(torch.bfloat16).npu()
    weight = _seq(4 * 16, 101.0).reshape(4, 16).to(torch.bfloat16).npu()
    states = _seq(7 * 3 * 16, 301.0).reshape(7, 3, 16).to(torch.bfloat16).npu()
    _conv1d_parity(
        "conv1d_update_gather_padding",
        dict(x=x, weight=weight,
             bias=_seq(16, 201.0).to(torch.bfloat16).npu(),
             conv_states=states, cache_indices=[1, 3, 5, -1, -1],
             activation_mode=1, pad_slot_id=-1, run_mode=1),
        defined_rows=[0, 1, 2],
        frozen_state_slots={0, 2, 4, 6})


def scenario_conv1d_varlen_pad_slot():
    """Reference: test_npu_causal_conv1d_varlen_pad_slot_matches_valid_segments.

    A whole sequence is a pad slot (cache_indices == pad_slot_id), which the
    kernel must skip without touching its state row.
    """

    x = _seq(7 * 16, 1.0).to(torch.bfloat16).npu()
    weight = _seq(4 * 16, 101.0).reshape(4, 16).to(torch.bfloat16).npu()
    states = _seq(2 * 3 * 16, 301.0).reshape(2, 3, 16).to(torch.bfloat16).npu()
    # Sequence 1 (tokens 2-3) is a pad slot: its output rows are uninitialised,
    # so only the tokens of sequences 0 (0-1) and 2 (4-6) are compared.
    _conv1d_parity(
        "conv1d_varlen_pad_slot",
        dict(x=x, weight=weight,
             bias=_seq(16, 201.0).to(torch.bfloat16).npu(),
             conv_states=states, query_start_loc=[0, 2, 4, 7],
             cache_indices=[0, -1, 1], initial_state_mode=[1, 0, 1],
             activation_mode=0, pad_slot_id=-1, run_mode=0),
        defined_rows=[0, 1, 4, 5, 6])


def scenario_conv1d_new_apis():
    """Upstream #390's replacement APIs, through the shared launcher.

    ``causal_conv1d_fn`` and ``causal_conv1d_update`` are what the integration
    is told to use from now on (the legacy ``npu_causal_conv1d`` is deprecated),
    and they are the ones that take *device* metadata (query_start_loc,
    cache_indices, has_initial_state) instead of host lists -- the shape the
    vLLM call site has.  Both go through the same aclnn ABI, so they exercise
    the same launch path with different marshalling above it.
    """

    if not _backend_carries("npu_causal_conv1d_fn"):
        reason = "the selected backend does not carry causal_conv1d_fn/_update"
        for name in ("causal_conv1d_fn(dense)", "causal_conv1d_fn(varlen)",
                     "causal_conv1d_update(dense)"):
            SKIPPED[name] = reason
            print(f"SKIP {name} ({reason})")
        return

    dt = torch.bfloat16
    # fn, dense batch: 3-D (B, S, D) x needs no query_start_loc.
    x = _seq(2 * 2 * 16, 1.0).reshape(2, 2, 16).to(dt).npu()
    weight = _seq(4 * 16, 101.0).reshape(4, 16).to(dt).npu()
    bias = _seq(16, 201.0).to(dt).npu()
    parity_or_domain_skip(
        "causal_conv1d_fn(dense)",
        lambda: _fn_case(ct, x, weight, bias),
        lambda: _fn_case(_launcher, x, weight, bias))
    # fn, varlen with device metadata (the vLLM-style call).
    xv = _seq(5 * 16, 1.0).reshape(5, 16).to(dt).npu()
    qsl = torch.tensor([0, 2, 5], dtype=torch.int32, device="npu")
    # Block ids must not collide with `null_block_id` (0 by default): a sequence
    # whose cache index is the null block is *skipped* by the kernel, which
    # leaves its output rows unwritten -- whatever the allocator had there is
    # then read back, so such a case is not a parity test at all (measured: the
    # same ctypes call twice differs by 9.5e6 in those rows).  Block 0 is
    # reserved in vLLM's paged cache for the same reason.
    cache = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    initial = torch.tensor([True, False], dtype=torch.bool, device="npu")
    parity_or_domain_skip(
        "causal_conv1d_fn(varlen, initial state)",
        lambda: _fn_case(ct, xv, weight, None, qsl=qsl, cache=cache,
                         initial=initial),
        lambda: _fn_case(_launcher, xv, weight, None, qsl=qsl, cache=cache,
                         initial=initial))
    # update: in-place on conv_state, returns the mutated x.
    parity_or_domain_skip(
        "causal_conv1d_update(dense)",
        lambda: _update_case(ct, dt),
        lambda: _update_case(_launcher, dt))
    # update with a caller-provided destination: the operator writes into it,
    # where the reference allocates its own output and copies.  Both have to
    # report the same values.
    parity_or_domain_skip(
        "causal_conv1d_update(out=)",
        lambda: _update_case(ct, dt, use_out=True),
        lambda: _update_case(_launcher, dt, use_out=True))


def _fn_case(backend, x, weight, bias, qsl=None, cache=None, initial=None):
    dt = x.dtype
    # Three blocks, so the two sequences can use ids 1 and 2 and leave the null
    # block (0) alone.
    states = _seq(3 * 3 * 16, 301.0).reshape(3, 3, 16).to(dt).npu()
    kwargs = {}
    if qsl is not None:
        kwargs = dict(query_start_loc=qsl, cache_indices=cache,
                      has_initial_state=initial)
    return backend.npu_causal_conv1d_fn(x, weight, bias, states,
                                        activation="silu", **kwargs)


def _update_case(backend, dt, use_out=False):
    x = _seq(2 * 16, 1.0).reshape(2, 16).to(dt).npu()
    weight = _seq(4 * 16, 101.0).reshape(4, 16).to(dt).npu()
    bias = _seq(16, 201.0).to(dt).npu()
    states = _seq(3 * 3 * 16, 301.0).reshape(3, 3, 16).to(dt).npu()
    # See the varlen case: block id 0 is the null block, and a sequence that
    # addresses it is skipped (its output row is never written).
    indices = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    destination = torch.zeros_like(x) if use_out else None
    out = backend.npu_causal_conv1d_update(
        x, states, weight, bias, activation="silu",
        conv_state_indices=indices, out=destination)
    torch.npu.synchronize()
    return out, states


def scenario_conv1d_bwd_bnsd():
    batch, num_heads, seqlen, head_dim, width = 2, 2, 9, 16, 2
    dim = num_heads * head_dim
    dt = torch.bfloat16
    x = (torch.arange(batch * seqlen * dim).reshape(batch, seqlen, dim).float()
         + 11).to(dt).npu()
    weight = (torch.arange(width * dim).reshape(width, dim).float()
              + 111).to(dt).npu()
    dy = (torch.arange(batch * seqlen * dim).reshape(batch, seqlen, dim).float()
          + 211).to(dt).npu()
    st = (torch.arange(batch * width * dim).reshape(batch, width, dim).float()
          + 311).to(dt).npu()
    dht = (torch.arange(batch * width * dim).reshape(batch, width, dim).float()
           + 411).to(dt).npu()
    ylog = torch.zeros_like(x)
    for i in range(width):
        if i == 0:
            ylog += x * weight[width - 1 - i].view(1, 1, -1)
        else:
            ylog[:, i:, :] += x[:, :-i, :] * weight[width - 1 - i].view(1, 1, -1)
    yb = (ylog.reshape(batch, seqlen, num_heads, head_dim)
          .permute(0, 2, 1, 3).contiguous())
    dyb = (dy.reshape(batch, seqlen, num_heads, head_dim)
           .permute(0, 2, 1, 3).contiguous())
    torch.npu.synchronize()
    kw = dict(x=x, y=yb, weight=weight, dy=dyb, initial_state=st, dht=dht,
              activation=2, input_layout="BNSD")
    parity_or_domain_skip("causal_conv1d_bwd(BNSD)",
                          lambda: ct.npu_causal_conv1d_bwd(**kw),
                          lambda: _launcher.npu_causal_conv1d_bwd(**kw))
    # The other declared input_layouts: BSND keeps x dim-last and moves y/dy to
    # (B,S,H,D); TND/NTD are the varlen spellings and need query_start_loc.
    ys = (ylog.reshape(batch, seqlen, num_heads, head_dim).contiguous())
    dys = dy.reshape(batch, seqlen, num_heads, head_dim).contiguous()
    parity_or_domain_skip(
        "causal_conv1d_bwd(BSND)",
        lambda: ct.npu_causal_conv1d_bwd(**dict(kw, y=ys, dy=dys,
                                                input_layout="BSND")),
        lambda: _launcher.npu_causal_conv1d_bwd(**dict(kw, y=ys, dy=dys,
                                                   input_layout="BSND")))
    flat = torch.arange(batch * seqlen * dim).reshape(1, batch * seqlen, dim)
    yflat = ylog.reshape(batch * seqlen, num_heads, head_dim)
    dyflat = dy.reshape(batch * seqlen, num_heads, head_dim)
    for layout in ("TND", "NTD"):
        xl = flat.clone().to(dt).npu()
        parity_or_domain_skip(
            f"causal_conv1d_bwd({layout})",
            lambda xl=xl, layout=layout: ct.npu_causal_conv1d_bwd(
                xl, yflat, weight, dyflat, initial_state=st, dht=dht,
                query_start_loc=[0, batch * seqlen], activation=2,
                input_layout=layout),
            lambda xl=xl, layout=layout: _launcher.npu_causal_conv1d_bwd(
                xl, yflat, weight, dyflat, initial_state=st, dht=dht,
                query_start_loc=[0, batch * seqlen], activation=2,
                input_layout=layout))


def _kda_fwd_tensors(layout, dt, *, B=1, T=128, H=4, HV=4, K=128, V=128):
    """Build layout-native q/k/v/g/beta for npu_chunk_kda_fwd."""

    def rnd(*shape, dtype=dt, scale=5e-2):
        return torch.randn(*shape, dtype=dtype, device="npu") * scale

    if layout == "TND":
        q, k = rnd(T, H, K), rnd(T, H, K)
        v = rnd(T, HV, V)
        g = rnd(T, HV, K, dtype=torch.float32, scale=1.0)
        beta = rnd(T, HV)
    elif layout == "NTD":
        q, k = rnd(H, T, K), rnd(H, T, K)
        v = rnd(HV, T, V)
        g = rnd(HV, T, K, dtype=torch.float32, scale=1.0)
        beta = rnd(HV, T)
    elif layout == "BSND":
        q, k = rnd(B, T, H, K), rnd(B, T, H, K)
        v = rnd(B, T, HV, V)
        g = rnd(B, T, HV, K, dtype=torch.float32, scale=1.0)
        beta = rnd(B, T, HV)
    else:  # BNSD
        q, k = rnd(B, H, T, K), rnd(B, H, T, K)
        v = rnd(B, HV, T, V)
        g = rnd(B, HV, T, K, dtype=torch.float32, scale=1.0)
        beta = rnd(B, HV, T)
    return q, k, v, g, beta


def scenario_chunk_kda_fwd():
    """kda_fwd 全域名（#491）：4 layout x dense/varlen x flag 矩阵 parity。

    合法域由 ctypes 参考实现界定；stable 只有在每个组合的逐输出 diff 都为 0、
    且 None 掩码与返回元组顺序都一致时才算覆盖（合法域记录见
    tools/stable_coverage.py 与 tools/stable_ctypes_fallbacks.py）。
    """
    H, HV, K, V = 4, 4, 128, 128
    layouts = ("BSND", "BNSD", "TND", "NTD")
    flags = ("output_final_state", "disable_recompute",
             "return_intermediate_states", "use_gate_in_kernel")
    combos = [dict(zip(flags, c))
              for c in itertools.product((False, True), repeat=len(flags))]
    total = 0
    for layout in layouts:
        for varlen in (False, True):
            for combo in combos:
                q, k, v, g, beta = _kda_fwd_tensors(layout, torch.bfloat16)
                cu = [0, 64, 128] if varlen else None
                seq_num = len(cu) - 1 if cu else 1
                svfs = ((False, True) if combo["output_final_state"]
                        else (False,))
                for svf in svfs:
                    kw = dict(layout=layout, chunk_size=64, scale=K ** -0.5,
                              cu_seqlens=cu, state_v_first=svf,
                              output_final_state=combo["output_final_state"],
                              disable_recompute=combo["disable_recompute"],
                              return_intermediate_states=combo[
                                  "return_intermediate_states"],
                              use_gate_in_kernel=combo["use_gate_in_kernel"])
                    if combo["output_final_state"]:
                        # K == V here, so state_v_first only reorders equal dims.
                        tail = (HV, V, K) if svf else (HV, K, V)
                        kw["initial_state"] = (
                            torch.randn((seq_num,) + tail, dtype=torch.float32,
                                        device="npu") * 1e-2)
                    if combo["use_gate_in_kernel"]:
                        kw["A_log"] = (
                            torch.randn(HV, dtype=torch.float32, device="npu")
                            * 0.1)
                        kw["dt_bias"] = (
                            torch.randn(HV * K, dtype=torch.float32,
                                        device="npu") * 0.5 - 3.0)
                        kw["safe_gate"] = True
                        kw["lower_bound"] = -1.0
                    tag = (f"chunk_kda_fwd({layout} varlen={int(varlen)} "
                           f"svf={int(svf)} out="
                           f"{int(combo['output_final_state'])} dis="
                           f"{int(combo['disable_recompute'])} ret="
                           f"{int(combo['return_intermediate_states'])} use="
                           f"{int(combo['use_gate_in_kernel'])})")
                    torch.npu.synchronize()
                    assert_parity(tag,
                                  ct.npu_chunk_kda_fwd(q, k, v, g, beta, **kw),
                                  _launcher.npu_chunk_kda_fwd(q, k, v, g, beta,
                                                          **kw))
                    total += 1
    print(f"PASS chunk_kda_fwd full-domain matrix ({total} combinations)")


def scenario_chunk_kda_fwd_variants():
    """kda_fwd shape/dtype 变体：V=256、B>1、GVA、bf16 g/beta、chunk128。"""
    variants = [
        # layout, B, T, H, HV, K, V, chunk, g_dtype, flags
        ("BSND", 1, 128, 4, 4, 128, 256, 64, torch.float32, {}),
        ("BSND", 2, 128, 4, 4, 128, 128, 64, torch.float32, {}),
        ("BSND", 2, 128, 4, 4, 128, 128, 64, torch.bfloat16, {}),
        ("BSND", 1, 128, 2, 4, 128, 128, 64, torch.float32, {}),
        ("BNSD", 1, 128, 2, 8, 128, 256, 128, torch.float32, {}),
        ("TND", 1, 128, 2, 4, 128, 128, 128, torch.float32, {}),
        ("NTD", 1, 128, 4, 4, 128, 256, 64, torch.float32, {}),
        ("BSND", 1, 256, 4, 4, 128, 128, 128, torch.float32,
         dict(output_final_state=True, disable_recompute=True,
              return_intermediate_states=True)),
        ("TND", 1, 192, 4, 4, 128, 128, 64, torch.float32,
         dict(output_final_state=True)),
    ]
    for (layout, B, T, H, HV, K, V, cs, gdt, extra) in variants:
        q, k, v, g, beta = _kda_fwd_tensors(
            layout, torch.bfloat16, B=B, T=T, H=H, HV=HV, K=K, V=V)
        if gdt is not torch.float32:
            g = g.to(dtype=gdt)
            beta = beta.to(dtype=gdt)
        kw = dict(layout=layout, chunk_size=cs, scale=K ** -0.5, **extra)
        if extra.get("output_final_state"):
            seq_num = B
            kw["initial_state"] = (
                torch.randn(seq_num, HV, K, V, dtype=torch.float32,
                            device="npu") * 1e-2)
        torch.npu.synchronize()
        # The T=256 / cs=128 / g=float32 variant overflows to inf on
        # Ascend950PR for some initial states -- measured: 89 entries, the same
        # 89 on both host paths, same shape.  It is input-dependent (a fresh
        # seed does not reproduce it) and identical on both paths, which is
        # what this comparison exists to check; `_value_diff` treats equal
        # infinities as equal so the gate does not fail for the wrong reason.
        assert_parity(
            f"chunk_kda_fwd(var {layout} B={B} T={T} H={H} HV={HV} "
            f"V={V} cs={cs} g={str(gdt).split('.')[-1]})",
            ct.npu_chunk_kda_fwd(q, k, v, g, beta, **kw),
            _launcher.npu_chunk_kda_fwd(q, k, v, g, beta, **kw))
    print(f"PASS chunk_kda_fwd variants ({len(variants)} cases)")


def scenario_chunk_kda_bwd_intra():
    B, H, T, K, cs = 2, 4, 256, 128, 64
    dt = torch.bfloat16
    q = torch.randn(B, H, T, K, dtype=dt, device="npu")
    k = torch.randn(B, H, T, K, dtype=dt, device="npu")
    gk = torch.randn(B, H, T, K, dtype=torch.float32, device="npu")
    beta = torch.randn(B, H, T, dtype=torch.float32, device="npu")
    dAqk = torch.randn(B, H, T, cs, dtype=torch.float32, device="npu")
    dAkk = torch.randn(B, H, T, cs, dtype=torch.float32, device="npu")
    dq = torch.randn(B, H, T, K, dtype=torch.float32, device="npu")
    dk = torch.randn(B, H, T, K, dtype=torch.float32, device="npu")
    db = torch.randn(B, H, T, dtype=torch.float32, device="npu")
    dg = torch.randn(B, H, T, K, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    kw = dict(layout="BNSD", safe_gate=True, chunk_size=cs)
    assert_parity(
        "chunk_kda_bwd_intra(BNSD dense)",
        ct.npu_chunk_kda_bwd_intra(q, k, gk, beta, dAqk, dAkk, dq, dk, db,
                                   dg, **kw),
        _launcher.npu_chunk_kda_bwd_intra(q, k, gk, beta, dAqk, dAkk, dq, dk, db,
                                      dg, **kw))
    # BSND is the operator's default layout; the tensors move the token and head
    # axes, the values do not change.
    kw_bsnd = dict(kw, layout="BSND")
    assert_parity(
        "chunk_kda_bwd_intra(BSND dense)",
        ct.npu_chunk_kda_bwd_intra(*[t.transpose(1, 2).contiguous()
                                     for t in (q, k)]
                                   + [gk.transpose(1, 2).contiguous(),
                                      beta.transpose(1, 2).contiguous(),
                                      dAqk.transpose(1, 2).contiguous(),
                                      dAkk.transpose(1, 2).contiguous()]
                                   + [t.transpose(1, 2).contiguous()
                                      for t in (dq, dk)]
                                   + [db.transpose(1, 2).contiguous(),
                                      dg.transpose(1, 2).contiguous()],
                                   **kw_bsnd),
        _launcher.npu_chunk_kda_bwd_intra(*[t.transpose(1, 2).contiguous()
                                        for t in (q, k)]
                                      + [gk.transpose(1, 2).contiguous(),
                                         beta.transpose(1, 2).contiguous(),
                                         dAqk.transpose(1, 2).contiguous(),
                                         dAkk.transpose(1, 2).contiguous()]
                                      + [t.transpose(1, 2).contiguous()
                                         for t in (dq, dk)]
                                      + [db.transpose(1, 2).contiguous(),
                                         dg.transpose(1, 2).contiguous()],
                                      **kw_bsnd))
    # TND is the packed varlen spelling: rank-3 tensors and cu_seqlens.  Both
    # sequences fit in one batch dimension, which is what the physical B=1
    # convention means here.
    # The packed tensors hold B*T tokens, and the reference requires cu_seqlens
    # to start at 0 and end at exactly that length.
    cu = [0, T, B * T]
    kw_tnd = dict(kw, layout="TND", cu_seqlens=cu)
    packed = [t.reshape(B * T, H, K) for t in (q, k)]
    packed += [gk.reshape(B * T, H, K), beta.reshape(B * T, H),
               dAqk.reshape(B * T, H, cs), dAkk.reshape(B * T, H, cs)]
    packed += [t.reshape(B * T, H, K) for t in (dq, dk)]
    packed += [db.reshape(B * T, H), dg.reshape(B * T, H, K)]
    parity_or_domain_skip(
        "chunk_kda_bwd_intra(TND varlen)",
        lambda: ct.npu_chunk_kda_bwd_intra(*packed, **kw_tnd),
        lambda: _launcher.npu_chunk_kda_bwd_intra(*packed, **kw_tnd))


def scenario_chunk_kda_bwd():
    B, H, T, K, V, cs = 2, 4, 256, 128, 128, 64
    dt = torch.bfloat16
    NT = T // cs
    q = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    k = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    v = torch.randn(B, H, T, V, dtype=dt, device="npu") * 5e-2
    beta = torch.randn(B, H, T, dtype=dt, device="npu")
    gk = torch.randn(B, H, T, K, dtype=torch.float32, device="npu")
    Aqk = torch.randn(B, H, T, cs, dtype=dt, device="npu") * 5e-2
    Akk = torch.randn(B, H, T, cs, dtype=dt, device="npu") * 5e-2
    w = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    qg = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    kg = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    v_new = torch.randn(B, H, T, V, dtype=dt, device="npu") * 5e-2
    h = torch.randn(B, NT, H, K, V, dtype=dt, device="npu") * 5e-2
    d_o = torch.randn(B, H, T, V, dtype=dt, device="npu") * 5e-2
    torch.npu.synchronize()
    kw = dict(raw_g=None, A_log=None, dt_bias=None, initial_state=None,
              dht=None, cu_seqlens=None, chunk_indices=None, chunk_size=cs,
              safe_gate=True, use_gate_in_kernel=False, disable_recompute=True,
              use_exp2=True, state_v_first=False)
    assert_parity(
        "chunk_kda_bwd(dense BNSD)",
        ct.npu_chunk_kda_bwd(q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new,
                             h, d_o, K ** -0.5, **kw),
        _launcher.npu_chunk_kda_bwd(q, k, v, beta, gk, Aqk, Akk, w, qg, kg,
                                v_new, h, d_o, K ** -0.5, **kw))
    # The remaining declared flags of this operator.
    for label, extra in (("state_v_first", dict(state_v_first=True)),
                         ("recompute", dict(disable_recompute=False))):
        torch.npu.synchronize()
        parity_or_domain_skip(
            f"chunk_kda_bwd({label})",
            lambda extra=extra: ct.npu_chunk_kda_bwd(
                q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o,
                K ** -0.5, **dict(kw, **extra)),
            lambda extra=extra: _launcher.npu_chunk_kda_bwd(
                q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o,
                K ** -0.5, **dict(kw, **extra)))


def scenario_chunk_kda_bwd_recompute():
    """The KDA saved tensors: gate cumsum plus the recomputed w/u/qg/kg.

    `use_gate_in_kernel` decides whether the fp32 gate cumsum is materialized,
    so both spellings are exercised -- that is also the only optional output
    here.  The `no gate` spelling is recorded rather than run on Ascend950:
    measured there, it raises an AI Core exception (error code 271) inside the
    kernel and takes the device down with it, so it cannot be a parity case
    until the OPP is fixed.  On 910B both spellings reject with 561103 and the
    helper records that.
    """

    B, H, T, K, cs = 2, 4, 256, 128, 64
    dt = torch.bfloat16
    q = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    k = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    v = torch.randn(B, H, T, K, dtype=dt, device="npu") * 5e-2
    g = torch.randn(B, H, T, K, dtype=torch.float32, device="npu")
    beta = torch.randn(B, H, T, dtype=dt, device="npu")
    a = torch.randn(B, H, T, cs, dtype=dt, device="npu") * 5e-2
    A_log = torch.randn(H, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    for label, extra in (("gate", dict(use_gate_in_kernel=True, A_log=A_log)),
                         ("no gate", dict(use_gate_in_kernel=False,
                                          A_log=None))):
        if label == "no gate" and "950" in str(
                torch.npu.get_device_name(0)):
            name = f"chunk_kda_bwd_recompute({label})"
            SKIPPED[name] = (
                "kernel defect: use_gate_in_kernel=False raises an AI Core "
                "exception (error code 271) on Ascend950 and leaves the device "
                "in an error state, so it is not run")
            print(f"SKIP {name} ({SKIPPED[name]})")
            continue
        parity_or_domain_skip(
            f"chunk_kda_bwd_recompute({label})",
            lambda extra=extra: ct.npu_chunk_kda_bwd_recompute(
                q, k, v, g, beta, a, cs, use_exp2=True, lower_bound=-5.0,
                **extra),
            lambda extra=extra: _launcher.npu_chunk_kda_bwd_recompute(
                q, k, v, g, beta, a, cs, use_exp2=True, lower_bound=-5.0,
                **extra))


def scenario_chunk_gated_delta_rule_bwd():
    """The composite GDN backward added by main (#532).

    One aclnn call for the whole backward graph, so the parity check is the
    interesting part: the operator's public tuple has eight slots, two of which
    (`d_a_log`, `d_dt_bias`) the implementation reserves and never fills, and
    `dh0` only exists when an initial state was passed.

    The reference requires `use_exp2=True`, `use_gate_in_kernel=False` and
    K=V=128 with `chunk_size=64`; the gates are exercised here so that the stable
    path's matching refusals are recorded rather than discovered later.
    """

    B, HK, HV, T, K, V, cs = 2, 2, 4, 256, 128, 128, 64
    dt = torch.bfloat16

    def make(*, with_state):
        q = torch.randn(B, HK, T, K, dtype=dt, device="npu") * 5e-2
        k = torch.randn(B, HK, T, K, dtype=dt, device="npu") * 5e-2
        v = torch.randn(B, HV, T, V, dtype=dt, device="npu") * 5e-2
        # BNSD: g/beta are head-major here, while A stays [B, HV, T, chunk].
        g = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
        beta = torch.randn(B, HV, T, dtype=dt, device="npu")
        a = torch.randn(B, HV, T, cs, dtype=dt, device="npu") * 5e-2
        d_o = torch.randn(B, HV, T, V, dtype=dt, device="npu") * 5e-2
        kw = dict(layout="BNSD", scale=K ** -0.5, chunk_size=cs,
                  use_exp2=True, use_gate_in_kernel=False)
        if with_state:
            state = torch.randn(B, HV, K, V, dtype=dt, device="npu") * 5e-2
            kw["initial_state"] = state
            kw["dht"] = torch.randn_like(state) * 5e-2
        torch.npu.synchronize()
        return (q, k, v, g, beta, a, d_o), kw

    args, kw = make(with_state=False)
    # Both paths may reject the inputs on an OPP whose tiling does not implement
    # this operator yet; the helper records that instead of failing, and the
    # case turns into a real parity test as soon as one accepts it.
    parity_or_domain_skip(
        "chunk_gated_delta_rule_bwd(dense BNSD)",
        lambda: ct.npu_chunk_gated_delta_rule_bwd(*args, **kw),
        lambda: _launcher.npu_chunk_gated_delta_rule_bwd(*args, **kw))
    # With an initial state the first output slot stops being None, which is the
    # branch the mask in the wrapper has to get right.
    args_state, kw_state = make(with_state=True)
    parity_or_domain_skip(
        "chunk_gated_delta_rule_bwd(initial state)",
        lambda: ct.npu_chunk_gated_delta_rule_bwd(*args_state, **kw_state),
        lambda: _launcher.npu_chunk_gated_delta_rule_bwd(*args_state, **kw_state))
    # The two flags the composite does not implement must be refused on both
    # paths; the helper records that instead of comparing anything.
    for label, extra in (("use_exp2", dict(use_exp2=False)),
                         ("use_gate_in_kernel", dict(use_gate_in_kernel=True))):
        parity_or_domain_skip(
            f"chunk_gated_delta_rule_bwd({label}=unsupported)",
            lambda extra=extra: ct.npu_chunk_gated_delta_rule_bwd(
                *args, **dict(kw, **extra)),
            lambda extra=extra: _launcher.npu_chunk_gated_delta_rule_bwd(
                *args, **dict(kw, **extra)))
    # The operator's other three declared spellings.  It accepts the TND/NTD
    # *names* but still reads a rank-4 tensor (layout_math::tokens4), so only the
    # token axis moves -- BSND/TND put it on dim 1, BNSD/NTD on dim 2 -- and the
    # two packed names additionally take cu_seqlens with a physical batch of 1.
    #
    # Two contract details the reference spells out (and rejected the first
    # version of these cases for): `A` stays BNSD-shaped whatever the layout says
    # ("A must have BNSD shape [B, HV, T, chunk_size]"), and a packed spelling
    # needs cu_seqlens *and* chunk_indices together.
    def token_major(tensor):
        return (tensor.transpose(1, 2).contiguous()
                if tensor.dim() >= 3 else tensor)

    # q, k, v, g, beta and d_o follow the layout; A does not (args[5]).
    bsnd = (tuple(token_major(tensor) for tensor in args[:5])
            + (args[5], token_major(args[6])))
    kw_bsnd = dict(kw, layout="BSND")
    parity_or_domain_skip(
        "chunk_gated_delta_rule_bwd(dense BSND)",
        lambda: ct.npu_chunk_gated_delta_rule_bwd(*bsnd, **kw_bsnd),
        lambda: _launcher.npu_chunk_gated_delta_rule_bwd(*bsnd, **kw_bsnd))
    # One sequence of T tokens, chunk_size tokens per chunk: the canonical
    # (sequence, chunk) pairs the reference asks for.
    chunk_indices = [pair for chunk in range(T // cs) for pair in (0, chunk)]
    for layout, spelled in (("TND", bsnd), ("NTD", args)):
        packed = tuple(tensor[:1].contiguous() for tensor in spelled)
        kw_packed = dict(kw, layout=layout, cu_seqlens=[0, T],
                         chunk_indices=chunk_indices)
        parity_or_domain_skip(
            f"chunk_gated_delta_rule_bwd(varlen {layout})",
            lambda s=packed, k=kw_packed: ct.npu_chunk_gated_delta_rule_bwd(
                *s, **k),
            lambda s=packed, k=kw_packed: _launcher
            .npu_chunk_gated_delta_rule_bwd(*s, **k))


def scenario_dqkwg():
    B, HK, HV, T, K, V, cs = 1, 4, 4, 1024, 128, 128, 64
    NT = T // cs
    dt = torch.float16

    def make4(*shape, scale_):
        return (torch.randn(shape) * scale_).to(dt).permute(
            0, 2, 1, 3).contiguous().npu()

    def make5(*shape, scale_):
        return (torch.randn(shape) * scale_).to(dt).permute(
            0, 2, 1, 3, 4).contiguous().npu()

    q = make4(B, T, HK, K, scale_=5e-2)
    k = make4(B, T, HK, K, scale_=5e-2)
    v = make4(B, T, HV, V, scale_=5e-2)
    do = make4(B, T, HV, V, scale_=5e-2)
    dv = make4(B, T, HV, V, scale_=5e-1)
    h = make5(B, NT, HV, K, V, scale_=5e-2)
    dh = make5(B, NT, HV, K, V, scale_=5e-2)
    g = (-torch.sort(torch.rand(B * T * HV), descending=False)[0]
         .reshape(B, T, HV)).permute(0, 2, 1).to(dt).contiguous().npu()
    torch.npu.synchronize()
    kw = dict(cu_seqlens=None, chunk_indices=None, w=None, g_gamma=None,
              scale=0.088, use_exp2=None, transpose_state_layout=None)
    assert_parity("chunk_bwd_dqkwg",
                  ct.npu_chunk_bwd_dqkwg(q, k, v, g, h, do, dh, dv, cs, **kw),
                  _launcher.npu_chunk_bwd_dqkwg(q, k, v, g, h, do, dh, dv, cs,
                                            **kw))
    # use_exp2 / transpose_state_layout: both are declared, neither is the
    # default, so they get their own cases (a rejected combination is recorded).
    for label, extra in (("use_exp2", dict(use_exp2=True)),
                         ("transpose_state_layout",
                          dict(transpose_state_layout=True))):
        torch.npu.synchronize()
        parity_or_domain_skip(
            f"chunk_bwd_dqkwg({label})",
            lambda extra=extra: ct.npu_chunk_bwd_dqkwg(
                q, k, v, g, h, do, dh, dv, cs, **dict(kw, **extra)),
            lambda extra=extra: _launcher.npu_chunk_bwd_dqkwg(
                q, k, v, g, h, do, dh, dv, cs, **dict(kw, **extra)))


def scenario_chunk_local_cumsum():
    # Dense rank-3 [B,H,T] domain: fixed length, reverse+scale, odd tail,
    # and a full-length varlen (single sequence) metadata path.
    for dt, suffix in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        fixed = torch.randn(2, 3, 128, dtype=dt, device="npu")
        torch.npu.synchronize()
        assert_parity(
            f"chunk_local_cumsum(fixed_{suffix})",
            ct.npu_chunk_local_cumsum(fixed, chunk_size=64),
            _launcher.npu_chunk_local_cumsum(fixed, chunk_size=64))
        odd = torch.randn(2, 3, 129, dtype=dt, device="npu")
        torch.npu.synchronize()
        assert_parity(
            f"chunk_local_cumsum(odd_t_{suffix})",
            ct.npu_chunk_local_cumsum(odd, chunk_size=64),
            _launcher.npu_chunk_local_cumsum(odd, chunk_size=64))
    reverse = torch.randn(2, 3, 128, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    kw = dict(chunk_size=64, reverse=True, scale=0.25)
    assert_parity(
        "chunk_local_cumsum(reverse_scale_fp16)",
        ct.npu_chunk_local_cumsum(reverse, **kw),
        _launcher.npu_chunk_local_cumsum(reverse, **kw))
    # output_dtype and head_first are both real parameters of the operator.
    for label, extra in (("output_dtype=bfloat16",
                          dict(output_dtype="bfloat16")),
                         ("output_dtype=float32",
                          dict(output_dtype="float32")),
                         ("head_first_false", dict(head_first=False))):
        torch.npu.synchronize()
        parity_or_domain_skip(
            f"chunk_local_cumsum({label})",
            lambda extra=extra: ct.npu_chunk_local_cumsum(
                reverse, chunk_size=64, **extra),
            lambda extra=extra: _launcher.npu_chunk_local_cumsum(
                reverse, chunk_size=64, **extra))
    varlen = torch.randn(1, 2, 128, dtype=torch.float16, device="npu")
    cu = [0, 128]
    ci = [0, 0]  # (seq_idx, chunk_idx) rows flattened for the single seq
    torch.npu.synchronize()
    assert_parity(
        "chunk_local_cumsum(varlen_single_fp16)",
        ct.npu_chunk_local_cumsum(varlen, chunk_size=64, cu_seqlens=cu,
                                  chunk_indices_out=ci),
        _launcher.npu_chunk_local_cumsum(varlen, chunk_size=64, cu_seqlens=cu,
                                     chunk_indices_out=ci))


def scenario_scaled_dot_kkt():
    B, Hk, Hv, T, K, cs = 2, 4, 4, 128, 64, 64
    for dt, suffix in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        k = (torch.randn(B, Hk, T, K) * 0.2).to(dt).npu()
        # The embedded OPP only ships k fp16/bf16 x g/beta fp32 variants.
        g = (torch.randn(B, Hv, T) * 0.02).npu()
        beta = torch.sigmoid(torch.randn(B, Hv, T)).npu()
        torch.npu.synchronize()
        assert_parity(
            f"chunk_scaled_dot_kkt({suffix})",
            ct.npu_chunk_scaled_dot_kkt(k, g, beta, chunk_size=cs),
            _launcher.npu_chunk_scaled_dot_kkt(k, g, beta, chunk_size=cs))


def scenario_solve_tri_dense():
    # Dense bsnd/bnsd is native stable; varlen (tnd/ntd) intentionally
    # delegates to ctypes inside the wrapper, so only dense is covered.
    B, H, T = 2, 4, 128
    for dt, suffix in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        for bt in (16, 64, 128):
            a_bsnd = (torch.randn(B, T, H, bt) * 0.1).to(dt).npu()
            torch.npu.synchronize()
            assert_parity(
                f"solve_tri(bsnd_{suffix}_bt{bt})",
                ct.npu_solve_tri(a_bsnd, layout="bsnd"),
                _launcher.npu_solve_tri(a_bsnd, layout="bsnd"))
        a_bnsd = ((torch.randn(B, T, H, 64) * 0.1).to(dt).npu()
                  .permute(0, 2, 1, 3).contiguous())
        torch.npu.synchronize()
        assert_parity(
            f"solve_tri(bnsd_{suffix})",
            ct.npu_solve_tri(a_bnsd, layout="bnsd"),
            _launcher.npu_solve_tri(a_bnsd, layout="bnsd"))
        # The two packed spellings are broken upstream on this OPP -- both kill
        # the process, measured with and without cu_seqlens -- so neither can be
        # a parity case; the wrappers refuse them instead, which
        # scenario_solve_tri_guards below records.


def scenario_solve_tri_guards():
    """The launcher must refuse the upstream-broken spelling, not crash.

    Measured first, then encoded: ``layout="tnd"`` kills the process on the
    ctypes reference *and* on the launcher, with and without cu_seqlens, so it
    is not a legal domain on this OPP.  Refusing it with a message is the
    documented contract ("illegal input errors, and does not have to error the
    same way"); matching the reference by crashing would not be.

    ``ntd`` behaves the same way -- measured while the missing-spelling work
    added a scenario for it: five of six shapes segfault, the sixth is rejected
    161001, so no ntd call produces a result.  It is *not* covered here because
    it is not intercepted: the reference forwards it too, and whether to refuse
    it is the operator owner's decision (see the inventory's known limits).
    A scenario that calls it would take the test process down.
    """

    a = (torch.randn(64, 4, 64) * 0.1).to(torch.float16).npu()
    torch.npu.synchronize()
    for backend, label in ((ct.npu_solve_tri, "ctypes"),
                           (_launcher.npu_solve_tri, "stable")):
        try:
            backend(a, layout="tnd")
        except RuntimeError as exc:
            assert "tnd" in str(exc), f"{label}: unexpected message {exc}"
            # A refusal is coverage too: recorded with its reason so the
            # scenario set shrinks visibly if the guard is ever dropped.
            name = f"solve_tri(tnd refused by {label})"
            SKIPPED[name] = ("layout='tnd' is refused because the operator "
                             "crashes the process for that spelling")
            print(f"SKIP {name} ({SKIPPED[name]})")
        else:
            raise AssertionError(f"{label} accepted the crashing tnd spelling")


def scenario_kda_gate_cumsum():
    # Dense KDA gate cumsum: g is head-major [B,H,T,K] fp16/bf16, while
    # A_log/dt_bias are fp32 (matches the embedded OPP kernel config).
    B, H, T, K, cs = 1, 4, 128, 64, 64
    for dt, suffix in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        raw = (torch.randn(B, T, H, K) * 1.25).to(dt).npu()
        g = raw.permute(0, 2, 1, 3).contiguous()
        a_log = torch.randn(H, dtype=torch.float32, device="npu") * 0.12
        dt_bias = (torch.randn(H * K, dtype=torch.float32) * 1.65 - 3.0).npu()
        torch.npu.synchronize()
        kw = dict(A_log=a_log, dt_bias=dt_bias,
                  use_gate_in_kernel=True, safe_gate=False, lower_bound=-5.0)
        assert_parity(
            f"kda_gate_cumsum({suffix})",
            ct.npu_kda_gate_cumsum(g, cs, **kw),
            _launcher.npu_kda_gate_cumsum(g, cs, **kw))
        # safe_gate is the operator's other declared flag.
        torch.npu.synchronize()
        parity_or_domain_skip(
            f"kda_gate_cumsum({suffix},safe_gate)",
            lambda g=g, kw=kw: ct.npu_kda_gate_cumsum(
                g, cs, **dict(kw, safe_gate=True)),
            lambda g=g, kw=kw: _launcher.npu_kda_gate_cumsum(
                g, cs, **dict(kw, safe_gate=True)))


def scenario_recurrent_kda():
    """Recurrent KDA forward through both backends, in-place state included.

    The shape/settings mirror the Ascend950 driver so the same scenario is
    exercised on A2 as well; the state tensor is written in place, so each
    backend gets its own clone and both are compared afterwards.
    """

    B, T, H, HV, K, V = 2, 2, 2, 4, 128, 128
    dt = torch.bfloat16
    q = torch.randn(B, T, H, K, dtype=dt, device="npu")
    k = torch.randn(B, T, H, K, dtype=dt, device="npu")
    v = torch.randn(B, T, HV, V, dtype=dt, device="npu")
    g = -torch.rand(B, T, HV, K, dtype=torch.float32, device="npu") * 5 - 1e-3
    beta = torch.rand(B, T, HV, dtype=torch.float32, device="npu") * 0.8 + 0.1
    cu = torch.tensor([0, T, 2 * T], dtype=torch.int64, device="npu")
    torch.npu.synchronize()
    st_c = torch.zeros(B, HV, V, K, dtype=torch.float32, device="npu")
    st_t = st_c.clone()
    kw = dict(cu_seqlens=cu, scale=K ** -0.5, layout="BSND",
              state_v_first=True)
    oc = ct.npu_recurrent_kda(q, k, v, g, beta, st_c, **kw)
    ot = _launcher.npu_recurrent_kda(q, k, v, g, beta, st_t, **kw)
    torch.npu.synchronize()
    assert_parity("recurrent_kda(dense BSND)", oc, ot)
    record_extra("recurrent_kda(state)", st_c, st_t)
    # TND: the same operator in the varlen spelling (T is the token axis).
    T_tnd = T * B
    q_t = q.reshape(T_tnd, H, K)
    k_t = k.reshape(T_tnd, H, K)
    v_t = v.reshape(T_tnd, HV, V)
    g_t = g.reshape(T_tnd, HV, K)
    beta_t = beta.reshape(T_tnd, HV)
    cu_t = torch.tensor([0, T, T_tnd], dtype=torch.int64, device="npu")
    st_ct = torch.zeros(2, HV, V, K, dtype=torch.float32, device="npu")
    st_tt = st_ct.clone()
    kw_t = dict(cu_seqlens=cu_t, scale=K ** -0.5, layout="TND",
                state_v_first=True)
    oc_t = ct.npu_recurrent_kda(q_t, k_t, v_t, g_t, beta_t, st_ct, **kw_t)
    ot_t = _launcher.npu_recurrent_kda(q_t, k_t, v_t, g_t, beta_t, st_tt, **kw_t)
    torch.npu.synchronize()
    assert_parity("recurrent_kda(dense TND)", oc_t, ot_t)
    record_extra("recurrent_kda(TND state)", st_ct, st_tt)


def scenario_chunk_gated_delta_rule_fwd():
    """Fused GDN forward (legacy Phase6 domain: BNSD dense).

    Upstream #495 fixed the op_api/ctypes parameter passing, so the fused op
    now runs on A2; the stable adapter covers the legacy path and falls back to
    ctypes for the A5 (use_exp2) / varlen / other-layout combos.
    """

    def make_case(B, Hk, Hv, T, V, chunk, suffix, with_final=True,
                  st_dtype=None):
        dt = torch.bfloat16
        q = (torch.randn(B, Hk, T, 128, device="npu") * 0.05).to(dt)
        k = (torch.randn(B, Hk, T, 128, device="npu") * 0.05).to(dt)
        v = (torch.randn(B, Hv, T, V, device="npu") * 0.05).to(dt)
        g = (torch.randn(B, T, Hv, device="npu") * 1.25).to(torch.float32)
        beta = torch.sigmoid(torch.randn(B, T, Hv, device="npu"))
        kw = dict(chunk_size=chunk, output_final_state=with_final)
        if st_dtype is not None:
            kw["initial_state"] = (
                torch.randn(B, Hv, 128, V, device="npu") * 0.02).to(st_dtype)
            kw["output_final_state"] = True
        torch.npu.synchronize()
        assert_parity(
            f"chunk_gated_delta_rule_fwd({suffix})",
            ct.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kw),
            _launcher.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kw))

    # GVA + final state + fp32 initial state (labels name the layout the case
    # drives, so the checked-in coverage record says which axes were covered:
    # BSND/TND/NTD are exercised by the Ascend950 driver).
    make_case(2, 2, 4, 128, 128, 64, "BNSD_B2_Hk2_Hv4_T128_V128_c64",
              st_dtype=torch.float32)
    # bf16 initial state, chunk 128, V=256
    make_case(2, 2, 4, 256, 256, 128, "B2_Hk2_Hv4_T256_V256_c128",
              st_dtype=torch.bfloat16)
    # no initial/final state
    make_case(1, 2, 2, 192, 128, 64, "B1_Hk2_Hv2_T192_V128_c64",
              with_final=False)

    # varlen (physical B=1, canonical chunk_indices)
    B, Hk, Hv, T, V, cs = 1, 2, 4, 128, 128, 64
    dt = torch.bfloat16
    q = (torch.randn(B, Hk, T, 128, device="npu") * 0.05).to(dt)
    k = (torch.randn(B, Hk, T, 128, device="npu") * 0.05).to(dt)
    v = (torch.randn(B, Hv, T, V, device="npu") * 0.05).to(dt)
    g = (torch.randn(B, T, Hv, device="npu") * 1.25).to(torch.float32)
    beta = torch.sigmoid(torch.randn(B, T, Hv, device="npu"))
    cu = [0, 30, 128]
    ci = []
    for seq, (begin, end) in enumerate(zip(cu[:-1], cu[1:])):
        for chunk in range((end - begin + cs - 1) // cs):
            ci.extend((seq, chunk))
    torch.npu.synchronize()
    kw = dict(chunk_size=cs, output_final_state=True, cu_seqlens=cu,
              chunk_indices=ci)

    def _finite_parity(name, oc, ot):
        # `A` is a chunk-local lower triangle: only the columns up to the
        # token's offset inside its chunk are written, and the rest keeps
        # whatever the allocator had.  On 910B both paths left NaN there; on
        # Ascend950 they leave *finite* garbage (measured 7e29 and 3.4e38) that
        # differs between the two allocations, which is what made this case fail
        # there.  The mask below compares the part the operator actually
        # computes -- it is derived from cu_seqlens, not from a tolerance.
        #
        # The result is recorded in SCENARIOS rather than only printed: the
        # scenario-set check is what notices if this case is ever dropped.
        assert len(oc) == len(ot)
        offsets = torch.zeros(T, dtype=torch.long)
        for begin, end in zip(cu, cu[1:]):
            offsets[begin:end] = torch.arange(end - begin) % cs
        columns = torch.arange(cs)
        triangle = (columns[None, :] <= offsets[:, None]).npu()
        for i, (a, b) in enumerate(zip(oc, ot)):
            if a is None or b is None:
                assert a is None and b is None, f"{name}[{i}]: None mismatch"
                continue
            assert tuple(a.shape) == tuple(b.shape), f"{name}[{i}]: shape"
            af, bf = a.float(), b.float()
            finite = torch.isfinite(af) & torch.isfinite(bf)
            if a.dim() == 4 and a.shape[-1] == cs:
                finite = finite & triangle[None, None, :, :]
            if finite.any():
                diff = float((af - bf).abs()[finite].max().item())
                assert diff == 0.0, f"{name}[{i}]: diff={diff}"
        SCENARIOS[name] = 0.0
        print(f"PASS {name}")

    _finite_parity(
        "chunk_gated_delta_rule_fwd(varlen_B1_T128_c64)",
        ct.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kw),
        _launcher.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kw))

    # The operator's other declared layouts.  Measured on A2: BSND is rejected
    # by the kernel (161002), and the rank-3 TND/NTD spellings are refused by the
    # reference's own validation ("q, k and v must be rank-4").  They are still
    # exercised -- by the Ascend950 driver, whose kernel does implement them --
    # so here they are recorded rather than left as unexplained gaps.
    for layout in ("BSND", "TND", "NTD"):
        B2, Hk2, Hv2, T2, V2, cs2 = 1, 2, 2, 128, 128, 64

        def rnd(*shape):
            return (torch.randn(*shape, device="npu") * 0.05).to(torch.bfloat16)

        if layout == "BSND":
            q2, k2 = rnd(B2, T2, Hk2, 128), rnd(B2, T2, Hk2, 128)
            v2 = rnd(B2, T2, Hv2, V2)
            g2 = (torch.randn(B2, T2, Hv2, device="npu") * 1.25).to(torch.float32)
            beta2 = torch.sigmoid(torch.randn(B2, T2, Hv2, device="npu"))
        elif layout == "TND":
            q2, k2 = rnd(T2, Hk2, 128), rnd(T2, Hk2, 128)
            v2 = rnd(T2, Hv2, V2)
            g2 = (torch.randn(T2, Hv2, device="npu") * 1.25).to(torch.float32)
            beta2 = torch.sigmoid(torch.randn(T2, Hv2, device="npu"))
        else:  # NTD
            q2, k2 = rnd(Hk2, T2, 128), rnd(Hk2, T2, 128)
            v2 = rnd(Hv2, T2, V2)
            g2 = (torch.randn(Hv2, T2, device="npu") * 1.25).to(torch.float32)
            beta2 = torch.sigmoid(torch.randn(Hv2, T2, device="npu"))
        torch.npu.synchronize()
        parity_or_domain_skip(
            f"chunk_gated_delta_rule_fwd(layout={layout})",
            lambda q2=q2, k2=k2, v2=v2, g2=g2, beta2=beta2, layout=layout,
            cs2=cs2: ct.npu_chunk_gated_delta_rule_fwd(
                q2, k2, v2, g2, beta2, chunk_size=cs2, layout=layout),
            lambda q2=q2, k2=k2, v2=v2, g2=g2, beta2=beta2, layout=layout,
            cs2=cs2: _launcher.npu_chunk_gated_delta_rule_fwd(
                q2, k2, v2, g2, beta2, chunk_size=cs2, layout=layout))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    group_cli(parser)
    args = parser.parse_args()

    if args.list_groups:
        # The listing needs no backend at all, so it answers even on a machine
        # that has no launcher built.
        print_groups([fn.__name__ for fn in _scenarios()])
        return

    # This driver runs the shipped backend directly; regression_stable_full.py
    # is the same run plus the checked-in baseline, so this is the quick form.
    if not _launcher.available():
        raise SystemExit(
            "no Stable-ABI launcher: set FLA_NPU_STABLE_LIB to a built "
            "libfla_npu_stable.so, or install a wheel that bundles one")
    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    scenarios = _scenarios()
    names = [fn.__name__ for fn in scenarios]
    missing_from_groups(names)
    chosen = set(select_groups(names, args.group))
    selected = [fn for fn in scenarios if fn.__name__ in chosen]
    if args.group:
        print(f"groups {', '.join(args.group)}: "
              f"{len(selected)} of {len(scenarios)} scenarios "
              f"({', '.join(select_groups(names, args.group))})")
    for fn in selected:
        fn()
    print(f"ALL PASS: {len(selected)} stable-op parity scenarios")


def _scenarios():
    """The scenario functions this driver runs, in order."""

    return [
        scenario_fast_gelu,
        scenario_recurrent_gated_delta_rule,
        scenario_recompute,
        scenario_pwy_full,
        scenario_pwy,
        scenario_dv_local,
        scenario_pwy_da,
        scenario_gated_fwd_h,
        scenario_chunk_fwd_h,
        scenario_chunk_fwd_o,
        scenario_bwd_dhu,
        scenario_conv1d_bwd_bnsd,
        scenario_chunk_kda_fwd,
        scenario_chunk_kda_fwd_variants,
        scenario_chunk_kda_bwd_intra,
        scenario_chunk_kda_bwd,
        scenario_chunk_kda_bwd_recompute,
        scenario_dqkwg,
        scenario_chunk_local_cumsum,
        scenario_scaled_dot_kkt,
        scenario_solve_tri_dense,
        scenario_kda_gate_cumsum,
        scenario_chunk_gated_delta_rule_fwd,
        scenario_conv1d_prefill,
        scenario_conv1d_varlen_initial_state,
        scenario_conv1d_update,
        scenario_conv1d_update_offset_state,
        scenario_conv1d_update_paged_state,
        scenario_conv1d_prefill_paged_state,
        scenario_recurrent_kda,
    ]


if __name__ == "__main__":
    main()
