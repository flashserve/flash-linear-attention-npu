"""Wheel 安装态数值回归：已迁移 thin 算子的 ctypes-vs-thin parity。

用法（221 上，wheel 已 pip install --target envXXX）:
    PYTHONPATH=/path/envXXX python tests/regression_thin_ops.py

每个场景对同一输入分别走 ctypes 与 thin 两条 host 路径，断言每个输出
tensor 的逐元素差为 0（同一 OPP kernel，期望 bitwise 相同）。host P50
仅作参考；合法域说明见 thin-migration-inventory.md。
"""
from __future__ import annotations

import itertools
import json
import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import _thin  # noqa: E402


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


def assert_parity(name, oc, ot):
    if not isinstance(oc, tuple):
        oc = (oc,)
        ot = (ot,)
    assert len(oc) == len(ot), f"{name}: output count mismatch"
    for i, (a, b) in enumerate(zip(oc, ot)):
        if a is None or b is None:
            assert a is None and b is None, f"{name}[{i}]: None mismatch"
            continue
        assert tuple(a.shape) == tuple(b.shape), f"{name}[{i}]: shape"
        diff = float((a.float() - b.float()).abs().max().item())
        assert diff == 0.0, f"{name}[{i}]: diff={diff}"
    print(f"PASS {name}")


def scenario_fast_gelu():
    x = torch.randn(4, 128, 256, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    assert_parity("fast_gelu_custom",
                  ct.npu_fast_gelu_custom(x), _thin.npu_fast_gelu_custom(x))
    grad = torch.randn_like(x)
    assert_parity("fast_gelu_custom_backward",
                  ct.npu_fast_gelu_custom_backward(grad, x),
                  _thin.npu_fast_gelu_custom_backward(grad, x))


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
    out_t = _thin.npu_recurrent_gated_delta_rule(query, key, value, state_t,
                                                 **kw)
    torch.npu.synchronize()
    assert_parity("recurrent_gated_delta_rule", out_c, out_t)
    assert float((state_c.float() - state_t.float()).abs().max().item()) == 0.0
    print("PASS recurrent_gated_delta_rule(state)")


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
                  _thin.npu_recompute_w_u_fwd(k, v, beta, A, cs, **kw))


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
                  _thin.npu_prepare_wy_repr_bwd_full(
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
                  _thin.npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, **kw))


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
                  _thin.npu_chunk_bwd_dv_local(q, k, d_o, g, **kw))


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
                  _thin.npu_prepare_wy_repr_bwd_da(
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
        _thin.npu_chunk_gated_delta_rule_fwd_h(k, w, u, g, chunk_size=cs))
    is0 = torch.randn(B, Hv, K, V, dtype=torch.float32, device="npu")
    assert_parity(
        "chunk_gated_delta_rule_fwd_h(final)",
        ct.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=is0, output_final_state=True,
            chunk_size=cs),
        _thin.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g, initial_state=is0, output_final_state=True,
            chunk_size=cs))


def scenario_chunk_fwd_h():
    B, Hk, Hv, T, K, V, cs = 1, 2, 2, 256, 128, 128, 64
    k, w, u, g = _fwd_h_inputs(B, Hk, Hv, T, K, V)
    torch.npu.synchronize()
    assert_parity("chunk_fwd_h",
                  ct.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs),
                  _thin.npu_chunk_fwd_h(k, w, u, g=g, chunk_size=cs))


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
        _thin.npu_chunk_fwd_o(q, k, v, h, scale, g=g, chunk_size=cs,
                              output_layout="BNSD"))


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
                  _thin.npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, do, dv, **kw))


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
    assert_parity("causal_conv1d_bwd(BNSD)",
                  ct.npu_causal_conv1d_bwd(**kw),
                  _thin.npu_causal_conv1d_bwd(**kw))


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

    合法域由 ctypes 参考实现界定；thin 只有在每个组合的逐输出 diff 都为 0、
    且 None 掩码与返回元组顺序都一致时才算覆盖（见 op_policy_check.py）。
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
                                  _thin.npu_chunk_kda_fwd(q, k, v, g, beta,
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
        assert_parity(
            f"chunk_kda_fwd(var {layout} B={B} T={T} H={H} HV={HV} "
            f"V={V} cs={cs} g={str(gdt).split('.')[-1]})",
            ct.npu_chunk_kda_fwd(q, k, v, g, beta, **kw),
            _thin.npu_chunk_kda_fwd(q, k, v, g, beta, **kw))
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
        _thin.npu_chunk_kda_bwd_intra(q, k, gk, beta, dAqk, dAkk, dq, dk, db,
                                      dg, **kw))


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
        _thin.npu_chunk_kda_bwd(q, k, v, beta, gk, Aqk, Akk, w, qg, kg,
                                v_new, h, d_o, K ** -0.5, **kw))


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
                  _thin.npu_chunk_bwd_dqkwg(q, k, v, g, h, do, dh, dv, cs,
                                            **kw))


def scenario_chunk_local_cumsum():
    # Dense rank-3 [B,H,T] domain: fixed length, reverse+scale, odd tail,
    # and a full-length varlen (single sequence) metadata path.
    for dt, suffix in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        fixed = torch.randn(2, 3, 128, dtype=dt, device="npu")
        torch.npu.synchronize()
        assert_parity(
            f"chunk_local_cumsum(fixed_{suffix})",
            ct.npu_chunk_local_cumsum(fixed, chunk_size=64),
            _thin.npu_chunk_local_cumsum(fixed, chunk_size=64))
        odd = torch.randn(2, 3, 129, dtype=dt, device="npu")
        torch.npu.synchronize()
        assert_parity(
            f"chunk_local_cumsum(odd_t_{suffix})",
            ct.npu_chunk_local_cumsum(odd, chunk_size=64),
            _thin.npu_chunk_local_cumsum(odd, chunk_size=64))
    reverse = torch.randn(2, 3, 128, dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    kw = dict(chunk_size=64, reverse=True, scale=0.25)
    assert_parity(
        "chunk_local_cumsum(reverse_scale_fp16)",
        ct.npu_chunk_local_cumsum(reverse, **kw),
        _thin.npu_chunk_local_cumsum(reverse, **kw))
    varlen = torch.randn(1, 2, 128, dtype=torch.float16, device="npu")
    cu = [0, 128]
    ci = [0, 0]  # (seq_idx, chunk_idx) rows flattened for the single seq
    torch.npu.synchronize()
    assert_parity(
        "chunk_local_cumsum(varlen_single_fp16)",
        ct.npu_chunk_local_cumsum(varlen, chunk_size=64, cu_seqlens=cu,
                                  chunk_indices_out=ci),
        _thin.npu_chunk_local_cumsum(varlen, chunk_size=64, cu_seqlens=cu,
                                     chunk_indices=ci))


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
            _thin.npu_chunk_scaled_dot_kkt(k, g, beta, chunk_size=cs))


def scenario_solve_tri_dense():
    # Dense bsnd/bnsd is native thin; varlen (tnd/ntd) intentionally
    # delegates to ctypes inside the _thin wrapper, so only dense is covered.
    B, H, T = 2, 4, 128
    for dt, suffix in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        for bt in (16, 64, 128):
            a_bsnd = (torch.randn(B, T, H, bt) * 0.1).to(dt).npu()
            torch.npu.synchronize()
            assert_parity(
                f"solve_tri(bsnd_{suffix}_bt{bt})",
                ct.npu_solve_tri(a_bsnd, layout="bsnd"),
                _thin.npu_solve_tri(a_bsnd, layout="bsnd"))
        a_bnsd = ((torch.randn(B, T, H, 64) * 0.1).to(dt).npu()
                  .permute(0, 2, 1, 3).contiguous())
        torch.npu.synchronize()
        assert_parity(
            f"solve_tri(bnsd_{suffix})",
            ct.npu_solve_tri(a_bnsd, layout="bnsd"),
            _thin.npu_solve_tri(a_bnsd, layout="bnsd"))


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
            _thin.npu_kda_gate_cumsum(g, cs, **kw))


def scenario_chunk_gated_delta_rule_fwd():
    """Fused GDN forward (legacy Phase6 domain: BNSD dense).

    Upstream #495 fixed the op_api/ctypes parameter passing, so the fused op
    now runs on A2; the thin adapter covers the legacy path and falls back to
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
            _thin.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kw))

    # GVA + final state + fp32 initial state
    make_case(2, 2, 4, 128, 128, 64, "B2_Hk2_Hv4_T128_V128_c64",
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
        # A's tail-padding rows are uninitialised on both paths; compare the
        # finite region only (see thin-migration-inventory.md).
        assert len(oc) == len(ot)
        for i, (a, b) in enumerate(zip(oc, ot)):
            if a is None or b is None:
                assert a is None and b is None, f"{name}[{i}]: None mismatch"
                continue
            assert tuple(a.shape) == tuple(b.shape), f"{name}[{i}]: shape"
            finite = torch.isfinite(a.float()) & torch.isfinite(b.float())
            if finite.any():
                diff = float(
                    (a.float() - b.float()).abs()[finite].max().item())
                assert diff == 0.0, f"{name}[{i}]: diff={diff}"
        print(f"PASS {name}")

    _finite_parity(
        "chunk_gated_delta_rule_fwd(varlen_B1_T128_c64)",
        ct.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kw),
        _thin.npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, **kw))


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    scenarios = [
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
        scenario_dqkwg,
        scenario_chunk_local_cumsum,
        scenario_scaled_dot_kkt,
        scenario_solve_tri_dense,
        scenario_kda_gate_cumsum,
        scenario_chunk_gated_delta_rule_fwd,
    ]
    for fn in scenarios:
        fn()
    print("ALL PASS: 21 thin-op parity scenarios")


if __name__ == "__main__":
    main()
