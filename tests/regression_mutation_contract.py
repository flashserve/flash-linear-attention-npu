"""安装态 mutation 契约回归：快路径重构后 version 计数 / grad 拒绝 / 数值一致。

用法（221 或 241，wheel 已 pip install --target envXXX）：
    PYTHONPATH=/path/envXXX python tests/regression_mutation_contract.py

覆盖 `MUTATED_ARGUMENTS` 里的原地更新算子。重构把 `inspect.signature().bind`
从热路径挪走（`MUTATION_FLAGS` 声明式 flag），正确性由三件事保证：

1. 该 bump 的还是 bump（inplace=True / 无 flag 的算子）；
2. 不该 bump 的不 bump（inplace=False 走 scratch state）；
3. requires_grad 的拒绝行为与 ctypes 完全一致。
"""
from __future__ import annotations

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import npu_recurrent_gated_delta_rule as gdr_public
from fla_npu.ops.ascendc import npu_recurrent_kda as kda_public


def version_of(tensor):
    return int(tensor._version)


def check_bump(label, tensor, before, expected):
    after = version_of(tensor)
    assert after - before == expected, (
        f"{label}: version {before} -> {after}, expected +{expected}")
    print(f"PASS {label} (version +{expected})")


def gdr_inputs():
    batch, nk, nv, dim = 8, 8, 16, 128
    gap, offset = 16384, 12288
    block_stride = nv * dim * dim + gap
    backing = torch.empty(batch * block_stride * 4, dtype=torch.int8,
                          device="npu")
    typed = backing.view(torch.float32)
    state = torch.as_strided(
        typed, size=(batch, nv, dim, dim),
        stride=(block_stride, dim * dim, dim, 1), storage_offset=offset)
    state.zero_()
    norm = lambda t: torch.nn.functional.normalize(t, p=2, dim=-1)  # noqa: E731
    query = norm(torch.randn(batch, nk, dim, device="npu")).to(
        torch.bfloat16)
    key = norm(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16)
    value = torch.randn(batch, nv, dim, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(batch, nv, dtype=torch.bfloat16, device="npu")
    g = torch.rand(batch, nv, dtype=torch.float32, device="npu")
    asl = torch.tensor([0] + [1] * batch, dtype=torch.int32, device="npu")
    ssi = torch.arange(batch, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    return query, key, value, state, dict(
        beta=beta, g=g, scale=dim ** -0.5, actual_seq_lengths=asl,
        ssm_state_indices=ssi, num_accepted_tokens=None)


def scenario_recurrent_gdr():
    query, key, value, state, kw = gdr_inputs()
    # 1) state 位置参数 → mutation 快路径，仍然 bump。
    before = version_of(state)
    gdr_public(query, key, value, state, **kw)
    torch.npu.synchronize()
    check_bump("recurrent_gdr(state positional)", state, before, 1)
    # 2) state 关键字 → 慢路径，语义必须一致。
    before = version_of(state)
    gdr_public(query, key, value, state=state, **kw)
    torch.npu.synchronize()
    check_bump("recurrent_gdr(state keyword)", state, before, 1)
    # 3) requires_grad 的 state 必须被拒（两条路径都拒）。
    grad_state = state.clone().detach().requires_grad_(True)
    for label, call in (
        ("positional", lambda: gdr_public(query, key, value, grad_state, **kw)),
        ("keyword", lambda: gdr_public(query, key, value, state=grad_state,
                                       **kw)),
    ):
        try:
            call()
        except RuntimeError as exc:
            assert "must not require gradients" in str(exc), str(exc)
            print(f"PASS recurrent_gdr(requires_grad rejected, {label})")
        else:
            raise AssertionError(
                f"recurrent_gdr({label}): requires_grad state was accepted")


def kda_inputs():
    B, T, H, HV, K, V = 2, 2, 2, 4, 128, 128
    dt = torch.bfloat16
    q = torch.randn(B, T, H, K, dtype=dt, device="npu")
    k = torch.randn(B, T, H, K, dtype=dt, device="npu")
    v = torch.randn(B, T, HV, V, dtype=dt, device="npu")
    g = -torch.rand(B, T, HV, K, dtype=torch.float32, device="npu") * 5 - 1e-3
    beta = torch.rand(B, T, HV, dtype=torch.float32, device="npu") * 0.8 + 0.1
    cu = torch.tensor([0, T, 2 * T], dtype=torch.int64, device="npu")
    torch.npu.synchronize()
    state = torch.zeros(B, HV, V, K, dtype=torch.float32, device="npu")
    kw = dict(cu_seqlens=cu, scale=K ** -0.5, layout="BSND",
              state_v_first=True, output_final_state=True)
    return q, k, v, g, beta, state, kw


def scenario_recurrent_kda():
    q, k, v, g, beta, state, kw = kda_inputs()
    # 1) inplace=True（默认）：写回调用方 tensor → bump。
    before = version_of(state)
    out = kda_public(q, k, v, g, beta, state, inplace_final_state=True, **kw)
    torch.npu.synchronize()
    check_bump("recurrent_kda(inplace=True positional)", state, before, 1)
    assert isinstance(out, tuple) and len(out) == 2
    # 2) flag 省略时默认 True，行为必须与显式 True 相同。
    before = version_of(state)
    kda_public(q, k, v, g, beta, state, **kw)
    torch.npu.synchronize()
    check_bump("recurrent_kda(flag omitted)", state, before, 1)
    # 3) inplace=False：算子写 scratch，调用方 tensor 不能被动 → 不 bump。
    before = version_of(state)
    out = kda_public(q, k, v, g, beta, state, inplace_final_state=False, **kw)
    torch.npu.synchronize()
    check_bump("recurrent_kda(inplace=False)", state, before, 0)
    _, final_state = out
    assert final_state.data_ptr() != state.data_ptr(), (
        "inplace=False must return a scratch final_state")
    print("PASS recurrent_kda(inplace=False returns scratch state)")
    # 4) requires_grad：inplace=True 拒绝；inplace=False 允许（不写回）。
    grad_state = state.clone().detach().requires_grad_(True)
    try:
        kda_public(q, k, v, g, beta, grad_state, inplace_final_state=True, **kw)
    except RuntimeError as exc:
        assert "must not require gradients" in str(exc), str(exc)
        print("PASS recurrent_kda(requires_grad rejected, inplace=True)")
    else:
        raise AssertionError("inplace=True accepted a requires_grad state")
    kda_public(q, k, v, g, beta, grad_state, inplace_final_state=False, **kw)
    torch.npu.synchronize()
    print("PASS recurrent_kda(requires_grad allowed, inplace=False)")


def scenario_values_still_match_ctypes():
    """快路径不得改变数值：同一输入下 thin public 与 ctypes 逐位一致。"""
    q, k, v, g, beta, state, kw = kda_inputs()
    ref = ct.npu_recurrent_kda(q, k, v, g, beta, state.clone(), **kw)
    got = kda_public(q, k, v, g, beta, state.clone(), **kw)
    torch.npu.synchronize()
    for i, (a, b) in enumerate(zip(ref, got)):
        if a is None or b is None:
            assert a is None and b is None, f"kda[{i}]: None mismatch"
            continue
        diff = float((a.float() - b.float()).abs().max().item())
        assert diff == 0.0, f"kda[{i}]: diff={diff}"
    print("PASS recurrent_kda(ctypes vs thin public values)")


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260910)
    scenario_recurrent_gdr()
    scenario_recurrent_kda()
    scenario_values_still_match_ctypes()
    print("ALL PASS: mutation contract regression")


if __name__ == "__main__":
    main()
