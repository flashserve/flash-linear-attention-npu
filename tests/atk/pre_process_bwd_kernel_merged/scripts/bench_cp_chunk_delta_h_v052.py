#!/usr/bin/env python3
"""fla **0.5.2** 专用：H20 上测 ``fla/ops/cp/chunk_delta_h.py`` 的三个 CP kernel。

按 pip 上的 fla-core 0.5.2 源码逐字对齐（`fla/ops/cp/chunk_delta_h.py`）：

  * ``pre_process_fwd_kernel_merged(k, v, w, g, gk, bg, u, hm, cu_seqlens, T,
    H, HV, K, V, BT, BLOCK_SIZE, BK1, USE_G*, USE_GK*, USE_BG*, IS_VARLEN*, MULTI_SEQS)``
    （带 * 的由 ``@triton.heuristics`` 注入，调用方不传）
  * ``pre_process_bwd_kernel_merged(q, k, w, g, gk, do, dhm, dv, cu_seqlens, scale, T,
    H, HV, K, V, BT, BLOCK_SIZE, BK1, USE_G*, USE_GK*, IS_VARLEN*, USE_BG)``
  * ``merge_fwd_bwd_kernel(h, ag_hm, pre_or_post_num_ranks, rank, seq_offsets, init_offsets,
    h0_seq_ids, h0, HV, K, V, BV†, BK, FORWARD, INTRACARD_MODE, NUM_SEQ_ENTRIES,
    HAS_H0*, STATE_V_FIRST)``（† BV 来自 autotune config；HAS_H0 由 heuristics 注入）

0.5.2 与 main 的差别（脚本已按 0.5.2 处理）：

  * **没有** ``AFFINE_CHAIN_PRECISION`` / ``input_precision``（tf32x3 链精度是后续版本才有的）；
  * **没有** ``h_seq_idx`` / ``NUM_RANKS_ON_DEVICE``（graph replay 相关参数）；
  * **没有 zigzag CP 布局**：``fla/ops/cp/context.py`` 只按 ``part_len = T // world_size``
    做连续等分，所以本脚本默认只跑 ``contiguous``；传 ``zigzag`` 会被跳过并提示。
  * ``build_cp_context(cu_seqlens, group, conv1d_kernel_size=None, cu_seqlens_cpu=None)``。

统计口径：device 侧 ``torch.cuda.Event``（同 stream），warmup 后循环 ``--iters``（默认 20）次取
mean/median/min/max；每个 cp_size 按“每 rank 的本地分片”造输入（``T_rank = T_global/cp``）；
``merge`` 取最坏 rank（链长 ``cp-1``，含 fwd+bwd 两次 launch）。rank 并行 ⇒ pre_process 的
per-rank 时间就是端到端里这一步的时间。

默认矩阵：cp ∈ {2,4,8,16,64} × contiguous × 4 个模型档 × 3 kernel = **60 个测量点**。
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass

torch = triton = None
pre_process_fwd_kernel_merged = None
pre_process_bwd_kernel_merged = None
merge_fwd_bwd_kernel = None


def _load_runtime() -> None:
    global torch, triton
    global pre_process_fwd_kernel_merged, pre_process_bwd_kernel_merged
    global merge_fwd_bwd_kernel
    import torch as _torch
    import triton as _triton

    from fla.ops.cp.chunk_delta_h import (  # noqa: PLC0415
        merge_fwd_bwd_kernel as _merge,
        pre_process_bwd_kernel_merged as _bwd,
        pre_process_fwd_kernel_merged as _fwd,
    )
    torch, triton = _torch, _triton
    pre_process_fwd_kernel_merged = _fwd
    pre_process_bwd_kernel_merged = _bwd
    merge_fwd_bwd_kernel = _merge


# ---------------------------------------------------------------------------
# 场景档位（H = q/k heads，HV = v heads，K/V = head dim，T_global = 全局序列长度）
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    name: str
    H: int
    HV: int
    K: int
    V: int
    T_global: int
    chunk: int = 64
    gate: str = "g"       # g（GDN 标量 gate）| gk（KDA 逐 K gate）| bg（DPLR）| none
    note: str = ""


DEFAULT_PROFILES = {
    "gdn_gva_h16_hv32": Profile(
        "gdn_gva_h16_hv32", 16, 32, 128, 128, 131072,
        note="GDN/GVA 主流档（H=16, HV=32, T=128k）"),
    "gdn_h8_hv8": Profile(
        "gdn_h8_hv8", 8, 8, 128, 128, 131072,
        note="GDN 小头数档（H=HV=8, T=128k）"),
    "kda_h64_hv64": Profile(
        "kda_h64_hv64", 64, 64, 128, 128, 131072, gate="gk",
        note="KDA 主流档（H=HV=64, 逐 K gate, T=128k）"),
    "gdn_long_t256k": Profile(
        "gdn_long_t256k", 16, 32, 128, 128, 262144,
        note="长序列档（H=16, HV=32, T=256k）"),
}

KERNELS = ("pre_process_fwd_kernel_merged",
           "pre_process_bwd_kernel_merged",
           "merge_fwd_bwd_kernel")


@dataclass
class Result:
    cp_size: int
    layout: str
    profile: str
    kernel: str
    t_rank: int
    chain_ranks: int
    iters: int
    mean_ms: float = float("nan")
    median_ms: float = float("nan")
    min_ms: float = float("nan")
    max_ms: float = float("nan")
    error: str = ""


SKIPPED: list[str] = []


def mean(values):
    return sum(values) / len(values)


def scenario_shape(profile: Profile, cp_size: int, layout: str):
    """0.5.2 只有连续等分：T_rank = T_global / cp（必须整除）。"""

    if layout != "contiguous":
        return None
    if profile.T_global % cp_size != 0:
        return None
    return profile.T_global // cp_size


# ---------------------------------------------------------------------------
# 版本自适应（0.5.2 已对齐，这里只是兜底：签名变了也不会抛 KeyError）
# ---------------------------------------------------------------------------
def _unwrap_chain(kernel):
    chain, node = [], kernel
    for _ in range(6):
        chain.append(node)
        nxt = getattr(node, "fn", None)
        if nxt is None or any(nxt is item for item in chain):
            break
        node = nxt
    jit = next((item for item in chain if hasattr(item, "params")
                or hasattr(item, "arg_names")), chain[-1])
    return chain, jit


def kernel_signature(kernel):
    chain, jit = _unwrap_chain(kernel)
    for node in reversed(chain):
        try:
            params = inspect.signature(node).parameters
        except (TypeError, ValueError):
            continue
        required = {name for name, p in params.items()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                       inspect.Parameter.VAR_KEYWORD)}
        return set(params), required
    names = set(getattr(jit, "arg_names", ()) or ())
    return names, set(names)


def kernel_constexpr(kernel) -> set:
    _, jit = _unwrap_chain(kernel)
    return {getattr(p, "name") for p in (getattr(jit, "params", []) or [])
            if getattr(p, "name", None) and getattr(p, "is_constexpr", False)}


def kernel_is_autotuned(kernel) -> bool:
    chain, _ = _unwrap_chain(kernel)
    return any(hasattr(node, "configs") for node in chain)


def kernel_heuristics(kernel) -> set:
    """``@triton.heuristics`` 注入的参数名：它们在源码签名里，但调用方不能传。

    triton >= 3.3 把 ``@triton.heuristics({...})`` 存到 ``Heuristics.values``；
    更早的版本可能用 ``.heuristics``；两种属性名都读，dict 取 key、其他可迭代对象
    取其中的字符串。
    """

    chain, _ = _unwrap_chain(kernel)
    names = set()
    for node in chain:
        for attr in ("values", "heuristics"):
            # ``values`` 只有 Heuristics 包装器同时带 ``arg_names`` 时才是 heuristics 表
            if attr == "values" and not hasattr(node, "arg_names"):
                continue
            table = getattr(node, attr, None)
            if isinstance(table, dict):
                names.update(table)
            elif attr == "heuristics" and isinstance(table, (list, tuple, set, frozenset)):
                names.update(item for item in table if isinstance(item, str))
    return names


def kernel_autotune_names(kernel) -> set:
    """``@triton.autotune`` 的 config 提供的参数名（例如 merge 的 BV）：调用方不传。"""

    chain, _ = _unwrap_chain(kernel)
    names = set()
    for node in chain:
        for config in getattr(node, "configs", []) or []:
            names.update(getattr(config, "kwargs", {}) or {})
            if hasattr(config, "all_kwargs"):
                names.update(config.all_kwargs() or {})
    return names


IGNORED: set[str] = set()
RESTORED: set[str] = set()


def _filter_kwargs(kernel, kwargs: dict, label: str) -> dict:
    accepted, required = kernel_signature(kernel)
    supplied = kernel_heuristics(kernel) | kernel_autotune_names(kernel)
    # heuristics / autotune config 提供的参数：调用方不传，也不算“缺必需参数”
    payload = {key: value for key, value in kwargs.items()
               if key in accepted and key not in supplied}
    IGNORED.update(f"{label}:{key}" for key in set(kwargs) - set(payload) - supplied)
    # 兜底：签名要求、但 heuristics/autotune 探测没识别出来、而我们本来就有值的参数，
    # 直接补回去。``@triton.heuristics`` 的 run() 会用自己算出的值覆盖同名 kwargs，
    # 多传是安全的；少传才会真报“缺少必需参数”。
    for name in sorted(required - set(payload) - supplied):
        if name in kwargs and kwargs[name] is not None:
            payload[name] = kwargs[name]
            RESTORED.add(f"{label}:{name}")
    missing = sorted(required - set(payload) - supplied)
    if missing:
        raise TypeError(f"{label} 缺少必需参数 {missing}；已安装版本签名={sorted(accepted)}")
    return payload


_UNRECOGNISED_KWARG = re.compile(
    r"Keyword argument ([A-Za-z_][A-Za-z0-9_]*) was specified but unrecognised")
_QUOTED_NAME = re.compile(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]")


def _launch(kernel, grid, payload: dict, label: str, superset: dict | None = None) -> None:
    """用已安装 triton 的真实 binding 规则自适应 launch。

    签名过滤已覆盖 0.5.2 的情况，这里再兜一层，避免不同 triton 版本 binding 行为有
    出入时直接失败：

      * ``Keyword argument X was specified but unrecognised`` → 丢掉 X 重试；
      * 报缺参数 → 从 kwargs 超集里把该参数补回去重试。
    """

    kwargs = dict(payload)
    superset = dict(superset or {})
    for _ in range(12):
        try:
            kernel[grid](**kwargs)
            return
        except (KeyError, TypeError) as exc:
            message = str(exc)
            match = _UNRECOGNISED_KWARG.search(message)
            if match and match.group(1) in kwargs:
                dropped = match.group(1)
                kwargs.pop(dropped)
                IGNORED.add(f"{label}:{dropped}@launch")
                continue
            if "missing" in message or "缺少必需参数" in message:
                added = False
                for name in _QUOTED_NAME.findall(message):
                    if name in superset and name not in kwargs:
                        kwargs[name] = superset[name]
                        RESTORED.add(f"{label}:{name}@launch")
                        added = True
                if added:
                    continue
            raise
    raise RuntimeError(f"{label}: 自适应 launch 重试次数用尽，fla/triton 版本可能不兼容")


def describe_kernels() -> None:
    for name, kernel in (("pre_process_fwd_kernel_merged", pre_process_fwd_kernel_merged),
                         ("pre_process_bwd_kernel_merged", pre_process_bwd_kernel_merged),
                         ("merge_fwd_bwd_kernel", merge_fwd_bwd_kernel)):
        _, required = kernel_signature(kernel)
        heuristics = kernel_heuristics(kernel)
        print(f"  {name}: required={sorted(required)}")
        print(f"      constexpr={sorted(kernel_constexpr(kernel))}"
              f"  autotuned={kernel_is_autotuned(kernel)}"
              f"  heuristics={sorted(heuristics)}")


# ---------------------------------------------------------------------------
# 输入构造（0.5.2 的 launcher 约定：[B, T_rank, H/HV, K/V]，B=1）
# ---------------------------------------------------------------------------
def make_inputs(profile: Profile, t_rank: int, device, dtype, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    H, HV, K, V = profile.H, profile.HV, profile.K, profile.V

    def randn(*shape):
        data = torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.05
        return data.to(dtype).to(device)

    def gate(shape, dim, out_dtype):
        data = torch.rand(*shape, generator=gen, dtype=torch.float32) * 0.01 + 0.001
        return torch.cumsum(-data, dim=dim).to(out_dtype).to(device)

    tensors = {
        "k": randn(1, t_rank, H, K),
        "q": randn(1, t_rank, H, K),
        "u": randn(1, t_rank, HV, V),
        "w": randn(1, t_rank, HV, K),
        "do": randn(1, t_rank, HV, V),
        "dv": randn(1, t_rank, HV, V),
        "g": None, "gk": None, "bg": None,
    }
    if profile.gate == "g":
        tensors["g"] = gate((1, t_rank, HV), 1, torch.float32)
    elif profile.gate == "gk":
        tensors["gk"] = gate((1, t_rank, HV, K), 1, dtype)
    elif profile.gate == "bg":
        tensors["bg"] = randn(1, t_rank, H, K)
    # 0.5.2 的 fwd launcher：v=None 时用 u 顶替（GDN/KDA）；DPLR 才有独立 v
    tensors["v"] = None if profile.gate != "bg" else randn(1, t_rank, HV, V)
    tensors["hm"] = torch.zeros(HV, K, V + K, dtype=torch.float32, device=device)
    tensors["dhm"] = torch.zeros(HV, K, V + K, dtype=torch.float32, device=device)
    tensors["h"] = torch.zeros(HV, K, V, dtype=torch.float32, device=device)
    tensors["cu_seqlens"] = torch.tensor([0, t_rank], dtype=torch.int32, device=device)
    return tensors


# ---------------------------------------------------------------------------
# 三个 kernel 的 launch（kwargs 与 fla 0.5.2 的 launcher 逐字一致）
# ---------------------------------------------------------------------------
def launch_fwd(t: dict, profile: Profile) -> None:
    block = 32 if profile.K <= 64 else 64
    grid = (triton.cdiv(profile.V, block) + triton.cdiv(profile.K, block), profile.HV)
    kwargs = dict(
        k=t["k"], v=t["u"] if t["v"] is None else t["v"], w=t["w"], g=t["g"],
        gk=t["gk"], bg=t["bg"], u=t["u"], hm=t["hm"],
        cu_seqlens=t["cu_seqlens"][-2:], T=int(t["k"].shape[1]),
        H=profile.H, HV=profile.HV, K=profile.K, V=profile.V, BT=profile.chunk,
        BK1=triton.next_power_of_2(profile.K), BLOCK_SIZE=block, MULTI_SEQS=False,
        # 这 4 个在 0.5.2 由 @triton.heuristics 注入（会被过滤掉不传）；万一当前
        # triton 的 heuristics 表探测不到，就按签名补回来，值与本用例的输入一致。
        USE_G=t["g"] is not None, USE_GK=t["gk"] is not None,
        USE_BG=t["bg"] is not None, IS_VARLEN=True,
    )
    payload = _filter_kwargs(pre_process_fwd_kernel_merged, kwargs,
                            "pre_process_fwd_kernel_merged")
    _launch(pre_process_fwd_kernel_merged, grid, payload,
            "pre_process_fwd_kernel_merged", kwargs)


def launch_bwd(t: dict, profile: Profile, scale: float) -> None:
    block = 32 if profile.K <= 64 else 64
    grid = (triton.cdiv(profile.V, block) + triton.cdiv(profile.K, block), profile.HV)
    kwargs = dict(
        q=t["q"], k=t["k"] if t["bg"] is None else t["bg"], w=t["w"], g=t["g"],
        gk=t["gk"], do=t["do"], dhm=t["dhm"], dv=t["dv"],
        cu_seqlens=t["cu_seqlens"][:2], scale=scale, T=int(t["q"].shape[1]),
        H=profile.H, HV=profile.HV, K=profile.K, V=profile.V, BT=profile.chunk,
        BK1=triton.next_power_of_2(profile.K), BLOCK_SIZE=block,
        USE_BG=t["bg"] is not None,
        USE_G=t["g"] is not None, USE_GK=t["gk"] is not None, IS_VARLEN=True,
    )
    payload = _filter_kwargs(pre_process_bwd_kernel_merged, kwargs,
                            "pre_process_bwd_kernel_merged")
    _launch(pre_process_bwd_kernel_merged, grid, payload,
            "pre_process_bwd_kernel_merged", kwargs)


def launch_merge(t: dict, profile: Profile, num_ranks: int, rank: int, forward: bool,
                 ag_hm) -> None:
    def grid(meta):
        return (triton.cdiv(profile.V, meta["BV"]), profile.HV)

    kwargs = dict(
        h=t["h"], ag_hm=ag_hm, pre_or_post_num_ranks=num_ranks, rank=rank,
        seq_offsets=None, init_offsets=None, h0_seq_ids=None, h0=None,
        HV=profile.HV, K=profile.K, V=profile.V,
        BK=triton.next_power_of_2(profile.K), FORWARD=forward,
        INTRACARD_MODE=False, NUM_SEQ_ENTRIES=0, STATE_V_FIRST=False,
        HAS_H0=False,
    )
    payload = _filter_kwargs(merge_fwd_bwd_kernel, kwargs, "merge_fwd_bwd_kernel")
    _launch(merge_fwd_bwd_kernel, grid, payload, "merge_fwd_bwd_kernel", kwargs)


def launch_merge_worst(t: dict, profile: Profile, cp_size: int, ag_hm) -> None:
    """最坏 rank：fwd 链 rank=cp-1 合并 cp-1 个；bwd 链 rank=0 合并 cp-1 个。"""

    num_ranks = max(cp_size - 1, 0)
    if num_ranks == 0:
        return
    launch_merge(t, profile, num_ranks, cp_size - 1, True, ag_hm)
    launch_merge(t, profile, num_ranks, 0, False, ag_hm)


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------
def time_kernel(launch, iters: int, warmup: int):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return mean(samples), sorted(samples)[len(samples) // 2], min(samples), max(samples)


def run_scenario(cp_size: int, layout: str, profile: Profile, args, device) -> list[Result]:
    dtype = getattr(torch, args.dtype)
    t_rank = scenario_shape(profile, cp_size, layout)
    if t_rank is None:
        if layout != "contiguous":
            reason = f"cp={cp_size} {layout} {profile.name}: fla 0.5.2 没有 zigzag 布局（只支持连续等分）"
        else:
            reason = (f"cp={cp_size} {layout} {profile.name}: T_global={profile.T_global} "
                      f"不能被 {cp_size} 整除（0.5.2 用 T//world_size 等分）")
        SKIPPED.append(reason)
        print(f"[cp={cp_size:<3} {layout:<10} {profile.name:<18}] 跳过：{reason}", flush=True)
        return []
    tensors = make_inputs(profile, t_rank, device, dtype, seed=args.seed)
    ag_hm = torch.zeros(cp_size, profile.HV, profile.K, profile.V + profile.K,
                        dtype=torch.float32, device=device)
    launches = {
        "pre_process_fwd_kernel_merged": lambda: launch_fwd(tensors, profile),
        "pre_process_bwd_kernel_merged": lambda: launch_bwd(tensors, profile, args.scale),
        "merge_fwd_bwd_kernel": lambda: launch_merge_worst(tensors, profile, cp_size, ag_hm),
    }
    results = []
    for name in KERNELS:
        result = Result(cp_size=cp_size, layout=layout, profile=profile.name, kernel=name,
                        t_rank=t_rank,
                        chain_ranks=(max(cp_size - 1, 0)
                                     if name == "merge_fwd_bwd_kernel" else 0),
                        iters=args.iters)
        try:
            (result.mean_ms, result.median_ms, result.min_ms,
             result.max_ms) = time_kernel(launches[name], args.iters, args.warmup)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
            if args.traceback:
                traceback.print_exc()
        results.append(result)
        status = f"ERR {result.error}" if result.error else f"{result.mean_ms:8.4f} ms"
        print(f"[cp={cp_size:<3} {layout:<10} {profile.name:<18}] {name:<30} "
              f"T_rank={t_rank:<7} chain={result.chain_ranks:<3} {status}", flush=True)
    return results


def print_table(results) -> None:
    if not results:
        return
    print("\n" + "=" * 116)
    print(f"{'cp':>3} {'layout':<10} {'profile':<18} {'kernel':<30} {'T_rank':>7} "
          f"{'chain':>5} {'mean(ms)':>10} {'median':>10} {'min':>9} {'max':>9}")
    print("-" * 116)
    for r in sorted(results, key=lambda x: (x.cp_size, x.layout, x.profile, x.kernel)):
        if r.error:
            print(f"{r.cp_size:>3} {r.layout:<10} {r.profile:<18} {r.kernel:<30} "
                  f"{r.t_rank:>7} {r.chain_ranks or '-':>5}   ERROR: {r.error}")
        else:
            print(f"{r.cp_size:>3} {r.layout:<10} {r.profile:<18} {r.kernel:<30} "
                  f"{r.t_rank:>7} {r.chain_ranks or '-':>5} {r.mean_ms:>10.4f} "
                  f"{r.median_ms:>10.4f} {r.min_ms:>9.4f} {r.max_ms:>9.4f}")
    print("=" * 116)
    print("pre_process_*：每 rank 本地耗时（ranks 并行 ⇒ 端到端里这一步≈该值）；"
          "merge：最坏 rank（链长=cp-1），含 fwd+bwd 两次 launch。")

    buckets = {}
    for r in results:
        if r.error:
            continue
        buckets.setdefault((r.cp_size, r.layout, r.profile), {})[r.kernel] = r.mean_ms
    if buckets:
        print("\n每 rank 合计（fwd 摘要 + bwd 摘要 + merge(fwd+bwd)）：")
        print(f"{'cp':>3} {'layout':<10} {'profile':<18} {'fwd(ms)':>9} {'bwd(ms)':>9} "
              f"{'merge(ms)':>10} {'total(ms)':>10}")
        for key in sorted(buckets):
            parts = buckets[key]
            fwd = parts.get("pre_process_fwd_kernel_merged", float("nan"))
            bwd = parts.get("pre_process_bwd_kernel_merged", float("nan"))
            mrg = parts.get("merge_fwd_bwd_kernel", 0.0)
            print(f"{key[0]:>3} {key[1]:<10} {key[2]:<18} {fwd:>9.4f} {bwd:>9.4f} "
                  f"{mrg:>10.4f} {fwd + bwd + mrg:>10.4f}")


def run_dist(args, profiles) -> int:
    """--dist：torchrun 起 cp_size 进程，走 0.5.2 的真实 CP 路径（含 all-gather + merge）。"""

    import torch.distributed as dist

    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
    )
    from fla.ops.cp.context import build_cp_context

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    profile = profiles[args.dist_profile]
    dtype = getattr(torch, args.dtype)
    t_rank = scenario_shape(profile, world, "contiguous")
    if t_rank is None:
        if rank == 0:
            print(f"T_global={profile.T_global} 必须能被 {world} 整除（0.5.2 的等分约束）",
                  file=sys.stderr)
        dist.destroy_process_group()
        return 2
    tensors = make_inputs(profile, t_rank, device, dtype, seed=args.seed)
    cu_global = torch.tensor([0, profile.T_global], dtype=torch.int32, device=device)
    # 0.5.2: build_cp_context(cu_seqlens, group, conv1d_kernel_size=None, cu_seqlens_cpu=None)
    ctx_kwargs = _filter_kwargs(build_cp_context,
                               dict(cu_seqlens=cu_global, group=dist.group.WORLD),
                               "build_cp_context")
    context = build_cp_context(**ctx_kwargs)
    cu_local = context.cu_seqlens

    def call_fwd():
        kwargs = dict(g=tensors["g"], gk=tensors["gk"], bg=tensors["bg"],
                      v=tensors["v"], chunk_size=profile.chunk, cu_seqlens=cu_local,
                      context=context)
        accepted, _ = kernel_signature(chunk_gated_delta_rule_fwd_h_pre_process)
        chunk_gated_delta_rule_fwd_h_pre_process(
            tensors["k"], tensors["w"], tensors["u"],
            **{k: v for k, v in kwargs.items() if k in accepted})

    def call_bwd():
        kwargs = dict(g=tensors["g"], gk=tensors["gk"], bg=tensors["bg"],
                      scale=args.scale, cu_seqlens=cu_local, context=context,
                      chunk_size=profile.chunk)
        accepted, _ = kernel_signature(chunk_gated_delta_rule_bwd_dhu_pre_process)
        chunk_gated_delta_rule_bwd_dhu_pre_process(
            tensors["q"], tensors["k"], tensors["w"], tensors["do"], tensors["dv"],
            **{k: v for k, v in kwargs.items() if k in accepted})

    summary = {}
    for name, fn in (("pre_process_fwd(+merge)", call_fwd),
                     ("pre_process_bwd(+merge)", call_bwd)):
        samples = []
        for step in range(args.warmup + args.iters):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            if step >= args.warmup:
                samples.append(start.elapsed_time(end))
        gathered = [None] * world
        dist.all_gather_object(gathered, samples)
        if rank == 0:
            per_rank = [mean(s) for s in gathered]
            summary[name] = {"per_rank_mean_ms": per_rank,
                             "max_rank_ms": max(per_rank), "mean_ms": mean(per_rank)}
            print(f"[dist cp={world} contiguous {profile.name}] {name:<24} "
                  f"per-rank mean={summary[name]['mean_ms']:.4f} ms  "
                  f"rank-max={summary[name]['max_rank_ms']:.4f} ms  (T_rank={t_rank})")
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, f"{args.out_prefix}_dist_cp{world}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"cp_size": world, "layout": "contiguous", "profile": profile.name,
                       "t_rank": t_rank, "iters": args.iters, "summary": summary},
                      handle, ensure_ascii=False, indent=2)
        print(f"JSON: {path}")
    dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cp-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 64])
    parser.add_argument("--layouts", nargs="+", default=["contiguous"],
                        choices=["contiguous", "zigzag"],
                        help="0.5.2 只有 contiguous；传 zigzag 会被跳过并提示")
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--custom", action="append", default=[],
                        metavar="NAME,H,HV,K,V,T[,GATE]",
                        help="自定义模型档，可重复；GATE ∈ g|gk|bg|none（默认 g）")
    parser.add_argument("--t-global", type=int, default=0,
                        help="覆盖所有 profile 的全局序列长度（0 = 用 profile 自带值）")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default="fla_cp_chunk_delta_h_v052")
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--traceback", action="store_true")
    parser.add_argument("--no-signature", action="store_true",
                        help="不打印 kernel 签名（默认打印）")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--dist-profile", default=list(DEFAULT_PROFILES)[0])
    args = parser.parse_args()

    for spec in args.custom:
        parts = spec.split(",")
        if len(parts) not in (6, 7):
            print(f"--custom 格式应为 NAME,H,HV,K,V,T[,GATE]，收到：{spec}", file=sys.stderr)
            return 2
        name, h, hv, k, v, t = parts[:6]
        gate = parts[6] if len(parts) == 7 else "g"
        if gate not in ("g", "gk", "bg", "none"):
            print(f"未知 gate {gate}（可选 g/gk/bg/none）", file=sys.stderr)
            return 2
        DEFAULT_PROFILES[name] = Profile(name, int(h), int(hv), int(k), int(v), int(t),
                                         gate=gate, note="自定义档")
        args.profiles.append(name)

    profiles = {}
    for name in args.profiles:
        if name not in DEFAULT_PROFILES:
            print(f"未知 profile {name}，可选：{', '.join(DEFAULT_PROFILES)}", file=sys.stderr)
            return 2
        profile = DEFAULT_PROFILES[name]
        if args.t_global > 0:
            profile = Profile(**{**profile.__dict__, "T_global": args.t_global})
        profiles[name] = profile

    if args.dry_run:
        print("scenario matrix（fla 0.5.2：只支持 contiguous 等分）:")
        skipped = 0
        for cp in args.cp_sizes:
            for layout in args.layouts:
                for profile in profiles.values():
                    t_rank = scenario_shape(profile, cp, layout)
                    if t_rank is None:
                        skipped += 1
                        note = ("跳过（0.5.2 没有 zigzag）" if layout != "contiguous"
                                else f"跳过（T_global 需被 {cp} 整除）")
                    else:
                        note = "整除 OK"
                    print(f"  cp={cp:<3} {layout:<10} {profile.name:<18} "
                          f"T_global={profile.T_global:<7} T_rank={str(t_rank):<8} "
                          f"chain_ranks={max(cp - 1, 0):<3} {note}")
        scenarios = len(args.cp_sizes) * len(args.layouts) * len(profiles)
        print(f"\nkernels({len(KERNELS)}): {', '.join(KERNELS)}")
        print(f"每个场景 warmup={args.warmup} + iters={args.iters}；场景 "
              f"{scenarios - skipped}/{scenarios} 可测（跳过 {skipped}），"
              f"测量点 {scenarios * len(KERNELS)} 个；dtype={args.dtype}")
        return 0

    _load_runtime()
    if args.dist:
        return run_dist(args, profiles)

    if not torch.cuda.is_available():
        print("没有可用 CUDA 设备（请在 H20 上运行；--dry-run 可先看场景矩阵）", file=sys.stderr)
        return 3
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    try:
        import fla
        fla_ver = f"{getattr(fla, '__version__', '?')} @ {os.path.dirname(fla.__file__)}"
    except Exception:
        fla_ver = "unknown"
    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"{torch.cuda.get_device_name(device)} | iters={args.iters} warmup={args.warmup} "
          f"dtype={args.dtype}\nfla: {fla_ver}", flush=True)
    if not args.no_signature:
        print("已安装版本 kernel 签名（应与 0.5.2 一致）：")
        describe_kernels()

    results = []
    for cp in args.cp_sizes:
        for layout in args.layouts:
            for profile in profiles.values():
                results.extend(run_scenario(cp, layout, profile, args, device))

    print_table(results)
    n_scenarios = len(args.cp_sizes) * len(args.layouts) * len(profiles)
    n_points = n_scenarios * len(KERNELS)
    ok_points = sum(1 for r in results if not r.error)
    print(f"\n覆盖：场景 {n_scenarios - len(SKIPPED)}/{n_scenarios}"
          f"（跳过 {len(SKIPPED)}）｜测量点 {ok_points}/{n_points} 有效"
          f"（失败 {sum(1 for r in results if r.error)}）")
    for reason in SKIPPED:
        print(f"  跳过：{reason}")
    if IGNORED:
        print(f"本版本不支持的参数已自动忽略：{sorted(IGNORED)}")
    if RESTORED:
        print(f"heuristics/autotune 未识别的参数已按签名补回：{sorted(RESTORED)}")

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f"{args.out_prefix}_results.csv")
    json_path = os.path.join(args.save_dir, f"{args.out_prefix}_results.json")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("cp_size,layout,profile,kernel,t_rank,chain_ranks,iters,"
                     "mean_ms,median_ms,min_ms,max_ms,error\n")
        for r in results:
            handle.write(f"{r.cp_size},{r.layout},{r.profile},{r.kernel},{r.t_rank},"
                         f"{r.chain_ranks},{r.iters},{r.mean_ms:.6f},{r.median_ms:.6f},"
                         f"{r.min_ms:.6f},{r.max_ms:.6f},\"{r.error}\"\n")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({
            "env": {"torch": torch.__version__, "triton": triton.__version__,
                    "device": torch.cuda.get_device_name(device), "fla": fla_ver,
                    "fla_expected": "0.5.2", "iters": args.iters, "warmup": args.warmup,
                    "dtype": args.dtype, "ignored_params": sorted(IGNORED),
                    "restored_params": sorted(RESTORED),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "profiles": {name: profile.__dict__ for name, profile in profiles.items()},
            "coverage": {"scenarios_expected": n_scenarios,
                         "scenarios_skipped": len(SKIPPED),
                         "points_expected": n_points, "points_ok": ok_points,
                         "skipped": list(SKIPPED)},
            "results": [r.__dict__ for r in results],
        }, handle, ensure_ascii=False, indent=2)
    print(f"\nCSV : {csv_path}\nJSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
