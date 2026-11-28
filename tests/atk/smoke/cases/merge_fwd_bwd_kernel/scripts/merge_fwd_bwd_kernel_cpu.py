"""merge_fwd_bwd_kernel 的纯 PyTorch 标杆。

与 kernel 一致的递推（不读调用方原来的 h）：

    h <- He_0
    h <- M_i @ h + He_i    i = 1 .. N-1

ag_hm 的每个 head 是 [128, 256]：前 128 列是 He，后 128 列是 M。
forward 的源 rank 是 rank-N .. rank-1；backward 是 rank+N .. rank+1。
S=N=1 且 rank=0 时源 rank 固定为 0。
state_v_first 只在最后把 [K, V] 转成 [V, K]。
BF16 先精确抬到 FP32 再算，写出时 round-to-nearest-even，对应 kernel 的 CAST_RINT。
"""

from __future__ import annotations

from typing import Any

import torch

K_DIM = 128
V_DIM = 128
VK_DIM = V_DIM + K_DIM
HE_SCALE = 0.05
M_SCALE = 0.02
H_SENTINEL = 3.0


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def tiling_key(dtype_name: str) -> int:
    name = str(dtype_name).lower()
    if name == "bf16":
        return 10
    if name == "fp32":
        return 30
    raise ValueError(f"unsupported dtype: {dtype_name}")


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out["dtype"] = str(spec.get("dtype", "bf16")).lower()
    out["S"] = int(spec["S"])
    out["HV"] = int(spec["HV"])
    n_key = "pre_or_post_num_ranks" if "pre_or_post_num_ranks" in spec else "N"
    out["pre_or_post_num_ranks"] = int(spec[n_key])
    out["rank"] = int(spec["rank"])
    out["forward"] = as_bool(spec.get("forward", True))
    out["state_v_first"] = as_bool(spec.get("state_v_first", False))
    out["ag_hm_noncontiguous"] = as_bool(spec.get("ag_hm_noncontiguous", False))
    out["seed"] = int(spec.get("seed", 20260817))
    out["tiling_key"] = tiling_key(out["dtype"])
    return out


def src_rank(rank: int, n_ranks: int, step: int, forward: bool, world: int) -> int:
    """与 kernel SrcRank 相同。step 是链上的下标 0 .. N-1。"""
    if world == 1 and n_ranks == 1:
        return 0
    if forward:
        return int(rank) - int(n_ranks) + int(step)
    return int(rank) + int(n_ranks) - int(step)


def _orig_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def make_ag_hm(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> torch.Tensor:
    """先在 FP32 采样，再量化到用例 dtype。DUT 和标杆吃的是同一份量化结果。"""
    del high_precision
    spec = normalize_spec(spec)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec["seed"])
    raw = torch.randn(
        (spec["S"], spec["HV"], K_DIM, VK_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    raw[..., :V_DIM] *= HE_SCALE
    raw[..., V_DIM:] *= M_SCALE
    data = raw.to(_orig_dtype(spec["dtype"])).to(device)
    if spec["ag_hm_noncontiguous"]:
        data = _noncontiguous_last_dim(data)
    return data


def make_h(spec: dict[str, Any], device: torch.device) -> torch.Tensor:
    """调用方缓冲。kernel 不读旧值；填一个有限哨兵，用来抓住误读。"""
    spec = normalize_spec(spec)
    return torch.full(
        (spec["HV"], K_DIM, K_DIM),
        H_SENTINEL,
        dtype=_orig_dtype(spec["dtype"]),
        device=device,
    )


def _noncontiguous_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    padded = torch.empty(
        *tensor.shape[:-1],
        tensor.shape[-1] * 2,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = padded[..., : tensor.shape[-1]]
    view.copy_(tensor)
    if view.is_contiguous():
        raise RuntimeError("ag_hm view is still contiguous")
    return view


def merge_reference(ag_hm: torch.Tensor, rank: int, n_ranks: int, forward: bool, state_v_first: bool) -> torch.Tensor:
    """在 ag_hm 的当前 dtype 上做链合并。ag_hm 可以不连续，这里会收成连续再读。"""
    data = ag_hm.contiguous()
    world = int(data.shape[0])
    step0 = src_rank(rank, n_ranks, 0, forward, world)
    state = data[step0, :, :, :V_DIM].contiguous()
    for step in range(1, int(n_ranks)):
        src = src_rank(rank, n_ranks, step, forward, world)
        block = data[src]
        he = block[:, :, :V_DIM]
        matrix = block[:, :, V_DIM:]
        state = torch.matmul(matrix, state) + he
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return state


def run_reference(spec: dict[str, Any], high_precision: bool = False) -> torch.Tensor:
    spec = normalize_spec(spec)
    ag_hm = make_ag_hm(spec, torch.device("cpu"))
    calc = torch.float64 if high_precision else torch.float32
    state = merge_reference(
        ag_hm.to(calc),
        spec["rank"],
        spec["pre_or_post_num_ranks"],
        spec["forward"],
        spec["state_v_first"],
    )
    if high_precision:
        return state
    return state.to(_orig_dtype(spec["dtype"]))


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    spec = normalize_spec(spec)
    return {
        "h": make_h(spec, device),
        "ag_hm": make_ag_hm(spec, device, high_precision),
        "pre_or_post_num_ranks": spec["pre_or_post_num_ranks"],
        "rank": spec["rank"],
        "forward": spec["forward"],
        "state_v_first": spec["state_v_first"],
    }


def assert_legal(spec: dict[str, Any]) -> None:
    spec = normalize_spec(spec)
    world, heads, n_ranks, rank = (spec["S"], spec["HV"], spec["pre_or_post_num_ranks"], spec["rank"])
    if not 1 <= world <= 1024:
        raise AssertionError(f"S out of range: {spec}")
    if not 1 <= heads <= 256:
        raise AssertionError(f"HV out of range: {spec}")
    if not 1 <= n_ranks <= world:
        raise AssertionError(f"N out of range: {spec}")
    if not 0 <= rank < world:
        raise AssertionError(f"rank out of range: {spec}")
    unit = world == 1 and n_ranks == 1 and rank == 0
    if not unit:
        if spec["forward"] and rank < n_ranks:
            raise AssertionError(f"forward requires rank >= N: {spec}")
        if (not spec["forward"]) and rank + n_ranks >= world:
            raise AssertionError(f"backward requires rank + N < S: {spec}")
    for step in range(n_ranks):
        src = src_rank(rank, n_ranks, step, spec["forward"], world)
        if not 0 <= src < world:
            raise AssertionError(f"src {src} out of range at step {step}: {spec}")


def self_check() -> None:
    """用独立展开核对源 rank、N=1 拷贝、转置，以及长链仍有限。"""
    world, heads, n_ranks, rank = 4, 2, 3, 3
    ag = torch.randn(world, heads, K_DIM, VK_DIM)
    ag[..., :V_DIM] *= HE_SCALE
    ag[..., V_DIM:] *= M_SCALE
    got = merge_reference(ag, rank, n_ranks, True, False)
    manual = ag[src_rank(rank, n_ranks, 0, True, world), :, :, :V_DIM].contiguous()
    for step in range(1, n_ranks):
        src = src_rank(rank, n_ranks, step, True, world)
        manual = torch.matmul(ag[src, :, :, V_DIM:], manual) + ag[src, :, :, :V_DIM]
    if not torch.allclose(got, manual):
        raise AssertionError("forward chain diverged from the explicit expansion")
    transposed = merge_reference(ag, rank, n_ranks, True, True)
    if not torch.allclose(transposed, manual.transpose(-1, -2).contiguous()):
        raise AssertionError("state_v_first is not a final transpose")
    copied = merge_reference(ag, 1, 1, True, False)
    if not torch.equal(copied, ag[0, :, :, :V_DIM].contiguous()):
        raise AssertionError("N=1 forward did not copy He at src=rank-N")
    unit = torch.randn(1, 2, K_DIM, VK_DIM)
    if not torch.equal(merge_reference(unit, 0, 1, False, False), unit[0, :, :, :V_DIM].contiguous()):
        raise AssertionError("unit world did not copy rank 0")
    backward = merge_reference(ag, 0, 3, False, False)
    if torch.allclose(got, backward):
        raise AssertionError("forward and backward unexpectedly matched")

    long_spec = {
        "dtype": "fp32",
        "S": 64,
        "HV": 1,
        "pre_or_post_num_ranks": 63,
        "rank": 63,
        "forward": True,
        "state_v_first": False,
        "seed": 20260817,
    }
    long_out = run_reference(long_spec, high_precision=False)
    if not torch.isfinite(long_out).all():
        raise AssertionError("N=63 chain produced a non-finite value")
    bf16_copy = run_reference(
        {
            "dtype": "bf16",
            "S": 2,
            "HV": 1,
            "pre_or_post_num_ranks": 1,
            "rank": 1,
            "forward": True,
            "state_v_first": True,
            "seed": 7,
        },
        high_precision=False,
    )
    if bf16_copy.dtype != torch.bfloat16 or bf16_copy.shape != (1, V_DIM, K_DIM):
        raise AssertionError(f"unexpected bf16 transpose store: {bf16_copy.dtype} {tuple(bf16_copy.shape)}")
    if not torch.isfinite(bf16_copy.float()).all():
        raise AssertionError("bf16 N=1 output is non-finite")
    contiguous = merge_reference(ag, rank, n_ranks, True, False)
    viewed = merge_reference(_noncontiguous_last_dim(ag), rank, n_ranks, True, False)
    if not torch.allclose(viewed, contiguous):
        raise AssertionError("noncontiguous ag_hm changed the reference")


if __name__ == "__main__":
    self_check()
    print("merge_fwd_bwd_kernel cpu reference: ok")
