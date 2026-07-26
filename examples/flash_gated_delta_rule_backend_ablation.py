#!/usr/bin/env python3
"""Run the 100-layer stress case with per-stage fla_npu/fla-org selection.

This script reuses every model/stress option from
``flash_gated_delta_rule_100layer_stress.py`` and adds the following switches:

    --fla-org-root=/path/to/flash-linear-attention
    --all-org
    --org-stages=kkt,solve
    --backend-kkt=fla-org
    --backend-solve=fla-npu

Run ``python3 this_file.py --backend-help`` for the complete stage list.

Gate-domain contract
--------------------
The boundary shared by all stages is the natural-log cumulative gate. fla_npu
kernels receive that tensor unchanged and use exp(). fla-org Triton-Ascend
kernels receive ``g_nat / ln(2)`` and use exp2(). The conversion is made only
at an adapter boundary, so a stage never interprets an exp2-domain tensor as
an exp-domain tensor (or the reverse). A mixed invocation creates at most one
converted gate tensor in forward and one in backward, regardless of how many
fla-org stages consume it.

The fla-org forward-output adapter intentionally avoids the upstream
``chunk_offsets[-1].item()`` call. Its total chunk count comes from the
host-prepared chunk index length, preserving the stress script's asynchronous
submission behavior.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import triton

import flash_gated_delta_rule_100layer_stress as stress


FLA_NPU = "fla-npu"
FLA_ORG = "fla-org"
RCP_LN2 = 1.4426950408889634074

# These are mathematical stage boundaries. In particular, fla_npu exposes WY
# backward as two AscendC calls while fla-org uses several Triton launches with
# shared FP32 scratch. Splitting that boundary further would not be equivalent.
STAGES = (
    "l2norm_fwd",
    "cumsum_fwd",
    "kkt",
    "solve",
    "recompute_wu",
    "fwd_h",
    "fwd_o",
    "dv_local",
    "bwd_dhu",
    "dqkwg",
    "wy_bwd",
    "cumsum_bwd",
    "l2norm_bwd",
)
FORWARD_GATE_STAGES = ("kkt", "recompute_wu", "fwd_h", "fwd_o")
BACKWARD_GATE_STAGES = (
    "recompute_wu",
    "fwd_h",
    "dv_local",
    "bwd_dhu",
    "dqkwg",
    "wy_bwd",
)


@dataclass(frozen=True)
class BackendConfig:
    stages: dict[str, str]
    fla_org_root: Optional[Path]

    def uses_org(self, stage: str) -> bool:
        return self.stages[stage] == FLA_ORG

    @property
    def any_org(self) -> bool:
        return any(backend == FLA_ORG for backend in self.stages.values())


_CONFIG: BackendConfig
_ORG: Optional[SimpleNamespace] = None


def _backend_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fla-org-root",
        default=os.environ.get("FLA_ORG_ROOT", ""),
        help="Checkout root of fla-org/flash-linear-attention.",
    )
    parser.add_argument(
        "--all-org",
        action="store_true",
        help="Select fla-org for every switchable GDR stage.",
    )
    parser.add_argument(
        "--org-stages",
        default="",
        help=f"Comma-separated fla-org stages. Valid names: {','.join(STAGES)}",
    )
    parser.add_argument(
        "--backend-help",
        action="store_true",
        help="Print backend-ablation options and exit.",
    )
    for stage in STAGES:
        parser.add_argument(
            f"--backend-{stage.replace('_', '-')}",
            choices=(FLA_NPU, FLA_ORG),
            default=None,
            help=f"Implementation used by {stage}.",
        )
    return parser


def _parse_backend_config(argv: list[str]) -> tuple[BackendConfig, list[str]]:
    parser = _backend_parser()
    args, remaining = parser.parse_known_args(argv[1:])
    if args.backend_help:
        parser.print_help()
        raise SystemExit(0)

    selected = {stage: FLA_ORG if args.all_org else FLA_NPU for stage in STAGES}
    if args.org_stages:
        requested = {item.strip().replace("-", "_") for item in args.org_stages.split(",") if item.strip()}
        unknown = sorted(requested.difference(STAGES))
        if unknown:
            raise ValueError(
                f"unknown --org-stages values {unknown}; valid stages are {list(STAGES)}"
            )
        for stage in requested:
            selected[stage] = FLA_ORG

    for stage in STAGES:
        explicit = getattr(args, f"backend_{stage}")
        if explicit is not None:
            selected[stage] = explicit

    root = Path(args.fla_org_root).expanduser().resolve() if args.fla_org_root else None
    config = BackendConfig(stages=selected, fla_org_root=root)
    return config, [argv[0], *remaining]


def _load_org_ops(root: Path) -> SimpleNamespace:
    marker = root / "fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py"
    if not marker.is_file():
        raise ValueError(
            f"--fla-org-root does not look like fla-org/flash-linear-attention: {root}"
        )

    loaded_fla = sys.modules.get("fla")
    if loaded_fla is not None:
        origin = Path(getattr(loaded_fla, "__file__", "")).resolve()
        if not origin.is_relative_to(root):
            # Importing the base stress module has already captured direct
            # references to fla_npu's Triton functions. Remove only their
            # module-cache entries so fla-org can occupy the same ``fla``
            # namespace; the captured function objects and globals remain
            # alive and continue to implement the fla_npu side of each switch.
            for module_name in tuple(sys.modules):
                if module_name == "fla" or module_name.startswith("fla."):
                    del sys.modules[module_name]

    sys.path.insert(0, str(root))
    importlib.invalidate_caches()

    from fla.modules.backends.triton_ascend.l2norm import (  # type: ignore[import-not-found]
        l2norm_bwd_npu,
        l2norm_fwd_npu,
    )
    from fla.ops.common.backends.triton_ascend import chunk_o  # type: ignore[import-not-found]
    from fla.ops.common.backends.triton_ascend.chunk_delta_h import (  # type: ignore[import-not-found]
        chunk_gated_delta_rule_bwd_dhu_npu,
        chunk_gated_delta_rule_fwd_h_npu,
    )
    from fla.ops.common.backends.triton_ascend.chunk_o import (  # type: ignore[import-not-found]
        chunk_bwd_dqkwg_npu,
        chunk_bwd_dv_local_npu,
    )
    from fla.ops.common.backends.triton_ascend.chunk_scaled_dot_kkt import (  # type: ignore[import-not-found]
        chunk_scaled_dot_kkt_fwd_npu,
    )
    from fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast import (  # type: ignore[import-not-found]
        prepare_wy_repr_bwd_npu,
        recompute_w_u_fwd_npu,
    )
    from fla.ops.utils.backends.triton_ascend.cumsum import (  # type: ignore[import-not-found]
        chunk_local_cumsum_npu,
    )
    from fla.ops.utils.backends.triton_ascend.solve_tril import (  # type: ignore[import-not-found]
        solve_tril_npu,
    )
    from fla.ops.utils.index import prepare_chunk_offsets  # type: ignore[import-not-found]

    return SimpleNamespace(
        l2norm_fwd=l2norm_fwd_npu,
        l2norm_bwd=l2norm_bwd_npu,
        cumsum=chunk_local_cumsum_npu,
        kkt=chunk_scaled_dot_kkt_fwd_npu,
        solve=solve_tril_npu,
        recompute_wu=recompute_w_u_fwd_npu,
        fwd_h=chunk_gated_delta_rule_fwd_h_npu,
        fwd_o_module=chunk_o,
        dv_local=chunk_bwd_dv_local_npu,
        bwd_dhu=chunk_gated_delta_rule_bwd_dhu_npu,
        dqkwg=chunk_bwd_dqkwg_npu,
        wy_bwd=prepare_wy_repr_bwd_npu,
        prepare_chunk_offsets=prepare_chunk_offsets,
    )


def _org() -> SimpleNamespace:
    if _ORG is None:
        raise RuntimeError("fla-org operators were not loaded")
    return _ORG


def _to_ntd4(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2).contiguous()


def _to_head_first4(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2).contiguous()


def _chunk_state_to_ntd(tensor: torch.Tensor) -> torch.Tensor:
    # [B, H, total_chunks, K, V] -> [B, total_chunks, H, K, V]
    return tensor.transpose(1, 2).contiguous()


def _chunk_state_to_head_first(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2).contiguous()


def _gate_for_org(g_nat: torch.Tensor) -> torch.Tensor:
    """Convert natural-log cumulative gates to fla-org's exp2 domain."""
    return g_nat.mul(RCP_LN2)


def _require_org_gate(g_log2: Optional[torch.Tensor]) -> torch.Tensor:
    if g_log2 is None:
        raise RuntimeError("missing fla-org log2-domain gate")
    return g_log2


def _chunk_tensor(
    chunk_indices: dict[str, torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    result = chunk_indices.get(str(chunk_size))
    if result is None:
        raise ValueError(f"missing tensor chunk indices for chunk_size={chunk_size}")
    return result


def _chunk_list(
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> list[int]:
    result = chunk_indices_list.get(str(chunk_size))
    if result is None:
        raise ValueError(f"missing host chunk indices for chunk_size={chunk_size}")
    return result


def _expect(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if tuple(tensor.shape) != shape:
        raise RuntimeError(f"{name} shape mismatch: got {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} dtype mismatch: got {tensor.dtype}, expected {dtype}")
    return tensor


def _l2norm_fwd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _CONFIG.uses_org("l2norm_fwd"):
        return _org().l2norm_fwd(x)
    return stress.l2norm_fwd(x)


def _l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    if _CONFIG.uses_org("l2norm_bwd"):
        return _org().l2norm_bwd(y, rstd, dy)
    return stress.l2norm_bwd(y, rstd, dy)


def _cumsum(
    g: torch.Tensor,
    *,
    reverse: bool,
    cu_seqlens: torch.Tensor,
    chunk_indices: dict[str, torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    stage = "cumsum_bwd" if reverse else "cumsum_fwd"
    if _CONFIG.uses_org(stage):
        # The shared boundary is natural-log. Do not use RCP_LN2 here.
        return _org().cumsum(
            g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=None,
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
            head_first=False,
            output_dtype=torch.float32,
        )
    return stress.chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
        output_dtype=torch.float32,
    )


def _kkt(
    k_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    beta_ntd: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    chunk_indices: dict[str, torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    indices = _chunk_tensor(chunk_indices, chunk_size)
    if _CONFIG.uses_org("kkt"):
        return _org().kkt(
            k=_to_ntd4(k_hf),
            g=_require_org_gate(g_log2_ntd),
            beta=beta_ntd,
            cu_seqlens=cu_seqlens,
            chunk_indices=indices,
            chunk_size=chunk_size,
            output_dtype=torch.float32,
        )
    return stress.chunk_scaled_dot_kkt_fwd(
        k=k_hf,
        g=g_nat_ntd,
        beta=beta_ntd,
        cu_seqlens=cu_seqlens,
        chunk_indices=indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )


def _solve(
    A_ntd: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> torch.Tensor:
    if _CONFIG.uses_org("solve"):
        return _org().solve(
            A=A_ntd,
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
            output_dtype=output_dtype,
        )
    return stress.solve_tri_ascendc(
        A_ntd,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        output_dtype=output_dtype,
    )


def _recompute_wu(
    k_hf: torch.Tensor,
    v_hf: torch.Tensor,
    beta_ntd: torch.Tensor,
    A_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _CONFIG.uses_org("recompute_wu"):
        w_ntd, u_ntd = _org().recompute_wu(
            k=_to_ntd4(k_hf),
            v=_to_ntd4(v_hf),
            beta=beta_ntd,
            A=_to_ntd4(A_hf),
            g=_require_org_gate(g_log2_ntd),
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        )
        return _to_head_first4(w_ntd), _to_head_first4(u_ntd)

    return stress.recompute_w_u(
        k_hf,
        v_hf,
        beta_ntd.transpose(1, 2).contiguous().float(),
        A_hf,
        g_nat_ntd.transpose(1, 2).contiguous(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )


def _fwd_h(
    k_hf: torch.Tensor,
    w_hf: torch.Tensor,
    u_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _CONFIG.uses_org("fwd_h"):
        h_ntd, v_new_ntd, _ = _org().fwd_h(
            k=_to_ntd4(k_hf),
            w=_to_ntd4(w_hf),
            u=_to_ntd4(u_hf),
            g=_require_org_gate(g_log2_ntd),
            gk=None,
            initial_state=None,
            output_final_state=False,
            chunk_size=chunk_size,
            save_new_value=True,
            state_v_first=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        )
        return _chunk_state_to_head_first(h_ntd), _to_head_first4(v_new_ntd)

    h_hf, v_new_hf, _ = stress.ascendc_chunk_gated_delta_rule_fwd_h(
        k_hf,
        w_hf,
        u_hf,
        g=g_nat_ntd.transpose(1, 2).contiguous(),
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    return h_hf, v_new_hf


def _org_fwd_o_async(
    q_ntd: torch.Tensor,
    k_ntd: torch.Tensor,
    v_ntd: torch.Tensor,
    h_ntd: torch.Tensor,
    g_log2_ntd: torch.Tensor,
    *,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Equivalent fla-org wrapper without its varlen Tensor.item() host wait."""
    module = _org().fwd_o_module
    B, T, H, K = q_ntd.shape
    V, HV = v_ntd.shape[-1], v_ntd.shape[2]
    BT = chunk_size
    N = len(cu_seqlens) - 1
    chunk_offsets = _org().prepare_chunk_offsets(cu_seqlens, BT)
    total_chunks = len(chunk_indices)
    BV = 128
    NV = triton.cdiv(V, BV)
    num_core = module.get_npu_properties()["num_aicore"]
    task_num = NV * HV * total_chunks
    output = torch.empty_like(v_ntd)
    g_head = g_log2_ntd.transpose(1, 2).contiguous()

    module.chunk_fwd_kernel_o_npu[(num_core,)](
        q=q_ntd,
        k=k_ntd,
        v=v_ntd,
        h=h_ntd,
        g=g_head,
        g_gamma=None,
        o=output,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N=N,
        total_chunks=total_chunks,
        task_num=task_num,
        num_core=num_core,
        BT=BT,
        BV=BV,
        STATE_V_FIRST=False,
    )
    return output


def _fwd_o(
    q_hf: torch.Tensor,
    k_hf: torch.Tensor,
    v_new_hf: torch.Tensor,
    h_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    *,
    scale: float,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> torch.Tensor:
    if _CONFIG.uses_org("fwd_o"):
        return _org_fwd_o_async(
            _to_ntd4(q_hf),
            _to_ntd4(k_hf),
            _to_ntd4(v_new_hf),
            _chunk_state_to_ntd(h_hf),
            _require_org_gate(g_log2_ntd),
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
            chunk_size=chunk_size,
        )

    output_hf = stress.ascendc_chunk_fwd_o(
        q_hf,
        k_hf,
        v_new_hf,
        h_hf,
        scale,
        g=g_nat_ntd.transpose(1, 2).contiguous(),
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    return _to_ntd4(output_hf)


def _forward_gdr(
    q_hf: torch.Tensor,
    k_hf: torch.Tensor,
    v_hf: torch.Tensor,
    g_raw_ntd: torch.Tensor,
    beta_ntd: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, K = k_hf.shape
    V = v_hf.shape[-1]
    total_chunks = len(_chunk_tensor(chunk_indices, chunk_size))

    g_nat_ntd = _cumsum(
        g_raw_ntd,
        reverse=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    _expect("g_cumsum", g_nat_ntd, shape=(B, T, H), dtype=torch.float32)
    g_log2_ntd = (
        _gate_for_org(g_nat_ntd)
        if any(_CONFIG.uses_org(stage) for stage in FORWARD_GATE_STAGES)
        else None
    )

    A_ntd = _kkt(
        k_hf,
        g_nat_ntd,
        g_log2_ntd,
        beta_ntd,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    _expect("A_kkt", A_ntd, shape=(B, T, H, chunk_size), dtype=torch.float32)

    A_ntd = _solve(
        A_ntd,
        output_dtype=k_hf.dtype,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    _expect("A_solve", A_ntd, shape=(B, T, H, chunk_size), dtype=k_hf.dtype)
    A_hf = _to_head_first4(A_ntd)

    w_hf, u_hf = _recompute_wu(
        k_hf,
        v_hf,
        beta_ntd,
        A_hf,
        g_nat_ntd,
        g_log2_ntd,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    _expect("w", w_hf, shape=(B, H, T, K), dtype=k_hf.dtype)
    _expect("u", u_hf, shape=(B, H, T, V), dtype=v_hf.dtype)

    h_hf, v_new_hf = _fwd_h(
        k_hf,
        w_hf,
        u_hf,
        g_nat_ntd,
        g_log2_ntd,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    _expect(
        "h",
        h_hf,
        shape=(B, H, total_chunks, K, V),
        dtype=k_hf.dtype,
    )
    _expect("v_new", v_new_hf, shape=(B, H, T, V), dtype=v_hf.dtype)

    output_ntd = _fwd_o(
        q_hf,
        k_hf,
        v_new_hf,
        h_hf,
        g_nat_ntd,
        g_log2_ntd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    _expect("output", output_ntd, shape=(B, T, H, V), dtype=v_hf.dtype)
    return g_nat_ntd, output_ntd, A_hf


def _dv_local(
    q_hf: torch.Tensor,
    k_hf: torch.Tensor,
    do_ntd: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    A_hf: torch.Tensor,
    *,
    scale: float,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> torch.Tensor:
    if _CONFIG.uses_org("dv_local"):
        result_ntd = _org().dv_local(
            q=_to_ntd4(q_hf),
            k=_to_ntd4(k_hf),
            do=do_ntd,
            g=_require_org_gate(g_log2_ntd),
            g_gamma=None,
            A=_to_ntd4(A_hf),
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        )
        return _to_head_first4(result_ntd)

    return stress.ascendc_chunk_bwd_dv_local(
        q_hf,
        k_hf,
        _to_head_first4(do_ntd),
        g_nat_ntd.transpose(1, 2).contiguous(),
        scale,
        chunk_size,
        g_gamma=None,
        A=A_hf,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )


def _bwd_dhu(
    q_hf: torch.Tensor,
    k_hf: torch.Tensor,
    w_hf: torch.Tensor,
    h_hf: torch.Tensor,
    do_ntd: torch.Tensor,
    dv_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    *,
    scale: float,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if _CONFIG.uses_org("bwd_dhu"):
        dh_ntd, dh0, dv_ntd = _org().bwd_dhu(
            q=_to_ntd4(q_hf),
            k=_to_ntd4(k_hf),
            w=_to_ntd4(w_hf),
            do=do_ntd,
            dv=_to_ntd4(dv_hf),
            g=_require_org_gate(g_log2_ntd),
            gk=None,
            h0=None,
            dht=None,
            scale=scale,
            state_v_first=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        )
        return (
            _chunk_state_to_head_first(dh_ntd),
            dh0,
            _to_head_first4(dv_ntd),
        )

    # Match the original stress script byte-for-byte in diagnostic mode.
    diagnostic_h0 = (
        torch.zeros_like(h_hf) if stress._STAGE_GRAD_RECORDS is not None else None
    )
    return stress.ascendc_chunk_gated_delta_rule_bwd_dhu(
        q_hf,
        k_hf,
        w_hf,
        _to_head_first4(do_ntd),
        dv_hf,
        scale,
        chunk_size,
        g=g_nat_ntd.transpose(1, 2).contiguous(),
        gK=None,
        h0=diagnostic_h0,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )


def _dqkwg(
    q_hf: torch.Tensor,
    k_hf: torch.Tensor,
    v_new_hf: torch.Tensor,
    w_hf: torch.Tensor,
    h_hf: torch.Tensor,
    do_ntd: torch.Tensor,
    dh_hf: torch.Tensor,
    dv_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    *,
    scale: float,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _CONFIG.uses_org("dqkwg"):
        dq, dk, dw, dg = _org().dqkwg(
            q=_to_ntd4(q_hf),
            k=_to_ntd4(k_hf),
            v=_to_ntd4(v_new_hf),
            do=do_ntd,
            h=_chunk_state_to_ntd(h_hf),
            dh=_chunk_state_to_ntd(dh_hf),
            w=_to_ntd4(w_hf),
            g=_require_org_gate(g_log2_ntd),
            g_gamma=None,
            dv=_to_ntd4(dv_hf),
            scale=scale,
            state_v_first=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        )
        if dw is None or dg is None:
            raise RuntimeError("fla-org dqkwg did not produce dw/dg")
        return (
            _to_head_first4(dq),
            _to_head_first4(dk),
            _to_head_first4(dw),
            dg,
        )

    dq, dk, dw, dg_hf = stress.ascendc_chunk_bwd_dqkwg(
        q_hf,
        k_hf,
        v_new_hf,
        g_nat_ntd.transpose(1, 2).contiguous(),
        h_hf,
        _to_head_first4(do_ntd),
        dh_hf,
        dv_hf,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )
    return dq, dk, dw, dg_hf.transpose(1, 2).contiguous()


def _wy_bwd(
    k_hf: torch.Tensor,
    v_hf: torch.Tensor,
    beta_ntd: torch.Tensor,
    A_hf: torch.Tensor,
    dw_hf: torch.Tensor,
    du_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    g_log2_ntd: Optional[torch.Tensor],
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _CONFIG.uses_org("wy_bwd"):
        dk, dv, db, dg = _org().wy_bwd(
            k=_to_ntd4(k_hf),
            v=_to_ntd4(v_hf),
            beta=beta_ntd,
            A=_to_ntd4(A_hf),
            dw=_to_ntd4(dw_hf),
            du=_to_ntd4(du_hf),
            g=_require_org_gate(g_log2_ntd),
            cu_seqlens=cu_seqlens,
            chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        )
        if dg is None:
            raise RuntimeError("fla-org WY backward did not produce dg")
        return _to_head_first4(dk), _to_head_first4(dv), db, dg

    beta_hf = beta_ntd.transpose(1, 2).contiguous().float()
    g_hf = g_nat_ntd.transpose(1, 2).contiguous()
    dA = stress.ascendc_prepare_wy_repr_bwd_da(
        k_hf,
        v_hf,
        beta_hf,
        A_hf,
        dw_hf,
        du_hf,
        g_hf,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    dk, dv, db_hf, dg_hf = stress.ascendc_prepare_wy_repr_bwd_full(
        k_hf,
        v_hf,
        beta_hf,
        A_hf,
        dA,
        dw_hf,
        du_hf,
        g_hf,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    return (
        dk,
        dv,
        db_hf.transpose(1, 2).contiguous(),
        dg_hf.transpose(1, 2).contiguous(),
    )


def _backward_gdr(
    q_hf: torch.Tensor,
    k_hf: torch.Tensor,
    v_hf: torch.Tensor,
    g_nat_ntd: torch.Tensor,
    beta_ntd: torch.Tensor,
    A_hf: torch.Tensor,
    scale: float,
    do_ntd: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g_log2_ntd = (
        _gate_for_org(g_nat_ntd)
        if any(_CONFIG.uses_org(stage) for stage in BACKWARD_GATE_STAGES)
        else None
    )
    w_hf, u_hf = _recompute_wu(
        k_hf,
        v_hf,
        beta_ntd,
        A_hf,
        g_nat_ntd,
        g_log2_ntd,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    h_hf, v_new_hf = _fwd_h(
        k_hf,
        w_hf,
        u_hf,
        g_nat_ntd,
        g_log2_ntd,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    dv_hf = _dv_local(
        q_hf,
        k_hf,
        do_ntd,
        g_nat_ntd,
        g_log2_ntd,
        A_hf,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    dh_hf, dh0, dv_hf = _bwd_dhu(
        q_hf,
        k_hf,
        w_hf,
        h_hf,
        do_ntd,
        dv_hf,
        g_nat_ntd,
        g_log2_ntd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    dq_hf, dk_hf, dw_hf, dg_ntd = _dqkwg(
        q_hf,
        k_hf,
        v_new_hf,
        w_hf,
        h_hf,
        do_ntd,
        dh_hf,
        dv_hf,
        g_nat_ntd,
        g_log2_ntd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    dk2_hf, dv_hf, db_ntd, dg2_ntd = _wy_bwd(
        k_hf,
        v_hf,
        beta_ntd,
        A_hf,
        dw_hf,
        dv_hf,
        g_nat_ntd,
        g_log2_ntd,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=chunk_size,
    )
    dk_hf.add_(dk2_hf)
    dg_ntd.add_(dg2_ntd)
    dg_ntd = _cumsum(
        dg_ntd,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    stress.record_stage_grad("gdr_dq", dq_hf)
    stress.record_stage_grad("gdr_dk", dk_hf)
    stress.record_stage_grad("gdr_dv", dv_hf)
    stress.record_stage_grad("gdr_db", db_ntd)
    stress.record_stage_grad("gdr_dg", dg_ntd)
    stress.record_stage_grad("gdr_dh", dh_hf)
    stress.record_stage_grad("gdr_dh0", dh0)
    return dq_hf, dk_hf, dv_hf, db_ntd, dg_ntd


class BackendAblationGatedDeltaRuleFunction(torch.autograd.Function):
    """GDR autograd node whose mathematical stages can be switched independently."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        cu_seqlens: torch.Tensor,
        cu_seqlens_list: list[int],
        chunk_indices: dict[str, torch.Tensor],
        chunk_indices_list: dict[str, list[int]],
        chunk_size: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> torch.Tensor:
        stress.trace_apply("stage enter: BackendAblationGatedDeltaRuleFunction.forward")
        if use_qk_l2norm_in_kernel:
            q, q_rstd = _l2norm_fwd(q)
            k, k_rstd = _l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, output, A = _forward_gdr(
            q,
            k,
            v,
            g,
            beta,
            scale,
            cu_seqlens,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
            chunk_size,
        )
        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        stress.trace_apply("stage return: BackendAblationGatedDeltaRuleFunction.forward")
        return output.to(q.dtype)

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        stress.trace_apply("stage enter: BackendAblationGatedDeltaRuleFunction.backward")
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg = _backward_gdr(
            q,
            k,
            v,
            g,
            beta,
            A,
            ctx.scale,
            do,
            ctx.cu_seqlens,
            ctx.cu_seqlens_list,
            ctx.chunk_indices,
            ctx.chunk_indices_list,
            ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = _l2norm_bwd(q, ctx.q_rstd, dq)
            dk = _l2norm_bwd(k, ctx.k_rstd, dk)
        stress.record_stage_grad("gdr_wrapper_dq", dq)
        stress.record_stage_grad("gdr_wrapper_dk", dk)
        stress.trace_apply("stage return: BackendAblationGatedDeltaRuleFunction.backward")
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def main() -> int:
    global _CONFIG, _ORG
    _CONFIG, remaining_argv = _parse_backend_config(sys.argv)
    sys.argv[:] = remaining_argv

    if _CONFIG.any_org:
        if _CONFIG.fla_org_root is None:
            raise ValueError(
                "at least one stage selects fla-org; pass --fla-org-root or set FLA_ORG_ROOT"
            )
        _ORG = _load_org_ops(_CONFIG.fla_org_root)

    if _CONFIG.any_org:
        org_stages = [stage for stage in STAGES if _CONFIG.uses_org(stage)]
        # Current fla-org NPU solve/KKT support the chunk sizes used by this
        # stress case. The base parser validates the concrete value later.
        print(
            "backend_ablation:",
            "canonical_gate=natural_log",
            "fla_org_gate_adapter=g_nat/ln2",
            f"fla_org_stages={','.join(org_stages)}",
            "fla_org_fwd_o_host_wait=disabled",
            flush=True,
        )
    else:
        print("backend_ablation: fla_org_stages=none", flush=True)

    # make_layer resolves this global at call time, so the original model,
    # causal-conv apply boundary, checkpointing, and stress reporting stay
    # unchanged.
    stress.StressGatedDeltaRuleFunction = BackendAblationGatedDeltaRuleFunction
    return stress.main()


if __name__ == "__main__":
    raise SystemExit(main())
