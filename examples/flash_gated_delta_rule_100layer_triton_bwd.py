#!/usr/bin/env python3
"""Run the PR #249 stress case with fla-org Triton-Ascend backward kernels.

The forward path is unchanged from
``flash_gated_delta_rule_100layer_stress.py``. During backward, the two
forward-recompute stages (``recompute_w_u`` and ``fwd_h``) also stay on the
original fla_npu AscendC path. The remaining GDR backward stages use the
fla-org Triton-Ascend implementation:

* ``chunk_bwd_dv_local_npu``
* ``chunk_gated_delta_rule_bwd_dhu_npu``
* ``chunk_bwd_dqkwg_npu``
* ``prepare_wy_repr_bwd_npu``
* reverse ``chunk_local_cumsum_npu``
* ``l2norm_bwd_npu``

Pass a fla-org checkout with ``--fla-org-root`` or ``FLA_ORG_ROOT``:

    python3 examples/flash_gated_delta_rule_100layer_triton_bwd.py \
        --fla-org-root /path/to/flash-linear-attention \
        --steps 20 --layers 100 --tokens 32768 --heads 16

The PR script stores cumulative gates in the natural-log domain and its
AscendC kernels use ``exp``. fla-org's Triton-Ascend backward kernels use
``exp2``, so this adapter converts the cumulative gate by ``1 / ln(2)`` only
at the fla-org boundary.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, TypeVar

import torch

import flash_gated_delta_rule_100layer_stress as stress


RCP_LN2 = 1.4426950408889634074
_T = TypeVar("_T")


def _parse_args(argv: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fla-org-root",
        default=os.environ.get("FLA_ORG_ROOT", ""),
        help="Checkout root of fla-org/flash-linear-attention.",
    )
    args, remaining = parser.parse_known_args(argv[1:])
    if not args.fla_org_root:
        parser.error("pass --fla-org-root or set FLA_ORG_ROOT")
    return Path(args.fla_org_root).expanduser().resolve(), [argv[0], *remaining]


def _load_org_ops(root: Path) -> SimpleNamespace:
    marker = root / "fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py"
    if not marker.is_file():
        raise ValueError(
            f"--fla-org-root does not look like fla-org/flash-linear-attention: {root}"
        )

    loaded_fla = sys.modules.get("fla")
    if loaded_fla is not None:
        origin = Path(getattr(loaded_fla, "__file__", "")).resolve()
        try:
            origin.relative_to(root)
        except ValueError:
            # The stress module has already captured direct references to the
            # PR wheel's Triton functions. Drop only the module-cache entries
            # so fla-org can occupy the same package name; the captured
            # function objects and their globals remain alive for the forward
            # path and backward recomputation.
            for module_name in tuple(sys.modules):
                if module_name == "fla" or module_name.startswith("fla."):
                    del sys.modules[module_name]

    sys.path.insert(0, str(root))
    importlib.invalidate_caches()

    from fla.modules.backends.triton_ascend.l2norm import (  # type: ignore[import-not-found]
        l2norm_bwd_npu,
    )
    from fla.ops.common.backends.triton_ascend.chunk_delta_h import (  # type: ignore[import-not-found]
        chunk_gated_delta_rule_bwd_dhu_npu,
    )
    from fla.ops.common.backends.triton_ascend.chunk_o import (  # type: ignore[import-not-found]
        chunk_bwd_dqkwg_npu,
        chunk_bwd_dv_local_npu,
    )
    from fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast import (  # type: ignore[import-not-found]
        prepare_wy_repr_bwd_npu,
    )
    from fla.ops.utils.backends.triton_ascend.cumsum import (  # type: ignore[import-not-found]
        chunk_local_cumsum_npu,
    )

    return SimpleNamespace(
        l2norm_bwd=l2norm_bwd_npu,
        dv_local=chunk_bwd_dv_local_npu,
        bwd_dhu=chunk_gated_delta_rule_bwd_dhu_npu,
        dqkwg=chunk_bwd_dqkwg_npu,
        wy_bwd=prepare_wy_repr_bwd_npu,
        cumsum=chunk_local_cumsum_npu,
    )


def _to_ntd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, H, T, D] to [B, T, H, D]."""
    return tensor.transpose(1, 2).contiguous()


def _to_head_first(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, T, H, D] to [B, H, T, D]."""
    return tensor.transpose(1, 2).contiguous()


def _chunk_state_to_ntd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, H, NT, K, V] to [B, NT, H, K, V]."""
    return tensor.transpose(1, 2).contiguous()


def _chunk_state_to_head_first(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, NT, H, K, V] to [B, H, NT, K, V]."""
    return tensor.transpose(1, 2).contiguous()


def _gate_for_org(g_natural: torch.Tensor) -> torch.Tensor:
    return g_natural.mul(RCP_LN2)


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


def _trace_call(name: str, function: Callable[..., _T], *args, **kwargs) -> _T:
    stress.trace_apply(f"stage enter: fla_org_{name}")
    result = function(*args, **kwargs)
    stress.trace_apply(f"stage return: fla_org_{name}")
    return result


def _gated_delta_rule_bwd(
    ops: SimpleNamespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep the forward-recompute portion exactly on the PR #249 path.
    g_head = g.transpose(1, 2).contiguous()
    beta_head = beta.transpose(1, 2).contiguous().float()
    w_head, u_head = stress.recompute_w_u(
        k,
        v,
        beta_head,
        A,
        g_head,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    h_head, v_new_head, _ = stress.ascendc_chunk_gated_delta_rule_fwd_h(
        k,
        w_head,
        u_head,
        g=g_head,
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

    # Everything below this boundary is the fla-org Triton-Ascend backward.
    q_ntd = _to_ntd(q)
    k_ntd = _to_ntd(k)
    v_ntd = _to_ntd(v)
    A_ntd = _to_ntd(A)
    w_ntd = _to_ntd(w_head)
    h_ntd = _chunk_state_to_ntd(h_head)
    v_new_ntd = _to_ntd(v_new_head)
    g_log2 = _gate_for_org(g)
    chunk_indices_tensor = _chunk_tensor(chunk_indices, chunk_size)

    dv_ntd = _trace_call(
        "chunk_bwd_dv_local",
        ops.dv_local,
        q=q_ntd,
        k=k_ntd,
        do=do,
        g=g_log2,
        g_gamma=None,
        A=A_ntd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices_tensor,
    )

    dh_ntd, dh0, dv_ntd = _trace_call(
        "chunk_gated_delta_rule_bwd_dhu",
        ops.bwd_dhu,
        q=q_ntd,
        k=k_ntd,
        w=w_ntd,
        do=do,
        dv=dv_ntd,
        g=g_log2,
        gk=None,
        h0=None,
        dht=None,
        scale=scale,
        state_v_first=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices_tensor,
    )

    dq_ntd, dk_ntd, dw_ntd, dg_ntd = _trace_call(
        "chunk_bwd_dqkwg",
        ops.dqkwg,
        q=q_ntd,
        k=k_ntd,
        v=v_new_ntd,
        do=do,
        h=h_ntd,
        dh=dh_ntd,
        w=w_ntd,
        g=g_log2,
        g_gamma=None,
        dv=dv_ntd,
        scale=scale,
        state_v_first=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices_tensor,
    )
    if dw_ntd is None or dg_ntd is None:
        raise RuntimeError("fla-org chunk_bwd_dqkwg_npu did not produce dw/dg")

    dk2_ntd, dv_ntd, db_ntd, dg2_ntd = _trace_call(
        "prepare_wy_repr_bwd",
        ops.wy_bwd,
        k=k_ntd,
        v=v_ntd,
        beta=beta,
        A=A_ntd,
        dw=dw_ntd,
        du=dv_ntd,
        g=g_log2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_tensor,
    )
    if dg2_ntd is None:
        raise RuntimeError("fla-org prepare_wy_repr_bwd_npu did not produce dg")

    dk_ntd.add_(dk2_ntd)
    dg_ntd.add_(dg2_ntd)
    # This derivative is in the natural-log gate domain. Do not apply RCP_LN2
    # to the reverse cumsum itself.
    dg_ntd = _trace_call(
        "chunk_local_cumsum_bwd",
        ops.cumsum,
        dg_ntd,
        chunk_size=chunk_size,
        reverse=True,
        scale=None,
        cu_seqlens=cu_seqlens,
        head_first=False,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices_tensor,
    )

    dq_head = _to_head_first(dq_ntd)
    dk_head = _to_head_first(dk_ntd)
    dv_head = _to_head_first(dv_ntd)
    dh_head = _chunk_state_to_head_first(dh_ntd)

    stress.record_stage_grad("gdr_dq", dq_head)
    stress.record_stage_grad("gdr_dk", dk_head)
    stress.record_stage_grad("gdr_dv", dv_head)
    stress.record_stage_grad("gdr_db", db_ntd)
    stress.record_stage_grad("gdr_dg", dg_ntd)
    stress.record_stage_grad("gdr_dh", dh_head)
    stress.record_stage_grad("gdr_dh0", dh0)
    return dq_head, dk_head, dv_head, db_ntd, dg_ntd


def _install_triton_backward(ops: SimpleNamespace) -> None:
    def gated_delta_rule_bwd(*args, **kwargs):
        return _gated_delta_rule_bwd(ops, *args, **kwargs)

    def l2norm_bwd(
        y: torch.Tensor,
        rstd: torch.Tensor,
        dy: torch.Tensor,
    ) -> torch.Tensor:
        return _trace_call("l2norm_bwd", ops.l2norm_bwd, y, rstd, dy)

    # StressGatedDeltaRuleFunction resolves both globals when backward runs.
    stress.gated_delta_rule_bwd = gated_delta_rule_bwd
    stress.l2norm_bwd = l2norm_bwd


def main() -> int:
    fla_org_root, remaining_argv = _parse_args(sys.argv)
    sys.argv[:] = remaining_argv
    ops = _load_org_ops(fla_org_root)
    _install_triton_backward(ops)
    print(
        "gdr_backward_backend:",
        "forward=fla_npu",
        "backward_recompute=fla_npu",
        "backward_non_recompute=fla_org_triton_ascend",
        "gate_adapter=g_natural/ln2",
        flush=True,
    )
    return stress.main()


if __name__ == "__main__":
    raise SystemExit(main())
