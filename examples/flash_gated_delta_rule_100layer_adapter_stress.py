#!/usr/bin/env python3
"""Run the PR #249 pure-async stress workload through the standalone adapter.

All workload generation, causal-conv forward/backward, checkpointing,
determinism checks, memory accounting, and end-of-run reporting come from
``flash_gated_delta_rule_100layer_stress.py``. Only the GDR apply boundary is
redirected to ``ChunkGatedDeltaRuleFunction``. The adapter requires
``FLA_ORG_ROOT`` and sources only solve_tri from fla-org's Triton-Ascend
backend.
"""

from __future__ import annotations

import torch

import flash_gated_delta_rule_100layer_stress as stress
from chunk_gated_delta_rule_function import (
    ChunkGatedDeltaRuleFunction,
    get_fla_org_solve_tril,
)


class _StressSignatureAdapter:
    """Map the original stress signature to the standalone GDR adapter."""

    @staticmethod
    def apply(
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
        output, _ = ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            None,
            False,
            cu_seqlens,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
            use_qk_l2norm_in_kernel,
            chunk_size,
        )
        return output


def main() -> int:
    solve_tril = get_fla_org_solve_tril()
    print(
        "solve_tri_backend=fla-org/triton-ascend "
        f"module={solve_tril.__module__}",
        flush=True,
    )
    stress.StressGatedDeltaRuleFunction = _StressSignatureAdapter
    return stress.main()


if __name__ == "__main__":
    raise SystemExit(main())
