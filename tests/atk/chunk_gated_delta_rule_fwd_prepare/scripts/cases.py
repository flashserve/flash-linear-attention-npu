"""Explicit GDN cases for the FLA span L2norm → ``fwd_intra`` / RecomputeWU.

Cases are written in ``gdn_cases()`` below. They are **not** loaded from CSV.

Used flags (``ChunkGatedDeltaRuleFunction.forward`` through ``fwd_intra``)::

    B, HK, HV, T, K, V, seqlens, chunk_size,
    use_qk_l2norm_in_kernel, use_gate_in_kernel,
    use_beta_sigmoid_in_kernel, allow_neg_eigval

Ignored (after ``fwd_intra``, or NPU-only)::

    scale              # only consumed by later ``fwd_o``
    output_final_state # ``fwd_h``
    state_v_first      # ``fwd_h``
    use_exp2           # NPU switch; Triton path is always exp2 after RCP_LN2
    chunk_indices      # FLA builds these from cu_seqlens
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import torch

if __package__ in (None, ""):
    from common import generate_inputs
else:
    from .common import generate_inputs


@dataclass(frozen=True)
class GdnCase:
    case_id: int
    batch: int
    hk: int
    hv: int
    seq_len: int
    head_k: int
    head_v: int
    chunk_size: int
    use_qk_l2norm_in_kernel: bool
    use_gate_in_kernel: bool
    use_beta_sigmoid_in_kernel: bool
    allow_neg_eigval: bool
    seqlens: tuple[int, ...] | None = None
    shape_id: int | None = None
    combo_id: int | None = None

    @property
    def varlen(self) -> bool:
        return self.seqlens is not None

    @property
    def base_id(self) -> int:
        return self.shape_id if self.shape_id is not None else self.case_id

    def skip_reason(self) -> str | None:
        if self.hv % self.hk != 0:
            return f"GVA requires HV % HK == 0, got HK={self.hk} HV={self.hv}"
        if self.chunk_size not in (16, 32, 64):
            return f"FLA Triton chunk_size must be 16/32/64, got {self.chunk_size}"
        if self.varlen and self.batch != 1:
            return f"FLA varlen requires B=1, got B={self.batch}"
        if self.varlen and sum(self.seqlens) != self.seq_len:
            return f"sum(seqlens)={sum(self.seqlens)} != T={self.seq_len}"
        if self.allow_neg_eigval and not self.use_beta_sigmoid_in_kernel:
            return "allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True"
        return None

    def flags(self) -> dict:
        return dict(
            chunk_size=self.chunk_size,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
            use_gate_in_kernel=self.use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=self.use_beta_sigmoid_in_kernel,
            allow_neg_eigval=self.allow_neg_eigval,
        )


# Order: l2norm, gate, beta_sigmoid, allow_neg_eigval. 16 combos, index 0-15.
FLAG_GRID = tuple(product((False, True), repeat=4))


def gdn_flag_grid_cases() -> list[GdnCase]:
    """Six shapes × 2^4 flag combos = 96 cases.

    ``allow_neg_eigval=True`` without ``use_beta_sigmoid_in_kernel`` is invalid
    (FLA raises); those 24 rows are still listed and skipped by ``skip_reason``.
    """
    out: list[GdnCase] = []
    for base in gdn_cases():
        for combo_id, (l2, gate, beta, neg) in enumerate(FLAG_GRID):
            out.append(
                replace(
                    base,
                    case_id=base.case_id * 100 + combo_id,
                    use_qk_l2norm_in_kernel=l2,
                    use_gate_in_kernel=gate,
                    use_beta_sigmoid_in_kernel=beta,
                    allow_neg_eigval=neg,
                    shape_id=base.case_id,
                    combo_id=combo_id,
                )
            )
    return out


def gdn_cases() -> list[GdnCase]:
    """All six cases, written explicitly here (not parsed from CSV).

    Case 2 is packed varlen: B=1, 64 sequences, ``cu_seqlens`` has 65 points
    (first 0, last T=65536). Lengths are mixed on purpose so chunk boundaries
    do not line up with a single fixed T.
    """
    # 8+8+8+16+16+8 = 64 sequences; sum = 65536
    varlen_seqlens = (
        (256,) * 8
        + (512,) * 8
        + (768,) * 8
        + (1024,) * 16
        + (1280,) * 16
        + (2048,) * 8
    )

    return [
        # 1. GVA, long fixed, L2norm on, gate off
        GdnCase(
            case_id=1, batch=2, hk=16, hv=32, seq_len=11264,
            head_k=128, head_v=128, chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
        ),
        # 2. packed varlen, L2norm off, gate off
        GdnCase(
            case_id=2, batch=1, hk=32, hv=32, seq_len=65536,
            head_k=128, head_v=128, chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            seqlens=varlen_seqlens,
        ),
        # 3. wide heads, short T, L2norm on, fused gate on
        GdnCase(
            case_id=3, batch=4, hk=96, hv=96, seq_len=128,
            head_k=128, head_v=128, chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
        ),
        # 4. T not multiple of chunk, L2norm off, fused gate on
        GdnCase(
            case_id=4, batch=1, hk=32, hv=32, seq_len=160,
            head_k=128, head_v=128, chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
        ),
        # 5. mid T, few heads, L2norm on, fused gate on
        GdnCase(
            case_id=5, batch=6, hk=6, hv=6, seq_len=1084,
            head_k=128, head_v=128, chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
        ),
        # 6. mid T, L2norm off, gate off
        GdnCase(
            case_id=6, batch=1, hk=12, hv=12, seq_len=1084,
            head_k=128, head_v=128, chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
        ),
    ]


def case_inputs(
    case: GdnCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    layout: str = "bsnd",
) -> dict[str, torch.Tensor]:
    return generate_inputs(
        batch=case.batch,
        seq_len=case.seq_len,
        num_k_heads=case.hk,
        num_v_heads=case.hv,
        head_k=case.head_k,
        head_v=case.head_v,
        dtype=dtype,
        device=device,
        seed=seed,
        use_gate_in_kernel=case.use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=case.use_beta_sigmoid_in_kernel,
        use_qk_l2norm_in_kernel=case.use_qk_l2norm_in_kernel,
        layout=layout,
        seqlens=None if case.seqlens is None else list(case.seqlens),
    )


def describe_case(case: GdnCase) -> str:
    if case.varlen:
        cu = f"packed n={len(case.seqlens)} T={case.seq_len}"
    else:
        cu = "None"
    prefix = f"case{case.case_id}"
    if case.shape_id is not None and case.combo_id is not None:
        prefix = f"shape{case.shape_id}/c{case.combo_id:02d}({case.case_id})"
    return (
        f"{prefix}: B={case.batch} HK={case.hk} HV={case.hv} "
        f"T={case.seq_len} K={case.head_k} V={case.head_v} "
        f"BT={case.chunk_size} cu_seqlens={cu} "
        f"l2norm={case.use_qk_l2norm_in_kernel} "
        f"gate={case.use_gate_in_kernel} "
        f"beta_sigmoid={case.use_beta_sigmoid_in_kernel} "
        f"neg_eig={case.allow_neg_eigval}"
    )


def select_cases(case_id: int | None = None, *, grid: bool = False) -> list[GdnCase]:
    cases = gdn_flag_grid_cases() if grid else gdn_cases()
    if case_id is None:
        return cases
    if grid and 1 <= case_id <= 6:
        picked = [c for c in cases if c.base_id == case_id]
    else:
        picked = [c for c in cases if c.case_id == case_id]
    if not picked:
        known = ", ".join(str(c.case_id) for c in cases)
        raise SystemExit(f"no case_id={case_id}; known: {known}")
    return picked
