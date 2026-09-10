"""与 FLA 语义对齐的 ChunkKdaFwd 可选输出策略。"""

from __future__ import annotations

from typing import Tuple


FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT = "0f0f0c97af39343855b43bbbaddcedfda5cb9d77"
FLA_ORG_KDA_FWD_ALIGNMENT_SOURCE = (
    "https://github.com/fla-org/flash-linear-attention/blob/"
    f"{FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT}/fla/ops/kda/chunk_fwd.py"
)


KDA_FWD_PREPARE_OUTPUT_NAMES = (
    "gk",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "qg_scaled",
    "q_hat",
    "k_hat",
    "q_rstd",
    "k_rstd",
    "beta_eff",
)

_KDA_FWD_PREPARE_OUTPUT_MASKS = {
    # 正向阶段间必需量：gk/Aqk/w/u/kg/qg_scaled。
    "none": (
        True, True, False, True, True, False, True, True,
        False, False, False, False, False,
    ),
    # 反向阶段重算中间量；w/u/kg 等正向阶段间必需量仍须输出。
    "recompute": (
        True, True, True, True, True, False, True, True,
        True, True, True, True, True,
    ),
    # 禁止反向重算时保留 Prepare 的全部结果。
    "save": (True,) * len(KDA_FWD_PREPARE_OUTPUT_NAMES),
}


def kda_fwd_prepare_output_mask(*, backward_mode: str) -> Tuple[bool, ...]:
    """返回固定 13 槽 Prepare 接口在指定反向策略下的输出掩码。"""

    if not isinstance(backward_mode, str) or backward_mode not in _KDA_FWD_PREPARE_OUTPUT_MASKS:
        raise ValueError("backward_mode must be one of: none, recompute, save")
    return _KDA_FWD_PREPARE_OUTPUT_MASKS[backward_mode]


def kda_fwd_optional_output_mask(
    *,
    output_final_state: bool,
    use_gate_in_kernel: bool,
    disable_recompute: bool,
    return_intermediate_states: bool,
) -> Tuple[bool, ...]:
    """返回 FLA 低层 12 槽接口的可见性掩码。"""

    return (
        True,
        output_final_state,
        not use_gate_in_kernel or disable_recompute,
        True,
        True,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute or return_intermediate_states,
        True,
    )
