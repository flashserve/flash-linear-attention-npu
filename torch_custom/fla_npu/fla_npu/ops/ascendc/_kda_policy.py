"""FLA-aligned host policy for ChunkKdaFwd optional outputs."""

from __future__ import annotations

from typing import Tuple


FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT = "0f0f0c97af39343855b43bbbaddcedfda5cb9d77"
FLA_ORG_KDA_FWD_ALIGNMENT_SOURCE = (
    "https://github.com/fla-org/flash-linear-attention/blob/"
    f"{FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT}/fla/ops/kda/chunk_fwd.py"
)


def kda_fwd_optional_output_mask(
    *,
    output_final_state: bool,
    use_gate_in_kernel: bool,
    disable_recompute: bool,
    return_intermediate_states: bool,
) -> Tuple[bool, ...]:
    """Return the visibility mask for the low-level 12-value FLA interface."""

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
