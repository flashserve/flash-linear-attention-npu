"""Triton-backed FLA NPU operators.

This module keeps the public package path stable:

    from fla_npu.ops.triton import chunk_local_cumsum
"""

from fla.ops.triton.triton_core.causal_conv1d import causal_conv1d_triton
from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.triton.triton_core.cumsum import (
    chunk_local_cumsum,
    chunk_local_cumsum_scalar,
)
from fla.ops.triton.triton_core.l2norm import L2Norm, l2norm, l2norm_bwd, l2norm_fwd
from fla.ops.triton.triton_core.solve_tril_fast import solve_tril_npu
from fla.ops.triton.triton_core.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)

causal_conv1d = causal_conv1d_triton
solve_tril = solve_tril_npu

__all__ = [
    "L2Norm",
    "autocast_custom_bwd",
    "autocast_custom_fwd",
    "causal_conv1d",
    "causal_conv1d_triton",
    "chunk_local_cumsum",
    "chunk_local_cumsum_scalar",
    "chunk_scaled_dot_kkt_fwd",
    "input_guard",
    "l2norm",
    "l2norm_bwd",
    "l2norm_fwd",
    "solve_tril",
    "solve_tril_npu",
]
