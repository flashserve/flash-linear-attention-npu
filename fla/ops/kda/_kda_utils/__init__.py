# fla/ops/kda/_kda_utils/__init__.py
# KDA-isolated utility functions — unified export
# All imports point to local copies within this package, NOT to the global fla.* tree.

from .utils import (
    input_guard,
    autocast_custom_fwd,
    autocast_custom_bwd,
    check_shared_mem,
    autotune_cache_kwargs,
    device,
    IS_NPU,
    IS_AMD,
    IS_NVIDIA,
    IS_NVIDIA_HOPPER,
    IS_GATHER_SUPPORTED,
    IS_TF32_SUPPORTED,
    IS_TMA_SUPPORTED,
    USE_CUDA_GRAPH,
    assert_close,
    tensor_cache,
)
from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp, exp2, log, log2, gather
from .constant import RCP_LN2
from .softplus import softplus
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd, l2norm_bwd

__all__ = [
    # utils
    "input_guard",
    "autocast_custom_fwd",
    "autocast_custom_bwd",
    "check_shared_mem",
    "autotune_cache_kwargs",
    "device",
    "IS_NPU",
    "IS_AMD",
    "IS_NVIDIA",
    "IS_NVIDIA_HOPPER",
    "IS_GATHER_SUPPORTED",
    "IS_TF32_SUPPORTED",
    "IS_TMA_SUPPORTED",
    "USE_CUDA_GRAPH",
    "assert_close",
    "tensor_cache",
    # index
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    # op
    "exp",
    "exp2",
    "log",
    "log2",
    "gather",
    # constant
    "RCP_LN2",
    # softplus
    "softplus",
    # cumsum
    "chunk_local_cumsum",
    # l2norm
    "l2norm_fwd",
    "l2norm_bwd",
]