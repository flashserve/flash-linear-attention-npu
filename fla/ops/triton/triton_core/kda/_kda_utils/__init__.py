# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

# fla/ops/kda/_kda_utils/__init__.py
# KDA-isolated utility functions — unified export
# Merged modules (constant, softplus, index, cumsum, utils) now live in triton_core/ directly.
# Remaining modules (op, l2norm) are still local copies.

from fla.ops.triton.triton_core.utils import (
    input_guard,
    autocast_custom_fwd,
    autocast_custom_bwd,
    autotune_cache_kwargs,
    device,
    IS_NPU,
    IS_GATHER_SUPPORTED,
    assert_close,
    tensor_cache,
)
from fla.ops.triton.triton_core.index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp, exp2, log, log2, gather
from fla.ops.triton.triton_core.constant import RCP_LN2
from fla.ops.triton.triton_core.softplus import softplus
from fla.ops.triton.triton_core.cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd, l2norm_bwd

__all__ = [
    # utils (merged to triton_core)
    "input_guard",
    "autocast_custom_fwd",
    "autocast_custom_bwd",
    "autotune_cache_kwargs",
    "device",
    "IS_NPU",
    "IS_GATHER_SUPPORTED",
    "assert_close",
    "tensor_cache",
    # index (merged to triton_core)
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    # op
    "exp",
    "exp2",
    "log",
    "log2",
    "gather",
    # constant (merged to triton_core)
    "RCP_LN2",
    # softplus (merged to triton_core)
    "softplus",
    # cumsum (merged to triton_core — KDA callers use use_per_head_kernel=True)
    "chunk_local_cumsum",
    # l2norm
    "l2norm_fwd",
    "l2norm_bwd",
]