# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

# _kda_cp: Context Parallel stub (not supported in NPU version)

class FLACPContext:
    """Stub: Context Parallel is not supported in NPU version."""
    pass

def build_cp_context(*args, **kwargs):
    raise NotImplementedError("Context Parallel is not supported in NPU version")
