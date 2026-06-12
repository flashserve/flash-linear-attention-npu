# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

# _kda_cp/chunk_delta_h.py: Context Parallel stubs (not supported in NPU version)

def chunk_gated_delta_rule_fwd_h_pre_process(*args, **kwargs):
    raise NotImplementedError("Context Parallel is not supported")

def compress_h0(*args, **kwargs):
    raise NotImplementedError("Context Parallel is not supported")

def chunk_gated_delta_rule_bwd_dhu_pre_process(*args, **kwargs):
    raise NotImplementedError("Context Parallel is not supported")

def expand_h0(*args, **kwargs):
    raise NotImplementedError("Context Parallel is not supported")
