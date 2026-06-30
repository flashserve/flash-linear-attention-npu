# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

import triton
from triton import language as tl


@triton.jit
def softplus(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)

@triton.jit
def softplus2(x):
    return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)