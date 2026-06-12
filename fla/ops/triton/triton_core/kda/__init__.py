# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

from .chunk import chunk_kda
from .fused_recurrent import fused_recurrent_kda

__all__ = [
    "chunk_kda",
    "fused_recurrent_kda",
]
