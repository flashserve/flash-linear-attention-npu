# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

# _kda_common: KDA-isolated common operators
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_h import chunk_fwd_h, chunk_bwd_dh
