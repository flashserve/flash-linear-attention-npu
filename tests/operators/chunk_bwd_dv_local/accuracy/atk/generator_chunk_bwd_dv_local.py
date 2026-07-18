# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import random
from typing import Union, List

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig


Q_INDEX = 0
K_INDEX = 1
D_O_INDEX = 2
G_INDEX = 3
MATRIX_INDEX = 4
A_INDEX = 6
CHUNK_SIZE_INDEX = 10
IS_MIX_INDEX = 11
IS_FIX_INDEX = 12
QKV_TYOE_INDEX = 13
H_RATIO_INDEX = 14

GVA_H_RATIOS = [1, 2, 4]

@GENERATOR_REGISTRY.register("generator_chunk_bwd_dv_local")
class ChunkBwdDvLocalGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        qkv_type = case_config.inputs[Q_INDEX].dtype
        case_config.inputs[K_INDEX].dtype = qkv_type
        case_config.inputs[D_O_INDEX].dtype = qkv_type
        case_config.inputs[A_INDEX].dtype = qkv_type
        case_config.inputs[QKV_TYOE_INDEX].range_values = qkv_type

        is_mix = case_config.inputs[IS_MIX_INDEX].range_values
        if not is_mix:
            case_config.inputs[G_INDEX].dtype = qkv_type
        is_fix = case_config.inputs[IS_FIX_INDEX].range_values
        B, H_qk, T, _ = case_config.inputs[Q_INDEX].shape
        if not is_fix:
            B = 1

        h_ratio = case_config.inputs[H_RATIO_INDEX].range_values if H_RATIO_INDEX < len(case_config.inputs) else 1
        if isinstance(h_ratio, str):
            h_ratio = int(h_ratio)
        H_do = H_qk * h_ratio

        K = 128
        V = 128
        case_config.inputs[Q_INDEX].shape = [B, H_qk, T, K]
        case_config.inputs[K_INDEX].shape = [B, H_qk, T, K]
        case_config.inputs[D_O_INDEX].shape = [B, H_do, T, V]

        chunkSize = case_config.inputs[CHUNK_SIZE_INDEX].range_values
        case_config.inputs[MATRIX_INDEX].shape = [chunkSize, chunkSize]

        return case_config