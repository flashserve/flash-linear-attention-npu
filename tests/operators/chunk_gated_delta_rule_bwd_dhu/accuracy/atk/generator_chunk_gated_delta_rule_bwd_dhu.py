# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


Q_INDEX = 0
K_INDEX = 1
W_INDEX = 2
D_O_INDEX = 3
DV_INDEX = 4
G_INDEX = 5
CHUNK_SIZE_INDEX = 9
IS_MIX_INDEX = 10
IS_FIX_INDEX = 11
QKV_TYPE_INDEX = 12

@GENERATOR_REGISTRY.register("generator_chunk_gated_delta_rule_bwd_dhu")
class ChunkGatedDeltaRuleBwdDhuGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        qkv_type = case_config.inputs[Q_INDEX].dtype
        case_config.inputs[K_INDEX].dtype = qkv_type
        case_config.inputs[W_INDEX].dtype = qkv_type
        case_config.inputs[D_O_INDEX].dtype = qkv_type
        case_config.inputs[DV_INDEX].dtype = qkv_type
        case_config.inputs[QKV_TYPE_INDEX].range_values = qkv_type

        is_mix = case_config.inputs[IS_MIX_INDEX].range_values
        if not is_mix:
            case_config.inputs[G_INDEX].dtype = qkv_type
        is_fix = case_config.inputs[IS_FIX_INDEX].range_values
        B, H, T, _ = case_config.inputs[Q_INDEX].shape
        if not is_fix:
            B = 1
        K = 128
        V = random.choice((128, 256))
        case_config.inputs[Q_INDEX].shape = [B, H, T, K]
        case_config.inputs[K_INDEX].shape = [B, H, T, K]
        case_config.inputs[W_INDEX].shape = [B, H, T, K]
        case_config.inputs[D_O_INDEX].shape = [B, H, T, V]
        case_config.inputs[DV_INDEX].shape = [B, H, T, V]
        case_config.inputs[G_INDEX].shape = [B, H, T]

        return case_config
