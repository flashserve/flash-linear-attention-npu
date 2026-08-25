"""chunk_gated_delta_rule_fwd_h 的 ATK 泛化用例生成器。

基于 GDN 泛化用例表整理的 100 个 shape × bf16/fp16 = 200 条；本文件内自维护 PROFILES。
case 按计算规模从小到大排序，便于从最小 case 开始逐步验证。
"""

from __future__ import annotations

import json
from copy import deepcopy

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None

OP_NAME = "chunk_gated_delta_rule_fwd_h"
PROFILES = [
    {'name': 'BSND_noGVA_V128_13_T512', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 0, 'seed': 20260817, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_T512', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 1, 'seed': 20260818, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_cs128_T512', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 2, 'seed': 20260819, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_cs128_T512', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 3, 'seed': 20260820, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 4, 'seed': 20260821, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 5, 'seed': 20260822, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 6, 'seed': 20260823, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 7, 'seed': 20260824, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28_T2048', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 2048, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 8, 'seed': 20260825, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28_T2048', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 2048, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 9, 'seed': 20260826, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28_cs128_T2048', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 2048, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 10, 'seed': 20260827, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28_cs128_T2048', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 2048, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 11, 'seed': 20260828, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_T3584', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 3584, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 12, 'seed': 20260829, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_T3584', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 3584, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 13, 'seed': 20260830, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64_T3584', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 3584, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 14, 'seed': 20260831, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64_T3584', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 3584, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 15, 'seed': 20260832, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 16, 'seed': 20260833, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 17, 'seed': 20260834, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 18, 'seed': 20260835, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_28_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 19, 'seed': 20260836, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33_T4480', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 4480, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 20, 'seed': 20260837, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33_T4480', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 4480, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 21, 'seed': 20260838, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33_cs64_T4480', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 4480, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 22, 'seed': 20260839, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33_cs64_T4480', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 4480, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 23, 'seed': 20260840, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 7178, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 24, 'seed': 20260841, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 7178, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 25, 'seed': 20260842, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 7178, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 26, 'seed': 20260843, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 7178, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 27, 'seed': 20260844, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_T640', 'dtype': 'bf16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 28, 'seed': 20260845, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_T640', 'dtype': 'fp16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 29, 'seed': 20260846, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_T640', 'dtype': 'bf16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 30, 'seed': 20260847, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_T640', 'dtype': 'fp16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 31, 'seed': 20260848, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_T640', 'dtype': 'bf16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 32, 'seed': 20260849, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_T640', 'dtype': 'fp16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 33, 'seed': 20260850, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_cs128_T640', 'dtype': 'bf16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 34, 'seed': 20260851, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_cs128_T640', 'dtype': 'fp16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 35, 'seed': 20260852, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_cs128_T640', 'dtype': 'bf16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 36, 'seed': 20260853, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_cs128_T640', 'dtype': 'fp16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 37, 'seed': 20260854, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_cs128_T640', 'dtype': 'bf16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 38, 'seed': 20260855, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_cs128_T640', 'dtype': 'fp16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 640, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 39, 'seed': 20260856, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_T256', 'dtype': 'bf16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 256, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 40, 'seed': 20260857, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_T256', 'dtype': 'fp16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 256, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 41, 'seed': 20260858, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_cs128_T256', 'dtype': 'bf16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 256, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 42, 'seed': 20260859, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_cs128_T256', 'dtype': 'fp16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 256, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 43, 'seed': 20260860, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_cs128_T4096', 'dtype': 'bf16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 44, 'seed': 20260861, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_cs128_T4096', 'dtype': 'fp16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 45, 'seed': 20260862, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_cs128_T5376', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 5376, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 46, 'seed': 20260863, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_cs128_T5376', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 5376, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 47, 'seed': 20260864, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_T5440', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 5440, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 48, 'seed': 20260865, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_T5440', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 5440, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 49, 'seed': 20260866, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_T4160', 'dtype': 'bf16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 4160, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 50, 'seed': 20260867, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_T4160', 'dtype': 'fp16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 4160, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 51, 'seed': 20260868, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64_T64', 'dtype': 'bf16', 'B': 128, 'HK': 4, 'HV': 32, 'T': 64, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 52, 'seed': 20260869, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64_T64', 'dtype': 'fp16', 'B': 128, 'HK': 4, 'HV': 32, 'T': 64, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 53, 'seed': 20260870, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_T512', 'dtype': 'bf16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 54, 'seed': 20260871, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_T512', 'dtype': 'fp16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 55, 'seed': 20260872, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_T512', 'dtype': 'bf16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 56, 'seed': 20260873, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_T512', 'dtype': 'fp16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 57, 'seed': 20260874, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_T512', 'dtype': 'bf16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 58, 'seed': 20260875, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_T512', 'dtype': 'fp16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 59, 'seed': 20260876, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_T512', 'dtype': 'bf16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 60, 'seed': 20260877, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_T512', 'dtype': 'fp16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 61, 'seed': 20260878, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_09_T512', 'dtype': 'bf16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 62, 'seed': 20260879, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_09_T512', 'dtype': 'fp16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 63, 'seed': 20260880, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_10_T512', 'dtype': 'bf16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 64, 'seed': 20260881, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_10_T512', 'dtype': 'fp16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 65, 'seed': 20260882, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_11_T512', 'dtype': 'bf16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 66, 'seed': 20260883, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_11_T512', 'dtype': 'fp16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 67, 'seed': 20260884, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_cs128_T512', 'dtype': 'bf16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 68, 'seed': 20260885, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_cs128_T512', 'dtype': 'fp16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 69, 'seed': 20260886, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_T1024', 'dtype': 'bf16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 70, 'seed': 20260887, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_T1024', 'dtype': 'fp16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 71, 'seed': 20260888, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_12_T1024', 'dtype': 'bf16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 72, 'seed': 20260889, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_12_T1024', 'dtype': 'fp16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 73, 'seed': 20260890, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_B32', 'dtype': 'bf16', 'B': 32, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 74, 'seed': 20260891, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_B32', 'dtype': 'fp16', 'B': 32, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 75, 'seed': 20260892, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_B16', 'dtype': 'bf16', 'B': 16, 'HK': 16, 'HV': 16, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 76, 'seed': 20260893, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_B16', 'dtype': 'fp16', 'B': 16, 'HK': 16, 'HV': 16, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 77, 'seed': 20260894, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_B64', 'dtype': 'bf16', 'B': 64, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 78, 'seed': 20260895, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_B64', 'dtype': 'fp16', 'B': 64, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 79, 'seed': 20260896, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_09_B32', 'dtype': 'bf16', 'B': 32, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 80, 'seed': 20260897, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_09_B32', 'dtype': 'fp16', 'B': 32, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 81, 'seed': 20260898, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_B4', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 2048, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 82, 'seed': 20260899, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_B4', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 2048, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 83, 'seed': 20260900, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 84, 'seed': 20260901, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 85, 'seed': 20260902, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 86, 'seed': 20260903, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 87, 'seed': 20260904, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_T4096', 'dtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 88, 'seed': 20260905, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_T4096', 'dtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 89, 'seed': 20260906, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_cs128_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 90, 'seed': 20260907, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_cs128_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 91, 'seed': 20260908, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_cs128_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 92, 'seed': 20260909, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_cs128_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 93, 'seed': 20260910, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_cs128_T4096', 'dtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 94, 'seed': 20260911, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_cs128_T4096', 'dtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 4096, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 95, 'seed': 20260912, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_T16384', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 96, 'seed': 20260913, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_T16384', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 97, 'seed': 20260914, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 98, 'seed': 20260915, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 99, 'seed': 20260916, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 100, 'seed': 20260917, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 101, 'seed': 20260918, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_cs128_T16384', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 102, 'seed': 20260919, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_cs128_T16384', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 103, 'seed': 20260920, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21_cs128_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 104, 'seed': 20260921, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21_cs128_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 105, 'seed': 20260922, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_cs64_T8192', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 106, 'seed': 20260923, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_cs64_T8192', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 107, 'seed': 20260924, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_T32768', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 108, 'seed': 20260925, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_T32768', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 109, 'seed': 20260926, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_cs128_T32768', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 110, 'seed': 20260927, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_cs128_T32768', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 111, 'seed': 20260928, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31', 'dtype': 'bf16', 'B': 176, 'HK': 2, 'HV': 64, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 112, 'seed': 20260929, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31', 'dtype': 'fp16', 'B': 176, 'HK': 2, 'HV': 64, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 113, 'seed': 20260930, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31_cs128', 'dtype': 'bf16', 'B': 176, 'HK': 2, 'HV': 64, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 114, 'seed': 20260931, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31_cs128', 'dtype': 'fp16', 'B': 176, 'HK': 2, 'HV': 64, 'T': 24, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 115, 'seed': 20260932, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 8999, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 116, 'seed': 20260933, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 8999, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 117, 'seed': 20260934, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33_cs64', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 8999, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 118, 'seed': 20260935, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_33_cs64', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 8999, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 119, 'seed': 20260936, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_scaled', 'dtype': 'bf16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 512, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 120, 'seed': 20260937, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_scaled', 'dtype': 'fp16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 512, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 121, 'seed': 20260938, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_cs128', 'dtype': 'bf16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 512, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 122, 'seed': 20260939, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_29_cs128', 'dtype': 'fp16', 'B': 16, 'HK': 21, 'HV': 63, 'T': 512, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 123, 'seed': 20260940, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_scaled', 'dtype': 'bf16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 124, 'seed': 20260941, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_scaled', 'dtype': 'fp16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 125, 'seed': 20260942, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_scaled', 'dtype': 'bf16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 126, 'seed': 20260943, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_scaled', 'dtype': 'fp16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 127, 'seed': 20260944, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_scaled', 'dtype': 'bf16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 128, 'seed': 20260945, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_scaled', 'dtype': 'fp16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 129, 'seed': 20260946, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_cs128', 'dtype': 'bf16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 130, 'seed': 20260947, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_cs128', 'dtype': 'fp16', 'B': 48, 'HK': 8, 'HV': 8, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 131, 'seed': 20260948, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_cs128', 'dtype': 'bf16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 132, 'seed': 20260949, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_cs128', 'dtype': 'fp16', 'B': 24, 'HK': 16, 'HV': 16, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 133, 'seed': 20260950, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_cs128', 'dtype': 'bf16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 134, 'seed': 20260951, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_cs128', 'dtype': 'fp16', 'B': 12, 'HK': 32, 'HV': 32, 'T': 1344, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 135, 'seed': 20260952, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 10880, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 136, 'seed': 20260953, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 10880, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 137, 'seed': 20260954, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 10880, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 138, 'seed': 20260955, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_32_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 48, 'T': 10880, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 139, 'seed': 20260956, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 8320, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 140, 'seed': 20260957, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 8320, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 141, 'seed': 20260958, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 8320, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 142, 'seed': 20260959, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_23_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 21, 'HV': 63, 'T': 8320, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 143, 'seed': 20260960, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_scaled', 'dtype': 'bf16', 'B': 128, 'HK': 4, 'HV': 32, 'T': 128, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 144, 'seed': 20260961, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_scaled', 'dtype': 'fp16', 'B': 128, 'HK': 4, 'HV': 32, 'T': 128, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 145, 'seed': 20260962, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64', 'dtype': 'bf16', 'B': 128, 'HK': 4, 'HV': 32, 'T': 128, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 146, 'seed': 20260963, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64', 'dtype': 'fp16', 'B': 128, 'HK': 4, 'HV': 32, 'T': 128, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 147, 'seed': 20260964, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31_T64_scaled', 'dtype': 'bf16', 'B': 128, 'HK': 2, 'HV': 64, 'T': 64, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 148, 'seed': 20260965, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31_T64_scaled', 'dtype': 'fp16', 'B': 128, 'HK': 2, 'HV': 64, 'T': 64, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 149, 'seed': 20260966, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31_cs128_T128_scaled', 'dtype': 'bf16', 'B': 64, 'HK': 2, 'HV': 64, 'T': 128, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 150, 'seed': 20260967, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_31_cs128_T128_scaled', 'dtype': 'fp16', 'B': 64, 'HK': 2, 'HV': 64, 'T': 128, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 151, 'seed': 20260968, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01', 'dtype': 'bf16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 152, 'seed': 20260969, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01', 'dtype': 'fp16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 153, 'seed': 20260970, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_scaled', 'dtype': 'bf16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 154, 'seed': 20260971, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_scaled', 'dtype': 'fp16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 155, 'seed': 20260972, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_scaled', 'dtype': 'bf16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 156, 'seed': 20260973, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_scaled', 'dtype': 'fp16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 157, 'seed': 20260974, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05', 'dtype': 'bf16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 158, 'seed': 20260975, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05', 'dtype': 'fp16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 159, 'seed': 20260976, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_09_scaled', 'dtype': 'bf16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 160, 'seed': 20260977, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_09_scaled', 'dtype': 'fp16', 'B': 64, 'HK': 8, 'HV': 8, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 161, 'seed': 20260978, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_10_scaled', 'dtype': 'bf16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 162, 'seed': 20260979, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_10_scaled', 'dtype': 'fp16', 'B': 32, 'HK': 16, 'HV': 16, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 163, 'seed': 20260980, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_11_scaled', 'dtype': 'bf16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 164, 'seed': 20260981, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_11_scaled', 'dtype': 'fp16', 'B': 16, 'HK': 32, 'HV': 32, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 165, 'seed': 20260982, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_cs128', 'dtype': 'bf16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 166, 'seed': 20260983, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_cs128', 'dtype': 'fp16', 'B': 128, 'HK': 4, 'HV': 4, 'T': 1024, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 167, 'seed': 20260984, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_scaled', 'dtype': 'bf16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 2048, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 168, 'seed': 20260985, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_scaled', 'dtype': 'fp16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 2048, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 169, 'seed': 20260986, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_12_scaled', 'dtype': 'bf16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 2048, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 170, 'seed': 20260987, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_12_scaled', 'dtype': 'fp16', 'B': 8, 'HK': 32, 'HV': 32, 'T': 2048, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 171, 'seed': 20260988, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 172, 'seed': 20260989, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 173, 'seed': 20260990, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 174, 'seed': 20260991, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 175, 'seed': 20260992, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 176, 'seed': 20260993, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 177, 'seed': 20260994, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 178, 'seed': 20260995, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 179, 'seed': 20260996, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 180, 'seed': 20260997, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 181, 'seed': 20260998, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 182, 'seed': 20260999, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_27_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 2, 'HV': 64, 'T': 8192, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 183, 'seed': 20261000, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 184, 'seed': 20261001, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 185, 'seed': 20261002, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 186, 'seed': 20261003, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 187, 'seed': 20261004, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 188, 'seed': 20261005, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 189, 'seed': 20261006, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 190, 'seed': 20261007, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 32768, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 191, 'seed': 20261008, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 192, 'seed': 20261009, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_21_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 193, 'seed': 20261010, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_cs64', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 194, 'seed': 20261011, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V256_24_cs64', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 32, 'T': 16384, 'K': 128, 'V': 256, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 195, 'seed': 20261012, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 65536, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 196, 'seed': 20261013, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 65536, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 197, 'seed': 20261014, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_cs128', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 65536, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 198, 'seed': 20261015, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_cs128', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 65536, 'K': 128, 'V': 128, 'chunk_size': 128, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 199, 'seed': 20261016, 'route': 'ascendc', 'soc': 'ascend950'},
]


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    return deepcopy(PROFILES[index % len(PROFILES)])


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_gated_delta_rule_fwd_h")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = _dtype(spec.get("dtype", "bf16"))
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config

