"""chunk_gated_delta_rule_fwd_h 的 ATK 泛化用例生成器。

基于 GDN 泛化用例表筛选出 34 个 shape × bf16/fp16 = 68 条；仅保留 V=128、chunk_size=64。
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
    {'name': 'BSND_noGVA_V128_13_T512_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 0, 'seed': 20260817, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_T512_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 1, 'seed': 20260818, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 2, 'seed': 20260821, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_13_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 3, 'seed': 20260822, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64_T3584_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 4, 'seed': 20260831, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64_T3584_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 5, 'seed': 20260832, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 6, 'seed': 20260843, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_26_cs64_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 4, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 7, 'seed': 20260844, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_T640_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 8, 'seed': 20260845, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_T640_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 9, 'seed': 20260846, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_T640_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 10, 'seed': 20260847, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_T640_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 11, 'seed': 20260848, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_T640_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 12, 'seed': 20260849, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_T640_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 13, 'seed': 20260850, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64_T64_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 4, 'HV': 32, 'T': 64, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 14, 'seed': 20260869, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64_T64_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 4, 'HV': 32, 'T': 64, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 15, 'seed': 20260870, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_T512_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 16, 'seed': 20260871, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_T512_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 17, 'seed': 20260872, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_T512_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 18, 'seed': 20260873, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_T512_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 19, 'seed': 20260874, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_T512_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 20, 'seed': 20260875, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_T512_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 21, 'seed': 20260876, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_T512_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 22, 'seed': 20260877, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_T512_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 23, 'seed': 20260878, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_T1024_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 24, 'seed': 20260887, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_T1024_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 25, 'seed': 20260888, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_B32_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 26, 'seed': 20260891, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_B32_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 27, 'seed': 20260892, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_B16_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 28, 'seed': 20260893, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_B16_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 29, 'seed': 20260894, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_B64_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 30, 'seed': 20260895, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_B64_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 31, 'seed': 20260896, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_B4_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 32, 'seed': 20260899, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_B4_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 33, 'seed': 20260900, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_T8192_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 34, 'seed': 20260901, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_T8192_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 35, 'seed': 20260902, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_T8192_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 36, 'seed': 20260903, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_T8192_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 37, 'seed': 20260904, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_T16384_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 38, 'seed': 20260913, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_T16384_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 39, 'seed': 20260914, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_T32768_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 40, 'seed': 20260925, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_T32768_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 41, 'seed': 20260926, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_scaled_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 42, 'seed': 20260941, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_14_scaled_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 43, 'seed': 20260942, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_scaled_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 44, 'seed': 20260943, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_15_scaled_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 45, 'seed': 20260944, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_scaled_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 46, 'seed': 20260945, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_16_scaled_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 47, 'seed': 20260946, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 4, 'HV': 32, 'T': 128, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 48, 'seed': 20260963, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_30_cs64_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 4, 'HV': 32, 'T': 128, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 49, 'seed': 20260964, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 50, 'seed': 20260969, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_01_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 51, 'seed': 20260970, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_scaled_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 52, 'seed': 20260971, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_02_scaled_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 53, 'seed': 20260972, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_scaled_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 54, 'seed': 20260973, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_03_scaled_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 55, 'seed': 20260974, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 56, 'seed': 20260975, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_05_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 4, 'HV': 4, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 57, 'seed': 20260976, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_scaled_scaled', 'dtype': 'bf16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 58, 'seed': 20260985, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_04_scaled_scaled', 'dtype': 'fp16', 'B': 4, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 59, 'seed': 20260986, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_scaled_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 60, 'seed': 20260989, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_19_scaled_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 32, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 61, 'seed': 20260990, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_scaled_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 62, 'seed': 20260991, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_GVA_V128_25_scaled_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 32, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 63, 'seed': 20260992, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 64, 'seed': 20261001, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_17_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 16, 'HV': 16, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 65, 'seed': 20261002, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_scaled', 'dtype': 'bf16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 66, 'seed': 20261013, 'route': 'ascendc', 'soc': 'ascend950'},
    {'name': 'BSND_noGVA_V128_18_scaled', 'dtype': 'fp16', 'B': 1, 'HK': 8, 'HV': 8, 'T': 512, 'K': 128, 'V': 128, 'chunk_size': 64, 'op': 'chunk_gated_delta_rule_fwd_h', 'case_id': 67, 'seed': 20261014, 'route': 'ascendc', 'soc': 'ascend950'},
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
