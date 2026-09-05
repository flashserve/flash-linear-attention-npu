#!/usr/bin/env python3
"""生成 5 个 GDN 语义场景、合计 500 条 A2 双标杆用例。"""

from __future__ import annotations

import math

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig

from case_matrix_contract import CASES_PER_SCENARIO, SCENARIOS, case_contract


Q = 0
K = 1
V = 2
G = 3
BETA = 4
SCALE = 5
CHUNK = 6
IS_VARLEN = 7
SCENARIO = 8
CU_SPEC = 9
QKV_DTYPE = 10

def _dtype_name(value) -> str:
    text = str(value).lower()
    return "bf16" if "bf16" in text or "bfloat16" in text else "fp16"


@GENERATOR_REGISTRY.register("generator_chunk_gated_delta_rule_fwd")
class ChunkGatedDeltaRuleFwdGenerator(CaseGenerator):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._case_index = 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        total_cases = len(SCENARIOS) * CASES_PER_SCENARIO
        case_index = self._case_index % total_cases
        self._case_index += 1
        scenario_index = case_index // CASES_PER_SCENARIO
        scenario = SCENARIOS[scenario_index]
        local_index = case_index % CASES_PER_SCENARIO
        (
            batch,
            k_heads,
            v_heads,
            tokens,
            value_dim,
            chunk_size,
            is_varlen,
            cu_spec,
        ) = case_contract(scenario, local_index)

        qkv_dtype = case_config.inputs[Q].dtype
        for index in (K, V, BETA):
            case_config.inputs[index].dtype = qkv_dtype
        case_config.inputs[Q].shape = [batch, k_heads, tokens, 128]
        case_config.inputs[K].shape = [batch, k_heads, tokens, 128]
        case_config.inputs[V].shape = [batch, v_heads, tokens, value_dim]
        case_config.inputs[G].shape = [batch, tokens, v_heads]
        case_config.inputs[BETA].shape = [batch, tokens, v_heads]
        case_config.inputs[SCALE].range_values = 1.0 / math.sqrt(128)
        case_config.inputs[CHUNK].range_values = chunk_size
        case_config.inputs[IS_VARLEN].range_values = is_varlen
        case_config.inputs[SCENARIO].range_values = scenario
        case_config.inputs[CU_SPEC].range_values = cu_spec
        case_config.inputs[QKV_DTYPE].range_values = _dtype_name(qkv_dtype)
        return case_config
