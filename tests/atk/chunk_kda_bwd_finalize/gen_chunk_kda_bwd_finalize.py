# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


def value(item):
    result = item.range_values
    return result[0] if isinstance(result, list) and len(result) == 1 else result


def token_shape(b, h, t, d, layout):
    if layout == "BNSD":
        return [b, h, t, d]
    if layout == "BSND":
        return [b, t, h, d]
    if layout == "NTD":
        return [h, t, d]
    return [t, h, d]  # TND


def chunk_count(t, variable):
    # Canonical packed metadata includes one empty sequence.
    lens = [0, t // 2, t - t // 2] if variable else [t]
    return sum((length + 63) // 64 for length in lens)


def metadata_shapes(items, t, variable):
    items["cu_seqlens"].shape = [4] if variable else [1]
    items["chunk_indices"].shape = [2 * chunk_count(t, variable)] if variable else [1]

@GENERATOR_REGISTRY.register("generator_chunk_kda_bwd_finalize")
class ChunkKdaBwdFinalizeGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x = {item.name: item for item in case_config.inputs}
        b, h, t, _ = x["q"].shape
        output_layout = value(x["output_layout"])
        rank3 = output_layout in ("NTD", "TND")
        variable = bool(value(x["is_varlen"]))
        if variable or rank3:
            b = 1
        shape = [h, t, 128] if rank3 else [b, h, t, 128]
        for name in ("q", "k", "v", "gk", "raw_g", "v_new", "dv_scan", "dq_raw"):
            x[name].shape = shape.copy()
        for name in ("akk", "d_aqk"):
            x[name].shape = shape[:-1] + [64]
        for name in ("beta", "q_rstd", "k_rstd"):
            x[name].shape = shape[:-1]
        n = chunk_count(t, variable)
        for name in ("h", "dh"):
            x[name].shape = [n, h, 128, 128] if rank3 else [b, n, h, 128, 128]
        x["a_log"].shape = [h]
        x["dt_bias"].shape = [h, 128]
        metadata_shapes(x, t, variable)
        return case_config
