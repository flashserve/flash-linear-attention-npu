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

@GENERATOR_REGISTRY.register("generator_chunk_kda_bwd_prepare")
class ChunkKdaBwdPrepareGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x = {item.name: item for item in case_config.inputs}
        b, h, t, _ = x["aqk"].shape  # YAML declares head-major seed dimensions.
        layout, variable = value(x["layout"]), bool(value(x["is_varlen"]))
        if variable or layout in ("NTD", "TND"):
            b = 1
        for name, width in (("aqk", 64), ("v_new", 128), ("d_o", 128)):
            x[name].shape = token_shape(b, h, t, width, layout)
        n = chunk_count(t, variable)
        x["h"].shape = ([n, h, 128, 128] if layout in ("NTD", "TND")
                         else [b, n, h, 128, 128])
        metadata_shapes(x, t, variable)
        return case_config
