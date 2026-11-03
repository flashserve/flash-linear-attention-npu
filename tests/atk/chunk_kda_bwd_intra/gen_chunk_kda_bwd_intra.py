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

@GENERATOR_REGISTRY.register("generator_chunk_kda_bwd_intra")
class ChunkKdaBwdIntraGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x = {item.name: item for item in case_config.inputs}
        b, h, t, k = x["q"].shape
        layout, variable = value(x["layout"]), bool(value(x["is_varlen"]))
        # ACLNN permits dense BSND/BNSD and variable BSND/TND.
        if layout == "TND":
            variable = True
            x["is_varlen"].range_values = True
        if variable and layout == "BNSD":
            layout = "BSND"
            x["layout"].range_values = layout
        if variable:
            b, k = 1, 128
        shape = token_shape(b, h, t, k, layout)
        for name in ("q", "k", "gk", "dq", "dk", "dg"):
            x[name].shape = shape.copy()
        for name in ("beta", "db"):
            x[name].shape = shape[:-1]
        for name in ("dAqk", "dAkk"):
            x[name].shape = shape[:-1] + [64]
        metadata_shapes(x, t, variable)
        return case_config
