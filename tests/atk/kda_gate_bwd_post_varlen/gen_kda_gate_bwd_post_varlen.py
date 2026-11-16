from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generator_kda_gate_bwd_post_varlen")
class KdaGatePostGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = {value.name: value for value in case_config.inputs}
        # Tiling4KdaGateBwdPost: positive B/H/T, K=128, C=64.
        # ATK synthetic upper_border may use INT32_MAX independently of
        # dim_values. Project it back to the YAML-declared finite test domain.
        design = next(value for value in self.config.inputs if value.name == "raw_g")
        declared_axes = design.shapes.dim_values
        raw_shape = inputs["raw_g"].shape
        shape = [max(1, min(int(dim), max(axis.values)))
                 for dim, axis in zip(raw_shape, declared_axes[-len(raw_shape):])]
        shape[-1] = 128
        inputs["raw_g"].shape = shape
        inputs["dg_act"].shape = list(shape)
        heads, tokens = shape[-3:-1]
        inputs["a_log"].shape = [heads]
        inputs["dt_bias"].shape = [heads * shape[-1]]
        inputs["chunk_size"].range_values = 64
        # Private launcher always feeds unscanned dg_base; kernel ignores true.
        inputs["input_scanned"].range_values = False
        inputs["cu_seqlens"].shape = [3]
        cuts = [0, tokens // 2, tokens]
        count = sum((end - start + 63) // 64 for start, end in zip(cuts, cuts[1:]))
        inputs["chunk_indices"].shape = [count, 2]
        return case_config
