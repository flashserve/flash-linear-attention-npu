from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generator_kda_gate_cumsum")
class KdaGateCumsumGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = {value.name: value for value in case_config.inputs}
        g = inputs["g"]
        # ACLNN rank/positive logical dimensions/K<=256; YAML declares candidates.
        # ATK synthetic upper_border may use INT32_MAX independently of
        # dim_values. Project it back to the YAML-declared finite test domain.
        design = next(value for value in self.config.inputs if value.name == "g")
        declared_axes = design.shapes.dim_values
        raw_shape = g.shape
        shape = [max(1, min(int(dim), max(axis.values)))
                 for dim, axis in zip(raw_shape, declared_axes[-len(raw_shape):])]
        if len(shape) == 4:
            batch, heads, tokens, key_dim = shape
        else:
            heads, tokens, key_dim = shape
            batch = 1
        key_dim = min(key_dim, 256)
        rank3 = bool(inputs["use_rank3"].range_values)
        varlen = bool(inputs["varlen"].range_values)
        if varlen or rank3:
            batch = 1
        g.shape = [heads, tokens, key_dim] if rank3 else [batch, heads, tokens, key_dim]
        inputs["a_log"].shape = [heads]
        inputs["dt_bias"].shape = [heads * key_dim]
        inputs["cu_seqlens"].shape = [3]
        # The int-array is materialized as [0, T//2, T] in build_inputs.
        return case_config
