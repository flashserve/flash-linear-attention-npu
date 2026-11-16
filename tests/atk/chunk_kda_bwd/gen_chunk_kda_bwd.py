"""Saved-cache legacy backward constraints from the public ACLNN interface."""
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generator_chunk_kda_bwd")
class ChunkKdaBwdGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = {value.name: value for value in case_config.inputs}
        q = inputs["q"]
        shape = [max(1, int(dim)) for dim in q.shape]
        if len(shape) == 4:
            batch, heads, tokens, _ = shape
        else:
            heads, tokens, _ = shape
            batch = 1
        packed = bool(inputs["is_varlen"].range_values)
        prefix = [heads, tokens] if packed else [batch, heads, tokens]
        key_shape = prefix + [128]
        for name in ["q", "k", "v", "w", "qg", "kg", "v_new", "d_o"]:
            inputs[name].shape = list(key_shape)
            inputs[name].dtype = q.dtype
        inputs["gk"].shape = list(key_shape)
        inputs["raw_g"].shape = list(key_shape)
        inputs["beta"].shape = list(prefix)
        for name in ["Aqk", "Akk"]:
            inputs[name].shape = prefix + [64]
            inputs[name].dtype = q.dtype
        chunk_size = int(inputs["chunk_size"].range_values)
        if packed:
            split = tokens // 2
            chunks = ((split + chunk_size - 1) // chunk_size
                      + (tokens - split + chunk_size - 1) // chunk_size)
            inputs["h"].shape = [chunks, heads, 128, 128]
        else:
            chunks = (tokens + chunk_size - 1) // chunk_size
            inputs["h"].shape = [batch, chunks, heads, 128, 128]
        inputs["h"].dtype = q.dtype
        inputs["A_log"].shape = [heads]
        inputs["dt_bias"].shape = [heads, 128]
        inputs["cu_seqlens"].shape = [3]
        inputs["chunk_indices"].shape = [2 * chunks]
        return case_config
