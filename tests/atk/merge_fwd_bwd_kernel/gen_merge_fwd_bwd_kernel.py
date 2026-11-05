"""Repair relationships from MergeFwdBwdKernel ACLNN/tiling validation."""
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generator_merge_fwd_bwd_kernel")
class MergeFwdBwdKernelGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = {value.name: value for value in case_config.inputs}
        ag = inputs["ag_hm"]
        world = min(1024, max(1, int(ag.shape[0])))
        heads = min(256, max(1, int(ag.shape[1])))
        ag.shape = [world, heads, 128, 256]
        inputs["h"].shape = [heads, 128, 128]
        inputs["h"].dtype = ag.dtype
        count = int(inputs["pre_or_post_num_ranks"].range_values)
        rank = int(inputs["rank"].range_values)
        if world == 1:
            count, rank = 1, 0
        else:
            count = min(world - 1, max(1, count))
            if bool(inputs["forward"].range_values):
                rank = min(world - 1, max(count, rank))
            else:
                rank = min(world - count - 1, max(0, rank))
        inputs["pre_or_post_num_ranks"].range_values = count
        inputs["rank"].range_values = rank
        return case_config
