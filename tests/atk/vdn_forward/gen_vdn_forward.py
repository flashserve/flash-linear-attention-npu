# SPDX-License-Identifier: BSD-3-Clause
"""Couple tensor shapes and dtypes to the public VDN contract."""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


@GENERATOR_REGISTRY.register("generator_vdn_forward")
class Generator(CaseGenerator):
    def after_case_config(self, case_config):
        fields = {field.name: field for field in case_config.inputs}
        q = fields["q"]
        frames, heads, tokens, dim = q.shape
        for name in ("k", "v"):
            fields[name].shape = list(q.shape)
            fields[name].dtype = q.dtype
        fields["beta"].shape = [frames, heads, tokens]
        if q.dtype == "fp32":
            fields["beta"].dtype = q.dtype
        fields["alpha"].shape = [frames, heads, dim]
        fields["bounds"].shape = [frames, 2]
        fields["bounds"].range_values = [0, frames - 1]
        fields["text_state"].shape = [heads, dim, dim]
        return case_config
