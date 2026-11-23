"""ATK generator for deterministic VDN solver and window coverage."""
import json

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


@GENERATOR_REGISTRY.register("generator_vdn_forward")
class Generator(CaseGenerator):
    def after_case_config(self, case_config):
        from pathlib import Path

        # Registered cases are the reviewed branch matrix. Additional generated
        # samples keep these configurations and vary the common CPU/device seed.
        cases = json.loads(Path(__file__).with_name("atk_vdn_forward.json").read_text())
        index = max(int(self.index) - 1, 0)
        config = cases[index % len(cases)]
        spec = json.loads(next(x["range_values"] for x in config["inputs"] if x["name"] == "case_spec"))
        spec["seed"] += index
        case_config.id = index
        case_config.name = f"vdn_forward_{index:04d}"
        case_config.default_seed = spec["seed"]
        for entry in case_config.inputs:
            item = entry[0] if isinstance(entry, list) else entry
            if item.name == "case_spec":
                item.range_values = json.dumps(spec)
            elif item.name == "low_precision_marker":
                item.dtype = spec["dtype"]
            elif item.name in spec:
                item.range_values = spec[item.name]
        return case_config

