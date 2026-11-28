import json

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY

PROFILES = [
    ("single", [1], 1, "BSH", "shared"),
    ("ragged", [1, 2, 3, 4], 3, "TH", "per_head"),
    ("cross", [63, 4], 3, "TH", "per_head"),
    ("long", [129], 2, "BSH", "flat"),
    ("slot_gaps", [5, 3, 8], 5, "TH", "per_head"),
    ("main_bsh", [4] * 40, 8, "BSH", "flat"),
    ("main_th", [4] * 40, 8, "TH", "flat"),
    ("wrap", [64, 64, 64, 64, 64], 2, "BSH", "flat"),
    ("max_s", [4096], 1, "BSH", "shared"),
]


@GENERATOR_REGISTRY.register("generator_mega_recurrent_kda")
class Generator(CaseGenerator):
    def after_case_config(self, case_config):
        index = (int(self.index) - 1) % len(PROFILES)
        name, lengths, heads, layout, conv = PROFILES[index]
        spec = dict(op="mega_recurrent_kda", name=name, lengths=lengths, heads=heads, layout=layout, conv=conv, seed=0)
        case_config.id = index
        case_config.name = "mega_recurrent_kda_" + name
        case_config.default_seed = spec["seed"]
        for item in case_config.inputs:
            cfg = item[0] if isinstance(item, list) else item
            if cfg.name == "case_spec":
                cfg.range_values = json.dumps(spec, separators=(",", ":"))
            elif cfg.name == "low_precision_marker":
                cfg.dtype = "bf16"
        return case_config
