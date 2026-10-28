import json
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    GENERATOR_REGISTRY = None


OP_NAME = "chunk_kda_bwd"
SEEDS = (0, 20260630, 7)


def _profile(key, disable_recompute, packed, norm, **updates):
    spec = dict(
        op=OP_NAME, case_key=key, B=1, H=8, T=100, dtype="bf16",
        beta_dtype="bf16", dt_bias=True, soc="ascend950", safe_gate=True,
        lower_bound=-5.0, disable_recompute=disable_recompute, norm=norm,
    )
    if packed:
        spec["T"] = 128
        spec["cu_seqlens"] = [0, 64, 128]
    spec.update(updates)
    return spec


PROFILES = [
    _profile("saved_dense", True, False, False, T=64),
    _profile("saved_packed", True, True, False),
    _profile("saved_dense_norm_beta32", True, False, True, beta_dtype="fp32"),
    _profile("saved_packed_norm", True, True, True, T=130, cu_seqlens=[0, 97, 130]),
    _profile("recompute_dense", False, False, False),
    _profile("recompute_packed", False, True, False),
    _profile("recompute_dense_norm_beta32", False, False, True, T=64, beta_dtype="fp32"),
    _profile("recompute_packed_norm", False, True, True, T=130, cu_seqlens=[0, 97, 130]),
    _profile("saved_dense_odd_norm", True, False, True, H=3),
    _profile("saved_packed_odd", True, True, False, H=3, T=130, cu_seqlens=[0, 97, 130]),
    _profile("saved_dense_b2_nobias_tail", True, False, False, B=2, dt_bias=False),
    _profile("recompute_dense_b2_nobias_tail", False, False, False, B=2, dt_bias=False),
]


def spec(index):
    values = dict(PROFILES[(index // len(SEEDS)) % len(PROFILES)])
    values["seed"] = SEEDS[index % len(SEEDS)]
    return values


def case(index, config):
    output_count = 7 if config.get("dt_bias", True) else 6
    dtypes = {str(i): "bf16" for i in range(output_count)}
    return dict(
        id=index, default_seed=config["seed"], name=OP_NAME,
        aclnn_name="ChunkKdaBwd", version="v2.1", api="pytorch",
        api_type="executor_chunk_kda_bwd", expected_error_msg=None, backward=False,
        standard={"acc": {"mixed_tolerance_bm": {"output_dtype_overrides": dtypes}},
                  "perf": "not_key"},
        outputs=None,
        inputs=[
            dict(name="low_precision_marker", type="tensor", required=True,
                 dtype="bf16", shape=[1], range_values=[0, 0], backward=False),
            dict(name="fp32_marker", type="tensor", required=True,
                 dtype="fp32", shape=[1], range_values=[0, 0], backward=False),
            dict(name="case_spec", type="attr", required=True, dtype="non_param",
                 shape=None, range_values=json.dumps(config, separators=(",", ":")),
                 backward=False),
        ],
    )


def _model_spec(saved):
    return dict(
        op=OP_NAME, case_key="model_h96_saved" if saved else "model_h96_recompute",
        B=1, H=96, T=8192, dtype="bf16", beta_dtype="bf16", dt_bias=True,
        seed=20260630, soc="ascend950", safe_gate=True, lower_bound=-5.0,
        disable_recompute=saved, norm=False,
    )


def write_cases(root):
    accuracy = [case(i, spec(i)) for i in range(len(PROFILES) * len(SEEDS))]
    perf = [case(i, _model_spec(saved)) for i, saved in enumerate((True, False))]
    mss = []
    for index, profile in enumerate(PROFILES):
        values = dict(profile, seed=SEEDS[0])
        mss.append(case(index, values))
    for suffix, values in (("", accuracy), ("_perf", perf), ("_mss", mss)):
        (root / f"atk_{OP_NAME}{suffix}.json").write_text(
            json.dumps(values, indent=2) + "\n", encoding="utf-8"
        )


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_kda_bwd")
    class Generator(CaseGenerator):
        def after_case_config(self, config):
            index = max(int(self.index) - 1, 0)
            values = spec(index)
            generated = case(index, values)
            config.id = index
            config.default_seed = values["seed"]
            config.expected_error_msg = None
            config.standard = type(config.standard)(**generated["standard"])
            for item in config.inputs:
                item = item[0] if isinstance(item, list) else item
                if item.name == "case_spec":
                    item.range_values = json.dumps(values, separators=(",", ":"))
            return config


if __name__ == "__main__":
    write_cases(Path(__file__).parent)
