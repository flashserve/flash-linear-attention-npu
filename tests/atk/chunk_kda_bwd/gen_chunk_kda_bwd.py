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
DIM = 128
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


def _normal(std, mean=0.0):
    return {"name": "nd", "mean": [mean], "std": [std]}


def _tensor(name, dtype, shape, range_values, required=True):
    return dict(
        name=name, type="tensor", required=required, dtype=dtype, shape=shape,
        range_values=range_values, backward=False,
    )


def _attr(name, dtype, value):
    return dict(
        name=name, type="attr", required=True, dtype=dtype, shape=None,
        range_values=value, backward=False,
    )


def _inputs(config):
    packed = bool(config.get("cu_seqlens"))
    batch, heads, tokens = (int(config[name]) for name in ("B", "H", "T"))
    token_shape = [heads, tokens, DIM] if packed else [batch, heads, tokens, DIM]
    beta_shape = [heads, tokens] if packed else [batch, heads, tokens]
    cu_seqlens = ",".join(str(value) for value in config.get("cu_seqlens", []))
    has_dt_bias = bool(config.get("dt_bias", True))
    return [
        _tensor("q", config["dtype"], token_shape, _normal(0.05)),
        _tensor("k", config["dtype"], token_shape, _normal(0.05)),
        _tensor("v", config["dtype"], token_shape, _normal(0.05)),
        _tensor("raw_g", "fp32", token_shape, _normal(1.25)),
        _tensor("beta", config["beta_dtype"], beta_shape, [0.1, 0.9]),
        _tensor("d_o", config["dtype"], token_shape, _normal(51.2)),
        _tensor("A_log", "fp32", [heads], _normal(0.12)),
        _tensor(
            "dt_bias", "fp32", [heads, DIM],
            _normal(1.65, mean=-3.0) if has_dt_bias else "default", required=False,
        ),
        _attr("cu_seqlens", "string", cu_seqlens),
        _attr("disable_recompute", "bool", bool(config["disable_recompute"])),
        _attr("norm", "bool", bool(config["norm"])),
        _attr("safe_gate", "bool", bool(config["safe_gate"])),
        _attr("lower_bound", "float", float(config["lower_bound"])),
        _attr("soc", "string", config["soc"]),
        _attr("dtype", "string", config["dtype"]),
        _attr("beta_dtype", "string", config["beta_dtype"]),
        _attr("has_dt_bias", "bool", has_dt_bias),
        _attr("seed", "int", int(config["seed"])),
    ]


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
        inputs=_inputs(config),
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
            expected_names = [item["name"] for item in generated["inputs"]]
            actual_names = [
                (item[0] if isinstance(item, list) else item).name
                for item in config.inputs
            ]
            if actual_names != expected_names:
                raise ValueError("YAML input names or order differ from the generated public inputs")
            config.id = index
            config.default_seed = values["seed"]
            config.expected_error_msg = None
            config.standard = type(config.standard)(**generated["standard"])
            for item, source in zip(config.inputs, generated["inputs"]):
                item = item[0] if isinstance(item, list) else item
                item.dtype = source["dtype"]
                item.shape = source["shape"]
                item.range_values = source["range_values"]
                item.required = source["required"]
            return config


if __name__ == "__main__":
    write_cases(Path(__file__).parent)
