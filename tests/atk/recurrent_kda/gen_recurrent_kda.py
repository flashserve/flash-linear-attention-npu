"""recurrent_kda 的 ATK 泛化用例生成器。"""

from __future__ import annotations

import json
from copy import deepcopy

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None

OP_NAME = "recurrent_kda"
def base_spec(**values):
    spec = dict(dtype="bf16", B=1, T=2, H=2, HV=4, K=128, V=128,
                layout="BSND", state_v_first=True, soc="ascend950", op=OP_NAME)
    spec.update(values)
    return spec


def profiles():
    output = [base_spec(name="bf16_small", B=2, state_zero=True)]
    for flags in range(8):
        output.append(base_spec(name=f"fused_{flags}", T=9,
            use_qk_l2norm_in_kernel=bool(flags & 1),
            use_gate_in_kernel=bool(flags & 2),
            use_beta_sigmoid_in_kernel=bool(flags & 4),
            g_dtype="bf16" if flags % 2 else "fp32",
            beta_dtype="bf16" if flags % 3 else "fp32",
            qkv_view="fused", inplace=bool(flags & 1)))
    for i, tokens in enumerate([1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 65, 129, 257, 1024]):
        output.append(base_spec(name=f"window_{tokens}", T=tokens,
            layout="TND" if i % 2 else "BSND", state_v_first=bool(i % 2),
            state_dtype="bf16" if i % 3 else "fp32",
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=bool(i % 2)))
    for ratio in [1, 3, 16]:
        output.append(base_spec(name=f"ratio_{ratio}", H=1, HV=ratio, T=17,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True))
    for pattern in ["zero", "small"]:
        output.append(base_spec(name=f"norm_{pattern}", T=17, pattern=pattern,
            use_qk_l2norm_in_kernel=True))
    for view in ["fused", "head_gap", "feature_gap"]:
        output.append(base_spec(name=f"view_{view}", T=17, B=2, qkv_view=view,
            strided_state=True, use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True))
    for packed in ["packed", "speculative"]:
        for layout in ["BSND", "TND"]:
            output.append(base_spec(name=f"slots_{packed}_{layout}", T=17,
                lengths=[0, 9, 17, 0, 33], layout=layout, padding=5,
                indices=packed, accepted=9, metadata_dtype="int32" if layout == "TND" else "int64",
                state_v_first=layout == "TND", state_dtype="bf16" if layout == "TND" else "fp32",
                strided_state=True, inplace=layout == "BSND",
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                safe_gate=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True))
    output += [
        base_spec(name="dense_no_metadata", B=2, T=17, cu=False),
        base_spec(name="tnd_no_metadata", T=33, layout="TND", cu=False),
        base_spec(name="zero_initial", T=17, initial_none=True, use_qk_l2norm_in_kernel=True),
        base_spec(name="large_gate", T=17, use_gate_in_kernel=True, large_gate=True),
        base_spec(name="bf16_kfirst_regression", T=9, V=128, state_v_first=False,
                  state_dtype="bf16", use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True),
        base_spec(name="empty_sequences", T=17, lengths=[0, 17, 0, 9], padding=3),
        base_spec(name="raw_gate_no_bias", B=2, T=17, dt_bias_none=True,
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True),
        base_spec(name="safe_gate_no_bias", B=2, T=17, dt_bias_none=True,
                  safe_gate=True, use_gate_in_kernel=True, state_v_first=False),
        base_spec(name="norm_signal", T=17, pattern="signal", state_zero=True,
                  use_qk_l2norm_in_kernel=True),
    ]
    return output


PROFILES = profiles()


def model_specs():
    result = []
    for batch in [32, 64, 72]:
        for hv in [32, 48, 128]:
            for tokens in [1, 2, 4, 8]:
                result.append(base_spec(name=f"B{batch}_H16_HV{hv}_T{tokens}",
                    B=batch, T=tokens, H=16, HV=hv, inplace=True, qkv_view="fused",
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, seed=20261020, case_id=len(result)))
    return result


def contract_specs():
    """Regression matrix for metadata/gate inference and unrestricted heads."""
    profiles = [
        base_spec(name="dense_ignored_layout", B=2, T=5, cu=False, layout="ignored"),
        base_spec(name="packed_ignored_layout", T=5, layout="ignored", lengths=[0, 5, 2]),
        base_spec(name="raw_gate_flag_false", T=5, a_log=True, use_gate_in_kernel=False),
        base_spec(name="raw_gate_no_bias_flag_false", T=5, a_log=True,
                  dt_bias_none=True, use_gate_in_kernel=False),
        base_spec(name="step_gate_flag_true", T=5, a_log=False, use_gate_in_kernel=True),
        base_spec(name="safe_gate_flag_false", T=5, a_log=True, safe_gate=True),
        base_spec(name="dense_zero_initial", B=2, T=5, cu=False, initial_none=True),
        base_spec(name="heads257", T=2, H=257, HV=257,
                  use_qk_l2norm_in_kernel=True, a_log=True),
        base_spec(name="ratio17_heads289", T=2, H=17, HV=289,
                  state_v_first=False, state_dtype="bf16", a_log=True),
        base_spec(name="ratio257_heads514", T=2, H=2, HV=514,
                  a_log=True, qkv_view="fused", inplace=True),
        base_spec(name="dense_fp16_aux", B=2, T=5, cu=False,
                  g_dtype="fp16", beta_dtype="fp16", a_log=True,
                  use_beta_sigmoid_in_kernel=True),
        base_spec(name="heads258_kfirst", T=2, H=129, HV=258, V=128,
                  a_log=True, state_v_first=False),
    ]
    return [dict(spec, seed=20261027 + repeat * len(profiles) + i)
            for repeat in range(3) for i, spec in enumerate(profiles)]


def typical_specs():
    names = {"bf16_small", "fused_7", "window_17", "window_1024",
             "view_feature_gap", "slots_packed_BSND", "slots_speculative_TND",
             "dense_no_metadata", "zero_initial", "large_gate", "bf16_kfirst_regression",
             "empty_sequences", "raw_gate_no_bias", "safe_gate_no_bias", "norm_signal"}
    return [dict(spec, seed=20261027 + i) for i, spec in enumerate(PROFILES)
            if spec["name"] in names]


def typical_model_specs():
    # One head count for each (batch, length); all head counts are represented
    # at every length across the three batches. Include B72/HV128/T8.
    head_counts = {32: [128, 32, 48, 32], 64: [32, 48, 128, 48],
                   72: [48, 128, 32, 128]}
    return [spec for spec in model_specs()
            if spec["HV"] == head_counts[spec["B"]][[1, 2, 4, 8].index(spec["T"])]]


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({"op": OP_NAME, "case_id": index, "seed": 20260817 + index, "route": "ascendc", "soc": "ascend950"})
    return profile

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_recurrent_kda")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = _dtype(spec.get("dtype", "bf16"))
                elif cfg.name == "case_spec":
                    cfg.dtype = "string"
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


def alignment_specs():
    """Small matrix for the revised A5 register/bank/metadata contract."""
    result = []
    for state_dtype in ("fp32", "bf16"):
        for v_first in (True, False):
            for snapshots in (False, True):
                result.append(base_spec(
                    name=f"align_{state_dtype}_{v_first}_{snapshots}", B=2, T=17, H=2, HV=32,
                    state_dtype=state_dtype, state_v_first=v_first, inplace=True,
                    indices=("packed" if v_first else "speculative") if snapshots else None,
                    accepted=2, metadata_dtype="int32" if snapshots else "int64",
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, safe_gate=not v_first,
                    strided_state=True))
    for length in (1, 2, 4, 8):
        result.append(base_spec(name=f"align_lmax_{length}", lengths=[0, length, length - 1],
                                padding=32, state_v_first=False, g_dtype="fp16", beta_dtype="fp16"))
    result.append(base_spec(name="align_all_empty", lengths=[0, 0], padding=16))
    for v_first in (True, False):
        result.append(base_spec(name=f"align_batch_gap_{v_first}", B=2, T=9, cu=False,
                                qkv_view="batch_gap", state_v_first=v_first,
                                state_dtype="bf16", use_qk_l2norm_in_kernel=True))
    for safe in (False, True):
        result.append(base_spec(name=f"align_ignored_gate_{safe}", T=9, a_log=False,
                                ignored_gate_parameters=True, safe_gate=safe, lower_bound=123.0))
    result.append(base_spec(name="align_dense32", B=2, T=32, cu=False,
                            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True))
    return [dict(spec, seed=20261029 + seed, case_id=3 * index + seed)
            for index, spec in enumerate(result) for seed in range(3)]


def generate_packages():
    # Accuracy candidates, user model shapes, and manually selected key paths
    # have separate sources; performance/mss are never synthesized by gen_cases.
    from pathlib import Path
    root = Path(__file__).resolve().parent
    template = json.loads((root / "atk_recurrent_kda.json").read_text(encoding="utf-8"))[0]
    groups = {
        "atk_recurrent_kda.json": [_spec(i) for i in range(len(PROFILES) * 3)],
        "atk_recurrent_kda_perf.json": model_specs(),
        "atk_recurrent_kda_contract.json": contract_specs(),
        "atk_recurrent_kda_alignment.json": alignment_specs(),
        "atk_recurrent_kda_typical.json": typical_specs(),
        "atk_recurrent_kda_perf_typical.json": typical_model_specs(),
        "atk_recurrent_kda_mss.json": [base_spec(**item) for item in [
            dict(name="key0_short", T=1, inplace=True),
            dict(name="key0_window", T=17, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True),
            dict(name="key0_tail", T=33, layout="TND", qkv_view="fused", state_dtype="bf16"),
            dict(name="key0_transpose", T=17, state_v_first=False, strided_state=True),
            dict(name="key0_slots", T=17, indices="speculative", inplace=True,
                 use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True),
            dict(name="key0_packed", T=17, lengths=[0,9,17], layout="TND", indices="packed"),
            dict(name="key0_fallback", T=9, qkv_view="feature_gap"),
            dict(name="key0_bf16_kfirst", T=9, V=128, state_dtype="bf16", state_v_first=False),
            dict(name="key0_no_bias", B=16, T=17, dt_bias_none=True,
                 use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                 use_beta_sigmoid_in_kernel=True),
            dict(name="key0_bf16_x1_general", T=1, state_dtype="bf16", inplace=True),
            dict(name="key0_bf16_x1_fused", T=1, state_dtype="bf16", inplace=True,
                 use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                 use_beta_sigmoid_in_kernel=True),
        ]],
    }
    from copy import deepcopy
    for filename, specs in groups.items():
        cases = []
        for index, spec in enumerate(specs):
            spec.setdefault("seed", 20261020); spec["case_id"] = index
            case = deepcopy(template)
            case.update(id=index, default_seed=spec["seed"], name=f"recurrent_kda_{index:04d}_{spec['name']}")
            for value in case["inputs"]:
                if value["name"] == "case_spec":
                    value["dtype"] = "string"
                    value["range_values"] = json.dumps(spec, separators=(",", ":"))
                elif value["name"] in spec:
                    value["range_values"] = spec[value["name"]]
            cases.append(case)
        # One case per line keeps changes to the spec/seed matrix reviewable.
        content = "[\n" + ",\n".join(
            "  " + json.dumps(case, ensure_ascii=False, separators=(",", ":"))
            for case in cases) + "\n]\n"
        (root / filename).write_text(content, encoding="utf-8")
        print(filename, len(cases))


if __name__ == "__main__":
    generate_packages()
