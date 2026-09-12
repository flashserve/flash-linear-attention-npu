"""从统一用例清单生成 chunk_kda_fwd_prepare 的冻结 ATK 用例。"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from itertools import product
from pathlib import Path

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


OP_NAME = "chunk_kda_fwd_prepare"
MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "op_cases" / f"{OP_NAME}.json"
)


def _load_generation_config() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("op") != OP_NAME:
        raise ValueError(
            f"{MANIFEST_PATH}: op must be {OP_NAME!r}, "
            f"got {manifest.get('op')!r}"
        )
    try:
        return manifest["atk_generation"]
    except KeyError as exc:
        raise ValueError(
            f"{MANIFEST_PATH}: missing atk_generation"
        ) from exc


GENERATION = _load_generation_config()
SEED_BASE = int(GENERATION["seed_base"])
STANDARD = deepcopy(GENERATION["standard"])
DEFAULT_SPEC = deepcopy(GENERATION["defaults"])
TEMPLATE_MATRIX = GENERATION["template_matrix"]
ACCURACY_SEED_PLAN = GENERATION["accuracy_seed_plan"]

DTYPE_TOKENS = {"bf16": 10, "fp32": 30}
BETA_MODE_TOKENS = {"raw": 0, "sigmoid": 1, "two_sigmoid": 2}
GATE_MODE_TOKENS = {"precomputed": 0, "softplus": 1, "safe": 2}
OUTPUT_MODE_TOKENS = {"none": 0, "recompute": 1, "save": 2}


def _positive(case_key: str, **updates) -> dict:
    spec = {"case_key": case_key, **deepcopy(DEFAULT_SPEC)}
    spec.update(updates)
    return spec


def _named_specs(group: str) -> list[dict]:
    specs = []
    for declaration in GENERATION[group]:
        overrides = deepcopy(declaration.get("overrides", {}))
        repeat = overrides.pop("cu_seqlens_repeat", None)
        if repeat is not None:
            values = [repeat["value"]] * int(repeat["count"])
            values.extend(repeat.get("tail", ()))
            overrides["cu_seqlens"] = ",".join(str(value) for value in values)
        specs.append(
            _positive(
                declaration["case_key"],
                **overrides,
            )
        )
    return specs


def _mode_values(beta_mode: str, gate_mode: str) -> dict:
    try:
        beta_values = TEMPLATE_MATRIX["beta_mode_values"][beta_mode]
        gate_values = TEMPLATE_MATRIX["gate_mode_values"][gate_mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown template mode: beta={beta_mode}, gate={gate_mode}"
        ) from exc
    return {**deepcopy(beta_values), **deepcopy(gate_values)}


def _as_template_spec(spec: dict) -> bool:
    axes = TEMPLATE_MATRIX["axes"]
    return (
        spec["gate_dtype"] in axes["gate_dtype"]
        and spec["beta_dtype"] in axes["beta_dtype"]
        and spec["backward_mode"] in axes["output_mode"]
        and (not spec["allow_neg_eigval"] or spec["use_beta_sigmoid_in_kernel"])
        and (
            spec["use_gate_in_kernel"]
            or (not spec["safe_gate"] and not spec["dt_bias"])
        )
    )


def _template_signature(spec: dict, *, include_output_mode: bool) -> tuple:
    if not _as_template_spec(spec):
        raise ValueError(f"invalid template spec: {spec['case_key']}")
    beta_mode = (
        "raw"
        if not spec["use_beta_sigmoid_in_kernel"]
        else ("two_sigmoid" if spec["allow_neg_eigval"] else "sigmoid")
    )
    gate_mode = (
        "precomputed"
        if not spec["use_gate_in_kernel"]
        else ("safe" if spec["safe_gate"] else "softplus")
    )
    signature = (
        spec["gate_dtype"],
        spec["beta_dtype"],
        bool(spec["use_qk_l2norm_in_kernel"]),
        beta_mode,
        gate_mode,
        bool(spec["use_exp2"]),
    )
    if include_output_mode:
        return signature + (spec["backward_mode"],)
    return signature


def _expected_tiling_key(spec: dict) -> int:
    signature = _template_signature(spec, include_output_mode=True)
    gate_dtype, beta_dtype, norm, beta_mode, gate_mode, use_exp2, output_mode = signature
    safe_gate = gate_mode == "safe"
    return (
        DTYPE_TOKENS[gate_dtype]
        + (DTYPE_TOKENS[beta_dtype] << 8)
        + (int(norm) << 16)
        + (BETA_MODE_TOKENS[beta_mode] << 17)
        + (GATE_MODE_TOKENS[gate_mode] << 19)
        + (int(use_exp2) << 21)
        + (int(safe_gate) << 22)
        + (OUTPUT_MODE_TOKENS[output_mode] << 23)
    )


def _template_specs(suite_name: str) -> list[dict]:
    axes = TEMPLATE_MATRIX["axes"]
    suite = TEMPLATE_MATRIX["suites"][suite_name]
    dt_bias_rule = TEMPLATE_MATRIX["dt_bias_rule"]
    if dt_bias_rule.get("selection") != "matrix_index_even":
        raise ValueError("unsupported dt_bias selection rule")
    include_output_mode = bool(suite.get("include_output_mode", False))
    output_modes = axes["output_mode"] if include_output_mode else (None,)
    profiles = suite.get("profiles", ())
    specs = []
    combinations = product(
        axes["gate_dtype"],
        axes["beta_dtype"],
        axes["norm"],
        axes["beta_mode"],
        axes["gate_mode"],
        axes["use_exp2"],
        output_modes,
    )
    for matrix_index, combination in enumerate(combinations):
        (
            gate_dtype,
            beta_dtype,
            norm,
            beta_mode,
            gate_mode,
            use_exp2,
            output_mode,
        ) = combination
        values = {
            "gate_dtype": gate_dtype,
            "beta_dtype": beta_dtype,
            "use_qk_l2norm_in_kernel": norm,
            "use_exp2": use_exp2,
            **_mode_values(beta_mode, gate_mode),
            **deepcopy(suite.get("overrides", {})),
        }
        values["dt_bias"] = (
            gate_mode in dt_bias_rule["gate_modes"]
            and matrix_index % 2 == 0
        )
        output_suffix = ""
        if output_mode is not None:
            values["backward_mode"] = output_mode
            output_suffix = f"_{output_mode}"
        profile_suffix = ""
        tags = suite["tags"]
        if profiles:
            profile_group = matrix_index // len(output_modes)
            profile_index = (
                profile_group % len(profiles)
                + profile_group // len(profiles)
            ) % len(profiles)
            profile = profiles[profile_index]
            values.update(deepcopy(profile["overrides"]))
            values["mss_profile"] = profile["name"]
            tags = f"{tags},{profile['tags']}"
            profile_suffix = f"_{profile['name']}"
        key = (
            f"{suite['prefix']}_{gate_dtype}_{beta_dtype}_"
            f"{'l2' if norm else 'identity'}_{beta_mode}_{gate_mode}_"
            f"{'exp2' if use_exp2 else 'exp'}{output_suffix}{profile_suffix}"
        )
        spec = _positive(key, tags=tags, **values)
        expected_signature = (
            gate_dtype,
            beta_dtype,
            bool(norm),
            beta_mode,
            gate_mode,
            bool(use_exp2),
        )
        if include_output_mode:
            expected_signature += (output_mode,)
        if _template_signature(
            spec, include_output_mode=include_output_mode
        ) != expected_signature:
            raise AssertionError(
                f"{suite_name}: profile changed template signature: {key}"
            )
        specs.append(spec)
    return specs


def _assert_template_matrix(
    name: str, specs: list[dict], *, include_output_mode: bool
) -> None:
    axes = TEMPLATE_MATRIX["axes"]
    expected_signatures = frozenset(
        product(
            axes["gate_dtype"],
            axes["beta_dtype"],
            axes["norm"],
            axes["beta_mode"],
            axes["gate_mode"],
            axes["use_exp2"],
        )
    )
    if include_output_mode:
        expected_signatures = frozenset(
            (*signature, output_mode)
            for signature in expected_signatures
            for output_mode in axes["output_mode"]
        )
    signatures = [
        _template_signature(spec, include_output_mode=include_output_mode)
        for spec in specs
    ]
    if len(signatures) != len(set(signatures)):
        raise AssertionError(f"{name}: duplicate template signatures")
    if frozenset(signatures) != expected_signatures:
        raise AssertionError(f"{name}: incomplete template matrix")


def _assert_unique(name: str, specs: list[dict]) -> None:
    keys = [spec["case_key"] for spec in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError(f"{name}: duplicate case_key")


def _number_specs(specs: list[dict], seed_offset: int) -> list[dict]:
    numbered = deepcopy(specs)
    _assert_unique("numbered", numbered)
    for case_id, spec in enumerate(numbered):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + seed_offset + case_id)
        spec["expected_tiling_key"] = _expected_tiling_key(spec)
    return numbered


def build_accuracy_specs() -> list[dict]:
    logical_specs = _named_specs("functional_cases")
    minimum = int(ACCURACY_SEED_PLAN["minimum_per_logical_case"])
    total = int(ACCURACY_SEED_PLAN["total_cases"])
    if minimum < 3:
        raise AssertionError("accuracy: every logical case needs at least 3 seeds")
    if len(logical_specs) * minimum > total:
        raise AssertionError("accuracy: seed plan cannot fit the requested total")
    specs = []
    remaining = total - len(logical_specs) * minimum
    for logical_id, logical_spec in enumerate(logical_specs):
        repeat_count = minimum + int(logical_id < remaining)
        for seed_index in range(repeat_count):
            spec = deepcopy(logical_spec)
            logical_key = spec["case_key"]
            spec["logical_case_key"] = logical_key
            spec["seed_index"] = seed_index
            spec["case_key"] = f"{logical_key}_seed{seed_index}"
            spec["seed"] = SEED_BASE + logical_id * 10 + seed_index
            specs.append(spec)
    if len(specs) != total:
        raise AssertionError(
            f"accuracy: expected {total} cases, got {len(specs)}"
        )
    return _number_specs(specs, 0)


def build_perf_specs() -> list[dict]:
    return _number_specs(_named_specs("performance_cases"), 1000)


def build_mss_specs() -> list[dict]:
    matrix = _template_specs("mss")
    _assert_template_matrix("determinism/mss", matrix, include_output_mode=True)
    specs = _number_specs(matrix, 2000)
    tiling_keys = [spec["expected_tiling_key"] for spec in specs]
    if len(tiling_keys) != 432 or len(set(tiling_keys)) != 432:
        raise AssertionError(
            "determinism/mss: expected 432 unique reachable tiling keys"
        )
    return specs


def _input(
    name: str,
    dtype: str,
    value,
    *,
    input_type: str = "attr",
    shape=None,
) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


ATTR_DTYPES = {
    "case_key": "string",
    "soc": "string",
    "route": "string",
    "dtype": "string",
    "B": "int",
    "HK": "int",
    "HV": "int",
    "T": "int",
    "K": "int",
    "V": "int",
    "layout": "string",
    "chunk_size": "int",
    "gate_dtype": "string",
    "beta_dtype": "string",
    "scale": "float",
    "epsilon": "float",
    "use_qk_l2norm_in_kernel": "bool",
    "use_gate_in_kernel": "bool",
    "use_beta_sigmoid_in_kernel": "bool",
    "allow_neg_eigval": "bool",
    "safe_gate": "bool",
    "lower_bound": "float",
    "use_exp2": "bool",
    "backward_mode": "string",
    "dt_bias": "bool",
    "cu_seqlens": "string",
    "explicit_chunk_indices": "bool",
    "tags": "string",
    "seed": "int",
}


def _case_payload(case_id: int, spec: dict) -> dict:
    metadata = deepcopy(spec)
    metadata["case_id"] = case_id
    inputs = [
        _input(
            "low_precision_marker",
            "bf16",
            [0, 0],
            input_type="tensor",
            shape=[1],
        ),
        _input(
            "fp32_marker",
            "fp32",
            [0, 0],
            input_type="tensor",
            shape=[1],
        ),
        _input(
            "case_spec",
            "non_param",
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    ]
    inputs.extend(
        _input(name, dtype, metadata[name])
        for name, dtype in ATTR_DTYPES.items()
    )
    return {
        "id": case_id,
        "default_seed": metadata["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{metadata['case_key']}",
        "aclnn_name": None,
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_kda_fwd_prepare",
        "expected_error_msg": None,
        "backward": False,
        "standard": STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def _payloads(specs: list[dict]) -> list[dict]:
    return [_case_payload(case_id, spec) for case_id, spec in enumerate(specs)]


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd_prepare")
    class ChunkKdaFwdPrepareGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            if CaseConfig is None:
                raise RuntimeError("ATK is required to build CaseConfig objects")
            self.cases = [
                CaseConfig(**payload)
                for payload in _payloads(build_accuracy_specs())
            ]
            self.length = len(self.cases)
            self.index = 0

        def generate(self) -> CaseConfig:
            case = self.cases[self.index]
            self.index += 1
            return case


def _write(path: Path, specs: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(_payloads(specs), ensure_ascii=False, indent=2) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy = build_accuracy_specs()
    perf = build_perf_specs()
    mss = build_mss_specs()
    _write(args.output_dir / f"atk_{OP_NAME}.json", accuracy)
    _write(args.output_dir / f"atk_{OP_NAME}_perf.json", perf)
    _write(args.output_dir / f"atk_{OP_NAME}_mss.json", mss)
    if args.summary:
        print(
            f"accuracy={len(accuracy)} perf={len(perf)} "
            f"determinism={len(mss)} mss={len(mss)}"
        )


if __name__ == "__main__":
    main()
