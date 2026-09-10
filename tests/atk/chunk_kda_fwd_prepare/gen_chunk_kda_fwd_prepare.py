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
        and (not spec["allow_neg_eigval"] or spec["use_beta_sigmoid_in_kernel"])
        and (
            spec["use_gate_in_kernel"]
            or (not spec["safe_gate"] and not spec["dt_bias"])
        )
    )


def _template_signature(spec: dict) -> tuple:
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
    return (
        spec["gate_dtype"],
        spec["beta_dtype"],
        bool(spec["use_qk_l2norm_in_kernel"]),
        beta_mode,
        gate_mode,
        bool(spec["use_exp2"]),
    )


def _template_specs(suite_name: str) -> list[dict]:
    axes = TEMPLATE_MATRIX["axes"]
    suite = TEMPLATE_MATRIX["suites"][suite_name]
    dt_bias_rule = TEMPLATE_MATRIX["dt_bias_rule"]
    specs = []
    combinations = product(
        axes["gate_dtype"],
        axes["beta_dtype"],
        axes["norm"],
        axes["beta_mode"],
        axes["gate_mode"],
        axes["use_exp2"],
    )
    for gate_dtype, beta_dtype, norm, beta_mode, gate_mode, use_exp2 in combinations:
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
            and bool(use_exp2) == bool(dt_bias_rule["use_exp2"])
        )
        key = (
            f"{suite['prefix']}_{gate_dtype}_{beta_dtype}_"
            f"{'l2' if norm else 'identity'}_{beta_mode}_{gate_mode}_"
            f"{'exp2' if use_exp2 else 'exp'}"
        )
        specs.append(_positive(key, tags=suite["tags"], **values))
    return specs


def _assert_template_matrix(name: str, specs: list[dict]) -> None:
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
    signatures = [_template_signature(spec) for spec in specs]
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
    return numbered


def build_accuracy_specs() -> list[dict]:
    matrix = _template_specs("accuracy")
    _assert_template_matrix("accuracy", matrix)
    return _number_specs(_named_specs("functional_cases") + matrix, 0)


def build_perf_specs() -> list[dict]:
    return _number_specs(_named_specs("performance_cases"), 1000)


def build_mss_specs() -> list[dict]:
    matrix = _template_specs("mss")
    _assert_template_matrix("mss", matrix)
    return _number_specs(matrix, 2000)


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
