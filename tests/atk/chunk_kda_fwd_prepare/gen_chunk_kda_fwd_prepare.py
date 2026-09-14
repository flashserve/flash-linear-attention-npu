"""生成 chunk_kda_fwd_prepare 的真实输入 ATK 用例。

生成的 JSON 直接描述 q/k/v/g/beta、可选张量和公开属性，不使用 marker
张量，也不把用例序列化进 case_spec。精度、性能和确定性集合共享同一输入契约。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from itertools import product
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig, InputCaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None
    InputCaseConfig = None


OP_NAME = "chunk_kda_fwd_prepare"
MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "op_cases"
    / "chunk_kda_fwd_prepare.json"
)


def _normalize_manifest_overrides(overrides: dict) -> dict:
    normalized = deepcopy(overrides)
    repeated = normalized.pop("cu_seqlens_repeat", None)
    if repeated is not None:
        normalized["cu_seqlens"] = tuple(
            [int(repeated["value"])] * int(repeated["count"])
            + [int(value) for value in repeated.get("tail", ())]
        )
    cu_seqlens = normalized.get("cu_seqlens")
    if isinstance(cu_seqlens, str):
        normalized["cu_seqlens"] = (
            tuple(int(value) for value in cu_seqlens.split(",") if value)
            if cu_seqlens
            else None
        )
    return normalized


def _load_generation_contract() -> tuple:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        generation = json.load(stream)["atk_generation"]

    defaults = _normalize_manifest_overrides(generation["defaults"])
    for metadata_name in (
        "tags",
        "route",
        "soc",
        "dtype",
        "data_scale",
        "gate_scale",
        "beta_scale",
    ):
        defaults.pop(metadata_name, None)

    def declarations(name: str) -> tuple:
        return tuple(
            {
                "case_key": item["case_key"],
                "overrides": _normalize_manifest_overrides(item["overrides"]),
            }
            for item in generation[name]
        )

    matrix = generation["template_matrix"]
    suite = matrix["suites"]["mss"]
    template_matrix = {
        "axes": {
            name: tuple(values)
            for name, values in matrix["axes"].items()
        },
        "beta_mode_values": deepcopy(matrix["beta_mode_values"]),
        "gate_mode_values": deepcopy(matrix["gate_mode_values"]),
        "profiles": tuple(
            (
                profile["name"],
                _normalize_manifest_overrides(profile["overrides"]),
            )
            for profile in suite["profiles"]
        ),
    }
    return (
        int(generation["seed_base"]),
        deepcopy(generation["standard"]),
        int(generation["accuracy_case_plan"]["total_cases"]),
        defaults,
        declarations("functional_cases"),
        declarations("performance_cases"),
        template_matrix,
    )


(
    SEED_BASE,
    STANDARD,
    ACCURACY_CASES,
    DEFAULT_SPEC,
    FUNCTIONAL_CASES,
    PERFORMANCE_CASES,
    TEMPLATE_MATRIX,
) = _load_generation_contract()
PERFORMANCE_DEFAULTS = {}

DTYPE_TOKENS = {"bf16": 10, "fp32": 30}
BETA_MODE_TOKENS = {"raw": 0, "sigmoid": 1, "two_sigmoid": 2}
GATE_MODE_TOKENS = {"precomputed": 0, "softplus": 1, "safe": 2}
OUTPUT_MODE_TOKENS = {"none": 0, "recompute": 1, "save": 2}


def _positive(case_key: str, **updates) -> dict:
    spec = {"case_key": case_key, **deepcopy(DEFAULT_SPEC)}
    spec.update(updates)
    return spec


def _named_specs(group: str) -> list[dict]:
    declarations = {
        "functional_cases": FUNCTIONAL_CASES,
        "performance_cases": PERFORMANCE_CASES,
    }[group]
    specs = []
    for declaration in declarations:
        overrides = deepcopy(declaration.get("overrides", {}))
        if group == "performance_cases":
            overrides = {**PERFORMANCE_DEFAULTS, **overrides}
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
    if suite_name != "mss":
        raise ValueError(f"unknown template suite: {suite_name}")
    axes = TEMPLATE_MATRIX["axes"]
    include_output_mode = True
    output_modes = axes["output_mode"]
    profiles = TEMPLATE_MATRIX["profiles"]
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
        }
        values["dt_bias"] = gate_mode != "precomputed" and matrix_index % 2 == 0
        output_suffix = ""
        if output_mode is not None:
            values["backward_mode"] = output_mode
            output_suffix = f"_{output_mode}"
        profile_suffix = ""
        if profiles:
            profile_group = matrix_index // len(output_modes)
            profile_index = (
                profile_group % len(profiles)
                + profile_group // len(profiles)
            ) % len(profiles)
            profile_name, profile_overrides = profiles[profile_index]
            values.update(deepcopy(profile_overrides))
            values["mss_profile"] = profile_name
            profile_suffix = f"_{profile_name}"
        key = (
            f"mss_matrix_{gate_dtype}_{beta_dtype}_"
            f"{'l2' if norm else 'identity'}_{beta_mode}_{gate_mode}_"
            f"{'exp2' if use_exp2 else 'exp'}{output_suffix}{profile_suffix}"
        )
        spec = _positive(key, **values)
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


def _number_specs(
    specs: list[dict], seed_offset: int, suite: str
) -> list[dict]:
    numbered = deepcopy(specs)
    _assert_unique("numbered", numbered)
    for case_id, spec in enumerate(numbered):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + seed_offset + case_id)
        spec["suite"] = suite
        spec["distribution"] = "uniform" if case_id % 2 == 0 else "normal"
        spec["expected_tiling_key"] = _expected_tiling_key(spec)
    return numbered


def _signature_pairs(signature: tuple) -> frozenset[tuple]:
    return frozenset(
        (left, signature[left], right, signature[right])
        for left in range(len(signature))
        for right in range(left + 1, len(signature))
    )


def _select_accuracy_matrix(specs: list[dict], count: int) -> list[dict]:
    """在固定额度内优先补齐所有合法属性二元组合。"""

    candidates = _template_specs("mss")
    candidate_data = [
        (
            candidate,
            _template_signature(candidate, include_output_mode=True),
        )
        for candidate in candidates
    ]
    universe = frozenset(
        pair
        for _, signature in candidate_data
        for pair in _signature_pairs(signature)
    )
    covered = set()
    axis_counts = Counter()
    pair_counts = Counter()
    for spec in specs:
        signature = _template_signature(spec, include_output_mode=True)
        pairs = _signature_pairs(signature)
        covered.update(pairs)
        pair_counts.update(pairs)
        axis_counts.update(enumerate(signature))

    selected = []
    remaining = list(candidate_data)
    while len(selected) < count:
        missing = universe.difference(covered)

        def score(item):
            _, signature = item
            pairs = _signature_pairs(signature)
            return (
                len(pairs.intersection(missing)),
                sum(1.0 / (1 + pair_counts[pair]) for pair in pairs),
                sum(
                    1.0 / (1 + axis_counts[(axis, value)])
                    for axis, value in enumerate(signature)
                ),
            )

        best = max(range(len(remaining)), key=lambda index: score(remaining[index]))
        spec, signature = remaining.pop(best)
        pairs = _signature_pairs(signature)
        covered.update(pairs)
        pair_counts.update(pairs)
        axis_counts.update(enumerate(signature))
        spec = deepcopy(spec)
        spec["case_key"] = spec["case_key"].replace(
            "mss_matrix_", "accuracy_matrix_", 1
        )
        selected.append(spec)

    missing = universe.difference(covered)
    if missing:
        raise AssertionError(
            f"accuracy: {len(missing)} legal attribute pairs are uncovered"
        )
    return selected


@lru_cache(maxsize=1)
def build_accuracy_specs() -> tuple[dict, ...]:
    specs = [
        *_named_specs("functional_cases"),
        *_named_specs("performance_cases"),
    ]
    specs.extend(_select_accuracy_matrix(specs, ACCURACY_CASES - len(specs)))
    if len(specs) != ACCURACY_CASES:
        raise AssertionError(
            f"accuracy: expected {ACCURACY_CASES} cases, got {len(specs)}"
        )
    return tuple(_number_specs(specs, 0, "accuracy"))


@lru_cache(maxsize=1)
def build_perf_specs() -> tuple[dict, ...]:
    keys = {
        declaration["case_key"] for declaration in PERFORMANCE_CASES
    }
    specs = [
        deepcopy(spec)
        for spec in build_accuracy_specs()
        if spec["case_key"] in keys
    ]
    if len(specs) != len(PERFORMANCE_CASES):
        raise AssertionError("performance: model cases are missing from accuracy")
    return tuple(specs)


@lru_cache(maxsize=1)
def build_mss_specs() -> tuple[dict, ...]:
    matrix = _template_specs("mss")
    _assert_template_matrix("determinism/mss", matrix, include_output_mode=True)
    specs = _number_specs(matrix, 2000, "determinism")
    tiling_keys = [spec["expected_tiling_key"] for spec in specs]
    if len(tiling_keys) != 432 or len(set(tiling_keys)) != 432:
        raise AssertionError(
            "determinism/mss: expected 432 unique reachable tiling keys"
        )
    return tuple(specs)


TENSOR_INPUTS = ("q", "k", "v", "g", "beta", "a_log", "dt_bias")
OPTIONAL_TENSORS = {"a_log", "dt_bias"}
BOOL_ATTRS = (
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "use_beta_sigmoid_in_kernel",
    "allow_neg_eigval",
    "safe_gate",
    "use_exp2",
)
EXPECTED_INPUTS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "scale",
    "layout",
    "chunk_size",
    "epsilon",
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "use_beta_sigmoid_in_kernel",
    "allow_neg_eigval",
    "safe_gate",
    "lower_bound",
    "use_exp2",
    "a_log",
    "dt_bias",
    "cu_seqlens",
    "chunk_indices",
    "backward_mode",
)


def _canonical_chunk_indices(cu_seqlens, chunk_size: int = 64):
    if cu_seqlens is None:
        return None
    indices = []
    for sequence, (begin, end) in enumerate(
        zip(cu_seqlens, cu_seqlens[1:])
    ):
        for chunk in range((end - begin + chunk_size - 1) // chunk_size):
            indices.extend((sequence, chunk))
    return tuple(indices)


def _case_shapes(spec: dict) -> dict[str, list[int]]:
    batch = int(spec["B"])
    key_heads = int(spec["HK"])
    value_heads = int(spec["HV"])
    tokens = int(spec["T"])
    key_dim = int(spec["K"])
    value_dim = int(spec["V"])
    layout = str(spec["layout"])
    if layout == "BNSD":
        q_shape = [batch, key_heads, tokens, key_dim]
        v_shape = [batch, value_heads, tokens, value_dim]
        g_shape = [batch, value_heads, tokens, key_dim]
        beta_shape = [batch, value_heads, tokens]
    elif layout == "BSND":
        q_shape = [batch, tokens, key_heads, key_dim]
        v_shape = [batch, tokens, value_heads, value_dim]
        g_shape = [batch, tokens, value_heads, key_dim]
        beta_shape = [batch, tokens, value_heads]
    elif layout == "NTD":
        q_shape = [key_heads, tokens, key_dim]
        v_shape = [value_heads, tokens, value_dim]
        g_shape = [value_heads, tokens, key_dim]
        beta_shape = [value_heads, tokens]
    elif layout == "TND":
        q_shape = [tokens, key_heads, key_dim]
        v_shape = [tokens, value_heads, value_dim]
        g_shape = [tokens, value_heads, key_dim]
        beta_shape = [tokens, value_heads]
    else:
        raise ValueError(f"unsupported layout: {layout}")
    return {
        "q": q_shape,
        "k": list(q_shape),
        "v": v_shape,
        "g": g_shape,
        "beta": beta_shape,
        "a_log": [value_heads],
        "dt_bias": [value_heads * key_dim],
    }


def _tensor_dtype(spec: dict, name: str) -> str:
    if name in {"q", "k", "v"}:
        return "bf16"
    if name == "g":
        return str(spec["gate_dtype"])
    if name == "beta":
        return str(spec["beta_dtype"])
    return "fp32"


def _tensor_present(spec: dict, name: str) -> bool:
    if name == "a_log":
        return bool(spec["use_gate_in_kernel"])
    if name == "dt_bias":
        return bool(spec["use_gate_in_kernel"] and spec["dt_bias"])
    return True


def _normal_range(
    case_id: int, name: str, *, quality: bool, spec: dict
):
    if quality:
        low_precision_std = (0.0625, 0.08838834764831843, 0.125, 0.25)
        if name in {"q", "k"}:
            mean, std = 0.0, low_precision_std[case_id % 4]
        elif name == "v":
            mean, std = 0.0, 0.05
        elif name == "g":
            mean, std = (
                (0.0, 1.25)
                if spec["use_gate_in_kernel"]
                else (-0.011, 0.004)
            )
        elif name == "beta":
            mean, std = 0.5, 0.1
        elif name == "a_log":
            mean, std = -4.0, 0.5
        else:
            mean, std = 0.0, 1.0
        return {"name": "nd", "mean": [mean], "std": [std]}
    stable = {
        "q": (0.0, 0.08),
        "k": (0.0, 0.08),
        "v": (0.0, 0.08),
        "g": (0.0, 0.5),
        "beta": (0.0, 0.5),
        "a_log": (-4.0, 1.0),
        "dt_bias": (0.0, 1.0),
    }
    mean, std = stable[name]
    return {"name": "nd", "mean": [mean], "std": [std]}


def _tensor_range(spec: dict, name: str):
    quality = spec["suite"] == "accuracy"
    if spec["distribution"] == "normal":
        return _normal_range(
            int(spec["case_id"]), name, quality=quality, spec=spec
        )
    if quality:
        ranges = {
            "q": [-0.08, 0.08],
            "k": [-0.08, 0.08],
            "v": [-0.08, 0.08],
            "g": (
                [-1.0, 1.0]
                if spec["use_gate_in_kernel"]
                else [-0.02, -0.002]
            ),
            "beta": [0.0, 1.0],
            "a_log": [-6.0, -2.0],
            "dt_bias": [-2.0, 2.0],
        }
        return ranges[name]
    stable = {
        "q": [-0.08, 0.08],
        "k": [-0.08, 0.08],
        "v": [-0.08, 0.08],
        "g": [-1.0, 1.0],
        "beta": [-1.0, 1.0],
        "a_log": [-6.0, -2.0],
        "dt_bias": [-2.0, 2.0],
    }
    return stable[name]


def _input_name(item) -> str:
    config = item[0] if isinstance(item, list) else item
    return str(config.name)


def _attrs_configs(name: str, values):
    if InputCaseConfig is None:
        raise RuntimeError("ATK is required to construct attrs inputs")
    if values is None:
        return [
            InputCaseConfig(
                name=name,
                type="attrs",
                required=False,
                dtype="string",
                shape=None,
                range_values="null",
                backward=False,
            )
        ]
    return [
        InputCaseConfig(
            name=name,
            type="attrs",
            required=False,
            dtype="int",
            shape=None,
            range_values=int(value),
            backward=False,
        )
        for value in values
    ]


def _configure_case(case_config, spec: dict, index: int):
    if [_input_name(item) for item in case_config.inputs] != list(EXPECTED_INPUTS):
        raise ValueError("YAML input order no longer matches the direct-input contract")
    shapes = _case_shapes(spec)
    cu_seqlens = spec.get("cu_seqlens")
    chunk_indices = (
        _canonical_chunk_indices(cu_seqlens, int(spec["chunk_size"]))
        if cu_seqlens is not None and spec["explicit_chunk_indices"]
        else None
    )
    list_values = {
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }
    for position, item in enumerate(case_config.inputs):
        name = _input_name(item)
        if name in list_values:
            case_config.inputs[position] = _attrs_configs(name, list_values[name])
            continue
        config = item[0] if isinstance(item, list) else item
        config.backward = False
        config.align_32B = None
        config.outlier_values = None
        if name in TENSOR_INPUTS:
            config.shape = shapes[name]
            config.dtype = _tensor_dtype(spec, name)
            config.required = name not in OPTIONAL_TENSORS
            config.range_values = (
                _tensor_range(spec, name)
                if _tensor_present(spec, name)
                else "null"
            )
            continue
        config.shape = None
        config.required = True
        if name in BOOL_ATTRS:
            config.dtype = "attr_bool"
            config.range_values = bool(spec[name])
        elif name in {"scale", "epsilon", "lower_bound"}:
            config.dtype = "float"
            config.range_values = float(spec[name])
        elif name == "chunk_size":
            config.dtype = "int"
            config.range_values = int(spec[name])
        elif name in {"layout", "backward_mode"}:
            config.dtype = "string"
            config.range_values = str(spec[name])
        else:
            raise ValueError(f"unhandled direct input: {name}")

    case_config.id = index
    case_config.default_seed = int(spec["seed"])
    case_config.name = (
        f"{OP_NAME}_{index:04d}_{spec['case_key']}_"
        f"h{spec['HK']}_hv{spec['HV']}_t{spec['T']}"
    )
    case_config.aclnn_name = "ChunkKdaFwdPrepare"
    case_config.api_type = "executor_chunk_kda_fwd_prepare"
    case_config.backward = False
    case_config.expected_error_msg = ""
    case_config.outputs = None
    case_config.save_name = OP_NAME
    case_config.is_boundary = any(
        token in spec["case_key"]
        for token in ("min", "tail", "varlen", "partition", "grid_stride")
    )
    return case_config


def _input_payload(
    name: str,
    input_type: str,
    required: bool,
    dtype: str,
    shape,
    value,
) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": required,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def _attrs_payload(name: str, values):
    if values is None:
        return [_input_payload(name, "attrs", False, "string", None, "null")]
    return [
        _input_payload(name, "attrs", False, "int", None, int(value))
        for value in values
    ]


def _case_payload(case_id: int, spec: dict) -> dict:
    shapes = _case_shapes(spec)
    inputs = []
    for name in ("q", "k", "v", "g", "beta", "a_log", "dt_bias"):
        present = _tensor_present(spec, name)
        inputs.append(
            _input_payload(
                name,
                "tensor",
                name not in OPTIONAL_TENSORS,
                _tensor_dtype(spec, name),
                shapes[name],
                _tensor_range(spec, name) if present else "null",
            )
        )
    scalar_attrs = (
        ("scale", "float", float(spec["scale"])),
        ("layout", "string", str(spec["layout"])),
        ("chunk_size", "int", int(spec["chunk_size"])),
        ("epsilon", "float", float(spec["epsilon"])),
        *( (name, "attr_bool", bool(spec[name])) for name in BOOL_ATTRS ),
        ("lower_bound", "float", float(spec["lower_bound"])),
        ("backward_mode", "string", str(spec["backward_mode"])),
    )
    # 按 YAML 的公开签名顺序重排，避免生成器和冻结 JSON 出现两套 ABI。
    attr_by_name = {
        name: _input_payload(name, "attr", True, dtype, None, value)
        for name, dtype, value in scalar_attrs
    }
    tensor_by_name = {item["name"]: item for item in inputs}
    cu_seqlens = spec.get("cu_seqlens")
    chunk_indices = (
        _canonical_chunk_indices(cu_seqlens, int(spec["chunk_size"]))
        if cu_seqlens is not None and spec["explicit_chunk_indices"]
        else None
    )
    direct_inputs = []
    for name in EXPECTED_INPUTS:
        if name in tensor_by_name:
            direct_inputs.append(tensor_by_name[name])
        elif name == "cu_seqlens":
            direct_inputs.append(_attrs_payload(name, cu_seqlens))
        elif name == "chunk_indices":
            direct_inputs.append(_attrs_payload(name, chunk_indices))
        else:
            direct_inputs.append(attr_by_name[name])
    return {
        "id": case_id,
        "default_seed": int(spec["seed"]),
        "name": (
            f"{OP_NAME}_{case_id:04d}_{spec['case_key']}_"
            f"h{spec['HK']}_hv{spec['HV']}_t{spec['T']}"
        ),
        "aclnn_name": "ChunkKdaFwdPrepare",
        "triton_name": None,
        "kernel_name": None,
        "version": "v2.1",
        "expected_error_msg": "",
        "api": "pytorch",
        "api_type": "executor_chunk_kda_fwd_prepare",
        "aclnn_api_type": "aclnn_function",
        "triton_api_type": "triton_function",
        "fusion_api_type": "fusion_function",
        "fusion_mode": None,
        "dist_api_type": "dist_function",
        "kernel_api_type": "kernel_function",
        "backward": False,
        "standard": deepcopy(STANDARD),
        "outputs": None,
        "inputs": direct_inputs,
        "acl_json": "",
        "method_inputs": None,
        "tensor_input": None,
        "compute_times": None,
        "save_name": OP_NAME,
        "uuid": None,
        "downloaded": False,
        "is_boundary": any(
            token in spec["case_key"]
            for token in ("min", "tail", "varlen", "partition", "grid_stride")
        ),
        "xrun_cs_name": None,
        "xrun_data": None,
        "strategy": None,
    }


def _payloads(specs: list[dict]) -> list[dict]:
    return [_case_payload(int(spec["case_id"]), spec) for spec in specs]


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd_prepare")
    class ChunkKdaFwdPrepareGenerator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            specs = build_accuracy_specs()
            if index >= len(specs):
                raise IndexError(
                    f"ATK requested profile {index}, but only {len(specs)} exist"
                )
            return _configure_case(case_config, specs[index], index)


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
