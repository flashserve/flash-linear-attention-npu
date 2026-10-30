"""校验并复用 chunk32 冻结 ATK 用例，不重写正式验收 JSON。"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig, InputCaseConfig
except ModuleNotFoundError as exc:
    if exc.name not in {"atk", "torch"}:
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None
    InputCaseConfig = None


OP_NAME = "chunk_kda_fwd_chunk32"
SUITES = {
    "accuracy": ("", 40),
    "performance": ("_perf", 2),
    "mss": ("_mss", 5),
}
LEGACY_MARKERS = ("low_precision_marker", "fp32_marker", "case_spec")


@lru_cache(maxsize=3)
def _cases(suite: str) -> list[dict]:
    suffix, expected_count = SUITES[suite]
    path = Path(__file__).resolve().parent / f"atk_{OP_NAME}{suffix}.json"
    with path.open(encoding="utf-8") as stream:
        cases = json.load(stream)
    if not isinstance(cases, list) or len(cases) != expected_count:
        actual_count = len(cases) if isinstance(cases, list) else "non-list"
        raise ValueError(
            f"{path.name}: expected {expected_count} cases, got {actual_count}"
        )

    names = [case.get("name") for case in cases]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{path.name}: every case must have a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: duplicate case name")

    expected_inputs = None
    for index, case in enumerate(cases):
        if case.get("id") != index:
            raise ValueError(
                f"{path.name}: expected contiguous id {index}, got {case.get('id')}"
            )
        inputs = case.get("inputs")
        if not isinstance(inputs, list):
            raise ValueError(f"{path.name}: case {index} inputs must be a list")
        input_names = [_input_name(item) for item in inputs]
        if expected_inputs is None:
            expected_inputs = input_names
            if tuple(input_names[:3]) != LEGACY_MARKERS:
                raise ValueError(
                    f"{path.name}: frozen cases must keep the legacy marker schema"
                )
        elif input_names != expected_inputs:
            raise ValueError(
                f"{path.name}: case {index} input order differs from case 0"
            )

        case_spec = next(
            item for item in inputs if _input_name(item) == "case_spec"
        )
        spec = json.loads(case_spec["range_values"])
        if not isinstance(spec, dict) or spec.get("chunk_size") != 32:
            raise ValueError(
                f"{path.name}: case {index} case_spec must describe chunk_size=32"
            )
        if suite == "accuracy":
            benchmark = case.get("standard", {}).get("acc", {}).get(
                "cv_fused_double_benchmark", {}
            )
            expected_benchmark = {
                "max_re_ratio": 5,
                "avg_re_ratio": 1.5,
                "root_mean_squared_ratio": 1.5,
            }
            if benchmark != expected_benchmark:
                raise ValueError(
                    f"{path.name}: case {index} must keep the frozen fused benchmark"
                )
    return cases


def _input_name(item) -> str:
    config = item[0] if isinstance(item, list) else item
    return config["name"] if isinstance(config, dict) else config.name


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd_chunk32")
    class ChunkKdaFwdChunk32Generator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = int(self.index) - 1
            cases = _cases("accuracy")
            if not 0 <= index < len(cases):
                raise IndexError(
                    f"ATK requested case {index}, but only {len(cases)} exist"
                )

            saved = cases[index]
            expected_inputs = [_input_name(item) for item in case_config.inputs]
            saved_inputs = [_input_name(item) for item in saved["inputs"]]
            if expected_inputs != saved_inputs:
                raise ValueError("YAML input order differs from the frozen public inputs")

            case_config.inputs = [
                [InputCaseConfig(**deepcopy(value)) for value in item]
                if isinstance(item, list)
                else InputCaseConfig(**deepcopy(item))
                for item in saved["inputs"]
            ]
            for name in (
                "id",
                "default_seed",
                "name",
                "aclnn_name",
                "api_type",
                "backward",
                "expected_error_msg",
                "outputs",
                "standard",
                "save_name",
            ):
                setattr(case_config, name, deepcopy(saved[name]))
            return case_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    for suite in SUITES:
        _cases(suite)
    if args.summary:
        print(" ".join(f"{suite}={len(_cases(suite))}" for suite in SUITES))


if __name__ == "__main__":
    main()
