"""将已冻结的公开输入用例提供给 ATK case 生成入口。"""

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
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None
    InputCaseConfig = None


OP_NAME = "chunk_kda_fwd_prepare"
SUITES = {"accuracy": ("", 200), "perf": ("_perf", 10), "mss": ("_mss", 432)}


@lru_cache(maxsize=3)
def _cases(suite: str) -> list[dict]:
    suffix, expected_count = SUITES[suite]
    path = Path(__file__).resolve().parent / f"atk_{OP_NAME}{suffix}.json"
    with path.open(encoding="utf-8") as stream:
        cases = json.load(stream)
    if len(cases) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} cases, got {len(cases)}")
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: duplicate case name")
    return cases


def _input_name(item) -> str:
    config = item[0] if isinstance(item, list) else item
    return config["name"] if isinstance(config, dict) else config.name


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd_prepare")
    class ChunkKdaFwdPrepareGenerator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = int(self.index) - 1
            cases = _cases("accuracy")
            if not 0 <= index < len(cases):
                raise IndexError(f"ATK requested case {index}, but only {len(cases)} exist")

            saved = cases[index]
            expected_inputs = [_input_name(item) for item in case_config.inputs]
            saved_inputs = [
                _input_name(item) for item in saved["inputs"]
            ]
            if expected_inputs != saved_inputs:
                raise ValueError("YAML input order differs from the frozen public inputs")

            case_config.inputs = [
                [InputCaseConfig(**value) for value in item]
                if isinstance(item, list)
                else InputCaseConfig(**item)
                for item in saved["inputs"]
            ]
            for name in (
                "id", "default_seed", "name", "aclnn_name", "api_type",
                "backward", "expected_error_msg", "outputs", "save_name",
                "is_boundary",
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
