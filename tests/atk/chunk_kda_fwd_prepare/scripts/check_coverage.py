"""校验冻结 ATK 用例数量、种子和全部可达 tiling key。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ATK_DIR = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ATK_DIR / "gen_chunk_kda_fwd_prepare.py"
EXPECTED_KEY_DIGEST = (
    "f28a869f07d8e3f65e8d0c85768759bde635759a09d047cd78b4a9b7136d3b19"
)


def _load_generator():
    module_name = "chunk_kda_fwd_prepare_coverage_generator"
    spec = importlib.util.spec_from_file_location(module_name, GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _check_frozen(generator, name: str, specs: list[dict]) -> None:
    path = ATK_DIR / name
    frozen = json.loads(path.read_text(encoding="utf-8"))
    expected = generator._payloads(specs)
    if frozen != expected:
        raise AssertionError(f"{name} 与统一用例清单不一致，请重新生成")


def _check_accuracy_seeds(specs: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for spec in specs:
        groups.setdefault(spec["logical_case_key"], []).append(spec)
    if len(specs) != 200 or len(groups) != 56:
        raise AssertionError("精度用例必须是 200 条、56 个逻辑场景")
    for logical_key, group in groups.items():
        seeds = [spec["seed"] for spec in group]
        seed_indices = [spec["seed_index"] for spec in group]
        if len(group) not in (3, 4):
            raise AssertionError(
                f"精度逻辑场景 {logical_key} 必须包含 3 或 4 条用例"
            )
        if len(set(seeds)) != len(group):
            raise AssertionError(f"精度逻辑场景 {logical_key} 存在重复 seed")
        if sorted(seed_indices) != list(range(len(group))):
            raise AssertionError(
                f"精度逻辑场景 {logical_key} 的 seed_index 必须从 0 连续编号"
            )


def main() -> None:
    generator = _load_generator()
    accuracy = generator.build_accuracy_specs()
    performance = generator.build_perf_specs()
    mss = generator.build_mss_specs()

    _check_frozen(generator, "atk_chunk_kda_fwd_prepare.json", accuracy)
    _check_frozen(generator, "atk_chunk_kda_fwd_prepare_perf.json", performance)
    _check_frozen(generator, "atk_chunk_kda_fwd_prepare_mss.json", mss)

    _check_accuracy_seeds(accuracy)
    logical_counts = Counter(spec["logical_case_key"] for spec in accuracy)

    expected_model_shapes = [
        (2, 16, 32, 11264),
        (1, 16, 32, 11264),
        (1, 32, 32, 65536),
        (4, 96, 96, 128),
        (1, 32, 32, 160),
        (6, 6, 6, 1084),
        (1, 12, 12, 1084),
        (1, 96, 96, 8192),
        (1, 96, 96, 16384),
        (1, 8, 24, 32768),
    ]
    model_shapes = [
        (spec["B"], spec["HK"], spec["HV"], spec["T"])
        for spec in performance
    ]
    if model_shapes != expected_model_shapes:
        raise AssertionError("性能用例没有原样保留 10 个用户模型 shape")

    keys = [spec["expected_tiling_key"] for spec in mss]
    key_digest = hashlib.sha256(
        "".join(
            f"{spec['case_id']},{spec['expected_tiling_key']}\n"
            for spec in mss
        ).encode("ascii")
    ).hexdigest()
    if len(keys) != 432 or len(set(keys)) != 432:
        raise AssertionError("确定性/内存用例未覆盖全部 432 个可达 tiling key")
    if key_digest != EXPECTED_KEY_DIGEST:
        raise AssertionError("tiling key 顺序或编码发生了未登记的变化")

    mode_counts = Counter(spec["backward_mode"] for spec in mss)
    print(
        "coverage ok: "
        f"accuracy={len(accuracy)} logical={len(logical_counts)} "
        f"performance={len(performance)} tiling_keys={len(set(keys))} "
        f"output_modes={dict(mode_counts)} sha256={key_digest}"
    )


if __name__ == "__main__":
    main()
