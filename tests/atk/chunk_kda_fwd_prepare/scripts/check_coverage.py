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


def main() -> None:
    generator = _load_generator()
    accuracy = generator.build_accuracy_specs()
    performance = generator.build_perf_specs()
    mss = generator.build_mss_specs()

    _check_frozen(generator, "atk_chunk_kda_fwd_prepare.json", accuracy)
    _check_frozen(generator, "atk_chunk_kda_fwd_prepare_perf.json", performance)
    _check_frozen(generator, "atk_chunk_kda_fwd_prepare_mss.json", mss)

    logical_counts = Counter(spec["logical_case_key"] for spec in accuracy)
    if len(accuracy) != 200 or min(logical_counts.values()) < 3:
        raise AssertionError("精度用例必须是 200 条且每个逻辑场景至少 3 个固定种子")

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
