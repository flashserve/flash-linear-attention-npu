"""本地验证 CPU 两路的 shape、空序列/状态重排、零梯度和独立 FP64 属性。

这不是 NPU 精度验收，不使用 CPU 两路接近来替代 mixed_tolerance_bm。
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import arch22_cpu as cpu

ROOT = Path(__file__).resolve().parents[1]


def materialize(payload):
    rng = np.random.default_rng(payload["default_seed"])
    values, gt = {}, None
    for value in payload["inputs"]:
        name = value["name"]
        if value["type"] != "tensor":
            values[name] = value["range_values"]
            continue
        low, high = value["range_values"]
        values[name] = rng.uniform(low, high, tuple(value["shape"])).astype(np.float32)
        if name == "g":
            gt = value["dtype"]
    return values, gt


def check(output):
    gen_spec = importlib.util.spec_from_file_location("finalize_generator", ROOT / "gen_chunk_gated_delta_rule_bwd_finalize.py")
    gen = importlib.util.module_from_spec(gen_spec)
    gen_spec.loader.exec_module(gen)
    cases = [value for value in gen.formal_arch22_payloads("accuracy") if value["api_type"] == gen.ARCH22_API][::3]
    measured = []
    for payload in cases:
        raw, gt = materialize(payload)
        case, data, actual = cpu.prepare_inputs(raw, gt)
        begin = time.perf_counter()
        lowp = cpu.finalize_cpu_lowp(case, data)
        lowp_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        highp = cpu.finalize_cpu_fp64(case, data)
        highp_seconds = time.perf_counter() - begin
        for name in cpu.OUTPUT_NAMES:
            assert lowp[name].shape == highp[name].shape
            assert lowp[name].dtype == np.float32 and highp[name].dtype == np.float64
            assert np.isfinite(lowp[name]).all() and np.isfinite(highp[name]).all()
        zeros = dict(data)
        for name in ("do", "p", "dh"):
            zeros[name] = np.zeros_like(data[name])
        for fn in (cpu.finalize_cpu_lowp, cpu.finalize_cpu_fp64):
            assert all(np.count_nonzero(value) == 0 for value in fn(case, zeros).values())
        measured.append(dict(source_case_id=payload["id"], name=payload["name"],
                             lowp_seconds=lowp_seconds, fp64_seconds=highp_seconds,
                             max_abs_lowp_vs_fp64={name:float(np.max(np.abs(lowp[name].astype(np.float64)-highp[name]))) for name in cpu.OUTPUT_NAMES}))
    # 同一合法 dense 输入变成含前/后空序列的 packed 输入，结果必须保持。
    raw, gt = materialize(cases[0])
    dense_case, dense_data, _ = cpu.prepare_inputs(raw, gt)
    raw["cu_seqlens"] = json.dumps([0, 0, dense_case.tokens, dense_case.tokens])
    raw["chunk_indices"] = json.dumps([1, 0])
    empty_case, empty_data, _ = cpu.prepare_inputs(raw, gt)
    for fn in (cpu.finalize_cpu_lowp, cpu.finalize_cpu_fp64):
        left, right = fn(dense_case, dense_data), fn(empty_case, empty_data)
        for name in cpu.OUTPUT_NAMES:
            np.testing.assert_array_equal(left[name], right[name])
    # 最后一条的 pair 重排与状态 bank 同步逆排，保持自然序列语义。
    raw, gt = materialize(cases[-1])
    case1, data1, _ = cpu.prepare_inputs(raw, gt)
    raw["chunk_indices"] = json.dumps([1, 0, 1, 1])
    raw["h"] = raw["h"][:, ::-1].copy()
    raw["dh"] = raw["dh"][:, ::-1].copy()
    case2, data2, _ = cpu.prepare_inputs(raw, gt)
    for fn in (cpu.finalize_cpu_lowp, cpu.finalize_cpu_fp64):
        left, right = fn(case1, data1), fn(case2, data2)
        for name in cpu.OUTPUT_NAMES:
            np.testing.assert_array_equal(left[name], right[name])
    # BF16 ties-to-even 的两个相邻 parity 边界。
    np.testing.assert_array_equal(cpu.cpu_bf16([1+2**-8, 1+3*2**-8]), np.array([1, 1+2**-6], np.float32))
    result = dict(schema="finalize-arch22-cpu-selfcheck/v1", passed=True,
                  logical_cases=len(cases), invariants=["finite_shapes", "independent_fp64_dtype", "zero_gradient",
                  "legal_empty_sequences", "reordered_state_banks", "bf16_rne_ties"], cases=measured,
                  npu_acceptance=False, tolerance_verdict="not evaluated; preserve native ATK mixed_tolerance_bm")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, logical_cases=len(cases), cpu_seconds=sum(v['lowp_seconds']+v['fp64_seconds'] for v in measured))))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    check(a.output)
