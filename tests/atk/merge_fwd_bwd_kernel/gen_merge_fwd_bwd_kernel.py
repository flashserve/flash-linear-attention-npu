"""merge_fwd_bwd_kernel 的 ATK 用例生成器。

精度用例按逻辑分支冻结，每条 3 个固定种子。性能用例取 CP 合并的模型档
（N = cp - 1，rank = cp - 1，以及对应的反向 rank）。内存用例按两个可达
TilingKey（BF16=10，FP32=30）覆盖 N=1、单步 GEMM、ping-pong、多 head 和 S 上界。

直接落地三份 JSON（不需要 ATK）：

    python gen_merge_fwd_bwd_kernel.py --summary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


OP_NAME = "merge_fwd_bwd_kernel"
SOC = "ascend910b"
ROUTE = "ascendc"
SEED0 = 20260817
SEEDS_PER_CASE = 3

STANDARD = {
    "acc": {
        "cv_fused_double_benchmark": {
            "max_re_ratio": 5,
            "avg_re_ratio": 1.5,
            "root_mean_squared_ratio": 1.5,
        }
    },
    "perf": "not_key",
}

_CPU_PY = Path(__file__).resolve().parent / "scripts" / "merge_fwd_bwd_kernel_cpu.py"


def _cpu():
    spec = importlib.util.spec_from_file_location("merge_fwd_bwd_kernel_cpu", _CPU_PY)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(_CPU_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _p(name, dtype, world, heads, n_ranks, rank, forward, state_v_first, noncontig=False, note=""):
    return {
        "name": name,
        "dtype": dtype,
        "S": int(world),
        "HV": int(heads),
        "pre_or_post_num_ranks": int(n_ranks),
        "rank": int(rank),
        "forward": bool(forward),
        "state_v_first": bool(state_v_first),
        "ag_hm_noncontiguous": bool(noncontig),
        "note": note,
    }


def accuracy_profiles() -> list[dict]:
    """28 条逻辑用例。case id = 逻辑下标 * 3 + 种子下标。"""
    return [
        _p("MFB-01_unit_n1_bf16", "bf16", 1, 1, 1, 0, True, False, note="N=1 单世界，只拷贝 He，不起 Cube"),
        _p("MFB-02_unit_n1_fp32_vfirst", "fp32", 1, 1, 1, 0, True, True, note="N=1 转置写出"),
        _p("MFB-03_n1_fwd_src0", "bf16", 4, 2, 1, 1, True, False, note="forward rank==N，src=0"),
        _p("MFB-04_n1_bwd_odd_vfirst", "fp32", 4, 3, 1, 0, False, True, note="backward N=1，奇数 HV"),
        _p("MFB-05_n1_fwd_lastrank", "bf16", 8, 4, 1, 7, True, False, note="forward 最后一个 rank，只拷贝"),
        _p("MFB-06_n2_fwd", "bf16", 4, 4, 2, 2, True, False, note="一步 GEMM，不写下一档 M"),
        _p("MFB-07_n2_bwd_odd_vfirst", "fp32", 4, 5, 2, 1, False, True, note="一步 GEMM + 转置 + 奇数 HV"),
        _p("MFB-08_n3_fwd_pp", "bf16", 8, 8, 3, 5, True, False, note="ping-pong 两块 M bank"),
        _p("MFB-09_n3_bwd_pp_vfirst", "fp32", 8, 7, 3, 2, False, True, note="ping-pong + 反向 + 转置"),
        _p("MFB-10_n8_fwd_last", "bf16", 16, 16, 8, 15, True, False, note="较长前向链，末 rank"),
        _p("MFB-11_n8_bwd_first_vfirst", "fp32", 16, 16, 8, 0, False, True, note="较长反向链，首 rank"),
        _p("MFB-12_hv1_chain_fwd", "bf16", 8, 1, 4, 4, True, False, note="单 head 多步，另一 AIV 不拥有 head"),
        _p("MFB-13_hv1_chain_bwd_vfirst", "fp32", 8, 1, 4, 0, False, True, note="单 head 反向并转置"),
        _p("MFB-14_model_cp8_fp32_fwd", "fp32", 8, 32, 7, 7, True, False, note="设计测量档：N=cp-1，rank=cp-1，FP32"),
        _p("MFB-15_model_cp8_bf16_bwd_vfirst", "bf16", 8, 32, 7, 0, False, True, note="cp=8 最坏反向，BF16 转置"),
        _p("MFB-16_model_cp8_fp32_vfirst", "fp32", 8, 32, 7, 7, True, True, note="设计测量档 + state_v_first"),
        _p("MFB-17_s1024_n1_fwd", "fp32", 1024, 1, 1, 1023, True, False, note="S 上界，只拷贝"),
        _p("MFB-18_s1024_n1_bwd_vfirst", "bf16", 1024, 1, 1, 0, False, True, note="S 上界反向 N=1"),
        _p("MFB-19_s1024_n2_fwd", "fp32", 1024, 2, 2, 1023, True, False, note="S 上界一步 GEMM"),
        _p("MFB-20_hv256_n2_fwd", "bf16", 4, 256, 2, 3, True, False, note="HV 上界，每核多个 head"),
        _p("MFB-21_hv256_n3_bwd_vfirst", "fp32", 6, 256, 3, 0, False, True, note="HV 上界 + ping-pong + 转置"),
        _p("MFB-22_noncontig_fwd", "bf16", 4, 4, 3, 3, True, False, True, note="ag_hm 最后一维不连续"),
        _p("MFB-23_noncontig_bwd_vfirst", "fp32", 4, 6, 3, 0, False, True, True, note="不连续 ag_hm + 反向转置"),
        _p("MFB-24_maxchain_fwd", "fp32", 8, 2, 7, 7, True, False, note="N=S-1 前向"),
        _p("MFB-25_maxchain_bwd_vfirst", "bf16", 8, 2, 7, 0, False, True, note="N=S-1 反向"),
        _p("MFB-26_both_aiv_hv2", "bf16", 4, 2, 2, 2, True, False, note="HV=2，两个 AIV 各拥有一个 head"),
        _p("MFB-27_cp2_n1_fp32", "fp32", 2, 32, 1, 1, True, False, note="cp=2，N=1 前向"),
        _p("MFB-28_cp2_n1_bf16_bwd_vfirst", "bf16", 2, 32, 1, 0, False, True, note="cp=2，N=1 反向"),
    ]


def perf_profiles() -> list[dict]:
    """模型档。设计测量是最后一档合并：N=cp-1，rank=cp-1，FP32。没有单独的达标线。"""
    return [
        _p("PERF-cp8-hv32-fp32-fwd", "fp32", 8, 32, 7, 7, True, False, note="设计测量档 cp=8 HV=32 FP32 前向"),
        _p("PERF-cp8-hv32-bf16-fwd", "bf16", 8, 32, 7, 7, True, False, note="同档 BF16"),
        _p("PERF-cp8-hv32-fp32-bwd", "fp32", 8, 32, 7, 0, False, False, note="cp=8 最坏反向"),
        _p("PERF-cp8-hv32-fp32-vfirst", "fp32", 8, 32, 7, 7, True, True, note="设计测量档 + 转置写出"),
        _p("PERF-cp16-hv32-fp32-fwd", "fp32", 16, 32, 15, 15, True, False, note="cp=16 HV=32"),
        _p("PERF-cp64-hv32-fp32-fwd", "fp32", 64, 32, 63, 63, True, False, note="cp=64 长链"),
        _p("PERF-cp8-hv64-fp32-fwd", "fp32", 8, 64, 7, 7, True, False, note="KDA 头数 HV=64"),
        _p("PERF-cp8-hv8-fp32-fwd", "fp32", 8, 8, 7, 7, True, False, note="GDN 小头数 HV=8"),
    ]


def mss_profiles() -> list[dict]:
    """每个可达 TilingKey 至少一条，并覆盖 N=1 / 单步 GEMM / ping-pong / 多 head / S 上界。"""
    return [
        _p("MSS-bf16-n1", "bf16", 2, 2, 1, 1, True, False, note="TilingKey 10，N=1 不写 scratch"),
        _p("MSS-fp32-n1-vfirst", "fp32", 1, 1, 1, 0, True, True, note="TilingKey 30，N=1 转置"),
        _p("MSS-bf16-n2", "bf16", 4, 2, 2, 2, True, False, note="TilingKey 10，一步 GEMM，两个 AIV 都是 owner"),
        _p("MSS-fp32-n3-odd-vfirst", "fp32", 8, 3, 3, 2, False, True, note="TilingKey 30，ping-pong、奇数 HV、转置"),
        _p("MSS-bf16-hv256-n3", "bf16", 4, 256, 3, 3, True, False, note="TilingKey 10，HV=256 每核多个 head"),
        _p("MSS-fp32-s1024-n2", "fp32", 1024, 1, 2, 1023, True, False, note="TilingKey 30，S 上界单 head GEMM"),
    ]


def _spec(profile: dict, case_id: int, seed: int, logical_id: int) -> dict:
    cpu = _cpu()
    spec = dict(profile)
    spec.update(
        {
            "op": OP_NAME,
            "case_id": case_id,
            "logical_id": logical_id,
            "seed": seed,
            "route": ROUTE,
            "soc": SOC,
        }
    )
    spec["tiling_key"] = cpu.tiling_key(spec["dtype"])
    cpu.assert_legal(spec)
    return spec


def _input(name, dtype, range_values, input_type="attr", shape=None):
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": range_values,
        "backward": False,
    }


def _case_payload(case_id: int, profile: dict, seed: int, logical_id: int) -> dict:
    spec = _spec(profile, case_id, seed, logical_id)
    inputs = [
        _input("low_precision_marker", "bf16", [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input("case_spec", "non_param", json.dumps(spec, ensure_ascii=False, separators=(",", ":"))),
        _input("dtype", "string", spec["dtype"]),
        _input("S", "int", spec["S"]),
        _input("HV", "int", spec["HV"]),
        _input("pre_or_post_num_ranks", "int", spec["pre_or_post_num_ranks"]),
        _input("rank", "int", spec["rank"]),
        _input("forward", "bool", spec["forward"]),
        _input("state_v_first", "bool", spec["state_v_first"]),
        _input("ag_hm_noncontiguous", "bool", spec["ag_hm_noncontiguous"]),
        _input("tiling_key", "int", spec["tiling_key"]),
        _input("case_id", "int", spec["case_id"]),
        _input("seed", "int", spec["seed"]),
        _input("soc", "string", spec["soc"]),
        _input("route", "string", spec["route"]),
    ]
    return {
        "id": case_id,
        "default_seed": spec["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{spec['name']}",
        "aclnn_name": "",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "expected_error_msg": None,
        "backward": False,
        "standard": STANDARD,
        "outputs": None,
        "inputs": inputs,
    }


def expand_seeds(profiles: list[dict]) -> list[tuple[dict, int, int]]:
    rows = []
    for logical_id, profile in enumerate(profiles):
        for _ in range(SEEDS_PER_CASE):
            rows.append((profile, SEED0 + len(rows), logical_id))
    return rows


def _write_json(path: Path, payloads: list) -> None:
    path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _check_coverage(accuracy: list[dict], perf: list[dict], mss: list[dict]) -> None:
    cpu = _cpu()
    for group in (accuracy, perf, mss):
        for profile in group:
            cpu.assert_legal(profile)
    names = [item["name"] for item in accuracy]
    if len(names) != len(set(names)):
        raise RuntimeError("accuracy profile names are not unique")
    if {cpu.tiling_key(item["dtype"]) for item in accuracy} != {10, 30}:
        raise RuntimeError("accuracy cases miss a tiling key")
    if {cpu.tiling_key(item["dtype"]) for item in mss} != {10, 30}:
        raise RuntimeError("mss cases miss a tiling key")
    n_values = {item["pre_or_post_num_ranks"] for item in accuracy}
    if not ({1, 2} <= n_values and any(n >= 3 for n in n_values)):
        raise RuntimeError(f"accuracy N coverage is incomplete: {sorted(n_values)}")
    required = {
        "dtype": {"bf16", "fp32"},
        "forward": {True, False},
        "state_v_first": {True, False},
        "ag_hm_noncontiguous": {True, False},
        "S": None,
        "HV": None,
    }
    for field, expected in required.items():
        actual = {item[field] for item in accuracy}
        if expected is not None and actual != expected:
            raise RuntimeError(f"{field} coverage {actual} != {expected}")
    worlds = {item["S"] for item in accuracy}
    heads = {item["HV"] for item in accuracy}
    if 1 not in worlds or 1024 not in worlds:
        raise RuntimeError(f"S boundary missing: {worlds}")
    if 1 not in heads or 256 not in heads:
        raise RuntimeError(f"HV boundary missing: {heads}")
    if not any(item["HV"] % 2 == 1 and item["pre_or_post_num_ranks"] > 1 for item in accuracy):
        raise RuntimeError("missing odd-HV gemm case")
    if not any(item["S"] == 1 and item["pre_or_post_num_ranks"] == 1 for item in accuracy):
        raise RuntimeError("missing unit-world case")


def emit(root: Path | None = None) -> None:
    root = Path(__file__).resolve().parent if root is None else Path(root)
    accuracy = accuracy_profiles()
    perf = perf_profiles()
    mss = mss_profiles()
    _check_coverage(accuracy, perf, mss)
    _cpu().self_check()
    expanded = expand_seeds(accuracy)
    _write_json(
        root / f"atk_{OP_NAME}.json",
        [_case_payload(i, profile, seed, logical_id) for i, (profile, seed, logical_id) in enumerate(expanded)],
    )
    _write_json(
        root / f"atk_{OP_NAME}_perf.json",
        [_case_payload(i, profile, SEED0 + i, i) for i, profile in enumerate(perf)],
    )
    _write_json(
        root / f"atk_{OP_NAME}_mss.json",
        [_case_payload(i, profile, SEED0 + i, i) for i, profile in enumerate(mss)],
    )


def _summary() -> str:
    accuracy = accuracy_profiles()
    mss = mss_profiles()
    keys = sorted({_cpu().tiling_key(item["dtype"]) for item in mss})
    return (
        f"logical={len(accuracy)} accuracy={len(accuracy) * SEEDS_PER_CASE} "
        f"perf={len(perf_profiles())} mss={len(mss)} tiling_keys={keys}"
    )


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register(f"generator_{OP_NAME}")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            self._rows = expand_seeds(accuracy_profiles())

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0) % len(self._rows)
            profile, seed, logical_id = self._rows[index]
            payload = _case_payload(index, profile, seed, logical_id)
            case_config.id = payload["id"]
            case_config.default_seed = payload["default_seed"]
            case_config.name = payload["name"]
            by_name = {item["name"]: item for item in payload["inputs"]}
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = "bf16"
                elif cfg.name in by_name:
                    cfg.range_values = by_name[cfg.name]["range_values"]
            return case_config


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize frozen {OP_NAME} ATK cases.")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    emit()
    if args.summary:
        print(_summary())


if __name__ == "__main__":
    main()
