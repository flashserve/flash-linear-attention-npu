"""pre_process_fwd_kernel_merged 的 ATK 用例生成器（把算子自带的 41 条用例表冻结成 ATK 用例）。

用例来源：算子开发期冻结的 41 条用例（37 条精度 + 4 条性能），
以及 `docs/design.md` 3.2.1 的 TilingKey 清单。

- `atk_pre_process_fwd_kernel_merged.json`      ：37 条精度用例 × **3 个固定种子** = 111 条
  （逻辑分支/边界/变长/GVA/dtype/并行度/子区间）
- `atk_pre_process_fwd_kernel_merged_perf.json` ：4 条性能用例（用户模型 case）
- `atk_pre_process_fwd_kernel_merged_mss.json`  ：每个**可达 TilingKey** 一条精简用例（确定性/内存检测）

取值空间由 `pre_process_fwd_kernel_merged.yaml` 的 valid/invalid 声明；本文件把该空间冻结为
确定性子集（`frozen_by_gen`），`case_spec` 足额写入每个字段，不再依赖随机采样。

可用 `python gen_pre_process_fwd_kernel_merged.py` 直接落地三份 JSON（不需要 ATK 环境）。
`--standard` 选择写入三份 JSON 的 `standard.acc`：默认 `mixed_tolerance_bm`（仓内统一标准，
NPU DUT + CPU 高精度 golden）；做 **GPU 双标杆**验收时用 `--standard cv_fused_double_benchmark`
（阈值与交付仓 `FLA_ATK` 的 `chunk_gated_delta_rule_fwd_h` 一致：5 / 1.5 / 1.5）。
"""

from __future__ import annotations

import argparse
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

OP_NAME = "pre_process_fwd_kernel_merged"
ACLNN_NAME = "PreProcessFwdKernelMerged"
SOC = "ascend950"
ROUTE = "ascendc"
K_DIM = 128
V_DIM = 128
BT = 64
SEED0 = 20260818

STANDARD = {
    "acc": "mixed_tolerance_bm",
    "perf": "not_key",
}
MSS_STANDARD = {
    "acc": "mixed_tolerance_bm",
    "perf": "not_key",
    "mem": 1.1,
}

# GPU 双标杆（DUT + 同精度标杆 vs FP64 真值）用 ATK 的 `cv_fused_double_benchmark`，
# 阈值与交付仓 `FLA_ATK` 里 `chunk_gated_delta_rule_fwd_h` / `chunk_kda_fwd` 一致。
DOUBLE_BENCHMARK_STANDARD = {
    "acc": {
        "cv_fused_double_benchmark": {
            "max_re_ratio": 5,
            "avg_re_ratio": 1.5,
            "root_mean_squared_ratio": 1.5,
        }
    },
    "perf": "not_key",
}
DOUBLE_BENCHMARK_MSS_STANDARD = {
    "acc": {
        "cv_fused_double_benchmark": {
            "max_re_ratio": 5,
            "avg_re_ratio": 1.5,
            "root_mean_squared_ratio": 1.5,
        }
    },
    "perf": "not_key",
    "mem": 1.1,
}

# `--standard` 的取值：仓内统一标准（CPU 单标杆）与 GPU 双标杆标准各一套。
STANDARD_CHOICES = {
    "mixed_tolerance_bm": (STANDARD, MSS_STANDARD),
    "cv_fused_double_benchmark": (DOUBLE_BENCHMARK_STANDARD, DOUBLE_BENCHMARK_MSS_STANDARD),
}
DEFAULT_STANDARD = "mixed_tolerance_bm"

# 每个精度用例至少 3 个固定种子（tests/atk/README.md「正式验收用例包」）。
SEEDS_PER_CASE = 3


def _eq_cu(total: int, seg: int) -> list[int]:
    """等长多段：`total/seg` 段，每段 `seg` 长。"""
    assert total % seg == 0, (total, seg)
    return [i * seg for i in range(total // seg + 1)]


def _p(name, gate, gate_dtype, hk, hv, t, cu, dtype="bf16", note="", group="精度"):
    """一条精度用例的描述（`t` 是**张量** T 轴长度；`cu` 给窗口/段边界）。"""
    return dict(
        name=name, group=group, gate=gate, gate_dtype=gate_dtype,
        HK=hk, HV=hv, T=int(t), cu=[int(x) for x in cu], dtype=dtype, note=note,
    )


def accuracy_profiles() -> list[dict]:
    """37 条精度用例（PPFM-01..37）。"""
    out: list[dict] = []
    # -- 窗口规模（GDN / KDA）: T=1 / 1023 / 4096 / 8191 / 32767
    for i, t in enumerate([1, 1023, 4096, 8191, 32767]):
        out.append(_p(f"PPFM-{i+1:02d}_win_g_t{t}", "g", "fp32", 32, 32, t, [0, t], note="窗口规模/GDN"))
    for i, t in enumerate([1, 1023, 4096, 8191, 32767]):
        out.append(_p(f"PPFM-{i+6:02d}_win_gk_t{t}", "gk", "fp32", 32, 32, t, [0, t], note="窗口规模/KDA"))
    # -- 变长（2/3 段、非整除边界、单段大 T）
    out.append(_p("PPFM-11_varlen_g_2seg", "g", "fp32", 8, 8, 512, [0, 256, 512], note="变长 2 段"))
    out.append(_p("PPFM-12_varlen_g_3seg", "g", "fp32", 8, 8, 1536, [0, 512, 1024, 1536], note="变长 3 段"))
    out.append(_p("PPFM-13_varlen_g_3seg_uneven", "g", "fp32", 8, 8, 1024, [0, 64, 320, 1024], note="变长 3 段/非整除"))
    out.append(_p("PPFM-14_varlen_g_1seg_big", "g", "fp32", 8, 8, 16387, [0, 16387], note="变长 1 段/大 T"))
    out.append(_p("PPFM-15_varlen_gk_2seg", "gk", "fp32", 8, 8, 512, [0, 256, 512], note="变长 2 段"))
    out.append(_p("PPFM-16_varlen_gk_3seg", "gk", "fp32", 8, 8, 1536, [0, 512, 1024, 1536], note="变长 3 段"))
    out.append(_p("PPFM-17_varlen_gk_3seg_uneven", "gk", "fp32", 8, 8, 1024, [0, 64, 320, 1024], note="变长 3 段/非整除"))
    out.append(_p("PPFM-18_varlen_gk_1seg_big", "gk", "fp32", 8, 8, 16387, [0, 16387], note="变长 1 段/大 T"))
    # -- GVA（HK < HV，成倍数）
    out.append(_p("PPFM-19_gva_2x1", "g", "fp32", 16, 32, 2048, [0, 2048], note="GVA 2:1"))
    out.append(_p("PPFM-20_gva_3x1_odd", "g", "fp32", 21, 63, 2048, [0, 2048], note="GVA 3:1/奇数 head"))
    out.append(_p("PPFM-21_gva_4x1", "g", "fp32", 8, 32, 2048, [0, 2048], note="GVA 4:1"))
    out.append(_p("PPFM-22_gva_8x1", "g", "fp32", 4, 32, 2048, [0, 2048], note="GVA 8:1"))
    out.append(_p("PPFM-23_gva_32x1", "g", "fp32", 2, 64, 2048, [0, 2048], note="GVA 32:1"))
    out.append(_p("PPFM-24_gva_1x2_multiseg", "g", "fp32", 16, 32, 8192, _eq_cu(8192, 1024), note="变长 + GVA 1:2"))
    out.append(_p("PPFM-25_gva_1x3_big", "g", "fp32", 21, 63, 16387, [0, 16387], note="GVA 1:3/大 T"))
    # -- gate dtype 分支
    out.append(_p("PPFM-26_gate_fp32", "g", "fp32", 8, 8, 1024, [0, 1024], note="g FP32"))
    out.append(_p("PPFM-27_gate_bf16", "g", "bf16", 8, 8, 1024, [0, 1024], note="g BF16"))
    out.append(_p("PPFM-28_gk_fp32", "gk", "fp32", 8, 8, 1024, [0, 1024], note="gk FP32"))
    out.append(_p("PPFM-29_gk_bf16", "gk", "bf16", 8, 8, 1024, [0, 1024], note="gk BF16"))
    # -- 并行度（Nseq × HV）
    out.append(_p("PPFM-30_par_hv8", "g", "fp32", 8, 8, 4096, [0, 4096], note="Nseq=1 × HV=8"))
    out.append(_p("PPFM-31_par_hv16_gva", "gk", "fp32", 8, 16, 4096, [0, 4096], note="Nseq=1 × HV=16（GVA）"))
    out.append(_p("PPFM-32_par_hv32", "g", "fp32", 8, 32, 4096, [0, 4096], note="Nseq=1 × HV=32（GVA）"))
    out.append(_p("PPFM-33_par_hv64_gva", "gk", "fp32", 8, 64, 4096, [0, 4096], note="Nseq=1 × HV=64（GVA）"))
    out.append(_p("PPFM-34_par_16seg", "g", "fp32", 8, 8, 65536, _eq_cu(65536, 4096), note="Nseq=16 × HV=8"))
    out.append(_p("PPFM-35_par_64seg", "gk", "fp32", 8, 8, 262144, _eq_cu(262144, 4096), note="Nseq=64 × HV=8"))
    # -- 子区间窗口（bos > 0）
    out.append(_p("PPFM-36_subinterval_tail", "g", "fp32", 8, 8, 512, [40, 512], note="子区间 [40,512)"))
    out.append(_p("PPFM-37_subinterval_tail_gk", "gk", "fp32", 8, 8, 512, [444, 512], note="子区间 [444,512)"))
    return out


def perf_profiles() -> list[dict]:
    """4 条性能用例（PPFM-38..41，用户模型 case）。"""
    return [
        _p("PPFM-38_perf_kda_model", "gk", "fp32", 32, 32, 11264, [0, 11264], note="对齐 H20 model-gk"),
        _p("PPFM-39_perf_gdn_model", "g", "fp32", 32, 32, 11264, [0, 11264], note="对齐 H20 model-g"),
        _p("PPFM-40_perf_cp2_gva", "g", "fp32", 16, 32, 5632, [0, 5632], note="CP=2 窗口 + GVA"),
        _p("PPFM-41_perf_long", "g", "fp32", 32, 32, 16384, [0, 16384], note="长窗口"),
    ]


def mss_profiles() -> list[dict]:
    """每个**可达 TilingKey** 一条精简用例（gate × gate dtype 的 4 个组合）。

    形状选最小可行集，只要求"能走到该 key 且内存/同步关键路径被覆盖"。
    """
    return [
        _p("MSS-gate-g-fp32", "g", "fp32", 2, 2, 256, [0, 256], note="TilingKey USE_G + gate fp32"),
        _p("MSS-gate-g-bf16", "g", "bf16", 2, 2, 256, [0, 256], note="TilingKey USE_G + gate bf16"),
        _p("MSS-gate-gk-fp32", "gk", "fp32", 2, 2, 256, [0, 256], note="TilingKey USE_GK + gate fp32"),
        _p("MSS-gate-gk-bf16", "gk", "bf16", 2, 2, 256, [0, 256], note="TilingKey USE_GK + gate bf16"),
        # 变长 + 多段（覆盖段枚举与 slot 复用）：同样最小规模
        _p("MSS-varlen-3seg", "gk", "fp32", 2, 2, 384, [0, 64, 192, 384], note="变长 3 段（段枚举/slot 复用）"),
    ]


def _spec(profile: dict, case_id: int, seed: int) -> dict:
    """把一条用例描述展开成完整的 case_spec（写进 JSON 与 attrs）。"""
    return {
        "name": profile["name"],
        "group": profile.get("group", "精度"),
        "note": profile.get("note", ""),
        "dtype": profile.get("dtype", "bf16"),
        "B": 1,
        "HK": int(profile["HK"]),
        "HV": int(profile["HV"]),
        "T": int(profile["T"]),
        "K": K_DIM,
        "V": V_DIM,
        "chunk_size": BT,
        "gate": profile.get("gate", "g"),
        "gate_dtype": profile.get("gate_dtype", "fp32"),
        "cu_seqlens": [int(x) for x in profile["cu"]],
        "op": OP_NAME,
        "case_id": case_id,
        "seed": seed,
        "route": ROUTE,
        "soc": SOC,
    }


def _input(name, dtype, range_values, input_type="attr", shape=None):
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": range_values,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def _case_payload(case_id: int, profile: dict, standard: dict, seed: int) -> dict:
    spec = _spec(profile, case_id, seed)
    marker_dtype = "bf16"
    inputs = [
        _input("low_precision_marker", marker_dtype, [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input("case_spec", "non_param", json.dumps(spec, ensure_ascii=False, separators=(",", ":"))),
        _input("dtype", "string", spec["dtype"]),
        _input("B", "int", spec["B"]),
        _input("HK", "int", spec["HK"]),
        _input("HV", "int", spec["HV"]),
        _input("T", "int", spec["T"]),
        _input("K", "int", spec["K"]),
        _input("V", "int", spec["V"]),
        _input("chunk_size", "int", spec["chunk_size"]),
        _input("gate", "string", spec["gate"]),
        _input("gate_dtype", "string", spec["gate_dtype"]),
        _input("soc", "string", spec["soc"]),
        _input("route", "string", spec["route"]),
    ]
    return {
        "id": case_id,
        "default_seed": spec["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{spec['name']}",
        "aclnn_name": ACLNN_NAME,
        "triton_name": None,
        "kernel_name": None,
        "version": "v2.1",
        "expected_error_msg": None,
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "aclnn_api_type": "aclnn_function",
        "triton_api_type": "triton_function",
        "fusion_api_type": "fusion_function",
        "fusion_mode": None,
        "dist_api_type": "dist_function",
        "kernel_api_type": "kernel_function",
        "backward": False,
        "standard": standard,
        "outputs": None,
        "inputs": inputs,
        "acl_json": "",
        "method_inputs": None,
        "tensor_input": None,
        "compute_times": None,
        "save_name": OP_NAME,
        "uuid": None,
        "downloaded": False,
        "is_boundary": False,
        "xrun_cs_name": None,
        "xrun_data": None,
        "strategy": None,
    }


def _write_json(path: Path, payloads: list) -> None:
    Path(path).write_text(json.dumps(payloads, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expand_seeds(profiles: list[dict], n: int = SEEDS_PER_CASE) -> list[tuple]:
    """把每条逻辑用例展开成 `n` 条（不同固定种子），种子按顺序唯一分配。"""
    out: list[tuple] = []
    for profile in profiles:
        for _ in range(n):
            out.append((profile, SEED0 + len(out)))
    return out


def _emit(acc, perf, mss, out_acc: Path, out_perf: Path, out_mss: Path,
          standard: str = DEFAULT_STANDARD) -> None:
    """三份 JSON 来源不同、不能互相替代（见 tests/atk/README.md「正式验收用例包」）。"""
    acc_standard, mss_standard = STANDARD_CHOICES[standard]
    expanded = expand_seeds(acc)
    _write_json(out_acc, [_case_payload(i, p, acc_standard, s) for i, (p, s) in enumerate(expanded)])
    _write_json(out_perf, [_case_payload(i, p, acc_standard, SEED0 + i) for i, p in enumerate(perf)])
    _write_json(out_mss, [_case_payload(i, p, mss_standard, SEED0 + i) for i, p in enumerate(mss)])


def build_cases() -> list:
    if CaseConfig is None:
        raise RuntimeError("ATK and PyTorch are required to instantiate CaseConfig objects.")
    expanded = expand_seeds(accuracy_profiles())
    return [CaseConfig(**_case_payload(i, p, STANDARD, s)) for i, (p, s) in enumerate(expanded)]


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register(f"generator_{OP_NAME}")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            self.cases = build_cases()
            self.length = len(self.cases)
            self.index = 0

        def generate(self):
            case = self.cases[self.index]
            self.index += 1
            return case


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize the frozen {OP_NAME} ATK matrix.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(f"atk_{OP_NAME}.json"))
    parser.add_argument("--perf", type=Path, default=Path(__file__).with_name(f"atk_{OP_NAME}_perf.json"))
    parser.add_argument("--mss", type=Path, default=Path(__file__).with_name(f"atk_{OP_NAME}_mss.json"))
    parser.add_argument(
        "--standard",
        choices=sorted(STANDARD_CHOICES),
        default=DEFAULT_STANDARD,
        help="写入三份 JSON 的 standard.acc；GPU 双标杆验收用 cv_fused_double_benchmark",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    acc = accuracy_profiles()
    perf = perf_profiles()
    mss = mss_profiles()
    _emit(acc, perf, mss, args.output, args.perf, args.mss, args.standard)
    if args.summary:
        keys = sorted({(p["gate"], p["gate_dtype"]) for p in mss})
        print(f"logical={len(acc)} accuracy={len(acc) * SEEDS_PER_CASE} perf={len(perf)} "
              f"mss={len(mss)} tiling_keys={len(keys)} -> {keys} standard={args.standard}")


if __name__ == "__main__":
    main()
