"""chunk_gated_delta_rule_fwd_prepare 的 ATK 泛化用例生成器。

50 个中型 shape × 6 条合法 flag = 300（bf16）。
中型按 tiling 的 totalChunks：dense 为 B*HV*ceil(T/64)，
varlen 为 HV*sum(ceil(s/64))，必须 **>256** 且 ≤384。
G≠3 时 pack=4：256 tiles = 64 packs = 32 AIC × 2 pack；>256 保证每核至少 2 pack。
约束：chunk_size=64，K=128，V∈{128,256}，HV/HK∈{1,2,3,4}，
use_exp2=True，use_gate=False。含 packed varlen（B=1 + seqlens）。
use_qk_l2norm True/False 都覆盖；False 时 executor 在调用前对 q/k 做 L2norm。

合法 flag（l2 / gate=F / sigmoid / neg）：

    T F T T   l2_sig1_neg1     核内 L2norm，beta_eff=2*sigmoid
    T F T F   l2_sig1_neg0     核内 L2norm，beta_eff=sigmoid
    T F F F   l2_sig0_neg0     核内 L2norm，不做 sigmoid
    F F T T   nol2_sig1_neg1   调用前归一化 qk，2*sigmoid
    F F T F   nol2_sig1_neg0   调用前归一化 qk，sigmoid
    F F F F   nol2_sig0_neg0   调用前归一化 qk，不做 sigmoid
"""

from __future__ import annotations

import json
from copy import deepcopy
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

OP_NAME = "chunk_gated_delta_rule_fwd_prepare"

# (tag, l2norm, sigmoid, neg). gate=False is fixed by the kernel.
SUPPORTED_FLAGS = (
    ("l2_sig1_neg1", True, True, True),
    ("l2_sig1_neg0", True, True, False),
    ("l2_sig0_neg0", True, False, False),
    ("nol2_sig1_neg1", False, True, True),
    ("nol2_sig1_neg0", False, True, False),
    ("nol2_sig0_neg0", False, False, False),
)

N_SHAPES = 50
N_FLAGS = len(SUPPORTED_FLAGS)
N_PROFILES = N_SHAPES * N_FLAGS

_BT = 64
_MEDIUM_CHUNKS_MIN = 256  # exclusive: tiles must be > 256
_MEDIUM_CHUNKS_MAX = 384
_MEDIUM_B = (1, 2, 4)
_MEDIUM_HK_MIN, _MEDIUM_HK_MAX = 2, 16
_MEDIUM_HV_MAX = 32


def _ok(hk: int, hv: int) -> bool:
    return hk > 0 and hv % hk == 0 and 1 <= (hv // hk) <= 4


def _n_time_chunks(length: int) -> int:
    return (int(length) + _BT - 1) // _BT


def _chunk_tiles(B: int, HV: int, T: int, seqlens: tuple[int, ...] | None = None) -> int:
    """Match host tiling totalChunks: B*HV*ceil(T/64), varlen HV*sum(ceil(s/64))."""
    if seqlens is not None:
        return int(HV) * sum(_n_time_chunks(s) for s in seqlens)
    return int(B) * int(HV) * _n_time_chunks(T)


def _assert_medium(name: str, B: int, HK: int, HV: int, T: int,
                   seqlens: tuple[int, ...] | None = None) -> None:
    if B not in _MEDIUM_B:
        raise ValueError(f"{name}: B={B} not in {_MEDIUM_B}")
    if not (_MEDIUM_HK_MIN <= HK <= _MEDIUM_HK_MAX):
        raise ValueError(f"{name}: HK={HK} not in [{_MEDIUM_HK_MIN},{_MEDIUM_HK_MAX}]")
    if HV > _MEDIUM_HV_MAX:
        raise ValueError(f"{name}: HV={HV} > {_MEDIUM_HV_MAX}")
    tiles = _chunk_tiles(B, HV, T, seqlens)
    if not (_MEDIUM_CHUNKS_MIN < tiles <= _MEDIUM_CHUNKS_MAX):
        raise ValueError(
            f"{name}: chunks={tiles} not in "
            f"({_MEDIUM_CHUNKS_MIN},{_MEDIUM_CHUNKS_MAX}] "
            f"(B={B} HV={HV} T={T} seqlens={seqlens})"
        )


def _shape_table() -> list[dict]:
    """50 条中型 shape：totalChunks ∈ (256, 384]，覆盖 G/V/对齐/尾块/不满 pack/B>1/奇数 HK/varlen。"""
    rows: list[dict] = []

    def add(name: str, B: int, HK: int, HV: int, T: int, V: int,
            seqlens: tuple[int, ...] | None = None) -> None:
        if not _ok(HK, HV):
            raise ValueError(f"{name}: invalid GVA HK={HK} HV={HV}")
        if seqlens is not None:
            seqlens = tuple(int(x) for x in seqlens)
            if B != 1 or any(n <= 0 for n in seqlens) or sum(seqlens) != T:
                raise ValueError(
                    f"varlen {name}: need B=1 and sum(seqlens)={T}, "
                    f"got B={B} seqlens={seqlens}"
                )
        _assert_medium(name, B, HK, HV, T, seqlens)
        row = dict(name=name, B=B, HK=HK, HV=HV, T=T, K=128, V=V, chunk_size=64)
        if seqlens is not None:
            row["seqlens"] = list(seqlens)
        rows.append(row)

    # 1. G=1..4 × V=128/256 对齐。T 随 HV 缩小，使 tiles>256。
    #    perf JSON 取前 6 条 T>=256 定长：G=1-4 V128 + G=1-2 V256。
    #    r1 HV=4 T=4160 → 260; r2 HV=8 T=2112 → 264;
    #    r3 HV=12 T=1408 → 264; r4 HV=16 T=1088 → 272.
    for V in (128, 256):
        add(f"r1_T4160_V{V}", 1, 4, 4, 4160, V)
        add(f"r2_T2112_V{V}", 1, 4, 8, 2112, V)
        add(f"r3_T1408_V{V}", 1, 4, 12, 1408, V)
        add(f"r4_T1088_V{V}", 1, 4, 16, 1088, V)

    # 2. 同上，尾块 ceil(T/64) = 整 chunk + 1
    for V in (128, 256):
        add(f"r1_T4192_V{V}", 1, 4, 4, 4192, V)   # 66*4=264
        add(f"r2_T2144_V{V}", 1, 4, 8, 2144, V)   # 34*8=272
        add(f"r3_T1440_V{V}", 1, 4, 12, 1440, V)  # 23*12=276
        add(f"r4_T1120_V{V}", 1, 4, 16, 1120, V)  # 18*16=288

    # 3. 其它中型 T / G
    add("g2_T1088_V128", 1, 8, 16, 1088, 128)     # 17*16=272
    add("g2_T1088_V256", 1, 8, 16, 1088, 256)
    add("g1_T2112_V128", 1, 8, 8, 2112, 128)      # 33*8=264
    add("g2_T3072_V128", 1, 4, 8, 3072, 128)      # 48*8=384
    add("g4_T1536_V256", 1, 4, 16, 1536, 256)     # 24*16=384
    add("g3_T704_V128", 1, 8, 24, 704, 128)       # 11*24=264

    # 4. B>1
    add("B2_g1_T1088", 2, 8, 8, 1088, 128)        # 2*8*17=272
    add("B2_g2_T1088_V256", 2, 4, 8, 1088, 256)
    add("B4_g1_T1088", 4, 4, 4, 1088, 128)        # 4*4*17=272
    add("B2_g2_T992", 2, 6, 12, 992, 128)         # 2*12*16=384

    # 5. 不满 pack（HV%4≠0）与奇数 HK
    add("partial_HV5_T3328", 1, 5, 5, 3328, 128)          # 52*5=260
    add("partial_HV6_T2816", 1, 6, 6, 2816, 128)          # 44*6=264
    add("partial_HV7_T2432", 1, 7, 7, 2432, 128)          # 38*7=266
    add("partial_HV9_T1920", 1, 9, 9, 1920, 128)          # 30*9=270
    add("odd_HK5_G2_T1728_V256", 1, 5, 10, 1728, 256)     # 27*10=270
    add("odd_HK7_G2_T1216_V256", 1, 7, 14, 1216, 256)     # 19*14=266
    add("odd_HK11_T1536", 1, 11, 11, 1536, 128)           # 24*11=264
    add("odd_HK5_G3_T1152", 1, 5, 15, 1152, 128)          # 18*15=270

    # 6. 更宽的中型头数
    add("HK12_G1_T1408", 1, 12, 12, 1408, 128)            # 22*12=264
    add("HK12_G2_T704_V256", 1, 12, 24, 704, 256)         # 11*24=264
    add("HK16_G1_T1088", 1, 16, 16, 1088, 128)            # 17*16=272
    add("HK16_G2_T576", 1, 16, 32, 576, 128)              # 9*32=288

    # 7. packed varlen：tiles = HV * sum(ceil(s/64))
    add("varlen_g1_align", 1, 4, 4, 4352, 128, (1088, 1088, 1088, 1088))     # 4*68=272
    add("varlen_g1_tail", 1, 4, 4, 4800, 128, (1600, 1536, 1664))            # 4*75=300
    add("varlen_g2_v256", 1, 4, 8, 2176, 256, (1088, 1088))                  # 8*34=272
    add("varlen_g3_mix", 1, 4, 12, 1408, 128, (448, 512, 448))               # 12*22=264
    add("varlen_g4_v256", 1, 2, 8, 2176, 256, (1088, 1088))                  # 8*34=272
    add("varlen_near_chunk", 1, 8, 8, 2112, 128, (1025, 1087))               # 8*34=272
    add("varlen_g2_tail", 1, 8, 16, 1088, 128, (544, 544))                   # 16*18=288
    add("varlen_four_seq", 1, 4, 4, 4160, 128, (1024, 1088, 960, 1088))      # 4*65=260

    # 8. 补齐到 50：G=3 尾块 / B>1 / G=4
    add("g3_T1504_tail", 1, 4, 12, 1504, 128)             # 24*12=288
    add("B2_HK8_G1_T1536", 2, 8, 8, 1536, 128)            # 2*8*24=384
    add("T2144_g1_tail", 1, 8, 8, 2144, 128)              # 34*8=272
    add("g4_T1152_V128", 1, 4, 16, 1152, 128)             # 18*16=288

    seen = set()
    uniq = []
    for row in rows:
        key = (
            row["B"], row["HK"], row["HV"], row["T"], row["V"],
            tuple(row.get("seqlens") or ()),
        )
        if key in seen:
            raise RuntimeError(f"duplicate shape {row['name']}: {key}")
        seen.add(key)
        uniq.append(row)
    if len(uniq) != N_SHAPES:
        raise RuntimeError(f"need {N_SHAPES} shapes, got {len(uniq)}")
    return uniq


def _make_profiles() -> list[dict]:
    profiles = []
    case_id = 0
    for shape in _shape_table():
        for tag, l2, sigmoid, neg in SUPPORTED_FLAGS:
            spec = dict(shape)
            spec.update(
                dtype="bf16",
                op=OP_NAME,
                case_id=case_id,
                seed=20260817 + case_id,
                route="ascendc",
                soc="ascend950",
                use_qk_l2norm_in_kernel=l2,
                use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=sigmoid,
                allow_neg_eigval=neg,
                use_exp2=True,
                flag_tag=tag,
            )
            spec["name"] = f"{shape['name']}_{tag}"
            profiles.append(spec)
            case_id += 1
    return profiles


PROFILES = _make_profiles()
if len(PROFILES) != N_PROFILES:
    raise RuntimeError(
        f"need {N_PROFILES} profiles ({N_SHAPES} shapes x {N_FLAGS} flags), "
        f"got {len(PROFILES)}"
    )


def _dtype(name: str) -> str:
    return "bf16"


def _spec(index: int) -> dict:
    return deepcopy(PROFILES[index % len(PROFILES)])


def _case_json(spec: dict, case_id: int) -> dict:
    inputs = [
        {
            "name": "low_precision_marker",
            "type": "tensor",
            "required": True,
            "dtype": "bf16",
            "shape": [1],
            "range_values": [0, 0],
            "backward": True,
            "align_32B": None,
            "outlier_values": None,
        },
        {
            "name": "fp32_marker",
            "type": "tensor",
            "required": True,
            "dtype": "fp32",
            "shape": [1],
            "range_values": [0, 0],
            "backward": True,
            "align_32B": None,
            "outlier_values": None,
        },
        {
            "name": "case_spec",
            "type": "attr",
            "required": True,
            "dtype": "non_param",
            "shape": None,
            "range_values": json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
            "backward": False,
            "align_32B": None,
            "outlier_values": None,
        },
    ]
    for key in (
        "dtype", "B", "HK", "HV", "T", "K", "V", "chunk_size", "case_id", "seed", "soc", "route",
    ):
        val = spec[key]
        dtype = "string" if isinstance(val, str) else "int"
        inputs.append(
            {
                "name": key,
                "type": "attr",
                "required": True,
                "dtype": dtype,
                "shape": None,
                "range_values": val,
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            }
        )
    if spec.get("seqlens"):
        inputs.append(
            {
                "name": "seqlens",
                "type": "attr",
                "required": True,
                "dtype": "string",
                "shape": None,
                "range_values": json.dumps(spec["seqlens"], separators=(",", ":")),
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            }
        )
    return {
        "id": case_id,
        "default_seed": spec["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{spec.get('name', 'case')}",
        "aclnn_name": None,
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
        "standard": {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1},
        "outputs": None,
        "inputs": inputs,
    }


def dump_json_files(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or Path(__file__).resolve().parent
    all_cases = [_case_json(spec, i) for i, spec in enumerate(PROFILES)]
    (out_dir / f"atk_{OP_NAME}.json").write_text(
        json.dumps(all_cases, indent=1, ensure_ascii=False) + "\n"
    )

    def _first(substr: str, tag: str = "l2_sig1_neg1") -> int:
        for i, spec in enumerate(PROFILES):
            if substr in spec.get("name", "") and spec.get("flag_tag") == tag:
                return i
        raise RuntimeError(f"no profile matching {substr!r} tag={tag!r}")

    # TilingKey 固定为 0。MSS 覆盖 V128/256、尾块、不满 pack、varlen、G=2/3/4、B>1、6 组 flag。
    mss_idx = [
        _first("r1_T4160_V128", "l2_sig1_neg1"),
        _first("r1_T4192_V128", "l2_sig1_neg1"),
        _first("partial_HV6_T2816", "l2_sig1_neg1"),
        _first("r2_T2112_V256", "l2_sig1_neg1"),
        _first("B4_g1_T1088", "l2_sig1_neg1"),
        _first("varlen_g1_tail", "l2_sig1_neg1"),
        _first("varlen_g2_v256", "l2_sig1_neg1"),
        _first("varlen_g3_mix", "l2_sig1_neg1"),
        _first("varlen_near_chunk", "l2_sig1_neg1"),
        _first("r1_T4160_V128", "l2_sig1_neg0"),
        _first("r1_T4160_V128", "l2_sig0_neg0"),
        _first("r1_T4160_V128", "nol2_sig1_neg1"),
        _first("r4_T1088_V256", "nol2_sig1_neg0"),
        _first("partial_HV5_T3328", "nol2_sig1_neg1"),
        _first("varlen_g2_v256", "nol2_sig0_neg0"),
    ]
    if len(mss_idx) != len(set(mss_idx)):
        raise RuntimeError(f"duplicate mss indices: {mss_idx}")
    mss = [all_cases[i] for i in mss_idx]
    (out_dir / f"atk_{OP_NAME}_mss.json").write_text(
        json.dumps(mss, indent=1, ensure_ascii=False) + "\n"
    )
    perf_idx = [
        i for i, s in enumerate(PROFILES)
        if s["T"] >= 256 and not s.get("seqlens") and s.get("flag_tag") == "l2_sig1_neg1"
    ][:6]
    if len(perf_idx) != 6:
        raise RuntimeError(f"need 6 perf cases, got {perf_idx}")
    perf = [all_cases[i] for i in perf_idx]
    (out_dir / f"atk_{OP_NAME}_perf.json").write_text(
        json.dumps(perf, indent=1, ensure_ascii=False) + "\n"
    )
    g2 = [
        all_cases[i] for i, s in enumerate(PROFILES)
        if s["HK"] > 0 and s["HV"] // s["HK"] == 2
    ]
    (out_dir / f"atk_{OP_NAME}_g2.json").write_text(
        json.dumps(g2, indent=1, ensure_ascii=False) + "\n"
    )
    return {
        "accuracy": len(all_cases),
        "mss": [(i, PROFILES[i]["name"]) for i in mss_idx],
        "perf": [(i, PROFILES[i]["name"]) for i in perf_idx],
        "g2": len(g2),
    }


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register(f"generator_{OP_NAME}")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = _dtype(spec.get("dtype", "bf16"))
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


if __name__ == "__main__":
    summary = dump_json_files()
    tags = {}
    tiles = []
    for spec in PROFILES:
        tags[spec["flag_tag"]] = tags.get(spec["flag_tag"], 0) + 1
        if spec["flag_tag"] != "l2_sig1_neg1":
            continue
        seq = tuple(spec["seqlens"]) if spec.get("seqlens") else None
        tiles.append(_chunk_tiles(spec["B"], spec["HV"], spec["T"], seq))
    print(f"wrote {len(PROFILES)} profiles {tags}")
    print(f"shape chunks min={min(tiles)} max={max(tiles)} n={len(tiles)}")
    print("mss:", summary["mss"])
    print("perf:", summary["perf"])
    print("g2:", summary["g2"])
