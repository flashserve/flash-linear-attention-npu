"""Generate the frozen ATK matrices for ChunkDeltaHBwdPreprocess.

用例设计来源是同一目录下的 ``cases.json``（正向 12 条 + 反向拦截 13 条）。正向用例在这里
展开成 ATK 用例矩阵 ``atk_<op>.json`` / ``_perf.json`` / ``_mss.json``；反向拦截用例由
``scripts/run_negative.sh`` 直接驱动 aclnn 校验返回码（ATK 的 runner 没有拦截 scope）。

``dhm`` 是 FP32，但本算子的链上状态按设计用模型 dtype（bf16/fp16）传递，所以每条用例在
``standard.acc`` 里用 ATK 原生的 ``output_dtype_overrides`` 声明"按模型 dtype 判精度"，
不自定义判据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generator_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:  # pragma: no cover - ATK 未安装时只走 main()
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None


OP_NAME = "chunk_delta_h_bwd_preprocess"
HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.json"
SEED_BASE = 20260927
CHUNK_SIZE = 64

# 输入由 executor 按 case_spec 里的 seed 确定性地构造（ATK 只传 marker + 用例元数据）。
# 值域与 tests/operators 下的 harness 一致：q/k/w/do/dv 是 randn*0.05，gate 是沿 token 的 -cumsum(rand*0.05)，
# 后者保证 2^{g_last-g} ≤ 1（gate 非单调会让长链溢出，实测 fp16 会出 NaN）。
MARKER_RANGE = [0, 0]
PERF_KEYS = ("pos_01_none_gate_dense", "pos_05_gk_varlen_first_segment",
             "pos_13_long_nt_chain_accumulation")
# 内存/确定性用例要覆盖本版全部 4 个 tilingKey：无门控 / g(model) / g(fp32) / gk。
MSS_KEYS = ("pos_01_none_gate_dense", "pos_02_g_bf16_dense", "pos_03_g_fp32_dense",
            "pos_04_gk_bf16_dense", "pos_05_gk_varlen_first_segment")


def _standard(model_dtype: str) -> dict:
    """ATK 原生标准 + 输出有效精度声明（dhm 的链按 model_dtype 传递）。"""

    return {
        "acc": {"mixed_tolerance_bm": {"output_dtype_overrides": {"0": model_dtype}}},
        "perf": "not_key",
        "mem": 1.1,
    }


def _load_cases() -> list[dict]:
    payload = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    return payload["cases"]


def _input(name: str, dtype: str, value, *, input_type: str = "attr", shape=None) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


def _metadata(spec: dict, index: int, tags: str) -> dict:
    shape = spec["shape"]
    return {
        "case_key": str(spec["id"]),
        "tags": tags,
        "route": "ascendc",
        "soc": "ascend950",
        "dtype": str(spec["dtype"]),
        "gate": str(spec.get("gate", "none")),
        "g_dtype": "fp32" if spec.get("g_dtype") == "fp32" else "model",
        "chunk_size": int(spec.get("chunk_size", CHUNK_SIZE)),
        "scale": float(spec["scale"]),
        "B": int(shape["B"]),
        "Hk": int(shape["Hk"]),
        "Hv": int(shape["Hv"]),
        "T": int(shape["T"]),
        "K": int(shape["K"]),
        "V": int(shape["V"]),
        "varlen": bool(spec.get("cu_seqlens")),
        "cu_seqlens": [int(x) for x in spec.get("cu_seqlens", [])],
        "note": str(spec.get("note", "")),
        "case_id": index,
        "seed": SEED_BASE + index,
    }


def _case_payload(case_id: int, spec: dict, tags: str) -> dict:
    metadata = _metadata(spec, case_id, tags)
    inputs = [
        _input("low_precision_marker", metadata["dtype"], MARKER_RANGE,
               input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", MARKER_RANGE,
               input_type="tensor", shape=[1]),
    ]
    inputs.append(
        _input("case_spec", "non_param",
               json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")))
    )
    for name, dtype in (
        ("case_key", "string"),
        ("tags", "string"),
        ("route", "string"),
        ("soc", "string"),
        ("dtype", "string"),
        ("gate", "string"),
        ("g_dtype", "string"),
        ("chunk_size", "int"),
        ("scale", "float"),
        ("B", "int"),
        ("Hk", "int"),
        ("Hv", "int"),
        ("T", "int"),
        ("K", "int"),
        ("V", "int"),
        ("varlen", "bool"),
        ("seed", "int"),
    ):
        inputs.append(_input(name, dtype, metadata[name]))
    return {
        "id": case_id,
        "default_seed": metadata["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{metadata['case_key']}",
        "aclnn_name": None,
        "version": "v2.1",
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "expected_error_msg": None,
        "backward": False,
        "standard": _standard(metadata["dtype"]),
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def _select(keys) -> list[dict]:
    wanted = set(keys)
    return [spec for spec in _load_cases() if spec["id"] in wanted]


def build_accuracy_specs() -> list[dict]:
    return _load_cases()


def build_perf_specs() -> list[dict]:
    return _select(PERF_KEYS)


def build_mss_specs() -> list[dict]:
    return _select(MSS_KEYS)


def _payloads(specs: list[dict], tags: str) -> list[dict]:
    return [_case_payload(index, spec, tags) for index, spec in enumerate(specs)]


if GENERATOR_REGISTRY is not None:  # pragma: no cover - 需要安装 ATK

    @GENERATOR_REGISTRY.register(f"generator_{OP_NAME}")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            if CaseConfig is None:
                raise RuntimeError("ATK is required to build CaseConfig objects")
            self.cases = [CaseConfig(**payload)
                          for payload in _payloads(build_accuracy_specs(), "accuracy")]
            self.length = len(self.cases)
            self.index = 0

        def generate(self) -> CaseConfig:
            case = self.cases[self.index]
            self.index += 1
            return case


def _write(path: Path, payloads: list[dict]) -> None:
    path.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy = build_accuracy_specs()
    perf = build_perf_specs()
    mss = build_mss_specs()
    _write(args.output_dir / f"atk_{OP_NAME}.json", _payloads(accuracy, "accuracy"))
    _write(args.output_dir / f"atk_{OP_NAME}_perf.json", _payloads(perf, "performance"))
    _write(args.output_dir / f"atk_{OP_NAME}_mss.json", _payloads(mss, "determinism,mss"))
    if args.summary:
        print(f"accuracy={len(accuracy)} perf={len(perf)} determinism={len(mss)} mss={len(mss)}")


if __name__ == "__main__":
    main()
