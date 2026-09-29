"""Generate the frozen ATK matrices for ChunkDeltaHBwdPreprocess.

用例设计来源是本文件内联的用例表（正向 12 条 + 反向拦截 13 条）。正向用例在这里
展开成 ATK 用例矩阵 ``atk_<op>.json`` / ``_perf.json`` / ``_mss.json``；反向拦截用例由
反向用例随精度矩阵一并输出并带 expected_return_code，由 ATK 的 --task run 校验 aclnn 返回码。

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
CASES_JSON = '{"cases":[{"chunk_size":64,"dtype":"bf16","gate":"none","id":"pos_01_none_gate_dense","note":"无门控（qg/kg 未门控）dense 基线","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":512,"V":128}},{"chunk_size":64,"dtype":"bf16","g_dtype":"model","gate":"g","id":"pos_02_g_bf16_dense","note":"GDN 标量 gate，g 与 q 同 dtype（tilingKey=2）","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":512,"V":128}},{"chunk_size":64,"dtype":"bf16","g_dtype":"fp32","gate":"g","id":"pos_03_g_fp32_dense","note":"GDN 标量 gate，g 为 FP32（tilingKey=3）；g 为 FP32 时不能走 Cast<float,float>（见算子 README 踩坑记录）","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":512,"V":128}},{"chunk_size":64,"dtype":"bf16","gate":"gk","id":"pos_04_gk_bf16_dense","note":"KDA/GDN2 逐 K gate（tilingKey=4）","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":512,"V":128}},{"chunk_size":64,"cu_seqlens":[0,300,512],"dtype":"bf16","gate":"gk","id":"pos_05_gk_varlen_first_segment","note":"varlen：packed 输入，本 launch 只处理 cu_seqlens[0:2] = [0,300)","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":512,"V":128}},{"chunk_size":64,"dtype":"bf16","gate":"gk","id":"pos_06_tail_chunk","note":"尾块：T=200 时最后一个 chunk 只有 8 行有效；[M,BT) 必须写零","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":200,"V":128}},{"chunk_size":64,"dtype":"bf16","gate":"gk","id":"pos_10_gva_hv_gt_hk","note":"GVA：Hv=8 为 Hk=4 的两倍，hk = hv // 2，不要求物理 repeat q/k","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":8,"K":128,"T":256,"V":128}},{"chunk_size":64,"dtype":"bf16","g_dtype":"fp32","gate":"g","id":"pos_11_single_chunk","note":"NT=1：只验证单 chunk 边界（链方向不被覆盖）","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":64,"V":128}},{"chunk_size":64,"dtype":"bf16","gate":"gk","id":"pos_12_two_chunk_chain_direction","note":"NT=2：验证 P_new = P_c @ P_old 与 dH ← decayK ⊙ dH + inc 的 chunk 顺序","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":128,"V":128}},{"chunk_size":64,"dtype":"bf16","gate":"gk","id":"pos_13_long_nt_chain_accumulation","note":"NT=32：链式 P_r 的误差累积用例，单独看相对误差","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":2048,"V":128}},{"chunk_size":64,"dtype":"fp16","g_dtype":"model","gate":"g","id":"pos_14_fp16_inputs","note":"FP16 输入路径：dhm 与链仍为 FP32","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":256,"V":128}},{"chunk_size":64,"dtype":"bf16","gate":"gk","id":"pos_15_head_contiguous_partition","note":"Hk=Hv=96 > 核数：每个工作组连续处理多个 head（groupHeads>1），覆盖跨 task 复用 workspace/UB 的同步路径","platform":["ascend910b","ascend910_93","ascend950"],"scale":0.08838834764831845,"shape":{"B":1,"Hk":96,"Hv":96,"K":128,"T":256,"V":128}}],"field_doc":{"cu_seqlens":"varlen 时给出；本算子只消费前两个元素 [bos, eos)","dtype":"q/k/w/do/dv 的 dtype：bf16 或 fp16","expected":"正向用例为精度判据；反向用例给 expected_return_code 与触发原因","g_dtype":"g 的 dtype：model（与 q 同）| fp32；gate=none/gk 时忽略","gate":"none | g（标量 gate，g dtype 见 g_dtype）| gk（逐 K gate）","shape":"B/Hk/Hv/T/K/V；B 必须为 1（一次 launch 一个 segment）"},"impl_type":"ascendc","negative_cases":[{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_01_g_and_gk_both","note":"g 与 gk 同时非空：必须拦截（GVA 路径下二者互斥）","trigger":"host tiling：hasG && hasGk"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_02_k_too_large","note":"K=512：本版只支持 K = V = 128","trigger":"host tiling：K != 128"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_03_hv_not_multiple_of_hk","note":"Hk=3 Hv=4：GVA 映射不成立","trigger":"host tiling：Hv % Hk != 0"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_04_dense_b_greater_than_one","note":"未传 cu_seqlens 且 B=2：一次 launch 只处理一个 segment","trigger":"host tiling：dense 路径要求 B == 1"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_05_varlen_b_greater_than_one","note":"传 cu_seqlens 且 B=2：packed 输入必须 B==1","trigger":"host tiling：varlen 路径要求 B == 1"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_06_chunk_size_not_64","note":"chunk_size=128 与本版 tiling 不匹配","trigger":"host tiling：chunk_size != 64"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_07_g_shape_mismatch","note":"g 为 [B,Hk,T] 而非 [B,Hv,T]","trigger":"host tiling：g shape != [B,Hv,T]"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_08_gk_dtype_fp32","note":"gk 为 FP32：逐 K gate 必须与 q/k 同 dtype","trigger":"host tiling：gk dtype != q dtype"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_09_cu_seqlens_too_short","note":"cu_seqlens 只有 1 个元素，无法给出 [bos, eos)","trigger":"host tiling：cu_seqlens shape dim0 - 1 < 1"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_10_empty_tensor","note":"T=0：空 tensor 必须给出明确错误","trigger":"host tiling：B/T/Hk/Hv/K/V 存在 0"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_11_k64_not_128","note":"本版只支持 K = V = 128：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 逐行的向量/寄存器读写也依赖该行宽；其它取值实测会设备报错或算错，必须拦截","trigger":"host tiling：K != 128（K=64，曾是历史用例，现按不支持拦截）"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_12_v96_not_128","note":"本版只支持 K = V = 128：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 逐行的向量/寄存器读写也依赖该行宽；其它取值实测会设备报错或算错，必须拦截","trigger":"host tiling：V != 128（V=96，曾是历史用例，现按不支持拦截）"},{"expected_return_code":"ACLNN_ERR_PARAM_INVALID","id":"neg_13_k256_not_128","note":"本版只支持 K = V = 128：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 逐行的向量/寄存器读写也依赖该行宽；其它取值实测会设备报错或算错，必须拦截","trigger":"host tiling：K != 128（K=256，曾是历史用例，现按不支持拦截）"}],"op_type":"ChunkDeltaHBwdPreprocess","operator":"chunk_delta_h_bwd_preprocess","precision":{"dhm":"FP32 输出，按绝对 + 相对双阈值判定；长 NT 用例单独给出相对误差","note":"P_r 是链式量，误差随 chunk 数放大；禁止通过收窄 range 或跳过用例制造通过结论"},"primary_entry":"fla_npu.ops.ascendc.chunk_delta_h_bwd_preprocess","reference":"tests/atk/chunk_delta_h_bwd_preprocess/reference.py"}'
SEED_BASE = 20260927
CHUNK_SIZE = 64

# ACLNN 返回码名 → 数字码：反向用例的 expected_error_msg 与 executor 的判定共用同一份口径。
_ACLNN_CODES = {
    "ACLNN_ERR_PARAM_INVALID": 161001,
    "ACLNN_ERR_PARAM_NULLPTR": 161002,
}

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
    return json.loads(CASES_JSON)['cases']


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
        # 反向用例 neg_07/neg_08 需要这两个 hint 才能构造出"非法但类型正确"的输入：
        # g_head=Hk → g 用 [B,Hk,T]；gk_dtype=fp32 → gk 用 FP32。
        "g_head": str(spec.get("g_head", "Hv")),
        "gk_dtype": str(spec.get("gk_dtype", "model")),
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
        # 反向（拦截）用例必须把期望返回码带进 case_spec：executor 据此走"预期失败"分支
        # （跳过 CPU 标杆、只在 NPU 侧校验 aclnn 返回码）。
        "expected_return_code": str(spec.get("expected_return_code") or ""),
        "expected_error_msg": (
            "{} ({}): {}".format(spec["expected_return_code"],
                                 _ACLNN_CODES.get(spec["expected_return_code"], "?"),
                                 spec.get("note", ""))
            if spec.get("expected_return_code") else ""
        ),
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
        ("g_head", "string"),
        ("gk_dtype", "string"),
        ("expected_return_code", "string"),
        ("expected_error_msg", "string"),
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
    payload = {
        "id": case_id,
        "default_seed": metadata["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{metadata['case_key']}",
        "aclnn_name": None,
        "version": "v2.1",
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "backward": False,
        "standard": _standard(metadata["dtype"]),
        # 反向（拦截）用例：标准与正向一致，靠 ATK 的 expected_error_msg 语义判定——
        # executor 命中预期拦截时抛出同一文本（见 executor 的 _ACLNN_CODES）。
        "expected_error_msg": (
            "{} ({}): {}".format(spec["expected_return_code"],
                                 _ACLNN_CODES.get(spec["expected_return_code"], "?"),
                                 spec.get("note", ""))
            if spec.get("expected_return_code") else None
        ),
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }
    if spec.get('expected_return_code'):
        payload["expected_return_code"] = spec['expected_return_code']
    return payload


NEG_SPECS_JSON = '[{"case_key":"neg_01_g_and_gk_both","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"both","gk_dtype":"model","id":"neg_01_g_and_gk_both","note":"g 与 gk 同时非空：必须拦截（GVA 路径下二者互斥）","scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：hasG && hasGk"},{"case_key":"neg_02_k_too_large","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_02_k_too_large","note":"K=512：本版只支持 K = V = 128","scale":0.04419417382415922,"shape":{"B":1,"Hk":4,"Hv":4,"K":512,"T":256,"V":128},"trigger":"host tiling：K != 128"},{"case_key":"neg_03_hv_not_multiple_of_hk","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_03_hv_not_multiple_of_hk","note":"Hk=3 Hv=4：GVA 映射不成立","scale":0.08838834764831845,"shape":{"B":1,"Hk":3,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：Hv % Hk != 0"},{"case_key":"neg_04_dense_b_greater_than_one","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_04_dense_b_greater_than_one","note":"未传 cu_seqlens 且 B=2：一次 launch 只处理一个 segment","scale":0.08838834764831845,"shape":{"B":2,"Hk":4,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：dense 路径要求 B == 1"},{"case_key":"neg_05_varlen_b_greater_than_one","chunk_size":64,"cu_seqlens":[0,128,256],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_05_varlen_b_greater_than_one","note":"传 cu_seqlens 且 B=2：packed 输入必须 B==1","scale":0.08838834764831845,"shape":{"B":2,"Hk":4,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：varlen 路径要求 B == 1"},{"case_key":"neg_06_chunk_size_not_64","chunk_size":128,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_06_chunk_size_not_64","note":"chunk_size=128 与本版 tiling 不匹配","scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：chunk_size != 64"},{"case_key":"neg_07_g_shape_mismatch","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hk","gate":"g","gk_dtype":"model","id":"neg_07_g_shape_mismatch","note":"g 为 [B,Hk,T] 而非 [B,Hv,T]","scale":0.08838834764831845,"shape":{"B":1,"Hk":2,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：g shape != [B,Hv,T]"},{"case_key":"neg_08_gk_dtype_fp32","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"fp32","g_head":"Hv","gate":"gk","gk_dtype":"fp32","id":"neg_08_gk_dtype_fp32","note":"gk 为 FP32：逐 K gate 必须与 q/k 同 dtype","scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：gk dtype != q dtype"},{"case_key":"neg_09_cu_seqlens_too_short","chunk_size":64,"cu_seqlens":[0],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_09_cu_seqlens_too_short","note":"cu_seqlens 只有 1 个元素，无法给出 [bos, eos)","scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":256,"V":128},"trigger":"host tiling：cu_seqlens shape dim0 - 1 < 1"},{"case_key":"neg_10_empty_tensor","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_10_empty_tensor","note":"T=0：空 tensor 必须给出明确错误","scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":0,"V":128},"trigger":"host tiling：B/T/Hk/Hv/K/V 存在 0"},{"case_key":"neg_11_k64_not_128","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_11_k64_not_128","note":"本版只支持 K = V = 128：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 逐行的向量/寄存器读写也依赖该行宽；其它取值实测会设备报错或算错，必须拦截","scale":0.125,"shape":{"B":1,"Hk":4,"Hv":4,"K":64,"T":256,"V":64},"trigger":"host tiling：K != 128（K=64，曾是历史用例，现按不支持拦截）"},{"case_key":"neg_12_v96_not_128","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_12_v96_not_128","note":"本版只支持 K = V = 128：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 逐行的向量/寄存器读写也依赖该行宽；其它取值实测会设备报错或算错，必须拦截","scale":0.08838834764831845,"shape":{"B":1,"Hk":4,"Hv":4,"K":128,"T":256,"V":96},"trigger":"host tiling：V != 128（V=96，曾是历史用例，现按不支持拦截）"},{"case_key":"neg_13_k256_not_128","chunk_size":64,"cu_seqlens":[],"dtype":"bf16","expected_return_code":"ACLNN_ERR_PARAM_INVALID","g_dtype":"model","g_head":"Hv","gate":"none","gk_dtype":"model","id":"neg_13_k256_not_128","note":"本版只支持 K = V = 128：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 逐行的向量/寄存器读写也依赖该行宽；其它取值实测会设备报错或算错，必须拦截","scale":0.0625,"shape":{"B":1,"Hk":4,"Hv":4,"K":256,"T":256,"V":128},"trigger":"host tiling：K != 128（K=256，曾是历史用例，现按不支持拦截）"}]'

def build_negative_specs() -> list[dict]:
    """反向（拦截）用例 spec：与正向共用 case_spec 通道，额外带 expected_return_code。"""
    return json.loads(NEG_SPECS_JSON)

def _select(keys) -> list[dict]:
    wanted = set(keys)
    return [spec for spec in _load_cases() if spec["id"] in wanted]


def build_accuracy_specs() -> list[dict]:
    return _load_cases()


def build_perf_specs() -> list[dict]:
    return _select(PERF_KEYS)


def build_mss_specs() -> list[dict]:
    return _select(MSS_KEYS)


def _payloads(specs: list[dict], tags: str, start: int = 0) -> list[dict]:
    return [_case_payload(start + index, spec, tags) for index, spec in enumerate(specs)]


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
    negative = build_negative_specs()
    perf = build_perf_specs()
    mss = build_mss_specs()
    _write(args.output_dir / f"atk_{OP_NAME}.json",
           _payloads(accuracy, "accuracy") + _payloads(negative, "negative", len(accuracy)))
    _write(args.output_dir / f"atk_{OP_NAME}_perf.json", _payloads(perf, "performance"))
    _write(args.output_dir / f"atk_{OP_NAME}_mss.json", _payloads(mss, "determinism,mss"))
    if args.summary:
        print(f"accuracy={len(accuracy)} negative={len(negative)} perf={len(perf)} determinism={len(mss)} mss={len(mss)}")


if __name__ == "__main__":
    main()
