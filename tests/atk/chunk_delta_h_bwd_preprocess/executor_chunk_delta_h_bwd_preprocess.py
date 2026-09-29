"""ChunkDeltaHBwdPreprocess ATK executor and CPU reference entry.

输入按 ``case_spec`` 里的 shape/seed 确定性构造（与算子在 tests/operators 下的 harness 同一套分布：
``randn*0.05`` 的 q/k/w/do/dv，以及沿 token 维 ``-cumsum(rand*0.05)`` 的 gate），
CPU 标杆复用 ``scripts/reference.py`` 的 ``preprocess_reference``；判据是 yaml/用例里声明的
ATK 原生 ``mixed_tolerance_bm``（按模型 dtype 判 ``dhm``）。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _gate,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _orig_dtype,
    _randn,
)


OP_NAME = "chunk_delta_h_bwd_preprocess"
CHUNK_SIZE = 64
_MODEL_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
_REFERENCE_FILE = Path(__file__).resolve().parent / "scripts" / "reference.py"
_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "atk_chunk_delta_h_bwd_preprocess_cpu", _REFERENCE_FILE)
if _REFERENCE_SPEC is None or _REFERENCE_SPEC.loader is None:
    raise ImportError(f"Unable to load CPU reference: {_REFERENCE_FILE}")
_REFERENCE = importlib.util.module_from_spec(_REFERENCE_SPEC)
_REFERENCE_SPEC.loader.exec_module(_REFERENCE)


def _segment(spec: dict):
    cu = [int(x) for x in spec.get("cu_seqlens", [])]
    if not cu:
        return None
    # 非空就原样交给算子：反向用例 neg_09 故意只给 1 个元素，必须让 host 拦到"至少 2 个元素"。
    return cu[:2] if len(cu) >= 2 else cu


def build_inputs(spec: dict, device: torch.device) -> dict:
    """按用例参数构造算子输入（CPU 节点与 NPU 节点用同一份确定性分布）。"""

    model = str(spec.get("dtype", "bf16"))
    model_dtype = _orig_dtype(model)
    seed = int(spec.get("seed", 20260927))
    batch = int(spec["B"])
    key_heads = int(spec["Hk"])
    value_heads = int(spec["Hv"])
    total_tokens = int(spec["T"])
    k_dim = int(spec["K"])
    v_dim = int(spec["V"])
    inputs = {
        "q": _randn((batch, key_heads, total_tokens, k_dim), model, model_dtype,
                    device, seed + 1),
        "k": _randn((batch, key_heads, total_tokens, k_dim), model, model_dtype,
                    device, seed + 2),
        "w": _randn((batch, value_heads, total_tokens, k_dim), model, model_dtype,
                    device, seed + 3),
        "d_o": _randn((batch, value_heads, total_tokens, v_dim), model, model_dtype,
                      device, seed + 4),
        "dv": _randn((batch, value_heads, total_tokens, v_dim), model, model_dtype,
                     device, seed + 5),
        "g": None,
        "gk": None,
    }
    gate = str(spec.get("gate", "none"))
    # gate="both" 用于反向用例：同时给出 g 与 gk，校验算子必须按互斥拦截（返回 161001）。
    if gate in ("g", "both"):
        gate_dtype = (torch.float32 if spec.get("g_dtype") == "fp32" else model_dtype)
        # 反向用例 neg_07：g 按 [B,Hk,T] 构造（正常 GVA 路径要求 [B,Hv,T]），必须被拦截。
        gate_heads = key_heads if str(spec.get("g_head", "Hv")) == "Hk" else value_heads
        inputs["g"] = _gate((batch, gate_heads, total_tokens), gate_dtype, device, seed + 6)
    if gate in ("gk", "both"):
        # 反向用例 neg_08：gk 按 FP32 构造（要求与 q/k 同 dtype），必须被拦截。
        gk_dtype = (torch.float32 if str(spec.get("gk_dtype", "model")) == "fp32" else model_dtype)
        inputs["gk"] = _kda_gate((batch, value_heads, total_tokens, k_dim), model,
                                 gk_dtype, device, seed + 7)
    return inputs


def _validate_outputs(outputs, spec: dict, tensors: dict) -> None:
    hv = int(spec["Hv"])
    k_dim = int(spec["K"])
    v_dim = int(spec["V"])
    if len(outputs) != 1:
        raise RuntimeError(f"{OP_NAME}: expected 1 output, got {len(outputs)}")
    dhm = outputs[0]
    expected = (hv, k_dim, v_dim + k_dim)
    if tuple(dhm.shape) != expected:
        raise RuntimeError(
            f"{OP_NAME}: dhm shape {tuple(dhm.shape)} != {expected}")
    if dhm.dtype != torch.float32:
        raise RuntimeError(f"{OP_NAME}: dhm dtype {dhm.dtype} != torch.float32")


_ACLNN_CODES = {
    "ACLNN_ERR_PARAM_INVALID": 161001,
    "ACLNN_ERR_PARAM_NULLPTR": 161002,
}


def _negative_expected_code(spec: dict):
    raw = spec.get("expected_return_code")
    if raw is None:
        # case_spec 里没带期望码时，用 case_key 兜底识别反向用例（本算子的反向用例统一以 neg_ 命名，
        # 期望码统一是 ACLNN_ERR_PARAM_INVALID = 161001）；避免依赖 ATK 传递哪些字段。
        if "neg_" in str(spec.get("case_key", "")):
            return _ACLNN_CODES["ACLNN_ERR_PARAM_INVALID"]
        return None
    text = str(raw)
    if text.isdigit():
        return int(text)
    if text not in _ACLNN_CODES:
        raise RuntimeError(f"{OP_NAME}: unknown expected_return_code {text!r}")
    return _ACLNN_CODES[text]


def _placeholder_output(spec: dict):
    """反向用例没有真实输出：按用例参数的形状给一个占位 dhm（标准里已标 not_key，不做比对）。"""

    shape = spec["shape"]
    value_heads = int(shape["Hv"])
    k_dim = int(shape["K"])
    v_dim = int(shape["V"])
    return (torch.zeros((value_heads, k_dim, v_dim + k_dim), dtype=torch.float32),)


def _call_op(tensors: dict, spec: dict, *, npu: bool):
    segment = _segment(spec)
    if segment is not None:
        # ATK 的后处理对"非张量属性"不稳；cu_seqlens 统一以张量下发（反向用例给的长度 1 也照发）。
        segment = _int_tensor(segment, tensors["q"].device)
    scale = float(spec.get("scale", 1.0))
    chunk_size = int(spec.get("chunk_size", CHUNK_SIZE))
    if npu:
        from fla_npu.ops import ascendc

        outputs = ascendc.chunk_delta_h_bwd_preprocess(
            q=tensors["q"], k=tensors["k"], w=tensors["w"], d_o=tensors["d_o"],
            dv=tensors["dv"], g=tensors.get("g"), gk=tensors.get("gk"),
            cu_seqlens=segment, scale=scale, chunk_size=chunk_size,
        )
        torch.npu.synchronize()
        return (outputs,)
    dhm = _REFERENCE.preprocess_reference(
        tensors["q"], tensors["k"], tensors["w"], tensors["d_o"], tensors["dv"],
        tensors.get("g"), tensors.get("gk"), scale=scale, chunk_size=chunk_size,
        bos=segment[0] if segment else 0, eos=segment[1] if segment else None,
    )
    return (dhm,)


def run_cpu(spec: dict, input_data: InputDataset):
    if _negative_expected_code(spec) is not None:
        # 反向用例没有真实输出，也不该跑 CPU 标杆（非法 shape 会让参考实现直接抛错）。
        return _placeholder_output(spec)
    tensors = build_inputs(spec, torch.device("cpu"))
    outputs = _call_op(tensors, spec, npu=False)
    _validate_outputs(outputs, spec, tensors)
    return outputs


def run_npu(spec: dict, input_data: InputDataset):
    want = _negative_expected_code(spec)
    tensors = build_inputs(spec, _marker_device(input_data))
    if want is not None:
        # 反向（拦截）用例：算子必须报错，且返回码与用例声明一致。返回码取自异常文本里的
        # ``aclnnStatus=<code>``（与 tests/atk/chunk_kda_fwd 的负向处理同一套口径）。
        try:
            _call_op(tensors, spec, npu=True)
        except Exception as exc:  # aclnn 抛 RuntimeError，取其中的 aclnnStatus
            match = re.search(r"aclnnStatus=(\d+)", str(exc))
            got = int(match.group(1)) if match else None
            if got != want:
                raise RuntimeError(
                    f"{OP_NAME}: negative case {spec.get('case_key')} returned {got}, "
                    f"expected {want}: {exc}"
                ) from exc
            # 返回码与声明一致：抛出与用例 expected_error_msg **完全一致**的文本，
            # 交由 ATK 的"预期失败"判定（照 tests/atk/chunk_kda_fwd 的口径）。
            print(f"NPU negative case intercepted as expected: {spec['case_key']} code={got}", flush=True)
            raise RuntimeError(str(spec.get("expected_error_msg")
                                   or "{} ({}): {}".format(spec["expected_return_code"], got,
                                                           spec.get("note", ""))))
        raise RuntimeError(
            f"{OP_NAME}: negative case {spec.get('case_key')} unexpectedly succeeded, "
            f"expected code {want}"
        )
    outputs = _call_op(tensors, spec, npu=True)
    print(f"NPU operator completed: {spec['case_key']}", flush=True)
    _validate_outputs(outputs, spec, tensors)
    return outputs


@register(f"executor_{OP_NAME}")
class FunctionApi(BaseApi):
    """ATK 执行入口：NPU DUT 与 CPU 标杆共用本文件。"""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, input_data)
        else:
            raise RuntimeError(
                f"{OP_NAME} only supports NPU DUT and CPU golden, got {self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
