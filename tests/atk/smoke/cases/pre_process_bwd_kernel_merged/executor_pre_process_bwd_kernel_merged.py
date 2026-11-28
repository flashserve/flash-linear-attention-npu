"""``pre_process_bwd_kernel_merged`` 的 ATK 执行插件（direct-input 口径，NPU DUT + 双标杆）。

用例直接在 JSON 里声明算子的公开输入（``q/k/w/d_o/dv`` + 可选 ``g/gk`` + ``scale`` /
``chunk_size`` + ``cu_seqlens``），因此本文件不做任何输入构造：``B/Hk/Hv/T/K/V`` 全部由
ATK 生成的张量形状推导，gate 的 dtype 与存在性由 ``g``/``gk`` 本身决定。

三路角色（由 ATK 的节点类型与 ``is_benchmark_task`` 决定）：

===============  ========================  =================================================
角色              触发条件                   行为
===============  ========================  =================================================
``DUT``          ``device == npu``         调 ``fla_npu.ops.ascendc.pre_process_bwd_kernel_merged``
``golden``       ``is_benchmark_task``     CPU FP64 升精度真值（``mode="fp64"``）
``benchmark``    其余 CPU 调用             CPU 同精度标杆（``mode="contract"``，模型 dtype 契约）
===============  ========================  =================================================

阈值只在用例 JSON 的 ``standard.acc.cv_fused_double_benchmark`` 里声明，executor 不自造指标。
反向（拦截）用例不再需要专门分支：算子会按 host tiling 拦截并抛出异常，ATK 用用例声明的
``expected_error_msg``（``aclnnStatus=161001``）比对，命中即通过，不做精度比较。
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

from _ascendc_common_executor import _finite_tuple


OP_NAME = "pre_process_bwd_kernel_merged"
CHUNK_SIZE = 64
REQUIRED_TENSORS = ("q", "k", "w", "d_o", "dv")
_REFERENCE_FILE = Path(__file__).resolve().parent / "scripts" / "reference.py"
_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "atk_pre_process_bwd_kernel_merged_cpu", _REFERENCE_FILE)
if _REFERENCE_SPEC is None or _REFERENCE_SPEC.loader is None:
    raise ImportError(f"Unable to load CPU reference: {_REFERENCE_FILE}")
_REFERENCE = importlib.util.module_from_spec(_REFERENCE_SPEC)
_REFERENCE_SPEC.loader.exec_module(_REFERENCE)

_NULL_TEXT = {"", "null", "none"}


def _optional_tensor(value):
    """可选输入缺省：ATK 会传 ``None``、空张量或字符串 ``"null"``。"""
    if value is None:
        return None
    if isinstance(value, str):
        return None if value.strip().lower() in _NULL_TEXT else value
    if torch.is_tensor(value) and value.numel() == 0:
        return None
    return value


def _int_list(value):
    """``cu_seqlens`` 是 ``type="attrs"`` 的整型属性组；缺省时是单条 ``"null"``。"""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and str(value[0]).strip().lower() in _NULL_TEXT:
            return None
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NULL_TEXT:
            return None
        return [int(token) for token in text.replace(" ", "").split(",") if token]
    raise RuntimeError(f"{OP_NAME}: unsupported cu_seqlens value {value!r}")


def _read_case(values: dict) -> tuple[dict, list | None, float, int]:
    """从 ATK 传入的输入取值里读出张量、``cu_seqlens``、``scale`` 与 ``chunk_size``。"""
    tensors = {}
    for name in REQUIRED_TENSORS:
        tensor = values.get(name)
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"{OP_NAME}: input {name!r} is missing or not a tensor")
        tensors[name] = tensor
    tensors["g"] = _optional_tensor(values.get("g"))
    tensors["gk"] = _optional_tensor(values.get("gk"))
    return tensors, _int_list(values.get("cu_seqlens")), float(values.get("scale", 1.0)), \
        int(values.get("chunk_size", CHUNK_SIZE))


def _facts(tensors: dict) -> dict:
    """形状事实全部来自输入张量本身（不做任何 case_spec 解析）。"""
    q, w, d_o = tensors["q"], tensors["w"], tensors["d_o"]
    return {
        "B": int(q.shape[0]), "Hk": int(q.shape[1]), "T": int(q.shape[2]), "K": int(q.shape[3]),
        "Hv": int(w.shape[1]), "V": int(d_o.shape[3]),
    }


def _validate_outputs(outputs, facts: dict, *, npu: bool) -> None:
    if len(outputs) != 1:
        raise RuntimeError(f"{OP_NAME}: expected 1 output, got {len(outputs)}")
    dhm = outputs[0]
    expected = (facts["Hv"], facts["K"], facts["V"] + facts["K"])
    if tuple(dhm.shape) != expected:
        raise RuntimeError(f"{OP_NAME}: dhm shape {tuple(dhm.shape)} != {expected}")
    if npu and dhm.dtype != torch.float32:
        raise RuntimeError(f"{OP_NAME}: dhm dtype {dhm.dtype} != torch.float32")


def _call_npu(tensors: dict, cu_seqlens, scale: float, chunk_size: int):
    from fla_npu.ops import ascendc

    segment = cu_seqlens[:2] if cu_seqlens else None
    outputs = ascendc.pre_process_bwd_kernel_merged(
        q=tensors["q"], k=tensors["k"], w=tensors["w"], d_o=tensors["d_o"], dv=tensors["dv"],
        g=tensors.get("g"), gk=tensors.get("gk"), cu_seqlens=segment,
        scale=scale, chunk_size=chunk_size,
    )
    torch.npu.synchronize()
    return (outputs,)


def _call_reference(tensors: dict, cu_seqlens, scale: float, chunk_size: int, high_precision: bool):
    bos = int(cu_seqlens[0]) if cu_seqlens else 0
    eos = int(cu_seqlens[1]) if len(cu_seqlens or []) >= 2 else None
    dhm = _REFERENCE.preprocess_reference(
        tensors["q"], tensors["k"], tensors["w"], tensors["d_o"], tensors["dv"],
        tensors.get("g"), tensors.get("gk"),
        scale=scale, chunk_size=chunk_size, bos=bos, eos=eos,
        mode="fp64" if high_precision else "contract",
    )
    # 三路角色都按输出精度 FP32 比较：真值在 FP64 上算完再降到 FP32（~6e-8 相对误差，
    # 远小于模型 dtype 链的 ~1e-2），保证三路 dtype 一致。
    return (dhm.float(),)


def run_cpu(values: dict, high_precision: bool):
    tensors, cu_seqlens, scale, chunk_size = _read_case(values)
    facts = _facts(tensors)
    outputs = _call_reference(tensors, cu_seqlens, scale, chunk_size, high_precision)
    _validate_outputs(outputs, facts, npu=False)
    return outputs


def run_npu(values: dict):
    tensors, cu_seqlens, scale, chunk_size = _read_case(values)
    facts = _facts(tensors)
    outputs = _call_npu(tensors, cu_seqlens, scale, chunk_size)
    print(f"NPU operator completed: {facts}", flush=True)
    _validate_outputs(outputs, facts, npu=True)
    return outputs


@register(f"executor_{OP_NAME}")
class FunctionApi(BaseApi):
    """ATK 执行入口：NPU DUT 与 CPU 双标杆角色共用本文件。"""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        # CPU 参考节点会被 ATK 以两种角色各调用一次：golden（FP64 真值）与
        # benchmark（同精度契约版）；NPU 侧永远是 DUT。
        self.high_precision = self.device in {"cpu", "gpu"} and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        values = input_data.kwargs
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(values)
        elif self.device == "cpu":
            outputs = run_cpu(values, self.high_precision)
        else:
            raise RuntimeError(
                f"{OP_NAME} only supports NPU DUT and CPU reference, got {self.device!r}")
        return _finite_tuple(outputs)
