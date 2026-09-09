"""chunk_gated_delta_rule_fwd_prepare 的 ATK executor。

输入生成、CPU 标杆（本目录 scripts/cpu_golden.py 的 cpu_gdn_fwd_l2norm_to_recompute）、
run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
精度标准为 mixed_tolerance_bm：NPU DUT vs CPU 高精度 golden。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _finite_tuple,
    _marker_device,
    _randn,
)
from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute
from layout import seqlens_to_cu

OP_NAME = "chunk_gated_delta_rule_fwd_prepare"


def _parse_seqlens(spec: dict[str, Any], batch: int, seq_len: int) -> list[int] | None:
    raw = spec.get("seqlens")
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip().startswith("[") else [
            int(x) for x in raw.replace(" ", "").split(",") if x
        ]
    seqlens = [int(x) for x in raw]
    if not seqlens or any(n <= 0 for n in seqlens):
        raise ValueError(f"seqlens must be positive ints, got {seqlens!r}")
    if batch != 1:
        raise ValueError(f"varlen requires B=1, got B={batch}")
    if sum(seqlens) != seq_len:
        raise ValueError(f"sum(seqlens)={sum(seqlens)} != T={seq_len}")
    return seqlens


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec.get("chunk_size", 64))
    g_dtype = torch.float64 if high_precision else torch.float32
    seqlens = _parse_seqlens(spec, B, T)
    cu = None if seqlens is None else seqlens_to_cu(seqlens, device=device, dtype=torch.int64)
    return {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "v": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": _randn((B, HV, T), "fp32", g_dtype, device, seed + 4, 0.2),
        "beta": _randn((B, HV, T), "fp32", g_dtype, device, seed + 5, 0.5),
        "chunk_size": chunk_size,
        "use_qk_l2norm_in_kernel": bool(spec.get("use_qk_l2norm_in_kernel", True)),
        "use_gate_in_kernel": bool(spec.get("use_gate_in_kernel", False)),
        "use_beta_sigmoid_in_kernel": bool(spec.get("use_beta_sigmoid_in_kernel", True)),
        "allow_neg_eigval": bool(spec.get("allow_neg_eigval", False)),
        "use_exp2": bool(spec.get("use_exp2", True)),
        "cu_seqlens": cu,
    }


def _forward_ref(inputs: dict[str, Any]):
    ref = cpu_gdn_fwd_l2norm_to_recompute(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        chunk_size=int(inputs["chunk_size"]),
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=inputs["use_gate_in_kernel"],
        use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=inputs["allow_neg_eigval"],
        cu_seqlens=inputs.get("cu_seqlens"),
        layout="bnsd",
    )
    return (
        ref.q,
        ref.k,
        ref.q_rstd,
        ref.k_rstd,
        ref.beta,
        ref.g,
        ref.w,
        ref.u,
        ref.a,
    )


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _forward_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)

    outputs = ascendc.chunk_gated_delta_rule_fwd_prepare(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        chunk_size=inputs["chunk_size"],
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=inputs["use_gate_in_kernel"],
        use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=inputs["allow_neg_eigval"],
        use_exp2=True,
        cu_seqlens=inputs.get("cu_seqlens"),
    )
    torch.npu.synchronize()
    # Host isfinite/compare: avoid queuing extra NPU kernels on the op stream.
    return tuple(None if t is None else t.detach().cpu() for t in outputs)


@register("executor_chunk_gated_delta_rule_fwd_prepare")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(getattr(task_result, "is_benchmark_task", False))
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, high_precision=self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
