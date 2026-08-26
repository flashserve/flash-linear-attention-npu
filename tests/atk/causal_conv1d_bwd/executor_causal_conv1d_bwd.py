"""causal_conv1d_bwd 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _RCP_LN2,
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _gate,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _num_chunks,
    _orig_dtype,
    _rand,
    _randn,
    _zeros,
)


OP_NAME = "causal_conv1d_bwd"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, T, D, W = (int(spec[x]) for x in ("B", "T", "D", "W"))
    return {
        "x": _randn((B, T, D), dtype_name, calc_dtype, device, seed + 1),
        "weight": _randn((W, D), dtype_name, calc_dtype, device, seed + 2),
        "dy": _randn((B, T, D), dtype_name, calc_dtype, device, seed + 3),
    }


def _causal_conv1d_ref(x, weight, bias):
    calc = torch.float64 if x.dtype == torch.float64 else torch.float32
    y = F.conv1d(
        x.to(calc).permute(0, 2, 1).contiguous(),
        weight.to(calc).t().unsqueeze(1).contiguous(),
        bias=bias.to(calc) if bias is not None else None,
        padding=weight.shape[0] - 1,
        groups=x.shape[-1],
    )
    return y[..., : x.shape[1]].permute(0, 2, 1).contiguous().to(x.dtype)


def _activation_bwd_grad(dy, y, activation: int):
    """把上游梯度 dy 转成预激活位置的有效梯度 g。"""
    if activation == 0:
        return dy
    if activation in (1, 2):  # SiLU / Swish（当前等价）
        if y is None:
            raise ValueError("activation 为 1/2 时必须提供 yOptional。")
        sig = torch.sigmoid(y)
        return dy * sig * (1.0 + y * (1.0 - sig))
    raise ValueError(f"activation 仅支持 0/1/2，当前为 {activation!r}")


def _causal_conv1d_bwd_ref(
    inputs: dict[str, Any],
    y_optional=None,
    initial_state=None,
    dht=None,
    activation: int = 0,
    input_layout: str = "BSND",
):
    if input_layout not in {"BSND", "BSH"}:
        raise NotImplementedError(
            f"CPU 标杆当前仅覆盖固定长度 BSND/BSH 场景，收到 layout={input_layout!r}"
        )

    x_in, w_in, dy_in = inputs["x"], inputs["weight"], inputs["dy"]
    calc = torch.float64 if x_in.dtype == torch.float64 else torch.float32

    B, T, D = x_in.shape
    W = w_in.shape[0]

    x = x_in.detach().clone().to(calc).requires_grad_(True)
    weight = w_in.detach().clone().to(calc).requires_grad_(True)
    dy = dy_in.detach().to(calc)

    # ---- 构造带历史状态的输入：h0 提供序列开头依赖的 W-1 个历史时刻 ----
    if initial_state is None:
        h0 = torch.zeros((B, W, D), dtype=calc, device=x_in.device)
    else:
        h0 = initial_state.detach().clone().to(calc)
    h0.requires_grad_(True)

    # 状态 layout [B, W, D]，最后 W-1 个 slot 为紧邻序列起点的历史，slot 顺序与时间递增一致
    hist = h0[:, W - 1 :, :] if W == 1 else h0[:, -(W - 1) :, :]
    x_pad = x if W == 1 else torch.cat([hist, x], dim=1)

    # ---- 前向预激活输出（显式左填充，故 conv1d 不再补 padding）----
    y = F.conv1d(
        x_pad.permute(0, 2, 1).contiguous(),
        weight.t().unsqueeze(1).contiguous(),
        bias=None,
        padding=0,
        groups=D,
    ).permute(0, 2, 1).contiguous()[:, :T, :]

    # ---- 有效梯度 g，并把 dht 反传到序列尾部 ----
    g = _activation_bwd_grad(dy, y_optional.detach().to(calc) if y_optional is not None else None, activation)

    y.backward(g, retain_graph=dht is not None)

    dx = x.grad
    dw = weight.grad
    db = g.sum(dim=(0, 1))
    dh0 = h0.grad if h0.grad is not None else torch.zeros_like(h0)
    if initial_state is None:
        dh0 = torch.zeros((B, W, D), dtype=calc, device=x_in.device)

    if dht is not None:
        # 最终卷积状态 ht 为序列尾部最后 W 个时刻的输入，其梯度直接叠加回 dx
        dht_c = dht.detach().to(calc)
        tail = min(W, T)
        dx = dx.clone()
        dx[:, T - tail :, :] += dht_c[:, W - tail :, :]

    return (
        dx.to(x_in.dtype),
        dw.to(w_in.dtype),
        db.to(dy_in.dtype),
    )


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _causal_conv1d_bwd_ref(
        inputs,
        y_optional=None,
        initial_state=None,
        dht=None,
        activation=0,
        input_layout="BSND",
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.causal_conv1d_bwd(inputs["x"], None, inputs["weight"], inputs["dy"], initial_state=None, dht=None, activation=0, input_layout="BSND")[:3]


@register("executor_causal_conv1d_bwd")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
