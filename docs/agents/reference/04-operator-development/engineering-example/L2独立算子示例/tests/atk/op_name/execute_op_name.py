"""示例文件：tests/atk/op_name/execute_op_name.py

规范来源：ATK 仓 skill/atk-quality-guard（SKILL.md 步骤 3b「执行插件关键要点」、
references/atk_user_guide.md 第 7/8/12 节）

注意事项：
  1. `@register` 的名称必须与 YAML 顶层 `api_type` 一致（本示例：`op_name_execute`）。
     若只跑 CPU/`atk node -b npu task`，有 `api_type` 插件即可；要跑 `atk aclnn`/`pyaclnn`
     还必须提供 YAML 的 `aclnn_name` 与 `aclnn_api_type`（默认 `aclnn_function` 够用时不写自定义插件）。
  2. 同一个插件里按 `self.device` 分支：`"npu"` 走算子真实接口（本仓为
     `fla_npu.ops.ascendc.op_name`，发布物路径按 `sys.path.insert(0, publish_path)` 导入），
     `"cpu"` 走 PyTorch 原生 golden。两条路径不要混写：CPU golden 不得依赖 NPU 专属类。
  3. CPU golden 用 PyTorch 原生算子实现；低精度输入先转 fp32 计算，最后 cast 回目标 dtype，
     避免把 CPU 侧的精度损失算到被测算子头上。
  4. `with_output=True` 时才返回结果；返回结构（个数、shape、dtype）要能支撑 pyaclnn/NPU 侧构造输出。
  5. 输入从 `InputDataset` 取（`args` / `kwargs` / `method_args` / `method_kwargs` / `tensor_args`），
     字段名与 YAML 的 `inputs[].name` 对应；当前 ATK 版本的取值方式可能有差异，按实际版本调整。
  6. 本文件不修改算子源码、不修改 ATK 框架代码，也不在失败时改判精度结论——精度失败按
     `troubleshooting.md` 区分"用例设计问题"与"算子实现问题"。
"""

from __future__ import annotations

import os
import sys

import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 算子发布物路径（Phase B 环境变量注入）：例如 custom OPP 的 python 包目录。
PUBLISH_PATH = os.environ.get("FLA_NPU_PUBLISH_PATH", "")


def forward_v2(x, g, a_log=None, initial_state=None, cu_seqlens=None, chunk_indices=None,
               layout="BSND", scale=1.0, chunk_size=64, epsilon=1.0e-6, return_saved=False):
    """NPU 路径：调用算子真实接口（本仓稳定入口 `fla_npu.ops.ascendc.op_name`）。"""
    if PUBLISH_PATH:
        sys.path.insert(0, PUBLISH_PATH)
    from fla_npu.ops.ascendc import op_name

    return op_name(x, g, a_log=a_log, initial_state=initial_state,
                   cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                   layout=layout, scale=scale, chunk_size=chunk_size,
                   epsilon=epsilon, return_saved=return_saved)


def forward_golden_v2(x, g, a_log=None, initial_state=None, cu_seqlens=None, chunk_indices=None,
                      layout="BSND", scale=1.0, chunk_size=64, epsilon=1.0e-6,
                      return_saved=False):
    """CPU 标杆：PyTorch 原生实现（低精度先转 fp32，再 cast 回目标 dtype）。

    示例算子语义：chunk 内 scan + L2 归一化 + 可选导出 state/x_norm。
    真实算子的 golden 必须按算子 README/docs/api.md 的语义逐步实现，不使用 NPU 专属算子。
    """
    calc_dtype = torch.float32
    x_f = x.to(calc_dtype)
    g_f = g.to(calc_dtype)
    dim = int(x_f.shape[-1])
    norm = torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) / float(dim) + float(epsilon))

    scan = torch.zeros_like(g_f)
    chunk_ends = []
    for start in range(0, g_f.shape[-1], int(chunk_size)):
        end = min(start + int(chunk_size), g_f.shape[-1])
        chunk_scan = torch.cumsum(g_f[..., start:end] * float(scale), dim=-1)
        scan[..., start:end] = chunk_scan
        chunk_ends.append(chunk_scan[..., end - 1])
    y = (x_f * scan.unsqueeze(-1) * norm).to(x.dtype)

    if not return_saved:
        return y, None, None
    state = torch.stack(chunk_ends, dim=-1).to(x.dtype)
    return y, state, norm.squeeze(-1)


def _collect_inputs(input_data):
    """按 YAML 的 inputs 顺序取出本次用例的实参（按当前 ATK 版本调整读取方式）。"""
    merged = {}
    for source in (getattr(input_data, "kwargs", None) or {},
                   getattr(input_data, "method_kwargs", None) or {}):
        merged.update(dict(source))
    for item in (getattr(input_data, "args", None) or ()):
        if isinstance(item, dict):
            merged.update(item)
    return merged


@register("op_name_execute")
class OpNameExecute(BaseApi):
    """ATK 执行插件：注册名与 YAML `api_type` 一致，按 device 分支。"""

    def __call__(self, input_data, with_output: bool = False):
        inputs = _collect_inputs(input_data)
        call_args = {
            "x": inputs.get("x"),
            "g": inputs.get("g"),
            "a_log": inputs.get("a_log"),
            "initial_state": inputs.get("initial_state"),
            "cu_seqlens": inputs.get("cu_seqlens"),
            "chunk_indices": inputs.get("chunk_indices"),
            "layout": inputs.get("layout", "BSND"),
            "scale": float(inputs.get("scale", 1.0)),
            "chunk_size": int(inputs.get("chunk_size", 64)),
            "epsilon": float(inputs.get("epsilon", 1.0e-6)),
            "return_saved": bool(inputs.get("return_saved", False)),
        }
        if call_args["x"] is None or call_args["g"] is None:
            raise RuntimeError("op_name_execute: 用例缺少 x/g 输入。")

        if self.device == "npu":
            outputs = forward_v2(**call_args)
        elif self.device == "cpu":
            outputs = forward_golden_v2(**call_args)
        else:
            raise RuntimeError(f"Unsupported device: {self.device}")

        if with_output:
            return outputs
        return None
