"""ChunkKdaFwdFinalize 的 ATK executor 和独立 CPU 参考。"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Optional

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


OP_NAME = "chunk_kda_fwd_finalize"
CHUNK_SIZE = 64
HEAD_DIM = 128


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return bool(value)


def _int_array(value: Any) -> Optional[tuple[int, ...]]:
    if value is None or (isinstance(value, str) and value.strip().lower() == "null"):
        return None
    if isinstance(value, (tuple, list)):
        if len(value) == 1 and value[0] in (None, "null"):
            return None
        return tuple(int(item) for item in value)
    raise TypeError("cu_seqlens/chunk_indices must be public integer arrays or null")


def _chunk_map(batch: int, tokens: int, cu: Optional[tuple[int, ...]]):
    if cu is None:
        return [[(start, min(start + CHUNK_SIZE, tokens), start // CHUNK_SIZE)
                 for start in range(0, tokens, CHUNK_SIZE)] for _ in range(batch)]
    chunks = []
    chunk_id = 0
    for begin, end in zip(cu, cu[1:]):
        for start in range(begin, end, CHUNK_SIZE):
            chunks.append((start, min(start + CHUNK_SIZE, end), chunk_id))
            chunk_id += 1
    return [chunks]


@dataclass
class Inputs:
    qg_scaled: torch.Tensor
    aqk: torch.Tensor
    v_new: torch.Tensor
    h: torch.Tensor
    cu_seqlens: Optional[tuple[int, ...]]
    chunk_indices: Optional[tuple[int, ...]]
    output_layout: str
    state_v_first: bool
    batch: int
    heads: int
    tokens: int


def _prepare(values: dict[str, Any]) -> Inputs:
    qg = values["qg_scaled"]
    aqk = values["aqk"]
    v_new = values["v_new"]
    h = values["h"]
    if not all(torch.is_tensor(value) for value in (qg, aqk, v_new, h)):
        raise TypeError("all four stage inputs must be ATK tensors")
    layout = str(values["output_layout"])
    if layout in ("BNSD", "BSND") and qg.ndim == 4:
        batch, heads, tokens, dim = qg.shape
    elif layout in ("NTD", "TND") and qg.ndim == 3:
        heads, tokens, dim = qg.shape
        batch = 1
    else:
        raise ValueError("output_layout does not match the head-major input rank")
    if dim != HEAD_DIM or heads <= 0 or tokens <= 0 or batch <= 0:
        raise ValueError("qg_scaled must have positive B/HV/T and K=128")
    if aqk.shape != qg.shape[:-1] + (CHUNK_SIZE,):
        raise ValueError("Aqk must use the same head/token axes and 64 columns")
    value_shapes = [(batch, heads, tokens, 128)]
    if qg.ndim == 3:
        value_shapes.append((heads, tokens, 128))
    if (tuple(v_new.shape) not in value_shapes
            or h.shape[:2] != (batch, heads)
            or h.shape[-2:] != (128, 128)):
        raise ValueError("v_new or h shape does not match stage inputs")
    cu = _int_array(values.get("cu_seqlens"))
    indices = _int_array(values.get("chunk_indices"))
    if cu is not None:
        if batch != 1 or len(cu) < 2 or cu[0] != 0 or cu[-1] != tokens:
            raise ValueError("varlen requires B=1 and cu_seqlens from 0 to T")
        if any(left >= right for left, right in zip(cu, cu[1:])):
            raise ValueError("cu_seqlens must be strictly increasing")
    if indices is not None:
        if cu is None:
            raise ValueError("chunk_indices requires cu_seqlens")
        canonical = tuple(
            element
            for sequence, (left, right) in enumerate(zip(cu, cu[1:]))
            for chunk in range((right - left + 63) // 64)
            for element in (sequence, chunk)
        )
        if indices != canonical:
            raise ValueError("chunk_indices must be canonical sequence-major pairs")
    expected_chunks = len(_chunk_map(batch, tokens, cu)[0])
    if h.shape != (batch, heads, expected_chunks, 128, 128):
        raise ValueError("h chunk dimension does not match cu_seqlens")
    return Inputs(qg, aqk, v_new, h, cu, indices, layout,
                  _as_bool(values["state_v_first"]), batch, heads, tokens)


def _head_major(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0) if value.ndim == 3 else value


def _output_layout(value: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "BNSD":
        return value.contiguous()
    if layout == "BSND":
        return value.permute(0, 2, 1, 3).contiguous()
    if layout == "NTD":
        return value[0].contiguous()
    return value[0].permute(1, 0, 2).contiguous()


def _output_shape(inputs: Inputs) -> tuple[int, ...]:
    if inputs.output_layout == "BNSD":
        return (inputs.batch, inputs.heads, inputs.tokens, 128)
    if inputs.output_layout == "BSND":
        return (inputs.batch, inputs.tokens, inputs.heads, 128)
    if inputs.output_layout == "NTD":
        return (inputs.heads, inputs.tokens, 128)
    return (inputs.tokens, inputs.heads, 128)


def run_cpu(inputs: Inputs) -> torch.Tensor:
    # CPU 参考独立消费已经 BF16 舍入的四个阶段输入。
    qg = _head_major(inputs.qg_scaled.to(torch.bfloat16)).float()
    aqk = _head_major(inputs.aqk.to(torch.bfloat16)).float()
    v_new = _head_major(inputs.v_new.to(torch.bfloat16)).float()
    h = inputs.h.to(torch.bfloat16).float()
    result = torch.zeros((inputs.batch, inputs.heads, inputs.tokens, 128), dtype=torch.float32)
    for batch_id, chunks in enumerate(_chunk_map(inputs.batch, inputs.tokens, inputs.cu_seqlens)):
        for begin, end, chunk_id in chunks:
            rows = end - begin
            state = h[batch_id, :, chunk_id]
            if inputs.state_v_first:
                state = state.transpose(-1, -2)
            qh = torch.matmul(qg[batch_id, :, begin:end], state)
            av = torch.matmul(aqk[batch_id, :, begin:end, :rows],
                              v_new[batch_id, :, begin:end])
            result[batch_id, :, begin:end] = (qh + av).to(torch.bfloat16).float()
    return _output_layout(result, inputs.output_layout)


def run_npu(inputs: Inputs) -> torch.Tensor:
    # 局限于 ATK 交付目录的 aclnn 直调，不向共享 Python 注册表添加临时入口。
    from fla_npu.ops.ascendc._runtime import ACL_FORMAT_ND, call_aclnn

    output = torch.empty(_output_shape(inputs), dtype=torch.bfloat16,
                         device=inputs.qg_scaled.device)
    layout_bytes = ctypes.create_string_buffer(inputs.output_layout.encode("ascii"))

    def nd_tensor(ctx, tensor, name):
        tensor = tensor.contiguous()
        if tensor.storage_offset():
            tensor = tensor.clone()
        return ctx.tensor(tensor, name, acl_format_override=ACL_FORMAT_ND,
                          storage_shape_override=tuple(tensor.shape))

    argtypes = [
        *([ctypes.c_void_p] * 6), ctypes.c_char_p, ctypes.c_bool,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    return call_aclnn(
        "aclnnChunkKdaFwdFinalize",
        lambda ctx: [
            nd_tensor(ctx, inputs.qg_scaled, "qg_scaled"),
            nd_tensor(ctx, inputs.aqk, "aqk"),
            nd_tensor(ctx, inputs.v_new, "v_new"),
            nd_tensor(ctx, inputs.h, "h"),
            ctx.int_array(inputs.cu_seqlens),
            ctx.int_array(inputs.chunk_indices),
            ctypes.cast(layout_bytes, ctypes.c_char_p),
            ctypes.c_bool(inputs.state_v_first),
            nd_tensor(ctx, output, "attn_out"),
        ],
        output,
        get_workspace_argtypes=argtypes,
    )


@register("executor_chunk_kda_fwd_finalize")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.inputs: Optional[Inputs] = None

    def init_by_input_data(self, input_data: InputDataset):
        self.inputs = _prepare(input_data.kwargs)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.inputs is None:
            self.init_by_input_data(input_data)
        if self.inputs is None:
            raise RuntimeError("ATK inputs were not initialized")
        if self.device == "cpu":
            output = run_cpu(self.inputs)
        elif self.device in ("npu", "pyaclnn"):
            output = run_npu(self.inputs)
        else:
            raise RuntimeError("unsupported ATK backend %r" % self.device)
        if (tuple(output.shape) != _output_shape(self.inputs)
                or output.dtype != (torch.float32 if self.device == "cpu"
                                    else torch.bfloat16)):
            raise RuntimeError("unexpected attn_out dtype or shape")
        if not with_output:
            return None
        if self.device != "cpu":
            torch.npu.synchronize()
        if not torch.isfinite(output.float()).all().item():
            raise RuntimeError("attn_out contains NaN or Inf")
        return (output,)

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        if self.inputs is None:
            raise RuntimeError("ATK inputs were not initialized")
        return {"output_layout": self.inputs.output_layout,
                "state_v_first": self.inputs.state_v_first,
                "chunks": int(self.inputs.h.shape[2])}
