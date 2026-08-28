"""chunk_bwd_dv_local 的 ATK executor。"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _finite_tuple,
    _gate,
    _marker_device,
    _orig_dtype,
    _randn,
)


OP_NAME = "chunk_bwd_dv_local"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = _prepare_lens(cu_seqlens)
    indices = torch.cat([torch.arange(int(_ceil_div(n.item(), chunk_size))) for n in lens])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def _build_cu_seqlens(spec: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if not _as_bool(spec.get("var_len", False)):
        return None
    total_t = int(spec["T"])
    seq_num = max(int(spec.get("cu_seqlens_len", spec.get("B", 1))), 1)
    base = total_t // seq_num
    rem = total_t % seq_num
    lens = [base + (1 if i < rem else 0) for i in range(seq_num)]
    lens = [max(x, 1) for x in lens]
    lens[-1] += total_t - sum(lens)
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, dtype=torch.int64, device=device)


def _dtype_name(spec: dict[str, Any], key: str, default: str) -> str:
    return str(spec.get(key, default)).lower()


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = _dtype_name(spec, "dtype", "bf16")
    g_dtype_name = _dtype_name(spec, "g_dtype", dtype_name)
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    g_calc_dtype = torch.float64 if high_precision else _orig_dtype(g_dtype_name)
    seed = int(spec.get("seed", 20260817))

    batch = int(spec["B"])
    hk = int(spec["HK"])
    hv = int(spec["HV"])
    total_t = int(spec["T"])
    k_dim = int(spec["K"])
    v_dim = int(spec["V"])
    chunk_size = int(spec["chunk_size"])
    if hv % hk != 0:
        raise RuntimeError(f"HV({hv}) must be divisible by HK({hk})")

    cu_seqlens = _build_cu_seqlens(spec, device)
    physical_batch = 1 if cu_seqlens is not None else batch
    q_shape = (physical_batch, hk, total_t, k_dim)
    do_shape = (physical_batch, hv, total_t, v_dim)
    g_shape = (physical_batch, hv, total_t)
    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens.cpu(), chunk_size).to(device)

    return {
        "q": _randn(q_shape, dtype_name, calc_dtype, device, seed + 1),
        "k": _randn(q_shape, dtype_name, calc_dtype, device, seed + 2),
        "do": _randn(do_shape, dtype_name, calc_dtype, device, seed + 3),
        "g": _gate(g_shape, g_calc_dtype, device, seed + 4).to(_orig_dtype(g_dtype_name)).to(g_calc_dtype),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(k_dim))),
        "chunk_size": chunk_size,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }


def _chunk_bwd_dv_local_fixed_ref(inputs: dict[str, Any], high_precision: bool = False) -> torch.Tensor:
    q, k, do, g = inputs["q"], inputs["k"], inputs["do"], inputs["g"]
    batch, hk, total_t, _ = q.shape
    hv, v_dim = do.shape[1], do.shape[3]
    chunk_size = int(inputs["chunk_size"])
    scale = float(inputs["scale"])
    calc = torch.float64 if high_precision else torch.float32
    out = torch.zeros_like(do, dtype=calc)
    h_ratio = hv // hk
    block_t = min(chunk_size, max(16, 2 ** math.ceil(math.log2(max(total_t, 1)))))

    for b_idx in range(batch):
        for chunk_idx in range(_ceil_div(total_t, chunk_size)):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_t)
            chunk_len = end - start
            if chunk_len <= 0:
                continue
            token_offsets = chunk_idx * block_t + torch.arange(block_t, device=q.device)
            valid = token_offsets < total_t
            mask = (token_offsets[:, None] <= token_offsets[None, :]) & valid[:, None] & valid[None, :]
            for hv_idx in range(hv):
                hk_idx = hv_idx // h_ratio
                q_chunk = q[b_idx, hk_idx, start:end].to(calc)
                k_chunk = k[b_idx, hk_idx, start:end].to(calc)
                g_chunk = g[b_idx, hv_idx, start:end].to(calc)
                attn = torch.zeros((block_t, block_t), dtype=calc, device=q.device)
                attn[:chunk_len, :chunk_len] = torch.matmul(k_chunk, q_chunk.t())
                attn[:chunk_len, :chunk_len] *= torch.exp(g_chunk[None, :] - g_chunk[:, None]) * scale
                attn = torch.where(mask, attn, torch.zeros_like(attn))
                attn = attn.to(torch.float32 if high_precision else q.dtype)
                for v_start in range(0, v_dim, 128):
                    v_end = min(v_start + 128, v_dim)
                    do_chunk = do[b_idx, hv_idx, start:end, v_start:v_end]
                    if high_precision:
                        do_chunk = do_chunk.to(torch.float32)
                    out[b_idx, hv_idx, start:end, v_start:v_end] += torch.matmul(
                        attn[:chunk_len, :chunk_len], do_chunk
                    ).to(calc)
    return out.to(do.dtype)


def _chunk_bwd_dv_local_varlen_ref(inputs: dict[str, Any], high_precision: bool = False) -> torch.Tensor:
    q, k, do, g = inputs["q"], inputs["k"], inputs["do"], inputs["g"]
    cu_seqlens = inputs["cu_seqlens"]
    chunk_indices = inputs["chunk_indices"]
    if cu_seqlens is None or chunk_indices is None:
        raise RuntimeError("var_len reference requires cu_seqlens and chunk_indices")

    hk, total_t = q.shape[1], q.shape[2]
    hv, v_dim = do.shape[1], do.shape[3]
    chunk_size = int(inputs["chunk_size"])
    scale = float(inputs["scale"])
    calc = torch.float64 if high_precision else torch.float32
    out = torch.zeros_like(do, dtype=calc)
    h_ratio = hv // hk
    block_t = min(chunk_size, max(16, 2 ** math.ceil(math.log2(max(total_t, 1)))))
    flat_indices = chunk_indices.view(-1).cpu()

    for chunk_idx in range(len(flat_indices) // 2):
        seq_idx = int(flat_indices[chunk_idx * 2].item())
        local_chunk = int(flat_indices[chunk_idx * 2 + 1].item())
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        seq_len = eos - bos
        start = bos + local_chunk * chunk_size
        end = min(start + chunk_size, eos)
        chunk_len = end - start
        if chunk_len <= 0:
            continue
        token_offsets = local_chunk * block_t + torch.arange(block_t, device=q.device)
        valid = token_offsets < seq_len
        mask = (token_offsets[:, None] <= token_offsets[None, :]) & valid[:, None] & valid[None, :]
        for hv_idx in range(hv):
            hk_idx = hv_idx // h_ratio
            q_chunk = q[0, hk_idx, start:end].to(calc)
            k_chunk = k[0, hk_idx, start:end].to(calc)
            g_chunk = g[0, hv_idx, start:end].to(calc)
            attn = torch.zeros((block_t, block_t), dtype=calc, device=q.device)
            attn[:chunk_len, :chunk_len] = torch.matmul(k_chunk, q_chunk.t())
            attn[:chunk_len, :chunk_len] *= torch.exp(g_chunk[None, :] - g_chunk[:, None]) * scale
            attn = torch.where(mask, attn, torch.zeros_like(attn))
            attn = attn.to(torch.float32 if high_precision else q.dtype)
            for v_start in range(0, v_dim, 128):
                v_end = min(v_start + 128, v_dim)
                do_chunk = do[0, hv_idx, start:end, v_start:v_end]
                if high_precision:
                    do_chunk = do_chunk.to(torch.float32)
                out[0, hv_idx, start:end, v_start:v_end] += torch.matmul(
                    attn[:chunk_len, :chunk_len], do_chunk
                ).to(calc)
    return out.to(do.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    if inputs["cu_seqlens"] is not None:
        return _chunk_bwd_dv_local_varlen_ref(inputs, high_precision=high_precision)
    return _chunk_bwd_dv_local_fixed_ref(inputs, high_precision=high_precision)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    route = str(spec.get("route", "ascendc"))
    kwargs = {
        "g_gamma": None,
        "A": None,
        "cu_seqlens": None if inputs["cu_seqlens"] is None else inputs["cu_seqlens"].detach().cpu().tolist(),
        "chunk_indices": None if inputs["chunk_indices"] is None else inputs["chunk_indices"].detach().cpu().view(-1).tolist(),
    }
    args = (inputs["q"], inputs["k"], inputs["do"], inputs["g"], inputs["scale"], inputs["chunk_size"])
    if route == "ascendc":
        from fla_npu.ops import ascendc

        if hasattr(ascendc, "chunk_bwd_dv_local"):
            return ascendc.chunk_bwd_dv_local(*args, **kwargs)
        return ascendc.npu_chunk_bwd_dv_local(*args, **kwargs)
    if route == "aclnn":
        from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_bwd_dv_local

        return npu_chunk_bwd_dv_local(*args, **kwargs)
    raise ValueError(f"unsupported route: {route}")


@register("executor_chunk_bwd_dv_local")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.spec = None
        self.runtime_case_id = None
        case_config = getattr(task_result, "case_config", None)
        case_id = case_config.get("id") if isinstance(case_config, dict) else getattr(case_config, "id", None)
        if case_id is not None:
            self.runtime_case_id = int(case_id)
        self.high_precision = self.device == "cpu"

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = _case_spec(input_data, OP_NAME)
        if os.environ.get("CHUNK_BWD_DV_LOCAL_ATK_TRACE_SEED") == "1":
            print(
                "CHUNK_BWD_DV_LOCAL_ATK_RUNTIME_SEED",
                self.device,
                "cpu_golden=" + str(self.high_precision),
                "case_id=" + str(self.runtime_case_id),
                "seed=" + str(self.spec.get("seed")),
                flush=True,
            )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        if self.spec is None:
            self.init_by_input_data(input_data)
        if self.device == "cpu":
            outputs = run_cpu(self.spec, self.high_precision)
        elif self.device in {"npu", "pyaclnn", "gpu"}:
            outputs = run_npu(self.spec, input_data)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")

    def export_custom_data(self, input_data: InputDataset):
        if self.spec is None:
            self.init_by_input_data(input_data)
        return {
            "case_id": self.runtime_case_id,
            "soc": str(self.spec.get("soc", "")),
            "route": str(self.spec.get("route", "ascendc")),
            "dtype": str(self.spec.get("dtype", "bf16")),
            "B": int(self.spec["B"]),
            "HK": int(self.spec["HK"]),
            "HV": int(self.spec["HV"]),
            "T": int(self.spec["T"]),
            "K": int(self.spec["K"]),
            "V": int(self.spec["V"]),
            "chunk_size": int(self.spec["chunk_size"]),
            "var_len": _as_bool(self.spec.get("var_len", False)),
        }
