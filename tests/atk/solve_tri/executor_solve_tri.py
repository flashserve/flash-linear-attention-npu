"""solve_tri 的本机混合容差 ATK executor。

精度标准为 mixed_tolerance_bm：
  dut    = 本机 NPU ``fla_npu.ops.ascendc.solve_tri``
  golden = 本机 CPU FP64 ``linalg.inv``（``--bm_device cpu``）

不依赖远程 GPU。输入构造与 CPU 逆与 GPU 双标杆套相同，保证同一份 case_spec 可复现。
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _finite_tuple, _marker_device, _orig_dtype, _to_python

OP_NAME = "solve_tri"


def _register_mixed_tolerance_bm_if_missing() -> None:
    """ATK 26.8.8+ 自带 mixed_tolerance_bm；26.7.8 用同角色的 single_bm 顶上。"""
    try:
        from atk.tasks.task_plugins_register import ACCURACY_REGISTRY
        import atk.tasks.post_process.single_benchmark_compare as sbc
    except Exception:
        return
    if "mixed_tolerance_bm" in ACCURACY_REGISTRY:
        return
    ACCURACY_REGISTRY.register_with_key(
        "mixed_tolerance_bm",
        sbc.SingleBenchmarkAccuracyCompare,
    )


_register_mixed_tolerance_bm_if_missing()


def _as_bool(value) -> bool:
    value = _to_python(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_seqlens(value) -> List[int] | None:
    value = _to_python(value)
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = json.loads(text) if text.startswith("[") else [part for part in text.split(",") if part.strip()]
    return [int(item) for item in value]


def _normalize_spec(spec: dict[str, Any], op_name: str) -> dict[str, Any]:
    spec = dict(spec)
    spec.setdefault("op", op_name)
    spec.setdefault("dtype", "bf16")
    for key in ("B", "H", "T", "chunk_size", "num_seqs", "seed", "case_id"):
        if spec.get(key) is not None:
            spec[key] = int(spec[key])
    if spec.get("aligned") is not None:
        spec["aligned"] = _as_bool(spec["aligned"])
    seqlens = _as_seqlens(spec.get("seqlens"))
    if seqlens:
        spec["seqlens"] = seqlens
        spec["num_seqs"] = len(seqlens)
        spec["T"] = sum(seqlens)
    elif "seqlens" in spec:
        spec.pop("seqlens", None)
    spec["layout"] = str(spec.get("layout", "bsnd")).lower()
    spec["dtype"] = str(spec.get("dtype", "bf16")).lower()
    return spec


def _case_spec(input_data, op_name: str) -> dict[str, Any]:
    raw = _to_python(input_data.kwargs.get("case_spec"))
    spec: dict[str, Any] = {}
    if raw:
        if isinstance(raw, str):
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                spec.update(loaded)
        elif isinstance(raw, dict):
            spec.update(raw)
        else:
            raise TypeError(f"case_spec 类型不支持：{type(raw)!r}")
    for key, value in input_data.kwargs.items():
        if key == "case_spec" or isinstance(value, torch.Tensor):
            continue
        parsed = _to_python(value)
        if parsed is None:
            continue
        spec.setdefault(key, parsed)
    return _normalize_spec(spec, op_name)


def _calc_dtype(name: str, high_precision: bool):
    if high_precision:
        return torch.float32
    return _orig_dtype(name)


def _make_lower_tri_block(actual_size: int, chunk_size: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    block = torch.randn(actual_size, chunk_size, generator=gen, dtype=torch.float32) * 0.1
    block[:, :actual_size] = torch.tril(block[:, :actual_size], diagonal=-1)
    if chunk_size > actual_size:
        block[:, actual_size:] = 0
    return block.to(dtype)


def _generate_cu_seqlens(num_seqs: int, total_T: int, chunk_size: int, seed: int) -> List[int]:
    random.seed(seed)
    if num_seqs == 1:
        return [0, total_T]
    min_len = chunk_size
    max_seqs = total_T // min_len
    if max_seqs < num_seqs:
        num_seqs = max(1, max_seqs)
    if num_seqs == 1:
        return [0, total_T]
    remaining = total_T - num_seqs * min_len
    seq_lens = [min_len] * num_seqs
    for _ in range(remaining):
        idx = random.randint(0, num_seqs - 1)
        seq_lens[idx] += 1
    cu_seqlens = [0]
    for length in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return cu_seqlens


def _prepare_chunk_indices(cu_seqlens: List[int], chunk_size: int) -> List[int]:
    indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        num_chunks = (eos - bos + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            indices.extend([seq_idx, chunk_idx])
    return indices


def _cu_seqlens_from_spec(spec: dict[str, Any], seed: int) -> List[int]:
    raw = spec.get("seqlens")
    if raw:
        seqlens = [int(value) for value in raw]
        if not seqlens or any(length <= 0 for length in seqlens):
            raise ValueError(f"非法 seqlens：{seqlens}")
        cu_seqlens = [0]
        for length in seqlens:
            cu_seqlens.append(cu_seqlens[-1] + length)
        return cu_seqlens
    return _generate_cu_seqlens(
        int(spec.get("num_seqs", 1)),
        int(spec["T"]),
        int(spec["chunk_size"]),
        seed,
    )


def build_inputs_bsnd(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    orig_dtype = _orig_dtype(dtype_name)
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, H, T, chunk_size = (int(spec[x]) for x in ("B", "H", "T", "chunk_size"))
    x = torch.zeros(B, T, H, chunk_size, dtype=calc_dtype, device="cpu")
    num_chunks = (T + chunk_size - 1) // chunk_size
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * chunk_size
                e = min(s + chunk_size, T)
                block = _make_lower_tri_block(e - s, chunk_size, calc_dtype, seed + b * 10000 + h * 1000 + c)
                x[b, s:e, h, :] = block
    return {"x": x.to(orig_dtype).to(calc_dtype).to(device), "cu_seqlens": None, "chunk_indices": None, "layout": "bsnd"}


def build_inputs_tnd(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    orig_dtype = _orig_dtype(dtype_name)
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    H, chunk_size = (int(spec[x]) for x in ("H", "chunk_size"))
    cu_seqlens = _cu_seqlens_from_spec(spec, seed)
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
    total_T = cu_seqlens[-1]
    x = torch.zeros(total_T, H, chunk_size, dtype=calc_dtype, device="cpu")
    for seq_idx in range(len(cu_seqlens) - 1):
        bos, eos = cu_seqlens[seq_idx], cu_seqlens[seq_idx + 1]
        num_chunks = (eos - bos + chunk_size - 1) // chunk_size
        for h in range(H):
            for c in range(num_chunks):
                s = bos + c * chunk_size
                e = min(s + chunk_size, eos)
                block = _make_lower_tri_block(e - s, chunk_size, calc_dtype, seed + seq_idx * 100000 + h * 1000 + c)
                x[s:e, h, :] = block
    return {
        "x": x.to(orig_dtype).to(calc_dtype).to(device),
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "layout": "tnd",
    }


def _to_canonical_x(x: torch.Tensor, layout: str):
    layout = str(layout).lower()
    if layout == "bnsd":
        return x.permute(0, 2, 1, 3).contiguous(), "bsnd"
    if layout == "ntd":
        return x.permute(1, 0, 2).contiguous(), "tnd"
    return x, layout


def _from_canonical_x(x: torch.Tensor, layout: str) -> torch.Tensor:
    layout = str(layout).lower()
    if layout == "bnsd":
        return x.permute(0, 2, 1, 3).contiguous()
    if layout == "ntd":
        return x.permute(1, 0, 2).contiguous()
    return x


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    layout = str(spec.get("layout", "bsnd")).lower()
    if layout in {"tnd", "ntd"}:
        inputs = build_inputs_tnd(spec, device, high_precision)
        if layout == "ntd":
            inputs["x"] = inputs["x"].permute(1, 0, 2).contiguous()
            inputs["layout"] = "ntd"
        return inputs
    if layout not in {"bsnd", "bnsd"}:
        raise ValueError(f"不支持的 layout：{layout}")
    inputs = build_inputs_bsnd(spec, device, high_precision)
    if layout == "bnsd":
        inputs["x"] = inputs["x"].permute(0, 2, 1, 3).contiguous()
        inputs["layout"] = "bnsd"
    return inputs


def _inv_chunk(block: torch.Tensor) -> torch.Tensor:
    n = block.shape[0]
    eye = torch.eye(n, dtype=block.dtype, device=block.device)
    return torch.linalg.inv(eye + block)


def _mask_canonical(out: torch.Tensor, inputs: dict, chunk_size: int) -> torch.Tensor:
    layout = inputs["layout"]
    cu_seqlens = inputs.get("cu_seqlens")
    if layout == "bsnd":
        _, tokens, _, _ = out.shape
        num_chunks = (tokens + chunk_size - 1) // chunk_size
        last_actual = tokens - (num_chunks - 1) * chunk_size
        if last_actual < chunk_size:
            start = (num_chunks - 1) * chunk_size
            out[:, start:, :, last_actual:] = 0
        return out
    if layout == "tnd" and cu_seqlens is not None:
        for seq_idx in range(len(cu_seqlens) - 1):
            bos = cu_seqlens[seq_idx]
            eos = cu_seqlens[seq_idx + 1]
            num_chunks = (eos - bos + chunk_size - 1) // chunk_size
            last_chunk_start = bos + (num_chunks - 1) * chunk_size
            last_actual = eos - last_chunk_start
            if last_actual < chunk_size:
                out[last_chunk_start:eos, :, last_actual:] = 0
    return out


def _mask_output(out: torch.Tensor, inputs: dict, chunk_size: int) -> torch.Tensor:
    layout = inputs["layout"]
    work, canon = _to_canonical_x(out, layout)
    canon_inputs = dict(inputs)
    canon_inputs["layout"] = canon
    return _from_canonical_x(_mask_canonical(work, canon_inputs, chunk_size), layout)


def run_cpu(spec: dict[str, Any], high_precision: bool = True) -> torch.Tensor:
    """CPU golden：同一份量化输入上做 FP64 ``linalg.inv``。"""
    del high_precision
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
    layout = inputs["layout"]
    x, canon = _to_canonical_x(inputs["x"].detach().cpu().to(torch.float64), layout)
    chunk_size = int(spec["chunk_size"])
    result = torch.zeros_like(x)
    if canon == "tnd":
        cu_seqlens = inputs["cu_seqlens"]
        _, heads, _ = x.shape
        for seq_idx in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[seq_idx], cu_seqlens[seq_idx + 1]
            num_chunks = (eos - bos + chunk_size - 1) // chunk_size
            for head in range(heads):
                for chunk in range(num_chunks):
                    start = bos + chunk * chunk_size
                    end = min(start + chunk_size, eos)
                    actual = end - start
                    result[start:end, head, :actual] = _inv_chunk(x[start:end, head, :actual])
    else:
        batch, tokens, heads, _ = x.shape
        num_chunks = (tokens + chunk_size - 1) // chunk_size
        for batch_idx in range(batch):
            for head in range(heads):
                for chunk in range(num_chunks):
                    start = chunk * chunk_size
                    end = min(start + chunk_size, tokens)
                    actual = end - start
                    result[batch_idx, start:end, head, :actual] = _inv_chunk(
                        x[batch_idx, start:end, head, :actual]
                    )
    out = _from_canonical_x(result.to(torch.float32), layout)
    return _mask_output(out, inputs, chunk_size)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    out = ascendc.solve_tri(
        inputs["x"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        layout=inputs["layout"],
    )
    torch.npu.synchronize()
    return _mask_output(out, inputs, int(spec["chunk_size"])).detach().cpu()


@register("executor_solve_tri")
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
            raise RuntimeError(f"solve_tri 仅支持本机 NPU DUT 与 CPU golden，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
