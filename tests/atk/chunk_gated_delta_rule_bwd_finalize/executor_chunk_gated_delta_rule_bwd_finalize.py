"""ChunkGatedDeltaRuleBwdFinalize ATK executor and CPU reference entry."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _case_spec, _finite_tuple


OP_NAME = "chunk_gated_delta_rule_bwd_finalize"
CHUNK_SIZE = 64
SCALE = 0.08838
_TENSOR_NAMES = (
    "q", "k", "v", "v_new", "do", "g", "beta", "beta_raw",
    "du", "h", "dh", "a",
)
_REFERENCE_FILE = Path(__file__).resolve().parent / "scripts" / "chunk_gated_delta_rule_bwd_finalize_cpu.py"
_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "atk_chunk_gated_delta_rule_bwd_finalize_cpu", _REFERENCE_FILE)
if _REFERENCE_SPEC is None or _REFERENCE_SPEC.loader is None:
    raise ImportError(f"Unable to load CPU reference: {_REFERENCE_FILE}")
_REFERENCE = importlib.util.module_from_spec(_REFERENCE_SPEC)
_REFERENCE_SPEC.loader.exec_module(_REFERENCE)


def _make_varlen(total_tokens: int):
    rng = random.Random(20260822)
    cuts = sorted(rng.sample(range(1, total_tokens), 63))
    cu_seqlens = [0] + cuts + [total_tokens]
    chunk_indices = []
    for seq_idx in range(64):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        for chunk_idx in range((seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE):
            chunk_indices.extend((seq_idx, chunk_idx))
    return cu_seqlens, chunk_indices


def _prepared(input_data: InputDataset, spec):
    values = [input_data.kwargs[name] for name in _TENSOR_NAMES]
    q = values[0]
    batch, key_heads, total_tokens, dim = q.shape
    value_heads = values[2].shape[1]
    if bool(spec.get("varlen", False)):
        cu_seqlens, chunk_indices = _make_varlen(total_tokens)
        state_chunk_num = len(chunk_indices) // 2
    else:
        cu_seqlens, chunk_indices = None, None
        state_chunk_num = (total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    metadata = {
        "name": str(spec["case_key"]),
        "batch": batch,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "total_tokens": total_tokens,
        "variable": bool(spec.get("varlen", False)),
        "state_chunk_num": state_chunk_num,
        "task_num": state_chunk_num * batch,
        "chunk_size": int(spec["chunk_size"]),
        "dim": dim,
        "seed": int(spec["seed"]),
    }
    device = q.device
    cu_tensor = (
        torch.tensor(cu_seqlens, dtype=torch.int64, device=device)
        if cu_seqlens is not None else None
    )
    chunk_tensor = (
        torch.tensor(chunk_indices, dtype=torch.int64, device=device)
        if chunk_indices is not None else None
    )
    return (*values, cu_tensor, chunk_tensor, None, metadata)


def _cpu_prepared(input_data: InputDataset, spec):
    prepared = list(_prepared(input_data, spec))
    scalar_dtype = (
        torch.float32 if spec.get("scalar_dtype") == "fp32" else torch.bfloat16
    )
    # ATK raises CPU inputs to its benchmark dtype. Recast every operator input
    # at the ABI boundary so the reference follows the same BF16/FP32 stages as NPU.
    for index in (0, 1, 2, 3, 4, 8, 9, 10, 11):
        prepared[index] = prepared[index].to(torch.bfloat16)
    for index in (5, 6, 7):
        prepared[index] = prepared[index].to(scalar_dtype)
    return tuple(prepared)


def _validate_outputs(outputs, prepared) -> None:
    q, _, v, _, _, g, beta = prepared[:7]
    expected_shapes = (q.shape, q.shape, v.shape, beta.shape, g.shape)
    expected_dtypes = (q.dtype, q.dtype, v.dtype, beta.dtype, g.dtype)
    if len(outputs) != len(expected_shapes):
        raise RuntimeError(f"{OP_NAME}: expected 5 outputs, got {len(outputs)}")
    for index, (output, shape, dtype) in enumerate(
        zip(outputs, expected_shapes, expected_dtypes)
    ):
        if tuple(output.shape) != tuple(shape):
            raise RuntimeError(
                f"{OP_NAME}: output {index} shape {tuple(output.shape)} != {tuple(shape)}")
        if output.dtype != dtype:
            raise RuntimeError(
                f"{OP_NAME}: output {index} dtype {output.dtype} != {dtype}")


def run_cpu(spec, input_data: InputDataset):
    prepared = _cpu_prepared(input_data, spec)
    (
        q, k, v, v_new, do, g, beta, beta_raw, du, h, dh, a,
        cu_seqlens, chunk_indices, _, _,
    ) = prepared
    use_qk_l2norm = bool(spec["use_qk_l2norm"])
    use_beta_sigmoid = bool(spec["use_beta_sigmoid"])
    q_rstd = (
        torch.rsqrt((q.float() * q.float()).sum(dim=-1) + 1.0e-6)
        if use_qk_l2norm else None
    )
    k_rstd = (
        torch.rsqrt((k.float() * k.float()).sum(dim=-1) + 1.0e-6)
        if use_qk_l2norm else None
    )
    outputs = _REFERENCE.chunk_gated_delta_rule_bwd_finalize_golden(
        q, k, v, v_new, do, du, g, beta, h, dh, a,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_raw=beta_raw if use_beta_sigmoid else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=float(spec.get("scale", SCALE)),
        chunk_size=int(spec.get("chunk_size", CHUNK_SIZE)),
        use_qk_l2_norm_in_kernel=use_qk_l2norm,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid,
        use_gate_in_kernel=False,
        state_v_first=bool(spec["state_v_first"]),
        use_exp2=True,
        nt_first=True,
    )
    _validate_outputs(outputs, prepared)
    return outputs


def run_npu(spec, input_data: InputDataset):
    prepared = _prepared(input_data, spec)
    (
        q, k, v, v_new, do, g, beta, beta_raw, du, h, dh, a,
        cu_seqlens, chunk_indices, _, _,
    ) = prepared
    use_qk_l2norm = bool(spec["use_qk_l2norm"])
    use_beta_sigmoid = bool(spec["use_beta_sigmoid"])
    q_rstd = (
        torch.rsqrt((q.float() * q.float()).sum(dim=-1) + 1.0e-6)
        if use_qk_l2norm else None
    )
    k_rstd = (
        torch.rsqrt((k.float() * k.float()).sum(dim=-1) + 1.0e-6)
        if use_qk_l2norm else None
    )

    from fla_npu.ops import ascendc

    outputs = ascendc.npu_chunk_gated_delta_rule_bwd_finalize(
        q, k, v, v_new, do, du, g, beta, h, dh, a,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_raw=beta_raw if use_beta_sigmoid else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=float(spec.get("scale", SCALE)),
        chunk_size=int(spec.get("chunk_size", CHUNK_SIZE)),
        use_qk_l2_norm_in_kernel=use_qk_l2norm,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid,
        use_gate_in_kernel=False,
        state_v_first=bool(spec["state_v_first"]),
        use_exp2=True,
    )
    torch.npu.synchronize()
    print(f"NPU operator completed: {spec['case_key']}", flush=True)
    _validate_outputs(outputs, prepared)
    return outputs


@register("executor_chunk_gated_delta_rule_bwd_finalize")
class FunctionApi(BaseApi):
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


# A2/A3 单独注册；上面的 A5 入口和 CPU 标杆保持不变。
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import arch22_cpu as _ARCH22


def _arch22_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy() if value.ndim else value.item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _arch22_runtime_identity(*, require_tiling=False, refresh=False):
    from arch22_identity import _arch22_runtime_identity as identity
    return identity(require_tiling=require_tiling, refresh=refresh)


def _arch22_input_identity(actual):
    digest = hashlib.sha256()
    for name, value in sorted(actual.items()):
        digest.update(name.encode())
        if isinstance(value, np.ndarray):
            digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def _arch22_call_npu(actual, device):
    from fla_npu.ops import ascendc
    _arch22_runtime_identity()
    kw = {}
    for name, value in actual.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
            dtype = torch.bfloat16 if name in ("q", "k", "v", "v_new", "do", "du", "h", "dh", "a") else torch.float32
            if name in ("g", "beta", "beta_raw") and actual["_gate_dtype"] == "bf16":
                dtype = torch.bfloat16
            kw[name] = tensor.to(dtype=dtype, device=device)
        elif not name.startswith("_"):
            kw[name] = value
    # metadata 本来就是公开 int64 list 参数，由 wrapper 建 aclIntArray。
    print(f"[finalize-arch22-key-begin] case={actual['_case_id']}", flush=True)
    outputs = ascendc.npu_chunk_gated_delta_rule_bwd_finalize(**kw)
    torch.npu.synchronize(device)
    print(f"[finalize-arch22-key-end] case={actual['_case_id']}", flush=True)
    if len(outputs) != 5:
        raise RuntimeError(f"A2/A3 actual output count={len(outputs)}; expected=5")
    expected_shapes = (kw["q"].shape, kw["k"].shape, kw["v"].shape, kw["beta"].shape, kw["g"].shape)
    expected_dtypes = (torch.bfloat16, torch.bfloat16, torch.bfloat16, kw["beta"].dtype, kw["g"].dtype)
    for name, output, shape, dtype in zip(_ARCH22.OUTPUT_NAMES, outputs, expected_shapes, expected_dtypes):
        if output.shape != shape or output.dtype != dtype:
            raise RuntimeError(f"A2/A3 {name}: actual={output.shape}/{output.dtype}; expected={shape}/{dtype}")
    return outputs


@register("executor_chunk_gated_delta_rule_bwd_finalize_arch22")
class Arch22FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.case_id = int(task_result.case_config.id)
        self.task_name = str(task_result.name or "")
        self.is_golden = bool(task_result.is_benchmark_task)
        self.gate_dtype = str(next(value.dtype for value in task_result.case_config.inputs if value.name == "g")).lower().split(".")[-1]
        if self.gate_dtype not in ("bf16", "fp32"):
            raise RuntimeError(f"A2/A3 unknown declared g dtype: {self.gate_dtype}")
        if str(self.device) in ("npu", "pyaclnn"):
            self.role = "dut"
        elif str(self.device) == "cpu" and self.is_golden:
            self.role = "fp64"
        elif str(self.device) == "cpu" and self.task_name == "cpu_lowp":
            self.role = "lowp"
        else:
            raise RuntimeError(f"A2/A3 unknown ATK role: {self.device}/{self.task_name}/{self.is_golden}")
        self.initialized = False

    def init_by_input_data(self, input_data: InputDataset):
        values = {name: _arch22_value(value) for name, value in input_data.kwargs.items()}
        self.case, self.data, self.actual = _ARCH22.prepare_inputs(values, str(self.gate_dtype))
        self.input_sha256 = _arch22_input_identity(self.actual)
        self.actual["_gate_dtype"] = str(self.gate_dtype)
        self.actual["_case_id"] = self.case_id
        self.device_arg = input_data.kwargs["q"].device
        self.initialized = True

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if not self.initialized:
            self.init_by_input_data(input_data)
        begin = time.monotonic()
        if self.role == "dut":
            values = _arch22_call_npu(self.actual, self.device_arg)
        else:
            fn = _ARCH22.finalize_cpu_fp64 if self.role == "fp64" else _ARCH22.finalize_cpu_lowp
            result = fn(self.case, self.data)
            values = [torch.from_numpy(np.ascontiguousarray(result[name])) for name in _ARCH22.OUTPUT_NAMES]
            if self.role == "lowp":
                # NumPy 用 FP32 表示已经舍入的 BF16 值；交 ATK 前恢复公开 ABI dtype。
                gate_dtype = torch.bfloat16 if self.gate_dtype == "bf16" else torch.float32
                dtypes = (torch.bfloat16, torch.bfloat16, torch.bfloat16, gate_dtype, gate_dtype)
                values = [value.to(dtype) for value, dtype in zip(values, dtypes)]
        outputs, metadata = [], {}
        for name, value in zip(_ARCH22.OUTPUT_NAMES, values):
            value = value.detach().cpu()
            if not torch.isfinite(value).all().item():
                raise RuntimeError(f"A2/A3 {self.role}/{name}: nonfinite output")
            # NumPy 不支持 BF16；只把独立临时副本用于摘要，保留生产角色的 dtype 记录。
            hash_value = value.float() if value.dtype == torch.bfloat16 else value
            # ATK 26.9.8 的前置 dtype 检查要求与真 FP64 golden 使用同一表示。
            # BF16/FP32 -> FP64 无损；判据由每条 case 的公开 ABI dtype override 决定。
            comparison_value = value.to(torch.float64)
            metadata[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                  sha256=hashlib.sha256(hash_value.contiguous().numpy().tobytes()).hexdigest(),
                                  comparison_dtype=str(comparison_value.dtype),
                                  comparison_sha256=hashlib.sha256(comparison_value.contiguous().numpy().tobytes()).hexdigest())
            outputs.append(comparison_value)
        record = dict(case_id=self.case_id, role=self.role, input_sha256=self.input_sha256,
                      seconds=time.monotonic() - begin, outputs=metadata,
                      reference_sha256=_ARCH22.REFERENCE_ARTIFACT_SHA256,
                      cpu_lowp_precision=_ARCH22.cpu_lowp_precision_ledger() if self.role == "lowp" else None)
        if self.role == "dut":
            record["runtime"] = _arch22_runtime_identity(require_tiling=True, refresh=True)
            record["measured_identity"] = record["runtime"]["measured_identity"]
        folder = os.getenv("FINALIZE_CPU_EVIDENCE")
        if folder:
            target = Path(folder) / f"{self.case_id:04d}"
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{self.role}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[finalize-arch22] case={self.case_id} role={self.role} input_sha256={self.input_sha256}", flush=True)
        return tuple(outputs) if with_output else None
