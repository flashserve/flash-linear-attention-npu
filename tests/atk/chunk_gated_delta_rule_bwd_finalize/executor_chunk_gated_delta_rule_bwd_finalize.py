"""ChunkGatedDeltaRuleBwdFinalize ATK executor and CPU reference entry."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

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
