#!/usr/bin/env python3
"""Validate model-parallel GDR shard shapes through the standalone adapter.

The cases model the local GDR tensors seen after TP, SP, or sequence-packed CP
partitioning. Run a case on one process to emulate a rank, or use torchrun with
the case's declared world size to exercise concurrent multi-NPU submission.
Results are emitted as one JSON object from rank 0.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401

import chunk_gated_delta_rule_function as gdr_adapter
from fla_npu.ops.ascendc import solve_tri as ascendc_solve_tri


ChunkGatedDeltaRuleFunction = gdr_adapter.ChunkGatedDeltaRuleFunction


@dataclass(frozen=True)
class ModelCase:
    name: str
    model: str
    scale: str
    parallel: str
    world_size: int
    mode: str
    global_tokens: int
    packed: bool
    value_dim: int = 128
    continuation: bool = False
    output_final_state: bool = False
    global_key_heads: int = 16
    global_value_heads: int = 32
    key_dim: int = 128
    chunk_size: int = 64


CASES = {
    case.name: case
    for case in (
        ModelCase(
            "qwen35_4b_tp4_train",
            "Qwen3.5-4B",
            "3.5B-class",
            "TP",
            4,
            "train",
            32768,
            True,
        ),
        ModelCase(
            "qwen35_4b_sp4_train_local_shard",
            "Qwen3.5-4B",
            "3.5B-class",
            "SP",
            4,
            "train",
            32768,
            False,
        ),
        ModelCase(
            "qwen35_35b_tp4_train",
            "Qwen3.5-35B-A3B",
            "35B",
            "TP",
            4,
            "train",
            32768,
            True,
        ),
        ModelCase(
            "qwen35_35b_cp4_train_pack",
            "Qwen3.5-35B-A3B",
            "35B",
            "CP",
            4,
            "train",
            32768,
            True,
        ),
        ModelCase(
            "qwen_next_tp4_infer_prefill",
            "Qwen3-Next-80B-A3B",
            "3B-active",
            "TP",
            4,
            "infer",
            32768,
            True,
        ),
        ModelCase(
            "qwen_next_sp4_infer_continuation",
            "Qwen3-Next-80B-A3B",
            "3B-active",
            "SP",
            4,
            "infer",
            32768,
            False,
            continuation=True,
            output_final_state=True,
        ),
        ModelCase(
            "qwen_next_cp4_infer_pack",
            "Qwen3-Next-80B-A3B",
            "3B-active",
            "CP",
            4,
            "infer",
            32768,
            True,
        ),
        ModelCase(
            "gva_v256_tp4_train",
            "GVA-generalization",
            "synthetic",
            "TP",
            4,
            "train",
            16384,
            True,
            value_dim=256,
        ),
    )
}


def _local_shape(case: ModelCase) -> tuple[int, int, int]:
    if case.parallel == "TP":
        if (
            case.global_key_heads % case.world_size
            or case.global_value_heads % case.world_size
        ):
            raise ValueError(f"{case.name}: heads are not divisible by TP size")
        return (
            case.global_tokens,
            case.global_key_heads // case.world_size,
            case.global_value_heads // case.world_size,
        )
    if case.global_tokens % case.world_size:
        raise ValueError(f"{case.name}: tokens are not divisible by parallel size")
    return (
        case.global_tokens // case.world_size,
        case.global_key_heads,
        case.global_value_heads,
    )


def _packed_offsets(tokens: int) -> list[int]:
    unit = tokens // 8
    if unit == 0:
        return [0, tokens]
    offsets = [0, unit, 3 * unit, 6 * unit, tokens]
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        return [0, tokens]
    return offsets


def _make_inputs(
    case: ModelCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    tuple[torch.Tensor, ...],
    Optional[torch.Tensor],
    Optional[list[int]],
    Optional[torch.Tensor],
]:
    tokens, key_heads, value_heads = _local_shape(case)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260729)

    q = torch.randn(
        1,
        key_heads,
        tokens,
        case.key_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        q.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        1,
        value_heads,
        tokens,
        case.value_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    gate_logits = torch.randn(
        1,
        tokens,
        value_heads,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    beta_logits = torch.randn(
        gate_logits.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    g = F.logsigmoid(gate_logits).to(dtype)
    beta = torch.sigmoid(beta_logits).to(dtype)

    offsets = _packed_offsets(tokens) if case.packed else None
    cu_seqlens = (
        torch.tensor(offsets, dtype=torch.int64, device=device)
        if offsets is not None
        else None
    )
    state_count = len(offsets) - 1 if offsets is not None else 1
    initial_state = (
        torch.randn(
            state_count,
            value_heads,
            case.key_dim,
            case.value_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        ).mul_(0.01)
        if case.continuation
        else None
    )
    return (q, k, v, g, beta), cu_seqlens, offsets, initial_state


def _run_once(
    case: ModelCase,
    base_inputs: tuple[torch.Tensor, ...],
    *,
    cu_seqlens: Optional[torch.Tensor],
    offsets: Optional[list[int]],
    initial_state: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor]]:
    training = case.mode == "train"
    inputs = [
        tensor.detach().clone().requires_grad_(training)
        for tensor in base_inputs
    ]
    output, final_state = ChunkGatedDeltaRuleFunction.apply(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        case.key_dim ** -0.5,
        initial_state,
        case.output_final_state,
        cu_seqlens,
        offsets,
        None,
        None,
        True,
        case.chunk_size,
    )
    retained = [output.detach().clone()]
    if final_state is not None:
        retained.append(final_state.detach().clone())
    if training:
        loss_scale = 1024.0 if output.dtype == torch.float16 else 1.0
        (output.float().mean() * loss_scale).backward()
        gradients = []
        for tensor in inputs:
            if tensor.grad is None:
                raise RuntimeError("a differentiable GDR input has no gradient")
            gradients.append(tensor.grad.detach().clone())
        retained.extend(gradients)
    return output, final_state, retained


def _all_finite(tensors: list[torch.Tensor]) -> torch.Tensor:
    checks = [torch.isfinite(tensor).all() for tensor in tensors]
    return torch.stack(checks).all()


def _init_distributed() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.npu.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("hccl")
    return rank, world_size


def _install_ascendc_solve_reference() -> None:
    """Install the pre-change solve path for controlled A/B validation."""

    def solve_reference(
        A: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices_out: Optional[dict[str, Optional[torch.Tensor]]] = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        A_input = A.to(output_dtype).contiguous()
        if cu_seqlens is None:
            return ascendc_solve_tri(A_input, layout="bsnd")
        if chunk_indices_out is None:
            raise ValueError("AscendC reference requires chunk metadata")
        chunk_indices = chunk_indices_out[str(A.shape[-1])]
        if chunk_indices is None:
            raise ValueError("AscendC reference is missing current chunk metadata")
        offsets = [int(item) for item in cu_seqlens.detach().cpu().tolist()]
        flattened = [
            int(item)
            for item in chunk_indices.detach().cpu().reshape(-1).tolist()
        ]
        return ascendc_solve_tri(
            A_input.squeeze(0),
            cu_seqlens=offsets,
            chunk_indices=flattened,
            layout="tnd",
        ).unsqueeze(0)

    gdr_adapter.triton_solve_tril = solve_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=False)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--determinism-runs", type=int, default=2)
    parser.add_argument(
        "--solve-backend",
        choices=("triton", "ascendc-reference"),
        default="triton",
        help="The reference mode is diagnostic and may synchronize metadata.",
    )
    args = parser.parse_args()
    if args.list:
        for case in CASES.values():
            print(json.dumps(asdict(case), sort_keys=True))
        return 0
    if args.case is None:
        parser.error("--case is required unless --list is used")
    if args.determinism_runs < 2:
        parser.error("--determinism-runs must be at least 2")

    case = CASES[args.case]
    if args.solve_backend == "ascendc-reference":
        _install_ascendc_solve_reference()
    rank, world_size = _init_distributed()
    if world_size not in (1, case.world_size):
        raise ValueError(
            f"{case.name} expects one emulation rank or {case.world_size} "
            f"distributed ranks, got {world_size}"
        )

    device = torch.device(f"npu:{int(os.environ.get('LOCAL_RANK', '0'))}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(20260729)
    torch.npu.manual_seed_all(20260729)
    torch.npu.reset_peak_memory_stats(device)
    base_inputs, cu_seqlens, offsets, initial_state = _make_inputs(
        case,
        device=device,
        dtype=dtype,
    )

    reference: Optional[list[torch.Tensor]] = None
    deterministic = True
    finite = True
    component_finite: dict[str, bool] = {}
    grad_norm = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(args.determinism_runs):
        _, _, retained = _run_once(
            case,
            base_inputs,
            cu_seqlens=cu_seqlens,
            offsets=offsets,
            initial_state=initial_state,
        )
        finite = finite and bool(_all_finite(retained).item())
        labels = ["output"]
        if case.output_final_state:
            labels.append("final_state")
        if case.mode == "train":
            labels.extend(("dq", "dk", "dv", "dg", "dbeta"))
        component_finite = {
            name: bool(torch.isfinite(tensor).all().item())
            for name, tensor in zip(labels, retained)
        }
        if case.mode == "train":
            gradient_start = 1 + int(case.output_final_state)
            grad_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(tensor.float())
                        for tensor in retained[gradient_start:]
                    ]
                )
            )
        if reference is None:
            reference = retained
        else:
            deterministic = deterministic and all(
                torch.equal(left, right)
                for left, right in zip(reference, retained)
            )

    torch.npu.synchronize()
    peak_bytes = torch.npu.max_memory_allocated(device)
    local_pass = finite and deterministic
    status = torch.tensor(
        [int(local_pass), peak_bytes],
        dtype=torch.int64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(status[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(status[1:], op=dist.ReduceOp.MAX)
    result = {
        **asdict(case),
        "dtype": args.dtype,
        "solve_backend": args.solve_backend,
        "local_tokens": _local_shape(case)[0],
        "local_key_heads": _local_shape(case)[1],
        "local_value_heads": _local_shape(case)[2],
        "actual_world_size": world_size,
        "loss_scale": (
            1024.0
            if case.mode == "train" and dtype == torch.float16
            else 1.0
        ),
        "finite": finite,
        "component_finite": component_finite,
        "deterministic": deterministic,
        "pass": bool(status[0].item()),
        "peak_memory_mib": round(status[1].item() / 1024 / 1024, 2),
        "grad_norm": (
            round(float(grad_norm.item()), 8)
            if case.mode == "train"
            else None
        ),
    }
    if rank == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
