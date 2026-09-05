"""在 NPU 上直接调用开发期 Stage 接口，排查 ABI 与同步问题。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

from executor_chunk_gdn_bwd_intra import (
    _stage0_ref,
    _stage1_ref,
    _stage2_ref,
    build_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--g-dtype", default="fp32", choices=("bf16", "fp32"))
    parser.add_argument("--beta-dtype", default="fp32", choices=("bf16", "fp32"))
    parser.add_argument("--group", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hk", type=int)
    parser.add_argument("--hv", type=int)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--use-exp2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    hk = args.hk or (4 if args.group in {1, 2} else 2 if args.group == 3 else 1)
    hv = args.hv or hk * args.group
    if hv != hk * args.group:
        parser.error("--hv must equal --hk * --group")
    spec = {
        "dtype": args.dtype,
        "g_dtype": args.g_dtype,
        "beta_dtype": args.beta_dtype,
        "B": args.batch,
        "HK": hk,
        "HV": hv,
        "T": args.tokens,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": args.use_exp2,
        "stage": args.stage,
        "seed": args.seed,
    }
    inputs = build_inputs(spec, torch.device("npu"))

    from fla_npu.ops import ascendc

    watchdog = subprocess.Popen(
        ["sh", "-c", 'sleep 60; kill -KILL "$1"', "operator-watchdog", str(os.getpid())]
    )
    try:
        print("operator_call_begin", flush=True)
        outputs = ascendc.chunk_gdn_bwd_intra(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A"],
            inputs["d_o"],
            inputs["scale"],
            inputs["chunk_size"],
            use_exp2=inputs["use_exp2"],
            stage=inputs["stage"],
        )
        print("operator_call_returned", flush=True)
        torch.npu.synchronize()
        print("operator_synchronize_done", flush=True)
    finally:
        watchdog.terminate()
        watchdog.wait()
    if args.stage == 0:
        visible = (outputs[0][..., : inputs["chunk_size"]],)
    elif args.stage == 1:
        visible = tuple(output[..., : inputs["chunk_size"]] for output in outputs)
    else:
        visible = outputs
    stats = [
        {
            "shape": tuple(output.shape),
            "finite": bool(torch.isfinite(output.float()).all().item()),
            "abs_max": float(output.float().abs().max().item()),
        }
        for output in visible
    ]
    print(stats)
    if args.compare:
        cpu_inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
        ref_fn = (_stage0_ref, _stage1_ref, _stage2_ref)[args.stage]
        refs = ref_fn(cpu_inputs)
        if isinstance(refs, torch.Tensor):
            refs = (refs,)
        for index, (output, ref) in enumerate(zip(visible, refs)):
            actual = output.float().cpu()
            expected = ref.float()
            error = (actual - expected).abs()
            denom = expected.abs().clamp_min(1e-8)
            flat_index = int(error.argmax().item())
            max_index = []
            for size in reversed(error.shape):
                max_index.append(flat_index % size)
                flat_index //= size
            max_index = tuple(reversed(max_index))
            print(
                {
                    "output": index,
                    "max_abs": float(error.max().item()),
                    "max_index": max_index,
                    "actual_at_max": float(actual[max_index].item()),
                    "expected_at_max": float(expected[max_index].item()),
                    "mean_abs": float(error.mean().item()),
                    "max_rel": float((error / denom).max().item()),
                    "allclose_5e-3": bool(
                        torch.allclose(actual, expected, rtol=5e-3, atol=5e-3)
                    ),
                }
            )
            if args.stage == 2:
                print(
                    {
                        "output": index,
                        "head_max_abs": error.amax(dim=(0, 2, 3)).tolist(),
                        "chunk_max_abs": [
                            float(error[..., start : start + 64, :].max().item())
                            for start in range(0, args.tokens, 64)
                        ],
                    }
                )
        if args.stage != 1:
            return
        alternate_spec = dict(spec, use_exp2=not spec["use_exp2"])
        alternate_inputs = build_inputs(
            alternate_spec, torch.device("cpu"), high_precision=True
        )
        alternate_w = _stage1_ref(alternate_inputs)[0].float()
        alternate_error = (visible[0].float().cpu() - alternate_w).abs()
        no_gate_error = (visible[0].float().cpu() - visible[1].float().cpu()).abs()
        print(
            {
                "output": 0,
                "alternate_gate": "exp2" if alternate_spec["use_exp2"] else "exp",
                "max_abs": float(alternate_error.max().item()),
                "mean_abs": float(alternate_error.mean().item()),
                "vs_a_beta_max_abs": float(no_gate_error.max().item()),
                "vs_a_beta_mean_abs": float(no_gate_error.mean().item()),
            }
        )
        columns = [0, 15, 31, 32, 47, 63]
        actual_w = visible[0].float().cpu()[0, 0]
        actual_a_beta = visible[1].float().cpu()[0, 0]
        gate = torch.exp2 if spec["use_exp2"] else torch.exp
        expected_gate = gate(cpu_inputs["g"].float()[0, 0])
        for row in (0, 64, 128):
            if row >= args.tokens:
                continue
            valid_columns = [column for column in columns if column < min(64, args.tokens - row)]
            ratio = actual_w[row, valid_columns] / actual_a_beta[row, valid_columns]
            print(
                {
                    "row": row,
                    "columns": valid_columns,
                    "actual_w_over_a_beta": ratio.tolist(),
                    "expected_gate": expected_gate[
                        row + torch.tensor(valid_columns)
                    ].tolist(),
                }
            )


if __name__ == "__main__":
    main()
