"""Profile the PR #20485 window matrix in one msprof collection."""

from __future__ import annotations

import argparse
import math

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import gdn_recoverssm_commit


WINDOW_CASES = (
    (4, 8, 128, 128, 1, torch.float32),
    (4, 8, 128, 128, 2, torch.bfloat16),
    (4, 8, 128, 128, 5, torch.float32),
    (8, 24, 128, 128, 9, torch.bfloat16),
    (1, 3, 128, 128, 5, torch.float32),
)


def run_case(case_index, _hk, hv, key, value, window, state_dtype):
    torch.manual_seed(419)
    batch = max(3, math.ceil(96 / hv)) if hv % 2 == 0 else 3
    pages = 3 * batch + 1
    state_cpu = torch.empty(
        pages, hv, value, key, dtype=state_dtype
    ).normal_(0.0, 0.02)
    replay_cpu = torch.empty(
        pages, hv, window, value + key + 1, dtype=torch.float32
    ).normal_(0.0, 0.03)
    replay_cpu[..., value + key].abs_().add_(0.5).clamp_max_(1.0)
    state = state_cpu.npu()
    replay = replay_cpu.npu()

    table = torch.tensor(
        [
            [[state.data_ptr()], [state.stride(0)]],
            [[replay.data_ptr()], [replay.stride(0)]],
        ],
        dtype=torch.int64,
    ).npu()
    source = torch.arange(1, batch + 1, dtype=torch.int32).npu()
    final = torch.arange(
        batch + 1, 2 * batch + 1, dtype=torch.int32
    ).npu()
    lengths = torch.full((batch,), window, dtype=torch.int32).npu()
    boundary_cpu = torch.zeros(batch, dtype=torch.int32)
    boundary_cpu[1] = 2 * batch + 1
    boundary = boundary_cpu.npu()
    recovery_cpu = torch.zeros(batch, dtype=torch.int32)
    recovery_cpu[1] = max(1, window // 2)
    recovery = recovery_cpu.npu()

    gdn_recoverssm_commit(
        state,
        replay,
        table,
        source,
        lengths,
        final,
        boundary,
        recovery,
        spec_query_len=window,
        align_mode=True,
    )
    torch.npu.synchronize()
    print(
        "CASE_ORDER",
        case_index,
        f"HV={hv}",
        "K=V=128",
        f"WINDOW={window}",
        f"DTYPE={state_dtype}",
        f"BATCH={batch}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", action="append", type=int)
    args = parser.parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("requires torch_npu")
    indices = (
        args.case_index
        if args.case_index is not None
        else range(len(WINDOW_CASES))
    )
    with torch.no_grad():
        for case_index in indices:
            run_case(case_index, *WINDOW_CASES[case_index])
    if args.case_index is None:
        print(
            "CASE_ORDER 5 HV=3 K=33 V=47 WINDOW=4 "
            "DTYPE=torch.float32 UNSUPPORTED",
            flush=True,
        )


if __name__ == "__main__":
    main()
