#!/usr/bin/env python3
"""Run 730 recurrent_gated_delta_rule state-layout workloads for msprof.

This script deliberately does not measure latency, compare layouts, or write
result files.  Use ``--state-layout`` to run continuous and
non-contiguous state in separate msprof processes.  Inputs are generated from
fixed CPU seeds, so matching cases have identical logical values across
layouts and invocations.

The non-contiguous state is selected from storage shaped
``(BlockNum, Nv, Dv * Dk + padding)``.  Removing a prefix from the last storage
dimension and splitting it into ``(Dv, Dk)`` gives the logical shape
``(BlockNum, Nv, Dv, Dk)`` with padded BlockNum/Nv strides.  The inner
``(Dv, Dk)`` matrix stays dense (``stride2 == Dk`` and ``stride3 == 1``), which
matches the currently safe state layout supported by PR #184.
"""

from __future__ import annotations

import argparse
import importlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F


STATE_HEAD_PREFIX_PADDING = 16
SUPPORTED_STATE_LAYOUTS = ("continuous", "noncontiguous")


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    nk: int
    nv: int
    dv: int
    use_g: bool
    use_gk: bool
    accepted_tokens: Optional[tuple[int, ...]] = None
    permute_state_indices: bool = False
    seed: int = 42
    dk: int = 128

    @property
    def batch_size(self) -> int:
        return len(self.lengths)

    @property
    def token_count(self) -> int:
        return sum(self.lengths)


HEAD_CONFIGS = (
    # (1, 1),
    # (2, 4),
    (4, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 32),
    (48, 96),
    (32, 96),
)  # (Nk, Nv) pairs


# Representative fixed values from test_performance.py:
# B in {1, 4, 16, 64, 128}, both g and gk, Nv in {1, 2, 8, 16}, and Dv 128/256.
FIXED_CASES = (
    Case(
        name="b1_l4_g_nv8_dv128",
        lengths=(4,),
        nk=4,
        nv=8,
        dv=128,
        use_g=True,
        use_gk=False,
        seed=300,
    ),
    Case(
        name="b16_l8_g_nv8_dv128",
        lengths=(8,) * 16,
        nk=4,
        nv=8,
        dv=128,
        use_g=True,
        use_gk=False,
        seed=301,
    ),
    Case(
        name="b16_l8_gk_nv16_dv128",
        lengths=(8,) * 16,
        nk=8,
        nv=16,
        dv=128,
        use_g=False,
        use_gk=True,
        permute_state_indices=True,
        seed=303,
    ),
    Case(
        name="b4_l2_accepted_g_nv8_dv256",
        lengths=(2,) * 4,
        nk=4,
        nv=8,
        dv=256,
        use_g=True,
        use_gk=False,
        accepted_tokens=(2, 1, 2, 1),
        seed=304,
    ),
    Case(
        name="b64_mixed_accepted_g_nv2_dv128",
        lengths=(1, 2) * 32,
        nk=2,
        nv=2,
        dv=128,
        use_g=True,
        use_gk=False,
        accepted_tokens=(1, 2) * 32,
        permute_state_indices=True,
        seed=96,
    ),
    Case(
        name="b128_l1_accepted_gk_nv1_dv128",
        lengths=(1,) * 128,
        nk=1,
        nv=1,
        dv=128,
        use_g=False,
        use_gk=True,
        accepted_tokens=(1,) * 128,
        permute_state_indices=True,
        seed=97,
    ),
)


def make_random_cases(
    count: int = 100, seed: int = 20260728
) -> tuple[Case, ...]:
    """Create 100 reproducible random state-layout workloads by default."""

    rng = random.Random(seed)
    cases = []
    for index in range(count):
        batch_size = rng.choice((1, 2, 4, 8))
        lengths = tuple(rng.randint(1, 8) for _ in range(batch_size))
        nk, nv = rng.choice(HEAD_CONFIGS)
        accepted_tokens = tuple(rng.randint(1, length) for length in lengths)
        permute_state_indices = rng.choice((False, True))
        dv = rng.choice((128, 256))
        use_g = bool(rng.getrandbits(1))
        cases.append(
            Case(
                name=f"random_{index:02d}",
                lengths=lengths,
                nk=nk,
                nv=nv,
                dv=dv,
                use_g=use_g,
                use_gk=not use_g,
                accepted_tokens=accepted_tokens,
                permute_state_indices=permute_state_indices,
                seed=seed + index,
            )
        )
    return tuple(cases)


def make_large_batch_cases(
    count: int = 100, seed: int = 20260729
) -> tuple[Case, ...]:
    """Create 100 reproducible cases with 128 < B < 256 by default."""

    rng = random.Random(seed)
    cases = []
    for index in range(count):
        batch_size = rng.randint(128, 256)
        lengths = tuple(rng.randint(1, 8) for _ in range(batch_size))
        accepted_tokens = tuple(rng.randint(1, length) for length in lengths)
        use_g = bool(index % 2)
        cases.append(
            Case(
                name=f"large_{index:02d}_b{batch_size}",
                lengths=lengths,
                nk=1,
                nv=1,
                dv=128,
                use_g=use_g,
                use_gk=not use_g,
                accepted_tokens=accepted_tokens,
                permute_state_indices=bool(index % 2),
                seed=seed + index,
            )
        )
    return tuple(cases)


RANDOM_CASES = make_random_cases()
LARGE_BATCH_CASES = make_large_batch_cases()
CASES = (*FIXED_CASES, *RANDOM_CASES, *LARGE_BATCH_CASES)


@dataclass
class StateLayout:
    name: str
    state: torch.Tensor
    storage: torch.Tensor


def validate_case(case: Case) -> None:
    if case.use_g == case.use_gk:
        raise ValueError(f"{case.name}: exactly one of use_g/use_gk must be true")
    if case.token_count <= 0:
        raise ValueError(f"{case.name}: token count must be positive")
    if case.dk != 128 or case.dv not in (128, 256):
        raise ValueError(f"{case.name}: expected Dk=128 and Dv in {{128, 256}}")
    if case.nv < case.nk or case.nv % case.nk:
        raise ValueError(f"{case.name}: expected Nv >= Nk and Nv % Nk == 0")
    if case.accepted_tokens is not None:
        if len(case.accepted_tokens) != case.batch_size:
            raise ValueError(f"{case.name}: accepted_tokens must have B entries")
        if any(
            accepted < 1 or accepted > length
            for accepted, length in zip(case.accepted_tokens, case.lengths)
        ):
            raise ValueError(f"{case.name}: accepted_tokens must be in [1, Li]")


def generate_inputs(case: Case) -> dict[str, Any]:
    """Generate the deterministic CPU inputs shared by all state layouts."""

    validate_case(case)
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    token_count = case.token_count

    query = F.normalize(
        torch.rand(
            (token_count, case.nk, case.dk),
            generator=generator,
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    key = F.normalize(
        torch.rand(
            (token_count, case.nk, case.dk),
            generator=generator,
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    value = torch.rand(
        (token_count, case.nv, case.dv),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    beta = torch.rand(
        (token_count, case.nv),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    state = torch.rand(
        (token_count, case.nv, case.dv, case.dk),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    if case.permute_state_indices:
        ssm_state_indices = torch.randperm(
            token_count, generator=generator
        ).to(torch.int32)
    else:
        ssm_state_indices = torch.arange(token_count, dtype=torch.int32)

    g = (
        -torch.rand(
            (token_count, case.nv),
            generator=generator,
            dtype=torch.float32,
        )
        if case.use_g
        else None
    )
    gk = (
        -torch.rand(
            (token_count, case.nv, case.dk),
            generator=generator,
            dtype=torch.float32,
        )
        if case.use_gk
        else None
    )

    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "state": state,
        "actual_seq_lengths": torch.tensor(
            (0, *case.lengths), dtype=torch.int32
        ),
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": (
            torch.tensor(case.accepted_tokens, dtype=torch.int32)
            if case.accepted_tokens is not None
            else None
        ),
        "g": g,
        "gk": gk,
        "scale": case.dk**-0.5,
    }


def resolve_operator() -> tuple[Callable[..., Any], str]:
    """Prefer the requested fla_npu.ops entry point, with API fallback."""

    ops_module = importlib.import_module("fla_npu.ops")
    direct_operator = getattr(ops_module, "recurrent_gated_delta_rule", None)
    if callable(direct_operator):
        return direct_operator, "fla_npu.ops.recurrent_gated_delta_rule"

    ascendc_module = importlib.import_module("fla_npu.ops.ascendc")
    ascendc_operator = getattr(ascendc_module, "recurrent_gated_delta_rule")
    return ascendc_operator, "fla_npu.ops.ascendc.recurrent_gated_delta_rule"


def make_state_layout(
    pristine_cpu_state: torch.Tensor,
    layout: str,
    device: torch.device,
) -> StateLayout:
    """Build the requested layout without an NPU-to-NPU ``copy_``."""

    if pristine_cpu_state.device.type != "cpu":
        raise ValueError("pristine state must remain on CPU")

    shape = tuple(pristine_cpu_state.shape)
    if layout == "continuous":
        storage = pristine_cpu_state.to(device)
        state = storage
    elif layout == "noncontiguous":
        # Prefix padding on every (Dv, Dk) head makes stride0/stride1
        # non-dense while retaining stride2 == Dk and stride3 == 1.  Fill the
        # view on CPU and transfer its contiguous backing storage so setup does
        # not require aclnnInplaceCopy/TensorMove on NPU.
        head_numel = shape[2] * shape[3]
        padded_head_numel = head_numel + STATE_HEAD_PREFIX_PADDING
        padded_strides = (
            shape[1] * padded_head_numel,
            padded_head_numel,
            shape[3],
            1,
        )
        cpu_storage = torch.empty(
            (shape[0], shape[1], padded_head_numel),
            dtype=pristine_cpu_state.dtype,
            device="cpu",
        )
        # ``slice().view()`` canonicalizes strides for size-one dimensions,
        # so Nv=1 would lose its padded stride even though the same storage
        # layout is retained.  Set all strides explicitly for every shape.
        cpu_state = cpu_storage.as_strided(
            shape, padded_strides, STATE_HEAD_PREFIX_PADDING
        )
        cpu_state.copy_(pristine_cpu_state)
        storage = cpu_storage.to(device)
        state = storage.as_strided(
            shape, padded_strides, STATE_HEAD_PREFIX_PADDING
        )
    else:
        raise ValueError(f"Unknown state layout: {layout}")

    if tuple(state.shape) != shape:
        raise AssertionError(f"{layout}: state shape changed")
    if layout == "continuous" and not state.is_contiguous():
        raise AssertionError("continuous state is unexpectedly non-contiguous")
    if layout == "noncontiguous":
        head_numel = shape[2] * shape[3]
        padded_head_numel = head_numel + STATE_HEAD_PREFIX_PADDING
        expected_strides = (
            shape[1] * padded_head_numel,
            padded_head_numel,
            shape[3],
            1,
        )
        if tuple(state.stride()) != expected_strides:
            raise AssertionError(
                "non-contiguous state must pad stride0/stride1 and keep the "
                "inner (Dv, Dk) matrix dense"
            )
        if (shape[0] > 1 or shape[1] > 1) and state.is_contiguous():
            raise AssertionError("non-contiguous state construction became contiguous")
        if state.storage_offset() != STATE_HEAD_PREFIX_PADDING:
            raise AssertionError("non-contiguous state has an unexpected offset")
    return StateLayout(layout, state, storage)


def move_non_state_inputs(
    inputs: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    device_inputs: dict[str, Any] = {"scale": inputs["scale"]}
    for name, value in inputs.items():
        if name in ("scale", "state"):
            continue
        device_inputs[name] = (
            value.to(device) if isinstance(value, torch.Tensor) else value
        )
    return device_inputs


def invoke_operator(
    operator: Callable[..., Any],
    device_inputs: dict[str, Any],
    state: torch.Tensor,
) -> Any:
    return operator(
        device_inputs["query"],
        device_inputs["key"],
        device_inputs["value"],
        state,
        beta=device_inputs["beta"],
        scale=device_inputs["scale"],
        actual_seq_lengths=device_inputs["actual_seq_lengths"],
        ssm_state_indices=device_inputs["ssm_state_indices"],
        num_accepted_tokens=device_inputs["num_accepted_tokens"],
        g=device_inputs["g"],
        gk=device_inputs["gk"],
    )


def selected_layouts(requested_layout: str) -> tuple[str, ...]:
    if requested_layout == "all":
        return SUPPORTED_STATE_LAYOUTS
    return (requested_layout,)


@torch.inference_mode()
def run_case(
    case: Case,
    layouts: tuple[str, ...],
    operator: Callable[..., Any],
    torch_npu_module: Any,
    device: torch.device,
    repeat: int,
) -> None:
    # The same non-state device tensors are reused for both 730 layouts.  State
    # layouts are prepared on CPU and transferred with their backing storage.
    inputs = generate_inputs(case)
    device_inputs = move_non_state_inputs(inputs, device)
    pristine_cpu_state = inputs["state"]

    for layout_name in layouts:
        layout = make_state_layout(pristine_cpu_state, layout_name, device)
        print(
            f"WORKLOAD_BEGIN case={case.name} layout={layout.name} "
            f"shape={tuple(layout.state.shape)} stride={layout.state.stride()} "
            f"storage_offset={layout.state.storage_offset()} repeat={repeat}",
            flush=True,
        )
        result = None
        for iteration in range(repeat):
            if iteration:
                # recurrent_gated_delta_rule mutates state.  Restore the same
                # logical initial values without invoking NPU TensorMove.
                del result
                layout = make_state_layout(
                    pristine_cpu_state, layout_name, device
                )
            torch_npu_module.npu.synchronize()
            print(
                f"OP_BEGIN case={case.name} layout={layout.name} "
                f"iteration={iteration}",
                flush=True,
            )
            result = invoke_operator(operator, device_inputs, layout.state)
            torch_npu_module.npu.synchronize()
            print(
                f"OP_END case={case.name} layout={layout.name} "
                f"iteration={iteration}",
                flush=True,
            )
        del result
        print(
            f"WORKLOAD_END case={case.name} layout={layout.name}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--state-layout",
        choices=("all", "continuous", "noncontiguous"),
        default="all",
        help="Layout to execute; 'all' runs both layouts (default: all)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Operator invocations per case and layout (default: 1)",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only this named case; may be repeated",
    )
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0")

    selected_cases = list(CASES)
    if args.case:
        requested_cases = set(args.case)
        selected_cases = [
            case for case in CASES if case.name in requested_cases
        ]
        missing_cases = requested_cases - {
            case.name for case in selected_cases
        }
        if missing_cases:
            raise ValueError(f"Unknown cases: {sorted(missing_cases)}")
    if args.list_cases:
        for case in selected_cases:
            print(case.name)
        return

    layouts = selected_layouts(args.state_layout)

    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError(
            "This workload must run in an environment with torch_npu"
        ) from error

    operator, operator_entrypoint = resolve_operator()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must be an NPU device such as npu:0")
    torch_npu.npu.set_device(device)

    print(
        f"device={args.device} operator={operator_entrypoint} "
        f"layouts={','.join(layouts)} "
        f"repeat={args.repeat}",
        flush=True,
    )
    for case in selected_cases:
        print(
            f"CASE case={case.name} B={case.batch_size} T={case.token_count} "
            f"Nk={case.nk} Nv={case.nv} Dk={case.dk} Dv={case.dv} "
            f"gate={'g' if case.use_g else 'gk'} seed={case.seed}",
            flush=True,
        )
        run_case(
            case,
            layouts,
            operator,
            torch_npu,
            device,
            args.repeat,
        )

    print("All requested workloads completed.", flush=True)


if __name__ == "__main__":
    main()
