#!/usr/bin/env python3
"""Cross-device correctness test for recurrent gated delta rule.

The same host-generated ``.pt`` input is consumed by three implementations:

* ``golden`` composes PyTorch CUDA operators in float64.
* ``triton`` runs FLA's CUDA ``fused_recurrent_gated_delta_rule``.
* ``npu`` runs ``fla_npu.ops.ascendc.recurrent_gated_delta_rule``.

NPU execution saves ``output`` and ``final_state`` as two standalone tensor
files.  Copy the output directory to the CUDA host.  CUDA runs golden and
Triton by default, loads the NPU tensors, and calls CT's ``dual`` Python API for
L1 double-benchmark checks.  Pass ``--save-gpu-tensors`` to save standalone
golden/Triton tensors instead of running CT online.  Complete CT terminal
reports are also saved under ``<output-dir>/ct_logs``.

Typical commands are ``--device cuda:0`` for golden plus Triton and
``--device npu:0`` for the Ascend C operator.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn.functional as F

from golden import recurrent_gated_delta_rule_golden


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    nk: int
    nv: int
    dv: int
    use_g: bool
    use_gk: bool
    ssm_pattern: str = "unique"
    ssm_indices: Optional[tuple[int, ...]] = None
    accepted_tokens: Optional[tuple[int, ...]] = None
    seed: int = 42
    expected_error: Optional[str] = None
    dk: int = 128

    @property
    def batch_size(self) -> int:
        return len(self.lengths)

    @property
    def token_count(self) -> int:
        return sum(self.lengths)


HEAD_CONFIGS = (
    (1, 1),
    (2, 4),
    (4, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 32),
    (48, 96),
    (32, 96),
)  # (Nk, Nv) pairs

QUICK_CASES = (
    Case(
        "q01_b1_l1_no_accepted",
        (1,),
        4,
        4,
        128,
        use_g=True,
        use_gk=False,
    ),
    Case(
        "q02_b1_l1_accepted",
        (1,),
        2,
        4,
        128,
        use_g=True,
        use_gk=False,
        accepted_tokens=(1,),
    ),
    Case(
        "q03_b4_l2_unique",
        (2, 2, 2, 2),
        4,
        8,
        128,
        use_g=False,
        use_gk=True,
    ),
    Case(
        "q04_b4_l2_accepted",
        (2, 2, 2, 2),
        4,
        8,
        256,
        use_g=True,
        use_gk=False,
        accepted_tokens=(2, 1, 2, 1),
    ),
    Case(
        "q05_empty_sequence",
        (3, 0, 2),
        2,
        4,
        128,
        use_g=True,
        use_gk=False,
    ),
    Case(
        "q06_permuted_state",
        (2, 2, 2, 2),
        4,
        8,
        256,
        ssm_pattern="permuted_unique",
        use_g=False,
        use_gk=True,
    ),
    Case(
        "q07_explicit_permutation",
        (3, 3, 2),
        4,
        8,
        128,
        use_g=True,
        use_gk=False,
        ssm_indices=(7, 2, 5, 0, 6, 1, 4, 3),
        accepted_tokens=(1, 3, 2),
    ),
    Case(
        "q08_head_vs_tail",
        (4, 4),
        2,
        4,
        256,
        use_g=True,
        use_gk=False,
        accepted_tokens=(1, 4),
    ),
    Case(
        "q09_mixed_lengths",
        (1, 8, 3, 2),
        4,
        4,
        128,
        use_g=True,
        use_gk=False,
        ssm_pattern="permuted_unique",
        seed=91,
    ),
    Case(
        "q10_mixed_lengths_accepted",
        (1, 8, 3, 2),
        4,
        8,
        256,
        use_g=True,
        use_gk=False,
        ssm_pattern="permuted_unique",
        accepted_tokens=(1, 8, 2, 1),
        seed=92,
    ),
    Case(
        "q11_invalid_accepted_zero",
        (2, 2),
        2,
        4,
        128,
        use_g=True,
        use_gk=False,
        accepted_tokens=(0, 1),
        expected_error="numAcceptedTokens must be in [1, Li]",
    ),
    Case(
        "q12_permuted_smoke",
        (8, 1, 0, 5),
        1,
        4,
        256,
        use_g=True,
        use_gk=False,
        ssm_pattern="permuted_unique",
        seed=93,
    ),
    Case(
        "q13_b16_l2_accepted_permuted_g",
        (2,) * 16,
        2,
        4,
        128,
        use_g=True,
        use_gk=False,
        ssm_pattern="permuted_unique",
        accepted_tokens=(1, 2) * 8,
        seed=94,
    ),
    Case(
        "q14_b32_l1_permuted_gk",
        (1,) * 32,
        1,
        2,
        128,
        use_g=False,
        use_gk=True,
        ssm_pattern="permuted_unique",
        seed=95,
    ),
    Case(
        "q15_b64_mixed_accepted_g",
        (1, 2) * 32,
        2,
        2,
        128,
        use_g=True,
        use_gk=False,
        accepted_tokens=(1, 2) * 32,
        seed=96,
    ),
    Case(
        "q16_b128_l1_accepted_permuted_gk",
        (1,) * 128,
        1,
        1,
        128,
        use_g=False,
        use_gk=True,
        ssm_pattern="permuted_unique",
        accepted_tokens=(1,) * 128,
        seed=97,
    ),
)

NEGATIVE_CASES = (
    Case(
        "neg_accepted_zero",
        (2, 2),
        2,
        4,
        128,
        use_g=True,
        use_gk=False,
        accepted_tokens=(0, 1),
        expected_error="numAcceptedTokens must be in [1, Li]",
    ),
    Case(
        "neg_accepted_gt_length",
        (2, 2),
        2,
        4,
        128,
        use_g=True,
        use_gk=False,
        accepted_tokens=(3, 1),
        expected_error="numAcceptedTokens must be in [1, Li]",
    ),
    Case(
        "neg_accepted_negative",
        (2, 2),
        2,
        4,
        256,
        use_g=True,
        use_gk=False,
        accepted_tokens=(-1, 1),
        expected_error="numAcceptedTokens must be in [1, Li]",
    ),
    Case(
        "neg_empty_with_accepted",
        (2, 0, 1),
        4,
        8,
        128,
        use_g=True,
        use_gk=False,
        accepted_tokens=(2, 0, 1),
        expected_error="empty sequences cannot have numAcceptedTokens",
    ),
)

STRESS_CASES = (
    Case(
        "stress_b16_unique",
        (8,) * 16,
        4,
        8,
        128,
        use_g=True,
        use_gk=False,
        ssm_pattern="unique",
        seed=301,
    ),
    Case(
        "stress_b16_permuted",
        (8,) * 16,
        4,
        8,
        256,
        use_g=True,
        use_gk=False,
        ssm_pattern="permuted_unique",
        seed=302,
    ),
    Case(
        "stress_b16_permuted_gk",
        (8,) * 16,
        8,
        16,
        128,
        ssm_pattern="permuted_unique",
        use_g=False,
        use_gk=True,
        seed=303,
    ),
)


def make_random_cases(count: int = 20, seed: int = 20260728) -> tuple[Case, ...]:
    """Create the requested reproducible random regression set."""
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        batch_size = rng.choice((1, 2, 4, 8))
        lengths = tuple(rng.randint(0, 8) for _ in range(batch_size))
        if not any(lengths):
            lengths = (1, *lengths[1:])
        nk, nv = rng.choice(HEAD_CONFIGS)
        accepted = None
        if all(length > 0 for length in lengths) and rng.random() < 0.5:
            accepted = tuple(rng.randint(1, length) for length in lengths)
        pattern = rng.choice(("unique", "permuted_unique"))
        dv = rng.choice((128, 256))
        use_g = bool(rng.getrandbits(1))
        cases.append(
            Case(
                name=f"random_{index:02d}",
                lengths=lengths,
                nk=nk,
                nv=nv,
                dv=dv,
                ssm_pattern=pattern,
                accepted_tokens=accepted,
                use_g=use_g,
                use_gk=not use_g,
                seed=seed + index,
            )
        )
    return tuple(cases)


def make_large_batch_cases(count: int = 20, seed: int = 20260729) -> tuple[Case, ...]:
    """Create 20 cases with 10 < B < 1000 and one state block per token."""
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        batch_size = rng.randint(11, 256)
        with_accepted = index % 2 == 1
        if with_accepted:
            # Keep the state tensor bounded while retaining large-B coverage.
            lengths = (1,) * batch_size
            accepted = (1,) * batch_size
        else:
            lengths = tuple(rng.randint(0, 1) for _ in range(batch_size))
            if not any(lengths):
                lengths = (1, *lengths[1:])
            accepted = None
        use_g = bool(index % 2)
        cases.append(
            Case(
                name=f"large_{index:02d}_b{batch_size}",
                lengths=lengths,
                nk=1,
                nv=1,
                dv=128,
                ssm_pattern=("unique", "permuted_unique")[index % 2],
                accepted_tokens=accepted,
                use_g=use_g,
                use_gk=not use_g,
                seed=seed + index,
            )
        )
    return tuple(cases)


RANDOM_CASES = make_random_cases()
LARGE_BATCH_CASES = make_large_batch_cases()


def _deduplicate_cases(cases: Iterable[Case]) -> list[Case]:
    unique: dict[str, Case] = {}
    for case in cases:
        unique.setdefault(case.name, case)
    return list(unique.values())


def cases_for_suite(suite: str) -> list[Case]:
    suites = {
        "quick": QUICK_CASES,
        "random": RANDOM_CASES,
        "large": LARGE_BATCH_CASES,
        "stress": STRESS_CASES,
        "negative": NEGATIVE_CASES,
    }
    if suite == "all":
        return _deduplicate_cases(
            (*QUICK_CASES, *RANDOM_CASES, *LARGE_BATCH_CASES, *STRESS_CASES, *NEGATIVE_CASES)
        )
    return list(suites[suite])


def validate_case(case: Case, allow_expected_error: bool = True) -> None:
    if case.use_g == case.use_gk:
        raise ValueError(f"{case.name}: exactly one of use_g and use_gk must be true")
    if not 1 <= case.batch_size < 1000:
        raise ValueError(f"{case.name}: batch size must be in [1, 999]")
    if case.dk != 128 or case.dv not in (128, 256):
        raise ValueError(f"{case.name}: dk must be 128 and dv must be 128 or 256")
    if case.nv < case.nk or case.nv % case.nk:
        raise ValueError(f"{case.name}: expected Nv >= Nk and Nv % Nk == 0")
    if not all(0 <= length <= 8 for length in case.lengths):
        raise ValueError(f"{case.name}: every Li must be in [0, 8]")
    if case.token_count <= 0:
        raise ValueError(f"{case.name}: the operator does not support an empty query tensor")
    if case.ssm_indices is not None and len(case.ssm_indices) != case.token_count:
        raise ValueError(f"{case.name}: explicit ssmStateIndices must have T entries")
    if case.accepted_tokens is not None:
        if len(case.accepted_tokens) != case.batch_size:
            raise ValueError(f"{case.name}: numAcceptedTokens must have B entries")
        invalid = any(
            length == 0 or accepted < 1 or accepted > length
            for length, accepted in zip(case.lengths, case.accepted_tokens)
        )
        if invalid and not (allow_expected_error and case.expected_error):
            raise ValueError(f"{case.name}: numAcceptedTokens must be in [1, Li] and Li must be positive")


def build_ssm_indices(case: Case) -> torch.Tensor:
    if case.ssm_indices is not None:
        indices = torch.tensor(case.ssm_indices, dtype=torch.int32)
    elif case.ssm_pattern == "unique":
        indices = torch.arange(case.token_count, dtype=torch.int32)
    elif case.ssm_pattern == "permuted_unique":
        generator = torch.Generator(device="cpu").manual_seed(case.seed + 17)
        indices = torch.randperm(case.token_count, generator=generator).to(torch.int32)
    else:
        raise ValueError(f"{case.name}: unsupported ssm pattern {case.ssm_pattern!r}")

    if indices.unique().numel() != case.token_count:
        raise ValueError(f"{case.name}: ssmStateIndices must contain one unique block per token")
    if int(indices.min()) < 0 or int(indices.max()) >= case.token_count:
        raise ValueError(f"{case.name}: ssmStateIndices must be a permutation of [0, T)")
    return indices


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def generate_inputs(case: Case) -> dict[str, Any]:
    validate_case(case)
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    t = case.token_count
    ssm_indices = build_ssm_indices(case)
    block_num = t

    query = F.normalize(
        torch.rand((t, case.nk, case.dk), generator=generator, dtype=torch.float32),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    key = F.normalize(
        torch.rand((t, case.nk, case.dk), generator=generator, dtype=torch.float32),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    value = torch.rand((t, case.nv, case.dv), generator=generator).to(torch.bfloat16)
    beta = torch.rand((t, case.nv), generator=generator).to(torch.bfloat16)
    g = (
        -torch.rand((t, case.nv), generator=generator, dtype=torch.float32)
        if case.use_g
        else None
    )
    gk = (
        -torch.rand((t, case.nv, case.dk), generator=generator, dtype=torch.float32)
        if case.use_gk
        else None
    )

    # A unique, bounded constant per block makes state writeback observable.
    block_ids = torch.arange(1, block_num + 1, dtype=torch.float32)
    block_ids = (block_ids / (block_num + 1) * 0.125).to(torch.bfloat16)
    state = block_ids[:, None, None, None].expand(
        block_num, case.nv, case.dv, case.dk
    ).clone()

    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "state": state,
        "actual_seq_lengths": torch.tensor((0, *case.lengths), dtype=torch.int32),
        "ssm_state_indices": ssm_indices,
        "num_accepted_tokens": (
            torch.tensor(case.accepted_tokens, dtype=torch.int32)
            if case.accepted_tokens is not None
            else None
        ),
        "g": g,
        "gk": gk,
    }
    metadata = {
        **asdict(case),
        "batch_size": case.batch_size,
        "token_count": t,
        "block_num": block_num,
        "scale": case.dk**-0.5,
        "state_v_first": True,
        "input_layout": {
            "query_key": "(T, Nk, Dk)",
            "value_output": "(T, Nv, Dv)",
            "state": "(BlockNum, Nv, Dv, Dk)",
            "triton_packed_batch": "(1, T, ...)",
        },
    }
    return {
        "metadata": metadata,
        "scale": case.dk**-0.5,
        **tensors,
        "checksums": {
            name: tensor_sha256(tensor)
            for name, tensor in tensors.items()
            if isinstance(tensor, torch.Tensor)
        },
    }


def validate_inputs(inputs: dict[str, Any], case: Case, check_accepted: bool = True) -> None:
    validate_case(case)
    t = case.token_count
    block_num = int(inputs["metadata"]["block_num"])
    specs = {
        "query": ((t, case.nk, case.dk), torch.bfloat16),
        "key": ((t, case.nk, case.dk), torch.bfloat16),
        "value": ((t, case.nv, case.dv), torch.bfloat16),
        "beta": ((t, case.nv), torch.bfloat16),
        "state": ((block_num, case.nv, case.dv, case.dk), torch.bfloat16),
        "actual_seq_lengths": ((case.batch_size + 1,), torch.int32),
        "ssm_state_indices": ((t,), torch.int32),
    }
    for name, (shape, dtype) in specs.items():
        tensor = inputs[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{case.name}: {name} expected shape={shape}, dtype={dtype}; "
                f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
    expected_lengths = torch.tensor((0, *case.lengths), dtype=torch.int32)
    if not torch.equal(inputs["actual_seq_lengths"], expected_lengths):
        raise ValueError(f"{case.name}: actualSeqLengths does not match the case")
    indices = inputs["ssm_state_indices"]
    if int(indices.min()) < 0 or int(indices.max()) >= block_num:
        raise ValueError(f"{case.name}: ssmStateIndices is outside [0, BlockNum)")
    g = inputs["g"]
    gk = inputs["gk"]
    if (g is None) == (gk is None):
        raise ValueError(f"{case.name}: exactly one of g and gk must be provided")
    if (g is not None) != case.use_g or (gk is not None) != case.use_gk:
        raise ValueError(f"{case.name}: g/gk inputs do not match the selected case")
    if g is not None and (tuple(g.shape) != (t, case.nv) or g.dtype != torch.float32):
        raise ValueError(f"{case.name}: g must have shape (T, Nv) and float32 dtype")
    if gk is not None and (
        tuple(gk.shape) != (t, case.nv, case.dk) or gk.dtype != torch.float32
    ):
        raise ValueError(f"{case.name}: gk must have shape (T, Nv, Dk) and float32 dtype")
    if check_accepted:
        validate_accepted_tokens(inputs)
    for name, checksum in inputs["checksums"].items():
        if tensor_sha256(inputs[name]) != checksum:
            raise ValueError(f"{case.name}: checksum mismatch for {name}")


def validate_accepted_tokens(inputs: dict[str, Any]) -> None:
    accepted = inputs["num_accepted_tokens"]
    if accepted is None:
        return
    lengths = inputs["actual_seq_lengths"][1:]
    if tuple(accepted.shape) != tuple(lengths.shape) or accepted.dtype != torch.int32:
        raise ValueError("numAcceptedTokens must be INT32 with shape (B,)")
    if bool(torch.any(lengths == 0)):
        raise ValueError("empty sequences cannot have numAcceptedTokens")
    if bool(torch.any(accepted < 1)) or bool(torch.any(accepted > lengths)):
        raise ValueError("numAcceptedTokens must be in [1, Li]")


def sequence_ranges(inputs: dict[str, Any]) -> list[tuple[int, int]]:
    ranges = []
    offset = int(inputs["actual_seq_lengths"][0])
    for length in inputs["actual_seq_lengths"][1:].tolist():
        ranges.append((offset, offset + length))
        offset += length
    return ranges


def representative_tokens(inputs: dict[str, Any]) -> list[Optional[int]]:
    accepted = inputs["num_accepted_tokens"]
    result = []
    for batch, (start, end) in enumerate(sequence_ranges(inputs)):
        if start == end:
            result.append(None)
        elif accepted is None:
            result.append(start)
        else:
            result.append(start + int(accepted[batch]) - 1)
    return result


def expected_state_blocks(inputs: dict[str, Any]) -> dict[str, list[int]]:
    ssm = inputs["ssm_state_indices"]
    token_blocks = sorted(set(int(index) for index in ssm.tolist()))
    representative = representative_tokens(inputs)
    representative_blocks = sorted(
        {int(ssm[token]) for token in representative if token is not None}
    )
    return {
        "per_token_kernel_writeback": token_blocks,
        "initial_state_selection": representative_blocks,
    }


def modified_state_blocks(before: torch.Tensor, after: torch.Tensor) -> list[int]:
    changed = torch.any(before != after, dim=(1, 2, 3))
    return torch.nonzero(changed, as_tuple=False).flatten().tolist()


def prepare_input_file(case: Case, input_dir: Path, overwrite: bool) -> tuple[Path, dict[str, Any]]:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{case.name}.pt"
    if path.exists() and not overwrite:
        inputs = torch.load(path, map_location="cpu")
        for key, expected in asdict(case).items():
            if inputs["metadata"].get(key) != expected:
                raise ValueError(
                    f"{path} metadata[{key!r}] does not match the selected case; "
                    "pass --overwrite-inputs to regenerate it"
                )
        validate_inputs(inputs, case, check_accepted=False)
        return path, inputs
    inputs = generate_inputs(case)
    validate_inputs(inputs, case, check_accepted=False)
    torch.save(inputs, path)
    return path, inputs


def synchronize(backend: str, torch_npu_module: Any = None) -> None:
    if backend == "npu":
        torch_npu_module.npu.synchronize()
    elif backend == "cuda":
        torch.cuda.synchronize()


def _extract_npu_result(result: Any, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(result, (tuple, list)):
        output = result[0]
        final_state = result[1] if len(result) > 1 and result[1] is not None else state
    else:
        output, final_state = result, state
    return output, final_state


@torch.inference_mode()
def run_golden(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    """Run the float64 small-operator golden on CUDA."""

    validate_accepted_tokens(inputs)
    if not torch.cuda.is_available():
        raise RuntimeError("golden execution requires a CUDA-capable NVIDIA GPU")
    cuda_device = torch.device(device)
    torch.cuda.set_device(cuda_device)

    math_names = ("query", "key", "value", "beta", "state")
    device_inputs = {
        name: inputs[name].to(device=cuda_device, dtype=torch.float64)
        for name in math_names
    }
    for name in ("g", "gk"):
        value = inputs[name]
        device_inputs[name] = (
            value.to(device=cuda_device, dtype=torch.float64)
            if value is not None
            else None
        )

    def invoke() -> tuple[torch.Tensor, torch.Tensor]:
        return recurrent_gated_delta_rule_golden(
            query=device_inputs["query"],
            key=device_inputs["key"],
            value=device_inputs["value"],
            state=device_inputs["state"],
            beta=device_inputs["beta"],
            scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"],
            num_accepted_tokens=inputs["num_accepted_tokens"],
            g=device_inputs["g"],
            gk=device_inputs["gk"],
            compute_dtype=torch.float64,
            output_dtype=torch.float64,
        )

    output_1, state_1 = invoke()
    synchronize("cuda")
    output_1, state_1 = output_1.cpu(), state_1.cpu()
    output_2, state_2 = invoke()
    synchronize("cuda")
    output_2, state_2 = output_2.cpu(), state_2.cpu()
    if tuple(output_1.shape) != tuple(inputs["value"].shape):
        raise AssertionError(
            f"golden output shape {tuple(output_1.shape)} != "
            f"{tuple(inputs['value'].shape)}"
        )

    deterministic_output = torch.equal(output_1, output_2)
    deterministic_state = torch.equal(state_1, state_2)
    modified_blocks = modified_state_blocks(inputs["state"], state_1)
    expected_blocks = expected_state_blocks(inputs)
    unexpected_blocks = set(modified_blocks) - set(
        expected_blocks["per_token_kernel_writeback"]
    )
    return {
        "backend": "golden",
        "device": device,
        "compute_dtype": "float64",
        "output": output_1.contiguous(),
        "final_state": state_1.contiguous(),
        "deterministic_output": deterministic_output,
        "deterministic_state": deterministic_state,
        "modified_state_blocks": modified_blocks,
        "expected_state_blocks": expected_blocks,
        "unexpected_state_blocks": sorted(unexpected_blocks),
        "checks": {
            "deterministic_output": deterministic_output,
            "deterministic_state": deterministic_state,
            "state_writeback_is_expected_subset": not unexpected_blocks,
        },
    }


@torch.inference_mode()
def run_npu(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    validate_accepted_tokens(inputs)
    try:
        import torch_npu
        from fla_npu.ops.ascendc import recurrent_gated_delta_rule
    except ImportError as error:
        raise RuntimeError("NPU execution requires torch_npu and fla_npu") from error

    npu_device = torch.device(device)
    torch_npu.npu.set_device(npu_device)
    names = ("query", "key", "value", "beta", "actual_seq_lengths", "ssm_state_indices")
    device_inputs = {name: inputs[name].to(npu_device) for name in names}
    for name in ("g", "gk", "num_accepted_tokens"):
        value = inputs[name]
        device_inputs[name] = value.to(npu_device) if value is not None else None
    pristine_state = inputs["state"].to(npu_device)
    working_state = pristine_state.clone()

    def prepare_iteration() -> None:
        working_state.copy_(pristine_state)

    def invoke() -> Any:
        return recurrent_gated_delta_rule(
            device_inputs["query"],
            device_inputs["key"],
            device_inputs["value"],
            working_state,
            beta=device_inputs["beta"],
            scale=inputs["scale"],
            actual_seq_lengths=device_inputs["actual_seq_lengths"],
            ssm_state_indices=device_inputs["ssm_state_indices"],
            num_accepted_tokens=device_inputs["num_accepted_tokens"],
            g=device_inputs["g"],
            gk=device_inputs["gk"],
        )

    prepare_iteration()
    output_1, state_1 = _extract_npu_result(invoke(), working_state)
    synchronize("npu", torch_npu)
    output_1, state_1 = output_1.cpu(), state_1.cpu()
    prepare_iteration()
    output_2, state_2 = _extract_npu_result(invoke(), working_state)
    synchronize("npu", torch_npu)
    output_2, state_2 = output_2.cpu(), state_2.cpu()
    if tuple(output_1.shape) != tuple(inputs["value"].shape):
        raise AssertionError(f"NPU output shape {tuple(output_1.shape)} != {tuple(inputs['value'].shape)}")
    deterministic_output = torch.equal(output_1, output_2)
    deterministic_state = torch.equal(state_1, state_2)
    modified_blocks = modified_state_blocks(inputs["state"], state_1)
    expected_blocks = expected_state_blocks(inputs)
    unexpected_blocks = set(modified_blocks) - set(expected_blocks["per_token_kernel_writeback"])
    return {
        "backend": "npu",
        "device": device,
        "output": output_1.contiguous(),
        "final_state": state_1.contiguous(),
        "deterministic_output": deterministic_output,
        "deterministic_state": deterministic_state,
        "modified_state_blocks": modified_blocks,
        "expected_state_blocks": expected_blocks,
        "unexpected_state_blocks": sorted(unexpected_blocks),
        "checks": {
            "deterministic_output": deterministic_output,
            "deterministic_state": deterministic_state,
            "state_writeback_is_expected_subset": not unexpected_blocks,
        },
    }


def _import_triton_operator(fla_repo: Optional[Path]) -> Callable[..., Any]:
    if fla_repo is not None:
        resolved = fla_repo.expanduser().resolve()
        if not (resolved / "fla" / "ops" / "gated_delta_rule" / "fused_recurrent.py").is_file():
            raise FileNotFoundError(f"FLA repository not found at {resolved}")
        sys.path.insert(0, str(resolved))
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule


def triton_initial_states(inputs: dict[str, Any], device: torch.device) -> torch.Tensor:
    state = inputs["state"]
    shape = (len(sequence_ranges(inputs)), *state.shape[1:])
    initial = torch.zeros(shape, dtype=state.dtype)
    ssm = inputs["ssm_state_indices"]
    for batch, token in enumerate(representative_tokens(inputs)):
        if token is not None:
            initial[batch].copy_(state[int(ssm[token])])
    return initial.to(device)


def replay_triton_state_writeback(
    operator: Callable[..., Any],
    inputs: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    physical_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay at most eight recurrent steps to expose per-token states.

    The raw Triton invocation processes every packed sequence in one launch,
    but FLA only returns one final state per sequence.  The NPU API writes the
    state after every token to ``state[ssm_state_indices[token]]``.  This replay
    batches all active sequences at each step and uses Triton's own kernel to
    materialize the otherwise hidden intermediate states for equivalent result
    comparison.
    """

    device = q.device
    output = torch.zeros_like(v)
    if physical_state is None:
        physical_state = inputs["state"].to(device).clone()
    sequence_state = initial_state.float().clone()
    ranges = sequence_ranges(inputs)
    max_length = max((end - start for start, end in ranges), default=0)
    ssm = inputs["ssm_state_indices"]

    for step in range(max_length):
        active_batches = [
            batch for batch, (start, end) in enumerate(ranges) if start + step < end
        ]
        if not active_batches:
            continue
        tokens = [ranges[batch][0] + step for batch in active_batches]
        batch_indices = torch.tensor(active_batches, dtype=torch.long, device=device)
        token_indices = torch.tensor(tokens, dtype=torch.long, device=device)
        step_g = g.index_select(0, token_indices).unsqueeze(1) if g is not None else None
        step_gk = (
            gk.index_select(0, token_indices).unsqueeze(1) if gk is not None else None
        )
        step_output, step_state = operator(
            q=q.index_select(0, token_indices).unsqueeze(1),
            k=k.index_select(0, token_indices).unsqueeze(1),
            v=v.index_select(0, token_indices).unsqueeze(1),
            g=step_g,
            gk=step_gk,
            beta=beta.index_select(0, token_indices).unsqueeze(1),
            scale=inputs["scale"],
            initial_state=sequence_state.index_select(0, batch_indices),
            output_final_state=True,
            state_v_first=True,
        )
        output.index_copy_(0, token_indices, step_output.squeeze(1))
        sequence_state.index_copy_(0, batch_indices, step_state)
        physical_indices = torch.tensor(
            [int(ssm[token]) for token in tokens], dtype=torch.long, device=device
        )
        physical_state.index_copy_(
            0, physical_indices, step_state.to(physical_state.dtype)
        )

    return output, physical_state, sequence_state


@torch.inference_mode()
def run_triton(
    inputs: dict[str, Any],
    device: str,
    fla_repo: Optional[Path],
) -> dict[str, Any]:
    validate_accepted_tokens(inputs)
    if not torch.cuda.is_available():
        raise RuntimeError("Triton execution requires a CUDA-capable NVIDIA GPU")
    operator = _import_triton_operator(fla_repo)
    cuda_device = torch.device(device)
    torch.cuda.set_device(cuda_device)
    q_flat = inputs["query"].to(cuda_device)
    k_flat = inputs["key"].to(cuda_device)
    v_flat = inputs["value"].to(cuda_device)
    beta_flat = inputs["beta"].to(cuda_device)
    g_flat = inputs["g"].to(cuda_device) if inputs["g"] is not None else None
    gk_flat = inputs["gk"].to(cuda_device) if inputs["gk"] is not None else None
    valid_start = int(inputs["actual_seq_lengths"][0])
    q = q_flat[valid_start:].unsqueeze(0)
    k = k_flat[valid_start:].unsqueeze(0)
    v = v_flat[valid_start:].unsqueeze(0)
    beta = beta_flat[valid_start:].unsqueeze(0)
    g = g_flat[valid_start:].unsqueeze(0) if g_flat is not None else None
    gk = gk_flat[valid_start:].unsqueeze(0) if gk_flat is not None else None
    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(inputs["actual_seq_lengths"][1:].to(torch.int64), 0).tolist()],
        dtype=torch.int64,
        device=cuda_device,
    )
    initial_state = triton_initial_states(inputs, cuda_device)

    def invoke() -> Any:
        return operator(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            beta=beta,
            scale=inputs["scale"],
            initial_state=initial_state,
            output_final_state=True,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

    physical_state_template = inputs["state"].to(cuda_device)
    working_physical_state = physical_state_template.clone()

    def prepare_equivalent_iteration() -> None:
        working_physical_state.copy_(physical_state_template)

    def invoke_equivalent() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return replay_triton_state_writeback(
            operator,
            inputs,
            q_flat,
            k_flat,
            v_flat,
            beta_flat,
            g_flat,
            gk_flat,
            initial_state,
            working_physical_state,
        )

    packed_output_1, sequence_state_1 = invoke()
    synchronize("cuda")
    output_1 = torch.zeros_like(v_flat)
    output_1[valid_start:] = packed_output_1.squeeze(0)
    packed_output_2, sequence_state_2 = invoke()
    synchronize("cuda")
    output_2 = torch.zeros_like(v_flat)
    output_2[valid_start:] = packed_output_2.squeeze(0)
    prepare_equivalent_iteration()
    replay_output, physical_state, replay_sequence_state = invoke_equivalent()
    synchronize("cuda")
    replay_matches_packed_output = torch.allclose(
        replay_output.float(), output_1.float(), rtol=2e-3, atol=2e-3
    )
    replay_matches_packed_state = torch.allclose(
        replay_sequence_state.float(), sequence_state_1.float(), rtol=2e-3, atol=2e-3
    )
    output_1, output_2 = output_1.cpu(), output_2.cpu()
    sequence_state_1, sequence_state_2 = sequence_state_1.cpu(), sequence_state_2.cpu()
    physical_state = physical_state.cpu()
    if tuple(output_1.shape) != tuple(inputs["value"].shape):
        raise AssertionError(
            f"Triton output shape {tuple(output_1.shape)} != {tuple(inputs['value'].shape)}"
        )
    deterministic_output = torch.equal(output_1, output_2)
    deterministic_state = torch.equal(sequence_state_1, sequence_state_2)
    modified_blocks = modified_state_blocks(inputs["state"], physical_state)
    expected_blocks = expected_state_blocks(inputs)
    unexpected_blocks = set(modified_blocks) - set(
        expected_blocks["per_token_kernel_writeback"]
    )
    return {
        # "backend": "triton",
        # "device": device,
        "output": output_1.contiguous(),
        "final_state": physical_state.contiguous(),
        # "sequence_final_state": sequence_state_1.contiguous(),
        # "deterministic_output": deterministic_output,
        # "deterministic_state": deterministic_state,
        # "replay_matches_packed_output": replay_matches_packed_output,
        # "replay_matches_packed_state": replay_matches_packed_state,
        # "modified_state_blocks": modified_blocks,
        # "expected_state_blocks": expected_blocks,
        # "unexpected_state_blocks": sorted(unexpected_blocks),
        # "checks": {
        #     "deterministic_output": deterministic_output,
        #     "deterministic_state": deterministic_state,
        #     "replay_matches_packed_output": replay_matches_packed_output,
        #     "replay_matches_packed_state": replay_matches_packed_state,
        #     "state_writeback_is_expected_subset": not unexpected_blocks,
        # },
    }


TENSOR_RESULT_NAMES = ("output", "final_state")
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class TerminalAndLogStream:
    """Mirror terminal output to a plain-text log without ANSI escapes."""

    def __init__(self, terminal: Any, log_file: Any) -> None:
        self.terminal = terminal
        self.log_file = log_file
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self.terminal.write(data)
        self.log_file.write(ANSI_ESCAPE.sub("", data))
        return len(data)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


def tensor_result_paths(
    output_dir: Path, backend: str, case_name: str
) -> dict[str, Path]:
    tensor_dir = output_dir / "tensors" / backend
    return {
        name: tensor_dir / f"{case_name}_{name}.pt"
        for name in TENSOR_RESULT_NAMES
    }


def save_backend_tensors(
    output_dir: Path,
    backend: str,
    case_name: str,
    backend_result: dict[str, Any],
) -> dict[str, Path]:
    """Save each result as a bare CPU tensor, never as a keyed payload."""

    paths = tensor_result_paths(output_dir, backend, case_name)
    for name, path in paths.items():
        tensor = backend_result.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{backend} {case_name} {name} is not a Tensor")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().cpu().contiguous(), path)
    return paths


def load_npu_tensors(output_dir: Path, case_name: str) -> dict[str, torch.Tensor]:
    paths = tensor_result_paths(output_dir, "npu", case_name)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing NPU tensor files. Run the NPU command first and copy its "
            f"output directory to this host: {missing}"
        )

    tensors = {}
    for name, path in paths.items():
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{path} must contain a bare Tensor, got {type(tensor).__name__}"
            )
        tensors[name] = tensor.contiguous()
    return tensors


def run_ct_l1_dual(
    output_dir: Path,
    case_name: str,
    npu_result: dict[str, torch.Tensor],
    golden_result: dict[str, Any],
    triton_result: dict[str, Any],
) -> dict[str, Any]:
    """Use NPU as test, float64 golden as gt, and Triton as bench."""

    try:
        from ct import dual
    except ImportError as error:
        raise RuntimeError(
            "Online GPU comparison requires the installed CT Tool Python package"
        ) from error

    log_path = output_dir / "ct_logs" / f"{case_name}_ct_dual_L1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = TerminalAndLogStream(sys.stdout, log_file)
        stderr_tee = TerminalAndLogStream(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(
                f"CT dual L1 case={case_name}: Test=NPU, GT=Golden, "
                "Bench=Triton",
                flush=True,
            )
            for name in TENSOR_RESULT_NAMES:
                gt = golden_result.get(name)
                bench = triton_result.get(name)
                if not isinstance(gt, torch.Tensor) or not isinstance(
                    bench, torch.Tensor
                ):
                    raise TypeError(
                        f"golden/Triton {name} must both be Tensor values"
                    )
                print(
                    f"  CT dual L1: case={case_name}, tensor={name}",
                    flush=True,
                )
                results[name] = dual(
                    test=npu_result[name],
                    gt=gt,
                    bench=bench,
                    level="L1",
                )
                print(f"    {results[name]}", flush=True)
    print(f"  complete CT log: {log_path}", flush=True)
    return results


BACKEND_DEVICE_TYPES = {
    "golden": "cuda",
    "triton": "cuda",
    "npu": "npu",
}


def infer_device_type(device: str) -> str:
    prefix = device.split(":", 1)[0].lower()
    if prefix not in ("cuda", "npu"):
        raise ValueError("--device must be cuda:N or npu:N")
    return prefix


def select_backends(device: str, requested: Optional[list[str]]) -> tuple[str, ...]:
    device_type = infer_device_type(device)
    backends = tuple(requested or (("golden", "triton") if device_type == "cuda" else ("npu",)))
    incompatible = [
        backend
        for backend in backends
        if BACKEND_DEVICE_TYPES[backend] != device_type
    ]
    if incompatible:
        raise ValueError(
            f"backends {incompatible} cannot run on {device}; "
            f"expected {device_type} backends only"
        )
    return tuple(dict.fromkeys(backends))


def default_fla_repo() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "flash-linear-attention-npu":
            candidate = parent.parent / "flash-linear-attention"
            return candidate if candidate.is_dir() else None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        required=True,
        help="npu:N or cuda:N; the prefix limits which backends can run",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=tuple(BACKEND_DEVICE_TYPES),
        help=(
            "implementation to run; may be repeated. Defaults to golden+triton "
            "on CUDA and npu on NPU"
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("quick", "random", "large", "stress", "negative", "all"),
        default="quick",
    )
    parser.add_argument("--case", action="append", help="Run only named cases; may be repeated")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recurrent_gdn_accuracy_outputs"),
    )
    parser.add_argument("--overwrite-inputs", action="store_true")
    parser.add_argument(
        "--save-gpu-tensors",
        action="store_true",
        help=(
            "On CUDA, save golden and Triton output/final_state tensors instead "
            "of loading NPU tensors and running CT dual L1 online"
        ),
    )
    parser.add_argument(
        "--fla-repo",
        type=Path,
        default=default_fla_repo(),
        help="Path containing fla/ops/gated_delta_rule/fused_recurrent.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = select_backends(args.device, args.backend)
    device_type = infer_device_type(args.device)
    if device_type == "npu" and args.save_gpu_tensors:
        raise ValueError("--save-gpu-tensors is valid only with --device cuda:N")
    if (
        device_type == "cuda"
        and not args.save_gpu_tensors
        and set(backends) != {"golden", "triton"}
    ):
        raise ValueError(
            "Online CT dual comparison requires both --backend golden and "
            "--backend triton (or omit --backend to use both by default)"
        )

    cases = cases_for_suite(args.suite)
    if args.case:
        requested = set(args.case)
        all_cases = cases_for_suite("all")
        cases = [case for case in all_cases if case.name in requested]
        missing = requested - {case.name for case in cases}
        if missing:
            raise ValueError(f"Unknown cases: {sorted(missing)}")
    if args.list_cases:
        for case in cases:
            print(case.name)
        return

    input_dir = args.output_dir / "inputs"
    failures = []

    for case in cases:
        print(
            f"[{case.name}] B={case.batch_size}, T={case.token_count}, "
            f"Nk/Nv={case.nk}/{case.nv}, Dk/Dv={case.dk}/{case.dv}, "
            f"decay={'g' if case.use_g else 'gk' if case.use_gk else 'none'}"
        )
        input_path, inputs = prepare_input_file(case, input_dir, args.overwrite_inputs)
        print(f"  input: {input_path}")
        backend_results = {}
        for backend in backends:
            try:
                if backend == "golden":
                    backend_result = run_golden(inputs, args.device)
                elif backend == "triton":
                    backend_result = run_triton(
                        inputs, args.device, args.fla_repo
                    )
                else:
                    backend_result = run_npu(inputs, args.device)
                if case.expected_error:
                    message = f"{case.name}/{backend}: unexpected success"
                    failures.append(message)
                    print(f"  {backend}: UNEXPECTED SUCCESS")
                    continue
                checks = backend_result.get("checks")
                failed_checks = (
                    [name for name, passed in checks.items() if not passed]
                    if checks is not None
                    else []
                )
                if failed_checks:
                    message = (
                        f"{case.name}/{backend}: validation failed: "
                        f"{failed_checks}"
                    )
                    failures.append(message)
                    print(f"  {backend}: VALIDATION FAILED {failed_checks}")
                    continue
                backend_results[backend] = backend_result
                print(f"  {backend}: ok")
            except Exception as error:
                expected = (
                    case.expected_error is not None
                    and case.expected_error in str(error)
                )
                if expected:
                    print(f"  {backend}: expected error: {error}")
                else:
                    message = (
                        f"{case.name}/{backend}: {type(error).__name__}: {error}"
                    )
                    failures.append(message)
                    print(f"  {backend}: ERROR: {error}")

        if case.expected_error:
            continue

        try:
            if device_type == "npu":
                npu_result = backend_results.get("npu")
                if npu_result is None:
                    continue
                paths = save_backend_tensors(
                    args.output_dir, "npu", case.name, npu_result
                )
                for name, path in paths.items():
                    print(f"  saved NPU {name}: {path}")
            elif args.save_gpu_tensors:
                for backend in backends:
                    backend_result = backend_results.get(backend)
                    if backend_result is None:
                        continue
                    paths = save_backend_tensors(
                        args.output_dir, backend, case.name, backend_result
                    )
                    for name, path in paths.items():
                        print(f"  saved {backend} {name}: {path}")
            elif {"golden", "triton"}.issubset(backend_results):
                npu_result = load_npu_tensors(args.output_dir, case.name)
                run_ct_l1_dual(
                    args.output_dir,
                    case.name,
                    npu_result,
                    backend_results["golden"],
                    backend_results["triton"],
                )
        except Exception as error:
            message = f"{case.name}/result: {type(error).__name__}: {error}"
            failures.append(message)
            print(f"  result handling: ERROR: {error}")

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("All requested cases completed.")

if __name__ == "__main__":
    main()
