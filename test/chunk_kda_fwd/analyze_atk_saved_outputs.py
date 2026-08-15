#!/usr/bin/env python3
"""Analyze saved ATK NPU, Triton, and FP64 golden outputs for chunk_kda_fwd."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


OUTPUT_NAMES = (
    "attn_out",
    "final_state",
    "gk",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state_out",
)
ROLES = ("npu", "triton", "golden")


@dataclass
class Metrics:
    count: int = 0
    nonfinite: int = 0
    sum_abs: float = 0.0
    sum_square: float = 0.0
    actual_square: float = 0.0
    reference_square: float = 0.0
    max_abs: float = 0.0

    @property
    def mae(self) -> float:
        return self.sum_abs / self.count if self.count else math.nan

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sum_square / self.count) if self.count else math.nan

    @property
    def relative_l2(self) -> float:
        if self.reference_square == 0.0:
            return math.inf if self.sum_square else 0.0
        return math.sqrt(self.sum_square / self.reference_square)

    @property
    def actual_rms(self) -> float:
        return math.sqrt(self.actual_square / self.count) if self.count else math.nan

    @property
    def reference_rms(self) -> float:
        return math.sqrt(self.reference_square / self.count) if self.count else math.nan


@dataclass
class Detail:
    global_metrics: Metrics
    chunks: list[Metrics]
    tokens: list[Metrics]
    heads: list[Metrics]
    top_elements: list[tuple[float, tuple[int, ...], float, float]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one ATK --save_data output run. The result root may be the "
            "configured --output_path or a timestamped ATK task directory."
        )
    )
    parser.add_argument("result_root", type=Path, help="ATK output directory to scan")
    parser.add_argument("--case-id", type=int, default=250)
    parser.add_argument(
        "--case-json",
        type=Path,
        default=Path(__file__).with_name("atk_chunk_kda_fwd.json"),
        help="ATK case JSON used to recover shape and visible-output metadata",
    )
    parser.add_argument(
        "--detail-output",
        type=int,
        default=0,
        help="compacted ATK output index for chunk/head diagnostics (default: 0)",
    )
    parser.add_argument(
        "--visible-outputs",
        default="",
        help="comma-separated KDA_ATK_VISIBLE_OUTPUTS value, when it was set for the run",
    )
    parser.add_argument(
        "--sequence-axis",
        type=int,
        default=None,
        help="override the inferred sequence axis for the detailed output",
    )
    parser.add_argument(
        "--head-axis",
        type=int,
        default=None,
        help="override the inferred head axis for the detailed output",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--absolute-rmse",
        type=float,
        default=1e-4,
        help="minimum NPU RMSE used by the diagnostic onset rule",
    )
    parser.add_argument(
        "--baseline-ratio",
        type=float,
        default=5.0,
        help="NPU/golden RMSE ratio over Triton/golden used by the onset rule",
    )
    return parser.parse_args()


def _role_for_part(part: str) -> Optional[str]:
    lowered = part.lower()
    if lowered == "gpu_benchmark" or ("gpu" in lowered and "benchmark" in lowered):
        return "golden"
    if lowered.startswith("npu_") or lowered == "npu":
        return "npu"
    if lowered.startswith("gpu_") or lowered == "gpu":
        return "triton"
    return None


def _discover_runs(root: Path, case_id: int) -> dict[Path, dict[str, dict[int, Path]]]:
    runs: dict[Path, dict[str, dict[int, Path]]] = {}
    case_component = str(case_id)
    for path in root.rglob("output_*.pt"):
        if case_component not in path.parts:
            continue
        try:
            output_index = int(path.stem.removeprefix("output_"))
        except ValueError:
            continue
        role_index = None
        role = None
        for index, part in enumerate(path.parts):
            candidate = _role_for_part(part)
            if candidate is not None:
                role_index, role = index, candidate
        if role_index is None or role is None:
            continue
        run_root = Path(*path.parts[:role_index])
        role_outputs = runs.setdefault(run_root, {}).setdefault(role, {})
        if output_index in role_outputs:
            raise RuntimeError(
                f"duplicate output_{output_index}.pt for role {role} under {run_root}; "
                "pass a narrower result_root"
            )
        role_outputs[output_index] = path
    return runs


def _select_run(root: Path, case_id: int) -> tuple[Path, dict[str, dict[int, Path]], int]:
    runs = _discover_runs(root, case_id)
    complete = [
        (run_root, outputs)
        for run_root, outputs in runs.items()
        if all(role in outputs for role in ROLES)
    ]
    if not complete:
        discovered = ", ".join(
            f"{path}: {sorted(outputs)}" for path, outputs in sorted(runs.items())
        )
        raise RuntimeError(
            "no complete NPU + regular GPU + gpu_benchmark result set found for "
            f"case {case_id} under {root}; discovered: {discovered or 'nothing'}"
        )

    def newest_mtime(item: tuple[Path, dict[str, dict[int, Path]]]) -> float:
        return max(path.stat().st_mtime for files in item[1].values() for path in files.values())

    complete.sort(key=newest_mtime, reverse=True)
    selected_root, selected = complete[0]
    return selected_root, selected, len(complete)


def _load_case_spec(path: Path, case_id: int) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        if int(case["id"]) != case_id:
            continue
        for item in case.get("inputs", []):
            if item.get("name") == "case_spec":
                return json.loads(item["range_values"])
        raise RuntimeError(f"case {case_id} has no case_spec input in {path}")
    raise RuntimeError(f"case {case_id} not found in {path}")


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _visible_output_names(spec: dict, selected_value: str) -> list[str]:
    available = {
        "attn_out": True,
        "final_state": _as_bool(spec.get("output_final_state")),
        "gk": not _as_bool(spec.get("use_gate_in_kernel"))
        or _as_bool(spec.get("disable_recompute")),
        "Aqk": True,
        "Akk": True,
        "w": _as_bool(spec.get("disable_recompute")),
        "u": _as_bool(spec.get("disable_recompute")),
        "qg": _as_bool(spec.get("disable_recompute")),
        "kg": _as_bool(spec.get("disable_recompute")),
        "v_new": _as_bool(spec.get("disable_recompute")),
        "h": _as_bool(spec.get("disable_recompute"))
        or _as_bool(spec.get("return_intermediate_states")),
        "initial_state_out": _as_bool(spec.get("initial_state")),
    }
    selected = {item.strip() for item in selected_value.split(",") if item.strip()}
    return [
        name
        for name in OUTPUT_NAMES
        if available[name] and (not selected or name in selected)
    ]


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} contains {type(value).__name__}, expected torch.Tensor")
    return value.detach()


def _add_metrics(total: Metrics, actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual = actual.to(torch.float64)
    reference = reference.to(torch.float64)
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    total.nonfinite += int(finite.numel() - finite.sum().item())
    if not finite.any().item():
        return
    actual = actual[finite]
    reference = reference[finite]
    difference = actual - reference
    total.count += difference.numel()
    total.sum_abs += float(difference.abs().sum().item())
    total.sum_square += float(difference.square().sum().item())
    total.actual_square += float(actual.square().sum().item())
    total.reference_square += float(reference.square().sum().item())
    total.max_abs = max(total.max_abs, float(difference.abs().max().item()))


def _merge_metrics(total: Metrics, item: Metrics) -> None:
    total.count += item.count
    total.nonfinite += item.nonfinite
    total.sum_abs += item.sum_abs
    total.sum_square += item.sum_square
    total.actual_square += item.actual_square
    total.reference_square += item.reference_square
    total.max_abs = max(total.max_abs, item.max_abs)


def _metrics_by_axis(
    actual: torch.Tensor, reference: torch.Tensor, axis: int
) -> list[Metrics]:
    axis_size = actual.shape[axis]
    actual = actual.to(torch.float64).movedim(axis, 0).reshape(axis_size, -1)
    reference = reference.to(torch.float64).movedim(axis, 0).reshape(axis_size, -1)
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    difference = torch.where(finite, actual - reference, torch.zeros_like(actual))
    actual = torch.where(finite, actual, torch.zeros_like(actual))
    reference = torch.where(finite, reference, torch.zeros_like(reference))
    count = finite.sum(dim=1).tolist()
    nonfinite = (finite.shape[1] - finite.sum(dim=1)).tolist()
    sum_abs = difference.abs().sum(dim=1).tolist()
    sum_square = difference.square().sum(dim=1).tolist()
    actual_square = actual.square().sum(dim=1).tolist()
    reference_square = reference.square().sum(dim=1).tolist()
    max_abs = difference.abs().max(dim=1).values.tolist()
    return [
        Metrics(
            count=int(count[index]),
            nonfinite=int(nonfinite[index]),
            sum_abs=float(sum_abs[index]),
            sum_square=float(sum_square[index]),
            actual_square=float(actual_square[index]),
            reference_square=float(reference_square[index]),
            max_abs=float(max_abs[index]),
        )
        for index in range(len(count))
    ]


def _infer_axis(shape: tuple[int, ...], size: int, preferred: int, label: str) -> int:
    candidates = [index for index, value in enumerate(shape) if value == size]
    if preferred in candidates:
        return preferred
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"cannot infer {label} axis: shape={shape}, expected dimension={size}, "
        f"candidates={candidates}; pass --{label}-axis"
    )


def _flat_to_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(result))


def _analyze_pair(
    actual_path: Path,
    reference_path: Path,
    sequence_axis: int,
    chunk_size: int,
    head_axis: Optional[int],
    top_k: int,
    detailed: bool,
) -> tuple[tuple[int, ...], torch.dtype, torch.dtype, Detail]:
    actual = _load_tensor(actual_path)
    reference = _load_tensor(reference_path)
    if actual.shape != reference.shape:
        raise RuntimeError(
            f"shape mismatch: {actual_path} has {tuple(actual.shape)}, "
            f"{reference_path} has {tuple(reference.shape)}"
        )

    shape = tuple(actual.shape)
    global_metrics = Metrics()
    chunks: list[Metrics] = []
    tokens: list[Metrics] = []
    head_totals = [Metrics() for _ in range(shape[head_axis])] if head_axis is not None else []
    top_elements: list[tuple[float, tuple[int, ...], float, float]] = []
    sequence_length = shape[sequence_axis]

    for start in range(0, sequence_length, chunk_size):
        length = min(chunk_size, sequence_length - start)
        actual_chunk = actual.narrow(sequence_axis, start, length)
        reference_chunk = reference.narrow(sequence_axis, start, length)
        chunk_metrics = Metrics()
        _add_metrics(chunk_metrics, actual_chunk, reference_chunk)
        _merge_metrics(global_metrics, chunk_metrics)
        chunks.append(chunk_metrics)

        if not detailed:
            continue
        tokens.extend(_metrics_by_axis(actual_chunk, reference_chunk, sequence_axis))

        if head_axis is not None:
            for total, item in zip(
                head_totals, _metrics_by_axis(actual_chunk, reference_chunk, head_axis)
            ):
                _merge_metrics(total, item)

        difference = (actual_chunk.to(torch.float64) - reference_chunk.to(torch.float64)).abs()
        finite_difference = torch.where(
            torch.isfinite(difference), difference, torch.full_like(difference, math.inf)
        )
        count = min(top_k, finite_difference.numel())
        values, indices = torch.topk(finite_difference.reshape(-1), count)
        for error, flat_index in zip(values.tolist(), indices.tolist()):
            local_index = list(_flat_to_index(int(flat_index), tuple(actual_chunk.shape)))
            local_index[sequence_axis] += start
            index = tuple(local_index)
            actual_value = float(actual[index].to(torch.float64).item())
            reference_value = float(reference[index].to(torch.float64).item())
            top_elements.append((float(error), index, actual_value, reference_value))
        top_elements.sort(key=lambda item: item[0], reverse=True)
        del top_elements[top_k:]

    actual_dtype, reference_dtype = actual.dtype, reference.dtype
    del actual, reference
    gc.collect()
    return shape, actual_dtype, reference_dtype, Detail(
        global_metrics=global_metrics,
        chunks=chunks,
        tokens=tokens,
        heads=head_totals,
        top_elements=top_elements,
    )


def _format_metrics(metrics: Metrics) -> str:
    return (
        f"count={metrics.count} nonfinite={metrics.nonfinite} "
        f"mae={metrics.mae:.9e} rmse={metrics.rmse:.9e} "
        f"max_abs={metrics.max_abs:.9e} relative_l2={metrics.relative_l2:.9e} "
        f"actual_rms={metrics.actual_rms:.9e} reference_rms={metrics.reference_rms:.9e}"
    )


def _is_significant(npu: Metrics, baseline: Metrics, absolute: float, ratio: float) -> bool:
    return npu.rmse > absolute and npu.rmse > ratio * max(baseline.rmse, 1e-30)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-30)


def _print_detail(
    npu: Detail,
    triton: Detail,
    chunk_size: int,
    top_k: int,
    absolute_rmse: float,
    baseline_ratio: float,
) -> None:
    significant_chunks = [
        index
        for index, (npu_item, triton_item) in enumerate(zip(npu.chunks, triton.chunks))
        if _is_significant(npu_item, triton_item, absolute_rmse, baseline_ratio)
    ]
    first_chunk = significant_chunks[0] if significant_chunks else None
    print(
        "ONSET_RULE "
        f"npu_rmse>{absolute_rmse:.3e} and "
        f"npu_rmse>{baseline_ratio:g}*triton_rmse"
    )
    print(f"FIRST_SIGNIFICANT_CHUNK {first_chunk if first_chunk is not None else 'none'}")

    if first_chunk is not None:
        start = max(0, first_chunk - 2)
        end = min(len(npu.chunks), first_chunk + 6)
    else:
        start, end = 0, min(8, len(npu.chunks))
    print("CHUNK_WINDOW chunk token_start npu_rmse triton_rmse ratio npu_max_abs")
    for index in range(start, end):
        print(
            f"CHUNK {index} {index * chunk_size} "
            f"{npu.chunks[index].rmse:.9e} {triton.chunks[index].rmse:.9e} "
            f"{_ratio(npu.chunks[index].rmse, triton.chunks[index].rmse):.6e} "
            f"{npu.chunks[index].max_abs:.9e}"
        )

    top_chunks = sorted(
        range(len(npu.chunks)), key=lambda index: npu.chunks[index].rmse, reverse=True
    )[:top_k]
    print("TOP_CHUNKS chunk token_start npu_rmse triton_rmse ratio npu_max_abs")
    for index in top_chunks:
        print(
            f"TOP_CHUNK {index} {index * chunk_size} "
            f"{npu.chunks[index].rmse:.9e} {triton.chunks[index].rmse:.9e} "
            f"{_ratio(npu.chunks[index].rmse, triton.chunks[index].rmse):.6e} "
            f"{npu.chunks[index].max_abs:.9e}"
        )

    if first_chunk is not None and npu.tokens and triton.tokens:
        token_start = first_chunk * chunk_size
        token_end = min(token_start + chunk_size, len(npu.tokens))
        significant_tokens = [
            index
            for index in range(token_start, token_end)
            if _is_significant(
                npu.tokens[index], triton.tokens[index], absolute_rmse, baseline_ratio
            )
        ]
        first_token = significant_tokens[0] if significant_tokens else None
        token_label = first_token if first_token is not None else "none"
        print(f"FIRST_SIGNIFICANT_TOKEN_IN_CHUNK {token_label}")
        top_tokens = sorted(
            range(token_start, token_end),
            key=lambda index: npu.tokens[index].rmse,
            reverse=True,
        )[:top_k]
        print("TOP_TOKENS token npu_rmse triton_rmse ratio npu_max_abs")
        for index in top_tokens:
            print(
                f"TOP_TOKEN {index} {npu.tokens[index].rmse:.9e} "
                f"{triton.tokens[index].rmse:.9e} "
                f"{_ratio(npu.tokens[index].rmse, triton.tokens[index].rmse):.6e} "
                f"{npu.tokens[index].max_abs:.9e}"
            )

    if npu.heads and triton.heads:
        top_heads = sorted(
            range(len(npu.heads)), key=lambda index: npu.heads[index].rmse, reverse=True
        )[:top_k]
        print("TOP_HEADS head npu_rmse triton_rmse ratio npu_max_abs")
        for index in top_heads:
            print(
                f"TOP_HEAD {index} {npu.heads[index].rmse:.9e} "
                f"{triton.heads[index].rmse:.9e} "
                f"{_ratio(npu.heads[index].rmse, triton.heads[index].rmse):.6e} "
                f"{npu.heads[index].max_abs:.9e}"
            )

    print("TOP_ELEMENTS rank index npu golden abs_error")
    for rank, (error, index, actual, reference) in enumerate(npu.top_elements, start=1):
        print(
            f"TOP_ELEMENT {rank} {index} {actual:.9e} {reference:.9e} {error:.9e}"
        )


def main() -> int:
    args = _parse_args()
    root = args.result_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    spec = _load_case_spec(args.case_json.expanduser().resolve(), args.case_id)
    run_root, files, complete_count = _select_run(root, args.case_id)
    common_indices = sorted(set.intersection(*(set(files[role]) for role in ROLES)))
    if not common_indices:
        raise RuntimeError(f"selected run has no common output indices: {run_root}")
    names = _visible_output_names(spec, args.visible_outputs)
    chunk_size = int(spec["chunk_size"])

    print("KDA_ATK_DIAGNOSTIC_BEGIN")
    print(f"CASE id={args.case_id} key={spec.get('case_key')} layout={spec.get('layout')}")
    print(
        f"SHAPE B={spec.get('B')} H={spec.get('H')} HV={spec.get('HV')} "
        f"T={spec.get('T')} K={spec.get('K')} V={spec.get('V')} chunk_size={chunk_size}"
    )
    print(f"SELECTED_RUN {run_root}")
    print(f"COMPLETE_RUNS_FOUND {complete_count}")
    for role in ROLES:
        print(f"ROLE {role} {files[role][common_indices[0]].parent.parent.parent}")
    print("OUTPUT_INDEX_MAP " + " ".join(
        f"{index}={names[index] if index < len(names) else 'unknown'}" for index in common_indices
    ))

    summaries: dict[int, dict[str, Metrics]] = {}
    detail_npu = None
    detail_triton = None
    detail_shape = None
    detail_dtypes = None
    for output_index in common_indices:
        probe = _load_tensor(files["golden"][output_index])
        shape = tuple(probe.shape)
        del probe
        gc.collect()
        detailed = output_index == args.detail_output
        preferred_sequence = 1 if len(shape) == 4 else 0
        try:
            sequence_axis = _infer_axis(
                shape, int(spec["T"]), preferred_sequence, "sequence"
            )
            output_chunk_size = chunk_size
        except RuntimeError:
            if detailed:
                raise
            sequence_axis = 0
            output_chunk_size = shape[0]
        head_axis = None
        if detailed:
            sequence_axis = args.sequence_axis if args.sequence_axis is not None else sequence_axis
            preferred_head = 2 if len(shape) == 4 and sequence_axis == 1 else 1
            head_axis = (
                args.head_axis
                if args.head_axis is not None
                else _infer_axis(shape, int(spec["HV"]), preferred_head, "head")
            )

        shape_ng, npu_dtype, golden_dtype, npu_golden = _analyze_pair(
            files["npu"][output_index],
            files["golden"][output_index],
            sequence_axis,
            output_chunk_size,
            head_axis,
            args.top_k,
            detailed,
        )
        _, triton_dtype, _, triton_golden = _analyze_pair(
            files["triton"][output_index],
            files["golden"][output_index],
            sequence_axis,
            output_chunk_size,
            head_axis,
            args.top_k,
            detailed,
        )
        _, _, _, npu_triton = _analyze_pair(
            files["npu"][output_index],
            files["triton"][output_index],
            sequence_axis,
            output_chunk_size,
            None,
            args.top_k,
            False,
        )
        summaries[output_index] = {
            "npu_vs_golden": npu_golden.global_metrics,
            "triton_vs_golden": triton_golden.global_metrics,
            "npu_vs_triton": npu_triton.global_metrics,
        }
        output_name = names[output_index] if output_index < len(names) else "unknown"
        print(
            f"OUTPUT {output_index} name={output_name} shape={shape_ng} "
            f"dtype_npu={npu_dtype} dtype_triton={triton_dtype} dtype_golden={golden_dtype}"
        )
        for pair_name, metrics in summaries[output_index].items():
            print(f"GLOBAL output={output_index} pair={pair_name} {_format_metrics(metrics)}")
        if detailed:
            detail_npu = npu_golden
            detail_triton = triton_golden
            detail_shape = shape_ng
            detail_dtypes = (npu_dtype, triton_dtype, golden_dtype)

    if detail_npu is None or detail_triton is None:
        raise RuntimeError(
            f"--detail-output {args.detail_output} is unavailable; common indices={common_indices}"
        )
    print(
        f"DETAIL output={args.detail_output} shape={detail_shape} dtypes={detail_dtypes} "
        "baseline=triton_vs_golden"
    )
    _print_detail(
        detail_npu,
        detail_triton,
        chunk_size,
        args.top_k,
        args.absolute_rmse,
        args.baseline_ratio,
    )
    print("KDA_ATK_DIAGNOSTIC_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
