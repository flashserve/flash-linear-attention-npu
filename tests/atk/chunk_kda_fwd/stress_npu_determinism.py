#!/usr/bin/env python3
"""Run fixed-input NPU determinism stress for modern ``chunk_kda_fwd``.

The mss manifest is the source of truth for this diagnostic.  By default all
records are executed, and the manifest must contain both host tiling keys.
The diagnostic intentionally does not run a CPU reference: every output from
every repeat is compared byte-for-byte with the first repeat.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional


_SOC_ALIASES = {
    "all": "all",
    "a2": "ascend910b",
    "a3": "ascend910_93",
    "a5": "ascend950",
    "ascend910b": "ascend910b",
    "ascend910_93": "ascend910_93",
    "ascend950": "ascend950",
}
_VALID_ROUTES = {"ascendc", "aclnn", "direct_launch"}
_TILING_KEYS = {1, 2}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chunk_kda_fwd repeatedly for every selected MSS case and "
            "compare every output bit against run 0."
        )
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--soc",
        default="all",
        help="restrict records to one SoC (ascend910b, ascend910_93, ascend950, or all)",
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=None,
        help="run one source manifest case instead of the complete MSS selection",
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--tokens",
        type=int,
        help="override T for a dense boundary stress; varlen records are rejected",
    )
    parser.add_argument(
        "--case-json",
        type=Path,
        default=Path(__file__).with_name("atk_chunk_kda_fwd_mss.json"),
    )
    return parser.parse_args()


def _normalise_soc(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _SOC_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_SOC_ALIASES))
        raise ValueError(f"unsupported --soc {value!r}; expected one of {choices}") from exc


def _case_spec(case: dict[str, Any], path: Path) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError(f"manifest entry in {path} is not an object")
    if "id" not in case:
        raise ValueError(f"manifest entry in {path} has no id")
    try:
        case_id = int(case["id"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest entry has a non-integer id: {case.get('id')!r}") from exc
    inputs = case.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"case {case_id} in {path} has no inputs list")
    candidates = [
        item for item in inputs
        if isinstance(item, dict) and item.get("name") == "case_spec"
    ]
    if len(candidates) != 1:
        raise ValueError(f"case {case_id} in {path} must contain exactly one case_spec input")
    raw = candidates[0].get("range_values")
    if isinstance(raw, str):
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"case {case_id} in {path} has invalid case_spec JSON") from exc
    elif isinstance(raw, dict):
        spec = raw
    else:
        raise ValueError(f"case {case_id} in {path} has an invalid case_spec value")
    if not isinstance(spec, dict):
        raise ValueError(f"case {case_id} in {path} case_spec is not an object")
    spec = dict(spec)
    if int(spec.get("case_id", case_id)) != case_id:
        raise ValueError(
            f"case {case_id} in {path} disagrees with case_spec.case_id={spec.get('case_id')!r}"
        )
    spec["case_id"] = case_id
    return spec


def _host_tiling_key(spec: dict[str, Any]) -> int:
    try:
        chunk_size = int(spec["chunk_size"])
        key_dim = int(spec["K"])
        value_dim = int(spec["V"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"case {spec.get('case_id')} lacks numeric chunk/K/V fields") from exc
    return 2 if (chunk_size, key_dim, value_dim) == (64, 128, 128) else 1


def _matches_soc(spec: dict[str, Any], requested: str) -> bool:
    if requested == "all":
        return True
    declared = str(spec.get("soc", "all")).strip().lower()
    if declared in {"all", requested.lower()}:
        return True
    platforms = spec.get("target_platforms", ())
    return isinstance(platforms, (list, tuple, set)) and requested in {
        str(item).strip().lower() for item in platforms
    }


def _load_specs(
    path: Path,
    *,
    soc: str = "all",
    case_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MSS manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"MSS manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"MSS manifest must be a non-empty JSON list: {path}")

    requested = _normalise_soc(soc)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for case in payload:
        spec = _case_spec(case, path)
        current_id = int(spec["case_id"])
        if case_id is not None and current_id != int(case_id):
            continue
        tags = {
            tag.strip() for tag in str(spec.get("tags", "")).split(",") if tag.strip()
        }
        if "negative" in tags or bool(spec.get("negative_case", False)):
            raise ValueError(f"MSS manifest contains a negative case {current_id}")
        if "mss" not in tags and str(spec.get("manifest", "mss")) != "mss":
            raise ValueError(f"case {current_id} is not marked as an MSS case")
        key = int(spec.get("tiling_key", -1))
        expected_key = int(spec.get("expected_tiling_key", -1))
        actual_key = _host_tiling_key(spec)
        if key not in _TILING_KEYS or expected_key != key or actual_key != key:
            raise ValueError(
                f"case {current_id} has inconsistent tiling key: "
                f"manifest={key}, expected={expected_key}, host={actual_key}"
            )
        route = str(spec.get("route", ""))
        if route not in _VALID_ROUTES:
            raise ValueError(f"case {current_id} has unsupported route {route!r}")
        if not _matches_soc(spec, requested):
            continue
        declared_soc = str(spec.get("soc", "all"))
        identity = (current_id, declared_soc, key)
        if identity in seen:
            raise ValueError(f"duplicate MSS case identity {identity}")
        seen.add(identity)
        selected.append(spec)

    if not selected:
        selector = f"case_id={case_id}" if case_id is not None else "the requested SoC"
        raise ValueError(f"no MSS cases selected for {selector}")
    selected.sort(key=lambda item: (int(item["case_id"]), int(item["tiling_key"])))
    if case_id is None and {int(item["tiling_key"]) for item in selected} != _TILING_KEYS:
        raise ValueError(
            "complete MSS selection must cover tiling keys {1, 2}; "
            f"selected {sorted({int(item['tiling_key']) for item in selected})}"
        )
    return selected


def _override_tokens(spec: dict[str, Any], tokens: Optional[int]) -> dict[str, Any]:
    if tokens is None:
        return dict(spec)
    value = int(tokens)
    chunk_size = int(spec["chunk_size"])
    if value <= 0 or value % chunk_size != 0:
        raise ValueError("--tokens must be a positive multiple of chunk_size")
    if str(spec.get("cu_seqlens", "")).strip():
        raise ValueError("--tokens cannot override a varlen MSS case")
    updated = dict(spec)
    updated["T"] = value
    updated["case_key"] = f"{updated.get('case_key', updated['case_id'])}_t{value}_stress"
    return updated


def _clone_prepared_inputs(torch, inputs):
    """Clone every tensor in the executor dataclass before each launch."""
    if not dataclasses.is_dataclass(inputs):
        raise TypeError("chunk_kda_fwd executor inputs are not a dataclass")
    values = {}
    for field in dataclasses.fields(inputs):
        value = getattr(inputs, field.name)
        if isinstance(value, torch.Tensor):
            values[field.name] = value.clone()
        elif isinstance(value, list):
            values[field.name] = list(value)
        elif isinstance(value, tuple):
            values[field.name] = tuple(value)
        else:
            values[field.name] = value
    return dataclasses.replace(inputs, **values)


def _output_name(index: int, names: Iterable[str]) -> str:
    names = tuple(names)
    return names[index] if index < len(names) else f"output_{index}"


def _ensure_finite(torch, tensor, label: str) -> None:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label} contains NaN or Inf")


def _snapshot_outputs(torch, outputs, names: Iterable[str]):
    names = tuple(names)
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError(f"executor returned {type(outputs).__name__}, expected a tuple/list")
    if len(outputs) != len(names):
        raise RuntimeError(
            f"executor returned {len(outputs)} outputs, expected {len(names)}"
        )
    snapshot = []
    for index, output in enumerate(outputs):
        name = _output_name(index, names)
        if output is None:
            snapshot.append(None)
            continue
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"{name} has unsupported output type {type(output).__name__}")
        _ensure_finite(torch, output, name)
        snapshot.append(output.detach().clone())
    return tuple(snapshot)


def _max_abs_delta(torch, current, baseline) -> Optional[float]:
    if not (current.is_floating_point() or current.is_complex()):
        return None
    if current.numel() == 0:
        return 0.0
    delta = (current.to(torch.float32) - baseline.to(torch.float32)).abs()
    value = float(delta.max().item())
    del delta
    return value


def _compare_outputs(torch, current, baseline, names: Iterable[str]) -> list[dict[str, Any]]:
    names = tuple(names)
    if not isinstance(current, (tuple, list)):
        raise RuntimeError(f"executor returned {type(current).__name__}, expected a tuple/list")
    if len(current) != len(names):
        raise RuntimeError(
            f"executor returned {len(current)} outputs, expected {len(names)}"
        )
    if len(current) != len(baseline):
        raise RuntimeError(
            f"output count changed between repeats: {len(current)} != {len(baseline)}"
        )
    differences: list[dict[str, Any]] = []
    for index, (value, reference) in enumerate(zip(current, baseline)):
        name = _output_name(index, names)
        if value is None or reference is None:
            if value is not None or reference is not None:
                differences.append(
                    {"name": name, "reason": "optional_output_presence_changed"}
                )
            continue
        if not isinstance(value, torch.Tensor) or not isinstance(reference, torch.Tensor):
            differences.append({"name": name, "reason": "output_type_changed"})
            continue
        _ensure_finite(torch, value, name)
        if value.dtype != reference.dtype or tuple(value.shape) != tuple(reference.shape):
            differences.append(
                {
                    "name": name,
                    "reason": "dtype_or_shape_changed",
                    "current_dtype": str(value.dtype),
                    "baseline_dtype": str(reference.dtype),
                    "current_shape": list(value.shape),
                    "baseline_shape": list(reference.shape),
                }
            )
            continue
        if value.device != reference.device:
            differences.append(
                {
                    "name": name,
                    "reason": "device_changed",
                    "current_device": str(value.device),
                    "baseline_device": str(reference.device),
                }
            )
            continue
        current_bytes = value.detach().contiguous().view(torch.uint8)
        baseline_bytes = reference.detach().contiguous().view(torch.uint8)
        if not bool(torch.equal(current_bytes, baseline_bytes)):
            mismatch_bytes = int(torch.count_nonzero(current_bytes != baseline_bytes).item())
            differences.append(
                {
                    "name": name,
                    "reason": "bitwise_mismatch",
                    "mismatched_bytes": mismatch_bytes,
                    "max_abs_delta": _max_abs_delta(torch, value, reference),
                }
            )
        del current_bytes, baseline_bytes
    return differences


def _run_case(
    torch,
    spec: dict[str, Any],
    *,
    device,
    repeats: int,
    prepare_inputs,
    run_positive_npu,
    output_names: Iterable[str],
) -> bool:
    low_marker = torch.zeros(1, dtype=torch.bfloat16, device=device)
    fp32_marker = torch.zeros(1, dtype=torch.float32, device=device)
    template_inputs = prepare_inputs(spec, low_marker, fp32_marker, high_precision=False)
    torch.npu.synchronize()

    case_id = int(spec["case_id"])
    case_key = str(spec.get("case_key", case_id))
    print(
        "CASE_BEGIN "
        + json.dumps(
            {
                "case_id": case_id,
                "case_key": case_key,
                "tiling_key": int(spec["tiling_key"]),
                "soc": str(spec.get("soc", "all")),
                "device": str(device),
                "repeats": repeats,
                "seed": int(spec["seed"]),
                "shape": {
                    name: int(spec[name])
                    for name in ("B", "H", "HV", "T", "K", "V", "chunk_size")
                },
                "comparison": "all_outputs_bitwise_against_run_0",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    baseline = None
    mismatch_runs: list[int] = []
    started = time.perf_counter()
    with torch.no_grad():
        for repeat in range(repeats):
            run_inputs = _clone_prepared_inputs(torch, template_inputs)
            launch_started = time.perf_counter()
            outputs = run_positive_npu(run_inputs, spec)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - launch_started
            if baseline is None:
                baseline = _snapshot_outputs(torch, outputs, output_names)
                record: dict[str, Any] = {
                    "run": repeat,
                    "elapsed_seconds": elapsed,
                    "binary_equal": True,
                    "baseline": True,
                    "outputs_compared": len(baseline),
                }
            else:
                differences = _compare_outputs(torch, outputs, baseline, output_names)
                equal = not differences
                if not equal:
                    mismatch_runs.append(repeat)
                record = {
                    "run": repeat,
                    "elapsed_seconds": elapsed,
                    "binary_equal": equal,
                    "differences": differences,
                    "outputs_compared": len(baseline),
                }
            print("RUN " + json.dumps(record, sort_keys=True), flush=True)
            del outputs, run_inputs
            gc.collect()

    passed = not mismatch_runs
    print(
        "CASE_SUMMARY "
        + json.dumps(
            {
                "case_id": case_id,
                "case_key": case_key,
                "tiling_key": int(spec["tiling_key"]),
                "repeats": repeats,
                "equal_to_run_0": repeats - len(mismatch_runs),
                "different_from_run_0": len(mismatch_runs),
                "mismatch_runs": mismatch_runs,
                "elapsed_seconds": time.perf_counter() - started,
                "passed": passed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del baseline, template_inputs, low_marker, fp32_marker
    gc.collect()
    return passed


def main() -> int:
    args = _parse_args()
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    requested_soc = _normalise_soc(args.soc)
    manifest = args.case_json.expanduser().resolve()
    specs = _load_specs(manifest, soc=requested_soc, case_id=args.case_id)
    specs = [_override_tokens(spec, args.tokens) for spec in specs]

    import torch
    import torch_npu  # noqa: F401

    from executor_chunk_kda_fwd import (
        _OUTPUT_NAMES,
        _prepare_inputs,
        _run_positive_npu,
    )

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    print("KDA_NPU_BINARY_STRESS_BEGIN", flush=True)
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for spec in specs:
        try:
            passed = _run_case(
                torch,
                spec,
                device=device,
                repeats=args.repeats,
                prepare_inputs=_prepare_inputs,
                run_positive_npu=_run_positive_npu,
                output_names=_OUTPUT_NAMES,
            )
            if not passed:
                failures.append(
                    {
                        "case_id": int(spec["case_id"]),
                        "tiling_key": int(spec["tiling_key"]),
                        "reason": "bitwise_mismatch",
                    }
                )
        except Exception as exc:  # Fail closed, but continue to report other selected cases.
            failure = {
                "case_id": int(spec.get("case_id", -1)),
                "tiling_key": int(spec.get("tiling_key", -1)),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print("CASE_ERROR " + json.dumps(failure, sort_keys=True), flush=True)
    summary = {
        "manifest": manifest.name,
        "soc": requested_soc,
        "case_count": len(specs),
        "tiling_keys": sorted({int(spec["tiling_key"]) for spec in specs}),
        "passed_cases": len(specs) - len(failures),
        "failed_cases": len(failures),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print("KDA_NPU_BINARY_STRESS_END", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
