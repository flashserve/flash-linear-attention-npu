"""Extract only ChunkKdaFwdFinalize target-kernel samples from one profile file.

GPU input is a PyTorch Chrome trace object with traceEvents.  NPU input is a
CANN op_summary CSV with an explicit microsecond task-duration column.  This
tool does not combine files or infer units from a generic Duration column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path


GPU_TARGET = "chunk_gla_fwd_kernel_o"
GPU_FORBIDDEN = "_chunk_kda_fwd_finalize_kernel"
NPU_TARGET = "ChunkKdaFwdFinalize"
NPU_TYPE_COLUMNS = ("OP Type", "Op Type")
NPU_NAME_COLUMNS = ("OP Name", "Op Name")
NPU_DURATION_COLUMNS = ("Task Duration(us)", "Task Duration (us)")


class ProfileError(ValueError):
    """Missing, ambiguous, or invalid evidence must not produce a score."""


def _positive_duration(value, where, *, csv_value=False):
    if isinstance(value, bool):
        raise ProfileError(f"{where}: duration must be a finite positive number")
    if csv_value:
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"{where}: missing duration")
        try:
            duration = float(value.strip())
        except ValueError as exc:
            raise ProfileError(f"{where}: invalid microsecond duration {value!r}") from exc
    else:
        if not isinstance(value, (int, float)):
            raise ProfileError(f"{where}: missing or nonnumeric duration")
        try:
            duration = float(value)
        except OverflowError as exc:
            raise ProfileError(f"{where}: duration exceeds the finite numeric range") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ProfileError(f"{where}: duration must be finite and positive, got {value!r}")
    return duration


def _gpu_samples(raw):
    try:
        trace = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"invalid GPU trace JSON: {exc}") from exc
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ProfileError("GPU trace must be an object containing a traceEvents list")
    samples, names = [], set()
    for index, event in enumerate(trace["traceEvents"]):
        if not isinstance(event, dict):
            raise ProfileError(f"traceEvents[{index}] is not an event object")
        if event.get("cat") != "kernel":
            continue
        name = event.get("name")
        if not isinstance(name, str):
            raise ProfileError(f"kernel event {index} has no string name")
        if GPU_FORBIDDEN in name:
            raise ProfileError(f"adapted contract kernel is forbidden in GPU evidence: {name}")
        if event.get("ph") != "X" or GPU_TARGET not in name:
            continue
        samples.append(_positive_duration(event.get("dur"), f"GPU target event {index}"))
        names.add(name)
    return samples, names


def _npu_samples(raw):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ProfileError(f"NPU CSV is not UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    try:
        fields = reader.fieldnames
    except csv.Error as exc:
        raise ProfileError(f"invalid NPU CSV header: {exc}") from exc
    if not fields:
        raise ProfileError("NPU CSV has no header")
    headers = [field.strip() for field in fields]
    if len(set(headers)) != len(headers):
        raise ProfileError("NPU CSV has duplicate columns")
    reader.fieldnames = headers
    type_columns = [name for name in NPU_TYPE_COLUMNS if name in headers]
    name_columns = [name for name in NPU_NAME_COLUMNS if name in headers]
    duration_column = next((name for name in NPU_DURATION_COLUMNS if name in headers), None)
    if not type_columns and not name_columns:
        raise ProfileError("unknown NPU CSV schema: expected OP Type/Op Type or OP Name/Op Name")
    if duration_column is None:
        raise ProfileError("unknown NPU duration unit: expected Task Duration(us)")
    samples, names = [], set()
    try:
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ProfileError(f"NPU CSV line {line} does not match its header")
            is_target = any(row[name].strip() == NPU_TARGET for name in type_columns)
            is_target = is_target or any(NPU_TARGET in row[name] for name in name_columns)
            if not is_target:
                continue
            samples.append(_positive_duration(row[duration_column], f"NPU target line {line}",
                                               csv_value=True))
            target_names = [row[name].strip() for name in name_columns if row[name].strip()]
            names.update(target_names or [NPU_TARGET])
    except csv.Error as exc:
        raise ProfileError(f"invalid NPU CSV: {exc}") from exc
    return samples, names


def check_profile(device, input_path, expected_count=10):
    path = Path(input_path).resolve()
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ProfileError("expected_count must be a positive integer")
    if not path.is_file():
        raise ProfileError(f"input must name one existing profile file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProfileError(f"cannot read profile file {path}: {exc}") from exc
    if device == "gpu":
        samples, names = _gpu_samples(raw)
    elif device == "npu":
        samples, names = _npu_samples(raw)
    else:
        raise ProfileError(f"unsupported device: {device}")
    if len(samples) != expected_count:
        raise ProfileError(f"expected {expected_count} target samples, found {len(samples)}")
    return {
        "passed": True,
        "device": device,
        "scope": "target_kernel",
        "time_unit": "us",
        "input_path": str(path),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "target_names": sorted(names),
        "expected_count": expected_count,
        "sample_count": len(samples),
        "all_samples": samples,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu", "npu"), required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="one GPU trace JSON or NPU op_summary CSV; directories are rejected")
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        print("ERROR: output must not overwrite the input profile", file=sys.stderr)
        return 1
    try:
        result = check_profile(args.device, args.input, args.expected_count)
    except (ProfileError, OSError) as exc:
        result = {
            "passed": False,
            "device": args.device,
            "scope": "target_kernel",
            "input_path": str(args.input.resolve()),
            "expected_count": args.expected_count,
            "error": str(exc),
        }
        print(f"ERROR: {exc}", file=sys.stderr)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot write result: {exc}", file=sys.stderr)
        return 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
