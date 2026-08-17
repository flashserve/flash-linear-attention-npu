#!/usr/bin/env python3
"""Execute and verify canonical non-accuracy ChunkKdaFwd cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from canonical_execution_adapter import (
    DEFAULT_MANIFEST,
    SUPPORTED_SOCS,
    materialize,
    project_records,
)
from persistent_reference_cache import (
    PinnedCatalog,
    ReferenceCacheError,
    VARIANT_MATERIALIZER_SCHEMA,
    default_catalog_reference,
)


HERE = Path(__file__).resolve().parent
EXECUTOR_PATH = HERE / "executor_chunk_kda_fwd.py"
EXIT_NOT_APPLICABLE = 3
SANITIZER_TOOLS = ("racecheck", "memcheck", "initcheck", "synccheck")
_DURATION_COLUMNS = (
    ("duration_us", 1.0),
    ("Duration(us)", 1.0),
    ("Task Duration(us)", 1.0),
    ("duration_ns", 0.001),
    ("Duration(ns)", 0.001),
    ("duration_ms", 1000.0),
    ("Duration(ms)", 1000.0),
)
_KERNEL_COLUMNS = ("kernel_name", "Kernel Name", "Op Name", "Name", "Task Name")
_OP_TYPE_COLUMNS = ("op_type", "Op Type", "OpType")
_PRIMARY_MIX_SUFFIX = "_2_mix_aic"
_MIX_AIV_SUFFIX = "_mix_aiv"
_UNAVAILABLE_DURATIONS = {"", "--", "NA", "N/A"}
_PROFILER_FAILURE_PATTERNS = (
    re.compile(r"507015", re.IGNORECASE),
    re.compile(r"RunDbiRecordTask[^\n]*(?:failed|error)", re.IGNORECASE),
    re.compile(r"Get profiling data failed", re.IGNORECASE),
    re.compile(r"\bDBI\b[^\n]*(?:failed|error)", re.IGNORECASE),
    re.compile(r"\btun(?:e|ing)\b[^\n]*(?:failed|error)", re.IGNORECASE),
    re.compile(r"\bAIC[_ -]?ERROR\b", re.IGNORECASE),
    re.compile(
        r"\bAICORE\b[^\n]*(?:error|fault|exception|fail)", re.IGNORECASE
    ),
    re.compile(r"\bnon[- ]?finite\b|\bNaN\b|\bInf(?:inity)?\b", re.IGNORECASE),
)


def _default_cache_dir() -> Path:
    configured = os.environ.get("KDA_ATK_PERSISTENT_CACHE_DIR", "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "fla_npu" / "chunk_kda_fwd_atk"
    )


def _pinned_execution_catalog(args) -> PinnedCatalog:
    catalog = PinnedCatalog(args.cache_dir, args.catalog)
    adapter = catalog.catalog.get("adapter")
    if adapter not in {
        "canonical_execution_adapter:materialize",
        "canonical_execution_adapter:materialize_all",
    }:
        raise ReferenceCacheError(
            "canonical execution requires a catalog built by the canonical execution adapter"
        )
    catalog.validate_source(
        args.source.resolve(),
        adapter=adapter,
        adapter_path=Path(__file__).resolve().parent / "canonical_execution_adapter.py",
        variant_materializer_schema=VARIANT_MATERIALIZER_SCHEMA,
    )
    return catalog


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _logical_record(path: Path, design_id: str) -> dict:
    matches = [
        record for record in materialize(path) if record["spec"]["design_id"] == design_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one logical record for {design_id}, got {len(matches)}")
    return matches[0]


def _physical_record(
    path: Path,
    design_id: str,
    soc: str,
    *,
    variant: str,
    sanitizer_tool: str | None = None,
) -> dict | None:
    logical = _logical_record(path, design_id)
    records = project_records(
        path,
        kind=logical["spec"]["execution_kind"],
        soc=soc,
        route="ascendc",
        variant=variant,
        sanitizer_tool=sanitizer_tool,
        include_not_applicable=True,
    )
    records = [record for record in records if record["spec"]["design_id"] == design_id]
    if len(records) != 1:
        raise ValueError(f"expected one physical record for {design_id}/{variant}, got {len(records)}")
    if records[0]["spec"].get("status") == "not_applicable":
        return None
    return records[0]


def _load_cached_inputs(
    logical_spec: dict,
    physical_spec: dict,
    catalog: PinnedCatalog,
    device,
):
    from executor_chunk_kda_fwd import (
        _apply_input_storage,
        _prepared_inputs_from_cpu,
        _select_cached_input_payload,
    )
    reader = catalog.reader_for(
        logical_spec,
        int(logical_spec["seed"]),
        EXECUTOR_PATH,
        include_references=False,
    )
    payload = _select_cached_input_payload(reader.load_shard("inputs"), physical_spec)
    inputs = _prepared_inputs_from_cpu(payload, device, high_precision=False)
    if inputs.seed != int(physical_spec["seed"]):
        raise RuntimeError(
            f"cached input seed {inputs.seed} does not match {physical_spec['seed']}"
        )
    return _apply_input_storage(inputs, physical_spec)


def _application_command(
    *,
    python: str,
    source: Path,
    cache_dir: Path,
    catalog_reference: str,
    design_id: str,
    soc: str,
    variant: str,
    device: int,
    repeats: int = 1,
) -> list[str]:
    if not catalog_reference:
        raise ValueError(
            "a cache catalog must be explicitly pinned with --catalog or "
            "KDA_ATK_PERSISTENT_CACHE_CATALOG"
        )
    return [
        python,
        str(Path(__file__).resolve()),
        "application",
        "--source",
        str(source),
        "--cache-dir",
        str(cache_dir),
        "--catalog",
        str(catalog_reference),
        "--design-id",
        design_id,
        "--soc",
        soc,
        "--variant",
        variant,
        "--device",
        str(device),
        "--repeats",
        str(repeats),
    ]


def materialize_msopprof_command(
    spec: dict,
    *,
    source: Path,
    cache_dir: Path,
    catalog_reference: str,
    output: Path,
    python: str,
    device: int,
) -> list[str]:
    profiler = spec["profiler"]
    application = _application_command(
        python=python,
        source=source,
        cache_dir=cache_dir,
        catalog_reference=catalog_reference,
        design_id=spec["design_id"],
        soc=spec["soc"],
        variant=spec["materialized_variant"],
        device=device,
        repeats=int(profiler["launch_count"]) + int(profiler["warm_up"]),
    )
    command = [
        "msopprof",
        f"--application={_shell_join(application)}",
        f"--output={output}",
        f"--aic-metrics={profiler['aic_metrics']}",
        f"--launch-count={profiler['launch_count']}",
        f"--warm-up={profiler['warm_up']}",
        f"--replay-mode={profiler['replay_mode']}",
        f"--kill={profiler['kill']}",
    ]
    if profiler.get("kernel_name"):
        command.append(f"--kernel-name={profiler['kernel_name']}")
    return command


def _record_kernel_name(record: dict) -> str:
    for column in _KERNEL_COLUMNS:
        value = record.get(column)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"profiler row has no supported kernel column: {sorted(record)}")


def _record_op_type(record: dict) -> str:
    for column in _OP_TYPE_COLUMNS:
        value = record.get(column)
        if value not in (None, ""):
            return str(value).strip().lower()
    raise ValueError(f"profiler row has no supported op type column: {sorted(record)}")


def _duration_field(record: dict):
    fields = [
        (column, multiplier, str(record.get(column, "")).strip())
        for column, multiplier in _DURATION_COLUMNS
        if column in record
    ]
    if len(fields) != 1:
        raise ValueError(
            "profiler row must have exactly one explicit duration unit column: "
            f"{sorted(record)}"
        )
    return fields[0]


def _record_duration_us(record: dict) -> float:
    column, multiplier, raw_value = _duration_field(record)
    if raw_value.upper() in _UNAVAILABLE_DURATIONS:
        raise ValueError(f"profiler target row has no duration in {column}")
    value = float(raw_value.replace(",", "")) * multiplier
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"profiler target duration must be positive and finite: {raw_value!r}")
    return value


def _record_has_unavailable_duration(record: dict) -> bool:
    column, _, raw_value = _duration_field(record)
    if raw_value.upper() in _UNAVAILABLE_DURATIONS:
        return True
    try:
        value = float(raw_value.replace(",", ""))
    except ValueError as error:
        raise ValueError(
            f"profiler companion has invalid duration {raw_value!r} in {column}"
        ) from error
    raise ValueError(f"profiler companion has numeric duration {value!r} in {column}")


def _json_records(value):
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            yield from value
        else:
            for item in value:
                yield from _json_records(item)
    elif isinstance(value, dict):
        if any(key in value for key in _KERNEL_COLUMNS) and any(
            key in value for key, _ in _DURATION_COLUMNS
        ):
            yield value
        else:
            for item in value.values():
                yield from _json_records(item)


def _read_profiler_records(path: Path):
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(
            [
                *path.rglob("OpBasicInfo*.csv"),
                *path.rglob("OpBasicInfo*.json"),
            ]
        )
    else:
        raise ValueError(f"profiler report does not exist: {path}")
    records = []
    for item in paths:
        if not item.parent.name.isdigit():
            raise ValueError(
                f"profiler BasicInfo file is not in a numeric replay directory: {item}"
            )
        replay_index = int(item.parent.name)
        if item.suffix.lower() == ".json":
            item_records = list(
                _json_records(json.loads(item.read_text(encoding="utf-8")))
            )
        elif item.suffix.lower() == ".csv":
            with item.open("r", encoding="utf-8-sig", newline="") as stream:
                item_records = list(csv.DictReader(stream))
        else:
            continue
        if not item_records:
            raise ValueError(f"empty profiler BasicInfo report: {item}")
        records.extend((replay_index, item, record) for record in item_records)
    if not records:
        raise ValueError(f"no structured profiler BasicInfo CSV/JSON records found in {path}")
    return records


def _primary_mix_runtime_name(kernel_json: Path) -> str:
    payload = json.loads(kernel_json.read_text(encoding="utf-8"))
    stem = payload.get("kernelName")
    if not isinstance(stem, str) or not stem:
        raise ValueError(f"kernel JSON has no non-empty kernelName: {kernel_json}")
    if payload.get("binFileName") != stem:
        raise ValueError(f"kernel JSON binFileName does not match kernelName: {kernel_json}")
    kernel_list = payload.get("kernelList")
    if not isinstance(kernel_list, list) or not kernel_list:
        raise ValueError(f"kernel JSON has no non-empty kernelList: {kernel_json}")
    names = []
    for entry in kernel_list:
        if not isinstance(entry, dict) or not isinstance(entry.get("kernelName"), str):
            raise ValueError(f"kernel JSON has an invalid kernelList entry: {kernel_json}")
        names.append(entry["kernelName"])
    if len(names) != len(set(names)):
        raise ValueError(f"kernel JSON contains duplicate kernel names: {kernel_json}")
    primary = f"{stem}_2"
    if names.count(primary) != 1:
        raise ValueError(
            f"kernel JSON must contain exactly one primary MIX entry {primary!r}: {kernel_json}"
        )
    return f"{primary}_mix_aic"


def _expected_mix_runtime_names(kernel_jsons):
    if not kernel_jsons:
        raise ValueError("at least one kernel JSON is required")
    names = tuple(_primary_mix_runtime_name(path) for path in kernel_jsons)
    if len(names) != len(set(names)):
        raise ValueError(f"kernel JSONs resolve to duplicate primary MIX names: {names}")
    return names


def _validate_profiler_log(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"profiler log does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in _PROFILER_FAILURE_PATTERNS:
        match = pattern.search(text)
        if match:
            raise ValueError(f"profiler log contains failure marker {match.group(0)!r}")


def parse_msopprof_report(
    path: Path,
    *,
    kernel_jsons,
    log: Path,
    launch_count: int,
) -> dict:
    if launch_count <= 0:
        raise ValueError("launch_count must be positive")
    expected_names = _expected_mix_runtime_names(kernel_jsons)
    expected_set = set(expected_names)
    companions = {
        name[: -len("_mix_aic")] + _MIX_AIV_SUFFIX for name in expected_names
    }
    _validate_profiler_log(log)

    replay_stages = {}
    replay_companions = {}
    for replay_index, source, record in _read_profiler_records(path):
        kernel_name = _record_kernel_name(record).strip()
        op_type = _record_op_type(record)
        if kernel_name in expected_set:
            if op_type != "mix":
                raise ValueError(
                    f"expected kernel {kernel_name!r} has op_type={op_type!r}, not 'mix'"
                )
            stages = replay_stages.setdefault(replay_index, {})
            if kernel_name in stages:
                raise ValueError(
                    f"replay {replay_index} has duplicate stage {kernel_name!r}: {source}"
                )
            stages[kernel_name] = _record_duration_us(record)
        elif kernel_name in companions:
            if op_type != "mix":
                raise ValueError(
                    f"companion {kernel_name!r} has op_type={op_type!r}, not 'mix'"
                )
            _record_has_unavailable_duration(record)
            seen = replay_companions.setdefault(replay_index, set())
            if kernel_name in seen:
                raise ValueError(
                    f"replay {replay_index} has duplicate companion {kernel_name!r}: {source}"
                )
            seen.add(kernel_name)
        else:
            raise ValueError(
                f"replay {replay_index} contains unknown profiler row "
                f"{kernel_name!r}/{op_type!r}: {source}"
            )

    replay_indices = sorted(set(replay_stages) | set(replay_companions))
    expected_indices = list(range(launch_count))
    if replay_indices != expected_indices:
        raise ValueError(
            f"profiler replay indices must be contiguous {expected_indices}, got {replay_indices}"
        )
    replay_durations = []
    for replay_index in expected_indices:
        stages = replay_stages.get(replay_index, {})
        missing = [name for name in expected_names if name not in stages]
        extra = [name for name in stages if name not in expected_set]
        if missing or extra or len(stages) != len(expected_names):
            raise ValueError(
                f"replay {replay_index} stage set mismatch: missing={missing}, extra={extra}"
            )
        replay_durations.append(sum(stages[name] for name in expected_names))
    if any(not math.isfinite(value) or value <= 0.0 for value in replay_durations):
        raise ValueError("profiler replay duration is not positive and finite")
    total = sum(replay_durations)
    ordered_durations = sorted(replay_durations)
    p95_rank = math.ceil(0.95 * len(ordered_durations))
    return {
        "source": "msopprof_structured_report",
        "kernel_names": list(expected_names),
        "stage_count": len(expected_names),
        "kernel_rows": launch_count * len(expected_names),
        "launch_count": launch_count,
        "replay_indices": replay_indices,
        "total_device_duration_us": total,
        "mean_application_us": statistics.mean(replay_durations),
        "median_application_us": statistics.median(replay_durations),
        "p95_application_us": ordered_durations[p95_rank - 1],
        "min_application_us": min(replay_durations),
        "max_application_us": max(replay_durations),
        "replay_duration_us": replay_durations,
    }


def evaluate_performance(spec: dict, current: dict, baseline: dict | None = None) -> dict:
    expectation = spec["performance_expectation"]
    current_us = float(current["mean_application_us"])
    checks = []
    absolute_ms = expectation.get("absolute_ms_lt")
    if absolute_ms is not None:
        checks.append(
            {
                "name": "absolute_ms_lt",
                "limit": float(absolute_ms),
                "actual": current_us / 1000.0,
                "passed": current_us / 1000.0 < float(absolute_ms),
            }
        )
    relative = expectation.get("max_relative_regression")
    if relative is not None:
        if baseline is None:
            raise ValueError("relative performance expectation requires a baseline report")
        baseline_us = float(baseline["mean_application_us"])
        relative_limit = float(relative)
        regression = current_us / baseline_us - 1.0
        checks.append(
            {
                "name": "max_relative_regression",
                "limit": relative_limit,
                "actual": regression,
                "passed": current_us <= baseline_us * (1.0 + relative_limit),
            }
        )
    status = (
        "measured"
        if not checks
        else ("passed" if all(check["passed"] for check in checks) else "failed")
    )
    return {
        "design_id": spec["design_id"],
        "variant": spec["materialized_variant"],
        "status": status,
        "checks": checks,
        "profiler_result": current,
    }


def verify_object_symbol(nm_text: str, symbol_regex: str) -> None:
    if not re.search(symbol_regex, nm_text, re.IGNORECASE | re.MULTILINE):
        raise RuntimeError(
            f"operator object does not contain required sanitizer symbol {symbol_regex!r}"
        )


_TOOL_FAILURE_PATTERNS = {
    "racecheck": (r"\bdata race detected\b", r"\bracecheck error\b"),
    "memcheck": (r"\binvalid (?:read|write|access)\b", r"\bout[- ]of[- ]bounds\b", r"\bmemory leak detected\b"),
    "initcheck": (r"\buninitialized (?:read|access|value)\b", r"\binitcheck error\b"),
    "synccheck": (r"\bsynchronization error\b", r"\bsynccheck error\b", r"\bbarrier divergence\b"),
}


def verify_sanitizer_log(text: str, *, tool: str, kernel_regex: str) -> dict:
    if tool not in SANITIZER_TOOLS:
        raise ValueError(f"unsupported sanitizer tool: {tool}")
    if re.search(r"No active sanitizer tool on kernel", text, re.IGNORECASE):
        raise RuntimeError("sanitizer log reports no active sanitizer tool")
    active = re.search(
        rf"Start\s+{re.escape(tool)}\s+sanitizer\s+on\s+kernel[^\n]*{kernel_regex}",
        text,
        re.IGNORECASE,
    )
    if active is None:
        raise RuntimeError(
            f"sanitizer log does not prove {tool} actually hit kernel {kernel_regex!r}"
        )
    failures = [
        pattern
        for pattern in (r"\bFATAL\b", r"\bSanitizer ERROR\b", *_TOOL_FAILURE_PATTERNS[tool])
        if re.search(pattern, text, re.IGNORECASE)
    ]
    if failures:
        raise RuntimeError(f"sanitizer reported failures matching {failures}")
    return {"status": "passed", "tool": tool, "kernel_regex": kernel_regex, "active": True}


def _tensor_bits(torch, tensor):
    integer_dtype = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64,
    }[tensor.element_size()]
    return tensor.detach().contiguous().view(integer_dtype)


def compare_outputs_bitwise(torch, baseline: tuple, current: tuple) -> list[str]:
    from executor_chunk_kda_fwd import _OUTPUT_NAMES

    if len(baseline) != len(_OUTPUT_NAMES) or len(current) != len(_OUTPUT_NAMES):
        raise RuntimeError("chunk_kda_fwd output tuple has an unexpected length")
    mismatches = []
    for name, expected, actual in zip(_OUTPUT_NAMES, baseline, current):
        if expected is None or actual is None:
            if expected is not actual:
                mismatches.append(f"{name}: visibility changed")
            continue
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            mismatches.append(f"{name}: shape/dtype changed")
            continue
        if not torch.equal(_tensor_bits(torch, expected), _tensor_bits(torch, actual)):
            mismatches.append(f"{name}: binary mismatch")
    return mismatches


def _run_stress(args, logical: dict) -> dict:
    import torch
    import torch_npu  # noqa: F401

    from executor_chunk_kda_fwd import _OUTPUT_NAMES, _run_positive_npu

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    catalog = _pinned_execution_catalog(args)
    variant_baselines = {}
    result = {"design_id": args.design_id, "soc": args.soc, "variants": {}, "status": "passed"}
    for variant in logical["spec"]["design_variants"]:
        physical = _physical_record(args.source, args.design_id, args.soc, variant=variant)
        if physical is None:
            return {"design_id": args.design_id, "soc": args.soc, "status": "not_applicable"}
        spec = physical["spec"]
        inputs = _load_cached_inputs(logical["spec"], spec, catalog, device)
        baseline = None
        with torch.no_grad():
            for repeat in range(int(spec["repeat_count"])):
                outputs = tuple(_run_positive_npu(inputs, spec))
                torch.npu.synchronize()
                for name, output in zip(_OUTPUT_NAMES, outputs):
                    if output is not None and not torch.isfinite(output).all().item():
                        raise RuntimeError(f"{variant} repeat {repeat} {name} contains NaN or Inf")
                if baseline is None:
                    baseline = tuple(
                        None if output is None else output.detach().clone() for output in outputs
                    )
                else:
                    mismatches = compare_outputs_bitwise(torch, baseline, outputs)
                    if mismatches:
                        raise RuntimeError(
                            f"{variant} repeat {repeat} is not bitwise deterministic: {mismatches}"
                        )
        variant_baselines[variant] = baseline
        result["variants"][variant] = {
            "repeats": int(spec["repeat_count"]),
            "bitwise_equal_to_run_0": True,
        }
    if logical["spec"].get("cross_variant_common_outputs_bitwise"):
        first_name, second_name = logical["spec"]["design_variants"][:2]
        first, second = variant_baselines[first_name], variant_baselines[second_name]
        common_first = tuple(a if b is not None else None for a, b in zip(first, second))
        common_second = tuple(b if a is not None else None for a, b in zip(first, second))
        mismatches = compare_outputs_bitwise(torch, common_first, common_second)
        if mismatches:
            raise RuntimeError(f"common outputs differ across masks: {mismatches}")
        result["cross_variant_common_outputs_bitwise"] = True
    return result


def _run_application(args, logical: dict, physical: dict) -> int:
    import torch
    import torch_npu  # noqa: F401

    from executor_chunk_kda_fwd import _run_positive_npu

    device = torch.device(f"npu:{args.device}")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    torch.npu.set_device(device)
    catalog = _pinned_execution_catalog(args)
    spec = physical["spec"]
    required_build_variant = str(spec.get("required_build_variant", "baseline"))
    selected_build_variant = os.environ.get(
        "KDA_CANONICAL_BUILD_VARIANT", "baseline"
    ).strip()
    if selected_build_variant != required_build_variant:
        raise RuntimeError(
            "selected operator build variant does not match the canonical experiment: "
            f"expected {required_build_variant!r}, got {selected_build_variant!r}"
        )
    inputs = _load_cached_inputs(logical["spec"], spec, catalog, device)
    with torch.no_grad():
        for _ in range(args.repeats):
            _run_positive_npu(inputs, spec)
        torch.npu.synchronize()
    return 0


def _add_case_args(parser, *, require_variant: bool = True):
    parser.add_argument("--source", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument(
        "--catalog",
        default=default_catalog_reference(),
        help="externally pinned catalog SHA256, filename, or in-cache path",
    )
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--soc", choices=SUPPORTED_SOCS, required=True)
    parser.add_argument("--variant", required=require_variant, default="fixed_input")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    application = subparsers.add_parser("application")
    _add_case_args(application)

    stress = subparsers.add_parser("stress")
    _add_case_args(stress, require_variant=False)

    command = subparsers.add_parser("msopprof-command")
    _add_case_args(command)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--python", default=sys.executable)

    parse = subparsers.add_parser("parse-msopprof")
    _add_case_args(parse)
    parse.add_argument("--report", type=Path, required=True)
    parse.add_argument(
        "--kernel-json",
        type=Path,
        action="append",
        required=True,
        help="kernel JSON for an expected stage; repeat for multi-stage applications",
    )
    parse.add_argument("--log", type=Path, required=True)
    parse.add_argument("--baseline-report", type=Path)
    parse.add_argument("--baseline-kernel-json", type=Path, action="append")
    parse.add_argument("--baseline-log", type=Path)

    sanitizer = subparsers.add_parser("run-sanitizer")
    _add_case_args(sanitizer)
    sanitizer.add_argument("--tool", choices=SANITIZER_TOOLS, required=True)
    sanitizer.add_argument("--operator-object", type=Path, required=True)
    sanitizer.add_argument("--log", type=Path, required=True)
    sanitizer.add_argument("--python", default=sys.executable)

    verify = subparsers.add_parser("verify-sanitizer")
    verify.add_argument("--tool", choices=SANITIZER_TOOLS, required=True)
    verify.add_argument("--operator-object", type=Path, required=True)
    verify.add_argument("--log", type=Path, required=True)
    verify.add_argument("--kernel-regex", default="chunk_kda_fwd")
    verify.add_argument("--symbol-regex", default="sanitizer")

    args = parser.parse_args()
    if args.command == "verify-sanitizer":
        nm = subprocess.run(
            ["nm", str(args.operator_object)], text=True, capture_output=True, check=True
        )
        verify_object_symbol(nm.stdout, args.symbol_regex)
        result = verify_sanitizer_log(
            args.log.read_text(encoding="utf-8", errors="replace"),
            tool=args.tool,
            kernel_regex=args.kernel_regex,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    logical = _logical_record(args.source, args.design_id)
    if args.command == "stress":
        result = _run_stress(args, logical)
        print(json.dumps(result, sort_keys=True))
        return EXIT_NOT_APPLICABLE if result["status"] == "not_applicable" else 0

    tool = getattr(args, "tool", None)
    physical = _physical_record(
        args.source,
        args.design_id,
        args.soc,
        variant=args.variant,
        sanitizer_tool=tool,
    )
    if physical is None:
        print(json.dumps({"design_id": args.design_id, "soc": args.soc, "status": "not_applicable"}))
        return EXIT_NOT_APPLICABLE
    spec = physical["spec"]
    if args.command == "application":
        return _run_application(args, logical, physical)
    if args.command == "msopprof-command":
        print(
            _shell_join(
                materialize_msopprof_command(
                    spec,
                    source=args.source,
                    cache_dir=args.cache_dir,
                    catalog_reference=args.catalog,
                    output=args.output,
                    python=args.python,
                    device=args.device,
                )
            )
        )
        return 0
    if args.command == "parse-msopprof":
        profiler = spec["profiler"]
        current = parse_msopprof_report(
            args.report,
            kernel_jsons=args.kernel_json,
            log=args.log,
            launch_count=int(profiler["launch_count"]),
        )
        baseline = None
        if args.baseline_report:
            if not args.baseline_kernel_json or not args.baseline_log:
                raise ValueError(
                    "--baseline-report requires --baseline-kernel-json and --baseline-log"
                )
            baseline = parse_msopprof_report(
                args.baseline_report,
                kernel_jsons=args.baseline_kernel_json,
                log=args.baseline_log,
                launch_count=int(profiler["launch_count"]),
            )
        elif args.baseline_kernel_json or args.baseline_log:
            raise ValueError(
                "--baseline-kernel-json/--baseline-log require --baseline-report"
            )
        result = evaluate_performance(spec, current, baseline)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] in {"passed", "measured"} else 1
    if args.command == "run-sanitizer":
        nm = subprocess.run(
            ["nm", str(args.operator_object)], text=True, capture_output=True, check=True
        )
        verify_object_symbol(nm.stdout, spec["sanitizer"]["object_symbol_regex"])
        application_command = _application_command(
            python=args.python,
            source=args.source,
            cache_dir=args.cache_dir,
            catalog_reference=args.catalog,
            design_id=args.design_id,
            soc=args.soc,
            variant=args.variant,
            device=args.device,
        )
        args.log.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "mssanitizer",
                f"--tool={args.tool}",
                *spec["sanitizer"]["tool_options"][args.tool],
                f"--log-file={args.log}",
                *application_command,
            ],
            check=True,
        )
        if not args.log.is_file():
            raise RuntimeError("mssanitizer did not create the requested raw log")
        result = verify_sanitizer_log(
            args.log.read_text(encoding="utf-8", errors="replace"),
            tool=args.tool,
            kernel_regex=spec["sanitizer"]["kernel_regex"],
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
