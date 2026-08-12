#!/usr/bin/env python3
"""Profile the requested KDA matrix with msopprof."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from xml.sax.saxutils import escape, quoteattr


HEADS = 96
KEY_DIM = 128
VALUE_DIM = 128
CHUNK_SIZE = 64
PROFILE_RANGE = "FLA_NPU_KDA_BENCH"
ATK_CASE_ID_START = 250
SEQUENCE_LENGTHS = (1024, 1536, 2048, 4096, 8192, 16384)
DISTRIBUTIONS = ("single", "balanced8", "mixed_tail", "short64")
SEED_BASE = 20260812
PREFILL_KERNEL_FILTER = "|".join(
    (
        "ChunkKdaFwd*",
        "chunk_kda_fwd*",
        "ChunkKdaFwdPrepare*",
        "chunk_kda_fwd_prepare*",
        "ChunkKdaFwdPostWu*",
        "chunk_kda_fwd_post_wu*",
        "ChunkGatedDeltaRuleFwdH*",
        "chunk_gated_delta_rule_fwd_h*",
        "ChunkKdaFwdFinalize*",
        "chunk_kda_fwd_finalize*",
    )
)
BASIC_INFO_FILE = re.compile(r"^OpBasicInfo(?:_.*)?\.csv$", re.IGNORECASE)
PROFILE_METRIC_NAMES = (
    "OpBasicInfo",
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
)
DETAIL_ALIASES = {
    "task_duration_us": ("Task Duration(us)", "task_duration(us)"),
    "aic_time_us": ("aic_time(us)",),
    "aiv_time_us": ("aiv_time(us)",),
    "mac_time_us": ("MAC Time(us)", "Mac Time(us)", "aic_cube_time(us)"),
    "cube_time_us": ("aic_cube_time(us)",),
    "cube_ratio": ("aic_cube_ratio",),
    "cube_fp_ratio": ("aic_cube_fp_ratio",),
    "cube_int_ratio": ("aic_cube_int_ratio",),
    "vec_time_us": ("aiv_vec_time(us)",),
    "vec_ratio": ("aiv_vec_ratio",),
    "vec_vf_ratio": ("aiv_vec_vf_ratio",),
    "vec_sfu_ratio": ("aiv_vec_sfu_ratio",),
    "vec_simt_vf_ratio": ("aiv_vec_simt_vf_ratio",),
    "aic_mte1_time_us": ("aic_mte1_time(us)",),
    "aic_mte1_ratio": ("aic_mte1_ratio",),
    "aic_mte1_active_bw_gb_s": ("aic_mte1_active_bw(GB/s)",),
    "aic_mte2_time_us": ("aic_mte2_time(us)",),
    "aic_mte2_ratio": ("aic_mte2_ratio",),
    "aic_mte2_active_bw_gb_s": ("aic_mte2_active_bw(GB/s)",),
    "aic_mte3_time_us": ("aic_mte3_time(us)",),
    "aic_mte3_ratio": ("aic_mte3_ratio",),
    "aic_mte3_active_bw_gb_s": ("aic_mte3_active_bw(GB/s)",),
    "aic_fixpipe_time_us": ("aic_fixpipe_time(us)",),
    "aic_fixpipe_ratio": ("aic_fixpipe_ratio",),
    "aic_fixpipe_active_bw_gb_s": ("aic_fixpipe_active_bw(GB/s)",),
    "aiv_mte2_time_us": ("aiv_mte2_time(us)",),
    "aiv_mte2_ratio": ("aiv_mte2_ratio",),
    "aiv_mte2_active_bw_gb_s": ("aiv_mte2_active_bw(GB/s)",),
    "aiv_mte3_time_us": ("aiv_mte3_time(us)",),
    "aiv_mte3_ratio": ("aiv_mte3_ratio",),
    "aiv_mte3_active_bw_gb_s": ("aiv_mte3_active_bw(GB/s)",),
    "aic_scalar_time_us": ("aic_scalar_time(us)",),
    "aic_scalar_ratio": ("aic_scalar_ratio",),
    "aiv_scalar_time_us": ("aiv_scalar_time(us)",),
    "aiv_scalar_ratio": ("aiv_scalar_ratio",),
}
DIAGNOSTIC_TREE_LIMIT = 500
DIAGNOSTIC_CSV_LIMIT = 100
DIAGNOSTIC_LOG_LINES = 240
INVALID_XML_CHARACTERS = re.compile(
    "[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]"
)


@dataclass(frozen=True)
class Case:
    atk_case_id: int
    case_id: str
    case_key: str
    phase: str
    direction: str
    batch: int
    sequence: int
    distribution: str
    disable_recompute: bool
    layout: str
    cu_seqlens: tuple[int, ...]
    explicit_chunk_indices: bool
    seed: int
    h100_us: Optional[float]
    optimized_npu_us: Optional[float]


def distribution_lengths(total: int, distribution: str) -> list[int]:
    if distribution == "single":
        return [total]
    if distribution == "balanced8":
        quotient, remainder = divmod(total, 8)
        return [quotient + (index < remainder) for index in range(8)]
    if distribution == "mixed_tail":
        base = total // 8
        values = [
            base - 47,
            base + 31,
            base - 1,
            base + 17,
            base - 33,
            base + 49,
            base - 15,
        ]
        values.append(total - sum(values))
        return values
    if distribution == "short64":
        quotient, remainder = divmod(total, CHUNK_SIZE)
        values = [CHUNK_SIZE] * quotient
        if remainder:
            values.append(remainder)
        return values
    raise ValueError(f"unknown distribution: {distribution}")


def cumulative_lengths(lengths: Iterable[int]) -> tuple[int, ...]:
    values = [0]
    for length in lengths:
        if length <= 0:
            raise ValueError(f"sequence length must be positive, got {length}")
        values.append(values[-1] + int(length))
    return tuple(values)


def build_atk_cases() -> tuple[Case, ...]:
    cases = []
    pair_id = 0
    for sequence in SEQUENCE_LENGTHS:
        for distribution in DISTRIBUTIONS:
            cu_seqlens = cumulative_lengths(
                distribution_lengths(sequence, distribution)
            )
            for disable_recompute in (False, True):
                atk_case_id = ATK_CASE_ID_START + pair_id * 2 + int(disable_recompute)
                mode = "export" if disable_recompute else "recompute"
                case_key = (
                    f"ascend950_h96_t{sequence}_c64_packed_{distribution}_{mode}"
                )
                cases.append(
                    Case(
                        atk_case_id=atk_case_id,
                        case_id=str(atk_case_id),
                        case_key=case_key,
                        phase="prefill",
                        direction="fwd",
                        batch=1,
                        sequence=sequence,
                        distribution=distribution,
                        disable_recompute=disable_recompute,
                        layout="BSND",
                        cu_seqlens=cu_seqlens,
                        explicit_chunk_indices=False,
                        seed=SEED_BASE + pair_id,
                        h100_us=None,
                        optimized_npu_us=None,
                    )
                )
            pair_id += 1
    result = tuple(cases)
    ids = tuple(case.atk_case_id for case in result)
    if len(result) != 48 or ids != tuple(range(250, 298)):
        raise RuntimeError("PR297 performance matrix must contain case IDs 250-297")
    return result


CASES = build_atk_cases()
CASE_BY_ID = {case.case_id: case for case in CASES}
CASE_BY_KEY = {case.case_key: case for case in CASES}
LEGACY_CASE_ALIASES = {
    "prefill_fwd_b1_s1024": "250",
    "prefill_fwd_b1_s8192": "282",
    "prefill_fwd_b1_s16384": "290",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-dir", type=Path)
    parser.add_argument("--repo-commit", default="unknown")
    parser.add_argument("--soc", default="unknown")
    parser.add_argument("--device-visible-id", type=int, default=0)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--warm-up", type=int, default=5)
    parser.add_argument("--launch-count", type=int, default=5000)
    parser.add_argument("--case-timeout", type=int, default=900)
    parser.add_argument("--decode-step", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--aic-metrics", default="Default")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--repair-workbook",
        type=Path,
        metavar="RESULTS_DIR",
        help="rebuild kernel_detail.xlsx from an existing results directory",
    )
    parser.add_argument("--worker")
    return parser.parse_args()


def selected_cases(value: str) -> list[Case]:
    if value == "all":
        return list(CASES)
    ids = [
        LEGACY_CASE_ALIASES.get(item.strip(), item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    selectors = {**CASE_BY_KEY, **CASE_BY_ID}
    unknown = sorted(set(ids) - set(selectors))
    if unknown:
        raise ValueError(f"unknown case IDs: {', '.join(unknown)}")
    return [selectors[case_id] for case_id in ids]


def shell_join(command: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def normal_quantized(
    torch,
    shape,
    generator,
    original_dtype,
    device,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    sigmoid: bool = False,
    l2_normalize: bool = False,
):
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    value = value.mul(std).add(mean)
    if sigmoid:
        value = torch.sigmoid(value)
    if l2_normalize:
        value = value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1e-6)
    return value.to(original_dtype).to(device)


def canonical_chunk_indices(
    cu_seqlens: tuple[int, ...], chunk_size: int
) -> tuple[int, ...]:
    indices = []
    for seq_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk_id in range((end - start + chunk_size - 1) // chunk_size):
            indices.extend((seq_id, chunk_id))
    return tuple(indices)


def make_prefill_inputs(torch, case: Case, device):
    shape = (case.batch, case.sequence, HEADS, KEY_DIM)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)
    q = normal_quantized(
        torch, shape, generator, torch.bfloat16, device,
        std=0.05, l2_normalize=True,
    )
    k = normal_quantized(
        torch, shape, generator, torch.bfloat16, device,
        std=0.05, l2_normalize=True,
    )
    v = normal_quantized(
        torch, shape, generator, torch.bfloat16, device, std=0.05,
    )
    gate = normal_quantized(
        torch, shape, generator, torch.float32, device, std=1.25,
    )
    beta = normal_quantized(
        torch,
        (case.batch, case.sequence, HEADS),
        generator,
        torch.bfloat16,
        device,
        mean=1.5,
        std=0.35,
        sigmoid=True,
    )
    a_log = normal_quantized(
        torch, (HEADS,), generator, torch.float32, device, std=0.12,
    )
    dt_bias = normal_quantized(
        torch,
        (HEADS * KEY_DIM,),
        generator,
        torch.float32,
        device,
        mean=-3.0,
        std=1.65,
    )
    return q, k, v, gate, beta, a_log, dt_bias


def run_in_profile_range(torch, operation):
    import torch_npu

    range_id = torch_npu.npu.mstx.range_start(PROFILE_RANGE)
    if not isinstance(range_id, int):
        raise RuntimeError("torch_npu MSTX range_start is unavailable")
    try:
        output = operation()
        torch.npu.synchronize()
        return output
    finally:
        torch_npu.npu.mstx.range_end(range_id)


def run_prefill_forward(torch, case: Case, device) -> None:
    from fla_npu.ops.ascendc import chunk_kda_fwd

    q, k, v, gate, beta, a_log, dt_bias = make_prefill_inputs(torch, case, device)
    chunk_indices = (
        canonical_chunk_indices(case.cu_seqlens, CHUNK_SIZE)
        if case.explicit_chunk_indices
        else None
    )
    torch.npu.synchronize()
    outputs = run_in_profile_range(
        torch,
        lambda: chunk_kda_fwd(
            q,
            k,
            v,
            gate,
            beta,
            KEY_DIM**-0.5,
            CHUNK_SIZE,
            layout="BSND",
            initial_state=None,
            output_final_state=False,
            cu_seqlens=list(case.cu_seqlens),
            chunk_indices=chunk_indices,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            disable_recompute=case.disable_recompute,
            return_intermediate_states=False,
            state_v_first=True,
        ),
    )
    print(
        f"BENCH_OK atk_case_id={case.atk_case_id} case_key={case.case_key} "
        f"output_shape={tuple(outputs[0].shape)}",
        flush=True,
    )


def run_worker(case: Case, device_visible_id: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    device = torch.device(f"npu:{device_visible_id}")
    torch.npu.set_device(device)
    torch.manual_seed(case.seed)
    run_prefill_forward(torch, case, device)
    return 0


def iter_profile_files(profile_dir: Path) -> Iterable[Path]:
    visited = set()
    for root, dirnames, filenames in os.walk(profile_dir, followlinks=True):
        try:
            stat = os.stat(root)
        except OSError:
            dirnames.clear()
            continue
        directory_id = (stat.st_dev, stat.st_ino)
        if directory_id in visited:
            dirnames.clear()
            continue
        visited.add(directory_id)
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            yield Path(root) / filename


def profile_metric_name(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    for name in sorted(PROFILE_METRIC_NAMES, key=len, reverse=True):
        lowered = name.lower()
        if stem == lowered or stem.startswith(f"{lowered}_"):
            return name
    return None


def normalized_csv_row(row: dict) -> dict[str, str]:
    return {
        (key or "").strip(): "" if value is None else value.strip()
        for key, value in row.items()
        if (key or "").strip()
    }


def profile_row_location(
    path: Path, row: dict[str, str], row_index: int
) -> tuple[str, str]:
    if path.parent.name.isdigit():
        return path.parent.parent.name, path.parent.name
    del row_index
    replay = row.get("Task ID") or row.get("task_id") or "0"
    return path.parent.name, replay


def merge_raw_fields(target: dict, metric_name: str, source: dict[str, str]) -> None:
    for key, value in source.items():
        if key not in target or target[key] in ("", None):
            target[key] = value
        elif value not in ("", None) and target[key] != value:
            target[f"{metric_name}.{key}"] = value


def read_kernel_detail_rows(profile_dir: Path) -> list[dict]:
    tables: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for path in iter_profile_files(profile_dir):
        metric_name = profile_metric_name(path)
        if metric_name is None:
            continue
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
            for row_index, row in enumerate(csv.DictReader(stream)):
                normalized = normalized_csv_row(row)
                instance, replay = profile_row_location(path, normalized, row_index)
                tables.setdefault((instance, replay), {}).setdefault(
                    metric_name, []
                ).append(
                    {
                        "row_index": row_index,
                    "source_csv": relative_display(path, profile_dir),
                        "fields": normalized,
                    }
                )

    records = []
    for (instance, replay), metric_tables in tables.items():
        basic_rows = metric_tables.get("OpBasicInfo", [])
        resource_rows: dict[tuple[str, str, str], dict] = {}
        for metric_name, rows in metric_tables.items():
            if metric_name == "OpBasicInfo":
                continue
            for row in rows:
                fields = row["fields"]
                block_id = fields.get("block_id", fields.get("Block Id", ""))
                sub_block_id = fields.get(
                    "sub_block_id", fields.get("Sub Block Id", "")
                )
                identity = (
                    block_id,
                    sub_block_id,
                    str(row["row_index"]) if block_id == "" else "",
                )
                record = resource_rows.setdefault(
                    identity,
                    {
                        "kernel_instance": instance,
                        "replay": replay,
                        "block_id": block_id,
                        "sub_block_id": sub_block_id,
                        "metric_file_types": set(),
                        "source_csvs": set(),
                    },
                )
                record["metric_file_types"].add(metric_name)
                record["source_csvs"].add(row["source_csv"])
                merge_raw_fields(record, metric_name, fields)

        if not resource_rows:
            for row_index, basic in enumerate(basic_rows or ({"fields": {}},)):
                resource_rows[("", "", str(row_index))] = {
                    "kernel_instance": instance,
                    "replay": replay,
                    "block_id": "",
                    "sub_block_id": "",
                    "metric_file_types": set(),
                    "source_csvs": set(),
                }

        for row_index, record in enumerate(resource_rows.values()):
            if basic_rows:
                basic = basic_rows[min(row_index, len(basic_rows) - 1)]
                record["metric_file_types"].add("OpBasicInfo")
                record["source_csvs"].add(basic["source_csv"])
                merge_raw_fields(record, "OpBasicInfo", basic["fields"])
            record["metric_file_types"] = ",".join(
                sorted(record["metric_file_types"])
            )
            record["source_csvs"] = ",".join(sorted(record["source_csvs"]))
            records.append(record)
    records.sort(
        key=lambda item: (
            natural_sort_key(str(item["kernel_instance"])),
            natural_sort_key(str(item["replay"])),
            natural_sort_key(str(item["block_id"])),
            natural_sort_key(str(item["sub_block_id"])),
        )
    )
    return records


def read_profile_rows(profile_dir: Path) -> list[dict]:
    records = []
    for path in iter_profile_files(profile_dir):
        if not BASIC_INFO_FILE.fullmatch(path.name):
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                normalized = normalized_csv_row(row)
                duration = normalized.get("Task Duration(us)")
                if not duration:
                    continue
                instance, replay = profile_row_location(
                    path, normalized, len(records)
                )
                records.append(
                    {
                        "instance": instance,
                        "replay": replay,
                        "op_name": normalized.get("Op Name", instance),
                        "op_type": normalized.get("Op Type", ""),
                        "duration_us": float(duration),
                        "discovery_index": len(records),
                    }
                )
    return records


def relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def profile_tree_lines(profile_dir: Path) -> list[str]:
    if not profile_dir.exists():
        return ["<profile directory does not exist>"]
    lines = []
    try:
        paths = sorted(profile_dir.rglob("*"), key=lambda item: str(item))
    except OSError as exc:
        return [f"<unable to enumerate profile directory: {exc}>"]
    for path in paths[:DIAGNOSTIC_TREE_LIMIT]:
        display = relative_display(path, profile_dir)
        try:
            if path.is_symlink():
                lines.append(f"LINK {display} -> {os.readlink(path)}")
            elif path.is_dir():
                lines.append(f"DIR  {display}/")
            else:
                lines.append(f"FILE {display} size={path.stat().st_size}")
        except OSError as exc:
            lines.append(f"ERR  {display}: {exc}")
    if len(paths) > DIAGNOSTIC_TREE_LIMIT:
        lines.append(f"... truncated {len(paths) - DIAGNOSTIC_TREE_LIMIT} entries")
    return lines or ["<profile directory is empty>"]


def csv_diagnostic_lines(profile_dir: Path) -> list[str]:
    csv_paths = [
        path for path in iter_profile_files(profile_dir) if path.suffix.lower() == ".csv"
    ]
    lines = [f"discovered_csv_count={len(csv_paths)}"]
    basic_info_paths = [path for path in csv_paths if BASIC_INFO_FILE.fullmatch(path.name)]
    lines.append(f"matching_basic_info_count={len(basic_info_paths)}")
    for path in csv_paths[:DIAGNOSTIC_CSV_LIMIT]:
        display = relative_display(path, profile_dir)
        try:
            with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                first_row = next(reader, [])
            lines.append(f"CSV {display}")
            lines.append(f"  header={json.dumps(header, ensure_ascii=False)}")
            lines.append(f"  first_row={json.dumps(first_row, ensure_ascii=False)}")
        except Exception as exc:
            lines.append(f"CSV {display} read_error={type(exc).__name__}: {exc}")
    if len(csv_paths) > DIAGNOSTIC_CSV_LIMIT:
        lines.append(f"... truncated {len(csv_paths) - DIAGNOSTIC_CSV_LIMIT} CSV files")
    return lines


def log_tail_lines(path: Path) -> list[str]:
    if not path.exists():
        return [f"<log does not exist: {path}>"]
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > DIAGNOSTIC_LOG_LINES:
        return [f"... last {DIAGNOSTIC_LOG_LINES} of {len(lines)} lines"] + lines[
            -DIAGNOSTIC_LOG_LINES:
        ]
    return lines or ["<log is empty>"]


def write_case_diagnostics(
    args: argparse.Namespace,
    result: dict,
    *,
    stage: str,
    returncode: Optional[int],
    log_paths: Iterable[Path],
) -> str:
    profile_dir = Path(result["profile_dir"])
    diagnostics_path = args.output_dir / "logs" / f"{result['case_id']}.diagnostics.txt"
    lines = [
        f"case={result['case_id']}",
        f"stage={stage}",
        f"status={result['status']}",
        f"note={result['note']}",
        f"returncode={returncode}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"msopprof={shutil.which('msopprof')}",
        f"worker_command={result.get('worker_command', '')}",
        "profile_commands="
        + json.dumps(result.get("profile_commands", []), ensure_ascii=False),
        f"profile_mode={result.get('profile_mode', '')}",
        f"profile_dir={profile_dir}",
        f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '')}",
        f"ASCEND_OPP_PATH={os.environ.get('ASCEND_OPP_PATH', '')}",
        f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}",
        f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')}",
        f"LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}",
        "",
        "== Profile tree ==",
        *profile_tree_lines(profile_dir),
        "",
        "== CSV inspection ==",
        *csv_diagnostic_lines(profile_dir),
    ]
    for path in log_paths:
        lines.extend(("", f"== Log tail: {path} ==", *log_tail_lines(path)))
    diagnostics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(diagnostics_path)


def summarize_profile(records: Iterable[dict]) -> tuple[float, list[dict]]:
    grouped: dict[str, list[float]] = {}
    names: dict[str, str] = {}
    for record in records:
        instance = record["instance"]
        grouped.setdefault(instance, []).append(record["duration_us"])
        names[instance] = record["op_name"]
    if not grouped:
        raise RuntimeError("msopprof produced no matching OpBasicInfo rows")
    breakdown = []
    for instance, values in grouped.items():
        median_us = statistics.median(values)
        breakdown.append(
            {
                "instance": instance,
                "op_name": names[instance],
                "samples": len(values),
                "median_us": median_us,
                "min_us": min(values),
                "max_us": max(values),
            }
        )
    breakdown.sort(key=lambda item: item["median_us"], reverse=True)
    return sum(item["median_us"] for item in breakdown), breakdown


def natural_sort_key(value: str) -> tuple:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", value)
    )


def classify_failure(log_text: str, returncode: int) -> tuple[str, str]:
    runtime_errors = re.findall(r"(?:RuntimeError|ValueError):\s*(.+)", log_text)
    if runtime_errors:
        return "ERROR", runtime_errors[-1].strip()
    if re.search(
        r"out of memory|\bOOM\b|cannot allocate memory|ACL_ERROR_RT_MEMORY",
        log_text,
        re.IGNORECASE,
    ):
        return "OOM", "device memory allocation failed"
    return "ERROR", f"msopprof/worker exited with status {returncode}"


def run_logged(command, *, cwd: Path, env: dict, log, timeout: int) -> tuple[int, bool]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )

    def copy_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"    {line}", end="", flush=True)

    output_thread = threading.Thread(target=copy_output, daemon=True)
    output_thread.start()
    try:
        returncode = process.wait(timeout=timeout)
        output_thread.join(timeout=10)
        return returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        output_thread.join(timeout=10)
        return process.returncode, True


def requested_shape(case: Case) -> str:
    return f"[{case.batch}, {case.sequence}, {HEADS}, {KEY_DIM}]"


def executed_shape(case: Case) -> str:
    return requested_shape(case)


def case_chunk_count(case: Case) -> int:
    return sum(
        (end - start + CHUNK_SIZE - 1) // CHUNK_SIZE
        for start, end in zip(case.cu_seqlens, case.cu_seqlens[1:])
    )


def speedup(baseline_us: Optional[float], actual_us: Optional[float]) -> Optional[float]:
    if baseline_us is None or actual_us is None or actual_us <= 0:
        return None
    return baseline_us / actual_us


def kernel_name_filter(case: Case) -> str:
    del case
    return PREFILL_KERNEL_FILTER


def profile_attempts(
    args: argparse.Namespace, case: Case, case_dir: Path, worker: list[str]
) -> list[dict]:
    common_options = [
        "msopprof",
        f"--application={shell_join(worker)}",
        f"--aic-metrics={args.aic_metrics}",
        f"--launch-count={args.launch_count}",
        f"--warm-up={args.warm_up}",
        "--replay-mode=application",
        "--kill=off",
    ]
    return [
        {
            "mode": "application_mstx",
            "profile_dir": case_dir / "application_mstx",
            "metric_scope": (
                "sum of median msopprof BasicInfo durations in the KDA MSTX range "
                "using application replay"
            ),
            "command": [
                *common_options,
                f"--output={case_dir / 'application_mstx'}",
                "--mstx=on",
                f"--mstx-include={PROFILE_RANGE}",
            ],
        },
        {
            "mode": "application_kernel_filter",
            "profile_dir": case_dir / "application_kernel_filter",
            "metric_scope": (
                "sum of median msopprof BasicInfo durations for explicitly selected "
                "KDA stages using application replay"
            ),
            "command": [
                *common_options,
                f"--output={case_dir / 'application_kernel_filter'}",
                f"--kernel-name={kernel_name_filter(case)}",
            ],
        },
    ]


def run_profile(args: argparse.Namespace, case: Case) -> dict:
    case_dir = args.output_dir / "profiles" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    preflight_log_path = args.output_dir / "logs" / f"{case.case_id}.preflight.log"
    worker = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        case.case_id,
        "--device-visible-id",
        str(args.device_visible_id),
    ]
    attempts = profile_attempts(args, case, case_dir, worker)
    env = os.environ.copy()
    env["TEST_DEVICE_ID"] = str(args.device_visible_id)
    result = {
        **asdict(case),
        "sequence_count": len(case.cu_seqlens) - 1,
        "chunk_count": case_chunk_count(case),
        "requested_qkv_shape": requested_shape(case),
        "executed_shape": executed_shape(case),
        "status": "PASS",
        "fla_npu_us": None,
        "speedup_vs_h100": None,
        "speedup_vs_optimized_npu": None,
        "metric_scope": "",
        "profile_dir": str(case_dir),
        "selected_profile_dir": "",
        "log": "",
        "diagnostics": "",
        "worker_command": shell_join(worker),
        "profile_commands": [shell_join(attempt["command"]) for attempt in attempts],
        "profile_mode": "",
        "aic_metrics": args.aic_metrics,
        "note": "",
        "breakdown": [],
        "metric_file_types": [],
        "missing_metric_file_types": [],
        "_kernel_detail": [],
        "profiled_segment_us": None,
    }
    with preflight_log_path.open("w", encoding="utf-8") as preflight_log:
        print(
            f"    preflight started (timeout={args.case_timeout}s, "
            f"log={preflight_log_path})",
            flush=True,
        )
        preflight_returncode, preflight_timed_out = run_logged(
            worker,
            cwd=args.repo_dir,
            env=env,
            log=preflight_log,
            timeout=args.case_timeout,
        )
    print(f"    preflight finished (status={preflight_returncode})", flush=True)
    if preflight_timed_out:
        result["status"] = "TIMEOUT"
        result["log"] = str(preflight_log_path)
        result["note"] = f"preflight exceeded {args.case_timeout} seconds"
        result["diagnostics"] = write_case_diagnostics(
            args,
            result,
            stage="preflight_timeout",
            returncode=preflight_returncode,
            log_paths=(preflight_log_path,),
        )
        return result
    if preflight_returncode != 0:
        log_text = preflight_log_path.read_text(encoding="utf-8", errors="replace")
        result["status"], note = classify_failure(log_text, preflight_returncode)
        result["log"] = str(preflight_log_path)
        result["note"] = f"preflight failed: {note}"
        result["diagnostics"] = write_case_diagnostics(
            args,
            result,
            stage="preflight",
            returncode=preflight_returncode,
            log_paths=(preflight_log_path,),
        )
        return result
    attempt_notes = []
    attempt_statuses = []
    attempt_log_paths = []
    profile_returncode = None
    total_us = None
    breakdown = []
    for attempt in attempts:
        mode = attempt["mode"]
        log_path = args.output_dir / "logs" / f"{case.case_id}.{mode}.log"
        attempt_log_paths.append(log_path)
        command = attempt["command"]
        result["log"] = str(log_path)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"command={shell_join(command)}\n")
            log.flush()
            print(
                f"    profiler started (mode={mode}, timeout={args.case_timeout}s, "
                f"log={log_path})",
                flush=True,
            )
            profile_returncode, profile_timed_out = run_logged(
                command,
                cwd=args.repo_dir,
                env=env,
                log=log,
                timeout=args.case_timeout,
            )
        print(
            f"    profiler finished (mode={mode}, status={profile_returncode})",
            flush=True,
        )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if profile_timed_out:
            attempt_statuses.append("TIMEOUT")
            attempt_notes.append(f"{mode}: exceeded {args.case_timeout} seconds")
            continue
        if profile_returncode != 0:
            status, note = classify_failure(log_text, profile_returncode)
            attempt_statuses.append(status)
            attempt_notes.append(f"{mode}: {note}")
            continue
        try:
            records = read_profile_rows(attempt["profile_dir"])
            total_us, breakdown = summarize_profile(records)
            detail_rows = read_kernel_detail_rows(attempt["profile_dir"])
            metric_file_types = sorted(
                {
                    metric_name
                    for row in detail_rows
                    for metric_name in row["metric_file_types"].split(",")
                    if metric_name
                }
            )
            if args.aic_metrics.lower() == "default":
                missing = sorted(set(PROFILE_METRIC_NAMES) - set(metric_file_types))
                if missing:
                    raise RuntimeError(
                        "msopprof Default output is incomplete; missing metric tables: "
                        + ", ".join(missing)
                    )
        except Exception as exc:
            attempt_statuses.append("ERROR")
            attempt_notes.append(f"{mode}: {exc}")
            continue
        result["profile_mode"] = mode
        result["metric_scope"] = attempt["metric_scope"]
        result["selected_profile_dir"] = str(attempt["profile_dir"])
        result["metric_file_types"] = metric_file_types
        result["_kernel_detail"] = detail_rows
        break

    if total_us is None:
        if "OOM" in attempt_statuses:
            result["status"] = "OOM"
        elif attempt_statuses and all(status == "TIMEOUT" for status in attempt_statuses):
            result["status"] = "TIMEOUT"
        else:
            result["status"] = "ERROR"
        result["note"] = "; ".join(attempt_notes)
        result["diagnostics"] = write_case_diagnostics(
            args,
            result,
            stage="profile_attempts",
            returncode=profile_returncode,
            log_paths=(preflight_log_path, *attempt_log_paths),
        )
        return result
    result["profiled_segment_us"] = total_us
    result["fla_npu_us"] = total_us
    result["speedup_vs_h100"] = speedup(case.h100_us, total_us)
    result["speedup_vs_optimized_npu"] = speedup(case.optimized_npu_us, total_us)
    result["breakdown"] = breakdown
    notes = []
    if attempt_notes:
        notes.append(
            f"profiled with {result['profile_mode']} after " + "; ".join(attempt_notes)
        )
    result["note"] = " ".join(notes)
    return result


def format_number(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def alias_value(row: dict, candidates: Iterable[str]):
    for candidate in candidates:
        if candidate in row and row[candidate] not in ("", None):
            return row[candidate]
    return ""


def case_matrix_row(case: Case, selected_ids: set[str]) -> dict:
    return {
        "selected": case.case_id in selected_ids,
        "atk_case_id": case.atk_case_id,
        "case_id": case.case_id,
        "case_key": case.case_key,
        "total_token_count": case.sequence,
        "sequence_distribution": case.distribution,
        "sequence_count": len(case.cu_seqlens) - 1,
        "chunk_count": case_chunk_count(case),
        "disable_recompute": case.disable_recompute,
        "layout": case.layout,
        "batch": case.batch,
        "head_num": HEADS,
        "key_dim": KEY_DIM,
        "value_dim": VALUE_DIM,
        "chunk_size": CHUNK_SIZE,
        "initial_state": "None",
        "output_final_state": False,
        "use_gate_in_kernel": True,
        "safe_gate": True,
        "return_intermediate_states": False,
        "state_v_first": True,
        "explicit_chunk_indices": case.explicit_chunk_indices,
        "cu_seqlens": csv_value(case.cu_seqlens),
        "seed": case.seed,
    }


def write_case_matrix(args: argparse.Namespace) -> None:
    selected_ids = {case.case_id for case in selected_cases(args.cases)}
    rows = [case_matrix_row(case, selected_ids) for case in CASES]
    with (args.output_dir / "case_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_kernel_detail_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        details = result.get("_kernel_detail") or ({},)
        for detail in details:
            row = {
                "atk_case_id": result["atk_case_id"],
                "case_id": result["case_id"],
                "case_key": result["case_key"],
                "case_status": result["status"],
                "total_token_count": result["sequence"],
                "sequence_distribution": result["distribution"],
                "sequence_count": result["sequence_count"],
                "chunk_count": result["chunk_count"],
                "disable_recompute": result["disable_recompute"],
                "layout": result["layout"],
                "batch": result["batch"],
                "head_num": HEADS,
                "key_dim": KEY_DIM,
                "value_dim": VALUE_DIM,
                "chunk_size": CHUNK_SIZE,
                "cu_seqlens": csv_value(result["cu_seqlens"]),
                "seed": result["seed"],
                "profile_mode": result["profile_mode"],
                "aic_metrics": result.get("aic_metrics", "Default"),
                **detail,
            }
            for alias, candidates in DETAIL_ALIASES.items():
                row[alias] = alias_value(row, candidates)
            rows.append(row)
    return rows


def excel_column_name(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def excel_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped not in ("", "NA") and re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", stripped
        ):
            try:
                return int(stripped) if re.fullmatch(r"[-+]?\d+", stripped) else float(stripped)
            except ValueError:
                pass
    return value


def xlsx_cell(reference: str, value, style: int = 0) -> str:
    value = excel_value(value)
    if value is None or value == "":
        return f'<c r={quoteattr(reference)} s={quoteattr(str(style))}/>'
    if isinstance(value, bool):
        return (
            f'<c r={quoteattr(reference)} s={quoteattr(str(style))} t="b">'
            f"<v>{int(value)}</v></c>"
        )
    if isinstance(value, (int, float)):
        numeric_style = 6 if isinstance(value, int) else 5
        return (
            f'<c r={quoteattr(reference)} s={quoteattr(str(style or numeric_style))}>'
            f"<v>{value}</v></c>"
        )
    text = escape(INVALID_XML_CHARACTERS.sub("", str(value)))
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<c r={quoteattr(reference)} s={quoteattr(str(style))} t="inlineStr">'
        f"<is><t{preserve}>{text}</t></is></c>"
    )


def xlsx_row(row_number: int, values: Iterable, style: int = 0) -> str:
    cells = "".join(
        xlsx_cell(f"{excel_column_name(column)}{row_number}", value, style)
        for column, value in enumerate(values)
    )
    return f'<row r="{row_number}">{cells}</row>'


def xlsx_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="10"/><name val="Aptos"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E2F3"/></left><right style="thin"><color rgb="FFD9E2F3"/></right><top style="thin"><color rgb="FFD9E2F3"/></top><bottom style="thin"><color rgb="FFD9E2F3"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="7">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
    <xf numFmtId="0" fontId="1" fillId="4" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="4" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
    <xf numFmtId="3" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def xlsx_sheet_xml(case: Case, result: Optional[dict], columns: tuple[str, ...]) -> str:
    status = result["status"] if result else "NOT_RUN"
    detail_rows = build_kernel_detail_rows([result]) if result else []
    detail_rows = detail_rows or ({},)
    metadata_rows = (
        ("ATK case ID", case.atk_case_id, "Status", status, "Total tokens", case.sequence, "Distribution", case.distribution),
        ("Sequence count", len(case.cu_seqlens) - 1, "Chunk count", case_chunk_count(case), "Disable recompute", case.disable_recompute, "Seed", case.seed),
        ("Case key", case.case_key, "cu_seqlens", csv_value(case.cu_seqlens), "Layout", case.layout, "Chunk size", CHUNK_SIZE),
    )
    rows = [xlsx_row(1, (f"Chunk KDA performance detail - ATK case {case.atk_case_id}",), 1)]
    for row_number, values in enumerate(metadata_rows, start=2):
        cells = []
        for column, value in enumerate(values):
            style = 2 if column % 2 == 0 else 3
            cells.append(xlsx_cell(f"{excel_column_name(column)}{row_number}", value, style))
        rows.append(f'<row r="{row_number}" ht="22">{"".join(cells)}</row>')
    rows.append(xlsx_row(5, columns, 4))
    for row_number, detail in enumerate(detail_rows, start=6):
        rows.append(xlsx_row(row_number, (detail.get(column, "") for column in columns)))
    last_column = excel_column_name(max(len(columns), 8) - 1)
    last_row = 5 + len(detail_rows)
    widths = []
    for index, column in enumerate(columns, start=1):
        if column in ("source_csvs", "Op Name", "Op Type"):
            width = 42
        elif len(column) > 28:
            width = 24
        else:
            width = min(max(len(column) + 2, 12), 22)
        widths.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
    tab_color = "FF70AD47" if status == "PASS" else ("FFC00000" if status not in ("PASS", "NOT_RUN") else "FFB7B7B7")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetPr><tabColor rgb="{tab_color}"/></sheetPr>
  <dimension ref="A1:{last_column}{last_row}"/>
  <sheetViews><sheetView showGridLines="0" workbookViewId="0"><pane ySplit="5" topLeftCell="A6" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>{''.join(widths)}</cols>
  <sheetData>{''.join(rows)}</sheetData>
  <mergeCells count="1"><mergeCell ref="A1:H1"/></mergeCells>
</worksheet>'''


def write_kernel_detail_workbook(args: argparse.Namespace, results: list[dict]) -> None:
    rows = build_kernel_detail_rows(results)
    resource_columns = (
        "metric_file_types",
        "source_csvs",
        "kernel_instance",
        "replay",
        "block_id",
        "sub_block_id",
        "Op Name",
        "Op Type",
        *DETAIL_ALIASES,
    )
    all_columns = {key for row in rows for key in row}
    case_columns = {
        "atk_case_id", "case_id", "case_key", "case_status", "total_token_count",
        "sequence_distribution", "sequence_count", "chunk_count", "disable_recompute",
        "layout", "batch", "head_num", "key_dim", "value_dim", "chunk_size",
        "cu_seqlens", "seed", "profile_mode", "aic_metrics",
    }
    dynamic_columns = sorted(all_columns - set(resource_columns) - case_columns)
    columns = (*resource_columns, *dynamic_columns)
    result_by_case = {result["case_id"]: result for result in results}
    workbook_path = args.output_dir / "kernel_detail.xlsx"
    with zipfile.ZipFile(workbook_path, "w", zipfile.ZIP_DEFLATED) as workbook:
        sheet_overrides = "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index in range(1, len(CASES) + 1)
        )
        workbook.writestr(
            "[Content_Types].xml",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  {sheet_overrides}
</Types>''',
        )
        workbook.writestr(
            "_rels/.rels",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''',
        )
        sheets_xml = "".join(
            f'<sheet name="case_{case.atk_case_id}" sheetId="{index}" r:id="rId{index}"/>'
            for index, case in enumerate(CASES, start=1)
        )
        workbook.writestr(
            "xl/workbook.xml",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView activeTab="0"/></bookViews>
  <sheets>{sheets_xml}</sheets>
</workbook>''',
        )
        relationships = "".join(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(CASES) + 1)
        )
        relationships += (
            f'<Relationship Id="rId{len(CASES) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{relationships}</Relationships>''',
        )
        workbook.writestr("xl/styles.xml", xlsx_styles_xml())
        for index, case in enumerate(CASES, start=1):
            workbook.writestr(
                f"xl/worksheets/sheet{index}.xml",
                xlsx_sheet_xml(case, result_by_case.get(case.case_id), columns),
            )


def repair_kernel_detail_workbook(results_dir: Path) -> Path:
    results_dir = results_dir.resolve()
    results_file = results_dir / "results.json"
    if not results_file.is_file():
        raise FileNotFoundError(f"results file not found: {results_file}")
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"invalid results payload: {results_file}")
    for result in results:
        profile_dir = result.get("selected_profile_dir")
        result["_kernel_detail"] = (
            read_kernel_detail_rows(Path(profile_dir)) if profile_dir else []
        )
    write_kernel_detail_workbook(argparse.Namespace(output_dir=results_dir), results)
    return results_dir / "kernel_detail.xlsx"


def write_reports(args: argparse.Namespace, results: list[dict]) -> None:
    from importlib import metadata as package_metadata

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": args.repo_commit,
        "soc": args.soc,
        "device_visible_id": args.device_visible_id,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "warm_up": args.warm_up,
        "launch_count": args.launch_count,
        "case_timeout": args.case_timeout,
        "aic_metrics": args.aic_metrics,
        "matrix_contract": "PR297 A5 positive case IDs 250-297",
        "profile_attempt_order": [
            "application replay with MSTX range filtering",
            "application replay with explicit KDA stage filtering",
        ],
    }
    try:
        metadata["torch"] = package_metadata.version("torch")
        metadata["torch_npu"] = package_metadata.version("torch-npu")
    except package_metadata.PackageNotFoundError:
        pass
    public_results = [
        {key: value for key, value in result.items() if not key.startswith("_")}
        for result in results
    ]
    payload = {"metadata": metadata, "results": public_results}
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    diagnostics = [result for result in results if result.get("diagnostics")]
    if diagnostics:
        sections = []
        for result in diagnostics:
            path = Path(result["diagnostics"])
            sections.extend(
                (
                    f"######## {result['case_id']} ########",
                    path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        (args.output_dir / "diagnostics.txt").write_text(
            "\n".join(sections), encoding="utf-8"
        )

    columns = (
        "atk_case_id",
        "case_id",
        "case_key",
        "phase",
        "direction",
        "batch",
        "sequence",
        "distribution",
        "sequence_count",
        "chunk_count",
        "disable_recompute",
        "cu_seqlens",
        "seed",
        "requested_qkv_shape",
        "executed_shape",
        "status",
        "h100_us",
        "optimized_npu_us",
        "fla_npu_us",
        "profiled_segment_us",
        "speedup_vs_h100",
        "speedup_vs_optimized_npu",
        "metric_scope",
        "profile_mode",
        "note",
        "profile_dir",
        "selected_profile_dir",
        "log",
        "diagnostics",
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    write_case_matrix(args)
    write_kernel_detail_workbook(args, results)

    lines = [
        "# fla_npu main KDA performance",
        "",
        f"- commit: `{args.repo_commit}`",
        f"- SOC: `{args.soc}`",
        "- unit: microseconds",
        f"- msopprof AI Core metrics: `{args.aic_metrics}`",
        "- matrix: PR297 A5 positive cases 250-297",
        "- metric: sum of median msopprof BasicInfo durations for selected KDA stages",
        "- full per-case kernel resources: `kernel_detail.xlsx` (one sheet per case)",
        "",
        "| ATK case | total tokens | sequence distribution | sequence count | chunk count | "
        "disable recompute | status | profiler | fla_npu us |",
        "| ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for result in results:
        lines.append(
            "| {atk_case_id} | {sequence} | {distribution} | {sequence_count} | "
            "{chunk_count} | {disable_recompute} | {status} | {profile_mode} | "
            "{actual} |".format(
                **result,
                actual=format_number(result["fla_npu_us"]),
            )
        )
    notes = [result for result in results if result["note"]]
    if notes:
        lines.extend(("", "## Notes", ""))
        lines.extend(f"- `{result['case_id']}`: {result['note']}" for result in notes)
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if sys.version_info < (3, 9):
        raise SystemExit(
            "benchmark_kda_matrix.py requires Python 3.9 or newer; "
            f"found {platform.python_version()}"
        )
    if not args.aic_metrics:
        raise ValueError("--aic-metrics must not be empty")
    if args.repair_workbook:
        print(f"Rebuilt workbook: {repair_kernel_detail_workbook(args.repair_workbook)}")
        return 0
    if args.list_cases:
        rows = [case_matrix_row(case, {item.case_id for item in CASES}) for case in CASES]
        writer = csv.DictWriter(sys.stdout, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return 0
    if args.worker:
        if args.worker not in CASE_BY_ID:
            raise ValueError(f"unknown worker case ID: {args.worker}")
        return run_worker(CASE_BY_ID[args.worker], args.device_visible_id)
    if args.output_dir is None or args.repo_dir is None:
        raise ValueError("--output-dir and --repo-dir are required in orchestrator mode")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir()
    results = []
    for index, case in enumerate(selected_cases(args.cases), start=1):
        print(f"[{index}] profiling {case.case_id}", flush=True)
        result = run_profile(args, case)
        results.append(result)
        print(
            f"    status={result['status']} fla_npu_us={format_number(result['fla_npu_us'])}",
            flush=True,
        )
        if result["status"] != "PASS":
            print(f"    note={result['note']}", flush=True)
            if result.get("diagnostics"):
                print(f"    diagnostics={result['diagnostics']}", flush=True)
        write_reports(args, results)
    print((args.output_dir / "results.md").read_text(encoding="utf-8"), flush=True)
    diagnostics_path = args.output_dir / "diagnostics.txt"
    if diagnostics_path.exists():
        print(f"Diagnostics: {diagnostics_path}", flush=True)
    failed_case_ids = [
        result["case_id"] for result in results if result["status"] != "PASS"
    ]
    if failed_case_ids:
        print(
            "Benchmark failed cases: " + ", ".join(failed_case_ids),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
