#!/usr/bin/env python3
"""Profile the requested KDA matrix with msopprof."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


HEADS = 96
KEY_DIM = 128
VALUE_DIM = 128
CHUNK_SIZE = 64
PROFILE_RANGE = "FLA_NPU_KDA_BENCH"
DECODE_TOTAL_CALLS = 10
DECODE_MEASURE_CALLS = 5
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
DECODE_KERNEL_FILTER = "RecurrentKda*|recurrent_kda*"
BASIC_INFO_FILE = re.compile(r"^OpBasicInfo(?:_.*)?\.csv$", re.IGNORECASE)
DIAGNOSTIC_TREE_LIMIT = 500
DIAGNOSTIC_CSV_LIMIT = 100
DIAGNOSTIC_LOG_LINES = 240


@dataclass(frozen=True)
class Case:
    case_id: str
    phase: str
    direction: str
    batch: int
    sequence: int
    h100_us: Optional[float]
    optimized_npu_us: Optional[float]


CASES = (
    Case("prefill_fwd_b1_s1024", "prefill", "fwd", 1, 1024, None, None),
    Case("prefill_fwd_b1_s8192", "prefill", "fwd", 1, 8192, None, None),
    Case("prefill_fwd_b1_s16384", "prefill", "fwd", 1, 16384, None, None),
    Case("prefill_fwd_b1_s65536", "prefill", "fwd", 1, 65536, None, None),
    Case("decode_fwd_b1_s8192", "decode", "fwd", 1, 8192, 17158.0, 109285.4),
    Case("decode_fwd_b4_s8192", "decode", "fwd", 4, 8192, 54987.0, 436946.2),
    Case("decode_fwd_b16_s8192", "decode", "fwd", 16, 8192, 188545.0, 1747650.0),
    Case("decode_fwd_b64_s8192", "decode", "fwd", 64, 8192, None, 6990387.0),
)
CASE_BY_ID = {case.case_id: case for case in CASES}


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
    parser.add_argument("--case-timeout", type=int, default=300)
    parser.add_argument("--decode-step", type=int, default=1)
    parser.add_argument("--aic-metrics", default="BasicInfo")
    parser.add_argument("--worker", choices=tuple(CASE_BY_ID))
    return parser.parse_args()


def selected_cases(value: str) -> list[Case]:
    if value == "all":
        return list(CASES)
    ids = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(ids) - set(CASE_BY_ID))
    if unknown:
        raise ValueError(f"unknown case IDs: {', '.join(unknown)}")
    return [CASE_BY_ID[case_id] for case_id in ids]


def shell_join(command: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def make_prefill_inputs(torch, case: Case, device):
    shape = (case.batch, case.sequence, HEADS, KEY_DIM)
    q = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    gate = torch.full(shape, -1.0, dtype=torch.float32, device=device)
    beta = torch.ones(
        (case.batch, case.sequence, HEADS), dtype=torch.bfloat16, device=device
    )
    a_log = torch.zeros((HEADS,), dtype=torch.float32, device=device)
    dt_bias = torch.zeros((HEADS * KEY_DIM,), dtype=torch.float32, device=device)
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
            output_final_state=False,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            disable_recompute=False,
            return_intermediate_states=False,
            state_v_first=True,
        ),
    )
    print(f"BENCH_OK output_shape={tuple(outputs[0].shape)}", flush=True)


def run_decode_forward(torch, case: Case, device, decode_step: int) -> None:
    from fla_npu.ops.ascendc import recurrent_kda

    query = torch.ones(
        (case.batch, decode_step, HEADS, KEY_DIM), dtype=torch.bfloat16, device=device
    )
    key = torch.zeros_like(query)
    value = torch.zeros(
        (case.batch, decode_step, HEADS, VALUE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    gate = torch.zeros(
        (case.batch, decode_step, HEADS, KEY_DIM), dtype=torch.float32, device=device
    )
    beta = torch.zeros(
        (case.batch, decode_step, HEADS), dtype=torch.float32, device=device
    )
    state = torch.zeros(
        (case.batch, HEADS, VALUE_DIM, KEY_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    cu_seqlens = (
        torch.arange(case.batch + 1, dtype=torch.int32, device=device) * decode_step
    )
    a_log = torch.zeros((HEADS,), dtype=torch.float32, device=device)
    dt_bias = torch.zeros((HEADS, KEY_DIM), dtype=torch.float32, device=device)

    def operation():
        return recurrent_kda(
            query,
            key,
            value,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            A_log=a_log,
            dt_bias=dt_bias,
            layout="BSND",
            output_final_state=False,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            safe_gate=True,
            lower_bound=-4.0,
            state_v_first=True,
        )

    def measured_operations():
        output = None
        for _ in range(DECODE_MEASURE_CALLS):
            output, _ = operation()
        return output

    torch.npu.synchronize()
    for _ in range(DECODE_TOTAL_CALLS - DECODE_MEASURE_CALLS):
        operation()
    torch.npu.synchronize()
    output = run_in_profile_range(torch, measured_operations)
    print(
        f"BENCH_OK output_shape={tuple(output.shape)} "
        f"decode_calls={DECODE_TOTAL_CALLS} measured_calls={DECODE_MEASURE_CALLS}",
        flush=True,
    )


def run_worker(case: Case, device_visible_id: int, decode_step: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    device = torch.device(f"npu:{device_visible_id}")
    torch.npu.set_device(device)
    torch.manual_seed(20260806)
    if case.phase == "decode":
        run_decode_forward(torch, case, device, decode_step)
    else:
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


def read_profile_rows(profile_dir: Path) -> list[dict]:
    records = []
    for path in iter_profile_files(profile_dir):
        if not BASIC_INFO_FILE.fullmatch(path.name):
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                normalized = {(key or "").strip(): value for key, value in row.items()}
                duration = normalized.get("Task Duration(us)")
                if not duration:
                    continue
                records.append(
                    {
                        "instance": path.parent.parent.name,
                        "replay": path.parent.name,
                        "op_name": normalized.get("Op Name", path.parent.parent.name),
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


def is_recurrent_kda_record(record: dict) -> bool:
    identity = " ".join(
        str(record.get(field, "")) for field in ("instance", "op_name", "op_type")
    )
    return "recurrentkda" in re.sub(r"[^a-z0-9]", "", identity.lower())


def summarize_decode_profile(records: Iterable[dict]) -> tuple[float, list[dict]]:
    recurrent_records = [record for record in records if is_recurrent_kda_record(record)]
    recurrent_records.sort(key=lambda item: item["discovery_index"])
    grouped: dict[str, list[float]] = {}
    names: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    for record in recurrent_records:
        instance = record["instance"]
        grouped.setdefault(instance, []).append(record["duration_us"])
        names[instance] = record["op_name"]
        first_seen.setdefault(instance, record["discovery_index"])
    if len(grouped) < DECODE_MEASURE_CALLS:
        if len(recurrent_records) < DECODE_MEASURE_CALLS:
            raise RuntimeError(
                "msopprof produced fewer than "
                f"{DECODE_MEASURE_CALLS} recurrent_kda duration samples"
            )
        breakdown = []
        selected_start = len(recurrent_records) - DECODE_MEASURE_CALLS
        for index, record in enumerate(recurrent_records):
            duration_us = record["duration_us"]
            breakdown.append(
                {
                    "instance": f"{record['instance']}#sample_{index + 1}",
                    "source_instance": record["instance"],
                    "op_name": record["op_name"],
                    "samples": 1,
                    "median_us": duration_us,
                    "min_us": duration_us,
                    "max_us": duration_us,
                    "first_seen": record["discovery_index"],
                    "selected": index >= selected_start,
                }
            )
        selected = recurrent_records[-DECODE_MEASURE_CALLS:]
        return statistics.mean(record["duration_us"] for record in selected), breakdown

    breakdown = []
    for instance, values in grouped.items():
        breakdown.append(
            {
                "instance": instance,
                "op_name": names[instance],
                "samples": len(values),
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
                "first_seen": first_seen[instance],
                "selected": False,
            }
        )
    breakdown.sort(
        key=lambda item: (natural_sort_key(item["instance"]), item["first_seen"])
    )
    selected = breakdown[-DECODE_MEASURE_CALLS:]
    for item in selected:
        item["selected"] = True
    return statistics.mean(item["median_us"] for item in selected), breakdown


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
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout), False
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
        return process.returncode, True


def requested_shape(case: Case) -> str:
    return f"[{case.batch}, {case.sequence}, {HEADS}, {KEY_DIM}]"


def executed_shape(case: Case, decode_step: int) -> str:
    if case.phase == "decode":
        return (
            f"[{case.batch}, {decode_step}, {HEADS}, {KEY_DIM}] BSND x "
            f"{DECODE_TOTAL_CALLS} consecutive calls; last {DECODE_MEASURE_CALLS} averaged; "
            f"state=[{case.batch}, {HEADS}, {VALUE_DIM}, {KEY_DIM}]"
        )
    return requested_shape(case)


def speedup(baseline_us: Optional[float], actual_us: Optional[float]) -> Optional[float]:
    if baseline_us is None or actual_us is None or actual_us <= 0:
        return None
    return baseline_us / actual_us


def kernel_name_filter(case: Case) -> str:
    if case.phase == "prefill":
        return PREFILL_KERNEL_FILTER
    return DECODE_KERNEL_FILTER


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
        "--decode-step",
        str(args.decode_step),
    ]
    attempts = profile_attempts(args, case, case_dir, worker)
    env = os.environ.copy()
    env["TEST_DEVICE_ID"] = str(args.device_visible_id)
    result = {
        **asdict(case),
        "requested_qkv_shape": requested_shape(case),
        "executed_shape": executed_shape(case, args.decode_step),
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
        "note": "",
        "breakdown": [],
        "profiled_segment_us": None,
        "aggregation_factor": 1,
        "decode_step": args.decode_step if case.phase == "decode" else None,
        "decode_total_calls": DECODE_TOTAL_CALLS if case.phase == "decode" else None,
        "decode_measure_calls": DECODE_MEASURE_CALLS if case.phase == "decode" else None,
    }
    with preflight_log_path.open("w", encoding="utf-8") as preflight_log:
        preflight_returncode, preflight_timed_out = run_logged(
            worker,
            cwd=args.repo_dir,
            env=env,
            log=preflight_log,
            timeout=args.case_timeout,
        )
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
            profile_returncode, profile_timed_out = run_logged(
                command,
                cwd=args.repo_dir,
                env=env,
                log=log,
                timeout=args.case_timeout,
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
            if case.phase == "decode":
                total_us, breakdown = summarize_decode_profile(records)
            else:
                total_us, breakdown = summarize_profile(records)
        except Exception as exc:
            attempt_statuses.append("ERROR")
            attempt_notes.append(f"{mode}: {exc}")
            continue
        result["profile_mode"] = mode
        if case.phase == "decode":
            result["metric_scope"] = (
                f"mean of the last {DECODE_MEASURE_CALLS} recurrent_kda call durations "
                f"after {DECODE_TOTAL_CALLS - DECODE_MEASURE_CALLS} warm-up calls"
            )
        else:
            result["metric_scope"] = attempt["metric_scope"]
        result["selected_profile_dir"] = str(attempt["profile_dir"])
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
    if case.phase == "decode":
        result["aggregation_factor"] = case.sequence // args.decode_step
        total_us *= result["aggregation_factor"]
    result["fla_npu_us"] = total_us
    result["speedup_vs_h100"] = speedup(case.h100_us, total_us)
    result["speedup_vs_optimized_npu"] = speedup(case.optimized_npu_us, total_us)
    result["breakdown"] = breakdown
    notes = []
    if attempt_notes:
        notes.append(
            f"profiled with {result['profile_mode']} after " + "; ".join(attempt_notes)
        )
    if case.phase == "decode":
        notes.append(
            f"recurrent_kda runs {DECODE_TOTAL_CALLS} consecutive calls on one in-place BF16 "
            f"[B,H,V,K] state; the mean of the last {DECODE_MEASURE_CALLS} call durations "
            f"is multiplied by {result['aggregation_factor']} calls for full S={case.sequence}."
        )
    result["note"] = " ".join(notes)
    return result


def format_number(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


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
        "decode_step": args.decode_step,
        "decode_total_calls": DECODE_TOTAL_CALLS,
        "decode_measure_calls": DECODE_MEASURE_CALLS,
        "decode_contract": (
            "S=8192 is total recurrent progression length. Each recurrent_kda call processes "
            f"T_step={args.decode_step} token(s) per sequence and updates the same state in place. "
            f"The worker runs {DECODE_TOTAL_CALLS} consecutive calls and averages the last "
            f"{DECODE_MEASURE_CALLS} device-kernel durations."
        ),
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
    payload = {"metadata": metadata, "results": results}
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
        "case_id",
        "phase",
        "direction",
        "batch",
        "sequence",
        "requested_qkv_shape",
        "executed_shape",
        "status",
        "h100_us",
        "optimized_npu_us",
        "fla_npu_us",
        "profiled_segment_us",
        "aggregation_factor",
        "decode_step",
        "decode_total_calls",
        "decode_measure_calls",
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

    lines = [
        "# fla_npu main KDA performance",
        "",
        f"- commit: `{args.repo_commit}`",
        f"- SOC: `{args.soc}`",
        "- unit: microseconds",
        f"- msopprof AI Core metrics: `{args.aic_metrics}`",
        "- prefill metric: sum of median msopprof BasicInfo durations for selected KDA stages",
        f"- decode metric: {DECODE_TOTAL_CALLS} consecutive calls, mean of the last "
        f"{DECODE_MEASURE_CALLS} device-kernel durations, multiplied by "
        f"`8192 / {args.decode_step}`",
        f"- decode semantic: `S=8192` total recurrent steps, `T_step={args.decode_step}`, state carried in place",
        "",
        "| case | phase | dir | B | S | status | profiler | H100 us | optimized NPU us | "
        "fla_npu us | vs H100 | vs optimized NPU |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {case_id} | {phase} | {direction} | {batch} | {sequence} | {status} | "
            "{profile_mode} | "
            "{h100} | {optimized} | {actual} | {vs_h100} | {vs_optimized} |".format(
                **result,
                h100=format_number(result["h100_us"]),
                optimized=format_number(result["optimized_npu_us"]),
                actual=format_number(result["fla_npu_us"]),
                vs_h100=(
                    f"{result['speedup_vs_h100']:.3f}x"
                    if result["speedup_vs_h100"] is not None
                    else "-"
                ),
                vs_optimized=(
                    f"{result['speedup_vs_optimized_npu']:.3f}x"
                    if result["speedup_vs_optimized_npu"] is not None
                    else "-"
                ),
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
    if args.decode_step < 1 or args.decode_step > 8:
        raise ValueError("--decode-step must be in [1, 8]")
    if any(case.sequence % args.decode_step for case in CASES if case.phase == "decode"):
        raise ValueError("--decode-step must divide every selected decode sequence length")
    if args.worker:
        return run_worker(CASE_BY_ID[args.worker], args.device_visible_id, args.decode_step)
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
