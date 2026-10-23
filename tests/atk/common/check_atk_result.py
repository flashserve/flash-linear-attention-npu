#!/usr/bin/env python3
"""Fail-closed checks for the official ATK statistic/summary XLSX reports.

The fields below come from ATK 26.9.8 report_title/{result,summary}_report_title.py.
An inapplicable summary field may be '-'; actual per-case evidence is mandatory.
The runner supplies a stage start time and JSON case count to reject stale or
partial reports. Standalone checks can omit the time when auditing old evidence.
"""

import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path
import re

import openpyxl


PHASES = {
    "accuracy": ("accuracy", "精度是否达标", "精度通过"),
    "performance": ("perf", "device性能是否达标", "Device性能（us）"),
    "determinism": ("determinism", "确定性计算是否达标", "确定性计算结果"),
    "mssanitizer": ("mssanitizer", "内存检测是否达标", "内存检测通过"),
}


class InvalidReport(ValueError):
    """Missing or contradictory evidence; never interpreted as a skipped check."""


def _missing(value):
    return value is None or (isinstance(value, str) and value.strip() in ("", "-"))


def _integer(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidReport(f"{label}: invalid count {value!r}") from None
    if isinstance(value, bool) or not math.isfinite(number) or number < 0 or not number.is_integer():
        raise InvalidReport(f"{label}: invalid count {value!r}")
    return int(number)


def _true(value):
    # bool and CSV strings are both emitted by official ATK. Numeric 1 is not
    # accepted as a substitute for a missing boolean result.
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _table(workbook, sheet, allow_empty=False):
    if sheet not in workbook.sheetnames:
        raise InvalidReport(f"missing {sheet} sheet")
    rows = list(workbook[sheet].values)
    if not rows:
        raise InvalidReport(f"empty {sheet} sheet")
    header = [str(v).strip() if v is not None else "" for v in rows[0]]
    if not all(header) or len(header) != len(set(header)):
        raise InvalidReport(f"{sheet}: empty or duplicate column names")
    data = [dict(zip(header, row)) for row in rows[1:] if any(not _missing(v) for v in row)]
    if not data and not allow_empty:
        raise InvalidReport(f"{sheet}: zero cases/rows")
    return header, data


def _summary(header, rows, phase, count):
    verdict = PHASES[phase][1]
    required = {"名称"}
    if phase != "performance":
        required.update(("总用例数", verdict))
    if not required.issubset(header):
        raise InvalidReport(f"summary: missing columns {sorted(required - set(header))}")
    active_nodes = []
    for row in rows:
        if _missing(row["名称"]):
            raise InvalidReport("summary: missing node name")
        for name in header:
            if name.endswith("是否达标") and not _missing(row[name]) and row[name] != "Pass":
                raise InvalidReport(f"summary: {name}={row[name]!r}")
        active = row.get(verdict) == "Pass"
        if active:
            active_nodes.append(str(row["名称"]))
        counts = {"总用例数": count, "执行成功用例个数": count,
                  "执行失败用例个数": 0, "通过用例个数": count}
        for name, expected in counts.items():
            value = row.get(name)
            must_exist = active and (name == "总用例数" or phase in ("accuracy", "determinism"))
            if _missing(value):
                if must_exist:
                    raise InvalidReport(f"summary: missing {name} for {row['名称']}")
            elif _integer(value, name) != expected:
                raise InvalidReport(f"summary: {name}={value!r}, statistic count={count}")
        rate_name = {"accuracy": "通过率", "determinism": "通过率",
                     "mssanitizer": "内存检测通过率"}.get(phase)
        rate = row.get(rate_name)
        if not _missing(rate) and _integer(rate, rate_name) != 100:
            raise InvalidReport(f"summary: {rate_name}={rate!r}")
    if phase != "performance" and not active_nodes:
        raise InvalidReport(f"summary: no explicit {verdict}=Pass")
    return active_nodes


def _determinism_outputs(value, case):
    """Require tensor output evidence; empty-vs-empty DC comparisons are True.

    ATK AccuracyInfoJson emits repr(list[dict(dtype, shape, stride)]). Optional
    None/empty outputs may coexist with real tensors. Byte counters belong to
    performance metadata and are not a substitute for this output contract.
    """
    try:
        outputs = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError, TypeError) as error:
        raise InvalidReport(f"case {case}: malformed output_info_list") from error
    if not isinstance(outputs, list):
        raise InvalidReport(f"case {case}: output_info_list must be a list")
    real_tensor = False
    for output in outputs:
        if output is None:
            continue
        if not isinstance(output, dict) or not {"dtype", "shape", "stride"}.issubset(output):
            raise InvalidReport(f"case {case}: malformed output descriptor {output!r}")
        dtype, shape, stride = output["dtype"], output["shape"], output["stride"]
        if dtype is None and shape in (None, []) and stride in (None, []):
            continue
        if not isinstance(dtype, str) or not re.fullmatch(
                r"torch\.(?:bool|u?int(?:8|16|32|64)|bfloat16|float(?:16|32|64)|complex(?:32|64|128)|float8_[a-z0-9]+)", dtype):
            raise InvalidReport(f"case {case}: invalid tensor dtype {dtype!r}")
        if (not isinstance(shape, list) or not isinstance(stride, list)
                or len(shape) != len(stride)
                or any(type(v) is not int or v < 0 for v in shape + stride)):
            raise InvalidReport(f"case {case}: invalid tensor shape/stride {shape!r}/{stride!r}")
        real_tensor |= math.prod(shape) > 0
    if not real_tensor:
        raise InvalidReport(f"case {case}: determinism has no nonempty tensor output evidence")


def _statistic(header, rows, phase, active_nodes):
    required = {"编号", "运行结果"}
    if phase == "determinism":
        required.add("output_info_list")
    if not required.issubset(header):
        raise InvalidReport(f"statistic: missing columns {sorted(required - set(header))}")
    suffix = PHASES[phase][2]
    columns = [h for h in header if h == suffix or h.endswith("_" + suffix)]
    if phase == "performance":
        columns = [h for h in columns if h.startswith(("npu_", "pyaclnn_"))]
    elif phase != "determinism":
        for node in active_nodes:
            if f"{node}_{suffix}" not in columns:
                raise InvalidReport(f"statistic: missing result for summary node {node}")
    if not columns:
        raise InvalidReport(f"statistic: missing {suffix} evidence")
    seen = set()
    for row in rows:
        case = _integer(row["编号"], "编号")
        if case in seen:
            raise InvalidReport(f"statistic: duplicate case ID {case}")
        seen.add(case)
        if row["运行结果"] != "SUCCESS":
            raise InvalidReport(f"case {case}: 运行结果={row['运行结果']!r}")
        if phase == "determinism":
            _determinism_outputs(row["output_info_list"], case)
        values = [row[h] for h in columns if not _missing(row[h])]
        if not values:
            raise InvalidReport(f"case {case}: missing {suffix} result")
        if phase == "performance":
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = math.nan
                if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
                    raise InvalidReport(f"case {case}: invalid device timing {value!r}")
            # Single-node performance has no comparison verdict. If ATK does
            # supply comparison or variance checks, they must also pass.
            for name in header:
                if name.endswith(("_device性能通过", "_性能波动校验结果")) and not _missing(row[name]):
                    if not _true(row[name]):
                        raise InvalidReport(f"case {case}: {name}={row[name]!r}")
        else:
            if not all(_true(v) for v in values):
                raise InvalidReport(f"case {case}: {suffix}={values!r}")
            if phase != "determinism":
                for node in active_nodes:
                    if not _true(row[f"{node}_{suffix}"]):
                        raise InvalidReport(f"case {case}: missing/pending result for {node}")
    return seen


def validate_report(path, phase, expected_count=None, expected_ids=None):
    """Return the proven case count, or raise InvalidReport on any gap."""
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            statistic_header, cases = _table(workbook, "statistic")
            # ATK's single-node performance report has only the 名称 header,
            # with no comparison nodes/summary rows. Statistic is authoritative.
            summary_header, summary = _table(workbook, "summary", phase == "performance")
        finally:
            workbook.close()
    except InvalidReport:
        raise
    except Exception as error:
        raise InvalidReport(f"cannot read report: {error}") from error
    if expected_count is not None:
        expected = _integer(expected_count, "expected count")
        if expected == 0 or len(cases) != expected:
            raise InvalidReport(f"statistic count={len(cases)}, expected={expected}")
    nodes = _summary(summary_header, summary, phase, len(cases))
    actual_ids = _statistic(statistic_header, cases, phase, nodes)
    if expected_ids is not None:
        if not isinstance(expected_ids, list) or not expected_ids:
            raise InvalidReport("expected IDs must be a nonempty list")
        ids = {_integer(value, "expected ID") for value in expected_ids}
        if len(ids) != len(expected_ids) or actual_ids != ids:
            raise InvalidReport(f"case ID mismatch: actual={sorted(actual_ids)}, expected={expected_ids}")
    return len(cases)


def check_report(output_root, op, phase, started_after_ns=None, expected_count=None, expected_ids=None):
    folder = PHASES[phase][0]
    candidates = list(Path(output_root).glob(f"{folder}/atk_output/atk_{op}_*/report/*.xlsx"))
    if started_after_ns is not None:
        candidates = [p for p in candidates if p.stat().st_mtime_ns >= started_after_ns]
    if not candidates:
        raise InvalidReport(f"no current {phase} report under {output_root}")
    # A corrupt newest report is a failure; never fall back to an older Pass.
    path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    return path, validate_report(path, phase, expected_count, expected_ids)


def validate_mss_log(path, started_after_ns, op, tool="memcheck", expected_count=None):
    if not path:
        raise InvalidReport("mssanitizer: --mss-log is required")
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidReport("mssanitizer: missing or empty sanitizer log")
    if started_after_ns is not None and path.stat().st_mtime_ns < started_after_ns:
        raise InvalidReport("mssanitizer: stale sanitizer log")
    text = path.read_text(encoding="utf-8", errors="strict")
    if "ERROR" in text:
        raise InvalidReport("mssanitizer: sanitizer log contains ERROR")
    if re.search(r"not instrumented|unsupported|not supported|skipp(?:ed|ing)", text, re.IGNORECASE):
        raise InvalidReport("mssanitizer: sanitizer reported an unsupported/skipped check")
    # CANN 9.2 emits these records to --log-file (confirmed on Ascend 950).
    # Startup messages, auxiliary IsFinite/ReduceAll kernels and ATK's True
    # cells alone do not prove that the custom OPP kernel was checked.
    starts = re.findall(r"Start (\w+) sanitizer on kernel (\w+)", text)
    finished = re.findall(r"Sanitizer finished on kernel (\w+)\. No error detected\.", text)
    if not starts or any(actual_tool != tool for actual_tool, _ in starts):
        raise InvalidReport(f"mssanitizer: no complete {tool} execution evidence")
    start_counts = Counter(kernel for _, kernel in starts)
    if start_counts != Counter(finished):
        raise InvalidReport("mssanitizer: unmatched kernel start/completion records")
    target = op.replace("_", "").lower()
    checked = sum(count for kernel, count in start_counts.items()
                  if kernel.split("_")[0].lower() == target)
    minimum = _integer(expected_count, "expected count") if expected_count is not None else 1
    if not target or checked < max(1, minimum):
        raise InvalidReport(f"mssanitizer: only {checked} target {op} launches checked, expected at least {minimum}")
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=[*PHASES, "all"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--started-after-ns", type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-ids", type=json.loads)
    parser.add_argument("--mss-log")
    parser.add_argument("--mss-tool", default="memcheck")
    args = parser.parse_args(argv)
    failed = False
    for phase in PHASES if args.type == "all" else [args.type]:
        try:
            if phase == "mssanitizer":
                validate_mss_log(args.mss_log, args.started_after_ns, args.op,
                                 args.mss_tool, args.expected_count)
            path, count = check_report(args.output_root, args.op, phase,
                                       args.started_after_ns, args.expected_count, args.expected_ids)
            print(f"[ATK结果检查] {phase}: Pass ({count} cases) [{path}]")
        except (InvalidReport, OSError) as error:
            failed = True
            print(f"[ATK结果检查] {phase}: Failed: {error}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
