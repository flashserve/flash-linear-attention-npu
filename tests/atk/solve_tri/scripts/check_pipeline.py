#!/usr/bin/env python3
"""Run A2 ATK long-pipeline cases with both sanitizers and reject skipped checks."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess


OP_DIR = Path(__file__).resolve().parents[1]
MSS_MANIFEST = OP_DIR / "atk_solve_tri_a2_mss.json"
EXECUTOR = OP_DIR / "executor_solve_tri_a2.py"
PIPELINE_IDS = (206, 207)


def read_core_count(path: Path, device: int) -> int:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row.get("Type") == "SolveTri"]
    if not records:
        raise ValueError("core evidence must be an ATK kernel_details.csv containing SolveTri")
    if any(int(row["Device_id"]) != device for row in records):
        raise ValueError("core evidence was collected on a different NPU device")
    for row in records:
        dims = [int(value) for value in row["Input Shapes"].split(";", 1)[0].strip('" ').split(",")]
        if dims not in ([1, 11264, 32, 64], [1, 32, 11264, 64], [65536, 32, 64]):
            raise ValueError("core evidence must come from a full model-size chunk-64 SolveTri run")
    counts = {int(row["Block Num"]) for row in records}
    if len(counts) != 1 or min(counts) <= 0:
        raise ValueError("core evidence has missing or inconsistent AIC block counts")
    return counts.pop()


def pipeline_coverage(path: Path, cores: int) -> list[dict]:
    if cores <= 0:
        raise ValueError("AIC count must be positive")
    cases = json.loads(path.read_text(encoding="utf-8"))
    selected = [case for case in cases if case["id"] in PIPELINE_IDS]
    if len(selected) != 2 or {case["id"] for case in selected} != set(PIPELINE_IDS):
        raise ValueError("mss manifest must contain both pipeline cases 206 and 207")
    result = []
    for case in selected:
        spec = json.loads(next(item["range_values"] for item in case["inputs"]
                               if item["name"] == "case_spec"))
        if case["api_type"] != "executor_solve_tri_a2" or spec.get("soc") != "ascend910b":
            raise ValueError("pipeline cases must select the independent A2 executor and SoC")
        chunk = int(spec["chunk_size"])
        if chunk != 64:
            raise ValueError("pipeline cases must select the A2/A3 chunk-64 implementation")
        lengths = spec.get("seqlens", [int(spec["T"])] * int(spec["B"]))
        tiles = int(spec["H"]) * sum((length + chunk - 1) // chunk for length in lengths)
        # Match the tiler: every core gets a ceil-sized interval except the tail.
        per_core = (tiles + cores - 1) // cores
        used_cores = (tiles + per_core - 1) // per_core if per_core else 0
        minimum = tiles - (used_cores - 1) * per_core if used_cores else 0
        batches = (minimum + 15) // 16
        if minimum <= 64:
            raise ValueError(f"case {case['id']}: only {minimum} tiles on the least loaded core; "
                             "every AIC must reach a fifth 16-tile batch")
        result.append({"id": case["id"], "tiles": tiles, "aic_blocks": used_cores,
                       "tiles_per_aic": per_core,
                       "min_tiles_per_aic": minimum, "min_batches_per_aic": batches})
    return result


def check_sanitizer(text: str, expected_tool: str, minimum: int = 2) -> dict:
    if expected_tool not in ("racecheck", "memcheck"):
        raise ValueError("expected_tool must be racecheck or memcheck")
    target = re.compile(r"\bSolveTri_[A-Za-z0-9_]+\b", re.I)
    started, finished, skipped, errors, warnings = [], [], [], [], []
    current_target = False
    for line in text.splitlines():
        match = target.search(line)
        if "kernel" in line.lower() and match:
            current_target = True
        elif "kernel" in line.lower() and re.search(r"(?:on|check|checking)\s+kernel\s+", line, re.I):
            current_target = False
        if match and "No active sanitizer tool" in line:
            skipped.append(line)
        start = re.search(r"Start (racecheck|memcheck) sanitizer on kernel", line, re.I)
        if match and start:
            if start.group(1).lower() != expected_tool:
                errors.append(f"expected {expected_tool}, observed {start.group(1)}: {line}")
            else:
                started.append(match.group())
        register_cleanup = "Register FFTS_BASE_ADDR was not reset to default" in line
        if current_target and register_cleanup:
            errors.append(line)
        if current_target and not register_cleanup and any(phrase in line.lower() for phrase in
                                  ("not supported", "temporarily ignored", "ffts_base_addr",
                                   "cannot check", "failed to initialize")):
            skipped.append(line)
        if current_target and ("WARN" in line or "warning" in line.lower()):
            warnings.append(line)
        if match and "Sanitizer finished on kernel" in line:
            if "No error detected" in line:
                finished.append(match.group())
            else:
                errors.append(line)
        if "see all detected errors above" in line.lower():
            errors.append(line)
        if re.search(r"\b(?:ERROR|FATAL)\b|\b(?:data\s+race|race\s+hazard|out[\s._-]+of[\s._-]+bounds)\b", line, re.I):
            if not re.search(r"\b(?:0 errors?|no error|no race)\b", line, re.I):
                errors.append(line)
    if errors:
        raise ValueError("sanitizer reported an error: " + errors[0])
    if skipped:
        raise ValueError("SolveTri sanitizer detection was skipped: " + skipped[0])
    if len(started) < minimum:
        raise ValueError(f"only {len(started)} active SolveTri checks; expected at least {minimum}")
    if len(finished) < minimum or len(finished) != len(started):
        raise ValueError(f"only {len(finished)} checked SolveTri launches; expected at least {minimum}")
    if Counter(started) != Counter(finished):
        raise ValueError("SolveTri sanitizer start/finish kernel counts do not match")
    return {"tool": expected_tool, "active_launches": len(started), "checked_launches": len(finished),
            "kernels": sorted(set(finished)),
            "register_cleanup_warnings": sum("Register FFTS_BASE_ADDR was not reset to default" in line
                                             for line in warnings),
            "warnings": warnings, "skipped": skipped, "errors": errors}


def pipeline_contract(case: dict) -> dict:
    spec = json.loads(next(item["range_values"] for item in case["inputs"]
                           if item["name"] == "case_spec"))
    return {"api_type": case["api_type"], "case_spec": spec}


def check_atk_reports(root: Path, manifest: Path = MSS_MANIFEST) -> list[str]:
    import openpyxl

    cases = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {case["id"]: pipeline_contract(case) for case in cases if case["id"] in PIPELINE_IDS}
    if set(expected) != set(PIPELINE_IDS):
        raise ValueError("mss manifest is missing a pipeline case contract")
    paths = list(root.rglob("*.xlsx"))
    reports = []
    observed = set()
    for path in paths:
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if "statistic" not in book.sheetnames:
                continue
            rows = iter(book["statistic"].values)
            headers = next(rows)
            for values in rows:
                row = dict(zip(headers, values))
                index = row.get("\u7f16\u53f7")
                if index not in PIPELINE_IDS:
                    continue
                if index in observed:
                    raise ValueError(f"duplicate ATK result for pipeline case {index}")
                observed.add(index)
                if row.get("\u8fd0\u884c\u7ed3\u679c") != "SUCCESS":
                    raise ValueError(f"ATK pipeline case {index} did not succeed")
                reported_case = json.loads(row["\u7528\u4f8bjson\u4fe1\u606f"])
                if reported_case.get("id") != index or pipeline_contract(reported_case) != expected[index]:
                    raise ValueError(f"ATK pipeline case {index} input contract differs from the mss manifest")
                checks = [value for name, value in row.items()
                          if "\u5185\u5b58\u68c0\u6d4b\u901a\u8fc7" in str(name) and value is not None]
                if not checks or any(value is not True for value in checks):
                    raise ValueError(f"ATK pipeline case {index} has no successful sanitizer verdict")
            reports.append(str(path))
        finally:
            book.close()
    if observed != set(PIPELINE_IDS):
        raise ValueError(f"missing ATK pipeline case results: {sorted(set(PIPELINE_IDS) - observed)}")
    return reports


def verify(root: Path, coverage: list[dict]) -> dict:
    result = {"passed": False, "coverage": coverage, "tools": {}, "errors": []}
    for tool in ("racecheck", "memcheck"):
        try:
            if (root / f"{tool}.exit").read_text().strip() != "0":
                raise ValueError("ATK/sanitizer command returned a nonzero exit code")
            text = (root / f"{tool}.sanitizer.log").read_text(encoding="utf-8", errors="replace")
            checked = check_sanitizer(text, tool)
            checked["reports"] = check_atk_reports(root / tool)
            result["tools"][tool] = checked
        except (OSError, ValueError, KeyError, StopIteration) as error:
            result["errors"].append(f"{tool}: {error}")
    result["passed"] = not result["errors"]
    return result


def run(root: Path, device: int, port: int):
    atk, sanitizer = shutil.which("atk"), shutil.which("mssanitizer")
    if not atk or not sanitizer:
        raise ValueError("atk and mssanitizer must be available in the initialized environment")
    cases = json.loads(MSS_MANIFEST.read_text(encoding="utf-8"))
    positions = [index for index, case in enumerate(cases) if case["id"] in PIPELINE_IDS]
    if len(positions) != 2 or positions[1] != positions[0] + 1:
        raise ValueError("pipeline cases must occupy two consecutive mss manifest positions")
    root.mkdir(parents=True, exist_ok=False)
    for tool in ("racecheck", "memcheck"):
        log = root / f"{tool}.sanitizer.log"
        command = [sanitizer, f"--tool={tool}", "--kernel-name=SolveTri", f"--log-file={log}", "--log-level=info", "--",
                   atk, "node", "--name", "npu_dut", "--backend", "npu", "--devices", str(device),
                   "--output_path", str(root / tool), "task", "-c", str(MSS_MANIFEST),
                   "-p", str(EXECUTOR), "--task", "run", "--mssanitizer", "-msl", str(log),
                   "-s", str(positions[0]), "-e", str(positions[-1] + 1), "--sqlite_web_port", str(port)]
        (root / f"{tool}.command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        with (root / f"{tool}.atk.log").open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=OP_DIR, stdout=stream, stderr=subprocess.STDOUT)
        (root / f"{tool}.exit").write_text(str(process.returncode) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check"))
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--core-profile", type=Path, required=True,
                        help="raw ATK kernel_details.csv from this device's chunk-64 SolveTri run")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--sqlite-web-port", type=int, default=18983)
    args = parser.parse_args()
    root = args.report_root.resolve()
    try:
        cores = read_core_count(args.core_profile, args.device)
        coverage = pipeline_coverage(MSS_MANIFEST, cores)
        if args.action == "run":
            run(root, args.device, args.sqlite_web_port)
        result = verify(root, coverage)
    except (OSError, ValueError, KeyError) as error:
        result = {"passed": False, "errors": [str(error)]}
    encoded = json.dumps(result, indent=2) + "\n"
    if root.is_dir():
        (root / "pipeline_gate.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
