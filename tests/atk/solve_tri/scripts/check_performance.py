#!/usr/bin/env python3
"""Check SolveTri's ATK profiles and the A2/A3 versus A5 evidence matrix.

inspect --report REPORT.xlsx --profile-root PROFILE_DIR --node npu_dut
check-a2 --manifest evidence.json --output a2.json
check --manifest evidence.json --max-duration-ratio RATIO --output gate.json

The ratio is DUT Task Duration / A5 Task Duration. It is deliberately mandatory:
Issue #660 says "0.5x A5" without defining latency versus throughput. Missing
platforms, measurements or baseline evidence fail the gate rather than passing
because ATK's performance standard is not_key. See performance_evidence.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics


SOCS = ("A2", "A3", "A5")
TARGETS = ("dense", "varlen")
MAX_EXTRA_BYTES = 200_000_000
DEVICE_NAMES = {"A2": "Ascend910B", "A3": "Ascend910_93", "A5": "Ascend950"}


def file_path(value: str | Path) -> Path:
    result = Path(value).resolve()
    # ATK nests profiles more deeply than Windows' legacy MAX_PATH permits.
    if os.name == "nt" and not str(result).startswith("\\\\?\\"):
        result = Path("\\\\?\\" + str(result))
    return result


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def contract(case: dict) -> dict:
    inputs = {item["name"]: item for item in case["inputs"]}
    attrs = {name: item.get("range_values") for name, item in inputs.items()}
    if "case_spec" in attrs:
        generator = case["api_type"]
        if generator not in {"executor_solve_tri", "executor_solve_tri_a2"}:
            raise ValueError(f"Unknown SolveTri input generator: {generator}")
        spec = json.loads(attrs["case_spec"])
        shape = [int(spec[key]) for key in ("B", "T", "H", "chunk_size")]
        layout = spec["layout"]
        if layout == "bnsd":
            shape = [shape[0], shape[2], shape[1], shape[3]]
        elif layout == "tnd":
            shape = shape[1:]
        elif layout == "ntd":
            shape = [shape[2], shape[1], shape[3]]
        return {"shape": shape, "dtype": spec["dtype"], "layout": layout,
                "seqlens": spec.get("seqlens"), "seed": spec["seed"],
                "generator": generator}
    k_shape = inputs["k"]["shape"]
    heads = inputs["g"]["shape"][-1]
    varlen = attrs["is_varlen"]
    if not isinstance(varlen, bool):
        raise ValueError("is_varlen must be a JSON boolean")
    bounds = [int(item) for item in attrs["cu_seqlens_spec"].split(",")] if varlen else None
    return {"shape": [k_shape[2], heads, attrs["chunk_size"]] if varlen
            else [k_shape[0], heads, k_shape[2], attrs["chunk_size"]],
            "dtype": attrs["qkv_dtype"], "layout": "tnd" if varlen else "bnsd",
            "seqlens": [right - left for left, right in zip(bounds, bounds[1:])] if bounds else None,
            "seed": case.get("default_seed"), "generator": "solve_tri_double_benchmark",
            "raw_inputs": {name: inputs[name] for name in ("k", "g", "beta")}}


def target_name(spec: dict) -> str:
    shape, layout = spec["shape"], spec["layout"]
    if spec["dtype"] != "bf16":
        raise ValueError("Issue #660 model cases require BF16 public inputs")
    if layout in ("bsnd", "bnsd"):
        expected = [1, 11264, 32, 64] if layout == "bsnd" else [1, 32, 11264, 64]
        if shape == expected and not spec["seqlens"]:
            return "dense"
    lengths = spec["seqlens"]
    if (layout == "tnd" and shape == [65536, 32, 64] and lengths
            and len(lengths) == 64 and min(lengths) > 0 and sum(lengths) == 65536):
        return "varlen"
    raise ValueError(f"Not an Issue #660 model case: {spec}")


def read_report(path: Path) -> dict[int, dict]:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(book["statistic"].values)
        headers = next(rows)
        result = {}
        for cells in rows:
            row = dict(zip(headers, cells))
            index = int(row["\u7f16\u53f7"])
            if index in result:
                raise ValueError(f"Duplicate ATK report case {index}")
            case = json.loads(row["\u7528\u4f8bjson\u4fe1\u606f"])
            row["contract"] = contract(case)
            result[index] = row
        return result
    finally:
        book.close()


def profile_durations(path: Path, expected: dict) -> tuple[list[float], list[int]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("OP Type") == "SolveTri"]
    if not rows:
        raise ValueError(f"No SolveTri Task Duration samples in {path}")
    durations = []
    for row in rows:
        value = float(row["Task Duration(us)"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid Task Duration in {path}")
        actual = row["Input Shapes"].split(";", 1)[0].strip('" ')
        if [int(item) for item in actual.split(",")] != expected["shape"]:
            raise ValueError(f"Profile input shape differs from ATK case in {path}")
        expected_dtype = {"bf16": {"DT_BF16", "BFLOAT16"}, "fp16": {"DT_FLOAT16", "FLOAT16"}}[expected["dtype"]]
        if row["Input Data Types"].split(";", 1)[0].strip('" ') not in expected_dtype:
            raise ValueError(f"Profile input dtype differs from model case in {path}")
        durations.append(value)
    device_ids = sorted({int(row["Device_id"]) for row in rows})
    return durations, device_ids


def collect(report: Path, profile_root: Path, node: str, case_ids=None) -> dict:
    rows = read_report(report)
    sources = {}
    for path in profile_root.rglob("op_summary*.csv"):
        parts = path.relative_to(profile_root).parts
        if node not in parts:
            continue
        offset = parts.index(node)
        if offset + 2 < len(parts) and parts[offset + 2].isdigit():
            index = int(parts[offset + 2])
            sources.setdefault(index, []).append(path)
    results = {}
    for index in (sorted(rows) if case_ids is None else case_ids):
        row = rows[index]
        if row["\u8fd0\u884c\u7ed3\u679c"] != "SUCCESS":
            raise ValueError(f"ATK execution failed for case {index}")
        paths = sources.get(index, [])
        if len(paths) != 1:
            raise ValueError(f"Expected one raw op_summary CSV for case {index}, got {len(paths)}")
        values, device_ids = profile_durations(paths[0], row["contract"])
        if len(device_ids) != 1:
            raise ValueError(f"Expected exactly one profiling device for case {index}")
        measured = row.get(node + "_Device\u6027\u80fd\uff08us\uff09")
        mean = statistics.mean(values)
        # Exported CSV samples have 0.001-us precision; XLSX retains more digits.
        if measured is None or not math.isclose(mean, float(measured), rel_tol=0, abs_tol=0.00051):
            raise ValueError(f"ATK XLSX and raw Task Duration disagree for case {index}")
        results[index] = {"contract": row["contract"], "samples": len(values), "mean_us": mean,
                          "device_ids": device_ids,
                          "task_duration_us": values, "profile": str(paths[0]),
                          "profile_sha256": digest(paths[0])}
    return {"report": str(report), "report_sha256": digest(report), "cases": results}


def check_accuracy(path: Path, ids: list[int], expected: dict) -> None:
    rows = read_report(path)
    for index in ids:
        row = rows[index]
        passed = [value for name, value in row.items()
                  if str(name).endswith("_\u7cbe\u5ea6\u901a\u8fc7") and value is not None]
        if row["\u8fd0\u884c\u7ed3\u679c"] != "SUCCESS" or not passed or any(value is not True for value in passed):
            raise ValueError(f"ATK accuracy not passed for case {index}")
        if row["contract"] != expected[index]["contract"]:
            raise ValueError(f"Accuracy and performance input contracts differ for case {index}")


def check_accuracy_worker(report: Path, log: Path, ids: list[int], pid: int, device_ids: list[int]) -> None:
    if report.parent.name != "report" or log.parent.name != "log" or report.parent.parent != log.parent.parent:
        raise ValueError("Accuracy report and ATK log must belong to the same task output directory")
    if len(device_ids) != 1 or not isinstance(pid, int) or pid <= 0:
        raise ValueError("Accuracy worker PID or device is invalid")
    text = log.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"start atk task:.*task=\['accuracy'\]", text):
        raise ValueError("ATK log does not identify an accuracy task")
    saved_reports = re.findall(r"save result excel file:\s*(.+)", text)
    if not any(Path(value.strip()).parts[-3:] == report.parts[-3:] for value in saved_reports):
        raise ValueError("Accuracy ATK log does not identify the supplied report")
    worker = re.compile(rf"\[.*@device_run_{device_ids[0]}\]\s+\[{pid}\].*?\[case (\d+)\](.*)")
    executed, completed = set(), set()
    for line in text.splitlines():
        match = worker.search(line)
        if not match:
            continue
        index, detail = int(match.group(1)), match.group(2)
        if "Start Execute OPP on ['accuracy']" in detail:
            executed.add(index)
        if "SUCCESS run NPU OPP TASK" in detail:
            completed.add(index)
    if not set(ids).issubset(executed & completed):
        raise ValueError("Accuracy log does not associate every case with the declared worker and device")


def asset_fingerprints(provenance: dict, opapi_sha256: str) -> dict:
    mapped = provenance["mapped_files_sha256"]
    if not any(Path(name).name == "libcust_opapi.so" and value == opapi_sha256
               for name, value in mapped.items()):
        raise ValueError("Worker did not load the declared operator API artifact")
    assets = provenance["kernel_assets_sha256"]
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Installed kernel asset fingerprints are missing")
    normalized = {}
    for name, value in assets.items():
        filename = Path(name).name
        if not filename or filename in normalized:
            raise ValueError("Duplicate or invalid kernel asset filename")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid kernel asset SHA256")
        normalized[filename] = value
    if not any(name.endswith(".o") for name in normalized):
        raise ValueError("No installed SolveTri kernel object fingerprint")
    return normalized


def check_memory(path: Path, ids: list[int], expected: dict, opapi_sha256: str,
                 device_name: str, cann_home: str, hostnames: set[str], kernel_assets: dict) -> dict:
    evidence = read_json(path)
    if evidence["method"] != "atk_memory_peak_allocated":
        raise ValueError("Memory evidence must come from ATK peak allocated measurements")
    result = {}
    for index in ids:
        record = evidence["cases"][str(index)]
        if record.get("device_name") != device_name:
            raise ValueError(f"Memory worker device differs from declared hardware for case {index}")
        if record.get("cann_home") != cann_home:
            raise ValueError(f"Memory worker CANN differs from performance worker for case {index}")
        if not hostnames.intersection(record.get("hostnames", [])):
            raise ValueError(f"Memory and performance workers ran on different hosts for case {index}")
        if expected[index]["device_ids"] != [record.get("device_id")]:
            raise ValueError(f"Memory and performance ran on different devices for case {index}")
        if asset_fingerprints(record, opapi_sha256) != kernel_assets:
            raise ValueError(f"Memory worker kernel assets differ for case {index}")
        if record["contract"] != expected[index]["contract"]:
            raise ValueError(f"Memory and performance input contracts differ for case {index}")
        before, peak, output = (int(record[key]) for key in ("baseline_bytes", "peak_bytes", "output_bytes"))
        output_size = math.prod(expected[index]["contract"]["shape"]) * 2
        if output != output_size or before < output_size or peak < before + output:
            raise ValueError(f"Invalid peak memory counters for case {index}")
        if record.get("warmup_count", 0) < 3 or record.get("measurement_count", 0) < 1:
            raise ValueError(f"Insufficient memory warmup or measurement for case {index}")
        extra = peak - before - output
        if extra > MAX_EXTRA_BYTES:
            raise ValueError(f"Case {index} extra memory {extra} bytes exceeds 200M limit")
        result[index] = extra
    return result


def check_matrix(manifest: dict, base: Path, maximum: float, socs=SOCS) -> dict:
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("max-duration-ratio must be finite and positive")
    result = {"passed": False, "max_duration_ratio": maximum, "max_extra_bytes": MAX_EXTRA_BYTES,
              "platforms": {}, "comparisons": [], "errors": []}
    for soc in socs:
        try:
            entry = manifest["platforms"][soc]
            resolve = lambda field: file_path(base / entry[field])
            for field in ("code_id", "cann_version", "device_name", "opapi_sha256"):
                if not entry.get(field):
                    raise ValueError(f"Missing {field}")
            if not entry["device_name"].startswith(DEVICE_NAMES[soc]):
                raise ValueError(f"Device {entry['device_name']} is not {soc}")
            if len(entry["opapi_sha256"]) != 64:
                raise ValueError("Invalid opapi SHA256")
            if resolve("build_exit").read_text().strip() != "0" or resolve("build_log").stat().st_size == 0:
                raise ValueError("Build did not complete successfully")
            if not resolve("cann_version_file").read_text(encoding="utf-8").strip():
                raise ValueError("CANN version evidence is empty")
            provenance = read_json(resolve("runtime_provenance"))
            hostnames = set(provenance["hostnames"])
            if not hostnames or any(not isinstance(value, str) or not value for value in hostnames):
                raise ValueError("Performance worker hostnames are missing")
            kernel_assets = asset_fingerprints(provenance, entry["opapi_sha256"])
            accuracy_provenance = read_json(resolve("accuracy_provenance"))
            if asset_fingerprints(accuracy_provenance, entry["opapi_sha256"]) != kernel_assets:
                raise ValueError("Accuracy worker kernel assets differ from performance worker")
            accuracy_hosts = set(accuracy_provenance["hostnames"])
            if not accuracy_hosts or not hostnames.intersection(accuracy_hosts):
                raise ValueError("Accuracy and performance workers ran on different hosts")
            ids = [int(entry["case_ids"][target]) for target in TARGETS]
            collected = collect(resolve("performance_report"), resolve("profile_root"), entry["node"], ids)
            cases = collected["cases"]
            for target, index in zip(TARGETS, ids):
                worker_tags = [part.split(f"_{provenance['pid']}_", 1)[0]
                               for part in Path(cases[index]["profile"]).parts
                               if f"_{provenance['pid']}_" in part]
                if not worker_tags:
                    raise ValueError(f"Profile and runtime provenance worker PIDs differ for case {index}")
                if not hostnames.intersection(worker_tags):
                    raise ValueError(f"Profile and runtime provenance hostnames differ for case {index}")
                if target_name(cases[index]["contract"]) != target:
                    raise ValueError(f"Wrong model contract for {target}")
                if cases[index]["contract"]["seed"] is None:
                    raise ValueError(f"Case {index} has no fixed seed for comparison with A5")
                if cases[index]["samples"] < 20:
                    raise ValueError(f"Case {index} has fewer than 20 Task Duration samples")
            check_accuracy(resolve("accuracy_report"), ids, cases)
            check_accuracy_worker(resolve("accuracy_report"), resolve("accuracy_log"), ids,
                                  accuracy_provenance["pid"], cases[ids[0]]["device_ids"])
            if any(cases[index]["device_ids"] != cases[ids[0]]["device_ids"] for index in ids):
                raise ValueError("Accuracy cases do not share the declared profiling device")
            cann_home = provenance["environment"]["ASCEND_HOME_PATH"]
            if not cann_home:
                raise ValueError("Performance worker CANN home is missing")
            if entry["cann_version"] not in resolve("cann_version_file").read_text(encoding="utf-8"):
                raise ValueError("Declared CANN version differs from version evidence")
            if accuracy_provenance["environment"]["ASCEND_HOME_PATH"] != cann_home:
                raise ValueError("Accuracy and performance workers used different CANN homes")
            memory = check_memory(resolve("memory_report"), ids, cases, entry["opapi_sha256"],
                                  entry["device_name"], cann_home, hostnames, kernel_assets)
            result["platforms"][soc] = {"code_id": entry["code_id"], "cann_version": entry["cann_version"],
                "device_name": entry["device_name"], "opapi_sha256": entry["opapi_sha256"],
                "kernel_assets_sha256": kernel_assets,
                "cases": {target: dict(cases[index], extra_bytes=memory[index])
                          for target, index in zip(TARGETS, ids)},
                "evidence_sha256": {field: digest(resolve(field)) for field in
                    ("build_exit", "build_log", "cann_version_file", "runtime_provenance", "accuracy_provenance", "accuracy_report", "accuracy_log", "memory_report")}}
        except (KeyError, OSError, ValueError, TypeError) as error:
            result["errors"].append(f"{soc}: {error}")
    if socs == ("A2",):
        result["scope"] = "A2 measurements and memory; no A5 performance comparison"
        result["max_duration_ratio"] = None
        result["passed"] = not result["errors"] and "A2" in result["platforms"]
        return result
    if len(result["platforms"]) == len(SOCS):
        if result["platforms"]["A2"]["code_id"] != result["platforms"]["A3"]["code_id"]:
            result["errors"].append("A2 and A3 tested different source versions")
        for soc in ("A2", "A3"):
            for target in TARGETS:
                actual = result["platforms"][soc]["cases"][target]
                baseline = result["platforms"]["A5"]["cases"][target]
                if actual["contract"] != baseline["contract"]:
                    result["errors"].append(f"{soc}/{target}: input contract differs from A5")
                    continue
                ratio = actual["mean_us"] / baseline["mean_us"]
                passed = ratio <= maximum
                result["comparisons"].append({"soc": soc, "target": target, "duration_ratio": ratio,
                    "actual_us": actual["mean_us"], "a5_us": baseline["mean_us"], "passed": passed})
                if not passed:
                    result["errors"].append(f"{soc}/{target}: duration ratio {ratio:.6f} exceeds {maximum}")
    result["passed"] = not result["errors"] and len(result["comparisons"]) == 4
    return result


def merge_memory(root: Path) -> dict:
    result = {"method": "atk_memory_peak_allocated", "cases": {}, "sources": []}
    for path in sorted(root.glob("case_*.json")):
        data = read_json(path)
        if data["method"] != result["method"]:
            raise ValueError(f"Wrong memory measurement method in {path}")
        result["sources"].append({"path": str(path), "sha256": digest(path)})
        for index, record in data["cases"].items():
            if index in result["cases"]:
                previous = result["cases"][index]
                if (record["contract"] != previous["contract"] or
                    record["mapped_files_sha256"] != previous["mapped_files_sha256"] or
                    record["kernel_assets_sha256"] != previous["kernel_assets_sha256"]):
                    raise ValueError(f"Mismatched memory records for case {index}")
                if record["extra_bytes"] <= previous["extra_bytes"]:
                    continue
            result["cases"][index] = record
    if not result["cases"]:
        raise ValueError("No ATK resource executor memory records")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    inspect = actions.add_parser("inspect")
    inspect.add_argument("--report", required=True, type=file_path)
    inspect.add_argument("--profile-root", required=True, type=file_path)
    inspect.add_argument("--node", required=True)
    check = actions.add_parser("check")
    check.add_argument("--manifest", required=True, type=file_path)
    check.add_argument("--max-duration-ratio", required=True, type=float)
    a2 = actions.add_parser("check-a2")
    a2.add_argument("--manifest", required=True, type=file_path)
    memory = actions.add_parser("memory")
    memory.add_argument("--input-root", required=True, type=file_path)
    for action in (inspect, check, a2, memory):
        action.add_argument("--output", type=file_path)
    args = parser.parse_args()
    try:
        if args.action == "inspect":
            result = collect(args.report, args.profile_root, args.node)
            status = 0
        elif args.action in ("check", "check-a2"):
            result = check_matrix(read_json(args.manifest), args.manifest.parent,
                                  args.max_duration_ratio if args.action == "check" else 1.0,
                                  SOCS if args.action == "check" else ("A2",))
            status = 0 if result["passed"] else 1
        else:
            result, status = merge_memory(args.input_root), 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result, status = {"passed": False, "errors": [str(error)]}, 2
    encoded = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
