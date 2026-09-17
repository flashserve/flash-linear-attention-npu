#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Optional


FIELD_ARGS = (
    (("B", "batch"), "--batch"),
    (("T", "tokens"), "--tokens"),
    (("chunk_size", "chunk-size"), "--chunk-size"),
    (("query_head", "query_heads", "query-heads"), "--query-heads"),
    (("value_head", "value_heads", "value-heads"), "--value-heads"),
    (("Kdim", "key_dim", "key-dim", "dim"), "--key-dim"),
    (("Vdim", "value_dim", "value-dim"), "--value-dim"),
    (("dtype",), "--dtype"),
    (("mean_len", "mean-len"), "--mean-len"),
    (("gate_source", "gate-source", "gate"), "--gate-source"),
    (("gate_function", "gate-function", "gate_fn", "gate-fn"), "--gate-function"),
    (("initial_state", "initial-state"), "--initial-state"),
    (("conv_kernel", "conv-kernel"), "--conv-kernel"),
)

BOOLEAN_FLAGS = (
    (("output_final_state", "output-final-state", "final_state", "final-state"), "--output-final-state"),
)

ACCURACY_TENSORS = {"o", "dq", "dk", "dv", "dbeta", "dg"}
DEFAULT_ACCURACY_TENSORS = ("o", "dq", "dk", "dv", "dbeta", "dg")
ACCURACY_THRESHOLDS = {
    "--accuracy-output-tol": 5e-3,
    "--accuracy-grad-tol": 8e-3,
    "--accuracy-beta-grad-tol": 2e-2,
    "--accuracy-gate-grad-tol": 2e-2,
    "--accuracy-output-cos-min": 0.999,
    "--accuracy-grad-cos-min": 0.999,
    "--accuracy-beta-grad-cos-min": 0.99,
    "--accuracy-gate-grad-cos-min": 0.99,
}
ACCURACY_THRESHOLD_CONTRACT = {
    option.replace("--accuracy-", "", 1).replace("-", "_"): value
    for option, value in ACCURACY_THRESHOLDS.items()
}
METRIC_THRESHOLD_FIELDS = {
    "o": ("output_tol", "output_cos_min"),
    "dq": ("grad_tol", "grad_cos_min"),
    "dk": ("grad_tol", "grad_cos_min"),
    "dv": ("grad_tol", "grad_cos_min"),
    "dbeta": ("beta_grad_tol", "beta_grad_cos_min"),
    "dg": ("gate_grad_tol", "gate_grad_cos_min"),
}
DEFAULT_EXAMPLE_SCRIPT = "examples/flash_gated_delta_rule.py"
REQUIRED_ACCURACY_CASES = {
    "case1_current_default",
    "gdr_accuracy_dense_b2_t128_h2_d128_fp16",
    "gdr_accuracy_varlen_64_64_h2_d128_fp16",
    "gdr_accuracy_tnd_3seq_t3991_h2_d128_fp16",
}
_BASE_REQUIRED_ACCURACY_CONTRACT = {
    "script": DEFAULT_EXAMPLE_SCRIPT,
    "chunk_size": 64,
    "key_dim": 128,
    "value_dim": 128,
    "gate_source": "g",
    "gate_function": "logsigmoid",
    "initial_state": "none",
    "output_final_state": False,
    "demo_model": False,
    "conv_kernel": 4,
    "accuracy_thresholds": ACCURACY_THRESHOLD_CONTRACT,
}
REQUIRED_ACCURACY_CONTRACTS = {
    "case1_current_default": {
        **_BASE_REQUIRED_ACCURACY_CONTRACT,
        "batch": 1,
        "tokens": 4087,
        "query_heads": 32,
        "value_heads": 32,
        "dtype": "bf16",
        "varlen": True,
        "cu_seqlens": [
            0,
            2049,
            3060,
            3573,
            3829,
            3957,
            4022,
            4054,
            4070,
            4077,
            4081,
            4086,
            4087,
        ],
        "mean_len": 1024,
        "qk_l2norm": True,
        "seed": 20260630,
        "scale": None,
        "accuracy_tensors": ["o"],
    },
    "gdr_accuracy_dense_b2_t128_h2_d128_fp16": {
        **_BASE_REQUIRED_ACCURACY_CONTRACT,
        "batch": 2,
        "tokens": 128,
        "query_heads": 2,
        "value_heads": 2,
        "dtype": "fp16",
        "varlen": False,
        "cu_seqlens": [],
        "mean_len": 128,
        "qk_l2norm": False,
        "seed": 42,
        "scale": 0.1,
        "accuracy_tensors": list(DEFAULT_ACCURACY_TENSORS),
    },
    "gdr_accuracy_varlen_64_64_h2_d128_fp16": {
        **_BASE_REQUIRED_ACCURACY_CONTRACT,
        "batch": 1,
        "tokens": 128,
        "query_heads": 2,
        "value_heads": 2,
        "dtype": "fp16",
        "varlen": True,
        "cu_seqlens": [0, 64, 128],
        "mean_len": 1024,
        "qk_l2norm": False,
        "seed": 43,
        "scale": 0.1,
        "accuracy_tensors": list(DEFAULT_ACCURACY_TENSORS),
    },
    "gdr_accuracy_tnd_3seq_t3991_h2_d128_fp16": {
        **_BASE_REQUIRED_ACCURACY_CONTRACT,
        "batch": 1,
        "tokens": 3991,
        "query_heads": 2,
        "value_heads": 2,
        "dtype": "fp16",
        "varlen": True,
        "cu_seqlens": [0, 1024, 2048, 3991],
        "mean_len": 1024,
        "qk_l2norm": False,
        "seed": 44,
        "scale": 0.1,
        "accuracy_tensors": ["o"],
    },
}
REQUIRED_EXACT_CU_SEQLENS = {
    "gdr_accuracy_dense_b2_t128_h2_d128_fp16": [],
    "gdr_accuracy_varlen_64_64_h2_d128_fp16": [0, 64, 128],
}
REQUIRED_SEQUENCE_COUNTS = {
    "case1_current_default": None,
    "gdr_accuracy_dense_b2_t128_h2_d128_fp16": None,
    "gdr_accuracy_varlen_64_64_h2_d128_fp16": 2,
    "gdr_accuracy_tnd_3seq_t3991_h2_d128_fp16": 3,
}
MANAGED_EXTRA_ARG_FLAGS = {
    *(arg_name for _, arg_name in FIELD_ARGS),
    *(arg_name for _, arg_name in BOOLEAN_FLAGS),
    "--case-name",
    "--demo-model",
    "--device",
    "--dim",
    "--heads",
    "--legacy-unfused-core",
    "--no-qk-l2norm",
    "--no-varlen",
    "--qk-l2norm",
    "--varlen",
    *ACCURACY_THRESHOLDS,
}
UNIQUE_REPORTED_EXTRA_OPTIONS = {
    "--accuracy-tensors",
    "--cu-seqlens",
    "--scale",
    "--seed",
}


def _read_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or an object with a cases list.")
    cases: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"case #{index} must be a JSON object.")
        name = str(case.get("name", "")).strip()
        if not name:
            raise ValueError(f"case #{index} is missing a non-empty name.")
        if name in names:
            raise ValueError(f"duplicate Example/ST case name: {name}")
        names.add(name)
        cases.append(case)
    return cases


def _case_get(case: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in case:
            return case[key]
    return None


def _normalize_extra_args(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("extra_args must be a string or a list of strings.")


def _case_script(case: dict[str, Any]) -> str:
    raw_script = str(case.get("script", DEFAULT_EXAMPLE_SCRIPT)).strip().replace("\\", "/")
    script = PurePosixPath(raw_script)
    if not raw_script or script.is_absolute() or ".." in script.parts:
        raise ValueError(f"invalid Example/ST script path: {raw_script!r}")
    normalized = script.as_posix()
    if case.get("name") in REQUIRED_ACCURACY_CASES and normalized != DEFAULT_EXAMPLE_SCRIPT:
        raise ValueError(
            f"required accuracy case {case['name']} must use {DEFAULT_EXAMPLE_SCRIPT}"
        )
    return normalized


def _extra_option_values(extra_args: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(extra_args):
        if item == option:
            if index + 1 >= len(extra_args) or extra_args[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a value in extra_args")
            values.append(extra_args[index + 1])
        elif item.startswith(f"{option}="):
            value = item.split("=", 1)[1]
            if not value:
                raise ValueError(f"{option} requires a value in extra_args")
            values.append(value)
    return values


def _validate_extra_args(extra_args: list[str]) -> None:
    protected_options = MANAGED_EXTRA_ARG_FLAGS | UNIQUE_REPORTED_EXTRA_OPTIONS
    for item in extra_args:
        option = item.split("=", 1)[0]
        if option in MANAGED_EXTRA_ARG_FLAGS:
            raise ValueError(
                f"extra_args cannot override runner-managed option {option}"
            )
        if option not in protected_options and any(
            protected.startswith(option) for protected in protected_options
        ):
            raise ValueError(
                f"extra_args cannot abbreviate protected option {option}"
            )
    for option in UNIQUE_REPORTED_EXTRA_OPTIONS:
        if len(_extra_option_values(extra_args, option)) > 1:
            raise ValueError(f"extra_args must specify {option} at most once")


def _extra_option_value(
    extra_args: list[str], option: str, default: Optional[str] = None
) -> Optional[str]:
    values = _extra_option_values(extra_args, option)
    return values[0] if values else default


def _extra_int(extra_args: list[str], option: str, default: int) -> int:
    value = _extra_option_value(extra_args, option)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{option} must be an integer") from error


def _extra_float(
    extra_args: list[str], option: str, default: Optional[float]
) -> Optional[float]:
    value = _extra_option_value(extra_args, option)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{option} must be a number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{option} must be finite")
    return parsed


def _accuracy_tensor_contract(extra_args: list[str]) -> list[str]:
    raw_value = _extra_option_value(
        extra_args, "--accuracy-tensors", ",".join(DEFAULT_ACCURACY_TENSORS)
    )
    assert raw_value is not None
    tensors = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not tensors or len(tensors) != len(set(tensors)):
        raise ValueError("--accuracy-tensors must contain unique tensor names")
    unknown = [item for item in tensors if item not in ACCURACY_TENSORS]
    if unknown:
        raise ValueError(f"--accuracy-tensors contains unsupported values: {', '.join(unknown)}")
    return tensors


def _normalize_initial_state(value: Any) -> str:
    if isinstance(value, bool):
        return "random" if value else "none"
    value = str(value).strip().lower()
    aliases = {
        "": "none",
        "false": "none",
        "no": "none",
        "none": "none",
        "null": "none",
        "true": "random",
        "yes": "random",
        "rand": "random",
        "random": "random",
        "zero": "zeros",
        "zeros": "zeros",
    }
    if value not in aliases:
        raise ValueError("initial_state must be one of none, zeros, random, true, or false.")
    return aliases[value]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _select_cases(cases: list[dict[str, Any]], case_filter: str) -> list[dict[str, Any]]:
    enabled = [case for case in cases if case.get("enabled", True)]
    if not case_filter.strip():
        return enabled

    wanted = [name.strip() for name in case_filter.split(",") if name.strip()]
    by_name = {case["name"]: case for case in cases}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"unknown Example/ST case(s): {', '.join(missing)}")

    disabled = [name for name in wanted if not by_name[name].get("enabled", True)]
    if disabled:
        raise ValueError(f"requested Example/ST case(s) are disabled: {', '.join(disabled)}")
    return [by_name[name] for name in wanted]


def _required_case_inventory_errors(cases: list[dict[str, Any]]) -> list[str]:
    by_name = {case["name"]: case for case in cases}
    errors = []
    for name in sorted(REQUIRED_ACCURACY_CASES):
        case = by_name.get(name)
        if case is None:
            errors.append(f"missing required accuracy case {name}")
        elif not case.get("enabled", True):
            errors.append(f"required accuracy case {name} is disabled")
    return errors


def _build_command(repo_root: Path, device: int, case: dict[str, Any]) -> list[str]:
    script_name = _case_script(case)
    script = (repo_root / script_name).resolve()
    resolved_root = repo_root.resolve()
    if script != resolved_root and resolved_root not in script.parents:
        raise ValueError(f"Example/ST script escapes the repository: {script_name}")
    extra_args = _normalize_extra_args(case.get("extra_args"))
    _validate_extra_args(extra_args)
    contract_errors = _required_case_contract_errors(str(case["name"]), _case_contract(case))
    if contract_errors:
        raise ValueError(
            f"required accuracy case {case['name']} contract mismatch: "
            + ", ".join(contract_errors)
        )
    cmd = [
        sys.executable,
        str(script),
        "--device",
        str(device),
        "--case-name",
        str(case["name"]),
    ]
    for aliases, arg_name in FIELD_ARGS:
        value = _case_get(case, aliases)
        if value is not None:
            if arg_name == "--initial-state":
                value = _normalize_initial_state(value)
            elif arg_name in ("--gate-source", "--gate-function"):
                value = str(value).strip().lower()
            cmd.extend([arg_name, str(value)])

    if _as_bool(case.get("demo_model")):
        cmd.append("--demo-model")
    for aliases, arg_name in BOOLEAN_FLAGS:
        if _as_bool(_case_get(case, aliases)):
            cmd.append(arg_name)
    if not _as_bool(case.get("varlen"), default=True):
        cmd.append("--no-varlen")
    if not _as_bool(case.get("qk_l2norm", case.get("qk-l2norm")), default=True):
        cmd.append("--no-qk-l2norm")
    if "--accuracy-check" in extra_args:
        for option, value in ACCURACY_THRESHOLDS.items():
            cmd.extend([option, str(value)])
    cmd.extend(extra_args)
    return cmd


def _parse_metric_value(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else value.lower()
    except ValueError:
        return value


def _parse_accuracy_metric(line: str) -> Optional[dict[str, Any]]:
    stripped = line.strip()
    if ": " not in stripped:
        return None
    name, values = stripped.split(": ", 1)
    if name not in ACCURACY_TENSORS:
        return None
    metric: dict[str, Any] = {"tensor": name, "raw": stripped}
    for item in values.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        metric[key] = _parse_metric_value(value)
    return metric


def _golden_action(line: str) -> str:
    if "reused" in line:
        return "reused"
    if "config mismatch" in line:
        return "regenerated"
    if "generated" in line:
        return "generated"
    return "unknown"


def _case_expects_accuracy(case: dict[str, Any]) -> bool:
    return "--accuracy-check" in _normalize_extra_args(case.get("extra_args"))


def _case_contract(case: dict[str, Any]) -> dict[str, Any]:
    extra_args = _normalize_extra_args(case.get("extra_args"))
    _validate_extra_args(extra_args)
    cu_seqlens: list[int] = []
    cu_seqlens_value = _extra_option_value(extra_args, "--cu-seqlens")
    if cu_seqlens_value is not None:
        try:
            cu_seqlens = [int(item) for item in cu_seqlens_value.split(",")]
        except ValueError as error:
            raise ValueError("--cu-seqlens must contain integers") from error
    expects_accuracy = _case_expects_accuracy(case)
    return {
        "script": _case_script(case),
        "batch": _case_get(case, ("B", "batch")),
        "tokens": _case_get(case, ("T", "tokens")),
        "chunk_size": _case_get(case, ("chunk_size", "chunk-size")),
        "query_heads": _case_get(case, ("query_head", "query_heads", "query-heads")),
        "value_heads": _case_get(case, ("value_head", "value_heads", "value-heads")),
        "key_dim": _case_get(case, ("Kdim", "key_dim", "key-dim")),
        "value_dim": _case_get(case, ("Vdim", "value_dim", "value-dim")),
        "dtype": str(case.get("dtype", "")).strip().lower(),
        "varlen": _as_bool(case.get("varlen"), default=True),
        "cu_seqlens": cu_seqlens,
        "mean_len": int(_case_get(case, ("mean_len", "mean-len")) or 1024),
        "gate_source": str(_case_get(case, ("gate_source", "gate-source", "gate")) or "g").strip().lower(),
        "gate_function": str(
            _case_get(case, ("gate_function", "gate-function", "gate_fn", "gate-fn"))
            or "logsigmoid"
        ).strip().lower(),
        "initial_state": _normalize_initial_state(
            _case_get(case, ("initial_state", "initial-state"))
        ),
        "output_final_state": _as_bool(
            _case_get(case, ("output_final_state", "output-final-state", "final_state", "final-state"))
        ),
        "qk_l2norm": _as_bool(
            case.get("qk_l2norm", case.get("qk-l2norm")), default=True
        ),
        "demo_model": _as_bool(case.get("demo_model")),
        "conv_kernel": int(_case_get(case, ("conv_kernel", "conv-kernel")) or 4),
        "seed": _extra_int(extra_args, "--seed", 42),
        "scale": _extra_float(extra_args, "--scale", None),
        "accuracy_tensors": _accuracy_tensor_contract(extra_args)
        if expects_accuracy
        else [],
        "accuracy_thresholds": ACCURACY_THRESHOLD_CONTRACT if expects_accuracy else {},
    }


def _required_case_contract_errors(name: str, contract: dict[str, Any]) -> list[str]:
    expected = REQUIRED_ACCURACY_CONTRACTS.get(name)
    if expected is None:
        return []
    if not isinstance(contract, dict):
        return ["contract"]
    errors = sorted(
        key
        for key in (set(expected) | set(contract)) - {"cu_seqlens"}
        if contract.get(key) != expected.get(key)
    )
    cu_seqlens = contract.get("cu_seqlens")
    expected_exact = REQUIRED_EXACT_CU_SEQLENS.get(name)
    valid_boundaries = (
        isinstance(cu_seqlens, list)
        and all(type(value) is int for value in cu_seqlens)
        and (
            (not expected["varlen"] and cu_seqlens == [])
            or (
                expected["varlen"]
                and len(cu_seqlens) >= 2
                and cu_seqlens[0] == 0
                and cu_seqlens[-1] == expected["tokens"]
                and all(right > left for left, right in zip(cu_seqlens, cu_seqlens[1:]))
            )
        )
    )
    sequence_count = REQUIRED_SEQUENCE_COUNTS[name]
    if expected_exact is not None and cu_seqlens != expected_exact:
        valid_boundaries = False
    if sequence_count is not None and (
        not isinstance(cu_seqlens, list) or len(cu_seqlens) != sequence_count + 1
    ):
        valid_boundaries = False
    if not valid_boundaries:
        errors.append("cu_seqlens")
    return sorted(errors)


def _required_case_report_errors(report: dict[str, Any]) -> list[str]:
    name = str(report.get("name", ""))
    expected = REQUIRED_ACCURACY_CONTRACTS.get(name)
    if expected is None:
        return []
    errors = [
        f"contract mismatch ({', '.join(fields)})"
        for fields in [_required_case_contract_errors(name, report.get("contract", {}))]
        if fields
    ]
    metrics = report.get("metrics")
    if not isinstance(metrics, list):
        return [*errors, "metrics is not an array"]
    metric_names = [
        metric.get("tensor", "") if isinstance(metric, dict) else ""
        for metric in metrics
    ]
    duplicates = sorted(
        {name for name in metric_names if name and metric_names.count(name) > 1}
    )
    if duplicates:
        errors.append(f"duplicate metrics ({', '.join(duplicates)})")
    expected_names = expected["accuracy_tensors"]
    if len(metric_names) != len(expected_names) or any(
        name not in metric_names for name in expected_names
    ):
        errors.append(
            "metric set mismatch "
            f"(expected {','.join(expected_names)}; actual {','.join(filter(None, metric_names)) or 'empty'})"
        )
    for metric in metrics:
        if not isinstance(metric, dict) or metric.get("tensor") not in METRIC_THRESHOLD_FIELDS:
            continue
        if any(metric.get(field) is not True for field in ("finite", "allclose", "cosine_ok")):
            errors.append(f"metric checks failed ({metric['tensor']})")
        tol_field, cos_min_field = METRIC_THRESHOLD_FIELDS[metric["tensor"]]
        if (
            metric.get("tol") != ACCURACY_THRESHOLD_CONTRACT[tol_field]
            or metric.get("cos_min") != ACCURACY_THRESHOLD_CONTRACT[cos_min_field]
        ):
            errors.append(f"metric threshold mismatch ({metric['tensor']})")
    return errors


def _enforce_required_case_report(report: dict[str, Any]) -> None:
    if report.get("status") != "passed" or report.get("accuracy_status") != "passed":
        return
    errors = _required_case_report_errors(report)
    if not errors:
        return
    report["validation_errors"] = errors
    report["status"] = "failed"
    report["return_code"] = report.get("return_code") or 2
    report["accuracy_status"] = "failed"
    for error in errors:
        print(f"[CI][ERROR] {report.get('name', 'unknown')}: {error}", flush=True)


def _enforce_accuracy_completion(report: dict[str, Any]) -> None:
    if (
        report.get("accuracy_check") is not True
        or report.get("return_code") != 0
        or report.get("accuracy_status") == "passed"
    ):
        return
    error = "accuracy check did not complete successfully"
    report.setdefault("validation_errors", []).append(error)
    report["status"] = "failed"
    report["return_code"] = 2
    report["accuracy_status"] = "failed"
    print(f"[CI][ERROR] {report.get('name', 'unknown')}: {error}", flush=True)


def _blank_case_report(case: dict[str, Any], status: str) -> dict[str, Any]:
    expects_accuracy = _case_expects_accuracy(case)
    return {
        "name": str(case["name"]),
        "description": str(case.get("description", "")).strip(),
        "contract": _case_contract(case),
        "status": status,
        "return_code": None,
        "duration_sec": 0.0,
        "accuracy_check": expects_accuracy,
        "accuracy_status": "not_run" if status == "not_run" and expects_accuracy else "not_requested",
        "golden": "not_run",
        "metrics": [],
    }


def _run_case(cmd: list[str], repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    report = _blank_case_report(case, "running")
    report["command"] = shlex.join(cmd)
    process = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    metrics: list[dict[str, Any]] = []
    accuracy_requested = "--accuracy-check" in cmd
    accuracy_started = False
    accuracy_passed = False
    golden = "not_requested"
    for line in process.stdout:
        print(line, end="", flush=True)
        stripped = line.strip()
        if stripped.startswith("accuracy golden:"):
            golden = _golden_action(stripped)
        elif stripped == "accuracy check:":
            accuracy_started = True
        elif stripped == "accuracy check passed":
            accuracy_started = True
            accuracy_passed = True
        metric = _parse_accuracy_metric(stripped)
        if metric is not None:
            accuracy_started = True
            metrics.append(metric)
    return_code = process.wait()
    accuracy_check = accuracy_requested or accuracy_started
    if accuracy_passed and return_code == 0:
        accuracy_status = "passed"
    elif accuracy_check:
        accuracy_status = "not_run" if return_code != 0 and not accuracy_started else "failed"
    else:
        accuracy_status = "not_requested"
    report.update(
        {
            "status": "passed" if return_code == 0 else "failed",
            "return_code": return_code,
            "duration_sec": round(time.monotonic() - started, 3),
            "accuracy_check": accuracy_check,
            "accuracy_status": accuracy_status,
            "golden": golden,
            "metrics": metrics,
        }
    )
    _enforce_accuracy_completion(report)
    _enforce_required_case_report(report)
    return report


def _summarize_report(cases: list[dict[str, Any]]) -> dict[str, int]:
    accuracy_cases = [case for case in cases if case.get("accuracy_check") or case.get("accuracy_status") == "not_run"]
    return {
        "total": len(cases),
        "passed": sum(1 for case in cases if case.get("status") == "passed"),
        "failed": sum(1 for case in cases if case.get("status") == "failed"),
        "not_run": sum(1 for case in cases if case.get("status") == "not_run"),
        "accuracy_total": len(accuracy_cases),
        "accuracy_passed": sum(1 for case in accuracy_cases if case.get("accuracy_status") == "passed"),
        "accuracy_failed": sum(1 for case in accuracy_cases if case.get("accuracy_status") == "failed"),
        "accuracy_not_run": sum(1 for case in accuracy_cases if case.get("accuracy_status") == "not_run"),
    }


def _report_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "platform": os.environ.get("CI_ACCURACY_PLATFORM", ""),
        "soc": os.environ.get("CI_SOC") or os.environ.get("FLA_NPU_SOC", ""),
        "head_sha": os.environ.get("CI_ACCURACY_HEAD_SHA") or os.environ.get("NPU_CI_TARGET_SHA", ""),
        "run_id": os.environ.get("CI_ACCURACY_RUN_ID") or os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("CI_ACCURACY_RUN_ATTEMPT") or os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "cases_file": args.cases_file,
        "case_filter": args.case_filter,
        "device": args.device,
    }


def _write_accuracy_report(
    path: Optional[Path],
    cases: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    complete: bool,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "gdr-accuracy-report-v1",
        "complete": complete,
        "metadata": metadata,
        "summary": _summarize_report(cases),
        "cases": cases,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configured Example/ST cases.")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--cases-file", default="ci/example_st_cases.json")
    parser.add_argument("--case-filter", default="", help="Comma-separated case names to run")
    parser.add_argument("--accuracy-report-file", default=os.environ.get("CI_ACCURACY_REPORT_FILE", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cases_file = (repo_root / args.cases_file).resolve()
    all_cases = _read_cases(cases_file)
    default_cases_file = (repo_root / "ci/example_st_cases.json").resolve()
    if cases_file == default_cases_file and not args.case_filter.strip():
        inventory_errors = _required_case_inventory_errors(all_cases)
        if inventory_errors:
            for error in inventory_errors:
                print(f"[CI][ERROR] {error}", flush=True)
            return 2
    cases = _select_cases(all_cases, args.case_filter)
    if not cases:
        raise SystemExit(f"No enabled Example/ST cases found in {cases_file}.")

    report_path = Path(args.accuracy_report_file) if args.accuracy_report_file else None
    if report_path is not None and not report_path.is_absolute():
        report_path = repo_root / report_path
    metadata = _report_metadata(args)

    print(f"[CI] Example/ST cases file: {cases_file}")
    case_reports: list[dict[str, Any]] = []
    failed_return_code = 0
    for index, case in enumerate(cases, start=1):
        name = case["name"]
        description = str(case.get("description", "")).strip()
        cmd = _build_command(repo_root, args.device, case)
        print(f"[CI] Example/ST case {index}/{len(cases)}: {name}")
        if description:
            print(f"[CI] {description}")
        print(f"[CI] Command: {shlex.join(cmd)}")
        if args.dry_run:
            continue
        case_report = _run_case(cmd, repo_root, case)
        case_reports.append(case_report)
        _write_accuracy_report(report_path, case_reports, metadata, complete=False)
        if case_report["return_code"] != 0:
            failed_return_code = int(case_report["return_code"])
            for remaining in cases[index:]:
                case_reports.append(_blank_case_report(remaining, "not_run"))
            break
    if not args.dry_run:
        _write_accuracy_report(report_path, case_reports, metadata, complete=True)
    return failed_return_code


if __name__ == "__main__":
    raise SystemExit(main())
