#!/usr/bin/env python3
import argparse
import json
import os
import signal
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


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


def _as_non_negative_int(value: Any, name: str) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if timeout < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return timeout


def _default_timeout_seconds() -> int:
    return _as_non_negative_int(os.environ.get("CI_EXAMPLE_CASE_TIMEOUT_SECONDS", "1800"), "CI_EXAMPLE_CASE_TIMEOUT_SECONDS")


def _case_timeout_seconds(case: dict[str, Any], default_timeout: int) -> int:
    value = case.get("timeout_seconds", case.get("timeout-seconds"))
    if value is None:
        return default_timeout
    return _as_non_negative_int(value, f"timeout_seconds for case {case['name']}")


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    proc.wait()


def _run_command(cmd: list[str], cwd: Path, timeout_seconds: int) -> None:
    timeout = None if timeout_seconds == 0 else timeout_seconds
    proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=(os.name == "posix"))
    try:
        return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc)
        raise exc
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


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


def _build_command(repo_root: Path, device: int, case: dict[str, Any]) -> list[str]:
    script = repo_root / str(case.get("script", "examples/flash_gated_delta_rule.py"))
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
    cmd.extend(_normalize_extra_args(case.get("extra_args")))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configured Example/ST cases.")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--cases-file", default="ci/example_st_cases.json")
    parser.add_argument("--case-filter", default="", help="Comma-separated case names to run")
    parser.add_argument("--timeout-seconds", type=int, default=_default_timeout_seconds(), help="Default per-case timeout; 0 disables timeout")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds < 0:
        raise SystemExit("--timeout-seconds must be a non-negative integer.")

    repo_root = Path(__file__).resolve().parents[1]
    cases_file = (repo_root / args.cases_file).resolve()
    cases = _select_cases(_read_cases(cases_file), args.case_filter)
    if not cases:
        raise SystemExit(f"No enabled Example/ST cases found in {cases_file}.")

    print(f"[CI] Example/ST cases file: {cases_file}", flush=True)
    for index, case in enumerate(cases, start=1):
        name = case["name"]
        description = str(case.get("description", "")).strip()
        cmd = _build_command(repo_root, args.device, case)
        timeout_seconds = _case_timeout_seconds(case, args.timeout_seconds)
        print(f"[CI] Example/ST case {index}/{len(cases)}: {name}", flush=True)
        if description:
            print(f"[CI] {description}", flush=True)
        print(f"[CI] Timeout: {'disabled' if timeout_seconds == 0 else str(timeout_seconds) + 's'}", flush=True)
        print(f"[CI] Command: {shlex.join(cmd)}", flush=True)
        if not args.dry_run:
            try:
                _run_command(cmd, repo_root, timeout_seconds)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"[CI][ERROR] Example/ST case {name} timed out after {timeout_seconds}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
