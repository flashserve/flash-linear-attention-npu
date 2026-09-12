#!/usr/bin/env python3
"""Create a compact, sanitized summary for an NPU CI run."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA = "npu-ci-diagnostics-v1"
MAX_INPUT_LINE_CHARS = 16_384
MAX_LINE_CHARS = 320
MAX_HEADLINE_CHARS = 240
MAX_LINES_PER_BLOCK = 8
MAX_BLOCKS_PER_CATEGORY = 4
MAX_FAILURE_CASES = 10
MAX_METRICS_PER_CASE = 6
MAX_TRACEBACK_SCAN_LINES = 80
MAX_JSON_BYTES = 64 * 1024
MAX_MARKDOWN_CHARS = 24 * 1024
MAX_MARKDOWN_BLOCKS_PER_CATEGORY = 2
MAX_MARKDOWN_LINES_PER_BLOCK = 4
MAX_MARKDOWN_FAILURE_CASES = 4
MAX_MARKDOWN_METRICS_PER_CASE = 3
MAX_REPRO_COMMAND_CHARS = 2_000

ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CARET_ANSI_RE = re.compile(r"\^\[\[[0-?]*[ -/]*[@-~]")
GITHUB_LOG_PREFIX_RE = re.compile(
    r"^[^\t]+\t[^\t]+\t\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s?"
)
GITHUB_COMMAND_ECHO_RE = re.compile(
    r"^(?:\x1b\[36;1m|\^\[\[36;1m).*(?:\x1b\[0m|\^\[\[0m)$"
)
PRIVATE_IP_RE = re.compile(
    r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|127(?:\.\d{1,3}){3}|"
    r"169\.254(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])"
)
PRIVATE_IPV6_RE = re.compile(
    r"(?i)(?<![0-9a-f:])(?:::1|(?:f[cd][0-9a-f]{2}|fe[89ab][0-9a-f]):"
    r"(?:[0-9a-f]{0,4}:){1,6}[0-9a-f]{0,4})(?![0-9a-f:])"
)
URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@")
USER_HOST_RE = re.compile(r"(?i)\b[a-z_][a-z0-9_.-]{0,63}@[a-z0-9_.-]+\b")
INTERNAL_DNS_RE = re.compile(
    r"(?i)\b(?:[a-z0-9-]+\.)+(?:local|internal|corp)\b"
)
MACHINE_NAME_RE = re.compile(
    r"(?i)\b(?=[a-z0-9-]*(?:runner|worker|host|node))"
    r"(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+)+\b"
)
CLI_SECRET_RE = re.compile(
    r"(?i)(?<![a-z0-9_-])(?P<key>--(?:[a-z][a-z0-9]*[-_])*"
    r"(?:authorization|token|password|passwd|secret|api[-_]?key|"
    r"access[-_]?key|secret[-_]?key|private[-_]?key))"
    r"(?P<separator>\s*=\s*|\s+)"
    r"(?:\[REDACTED\]|\"(?:\\.|[^\"\\\r\n])*\"|"
    r"'(?:\\.|[^'\\\r\n])*'|[^\s,;\r\n}\]]+)"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![a-z0-9_-])(?P<quote>[\"']?)"
    r"(?P<key>(?:[a-z][a-z0-9]*[-_])*(?:authorization|token|password|passwd|"
    r"secret|api[-_]?key|access[-_]?key|secret[-_]?key|private[-_]?key))(?P=quote)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<value>\[REDACTED\]|\"(?:\\.|[^\"\\\r\n])*\"|"
    r"'(?:\\.|[^'\\\r\n])*'|(?:bearer\s+|basic\s+)?[^,;\r\n}\]]+?"
    r"(?=\s+[\"']?[a-z][a-z0-9_-]*[\"']?\s*[:=]|[,;\r\n}\]]|$))"
)
HOST_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![a-z0-9_-])(?P<quote>[\"']?)"
    r"(?P<key>host(?:name)?|runner|machine|worker|node)(?P=quote)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<value>\[INTERNAL_HOST\]|\"(?:\\.|[^\"\\\r\n])*\"|"
    r"'(?:\\.|[^'\\\r\n])*'|[a-z0-9][a-z0-9_.-]*)"
)
HOST_CONTEXT_RE = re.compile(
    r"(?i)\b((?:on|via|from)\s+host\s+)([a-z0-9][a-z0-9_.-]*)"
)
SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|"
    r"glpat-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16})\b"
)
WINDOWS_PATH_RE = re.compile(r"(?i)(?<![\w])(?:[a-z]:[\\/])[^\s\"'<>|]+")
REPO_PREFIX_RE = re.compile(
    r"(?i)(?:[a-z]:)?/(?:[^\s/:]+/)*"
    r"(?:flash-linear-attention-npu(?:-[^/\s:]*)?|workspace/repo|github/workspace)/"
)
ENV_PATH_RE = re.compile(
    r"(?<![\w.])/(?:data|workspace|root|home|tmp|opt|usr|var|mnt|etc|srv|run|"
    r"__w|github|runner)"
    r"(?:/[^\s\"'<>:]*)*(?=$|[\s\"'<>:,;.\)\]])"
)
UNIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![:/\w.])/(?:[^\s/\"'<>:]+/)+[^\s\"'<>:]*"
)

COMPILER_ERROR_RE = re.compile(
    r"(?i)^(?:.*?:\d+(?::\d+)?:\s*|[A-Za-z0-9_.+-]+:\s*)?"
    r"(?:fatal\s+)?error:\s*(.+)$"
)
COMPILE_SPECIFIC_RE = re.compile(
    r"(?i)(?:cmake error|undefined reference|ld(?:\.lld)?: error|"
    r"(?:clang\+\+|clang|g\+\+|gcc): error|linker command failed|"
    r"compiler command failed)"
)
COMPILE_GENERIC_RE = re.compile(
    r"(?i)(?:ninja: build stopped|subcommand failed|make(?:\[\d+\])?: \*\*\* .*error|"
    r"calledprocesserror.*(?:cmake|make|ninja|compiler)|compilation failed|"
    r"compilation terminated\.|all warnings being treated as errors|error generated\.)"
)
INFRASTRUCTURE_RE = re.compile(
    r"(?i)(?:no space left on device|docker daemon|cannot connect to (?:the )?docker|"
    r"required preloaded docker image is missing|no npu device was found|"
    r"timed out waiting for an unlocked npu|runner.*(?:offline|lost|disconnect)|"
    r"connection (?:reset|refused|timed out)|network is unreachable|"
    r"temporary failure in name resolution|failed to (?:download|fetch)|"
    r"disk quota exceeded)"
)
RUNTIME_RE = re.compile(
    r"(?i)(?:^\s*\[(?:ERROR|FAIL|FATAL|TIMEOUT)\]|\[CI\]\[ERROR\]|"
    r"runtimeerror:|assertionerror:|valueerror:|typeerror:|"
    r"segmentation fault|core dumped|accuracy check failed|"
    r"\b(?:acl|aclnn|npu|kernel|device|stream)\b.*(?:error|failed)|"
    r"(?:error|failed).*\b(?:acl|aclnn|npu|kernel|device|stream)\b|"
    r"\b(?:EZ|EE|EJ)\d{4,}\b)"
)
FAILURE_MARKER_RE = re.compile(r"(?i)^\s*\[(?:FAIL|FATAL|TIMEOUT)\]")
TRACEBACK_END_RE = re.compile(r"^[\w.]+(?:Error|Exception):\s*.+")
WARNING_RE = re.compile(r"(?i)(?:\bwarning:|\[warn(?:ing)?\])")
PROMOTED_WARNING_RE = re.compile(r"(?i)\bwarning:.*\[-Werror(?:[=,\]])")
SOURCE_EXCERPT_RE = re.compile(r"^\s*\d+\s*\|")
SOURCE_COMPILER_ERROR_RE = re.compile(
    r"(?i)\.(?:c|cc|cpp|cxx|h|hh|hpp|inc|cu|cuh):\d+(?::\d+)?:\s*"
    r"(?:fatal\s+)?error:"
)
SOURCE_LOCATION_RE = re.compile(
    r"(?i)([^\s:|]+\.(?:c|cc|cpp|cxx|h|hh|hpp|inc|cu|cuh):\d+(?::\d+)?)"
)
CASCADE_MISSING_OBJECT_RE = re.compile(
    r"(?i)ld(?:\.lld)?: error: cannot open .*kernel_meta"
)


def _truncate(value: str, limit: int) -> str:
    value = value.replace("\x00", "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 15)] + "... [truncated]"


def _redact_secret_assignment(match: re.Match[str]) -> str:
    key = match.group("key")
    separator = match.group("separator")
    quote = match.group("quote")
    is_spaced_lowercase_token_variable = (
        not quote
        and key == key.lower()
        and re.search(r"(?:^|[-_])token$", key) is not None
        and re.fullmatch(r"\s+[:=]\s+", separator) is not None
    )
    if is_spaced_lowercase_token_variable:
        return match.group(0)
    return f"{quote}{key}{quote}{separator}[REDACTED]"


def _redact_host_assignment(match: re.Match[str]) -> str:
    key = match.group("key")
    separator = match.group("separator")
    quote = match.group("quote")
    is_spaced_lowercase_business_variable = (
        not quote
        and key in {"host", "worker", "node"}
        and re.fullmatch(r"\s+[:=]\s+", separator) is not None
    )
    if is_spaced_lowercase_business_variable:
        return match.group(0)
    return f"{quote}{key}{quote}{separator}[INTERNAL_HOST]"


def sanitize_text(value: Any, limit: int = MAX_LINE_CHARS) -> str:
    """Remove terminal control sequences and non-public environment details."""

    text = CARET_ANSI_RE.sub("", ANSI_RE.sub("", str(value)))
    text = URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", text)
    text = CLI_SECRET_RE.sub(
        lambda match: (
            f"{match.group('key')}{match.group('separator')}[REDACTED]"
        ),
        text,
    )
    text = SECRET_ASSIGNMENT_RE.sub(_redact_secret_assignment, text)
    text = SECRET_VALUE_RE.sub("[REDACTED]", text)
    text = PRIVATE_IP_RE.sub("[PRIVATE_IP]", text)
    text = PRIVATE_IPV6_RE.sub("[PRIVATE_IP]", text)
    text = USER_HOST_RE.sub("[REDACTED_USER_HOST]", text)
    text = INTERNAL_DNS_RE.sub("[INTERNAL_HOST]", text)
    text = HOST_ASSIGNMENT_RE.sub(_redact_host_assignment, text)
    text = HOST_CONTEXT_RE.sub(
        lambda match: f"{match.group(1)}[INTERNAL_HOST]", text
    )
    text = MACHINE_NAME_RE.sub("[INTERNAL_HOST]", text)
    text = REPO_PREFIX_RE.sub("", text)
    text = WINDOWS_PATH_RE.sub("<PATH>", text)
    text = ENV_PATH_RE.sub("<PATH>", text)
    text = UNIX_ABSOLUTE_PATH_RE.sub("<PATH>", text)
    text = "".join(char for char in text if char in "\t\n\r" or ord(char) >= 32)
    return _truncate(text.rstrip("\r\n"), limit)


def sanitize_inline(value: Any, limit: int = MAX_LINE_CHARS) -> str:
    """Sanitize a value that must stay on one Markdown or shell line."""

    text = sanitize_text(value, max(limit * 2, MAX_LINE_CHARS))
    return _truncate(" ".join(text.splitlines()), limit)


def sanitize_markdown_inline(value: Any, limit: int = MAX_LINE_CHARS) -> str:
    """Sanitize untrusted text embedded in prose, inline code, or tables."""

    text = sanitize_inline(value, limit)
    return (
        text.replace("@", "_at_")
        .replace("`", "'")
        .replace("<", "(")
        .replace(">", ")")
        .replace("[", "(")
        .replace("]", ")")
    )


def _iter_log_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line[:MAX_INPUT_LINE_CHARS]
            exported_github_log = GITHUB_LOG_PREFIX_RE.match(line) is not None
            line = GITHUB_LOG_PREFIX_RE.sub("", line)
            if exported_github_log and GITHUB_COMMAND_ECHO_RE.match(line):
                continue
            yield CARET_ANSI_RE.sub("", ANSI_RE.sub("", line)).rstrip("\r\n")


def _diagnostic_key(headline: str, lines: list[str]) -> str:
    locations = []
    for line in lines:
        locations.extend(match.group(1) for match in SOURCE_LOCATION_RE.finditer(line))
    source_context = [
        line.strip()
        for line in lines[1:]
        if "^" in line or "~" in line or "memcpy" in line or "error:" in line.lower()
    ]
    return "\n".join([headline.lower(), *dict.fromkeys(locations), *source_context])


def _add_block(
    blocks: list[dict[str, Any]],
    seen: dict[str, dict[str, Any]],
    headline: str,
    lines: list[str],
    *,
    key: Optional[str] = None,
    priority: bool = False,
) -> None:
    clean_headline = sanitize_text(headline, MAX_HEADLINE_CHARS).strip()
    clean_lines = [
        sanitize_text(line, MAX_LINE_CHARS)
        for line in lines[:MAX_LINES_PER_BLOCK]
        if sanitize_text(line, MAX_LINE_CHARS).strip()
    ]
    if not clean_headline:
        clean_headline = "CI failure"
    if not clean_lines:
        clean_lines = [clean_headline]
    normalized_key = key or _diagnostic_key(clean_headline, clean_lines)
    if normalized_key in seen:
        seen[normalized_key]["occurrences"] += 1
        return
    if len(blocks) >= MAX_BLOCKS_PER_CATEGORY:
        if not priority:
            return
        removed_index = next(
            (
                index
                for index, existing_block in enumerate(blocks)
                if not existing_block.get("_priority", False)
            ),
            len(blocks) - 1,
        )
        removed = blocks.pop(removed_index)
        for seen_key, seen_block in list(seen.items()):
            if seen_block is removed:
                del seen[seen_key]
                break
    block = {
        "headline": clean_headline,
        "lines": clean_lines,
        "occurrences": 1,
        "_priority": priority,
    }
    blocks.append(block)
    seen[normalized_key] = block


def _is_compile_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped or WARNING_RE.search(stripped):
        return False
    if stripped.startswith("[") and not stripped.startswith("[ERROR]"):
        return False
    return bool(
        line[:1].isspace()
        or re.match(r"^\s*(?:\d+\s*)?\|", line)
        or re.match(r"^\s*[\^~]+", line)
        or "memcpy(" in line
        or re.search(r"(?i)\bnote:", line)
    )


def extract_diagnostics(log_file: Path, enabled: bool = True) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {
        "compile": [],
        "runtime": [],
        "infrastructure": [],
    }
    if not enabled or not log_file.is_file():
        return diagnostics

    seen = {category: {} for category in diagnostics}
    pending_compile: Optional[dict[str, Any]] = None
    traceback_lines: Optional[list[str]] = None
    traceback_seen = 0
    generic_compile: list[tuple[str, list[str]]] = []
    failure_markers: list[tuple[str, list[str]]] = []

    def finish_compile() -> None:
        nonlocal pending_compile
        if pending_compile is None:
            return
        _add_block(
            diagnostics["compile"],
            seen["compile"],
            pending_compile["headline"],
            pending_compile["lines"],
            key=_diagnostic_key(
                sanitize_text(pending_compile["headline"], MAX_HEADLINE_CHARS),
                [sanitize_text(item) for item in pending_compile["lines"]],
            ),
        )
        pending_compile = None

    def finish_traceback() -> None:
        nonlocal traceback_lines, traceback_seen
        if not traceback_lines:
            traceback_lines = None
            return
        final_line = next(
            (line.strip() for line in reversed(traceback_lines) if line.strip()),
            "Python traceback",
        )
        _add_block(
            diagnostics["runtime"],
            seen["runtime"],
            final_line,
            traceback_lines,
            priority=True,
        )
        traceback_lines = None
        traceback_seen = 0

    for line in _iter_log_lines(log_file):
        stripped = line.strip()

        compiler_match = COMPILER_ERROR_RE.search(stripped)
        if pending_compile is not None:
            if compiler_match:
                finish_compile()
            elif _is_compile_continuation(line) and len(pending_compile["lines"]) < MAX_LINES_PER_BLOCK:
                pending_compile["lines"].append(line)
                continue
            else:
                finish_compile()

        if traceback_lines is not None:
            traceback_seen += 1
            if len(traceback_lines) < MAX_LINES_PER_BLOCK:
                traceback_lines.append(line)
            else:
                # Keep the exception line visible even for a long traceback.
                traceback_lines[-1] = line
            if TRACEBACK_END_RE.match(stripped):
                finish_traceback()
            elif traceback_seen >= MAX_TRACEBACK_SCAN_LINES:
                finish_traceback()
            continue

        if not stripped or (
            WARNING_RE.search(stripped) and not PROMOTED_WARNING_RE.search(stripped)
        ):
            continue

        if compiler_match:
            pending_compile = {
                "headline": compiler_match.group(1),
                "lines": [line],
            }
            continue

        if stripped.startswith("Traceback (most recent call last):"):
            traceback_lines = [line]
            traceback_seen = 1
            continue

        if INFRASTRUCTURE_RE.search(stripped):
            _add_block(
                diagnostics["infrastructure"],
                seen["infrastructure"],
                stripped,
                [line],
            )
            continue

        if COMPILE_SPECIFIC_RE.search(stripped):
            pending_compile = {"headline": stripped, "lines": [line]}
            continue

        if PROMOTED_WARNING_RE.search(stripped):
            pending_compile = {"headline": stripped, "lines": [line]}
            continue

        if COMPILE_GENERIC_RE.search(stripped):
            generic_compile.append((stripped, [line]))
            continue

        if FAILURE_MARKER_RE.match(stripped):
            failure_markers.append((stripped, [line]))
            continue

        if not SOURCE_EXCERPT_RE.match(line) and RUNTIME_RE.search(stripped):
            _add_block(
                diagnostics["runtime"],
                seen["runtime"],
                stripped,
                [line],
                priority=FAILURE_MARKER_RE.match(stripped) is None,
            )

    finish_compile()
    finish_traceback()
    if not diagnostics["compile"] and generic_compile:
        for headline, lines in generic_compile:
            _add_block(
                diagnostics["compile"], seen["compile"], headline, lines
            )
    if not any(diagnostics.values()):
        for headline, lines in failure_markers:
            _add_block(
                diagnostics["runtime"], seen["runtime"], headline, lines
            )
    if any(
        SOURCE_COMPILER_ERROR_RE.search(line)
        for block in diagnostics["compile"]
        for line in block["lines"]
    ):
        diagnostics["compile"] = [
            block
            for block in diagnostics["compile"]
            if not CASCADE_MISSING_OBJECT_RE.search(" ".join(block["lines"]))
        ]
    for blocks in diagnostics.values():
        for block in blocks:
            block.pop("_priority", None)
    return diagnostics


def _empty_accuracy() -> dict[str, Any]:
    return {
        "status": "not_available",
        "total": 0,
        "passed": 0,
        "failed": 0,
        "not_run": 0,
        "failures": [],
    }


def _as_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} is not an integer")
    if value < 0:
        raise ValueError(f"{field} is negative")
    return value


def _metric_failed(metric: dict[str, Any]) -> bool:
    for key in ("finite", "allclose", "cosine_ok"):
        if key in metric and metric[key] is False:
            return True
        if isinstance(metric.get(key), str) and metric[key].lower() == "false":
            return True
    status = str(metric.get("status", "")).lower()
    return status in {"failed", "failure", "error"}


def _metric_details(metric: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for key in (
        "finite",
        "allclose",
        "cosine_ok",
        "tol",
        "cosine",
        "cos_min",
        "max_abs",
        "mean_abs",
        "rmse",
        "bad_frac",
    ):
        if key not in metric:
            continue
        value = metric[key]
        if isinstance(value, float) and not math.isfinite(value):
            details[key] = str(value).lower()
        elif isinstance(value, (bool, int, float)):
            details[key] = value
        else:
            details[key] = sanitize_inline(value, 100)
    return details


def load_accuracy_report(
    report_file: Optional[Path], require_report: bool
) -> tuple[dict[str, Any], Optional[str]]:
    accuracy = _empty_accuracy()
    if report_file is None or not report_file.is_file():
        if require_report:
            return accuracy, "Required accuracy report was not generated"
        return accuracy, None

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("report root is not an object")
        if report.get("schema") != "gdr-accuracy-report-v1":
            raise ValueError("report schema is not gdr-accuracy-report-v1")
        if report.get("complete") is not True:
            raise ValueError("report did not finish writing all selected cases")
        summary = report.get("summary")
        cases = report.get("cases")
        if not isinstance(summary, dict) or not isinstance(cases, list):
            raise ValueError("report summary or cases is invalid")
        summary_counts = {
            field: _as_nonnegative_int(summary.get(field), field)
            for field in (
                "total",
                "passed",
                "failed",
                "not_run",
                "accuracy_total",
                "accuracy_passed",
                "accuracy_failed",
                "accuracy_not_run",
            )
        }
        if (
            summary_counts["passed"]
            + summary_counts["failed"]
            + summary_counts["not_run"]
            != summary_counts["total"]
        ):
            raise ValueError("execution report counts are inconsistent")
        total = summary_counts["accuracy_total"]
        passed = summary_counts["accuracy_passed"]
        failed = summary_counts["accuracy_failed"]
        not_run = summary_counts["accuracy_not_run"]
        if passed + failed + not_run != total:
            raise ValueError("accuracy report counts are inconsistent")

        failures: list[dict[str, Any]] = []
        derived_execution = {"passed": 0, "failed": 0, "not_run": 0}
        derived_accuracy = {"passed": 0, "failed": 0, "not_run": 0}
        for case_index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise ValueError(f"case {case_index} is not an object")
            case_status = str(case.get("status", "unknown")).lower()
            accuracy_status = str(case.get("accuracy_status", "not_requested")).lower()
            if case_status not in derived_execution:
                raise ValueError(f"case {case_index} status is invalid")
            return_code = case.get("return_code")
            is_integer_return_code = isinstance(return_code, int) and not isinstance(
                return_code, bool
            )
            if case_status == "passed" and (
                not is_integer_return_code or return_code != 0
            ):
                raise ValueError(
                    f"case {case_index} passed with an invalid return_code"
                )
            if case_status == "failed" and (
                not is_integer_return_code or return_code == 0
            ):
                raise ValueError(
                    f"case {case_index} failed with an invalid return_code"
                )
            if case_status == "not_run" and return_code is not None:
                raise ValueError(
                    f"case {case_index} not_run has a non-null return_code"
                )
            if accuracy_status not in {"passed", "failed", "not_run", "not_requested"}:
                raise ValueError(f"case {case_index} accuracy_status is invalid")
            accuracy_check = case.get("accuracy_check")
            if not isinstance(accuracy_check, bool):
                raise ValueError(f"case {case_index} accuracy_check is not boolean")
            is_accuracy_case = accuracy_check or accuracy_status == "not_run"
            if accuracy_check and accuracy_status == "not_requested":
                raise ValueError(f"case {case_index} did not report an accuracy result")
            if not is_accuracy_case and accuracy_status != "not_requested":
                raise ValueError(f"case {case_index} has an unexpected accuracy result")

            derived_execution[case_status] += 1
            if is_accuracy_case:
                derived_accuracy[accuracy_status] += 1

            failed_metrics = []
            metrics = case.get("metrics", [])
            if not isinstance(metrics, list):
                raise ValueError(f"case {case_index} metrics is not an array")
            if accuracy_status == "passed" and not metrics:
                raise ValueError(f"case {case_index} passed without accuracy metrics")
            for metric_index, metric in enumerate(metrics):
                if not isinstance(metric, dict):
                    raise ValueError(
                        f"case {case_index} metric {metric_index} is not an object"
                    )
                if is_accuracy_case and any(
                    not isinstance(metric.get(field), bool)
                    for field in ("finite", "allclose", "cosine_ok")
                ):
                    raise ValueError(
                        f"case {case_index} metric {metric_index} has invalid checks"
                    )
                if not _metric_failed(metric):
                    continue
                output = metric.get("tensor") or metric.get("output") or metric.get("name") or "output"
                failed_metrics.append(
                    {
                        "output": sanitize_inline(output, 100),
                        "status": "failed",
                        "details": _metric_details(metric),
                    }
                )
                if len(failed_metrics) >= MAX_METRICS_PER_CASE:
                    break
            if (
                case_status not in {"failed", "not_run"}
                and accuracy_status not in {"failed", "not_run"}
                and not failed_metrics
            ):
                continue
            if len(failures) < MAX_FAILURE_CASES:
                failures.append(
                    {
                        "name": sanitize_inline(case.get("name", "unnamed-case"), 160),
                        "status": sanitize_inline(
                            accuracy_status
                            if accuracy_status in {"failed", "not_run"}
                            else case_status,
                            40,
                        ),
                        "return_code": return_code if is_integer_return_code else None,
                        "metrics": failed_metrics,
                    }
                )

        expected_counts = {
            "total": len(cases),
            **derived_execution,
            "accuracy_total": sum(derived_accuracy.values()),
            **{f"accuracy_{key}": value for key, value in derived_accuracy.items()},
        }
        if any(summary_counts[field] != value for field, value in expected_counts.items()):
            raise ValueError("report summary does not match case results")

        accuracy.update(
            {
                "status": "success"
                if total > 0 and passed == total and not failures
                else "failure",
                "total": total,
                "passed": passed,
                "failed": failed,
                "not_run": not_run,
                "failures": failures,
            }
        )
        return accuracy, None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return accuracy, f"Accuracy report is invalid: {sanitize_text(error)}"


def _add_synthetic_block(
    diagnostics: dict[str, list[dict[str, Any]]],
    category: str,
    message: str,
    *,
    priority: bool = False,
) -> None:
    existing = diagnostics[category]
    if any(block.get("headline") == message for block in existing):
        return
    if len(existing) >= MAX_BLOCKS_PER_CATEGORY:
        if not priority:
            return
        existing.pop()
    existing.append(
        {"headline": message, "lines": [message], "occurrences": 1}
    )


def _shell_assignment(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def build_reproduction(
    platform: str,
    soc: str,
    mode: str,
    ops: str,
    failed_cases: list[dict[str, Any]],
) -> list[str]:
    assignments = [
        _shell_assignment("CI_MODE", mode),
        _shell_assignment("CI_SOC", soc),
        _shell_assignment("FLA_NPU_SOC", soc),
        _shell_assignment("CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK", "true"),
        _shell_assignment("CI_RUN_SCOPED_WHEEL_INSTALL_CHECK", "true"),
    ]
    if platform.lower() == "a5" or soc == "ascend950":
        assignments.extend(
            [
                _shell_assignment("CI_IMAGE", "fla-npu-ci:9.1.0-950"),
                _shell_assignment("CI_DOCKERFILE", "ci/Dockerfile.ascend950"),
                _shell_assignment("CI_REQUIRE_PRELOADED_IMAGE", "true"),
            ]
        )
    else:
        assignments.extend(
            [
                _shell_assignment("CI_IMAGE", "fla-npu-ci:9.1.0-910b"),
                _shell_assignment("CI_DOCKERFILE", "ci/Dockerfile"),
            ]
        )
    suffix = ["bash", "ci/run_ci_container.sh"]

    def render(candidate_assignments: list[str]) -> str:
        return " ".join([*candidate_assignments, *suffix])

    if ops:
        candidate = [*assignments, _shell_assignment("CI_OPS", ops)]
        if len(render(candidate)) <= MAX_REPRO_COMMAND_CHARS:
            assignments = candidate

    selected_case_names: list[str] = []
    for case in failed_cases:
        case_name = case.get("name")
        if not case_name:
            continue
        candidate_names = [*selected_case_names, str(case_name)]
        candidate = [
            *assignments,
            _shell_assignment("CI_EXAMPLE_CASE_FILTER", ",".join(candidate_names)),
        ]
        if len(render(candidate)) <= MAX_REPRO_COMMAND_CHARS:
            selected_case_names = candidate_names
    if selected_case_names:
        assignments.append(
            _shell_assignment(
                "CI_EXAMPLE_CASE_FILTER", ",".join(selected_case_names)
            )
        )
    return [render(assignments)]


def _payload_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _fit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fitted = copy.deepcopy(payload)
    while _payload_size(fitted) > MAX_JSON_BYTES:
        candidates = [
            blocks
            for blocks in fitted["diagnostics"].values()
            if len(blocks) > 1
        ]
        if candidates:
            max(candidates, key=len).pop()
            fitted["truncated"] = True
            continue
        failures = fitted["accuracy"]["failures"]
        if len(failures) > 1:
            failures.pop()
            fitted["truncated"] = True
            continue
        metric_lists = [case["metrics"] for case in failures if len(case["metrics"]) > 1]
        if metric_lists:
            max(metric_lists, key=len).pop()
            fitted["truncated"] = True
            continue
        for blocks in fitted["diagnostics"].values():
            for block in blocks:
                if len(block["lines"]) > 2:
                    block["lines"].pop()
                    fitted["truncated"] = True
                    break
            else:
                continue
            break
        else:
            break
    return fitted


def build_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    execution = payload["execution"]
    accuracy = payload["accuracy"]
    platform = sanitize_markdown_inline(metadata["platform"], 40).upper()
    soc = sanitize_markdown_inline(metadata["soc"], 80)
    lines = [f"### {platform} / `{soc}`"]
    if execution["status"] == "success":
        lines.append("- 执行：通过")
    else:
        lines.append(f"- 执行：失败（退出码 {execution['exit_code']}）")
    if accuracy["status"] == "not_available":
        lines.append(
            "- 精度：未执行（执行阶段失败）"
            if execution["status"] == "failure"
            else "- 精度：未生成报告"
        )
    elif accuracy["status"] == "success":
        lines.append(f"- 精度：{accuracy['passed']}/{accuracy['total']}")
    else:
        lines.append(
            f"- 精度：{accuracy['passed']}/{accuracy['total']}"
            f"（失败 {accuracy['failed']}，未执行 {accuracy['not_run']}）"
        )

    if payload["status"] == "success":
        return "\n".join(lines) + "\n"

    truncated = bool(payload.get("truncated"))

    def append_section(section: list[str]) -> bool:
        nonlocal truncated
        candidate = "\n".join([*lines, *section]) + "\n"
        if len(candidate) > MAX_MARKDOWN_CHARS:
            truncated = True
            return False
        lines.extend(section)
        return True

    populated = [
        category
        for category in ("compile", "runtime", "infrastructure")
        if payload["diagnostics"][category]
    ]
    labels = {
        "compile": "编译错误",
        "runtime": "执行错误",
        "infrastructure": "基础设施错误",
    }
    if populated:
        lines.append(f"- 失败分类：{'、'.join(labels[item] for item in populated)}")

    if accuracy["failures"]:
        section = ["", "#### 失败用例"]
        shown_failures = accuracy["failures"][:MAX_MARKDOWN_FAILURE_CASES]
        truncated = truncated or len(accuracy["failures"]) > len(shown_failures)
        for case in shown_failures:
            case_name = sanitize_markdown_inline(case["name"], 160)
            return_code = case.get("return_code")
            return_code_text = (
                f"，退出码 {return_code}" if isinstance(return_code, int) else ""
            )
            section.append(f"- `{case_name}`：{case['status']}{return_code_text}")
            shown_metrics = case["metrics"][:MAX_MARKDOWN_METRICS_PER_CASE]
            truncated = truncated or len(case["metrics"]) > len(shown_metrics)
            for metric in shown_metrics:
                output = sanitize_markdown_inline(metric["output"], 100)
                details = " ".join(
                    f"{sanitize_markdown_inline(key, 60)}="
                    f"{sanitize_markdown_inline(value, 100)}"
                    for key, value in metric["details"].items()
                )
                section.append(
                    _truncate(f"  - `{output}`：{details or 'metric failed'}", 700)
                )
        append_section(section)

    for category in ("compile", "runtime", "infrastructure"):
        blocks = payload["diagnostics"][category]
        if not blocks:
            continue
        shown_blocks = blocks[:MAX_MARKDOWN_BLOCKS_PER_CATEGORY]
        truncated = truncated or len(blocks) > len(shown_blocks)
        for index, block in enumerate(shown_blocks, start=1):
            suffix = (
                f"（重复 {block['occurrences']} 次）"
                if block["occurrences"] > 1
                else ""
            )
            shown_lines = block["lines"][:MAX_MARKDOWN_LINES_PER_BLOCK]
            truncated = truncated or len(block["lines"]) > len(shown_lines)
            section = [
                "",
                f"#### {labels[category]} {index}{suffix}",
                "",
                "```text",
                *(str(item).replace("```", "` ` `") for item in shown_lines),
                "```",
            ]
            append_section(section)

    if payload["reproduction"]:
        append_section(
            [
                "",
                "#### 复现",
                "",
                "```bash",
                *(str(item).replace("```", "` ` `") for item in payload["reproduction"]),
                "```",
            ]
        )

    if truncated:
        append_section(["", "_其余诊断已省略，完整结构化结果见平台 artifact。_"])
    return "\n".join(lines) + "\n"


def summarize(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    accuracy_file = Path(args.accuracy_report_file) if args.accuracy_report_file else None
    accuracy, report_error = load_accuracy_report(
        accuracy_file, args.require_accuracy_report
    )
    execution_failed = args.exit_code != 0
    accuracy_failed = accuracy["status"] == "failure"
    overall_failed = execution_failed or accuracy_failed or report_error is not None
    diagnostics = extract_diagnostics(Path(args.log_file), enabled=overall_failed)

    if args.exit_code == 137:
        _add_synthetic_block(
            diagnostics,
            "infrastructure",
            "CI process exited with code 137 (SIGKILL or resource exhaustion)",
            priority=True,
        )
    if report_error and not execution_failed:
        _add_synthetic_block(
            diagnostics, "infrastructure", report_error, priority=True
        )
    if accuracy_failed:
        _add_synthetic_block(
            diagnostics,
            "runtime",
            f"Accuracy checks failed ({accuracy['passed']}/{accuracy['total']} passed)",
        )
    if execution_failed and not any(diagnostics.values()):
        _add_synthetic_block(
            diagnostics,
            "runtime",
            f"CI command exited with code {args.exit_code}",
        )

    metadata = {
        "platform": sanitize_inline(args.platform, 40),
        "soc": sanitize_inline(args.soc, 80),
        "mode": sanitize_inline(args.mode, 40),
        "ops": sanitize_inline(args.ops, 500),
        "head_sha": sanitize_inline(args.head_sha, 80),
        "run_id": sanitize_inline(args.run_id, 80),
        "run_attempt": sanitize_inline(args.run_attempt, 40),
    }
    payload = {
        "schema": SCHEMA,
        "metadata": metadata,
        "status": "failure" if overall_failed else "success",
        "execution": {
            "status": "failure" if execution_failed else "success",
            "exit_code": args.exit_code,
        },
        "accuracy": accuracy,
        "diagnostics": diagnostics,
        "reproduction": build_reproduction(
            metadata["platform"],
            metadata["soc"],
            metadata["mode"],
            metadata["ops"],
            accuracy["failures"],
        )
        if overall_failed
        else [],
    }
    payload = _fit_payload(payload)
    return payload, build_markdown(payload)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--ops", default="")
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--accuracy-report-file", default="")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument("--require-accuracy-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    payload, markdown = summarize(args)
    _write_text(
        Path(args.json_out),
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    _write_text(Path(args.markdown_out), markdown)
    return 1 if payload["status"] == "failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
