"""Read and materialize the Chapter 21 ChunkKdaFwd design matrix."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = ROOT / "tests" / "op_cases" / "chunk_kda_fwd.json"
DESIGN_PATH = (
    ROOT
    / "fla"
    / "ops"
    / "ascendc"
    / "kda"
    / "chunk_kda_fwd"
    / "docs"
    / "design.md"
)

EXPECTED_PREFIX_COUNTS = {"P": 96, "N": 84, "M": 12, "S": 8, "G": 100}
EXPECTED_KIND_COUNTS = {
    "accuracy": 176,
    "run": 92,
    "msopprof": 20,
    "stress": 6,
    "sanitizer": 6,
}
SUPPORTED_SOCS = ("ascend910b", "ascend910_93", "ascend950")
EXECUTABLE_ENTRYPOINTS = {
    "accuracy": {
        "adapter_status": "implemented",
        "adapter": "test/chunk_kda_fwd/canonical_case_adapter.py",
        "runner": "test/chunk_kda_fwd/executor_chunk_kda_fwd.py",
    },
    "run": {
        "adapter_status": "implemented",
        "adapter": "test/chunk_kda_fwd/canonical_execution_adapter.py",
        "runner": "test/chunk_kda_fwd/executor_chunk_kda_fwd.py",
    },
    "msopprof": {
        "adapter_status": "implemented",
        "adapter": "test/chunk_kda_fwd/canonical_execution_adapter.py",
        "runner": "test/chunk_kda_fwd/canonical_execution_runner.py",
    },
    "stress": {
        "adapter_status": "implemented",
        "adapter": "test/chunk_kda_fwd/canonical_execution_adapter.py",
        "runner": "test/chunk_kda_fwd/stress_npu_determinism.py",
    },
    "sanitizer": {
        "adapter_status": "implemented",
        "adapter": "test/chunk_kda_fwd/canonical_execution_adapter.py",
        "runner": "test/chunk_kda_fwd/canonical_execution_runner.py",
    },
}
DESIGN_ID_RE = re.compile(r"^KDA-FWD-([PNMSG])(\d{3})$")
DESIGN_ROW_RE = re.compile(r"^\| (KDA-FWD-[PNMSG]\d{3}) \|")


def load_manifest(path=MANIFEST_PATH):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_design_matrix(path=MANIFEST_PATH):
    manifest = load_manifest(path)
    matrix = manifest.get("design_matrix")
    if not isinstance(matrix, dict):
        raise ValueError("chunk_kda_fwd manifest has no design_matrix object")
    return matrix


def design_row_digest(path=DESIGN_PATH):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.rstrip("\n")
            if DESIGN_ROW_RE.match(stripped):
                rows.append(stripped)
    payload = "\n".join(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(rows)


def _markdown_cells(line):
    return [item.strip() for item in line.strip().strip("|").split("|")]


def read_design_rows(path=DESIGN_PATH):
    rows = []
    section = ""
    headers = None
    with Path(path).open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if line.startswith("### ") or line.startswith("#### "):
                section = line.lstrip("# ")
            if line.startswith("| ID |"):
                headers = _markdown_cells(line)
                continue
            match = DESIGN_ROW_RE.match(line)
            if not match:
                continue
            if headers is None:
                raise ValueError("Chapter 21 row has no table header: {}".format(match.group(1)))
            values = _markdown_cells(line)
            if len(values) != len(headers):
                raise ValueError("Chapter 21 table width mismatch: {}".format(match.group(1)))
            rows.append(
                {
                    "id": match.group(1),
                    "section": section,
                    "source_fields": dict(zip(headers[1:], values[1:])),
                }
            )
    return rows


def _expected_ids():
    result = []
    for prefix in ("P", "N", "M", "S", "G"):
        result.extend(
            "KDA-FWD-{}{:03d}".format(prefix, index)
            for index in range(1, EXPECTED_PREFIX_COUNTS[prefix] + 1)
        )
    return result


def _expected_platforms(case):
    match = DESIGN_ID_RE.match(case["id"])
    prefix, index = match.group(1), int(match.group(2))
    fields = case["source_fields"]
    text = " ".join(
        fields.get(key, "")
        for key in ("平台/路由", "平台", "平台/布局", "场景")
    )
    if "ALL" in text or "三平台" in text:
        return list(SUPPORTED_SOCS)
    platforms = []
    for token, soc in (
        ("A2", "ascend910b"),
        ("A3", "ascend910_93"),
        ("A5", "ascend950"),
    ):
        if re.search(r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])".format(token), text):
            platforms.append(soc)
    if platforms:
        return platforms
    if prefix == "P" and (index <= 16 or index >= 73):
        return ["ascend950"]
    return list(SUPPORTED_SOCS)


def _expected_routes(case):
    fields = case["source_fields"]
    explicit = fields.get("平台/路由") or fields.get("路由") or fields.get("平台") or ""
    route_text = explicit.split("/", 1)[1] if "/" in explicit else explicit
    tokens = [token for token in route_text.split("+") if token in {"P", "A", "D"}]
    if tokens:
        names = {"P": "ascendc", "A": "aclnn", "D": "direct_launch"}
        return [names[token] for token in tokens]
    prefix = DESIGN_ID_RE.match(case["id"]).group(1)
    if prefix == "N" or (prefix == "G" and case["kind"] == "run"):
        return ["aclnn"]
    if case["kind"] in {"msopprof", "stress", "sanitizer"}:
        return ["ascendc"]
    return ["ascendc", "aclnn"]


def validate_design_matrix(matrix=None, design_path=DESIGN_PATH):
    matrix = load_design_matrix() if matrix is None else matrix
    cases = matrix.get("cases")
    if not isinstance(cases, list):
        raise ValueError("design_matrix.cases must be a list")

    ids = [case.get("id") for case in cases]
    if ids != _expected_ids():
        raise ValueError("design matrix IDs are not the exact ordered 300-case sequence")
    if len(ids) != len(set(ids)):
        raise ValueError("design matrix IDs must be unique")

    prefix_counts = Counter(
        DESIGN_ID_RE.match(case_id).group(1) for case_id in ids
    )
    if dict(prefix_counts) != EXPECTED_PREFIX_COUNTS:
        raise ValueError("unexpected design prefix counts: {}".format(dict(prefix_counts)))

    kind_counts = Counter(case.get("kind") for case in cases)
    if dict(kind_counts) != EXPECTED_KIND_COUNTS:
        raise ValueError("unexpected execution kind counts: {}".format(dict(kind_counts)))

    entrypoints = matrix.get("entrypoints", {})
    if set(entrypoints) != set(EXPECTED_KIND_COUNTS):
        raise ValueError("every execution kind must have exactly one entrypoint")
    if matrix.get("logical_case_counts") != EXPECTED_PREFIX_COUNTS:
        raise ValueError("declared logical case counts do not match Chapter 21")
    if matrix.get("execution_kind_counts") != EXPECTED_KIND_COUNTS:
        raise ValueError("declared execution kind counts do not match Chapter 21")

    for case in cases:
        case_id = case["id"]
        if not case.get("platforms"):
            raise ValueError("{} has no target platform".format(case_id))
        unknown_soc = set(case["platforms"]).difference(SUPPORTED_SOCS)
        if unknown_soc:
            raise ValueError("{} has unknown platforms {}".format(case_id, sorted(unknown_soc)))
        if not case.get("routes"):
            raise ValueError("{} has no execution route".format(case_id))
        if not case.get("variants"):
            raise ValueError("{} has no execution variant".format(case_id))
        if case["platforms"] != _expected_platforms(case):
            raise ValueError("{} has an incorrect platform projection".format(case_id))
        if case["routes"] != _expected_routes(case):
            raise ValueError("{} has an incorrect route projection".format(case_id))
        if not isinstance(case.get("source_fields"), dict) or not case["source_fields"]:
            raise ValueError("{} lost its Chapter 21 source row".format(case_id))
        if case["kind"] == "sanitizer" and not case.get("sanitizer_tools"):
            raise ValueError("{} has no sanitizer tool".format(case_id))
        if case["kind"] == "sanitizer":
            source_text = " ".join(case["source_fields"].values())
            missing_tools = [
                tool for tool in case["sanitizer_tools"] if tool not in source_text
            ]
            if missing_tools:
                raise ValueError(
                    "{} sanitizer tools do not match design.md: {}".format(
                        case_id, ",".join(missing_tools)
                    )
                )
        if case["kind"] == "stress" and int(case.get("repeat_count", 0)) < 2:
            raise ValueError("{} has no repeat count".format(case_id))

    source = matrix.get("source", {})
    actual_digest, row_count = design_row_digest(design_path)
    design_rows = read_design_rows(design_path)
    if row_count != 300:
        raise ValueError("Chapter 21 must contain exactly 300 table rows")
    for case, source_row in zip(cases, design_rows):
        for key in ("id", "section", "source_fields"):
            if case.get(key) != source_row[key]:
                raise ValueError(
                    "{} does not match design.md field {}".format(case["id"], key)
                )
    if source.get("row_sha256") != actual_digest:
        raise ValueError("design_matrix is stale relative to design.md Chapter 21 rows")
    if int(source.get("row_count", -1)) != row_count:
        raise ValueError("design_matrix source row count is stale")
    return {
        "logical_cases": len(cases),
        "prefix_counts": dict(prefix_counts),
        "kind_counts": dict(kind_counts),
        "source_row_sha256": actual_digest,
    }


def _task_id(case_id, soc, route, variant, tool):
    parts = [case_id, soc, route]
    if variant:
        parts.append(variant)
    if tool:
        parts.append(tool)
    return "@".join(parts)


def materialize_tasks(
    matrix=None,
    case_ids=(),
    kinds=(),
    soc=None,
    include_not_applicable=True,
):
    """Expand logical cases into read-only execution-plan records.

    The returned records are plans, not test results. A non-target SOC produces
    an explicit ``not_applicable`` record so it cannot be aggregated as PASS.
    """

    matrix = load_design_matrix() if matrix is None else matrix
    selected_ids = set(case_ids)
    selected_kinds = set(kinds)
    if soc is not None and soc not in SUPPORTED_SOCS:
        raise ValueError("unsupported SOC: {}".format(soc))
    known_ids = {case["id"] for case in matrix["cases"]}
    unknown_ids = selected_ids.difference(known_ids)
    if unknown_ids:
        raise ValueError(
            "unknown design case IDs: {}".format(", ".join(sorted(unknown_ids)))
        )
    unknown_kinds = selected_kinds.difference(EXPECTED_KIND_COUNTS)
    if unknown_kinds:
        raise ValueError(
            "unknown execution kinds: {}".format(", ".join(sorted(unknown_kinds)))
        )

    tasks = []
    for case in matrix["cases"]:
        if selected_ids and case["id"] not in selected_ids:
            continue
        if selected_kinds and case["kind"] not in selected_kinds:
            continue

        if soc is not None and soc not in case["platforms"]:
            if include_not_applicable:
                tasks.append(
                    {
                        "task_id": "{}@{}@not_applicable".format(case["id"], soc),
                        "logical_id": case["id"],
                        "kind": case["kind"],
                        "soc": soc,
                        "status": "not_applicable",
                        "reason": "case targets {}".format(",".join(case["platforms"])),
                    }
                )
            continue

        target_socs = [soc] if soc is not None else case["platforms"]
        variants = case.get("variants") or [None]
        tools = case.get("sanitizer_tools") or [None]
        for target_soc in target_socs:
            for route in case["routes"]:
                for variant in variants:
                    for tool in tools:
                        tasks.append(
                            {
                                "task_id": _task_id(case["id"], target_soc, route, variant, tool),
                                "logical_id": case["id"],
                                "kind": case["kind"],
                                "soc": target_soc,
                                "route": route,
                                "variant": variant,
                                "sanitizer_tool": tool,
                                "status": "planned",
                                "entrypoint": {
                                    **matrix["entrypoints"][case["kind"]],
                                    **EXECUTABLE_ENTRYPOINTS[case["kind"]],
                                },
                                "section": case["section"],
                                "source_fields": case["source_fields"],
                            }
                        )
    return tasks


def summarize_tasks(tasks):
    status_counts = Counter(task["status"] for task in tasks)
    planned = [task for task in tasks if task["status"] == "planned"]
    not_applicable = [task for task in tasks if task["status"] == "not_applicable"]
    return {
        "physical_tasks": len(tasks),
        "logical_ids": len(set(task["logical_id"] for task in tasks)),
        "applicable_logical_ids": len(set(task["logical_id"] for task in planned)),
        "not_applicable_logical_ids": len(
            set(task["logical_id"] for task in not_applicable)
        ),
        "status_counts": dict(status_counts),
        "kind_counts": dict(Counter(task["kind"] for task in planned)),
        "soc_counts": dict(Counter(task["soc"] for task in planned)),
    }
