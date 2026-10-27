#!/usr/bin/env python3
"""Run deterministic checks for the four operator-review layers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


REQUIRED_OPERATOR_FILES = [
    "CMakeLists.txt",
    "README.md",
    "docs/api.md",
    "docs/design.md",
    "op_host/CMakeLists.txt",
]

FORBIDDEN_ARTIFACT_PARTS = {
    "build",
    "build_out",
    "dist",
    "output",
    "outputs",
    "reports",
    "__pycache__",
    ".ci-cache",
    "PROF",
    "OPPROF",
}

HOST_SOURCE_SUFFIXES = {".cc", ".cpp", ".h", ".hpp"}


@dataclass(frozen=True)
class Finding:
    layer: str
    rule: str
    severity: str
    file: str
    line: int
    message: str
    suggestion: str


def run(command: list[str], root: Path) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def changed_paths(root: Path, base: str) -> list[Path]:
    output = run(["git", "diff", "--name-only", f"{base}...HEAD"], root)
    paths = [root / line for line in output.splitlines() if line.strip()]
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"], root)
    paths.extend(root / line for line in untracked.splitlines() if line.strip())
    return sorted({path for path in paths if path.exists()})


def check_environment_contracts(root: Path, base: str) -> list[Finding]:
    diff = run(["git", "diff", "--unified=0", f"{base}...HEAD"], root)
    current_file = ""
    added_by_file: dict[str, list[str]] = {}
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            added_by_file.setdefault(current_file, [])
        elif line.startswith("+") and not line.startswith("+++"):
            if current_file:
                added_by_file[current_file].append(line[1:])

    candidate_patterns = [
        re.compile(r"(?:os\.environ(?:\.get)?|os\.getenv|getenv)\s*\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']"),
        re.compile(r"(?:^|\s)export\s+([A-Z][A-Z0-9_]*)="),
        re.compile(r"\$ENV\{([A-Z][A-Z0-9_]*)\}"),
        re.compile(r"^\s*option\(\s*([A-Z][A-Z0-9_]*)"),
        re.compile(r"^\s*set\(\s*([A-Z][A-Z0-9_]*)\s+.*\bCACHE\b"),
    ]
    compile_definition = re.compile(r"^\s*(?:-D)?([A-Z][A-Z0-9_]*)=[^\s]+")
    candidates: dict[str, str] = {}
    for path, lines in added_by_file.items():
        is_cmake = path.endswith("CMakeLists.txt") or path.endswith(".cmake")
        for line in lines:
            for pattern in candidate_patterns:
                for match in pattern.finditer(line):
                    candidates.setdefault(match.group(1), path)
            if is_cmake:
                match = compile_definition.match(line)
                if match:
                    candidates.setdefault(match.group(1), path)

    if not candidates:
        return []

    doc_additions = "\n".join(
        line
        for path, lines in added_by_file.items()
        if path.endswith(".md")
        for line in lines
    )
    findings = []
    for name, path in sorted(candidates.items()):
        documented = name in doc_additions and any(
            word in doc_additions for word in ("环境变量", "默认值", "非法值", "回退")
        )
        if not documented:
            findings.append(
                Finding(
                    "engineering",
                    "STRUCT-ENV-CONTRACT",
                    "warning",
                    path,
                    1,
                    f"new environment/build switch {name} has no warning contract in changed docs",
                    "document name, default, scope, consumers, precedence, invalid-value behavior, "
                    "lifecycle, test matrix, and rollback in the PR",
                )
            )
    return findings


def all_operator_paths(root: Path) -> list[Path]:
    base = root / "fla" / "ops" / "ascendc"
    result = []
    for path in base.rglob("README.md"):
        operator_root = path.parent
        if (operator_root / "op_host").is_dir() or (operator_root / "op_kernel").is_dir():
            result.append(operator_root)
    return sorted(result)


def operator_paths_by_name(root: Path, names: set[str]) -> list[Path]:
    return [
        path
        for path in all_operator_paths(root)
        if any(
            part in names or any(part.endswith(f"_{name}") for name in names)
            for part in path.parts
        )
    ]


def find_operator_root(path: Path, root: Path) -> Path | None:
    try:
        current = path if path.is_dir() else path.parent
    except OSError:
        return None
    boundary = root / "fla" / "ops" / "ascendc"
    while current != root and boundary in current.parents:
        if current == boundary / "common":
            return None
        if (current / "op_host").is_dir() or (current / "op_kernel").is_dir():
            return current
        current = current.parent
    return None


def operator_roots_for_selection(
    root: Path, selected_paths: list[Path], op_names: set[str]
) -> list[Path]:
    roots = {find_operator_root(path, root) for path in selected_paths}
    roots.discard(None)
    if op_names:
        roots.update(operator_paths_by_name(root, op_names))
    return sorted(roots)


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def check_structure(root: Path, operator_roots: list[Path]) -> list[Finding]:
    findings = []
    for operator_root in operator_roots:
        for relative_required in REQUIRED_OPERATOR_FILES:
            required_path = operator_root / relative_required
            if not required_path.is_file():
                findings.append(
                    Finding(
                        "engineering",
                        "STRUCT-REQUIRED-FILE",
                        "error",
                        relative(operator_root, root),
                        1,
                        f"missing required operator file: {relative_required}",
                        "restore the standard operator layout before review",
                    )
                )
        for path in files_under(operator_root):
            text = relative(path, root)
            if any(part in FORBIDDEN_ARTIFACT_PARTS for part in path.parts):
                findings.append(
                    Finding(
                        "engineering",
                        "STRUCT-ARTIFACT",
                        "error",
                        text,
                        1,
                        "build, profiling, report, cache, or generated artifact is inside the operator",
                        "remove the artifact from the PR and keep only source-controlled deliverables",
                    )
                )
            if "/tests/" in f"/{text}" or "/test/" in f"/{text}":
                findings.append(
                    Finding(
                        "engineering",
                        "STRUCT-TEST-LOCATION",
                        "warning",
                        text,
                        1,
                        "operator-local test asset is outside the repository test layout",
                        "move durable cases to tests/atk/<op>/ or tests/operators/<op>/",
                    )
                )
    return findings


def is_host_surface(path: Path) -> bool:
    text = path.as_posix()
    if "/op_host/" not in text:
        return False
    return (
        "/op_api/" in text
        or "/op_tiling/" in text
        or "tiling" in path.name
        or path.name.startswith("aclnn_")
    )


def check_dfx(root: Path, operator_roots: list[Path]) -> list[Finding]:
    findings = []
    patterns = [
        (
            "DFX-MESSAGE",
            re.compile(r"\b(CHECK_RET|CHECK_COND)\s*\("),
            "legacy guard may return only a status code without parameter context",
            "for an implementation PR, report through typed OP_LOGE_FOR_*; do not change code in a review-skill PR",
        ),
        (
            "DFX-MESSAGE",
            re.compile(r"\bOP_LOGE\s*\("),
            "generic OP_LOGE reports only generic/internal context",
            "for public parameter constraints, use the parameter-level OP_LOGE_FOR_* report",
        ),
    ]
    for operator_root in operator_roots:
        for path in files_under(operator_root):
            if path.suffix not in HOST_SOURCE_SUFFIXES or not is_host_surface(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            test_only_shim = "TORCH_MODE" in text or "/tests/" in f"/{path.as_posix()}/"
            if test_only_shim:
                continue
            for rule, pattern, message, suggestion in patterns:
                for match in pattern.finditer(text):
                    findings.append(
                        Finding(
                            "dfx",
                            rule,
                            "error",
                            relative(path, root),
                            line_number(text, match.start()),
                            message,
                            suggestion,
                        )
                    )
    return findings


def markdown_links(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    links = []
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = match.group(1).strip().split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        links.append((line_number(text, match.start()), target))
    return links


def check_docs(root: Path, markdown_paths: list[Path], operator_roots: list[Path]) -> list[Finding]:
    findings = []
    markdown_set = {path.resolve() for path in markdown_paths if path.suffix == ".md"}
    for operator_root in operator_roots:
        markdown_set.update((operator_root / "README.md", operator_root / "docs" / "api.md",
                             operator_root / "docs" / "design.md"))
    for path in sorted(markdown_set):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line, target in markdown_links(path):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                findings.append(
                    Finding(
                        "docs",
                        "DOC-BROKEN-LINK",
                        "error",
                        relative(path, root),
                        line,
                        f"relative link target does not exist: {target}",
                        "fix the link or remove it",
                    )
                )
        for token in ("/home/", "/mnt/", "C:\\Users\\", "sk-", "ghp_", "BEGIN PRIVATE KEY"):
            position = text.find(token)
            if position >= 0:
                findings.append(
                    Finding(
                        "docs",
                        "DOC-SENSITIVE",
                        "error",
                        relative(path, root),
                        line_number(text, position),
                        f"document contains a local path or credential-like token: {token}",
                        "replace it with a public, relative description",
                    )
                )

    required_docs = {
        "README.md": ("支持范围", "输入", "输出"),
        "docs/api.md": ("接口", "参数", "返回"),
        "docs/design.md": ("设计",),
    }
    for operator_root in operator_roots:
        for child, headings in required_docs.items():
            path = operator_root / child
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for heading in headings:
                if heading not in text:
                    findings.append(
                        Finding(
                            "docs",
                            "DOC-REQUIRED-SECTION",
                            "warning",
                            relative(path, root),
                            1,
                            f"required operator-document concept is missing: {heading}",
                            f"add a clear `{heading}` section following the repository template",
                        )
                    )
    return findings


def load_issue_catalog(root: Path) -> dict:
    path = root / "docs/agents/skills/operator-review/references/issue-corpus.json"
    return json.loads(path.read_text(encoding="utf-8"))


def check_functional(root: Path, op_names: set[str]) -> tuple[list[Finding], dict]:
    findings = []
    catalog = load_issue_catalog(root)
    issues = catalog.get("issues", [])
    if catalog.get("completed_issue_count") != len(issues):
        findings.append(
            Finding(
                "functional",
                "FUNC-ISSUE-CORPUS",
                "error",
                "docs/agents/skills/operator-review/references/issue-corpus.json",
                1,
                "completed issue count does not match the catalog records",
                "rerun sync_issue_corpus.py and review classification changes",
            )
        )
    for issue in issues:
        if not issue.get("rules"):
            findings.append(
                Finding(
                    "functional",
                    "FUNC-ISSUE-COVERAGE",
                    "error",
                    "docs/agents/skills/operator-review/references/issue-corpus.json",
                    1,
                    f"completed issue #{issue.get('number')} has no review rule",
                    "assign at least one rule in sync_issue_corpus.py",
                )
            )
    applicable = []
    for issue in issues:
        labels = set(issue.get("labels", []))
        if any(f"op:{name}" in labels for name in op_names):
            applicable.append(issue)
    return findings, {
        "completed_issue_count": len(issues),
        "applicable_operator_issues": [issue["number"] for issue in applicable],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--op", action="append", default=[])
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--diff-base")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    root = args.repo.resolve()
    op_names = {name[3:] if name.startswith("op:") else name for name in args.op}
    if args.path:
        selected_paths = [(root / item).resolve() for item in args.path]
    elif args.diff_base:
        selected_paths = changed_paths(root, args.diff_base)
    else:
        selected_paths = []

    if args.all:
        operator_roots = all_operator_paths(root)
    else:
        operator_roots = operator_roots_for_selection(root, selected_paths, op_names)
    markdown_paths = [path for path in selected_paths if path.suffix == ".md"]

    findings = []
    findings.extend(check_structure(root, operator_roots))
    if args.diff_base:
        findings.extend(check_environment_contracts(root, args.diff_base))
    findings.extend(check_dfx(root, operator_roots))
    findings.extend(check_docs(root, markdown_paths, operator_roots))
    functional_findings, functional_summary = check_functional(root, op_names)
    findings.extend(functional_findings)

    severity = {"error": 0, "warning": 1}
    findings.sort(key=lambda item: (severity[item.severity], item.layer, item.file, item.line))
    payload = {
        "schema_version": 1,
        "selected_operators": [relative(path, root) for path in operator_roots],
        "functional": functional_summary,
        "summary": {
            "errors": sum(item.severity == "error" for item in findings),
            "warnings": sum(item.severity == "warning" for item in findings),
        },
        "findings": [asdict(item) for item in findings],
    }

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in findings:
            print(
                f"[{item.layer}] {item.severity} {item.rule} "
                f"{item.file}:{item.line}: {item.message}"
            )
            print(f"  fix: {item.suggestion}")
        print(
            f"review summary: {payload['summary']['errors']} errors, "
            f"{payload['summary']['warnings']} warnings, "
            f"{functional_summary['completed_issue_count']} completed issues covered"
        )

    has_warning_failure = args.fail_on_warning and payload["summary"]["warnings"] > 0
    return 1 if payload["summary"]["errors"] > 0 or has_warning_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
