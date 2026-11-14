#!/usr/bin/env python3
"""Classify the files changed by a PR for the device CI pipeline.

The device job merges the PR before invoking this script.  A merge commit is
therefore the most reliable source of the PR file list: ``HEAD^1`` is the
base that was tested and ``HEAD`` contains the PR changes.  The classifier is
kept in ``ci/`` so the workflow can evolve without changing the workflow
definition itself.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


ASCENDC_PREFIX = "fla/ops/ascendc/"
TEST_PREFIXES = ("tests/", "examples/", "ci/", ".github/")


def _normalise(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def paths_from_name_status(lines: Iterable[str]) -> list[str]:
    """Extract all paths from ``git diff --name-status`` output.

    Renames and copies contain both the old and new path.  Considering both
    sides keeps a rename out of the fast paths if the old path was outside
    the new path's classification.
    """

    paths: list[str] = []
    for line in lines:
        fields = line.rstrip("\r\n").split("\t")
        if not fields:
            continue
        status = fields[0]
        if status.startswith(("R", "C")) and len(fields) >= 3:
            candidates = fields[1:3]
        elif len(fields) >= 2:
            candidates = [fields[1]]
        else:
            candidates = [fields[0]]
        for path in candidates:
            path = _normalise(path)
            if path:
                paths.append(path)
    return paths


def _operator_roots(repo_root: Path) -> list[tuple[str, str]]:
    """Return ``(repository-relative-root, --ops-name)`` pairs."""

    root = repo_root / "fla" / "ops" / "ascendc"
    roots: set[tuple[str, str]] = set()
    if not root.is_dir():
        return []
    for cmake in root.rglob("CMakeLists.txt"):
        parts = cmake.parts
        if "tests" in parts:
            continue
        if "op_host" in parts:
            index = len(parts) - 1 - parts[::-1].index("op_host")
            op_root = Path(*parts[:index])
        else:
            op_root = cmake.parent
        try:
            relative = op_root.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        roots.add((relative, op_root.name))
    return sorted(roots, key=lambda item: len(item[0]), reverse=True)


def operator_for_path(path: str, repo_root: Path) -> str | None:
    """Return the deepest supported operator ancestor for an AscendC path."""

    path = _normalise(path)
    if not path.startswith(ASCENDC_PREFIX):
        return None
    # Match an actual operator root, instead of merely matching a directory
    # name.  A kernel can contain an ``internal/operators/chunk_fwd_o``
    # directory whose name is itself a supported operator, but that directory
    # is not the build root for a file below the parent operator.
    for operator_root, operator_name in _operator_roots(repo_root):
        if path == operator_root or path.startswith(operator_root + "/"):
            return operator_name
    return None


def _is_documentation(path: str) -> bool:
    lower = path.lower()
    return (
        lower.endswith(".md")
        or lower == "licenses"
        or lower.startswith("licenses/")
        or lower == "docs"
        or lower.startswith("docs/")
    )


def _is_test_only(path: str) -> bool:
    return any(path == prefix[:-1] or path.startswith(prefix) for prefix in TEST_PREFIXES)


def classify_paths(paths: Sequence[str], repo_root: Path | None = None) -> tuple[str, list[str]]:
    """Return ``(scope, operators)`` for a changed path list.

    Scope values are ``docs-only``, ``tests-only``, ``partial`` and ``full``.
    An empty or unrecognised change list fails closed to ``full``.
    """

    normalised = sorted({_normalise(path) for path in paths if _normalise(path)})
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    if not normalised:
        return "full", []

    has_documentation = any(_is_documentation(path) for path in normalised)
    has_tests = any(
        _is_test_only(path) and not _is_documentation(path) for path in normalised
    )
    operators: set[str] = set()
    has_other = False
    for path in normalised:
        if _is_documentation(path) or _is_test_only(path):
            continue
        operator = operator_for_path(path, root)
        if operator is None:
            has_other = True
        else:
            operators.add(operator)

    # Documentation is neutral when mixed with tests or operator sources.
    # Any unrelated source/configuration change still requires a full build.
    if has_other:
        return "full", []
    if operators:
        return "partial", sorted(operators)
    if has_tests:
        return "tests-only", []
    if has_documentation:
        return "docs-only", []
    return "full", []


def _git_changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", "diff", "--name-status", "-M", base, head],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return paths_from_name_status(result.stdout.splitlines())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-range", nargs=2, metavar=("BASE", "HEAD"))
    parser.add_argument("--paths-from-stdin", action="store_true")
    args = parser.parse_args(argv)

    if args.git_range:
        try:
            paths = _git_changed_paths(*args.git_range)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"[CI][ERROR] Unable to inspect changed files: {exc}", file=sys.stderr)
            return 2
    else:
        # This mode is useful for unit tests and local reproductions.  It
        # accepts either one path per line or name-status records.
        paths = paths_from_name_status(sys.stdin)
    scope, operators = classify_paths(paths)
    print(scope)
    print(",".join(operators))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
