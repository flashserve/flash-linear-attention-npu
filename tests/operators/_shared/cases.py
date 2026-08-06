"""Load and filter the canonical operator case manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_ROOT = REPO_ROOT / "tests" / "op_cases"


def load_cases(op_name: str) -> dict[str, Any]:
    path = CASE_ROOT / f"{op_name}.json"
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("op") != op_name:
        raise ValueError(f"{path}: op must be {op_name!r}, got {manifest.get('op')!r}")
    return manifest


def _contains_all(values: Iterable[str], expected: set[str]) -> bool:
    return expected.issubset(set(values))


def select_cases(
    op_name: str,
    *,
    tags: Iterable[str] = (),
    route: str | None = None,
    soc: str | None = None,
    include_negative: bool = True,
) -> list[dict[str, Any]]:
    """Return cases matching all requested dimensions.

    Environment variables are intentionally supported so CI can use the same
    entry point without rewriting case lists:

    - ``FLA_NPU_CASE_IDS``: comma-separated explicit IDs.
    - ``FLA_NPU_CASE_TAGS``: comma-separated tags that every selected case has.
    - ``FLA_NPU_SOC``: SOC name used when ``soc`` is not passed.
    """

    manifest = load_cases(op_name)
    required_tags = set(tags)
    required_tags.update(filter(None, os.environ.get("FLA_NPU_CASE_TAGS", "").split(",")))
    explicit_ids = set(filter(None, os.environ.get("FLA_NPU_CASE_IDS", "").split(",")))
    soc = soc or os.environ.get("FLA_NPU_SOC")

    selected = []
    for case in manifest["cases"]:
        if explicit_ids and case["id"] not in explicit_ids:
            continue
        if required_tags and not _contains_all(case.get("tags", ()), required_tags):
            continue
        if route is not None and route not in case.get("run_on", ()):
            continue
        if soc is not None and soc not in case.get("soc", ()):
            continue
        if not include_negative and "negative" in case.get("tags", ()):
            continue
        selected.append(case)
    return selected


def case_ids(op_name: str, **filters: Any) -> list[str]:
    return [case["id"] for case in select_cases(op_name, **filters)]
