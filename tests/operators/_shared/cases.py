"""Shared JSON case loading and selection helpers for operator tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


CASES_ROOT = Path(__file__).resolve().parents[2] / "op_cases"


def _csv_values(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def load_cases(op: str) -> dict:
    path = CASES_ROOT / f"{op}.json"
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("op") != op or not isinstance(manifest.get("cases"), list):
        raise ValueError(f"invalid case manifest: {path}")
    return manifest


def select_cases(
    op: str,
    *,
    tags: Iterable[str] | None = None,
    route: str | None = None,
    include_negative: bool = True,
) -> list[dict]:
    required_tags = set(tags or ()) | _csv_values(os.environ.get("FLA_NPU_CASE_TAGS"))
    selected_ids = _csv_values(os.environ.get("FLA_NPU_CASE_IDS"))
    selected = []
    for case in load_cases(op)["cases"]:
        case_tags = set(case.get("tags", ()))
        if required_tags and not required_tags <= case_tags:
            continue
        if route is not None and route not in case.get("run_on", ()):
            continue
        if not include_negative and "negative" in case_tags:
            continue
        if selected_ids and case.get("id") not in selected_ids:
            continue
        selected.append(case)
    return selected
