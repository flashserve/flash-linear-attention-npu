"""读取 ChunkKdaFwdPrepare 的统一用例清单。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


OP_NAME = "chunk_kda_fwd_prepare"
REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "tests" / "op_cases" / f"{OP_NAME}.json"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("op") != OP_NAME:
        raise ValueError(
            f"{MANIFEST_PATH}: op must be {OP_NAME!r}, "
            f"got {manifest.get('op')!r}"
        )
    return manifest


def select_cases(*, route: str | None = None, tag: str | None = None):
    requested_ids = set(
        filter(None, os.environ.get("FLA_NPU_CASE_IDS", "").split(","))
    )
    soc = os.environ.get("FLA_NPU_SOC")
    selected = []
    for case in load_manifest()["cases"]:
        if requested_ids and case["id"] not in requested_ids:
            continue
        if route is not None and route not in case.get("run_on", ()):
            continue
        if tag is not None and tag not in case.get("tags", ()):
            continue
        if soc is not None and soc not in case.get("soc", ()):
            continue
        selected.append(case)
    return selected
