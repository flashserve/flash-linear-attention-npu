"""JSON selectors shared by chunk_scaled_dot_kkt tests."""

from __future__ import annotations

import os

from tests.operators._shared.cases import load_cases, select_cases


OP = "chunk_scaled_dot_kkt"


def manifest():
    return load_cases(OP)


def cases(*, tag=None, route=None):
    old_tags = os.environ.get("FLA_NPU_CASE_TAGS")
    if tag is not None:
        os.environ["FLA_NPU_CASE_TAGS"] = tag
    try:
        return select_cases(OP, route=route)
    finally:
        if tag is not None:
            if old_tags is None:
                os.environ.pop("FLA_NPU_CASE_TAGS", None)
            else:
                os.environ["FLA_NPU_CASE_TAGS"] = old_tags


def case_ids(*, tag=None, route=None):
    return [case["id"] for case in cases(tag=tag, route=route)]
