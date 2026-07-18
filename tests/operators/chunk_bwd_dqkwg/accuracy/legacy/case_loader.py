"""Compatibility adapter backed by tests/op_cases/chunk_bwd_dqkwg.json."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tests.operators._shared.legacy_cases import dqkwg_case_info  # noqa: E402


def get_case_info(case_name):
    case = dqkwg_case_info(case_name)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    case["dtype"] = dtype_map[case["dtype"]]
    case["Gtype"] = dtype_map[case["Gtype"]]
    return case
