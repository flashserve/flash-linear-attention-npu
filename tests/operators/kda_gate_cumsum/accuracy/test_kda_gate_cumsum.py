"""Canonical JSON entry for KdaGateCumsum accuracy/generalization tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.operators._shared.cases import load_cases, select_cases
from tests.operators._shared.npu_generalization import run_generalization_cases


OP = "kda_gate_cumsum"
ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "torch_custom/fla_npu/test/test_npu_chunk_kda.py"


def test_case_manifest_covers_required_matrix():
    manifest = load_cases(OP)
    cases = manifest["cases"]
    tags = {tag for case in cases for tag in case["tags"]}
    assert {"accuracy", "generalization", "boundary", "negative", "route", "performance", "example"} <= tags
    assert {"ascend910b", "ascend910_93", "ascend950"} <= set(manifest["capability"]["soc"])
    assert all("ascendc" in case["run_on"] for case in cases if "accuracy" in case["tags"])


def test_selected_case_ids_are_unique():
    selected = select_cases(OP)
    ids = [case["id"] for case in selected]
    assert len(ids) == len(set(ids))


@pytest.mark.npu
def test_json_generalization_cases():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    cases = select_cases(
        OP,
        tags=("generalization",),
        route="ascendc",
        include_negative=False,
    )
    assert cases, f"{OP} has no executable generalization cases"
    run_generalization_cases(OP, cases)


@pytest.mark.npu
def test_main_ascendc_accuracy_backend():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert RUNNER.is_file(), RUNNER
    env = os.environ.copy()
    env["FLA_NPU_OPERATOR"] = OP
    env["FLA_NPU_CASE_MANIFEST"] = str(ROOT / "tests" / "op_cases" / f"{OP}.json")
    env["FLA_NPU_CASE_IDS"] = ",".join(case["id"] for case in select_cases(OP, route="ascendc"))
    subprocess.run([sys.executable, str(RUNNER)], cwd=RUNNER.parent, env=env, check=True)
