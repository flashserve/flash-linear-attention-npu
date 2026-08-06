"""FLA-aligned optional-output policy contract for chunk_kda_fwd."""

import importlib.util
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
POLICY_PATH = (
    ROOT
    / "torch_custom/fla_npu/fla_npu/ops/ascendc/_kda_policy.py"
)
MANIFEST_PATH = ROOT / "tests/op_cases/chunk_kda_fwd.json"


def _load_policy_module():
    spec = importlib.util.spec_from_file_location("fla_npu_kda_policy", POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_mask(output_final_state, use_gate_in_kernel, disable_recompute, return_states):
    return (
        True,
        output_final_state,
        not use_gate_in_kernel or disable_recompute,
        True,
        True,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute or return_states,
        True,
    )


def test_optional_output_policy_covers_all_16_boolean_combinations():
    policy = _load_policy_module()
    combinations = list(itertools.product((False, True), repeat=4))
    assert len(combinations) == 16
    for output_final_state, use_gate_in_kernel, disable_recompute, return_states in combinations:
        actual = policy.kda_fwd_optional_output_mask(
            output_final_state=output_final_state,
            use_gate_in_kernel=use_gate_in_kernel,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_states,
        )
        assert actual == _expected_mask(
            output_final_state,
            use_gate_in_kernel,
            disable_recompute,
            return_states,
        )


def test_optional_output_alignment_is_pinned_to_manifest_commit():
    policy = _load_policy_module()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    alignment = manifest["upstream_alignment"]
    assert alignment["repository"] == "fla-org/flash-linear-attention"
    assert alignment["commit"] == policy.FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT
    assert alignment["source"] == "fla/ops/kda/chunk_fwd.py"
    assert alignment["return_policy"] == {
        "final_state": "output_final_state",
        "gk": "not use_gate_in_kernel or disable_recompute",
        "w_u_qg_kg_v_new": "disable_recompute",
        "h": "disable_recompute or return_intermediate_states",
    }

    case_id = manifest["coverage_requirements"]["optional_output_case_ids"]
    assert case_id == ["chunk_kda_fwd_optional_output_matrix"]
    case = next(item for item in manifest["cases"] if item["id"] == case_id[0])
    assert policy.FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT in case["reference"]
    matrix = case["attrs"]["optional_output_matrix"]
    assert all(values == [False, True] for values in matrix.values())
    assert case["expect"]["matrix_size"] == 16
