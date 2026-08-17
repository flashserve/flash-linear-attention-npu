"""Host-only contracts for the canonical 176-case ATK accuracy adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[5]
ADAPTER_PATH = ROOT / "test/chunk_kda_fwd/canonical_case_adapter.py"
MANIFEST_PATH = ROOT / "tests/op_cases/chunk_kda_fwd.json"
CACHE_PATH = ROOT / "test/chunk_kda_fwd/persistent_reference_cache.py"
ATK_YAML_PATH = ROOT / "test/chunk_kda_fwd/chunk_kda_fwd.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _adapter():
    return _load(ADAPTER_PATH, "chunk_kda_canonical_case_adapter")


def _specs_by_design_id():
    return {
        record["spec"]["design_id"]: record["spec"]
        for record in _adapter().materialize(MANIFEST_PATH)
    }


def _cu(spec):
    return [int(value) for value in spec["cu_seqlens"].split(",")]


def test_atk_schema_accepts_every_canonical_target_soc():
    schema = ATK_YAML_PATH.read_text(encoding="utf-8")
    soc_block = schema.split("- name: soc", 1)[1].split("- name: route", 1)[0]
    assert "values: [ascend910b, ascend910_93, ascend950]" in soc_block


def test_materializer_has_exact_ordered_176_logical_accuracy_specs():
    adapter = _adapter()
    records = adapter.materialize(MANIFEST_PATH)

    assert len(records) == 176
    assert all(set(record) == {"id", "spec"} for record in records)
    assert [record["id"] for record in records] == [
        *range(1001, 1097),
        *range(2001, 2081),
    ]
    assert tuple(record["spec"]["design_id"] for record in records) == (
        adapter.EXPECTED_ACCURACY_IDS
    )
    assert len({record["spec"]["seed"] for record in records}) == 176
    assert all(
        adapter.EXECUTOR_REQUIRED_FIELDS <= set(record["spec"])
        for record in records
    )
    assert records == adapter.materialize(MANIFEST_PATH)


def test_accuracy_variants_materialize_256_distinct_numeric_cache_entries():
    adapter = _adapter()
    records = adapter.materialize(MANIFEST_PATH)
    variants = [
        item
        for record in records
        for item in adapter.materialize_cache_variants(record["spec"])
    ]

    assert len(variants) == 256
    assert sum(item["variant"] == "random" for item in variants) == 176
    assert sum(item["variant"] == "traceable_metamorphic" for item in variants) == 80
    assert all(
        item["spec"]["materialized_variant"] == item["variant"]
        for item in variants
    )
    assert all(
        item["spec"]["traceable_head_mapping"]
        for item in variants
        if item["variant"] == "traceable_metamorphic"
    )

    g039 = next(
        item["spec"]
        for item in variants
        if item["spec"]["design_id"] == "KDA-FWD-G039"
        and item["variant"] == "traceable_metamorphic"
    )
    g043 = next(
        item["spec"]
        for item in variants
        if item["spec"]["design_id"] == "KDA-FWD-G043"
        and item["variant"] == "traceable_metamorphic"
    )
    assert g039["data_variant"] == "head_distinct_a_log"
    assert g043["data_variant"] == "initial_state_pulse_hv_2"


def test_positive_output_layout_dtype_and_gate_rules_are_explicit():
    cases = _specs_by_design_id()

    assert (
        cases["KDA-FWD-P001"]["output_final_state"],
        cases["KDA-FWD-P001"]["use_gate_in_kernel"],
        cases["KDA-FWD-P001"]["disable_recompute"],
        cases["KDA-FWD-P001"]["return_intermediate_states"],
    ) == (False, False, False, False)
    assert (
        cases["KDA-FWD-P016"]["output_final_state"],
        cases["KDA-FWD-P016"]["use_gate_in_kernel"],
        cases["KDA-FWD-P016"]["disable_recompute"],
        cases["KDA-FWD-P016"]["return_intermediate_states"],
    ) == (True, True, True, True)
    assert cases["KDA-FWD-P017"]["layout"] == "BSND"
    assert cases["KDA-FWD-P020"]["layout"] == "NTD"
    assert _cu(cases["KDA-FWD-P023"]) == [0, 64, 128]
    assert cases["KDA-FWD-P024"]["input_storage"] == ["q", "k", "v", "g", "beta"]
    assert (
        cases["KDA-FWD-P032"]["q_dtype"],
        cases["KDA-FWD-P032"]["g_dtype"],
        cases["KDA-FWD-P032"]["beta_dtype"],
    ) == ("fp16", "bf16", "bf16")
    assert (
        cases["KDA-FWD-P036"]["a_log_dtype"],
        cases["KDA-FWD-P036"]["dt_bias_dtype"],
        cases["KDA-FWD-P036"]["safe_gate"],
    ) == ("bf16", "bf16", False)
    assert cases["KDA-FWD-P038"]["lower_bound"] == -0.001


def test_positive_shape_gva_state_and_varlen_rules_are_explicit():
    cases = _specs_by_design_id()

    assert cases["KDA-FWD-P042"]["initial_state"] is True
    assert cases["KDA-FWD-P042"]["state_v_first"] is True
    assert (cases["KDA-FWD-P044"]["H"], cases["KDA-FWD-P044"]["HV"]) == (2, 8)
    assert (
        cases["KDA-FWD-P057"]["chunk_size"],
        cases["KDA-FWD-P057"]["T"],
        cases["KDA-FWD-P057"]["K"],
        cases["KDA-FWD-P057"]["V"],
    ) == (128, 257, 128, 256)
    assert _cu(cases["KDA-FWD-P068"]) == [0, 65, 193]
    assert (
        cases["KDA-FWD-P071"]["H"],
        cases["KDA-FWD-P071"]["HV"],
        cases["KDA-FWD-P071"]["K"],
        cases["KDA-FWD-P071"]["V"],
    ) == (1, 4, 256, 256)

    p073 = cases["KDA-FWD-P073"]
    assert (p073["layout"], p073["T"], p073["H"], p073["HV"]) == (
        "TND",
        1024,
        96,
        96,
    )
    assert _cu(p073) == [0, 1024]
    assert p073["explicit_chunk_indices"] is False
    p096 = cases["KDA-FWD-P096"]
    assert p096["T"] == 16384
    assert len(_cu(p096)) == 257
    assert all(
        right - left == 64 for left, right in zip(_cu(p096), _cu(p096)[1:])
    )


def test_gva_rules_cover_head_layout_gate_state_and_long_varlen_cases():
    cases = _specs_by_design_id()

    assert (cases["KDA-FWD-G001"]["H"], cases["KDA-FWD-G001"]["HV"]) == (1, 2)
    assert (cases["KDA-FWD-G016"]["H"], cases["KDA-FWD-G016"]["HV"]) == (96, 96)
    assert (
        cases["KDA-FWD-G024"]["layout"],
        cases["KDA-FWD-G024"]["q_dtype"],
        _cu(cases["KDA-FWD-G024"]),
    ) == ("NTD", "fp16", [0, 65, 128])
    assert (
        cases["KDA-FWD-G030"]["H"],
        cases["KDA-FWD-G030"]["HV"],
        cases["KDA-FWD-G030"]["g_dtype"],
        cases["KDA-FWD-G030"]["beta_dtype"],
    ) == (3, 96, "bf16", "bf16")
    assert cases["KDA-FWD-G032"]["input_storage"] == ["v", "g", "beta"]
    assert (
        cases["KDA-FWD-G036"]["a_log_dtype"],
        cases["KDA-FWD-G037"]["dt_bias_dtype"],
        cases["KDA-FWD-G038"]["dt_bias"],
    ) == ("bf16", "bf16", False)
    assert cases["KDA-FWD-G039"]["data_variant"] == "head_distinct_a_log"
    assert cases["KDA-FWD-G040"]["data_variant"] == "head_distinct_dt_bias"
    assert cases["KDA-FWD-G043"]["data_variant"] == "initial_state_pulse_hv_2"
    assert cases["KDA-FWD-G044"]["data_variant"] == "initial_state_pulse_hv_3"
    assert cases["KDA-FWD-G045"]["output_final_state"] is False
    assert cases["KDA-FWD-G046"]["disable_recompute"] is False
    assert cases["KDA-FWD-G048"]["return_intermediate_states"] is False
    assert (
        cases["KDA-FWD-G060"]["K"],
        cases["KDA-FWD-G060"]["V"],
        cases["KDA-FWD-G060"]["H"],
        cases["KDA-FWD-G060"]["HV"],
    ) == (16, 16, 2, 8)
    assert _cu(cases["KDA-FWD-G071"]) == [0, 64, 64, 128]
    assert (
        cases["KDA-FWD-G080"]["T"],
        cases["KDA-FWD-G080"]["H"],
        cases["KDA-FWD-G080"]["HV"],
        cases["KDA-FWD-G080"]["distribution"],
    ) == (16384, 3, 96, "mixed")


def test_soc_and_route_projection_reuses_the_same_cached_numeric_identity():
    adapter = _adapter()
    cache = _load(CACHE_PATH, "chunk_kda_persistent_reference_cache")
    projections = {}
    records_by_route = {}
    for route in ("ascendc", "aclnn", "direct_launch"):
        records = adapter.project_records(MANIFEST_PATH, soc="ascend950", route=route)
        records_by_route[route] = records
        projections[route] = {
            (
                record["spec"]["design_id"],
                record["spec"]["materialized_variant"],
            ): record["spec"]
            for record in records
        }

    assert {route: len(records) for route, records in records_by_route.items()} == {
        "ascendc": 254,
        "aclnn": 234,
        "direct_launch": 21,
    }
    key = ("KDA-FWD-P018", "random")
    assert key in projections["direct_launch"]
    identities = [
        cache.normalize_spec(projections[route][key])
        for route in ("ascendc", "aclnn", "direct_launch")
    ]
    assert identities[0] == identities[1] == identities[2]
    assert {
        projections[route][key]["route"]
        for route in projections
    } == {"ascendc", "aclnn", "direct_launch"}

    all_specs = _specs_by_design_id()
    p033_spec = all_specs["KDA-FWD-P033"]
    p033 = cache.normalize_spec(p033_spec)
    a_log_changed = cache.normalize_spec({**p033_spec, "a_log_dtype": "bf16"})
    dt_bias_changed = cache.normalize_spec({**p033_spec, "dt_bias_dtype": "bf16"})
    assert p033["a_log_dtype"] == "fp32"
    assert p033["dt_bias_dtype"] == "fp32"
    assert p033 != a_log_changed
    assert p033 != dt_bias_changed


@pytest.mark.parametrize(
    "soc,route_counts,total",
    [
        (
            "ascend910b",
            {"ascendc": 164, "aclnn": 144, "direct_launch": 21},
            329,
        ),
        (
            "ascend950",
            {"ascendc": 254, "aclnn": 234, "direct_launch": 21},
            509,
        ),
    ],
)
def test_accuracy_projection_preserves_logical_and_physical_count_contracts(
    soc, route_counts, total
):
    adapter = _adapter()
    actual = {
        route: len(adapter.project_records(MANIFEST_PATH, soc=soc, route=route))
        for route in route_counts
    }

    assert len(adapter.materialize(MANIFEST_PATH)) == 176
    assert actual == route_counts
    assert sum(actual.values()) == total
    assert all(
        len(
            {
                (record["id"], record["spec"]["materialized_variant"])
                for record in adapter.project_records(
                    MANIFEST_PATH, soc=soc, route=route
                )
            }
        )
        == count
        for route, count in route_counts.items()
    )


def test_atk_payload_embeds_executable_spec_and_stable_design_id():
    adapter = _adapter()
    payloads = adapter.build_atk_payloads(
        MANIFEST_PATH, soc="ascend950", route="direct_launch"
    )
    case = next(item for item in payloads if item["id"] == 1018)
    attrs = {item["name"]: item["range_values"] for item in case["inputs"]}
    spec = json.loads(attrs["case_spec"])

    assert attrs["design_id"] == "KDA-FWD-P018"
    assert spec["design_id"] == "KDA-FWD-P018"
    assert spec["soc"] == "ascend950"
    assert spec["route"] == "direct_launch"
    assert case["api_type"] == "executor_chunk_kda_fwd"
    assert "cv_fused_double_benchmark" in case["standard"]["acc"]

    traceable = adapter.build_atk_payloads(
        MANIFEST_PATH,
        soc="ascend950",
        route="ascendc",
        variant="traceable_metamorphic",
    )
    trace_case = next(item for item in traceable if item["id"] == 12001)
    trace_attrs = {
        item["name"]: item["range_values"] for item in trace_case["inputs"]
    }
    trace_spec = json.loads(trace_attrs["case_spec"])
    assert trace_spec["design_id"] == "KDA-FWD-G001"
    assert trace_spec["materialized_variant"] == "traceable_metamorphic"
    assert trace_spec["traceable_head_mapping"] is True

    with pytest.raises(ValueError, match="unsupported accuracy variant"):
        adapter.project_records(MANIFEST_PATH, variant="not-a-variant")


def test_adapter_rejects_stale_canonical_source_digest(tmp_path):
    adapter = _adapter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["design_matrix"]["source"]["row_sha256"] = "0" * 64
    source = tmp_path / "chunk_kda_fwd.json"
    source.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="design source changed"):
        adapter.materialize(source)
