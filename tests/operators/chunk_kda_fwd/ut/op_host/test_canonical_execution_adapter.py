"""Host-only contracts for the 124 canonical non-accuracy cases."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[5]
CASE_DIR = ROOT / "test/chunk_kda_fwd"
ADAPTER_PATH = CASE_DIR / "canonical_execution_adapter.py"
RUNNER_PATH = CASE_DIR / "canonical_execution_runner.py"
BUILDER_PATH = CASE_DIR / "build_reference_cache.py"
CACHE_PATH = CASE_DIR / "persistent_reference_cache.py"
MANIFEST_PATH = ROOT / "tests/op_cases/chunk_kda_fwd.json"


def _catalog_content_pin(required_shards):
    return {
        "manifest_generation": "b" * 64,
        "shard_sha256": {name: "c" * 64 for name in required_shards},
    }


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def _adapter():
    return _load(ADAPTER_PATH, "chunk_kda_canonical_execution_adapter")


def _runner():
    return _load(RUNNER_PATH, "chunk_kda_canonical_execution_runner")


def _by_id(kind=None):
    return {
        record["spec"]["design_id"]: record["spec"]
        for record in _adapter().materialize(MANIFEST_PATH, kind=kind)
    }


def _cu(spec):
    return [int(value) for value in spec["cu_seqlens"].split(",")]


def test_materializer_has_exact_92_20_6_6_logical_counts_and_stable_ids():
    adapter = _adapter()
    records = adapter.materialize(MANIFEST_PATH)

    assert len(records) == 124
    assert all(set(record) == {"id", "spec"} for record in records)
    assert len({record["id"] for record in records}) == 124
    assert tuple(record["spec"]["design_id"] for record in records) == (
        adapter.EXPECTED_NON_ACCURACY_IDS
    )
    assert Counter(record["spec"]["execution_kind"] for record in records) == {
        "run": 92,
        "msopprof": 20,
        "stress": 6,
        "sanitizer": 6,
    }
    assert [record["id"] for record in adapter.materialize(MANIFEST_PATH, kind="run")][
        :3
    ] == [3001, 3002, 3003]
    assert records == adapter.materialize(MANIFEST_PATH)


def test_combined_cache_adapter_is_exactly_the_ordered_300_source_rows():
    adapter = _adapter()
    records = adapter.materialize_all(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert len(records) == 300
    assert len({record["id"] for record in records}) == 300
    assert [record["spec"]["design_id"] for record in records] == [
        row["id"] for row in manifest["design_matrix"]["cases"]
    ]


def test_materializer_rejects_any_changed_canonical_case_row(tmp_path):
    adapter = _adapter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["design_matrix"]["cases"][0]["variants"] = ["changed"]
    source = tmp_path / "chunk_kda_fwd.json"
    source.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="300 case rows changed"):
        adapter.materialize(source)


def test_run_specs_bind_every_id_to_an_explicit_mutation_and_route_outcome():
    cases = _by_id("run")

    assert cases["KDA-FWD-N001"]["mutation"] == "null_q"
    assert cases["KDA-FWD-N001"]["expected_return_code"] == 161001
    assert "nullptr" in cases["KDA-FWD-N001"]["expected_message"]
    assert cases["KDA-FWD-N081"]["mutation"] == "aqk_last_dim"
    assert cases["KDA-FWD-N084"]["mutation"] == "h_layout_or_state"
    assert cases["KDA-FWD-G083"]["mutation"] == "hv_lt_h_4_2"
    assert cases["KDA-FWD-G084"]["mutation"] == "hv_not_divisible_3_8"
    assert (cases["KDA-FWD-G084"]["H"], cases["KDA-FWD-G084"]["HV"]) == (3, 8)
    assert cases["KDA-FWD-G088"]["mutation"] == "initial_state_key_heads"
    assert set(cases["KDA-FWD-G088"]["expected_outcomes"]) == {
        "ascendc",
        "aclnn",
    }
    assert len(cases) == 92
    assert all(spec["mutation"] for spec in cases.values())


def test_run_projection_selects_real_route_specific_code_and_message():
    adapter = _adapter()
    public = next(
        record["spec"]
        for record in adapter.project_records(
            MANIFEST_PATH,
            kind="run",
            soc="ascend950",
            route="ascendc",
        )
        if record["spec"]["design_id"] == "KDA-FWD-G084"
    )
    aclnn = next(
        record["spec"]
        for record in adapter.project_records(
            MANIFEST_PATH,
            kind="run",
            soc="ascend950",
            route="aclnn",
        )
        if record["spec"]["design_id"] == "KDA-FWD-G084"
    )

    assert public["expected_return_code"] == "RuntimeError"
    assert "H/HV must satisfy" in public["expected_message"]
    assert aclnn["expected_return_code"] == 161002
    assert "divisible" in aclnn["expected_message"]


def test_run_atk_generation_is_full_route_projection_not_the_48_subset():
    adapter = _adapter()
    aclnn = adapter.build_run_atk_payloads(
        MANIFEST_PATH, soc="ascend950", route="aclnn"
    )
    public = adapter.build_run_atk_payloads(
        MANIFEST_PATH, soc="ascend950", route="ascendc"
    )

    assert len(aclnn) == 92
    assert len(public) == 12
    assert all(case["expected_error_msg"] for case in aclnn + public)
    assert all(
        next(item for item in case["inputs"] if item["name"] == "negative_case")[
            "range_values"
        ]
        is True
        for case in aclnn + public
    )


def test_performance_specs_are_device_profiler_contracts_with_explicit_thresholds():
    cases = _by_id("msopprof")
    m003, m006, m007, m008 = (
        cases[design_id]
        for design_id in (
            "KDA-FWD-M003",
            "KDA-FWD-M006",
            "KDA-FWD-M007",
            "KDA-FWD-M008",
        )
    )

    assert (m003["T"], m003["H"], m003["HV"]) == (16384, 96, 96)
    assert len(_cu(m006)) == 9
    assert m006["performance_expectation"] == {
        "baseline_design_id": "KDA-FWD-M003",
        "max_relative_regression": 0.05,
        "absolute_ms_lt": 12.0,
        "short_chain_exemption": False,
    }
    lengths = [right - left for left, right in zip(_cu(m007), _cu(m007)[1:])]
    assert sum(length // 64 for length in lengths) == 253
    assert sum(length % 64 != 0 for length in lengths) == 6
    assert m008["performance_expectation"]["short_chain_exemption"] is True
    assert cases["KDA-FWD-G092"]["performance_expectation"][
        "baseline_design_id"
    ] == "KDA-FWD-G091"
    assert all(spec["profiler"]["tool"] == "msopprof" for spec in cases.values())


def test_input_variant_plan_deduplicates_only_byte_identical_inputs():
    adapter = _adapter()
    cases = _by_id()

    m001 = adapter.materialize_input_variants(cases["KDA-FWD-M001"])
    s004 = adapter.materialize_input_variants(cases["KDA-FWD-S004"])
    s006 = adapter.materialize_input_variants(cases["KDA-FWD-S006"])
    g099 = adapter.materialize_input_variants(cases["KDA-FWD-G099"])

    assert len(m001["variant_specs"]) == 1
    assert len(set(m001["aliases"].values())) == 1
    assert len(s004["variant_specs"]) == 1
    assert len(s006["variant_specs"]) == 3
    assert len(g099["variant_specs"]) == 2


def test_stress_and_sanitizer_specs_cover_repeats_masks_tools_and_boundaries():
    stress = _by_id("stress")
    sanitizer = _by_id("sanitizer")

    assert stress["KDA-FWD-S001"]["repeat_count"] == 100
    assert (
        stress["KDA-FWD-S001"]["output_final_state"],
        stress["KDA-FWD-S001"]["disable_recompute"],
        stress["KDA-FWD-S001"]["return_intermediate_states"],
    ) == (True, True, True)
    assert stress["KDA-FWD-S004"]["repeat_count"] == 20
    assert stress["KDA-FWD-S004"]["cross_variant_common_outputs_bitwise"] is True
    assert stress["KDA-FWD-G097"]["data_variant"] == "gva_head_traceable"
    assert stress["KDA-FWD-G098"]["repeat_count"] == 100

    assert sanitizer["KDA-FWD-S005"]["sanitizer_tools"] == ["racecheck"]
    assert sanitizer["KDA-FWD-S006"]["design_variants"] == [
        "dense_key2",
        "mixed_tail_key2",
        "max_kv_key1",
    ]
    assert sanitizer["KDA-FWD-G099"]["sanitizer_tools"] == [
        "racecheck",
        "synccheck",
    ]
    assert sanitizer["KDA-FWD-S006"]["sanitizer"]["tool_options"]["memcheck"] == [
        "--leak-check=yes"
    ]
    assert (sanitizer["KDA-FWD-G100"]["H"], sanitizer["KDA-FWD-G100"]["HV"]) == (
        1,
        128,
    )


@pytest.mark.parametrize(
    "soc,total,planned,not_applicable,kind_counts",
    [
        (
            "ascend910b",
            139,
            114,
            25,
            {"run": 104, "msopprof": 5, "sanitizer": 5},
        ),
        (
            "ascend950",
            148,
            140,
            8,
            {"run": 104, "msopprof": 14, "stress": 8, "sanitizer": 14},
        ),
    ],
)
def test_a2_a5_physical_projection_matches_design_matrix_without_counting_na_as_pass(
    soc, total, planned, not_applicable, kind_counts
):
    records = _adapter().project_records(
        MANIFEST_PATH, soc=soc, include_not_applicable=True
    )
    statuses = Counter(record["spec"]["status"] for record in records)

    assert len(records) == total
    assert statuses == {"planned": planned, "not_applicable": not_applicable}
    assert Counter(
        record["spec"]["execution_kind"]
        for record in records
        if record["spec"]["status"] == "planned"
    ) == kind_counts
    assert not any(record["spec"]["status"] == "passed" for record in records)


def test_msopprof_command_uses_profiler_device_time_and_never_python_wall_time(tmp_path):
    adapter = _adapter()
    runner = _runner()
    spec = next(
        record["spec"]
        for record in adapter.project_records(
            MANIFEST_PATH,
            kind="msopprof",
            soc="ascend950",
            route="ascendc",
            variant="baseline",
        )
        if record["spec"]["design_id"] == "KDA-FWD-M006"
    )
    command = runner.materialize_msopprof_command(
        spec,
        source=MANIFEST_PATH,
        cache_dir=tmp_path / "cache",
        catalog_reference="a" * 64,
        output=tmp_path / "profile",
        python="python3",
        device=0,
    )

    joined = " ".join(command)
    assert command[0] == "msopprof"
    assert "--launch-count=20" in command
    assert "canonical_execution_runner.py application" in joined
    assert f"--catalog {'a' * 64}" in joined
    assert "--repeats 25" in joined
    assert "perf_counter" not in joined
    assert "wall" not in joined.lower()


def _write_kernel_json(directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"ChunkKdaFwd_{label}"
    path = directory / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "binFileName": stem,
                "kernelName": stem,
                "kernelList": [
                    {"kernelName": f"{stem}_1"},
                    {"kernelName": f"{stem}_2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path, f"{stem}_2_mix_aic"


def _write_basic_info(report, replay_index, rows):
    directory = report / str(replay_index)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "OpBasicInfo_test.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("Op Name", "Op Type", "Task Duration(us)"),
        )
        writer.writeheader()
        for name, op_type, duration in rows:
            writer.writerow(
                {
                    "Op Name": name,
                    "Op Type": op_type,
                    "Task Duration(us)": duration,
                }
            )


def _write_clean_profiler_log(directory):
    path = directory / "msopprof.log"
    path.write_text("msopprof completed successfully\n", encoding="utf-8")
    return path


def test_msopprof_structured_csv_parser_and_threshold_evaluation(tmp_path):
    runner = _runner()
    first_json, first_name = _write_kernel_json(tmp_path / "kernels", "a" * 32)
    second_json, second_name = _write_kernel_json(tmp_path / "kernels", "b" * 32)
    report = tmp_path / "profile"
    for replay_index in range(20):
        _write_basic_info(
            report,
            replay_index,
            [
                (first_name, "mix", str(400 + replay_index)),
                (first_name[: -len("_mix_aic")] + "_mix_aiv", "mix", "NA"),
                (second_name, "MIX", "100"),
                (second_name[: -len("_mix_aic")] + "_mix_aiv", "mix", ""),
            ],
        )
    parsed = runner.parse_msopprof_report(
        report,
        kernel_jsons=[first_json, second_json],
        log=_write_clean_profiler_log(tmp_path),
        launch_count=20,
    )
    spec = _by_id("msopprof")["KDA-FWD-M006"]
    baseline = {**parsed, "mean_application_us": 490.0}
    result = runner.evaluate_performance(spec, parsed, baseline)

    assert parsed["source"] == "msopprof_structured_report"
    assert parsed["mean_application_us"] == 509.5
    assert parsed["p95_application_us"] == 518.0
    assert parsed["stage_count"] == 2
    assert parsed["kernel_rows"] == 40
    assert parsed["replay_indices"] == list(range(20))
    assert parsed["replay_duration_us"] == [500.0 + index for index in range(20)]
    assert result["status"] == "passed"


@pytest.mark.parametrize(
    ("scenario", "error"),
    (
        ("duplicate", "duplicate stage"),
        ("wrong_type", "op_type"),
        ("unknown", "unknown profiler row"),
        ("missing", "stage set mismatch"),
        ("numeric_companion", "companion has numeric duration"),
        ("nonfinite", "positive and finite"),
    ),
)
def test_msopprof_parser_rejects_ambiguous_or_incomplete_rows(
    tmp_path, scenario, error
):
    runner = _runner()
    kernel_json, name = _write_kernel_json(tmp_path / "kernels", "c" * 32)
    companion = name[: -len("_mix_aic")] + "_mix_aiv"
    rows = [(name, "mix", "25"), (companion, "mix", "NA")]
    if scenario == "duplicate":
        rows.append((name, "mix", "25"))
    elif scenario == "wrong_type":
        rows[0] = (name, "aicore", "25")
    elif scenario == "unknown":
        rows.append((name + "_extra", "mix", "1"))
    elif scenario == "missing":
        rows = [(companion, "mix", "NA")]
    elif scenario == "numeric_companion":
        rows[1] = (companion, "mix", "1")
    elif scenario == "nonfinite":
        rows[0] = (name, "mix", "nan")
    _write_basic_info(tmp_path / "profile", 0, rows)

    with pytest.raises(ValueError, match=error):
        runner.parse_msopprof_report(
            tmp_path / "profile",
            kernel_jsons=[kernel_json],
            log=_write_clean_profiler_log(tmp_path),
            launch_count=1,
        )


def test_msopprof_parser_rejects_noncontiguous_replay_directories(tmp_path):
    runner = _runner()
    kernel_json, name = _write_kernel_json(tmp_path / "kernels", "d" * 32)
    for replay_index in (0, 2):
        _write_basic_info(tmp_path / "profile", replay_index, [(name, "mix", "10")])

    with pytest.raises(ValueError, match="replay indices must be contiguous"):
        runner.parse_msopprof_report(
            tmp_path / "profile",
            kernel_jsons=[kernel_json],
            log=_write_clean_profiler_log(tmp_path),
            launch_count=2,
        )


@pytest.mark.parametrize(
    "marker",
    (
        "507015",
        "RunDbiRecordTask failed",
        "DBI tune failed",
        "AIC_ERROR",
        "AICORE FAULT",
        "non-finite output",
    ),
)
def test_msopprof_parser_rejects_profiler_failure_markers(tmp_path, marker):
    runner = _runner()
    kernel_json, name = _write_kernel_json(tmp_path / "kernels", "e" * 32)
    _write_basic_info(tmp_path / "profile", 0, [(name, "mix", "10")])
    log = tmp_path / "msopprof.log"
    log.write_text(marker + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="failure marker"):
        runner.parse_msopprof_report(
            tmp_path / "profile",
            kernel_jsons=[kernel_json],
            log=log,
            launch_count=1,
        )


def test_performance_relative_threshold_includes_exact_boundary():
    runner = _runner()
    spec = {
        "design_id": "KDA-FWD-M-BOUNDARY",
        "materialized_variant": "baseline",
        "performance_expectation": {"max_relative_regression": 0.05},
    }
    baseline = {"mean_application_us": 1000.0}

    boundary = runner.evaluate_performance(
        spec, {"mean_application_us": 1050.0}, baseline
    )
    above = runner.evaluate_performance(
        spec, {"mean_application_us": 1050.0001}, baseline
    )

    assert boundary["status"] == "passed"
    assert boundary["checks"][0]["actual"] == pytest.approx(0.05)
    assert above["status"] == "failed"


@pytest.mark.parametrize("tool", ["racecheck", "memcheck", "initcheck", "synccheck"])
def test_sanitizer_requires_object_symbol_and_actual_kernel_hit(tool):
    runner = _runner()
    runner.verify_object_symbol("0000 T __mssanitizer_instrumented", "sanitizer")
    result = runner.verify_sanitizer_log(
        f"Start {tool} sanitizer on kernel chunk_kda_fwd_main\n0 errors\n",
        tool=tool,
        kernel_regex="chunk_kda_fwd",
    )
    assert result["status"] == "passed"

    with pytest.raises(RuntimeError, match="required sanitizer symbol"):
        runner.verify_object_symbol("0000 T ordinary_kernel", "sanitizer")
    with pytest.raises(RuntimeError, match="no active sanitizer"):
        runner.verify_sanitizer_log(
            "No active sanitizer tool on kernel chunk_kda_fwd\n",
            tool=tool,
            kernel_regex="chunk_kda_fwd",
        )
    with pytest.raises(RuntimeError, match="does not prove"):
        runner.verify_sanitizer_log(
            f"Start {tool} sanitizer on kernel another_op\n",
            tool=tool,
            kernel_regex="chunk_kda_fwd",
        )


def test_sanitizer_rejects_tool_specific_failure():
    runner = _runner()
    with pytest.raises(RuntimeError, match="reported failures"):
        runner.verify_sanitizer_log(
            "Start racecheck sanitizer on kernel chunk_kda_fwd\nData race detected\n",
            tool="racecheck",
            kernel_regex="chunk_kda_fwd",
        )


def test_cache_identity_reuses_routes_but_binds_negative_mutation(monkeypatch):
    cache = _load(CACHE_PATH, "chunk_kda_execution_cache")
    adapter = _adapter()
    public = next(
        record["spec"]
        for record in adapter.project_records(
            MANIFEST_PATH, kind="run", soc="ascend950", route="ascendc"
        )
        if record["spec"]["design_id"] == "KDA-FWD-G084"
    )
    aclnn = next(
        record["spec"]
        for record in adapter.project_records(
            MANIFEST_PATH, kind="run", soc="ascend950", route="aclnn"
        )
        if record["spec"]["design_id"] == "KDA-FWD-G084"
    )

    assert cache.normalize_spec(public) == cache.normalize_spec(aclnn)
    changed = dict(public)
    changed["mutation"] = "hv_lt_h"
    assert cache.normalize_spec(changed) != cache.normalize_spec(public)


def test_input_only_cache_manifest_requires_no_reference_shards(tmp_path):
    cache = _load(CACHE_PATH, "chunk_kda_input_only_cache")
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_chunk_kda_metadata(
        {"tags": "stress,canonical_300", "seed": 7},
        7,
        executor_path,
        producer_torch_version="test",
        include_references=False,
    )
    with cache.CacheWriter(tmp_path / "cache", metadata) as writer:
        writer.write_shard("inputs", {"seed": 7})
        writer.commit()

    reader = cache.CacheReader(tmp_path / "cache", metadata)
    reader.validate_all()
    assert set(reader.shards) == {"inputs"}
    with pytest.raises(cache.ReferenceCacheError, match="unknown cache shard"):
        reader.load_shard("cpu_fp64")


def test_builder_accepts_exact_300_adapter_and_preserves_execution_kinds(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE_DIR))
    builder = _load(BUILDER_PATH, "chunk_kda_all_cache_builder")
    cases = builder._load_cases(
        MANIFEST_PATH,
        set(),
        "canonical_execution_adapter:materialize_all",
    )

    assert len(cases) == 300
    assert len({case_id for case_id, _ in cases}) == 300
    assert Counter(
        next(
            tag
            for tag in ("accuracy", "run", "msopprof", "stress", "sanitizer")
            if tag in {item.strip() for item in spec["tags"].split(",")}
        )
        for _, spec in cases
    ) == {
        "accuracy": 176,
        "run": 92,
        "msopprof": 20,
        "stress": 6,
        "sanitizer": 6,
    }
    entries = []
    for case_id, spec in cases:
        required_shards = (
            ["inputs", "cpu_fp64", "cpu_same_precision"]
            if builder._include_references(spec)
            else ["inputs"]
        )
        entries.append(
            {
                "case_id": case_id,
                "cache_entries": [
                    {
                        "variant": item["variant"],
                        "cache_key": hashlib.sha256(
                            f"{case_id}:{item['variant']}".encode("ascii")
                        ).hexdigest(),
                        "required_shards": required_shards,
                        **_catalog_content_pin(required_shards),
                    }
                    for item in builder._cache_specs(spec)
                ],
            }
        )
    catalog = builder.build_catalog(
        MANIFEST_PATH,
        "canonical-all:v1",
        entries,
        adapter_sha256="1" * 64,
        variant_materializer_schema=builder.VARIANT_MATERIALIZER_SCHEMA,
        producer_torch_version="2.test",
        producer_executor_sha256="2" * 64,
        producer_golden_executor_sha256="3" * 64,
        producer_benchmark_executor_sha256="4" * 64,
    )
    assert catalog["case_count"] == 300
    assert catalog["cache_entry_count"] == 380
    assert Counter(
        tuple(cache_entry["required_shards"])
        for entry in catalog["entries"]
        for cache_entry in entry["cache_entries"]
    ) == {
        ("inputs", "cpu_fp64", "cpu_same_precision"): 256,
        ("inputs",): 124,
    }
    assert sum(
        cache_entry["variant"] == "traceable_metamorphic"
        for entry in catalog["entries"]
        for cache_entry in entry["cache_entries"]
    ) == 80
