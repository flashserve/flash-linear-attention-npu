"""Integrity checks for the canonical Chapter 21 KDA design matrix."""

from collections import Counter

from tests.operators.chunk_kda_fwd.common.design_matrix import (
    EXPECTED_KIND_COUNTS,
    EXPECTED_PREFIX_COUNTS,
    load_design_matrix,
    materialize_tasks,
    summarize_tasks,
    validate_design_matrix,
)


def test_design_matrix_has_all_300_logical_cases():
    result = validate_design_matrix()
    assert result["logical_cases"] == 300
    assert result["prefix_counts"] == EXPECTED_PREFIX_COUNTS
    assert result["kind_counts"] == EXPECTED_KIND_COUNTS


def test_every_logical_case_materializes_an_execution_plan():
    tasks = materialize_tasks()
    summary = summarize_tasks(tasks)
    assert summary["logical_ids"] == 300
    assert summary["physical_tasks"] > 300
    assert summary["status_counts"] == {"planned": summary["physical_tasks"]}
    task_ids = [task["task_id"] for task in tasks]
    assert len(task_ids) == len(set(task_ids))
    assert {task["entrypoint"]["adapter_status"] for task in tasks} == {
        "implemented"
    }


def test_accuracy_physical_plan_keeps_exact_a2_a5_variant_and_route_counts():
    expected = {
        "ascend910b": {
            "total": 329,
            "routes": {"ascendc": 164, "aclnn": 144, "direct_launch": 21},
        },
        "ascend950": {
            "total": 509,
            "routes": {"ascendc": 254, "aclnn": 234, "direct_launch": 21},
        },
    }
    for soc, contract in expected.items():
        tasks = materialize_tasks(
            kinds=("accuracy",), soc=soc, include_not_applicable=False
        )
        assert len(tasks) == contract["total"]
        assert Counter(task["route"] for task in tasks) == contract["routes"]
        assert {task["variant"] for task in tasks} == {
            "random",
            "traceable_metamorphic",
        }
        assert {task["status"] for task in tasks} == {"planned"}


def test_a3_only_cases_are_not_applicable_on_a2_or_a5():
    a3_only = ("KDA-FWD-M011", "KDA-FWD-M012", "KDA-FWD-G095", "KDA-FWD-G096")
    for soc in ("ascend910b", "ascend950"):
        tasks = materialize_tasks(case_ids=a3_only, soc=soc)
        assert len(tasks) == len(a3_only)
        assert {task["status"] for task in tasks} == {"not_applicable"}


def test_platform_projection_reports_not_applicable_separately_from_planned():
    summary = summarize_tasks(materialize_tasks(soc="ascend950"))
    assert summary["logical_ids"] == 300
    assert summary["applicable_logical_ids"] == 291
    assert summary["not_applicable_logical_ids"] == 9


def test_a3_only_cases_materialize_normally_on_a3():
    a3_only = ("KDA-FWD-M011", "KDA-FWD-M012", "KDA-FWD-G095", "KDA-FWD-G096")
    matrix = load_design_matrix()
    declared_a3_only = {
        case["id"]
        for case in matrix["cases"]
        if case["platforms"] == ["ascend910_93"]
    }
    assert declared_a3_only == set(a3_only)
    tasks = materialize_tasks(case_ids=a3_only, soc="ascend910_93")
    assert {task["logical_id"] for task in tasks} == set(a3_only)
    assert {task["status"] for task in tasks} == {"planned"}


def test_sanitizer_cases_keep_their_required_tools():
    matrix = load_design_matrix()
    sanitizer_cases = {
        case["id"]: case
        for case in matrix["cases"]
        if case["kind"] == "sanitizer"
    }
    assert sanitizer_cases["KDA-FWD-S005"]["sanitizer_tools"] == ["racecheck"]
    assert sanitizer_cases["KDA-FWD-S006"]["sanitizer_tools"] == ["memcheck"]
    assert sanitizer_cases["KDA-FWD-S007"]["sanitizer_tools"] == ["initcheck"]
    assert sanitizer_cases["KDA-FWD-S008"]["sanitizer_tools"] == ["synccheck"]
    assert sanitizer_cases["KDA-FWD-G099"]["sanitizer_tools"] == ["racecheck", "synccheck"]
    assert sanitizer_cases["KDA-FWD-G100"]["sanitizer_tools"] == ["memcheck", "initcheck"]


def test_profile_and_mask_subexperiments_are_physical_tasks_not_new_logical_ids():
    matrix = load_design_matrix()
    cases = {case["id"]: case for case in matrix["cases"]}
    assert cases["KDA-FWD-M001"]["variants"] == [
        "baseline",
        "l2_streaming_single_read_disabled",
    ]
    assert cases["KDA-FWD-M002"]["variants"] == ["baseline"]
    assert cases["KDA-FWD-S004"]["variants"] == ["all_outputs", "hidden_outputs"]
    assert cases["KDA-FWD-G098"]["variants"] == ["all_outputs", "hidden_outputs"]


def test_unknown_design_case_cannot_silently_materialize_an_empty_plan():
    try:
        materialize_tasks(case_ids=("KDA-FWD-P999",))
    except ValueError as error:
        assert "unknown design case IDs" in str(error)
    else:
        raise AssertionError("unknown design ID unexpectedly produced a plan")
