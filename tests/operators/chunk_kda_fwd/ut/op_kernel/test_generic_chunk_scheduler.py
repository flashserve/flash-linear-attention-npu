"""Static contracts for the generic A2/A3 chunk-owner scheduler."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
KERNEL_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel"
PLAN = KERNEL_ROOT / "chunk_kda_fwd_plan.h"
PREPARE = KERNEL_ROOT / "chunk_kda_fwd_prepare.h"
POST_WU = KERNEL_ROOT / "chunk_kda_fwd_post_wu.h"
FINALIZE = KERNEL_ROOT / "chunk_kda_fwd_finalize.h"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_body(path: Path, function_name: str) -> str:
    source = _source(path)
    signature = source.index(f"void {function_name}(")
    opening = source.index("{", signature)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {function_name}")


ENTRY_POINTS = {
    PREPARE: ("ProcessPreAiv", "ProcessPreAic"),
    POST_WU: (
        "ProcessPostAiv",
        "ProcessPostAic",
        "ProcessVarlenTailSeedCopyAiv",
        "ProcessVarlenTailAic",
        "ProcessVarlenTailAiv",
    ),
    FINALIZE: ("ProcessOutAiv", "ProcessOutAic"),
}


HEAD_RANGES = {
    PREPARE: (
        ("ProcessCompactPreAivHeadRange", "ProcessCompactPreAivHeadWindow", True),
        ("ProcessCompactPreAicHeadRange", "ProcessCompactPreAicHeadWindow", True),
    ),
    POST_WU: (
        ("ProcessCompactPostAivHeadRange", "ProcessCompactPostAivHeadWindow", True),
        ("ProcessCompactPostAicHeadRange", "ProcessCompactPostAicHeadWindow", True),
        ("ProcessCompactTailSeedAivHeadRange", "ProcessCompactTailSeedAivHeadWindow", False),
        (
            "ProcessCompactTailSnapshotAicHeadRange",
            "ProcessCompactTailSnapshotAicHeadWindow",
            False,
        ),
        (
            "ProcessCompactTailSnapshotAivHeadRange",
            "ProcessCompactTailSnapshotAivHeadWindow",
            False,
        ),
    ),
    FINALIZE: (
        ("ProcessCompactOutAivHeadRange", "ProcessCompactOutAivHeadWindow", True),
        ("ProcessCompactOutAicHeadRange", "ProcessCompactOutAicHeadWindow", True),
    ),
}


G1_PHASES = {
    PREPARE: (
        ("ProcessG1FullPreAivPhase", "LoadOwnedFullChunk"),
        ("ProcessG1TailPreAivPhase", "LoadOwnedTailChunk"),
        ("ProcessG1FullPreAicPhase", "LoadOwnedFullChunk"),
        ("ProcessG1TailPreAicPhase", "LoadOwnedTailChunk"),
    ),
    POST_WU: (
        ("ProcessG1FullPostAivPhase", "LoadCompactFullChunk"),
        ("ProcessG1TailPostAivPhase", "LoadCompactTailChunk"),
        ("ProcessG1FullPostAicPhase", "LoadCompactFullChunk"),
        ("ProcessG1TailPostAicPhase", "LoadCompactTailChunk"),
        ("ProcessG1TailSeedAivPhase", "LoadCompactTailChunk"),
        ("ProcessG1TailSnapshotAicPhase", "LoadCompactTailChunk"),
        ("ProcessG1TailSnapshotAivPhase", "LoadCompactTailChunk"),
    ),
    FINALIZE: (
        ("ProcessG1FullOutAivPhase", "LoadCompactFullChunk"),
        ("ProcessG1TailOutAivPhase", "LoadCompactTailChunk"),
        ("ProcessG1FullOutAicPhase", "LoadCompactFullChunk"),
        ("ProcessG1TailOutAicPhase", "LoadCompactTailChunk"),
    ),
}


ENUMERATION_PAIRS = (
    (PREPARE, "ProcessG1FullPreAivPhase", "ProcessG1FullPreAicPhase", "g1_full"),
    (PREPARE, "ProcessG1TailPreAivPhase", "ProcessG1TailPreAicPhase", "g1_tail"),
    (
        PREPARE,
        "ProcessGroupedFullPreAivPhase",
        "ProcessGroupedFullPreAicPhase",
        "grouped_full",
    ),
    (
        PREPARE,
        "ProcessGroupedTailPreAivPhase",
        "ProcessGroupedTailPreAicPhase",
        "grouped_tail",
    ),
    (POST_WU, "ProcessG1FullPostAivPhase", "ProcessG1FullPostAicPhase", "g1_full"),
    (POST_WU, "ProcessG1TailPostAivPhase", "ProcessG1TailPostAicPhase", "g1_tail"),
    (
        POST_WU,
        "ProcessGroupedFullPostAivPhase",
        "ProcessGroupedFullPostAicPhase",
        "grouped_full",
    ),
    (
        POST_WU,
        "ProcessGroupedTailPostAivPhase",
        "ProcessGroupedTailPostAicPhase",
        "grouped_tail",
    ),
    (
        POST_WU,
        "ProcessG1TailSnapshotAivPhase",
        "ProcessG1TailSnapshotAicPhase",
        "g1_tail",
    ),
    (
        POST_WU,
        "ProcessGroupedTailSnapshotAivPhase",
        "ProcessGroupedTailSnapshotAicPhase",
        "grouped_tail",
    ),
    (FINALIZE, "ProcessG1FullOutAivPhase", "ProcessG1FullOutAicPhase", "g1_full"),
    (FINALIZE, "ProcessG1TailOutAivPhase", "ProcessG1TailOutAicPhase", "g1_tail"),
    (
        FINALIZE,
        "ProcessGroupedFullOutAivPhase",
        "ProcessGroupedFullOutAicPhase",
        "grouped_full",
    ),
    (
        FINALIZE,
        "ProcessGroupedTailOutAivPhase",
        "ProcessGroupedTailOutAicPhase",
        "grouped_tail",
    ),
)


ENUMERATION_TOKENS = {
    "g1_full": (
        "cursor.fullStartSequence",
        "cursor.fullStartLocalChunk",
        "ordinal = cursor.fullBegin",
        "ordinal < cursor.fullEnd",
    ),
    "g1_tail": (
        "ordinal = cursor.tailBegin",
        "ordinal < cursor.tailEnd",
    ),
    "grouped_full": (
        "cursor.fullStartSequence",
        "cursor.fullStartLocalChunk",
        "task = cursor.fullBegin",
        "task < cursor.fullEnd",
    ),
    "grouped_tail": (
        "task = cursor.tailBegin",
        "task < cursor.tailEnd",
    ),
}


def test_compact_stages_dispatch_g1_or_grouped_once_at_stage_entry():
    for path, entry_points in ENTRY_POINTS.items():
        source = _source(path)
        assert "ComputeChunkHeadGroupCount" not in source
        for entry_point in entry_points:
            body = _function_body(path, entry_point)
            assert body.count("plan.HeadGroupCount() == 1") == 1
            assert body.index("LoadChunkCoreCursor") < body.index(
                "plan.HeadGroupCount() == 1"
            )


def test_g1_phases_use_direct_chunk_ownership_without_group_decode():
    for path, phases in G1_PHASES.items():
        for phase, direct_loader in phases:
            body = _function_body(path, phase)
            assert direct_loader in body
            assert "DecodeChunkHeadGroupTask" not in body
            assert "LoadGrouped" not in body
            assert "task %" not in body
            assert "task /" not in body


def test_head_ranges_use_runtime_capped_windows_and_shared_semaphore_loop():
    plan_source = _source(PLAN)
    assert plan_source.count("constexpr uint32_t KDA_HEADS_PER_TASK = 4;") == 1
    for path, ranges in HEAD_RANGES.items():
        source = _source(path)
        assert '#include "chunk_kda_fwd_plan.h"' in source
        assert "constexpr uint32_t KDA_HEADS_PER_TASK = 4;" not in source
        assert "template <bool IS_TAIL, uint32_t HEAD_COUNT>" not in source
        assert "template <uint32_t HEAD_COUNT>" not in source
        for range_name, window_name, has_tail_template in ranges:
            range_body = _function_body(path, range_name)
            window_body = _function_body(path, window_name)
            assert "headBase < headEnd" in range_body
            assert "const uint32_t headCnt" in range_body
            assert "KdaForward::HeadWindowHeadCount(" in range_body
            assert "H_" in range_body
            assert "HV_" in range_body
            assert "headBase += headCnt" in range_body
            assert "headBase += KDA_HEADS_PER_TASK" not in range_body
            runtime_call = (
                f"{window_name}<IS_TAIL>"
                if has_tail_template
                else window_name
            )
            assert runtime_call in range_body
            assert "headBase, headCnt" in range_body
            assert "for (uint32_t lane = 0; lane < headCnt; ++lane)" in window_body
            assert "#pragma unroll" not in window_body
            assert "lane < HEAD_COUNT" not in window_body
            assert "switch (headEnd - headBase)" not in range_body
            for width in (1, 2, 3, 4):
                assert f"{window_name}<{width}>" not in range_body
                assert f"{window_name}<IS_TAIL, {width}>" not in range_body


def test_full_and_tail_math_paths_are_statically_separate():
    forbidden_runtime_tail_dispatch = (
        "IS_TAIL ?",
        "if (IS_TAIL)",
        "? cursor.tailBegin",
        "const bool isTail",
    )
    for path in (PREPARE, POST_WU, FINALIZE):
        source = _source(path)
        for forbidden in forbidden_runtime_tail_dispatch:
            assert forbidden not in source
        assert "HeadRange<false>" in source
        assert "HeadRange<true>" in source


def test_aic_and_aiv_consume_identical_cursor_order():
    for path, aiv_phase, aic_phase, ownership in ENUMERATION_PAIRS:
        aiv_body = _function_body(path, aiv_phase)
        aic_body = _function_body(path, aic_phase)
        for token in ENUMERATION_TOKENS[ownership]:
            assert token in aiv_body
            assert token in aic_body


def test_stage_dispatch_keeps_full_phase_before_tail_phase():
    for path, entry_points in ENTRY_POINTS.items():
        for entry_point in entry_points:
            body = _function_body(path, entry_point)
            if "Full" not in body:
                continue
            g1_full = body.index("ProcessG1Full")
            g1_tail = body.index("ProcessG1Tail")
            grouped_full = body.index("ProcessGroupedFull")
            grouped_tail = body.index("ProcessGroupedTail")
            assert g1_full < g1_tail
            assert grouped_full < grouped_tail
