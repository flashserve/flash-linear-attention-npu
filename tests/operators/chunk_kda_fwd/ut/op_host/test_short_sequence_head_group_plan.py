"""Host-only contracts for short-sequence chunk/head-group scheduling."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[5]
KERNEL_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel"
TILING_SOURCE = (
    ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp"
)
PLAN_HEADER = KERNEL_ROOT / "chunk_kda_fwd_plan.h"


def _head_window_count(query_head_count: int, value_head_count: int) -> int:
    if (
        not query_head_count
        or value_head_count < query_head_count
        or value_head_count % query_head_count
    ):
        return 0
    head_ratio = value_head_count // query_head_count
    if head_ratio <= 4:
        heads_per_window = (4 // head_ratio) * head_ratio
        return (
            value_head_count // heads_per_window
            + int(value_head_count % heads_per_window != 0)
        )
    windows_per_query = head_ratio // 4 + int(head_ratio % 4 != 0)
    return query_head_count * windows_per_query


def _head_window_begin(
    window_ordinal: int, query_head_count: int, value_head_count: int
) -> int:
    window_count = _head_window_count(query_head_count, value_head_count)
    if not window_count:
        return 0
    if window_ordinal >= window_count:
        return value_head_count if window_ordinal == window_count else 0
    head_ratio = value_head_count // query_head_count
    if head_ratio <= 4:
        return window_ordinal * (4 // head_ratio) * head_ratio
    windows_per_query = head_ratio // 4 + int(head_ratio % 4 != 0)
    query_head, local_window = divmod(window_ordinal, windows_per_query)
    return query_head * head_ratio + local_window * 4


def _head_window_head_count(
    hv_base: int, query_head_count: int, value_head_count: int
) -> int:
    if (
        hv_base >= value_head_count
        or not _head_window_count(query_head_count, value_head_count)
    ):
        return 0
    head_ratio = value_head_count // query_head_count
    if head_ratio <= 4:
        head_count = (4 // head_ratio) * head_ratio
    else:
        query_head_end = (hv_base // head_ratio + 1) * head_ratio
        head_count = min(4, query_head_end - hv_base)
    return min(head_count, value_head_count - hv_base)


def _head_windows(query_head_count: int, value_head_count: int):
    windows = []
    hv_base = 0
    while hv_base < value_head_count:
        head_count = _head_window_head_count(
            hv_base, query_head_count, value_head_count
        )
        assert head_count > 0
        windows.append((hv_base, hv_base + head_count))
        hv_base += head_count
    return windows


def _resident_query_loads(query_head_count: int, value_head_count: int):
    head_ratio = value_head_count // query_head_count
    loads = []
    for head_begin, head_end in _head_windows(
        query_head_count, value_head_count
    ):
        window_loads = []
        for value_head in range(head_begin, head_end):
            query_head = value_head // head_ratio
            if not window_loads or window_loads[-1] != query_head:
                window_loads.append(query_head)
        loads.append(window_loads)
    return loads


def _group_count(
    full_chunk_count: int,
    tail_chunk_count: int,
    physical_core_count: int,
    query_head_count: int,
    value_head_count: int,
) -> int:
    grouping_chunk_count = full_chunk_count + tail_chunk_count
    window_count = _head_window_count(query_head_count, value_head_count)
    if not grouping_chunk_count or not physical_core_count or not window_count:
        return 0
    if grouping_chunk_count >= physical_core_count:
        return 1
    return min(
        window_count,
        (physical_core_count + grouping_chunk_count - 1)
        // grouping_chunk_count,
    )


def _head_range(
    group: int,
    group_count: int,
    query_head_count: int,
    value_head_count: int,
):
    window_count = _head_window_count(query_head_count, value_head_count)
    window_begin = group * window_count // group_count
    window_end = (group + 1) * window_count // group_count
    return (
        _head_window_begin(window_begin, query_head_count, value_head_count),
        _head_window_begin(window_end, query_head_count, value_head_count),
    )


def _build_cursors(
    full_chunk_count: int,
    tail_chunk_count: int,
    group_count: int,
    physical_core_count: int,
):
    full_task_count = full_chunk_count * group_count
    tail_task_count = tail_chunk_count * group_count
    total_task_count = full_task_count + tail_task_count
    active_core_count = min(physical_core_count, total_task_count)
    cursors = []
    for core in range(active_core_count):
        task_begin = total_task_count * core // active_core_count
        task_end = total_task_count * (core + 1) // active_core_count
        full_begin = min(task_begin, full_task_count)
        full_end = min(task_end, full_task_count)
        tail_begin = min(
            max(task_begin - full_task_count, 0), tail_task_count
        )
        tail_end = min(
            max(task_end - full_task_count, 0), tail_task_count
        )
        cursors.append((full_begin, full_end, tail_begin, tail_end))
    return cursors


def _owned_head_workloads(
    cursors, group_count, query_head_count, value_head_count
):
    workloads = []
    for full_begin, full_end, tail_begin, tail_end in cursors:
        workload = 0
        for task in list(range(full_begin, full_end)) + list(
            range(tail_begin, tail_end)
        ):
            group = task % group_count
            head_begin, head_end = _head_range(
                group, group_count, query_head_count, value_head_count
            )
            workload += head_end - head_begin
        workloads.append(workload)
    return workloads


@pytest.mark.parametrize(
    ("full_chunk_count", "expected_group_count", "expected_active"),
    [(1, 24, 24), (16, 2, 32), (31, 2, 32), (32, 1, 32)],
)
def test_h96_short_sequence_group_count(
    full_chunk_count, expected_group_count, expected_active
):
    group_count = _group_count(full_chunk_count, 0, 32, 96, 96)
    assert group_count == expected_group_count
    assert min(32, full_chunk_count * group_count) == expected_active


def test_runtime_head_windows_cover_all_supported_integer_gva_ratios():
    ratios_seen = set()
    multi_query_ratios_seen = set()

    # 穷举能力范围内所有合法H/HV组合，而不是只验证几个常见比例。
    for query_head_count in range(1, 129):
        for head_ratio in range(1, 128 // query_head_count + 1):
            value_head_count = query_head_count * head_ratio
            shape = (query_head_count, value_head_count, head_ratio)
            ratios_seen.add(head_ratio)
            if query_head_count > 1:
                multi_query_ratios_seen.add(head_ratio)

            windows = _head_windows(query_head_count, value_head_count)
            assert windows, shape
            assert _head_window_count(
                query_head_count, value_head_count
            ) == len(windows), shape
            assert (
                windows[0][0] == 0
                and windows[-1][1] == value_head_count
            ), shape
            assert all(
                left_end == right_begin
                for (_, left_end), (right_begin, _) in zip(windows, windows[1:])
            ), shape
            assert [
                head
                for head_begin, head_end in windows
                for head in range(head_begin, head_end)
            ] == list(range(value_head_count)), shape
            assert all(
                0 < head_end - head_begin <= 4
                for head_begin, head_end in windows
            ), shape

            if head_ratio <= 4:
                heads_per_window = (4 // head_ratio) * head_ratio
                assert all(
                    head_begin % head_ratio == 0
                    and head_end % head_ratio == 0
                    and head_end - head_begin
                    == min(heads_per_window, value_head_count - head_begin)
                    for head_begin, head_end in windows
                ), shape
            else:
                assert all(
                    head_begin // head_ratio == (head_end - 1) // head_ratio
                    and head_begin % head_ratio % 4 == 0
                    and head_end - head_begin
                    == min(
                        4,
                        (head_begin // head_ratio + 1) * head_ratio - head_begin,
                    )
                    for head_begin, head_end in windows
                ), shape

            loads = _resident_query_loads(
                query_head_count, value_head_count
            )
            flattened_loads = [head for window in loads for head in window]
            loads_per_query = 1 if head_ratio <= 4 else (head_ratio + 3) // 4
            assert (
                len(flattened_loads)
                == query_head_count * loads_per_query
            ), shape
            assert flattened_loads == [
                query_head
                for query_head in range(query_head_count)
                for _ in range(loads_per_query)
            ], shape

            window_boundaries = {windows[0][0]}
            window_boundaries.update(head_end for _, head_end in windows)
            for group_count in range(1, len(windows) + 1):
                ranges = [
                    _head_range(
                        group,
                        group_count,
                        query_head_count,
                        value_head_count,
                    )
                    for group in range(group_count)
                ]
                expected_ranges = []
                for group in range(group_count):
                    window_begin = group * len(windows) // group_count
                    window_end = (group + 1) * len(windows) // group_count
                    expected_ranges.append(
                        (
                            windows[window_begin][0],
                            windows[window_end - 1][1],
                        )
                    )
                assert (
                    ranges[0][0] == 0
                    and ranges[-1][1] == value_head_count
                ), shape
                assert all(
                    head_begin < head_end
                    for head_begin, head_end in ranges
                ), shape
                assert ranges == expected_ranges, shape
                assert all(
                    head_begin in window_boundaries
                    and head_end in window_boundaries
                    for head_begin, head_end in ranges
                ), shape
                assert all(
                    left_end == right_begin
                    for (_, left_end), (right_begin, _) in zip(
                        ranges, ranges[1:]
                    )
                ), shape

                # kernel按真实headCnt推进时必须恰好消费owner拥有的窗口段。
                for head_begin, head_end in ranges:
                    head = head_begin
                    while head < head_end:
                        head_count = _head_window_head_count(
                            head,
                            query_head_count,
                            value_head_count,
                        )
                        assert head_count > 0, shape
                        assert head + head_count <= head_end, shape
                        head += head_count
                    assert head == head_end, shape

    assert ratios_seen == set(range(1, 129))
    assert set(range(1, 65)).issubset(multi_query_ratios_seen)


def test_ratio_7_and_19_keep_single_query_remainders_explicitly():
    # 两个非4整除比例单列，便于检视时直接看到真实尾窗和重复读次数。
    assert _head_windows(1, 7) == [(0, 4), (4, 7)]
    assert _resident_query_loads(1, 7) == [[0], [0]]
    assert _head_windows(1, 19) == [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 19),
    ]
    assert _resident_query_loads(1, 19) == [[0], [0], [0], [0], [0]]


@pytest.mark.parametrize(
    ("full_chunk_count", "tail_chunk_count", "group_count"),
    [(1, 1, 4), (3, 5, 4), (15, 1, 2), (16, 0, 2)],
)
def test_combined_cursor_has_balanced_nonempty_owners_and_phase_coverage(
    full_chunk_count, tail_chunk_count, group_count
):
    cursors = _build_cursors(
        full_chunk_count, tail_chunk_count, group_count, 32
    )
    full_tasks = Counter()
    tail_tasks = Counter()
    owner_loads = []
    for full_begin, full_end, tail_begin, tail_end in cursors:
        full_tasks.update(range(full_begin, full_end))
        tail_tasks.update(range(tail_begin, tail_end))
        owner_loads.append(
            (full_end - full_begin) + (tail_end - tail_begin)
        )

    assert full_tasks == Counter(
        range(full_chunk_count * group_count)
    )
    assert tail_tasks == Counter(
        range(tail_chunk_count * group_count)
    )
    assert min(owner_loads) >= 1
    assert max(owner_loads) - min(owner_loads) <= 1


def test_one_full_one_tail_uses_all_eight_available_tasks():
    cursors = _build_cursors(1, 1, 4, 32)
    assert len(cursors) == 8
    assert all(
        (full_end - full_begin) + (tail_end - tail_begin) == 1
        for full_begin, full_end, tail_begin, tail_end in cursors
    )


def test_one_query_with_many_values_uses_four_head_windows():
    full_chunk_count = 1
    group_count = _group_count(full_chunk_count, 0, 32, 1, 128)
    assert group_count == 32
    assert len(_build_cursors(1, 0, group_count, 32)) == 32


def test_t1024_h96_doubles_available_chunk_stage_cores():
    full_chunk_count = 1024 // 64
    group_count = _group_count(full_chunk_count, 0, 32, 96, 96)
    assert group_count == 2
    assert min(32, full_chunk_count * group_count) == 32


@pytest.mark.parametrize(
    (
        "full_chunk_count",
        "tail_chunk_count",
        "expected_group_count",
    ),
    [
        (31, 32, 1),
        (31, 0, 2),
        (0, 16, 2),
        (16, 16, 1),
    ],
)
def test_full_and_tail_tasks_jointly_determine_short_stream_grouping(
    full_chunk_count, tail_chunk_count, expected_group_count
):
    group_count = _group_count(
        full_chunk_count, tail_chunk_count, 32, 96, 96
    )
    assert group_count == expected_group_count
    assert min(
        32,
        (full_chunk_count + tail_chunk_count) * group_count,
    ) == 32


@pytest.mark.parametrize(
    (
        "physical_core_count",
        "full_chunk_count",
        "tail_chunk_count",
        "expected_group_count",
    ),
    [
        (7, 2, 1, 3),
        (13, 5, 1, 3),
        (16, 7, 1, 2),
        (24, 11, 1, 2),
        (40, 31, 32, 1),
        (48, 31, 0, 2),
        (64, 31, 32, 2),
    ],
)
def test_grouping_uses_runtime_physical_core_count(
    physical_core_count,
    full_chunk_count,
    tail_chunk_count,
    expected_group_count,
):
    group_count = _group_count(
        full_chunk_count,
        tail_chunk_count,
        physical_core_count,
        96,
        96,
    )
    assert group_count == expected_group_count
    cursors = _build_cursors(
        full_chunk_count,
        tail_chunk_count,
        group_count,
        physical_core_count,
    )
    assert len(cursors) == min(
        physical_core_count,
        (full_chunk_count + tail_chunk_count) * group_count,
    )
    owner_loads = [
        (full_end - full_begin) + (tail_end - tail_begin)
        for full_begin, full_end, tail_begin, tail_end in cursors
    ]
    assert max(owner_loads) - min(owner_loads) <= 1


def test_runtime_p32_h96_three_chunks_keeps_window_aligned_boundaries():
    physical_core_count = 32
    chunk_count = 3
    value_head_count = 96
    group_count = _group_count(
        chunk_count,
        0,
        physical_core_count,
        value_head_count,
        value_head_count,
    )
    ranges = [
        _head_range(
            group, group_count, value_head_count, value_head_count
        )
        for group in range(group_count)
    ]

    assert group_count == 11
    assert all(
        head_begin % 4 == 0 and head_end % 4 == 0
        for head_begin, head_end in ranges
    )
    covered_heads = [
        head
        for head_begin, head_end in ranges
        for head in range(head_begin, head_end)
    ]
    assert covered_heads == list(range(value_head_count))
    cursors = _build_cursors(
        chunk_count, 0, group_count, physical_core_count
    )
    assert len(cursors) == physical_core_count
    assert sum(
        full_end - full_begin
        for full_begin, full_end, _, _ in cursors
    ) == chunk_count * group_count


@pytest.mark.parametrize(
    ("full_chunk_count", "tail_chunk_count", "expected_group_count"),
    [(0, 1, 24), (0, 16, 2), (0, 32, 1)],
)
def test_all_tail_stream_uses_the_same_logical_chunk_rule(
    full_chunk_count, tail_chunk_count, expected_group_count
):
    assert (
        _group_count(full_chunk_count, tail_chunk_count, 32, 96, 96)
        == expected_group_count
    )


@pytest.mark.parametrize(
    (
        "physical_core_count",
        "value_head_count",
        "expected_owned_heads",
    ),
    [
        (32, 32, {4}),
        (32, 64, {4}),
        (32, 96, {4}),
        (32, 128, {4}),
        (16, 96, {4, 8}),
    ],
)
def test_each_core_actual_owned_head_workload_covers_pipeline_widths(
    physical_core_count, value_head_count, expected_owned_heads
):
    group_count = _group_count(
        1,
        0,
        physical_core_count,
        value_head_count,
        value_head_count,
    )
    cursors = _build_cursors(
        1, 0, group_count, physical_core_count
    )
    assert set(
        _owned_head_workloads(
            cursors, group_count, value_head_count, value_head_count
        )
    ) == expected_owned_heads


def test_unequal_task_counts_and_full_tail_phase_remainder_preserve_work():
    full_chunk_count = 3
    tail_chunk_count = 4
    group_count = 3
    value_head_count = 10
    cursors = _build_cursors(
        full_chunk_count, tail_chunk_count, group_count, 8
    )
    task_loads = [
        (full_end - full_begin) + (tail_end - tail_begin)
        for full_begin, full_end, tail_begin, tail_end in cursors
    ]
    workloads = _owned_head_workloads(
        cursors, group_count, value_head_count, value_head_count
    )

    assert max(task_loads) - min(task_loads) == 1
    assert any(
        full_begin < full_end and tail_begin < tail_end
        for full_begin, full_end, tail_begin, tail_end in cursors
    )
    assert sum(workloads) == (
        full_chunk_count + tail_chunk_count
    ) * value_head_count
    assert min(workloads) > 0


def test_host_plan_uses_combined_owner_range_and_real_active_count():
    source = TILING_SOURCE.read_text(encoding="utf-8")
    assert (
        "sequenceInfo.totalFullChunks, sequenceInfo.totalTailChunks"
        in source
    )
    assert "logicalChunkCount * headGroupCount" in source
    assert "std::min<uint64_t>(blockDim, groupedTaskCount)" in source
    assert "groupedTaskCount * core / header.chunkUsedCoreNum" in source
    assert "taskBegin > fullTaskCount ? taskBegin - fullTaskCount : 0" in source


def test_compact_plan_serialization_checks_before_uint32_narrowing():
    source = TILING_SOURCE.read_text(encoding="utf-8")

    assert "bool AlignPlanOffset(uint64_t offset" in source
    assert "bool AppendPlanVector(" in source
    assert "uint32_t &encodedOffset" in source
    assert "CheckedMul(static_cast<uint64_t>(values.size())" in source
    assert "CheckedAdd(offset, vectorBytes, payloadEnd)" in source
    assert "payloadEnd > std::numeric_limits<uint32_t>::max()" in source
    assert "payloadEnd > static_cast<uint64_t>(payload.max_size())" in source
    assert "encodedOffset = static_cast<uint32_t>(offset);" in source
    assert source.index("payloadEnd > std::numeric_limits<uint32_t>::max()") < source.index(
        "encodedOffset = static_cast<uint32_t>(offset);"
    )
    assert "static_cast<uint32_t>(payload.size())," not in source
    assert "offset + values.size() * sizeof(T)" not in source


def test_host_plan_gets_core_count_from_platform_instead_of_a_literal():
    source = TILING_SOURCE.read_text(encoding="utf-8")
    assert "platform.GetCoreNumAic()" in source
    assert "sequenceInfo, blockDim" in source
    assert "context->SetBlockDim(blockDim)" in source
    assert not re.search(r"\bblockDim\s*=\s*32\b", source)


@pytest.mark.parametrize(
    (
        "physical_core_count",
        "logical_chunk_count",
        "expected_group_count",
    ),
    [
        (24, 1, 24),
        (32, 1, 24),
        (40, 1, 24),
        (48, 1, 24),
        (32, 3, 11),
        (48, 3, 16),
        (32, 16, 2),
        (40, 16, 3),
        (32, 31, 2),
        (64, 31, 3),
    ],
)
def test_dynamic_core_count_caps_groups_by_runtime_windows(
    physical_core_count,
    logical_chunk_count,
    expected_group_count,
):
    group_count = _group_count(
        logical_chunk_count, 0, physical_core_count, 96, 96
    )
    assert group_count == expected_group_count
    assert group_count <= _head_window_count(96, 96)
    ranges = [
        _head_range(group, group_count, 96, 96)
        for group in range(group_count)
    ]
    assert all(
        head_begin % 4 == 0 and head_end % 4 == 0
        for head_begin, head_end in ranges
    )


def test_internal_plan_keeps_stage_flags_reserved_and_zero():
    tiling = TILING_SOURCE.read_text(encoding="utf-8")
    plan = PLAN_HEADER.read_text(encoding="utf-8")

    assert "KDA_COMPACT_PLAN_VERSION = 6" in plan
    assert "uint32_t fullStartSequence;" in plan
    assert "uint32_t fullStartLocalChunk;" in plan
    assert "cursor.fullStartSequence = LoadU32(" in plan
    assert "cursor.fullStartLocalChunk = LoadU32(" in plan
    assert "sizeof(CompactSequencePlanHeader) == 21 * sizeof(uint32_t)" in plan
    assert "sizeof(ChunkCoreCursor) == 6 * sizeof(uint32_t)" in plan
    assert "offsetof(ChunkCoreCursor, tailBegin)" in plan
    assert "uint32_t chunkStageFlags;" in plan
    assert "HeadWindowCount(" in plan
    assert "HeadWindowBegin(" in plan
    assert "HeadWindowHeadCount(" in plan
    assert "KDA_CHUNK_STAGE_HEAD_GROUPS_PAIR_ALIGNED" not in plan
    assert "HeadGroupsPairAligned" not in plan
    assert "header.chunkStageFlags = 0;" in tiling
    assert "header.chunkStageFlags |=" not in tiling


def test_all_chunk_independent_stages_decode_group_tasks():
    sources = [
        KERNEL_ROOT / "chunk_kda_fwd_prepare.h",
        KERNEL_ROOT / "chunk_kda_fwd_post_wu.h",
        KERNEL_ROOT / "chunk_kda_fwd_finalize.h",
        KERNEL_ROOT / "arch35/chunk_kda_fwd_prepare.h",
        KERNEL_ROOT / "arch35/chunk_kda_fwd_finalize.h",
    ]
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        assert "DecodeChunkHeadGroupTask" in source
        assert "ComputeChunkHeadGroupCount" not in source
        assert "cursor.fullBegin" in source
        assert "cursor.tailBegin" in source


def test_kernel_only_consumes_host_computed_head_group_policy():
    for source_path in KERNEL_ROOT.rglob("*"):
        if source_path.suffix not in {".h", ".hpp", ".cpp"}:
            continue
        if source_path.name == "chunk_kda_fwd_plan.h":
            continue
        source = source_path.read_text(encoding="utf-8")
        assert "ComputeChunkHeadGroupCount" not in source


def test_head_state_does_not_consume_chunk_head_group_cursors():
    common_root = KERNEL_ROOT / "common"
    sources = [
        common_root / "chunk_kda_head_state.h",
        common_root / "chunk_kda_head_state_arch35.h",
    ]
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        assert "HeadGroupCount" not in source
        assert "DecodeChunkHeadGroupTask" not in source
