"""Static contracts for the A2/A3 compact-plan FwdH scheduler."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd"
COMMON = OP_ROOT / "op_kernel/chunk_kda_fwd_common.h"
PLAN = OP_ROOT / "op_kernel/chunk_kda_fwd_plan.h"
GDN_ROOT = (
    ROOT
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
    "chunk_gated_delta_rule_fwd_h/op_kernel/gemm"
)
SCHEDULER = GDN_ROOT / "block/block_scheduler_gdn_fwd_h.hpp"
KERNEL = GDN_ROOT / "kernel/gdn_fwd_h_kernel.hpp"
HOST_TILING = OP_ROOT / "op_host/chunk_kda_fwd_tiling.cpp"


UINT32_MAX = (1 << 32) - 1


def _affine_wave_tasks(heads, physical_cores, sequences):
    used_cores = min(heads, physical_cores)
    max_heads = (heads + used_cores - 1) // used_cores
    tasks = {core: [] for core in range(physical_cores)}
    for core in range(physical_cores):
        head_begin = core * heads // used_cores if core < used_cores else 0
        head_end = (core + 1) * heads // used_cores if core < used_cores else 0
        for wave in range(sequences * max_heads):
            sequence = wave // max_heads
            head = head_begin + wave % max_heads
            if head < head_end:
                tasks[core].append((sequence, head))
    return tasks, sequences * max_heads


def _resolve_affine_wave_task(heads, physical_cores, sequences, core, wave):
    used_cores = min(heads, physical_cores)
    max_heads = (heads + used_cores - 1) // used_cores
    if core >= used_cores:
        return None
    head_begin = core * heads // used_cores
    head_end = (core + 1) * heads // used_cores
    sequence = wave // max_heads
    head = head_begin + wave % max_heads
    if sequence >= sequences or head >= head_end:
        return None
    return sequence * heads + head


def test_affine_head_ranges_cover_every_sequence_head_once():
    for heads, cores, sequences in ((96, 32, 4), (5, 3, 6), (3, 8, 5)):
        tasks, wave_count = _affine_wave_tasks(heads, cores, sequences)
        assert wave_count > 0
        flattened = [task for core_tasks in tasks.values() for task in core_tasks]
        assert len(flattened) == sequences * heads
        assert len(set(flattened)) == sequences * heads
        assert set(flattened) == {
            (sequence, head)
            for sequence in range(sequences)
            for head in range(heads)
        }
        for core_tasks in tasks.values():
            assert core_tasks == sorted(core_tasks)


def test_head_ranges_are_affine_runtime_state_not_plan_payload():
    plan = PLAN.read_text(encoding="utf-8")
    header = plan.split("struct CompactSequencePlanHeader", 1)[1].split("};", 1)[0]
    assert "fwdUsedCoreNum" in header
    assert "headBegin" not in header
    assert "headEnd" not in header
    assert "headRange" not in header

    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert "cubeCoreIdx) * vNumHead / fwdUsedCoreNum" in scheduler
    assert "cubeCoreIdx + 1) * vNumHead / fwdUsedCoreNum" in scheduler
    assert "return static_cast<uint64_t>(sequenceCount) * maxHeadsPerCore;" in scheduler
    assert "return sequenceIdx * static_cast<uint64_t>(vNumHead) + headIdx;" in scheduler


def test_compact_plan_resolves_original_sequence_without_recompacting():
    common = COMMON.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert "plan.SequenceCount()" in common
    assert "plan.FwdUsedCoreNum()" in common
    assert "plan.SequenceChunkOffsetsOffset()" in common
    assert "ResolveCompactSequence(compactBatchIdx, stream);" in scheduler
    assert "gmSeqlen.GetValue(sequenceIdx)" in scheduler
    assert "gmSeqlen.GetValue(sequenceIdx + 1)" in scheduler
    assert "gmSeqChunkOffsets.GetValue(sequenceIdx)" in scheduler
    assert "gmSeqChunkOffsets.GetValue(sequenceIdx + 1)" in scheduler
    assert "cachedSequenceIdx == sequenceIdx" in scheduler


def test_zero_length_sequence_keeps_its_original_state_slot():
    kernel = KERNEL.read_text(encoding="utf-8")
    assert "const bool hasChunks = initialStream.batchChunks != 0;" in kernel
    assert "if (!hasChunks)" in kernel
    assert "(static_cast<uint64_t>(batchIdx) * vNumHead + vHeadIdx)" in kernel
    assert "gmFinalState[finalStateOffset]" in kernel
    assert "gmInitialState[finalStateOffset]" in kernel


def test_large_affine_task_and_wave_indices_do_not_wrap_at_uint32():
    sequences = UINT32_MAX
    heads = 96
    physical_cores = 32
    max_heads = (heads + physical_cores - 1) // physical_cores

    task_count = sequences * heads
    wave_count = sequences * max_heads
    assert task_count > UINT32_MAX
    assert wave_count > UINT32_MAX

    last_wave = wave_count - 1
    assert _resolve_affine_wave_task(
        heads, physical_cores, sequences, physical_cores - 1, last_wave
    ) == task_count - 1

    dense_last_task = task_count - 1
    dense_wave = dense_last_task // physical_cores
    dense_core = dense_last_task % physical_cores
    assert dense_wave * physical_cores + dense_core == dense_last_task


def test_large_token_and_address_offsets_do_not_wrap_at_uint32():
    token_offset = UINT32_MAX + 64
    total_tokens = token_offset + 8192
    v_heads = 96
    v_head = 95
    v_dim = 128
    chunk = 127
    chunk_size = 64

    chunk_token_offset = token_offset + chunk * chunk_size
    token_slot = v_head * total_tokens + chunk_token_offset
    uv_offset = token_slot * v_dim
    state_offset = (UINT32_MAX * v_heads + v_head) * 128 * v_dim
    assert chunk_token_offset > UINT32_MAX
    assert uv_offset > UINT32_MAX
    assert state_offset > UINT32_MAX


def test_scheduler_and_kernel_keep_64_bit_internal_index_contract():
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")

    for field in (
        "hSrcOffset",
        "hDstOffset",
        "uvOffset",
        "wkOffset",
        "wOffset",
        "gOffset",
        "gkOffset",
        "hWorkOffset",
        "vWorkOffset",
        "kDecayWorkOffset",
        "initialStateOffset",
        "finalStateOffset",
    ):
        assert f"uint64_t {field};" in scheduler
    for field in (
        "seqlen",
        "numSeqWorkspaceOffset",
        "numChunksWorkspaceOffset",
        "taskIdx",
        "taskStride",
        "taskNum",
        "totalTokens",
    ):
        assert f"uint64_t {field};" in scheduler
    assert "uint64_t tokenOffset;" in scheduler
    assert "uint64_t batchTokens;" in scheduler
    assert "uint64_t nextTaskIdx{0};" in scheduler
    assert "uint64_t cachedTokenOffset{0};" in scheduler
    assert "uint64_t cachedBatchTokens{0};" in scheduler
    assert "uint64_t seqChunkOffsetsOffset{0};" in scheduler
    assert "uint64_t ResolveWaveTask(uint64_t waveIdx) const" in scheduler
    assert "void InitTaskWave(uint64_t waveIdx)" in scheduler
    assert "uint64_t GetTaskWaveCount() const" in scheduler
    assert "bool ResolveWaveStream(uint64_t waveIdx" in scheduler
    assert "taskNum = static_cast<uint64_t>(batch) * vNumHead;" in scheduler
    assert "static_cast<uint64_t>(gmSeqlen.GetValue(sequenceIdx))" in scheduler
    assert "static_cast<uint32_t>(gmSeqlen.GetValue" not in scheduler
    assert "static_cast<uint64_t>(stream.shapeBatchIdx)" in scheduler
    assert "static_cast<uint64_t>(cubeCoreIdx) * PING_PONG_STAGES" in scheduler

    for field in (
        "seqlen",
        "vWorkspaceOffset",
        "vUpdateWorkspaceOffset",
        "hWorkspaceOffset",
        "numSeqWorkspaceOffset",
        "numChunksWorkspaceOffset",
        "kDecayWorkspaceOffset",
    ):
        assert f"uint64_t {field};" in kernel
    for local_offset in (
        "cube1OffsetW",
        "cube1OffsetH",
        "cube1OffsetVwork",
        "cube2OffsetKwork",
        "cube2OffsetVwork",
        "cube2OffsetH",
    ):
        assert f"const uint64_t {local_offset}" in kernel
    assert kernel.count("const uint64_t taskWaveCount") == 2
    assert kernel.count("for (uint64_t waveIdx") == 2
    assert "const uint64_t stateBlockSize" in kernel
    assert "const uint64_t hBaseOffset" in kernel
    assert "const uint64_t stateBaseOffset" in kernel
    assert "const uint64_t hOffset" in kernel
    assert "uint32_t taskWaveCount" not in kernel


def test_dense_batch_does_not_inherit_the_varlen_sequence_limit():
    host = HOST_TILING.read_text(encoding="utf-8")
    assert "sequenceInfo.isVarLen && sequenceInfo.seqNum > 1024" in host
    dense_branch = host.split("if (!info.isVarLen) {", 1)[1].split(
        "info.seqNum = cuTensor", 1
    )[0]
    assert "1024" not in dense_branch
