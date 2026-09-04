/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_STRUCT_H
#define PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_STRUCT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "chunk_kda_fwd_prepare_tiling_key.h"

namespace kda_prepare_pseudocode {

enum class Stage : std::uint8_t {
    V0,
    V1,
    C2,
    V3,
    C4,
    C5,
    V6,
    C7,
};

enum class Pipe : std::uint8_t {
    Control,
    Mte2,
    Mte1,
    Vector,
    Cube,
    Fixpipe,
    Mte3,
};

// Named same-core dependencies. They are observable design obligations, not
// claims about a concrete HardEvent spelling in any CANN release.
enum class LocalDependency : std::uint8_t {
    Mte2ToVectorInputs,
    VectorToMte3Outputs,
    Mte2ToMte3SourceFree,
    Mte2ToMte1Inputs,
    CubeToMte1OperandReuse,
    CubeToFixpipeOutput,
    Mte2ToFixpipePayloadReuse,
    FixpipeToMte2Relay,
    Mte2FillToLoadWaw,
};

enum class CoreRole : std::uint8_t {
    Aic,
    Aiv,
    Shared,
};

enum class MemorySpace : std::uint8_t {
    Gm,
    Workspace,
    Ub,
    L1,
    L0, // Symbolic L0C result storage retained for existing trace consumers.
    L0A,
    L0B,
};

enum class SharedArenaUse : std::uint8_t {
    V01,
    V6,
    Count,
};

enum class L0cStageUse : std::uint8_t {
    C2,
    C4,
    C5,
    C7,
    Count,
};

enum class L0OperandUse : std::uint8_t {
    C2Band0,
    C2Band1,
    C2Band2,
    C2Band3,
    C4,
    C5,
    C7,
    Count,
};

// PROPOSED Arch22 adapter channels. Logical SyncPoint values remain distinct;
// an implementation may serialize compatible phases over these bounded
// reverse-aware channels instead of assigning one physical flag per enum.
enum class Arch22ReverseChannel : std::uint8_t {
    AivToAicPhase,
    AicToAivPhase,
    SlotCredit,
    Count,
};

constexpr std::size_t kSharedArenaUseCount =
    static_cast<std::size_t>(SharedArenaUse::Count);
constexpr std::size_t kL0cStageUseCount =
    static_cast<std::size_t>(L0cStageUse::Count);
constexpr std::size_t kL0OperandUseCount =
    static_cast<std::size_t>(L0OperandUse::Count);
constexpr std::size_t kMaxL0OperandBankCount = 2U;
constexpr std::uint32_t kAllGroupLocalHeads =
    std::numeric_limits<std::uint32_t>::max();

enum class WorkspaceRegion : std::uint8_t {
    WholeSlot,
    Context,
    SharedPayload,
};

enum class SyncPoint : std::uint8_t {
    // SlotFree owns the per-HV G/beta/payload parts of a workspace slot. A live
    // owner-slot Qhat/Khat cache is a disjoint subresource governed only by
    // QkCacheFree/Ready. It remains live until C7 has observed V6RhsReady from
    // every mapped HV cache reader and publishes the next free generation.
    SlotFree,
    QkCacheFree,
    LocalBankFree,
    SharedArenaFree,
    L1BankFree,
    L0cBankFree,
    QkCacheReady,
    V0ContextReady,
    V0BetaReady,
    V0ExportDone,
    V1ScoreReady,
    V1MainSourceFree,
    C2ScorePayloadFree,
    C2ScoreL1Free,
    C2RawDstFree,
    C2RawReady,
    V3VcsReady,
    V3LocalSourceFree,
    C4PayloadFree,
    C4TReady,
    C4AkkPrepReady,
    C5AkkReady,
    V6RhsReady,
};

struct ChunkTask {
    std::uint32_t sequenceId = 0;
    std::uint32_t chunkInSequence = 0;
    std::uint32_t globalChunk = 0;
    std::uint32_t validRows = 0;
};

struct HeadTask {
    // headId is the value/gate head (HV). qkHeadId is its grouped Q/K source
    // head (HK), derived from the public HV/HK ratio.
    std::uint32_t headId = 0;
    std::uint32_t qkHeadId = 0;
    std::uint32_t qkCacheSlot = 0;
    std::uint64_t qkCacheGeneration = 0;
    bool qkOwner = true;
    bool qkLastConsumer = true;
    std::uint32_t groupLocalHead = 0;
    std::uint32_t workspaceSlot = 0;
    std::uint64_t workspaceGeneration = 0;
    std::uint32_t aivId = 0;
    std::uint32_t aivLocalSlot = 0;
    std::uint32_t localBankId = 0;
    std::uint64_t localGeneration = 0;
    std::uint32_t sharedArenaId = 0;
    std::array<std::uint64_t, kSharedArenaUseCount> sharedGenerations{};
    std::uint32_t l1BankId = 0;
    std::uint64_t l1Generation = 0;
    std::uint32_t l0cBankId = 0;
    std::uint32_t l0cPairWave = 0;
    std::uint64_t l0cGeneration = 0;
    std::array<std::uint64_t, kL0cStageUseCount> l0cStageGenerations{};
    std::uint32_t l0OperandBankId = 0;
    std::array<std::uint64_t, kL0OperandUseCount>
        l0OperandGenerations{};
    bool active = false;
};

constexpr std::uint64_t SharedGenerationFor(
    const HeadTask &head, SharedArenaUse use) noexcept
{
    return head.sharedGenerations[static_cast<std::size_t>(use)];
}

constexpr std::uint64_t L0cGenerationFor(
    const HeadTask &head, L0cStageUse use,
    Architecture architecture) noexcept
{
    return architecture == Architecture::Arch22
               ? head.l0cStageGenerations[static_cast<std::size_t>(use)]
               : head.l0cGeneration;
}

constexpr std::uint64_t L0OperandGenerationFor(
    const HeadTask &head, L0OperandUse use) noexcept
{
    return head.l0OperandGenerations[static_cast<std::size_t>(use)];
}

constexpr L0OperandUse C2L0OperandUse(std::uint32_t scoreBlock) noexcept
{
    return static_cast<L0OperandUse>(
        static_cast<std::uint32_t>(L0OperandUse::C2Band0) + scoreBlock);
}

struct HeadGroup {
    ChunkTask chunk{};
    std::uint32_t headGroupId = 0;
    std::uint32_t wavefront = 0;
    std::uint32_t activeHeads = 0;
    std::array<std::uint64_t, kArch22PairCount>
        pairCollectiveGenerations{};
    std::array<HeadTask, kHeadsPerGroup> heads{};
};

constexpr bool PairHasActiveHead(const HeadGroup &group,
                                 std::uint32_t pairWave) noexcept
{
    return pairWave < kArch22PairCount &&
           pairWave * kAivPerWorkgroup < group.activeHeads;
}

constexpr std::uint64_t PairCollectiveGenerationFor(
    const HeadGroup &group, std::uint32_t pairWave) noexcept
{
    return group.pairCollectiveGenerations[pairWave];
}

struct WorkItem {
    std::uint64_t ordinal = 0;
    std::uint32_t chunkOrdinal = 0;
    std::uint32_t headGroupOrdinal = 0;
    HeadGroup group{};
};

struct CorePlan {
    std::uint32_t workgroupId = 0;
    std::uint32_t workgroupCount = 0;
    PartitionMode mode = PartitionMode::ChunkOnly;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t primaryWorkItemCount = 0;
    std::uint32_t totalChunks = 0;
    std::uint32_t headGroupCount = 0;
    std::uint32_t headPartitionCount = 0;
    Architecture architecture = Architecture::Arch35;
};

struct OwnerTicketState {
    std::array<std::uint64_t, kMaxWorkspaceSlotCount> workspaceNext{};
    std::array<std::uint64_t, kHeadsPerGroup> localNext{};
    std::array<std::uint64_t, kAivPerWorkgroup> sharedNext{};
    std::array<std::uint64_t, kArch22PairCount> collectiveNext{};
    std::array<std::uint64_t, kHeadsPerGroup> l1Next{};
    std::array<std::uint64_t, kHeadsPerGroup> l0cNext{};
    std::array<std::uint64_t, kMaxL0OperandBankCount> l0OperandNext{};
    std::array<std::uint64_t, kMaxWorkspaceSlotCount> qkCacheNext{};
    std::array<std::uint64_t, kMaxWorkspaceSlotCount>
        qkCacheCurrentGeneration{};
    std::array<std::uint64_t, kMaxWorkspaceSlotCount> qkCacheCurrentKey{};
    std::array<bool, kMaxWorkspaceSlotCount> qkCacheValid{};
};

struct BufferSpan {
    const char *name = nullptr;
    MemorySpace space = MemorySpace::Workspace;
    std::uint64_t byteOffset = 0;
    // Contiguous enclosing byte range. For a strided matrix this is
    // ((rows - 1) * leadingDimension + columns) * elementBytes, not merely
    // rows * columns * elementBytes.
    std::size_t byteSize = 0;
    std::uint32_t slot = 0;
    std::uint64_t generation = 0;
    CoreRole ownerRole = CoreRole::Shared;
    std::uint32_t ownerId = 0;
    // Optional logical matrix view. Zero values mean that the span is a
    // byte-range only. Strided transfers must populate all four fields so row
    // stride and the logical rectangle are not lost.
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t leadingDimension = 0;
    std::uint32_t elementBytes = 0;
    // Logical tensor head used by the symbolic GM descriptor. It is separate
    // from memory ownership: raw q/k use HK, while gate/value/outputs use HV.
    std::uint32_t logicalHeadId = kAllGroupLocalHeads;
};

// A shared host-only clock makes operation and synchronization records
// comparable without introducing allocation or runtime behavior in the
// default (untraced) pseudocode path.
struct TraceClock {
    std::uint64_t next = 1U;

    std::uint64_t Tick() noexcept
    {
        return next++;
    }
};

struct WorkspaceView {
    // Total bytes owned by the launch. The symbolic entry rejects an
    // undersized backing allocation before deriving any per-workgroup span.
    std::uint64_t backingBytes = 0;
    std::uint64_t workgroupBase = 0;
    std::size_t slotStrideBytes = 0;
    std::size_t contextOffsetBytes = 0;
    std::size_t contextBytes = 0;
    std::size_t payloadOffsetBytes = 0;
    std::size_t payloadBytes = 0;

    BufferSpan Span(WorkspaceRegion region, std::uint32_t slot,
                    std::uint64_t generation) const noexcept
    {
        const std::uint64_t slotBase =
            workgroupBase + static_cast<std::uint64_t>(slot) * slotStrideBytes;
        switch (region) {
            case WorkspaceRegion::Context:
                return {"context", MemorySpace::Workspace,
                        slotBase + contextOffsetBytes, contextBytes, slot, generation};
            case WorkspaceRegion::SharedPayload:
                return {"shared-payload", MemorySpace::Workspace,
                        slotBase + payloadOffsetBytes, payloadBytes, slot, generation};
            case WorkspaceRegion::WholeSlot:
            default:
                return {"slot", MemorySpace::Workspace, slotBase, slotStrideBytes,
                        slot, generation};
        }
    }
};

enum class SyncAction : std::uint8_t {
    Wait,
    Set,
    AivArrive,
    AicWait,
    AicPublish,
    AivWait,
};

struct SyncRecord {
    SyncAction action = SyncAction::Wait;
    SyncPoint point = SyncPoint::SlotFree;
    // A regular event records its owner slot; a pair event records pairWave.
    std::uint32_t ownerId = 0;
    std::uint64_t generation = 0;
    std::uint32_t aivId = std::numeric_limits<std::uint32_t>::max();
    bool hasActiveHead = true;
    Stage stage = Stage::V0;
    Pipe pipe = Pipe::Control;
    std::uint64_t order = 0U;
};

struct SyncTrace {
    static constexpr std::size_t kCapacity = 2048U;
    std::array<SyncRecord, kCapacity> records{};
    std::size_t size = 0;
    bool overflow = false;

    void Push(SyncAction action, SyncPoint point, std::uint32_t ownerId,
              std::uint64_t generation, std::uint32_t aivId,
              bool hasActiveHead, Stage stage, Pipe pipe,
              std::uint64_t order) noexcept
    {
        if (size == records.size()) {
            overflow = true;
            return;
        }
        records[size++] = {action, point, ownerId, generation, aivId,
                           hasActiveHead, stage, pipe, order};
    }
};

struct LocalSyncRecord {
    LocalDependency dependency = LocalDependency::Mte2ToVectorInputs;
    Stage stage = Stage::V0;
    std::uint64_t order = 0U;
};

struct LocalSyncTrace {
    static constexpr std::size_t kCapacity = 2048U;
    std::array<LocalSyncRecord, kCapacity> records{};
    std::size_t size = 0U;
    bool overflow = false;

    void Push(LocalDependency dependency, Stage stage,
              std::uint64_t order) noexcept
    {
        if (size == records.size()) {
            overflow = true;
            return;
        }
        records[size++] = {dependency, stage, order};
    }
};

enum class OperationKind : std::uint8_t {
    Load,
    RunVf,
    Store,
    Zero,
    ZeroUndefined,
    SetHf32Mode,
    Mmad,
    MmadQuadrantPackedLhs,
    Fill,
    StoreRounded,
};

enum class RuntimeScaleUse : std::uint8_t {
    None,
    Aqk,
    FusedQg,
};

struct OperationRecord {
    OperationKind kind = OperationKind::Load;
    Stage stage = Stage::V0;
    BufferSpan source{};
    BufferSpan auxiliary{};
    BufferSpan destination{};
    BufferSpan l0aOperand{};
    BufferSpan l0bOperand{};
    MatrixStorage lhsStorage = MatrixStorage::Fp32;
    MatrixStorage rhsStorage = MatrixStorage::Fp32;
    InputStorage inputStorage = InputStorage::Bf16;
    Pow2Primitive pow2Primitive = Pow2Primitive::ExpLn2;
    std::uint32_t m = 0U;
    std::uint32_t n = 0U;
    std::uint32_t k = 0U;
    std::uint32_t quadrantRows = 0U;
    std::uint32_t quadrantColumns = 0U;
    std::uint32_t value = 0U;
    std::uint32_t headId = 0U;
    std::uint32_t runtimeScaleMultiplyCount = 0U;
    float runtimeScale = 1.0F;
    RuntimeScaleUse runtimeScaleUse = RuntimeScaleUse::None;
    bool transposeRhs = false;
    bool negate = false;
    bool enabled = false;
    bool hasPow2Primitive = false;
    bool hasRuntimeScale = false;
    std::uint64_t order = 0U;
};

struct OperationTrace {
    static constexpr std::size_t kCapacity = 2048U;
    std::array<OperationRecord, kCapacity> records{};
    std::size_t size = 0U;
    bool overflow = false;

    void Push(const OperationRecord &record) noexcept
    {
        if (size == records.size()) {
            overflow = true;
            return;
        }
        records[size++] = record;
    }
};

// PROPOSED: these methods describe ready/free ownership only. They do not map
// to a chosen CrossCore flag API, flag ID, counter depth, or HardEvent. The
// optional trace exists only for the host contract test. It records both
// ordinary owner tickets and pair collectives so the test can prove ordering,
// dummy participation, generations, and ready/free closure. QkCacheReady is
// the sole level-triggered exception: one owner publishes a generation into
// the workspace control page and every mapped HV may acquire it without
// consuming the state. QkCacheFree remains a single next-generation credit.
struct SyncLedger {
    SyncTrace *trace = nullptr;
    LocalSyncTrace *localTrace = nullptr;
    TraceClock *clock = nullptr;

    std::uint64_t NextOrder() const noexcept
    {
        return clock == nullptr ? 0U : clock->Tick();
    }

    void Local(LocalDependency dependency, Stage stage) const noexcept
    {
        if (localTrace != nullptr) {
            localTrace->Push(dependency, stage, NextOrder());
        }
    }

    void Wait(SyncPoint point, std::uint32_t slot, std::uint64_t generation,
              Stage consumer, Pipe consumerPipe) const noexcept
    {
        if (trace != nullptr) {
            trace->Push(SyncAction::Wait, point, slot, generation,
                        std::numeric_limits<std::uint32_t>::max(), true,
                        consumer, consumerPipe, NextOrder());
        }
    }

    void Set(SyncPoint point, std::uint32_t slot, std::uint64_t generation,
             Stage producer, Pipe producerPipe) const noexcept
    {
        if (trace != nullptr) {
            trace->Push(SyncAction::Set, point, slot, generation,
                        std::numeric_limits<std::uint32_t>::max(), true,
                        producer, producerPipe, NextOrder());
        }
    }

    void Join(SyncPoint output, SyncPoint lhs, SyncPoint rhs,
              std::uint32_t slot, std::uint64_t generation, Stage consumer,
              Pipe consumerPipe) const noexcept
    {
        Wait(lhs, slot, generation, consumer, consumerPipe);
        Wait(rhs, slot, generation, consumer, consumerPipe);
        Set(output, slot, generation, consumer, consumerPipe);
    }

    void Join(SyncPoint output, SyncPoint first, SyncPoint second,
              SyncPoint third, std::uint32_t slot, std::uint64_t generation,
              Stage consumer, Pipe consumerPipe) const noexcept
    {
        Wait(first, slot, generation, consumer, consumerPipe);
        Wait(second, slot, generation, consumer, consumerPipe);
        Wait(third, slot, generation, consumer, consumerPipe);
        Set(output, slot, generation, consumer, consumerPipe);
    }

    // PROPOSED Arch22 mode-0x2 adapter. Each AIV arrives exactly once per
    // active pair; hasActiveHead=false is the mandatory dummy participant.
    // The AIC waits/publishes once per pair, and both AIVs consume a publish.
    void AivArrivePair(SyncPoint point, std::uint32_t pairWave,
                       std::uint64_t generation, std::uint32_t aivId,
                       bool hasActiveHead, Stage producer,
                       Pipe producerPipe) const noexcept
    {
        if (trace != nullptr) {
            trace->Push(SyncAction::AivArrive, point, pairWave, generation,
                        aivId, hasActiveHead, producer, producerPipe,
                        NextOrder());
        }
    }

    void AicWaitPair(SyncPoint point, std::uint32_t pairWave,
                     std::uint64_t generation, Stage consumer,
                     Pipe consumerPipe) const noexcept
    {
        if (trace != nullptr) {
            trace->Push(
                SyncAction::AicWait, point, pairWave, generation,
                std::numeric_limits<std::uint32_t>::max(), true, consumer,
                consumerPipe, NextOrder());
        }
    }

    void AicPublishPair(SyncPoint point, std::uint32_t pairWave,
                        std::uint64_t generation, Stage producer,
                        Pipe producerPipe) const noexcept
    {
        if (trace != nullptr) {
            trace->Push(
                SyncAction::AicPublish, point, pairWave, generation,
                std::numeric_limits<std::uint32_t>::max(), true, producer,
                producerPipe, NextOrder());
        }
    }

    void AivWaitPair(SyncPoint point, std::uint32_t pairWave,
                     std::uint64_t generation, std::uint32_t aivId,
                     Stage consumer, Pipe consumerPipe) const noexcept
    {
        if (trace != nullptr) {
            trace->Push(SyncAction::AivWait, point, pairWave, generation,
                        aivId, true, consumer, consumerPipe, NextOrder());
        }
    }
};

// Symbolic no-ops. Their names state dataflow intent and are not claims about
// an Ascend C API declaration, overload, memory position, or synchronization.
struct VectorOps {
    OperationTrace *trace = nullptr;
    TraceClock *clock = nullptr;

    std::uint64_t NextOrder() const noexcept
    {
        return clock == nullptr ? 0U : clock->Tick();
    }

    void Load(Stage stage, const BufferSpan &source,
              const BufferSpan &destination) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::Load;
            record.stage = stage;
            record.source = source;
            record.destination = destination;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void RunVf(Stage stage, const HeadTask &task) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::RunVf;
            record.stage = stage;
            record.headId = task.headId;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void RunVf(Stage stage, const HeadTask &task,
               Pow2Primitive pow2Primitive) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::RunVf;
            record.stage = stage;
            record.headId = task.headId;
            record.pow2Primitive = pow2Primitive;
            record.hasPow2Primitive = true;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void RunVf(Stage stage, const HeadTask &task, float runtimeScale,
               RuntimeScaleUse runtimeScaleUse,
               std::uint32_t runtimeScaleMultiplyCount) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::RunVf;
            record.stage = stage;
            record.headId = task.headId;
            record.runtimeScale = runtimeScale;
            record.runtimeScaleUse = runtimeScaleUse;
            record.runtimeScaleMultiplyCount = runtimeScaleMultiplyCount;
            record.hasRuntimeScale = true;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void RunVf(Stage stage, const HeadTask &task,
               Pow2Primitive pow2Primitive, float runtimeScale,
               RuntimeScaleUse runtimeScaleUse,
               std::uint32_t runtimeScaleMultiplyCount) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::RunVf;
            record.stage = stage;
            record.headId = task.headId;
            record.pow2Primitive = pow2Primitive;
            record.hasPow2Primitive = true;
            record.runtimeScale = runtimeScale;
            record.runtimeScaleUse = runtimeScaleUse;
            record.runtimeScaleMultiplyCount = runtimeScaleMultiplyCount;
            record.hasRuntimeScale = true;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void Store(Stage stage, const BufferSpan &source,
               const BufferSpan &destination) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::Store;
            record.stage = stage;
            record.source = source;
            record.destination = destination;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void Zero(Stage stage, const BufferSpan &destination) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::Zero;
            record.stage = stage;
            record.destination = destination;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void ZeroUndefined(Stage stage, const BufferSpan &span) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::ZeroUndefined;
            record.stage = stage;
            record.destination = span;
            record.order = NextOrder();
            trace->Push(record);
        }
    }
};

struct CubeOps {
    OperationTrace *trace = nullptr;
    TraceClock *clock = nullptr;

    std::uint64_t NextOrder() const noexcept
    {
        return clock == nullptr ? 0U : clock->Tick();
    }

    // PROPOSED symbolic mode boundary. Arch22 C4/C5 must explicitly disable
    // HF32 before their FP32 MMADs; this is not a frozen Ascend C API call.
    void SetHf32Mode(Stage stage, bool enabled) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::SetHf32Mode;
            record.stage = stage;
            record.enabled = enabled;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void Load(Stage stage, const BufferSpan &source,
              const BufferSpan &destination) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::Load;
            record.stage = stage;
            record.source = source;
            record.destination = destination;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    // The L1 source descriptors and their concrete L0A/L0B operand spans are
    // both explicit. This makes shared-vs-disjoint operand residency and the
    // subsequent release edge auditable without claiming a concrete MTE1 API.
    // The symbolic MMAD accumulator and its L0C output are always FP32.
    void Mmad(Stage stage, const BufferSpan &lhs, const BufferSpan &rhs,
              const BufferSpan &output, const BufferSpan &l0aOperand,
              const BufferSpan &l0bOperand, MatrixStorage lhsStorage,
              MatrixStorage rhsStorage, std::uint32_t m,
              std::uint32_t n, std::uint32_t k,
              bool transposeRhs = false, bool negate = false) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::Mmad;
            record.stage = stage;
            record.source = lhs;
            record.auxiliary = rhs;
            record.destination = output;
            record.l0aOperand = l0aOperand;
            record.l0bOperand = l0bOperand;
            record.headId = l0aOperand.logicalHeadId;
            record.lhsStorage = lhsStorage;
            record.rhsStorage = rhsStorage;
            record.m = m;
            record.n = n;
            record.k = k;
            record.transposeRhs = transposeRhs;
            record.negate = negate;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void MmadQuadrantPackedLhs(
        Stage stage, const BufferSpan &lhs, const BufferSpan &rhs,
        const BufferSpan &output, const BufferSpan &l0aOperand,
        const BufferSpan &l0bOperand, MatrixStorage lhsStorage,
        MatrixStorage rhsStorage, std::uint32_t quadrantRows,
        std::uint32_t quadrantColumns, std::uint32_t n,
        std::uint32_t k) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::MmadQuadrantPackedLhs;
            record.stage = stage;
            record.source = lhs;
            record.auxiliary = rhs;
            record.destination = output;
            record.l0aOperand = l0aOperand;
            record.l0bOperand = l0bOperand;
            record.headId = l0aOperand.logicalHeadId;
            record.lhsStorage = lhsStorage;
            record.rhsStorage = rhsStorage;
            record.quadrantRows = quadrantRows;
            record.quadrantColumns = quadrantColumns;
            record.n = n;
            record.k = k;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void Fill(Stage stage, const BufferSpan &destination,
              std::uint32_t value = 0) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::Fill;
            record.stage = stage;
            record.destination = destination;
            record.value = value;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void Store(Stage stage, const BufferSpan &source,
               const BufferSpan &destination) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::Store;
            record.stage = stage;
            record.source = source;
            record.destination = destination;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    // PROPOSED Fixpipe conversion: FP16 means saturate to +/-65504 then RINT;
    // BF16 means RINT without a finite-magnitude saturation.
    void StoreRounded(Stage stage, const BufferSpan &source,
                      const BufferSpan &destination,
                      InputStorage storage) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::StoreRounded;
            record.stage = stage;
            record.source = source;
            record.destination = destination;
            record.inputStorage = storage;
            record.order = NextOrder();
            trace->Push(record);
        }
    }
};

struct VectorStageArgs {
    const WorkItem *work = nullptr;
    WorkspaceView *workspace = nullptr;
    SyncLedger *sync = nullptr;
    VectorOps *ops = nullptr;
    Architecture architecture = Architecture::Arch35;
    ProposedTilingKey key{};
    float epsilon = 0.0F;
    float lowerBound = 0.0F;
    float scale = 1.0F;
    bool hasDtBias = false;
    std::uint32_t workgroupId = 0;
    std::uint32_t aivId = 0;
    std::uint32_t selectedGroupLocalHead = kAllGroupLocalHeads;
};

struct CubeStageArgs {
    const WorkItem *work = nullptr;
    WorkspaceView *workspace = nullptr;
    SyncLedger *sync = nullptr;
    CubeOps *ops = nullptr;
    Architecture architecture = Architecture::Arch35;
    ProposedTilingKey key{};
    float epsilon = 0.0F;
    float lowerBound = 0.0F;
    float scale = 1.0F;
    std::uint32_t workgroupId = 0;
};

} // namespace kda_prepare_pseudocode

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_STRUCT_H
