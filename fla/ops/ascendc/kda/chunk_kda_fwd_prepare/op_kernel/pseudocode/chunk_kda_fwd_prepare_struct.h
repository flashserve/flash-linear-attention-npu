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

// 具名的同核依赖。这些是可观测的语义约束：Arch22 由成对 HardEvent 落地，
// Arch35 则由下方独立的 Mutex 轨迹证明具体流水和物理槽生命周期。
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
    L0, // 为现有轨迹消费者保留的符号化 L0C 结果存储。
    L0A,
    L0B,
};

enum class NativeMatrixLayout : std::uint8_t {
    Unspecified,
    L0aZN,
    L0aZZ,
    L0cFractal,
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

// 待实现的 Arch22 适配器通道。逻辑 SyncPoint 值保持互异；实现可以在这些有界、
// 支持反向确认的通道上串行化兼容阶段，而不必为每个枚举值分配一个物理标志。
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
    // SlotFree 管理工作区槽中每个 HV 的 G/beta/载荷区域。存活的所有者槽
    // Qhat/Khat 缓存是独立子资源，仅由 QkCacheFree/QkCacheReady 管理。它持续存活，直到 C7
    // 观察到每个映射 HV 缓存读取者的 V6RhsReady，并发布下一空闲代际。
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
    // headId 表示值/门控头（HV）。qkHeadId 表示根据公开 HV/HK 比例推导出的
    // 分组 Q/K 来源头（HK）。
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
    // 连续包络字节范围。对于跨步矩阵，该值为
    // ((rows - 1) * leadingDimension + columns) * elementBytes，
    // 而不只是 rows * columns * elementBytes。
    std::size_t byteSize = 0;
    std::uint32_t slot = 0;
    std::uint64_t generation = 0;
    CoreRole ownerRole = CoreRole::Shared;
    std::uint32_t ownerId = 0;
    // 可选的逻辑矩阵视图。值为零表示该范围只描述字节区间。
    // 跨步搬运必须填写全部四个字段，避免丢失行步长和逻辑矩形信息。
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t leadingDimension = 0;
    std::uint32_t elementBytes = 0;
    // 符号化 GM 描述符使用的逻辑张量头。它与内存所有权相互独立：
    // 原始 q/k 使用 HK，门控/值/输出使用 HV。
    std::uint32_t logicalHeadId = kAllGroupLocalHeads;
    // 原生 L0 分形布局的逻辑视图。byteOffset/byteSize 始终保留完整、连续的
    // 所属分配区；逻辑子块通过父矩阵形状、起点和 rows/columns 描述，
    // 不能按行主序公式把子块起点换算成字节偏移。
    NativeMatrixLayout nativeLayout = NativeMatrixLayout::Unspecified;
    std::uint32_t parentRows = 0U;
    std::uint32_t parentColumns = 0U;
    std::uint32_t logicalRowOffset = 0U;
    std::uint32_t logicalColumnOffset = 0U;
};

inline BufferSpan NativeMatrixOwner(
    const BufferSpan &storage, const char *name, NativeMatrixLayout layout,
    std::uint32_t rows, std::uint32_t columns,
    std::uint32_t elementBytes) noexcept
{
    BufferSpan owner = storage;
    owner.name = name;
    owner.rows = rows;
    owner.columns = columns;
    owner.leadingDimension = 0U;
    owner.elementBytes = elementBytes;
    owner.nativeLayout = layout;
    owner.parentRows = rows;
    owner.parentColumns = columns;
    owner.logicalRowOffset = 0U;
    owner.logicalColumnOffset = 0U;
    return owner;
}

inline BufferSpan NativeMatrixTile(
    const BufferSpan &owner, const char *name, std::uint32_t row,
    std::uint32_t column, std::uint32_t rows,
    std::uint32_t columns) noexcept
{
    BufferSpan tile = owner;
    tile.name = name;
    tile.rows = rows;
    tile.columns = columns;
    tile.logicalRowOffset = row;
    tile.logicalColumnOffset = column;
    return tile;
}

// 共享的仅主机时钟使操作记录与同步记录可比较，
// 同时不在默认的无轨迹伪代码路径中引入内存分配或运行时行为。
struct TraceClock {
    std::uint64_t next = 1U;

    std::uint64_t Tick() noexcept
    {
        return next++;
    }
};

struct WorkspaceView {
    // 本次启动拥有的总字节数。符号化入口在推导任何每个工作组的内存范围前，
    // 会拒绝容量不足的后备分配空间。
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
    // 普通事件记录其所有者槽；配对事件记录 pairWave。
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

// Mutex 只管理 Arch35 同一个 AI Core 内的异步流水，不能替代 AIC/AIV 之间的
// 就绪/空闲票据。本方案采用静态 Tensor/静态 UB-L1 地址范式，因此 MutexID
// 由编译期表管理；按官方约束只使用建议的 0..27，不占用系统预留的 28..31。
using SymbolicMutexId = std::uint8_t;
constexpr SymbolicMutexId kInvalidMutexId =
    std::numeric_limits<SymbolicMutexId>::max();
constexpr SymbolicMutexId kArch35StaticMutexIdLimit = 28U;

enum class MutexAction : std::uint8_t {
    Lock,
    Unlock,
    PipeBarrier,
};

enum class MutexResource : std::uint8_t {
    AivUbBank,
    AicL1Bank,
    AicL0OperandBank,
    AicL0cLowerHalf,
    AicL0cUpperHalf,
    Mte2Overwrite,
};

struct MutexRecord {
    MutexAction action = MutexAction::Lock;
    MutexResource resource = MutexResource::AivUbBank;
    SymbolicMutexId mutexId = kInvalidMutexId;
    std::uint32_t ownerId = 0U;
    Stage stage = Stage::V0;
    Pipe pipe = Pipe::Control;
    std::uint64_t order = 0U;
};

struct MutexTrace {
    static constexpr std::size_t kCapacity = 4096U;
    std::array<MutexRecord, kCapacity> records{};
    std::size_t size = 0U;
    bool overflow = false;

    void Push(MutexAction action, MutexResource resource,
              SymbolicMutexId mutexId, std::uint32_t ownerId, Stage stage,
              Pipe pipe, std::uint64_t order) noexcept
    {
        if (size == records.size()) {
            overflow = true;
            return;
        }
        records[size++] = {action, resource, mutexId, ownerId,
                           stage, pipe, order};
    }
};

struct Arch35VectorMutexIds {
    static constexpr SymbolicMutexId UbBank(std::uint32_t localSlot) noexcept
    {
        return static_cast<SymbolicMutexId>(localSlot);
    }

    static constexpr SymbolicMutexId kCount = kHeadsPerAiv;
};

struct Arch35CubeMutexIds {
    static constexpr SymbolicMutexId L1Bank(std::uint32_t bank) noexcept
    {
        return static_cast<SymbolicMutexId>(bank);
    }

    static constexpr SymbolicMutexId L0OperandBank(
        std::uint32_t bank) noexcept
    {
        return static_cast<SymbolicMutexId>(kHeadsPerGroup + bank);
    }

    static constexpr SymbolicMutexId L0cLowerHalf(
        std::uint32_t bank) noexcept
    {
        return static_cast<SymbolicMutexId>(kHeadsPerGroup + 1U + bank);
    }

    static constexpr SymbolicMutexId L0cUpperHalf(
        std::uint32_t bank) noexcept
    {
        return static_cast<SymbolicMutexId>(2U * kHeadsPerGroup + 1U + bank);
    }

    static constexpr SymbolicMutexId kCount =
        3U * kHeadsPerGroup + 1U;
};

enum class OperationKind : std::uint8_t {
    Load,
    RunVf,
    Store,
    Zero,
    ZeroUndefined,
    SetHf32Mode,
    Mmad,
    MmadRowStackedLhs,
    MmadQuadrantPackedLhs,
    Fill,
    StoreRounded,
};

enum class RuntimeScaleUse : std::uint8_t {
    None,
    Aqk,
    FusedQg,
};

// 一次 VF 调用与其数学体的静态绑定。Stage 代码必须选择其中一个具名公式，
// 不能再用只有 Stage 编号的通用 RunVf 代替实际计算。
enum class VectorFormula : std::uint8_t {
    None,
    QkNormGateCumsumBeta,
    S4ScoreOperands,
    AqkAndAkkFactors,
    PostWuOperands,
};

// 每次逻辑 MMAD 对应的矩阵公式。布局、M/N/K 和物理打包方式仍由同一条
// OperationRecord 记录，公式名只用于消除通用 Mmad 调用的语义歧义。
enum class MatrixFormula : std::uint8_t {
    None,
    RawAqkAndAkk,
    TEqualsBMatmulX0,
    AkkLowerLeftEqualsNegX1MatmulT,
    WEqualsAkkMatmulKBetaG,
    UEqualsAkkMatmulVBeta,
};

struct OperationRecord {
    OperationKind kind = OperationKind::Load;
    Stage stage = Stage::V0;
    BufferSpan source{};
    BufferSpan secondarySource{};
    BufferSpan rhsOperand{};
    BufferSpan destination{};
    BufferSpan l0aOperand{};
    BufferSpan lhsTopL0aTile{};
    BufferSpan lhsBottomL0aTile{};
    BufferSpan l0bOperand{};
    MatrixStorage lhsStorage = MatrixStorage::Fp32;
    MatrixStorage rhsStorage = MatrixStorage::Fp32;
    InputStorage inputStorage = InputStorage::Bf16;
    Pow2Primitive pow2Primitive = Pow2Primitive::ExpLn2;
    VectorFormula vectorFormula = VectorFormula::None;
    MatrixFormula matrixFormula = MatrixFormula::None;
    std::uint32_t m = 0U;
    std::uint32_t n = 0U;
    std::uint32_t k = 0U;
    std::uint32_t lhsTopRows = 0U;
    std::uint32_t lhsBottomRows = 0U;
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

// 公式合同向底层轨迹接口提交的元数据。数学体、Stage 和公式枚举由架构内
// 同一个合同类型绑定；VectorOps 只负责记录，不伪装成公式执行入口。
struct VectorVfTraceMetadata {
    Stage stage = Stage::V0;
    VectorFormula formula = VectorFormula::None;
    Pow2Primitive pow2Primitive = Pow2Primitive::ExpLn2;
    RuntimeScaleUse runtimeScaleUse = RuntimeScaleUse::None;
    std::uint32_t runtimeScaleMultiplyCount = 0U;
    float runtimeScale = 1.0F;
    bool hasPow2Primitive = false;
    bool hasRuntimeScale = false;
};

// 待实现：这些方法只描述就绪/空闲所有权，不映射到选定的跨核标志 API、
// 标志 ID、计数器深度或 HardEvent。可选轨迹仅供主机合同测试使用，同时记录
// 普通所有者票据和配对集合操作，使测试可以证明执行顺序、空参与者、代际以及
// 就绪/空闲闭环。QkCacheReady 是唯一的电平式例外：一个所有者将代际发布到
// 工作区控制页，每个映射的 HV 都可以获取该状态而不消费它。
// QkCacheFree 仍只保留一个下一代许可。
struct SyncLedger {
    SyncTrace *trace = nullptr;
    LocalSyncTrace *localTrace = nullptr;
    TraceClock *clock = nullptr;
    MutexTrace *mutexTrace = nullptr;

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

    void MutexLock(MutexResource resource, SymbolicMutexId mutexId,
                   std::uint32_t ownerId, Stage stage, Pipe pipe) const noexcept
    {
        if (mutexTrace != nullptr) {
            mutexTrace->Push(MutexAction::Lock, resource, mutexId, ownerId,
                             stage, pipe, NextOrder());
        }
    }

    void MutexUnlock(MutexResource resource, SymbolicMutexId mutexId,
                     std::uint32_t ownerId, Stage stage,
                     Pipe pipe) const noexcept
    {
        if (mutexTrace != nullptr) {
            mutexTrace->Push(MutexAction::Unlock, resource, mutexId, ownerId,
                             stage, pipe, NextOrder());
        }
    }

    void PipeBarrier(MutexResource resource, std::uint32_t ownerId,
                     Stage stage, Pipe pipe) const noexcept
    {
        if (mutexTrace != nullptr) {
            mutexTrace->Push(MutexAction::PipeBarrier, resource,
                             kInvalidMutexId, ownerId, stage, pipe,
                             NextOrder());
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

    // 待实现的 Arch22 0x2 模式适配器。每个 AIV 对每个有效配对恰好到达一次；
    // hasActiveHead=false 表示必须参加的空参与者。AIC 对每个配对执行一次等待和发布，
    // 两个 AIV 都消费同一次发布。
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

// 符号化空操作。其名称只表达数据流意图，不表示 Ascend C API 的声明、重载、
// 内存位置或同步语义。
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

    void RecordOneVf(const HeadTask &task,
                     const VectorVfTraceMetadata &metadata) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::RunVf;
            record.stage = metadata.stage;
            record.headId = task.headId;
            record.vectorFormula = metadata.formula;
            record.pow2Primitive = metadata.pow2Primitive;
            record.hasPow2Primitive = metadata.hasPow2Primitive;
            record.runtimeScale = metadata.runtimeScale;
            record.runtimeScaleUse = metadata.runtimeScaleUse;
            record.runtimeScaleMultiplyCount =
                metadata.runtimeScaleMultiplyCount;
            record.hasRuntimeScale = metadata.hasRuntimeScale;
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

    // 待实现的符号化模式边界。Arch22 C4/C5 必须在 FP32 MMAD 前显式关闭 HF32；
    // 这不是已冻结的 Ascend C API 调用。
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

    // L1 源描述符及其具体 L0A/L0B 操作数范围都显式记录。这样无需声明具体
    // MTE1 API，也能审计操作数驻留区是共享还是独立，以及后续释放依赖边。
    // 符号化 MMAD 累加器及其 L0C 输出始终为 FP32。
    void Mmad(Stage stage, MatrixFormula formula, const BufferSpan &lhs,
              const BufferSpan &rhs,
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
            record.matrixFormula = formula;
            record.source = lhs;
            record.rhsOperand = rhs;
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

    // 将两个独立的 L1 行平面分别装入同一 L0A 所属存储的上下逻辑行子块，
    // 再用一次 MMAD 共享同一份 L0B。该记录显式保留两个 L1 源及两个原生
    // L0A 子块，避免把分形地址伪装成连续半区；紧凑拼装的 L0C 输出仍为 FP32。
    void MmadRowStackedLhs(
        Stage stage, MatrixFormula formula, const BufferSpan &lhsTop,
        const BufferSpan &lhsBottom,
        const BufferSpan &rhs, const BufferSpan &output,
        const BufferSpan &l0aOperand, const BufferSpan &lhsTopL0aTile,
        const BufferSpan &lhsBottomL0aTile, const BufferSpan &l0bOperand,
        MatrixStorage lhsStorage, MatrixStorage rhsStorage,
        std::uint32_t lhsTopRows, std::uint32_t lhsBottomRows,
        std::uint32_t n, std::uint32_t k, bool transposeRhs = false) const noexcept
    {
        if (trace != nullptr) {
            OperationRecord record{};
            record.kind = OperationKind::MmadRowStackedLhs;
            record.stage = stage;
            record.matrixFormula = formula;
            record.source = lhsTop;
            record.secondarySource = lhsBottom;
            record.rhsOperand = rhs;
            record.destination = output;
            record.l0aOperand = l0aOperand;
            record.lhsTopL0aTile = lhsTopL0aTile;
            record.lhsBottomL0aTile = lhsBottomL0aTile;
            record.l0bOperand = l0bOperand;
            record.headId = l0aOperand.logicalHeadId;
            record.lhsStorage = lhsStorage;
            record.rhsStorage = rhsStorage;
            record.m = lhsTopRows + lhsBottomRows;
            record.n = n;
            record.k = k;
            record.lhsTopRows = lhsTopRows;
            record.lhsBottomRows = lhsBottomRows;
            record.transposeRhs = transposeRhs;
            record.order = NextOrder();
            trace->Push(record);
        }
    }

    void MmadQuadrantPackedLhs(
        Stage stage, MatrixFormula formula, const BufferSpan &lhs,
        const BufferSpan &rhs,
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
            record.matrixFormula = formula;
            record.source = lhs;
            record.rhsOperand = rhs;
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

    // 待实现的 Fixpipe 转换：FP16 先饱和到 +/-65504，再执行 RINT；
    // BF16 执行 RINT，但不做有限幅值饱和。
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
    ProposedTilingKey key{};
    float epsilon = 0.0F;
    float lowerBound = 0.0F;
    float scale = 1.0F;
    std::uint32_t workgroupId = 0;
};

} // namespace kda_prepare_pseudocode

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_STRUCT_H
