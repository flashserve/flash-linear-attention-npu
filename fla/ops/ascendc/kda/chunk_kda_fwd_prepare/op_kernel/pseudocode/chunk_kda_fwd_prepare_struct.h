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
    L0,
};

enum class WorkspaceRegion : std::uint8_t {
    WholeSlot,
    Context,
    SharedPayload,
};

enum class SyncPoint : std::uint8_t {
    SlotFree,
    LocalBankFree,
    L1BankFree,
    L0cBankFree,
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
    std::uint32_t headId = 0;
    std::uint32_t groupLocalHead = 0;
    std::uint32_t workspaceSlot = 0;
    std::uint64_t workspaceGeneration = 0;
    std::uint32_t aivId = 0;
    std::uint32_t aivLocalSlot = 0;
    std::uint32_t localBankId = 0;
    std::uint64_t localGeneration = 0;
    std::uint32_t l1BankId = 0;
    std::uint64_t l1Generation = 0;
    std::uint32_t l0cBankId = 0;
    std::uint64_t l0cGeneration = 0;
    bool active = false;
};

struct HeadGroup {
    ChunkTask chunk{};
    std::uint32_t headGroupId = 0;
    std::uint32_t wavefront = 0;
    std::uint32_t activeHeads = 0;
    std::array<HeadTask, kHeadsPerGroup> heads{};
};

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
};

struct OwnerTicketState {
    std::array<std::uint64_t, kWorkspaceSlotCount> workspaceNext{};
    std::array<std::uint64_t, kHeadsPerGroup> localNext{};
    std::array<std::uint64_t, kHeadsPerGroup> l1Next{};
    std::array<std::uint64_t, kHeadsPerGroup> l0cNext{};
};

struct BufferSpan {
    const char *name = nullptr;
    MemorySpace space = MemorySpace::Workspace;
    std::uint64_t byteOffset = 0;
    std::size_t byteSize = 0;
    std::uint32_t slot = 0;
    std::uint64_t generation = 0;
    CoreRole ownerRole = CoreRole::Shared;
    std::uint32_t ownerId = 0;
};

struct WorkspaceView {
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

// PROPOSED: these methods describe ready/free ownership only. They do not map
// to a chosen CrossCore flag API, flag ID, counter depth, or HardEvent.
struct SyncLedger {
    void Wait(SyncPoint point, std::uint32_t slot, std::uint64_t generation,
              Stage consumer, Pipe consumerPipe) const noexcept
    {
        (void)point;
        (void)slot;
        (void)generation;
        (void)consumer;
        (void)consumerPipe;
    }

    void Set(SyncPoint point, std::uint32_t slot, std::uint64_t generation,
             Stage producer, Pipe producerPipe) const noexcept
    {
        (void)point;
        (void)slot;
        (void)generation;
        (void)producer;
        (void)producerPipe;
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
};

// Symbolic no-ops. Their names state dataflow intent and are not claims about
// an Ascend C API declaration, overload, memory position, or synchronization.
struct VectorOps {
    void Load(Stage stage, const BufferSpan &source,
              const BufferSpan &destination) const noexcept
    {
        (void)stage;
        (void)source;
        (void)destination;
    }

    void RunVf(Stage stage, const HeadTask &task) const noexcept
    {
        (void)stage;
        (void)task;
    }

    void Store(Stage stage, const BufferSpan &source,
               const BufferSpan &destination) const noexcept
    {
        (void)stage;
        (void)source;
        (void)destination;
    }

    void Zero(Stage stage, const BufferSpan &destination) const noexcept
    {
        (void)stage;
        (void)destination;
    }

    void ZeroUndefined(Stage stage, const BufferSpan &span) const noexcept
    {
        (void)stage;
        (void)span;
    }
};

struct CubeOps {
    void Load(Stage stage, const BufferSpan &source,
              const BufferSpan &destination) const noexcept
    {
        (void)stage;
        (void)source;
        (void)destination;
    }

    // lhsStorage/rhsStorage make the Cube interpretation auditable. The
    // symbolic MMAD accumulator and its L0C output are always FP32.
    void Mmad(Stage stage, const BufferSpan &lhs, const BufferSpan &rhs,
              const BufferSpan &output, MatrixStorage lhsStorage,
              MatrixStorage rhsStorage, std::uint32_t m, std::uint32_t n,
              std::uint32_t k, bool transposeRhs = false,
              bool negate = false) const noexcept
    {
        (void)stage;
        (void)lhs;
        (void)rhs;
        (void)output;
        (void)lhsStorage;
        (void)rhsStorage;
        (void)m;
        (void)n;
        (void)k;
        (void)transposeRhs;
        (void)negate;
    }

    void MmadQuadrantPackedLhs(
        Stage stage, const BufferSpan &lhs, const BufferSpan &rhs,
        const BufferSpan &output, MatrixStorage lhsStorage,
        MatrixStorage rhsStorage, std::uint32_t quadrantRows,
        std::uint32_t quadrantColumns, std::uint32_t n,
        std::uint32_t k) const noexcept
    {
        (void)stage;
        (void)lhs;
        (void)rhs;
        (void)output;
        (void)lhsStorage;
        (void)rhsStorage;
        (void)quadrantRows;
        (void)quadrantColumns;
        (void)n;
        (void)k;
    }

    void Fill(Stage stage, const BufferSpan &destination,
              std::uint32_t value = 0) const noexcept
    {
        (void)stage;
        (void)destination;
        (void)value;
    }

    void Store(Stage stage, const BufferSpan &source,
               const BufferSpan &destination) const noexcept
    {
        (void)stage;
        (void)source;
        (void)destination;
    }

    // PROPOSED Fixpipe conversion: FP16 means saturate to +/-65504 then RINT;
    // BF16 means RINT without a finite-magnitude saturation.
    void StoreRounded(Stage stage, const BufferSpan &source,
                      const BufferSpan &destination,
                      InputStorage storage) const noexcept
    {
        (void)stage;
        (void)source;
        (void)destination;
        (void)storage;
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
