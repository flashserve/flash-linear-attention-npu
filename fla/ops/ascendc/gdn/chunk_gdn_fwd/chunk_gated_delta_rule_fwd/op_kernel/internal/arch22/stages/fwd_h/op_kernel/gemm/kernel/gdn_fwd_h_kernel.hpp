/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#define CATLASS_ARCH 2201

#include "../../../../../common/qkv_input_layout.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_h.hpp"
#include "../../../../../pipeline/chunk_gated_delta_rule_ho_pipeline.h"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"



#include "kernel_operator.h"
using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

struct GDNFwdHTileShapes128 {
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
};

struct GDNFwdHTileShapes256 {
    using L1TileShape = tla::Shape<_128, _256, _128>;
    using L0TileShape = tla::Shape<_128, _256, _64>;
};

template <bool KGated, bool ScalarGated, bool UseExp2>
struct GDNFwdHGateTag {
    static constexpr bool value = KGated;
    static constexpr bool scalarGated = ScalarGated;
    static constexpr bool useExp2 = UseExp2;
};

template <class BaseCopy>
struct GDNFwdHPaddedGmToL1Copy : BaseCopy {
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src)
    {
        // 先完成同属 MTE2 的 InitConstValue，再覆盖有效的 GM 输入。
        AscendC::PipeBarrier<PIPE_MTE2>();
        BaseCopy::operator()(dst, src);
    }
};

template <class BaseTileCopy>
struct GDNFwdHKVTailTileCopy : BaseTileCopy {
    template <class TensorA>
    using CopyGmToL1A = GDNFwdHPaddedGmToL1Copy<typename BaseTileCopy::template CopyGmToL1A<TensorA>>;

    template <class TensorB>
    using CopyGmToL1B = GDNFwdHPaddedGmToL1Copy<typename BaseTileCopy::template CopyGmToL1B<TensorB>>;
};

template <class BaseMmad>
struct GDNFwdHCurrentAPrefetchMmad : BaseMmad {
    using ArchTag = typename BaseMmad::ArchTag;
    using TileCopy = typename BaseMmad::TileCopy;

    CATLASS_DEVICE
    GDNFwdHCurrentAPrefetchMmad(Arch::Resource<ArchTag> &resource) : BaseMmad(resource) {}

    template <class TensorA>
    CATLASS_DEVICE void PrefetchCurrentA(TensorA &tensorA, GemmCoord const &actualShape)
    {
        // 前序 WH/KV 的有向退休链已释放当前槽，保留 Base 的 A-ring。
        auto tensorL1A = tla::MakeTensor(
            this->l1ATensorList[this->l1AListId], BaseMmad::L1A_LAYOUT, Arch::PositionL1{});
        auto tensorTileA = this->GetTileA(
            tensorA, 0, 0, actualShape.m(), min(actualShape.k(), BaseMmad::L1_TILE_K));
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;
        copyGmToL1A(tensorL1A, tensorTileA);
        // 下次原 operator 的 MTE2_MTE1 覆盖本次 copy，并消费一次 Skip。
        this->SkipNextADataCopy();
    }
};

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE,
    typename TileShapes = GDNFwdHTileShapes128,
    bool kGated = false,
    bool scalarGated = true,
    bool useExp2 = false,
    bool kChunkPipeline = false,
    int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1
>
class GDNFwdHKernel {
public:

    static constexpr uint32_t HO_PIPELINE_CHUNK_SIZE = 64;
    static constexpr uint32_t HO_PIPELINE_VALUE_HEADS = 8;
    static constexpr uint32_t HO_PIPELINE_VALUE_DIM = 128;
    static constexpr uint32_t HO_PIPELINE_CUBE_CORES = 24;
    static constexpr uint32_t HO_PIPELINE_EVENT_COUNT = 2;
    static constexpr uint32_t HO_PIPELINE_SYNC_UB_OFFSET = 188 * 1024;
    static constexpr uint64_t HO_PIPELINE_WORKSPACE_ALIGNMENT = 512;

    using ArchTag = Arch::AtlasA2;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube<kStaticV, kStaticBT>;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHVec<kStaticV, kStaticBT>;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using L1TileShapeVTla = typename TileShapes::L1TileShape;
    using L0TileShapeVTla = typename TileShapes::L0TileShape;

    using WType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;

    // cube 1
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadKVBase = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;
    using BlockMmadKV = GDNFwdHCurrentAPrefetchMmad<BlockMmadKVBase>;
    using TileCopyKVTail = GDNFwdHKVTailTileCopy<TileCopyKV>;
    using BlockMmadKVTail = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKVTail>;

    static constexpr uint32_t W_PREFETCH_SLOT_BYTES = BlockMmadWH::L1A_TILE_SIZE;
    static constexpr uint32_t W_PREFETCH_L1_OFFSET =
        BlockMmadWH::L1A_TILE_SIZE * BlockMmadWH::L1A_STAGES +
        BlockMmadWH::L1B_TILE_SIZE * BlockMmadWH::L1B_STAGES;
    static_assert(W_PREFETCH_L1_OFFSET + PING_PONG_STAGES * W_PREFETCH_SLOT_BYTES <= ArchTag::L1_SIZE,
        "The private W prefetch slots must fit after the shared MMAD L1 scratch.");

    // vec 1
    using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasGDNFwdHVnew;
    using GateTag = GDNFwdHGateTag<kGated, scalarGated, useExp2>;
    using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, VworkType, FinalStateType, GateTag>;

    // vec 2
    using DispatchPolicyGDNFwdHUpdate = Epilogue::EpilogueAtlasGDNFwdHUpdate;
    using EpilogueGDNFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHUpdate, HType, GType, HType, HworkType, FinalStateType, GateTag>;

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;


    uint32_t batch;
    __aicore__ inline bool InputSequenceMajor() const
    {
        if constexpr (kQkvLayout >= 0) {
            return kQkvLayout == 1;
        } else {
            return inputSequenceMajorRuntime_;
        }
    }

    bool inputSequenceMajorRuntime_{false};
    __aicore__ inline void ConfigureInputLayout(bool sequenceMajor)
    {
        if constexpr (kQkvLayout < 0) {
            inputSequenceMajorRuntime_ = sequenceMajor;
        }
    }
    GM_ADDR gLastExpAddr{nullptr};
    GM_ADDR gDecayAddr{nullptr};
    __aicore__ inline void ConfigureGDecay(GM_ADDR addr) { gDecayAddr = addr; }
    uint64_t gLastExpStride{0};
    __aicore__ inline void ConfigureGLastExp(GM_ADDR addr)
    {
        gLastExpAddr = addr;
        gLastExpStride = (static_cast<uint64_t>(vNumHead) + 7) & ~7ULL;
    }
    __aicore__ inline uint64_t RawQkOffset(uint64_t offset) const
    {
        return InputSequenceMajor() ? GDN::QkvSequenceMajorOffset(offset, seqlen, kNumHead, kHeadDim) : offset;
    }
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    __aicore__ inline uint32_t ValueDim() const
    {
        if constexpr (kStaticV != 0) {
            return static_cast<uint32_t>(kStaticV);
        } else {
            return vHeadDimRuntime_;
        }
    }

    uint32_t vHeadDimRuntime_;
    __aicore__ inline uint32_t ChunkSize() const
    {
        if constexpr (kStaticBT != 0) {
            return static_cast<uint32_t>(kStaticBT);
        } else {
            return chunkSizeRuntime_;
        }
    }

    uint32_t chunkSizeRuntime_;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t vUpdateWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t numSeqWorkspaceOffset;
    uint32_t numChunksWorkspaceOffset;
    uint32_t kDecayWorkspaceOffset;

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;

    AscendC::GlobalTensor<ElementG> gmGk;
    AscendC::GlobalTensor<ElementK> gmKDecayWorkspace;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;
    AscendC::GlobalTensor<int32_t> gmPipelineSync;

    bool chunkPipelineEnabled{false};

    GdnHoPipeline::HoPipelineConfig idleCfg{};
    AscendC::GlobalTensor<int32_t> gmHoReady;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> &resource;


    __aicore__ inline uint64_t PipelineVNewBytes() const
    {
        const uint64_t bytes = static_cast<uint64_t>(batch) * vNumHead * seqlen * ValueDim() * sizeof(ElementV);
        return (bytes + HO_PIPELINE_WORKSPACE_ALIGNMENT - 1) / HO_PIPELINE_WORKSPACE_ALIGNMENT *
               HO_PIPELINE_WORKSPACE_ALIGNMENT;
    }

    __aicore__ inline bool CanRunChunkPipeline() const
    {
        if constexpr (!kChunkPipeline) {
            return false;
        }
        return isVariedLen == 0 && batch == 1 && vNumHead == HO_PIPELINE_VALUE_HEADS &&
               ValueDim() == HO_PIPELINE_VALUE_DIM &&
               (ChunkSize() == HO_PIPELINE_CHUNK_SIZE || ChunkSize() == 2 * HO_PIPELINE_CHUNK_SIZE) &&
               AscendC::GetBlockNum() == HO_PIPELINE_CUBE_CORES;
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> GetPipelineSyncLocal()
    {
        return resource.ubBuf.template GetBufferByByte<int32_t>(HO_PIPELINE_SYNC_UB_OFFSET);
    }

    __aicore__ inline uint32_t GetPipelineAivIdx() const
    {
        return vecBlockScheduler.cubeCoreIdx * AscendC::GetSubBlockNum() + AscendC::GetSubBlockIdx();
    }

    __aicore__ inline void InitPipelineSync()
    {
        if (!chunkPipelineEnabled) {
            return;
        }
        auto syncLocal = GetPipelineSyncLocal();
        AscendC::Duplicate(syncLocal, static_cast<int32_t>(0), 8);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t logicalAivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        const uint32_t logicalAivIdx = GetPipelineAivIdx();
        for (uint32_t eventId = 0; eventId < HO_PIPELINE_EVENT_COUNT; ++eventId) {
            const uint32_t offset = logicalAivNum * 8 * eventId + logicalAivIdx * 8;
            AscendC::DataCopy(gmPipelineSync[offset], syncLocal, 8);
        }
    }

    __aicore__ inline void SignalChunkReady(const GDNFwdHOffsets &offsets)
    {
        if (!chunkPipelineEnabled) {
            return;
        }
        const uint32_t eventId = offsets.chunkIdx % HO_PIPELINE_EVENT_COUNT;
        AscendC::IBSet<false>(gmPipelineSync, GetPipelineSyncLocal(), GetPipelineAivIdx(), eventId);
    }

    // 新通知协议（仅 AIV）：清零本 AIV 在全部 bank 的自身 32B 槽；跨核“先清
    // 零、后收发”顺序由随后的原 wave 初始化 SyncAll（含空闲后缀核）建立。
    __aicore__ inline void InitIdlePipelineSlots()
    {
        if (!idleCfg.enabled) {
            return;
        }
        GdnHoPipeline::HoNotifyPipeline(idleCfg, gmHoReady).InitOwnSlots(GetPipelineSyncLocal());
    }

    // 新通知协议（仅 AIV）：V2 共同出口按 (seq, head, vBlock, localChunk) 发
    // 布一次；末块跳过更新的分支同样经过此处发布。
    __aicore__ inline void PublishIdleChunk(uint32_t compactSequence, uint32_t head, uint32_t vBlock,
                                            uint32_t localChunk)
    {
        if (!idleCfg.enabled) {
            return;
        }
        GdnHoPipeline::HoNotifyPipeline(idleCfg, gmHoReady).Publish(
            compactSequence, head, vBlock, localChunk, GetPipelineSyncLocal());
    }


    __aicore__ inline explicit GDNFwdHKernel(Arch::Resource<ArchTag> &sharedResource)
        : resource(sharedResource) {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {

        __gm__ GdnMegaArch22FwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ GdnMegaArch22FwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        if constexpr (kStaticV == 0) {
            vHeadDimRuntime_ = gdnFwdHTilingData->vHeadDim;
        }
        if constexpr (kStaticBT == 0) {
            chunkSizeRuntime_ = gdnFwdHTilingData->chunkSize;
        }
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = gdnFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = gdnFwdHTilingData->hWorkspaceOffset;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;
        kDecayWorkspaceOffset = gdnFwdHTilingData->kDecayWorkspaceOffset;

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));
        gmKDecayWorkspace.SetGlobalBuffer((__gm__ ElementK *)(user + kDecayWorkspaceOffset));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        chunkPipelineEnabled = CanRunChunkPipeline();
        if (chunkPipelineEnabled) {
            gmPipelineSync.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(v_new + PipelineVNewBytes()));
        }

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
            cubeBlockScheduler.ConfigureTaskStreams(!kGated && !chunkPipelineEnabled);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
            vecBlockScheduler.ConfigureTaskStreams(!kGated && !chunkPipelineEnabled);
        }
    }

    // 生命周期：统一 mega 入口在 Init 后、Process 前调用；默认关闭保持原行
    // 为。enabled 时旧 ring 握手退出并绑定 host 独立预留的 GM ready；配置合
    // 法性与全核一致性由入口保证，本核不做局部资格判断。
    __aicore__ inline void ConfigureIdlePipeline(const GdnHoPipeline::HoPipelineConfig &cfg, GM_ADDR readyAddr)
    {
        if (!cfg.enabled) {
            return;
        }
        idleCfg = cfg;
        chunkPipelineEnabled = false;
        gmHoReady.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(readyAddr));
    }

    template <typename TilingData>
    __aicore__ inline void InitFromData(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state,
        GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
        GM_ADDR final_state, const TilingData& tilingData, GM_ADDR user) {
        batch = tilingData.batch;
        seqlen = tilingData.seqlen;
        kNumHead = tilingData.kNumHead;
        vNumHead = tilingData.vNumHead;
        kHeadDim = tilingData.kHeadDim;
        if constexpr (kStaticV == 0) {
            vHeadDimRuntime_ = tilingData.vHeadDim;
        }
        if constexpr (kStaticBT == 0) {
            chunkSizeRuntime_ = tilingData.chunkSize;
        }
        useInitialState = tilingData.useInitialState;
        storeFinalState = tilingData.storeFinalState;
        isVariedLen = tilingData.isVariedLen;
        shapeBatch = tilingData.shapeBatch;
        tokenBatch = tilingData.tokenBatch;
        vWorkspaceOffset = tilingData.vWorkspaceOffset;
        vUpdateWorkspaceOffset = tilingData.vUpdateWorkspaceOffset;
        hWorkspaceOffset = tilingData.hWorkspaceOffset;
        numSeqWorkspaceOffset = tilingData.numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = tilingData.numChunksWorkspaceOffset;
        kDecayWorkspaceOffset = tilingData.kDecayWorkspaceOffset;

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)(scalarGated ? g : gk));
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmGk.SetGlobalBuffer((__gm__ ElementG *)(kGated ? gk : g));
        gmKDecayWorkspace.SetGlobalBuffer((__gm__ ElementK *)(user + kDecayWorkspaceOffset));
        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
            cubeBlockScheduler.ConfigureTaskStreams(!kGated && !chunkPipelineEnabled);
        }
        if ASCEND_IS_AIV {
            vecBlockScheduler.InitFromData(cu_seqlens, chunk_indices, tilingData, user);
            vecBlockScheduler.ConfigureTaskStreams(!kGated && !chunkPipelineEnabled);
        }
    }

    __aicore__ inline void PresetVectorPipelineEvents()
    {
        constexpr uint32_t pongBaseEvent = 4;
        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);
    }

    __aicore__ inline void DrainVectorPipelineEvents(
        const bool (&event0FromMte3)[PING_PONG_STAGES],
        const bool (&event2FromMte3)[PING_PONG_STAGES])
    {
        constexpr uint32_t pongBaseEvent = 4;
        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
            for (uint32_t streamId = 0; streamId < PING_PONG_STAGES; ++streamId) {
                uint32_t eventOffset = streamId * pongBaseEvent;
                if (event0FromMte3[streamId]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + eventOffset);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventOffset);
                }
                if (event2FromMte3[streamId]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventOffset);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + eventOffset);
                }
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pongBaseEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pongBaseEvent);
    }

    __aicore__ inline AscendC::LocalTensor<ElementW> GetPrefetchedW(uint32_t streamId)
    {
        return resource.l1Buf.template GetBufferByByte<ElementW>(
            W_PREFETCH_L1_OFFSET + streamId * W_PREFETCH_SLOT_BYTES);
    }

    template <class WLayout>
    __aicore__ inline void PrefetchNextChunkW(const WLayout& wLayout)
    {
        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
            const uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
            const auto& stream = cubeBlockScheduler.GetStream(i);
            if (cubeBlockScheduler.StreamIsDone(stream) || stream.offset.isFinalState) {
                continue;
            }
            const uint32_t nextChunk = stream.chunkIdx + 1;
            const uint32_t nextWOffset = stream.offset.wOffset + ChunkSize() * kHeadDim;
            const uint32_t nextTokens = Min(ChunkSize(), stream.batchTokens - nextChunk * ChunkSize());
            auto tensorW = tla::MakeTensor(gmW[nextWOffset], wLayout, Catlass::Arch::PositionGM{});
            GemmCoord nextShape{nextTokens, stream.offset.vBlockDim, kHeadDim};
            auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0),
                tla::MakeShape(nextShape.m(), nextShape.k()));
            // Match the RowMajor GetTileA used by the ordinary WH copy.
            auto tensorTileW = GetTile(tensorBlockW, tla::MakeCoord(0, 0),
                tla::MakeShape(nextShape.m(), min(nextShape.k(), BlockMmadWH::L1_TILE_K)));
            auto l1W = GetPrefetchedW(streamId);
            if (nextTokens < ChunkSize()) {
                AscendC::InitConstValueParams<ElementW> clearParams(
                    1, static_cast<uint16_t>(W_PREFETCH_SLOT_BYTES / 32), 0,
                    static_cast<ElementW>(0));
                AscendC::InitConstValue(l1W, clearParams);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            auto tensorL1W = tla::MakeTensor(l1W, BlockMmadWH::L1A_LAYOUT, Arch::PositionL1{});
            using CopyW = typename TileCopyWH::template CopyGmToL1A<decltype(tensorBlockW)>;
            CopyW copyW;
            copyW(tensorL1W, tensorTileW);
            // The next WH's original MTE2_MTE1 Set/Wait covers this earlier copy.
        }
    }

    template <bool kFinalStored>
    __aicore__ inline void PrepareNextStreamH0(uint32_t streamId)
    {
        static_assert(EpilogueGDNFwdHUpdate::kSharedLowResident,
            "Stream refill requires the private A2 SharedLowResident path.");
        constexpr uint32_t ROWS = 64; // 原 MIX_AIC_1_2 的每 AIV 半个 K128 state。
        constexpr uint32_t COLS = 128;
        constexpr uint32_t ELEMS = ROWS * COLS;
        static_assert(ELEMS * sizeof(float) == 32 * 1024 &&
            ELEMS * sizeof(ElementH) == 16 * 1024, "Reuse only the existing stream slots.");
        uint32_t nextInitialOffset = 0;
        uint32_t nextHOffset = 0;
        if (!vecBlockScheduler.PeekNextTaskH0(streamId, nextInitialOffset, nextHOffset)) {
            return;
        }
        const uint32_t rowBegin = AscendC::GetSubBlockIdx() * ROWS;
        const uint32_t eventOffset = streamId * 4;
        auto state = resource.ubBuf.template GetBufferByByte<float>(
            (streamId == 0 ? 64 : 128) * 1024);
        auto low = resource.ubBuf.template GetBufferByByte<ElementH>(
            (streamId == 0 ? 32 : 96) * 1024);
        if constexpr (kFinalStored) {
            // 原 final-state MTE3 最后读者先退休；原 EVENT0 的正式 Set 保留在 caller。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + eventOffset);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + eventOffset);
        } else {
            // 原最后 V2 skip 分支仍持有 V1 留下的 V_MTE2 EVENT0。
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventOffset);
        }
        if (useInitialState) {
            AscendC::DataCopy(state, gmInitialState[nextInitialOffset + rowBegin * COLS], ELEMS);
        }
        // 零初态也必须把旧读者的退休顺序传递到 Vector。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + eventOffset);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + eventOffset);
        if (useInitialState) {
            AscendC::Cast(low, state, AscendC::RoundMode::CAST_RINT, ELEMS);
        } else {
            AscendC::Duplicate(low, static_cast<ElementH>(0), ELEMS);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + eventOffset);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + eventOffset);
        AscendC::DataCopy(gmH[nextHOffset + rowBegin * COLS], low, ELEMS);
        if constexpr (!kFinalStored) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventOffset);
        }
    }

    template <bool kEnabled>
    struct StreamFinalRefill {
        GDNFwdHKernel* kernel;
        uint32_t streamId;
        CATLASS_DEVICE void operator()() const
        {
            if constexpr (kEnabled) {
                kernel->template PrepareNextStreamH0<true>(streamId);
            }
        }
    };

    __aicore__ inline void Process()
    {
        if constexpr (EpilogueGDNFwdHUpdate::kSharedLowResident) {
            bool twoStreams = false;
            if ASCEND_IS_AIC {
                twoStreams = cubeBlockScheduler.streamsPerWave == PING_PONG_STAGES;
            }
            if ASCEND_IS_AIV {
                twoStreams = vecBlockScheduler.streamsPerWave == PING_PONG_STAGES;
            }
            // 仅入口分派一次；其它模式保留原 wave 实例，不在每 chunk 判断模式。
            if (isVariedLen && twoStreams && !chunkPipelineEnabled && !idleCfg.enabled &&
                kHeadDim == 128 && ValueDim() == 128 &&
                (ChunkSize() == 64 || ChunkSize() == 128)) {
                ProcessTaskEpoch<true>();
                return;
            }
        }
        ProcessTaskEpoch<false>();
    }

    template <bool kStreamRefill>
    __aicore__ inline void ProcessTaskEpoch() {
        if (isVariedLen) {
            AscendC::SyncAll<false>();
        }

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, ValueDim());
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * ChunkSize() * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);

            const int64_t kStride = static_cast<int64_t>(kHeadDim) *
                (InputSequenceMajor() && !kGated ? kNumHead : 1);
            auto kLayout = tla::MakeLayoutFromTag(
                LayoutK(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kStride));
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(coreNum * ChunkSize() * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(coreNum * kHeadDim * PING_PONG_STAGES, cubeBlockScheduler.vBlockSize);
            uint32_t taskWaveCount = cubeBlockScheduler.GetTaskWaveCount();
            if constexpr (kStreamRefill) {
                taskWaveCount = 1;
            }
            for (uint32_t waveIdx = 0; waveIdx < taskWaveCount; ++waveIdx) {
                BlockMmadWH blockMmadWH(resource);
                BlockMmadKV blockMmadKV(resource);
                BlockMmadWH blockMmadWHTail(resource);
                BlockMmadKVTail blockMmadKVTail(resource);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
                if (waveIdx == 0) {
                    AscendC::SyncAll<false>();
                } else {
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
#else
                AscendC::SyncAll<false>();
#endif
                if constexpr (kStreamRefill) {
                    cubeBlockScheduler.InitStreamEpoch();
                } else {
                    cubeBlockScheduler.InitTaskWave(waveIdx);
                }
                uint32_t currStage = 0; // 0: C1, 1: C2
                while (cubeBlockScheduler.isRunning) {
                    if (currStage == 0) {
                    /* C1: v_work = w @ h[i] */
                    cubeBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }

                        const GDNFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                        int64_t cube1OffsetW = cube1Offsets.wOffset;
                        int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                        int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                        auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                        auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                        auto tensorV = tla::MakeTensor(gmVWorkspace[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                        GemmCoord cube1Shape {cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                        auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                        auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                        auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                        if (cube1Offsets.blockTokens < ChunkSize()) {
                            blockMmadWHTail.preSetFlags();
                            if (!cube1Offsets.isInitialState) {
                                blockMmadWHTail.UseExternalL1ATensor(GetPrefetchedW(streamId));
                            }
                            blockMmadWHTail(
                                tensorBlockW, tensorBlockH, tensorBlockV,
                                cube1Shape);
                            blockMmadWHTail.finalWaitFlags();
                        } else {
                            blockMmadWH.preSetFlags();
                            if (!cube1Offsets.isInitialState) {
                                blockMmadWH.UseExternalL1ATensor(GetPrefetchedW(streamId));
                            }
                            blockMmadWH(
                                tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                            blockMmadWH.finalWaitFlags();
                        }
                        // 退休 MTE2/MTE1 旧 credit Wait 与 FIX，再允许后续预取、seed 和配置。
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
                    }
                    } else {
                    /* C2: h[i+1] = k.T @ v_work */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);

                        if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                            // step 3: h[i+1] = k.T @ v_work
                            int64_t cube2OffsetKwork = kGated ? cube2Offsets.kDecayWorkOffset : cube2Offsets.wkOffset;
                            int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                            int64_t cube2OffsetH = cube2Offsets.hWorkOffset;
                            auto tensorK = kGated
                                ? tla::MakeTensor(gmKDecayWorkspace[cube2OffsetKwork], kLayout, Catlass::Arch::PositionGM{})
                                : tla::MakeTensor(gmK[RawQkOffset(cube2OffsetKwork)], kLayout, Catlass::Arch::PositionGM{});
                            auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vworkLayout, Catlass::Arch::PositionGM{});
                            auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2OffsetH], hworkLayout, Catlass::Arch::PositionGM{});
                            GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                            auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                            auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                            auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                            if (cube2Offsets.blockTokens < ChunkSize()) {
                                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                                blockMmadKVTail.preSetFlags();
                                // 归约 K 的尾分形必须清零，不能复用前一调用的 L1 数据。
                                blockMmadKVTail(
                                    tensorBlockK, tensorBlockVwork, tensorBlockHwork,
                                    cube2Shape, Catlass::EmptyClass{}, true);
                                blockMmadKVTail.finalWaitFlags();
                            } else {
                                if constexpr (!kGated) {
                                    blockMmadKV.PrefetchCurrentA(tensorBlockK, cube2Shape);
                                }
                                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                                blockMmadKV.preSetFlags();
                                blockMmadKV(
                                    tensorBlockK, tensorBlockVwork, tensorBlockHwork,
                                    cube2Shape);
                                blockMmadKV.finalWaitFlags();
                            }
                            // 退休必要流水及旧 credits；尾块与完整块共用同一交接链。
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
                            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                        } else {
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done[streamId]);
                    }
                    // Keep all current KV copies ahead of the next W on MTE2.
                    PrefetchNextChunkW(wLayout);
                    }
                    currStage ^= 0x01;
                }
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
            }

        }

        if ASCEND_IS_AIV {
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t logicalHeadTasks =
                (isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch) * vNumHead;
            uint32_t taskCount = logicalHeadTasks * vecBlockScheduler.vBlockCount;
            uint32_t rowsPerSubBlock = (kHeadDim + subBlockNum - 1) / subBlockNum;
            uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
            uint32_t rowEnd = Min(rowBegin + rowsPerSubBlock, kHeadDim);
            uint32_t hRowsPerTile = (32 * 1024) / (ValueDim() * sizeof(ElementH));
            uint32_t stateRowsPerTile =
                (64 * 1024) / (ValueDim() * sizeof(ElementInitialState));
            uint32_t rowsPerTile = Min(hRowsPerTile, stateRowsPerTile);
            uint32_t totalChunks =
                isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + ChunkSize() - 1) / ChunkSize());
            uint32_t stateBlockSize = kHeadDim * ValueDim();
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPing =
                resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPong =
                resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
            uint32_t taskWaveCount = vecBlockScheduler.GetTaskWaveCount();
            if constexpr (kStreamRefill) {
                taskWaveCount = 1;
            }
            InitPipelineSync();
            InitIdlePipelineSlots();
            for (uint32_t waveIdx = 0; waveIdx < taskWaveCount; ++waveIdx) {
                EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);
                epilogueGDNFwdHVnew.ConfigureGDecay(gDecayAddr);
                EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
                epilogueGDNFwdHUpdate.ConfigureGLastExp(gLastExpAddr);
                for (uint32_t initStreamId = 0; initStreamId < vecBlockScheduler.streamsPerWave; ++initStreamId) {
                    uint32_t taskIdx = vecBlockScheduler.GetWaveTaskIndex(waveIdx, initStreamId);
                    uint32_t pingpongFlag = 1;
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                    if (taskIdx < taskCount) {
                        uint32_t vBlockIdx = taskIdx / logicalHeadTasks;
                        uint32_t logicalTask = taskIdx % logicalHeadTasks;
                        uint32_t batchIdx = logicalTask / vNumHead;
                        uint32_t vHeadIdx = logicalTask % vNumHead;
                        uint32_t vBlockOffset = vBlockIdx * vecBlockScheduler.vBlockSize;
                        uint32_t vBlockDim = Min(
                            vecBlockScheduler.vBlockSize, ValueDim() - vBlockOffset);
                        uint32_t chunkOffset =
                            isVariedLen ? vecBlockScheduler.GetVarlenChunkOffset(batchIdx) : 0;
                        uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                        uint32_t hBaseOffset =
                            ((shapeBatchIdx * totalChunks + chunkOffset) * vNumHead + vHeadIdx) *
                            stateBlockSize + vBlockOffset;
                        uint32_t initialStateBaseOffset =
                            (batchIdx * vNumHead + vHeadIdx) * stateBlockSize + vBlockOffset;
                        // A split V block is strided by the original vHeadDim in
                        // GM. Process one state row at a time so both 128-wide
                        // blocks retain the original row layout.
                        uint32_t stateRowsPerStep =
                            vBlockDim == ValueDim() ? rowsPerTile : 1;
                        for (uint32_t rowOffset = rowBegin; rowOffset < rowEnd;
                             rowOffset += stateRowsPerStep) {
                            uint32_t rowsThisTile = Min(stateRowsPerStep, rowEnd - rowOffset);
                            uint32_t stateTileElems = rowsThisTile * vBlockDim;
                            uint32_t hOffset = hBaseOffset + rowOffset * ValueDim();
                            AscendC::LocalTensor<ElementInitialState> stateUbTensor =
                                pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                            AscendC::LocalTensor<ElementH> hUbTensor =
                                pingpongFlag ? hUbTensorPing : hUbTensorPong;
                            auto eventId = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                            if (useInitialState) {
                                uint32_t initialStateOffset =
                                    initialStateBaseOffset + rowOffset * ValueDim();
                                if constexpr (!std::is_same<ElementInitialState, ElementH>::value) {
                                    AscendC::DataCopy(
                                        stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                    AscendC::Cast(
                                        hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT,
                                        stateTileElems);
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                    AscendC::DataCopy(gmH[hOffset], hUbTensor, stateTileElems);
                                } else {
                                    AscendC::DataCopy(
                                        stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                                    AscendC::DataCopy(gmH[hOffset], stateUbTensor, stateTileElems);
                                }
                            } else {
                                // 将旧搬出完成的MTE2等待传递给复用UB的Vector写入。
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                                AscendC::Duplicate(hUbTensor, static_cast<ElementH>(0), stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                                AscendC::DataCopy(gmH[hOffset], hUbTensor, stateTileElems);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                            pingpongFlag = 1 - pingpongFlag;
                        }
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

                }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
                if (waveIdx == 0) {
                    AscendC::SyncAll<false>();
                } else {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                }
#else
                AscendC::SyncAll<false>();
#endif
                if constexpr (kStreamRefill) {
                    vecBlockScheduler.InitStreamEpoch();
                } else {
                    vecBlockScheduler.InitTaskWave(waveIdx);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);
                PresetVectorPipelineEvents();
                uint32_t currStage = 0; // 0: V1, 1: V2
                bool waitStageFence = false;
                bool event0FromMte3[PING_PONG_STAGES] = {false, false};
                bool event2FromMte3[PING_PONG_STAGES] = {
                    !(storeFinalState && std::is_same<ElementFinalState, float>::value),
                    !(storeFinalState && std::is_same<ElementFinalState, float>::value)};
                while (vecBlockScheduler.isRunning) {
                    if (waitStageFence) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    }
                    if (currStage == 0) {
                    /* V1:
                     * gmV = gmU - gmVWorkspace
                     * g_buf = gmG[-1] - gmG
                     * g_buf = exp(g_buf)
                     * gmVWorkspace = g_buf * gmV
                     */
                    vecBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        bool waitWsFromMte3 = storeFinalState && std::is_same<ElementFinalState, float>::value &&
                                              event0FromMte3[streamId];
                        epilogueGDNFwdHVnew(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset],
                            gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset],
                            gmGk[vec1Offsets.gkOffset], gmK[vec1Offsets.wkOffset], gmKDecayWorkspace[vec1Offsets.kDecayWorkOffset],
                            vec1Offsets.blockTokens, kHeadDim, vec1Offsets.vBlockDim, ValueDim(),
                            vecBlockScheduler.cube1Done[streamId], vecBlockScheduler.vec1Done[streamId],
                            vec1Offsets.isInitialState, vec1Offsets.isFinalState, storeFinalState,
                            waitWsFromMte3, (streamId == 0), false, vec1Offsets.gOffset
                        );
                        SignalChunkReady(vec1Offsets);
                        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                            event0FromMte3[streamId] = false;
                        }
                    }
                    } else {
                    /* V2: h[i+1] += h_work if i < num_chunks - 1 else None */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        if (vecBlockScheduler.NeedProcessStage2(stream)) {
                            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                                event0FromMte3[streamId] = true;
                                event2FromMte3[streamId] = !vec2Offsets.isFinalState;
                            }
                            // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                            epilogueGDNFwdHUpdate(
                                gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                gmG[vec2Offsets.gOffset],
                                gmH[vec2Offsets.hSrcOffset],
                                gmHWorkspace[vec2Offsets.hWorkOffset],
                                gmGk[vec2Offsets.gkOffset],
                                gmInitialState[vec2Offsets.initialStateOffset],
                                vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, ValueDim(), vecBlockScheduler.cube2Done[streamId],
                                vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState,
                                useInitialState, (streamId == 0), false,
                                vec2Offsets.globalChunkIdx * gLastExpStride + vec2Offsets.headIdx,
                                StreamFinalRefill<kStreamRefill>{this, streamId}
                            );
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                        } else {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                            if constexpr (kStreamRefill) {
                                PrepareNextStreamH0<false>(streamId);
                            }
                        }
                        PublishIdleChunk(stream.batchIdx, stream.vHeadIdx, stream.vBlockIdx, stream.chunkIdx);
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                    }
                    }
                    waitStageFence = vecBlockScheduler.isRunning;
                    if (waitStageFence) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    }
                    currStage ^= 0x01;
                }

                DrainVectorPipelineEvents(event0FromMte3, event2FromMte3);
            }

        }
        if constexpr (kChunkPipeline) {
            // The dense fused path publishes individual chunks through IBSet/IBWait.
            // Varlen currently uses producer affinity without that handshake, so close
            // H globally before the following O stage consumes h/vNew. The idle
            // pipeline path skips this barrier and leaves the final producer/consumer
            // rendezvous to the future mega entry.
            if (isVariedLen && !idleCfg.enabled) {
                AscendC::SyncAll<false>();
            }
        }
    }

};

}
