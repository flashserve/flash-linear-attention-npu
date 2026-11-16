/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include "gdn_ho_owner_local_config.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
namespace Catlass::Gemm::Kernel {

template <class Original>
struct GDNHoOwnerWHMmad : Original {
    CATLASS_DEVICE explicit GDNHoOwnerWHMmad(Arch::Resource<typename Original::ArchTag> &r) : Original(r) {}
    CATLASS_DEVICE void BindOwnerH(AscendC::LocalTensor<typename Original::ElementB> h)
    {
        // full/tail各有自己的游标；绑定本次当前B index，不把bank当stream。
        this->l1BTensorList[this->l1BListId] = h;
    }
};

template <typename InputT, typename TileShapes, int kQkvLayout>
class GDNFwdHOOwnerLocalKernel : public GDNFwdHKernel<
    InputT, float, float, float, TileShapes, false, true, false, true, 128, 64, kQkvLayout> {
    using Base = GDNFwdHKernel<InputT, float, float, float, TileShapes,
        false, true, false, true, 128, 64, kQkvLayout>;
    using OTypes = GDNFwdOKernel<InputT, float, float, true, 128, 64, kQkvLayout, 1>;
    using L = GDN::HoOwnerLocal::Layout;
    static constexpr bool kGated = false;
    static constexpr uint32_t W_PREFETCH_SLOT_BYTES = Base::W_PREFETCH_SLOT_BYTES;
    using BlockMmadWH = GDNHoOwnerWHMmad<typename Base::BlockMmadWH>;
    using BlockMmadKV = typename Base::BlockMmadKV;
    using BlockMmadKVTail = typename Base::BlockMmadKVTail;
    using BlockMmadQKH = Gemm::Block::GDNFwdOQKHConcatMmad<
        typename OTypes::BlockMmadQK, typename OTypes::TileCopyQH>;
    using BlockMmadAV = typename OTypes::BlockMmadAttenVNEW128;
    using Qmask = typename OTypes::EpilogueGDNFwdOQkmask;
    using Output = typename OTypes::EpilogueGDNFwdOOutput;
    using ArchTag = typename Base::ArchTag;
    using ElementK = typename Base::ElementK;
    using ElementW = typename Base::ElementW;
    using ElementU = typename Base::ElementU;
    using ElementG = typename Base::ElementG;
    using ElementH = typename Base::ElementH;
    using ElementV = typename Base::ElementV;
    using ElementVWork = typename Base::ElementVWork;
    using ElementHWork = typename Base::ElementHWork;
    using ElementInitialState = typename Base::ElementInitialState;
    using ElementFinalState = typename Base::ElementFinalState;
    using LayoutW = typename Base::LayoutW;
    using LayoutH = typename Base::LayoutH;
    using LayoutV = typename Base::LayoutV;
    using LayoutK = typename Base::LayoutK;
    using TileCopyWH = typename Base::TileCopyWH;
    using GDNFwdHOffsets = typename Base::GDNFwdHOffsets;
    using EpilogueGDNFwdHVnew = typename Base::EpilogueGDNFwdHVnew;
    using EpilogueGDNFwdHUpdate = typename Base::EpilogueGDNFwdHUpdate;
    using Base::resource;
    using Base::cubeBlockScheduler;
    using Base::vecBlockScheduler;
    using Base::shapeBatch;
    using Base::seqlen;
    using Base::batch;
    using Base::tokenBatch;
    using Base::kHeadDim;
    using Base::kNumHead;
    using Base::vNumHead;
    using Base::isVariedLen;
    using Base::useInitialState;
    using Base::storeFinalState;
    using Base::gmW;
    using Base::gmH;
    using Base::gmK;
    using Base::gmG;
    using Base::gmU;
    using Base::gmV;
    using Base::gmGk;
    using Base::gmKDecayWorkspace;
    using Base::gmInitialState;
    using Base::gmFinalState;
    using Base::gmVWorkspace;
    using Base::gmVUpdateWorkspace;
    using Base::gmHWorkspace;
    using Base::gDecayAddr;
    using Base::gLastExpAddr;
    using Base::gLastExpStride;
    using Base::RawQkOffset;
    using Base::InputSequenceMajor;
    using Base::ChunkSize;
    using Base::ValueDim;
    using Base::PresetVectorPipelineEvents;
    using Base::DrainVectorPipelineEvents;

    GDN::HoOwnerLocal::Config owner_{};
    Arch::CrossCoreFlag ownerCubeDone_[2] = {4, 5};
    Arch::CrossCoreFlag ownerVecDone_[2] = {6, 7};
    AscendC::GlobalTensor<InputT> gmQ_, gmO_, gmMasked_;
    AscendC::GlobalTensor<float> gmRaw_, gmQH_, gmAV_;
    AscendC::GlobalTensor<bool> gmMask_;

    CATLASS_DEVICE uint32_t OwnerIndex() const
    {
        if ASCEND_IS_AIC { return AscendC::GetBlockIdx(); }
        return AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    }
    CATLASS_DEVICE uint64_t AttnSlot(uint32_t s) const { return (OwnerIndex() * 2ULL + s) * 64 * 64; }
    CATLASS_DEVICE uint64_t AVSlot(uint32_t s) const { return (OwnerIndex() * 2ULL + s) * 64 * 128; }
    CATLASS_DEVICE uint64_t QHSlot(uint32_t s) const { return (OwnerIndex() * 4ULL + s) * 64 * 128; }
    CATLASS_DEVICE AscendC::LocalTensor<InputT> GetOwnerB(uint32_t s)
    { return resource.l1Buf.template GetBufferByByte<InputT>(L::B_L1 + s * L::B_SLOT); }
    CATLASS_DEVICE AscendC::LocalTensor<InputT> GetOwnerH(uint32_t s)
    { return resource.l1Buf.template GetBufferByByte<InputT>(L::B_L1 + s * L::B_SLOT + L::H_IN_B); }
    CATLASS_DEVICE AscendC::LocalTensor<InputT> GetOwnerPrefetchedW(uint32_t s)
    { return resource.l1Buf.template GetBufferByByte<InputT>(L::W_L1 + s * Base::W_PREFETCH_SLOT_BYTES); }

    template <class Scheduler>
    CATLASS_DEVICE void MapHFlags(Scheduler &scheduler)
    {
        for (uint32_t s = 0; s < 2; ++s) {
            scheduler.cube1Done[s] = s;
            scheduler.cube2Done[s] = s;
            scheduler.vec1Done[s] = 2 + s;
            scheduler.vec2Done[s] = 2 + s;
        }
    }
    CATLASS_DEVICE static void RetireOInputCredits()
    {
        // 保留原QKH/AV调用后的有向退休，不提升MM对象信用生命周期。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    }
    template <class WLayout>
    __aicore__ inline void PrefetchNextOwnerW(const WLayout& wLayout)
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
            auto l1W = GetOwnerPrefetchedW(streamId);
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
    __aicore__ inline void PrepareOwnerH0(uint32_t streamId)
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
            streamId == 0 ? L::CARRY0 : L::CARRY1);
        auto low = resource.ubBuf.template GetBufferByByte<ElementH>(
            streamId == 0 ? L::LOW0 : L::LOW1);
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
    struct OwnerFinalRefill {
        GDNFwdHOOwnerLocalKernel *kernel;
        uint32_t streamId;
        CATLASS_DEVICE void operator()() const
        {
            if constexpr (kEnabled) { kernel->template PrepareOwnerH0<true>(streamId); }
        }
    };

    template <bool kStreamRefill>
    CATLASS_DEVICE void ProcessEpoch()
    {
        if (isVariedLen) { AscendC::SyncAll<false>(); }
        if ASCEND_IS_AIC {
            const uint32_t coreNum = AscendC::GetBlockNum();
            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, ValueDim());
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * ChunkSize() * PING_PONG_STAGES, 128);
            const int64_t kStride = static_cast<int64_t>(128) * (InputSequenceMajor() ? kNumHead : 1);
            auto kLayout = tla::MakeLayoutFromTag(LayoutK(128, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kStride));
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(coreNum * ChunkSize() * PING_PONG_STAGES, 128);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(coreNum * 128 * PING_PONG_STAGES, 128);
            auto qLayout = tla::MakeLayoutFromTag(layout::RowMajor(shapeBatch * kNumHead * seqlen, 128, kStride));
            auto qhLayout = tla::MakeLayout<float, layout::RowMajor>(coreNum * 4 * 64, 128);
            auto avLayout = tla::MakeLayout<float, layout::RowMajor>(coreNum * 2 * 64, 128);
            auto rawVLayout = tla::MakeLayout<InputT, layout::RowMajor>(shapeBatch * vNumHead * seqlen, 128);
            uint32_t taskWaveCount = kStreamRefill ? 1 : cubeBlockScheduler.GetTaskWaveCount();
            for (uint32_t waveIdx = 0; waveIdx < taskWaveCount; ++waveIdx) {
                BlockMmadWH blockMmadWH(resource), blockMmadWHTail(resource);
                BlockMmadKV blockMmadKV(resource);
                BlockMmadKVTail blockMmadKVTail(resource);
                BlockMmadQKH blockMmadQKH(resource);
                BlockMmadAV blockMmadAV(resource);
                if (waveIdx == 0) { AscendC::SyncAll<false>(); }
                else { AscendC::PipeBarrier<PIPE_ALL>(); } // 原wave切换合同
                if constexpr (kStreamRefill) { cubeBlockScheduler.InitStreamEpoch(); }
                else { cubeBlockScheduler.InitTaskWave(waveIdx); }
                MapHFlags(cubeBlockScheduler);
                while (cubeBlockScheduler.isRunning) {
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
                            blockMmadWHTail.BindOwnerH(GetOwnerH(streamId));
                            blockMmadWHTail.preSetFlags();
                            if (!cube1Offsets.isInitialState) {
                                blockMmadWHTail.UseExternalL1ATensor(GetOwnerPrefetchedW(streamId));
                            }
                            blockMmadWHTail(
                                tensorBlockW, tensorBlockH, tensorBlockV,
                                cube1Shape);
                            blockMmadWHTail.finalWaitFlags();
                        } else {
                            blockMmadWH.BindOwnerH(GetOwnerH(streamId));
                            blockMmadWH.preSetFlags();
                            if (!cube1Offsets.isInitialState) {
                                blockMmadWH.UseExternalL1ATensor(GetOwnerPrefetchedW(streamId));
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

                    // WH已发布给HV1；同stream原H只载一次，QKH只补K。
                    for (uint32_t s = 0; s < 2; ++s) {
                        const auto &stream = cubeBlockScheduler.GetStream(s);
                        if (cubeBlockScheduler.StreamIsDone(stream)) { continue; }
                        const auto &off = cubeBlockScheduler.GetCurTaskOffsets(stream);
                        Arch::CrossCoreWaitFlag(ownerVecDone_[s]); // Ofree先于raw/QH写入
                        auto attnLayout = tla::MakeLayout<float, layout::RowMajor>(coreNum * 2 * 64, off.blockTokens);
                        auto tq = tla::MakeTensor(gmQ_[RawQkOffset(off.wkOffset)], qLayout, Arch::PositionGM{});
                        auto tk = tla::MakeTensor(gmK[RawQkOffset(off.wkOffset)], kLayout, Arch::PositionGM{});
                        auto th = tla::MakeTensor(gmH[off.hSrcOffset], hLayout, Arch::PositionGM{});
                        auto tr = tla::MakeTensor(gmRaw_[AttnSlot(s)], attnLayout, Arch::PositionGM{});
                        auto tqh = tla::MakeTensor(gmQH_[QHSlot(s)], qhLayout, Arch::PositionGM{});
                        blockMmadQKH.BindOwnerL1B(GetOwnerB(s));
                        blockMmadQKH.preSetFlags();
                        blockMmadQKH.template RunRaw<true>(tq, tk, th, tr, off.blockTokens);
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ownerCubeDone_[s]);
                        blockMmadQKH.StoreH(tqh, off.blockTokens);
                        blockMmadQKH.finalWaitFlags();
                        RetireOInputCredits();
                    }
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

                    PrefetchNextOwnerW(wLayout);
                    for (uint32_t s = 0; s < 2; ++s) {
                        const auto &stream = cubeBlockScheduler.GetStream(s);
                        if (cubeBlockScheduler.StreamIsDone(stream)) { continue; }
                        const auto &off = cubeBlockScheduler.GetCurTaskOffsets(stream);
                        Arch::CrossCoreWaitFlag(ownerVecDone_[s]); // mask也承接HV1 raw-vnew drain
                        auto attnLayout = tla::MakeLayout<InputT, layout::RowMajor>(coreNum * 2 * 64, off.blockTokens);
                        auto ta = tla::MakeTensor(gmMasked_[AttnSlot(s)], attnLayout, Arch::PositionGM{});
                        auto tv = tla::MakeTensor(gmV[off.uvOffset], rawVLayout, Arch::PositionGM{});
                        auto to = tla::MakeTensor(gmAV_[AVSlot(s)], avLayout, Arch::PositionGM{});
                        GemmCoord shape{off.blockTokens, 128, off.blockTokens};
                        auto ba = GetTile(ta, tla::MakeCoord(0,0), tla::MakeShape(shape.m(), shape.k()));
                        auto bv = GetTile(tv, tla::MakeCoord(0,0), tla::MakeShape(shape.k(), shape.n()));
                        auto bo = GetTile(to, tla::MakeCoord(0,0), tla::MakeShape(shape.m(), shape.n()));
                        blockMmadAV.preSetFlags();
                        blockMmadAV(ba, bv, bo, shape);
                        blockMmadAV.finalWaitFlags();
                        RetireOInputCredits();
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ownerCubeDone_[s]);
                    }
                }
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
                Arch::CrossCoreWaitFlag(ownerVecDone_[0]);
                Arch::CrossCoreWaitFlag(ownerVecDone_[1]);
            }
        }
        if ASCEND_IS_AIV {
            const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            const uint32_t subBlockNum = AscendC::GetSubBlockNum();
            const uint32_t logicalHeadTasks = (isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch) * vNumHead;
            const uint32_t taskCount = logicalHeadTasks * vecBlockScheduler.vBlockCount;
            const uint32_t rowBegin = subBlockIdx * 64;
            const uint32_t rowEnd = rowBegin + 64;
            const uint32_t rowsPerTile = 64;
            const uint32_t totalChunks = isVariedLen ? vecBlockScheduler.totalChunks : (seqlen + 63) / 64;
            const uint32_t stateBlockSize = 128 * 128;
            auto stateUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementInitialState>(L::CALC);
            auto stateUbTensorPong = stateUbTensorPing;
            auto hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(L::LOW0);
            auto hUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementH>(L::LOW1);
            auto mask = resource.ubBuf.template GetBufferByByte<float>(L::MASK);
            AscendC::Duplicate<float>(mask, 0.0f, 64 * 64);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < 64; ++i) { AscendC::Duplicate<float>(mask[i * 64], 1.0f, i + 1); }
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t taskWaveCount = kStreamRefill ? 1 : vecBlockScheduler.GetTaskWaveCount();
            for (uint32_t waveIdx = 0; waveIdx < taskWaveCount; ++waveIdx) {
                EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);
                EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
                epilogueGDNFwdHVnew.ConfigureOwnerLocal(resource);
                epilogueGDNFwdHUpdate.ConfigureOwnerLocal(resource);
                epilogueGDNFwdHVnew.ConfigureGDecay(gDecayAddr);
                epilogueGDNFwdHUpdate.ConfigureGLastExp(gLastExpAddr);
                Qmask qmask(resource); Output output(resource);
                qmask.ConfigureOwnerLocal(resource); output.ConfigureOwnerLocal(resource);
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
                                initStreamId == 0 ? hUbTensorPing : hUbTensorPong;
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

                if (waveIdx == 0) { AscendC::SyncAll<false>(); }
                else {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Arch::CrossCoreBarrier<0x1, PIPE_MTE3>(); // 原H0 wave handoff
                }
                if constexpr (kStreamRefill) { vecBlockScheduler.InitStreamEpoch(); }
                else { vecBlockScheduler.InitTaskWave(waveIdx); }
                MapHFlags(vecBlockScheduler);
                for (uint32_t s = 0; s < 2; ++s) {
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[s]);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ownerVecDone_[s]);
                }
                PresetVectorPipelineEvents();
                bool pendingFence = false;
                bool event0FromMte3[2] = {false, false};
                bool event2FromMte3[2] = {!storeFinalState, !storeFinalState};
                while (vecBlockScheduler.isRunning) {
                    if (pendingFence) { AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1); }
                    vecBlockScheduler.InitTasks();
                    if (!vecBlockScheduler.isRunning) { break; }
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3 + streamId * 4);
                            continue;
                        }
                        const GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        bool waitWsFromMte3 = storeFinalState && std::is_same<ElementFinalState, float>::value &&
                                              event0FromMte3[streamId];
                        epilogueGDNFwdHVnew.template operator()<true>(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset],
                            gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset],
                            gmGk[vec1Offsets.gkOffset], gmK[vec1Offsets.wkOffset], gmKDecayWorkspace[vec1Offsets.kDecayWorkOffset],
                            vec1Offsets.blockTokens, kHeadDim, vec1Offsets.vBlockDim, ValueDim(),
                            vecBlockScheduler.cube1Done[streamId], vecBlockScheduler.vec1Done[streamId],
                            vec1Offsets.isInitialState, vec1Offsets.isFinalState, storeFinalState,
                            waitWsFromMte3, (streamId == 0), false, vec1Offsets.gOffset
                        );
                        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                            event0FromMte3[streamId] = false;
                        }
                    }

                    // 不能以一般Wait阻塞S：显式MTE2->V->S回执确认两个G lease及HV1退休。
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(2);
                    for (uint32_t s = 0; s < 2; ++s) {
                        const auto &stream = vecBlockScheduler.GetStream(s);
                        if (vecBlockScheduler.StreamIsDone(stream)) { continue; }
                        const auto &off = vecBlockScheduler.GetCurTaskOffsets(stream);
                        const uint32_t n0 = (off.blockTokens + 1) / 2;
                        if ((subBlockIdx == 0 ? n0 : off.blockTokens - n0) != 0) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1 + 4 * s);
                        }
                    }
                    for (uint32_t s = 0; s < 2; ++s) {
                        const auto &stream = vecBlockScheduler.GetStream(s);
                        if (vecBlockScheduler.StreamIsDone(stream)) { continue; }
                        const auto &off = vecBlockScheduler.GetCurTaskOffsets(stream);
                        Arch::CrossCoreWaitFlag(ownerCubeDone_[s]);
                        uint32_t pp = 0;
                        qmask.template operator()<true>(gmMasked_[AttnSlot(s)], gmG[off.gOffset],
                            gmRaw_[AttnSlot(s)], gmMask_, 64, off.blockTokens, 128, 128, pp,
                            stream.batchIdx, off.headIdx, stream.chunkIdx);
                        if (isVariedLen != 0) { Arch::CrossCoreBarrier<0x1, PIPE_MTE3>(); }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ownerVecDone_[s]);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
                    // 原HV1宏fence挪到所有Q之后。
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
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
                                OwnerFinalRefill<kStreamRefill>{this, streamId}
                            );
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                        } else {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                            if constexpr (kStreamRefill) {
                                PrepareOwnerH0<false>(streamId);
                            }
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                    }

                    for (uint32_t s = 0; s < 2; ++s) {
                        const auto &stream = vecBlockScheduler.GetStream(s);
                        if (vecBlockScheduler.StreamIsDone(stream)) { continue; }
                        const auto &off = vecBlockScheduler.GetCurTaskOffsets(stream);
                        Arch::CrossCoreWaitFlag(ownerCubeDone_[s]);
                        const uint64_t out = owner_.outputSequenceMajor
                            ? GDN::QkvSequenceMajorOffset(off.uvOffset, seqlen, vNumHead, 128) : off.uvOffset;
                        const uint32_t outStride = owner_.outputSequenceMajor ? vNumHead * 128 : 128;
                        uint32_t pp = 0;
                        // zero行仍走原全部本地事件，退休H0低源及O源；只有GM输出自身跳零。
                        output.template operator()<true>(gmO_[out], gmG[off.gOffset], gmAV_[AVSlot(s)],
                            gmQH_[QHSlot(s)], owner_.oTiling.scale, off.blockTokens, 128, 128,
                            outStride, pp, stream.batchIdx, off.headIdx, stream.chunkIdx);
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ownerVecDone_[s]);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
                    // 原HV2宏fence挪到所有Output后；退出轮也恰消费一次。
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    pendingFence = true;
                }
                DrainVectorPipelineEvents(event0FromMte3, event2FromMte3);
                if constexpr (!kStreamRefill) {
                    if (waveIdx + 1 < taskWaveCount) {
                        // dense跨wave旧Wait(1/5)在MTE2；收到定向S回执后才能重新种H0票。
                        // 只在非末wave交界增加4调用，varlen和每chunk均不增加此排空。
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(2);
                    }
                }
            }
        }
    }

public:
    CATLASS_DEVICE explicit GDNFwdHOOwnerLocalKernel(Arch::Resource<ArchTag> &r) : Base(r) {}
    CATLASS_DEVICE void ProcessOwnerLocal(const GDN::HoOwnerLocal::Config &cfg)
    {
        static_assert(L::END <= L::SYNC && L::L1_END <= ArchTag::L1_SIZE,
            "The private owner resource map must fit the original synchronization boundary");
        static_assert(BlockMmadWH::L1_TILE_K == 128 && Base::W_PREFETCH_SLOT_BYTES == 32 * 1024,
            "Owner H pin requires a single full K128 step and two original W slots");
        owner_ = cfg;
        gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(cfg.q));
        gmO_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(cfg.o));
        gmMasked_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(cfg.userWorkspace + cfg.oTiling.aftermaskWorkspaceOffset));
        gmRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cfg.userWorkspace + cfg.oTiling.attnWorkspaceOffset));
        gmQH_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cfg.userWorkspace + cfg.oTiling.hWorkspaceOffset));
        gmAV_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cfg.userWorkspace + cfg.oTiling.vWorkspaceOffset));
        gmMask_.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(cfg.userWorkspace + cfg.oTiling.maskWorkspaceOffset));
        if (isVariedLen) { ProcessEpoch<true>(); }
        else { ProcessEpoch<false>(); }
    }
};

} // namespace Catlass::Gemm::Kernel
#endif
