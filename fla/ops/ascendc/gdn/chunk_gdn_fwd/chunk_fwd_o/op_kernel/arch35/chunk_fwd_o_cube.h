/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * AIC Stage2: L0/L1 ping-pong + Fixpipe→UB (A_raw/O_s_raw FP32, design §354).
 * Cross-core synchronization uses two ordered ready chains.
 */

#ifndef CHUNK_FWD_O_ARCH35_CUBE_H
#define CHUNK_FWD_O_ARCH35_CUBE_H

#define CATLASS_ARCH 3510

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN {

using namespace AscendC;

class ChunkFwdOA5CubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using Element = bfloat16_t;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;

    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutCM, Element,
                                                              LayoutRM>;
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutCM, Element,
                                                              LayoutRM>;
    using TileCopyAV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutRM, Element,
                                                              LayoutRM>;
    using DirectTileCopyCC = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, Element, LayoutRM, Element, LayoutCM, float, LayoutRM, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using DirectTileCopyRM = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, Element, LayoutRM, Element, LayoutCM, float, LayoutRM>;

    static constexpr uint32_t kBt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
    static constexpr uint32_t kK = static_cast<uint32_t>(CHUNK_FWD_O_A5_K);
    static constexpr uint32_t kV = static_cast<uint32_t>(CHUNK_FWD_O_A5_V);
    static constexpr uint32_t kL0BufferCount = CHUNK_FWD_O_L0_BUFFER_COUNT;
    static constexpr uint32_t kL1ResidentHeadCount = CHUNK_FWD_O_L1_RESIDENT_HEAD_COUNT;
    static constexpr uint32_t kL1EventCount = 2U * kL1ResidentHeadCount;
    static constexpr TEventID kL1EventBase = 5;

    static_assert(CHUNK_FWD_O_L0_A_BYTES * kL0BufferCount <= ArchTag::L0A_SIZE,
                  "Stage2 L0A ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L0_B_BYTES * kL0BufferCount <= ArchTag::L0B_SIZE,
                  "Stage2 L0B ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L0_C_BYTES * kL0BufferCount <= ArchTag::L0C_SIZE,
                  "Stage2 L0C ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L1_STAGE2_END <= CHUNK_FWD_O_L1_APRIME_BASE,
                  "Stage2 L1 stream slots overlap Stage4 resident buffers.");
    static_assert(CHUNK_FWD_O_L1_STAGE4_END <= ArchTag::L1_SIZE,
                  "Stage4 A-prime/V resident slots exceed architecture limit.");

    __aicore__ inline ChunkFwdOA5CubeProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                             GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o,
                                             GM_ADDR workspace)
        : q_(q), k_(k), v_(v), h_(h), g_(g), cuSeqlens_(cuSeqlens), chunkOffsets_(chunkOffsets), o_(o),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkFwdOTilingData &tiling)
    {
        tiling_ = tiling;
        qGm_.SetGlobalBuffer((__gm__ Element *)q_);
        kGm_.SetGlobalBuffer((__gm__ Element *)k_);
        vGm_.SetGlobalBuffer((__gm__ Element *)v_);
        hGm_.SetGlobalBuffer((__gm__ Element *)h_);
        if ASCEND_IS_AIC {
            SetLoadDataPaddingValue<Element>(static_cast<Element>(0));
            for (uint32_t eventIdx = 0; eventIdx < kL1EventCount; ++eventIdx) {
                SetFlag<HardEvent::MTE1_MTE2>(L1Event(eventIdx));
            }
            for (uint32_t slotIdx = 0; slotIdx < kL0BufferCount; ++slotIdx) {
                SetFlag<HardEvent::M_MTE1>(L0AEvent(slotIdx));
                SetFlag<HardEvent::M_MTE1>(L0BEvent(slotIdx));
                SetFlag<HardEvent::FIX_M>(slotIdx);
            }
            l0ASlot_ = 0U;
            l0BSlot_ = 0U;
            l0CSlot_ = 0U;
        }
    }

    __aicore__ inline void Process(uint32_t coreIdx, uint32_t coreNum)
    {
        ChunkFwdOChunkLoc loc;
        const uint32_t headGroupNum = ChunkFwdOHeadGroupNum(tiling_);
        const uint32_t groupTaskNum = static_cast<uint32_t>(tiling_.chunkNum) * headGroupNum;
        for (uint32_t groupTaskIdx = coreIdx; groupTaskIdx < groupTaskNum; groupTaskIdx += coreNum) {
            const uint32_t loopIdx = groupTaskIdx / headGroupNum;
            const uint32_t headGroupIdx = groupTaskIdx % headGroupNum;
            const int64_t hvBase = static_cast<int64_t>(headGroupIdx) * tiling_.taskGroupSize;
            ChunkFwdOResolveChunkLoc(cuSeqlens_, chunkOffsets_, tiling_, loopIdx, loc);
            const int64_t remaining = tiling_.vNumHead - hvBase;
            const int64_t taskCount = remaining < tiling_.taskGroupSize ? remaining : tiling_.taskGroupSize;
            const int64_t groupRound = ChunkFwdOGroupRound(groupTaskIdx, coreNum);
            // Stage 2: consume the Stage 1 ready chain and produce QK/QH
            // for every HEAD in this task group before entering Stage 4.
            if ASCEND_IS_AIC {
                for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                    const int64_t hv = hvBase + headOffset;
                    const int64_t hk = hv / tiling_.hvPerHk;
                    const int64_t firstHvForHk = hk * tiling_.hvPerHk;
                    const int64_t firstHvInGroup = firstHvForHk < hvBase ? hvBase : firstHvForHk;
                    const uint32_t qkL1Slot = static_cast<uint32_t>(firstHvInGroup - hvBase);
                    const bool loadQK = hv == firstHvInGroup;
                    const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                    const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                    ProcessStage2Head(loc, hk, hv, ownerSubBlock, localSlot, qkL1Slot,
                                      static_cast<uint32_t>(headOffset), loadQK);
                }
            }

            // Stage 4: consume Stage 3 A-prime ready signals, compute
            // A-prime@V, and publish O_l for every HEAD in this group.
            if ASCEND_IS_AIC {
                for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
                    const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                    const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                    ProcessStage4Head(loc, hvBase + headOffset, ownerSubBlock, localSlot,
                                      static_cast<uint32_t>(headOffset), groupRound);
                }
            }
            Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
        }
        if ASCEND_IS_AIC {
            for (uint32_t eventIdx = 0; eventIdx < kL1EventCount; ++eventIdx) {
                WaitFlag<HardEvent::MTE1_MTE2>(L1Event(eventIdx));
            }
        }
    }

private:
    static constexpr TEventID L0AEvent(uint32_t slot)
    {
        return static_cast<TEventID>(2U * slot);
    }

    static constexpr TEventID L0BEvent(uint32_t slot)
    {
        return static_cast<TEventID>(2U * slot + 1U);
    }

    static constexpr TEventID L1Event(uint32_t eventIdx)
    {
        return static_cast<TEventID>(kL1EventBase + eventIdx);
    }

    __aicore__ inline void ProcessStage4Head(const ChunkFwdOChunkLoc &loc, int64_t hv,
                                             uint32_t ownerSubBlock, uint32_t localSlot,
                                             uint32_t headOffset, int64_t groupRound)
    {
        const uint32_t mActual = static_cast<uint32_t>(loc.chunkLen);
        const TEventID aPrimeEvent = L1Event(headOffset);
        const TEventID vEvent = L1Event(kL1ResidentHeadCount + headOffset);

        // Load A-prime and V from GM to their Stage4 L1 buffers.
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const int64_t vOffset = ChunkFwdOVOOffset(tiling_, loc, hv);
        WaitFlag<HardEvent::MTE1_MTE2>(aPrimeEvent);
        GlobalTensor<Element> aPrimeGm;
        aPrimeGm.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(
            ChunkFwdOAPrimeGmOffset(workspace_, tiling_, coreIdx, groupRound, headOffset)));
        auto layoutAPrimeGm = tla::MakeLayout<Element, LayoutRM>(kBt, kBt);
        auto layoutVGm = tla::MakeLayout<Element, LayoutRM>(kBt, kV);
        auto tensorAPrimeGm = tla::MakeTensor(aPrimeGm, layoutAPrimeGm, Catlass::Arch::PositionGM{});
        auto tensorVGm = tla::MakeTensor(vGm_[vOffset], layoutVGm, Catlass::Arch::PositionGM{});
        auto blockAPrime = GetTile(tensorAPrimeGm, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kBt));
        auto blockV = GetTile(tensorVGm, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kV));
        using LayoutTagL1A = typename TileCopyAV::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyAV::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyAV::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyAV::LayoutTagL0B;
        using CopyGmToL1A = typename TileCopyAV::template CopyGmToL1A<decltype(blockAPrime)>;
        using CopyGmToL1B = typename TileCopyAV::template CopyGmToL1B<decltype(blockV)>;
        using CopyL1ToL0A = typename TileCopyAV::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyAV::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;
        LocalTensor<Element> l1APrime =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1APrimeOffset(headOffset));
        LocalTensor<Element> l1V =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1VOffset(headOffset));
        auto layoutL1APrime = tla::MakeLayout<Element, LayoutTagL1A>(kBt, kBt);
        auto layoutL1V = tla::MakeLayout<Element, LayoutTagL1B>(kBt, kV);
        auto tensorL1APrime = tla::MakeTensor(l1APrime, layoutL1APrime, Catlass::Arch::PositionL1{});
        auto tensorL1V = tla::MakeTensor(l1V, layoutL1V, Catlass::Arch::PositionL1{});
        CopyGmToL1A{}(tensorL1APrime, blockAPrime);
        SetFlag<HardEvent::MTE2_MTE1>(aPrimeEvent);
        WaitFlag<HardEvent::MTE1_MTE2>(vEvent);
        CopyGmToL1B{}(tensorL1V, blockV);
        SetFlag<HardEvent::MTE2_MTE1>(vEvent);
        WaitFlag<HardEvent::MTE2_MTE1>(aPrimeEvent);
        WaitFlag<HardEvent::MTE2_MTE1>(vEvent);

        const uint32_t avASlot = l0ASlot_;
        const uint32_t avBSlot = l0BSlot_;
        const uint32_t avCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;

        // Move A-prime and V into L0, then compute A-prime @ V.
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(avASlot));
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(avBSlot));
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(avCSlot));
        WaitFlag<HardEvent::M_MTE1>(L0AEvent(avASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(avBSlot));
        WaitFlag<HardEvent::FIX_M>(avCSlot);
        if (mActual == kBt) {
            auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(kBt, kBt);
            auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(kBt, kV);
            auto layoutL0C = tla::MakeLayoutL0C(kBt, kV);
            auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
            auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
            auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
            auto tileL1APrime = GetTile(tensorL1APrime, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kBt));
            auto tileL1V = GetTile(tensorL1V, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kV));
            auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kBt));
            auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kV));
            auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kV));
            CopyL1ToL0A{}(tileL0A, tileL1APrime);
            CopyL1ToL0B{}(tileL0B, tileL1V);
            SetFlag<HardEvent::MTE1_M>(avCSlot);
            WaitFlag<HardEvent::MTE1_M>(avCSlot);
            TileMmad{}(tileL0C, tileL0A, tileL0B, kBt, kV, kBt, true, 0);
        } else {
            auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(mActual, mActual);
            auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(mActual, kV);
            auto layoutL0C = tla::MakeLayoutL0C(mActual, kV);
            auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
            auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
            auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
            auto tileL1APrime = GetTile(tensorL1APrime, tla::MakeCoord(0, 0), tla::MakeShape(mActual, mActual));
            auto tileL1V = GetTile(tensorL1V, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kV));
            auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, mActual));
            auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kV));
            CopyL1ToL0A{}(tileL0A, tileL1APrime);
            CopyL1ToL0B{}(tileL0B, tileL1V);
            SetFlag<HardEvent::MTE1_M>(avCSlot);
            WaitFlag<HardEvent::MTE1_M>(avCSlot);
            TileMmad{}(tensorL0C, tensorL0A, tensorL0B, true, 0);
        }
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(L0AEvent(avASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(avBSlot));
        SetFlag<HardEvent::MTE1_MTE2>(aPrimeEvent);
        SetFlag<HardEvent::MTE1_MTE2>(vEvent);

        // Publish O_l from L0C to the owner AIV's UB slot.
        auto olLayout = tla::MakeLayoutL0C(mActual, kV);
        auto olTensor = tla::MakeTensor(l0C, olLayout, Catlass::Arch::PositionL0C{});
        auto olTile = GetTile(olTensor, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kV));
        auto layoutOlUb = tla::MakeLayout<float, LayoutRM>(mActual, kV);
        auto tensorOlUb = tla::MakeTensor(
            resource_.ubBuf.template GetBufferByByte<float>(ChunkFwdOOlOffset(localSlot)), layoutOlUb,
            Catlass::Arch::PositionUB{});
        using CopyOlToUb = typename DirectTileCopyRM::template CopyL0CToDst<decltype(tensorOlUb)>;
        CopyOlToUb copyOlToUb;
        SetFlag<HardEvent::M_FIX>(avCSlot);
        WaitFlag<HardEvent::M_FIX>(avCSlot);
        copyOlToUb(tensorOlUb, olTile, static_cast<uint8_t>(ownerSubBlock), 0);
        SetFlag<HardEvent::FIX_M>(avCSlot);

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
    }

    __aicore__ inline void ProcessStage2Head(const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv,
                                             uint32_t ownerSubBlock, uint32_t localSlot,
                                             uint32_t qkL1Slot, uint32_t hL1Slot, bool loadQK)
    {
        const uint32_t m = kBt;
        const uint32_t mActual = static_cast<uint32_t>(loc.chunkLen);
        const TEventID qkEvent = L1Event(qkL1Slot);
        const TEventID hEvent = L1Event(kL1ResidentHeadCount + hL1Slot);

        // Load Q and K once for all GVA HEADs that share the same HK in this group.
        if (loadQK) {
            WaitFlag<HardEvent::MTE1_MTE2>(qkEvent);
            const int64_t qOffset = ChunkFwdOQKOffset(tiling_, loc, hk);
            const int64_t kOffset = ChunkFwdOQKOffset(tiling_, loc, hk);
            using LayoutTagL1Q = typename TileCopyQK::LayoutTagL1A;
            using LayoutTagL1K = typename TileCopyQK::LayoutTagL1B;
            auto layoutQGm = tla::MakeLayout<Element, LayoutRM>(m, kK);
            auto layoutKGm = tla::MakeLayout<Element, LayoutCM>(kK, m);
            auto tensorQGm = tla::MakeTensor(qGm_[qOffset], layoutQGm, Catlass::Arch::PositionGM{});
            auto tensorKGm = tla::MakeTensor(kGm_[kOffset], layoutKGm, Catlass::Arch::PositionGM{});
            auto blockQ = GetTile(tensorQGm, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kK));
            auto blockK = GetTile(tensorKGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, mActual));
            using CopyGmToL1Q = typename TileCopyQK::template CopyGmToL1A<decltype(blockQ)>;
            using CopyGmToL1K = typename TileCopyQK::template CopyGmToL1B<decltype(blockK)>;
            LocalTensor<Element> l1Q =
                resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(qkL1Slot));
            LocalTensor<Element> l1K =
                resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(qkL1Slot));
            auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1Q>(m, kK);
            auto layoutL1K = tla::MakeLayout<Element, LayoutTagL1K>(kK, m);
            auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
            auto tensorL1K = tla::MakeTensor(l1K, layoutL1K, Catlass::Arch::PositionL1{});
            CopyGmToL1Q{}(tensorL1Q, blockQ);
            CopyGmToL1K{}(tensorL1K, blockK);
            SetFlag<HardEvent::MTE2_MTE1>(qkEvent);
            WaitFlag<HardEvent::MTE2_MTE1>(qkEvent);
        }

        // Move Q/K from L1 to L0 and launch Q @ K^T.
        const uint32_t qktASlot = l0ASlot_;
        const uint32_t qktBSlot = l0BSlot_;
        const uint32_t qktCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        using LayoutTagQkL1A = typename TileCopyQK::LayoutTagL1A;
        using LayoutTagQkL1B = typename TileCopyQK::LayoutTagL1B;
        using LayoutTagQkL0A = typename TileCopyQK::LayoutTagL0A;
        using LayoutTagQkL0B = typename TileCopyQK::LayoutTagL0B;
        using CopyQkL1ToL0A = typename TileCopyQK::CopyL1ToL0A;
        using CopyQkL1ToL0B = typename TileCopyQK::CopyL1ToL0B;
        using QkTileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagQkL1A>;
        LocalTensor<Element> l1Q =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(qkL1Slot));
        LocalTensor<Element> l1K =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(qkL1Slot));
        LocalTensor<Element> qktL0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(qktASlot));
        LocalTensor<Element> qktL0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(qktBSlot));
        LocalTensor<float> qktL0C =
            resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(qktCSlot));
        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagQkL1A>(m, kK);
        auto layoutL1K = tla::MakeLayout<Element, LayoutTagQkL1B>(kK, m);
        auto layoutQktL0A = tla::MakeLayout<Element, LayoutTagQkL0A>(m, kK);
        auto layoutQktL0B = tla::MakeLayout<Element, LayoutTagQkL0B>(kK, m);
        auto layoutQktL0C = tla::MakeLayoutL0C(m, m);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(l1K, layoutL1K, Catlass::Arch::PositionL1{});
        auto tensorQktL0A = tla::MakeTensor(qktL0A, layoutQktL0A, Catlass::Arch::PositionL0A{});
        auto tensorQktL0B = tla::MakeTensor(qktL0B, layoutQktL0B, Catlass::Arch::PositionL0B{});
        auto tensorQktL0C = tla::MakeTensor(qktL0C, layoutQktL0C, Catlass::Arch::PositionL0C{});
        auto tileL1Q = GetTile(tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL1K = GetTile(tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));
        auto tileQktL0A = GetTile(tensorQktL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileQktL0B = GetTile(tensorQktL0B, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));
        auto tileQktL0C = GetTile(tensorQktL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
        WaitFlag<HardEvent::M_MTE1>(L0AEvent(qktASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(qktBSlot));
        WaitFlag<HardEvent::FIX_M>(qktCSlot);
        CopyQkL1ToL0A{}(tileQktL0A, tileL1Q);
        CopyQkL1ToL0B{}(tileQktL0B, tileL1K);
        SetFlag<HardEvent::MTE1_M>(qktCSlot);
        WaitFlag<HardEvent::MTE1_M>(qktCSlot);
        QkTileMmad{}(tileQktL0C, tileQktL0A, tileQktL0B, m, m, kK, true, 0);
        SetFlag<HardEvent::M_MTE1>(L0AEvent(qktASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(qktBSlot));

        // Load H to L1 while Q @ K^T runs on M/FIX.
        WaitFlag<HardEvent::MTE1_MTE2>(hEvent);
        const int64_t hOffset = ChunkFwdOHOffset(tiling_, loc, hv);
        using LayoutTagL1H = typename TileCopyQH::LayoutTagL1B;
        auto layoutHGm = tla::MakeLayout<Element, LayoutCM>(kK, kV);
        auto tensorHGm = tla::MakeTensor(hGm_[hOffset], layoutHGm, Catlass::Arch::PositionGM{});
        auto blockH = GetTile(tensorHGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        using CopyGmToL1H = typename TileCopyQH::template CopyGmToL1B<decltype(blockH)>;
        LocalTensor<Element> l1H =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1HOffset(hL1Slot));
        auto layoutL1H = tla::MakeLayout<Element, LayoutTagL1H>(kK, kV);
        auto tensorL1H = tla::MakeTensor(l1H, layoutL1H, Catlass::Arch::PositionL1{});
        CopyGmToL1H{}(tensorL1H, blockH);
        SetFlag<HardEvent::MTE2_MTE1>(hEvent);

        // Publish Q @ K^T from L0C to the owner AIV's A_raw UB slot.
        auto qktLayout = tla::MakeLayoutL0C(m, m);
        auto qktTensor = tla::MakeTensor(qktL0C, qktLayout, Catlass::Arch::PositionL0C{});
        auto qktTile = GetTile(qktTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
        auto layoutARawUb = tla::MakeLayout<float, LayoutRM>(m, m);
        auto tensorARawUb = tla::MakeTensor(
            resource_.ubBuf.template GetBufferByByte<float>(ChunkFwdOARawOffset(localSlot)), layoutARawUb,
            Catlass::Arch::PositionUB{});
        using CopyQktToUb = typename DirectTileCopyCC::template CopyL0CToDst<decltype(tensorARawUb)>;
        CopyQktToUb copyQktToUb;
        SetFlag<HardEvent::M_FIX>(qktCSlot);
        WaitFlag<HardEvent::M_FIX>(qktCSlot);
        copyQktToUb(tensorARawUb, qktTile, static_cast<uint8_t>(ownerSubBlock), 0);
        SetFlag<HardEvent::FIX_M>(qktCSlot);
        WaitFlag<HardEvent::MTE2_MTE1>(hEvent);

        // Move Q/H from L1 to L0 and compute Q @ H.
        const uint32_t qhASlot = l0ASlot_;
        const uint32_t qhBSlot = l0BSlot_;
        const uint32_t qhCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        using LayoutTagQhL1A = typename TileCopyQH::LayoutTagL1A;
        using LayoutTagQhL1B = typename TileCopyQH::LayoutTagL1B;
        using LayoutTagQhL0A = typename TileCopyQH::LayoutTagL0A;
        using LayoutTagQhL0B = typename TileCopyQH::LayoutTagL0B;
        using CopyQhL1ToL0A = typename TileCopyQH::CopyL1ToL0A;
        using CopyQhL1ToL0B = typename TileCopyQH::CopyL1ToL0B;
        using QhTileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagQhL1A>;
        LocalTensor<Element> qhL0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(qhASlot));
        LocalTensor<Element> qhL0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(qhBSlot));
        LocalTensor<float> qhL0C =
            resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(qhCSlot));
        auto layoutQhL1Q = tla::MakeLayout<Element, LayoutTagQhL1A>(m, kK);
        auto layoutQhL1H = tla::MakeLayout<Element, LayoutTagQhL1B>(kK, kV);
        auto layoutQhL0A = tla::MakeLayout<Element, LayoutTagQhL0A>(m, kK);
        auto layoutQhL0B = tla::MakeLayout<Element, LayoutTagQhL0B>(kK, kV);
        auto layoutQhL0C = tla::MakeLayoutL0C(m, kV);
        auto tensorQhL1Q = tla::MakeTensor(l1Q, layoutQhL1Q, Catlass::Arch::PositionL1{});
        auto tensorQhL1H = tla::MakeTensor(l1H, layoutQhL1H, Catlass::Arch::PositionL1{});
        auto tensorQhL0A = tla::MakeTensor(qhL0A, layoutQhL0A, Catlass::Arch::PositionL0A{});
        auto tensorQhL0B = tla::MakeTensor(qhL0B, layoutQhL0B, Catlass::Arch::PositionL0B{});
        auto tensorQhL0C = tla::MakeTensor(qhL0C, layoutQhL0C, Catlass::Arch::PositionL0C{});
        auto tileQhL1Q = GetTile(tensorQhL1Q, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileQhL1H = GetTile(tensorQhL1H, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        auto tileQhL0A = GetTile(tensorQhL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileQhL0B = GetTile(tensorQhL0B, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        auto tileQhL0C = GetTile(tensorQhL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));
        WaitFlag<HardEvent::M_MTE1>(L0AEvent(qhASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(qhBSlot));
        WaitFlag<HardEvent::FIX_M>(qhCSlot);
        CopyQhL1ToL0A{}(tileQhL0A, tileQhL1Q);
        CopyQhL1ToL0B{}(tileQhL0B, tileQhL1H);
        SetFlag<HardEvent::MTE1_M>(qhCSlot);
        WaitFlag<HardEvent::MTE1_M>(qhCSlot);
        QhTileMmad{}(tileQhL0C, tileQhL0A, tileQhL0B, m, kV, kK, true, 0);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(L0AEvent(qhASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(qhBSlot));

        // Publish Q @ H from L0C to the owner AIV's O_s_raw UB slot.
        auto qhLayout = tla::MakeLayoutL0C(m, kV);
        auto qhTensor = tla::MakeTensor(qhL0C, qhLayout, Catlass::Arch::PositionL0C{});
        auto qhTile = GetTile(qhTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));
        auto layoutOSRawUb = tla::MakeLayout<float, LayoutRM>(m, kV);
        auto tensorOSRawUb = tla::MakeTensor(
            resource_.ubBuf.template GetBufferByByte<float>(ChunkFwdOOSRawOffset(localSlot)), layoutOSRawUb,
            Catlass::Arch::PositionUB{});
        using CopyQhToUb = typename DirectTileCopyRM::template CopyL0CToDst<decltype(tensorOSRawUb)>;
        CopyQhToUb copyQhToUb;
        SetFlag<HardEvent::M_FIX>(qhCSlot);
        WaitFlag<HardEvent::M_FIX>(qhCSlot);
        copyQhToUb(tensorOSRawUb, qhTile, static_cast<uint8_t>(ownerSubBlock), 0);
        SetFlag<HardEvent::FIX_M>(qhCSlot);

        if (loadQK) {
            SetFlag<HardEvent::MTE1_MTE2>(qkEvent);
        }
        SetFlag<HardEvent::MTE1_MTE2>(hEvent);

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
    }

    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR h_;
    GM_ADDR g_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkOffsets_;
    GM_ADDR o_;
    GM_ADDR workspace_;
    ChunkFwdOTilingData tiling_{};
    Catlass::Arch::Resource<ArchTag> resource_;
    GlobalTensor<Element> qGm_;
    GlobalTensor<Element> kGm_;
    GlobalTensor<Element> vGm_;
    GlobalTensor<Element> hGm_;
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{CHUNK_FWD_O_VEC_TO_CUBE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CHUNK_FWD_O_CUBE_TO_VEC_READY_FLAG};
    uint32_t l0ASlot_ = 0U;
    uint32_t l0BSlot_ = 0U;
    uint32_t l0CSlot_ = 0U;
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_CUBE_H
