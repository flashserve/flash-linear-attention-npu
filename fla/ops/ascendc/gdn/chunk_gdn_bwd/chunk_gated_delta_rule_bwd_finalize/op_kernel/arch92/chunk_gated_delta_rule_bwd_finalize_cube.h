#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_CUBE_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_CUBE_H

#include "chunk_gated_delta_rule_bwd_finalize_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN::FinalizeArch92 {

template <typename DT>
class ChunkGatedDeltaRuleBwdFinalizeCube {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR dO, GM_ADDR du, GM_ADDR h, GM_ADDR dh, GM_ADDR a,
        GM_ADDR kbg, GM_ADDR vb, GM_ADDR dA0, GM_ADDR ds0Out,
        GM_ADDR dkbOut, GM_ADDR dkbTOut, GM_ADDR dsIn,
        GM_ADDR doGIn, GM_ADDR vDecayIn, GM_ADDR workspace,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling)
    {
        q_ = q;
        k_ = k;
        vNew_ = vNew;
        do_ = dO;
        du_ = du;
        h_ = h;
        dh_ = dh;
        a_ = a;
        kbg_ = kbg;
        vb_ = vb;
        dA0_ = dA0;
        ds0Out_ = ds0Out;
        dkbOut_ = dkbOut;
        dkbTOut_ = dkbTOut;
        dsIn_ = dsIn;
        doGIn_ = doGIn;
        vDecayIn_ = vDecayIn;
        workspace_ = workspace;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        stateChunkNum_ = tiling_->isVariable != 0 ? tiling_->totalChunkNum : tiling_->chunkNumForT;
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
    }

    __aicore__ inline void Process()
    {
        if (tiling_->headRatio > HEAD_COUNT) {
            ProcessImpl<true>();
        } else {
            ProcessImpl<false>();
        }
    }

    template <bool WINDOWED>
    __aicore__ inline void ProcessImpl()
    {
        Catlass::Arch::Resource<ArchTag> resource;

        AscendC::TPipe eventPipe;
        for (uint32_t slot = 0; slot < HEAD_COUNT; ++slot) {
            inputFixToMte2_[slot] = eventPipe.AllocEventID<AscendC::HardEvent::FIX_MTE2>();
        }
        // Two resident-event groups and three independent L0 slots.
        for (uint32_t i = 0; i < 6; ++i) {
            MTE1_MTE2Events_[i] = eventPipe.AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
        }
        for (uint32_t i = 0; i < 6; ++i) {
            MTE2_MTE1Events_[i] = eventPipe.AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
        }
        for (uint32_t i = 0; i < 6; ++i) {
            FIX_MTE1Events_[i] = eventPipe.AllocEventID<AscendC::HardEvent::FIX_MTE1>();
        }
        for (uint32_t i = 0; i < 6; ++i) {
            M_MTE1Events_[i] = eventPipe.AllocEventID<AscendC::HardEvent::M_MTE1>();
        }
        for (uint32_t i = 0; i < 3; ++i) {
            MTE1_MEvents_[i] = eventPipe.AllocEventID<AscendC::HardEvent::MTE1_M>();
        }
        for (uint32_t i = 0; i < 3; ++i) {
            FIX_MEvents_[i] = eventPipe.AllocEventID<AscendC::HardEvent::FIX_M>();
        }
        for (uint32_t i = 0; i < 3; ++i) {
            M_FIXEvents_[i] = eventPipe.AllocEventID<AscendC::HardEvent::M_FIX>();
        }
        AscendC::LocalTensor<DT> hL1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            hL1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_H_OFFSET + slot * STATE_BYTES);
        }
        AscendC::LocalTensor<DT> duL1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            duL1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_DU_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> vbL1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            vbL1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_VB_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> kbgL1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            kbgL1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_KBG_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> dw0L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dw0L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_DW0_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> aL1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            aL1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_A_OFFSET + slot * MATRIX_BYTES);
        }
        AscendC::LocalTensor<DT> dA0L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dA0L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_DA0_OFFSET + slot * MATRIX_BYTES);
        }
        AscendC::LocalTensor<DT> kStage5L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            kStage5L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_K_A_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> doStage11L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            doStage11L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_DO_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> vNewStage11L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            vNewStage11L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_VNEW_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> qStage13L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            qStage13L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_Q_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> dhStage13L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dhStage13L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DH_OFFSET + slot * STATE_BYTES);
        }
        AscendC::LocalTensor<DT> dsStage13L1[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dsStage13L1[slot] = resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DS_OFFSET + slot * MATRIX_BYTES);
        }
        AscendC::LocalTensor<DT> l0A[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            l0A[slot] = resource.l0ABuf.template GetBufferByByte<DT>(0 + slot * L0_A_BYTES);
        }
        AscendC::LocalTensor<DT> l0B[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            l0B[slot] = resource.l0BBuf.template GetBufferByByte<DT>(0 + slot * L0_B_BYTES);
        }
        AscendC::LocalTensor<ElementAccumulator> l0C[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            l0C[slot] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0 + slot * L0_C_BYTES);
        }
        AscendC::LocalTensor<DT> dAuCv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dAuCv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_DAU_OFFSET + slot * MATRIX_BYTES);
        }
        AscendC::LocalTensor<DT> dvbCv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dvbCv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_DVB_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> dAw0Cv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dAw0Cv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_DAW0_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> a2Cv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            a2Cv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_A2_OFFSET + slot * MATRIX_BYTES);
        }
        AscendC::LocalTensor<DT> dA2Cv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dA2Cv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_DA2_OFFSET + slot * MATRIX_BYTES);
        }
        AscendC::LocalTensor<DT> dkbg0Cv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dkbg0Cv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_DKBG0_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> ds0Cv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            ds0Cv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE12_DS_OFFSET + slot * MATRIX_BYTES);
        }
        AscendC::LocalTensor<DT> dk0Cv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dk0Cv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DK0_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> dqIntraCv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dqIntraCv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DQ_INTRA_OFFSET + slot * VECTOR_BYTES);
        }
        AscendC::LocalTensor<DT> dkIntraCv[SLOT_COUNT];
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            dkIntraCv[slot] = resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DK_INTRA_OFFSET + slot * VECTOR_BYTES);
        }

        uint32_t eventIndex = 0;
        int64_t taskIdx = 0;
        int64_t chunkTaskIdx = 0;
        int64_t headGroupIdx = 0;
        int64_t windowBank = 0;
        int64_t hvBase = 0;
        int64_t taskCount = 0;
        int64_t headOffset = 0;
        ChunkInfo chunk;

        for (eventIndex = 0; eventIndex < L1_EVENT_COUNT; ++eventIndex) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[eventIndex]);
        }
        for (eventIndex = 0; eventIndex < SLOT_COUNT; ++eventIndex) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[2 * eventIndex]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[2 * eventIndex + 1]);
        }
        for (eventIndex = 0; eventIndex < SLOT_COUNT; ++eventIndex) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[eventIndex]);
        }

        for (taskIdx = coreIdx_; taskIdx < tiling_->taskNum; taskIdx += coreNum_) {
            chunkTaskIdx = taskIdx / tiling_->headGroupNum;
            headGroupIdx = taskIdx % tiling_->headGroupNum;
            GetChunkInfo(chunkTaskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            const int64_t windowCount = WINDOWED ?
                (tiling_->headRatio + HEAD_COUNT - 1) / HEAD_COUNT : 1;
            for (int64_t window = 0; window < windowCount; ++window) {
                hvBase = WINDOWED ?
                    headGroupIdx * tiling_->headRatio + window * HEAD_COUNT :
                    headGroupIdx * tiling_->taskGroupSize;
                taskCount = Min(static_cast<int64_t>(HEAD_COUNT), WINDOWED ?
                    tiling_->headRatio - window * HEAD_COUNT : tiling_->taskGroupSize);
                workspaceGroupRound_ = windowBank;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage1Head(
                        hL1, duL1, vbL1, dw0L1, l0A, l0B, l0C, dAuCv,
                        chunk, hvBase + headOffset, headOffset);
                }
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage3Head(
                        aL1, duL1, kbgL1, dw0L1, l0A, l0B, l0C, dvbCv, dAw0Cv,
                        chunk, hvBase + headOffset, headOffset);
                    // Publish readiness on FIX after all S3 copies for this head.
                    SetReady<PIPE_FIX>(FinalizeCvFlag::S3Ready);
                }
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage5Head(
                        aL1, dA0L1, kStage5L1,
                        l0A, l0B, l0C, a2Cv,
                        chunk, hvBase + headOffset, headOffset);
                }
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage7Head(
                        aL1, dA0L1, dw0L1, l0A, l0B, l0C, dA2Cv, dkbg0Cv,
                        chunk, hvBase + headOffset, headOffset);
                    SetReady<PIPE_FIX>(FinalizeCvFlag::S7Ready);
                }
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage9Head(
                        dA0L1, vbL1, kStage5L1, l0A, l0B, l0C,
                        chunk, hvBase + headOffset, headOffset);
                }
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage11Head(
                        doStage11L1, vNewStage11L1, l0A, l0B, l0C, ds0Cv,
                        chunk, hvBase + headOffset, headOffset);
                    SetReady<PIPE_FIX>(FinalizeCvFlag::S11Ready);
                }
                // S13 dk_base spans the compact ds slots of adjacent heads.
                // All S12 MTE3 consumers must finish before any of these CV writes.
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    WaitVectorInput(FinalizeCvFlag::S12Ready, static_cast<uint32_t>(headOffset));
                }
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage13Head(
                        qStage13L1, dhStage13L1, dsStage13L1,
                        kStage5L1, hL1, vNewStage11L1, vbL1, l0A, l0B, l0C,
                        dk0Cv, dqIntraCv, dkIntraCv,
                        chunk, hvBase + headOffset, headOffset);
                    SetReady<PIPE_FIX>(FinalizeCvFlag::S13Ready);
                }
                WaitReady<PIPE_FIX>(FinalizeCvFlag::S14Free);
                ++windowBank;
            }
        }

        for (eventIndex = 0; eventIndex < L1_EVENT_COUNT; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[eventIndex]);
        }
        for (eventIndex = 0; eventIndex < SLOT_COUNT; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[2 * eventIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[2 * eventIndex + 1]);
        }
        for (eventIndex = 0; eventIndex < SLOT_COUNT; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[eventIndex]);
        }
        for (uint32_t i = 0; i < 6; ++i) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[i]);
        }
        for (uint32_t i = 0; i < 6; ++i) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[i]);
        }
        for (uint32_t i = 0; i < 6; ++i) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::FIX_MTE1>(FIX_MTE1Events_[i]);
        }
        for (uint32_t i = 0; i < 6; ++i) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[i]);
        }
        for (uint32_t i = 0; i < 3; ++i) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[i]);
        }
        for (uint32_t i = 0; i < 3; ++i) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::FIX_M>(FIX_MEvents_[i]);
        }
        for (uint32_t i = 0; i < 3; ++i) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::M_FIX>(M_FIXEvents_[i]);
        }
        for (uint32_t slot = 0; slot < HEAD_COUNT; ++slot) {
            eventPipe.ReleaseEventID<AscendC::HardEvent::FIX_MTE2>(inputFixToMte2_[slot]);
        }
    }

private:
    using ArchTag = Catlass::Arch::Ascend960DT;
    using LayoutRowMajor = Catlass::layout::RowMajor;
    using LayoutColumnMajor = Catlass::layout::ColumnMajor;
    using TileCopyDau = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRowMajor, DT, LayoutColumnMajor, DT, LayoutRowMajor>;
    using TileCopyStateVFirst = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyDw0 = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyDvb = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutColumnMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyStage9DkbT = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutColumnMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyToUb = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    template <class Dst>
    using CopyL0CToUb = typename TileCopyToUb::template CopyL0CToDst<Dst>;
    using ElementAccumulator = typename TileCopyDau::ElementAccumulator;
    using CopyDauL1ToL0A = typename TileCopyDau::CopyL1ToL0A;
    using CopyDauL1ToL0B = typename TileCopyDau::CopyL1ToL0B;
    using CopyStateVFirstL1ToL0B = typename TileCopyStateVFirst::CopyL1ToL0B;
    using CopyDw0L1ToL0A = typename TileCopyDw0::CopyL1ToL0A;
    using CopyDw0L1ToL0B = typename TileCopyDw0::CopyL1ToL0B;
    using CopyStage9DkbTL1ToL0A = typename TileCopyStage9DkbT::CopyL1ToL0A;
    using CopyStage9DkbTL1ToL0B = typename TileCopyStage9DkbT::CopyL1ToL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, DT, typename TileCopyDw0::LayoutTagL1A>;
    using TileMmadDvb = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, DT, typename TileCopyDvb::LayoutTagL1A>;

    static constexpr uint32_t VECTOR_ELEMS = CHUNK_SIZE_64 * K_SIZE_128;
    static constexpr uint32_t MATRIX_ELEMS = CHUNK_SIZE_64 * CHUNK_SIZE_64;
    static constexpr uint32_t STATE_ELEMS = K_SIZE_128 * V_SIZE_128;
    static constexpr uint32_t VECTOR_BYTES = VECTOR_ELEMS * sizeof(DT);
    static constexpr uint32_t MATRIX_BYTES = MATRIX_ELEMS * sizeof(DT);
    static constexpr uint32_t STATE_BYTES = STATE_ELEMS * sizeof(DT);

    static constexpr uint32_t L1_H_OFFSET = L1::H;
    static constexpr uint32_t L1_A_OFFSET = L1::A;
    static constexpr uint32_t L1_DU_OFFSET = L1::DU;
    static constexpr uint32_t L1_VB_OFFSET = L1::VB;
    static constexpr uint32_t L1_KBG_OFFSET = L1::KBG;
    static constexpr uint32_t L1_DW0_OFFSET = L1::DW0;
    static constexpr uint32_t L1_STAGE5_DA0_OFFSET = L1::DA;
    static constexpr uint32_t L1_STAGE5_K_A_OFFSET = L1::DU;
    static constexpr uint32_t L1_STAGE11_DO_OFFSET = L1::A;
    static constexpr uint32_t L1_STAGE11_VNEW_OFFSET = L1::KBG;
    static constexpr uint32_t L1_STAGE13_Q_OFFSET = L1::A;
    static constexpr uint32_t L1_STAGE13_DH_OFFSET = L1::DH;
    static constexpr uint32_t L1_STAGE13_DS_OFFSET = L1::DA;
    static constexpr uint32_t UB_DAU_OFFSET = UB::DAU;
    static constexpr uint32_t UB_DVB_OFFSET = UB::VB;
    static constexpr uint32_t UB_DAW0_OFFSET = UB::KBG;
    static constexpr uint32_t UB_A2_OFFSET = UB::KBG;
    static constexpr uint32_t UB_DA2_OFFSET = UB::DA2;
    static constexpr uint32_t UB_DKBG0_OFFSET = UB::DKBG0;
    static constexpr uint32_t UB_STAGE12_DS_OFFSET = UB::DS;
    static constexpr uint32_t UB_STAGE13_DK0_OFFSET = UB::DK_BASE;
    static constexpr uint32_t UB_STAGE13_DQ_INTRA_OFFSET = UB::DQ_HV;
    static constexpr uint32_t UB_STAGE13_DK_INTRA_OFFSET = UB::DK_INTRA;

    static constexpr uint32_t L0_A_BYTES = L0_SLOT_BYTES;
    static constexpr uint32_t L0_B_BYTES = L0_SLOT_BYTES;
    static constexpr uint32_t L0_C_BYTES = L0_SLOT_BYTES;
    static constexpr AscendC::FixpipeConfig FIXPIPE_NZ_L1_CONFIG = {
        AscendC::CO2Layout::NZ, false};

    static constexpr auto GM_DU_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<V_SIZE_128>{});
    __aicore__ inline auto MakePublicQkLayout() const
    {
        const int64_t rowStride =
            (tiling_->layout == FinalizeLayout::BSND)
                ? tiling_->HK * K_SIZE_128 : K_SIZE_128;
        return tla::MakeLayoutFromTag(LayoutRowMajor(CHUNK_SIZE_64, K_SIZE_128, rowStride));
    }
    static constexpr auto GM_STATE_TRANSPOSE_LAYOUT = tla::MakeLayout<DT, LayoutColumnMajor>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    static constexpr auto GM_STATE_V_FIRST_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    static constexpr auto GM_A_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<CHUNK_SIZE_64>{});
    static constexpr auto GM_VECTOR_TRANSPOSE_LAYOUT = tla::MakeLayout<DT, LayoutColumnMajor>(
        tla::Int<K_SIZE_128>{}, tla::Int<CHUNK_SIZE_64>{});
    static constexpr auto L1_DU_LAYOUT = tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<V_SIZE_128>{});
    static constexpr auto L1_VB_LAYOUT = tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
        tla::Int<V_SIZE_128>{}, tla::Int<CHUNK_SIZE_64>{});
    static constexpr auto L1_STATE_TRANSPOSE_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    static constexpr auto L1_STATE_V_FIRST_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL1B>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    static constexpr auto UB_DAU_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<CHUNK_SIZE_64>{});
    static constexpr auto UB_DVB_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<V_SIZE_128>{});

    static_assert(UB_TOTAL_BYTES <= ArchTag::UB_SIZE);
    static_assert(L1_TOTAL_BYTES <= ArchTag::L1_SIZE);
    static_assert(L0_A_BYTES * SLOT_COUNT <= ArchTag::L0A_SIZE);
    static_assert(L0_B_BYTES * SLOT_COUNT <= ArchTag::L0B_SIZE);
    static_assert(L0_C_BYTES * SLOT_COUNT <= ArchTag::L0C_SIZE);
    static_assert(L1_DW0_OFFSET + HEAD_COUNT * VECTOR_BYTES == L1_TOTAL_BYTES);
    static_assert(UB_DAU_OFFSET + HEAD_COUNT * MATRIX_BYTES == UB::KBG);

    __aicore__ inline void ProcessStage1Head(
        AscendC::LocalTensor<DT> (&hL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&duL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&vbL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&dw0L1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&l0A)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&l0B)[SLOT_COUNT],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dAuCv)[SLOT_COUNT],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t mAlign = (chunkLen + 15U) / 16U * 16U;
        const uint32_t localHv = static_cast<uint32_t>(headOffset);
        const uint32_t duEvent = static_cast<uint32_t>(headOffset);
        const uint32_t hEvent = HEAD_COUNT + static_cast<uint32_t>(headOffset);
        const int64_t tokenBase =
            ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * tiling_->V;
        const int64_t stateBase =
            ((chunk.bIdx * stateChunkNum_ + chunk.stateChunkIdx) * tiling_->HV + hv) *
            tiling_->K * tiling_->V;

        AscendC::GlobalTensor<DT> gmDu;
        AscendC::GlobalTensor<DT> gmH;
        AscendC::GlobalTensor<DT> gmVb;
        gmDu.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(du_) + tokenBase);
        gmH.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(h_) + stateBase);
        gmVb.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT *>(vb_) + WorkspaceHeadOffset(headOffset));
        auto tensorGmDu = tla::MakeTensor(gmDu, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmVb = tla::MakeTensor(
            gmVb, GM_VECTOR_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmDu = tla::GetTile(
            tensorGmDu, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto blockGmVb = tla::GetTile(
            tensorGmVb, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, chunkLen));
        auto tensorL1Du = tla::MakeTensor(
            duL1[headOffset], L1_DU_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1HTranspose = tla::MakeTensor(
            hL1[headOffset], L1_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1HVFirst = tla::MakeTensor(
            hL1[headOffset], L1_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1Vb = tla::MakeTensor(
            vbL1[headOffset], L1_VB_LAYOUT, Catlass::Arch::PositionL1{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[duEvent]);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmDu)>{}(tensorL1Du, blockGmDu);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[hEvent]);
        if (tiling_->stateVFirst != 0) {
            auto tensorGmH = tla::MakeTensor(
                gmH, GM_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmH = tla::GetTile(
                tensorGmH, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyStateVFirst::template CopyGmToL1B<decltype(blockGmH)>{}(
                tensorL1HVFirst, blockGmH);
        } else {
            auto tensorGmH = tla::MakeTensor(
                gmH, GM_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmH = tla::GetTile(
                tensorGmH, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyDau::template CopyGmToL1B<decltype(blockGmH)>{}(
                tensorL1HTranspose, blockGmH);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[hEvent]);

        WaitVectorInput(FinalizeCvFlag::S0Ready, localHv);
        typename TileCopyDau::template CopyGmToL1B<decltype(blockGmVb)>{}(tensorL1Vb, blockGmVb);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[duEvent]);

        const uint32_t dAuL0ASlot = l0ASlot_;
        const uint32_t dAuL0BSlot = l0BSlot_;
        const uint32_t dAuL0CSlot = l0CSlot_;
        const uint32_t dAuL0AEvent = 2 * dAuL0ASlot;
        const uint32_t dAuL0BEvent = 2 * dAuL0BSlot + 1;
        l0ASlot_ = NextSlot(l0ASlot_);
        l0BSlot_ = NextSlot(l0BSlot_);
        l0CSlot_ = NextSlot(l0CSlot_);
        auto tensorDauL0A = tla::MakeTensor(
            l0A[dAuL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, V_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorDauL0B = tla::MakeTensor(
            l0B[dAuL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(V_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDauL1A = tla::GetTile(
            tensorL1Du, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        auto tileDauL1B = tla::GetTile(
            tensorL1Vb, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, chunkLen));
        auto tensorDauL0C = tla::MakeTensor(
            l0C[dAuL0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileDauL0C = tla::GetTile(
            tensorDauL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[duEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAuL0AEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAuL0BEvent]);
        CopyDauL1ToL0A{}(tensorDauL0A, tileDauL1A);
        CopyDauL1ToL0B{}(tensorDauL0B, tileDauL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dAuL0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dAuL0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dAuL0CSlot]);
        tileMmad_(tileDauL0C, tensorDauL0A, tensorDauL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAuL0AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAuL0BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dAuL0CSlot]);

        auto tensorDauCv = tla::MakeTensor(
            dAuCv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDauCv = tla::GetTile(
            tensorDauCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, FixpipeBf16Columns(chunkLen)));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dAuL0CSlot]);
        CopyL0CToUb<decltype(blockDauCv)>{}(blockDauCv, tensorDauL0C, false, uint8_t{0});
        SetReady<PIPE_FIX>(FinalizeCvFlag::S1Ready);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dAuL0CSlot]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[hEvent]);
        // N=64 keeps every L0A/L0B/L0C slot within 16 KiB.
        for (uint32_t nOffset = 0; nOffset < K_SIZE_128; nOffset += 64) {
            constexpr uint32_t nSize = 64;
            const uint32_t dw0L0CSlot = l0CSlot_;
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tensorDw0L0C = tla::MakeTensor(
                l0C[dw0L0CSlot], tla::MakeLayoutL0C(mActual, nSize),
                Catlass::Arch::PositionL0C{});
            auto tileDw0L0C = tla::GetTile(
                tensorDw0L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            const uint32_t dw0L0ASlot = l0ASlot_;
            const uint32_t dw0L0BSlot = l0BSlot_;
            const uint32_t dw0L0AEvent = 2 * dw0L0ASlot;
            const uint32_t dw0L0BEvent = 2 * dw0L0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            auto tensorDw0L0A = tla::MakeTensor(
                l0A[dw0L0ASlot],
                tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0A>(mActual, V_SIZE_128),
                Catlass::Arch::PositionL0A{});
            auto tensorDw0L0BTranspose = tla::MakeTensor(
                l0B[dw0L0BSlot],
                tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(V_SIZE_128, nSize),
                Catlass::Arch::PositionL0B{});
            auto tensorDw0L0BVFirst = tla::MakeTensor(
                l0B[dw0L0BSlot],
                tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL0B>(V_SIZE_128, nSize),
                Catlass::Arch::PositionL0B{});
            auto tileDw0L1A = tla::GetTile(
                tensorL1Du, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dw0L0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dw0L0BEvent]);
            CopyDw0L1ToL0A{}(tensorDw0L0A, tileDw0L1A);
            if (tiling_->stateVFirst != 0) {
                auto tileDw0L1B = tla::GetTile(
                    tensorL1HVFirst, tla::MakeCoord(0, nOffset), tla::MakeShape(V_SIZE_128, nSize));
                CopyStateVFirstL1ToL0B{}(tensorDw0L0BVFirst, tileDw0L1B);
            } else {
                auto tileDw0L1B = tla::GetTile(
                    tensorL1HTranspose, tla::MakeCoord(0, nOffset), tla::MakeShape(V_SIZE_128, nSize));
                CopyDauL1ToL0B{}(tensorDw0L0BTranspose, tileDw0L1B);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dw0L0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dw0L0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dw0L0CSlot]);
            if (tiling_->stateVFirst != 0) {
                tileMmad_(tileDw0L0C, tensorDw0L0A, tensorDw0L0BVFirst, true, 0);
            } else {
                tileMmad_(tileDw0L0C, tensorDw0L0A, tensorDw0L0BTranspose, true, 0);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dw0L0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dw0L0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dw0L0CSlot]);

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dw0L0CSlot]);
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> dw0FixpipeParams;
            dw0FixpipeParams.nSize = nSize;
            dw0FixpipeParams.mSize = chunkLen;
            dw0FixpipeParams.srcStride = mAlign;
            dw0FixpipeParams.dstStride = CHUNK_SIZE_64 * 16U;
            dw0FixpipeParams.quantPre = QuantMode_t::F322BF16;
            dw0FixpipeParams.unitFlag = 0;
            AscendC::Fixpipe<DT, ElementAccumulator, FIXPIPE_NZ_L1_CONFIG>(
                dw0L1[headOffset][nOffset * CHUNK_SIZE_64], l0C[dw0L0CSlot], dw0FixpipeParams);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dw0L0CSlot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[duEvent]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[hEvent]);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(FIX_MTE1Events_[static_cast<uint32_t>(headOffset) + HEAD_COUNT]);
    }

    __aicore__ inline void ProcessStage3Head(
        AscendC::LocalTensor<DT> (&aL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&duL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&kbgL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&dw0L1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&l0A)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&l0B)[SLOT_COUNT],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dvbCv)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dAw0Cv)[SLOT_COUNT],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t mAlign = (chunkLen + 15U) / 16U * 16U;
        const uint32_t localHv = static_cast<uint32_t>(headOffset);
        const uint32_t aEvent = static_cast<uint32_t>(headOffset);
        const int64_t aBase =
            ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * CHUNK_SIZE_64;

        AscendC::GlobalTensor<DT> gmA;
        AscendC::GlobalTensor<DT> gmKbg;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(a_) + aBase);
        gmKbg.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT *>(kbg_) + WorkspaceHeadOffset(headOffset));
        auto tensorGmA = tla::MakeTensor(gmA, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmKbg = tla::MakeTensor(
            gmKbg, GM_VECTOR_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmA = tla::GetTile(
            tensorGmA, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmKbg = tla::GetTile(
            tensorGmKbg, tla::MakeCoord(0, 0), tla::MakeShape(K_SIZE_128, chunkLen));
        auto tensorL1A = tla::MakeTensor(
            aL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[aEvent]);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmA)>{}(tensorL1A, blockGmA);
        auto tensorKbgL1 = tla::MakeTensor(
            kbgL1[headOffset], L1_VB_LAYOUT, Catlass::Arch::PositionL1{});
        typename TileCopyDau::template CopyGmToL1B<decltype(blockGmKbg)>{}(tensorKbgL1, blockGmKbg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[aEvent]);

        const uint32_t dAwL0ASlot = l0ASlot_;
        const uint32_t dAwL0BSlot = l0BSlot_;
        const uint32_t dAwL0CSlot = l0CSlot_;
        const uint32_t dAwL0AEvent = 2 * dAwL0ASlot;
        const uint32_t dAwL0BEvent = 2 * dAwL0BSlot + 1;
        l0ASlot_ = NextSlot(l0ASlot_);
        l0BSlot_ = NextSlot(l0BSlot_);
        l0CSlot_ = NextSlot(l0CSlot_);
        auto tensorDw0L1 = tla::MakeTensor(
            dw0L1[headOffset], L1_DU_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorDAwL0A = tla::MakeTensor(
            l0A[dAwL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorDAwL0B = tla::MakeTensor(
            l0B[dAwL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(K_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDw0L1 = tla::GetTile(
            tensorDw0L1, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        auto tileKbgL1 = tla::GetTile(
            tensorKbgL1, tla::MakeCoord(0, 0), tla::MakeShape(K_SIZE_128, chunkLen));
        auto tensorDAwL0C = tla::MakeTensor(
            l0C[dAwL0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileDAwL0C = tla::GetTile(
            tensorDAwL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(FIX_MTE1Events_[static_cast<uint32_t>(headOffset) + HEAD_COUNT]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[aEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAwL0AEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAwL0BEvent]);
        CopyDauL1ToL0A{}(tensorDAwL0A, tileDw0L1);
        CopyDauL1ToL0B{}(tensorDAwL0B, tileKbgL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dAwL0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dAwL0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dAwL0CSlot]);
        tileMmad_(tileDAwL0C, tensorDAwL0A, tensorDAwL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAwL0AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dAwL0BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dAwL0CSlot]);
        auto tensorDAwCv = tla::MakeTensor(
            dAw0Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDAwCv = tla::GetTile(
            tensorDAwCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, FixpipeBf16Columns(chunkLen)));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dAwL0CSlot]);
        CopyL0CToUb<decltype(blockDAwCv)>{}(blockDAwCv, tensorDAwL0C, false, uint8_t{0});
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dAwL0CSlot]);

        // N=64 keeps every L0A/L0B/L0C slot within 16 KiB.
        for (uint32_t nOffset = 0; nOffset < K_SIZE_128; nOffset += 64) {
            constexpr uint32_t nSize = 64;
            const uint32_t dvbL0ASlot = l0ASlot_;
            const uint32_t dvbL0BSlot = l0BSlot_;
            const uint32_t dvbL0CSlot = l0CSlot_;
            const uint32_t dvbL0AEvent = 2 * dvbL0ASlot;
            const uint32_t dvbL0BEvent = 2 * dvbL0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tensorDuL1B = tla::MakeTensor(
                duL1[headOffset],
                tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                    CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});
            auto tensorL1AForDvb = tla::MakeTensor(
                aL1[headOffset],
                tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1A>(
                    CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
            auto tensorDvbL0A = tla::MakeTensor(
                l0A[dvbL0ASlot],
                tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0A>(mActual, chunkLen),
                Catlass::Arch::PositionL0A{});
            auto tensorDvbL0B = tla::MakeTensor(
                l0B[dvbL0BSlot],
                tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0B>(chunkLen, nSize),
                Catlass::Arch::PositionL0B{});
            auto tileAL1 = tla::GetTile(
                tensorL1AForDvb, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
            auto tileDuL1B = tla::GetTile(
                tensorDuL1B, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            auto tensorDvbL0C = tla::MakeTensor(
                l0C[dvbL0CSlot], tla::MakeLayoutL0C(mActual, nSize),
                Catlass::Arch::PositionL0C{});
            auto tileDvbL0C = tla::GetTile(
                tensorDvbL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dvbL0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dvbL0BEvent]);
            typename TileCopyDvb::CopyL1ToL0A{}(tensorDvbL0A, tileAL1);
            typename TileCopyDvb::CopyL1ToL0B{}(tensorDvbL0B, tileDuL1B);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dvbL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dvbL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dvbL0CSlot]);
            TileMmadDvb{}(tileDvbL0C, tensorDvbL0A, tensorDvbL0B, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dvbL0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dvbL0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dvbL0CSlot]);
            auto tensorDvbCv = tla::MakeTensor(
                dvbCv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
            auto blockDvbCv = tla::GetTile(
                tensorDvbCv, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dvbL0CSlot]);
            CopyL0CToUb<decltype(blockDvbCv)>{}(blockDvbCv, tensorDvbL0C, false, uint8_t{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dvbL0CSlot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[aEvent]);
    }

    __aicore__ inline void ProcessStage5Head(
        AscendC::LocalTensor<DT> (&aL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&dA0L1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&kStage5L1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&l0A)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&l0B)[SLOT_COUNT],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&a2Cv)[SLOT_COUNT],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t mAlign = (chunkLen + 15U) / 16U * 16U;
        constexpr uint32_t tileBatch = 1;
        const uint32_t localHv = static_cast<uint32_t>(headOffset);
        const uint32_t l1Event = static_cast<uint32_t>(headOffset);
        const int64_t hk = hv / tiling_->headRatio;
        const int64_t kBase = (tiling_->layout == FinalizeLayout::BSND)
                                  ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HK + hk) * tiling_->K
                                  : ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * tiling_->K;

        AscendC::GlobalTensor<DT> gmDA0;
        AscendC::GlobalTensor<DT> gmK;
        gmDA0.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT *>(dA0_) + WorkspaceHeadOffset(headOffset));
        gmK.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(k_) + kBase);
        auto tensorGmDA0 = tla::MakeTensor(gmDA0, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmK = tla::MakeTensor(
            gmK,
            MakePublicQkLayout(),
            Catlass::Arch::PositionGM{});
        auto blockGmDA0 = tla::GetTile(
            tensorGmDA0, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmK = tla::GetTile(
            tensorGmK, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));

        auto tensorL1DA0 = tla::MakeTensor(
            dA0L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(
            kStage5L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[l1Event]);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmK)>{}(
            tensorL1K, blockGmK);
        WaitVectorInput(FinalizeCvFlag::S4Ready, l1Event);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmDA0)>{}(
            tensorL1DA0, blockGmDA0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[l1Event]);

        const uint32_t dA1L0ASlot = l0ASlot_;
        const uint32_t dA1L0BSlot = l0BSlot_;
        const uint32_t dA1L0CSlot = l0CSlot_;
        const uint32_t dA1L0AEvent = 2 * dA1L0ASlot;
        const uint32_t dA1L0BEvent = 2 * dA1L0BSlot + 1;
        l0ASlot_ = NextSlot(l0ASlot_);
        l0BSlot_ = NextSlot(l0BSlot_);
        l0CSlot_ = NextSlot(l0CSlot_);
        auto tensorL1AB = tla::MakeTensor(
            aL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorDA1L0A = tla::MakeTensor(
            l0A[dA1L0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDA1L0B = tla::MakeTensor(
            l0B[dA1L0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(chunkLen, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDA0L1 = tla::GetTile(
            tensorL1DA0, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileAL1B = tla::GetTile(
            tensorL1AB, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto tensorDA1L0C = tla::MakeTensor(
            l0C[dA1L0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[l1Event]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA1L0AEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA1L0BEvent]);
        CopyDauL1ToL0A{}(tensorDA1L0A, tileDA0L1);
        CopyDauL1ToL0B{}(tensorDA1L0B, tileAL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dA1L0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dA1L0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dA1L0CSlot]);
        // Each Stage 5 GEMM initializes a separate L0C slot. Use the CATLASS
        // batch overload to avoid the conditional PIPE_M barrier for small tiles.
        tileMmad_(tensorDA1L0C, tensorDA1L0A, tensorDA1L0B, mActual, chunkLen, chunkLen, tileBatch);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA1L0AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA1L0BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dA1L0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dA1L0CSlot]);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> dA1FixpipeParams;
        dA1FixpipeParams.nSize = mAlign;
        dA1FixpipeParams.mSize = mAlign;
        dA1FixpipeParams.srcStride = mAlign;
        dA1FixpipeParams.dstStride = CHUNK_SIZE_64 * 16U;
        dA1FixpipeParams.quantPre = QuantMode_t::F322BF16;
        dA1FixpipeParams.unitFlag = 0;
        AscendC::Fixpipe<DT, ElementAccumulator, FIXPIPE_NZ_L1_CONFIG>(
            dA0L1[headOffset], l0C[dA1L0CSlot], dA1FixpipeParams);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(FIX_MTE1Events_[static_cast<uint32_t>(headOffset) + HEAD_COUNT]);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dA1L0CSlot]);

        const uint32_t a2L0ASlot = l0ASlot_;
        const uint32_t a2L0BSlot = l0BSlot_;
        const uint32_t a2L0CSlot = l0CSlot_;
        const uint32_t a2L0AEvent = 2 * a2L0ASlot;
        const uint32_t a2L0BEvent = 2 * a2L0BSlot + 1;
        l0ASlot_ = NextSlot(l0ASlot_);
        l0BSlot_ = NextSlot(l0BSlot_);
        l0CSlot_ = NextSlot(l0CSlot_);
        auto tensorL1KT = tla::MakeTensor(
            kStage5L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
                K_SIZE_128, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorA2L0A = tla::MakeTensor(
            l0A[a2L0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorA2L0B = tla::MakeTensor(
            l0B[a2L0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(K_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileKL1 = tla::GetTile(
            tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        auto tileKTL1 = tla::GetTile(
            tensorL1KT, tla::MakeCoord(0, 0), tla::MakeShape(K_SIZE_128, chunkLen));
        auto tensorA2L0C = tla::MakeTensor(
            l0C[a2L0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[a2L0AEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[a2L0BEvent]);
        CopyDauL1ToL0A{}(tensorA2L0A, tileKL1);
        CopyDauL1ToL0B{}(tensorA2L0B, tileKTL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[a2L0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[a2L0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[a2L0CSlot]);
        tileMmad_(tensorA2L0C, tensorA2L0A, tensorA2L0B, mActual, chunkLen, K_SIZE_128, tileBatch);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[a2L0AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[a2L0BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[a2L0CSlot]);
        auto tensorA2Cv = tla::MakeTensor(
            a2Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockA2Cv = tla::GetTile(
            tensorA2Cv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, FixpipeBf16Columns(chunkLen)));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[a2L0CSlot]);
        CopyL0CToUb<decltype(blockA2Cv)>{}(blockA2Cv, tensorA2L0C, false, uint8_t{0});
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[a2L0CSlot]);
    }

    __aicore__ inline void ProcessStage7Head(
        AscendC::LocalTensor<DT> (&aL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&dA1L1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&dw0L1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&l0A)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&l0B)[SLOT_COUNT],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dA2Cv)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dkbg0Cv)[SLOT_COUNT],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t localHv = static_cast<uint32_t>(headOffset);
        const uint32_t l1Event = static_cast<uint32_t>(headOffset);
        auto tensorL1DA1 = tla::MakeTensor(
            dA1L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1B>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});

        auto tensorL1A = tla::MakeTensor(
            aL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});

        const uint32_t dA2L0ASlot = l0ASlot_;
        const uint32_t dA2L0BSlot = l0BSlot_;
        const uint32_t dA2L0CSlot = l0CSlot_;
        const uint32_t dA2L0AEvent = 2 * dA2L0ASlot;
        const uint32_t dA2L0BEvent = 2 * dA2L0BSlot + 1;
        l0ASlot_ = NextSlot(l0ASlot_);
        l0BSlot_ = NextSlot(l0BSlot_);
        l0CSlot_ = NextSlot(l0CSlot_);
        auto tensorDA2L0A = tla::MakeTensor(
            l0A[dA2L0ASlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDA2L0B = tla::MakeTensor(
            l0B[dA2L0BSlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0B>(chunkLen, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileAL1 = tla::GetTile(
            tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileDA1L1 = tla::GetTile(
            tensorL1DA1, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto tensorDA2L0C = tla::MakeTensor(
            l0C[dA2L0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileDA2L0C = tla::GetTile(
            tensorDA2L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(FIX_MTE1Events_[static_cast<uint32_t>(headOffset) + HEAD_COUNT]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA2L0AEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA2L0BEvent]);
        typename TileCopyDvb::CopyL1ToL0A{}(tensorDA2L0A, tileAL1);
        typename TileCopyDvb::CopyL1ToL0B{}(tensorDA2L0B, tileDA1L1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[l1Event]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dA2L0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dA2L0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dA2L0CSlot]);
        TileMmadDvb{}(tileDA2L0C, tensorDA2L0A, tensorDA2L0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA2L0AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dA2L0BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dA2L0CSlot]);
        auto tensorDA2Cv = tla::MakeTensor(
            dA2Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDA2Cv = tla::GetTile(
            tensorDA2Cv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, FixpipeBf16Columns(chunkLen)));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dA2L0CSlot]);
        CopyL0CToUb<decltype(blockDA2Cv)>{}(blockDA2Cv, tensorDA2L0C, false, uint8_t{0});
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dA2L0CSlot]);

        // N=64 keeps every L0A/L0B/L0C slot within 16 KiB.
        for (uint32_t nOffset = 0; nOffset < K_SIZE_128; nOffset += 64) {
            constexpr uint32_t nSize = 64;
            const uint32_t dkbgL0ASlot = l0ASlot_;
            const uint32_t dkbgL0BSlot = l0BSlot_;
            const uint32_t dkbgL0CSlot = l0CSlot_;
            const uint32_t dkbgL0AEvent = 2 * dkbgL0ASlot;
            const uint32_t dkbgL0BEvent = 2 * dkbgL0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tensorL1Dw0B = tla::MakeTensor(
                dw0L1[headOffset],
                tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1B>(
                    CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
            auto tensorDkbgL0A = tla::MakeTensor(
                l0A[dkbgL0ASlot],
                tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0A>(mActual, chunkLen),
                Catlass::Arch::PositionL0A{});
            auto tensorDkbgL0B = tla::MakeTensor(
                l0B[dkbgL0BSlot],
                tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0B>(chunkLen, nSize),
                Catlass::Arch::PositionL0B{});
            auto tileDw0L1B = tla::GetTile(
                tensorL1Dw0B, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            auto tensorDkbgL0C = tla::MakeTensor(
                l0C[dkbgL0CSlot], tla::MakeLayoutL0C(mActual, nSize),
                Catlass::Arch::PositionL0C{});
            auto tileDkbgL0C = tla::GetTile(
                tensorDkbgL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbgL0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbgL0BEvent]);
            typename TileCopyDvb::CopyL1ToL0A{}(tensorDkbgL0A, tileAL1);
            typename TileCopyDvb::CopyL1ToL0B{}(tensorDkbgL0B, tileDw0L1B);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkbgL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkbgL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkbgL0CSlot]);
            TileMmadDvb{}(tileDkbgL0C, tensorDkbgL0A, tensorDkbgL0B, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbgL0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbgL0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkbgL0CSlot]);
            auto tensorDkbgCv = tla::MakeTensor(
                dkbg0Cv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
            auto blockDkbgCv = tla::GetTile(
                tensorDkbgCv, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkbgL0CSlot]);
            CopyL0CToUb<decltype(blockDkbgCv)>{}(blockDkbgCv, tensorDkbgL0C, false, uint8_t{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkbgL0CSlot]);
        }
    }

    __aicore__ inline void ProcessStage9Head(
        AscendC::LocalTensor<DT> (&dAL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&kbL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&kL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&l0A)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&l0B)[SLOT_COUNT],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[SLOT_COUNT],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t l1Event = static_cast<uint32_t>(headOffset);
        const int64_t workspaceBase = WorkspaceHeadOffset(headOffset);

        AscendC::GlobalTensor<DT> gmDA;
        AscendC::GlobalTensor<DT> gmKb;
        AscendC::GlobalTensor<DT> gmDkb;
        AscendC::GlobalTensor<DT> gmDkbT;
        gmDA.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dA0_) + workspaceBase);
        gmKb.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + workspaceBase);
        gmDkb.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbOut_) + workspaceBase,
                             VECTOR_ELEMS);
        gmDkbT.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbTOut_) + workspaceBase,
                              VECTOR_ELEMS);
        auto tensorGmDA = tla::MakeTensor(gmDA, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmKb = tla::MakeTensor(gmKb, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmDkb = tla::MakeTensor(gmDkb, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmDkbT = tla::MakeTensor(gmDkbT, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmDA = tla::GetTile(
            tensorGmDA, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmKb = tla::GetTile(
            tensorGmKb, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));

        auto tensorL1DA = tla::MakeTensor(
            dAL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1Kb = tla::MakeTensor(
            kbL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL1B>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[l1Event]);
        typename TileCopyStage9DkbT::template CopyGmToL1B<decltype(blockGmKb)>{}(
            tensorL1Kb, blockGmKb);
        WaitVectorInput(FinalizeCvFlag::S8Ready, l1Event);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmDA)>{}(
            tensorL1DA, blockGmDA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[l1Event]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[l1Event]);
        // N=64 keeps every L0A/L0B/L0C slot within 16 KiB.
        for (uint32_t nOffset = 0; nOffset < K_SIZE_128; nOffset += 64) {
            constexpr uint32_t nSize = 64;
            auto blockGmDkb = tla::GetTile(
                tensorGmDkb, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            const uint32_t dkbL0ASlot = l0ASlot_;
            const uint32_t dkbL0BSlot = l0BSlot_;
            const uint32_t dkbL0CSlot = l0CSlot_;
            const uint32_t dkbL0AEvent = 2 * dkbL0ASlot;
            const uint32_t dkbL0BEvent = 2 * dkbL0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tensorL1K = tla::MakeTensor(
                kL1[headOffset],
                tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                    CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
            auto tensorDkbL0A = tla::MakeTensor(
                l0A[dkbL0ASlot],
                tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0A>(mActual, chunkLen),
                Catlass::Arch::PositionL0A{});
            auto tensorDkbL0B = tla::MakeTensor(
                l0B[dkbL0BSlot],
                tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0B>(chunkLen, nSize),
                Catlass::Arch::PositionL0B{});
            auto tileDAL1 = tla::GetTile(
                tensorL1DA, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
            auto tileKL1 = tla::GetTile(
                tensorL1K, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            auto tensorDkbL0C = tla::MakeTensor(
                l0C[dkbL0CSlot], tla::MakeLayoutL0C(mActual, nSize),
                Catlass::Arch::PositionL0C{});
            auto tileDkbL0C = tla::GetTile(
                tensorDkbL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbL0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbL0BEvent]);
            CopyDw0L1ToL0A{}(tensorDkbL0A, tileDAL1);
            CopyDw0L1ToL0B{}(tensorDkbL0B, tileKL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkbL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkbL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkbL0CSlot]);
            tileMmad_(tileDkbL0C, tensorDkbL0A, tensorDkbL0B, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbL0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbL0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkbL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkbL0CSlot]);
            typename TileCopyDw0::template CopyL0CToDst<decltype(blockGmDkb)>{}(
                blockGmDkb, tensorDkbL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkbL0CSlot]);
        }
        // N=64 keeps every L0A/L0B/L0C slot within 16 KiB.
        for (uint32_t nOffset = 0; nOffset < K_SIZE_128; nOffset += 64) {
            constexpr uint32_t nSize = 64;
            auto blockGmDkbT = tla::GetTile(
                tensorGmDkbT, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            const uint32_t dkbTL0ASlot = l0ASlot_;
            const uint32_t dkbTL0BSlot = l0BSlot_;
            const uint32_t dkbTL0CSlot = l0CSlot_;
            const uint32_t dkbTL0AEvent = 2 * dkbTL0ASlot;
            const uint32_t dkbTL0BEvent = 2 * dkbTL0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tensorL1DAT = tla::MakeTensor(
                dAL1[headOffset],
                tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL1A>(
                    CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
            auto tensorDkbTL0A = tla::MakeTensor(
                l0A[dkbTL0ASlot],
                tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0A>(mActual, chunkLen),
                Catlass::Arch::PositionL0A{});
            auto tensorDkbTL0B = tla::MakeTensor(
                l0B[dkbTL0BSlot],
                tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0B>(chunkLen, nSize),
                Catlass::Arch::PositionL0B{});
            auto tileDATL1 = tla::GetTile(
                tensorL1DAT, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
            auto tileKbL1 = tla::GetTile(
                tensorL1Kb, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            auto tensorDkbTL0C = tla::MakeTensor(
                l0C[dkbTL0CSlot], tla::MakeLayoutL0C(mActual, nSize),
                Catlass::Arch::PositionL0C{});
            auto tileDkbTL0C = tla::GetTile(
                tensorDkbTL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbTL0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbTL0BEvent]);
            CopyStage9DkbTL1ToL0A{}(tensorDkbTL0A, tileDATL1);
            CopyStage9DkbTL1ToL0B{}(tensorDkbTL0B, tileKbL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkbTL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkbTL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkbTL0CSlot]);
            tileMmad_(tileDkbTL0C, tensorDkbTL0A, tensorDkbTL0B, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbTL0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkbTL0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkbTL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkbTL0CSlot]);
            typename TileCopyStage9DkbT::template CopyL0CToDst<decltype(blockGmDkbT)>{}(
                blockGmDkbT, tensorDkbTL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkbTL0CSlot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[l1Event]);
    }

    __aicore__ inline void ProcessStage11Head(
        AscendC::LocalTensor<DT> (&doL1)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&vNewL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&l0A)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&l0B)[SLOT_COUNT],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&ds0Cv)[SLOT_COUNT],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t doSlot = static_cast<uint32_t>(headOffset);
        const uint32_t localHv = static_cast<uint32_t>(headOffset);
        const uint32_t doEvent = doSlot;
        const uint32_t vNewEvent = HEAD_COUNT + static_cast<uint32_t>(headOffset);
        const int64_t vectorBase =
            ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * V_SIZE_128;
        AscendC::GlobalTensor<DT> gmDo;
        AscendC::GlobalTensor<DT> gmVNew;
        gmDo.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(do_) + vectorBase);
        gmVNew.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vNew_) + vectorBase);
        auto tensorGmDo = tla::MakeTensor(gmDo, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmVNew = tla::MakeTensor(
            gmVNew, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmDo = tla::GetTile(
            tensorGmDo, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto blockGmVNew = tla::GetTile(
            tensorGmVNew, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto tensorDs0Cv = tla::MakeTensor(
            ds0Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDs0Cv = tla::GetTile(
            tensorDs0Cv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, FixpipeBf16Columns(chunkLen)));
        auto tensorL1Do = tla::MakeTensor(
            doL1[doSlot], L1_DU_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1VNewA = tla::MakeTensor(
            vNewL1[headOffset], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorL1VNewB = tla::MakeTensor(
            vNewL1[headOffset], L1_VB_LAYOUT, Catlass::Arch::PositionL1{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[doEvent]);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmDo)>{}(tensorL1Do, blockGmDo);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[vNewEvent]);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmVNew)>{}(
            tensorL1VNewA, blockGmVNew);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[doEvent]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[vNewEvent]);

        const uint32_t dsL0ASlot = l0ASlot_;
        const uint32_t dsL0BSlot = l0BSlot_;
        const uint32_t dsL0CSlot = l0CSlot_;
        const uint32_t dsL0AEvent = 2 * dsL0ASlot;
        const uint32_t dsL0BEvent = 2 * dsL0BSlot + 1;
        l0ASlot_ = NextSlot(l0ASlot_);
        l0BSlot_ = NextSlot(l0BSlot_);
        l0CSlot_ = NextSlot(l0CSlot_);
        auto tensorDsL0A = tla::MakeTensor(
            l0A[dsL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, V_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorDsL0B = tla::MakeTensor(
            l0B[dsL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(V_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDoL1 = tla::GetTile(
            tensorL1Do, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        auto tileVNewL1 = tla::GetTile(
            tensorL1VNewB, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, chunkLen));
        auto tensorDsL0C = tla::MakeTensor(
            l0C[dsL0CSlot], tla::MakeLayoutL0C(mActual, chunkLen), Catlass::Arch::PositionL0C{});
        auto tileDsL0C = tla::GetTile(
            tensorDsL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[doEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[vNewEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dsL0AEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dsL0BEvent]);
        CopyDauL1ToL0A{}(tensorDsL0A, tileDoL1);
        CopyDauL1ToL0B{}(tensorDsL0B, tileVNewL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[vNewEvent]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dsL0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dsL0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dsL0CSlot]);
        tileMmad_(tileDsL0C, tensorDsL0A, tensorDsL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dsL0AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dsL0BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dsL0CSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dsL0CSlot]);
        CopyL0CToUb<decltype(blockDs0Cv)>{}(blockDs0Cv, tensorDsL0C, false, uint8_t{0});
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dsL0CSlot]);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[doEvent]);
    }

    __aicore__ inline void ProcessStage13Head(
        AscendC::LocalTensor<DT> (&qL1)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dhL1)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dsL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&kL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&hL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&doGL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&vDecayL1)[HEAD_COUNT],
        AscendC::LocalTensor<DT> (&l0A)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&l0B)[SLOT_COUNT],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dk0Cv)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dqIntraCv)[SLOT_COUNT],
        AscendC::LocalTensor<DT> (&dkIntraCv)[SLOT_COUNT],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t localHv = static_cast<uint32_t>(headOffset);
        const int64_t hk = hv / tiling_->headRatio;
        const uint32_t qSlot = static_cast<uint32_t>(headOffset);
        const uint32_t dhSlot = static_cast<uint32_t>(headOffset);
        const uint32_t dsEvent = static_cast<uint32_t>(headOffset);
        const int64_t qBase = (tiling_->layout == FinalizeLayout::BSND)
                                  ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HK + hk) * K_SIZE_128
                                  : ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * K_SIZE_128;
        const int64_t workspaceBase = WorkspaceHeadOffset(headOffset);
        const int64_t stateBase =
            ((chunk.bIdx * stateChunkNum_ + chunk.stateChunkIdx) * tiling_->HV + hv) *
            K_SIZE_128 * V_SIZE_128;

        AscendC::GlobalTensor<DT> gmQ;
        AscendC::GlobalTensor<DT> gmDh;
        AscendC::GlobalTensor<DT> gmDs;
        AscendC::GlobalTensor<DT> gmDoG;
        AscendC::GlobalTensor<DT> gmVDecay;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(q_) + qBase);
        gmDh.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh_) + stateBase);
        gmDs.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dsIn_) + workspaceBase);
        gmDoG.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(doGIn_) + workspaceBase);
        gmVDecay.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vDecayIn_) + workspaceBase);
        auto tensorGmQ = tla::MakeTensor(
            gmQ,
            MakePublicQkLayout(),
            Catlass::Arch::PositionGM{});
        auto tensorGmDs = tla::MakeTensor(gmDs, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmDoG = tla::MakeTensor(gmDoG, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmVDecay = tla::MakeTensor(gmVDecay, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmQ = tla::GetTile(
            tensorGmQ, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto blockGmDs = tla::GetTile(
            tensorGmDs, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmDoG = tla::GetTile(
            tensorGmDoG, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto blockGmVDecay = tla::GetTile(
            tensorGmVDecay, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto tensorL1Q = tla::MakeTensor(
            qL1[qSlot], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorL1DhTranspose = tla::MakeTensor(
            dhL1[dhSlot], L1_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1DhVFirst = tla::MakeTensor(
            dhL1[dhSlot], L1_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1Ds = tla::MakeTensor(
            dsL1[headOffset], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1DsT = tla::MakeTensor(
            dsL1[headOffset], tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1DoG = tla::MakeTensor(
            doGL1[headOffset], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorL1VDecay = tla::MakeTensor(
            vDecayL1[headOffset], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});

        // S12 tokens were acquired for every active head at the stage boundary.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[dsEvent]);
        if (tiling_->stateVFirst != 0) {
            auto tensorGmDh = tla::MakeTensor(
                gmDh, GM_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmDh = tla::GetTile(
                tensorGmDh, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyStateVFirst::template CopyGmToL1B<decltype(blockGmDh)>{}(
                tensorL1DhVFirst, blockGmDh);
        } else {
            auto tensorGmDh = tla::MakeTensor(
                gmDh, GM_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmDh = tla::GetTile(
                tensorGmDh, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyDau::template CopyGmToL1B<decltype(blockGmDh)>{}(
                tensorL1DhTranspose, blockGmDh);
        }
        typename TileCopyDw0::template CopyGmToL1B<decltype(blockGmQ)>{}(tensorL1Q, blockGmQ);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmDs)>{}(tensorL1Ds, blockGmDs);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmDoG)>{}(tensorL1DoG, blockGmDoG);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmVDecay)>{}(tensorL1VDecay, blockGmVDecay);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[dsEvent]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1Events_[dsEvent]);
        // N=64 keeps every L0A/L0B/L0C slot within 16 KiB.
        for (uint32_t nOffset = 0; nOffset < K_SIZE_128; nOffset += 64) {
            constexpr uint32_t nSize = 64;
            auto tensorL1K = tla::MakeTensor(
                kL1[headOffset], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                    CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
            auto tileDsL1 = tla::GetTile(
                tensorL1Ds, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
            auto tileKL1 = tla::GetTile(
                tensorL1K, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            auto tileDoGL1 = tla::GetTile(
                tensorL1DoG, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
            auto tensorL1HTranspose = tla::MakeTensor(
                hL1[headOffset], L1_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionL1{});
            auto tensorL1HVFirst = tla::MakeTensor(
                hL1[headOffset], L1_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionL1{});

            const uint32_t dqL0ASlot = l0ASlot_;
            const uint32_t dqL0BSlot = l0BSlot_;
            const uint32_t dqL0CSlot = l0CSlot_;
            const uint32_t dqL0AEvent = 2 * dqL0ASlot;
            const uint32_t dqL0BEvent = 2 * dqL0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tensorDqL0A = tla::MakeTensor(
                l0A[dqL0ASlot], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0A>(
                    mActual, chunkLen), Catlass::Arch::PositionL0A{});
            auto tensorDqL0B = tla::MakeTensor(
                l0B[dqL0BSlot], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0B>(
                    chunkLen, nSize), Catlass::Arch::PositionL0B{});
            auto tensorDqL0C = tla::MakeTensor(
                l0C[dqL0CSlot], tla::MakeLayoutL0C(mActual, nSize), Catlass::Arch::PositionL0C{});
            auto tileDqL0C = tla::GetTile(
                tensorDqL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            auto tensorDqCv = tla::MakeTensor(
                dqIntraCv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
            auto blockDqCv = tla::GetTile(
                tensorDqCv, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0BEvent]);
            auto tensorDoGL0A = tla::MakeTensor(
                l0A[dqL0ASlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(
                    mActual, V_SIZE_128), Catlass::Arch::PositionL0A{});
            auto tensorHL0BTranspose = tla::MakeTensor(
                l0B[dqL0BSlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(
                    V_SIZE_128, nSize), Catlass::Arch::PositionL0B{});
            auto tensorHL0BVFirst = tla::MakeTensor(
                l0B[dqL0BSlot], tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL0B>(
                    V_SIZE_128, nSize), Catlass::Arch::PositionL0B{});
            CopyDauL1ToL0A{}(tensorDoGL0A, tileDoGL1);
            if (tiling_->stateVFirst != 0) {
                auto tileHL1 = tla::GetTile(
                    tensorL1HVFirst, tla::MakeCoord(0, nOffset), tla::MakeShape(V_SIZE_128, nSize));
                CopyStateVFirstL1ToL0B{}(tensorHL0BVFirst, tileHL1);
            } else {
                auto tileHL1 = tla::GetTile(
                    tensorL1HTranspose, tla::MakeCoord(0, nOffset), tla::MakeShape(V_SIZE_128, nSize));
                CopyDauL1ToL0B{}(tensorHL0BTranspose, tileHL1);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dqL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dqL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dqL0CSlot]);
            if (tiling_->stateVFirst != 0) {
                tileMmad_(tileDqL0C, tensorDoGL0A, tensorHL0BVFirst, true, 0);
            } else {
                tileMmad_(tileDqL0C, tensorDoGL0A, tensorHL0BTranspose, true, 0);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0BEvent]);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0BEvent]);
            CopyDw0L1ToL0A{}(tensorDqL0A, tileDsL1);
            CopyDw0L1ToL0B{}(tensorDqL0B, tileKL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dqL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dqL0CSlot]);
            tileMmad_(tileDqL0C, tensorDqL0A, tensorDqL0B, false, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dqL0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dqL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dqL0CSlot]);
            CopyL0CToUb<decltype(blockDqCv)>{}(blockDqCv, tensorDqL0C, false, uint8_t{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dqL0CSlot]);

            const uint32_t dkL0ASlot = l0ASlot_;
            const uint32_t dkL0BSlot = l0BSlot_;
            const uint32_t dkL0CSlot = l0CSlot_;
            const uint32_t dkL0AEvent = 2 * dkL0ASlot;
            const uint32_t dkL0BEvent = 2 * dkL0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tensorDkL0A = tla::MakeTensor(
                l0A[dkL0ASlot], tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0A>(
                    mActual, chunkLen), Catlass::Arch::PositionL0A{});
            auto tensorDkL0B = tla::MakeTensor(
                l0B[dkL0BSlot], tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0B>(
                    chunkLen, nSize), Catlass::Arch::PositionL0B{});
            auto tensorDkL0C = tla::MakeTensor(
                l0C[dkL0CSlot], tla::MakeLayoutL0C(mActual, nSize), Catlass::Arch::PositionL0C{});
            auto tileDkL0C = tla::GetTile(
                tensorDkL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            auto tileQL1 = tla::GetTile(
                tensorL1Q, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            auto tileDsTL1 = tla::GetTile(
                tensorL1DsT, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkL0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkL0BEvent]);
            CopyStage9DkbTL1ToL0A{}(tensorDkL0A, tileDsTL1);
            CopyStage9DkbTL1ToL0B{}(tensorDkL0B, tileQL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dkL0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkL0CSlot]);
            tileMmad_(tileDkL0C, tensorDkL0A, tensorDkL0B, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkL0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dkL0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkL0CSlot]);
            auto tensorDkCv = tla::MakeTensor(
                dkIntraCv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
            auto blockDkCv = tla::GetTile(
                tensorDkCv, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));

            const uint32_t dk0L0ASlot = l0ASlot_;
            const uint32_t dk0L0BSlot = l0BSlot_;
            const uint32_t dk0L0CSlot = l0CSlot_;
            const uint32_t dk0L0AEvent = 2 * dk0L0ASlot;
            const uint32_t dk0L0BEvent = 2 * dk0L0BSlot + 1;
            l0ASlot_ = NextSlot(l0ASlot_);
            l0BSlot_ = NextSlot(l0BSlot_);
            l0CSlot_ = NextSlot(l0CSlot_);
            auto tileVDecayL1 = tla::GetTile(
                tensorL1VDecay, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
            auto tensorDk0L0A = tla::MakeTensor(
                l0A[dk0L0ASlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(
                    mActual, V_SIZE_128), Catlass::Arch::PositionL0A{});
            auto tensorDk0L0BTranspose = tla::MakeTensor(
                l0B[dk0L0BSlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(
                    V_SIZE_128, nSize), Catlass::Arch::PositionL0B{});
            auto tensorDk0L0BVFirst = tla::MakeTensor(
                l0B[dk0L0BSlot], tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL0B>(
                    V_SIZE_128, nSize), Catlass::Arch::PositionL0B{});
            auto tensorDk0L0C = tla::MakeTensor(
                l0C[dk0L0CSlot], tla::MakeLayoutL0C(mActual, nSize), Catlass::Arch::PositionL0C{});
            auto tileDk0L0C = tla::GetTile(
                tensorDk0L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, nSize));
            auto tensorDk0Cv = tla::MakeTensor(
                dk0Cv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
            auto blockDk0Cv = tla::GetTile(
                tensorDk0Cv, tla::MakeCoord(0, nOffset), tla::MakeShape(chunkLen, nSize));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dk0L0AEvent]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dk0L0BEvent]);
            CopyDauL1ToL0A{}(tensorDk0L0A, tileVDecayL1);
            if (tiling_->stateVFirst != 0) {
                auto tileDhL1 = tla::GetTile(
                    tensorL1DhVFirst, tla::MakeCoord(0, nOffset), tla::MakeShape(V_SIZE_128, nSize));
                CopyStateVFirstL1ToL0B{}(tensorDk0L0BVFirst, tileDhL1);
            } else {
                auto tileDhL1 = tla::GetTile(
                    tensorL1DhTranspose, tla::MakeCoord(0, nOffset), tla::MakeShape(V_SIZE_128, nSize));
                CopyDauL1ToL0B{}(tensorDk0L0BTranspose, tileDhL1);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dk0L0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_MEvents_[dk0L0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dk0L0CSlot]);
            if (tiling_->stateVFirst != 0) {
                tileMmad_(tileDk0L0C, tensorDk0L0A, tensorDk0L0BVFirst, true, 0);
            } else {
                tileMmad_(tileDk0L0C, tensorDk0L0A, tensorDk0L0BTranspose, true, 0);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dk0L0AEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1Events_[dk0L0BEvent]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dk0L0CSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dk0L0CSlot]);
            CopyL0CToUb<decltype(blockDk0Cv)>{}(blockDk0Cv, tensorDk0L0C, false, uint8_t{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dk0L0CSlot]);

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(M_FIXEvents_[dkL0CSlot]);
            CopyL0CToUb<decltype(blockDkCv)>{}(blockDkCv, tensorDkL0C, false, uint8_t{0});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIX_MEvents_[dkL0CSlot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2Events_[dsEvent]);
    }

    __aicore__ static inline uint32_t FixpipeBf16Columns(uint32_t validColumns)
    {
        // The explicit sub-block overload needs 32-byte rows. Both the L0C fractal and UB row fit the padding.
        constexpr uint32_t columnsPerBlock = UB_ALIGN_BYTES / sizeof(DT);
        return (validColumns + columnsPerBlock - 1U) / columnsPerBlock * columnsPerBlock;
    }

    __aicore__ inline void WaitVectorInput(FinalizeCvFlag flag, uint32_t slot)
    {
        // Let FIX publish earlier Cube results before acquiring the next Vector input.
        // Only the subsequent GM loads depend on this local event.
        WaitReady<PIPE_FIX>(flag);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(inputFixToMte2_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(inputFixToMte2_[slot]);
    }

    __aicore__ inline int64_t WorkspaceHeadOffset(int64_t headOffset) const
    {
        return GetWorkspaceHeadOffset(coreIdx_, workspaceGroupRound_, headOffset);
    }

    AscendC::TEventID inputFixToMte2_[HEAD_COUNT];
    AscendC::TEventID MTE1_MTE2Events_[6];
    AscendC::TEventID MTE2_MTE1Events_[6];
    AscendC::TEventID FIX_MTE1Events_[6];
    AscendC::TEventID M_MTE1Events_[6];
    AscendC::TEventID MTE1_MEvents_[3];
    AscendC::TEventID FIX_MEvents_[3];
    AscendC::TEventID M_FIXEvents_[3];
    GM_ADDR q_ = nullptr;
    GM_ADDR k_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR do_ = nullptr;
    GM_ADDR du_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dh_ = nullptr;
    GM_ADDR a_ = nullptr;
    GM_ADDR kbg_ = nullptr;
    GM_ADDR vb_ = nullptr;
    GM_ADDR dA0_ = nullptr;
    GM_ADDR ds0Out_ = nullptr;
    GM_ADDR dkbOut_ = nullptr;
    GM_ADDR dkbTOut_ = nullptr;
    GM_ADDR dsIn_ = nullptr;
    GM_ADDR doGIn_ = nullptr;
    GM_ADDR vDecayIn_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling_ = nullptr;
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    int64_t stateChunkNum_ = 0;
    int64_t workspaceGroupRound_ = 0;
    uint32_t l0ASlot_ = 0;
    uint32_t l0BSlot_ = 0;
    uint32_t l0CSlot_ = 0;
    TileMmad tileMmad_;
};

} // namespace GDN::FinalizeArch92

#endif
