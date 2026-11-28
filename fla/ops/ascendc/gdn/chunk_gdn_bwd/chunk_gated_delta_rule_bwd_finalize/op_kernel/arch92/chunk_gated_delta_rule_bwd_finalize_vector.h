#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_VECTOR_H

#include "chunk_gated_delta_rule_bwd_finalize_vf.h"

namespace GDN::FinalizeArch92 {
template <typename DT, typename GT, typename BT, bool USE_QK_L2NORM, bool USE_BETA_SIGMOID,
          bool USE_EXP2>
class ChunkGatedDeltaRuleBwdFinalizeVector {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR dO,
        GM_ADDR g, GM_ADDR beta, GM_ADDR h, GM_ADDR dh,
        GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR dqOut, GM_ADDR dkOut, GM_ADDR dgOut,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR kbgOut, GM_ADDR vbOut,
        GM_ADDR dvbOut, GM_ADDR dA0Out,
        GM_ADDR dkbIn, GM_ADDR dkbTIn, GM_ADDR doGOut,
        GM_ADDR betaRawIn, GM_ADDR dbetaOut, GM_ADDR dsOut,
        GM_ADDR dqStage12Out, GM_ADDR dkStage12Out, GM_ADDR dk0Stage13Out,
        GM_ADDR workspace,
        const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(v));
        vNewGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vNew));
        doGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(beta));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(h));
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh));
        if constexpr (USE_QK_L2NORM) {
            qRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
            kRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        }
        dqOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dqOut));
        dkOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkOut));
        dgOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(dgOut));
        kbgOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(kbgOut));
        vbOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vbOut));
        a2OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dvbOut));
        dA0OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dA0Out));
        dkbInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbIn));
        dkbTInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbTIn));
        doGOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(doGOut));
        if constexpr (USE_BETA_SIGMOID) {
            betaRawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(betaRawIn));
        }
        dbetaOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(dbetaOut));
        dsOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dsOut));
        dqStage12OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dqStage12Out));
        dkStage12OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkStage12Out));
        dk0Stage13OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dk0Stage13Out));
        kbWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        stateChunkNum_ = tiling_->isVariable != 0 ? tiling_->totalChunkNum : tiling_->chunkNumForT;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }

        // Three full-head residents; aliases follow the S0--S15 lifetimes.
        pipe_->InitBuffer(ubBuf_, UB_TOTAL_BYTES);
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            k_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::K + slot * VECTOR_BYTES);
            v_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::V + slot * VECTOR_BYTES);
            gateDA_[slot] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, UB::GATE_DA + slot * MATRIX_BYTES);
            vb_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::VB + slot * VECTOR_BYTES);
            dvb_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::VB + slot * VECTOR_BYTES);
            kbg_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::KBG + slot * VECTOR_BYTES);
            dAu_[slot] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, UB::DAU + slot * MATRIX_BYTES);
            dAw0_[slot] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, UB::KBG + slot * VECTOR_BYTES);
            a2_[slot] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, UB::KBG + slot * MATRIX_BYTES);
            dbVPartial_[slot] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::DB_V + slot * SMALL_SLOT_BYTES);
            dgPrepare_[slot] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::G + slot * SMALL_SLOT_BYTES);
            hState_[slot] = ubBuf_.GetWithOffset<DT>(STATE_ELEMS, UB::H_STATE + slot * STATE_BYTES);
            dhState_[slot] = ubBuf_.GetWithOffset<DT>(STATE_ELEMS, UB::DH_STATE + slot * STATE_BYTES);
            dA2_[slot] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, UB::DA2 + slot * MATRIX_BYTES);
            dkbg0_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DKBG0 + slot * VECTOR_BYTES);
            kb_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::KB + slot * VECTOR_BYTES);
            g_[slot] = ubBuf_.GetWithOffset<GT>(CHUNK_SIZE_64, UB::G + slot * SMALL_SLOT_BYTES);
            beta_[slot] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, UB::BETA + slot * SMALL_SLOT_BYTES);
            betaSaved_[slot] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::BETA + slot * SMALL_SLOT_BYTES);
            gExp_[slot] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::G_EXP + slot * SMALL_SLOT_BYTES);
            gLast_[slot] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::G_LAST + slot * SMALL_SLOT_BYTES);
            decay_[slot] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::DECAY + slot * SMALL_SLOT_BYTES);
            bg_[slot] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::BG + slot * SMALL_SLOT_BYTES);
            dkb_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DKB + slot * VECTOR_BYTES);
            dkbT_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::VB + slot * VECTOR_BYTES);
            ds_[slot] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, UB::DS + slot * MATRIX_BYTES);
            doG_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DO_G + slot * VECTOR_BYTES);
            vDecay_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::V_DECAY + slot * VECTOR_BYTES);
            betaRaw_[slot] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, UB::BETA_RAW + slot * SMALL_SLOT_BYTES);
            dbetaOutput_[slot] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, UB::BETA_RAW + slot * SMALL_SLOT_BYTES);
            dk0Stage13_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DK_BASE + slot * VECTOR_BYTES);
            dqIntraStage13_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DQ_HV + slot * VECTOR_BYTES);
            dkIntraStage13_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DK_INTRA + slot * VECTOR_BYTES);
            qStage14_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DO_G + slot * VECTOR_BYTES);
            dqAcc_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DQ_ACC + slot * VECTOR_BYTES);
            dkAcc_[slot] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DK_ACC + slot * VECTOR_BYTES);
        }
        dqPartialInput_ = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DQ_PARTIAL);
        dkPartialInput_ = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::DK_PARTIAL);
        qStage15_ = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::NORM_Q);
        kStage15_ = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, UB::NORM_K);
        qRstdStage15_ = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::BETA_RAW);
        kRstdStage15_ = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, UB::BETA_RAW + SMALL_SLOT_BYTES);

        uint32_t eventIndex = 0;
        for (eventIndex = 0; eventIndex < SLOT_COUNT; ++eventIndex) {
            mte2ToV_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToV_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            mte3ToMte2_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            stage12Mte3ToMte2_[eventIndex] =
                pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            stateVToMte2_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[eventIndex]);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                stage12Mte3ToMte2_[eventIndex]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[eventIndex]);
        }
        stage15VToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
    }

    __aicore__ inline void SetGvaAccumulators(GM_ADDR ws0, GM_ADDR ws1)
    {
        gvaAcc_[0].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws0 + coreIdx_ * GVA_ACC_CORE_BYTES));
        gvaAcc_[1].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws1 + coreIdx_ * GVA_ACC_CORE_BYTES));
        scratch_ = ubBuf_.GetWithOffset<float>(VECTOR_ELEMS, UB::DK_PARTIAL);
        mergeAcc_ = ubBuf_.GetWithOffset<float>(VECTOR_ELEMS, UB::DQ_ACC);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->headRatio > 4) {
            ProcessImpl<true>();
        } else {
            ProcessImpl<false>();
        }
    }

    template <bool GVA_GT4>
    __aicore__ inline void ProcessImpl()
    {
        int64_t taskIdx = 0;
        int64_t chunkTaskIdx = 0;
        int64_t headGroupIdx = 0;
        int64_t windowBank = 0;
        int64_t hvBase = 0;
        int64_t taskCount = 0;
        int64_t headOffset = 0;
        int64_t hv = 0;
        int64_t hk = 0;
        int64_t gOffset = 0;
        int64_t kOffset = 0;
        int64_t vOffset = 0;
        int64_t stateOffset = 0;
        int64_t vectorBase = 0;
        int64_t qOffset = 0;
        int64_t workspaceVectorOffset = 0;
        int64_t workspaceGroupRound = 0;
        int64_t localHv = 0;
        int64_t stage15Begin = 0;
        int64_t stage15End = 0;
        int64_t stage15Head = 0;
        int64_t stage15Hk = 0;
        int64_t stage15AccId = 0;
        int64_t stage15FirstHead = 0;
        int64_t stage15OutputCount = 0;
        int64_t stage15Output = 0;
        uint32_t qSlot = 0;
        uint32_t matrixBlockBytes = 0;
        uint32_t matrixAlignedBlockBytes = 0;
        AscendC::DataCopyExtParams vectorCopyParams;
        AscendC::DataCopyExtParams matrixCopyParams;
        AscendC::DataCopyExtParams scalarCopyParams;
        ChunkInfo chunk;

        for (taskIdx = coreIdx_; taskIdx < tiling_->taskNum; taskIdx += coreNum_) {
            chunkTaskIdx = taskIdx / tiling_->headGroupNum;
            headGroupIdx = taskIdx % tiling_->headGroupNum;
            GetChunkInfo(chunkTaskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            matrixBlockBytes = static_cast<uint32_t>(chunk.chunkLen * sizeof(DT));
            matrixAlignedBlockBytes = (matrixBlockBytes + UB_ALIGN_BYTES - 1U) /
                UB_ALIGN_BYTES * UB_ALIGN_BYTES;
            matrixCopyParams = {
                static_cast<uint16_t>(chunk.chunkLen),
                matrixBlockBytes,
                static_cast<uint32_t>((CHUNK_SIZE_64 * sizeof(DT) - matrixAlignedBlockBytes) /
                    UB_ALIGN_BYTES),
                static_cast<uint32_t>((CHUNK_SIZE_64 - chunk.chunkLen) * sizeof(DT)), 0};

            scalarCopyParams = {
                1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0};
            if constexpr (GVA_GT4) {
                InitializeGvaAccumulators();
            }
            const int64_t windowCount = tiling_->headRatio > HEAD_COUNT ?
                (tiling_->headRatio + HEAD_COUNT - 1) / HEAD_COUNT : 1;
            const int64_t firstWindowSerial = windowBank;
            for (int64_t window = 0; window < windowCount; ++window) {
                hvBase = tiling_->headRatio > HEAD_COUNT ?
                    headGroupIdx * tiling_->headRatio + window * HEAD_COUNT :
                    headGroupIdx * tiling_->taskGroupSize;
                taskCount = Min(static_cast<int64_t>(HEAD_COUNT), tiling_->headRatio > HEAD_COUNT ?
                    tiling_->headRatio - window * HEAD_COUNT : tiling_->taskGroupSize);
                workspaceGroupRound = windowBank;
                int64_t qCachedHk[SLOT_COUNT] = {-1, -1, -1};

                streamSlot_ = 0;

                for (uint32_t slot = 0; slot < taskCount; ++slot) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[slot]);
                }

                for (headOffset = 0; headOffset < taskCount; ++headOffset) {

                    hv = hvBase + headOffset;
                    hk = hv / tiling_->headRatio;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    kOffset = (tiling_->layout == GDN::FinalizeLayout::BSND)
                                   ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HK + hk) * tiling_->K
                                   : ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * tiling_->K;
                    vOffset = (tiling_->layout == GDN::FinalizeLayout::BSND)
                                   ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HV + hv) * tiling_->V
                                   : ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * tiling_->V;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[streamSlot_]);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        stage12Mte3ToMte2_[streamSlot_]);
                    AscendC::DataCopyPad(g_[streamSlot_], gGm_[gOffset],
                        {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(GT)), 0, 0, 0},
                        {false, 0, 0, 0});
                    AscendC::DataCopyPad(beta_[streamSlot_], betaGm_[gOffset],
                        {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0},
                        {false, 0, 0, 0});
                    if ((tiling_->layout == GDN::FinalizeLayout::BSND)) {
                        AscendC::DataCopyPad(k_[streamSlot_], kGm_[kOffset],
                            {static_cast<uint16_t>(chunk.chunkLen),
                             static_cast<uint32_t>(K_SIZE_128 * sizeof(DT)),
                             static_cast<uint32_t>((tiling_->HK - 1) * K_SIZE_128 * sizeof(DT)), 0, 0},
                            {false, 0, 0, 0});
                        AscendC::DataCopyPad(v_[streamSlot_], vGm_[vOffset],
                            {static_cast<uint16_t>(chunk.chunkLen),
                             static_cast<uint32_t>(V_SIZE_128 * sizeof(DT)),
                             static_cast<uint32_t>((tiling_->HV - 1) * V_SIZE_128 * sizeof(DT)), 0, 0},
                            {false, 0, 0, 0});
                    } else {
                        AscendC::DataCopy(k_[streamSlot_], kGm_[kOffset], chunk.chunkLen * K_SIZE_128);
                        AscendC::DataCopy(v_[streamSlot_], vGm_[vOffset], chunk.chunkLen * V_SIZE_128);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);

                    Stage0VF<DT, GT, BT, USE_EXP2>(
                        reinterpret_cast<__ubuf__ DT *>(gateDA_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(kbg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(vb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gExp_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gLast_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(decay_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(bg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(v_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ GT *>(g_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ BT *>(beta_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);

                    vectorCopyParams = {
                        1, static_cast<uint32_t>(chunk.chunkLen * K_SIZE_128 * sizeof(DT)), 0, 0, 0};

                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        kbgOutGm_[workspaceVectorOffset], kbg_[streamSlot_], vectorCopyParams);
                    AscendC::DataCopyPad(
                        vbOutGm_[workspaceVectorOffset], vb_[streamSlot_], vectorCopyParams);

                    NotifyCube();

                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {

                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    WaitCube();
                    Stage2VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dAu_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(kb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);

                    vectorCopyParams = {
                        1, static_cast<uint32_t>(chunk.chunkLen * K_SIZE_128 * sizeof(DT)), 0, 0, 0};
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        kbWorkspaceGm_[workspaceVectorOffset],
                        kb_[streamSlot_], vectorCopyParams);
                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {

                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;

                    WaitCube();
                    Stage4VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dAw0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dAu_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        dA0OutGm_[workspaceVectorOffset], dAw0_[streamSlot_], matrixCopyParams);
                    NotifyCube();
                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {

                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    vOffset = (tiling_->layout == GDN::FinalizeLayout::BSND)
                                  ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HV + hv) * tiling_->V
                                  : ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * tiling_->V;
                    Stage6VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dvb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(v_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dbVPartial_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);

                    if ((tiling_->layout == GDN::FinalizeLayout::BSND)) {
                        const AscendC::DataCopyExtParams dvOutputCopyParams = {
                            static_cast<uint16_t>(chunk.chunkLen),
                            static_cast<uint32_t>(V_SIZE_128 * sizeof(DT)),
                            0,
                            static_cast<uint32_t>((tiling_->HV - 1) * V_SIZE_128 * sizeof(DT)), 0};
                        AscendC::DataCopyPad(dkStage12OutGm_[vOffset], dvb_[streamSlot_], dvOutputCopyParams);
                    } else {
                        AscendC::DataCopyPad(
                            dkStage12OutGm_[vOffset], dvb_[streamSlot_], vectorCopyParams);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);
                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {

                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    vOffset = gOffset * tiling_->V;

                    WaitCube();
                    Stage8VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dA2_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(gateDA_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(a2_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbg0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gExp_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(bg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dbVPartial_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dgPrepare_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);

                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        dA0OutGm_[workspaceVectorOffset], dA2_[streamSlot_], matrixCopyParams);

                    NotifyCube();

                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[streamSlot_]);
                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                // S10 overwrites several S8 slots, including other heads' matrix buffers.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
                for (uint32_t slot = 0; slot < taskCount; ++slot) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[slot]);
                }

                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {

                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    stateOffset =
                        ((chunk.bIdx * stateChunkNum_ + chunk.stateChunkIdx) * tiling_->HV + hv) *
                        STATE_ELEMS;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[streamSlot_]);
                    AscendC::DataCopy(hState_[streamSlot_], hGm_[stateOffset], STATE_ELEMS);
                    AscendC::DataCopy(dhState_[streamSlot_], dhGm_[stateOffset], STATE_ELEMS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    Stage10VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(hState_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dhState_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gLast_[streamSlot_].GetPhyAddr()));

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[streamSlot_]);
                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                for (uint32_t slot = 0; slot < taskCount; ++slot) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[slot]);
                }
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {

                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    vectorBase = gOffset * K_SIZE_128;
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopy(
                        doG_[streamSlot_], doGm_[vectorBase], chunk.chunkLen * V_SIZE_128);
                    AscendC::DataCopy(
                        vDecay_[streamSlot_], vNewGm_[vectorBase], chunk.chunkLen * V_SIZE_128);
                    if constexpr (USE_BETA_SIGMOID) {
                        AscendC::DataCopyPad(betaRaw_[streamSlot_], betaRawGm_[gOffset],
                            {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0},
                            {false, 0, 0, 0});
                    }
                    WaitCube<PIPE_MTE2>();
                    AscendC::DataCopy(
                        dkb_[streamSlot_], dkbInGm_[workspaceVectorOffset], chunk.chunkLen * K_SIZE_128);
                    AscendC::DataCopy(
                        dkbT_[streamSlot_], dkbTInGm_[workspaceVectorOffset], chunk.chunkLen * K_SIZE_128);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    Stage12VF<DT, BT, USE_BETA_SIGMOID>(
                        reinterpret_cast<__ubuf__ DT *>(dkb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbT_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(ds_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(doG_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(vDecay_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbg0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(gateDA_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gExp_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(decay_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(bg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dbVPartial_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ BT *>(betaRaw_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaRaw_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ BT *>(dbetaOutput_[streamSlot_].GetPhyAddr()),
                        tiling_->scale, static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    if constexpr (std::is_same<BT, float>::value) {
                        AscendC::DataCopyPad(dbetaOutGm_[gOffset], dbVPartial_[streamSlot_],
                            {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0});
                    } else {
                        AscendC::DataCopyPad(dbetaOutGm_[gOffset], dbetaOutput_[streamSlot_],
                            {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0});
                    }
                    AscendC::DataCopyPad(
                        dsOutGm_[workspaceVectorOffset], ds_[streamSlot_], matrixCopyParams);
                    AscendC::DataCopyPad(
                        doGOutGm_[workspaceVectorOffset], doG_[streamSlot_], vectorCopyParams);
                    AscendC::DataCopyPad(
                        dk0Stage13OutGm_[workspaceVectorOffset], vDecay_[streamSlot_], vectorCopyParams);

                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);

                    NotifyCube();

                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    WaitCube();

                    hv = hvBase + headOffset;
                    hk = hv / tiling_->headRatio;
                    localHv = hv - hk * tiling_->headRatio;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    qOffset = (tiling_->layout == GDN::FinalizeLayout::BSND)
                                  ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HK + hk) * tiling_->K
                                  : ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * tiling_->K;

                    qSlot = tiling_->headRatio == 1 ? streamSlot_ : 0;
                    if (qCachedHk[qSlot] != hk) {
                        qCachedHk[qSlot] = hk;
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[qSlot]);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[qSlot]);
                        if ((tiling_->layout == GDN::FinalizeLayout::BSND)) {
                            AscendC::DataCopyPad(qStage14_[qSlot], qGm_[qOffset],
                                {static_cast<uint16_t>(chunk.chunkLen),
                                 static_cast<uint32_t>(K_SIZE_128 * sizeof(DT)),
                                 static_cast<uint32_t>((tiling_->HK - 1) * K_SIZE_128 * sizeof(DT)), 0, 0},
                                {false, 0, 0, 0});
                        } else {
                            AscendC::DataCopy(qStage14_[qSlot], qGm_[qOffset], chunk.chunkLen * K_SIZE_128);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[qSlot]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[qSlot]);
                    }
                    Stage14VF<DT, GT>(
                        reinterpret_cast<__ubuf__ DT *>(dk0Stage13_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dqIntraStage13_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkIntraStage13_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbg0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(qStage14_[qSlot].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dgPrepare_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gLast_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(decay_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    if constexpr (GVA_GT4) {
                        AccumulateGva(dqIntraStage13_[streamSlot_], 0, chunk.chunkLen, localHv % 2);
                        AccumulateGva(dk0Stage13_[streamSlot_], VECTOR_ELEMS, chunk.chunkLen, localHv % 2);
                    } else {
                        workspaceVectorOffset = GetWorkspaceHeadOffset(
                            coreIdx_, workspaceGroupRound, headOffset);
                        AscendC::DataCopyPad(
                            dqStage12OutGm_[workspaceVectorOffset],
                            dqIntraStage13_[streamSlot_], vectorCopyParams);
                        AscendC::DataCopyPad(
                            a2OutGm_[workspaceVectorOffset],
                            dk0Stage13_[streamSlot_], vectorCopyParams);
                    }
                    AscendC::DataCopyPad(
                        dgOutGm_[gOffset],
                        decay_[streamSlot_].template ReinterpretCast<GT>(),
                        {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(GT)), 0, 0, 0});
                    streamSlot_ = (streamSlot_ + 1) % SLOT_COUNT;
                }

                for (uint32_t slot = 0; slot < taskCount; ++slot) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[slot]);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
                // All S14 consumers are done. The AIC may advance to the next window,
                // whose first CV write still waits for the new S0 notification.
                NotifyCube();
                for (uint32_t slot = 0; slot < taskCount; ++slot) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[slot]);
                }

                if constexpr (GVA_GT4) {
                    if (window + 1 == windowCount) {
                        MergeGvaAccumulators(chunk, headGroupIdx);
                    }
                } else if (window + 1 == windowCount) {
                    const int64_t stage15HvBase = tiling_->headRatio > HEAD_COUNT ?
                        headGroupIdx * tiling_->headRatio : hvBase;
                    stage15Begin = 0;
                    stage15End = tiling_->headRatio > HEAD_COUNT ? tiling_->headRatio : taskCount;
                    stage15OutputCount = 0;
                    stage15AccId = 0;
                    stage15FirstHead = stage15Begin;
                    for (stage15Head = stage15Begin; stage15Head < stage15End; ++stage15Head) {
                        stage15AccId = (stage15Head - stage15Begin) / tiling_->headRatio;
                        hv = stage15HvBase + stage15Head;
                        gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T +
                            chunk.tokenStart;
                        workspaceVectorOffset = GetWorkspaceHeadOffset(
                            coreIdx_, firstWindowSerial + stage15Head / HEAD_COUNT, stage15Head % HEAD_COUNT);
                        if ((stage15Head - stage15Begin) % tiling_->headRatio == 0) {
                            AscendC::DataCopy(
                                dqAcc_[stage15AccId], dqStage12OutGm_[workspaceVectorOffset],
                                chunk.chunkLen * K_SIZE_128);
                            AscendC::DataCopy(
                                dkAcc_[stage15AccId], a2OutGm_[workspaceVectorOffset],
                                chunk.chunkLen * K_SIZE_128);
                            stage15OutputCount = stage15AccId + 1;
                        } else {
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                            AscendC::DataCopy(
                                dqPartialInput_, dqStage12OutGm_[workspaceVectorOffset],
                                chunk.chunkLen * K_SIZE_128);
                            AscendC::DataCopy(
                                dkPartialInput_, a2OutGm_[workspaceVectorOffset],
                                chunk.chunkLen * K_SIZE_128);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                            Stage15AccumulateVF<DT>(
                                reinterpret_cast<__ubuf__ DT *>(dqAcc_[stage15AccId].GetPhyAddr()),
                                reinterpret_cast<__ubuf__ DT *>(dkAcc_[stage15AccId].GetPhyAddr()),
                                reinterpret_cast<__ubuf__ DT *>(dqPartialInput_.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ DT *>(dkPartialInput_.GetPhyAddr()),
                                static_cast<uint16_t>(chunk.chunkLen));

                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                        }
                    }

                    for (stage15Output = 0; stage15Output < stage15OutputCount; ++stage15Output) {
                        stage15FirstHead = stage15Begin + stage15Output * tiling_->headRatio;
                        stage15Hk = (stage15HvBase + stage15FirstHead) / tiling_->headRatio;
                        qOffset = (tiling_->layout == GDN::FinalizeLayout::BSND)
                                      ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HK + stage15Hk) * tiling_->K
                                      : ((chunk.bIdx * tiling_->HK + stage15Hk) * tiling_->T + chunk.tokenStart) * tiling_->K;
                        const int64_t rstdOffset = (chunk.bIdx * tiling_->HK + stage15Hk) * tiling_->T + chunk.tokenStart;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                        if constexpr (USE_QK_L2NORM) {
                            if ((tiling_->layout == GDN::FinalizeLayout::BSND)) {
                                AscendC::DataCopyPad(qStage15_, qGm_[qOffset],
                                    {static_cast<uint16_t>(chunk.chunkLen),
                                     static_cast<uint32_t>(K_SIZE_128 * sizeof(DT)),
                                     static_cast<uint32_t>((tiling_->HK - 1) * K_SIZE_128 * sizeof(DT)), 0, 0},
                                    {false, 0, 0, 0});
                                AscendC::DataCopyPad(kStage15_, kGm_[qOffset],
                                    {static_cast<uint16_t>(chunk.chunkLen),
                                     static_cast<uint32_t>(K_SIZE_128 * sizeof(DT)),
                                     static_cast<uint32_t>((tiling_->HK - 1) * K_SIZE_128 * sizeof(DT)), 0, 0},
                                    {false, 0, 0, 0});
                            } else {
                                AscendC::DataCopy(qStage15_, qGm_[qOffset], chunk.chunkLen * K_SIZE_128);
                                AscendC::DataCopy(kStage15_, kGm_[qOffset], chunk.chunkLen * K_SIZE_128);
                            }
                            AscendC::DataCopyPad(qRstdStage15_,
                                qRstdGm_[rstdOffset], scalarCopyParams,
                                {false, 0, 0, 0});
                            AscendC::DataCopyPad(kRstdStage15_,
                                kRstdGm_[rstdOffset], scalarCopyParams,
                                {false, 0, 0, 0});
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                        if constexpr (USE_QK_L2NORM) {
                            Stage15NormVF<DT>(
                                reinterpret_cast<__ubuf__ DT *>(dqAcc_[stage15Output].GetPhyAddr()),
                                reinterpret_cast<__ubuf__ DT *>(dkAcc_[stage15Output].GetPhyAddr()),
                                reinterpret_cast<__ubuf__ DT *>(qStage15_.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ DT *>(kStage15_.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float *>(qRstdStage15_.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float *>(kRstdStage15_.GetPhyAddr()),
                                static_cast<uint16_t>(chunk.chunkLen));
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
                        if ((tiling_->layout == GDN::FinalizeLayout::BSND)) {
                            const AscendC::DataCopyExtParams outputCopyParams = {
                                static_cast<uint16_t>(chunk.chunkLen),
                                static_cast<uint32_t>(K_SIZE_128 * sizeof(DT)),
                                0,
                                static_cast<uint32_t>((tiling_->HK - 1) * K_SIZE_128 * sizeof(DT)), 0};
                            AscendC::DataCopyPad(dqOutGm_[qOffset], dqAcc_[stage15Output], outputCopyParams);
                            AscendC::DataCopyPad(dkOutGm_[qOffset], dkAcc_[stage15Output], outputCopyParams);
                        } else {
                            AscendC::DataCopyPad(dqOutGm_[qOffset], dqAcc_[stage15Output], vectorCopyParams);
                            AscendC::DataCopyPad(dkOutGm_[qOffset], dkAcc_[stage15Output], vectorCopyParams);
                        }
                    }
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);

                // Match exactly the S0 waits of the active slots; unused tail slots
                // keep their existing initial/free credits.
                for (uint32_t slot = 0; slot < taskCount; ++slot) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stage12Mte3ToMte2_[slot]);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[slot]);
                }
                ++windowBank;
            }
        }

        uint32_t eventIndex = 0;
        for (eventIndex = 0; eventIndex < SLOT_COUNT; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[eventIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                stage12Mte3ToMte2_[eventIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(
                stage12Mte3ToMte2_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(stateVToMte2_[eventIndex]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
    }

private:
    __aicore__ inline void InitializeGvaAccumulators()
    {
        ZeroGvaScratchVF(reinterpret_cast<__ubuf__ float *>(scratch_.GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::DisableDmaAtomic();
        for (uint32_t parity = 0; parity < 2; ++parity) {
            AscendC::DataCopy(gvaAcc_[parity], scratch_, VECTOR_ELEMS);
            AscendC::DataCopy(gvaAcc_[parity][VECTOR_ELEMS], scratch_, VECTOR_ELEMS);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
    }

    __aicore__ inline void AccumulateGva(
        const AscendC::LocalTensor<DT> &partial, uint32_t offset, int64_t rows, uint32_t parity)
    {
        CastBf16ToFp32VF<DT>(
            reinterpret_cast<__ubuf__ float *>(scratch_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ DT *>(partial.GetPhyAddr()), static_cast<uint16_t>(rows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(gvaAcc_[parity][offset], scratch_, rows * K_SIZE_128);
        AscendC::DisableDmaAtomic();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
    }

    __aicore__ inline void MergeGvaGradient(
        const AscendC::LocalTensor<DT> &output, uint32_t offset, int64_t rows)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
        AscendC::DataCopy(mergeAcc_, gvaAcc_[0][offset], rows * K_SIZE_128);
        AscendC::DataCopy(scratch_, gvaAcc_[1][offset], rows * K_SIZE_128);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
        MergeFp32ToBf16VF<DT>(
            reinterpret_cast<__ubuf__ DT *>(output.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(mergeAcc_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(scratch_.GetPhyAddr()),
            static_cast<uint16_t>(rows));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
    }

    __aicore__ inline void MergeGvaAccumulators(const ChunkInfo &chunk, int64_t hk)
    {

        auto dqFinal = dqPartialInput_;
        auto dkFinal = dkAcc_[0];
        MergeGvaGradient(dqFinal, 0, chunk.chunkLen);
        MergeGvaGradient(dkFinal, VECTOR_ELEMS, chunk.chunkLen);
        const int64_t qOffset = tiling_->layout == GDN::FinalizeLayout::BSND
            ? ((chunk.bIdx * tiling_->T + chunk.tokenStart) * tiling_->HK + hk) * tiling_->K
            : ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * tiling_->K;
        if constexpr (USE_QK_L2NORM) {

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
            if (tiling_->layout == GDN::FinalizeLayout::BSND) {
                const AscendC::DataCopyExtParams inputCopyParams = {
                    static_cast<uint16_t>(chunk.chunkLen),
                    static_cast<uint32_t>(K_SIZE_128 * sizeof(DT)),
                    static_cast<uint32_t>((tiling_->HK - 1) * K_SIZE_128 * sizeof(DT)), 0, 0};
                AscendC::DataCopyPad(qStage15_, qGm_[qOffset], inputCopyParams, {false, 0, 0, 0});
                AscendC::DataCopyPad(kStage15_, kGm_[qOffset], inputCopyParams, {false, 0, 0, 0});
            } else {
                AscendC::DataCopy(qStage15_, qGm_[qOffset], chunk.chunkLen * K_SIZE_128);
                AscendC::DataCopy(kStage15_, kGm_[qOffset], chunk.chunkLen * K_SIZE_128);
            }
            const int64_t rstdOffset = (chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart;
            AscendC::DataCopyExtParams scalarParams = {
                1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(qRstdStage15_, qRstdGm_[rstdOffset],
                                scalarParams, {false, 0, 0, 0});
            AscendC::DataCopyPad(kRstdStage15_, kRstdGm_[rstdOffset],
                                scalarParams, {false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
            Stage15NormVF<DT>(
                reinterpret_cast<__ubuf__ DT *>(dqFinal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ DT *>(dkFinal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ DT *>(qStage15_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ DT *>(kStage15_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(qRstdStage15_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(kRstdStage15_.GetPhyAddr()),
                static_cast<uint16_t>(chunk.chunkLen));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
        if (tiling_->layout == GDN::FinalizeLayout::BSND) {
            const AscendC::DataCopyExtParams outputCopyParams = {
                static_cast<uint16_t>(chunk.chunkLen),
                static_cast<uint32_t>(K_SIZE_128 * sizeof(DT)), 0,
                static_cast<uint32_t>((tiling_->HK - 1) * K_SIZE_128 * sizeof(DT)), 0};
            AscendC::DataCopyPad(dqOutGm_[qOffset], dqFinal, outputCopyParams);
            AscendC::DataCopyPad(dkOutGm_[qOffset], dkFinal, outputCopyParams);
        } else {
            AscendC::DataCopy(dqOutGm_[qOffset], dqFinal, chunk.chunkLen * K_SIZE_128);
            AscendC::DataCopy(dkOutGm_[qOffset], dkFinal, chunk.chunkLen * K_SIZE_128);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[0]);
    }

    AscendC::GlobalTensor<float> gvaAcc_[2];
    AscendC::LocalTensor<float> scratch_;
    AscendC::LocalTensor<float> mergeAcc_;

    const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    uint32_t streamSlot_ = 0;

    AscendC::GlobalTensor<DT> kGm_;
    AscendC::GlobalTensor<DT> qGm_;
    AscendC::GlobalTensor<DT> vGm_;
    AscendC::GlobalTensor<DT> vNewGm_;
    AscendC::GlobalTensor<DT> doGm_;
    AscendC::GlobalTensor<GT> gGm_;
    AscendC::GlobalTensor<BT> betaGm_;
    AscendC::GlobalTensor<DT> hGm_;
    AscendC::GlobalTensor<DT> dhGm_;
    AscendC::GlobalTensor<float> qRstdGm_;
    AscendC::GlobalTensor<float> kRstdGm_;
    AscendC::GlobalTensor<DT> dqOutGm_;
    AscendC::GlobalTensor<DT> dkOutGm_;
    AscendC::GlobalTensor<GT> dgOutGm_;
    AscendC::GlobalTensor<DT> kbgOutGm_;
    AscendC::GlobalTensor<DT> vbOutGm_;
    AscendC::GlobalTensor<DT> a2OutGm_;
    AscendC::GlobalTensor<DT> dA0OutGm_;
    AscendC::GlobalTensor<DT> kbWorkspaceGm_;
    AscendC::GlobalTensor<DT> dkbInGm_;
    AscendC::GlobalTensor<DT> dkbTInGm_;
    AscendC::GlobalTensor<DT> doGOutGm_;
    AscendC::GlobalTensor<BT> betaRawGm_;
    AscendC::GlobalTensor<BT> dbetaOutGm_;
    AscendC::GlobalTensor<DT> dsOutGm_;
    AscendC::GlobalTensor<DT> dqStage12OutGm_;
    AscendC::GlobalTensor<DT> dkStage12OutGm_;
    AscendC::GlobalTensor<DT> dk0Stage13OutGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<DT> k_[SLOT_COUNT];
    AscendC::LocalTensor<DT> v_[SLOT_COUNT];
    AscendC::LocalTensor<DT> gateDA_[SLOT_COUNT];
    AscendC::LocalTensor<DT> vb_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dvb_[SLOT_COUNT];
    AscendC::LocalTensor<DT> kbg_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dAu_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dAw0_[SLOT_COUNT];
    AscendC::LocalTensor<DT> a2_[SLOT_COUNT];
    AscendC::LocalTensor<float> dbVPartial_[SLOT_COUNT];
    AscendC::LocalTensor<float> dgPrepare_[SLOT_COUNT];
    AscendC::LocalTensor<DT> hState_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dhState_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dA2_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dkbg0_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dkb_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dkbT_[SLOT_COUNT];
    AscendC::LocalTensor<DT> ds_[SLOT_COUNT];
    AscendC::LocalTensor<DT> doG_[SLOT_COUNT];
    AscendC::LocalTensor<DT> vDecay_[SLOT_COUNT];
    AscendC::LocalTensor<BT> betaRaw_[SLOT_COUNT];
    AscendC::LocalTensor<BT> dbetaOutput_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dk0Stage13_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dqIntraStage13_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dkIntraStage13_[SLOT_COUNT];
    AscendC::LocalTensor<DT> qStage14_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dqAcc_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dkAcc_[SLOT_COUNT];
    AscendC::LocalTensor<DT> dqPartialInput_;
    AscendC::LocalTensor<DT> dkPartialInput_;
    AscendC::LocalTensor<DT> qStage15_;
    AscendC::LocalTensor<DT> kStage15_;
    AscendC::LocalTensor<float> qRstdStage15_;
    AscendC::LocalTensor<float> kRstdStage15_;
    AscendC::LocalTensor<DT> kb_[SLOT_COUNT];
    AscendC::LocalTensor<GT> g_[SLOT_COUNT];
    AscendC::LocalTensor<BT> beta_[SLOT_COUNT];
    AscendC::LocalTensor<float> betaSaved_[SLOT_COUNT];
    AscendC::LocalTensor<float> gExp_[SLOT_COUNT];
    AscendC::LocalTensor<float> gLast_[SLOT_COUNT];
    AscendC::LocalTensor<float> decay_[SLOT_COUNT];
    AscendC::LocalTensor<float> bg_[SLOT_COUNT];
    AscendC::TEventID mte2ToV_[SLOT_COUNT];
    AscendC::TEventID vToMte3_[SLOT_COUNT];
    AscendC::TEventID mte3ToV_[SLOT_COUNT];
    AscendC::TEventID mte3ToMte2_[SLOT_COUNT];
    AscendC::TEventID stage12Mte3ToMte2_[SLOT_COUNT];
    AscendC::TEventID stateVToMte2_[SLOT_COUNT];
    AscendC::TEventID stage15VToMte2_;
    int64_t stateChunkNum_ = 0;
};


} // namespace GDN::FinalizeArch92

#endif
