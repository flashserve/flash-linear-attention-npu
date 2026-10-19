/**
 * Fused GDN forward. Entry is Init + Process, same shape as prepare.
 * AIV0 owns the vector stages. AIC consumes L1 and Fixpipes FP32 back to that UB.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_H

#include "chunk_gated_delta_rule_fwd_32_matmul.h"
#include "chunk_gated_delta_rule_fwd_32_vf.h"
#include "chunk_gated_delta_rule_fwd_32_tiling_data.h"

namespace GdnFwd32 {

class Kernel {
public:
    __aicore__ inline Kernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR initial, GM_ADDR o,
                                GM_ADDR finalState, GM_ADDR workspace, const ChunkGatedDeltaRuleFwd32TilingData &td)
    {
        subBlockIdx_ = GetSubBlockIdx();
        batchSize_ = td.B;
        valueHeadNum_ = td.HV;
        seqLen_ = td.T;
        scale_ = td.scale;
        hasInitial_ = td.hasInitial;

        // MIX is AIC:AIV = 1:2. AIV block ids are aicIdx * subBlockNum + sub, so fold them back.
        uint32_t aicIdx = GetBlockIdx();
        if ASCEND_IS_AIV {
            const uint32_t subBlockNum = GetSubBlockNum();
            if (subBlockNum > 0) {
                aicIdx /= subBlockNum;
            }
        }
        aicNum_ = td.coreNum == 0 ? 1 : td.coreNum;
        const uint32_t totalHeads = batchSize_ * valueHeadNum_;
        isActive_ = valueHeadNum_ != 0 && aicIdx < totalHeads;
        headBegin_ = isActive_ ? aicIdx : 0;

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(beta));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(o));
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(finalState));
        wsAddr_ = workspace + td.workspaceOffset;
        stateWs_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(wsAddr_));
        if (hasInitial_ != 0 && initial != nullptr) {
            initGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initial));
        }

        ubR_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_R);
        ubQ_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_Q);
        ubK_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_K);
        ubV_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_V);
        ubKbg_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_KBG);
        ubQexp_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_QEXP);
        ubG_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_G);
        ubBeta_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_BETA);
        ubBe_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_BE);
        ubGp_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_GP);
        ubEg_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_EG);
        ubKkt_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_KKT);
        ubQk_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_QK);
        ubL_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_L);
        ubX00_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_X00);
        ubX11_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_X11);
        ubAfp_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_AFP);
        ubAbf_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_ABF);
        ubAttn_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(UB_ATTN);
        ubU_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_U);
        ubP_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_P);
        ubOacc_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_OACC);
        ubD_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_D);
        ubKt_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(224 * 1024);

        l1K_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_K);
        l1Q_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_Q);
        l1Vb_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_VB);
        l1Kbg_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_KBG);
        l1Qexp_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_QEXP);
        l1A_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_A);
        l1H_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_H);
        l1W_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_W);
        l1Vnew_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_VNEW);
        l1VnewG_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_VNEWG);
        l1Attn_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(L1_ATTN);
        l1Kt_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(256 * 1024);
        l0A_ = buf.template GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(0);
        l0B_ = buf.template GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(0);
        l0C_ = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);

        for (int32_t db = 0; db < 2; ++db) {
            ubQDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kS1Q + static_cast<uint32_t>(db) * kS1SlotBf16);
            ubKDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kS1K + static_cast<uint32_t>(db) * kS1SlotBf16);
            ubVDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kS1V + static_cast<uint32_t>(db) * kS1SlotBf16);
            ubKbgDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kS1Kbg + static_cast<uint32_t>(db) * kS1SlotBf16);
            ubQexpDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kS1Qexp + static_cast<uint32_t>(db) * kS1SlotBf16);
            ubGDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kS1G + static_cast<uint32_t>(db) * kS1SlotG);
            ubBetaDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kS1Beta + static_cast<uint32_t>(db) * kS1SlotBeta);
            ubBeDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kS1Be + static_cast<uint32_t>(db) * kS1SlotBe);
            ubGpDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kS1Gp + static_cast<uint32_t>(db) * kS1SlotGp);
            ubEgDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kS1Eg + static_cast<uint32_t>(db) * kS1SlotEg);
            ubKktDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kS1Kkt + static_cast<uint32_t>(db) * kS1SlotKkt);
            ubQkDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kS1Qk + static_cast<uint32_t>(db) * kS1SlotKkt);
            ubLDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbL + static_cast<uint32_t>(db) * 4 * 1024);
            ubAttnDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbAttn + static_cast<uint32_t>(db) * 2 * 1024);
            ubAbfDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbAbf + static_cast<uint32_t>(db) * 2 * 1024);
            ubUDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbU + static_cast<uint32_t>(db) * 16 * 1024);
            ubPDb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbP + static_cast<uint32_t>(db) * 16 * 1024);
        }
        ubX00_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbX00);
        ubX11_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbX11);
        ubAfp_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbAfp);
        ubR_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbR);
        ubD_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbD);
        for (uint32_t t = 0; t < static_cast<uint32_t>(kTasksPerRound); ++t) {
            l1KDb_[t] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(t * kL1TaskStride);
            l1QDb_[t] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(t * kL1TaskStride + 8 * 1024);
            l1VbDb_[t] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1Vb0 + t * 8 * 1024);
            l1KbgDb_[t] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1Kbg0 + t * 8 * 1024);
            l1ADb_[t] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1A0 + t * 2 * 1024);
            l1HDb_[t] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1H0 + t * 32 * 1024);
            l1WDb_[t] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1W0 + t * 8 * 1024);
        }
        for (uint32_t b = 0; b < 2; ++b) {
            l0ADb_[b] = buf.template GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(b * kL0AbBank);
            l0BDb_[b] = buf.template GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(b * kL0AbBank);
            l0CDb_[b] = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(b * kL0CBank);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
        if ASCEND_IS_AIC {
            ProcessAic();
        }
    }

private:
    template <typename T>
    __aicore__ inline void CopyOut(GlobalTensor<T> dst, LocalTensor<T> src, uint64_t elem, uint32_t nElem)
    {
        DataCopy(dst[elem], src, nElem);
    }

    __aicore__ inline void CubeAB(LocalTensor<bfloat16_t> a, LocalTensor<bfloat16_t> b, LocalTensor<float> c, int m,
                                  int n, int k, bool transposeB, uint16_t kStart = 0)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        LoadL1NzToL0AB(a, b, l0A_, l0B_, m, n, k, transposeB, kStart);
        MmadL0ToL0C(l0A_, l0B_, l0C_, m, n, k, true);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeFp32Ub(c, l0C_, m, n);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    }

    __aicore__ inline void CubeKSplit(LocalTensor<bfloat16_t> a, uint32_t bBaseByte, LocalTensor<float> c)
    {
        for (int tile = 0; tile < 4; ++tile) {
            const uint32_t bByte = static_cast<uint32_t>(tile * kBt * kDim * static_cast<int>(sizeof(bfloat16_t)));
            auto b = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(bBaseByte + bByte);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
            LoadL1NzToL0AB(a, b, l0A_, l0B_, kBt, kDim, kBt, true, static_cast<uint16_t>(tile * 2));
            MmadL0ToL0C(l0A_, l0B_, l0C_, kBt, kDim, kBt, tile == 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeFp32Ub(c, l0C_, kBt, kDim);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    }

    __aicore__ inline void InitState()
    {
        if (hasInitial_ != 0) {
            const int64_t elem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim);
            DataCopy(ubR_, initGm_[elem], kDim * kDim);
        } else {
            Duplicate(ubR_, static_cast<bfloat16_t>(0), kDim * kDim);
        }
        SyncAll();
    }

    __aicore__ inline int64_t TokenBase() const
    {
        return static_cast<int64_t>((static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * seqLen_);
    }

    __aicore__ inline int64_t StateWsElem() const
    {
        const int64_t head = static_cast<int64_t>(batchIdx_) * valueHeadNum_ + headIdx_;
        return head * static_cast<int64_t>(kGdn32WsStride / sizeof(bfloat16_t)) +
               static_cast<int64_t>(kGdn32WsH / sizeof(bfloat16_t));
    }

    // One chunk on this AIV. db 0 is ping, db 1 is pong.
    // q/k land, L2, then L1; the MTE3 flag lets Cube start Stage2 while the
    // rest of this function does sigmoid, cumsum and the row scales.
    __aicore__ inline void Stage1(int32_t db, uint32_t task, uint32_t chunk)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        const int32_t nRow = kBt * kDim;
        const uint16_t evQk = static_cast<uint16_t>(db);
        const uint16_t evV = static_cast<uint16_t>(2 + db);
        const uint16_t evG = static_cast<uint16_t>(4 + db);
        const uint16_t evBeta = static_cast<uint16_t>(6 + db);
        DataCopy(ubQDb_[db], qGm_[token * kDim], nRow);
        DataCopy(ubKDb_[db], kGm_[token * kDim], nRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evQk);
        DataCopy(ubVDb_[db], vGm_[token * kDim], nRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evV);
        DataCopy(ubGDb_[db], gGm_[token], kBt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evG);
        DataCopy(ubBetaDb_[db], betaGm_[token], kBt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evBeta);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evQk);
        L2InplaceQKVF(ubQDb_[db], ubKDb_[db]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        UploadNdToL1(l1KDb_[task], ubKDb_[db], kBt, kDim);
        UploadNdToL1(l1QDb_[task], ubQDb_[db], kBt, kDim);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(task));

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evBeta);
        BetaSigmoidVF(ubBetaDb_[db], ubBeDb_[db]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evG);
        CumsumExp2VF(ubGDb_[db], ubGpDb_[db], ubEgDb_[db]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evV);
        ScaleRowsVF(ubVDb_[db], ubVDb_[db], ubBeDb_[db]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db + 2));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db + 2));
        UploadNdToL1(l1VbDb_[task], ubVDb_[db], kBt, kDim);
        ScaleRowsBetaExpVF(ubKDb_[db], ubKbgDb_[db], ubBeDb_[db], ubEgDb_[db]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db + 2));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db + 2));
        UploadNdToL1(l1KbgDb_[task], ubKbgDb_[db], kBt, kDim);
        ScaleRowsVF(ubQDb_[db], ubQexpDb_[db], ubEgDb_[db]);
    }

    // R is the running state. Chunk 0 starts from initial_state, or zero.
    __aicore__ inline void InitR()
    {
        if (hasInitial_ != 0) {
            const int64_t elem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim);
            DataCopy(ubR_, initGm_[elem], kDim * kDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        } else {
            Duplicate(ubR_, static_cast<bfloat16_t>(0), kDim * kDim);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        }
    }

    // Four [32,128] NZ panels. Same view CubeP and q_exp@H use.
    __aicore__ inline void UploadH(uint32_t task)
    {
        for (int tile = 0; tile < 4; ++tile) {
            auto dst = l1HDb_[task][static_cast<uint32_t>(tile * kBt * kDim)];
            auto src = ubR_[static_cast<uint32_t>(tile * kBt * kDim)];
            UploadNdToL1(dst, src, kBt, kDim);
        }
    }

    // Inverse of UploadNdToL1. Runs on AIC: L1->UB is a cube copy, and the
    // subblock picks which AIV receives this chunk's entry state.
    template <uint8_t Sub>
    __aicore__ inline void DownloadPanel(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> ub)
    {
        constexpr uint16_t kRows = static_cast<uint16_t>(kBt);
        constexpr uint16_t kGap = static_cast<uint16_t>(kDim / 16 - 1);
        for (uint32_t fc = 0; fc < static_cast<uint32_t>(kDim / 16); ++fc) {
            DataCopyL1ToUB<bfloat16_t, Sub>(ub[fc * 16], l1[fc * 16 * kBt], DataCopyParams(kRows, 1, 0, kGap));
        }
    }

    __aicore__ inline void PushH(uint32_t task)
    {
        for (int tile = 0; tile < 4; ++tile) {
            auto src = l1HDb_[task][static_cast<uint32_t>(tile * kBt * kDim)];
            auto dst = ubR_[static_cast<uint32_t>(tile * kBt * kDim)];
            if ((task & 1U) != 0) {
                DownloadPanel<1>(src, dst);
            } else {
                DownloadPanel<0>(src, dst);
            }
        }
    }


    // Tril, 16x16 leaf inverse, A onto L1, then this chunk's Cube can start.
    __aicore__ inline void Stage3(int32_t db, uint32_t task)
    {
        GateAttnVF(ubGpDb_[db], ubBeDb_[db], ubKktDb_[db], ubQkDb_[db], ubLDb_[db], ubAttnDb_[db]);
        InvertAVF(ubLDb_[db], ubX00_, ubX11_, ubPDb_[db], ubAfp_, ubAbfDb_[db]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        UploadNdToL1(l1ADb_[task], ubAbfDb_[db], kBt, kBt);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(task + 4));
    }

    __aicore__ inline void StageVnew()
    {
        Sub(ubOacc_, ubU_, ubP_, kBt * kDim);
        SyncAll();
        Cast(ubQexp_, ubOacc_, RoundMode::CAST_RINT, kBt * kDim);
        __ubuf__ float *gpAddr = (__ubuf__ float *)ubGp_.GetPhyAddr();
        __ubuf__ float *egAddr = (__ubuf__ float *)ubEg_.GetPhyAddr();
        __VEC_SCOPE__
        {
            using namespace AscendC::MicroAPI;
            uint32_t n = static_cast<uint32_t>(kBt);
            MaskReg preg = UpdateMask<float>(n);
            RegTensor<float> gp, last;
            LoadAlign(gp, gpAddr);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(last, gpAddr + (kBt - 1));
            Sub(gp, last, gp, preg);
            Muls(gp, gp, kLn2, preg);
            Exp(gp, gp, preg);
            StoreAlign(egAddr, gp, preg);
        }
        ScaleFp32RowsVF(ubOacc_, ubEg_, ubKbg_);
        SyncAll();
        TransposeKVf(ubK_, ubQ_);
        SyncAll();
        UploadNdToL1(l1Vnew_, ubQexp_, kBt, kDim);
        UploadNdToL1(l1VnewG_, ubKbg_, kBt, kDim);
        for (int tile = 0; tile < 4; ++tile) {
            const uint32_t elems = static_cast<uint32_t>(tile * kBt * kBt);
            UploadNdToL1(l1A_[elems], ubQ_[elems], kBt, kBt);
        }
    }

    __aicore__ inline void StageOutput(uint32_t chunk, bool last)
    {
        ubEg_.SetValue(0, ubGp_.GetValue(kBt - 1) * kLn2);
        SyncAll();
        Exp(ubEg_, ubEg_, 8);
        SyncAll();
        Add(ubOacc_, ubOacc_, ubP_, kBt * kDim);
        Muls(ubOacc_, ubOacc_, scale_, kBt * kDim);
        Cast(ubQ_, ubOacc_, RoundMode::CAST_RINT, kBt * kDim);
        SyncAll();
        const uint64_t token =
            (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * seqLen_ + static_cast<uint64_t>(chunk) * kBt;
        CopyOut(oGm_, ubQ_, token * kDim, kBt * kDim);
        ApplyStateVF(ubR_, ubD_, ubEg_.GetValue(0));
        // Vector store of R must land before MTE3 reads it. Without this, the
        // GM final_state is the pre-update buffer (often zeros on the last chunk).
        SyncAll();
        if (last) {
            const uint64_t stateElem = (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim;
            CopyOut(stateGm_, ubR_, stateElem, kDim * kDim);
        }
        SyncAll();
    }


    // One chunk. Even tasks use L0 bank 0, odd tasks bank 1.
    // kkt and qk are FP32 row-major in that AIV's ping or pong UB slot.
    __aicore__ inline void Stage2(uint32_t task)
    {
        const uint8_t bank = static_cast<uint8_t>(task & 1U);
        const int32_t db = PingPongDb(task);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlag(task));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);

        LoadL1NzToL0AB(l1KDb_[task], l1KDb_[task], l0ADb_[bank], l0BDb_[bank], kBt, kBt, kDim, false, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kBt, kDim, true, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeFp32Ub(ubKktDb_[db], l0CDb_[bank], kBt, kBt, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
        LoadL1NzToL0AB(l1QDb_[task], l1KDb_[task], l0ADb_[bank], l0BDb_[bank], kBt, kBt, kDim, false, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kBt, kDim, true, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeFp32Ub(ubQkDb_[db], l0CDb_[bank], kBt, kBt, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlag(task));
    }

    // u = A @ vb, w = A @ kbg, P = w @ H. u and P return as FP32.
    __aicore__ inline void Stage4(uint32_t task)
    {
        const uint8_t bank = static_cast<uint8_t>(task & 1U);
        const int32_t db = PingPongDb(task);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlag(task + 4));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);

        LoadL1NzToL0AB(l1ADb_[task], l1VbDb_[task], l0ADb_[bank], l0BDb_[bank], kBt, kDim, kBt, true, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kBt, true, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeFp32Ub(ubUDb_[db], l0CDb_[bank], kBt, kDim, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
        LoadL1NzToL0AB(l1ADb_[task], l1KbgDb_[task], l0ADb_[bank], l0BDb_[bank], kBt, kDim, kBt, true, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kBt, true, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeBf16L1(l1WDb_[task], l0CDb_[bank], kBt, kDim);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(bank);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlag(task + 4));
    }

    // P = w @ H. H is this chunk's entry state, already on L1.
    // waitEntry is only chunk 0; later chunks already waited on the producer's kFlagNext.
    __aicore__ inline void StageP(uint32_t task, bool waitEntry)
    {
        const uint8_t bank = static_cast<uint8_t>(task & 1U);
        const int32_t db = PingPongDb(task);
        if (waitEntry) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagH, task));
        }
        // Stage4 posts M_MTE1 before the w fixpipe. Stall MTE1 on that store.
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(bank);
        for (int tile = 0; tile < 4; ++tile) {
            auto hTile = l1HDb_[task][static_cast<uint32_t>(tile * kBt * kDim)];
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
            LoadL1NzToL0AB(l1WDb_[task], hTile, l0ADb_[bank], l0BDb_[bank], kBt, kDim, kBt, true,
                           static_cast<uint16_t>(tile * 2), bank);
            MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kBt, tile == 0, bank);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeFp32Ub(ubPDb_[db], l0CDb_[bank], kBt, kDim, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(AicSeen(kFlagP, task));
    }

    // D = k_hat^T @ v_new_g, then o_acc = q_exp @ H + attn @ v_new.
    __aicore__ inline void Stage6(uint32_t task)
    {
        const uint8_t bank = static_cast<uint8_t>(task & 1U);
        const int32_t db = PingPongDb(task);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagV, task));

        // k^T @ v_new_g, 32 rows at a time. A full 128-row Fixpipe misses on AIV1.
        // D always uses L0 bank 0. Bank 1's transposed load does not match bank 0.
        constexpr uint8_t dBank = 0;
        for (int part = 0; part < 4; ++part) {
            auto kPart = l1KDb_[task][static_cast<uint32_t>(part * 2 * kBt * 16)];
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dBank);
            LoadL1NzToL0AB(kPart, l1KbgDb_[task], l0ADb_[dBank], l0BDb_[dBank], kBt, kDim, kBt, true, 0, dBank,
                           true);
            MmadL0ToL0C(l0ADb_[dBank], l0BDb_[dBank], l0CDb_[dBank], kBt, kDim, kBt, true, dBank);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dBank);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dBank);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dBank);
            FixpipeFp32Ub(ubD_[static_cast<uint32_t>(part * kBt * kDim)], l0CDb_[dBank], kBt, kDim, bank);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dBank);
        }

        for (int tile = 0; tile < 4; ++tile) {
            auto hTile = l1HDb_[task][static_cast<uint32_t>(tile * kBt * kDim)];
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
            LoadL1NzToL0AB(l1QDb_[task], hTile, l0ADb_[bank], l0BDb_[bank], kBt, kDim, kBt, true,
                           static_cast<uint16_t>(tile * 2), bank);
            MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kBt, tile == 0, bank);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
        LoadL1NzToL0AB(l1ADb_[task], l1VbDb_[task], l0ADb_[bank], l0BDb_[bank], kBt, kDim, kBt, true, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kBt, false, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeFp32Ub(ubPDb_[db], l0CDb_[bank], kBt, kDim, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(AicSeen(kFlagO, task));
    }

    // v_new, v_new_g, attn and q_exp onto L1, then this chunk's Stage 6 can start.
    __aicore__ inline void Stage5(int32_t db, uint32_t task)
    {
        const float gLast = ubGpDb_[db].GetValue(kBt - 1);
        VnewVF(ubUDb_[db], ubPDb_[db], ubGpDb_[db], ubVDb_[db], ubKbgDb_[db], gLast);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        UploadNdToL1(l1VbDb_[task], ubVDb_[db], kBt, kDim);
        UploadNdToL1(l1KbgDb_[task], ubKbgDb_[db], kBt, kDim);
        UploadNdToL1(l1ADb_[task], ubAttnDb_[db], kBt, kBt);
        UploadNdToL1(l1QDb_[task], ubQexpDb_[db], kBt, kDim);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagV);
    }

    __aicore__ inline void Stage7(int32_t db, uint32_t task, uint32_t chunk, bool hasNext, bool last)
    {
        ubEgDb_[db].SetValue(0, ubGpDb_[db].GetValue(kBt - 1) * kLn2);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        Exp(ubEgDb_[db], ubEgDb_[db], 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        ApplyStateVF(ubR_, ubD_, ubEgDb_[db].GetValue(0));
        ScaleOVf(ubPDb_[db], ubVDb_[db], scale_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        DataCopy(oGm_[token * kDim], ubVDb_[db], kBt * kDim);
        if (hasNext) {
            UploadH(static_cast<uint32_t>((task + 1) % kTasksPerRound));
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagNext);
        }
        if (last) {
            const int64_t stateElem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim);
            DataCopy(stateGm_[stateElem], ubR_, kDim * kDim);
        }
    }


    __aicore__ inline void ProcessAiv()
    {
        if (!isActive_) {
            return;
        }
        const uint32_t sub = subBlockIdx_;
        const uint32_t chunkNum = seqLen_ / kBt;
        const uint32_t totalHeads = batchSize_ * valueHeadNum_;
        const uint32_t nRound = (chunkNum + static_cast<uint32_t>(kTasksPerRound) - 1) / static_cast<uint32_t>(kTasksPerRound);
        for (uint32_t headLinear = headBegin_; headLinear < totalHeads; headLinear += aicNum_) {
            batchIdx_ = valueHeadNum_ == 0 ? 0 : headLinear / valueHeadNum_;
            headIdx_ = valueHeadNum_ == 0 ? 0 : headLinear - batchIdx_ * valueHeadNum_;
            InitR();
            for (uint32_t round = 0; round < nRound; ++round) {
                // Prologue is this AIV's first chunk. Its Stage1/Stage3 do not read H.
                // The second chunk's Stage1/Stage3 run after that tail, overlapping the
                // other AIV's StageP/Stage6.
                const uint32_t task0 = sub;
                const uint32_t chunk0 = round * static_cast<uint32_t>(kTasksPerRound) + task0;
                if (chunk0 < chunkNum) {
                    Stage1(0, task0, chunk0);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(task0));
                    Stage3(0, task0);
                }
                for (int db = 0; db < 2; ++db) {
                    const uint32_t task = sub + static_cast<uint32_t>(db) * 2;
                    const uint32_t chunk = round * static_cast<uint32_t>(kTasksPerRound) + task;
                    if (chunk >= chunkNum) {
                        continue;
                    }
                    if (db != 0) {
                        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(task));
                        Stage3(db, task);
                    }
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(task + 4));
                    if (chunk == 0) {
                        UploadH(task);
                        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagH);
                    }
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagP);
                    Stage5(db, task);
                    if (chunk > 0) {
                        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagGo);
                    }
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagO);
                    const bool hasNext = chunk + 1 < chunkNum;
                    const bool last = chunk + 1 == chunkNum;
                    Stage7(db, task, chunk, hasNext, last);
                    if (db == 0) {
                        const uint32_t taskN = sub + 2;
                        const uint32_t chunkN = round * static_cast<uint32_t>(kTasksPerRound) + taskN;
                        if (chunkN < chunkNum) {
                            Stage1(1, taskN, chunkN);
                        }
                    }
                }
            }
            if (headLinear + aicNum_ < totalHeads) {
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagDone);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagHead);
                AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessAic()
    {
        if (!isActive_) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
        const uint32_t chunkNum = seqLen_ / kBt;
        const uint32_t totalHeads = batchSize_ * valueHeadNum_;
        const uint32_t nRound = (chunkNum + static_cast<uint32_t>(kTasksPerRound) - 1) / static_cast<uint32_t>(kTasksPerRound);
        for (uint32_t headLinear = headBegin_; headLinear < totalHeads; headLinear += aicNum_) {
            for (uint32_t round = 0; round < nRound; ++round) {
                const uint32_t base = round * static_cast<uint32_t>(kTasksPerRound);
                // Prologue covers each AIV's first chunk. The other chunk's Stage2
                // fills the next Stage5 bubble; its Stage4 follows that Stage6,
                // after the AIV has run Stage3.
                for (uint32_t t = 0; t < 2; ++t) {
                    if (base + t >= chunkNum) {
                        break;
                    }
                    Stage2(t);
                }
                for (uint32_t t = 0; t < 2; ++t) {
                    if (base + t >= chunkNum) {
                        break;
                    }
                    Stage4(t);
                }
                for (uint32_t t = 0; t < static_cast<uint32_t>(kTasksPerRound); ++t) {
                    if (base + t >= chunkNum) {
                        break;
                    }
                    const bool first = round == 0 && t == 0;
                    if (!first) {
                        const uint32_t srcTask = (t + static_cast<uint32_t>(kTasksPerRound) - 1) %
                                                  static_cast<uint32_t>(kTasksPerRound);
                        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagNext, srcTask));
                    }
                    StageP(t, first);
                    if (!first) {
                        PushH(t);
                        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(AicSeen(kFlagGo, t));
                    }
                    const bool prefetch = t == 1 || t == 2;
                    const uint32_t ahead = t + 1;
                    if (prefetch && base + ahead < chunkNum) {
                        Stage2(ahead);
                    }
                    Stage6(t);
                    if (prefetch && base + ahead < chunkNum) {
                        Stage4(ahead);
                    }
                }
            }
            if (headLinear + aicNum_ < totalHeads) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagDone, 0));
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagDone, 1));
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(AicSeen(kFlagHead, 0));
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(AicSeen(kFlagHead, 1));
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    }

    GlobalTensor<bfloat16_t> qGm_;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<bfloat16_t> oGm_;
    GlobalTensor<bfloat16_t> stateGm_;
    GlobalTensor<bfloat16_t> stateWs_;
    GlobalTensor<float> flagGm_;
    GM_ADDR wsAddr_ = nullptr;
    GlobalTensor<bfloat16_t> initGm_;

    LocalTensor<bfloat16_t> ubR_, ubQ_, ubK_, ubV_, ubKbg_, ubQexp_, ubBeta_, ubAbf_, ubAttn_, ubKt_;
    LocalTensor<float> ubG_, ubBe_, ubGp_, ubEg_, ubKkt_, ubQk_, ubL_, ubX00_, ubX11_, ubAfp_, ubU_, ubP_, ubOacc_, ubD_;
    LocalTensor<bfloat16_t> l1K_, l1Q_, l1Vb_, l1Kbg_, l1Qexp_, l1A_, l1H_, l1W_, l1Vnew_, l1VnewG_, l1Attn_, l1Kt_;
    LocalTensor<bfloat16_t> l0A_, l0B_;
    LocalTensor<float> l0C_;

    LocalTensor<bfloat16_t> ubQDb_[2], ubKDb_[2], ubVDb_[2], ubKbgDb_[2], ubQexpDb_[2], ubBetaDb_[2];
    LocalTensor<bfloat16_t> ubAttnDb_[2], ubAbfDb_[2];
    LocalTensor<float> ubGDb_[2], ubBeDb_[2], ubGpDb_[2], ubEgDb_[2], ubKktDb_[2], ubQkDb_[2];
    LocalTensor<float> ubLDb_[2], ubUDb_[2], ubPDb_[2];
    LocalTensor<bfloat16_t> l1KDb_[4], l1QDb_[4], l1VbDb_[4], l1KbgDb_[4], l1ADb_[4], l1HDb_[4], l1WDb_[4];
    LocalTensor<bfloat16_t> l0ADb_[2], l0BDb_[2];
    LocalTensor<float> l0CDb_[2];

    uint32_t subBlockIdx_ = 0;
    uint32_t aicNum_ = 1;
    uint32_t headBegin_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t valueHeadNum_ = 0;
    uint32_t seqLen_ = 0;
    uint32_t batchIdx_ = 0;
    uint32_t headIdx_ = 0;
    uint32_t hasInitial_ = 0;
    float scale_ = 0.f;
    bool isActive_ = false;
    OnChipBuffer buf;
};

}  // namespace GdnFwd32

#endif
