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
        // The other half of a 64-token step shares this AIV's ping-pong db, so its
        // cumsum lives past ubR_ (224 KiB, 16 KiB) and stays inside the 248 KiB UB.
        ubGd_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(240 * 1024);
        ubGpd_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(240 * 1024 + 512);
        ubEgd_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(240 * 1024 + 1024);
        // 16x16 identity for the register solve. ubEgd_ occupies 512 bytes from +1024.
        ubEye_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(242 * 1024);
        // Five 64-lane gather indices. Identity above is 32x32, so the indices sit after it.
        ubSh_ = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(242 * 1024 + 4096);
        futBe_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbFutBe);
        futGp_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbFutGp);
        futEg_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbFutEg);

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
            // u, P and o are fixpiped bf16. Same byte address as the float slot.
            ubUb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbU + static_cast<uint32_t>(db) * 16 * 1024);
            ubPb_[db] = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbP + static_cast<uint32_t>(db) * 16 * 1024);
        }
        ubX00_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbX00);
        ubX11_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbX11);
        ubAfp_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbAfp);
        ubR_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbR);
        ubD_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbD);
        ubDb_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbD);
        // Second half's delta. Aliases ubU (dead from the end of Stage5 until Stage4
        // of the next step). Stage6 of the high half writes it; Apply reads it
        // before that Stage4. bf16, so the 64 rows occupy the first 16KB.
        ubD1_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbU);
        ubD1b_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbU);
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
    __aicore__ inline void Stage1(int32_t db, uint32_t task, uint32_t chunk, int32_t prefetchDb = 0,
                                 uint32_t prefetchChunk = 0xFFFFFFFFu)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        const int32_t nRow = kBt * kDim;
        const uint16_t evQk = static_cast<uint16_t>(db);
        const uint16_t evV = static_cast<uint16_t>(2 + db);
        const uint16_t evG = static_cast<uint16_t>(4 + db);
        const uint16_t evBeta = static_cast<uint16_t>(6 + db);
        const bool l2Done = l2Ready_[db] != 0;
        const bool onL1 = qkOnL1_[db] != 0;
        const bool qkReady = !l2Done && qkReady_[db] != 0;
        const bool vReady = vReady_[db] != 0;
        qkReady_[db] = 0;
        vReady_[db] = 0;
        l2Ready_[db] = 0;
        qkOnL1_[db] = 0;
        // Previous step's MTE3 is done with these ND slots: q/k, v/o, q_exp/k_beta_g.
        // A hit from PrefetchQKV already consumed MTE3_MTE2(evV) and left MTE2_V pending.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evQk);
        if (!vReady) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evV);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(4 + db));
        if (!l2Done && !qkReady) {
            DataCopy(ubQDb_[db], qGm_[token * kDim], nRow);
            DataCopy(ubKDb_[db], kGm_[token * kDim], nRow);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evQk);
        }
        if (!vReady) {
            DataCopy(ubVDb_[db], vGm_[token * kDim], nRow);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evV);
        }
        if (!l2Done) {
            DataCopy(ubGDb_[db], gGm_[token], kBt);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evG);
            DataCopy(ubBetaDb_[db], betaGm_[token], kBt);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evBeta);
        }
        // Future q/k goes out after this chunk's loads, so L2 below overlaps it
        // instead of waiting behind it.
        if (prefetchChunk < seqLen_ / kBt) {
            PrefetchQK(prefetchDb, prefetchChunk);
        }

        LocalTensor<bfloat16_t> vNz = ubDb_[static_cast<uint32_t>(kDim / 2) * kDim];
        if (l2Done) {
            // EarlyL2 already wrote q_exp, k_beta_g and the gate vectors.
            CopyF32VF(ubBeDb_[db], futBe_, static_cast<uint32_t>(kBt));
            CopyF32VF(ubGpDb_[db], futGp_, static_cast<uint32_t>(kBt));
            CopyF32VF(ubEgDb_[db], futEg_, static_cast<uint32_t>(kBt * 2));
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evBeta);
            BetaSigmoidVF(ubBetaDb_[db], ubBeDb_[db]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evG);
            CumsumExp2VF<true>(ubGDb_[db], ubGpDb_[db], ubEgDb_[db], ubSh_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evQk);
            // q/k NZ fill this chunk's P slot (16KB, P is not written until next step's
            // StageP). v NZ sits in the unused upper half of D.
            LocalTensor<bfloat16_t> qNz = ubPb_[db];
            LocalTensor<bfloat16_t> kNz = ubPb_[db][static_cast<uint32_t>(kBt * kDim)];
            L2InplaceQKVF(ubQDb_[db], ubKDb_[db], ubKbgDb_[db], ubQexpDb_[db], ubBeDb_[db], ubEgDb_[db], qNz, kNz);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evQk);
        }
        // ScaleRows is queued on V before the q/k copy wait, so the copy overlaps it.
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evV);
        ScaleRowsVF<true>(ubVDb_[db], ubVDb_[db], ubBeDb_[db],
                          reinterpret_cast<__ubuf__ bfloat16_t *>(vNz.GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV);
        if (!onL1) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evQk);
            CopyNzToL1(l1KDb_[task], ubPb_[db][static_cast<uint32_t>(kBt * kDim)], kBt, kDim);
            CopyNzToL1(l1QDb_[task], ubPb_[db], kBt, kDim);
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(task));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evQk);
        // k_beta_g must land before the next Stage5 of this db rewrites ubKbg.
        // The scaled-v copy is queued behind that flag and overlaps the other db's Vnew.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV);
        CopyNzToL1(l1KbgDb_[task], ubKbgDb_[db], kBt, kDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evV);
        CopyNzToL1(l1VbDb_[task], vNz, kBt, kDim);
    }

    // Normalize the prefetched chunk while the cube is still in Stage6. q/k NZ
    // land in the other db's P slot, which StageP will not write until the next
    // step, and only after PublishEarlyQk's copy has finished.
    __aicore__ inline void EarlyL2(int32_t db, int32_t nzDb, uint32_t chunk)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        const uint16_t evQk = static_cast<uint16_t>(db);
        const uint16_t evG = static_cast<uint16_t>(4 + db);
        const uint16_t evBeta = static_cast<uint16_t>(6 + db);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(4 + db));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(4 + db));
        DataCopy(ubGDb_[db], gGm_[token], kBt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evG);
        DataCopy(ubBetaDb_[db], betaGm_[token], kBt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evBeta);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evQk);
        qkReady_[db] = 0;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evBeta);
        BetaSigmoidVF(ubBetaDb_[db], futBe_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evG);
        CumsumExp2VF<true>(ubGDb_[db], futGp_, futEg_, ubSh_);
        LocalTensor<bfloat16_t> qNz = ubPb_[nzDb];
        LocalTensor<bfloat16_t> kNz = ubPb_[nzDb][static_cast<uint32_t>(kBt * kDim)];
        L2InplaceQKVF(ubQDb_[db], ubKDb_[db], ubKbgDb_[db], ubQexpDb_[db], futBe_, futEg_, qNz, kNz);
        l2Ready_[db] = 1;
    }

    __aicore__ inline void PublishEarlyQk(int32_t nzDb, uint32_t task)
    {
        CopyNzToL1(l1KDb_[task], ubPb_[nzDb][static_cast<uint32_t>(kBt * kDim)], kBt, kDim);
        CopyNzToL1(l1QDb_[task], ubPb_[nzDb], kBt, kDim);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(task));
    }

    // q and k of the chunk this db serves next iteration. ubQ/ubK were consumed by
    // the previous step's L2, so the copy is issued with no scalar wait and runs
    // under this step's L2.
    __aicore__ inline void PrefetchQK(int32_t db, uint32_t chunk)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        const int32_t nRow = kBt * kDim;
        const uint16_t evQk = static_cast<uint16_t>(db);
        DataCopy(ubQDb_[db], qGm_[token * kDim], nRow);
        DataCopy(ubKDb_[db], kGm_[token * kDim], nRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evQk);
        qkReady_[db] = 1;
    }

    // v lands after Apply releases ubV[db]. Stage1 waits for it only after L2, so
    // this copy overlaps the next step's normalize.
    __aicore__ inline void PrefetchV(int32_t db, uint32_t chunk)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        const int32_t nRow = kBt * kDim;
        const uint16_t evV = static_cast<uint16_t>(2 + db);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evV);
        DataCopy(ubVDb_[db], vGm_[token * kDim], nRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evV);
        vReady_[db] = 1;
    }

    // This AIV's 64 K rows. Chunk 0 starts from initial_state, or zero.
    // The half stays in ubR_ for the whole head and is updated in place.
    __aicore__ inline void InitH()
    {
        constexpr uint32_t nElem = static_cast<uint32_t>(kDim / 2) * kDim;
        if (hasInitial_ != 0) {
            const int64_t elem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim +
                static_cast<uint64_t>(subBlockIdx_) * nElem);
            DataCopy(ubR_, initGm_[elem], nElem);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        } else {
            Duplicate(ubR_, static_cast<bfloat16_t>(0), nElem);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        }
    }

    // g' of the half this AIV does not own. Both halves of a 64-token step use the
    // same ping-pong db, so the other cumsum lives in ubGd_/ubGpd_.
    __aicore__ inline void LoadDecay(uint32_t chunk)
    {
        const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
        DataCopy(ubGd_, gGm_[token], kBt);
        // Event 0/1 stay with the q/k prefetch. Event 4 is free once Stage1 has
        // consumed g's MTE2_V.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        SumLastVF(ubGd_, ubGpd_);
    }

    // This AIV's 64 ND rows land in the one [128,128] NZ at l1H.
    // Panel height is 128, so StageP and Stage6 load H once with K=128.
    __aicore__ inline void UploadH()
    {
        const uint32_t row0 = subBlockIdx_ * (kDim / 2);
        for (int tile = 0; tile < 2; ++tile) {
            CopyNdRowsToFullNz(l1HDb_[0], ubR_[static_cast<uint32_t>(tile * kBt * kDim)],
                               row0 + static_cast<uint32_t>(tile) * kBt, kBt, kDim, kDim);
        }
    }

    // Tiles already stored as tight NZ by Apply. Two copies, no vector pack.
    __aicore__ inline void UploadHnz(LocalTensor<bfloat16_t> nz0, LocalTensor<bfloat16_t> nz1)
    {
        const uint32_t tileBase = subBlockIdx_ * 2;
        CopyNzToL1(l1HDb_[0][tileBase * kBt * kDim], nz0, kBt, kDim);
        CopyNzToL1(l1HDb_[0][(tileBase + 1) * kBt * kDim], nz1, kBt, kDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
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
        GateAttnVF(ubEgDb_[db], ubBeDb_[db], ubKktDb_[db], ubQkDb_[db], ubLDb_[db], ubAttnDb_[db]);
        InvertAVF(ubLDb_[db], ubX00_, ubX11_, ubPDb_[db], ubAfp_, ubAbfDb_[db], ubEye_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        CopyNdToL1(l1ADb_[task], ubAbfDb_[db], kBt, kBt);
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
        UploadNdToL1(l1Vnew_, ubQexp_, ubK_, kBt, kDim);
        UploadNdToL1(l1VnewG_, ubKbg_, ubQ_, kBt, kDim);
        for (int tile = 0; tile < 4; ++tile) {
            const uint32_t elems = static_cast<uint32_t>(tile * kBt * kBt);
            UploadNdToL1(l1A_[elems], ubQ_[elems], ubK_, kBt, kBt);
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
        ApplyStateVF<false>(ubR_, ubDb_, ubDb_, ubEg_.GetValue(0), 0.f, static_cast<uint16_t>(kBt * 2));
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
        // w uses the other L0 bank so its load overlaps u's mad.
        const uint8_t wBank = static_cast<uint8_t>(bank ^ 1U);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(wBank);
        LoadL1NzToL0AB(l1ADb_[task], l1KbgDb_[task], l0ADb_[wBank], l0BDb_[wBank], kBt, kDim, kBt, true, 0, wBank);
        MmadL0ToL0C(l0ADb_[wBank], l0BDb_[wBank], l0CDb_[wBank], kBt, kDim, kBt, true, wBank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(wBank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(wBank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeBf16Ub(ubUb_[db], l0CDb_[bank], kBt, kDim, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(wBank);
        FixpipeBf16L1(l1WDb_[task], l0CDb_[wBank], kBt, kDim);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(wBank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(bank);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlag(task + 4));
    }

    // Issue w@H. The two halves use different L0 banks, so the second load
    // overlaps the first mad. StagePStore fixpipes after both are in flight.
    __aicore__ inline void StagePCompute(uint32_t task)
    {
        const uint8_t bank = static_cast<uint8_t>(task & 1U);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
        LoadL1NzToL0AB(l1WDb_[task], l1HDb_[0], l0ADb_[bank], l0BDb_[bank], kBt, kDim, kDim, true, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kDim, true, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
    }

    __aicore__ inline void StagePStore(uint32_t task)
    {
        const uint8_t bank = static_cast<uint8_t>(task & 1U);
        const int32_t db = PingPongDb(task);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeBf16Ub(ubPb_[db], l0CDb_[bank], kBt, kDim, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(AicSeen(kFlagP, task));
    }

    // D = k_hat^T @ v_new_g, then o_acc = q_exp @ H + attn @ v_new.
    // dBuf is ubD_ for the low half and ubD1_ for the high half, so the second
    // Stage6 does not wait for the first delta to be consumed.
    __aicore__ inline void Stage6(uint32_t task, LocalTensor<bfloat16_t> dBuf)
    {
        const uint8_t bank = static_cast<uint8_t>(task & 1U);
        const int32_t db = PingPongDb(task);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagV, task));

        // k^T @ v_new_g, 32 rows at a time. A full 128-row Fixpipe misses on AIV1.
        // Transposed K stays on L0 bank 0. v_new_g is loaded once. Two L0C slices
        // let part N+1's mad run while part N's fixpipe is still draining, and the
        // next K slice is issued before that fixpipe's wait.
        constexpr uint8_t dBank = 0;
        LocalTensor<float> slice0 = l0CDb_[dBank];
        LocalTensor<float> slice1 = l0CDb_[dBank][static_cast<uint32_t>(kBt * kDim)];
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dBank);
        LoadL1NzToL0AB(l1KDb_[task], l1KbgDb_[task], l0ADb_[dBank], l0BDb_[dBank], kBt, kDim, kBt, true, 0, dBank,
                       true, false);
        for (int part = 0; part < 4; ++part) {
            const uint8_t ev = static_cast<uint8_t>(part & 1);
            LocalTensor<float> l0c = (ev == 0) ? slice0 : slice1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dBank);
            MmadL0ToL0C(l0ADb_[dBank], l0BDb_[dBank], l0c, kBt, kDim, kBt, true, ev);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ev);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dBank);
            if (part + 1 < 4) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dBank);
                auto kPart = l1KDb_[task][static_cast<uint32_t>((part + 1) * 2 * kBt * 16)];
                LoadL1NzToL0A(kPart, l0ADb_[dBank], kBt, kBt, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dBank);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ev);
            const uint8_t dstSub = static_cast<uint8_t>(part >= 2);
            const uint32_t localRow = static_cast<uint32_t>(part & 1) * kBt;
            FixpipeBf16Ub(dBuf[localRow * kDim], l0c, kBt, kDim, dstSub);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ev);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
        LoadL1NzToL0AB(l1QDb_[task], l1HDb_[0], l0ADb_[bank], l0BDb_[bank], kBt, kDim, kDim, true, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kDim, true, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bank);
        LoadL1NzToL0AB(l1ADb_[task], l1VbDb_[task], l0ADb_[bank], l0BDb_[bank], kBt, kDim, kBt, true, 0, bank);
        MmadL0ToL0C(l0ADb_[bank], l0BDb_[bank], l0CDb_[bank], kBt, kDim, kBt, false, bank);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        FixpipeBf16Ub(ubPb_[db], l0CDb_[bank], kBt, kDim, bank);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(kFlagO);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagO + kAiv1FlagOff));
    }

    // v_new, v_new_g and q_exp are already tight NZ. attn stays ND (32 columns,
    // two bursts). ScaleO reuses ubV after the v_new copy.
    __aicore__ inline void Stage5(int32_t db, uint32_t task)
    {
        const uint16_t evV = static_cast<uint16_t>(2 + db);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evV);
        VnewVF(ubUb_[db], ubPb_[db], ubEgDb_[db], ubVDb_[db], ubKbgDb_[db], ubBetaDb_[db]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        CopyNzToL1(l1VbDb_[task], ubVDb_[db], kBt, kDim);
        CopyNzToL1(l1KbgDb_[task], ubKbgDb_[db], kBt, kDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(db));
        CopyNdToL1(l1ADb_[task], ubAttnDb_[db], kBt, kBt);
        CopyNzToL1(l1QDb_[task], ubQexpDb_[db], kBt, kDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(4 + db));
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagV);
    }

    // AIV0 owns even chunks, AIV1 owns odd chunks. Each AIV ping-pongs its own
    // two chunks, so the L1 slot is owner + slot*2: tasks 0,2 then 1,3.
    __aicore__ inline uint32_t TaskOf(uint32_t chunk) const
    {
        const uint32_t owner = chunk & 1U;
        const uint32_t slot = (chunk >> 1) & 1U;
        return owner + slot * 2U;
    }

    // exp(g'[idx]). Own half keeps the full prefix in gp[31]. The other half
    // stores sum(g)/ln2 at gp[0]. CrossCoreWait on PIPE_V does not order GetValue.
    __aicore__ inline float DecayOf(LocalTensor<float> gp, int32_t idx)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        ubEgd_.SetValue(0, gp.GetValue(idx) * kLn2);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        Exp(ubEgd_, ubEgd_, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return ubEgd_.GetValue(0);
    }

    // writeO stores this half's o. last stores this AIV's half of final_state.
    // State itself is updated once per 64-token step, before these stores.
    __aicore__ inline void ApplyHalf(int32_t db, uint32_t chunk, bool last, bool writeO)
    {
        if (writeO) {
            // v_new's UB->L1 copy still reads ubV.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(db));
            ScaleOVf(ubPb_[db], ubVDb_[db], scale_);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<uint16_t>(db));
        if (writeO) {
            const int64_t token = TokenBase() + static_cast<int64_t>(chunk) * kBt;
            DataCopy(oGm_[token * kDim], ubVDb_[db], kBt * kDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<uint16_t>(2 + db));
        }
        if (last) {
            constexpr uint32_t nElem = static_cast<uint32_t>(kDim / 2) * kDim;
            const int64_t stateElem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim +
                static_cast<uint64_t>(subBlockIdx_) * nElem);
            DataCopy(stateGm_[stateElem], ubR_, nElem);
        }
    }

    __aicore__ inline void Stage7(int32_t db, uint32_t chunk, bool last, bool writeO)
    {
        ApplyHalf(db, chunk, last, writeO);
    }


    // Diagonal of the 32x32 identity. Filled once; Solve32 reloads it per group.
    __aicore__ inline void InitEye()
    {
        Duplicate(ubEye_, 0.f, kBt * kBt);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        for (int i = 0; i < kBt; ++i) {
            ubEye_.SetValue(i * kBt + i, 1.f);
        }
        for (int s = 0; s < 5; ++s) {
            const uint32_t dist = 1U << s;
            for (int i = 0; i < 64; ++i) {
                const uint32_t src = (static_cast<uint32_t>(i) >= dist && i < kBt) ? static_cast<uint32_t>(i) - dist : 0U;
                ubSh_.SetValue(s * 64 + i, src);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }

    __aicore__ inline void ProcessAiv()
    {
        if (!isActive_) {
            return;
        }
        InitEye();
        // First prepare has no previous MTE3 reader of these ND slots.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
        const uint32_t sub = subBlockIdx_;
        const uint32_t chunkNum = seqLen_ / kBt;
        const uint32_t totalHeads = batchSize_ * valueHeadNum_;
        for (uint32_t headLinear = headBegin_; headLinear < totalHeads; headLinear += aicNum_) {
            batchIdx_ = valueHeadNum_ == 0 ? 0 : headLinear / valueHeadNum_;
            headIdx_ = valueHeadNum_ == 0 ? 0 : headLinear - batchIdx_ * valueHeadNum_;
            qkReady_[0] = 0;
            qkReady_[1] = 0;
            vReady_[0] = 0;
            vReady_[1] = 0;
            l2Ready_[0] = 0;
            l2Ready_[1] = 0;
            qkOnL1_[0] = 0;
            qkOnL1_[1] = 0;
            InitH();
            // One KDA step is 64 tokens. AIV0 takes the low 32, AIV1 the high 32.
            // Both halves read the same entry H. State is H = g1*(g0*H+D0)+D1.
            // Stage5 is issued as soon as P lands so Stage6 is not stuck behind the
            // next step's prepare. That prepare overlaps Stage6. D1 sits in ubD1_
            // (the ubU window) so the high Stage6 does not wait for Apply(D0).
            if (sub == 0 && chunkNum > 0) {
                Stage1(0, TaskOf(0), 0);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(TaskOf(0)));
                Stage3(0, TaskOf(0));
            }
            if (sub == 1 && chunkNum > 1) {
                Stage1(0, TaskOf(1), 1);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(TaskOf(1)));
                Stage3(0, TaskOf(1));
            }
            for (uint32_t c0 = 0; c0 < chunkNum; c0 += 2) {
                const uint32_t c1 = c0 + 1;
                const bool has1 = c1 < chunkNum;
                const int32_t db = static_cast<int32_t>((c0 >> 1) & 1U);
                const int32_t dbN = 1 - db;
                const bool own0 = sub == 0;
                const uint32_t n0 = c0 + 2;
                const uint32_t n1 = c0 + 3;
                const bool hasN0 = n0 < chunkNum;
                const bool hasN1 = n1 < chunkNum;
                const bool hasNext = hasN0 || hasN1;
                if (c0 != 0) {
                    // Previous step's Apply set event 6 after writing ubR_.
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
                }
                // Previous step published q/k from this step's P slot. StageP
                // fixpipes that slot, so the copy has to finish before H is visible.
                const int32_t held = db ^ 1;
                if (qkOnL1_[held] != 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(6 + held));
                }
                UploadH();
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagH);
                // Next chunk's q/k hit L1 while StageP is in flight, so Stage2
                // overlaps the rest of this step and Stage6 covers Stage5+Stage3.
                const uint32_t fut = own0 ? (c0 + 4) : (c1 + 4);
                if (own0 && hasN0) {
                    Stage1(dbN, TaskOf(n0), n0, db, fut);
                }
                if (!own0 && hasN1) {
                    Stage1(dbN, TaskOf(n1), n1, db, fut);
                }
                if (own0) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(TaskOf(c0) + 4));
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagP);
                    Stage5(db, TaskOf(c0));
                } else if (has1) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(TaskOf(c1) + 4));
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagP);
                    Stage5(db, TaskOf(c1));
                }
                if (own0) {
                    if (has1) {
                        LoadDecay(c1);
                    }
                } else {
                    LoadDecay(c0);
                }
                if (own0 && hasN0) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(TaskOf(n0)));
                    Stage3(dbN, TaskOf(n0));
                }
                if (!own0 && hasN1) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(TaskOf(n1)));
                    Stage3(dbN, TaskOf(n1));
                }
                int32_t earlyDb = -1;
                int32_t earlyNz = 0;
                uint32_t earlyTask = 0;
                if (fut < chunkNum && qkReady_[db] != 0) {
                    EarlyL2(db, dbN, fut);
                    earlyDb = db;
                    earlyNz = dbN;
                    earlyTask = TaskOf(fut);
                }
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagO);
                if (earlyDb >= 0) {
                    // Vector pipe: L2, then the O flag, then this set. Scalar waits
                    // here so the L1 copy neither races L2 nor the cube's Stage6.
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
                    PublishEarlyQk(earlyNz, earlyTask);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint16_t>(6 + earlyDb));
                    qkOnL1_[earlyDb] = 1;
                }
                const float decay0 = own0 ? DecayOf(ubGpDb_[db], kBt - 1) : DecayOf(ubGpd_, 0);
                float decay1 = 0.f;
                if (has1) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagO);
                    decay1 = own0 ? DecayOf(ubGpd_, 0) : DecayOf(ubGpDb_[db], kBt - 1);
                }
                if (has1) {
                    ApplyStateVF<true>(ubR_, ubDb_, ubD1b_, decay0, decay1, static_cast<uint16_t>(kDim / 2));
                } else {
                    ApplyStateVF<false>(ubR_, ubDb_, ubDb_, decay0, decay0, static_cast<uint16_t>(kDim / 2));
                }
                // D1 is consumed once the VF is queued. Stage4 overlaps the saxpy.
                if (hasNext) {
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(kFlagGo);
                }
                if (c0 + 2 < chunkNum) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
                }
                ApplyHalf(db, c0, !has1, own0);
                if (has1) {
                    ApplyHalf(db, c1, !hasN0, !own0);
                }
                if (fut < chunkNum) {
                    PrefetchV(db, fut);
                }
            }
            if (headLinear + aicNum_ < totalHeads) {
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagDone);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kFlagHead);
                AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
                PipeBarrier<PIPE_ALL>();
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
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
        for (uint32_t headLinear = headBegin_; headLinear < totalHeads; headLinear += aicNum_) {
            // Prologue prepares both halves of the first 64-token step. Later steps
            // are issued from the tail: Stage2 under Stage6, Stage4 after the
            // inverse of that half is on L1.
            if (chunkNum > 0) {
                Stage2(TaskOf(0));
                Stage4(TaskOf(0));
            }
            if (chunkNum > 1) {
                Stage2(TaskOf(1));
                Stage4(TaskOf(1));
            }
            for (uint32_t c0 = 0; c0 < chunkNum; c0 += 2) {
                const uint32_t c1 = c0 + 1;
                const bool has1 = c1 < chunkNum;
                const uint32_t n0 = c0 + 2;
                const uint32_t n1 = c0 + 3;
                const bool hasN0 = n0 < chunkNum;
                const bool hasN1 = n1 < chunkNum;
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagH, 0));
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagH, 1));
                // Both halves read this step's entry H. H is not uploaded again
                // until both deltas have been applied.
                StagePCompute(TaskOf(c0));
                if (has1) {
                    StagePCompute(TaskOf(c1));
                }
                StagePStore(TaskOf(c0));
                if (has1) {
                    StagePStore(TaskOf(c1));
                }
                // Stage2 before Stage6: kkt is ready while Stage6 runs, so the
                // vector inverse stays under the long cube chain.
                if (hasN0) {
                    Stage2(TaskOf(n0));
                }
                if (hasN1) {
                    Stage2(TaskOf(n1));
                }
                Stage6(TaskOf(c0), ubDb_);
                if (has1) {
                    Stage6(TaskOf(c1), ubD1b_);
                }
                if (hasN0 || hasN1) {
                    // u's fixpipe reuses the UB that still holds D1 until Apply sets Go.
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagGo, 0));
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(AicSeen(kFlagGo, 1));
                }
                if (hasN0) {
                    Stage4(TaskOf(n0));
                }
                if (hasN1) {
                    Stage4(TaskOf(n1));
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
    LocalTensor<float> ubD1_, ubGd_, ubGpd_, ubEgd_, ubEye_;
    LocalTensor<uint32_t> ubSh_;
    LocalTensor<bfloat16_t> l1K_, l1Q_, l1Vb_, l1Kbg_, l1Qexp_, l1A_, l1H_, l1W_, l1Vnew_, l1VnewG_, l1Attn_, l1Kt_;
    LocalTensor<bfloat16_t> l0A_, l0B_;
    LocalTensor<float> l0C_;

    LocalTensor<bfloat16_t> ubQDb_[2], ubKDb_[2], ubVDb_[2], ubKbgDb_[2], ubQexpDb_[2], ubBetaDb_[2];
    LocalTensor<bfloat16_t> ubAttnDb_[2], ubAbfDb_[2];
    LocalTensor<float> ubGDb_[2], ubBeDb_[2], ubGpDb_[2], ubEgDb_[2], ubKktDb_[2], ubQkDb_[2];
    LocalTensor<float> ubLDb_[2], ubUDb_[2], ubPDb_[2];
    LocalTensor<bfloat16_t> ubUb_[2], ubPb_[2], ubDb_, ubD1b_;
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
    uint8_t qkReady_[2] = {0, 0};
    uint8_t vReady_[2] = {0, 0};
    uint8_t l2Ready_[2] = {0, 0};
    uint8_t qkOnL1_[2] = {0, 0};
    LocalTensor<float> futBe_, futGp_, futEg_;
    OnChipBuffer buf;
};

}  // namespace GdnFwd32

#endif
