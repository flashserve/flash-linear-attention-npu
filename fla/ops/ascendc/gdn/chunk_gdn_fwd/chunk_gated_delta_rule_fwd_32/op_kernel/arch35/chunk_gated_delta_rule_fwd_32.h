/**
 * GDN forward scheduled like kda_32_var.
 *
 * One step is 32 tokens. AIV0 owns even chunks, AIV1 owns odd chunks: that AIV
 * does L2, sigmoid, cumsum and the 32x32 inverse. Both AIVs take part in every
 * chunk. v_new and o are split 16/16 (subblock 0 gets rows 0..15). State is
 * split by K: each AIV holds 64 of the 128 rows and adds its D slices as the
 * fixpipe lands. The next chunk's P and o_inter use the updated state.
 *
 * Cube order inside a chunk matches KDA's consumers: kkt/qk, then A@vb and
 * w@H, then v_new, then D, then o. D's fixpipe releases the saxpy; o is stored
 * after the next H is queued. Chunk c+2's K L2 runs under
 * o and D; its L1 copy and q L2 run after D, and H is uploaded before q's L2.
 * H stays in L0B across q@H. q_exp is loaded before the v_new wait, so
 * q@H does not move H a second time. D then uses the other L0 bank, so its
 * load overlaps q@H. The second K tile overlaps the first D mad, and that
 * mad's fixpipe overlaps the second D mad. A@kbg uses the other L0 bank.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_H

#include "chunk_gated_delta_rule_fwd_32_matmul.h"
#include "chunk_gated_delta_rule_fwd_32_tiling_data.h"
#include "chunk_gated_delta_rule_fwd_32_vf.h"

namespace GdnFwd32 {

using namespace AscendC;

constexpr uint16_t kAiv1 = 16;
constexpr uint16_t kFlagQk = 0;
constexpr uint16_t kFlagQ = 9;
constexpr uint16_t kFlagKkt = 1;
constexpr uint16_t kFlagInv = 2;
constexpr uint16_t kFlagH = 3;
// Columns 0-63 of H. Both AIVs post this once their first four panels are in
// L1, so w@H's first N-half overlaps the second saxpy.
constexpr uint16_t kFlagH0 = 11;
constexpr uint16_t kFlagP = 4;
constexpr uint16_t kFlagV = 5;
constexpr uint16_t kFlagO = 6;
constexpr uint16_t kFlagD = 7;
constexpr uint16_t kFlagOd = 8;
constexpr uint16_t kFlagAttn = 10;

constexpr uint32_t kUbState = 0;
constexpr uint32_t kUbQ = 16 * 1024;
constexpr uint32_t kUbK = 24 * 1024;
constexpr uint32_t kUbV = 32 * 1024;
constexpr uint32_t kUbQnz = 40 * 1024;
constexpr uint32_t kUbKnz = 48 * 1024;
constexpr uint32_t kUbKbg = 56 * 1024;
constexpr uint32_t kUbQexp = 64 * 1024;
constexpr uint32_t kUbVnz = 72 * 1024;
constexpr uint32_t kUbG = 80 * 1024;
constexpr uint32_t kUbBeta = 81 * 1024;
constexpr uint32_t kUbBe = 82 * 1024;
constexpr uint32_t kUbEg = 84 * 1024;
constexpr uint32_t kUbKkt = 88 * 1024;
constexpr uint32_t kUbQk = 92 * 1024;
constexpr uint32_t kUbL = 96 * 1024;
constexpr uint32_t kUbA = 100 * 1024;
constexpr uint32_t kUbAbf = 104 * 1024;
constexpr uint32_t kUbAttn = 106 * 1024;
constexpr uint32_t kUbU = 108 * 1024;
constexpr uint32_t kUbVn = 124 * 1024;
constexpr uint32_t kUbVg = 128 * 1024;
constexpr uint32_t kUbO = 132 * 1024;
// Two 32-row D slices per AIV, so the cube can land both before the saxpy.
constexpr uint32_t kUbD = 176 * 1024;
constexpr uint32_t kUbEye = 144 * 1024;
constexpr uint32_t kUbSh = 148 * 1024;
constexpr uint32_t kUbScratch = 152 * 1024;
constexpr uint32_t kUbG2 = 160 * 1024;
constexpr uint32_t kUbEg2 = 162 * 1024;

constexpr uint32_t kL1H = 0;
constexpr uint32_t kL1Slot = 40 * 1024;
constexpr uint32_t kL1Stride = 56 * 1024;
constexpr uint32_t kOffQ = 0;
constexpr uint32_t kOffK = 8 * 1024;
constexpr uint32_t kOffVb = 16 * 1024;
constexpr uint32_t kOffKbg = 24 * 1024;
constexpr uint32_t kOffQexp = 32 * 1024;
constexpr uint32_t kOffA = 40 * 1024;
constexpr uint32_t kOffAttn = 42 * 1024;
constexpr uint32_t kOffW = 44 * 1024;
constexpr uint32_t kL1Vn = 160 * 1024;
constexpr uint32_t kL1Vg = 168 * 1024;

class Kernel {
public:
    __aicore__ inline Kernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR initial,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR finalState, GM_ADDR workspace,
                                const ChunkGatedDeltaRuleFwd32TilingData &tilingData)
    {
        sub_ = GetSubBlockIdx();
        batchSize_ = tilingData.B;
        valueHeadNum_ = tilingData.HV;
        keyHeadNum_ = tilingData.HK == 0 ? tilingData.HV : tilingData.HK;
        seqLen_ = tilingData.T;
        scale_ = tilingData.scale;
        hasInitial_ = tilingData.hasInitial;
        seqNum_ = tilingData.seqNum;
        uint32_t aicIdx = GetBlockIdx();
        if ASCEND_IS_AIV {
            const uint32_t subNum = GetSubBlockNum();
            if (subNum > 0) {
                aicIdx /= subNum;
            }
        }
        aicNum_ = tilingData.coreNum == 0 ? 1 : tilingData.coreNum;
        totalHeads_ = batchSize_ * valueHeadNum_;
        group_ = keyHeadNum_ == 0 ? 1 : valueHeadNum_ / keyHeadNum_;
        // tailRows_ == 0 means every chunk is a full 32. Otherwise the last chunk
        // has that many valid tokens and the kernel zero-pads the rest.
        tailRows_ = seqLen_ % static_cast<uint32_t>(kBt);
        nChunk_ = tailRows_ == 0 ? seqLen_ / static_cast<uint32_t>(kBt)
                                 : seqLen_ / static_cast<uint32_t>(kBt) + 1;
        hasHead_ = valueHeadNum_ != 0 && aicIdx < totalHeads_;
        headBegin_ = hasHead_ ? aicIdx : 0;
        (void)workspace;

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(beta));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(o));
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(finalState));
        if (hasInitial_ != 0 && initial != nullptr) {
            initGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initial));
        }
        (void)chunkIndices;
        if (seqNum_ != 0 && cuSeqlens != nullptr) {
            cuGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        }

        ubState_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbState);
        ubQ_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbQ);
        ubK_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbK);
        ubV_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbV);
        ubQnz_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbQnz);
        ubKnz_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbKnz);
        ubKbg_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbKbg);
        ubQexp_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbQexp);
        ubVnz_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbVnz);
        ubG_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbG);
        ubBeta_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbBeta);
        ubBe_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbBe);
        ubEg_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbEg);
        ubKkt_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbKkt);
        ubQk_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbQk);
        ubL_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbL);
        ubA_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbA);
        ubAbf_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbAbf);
        ubAttn_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbAttn);
        ubU_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbU);
        ubVn_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbVn);
        ubVg_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbVg);
        ubO_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbO);
        ubD_ = buf.template GetBuffer<BufferType::ASCEND_UB, bfloat16_t>(kUbD);
        ubEye_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbEye);
        ubSh_ = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(kUbSh);
        ubScratch_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbScratch);
        l1H_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1H);
        l1Vn_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1Vn);
        l1Vg_ = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(kL1Vg);
        for (uint32_t s = 0; s < 2; ++s) {
            const uint32_t base = kL1Slot + s * kL1Stride;
            l1Q_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffQ);
            l1K_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffK);
            l1Vb_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffVb);
            l1Kbg_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffKbg);
            l1Qexp_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffQexp);
            l1A_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffA);
            l1Attn_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffAttn);
            l1W_[s] = buf.template GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(base + kOffW);
        }
        // 32KB banks. H is 128*128 bf16 and fills bank 0; bank 1 never holds H.
        l0A_[0] = buf.template GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(0);
        l0A_[1] = buf.template GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(32 * 1024);
        l0B_[0] = buf.template GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(0);
        l0B_[1] = buf.template GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(32 * 1024);
        l0C0_ = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0C1_ = l0C0_[static_cast<uint32_t>(kBt * kDim)];
        ubG2_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbG2);
        ubEg2_ = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbEg2);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            if (seqNum_ == 0) {
                ProcessAiv();
            } else {
                ProcessAivVarlen();
            }
        }
        if ASCEND_IS_AIC {
            if (seqNum_ == 0) {
                ProcessAic();
            } else {
                ProcessAicVarlen();
            }
        }
    }

private:
    // One flat (batch, value-head) task. q/k use the shared key head.
    __aicore__ inline void GetCurIdx(uint32_t head)
    {
        batchIdx_ = valueHeadNum_ == 0 ? 0 : head / valueHeadNum_;
        headIdx_ = valueHeadNum_ == 0 ? 0 : head - batchIdx_ * valueHeadNum_;
        const uint32_t keyHead = group_ <= 1 ? headIdx_ : headIdx_ / group_;
        vHeadBase_ = static_cast<int64_t>(
            (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * seqLen_);
        qHeadBase_ = static_cast<int64_t>(
            (static_cast<uint64_t>(batchIdx_) * keyHeadNum_ + keyHead) * seqLen_);
        vTokenBase_ = vHeadBase_;
        qTokenBase_ = qHeadBase_;
    }

    // 0 means this chunk is a full 32 rows.
    __aicore__ inline uint32_t ChunkRows(uint32_t chunk) const
    {
        if (tailRows_ == 0 || chunk + 1U < nChunk_) {
            return static_cast<uint32_t>(kBt);
        }
        return tailRows_;
    }

    template <typename T>
    __aicore__ inline void CopyGmToUb(LocalTensor<T> dst, GlobalTensor<T> src, int64_t offset, uint32_t count)
    {
        const uint32_t bytes = count * static_cast<uint32_t>(sizeof(T));
        if (bytes >= 32U && (bytes & 31U) == 0U) {
            DataCopy(dst, src[offset], count);
            return;
        }
        // An unpadded tail is still written out to 32 bytes, and those extra lanes
        // are not the zeros Duplicate left behind. Pad them with 0 so a short g
        // keeps eg[31] equal to the last valid token.
        const uint32_t padBytes = (32U - (bytes & 31U)) & 31U;
        const uint32_t padElems = padBytes / static_cast<uint32_t>(sizeof(T));
        DataCopyParams params{1, static_cast<uint16_t>(bytes), 0, 0};
        DataCopyPadParams pad{true, 0, static_cast<uint8_t>(padElems), 0};
        DataCopyPad(dst, src[offset], params, pad);
    }

    // Same bytes as CopyGmToUb, and also accepts a GM address that is not 32-byte aligned.
    // A packed sequence can start at any token. q/k/v rows are 256 bytes, so they stay aligned;
    // g is 4 bytes and beta is 2, so a full chunk of those two may not be.
    template <typename T>
    __aicore__ inline void CopyGmToUbRelaxed(LocalTensor<T> dst, GlobalTensor<T> src, int64_t offset, uint32_t count)
    {
        const uint32_t bytes = count * static_cast<uint32_t>(sizeof(T));
        const bool addrAligned = ((static_cast<uint64_t>(offset) * sizeof(T)) & 31U) == 0U;
        if (addrAligned && bytes >= 32U && (bytes & 31U) == 0U) {
            DataCopy(dst, src[offset], count);
            return;
        }
        const uint32_t padBytes = (32U - (bytes & 31U)) & 31U;
        const uint32_t padElems = padBytes / static_cast<uint32_t>(sizeof(T));
        DataCopyParams params{1, static_cast<uint16_t>(bytes), 0, 0};
        DataCopyPadParams pad{padElems != 0U, 0, static_cast<uint8_t>(padElems), 0};
        DataCopyPad(dst, src[offset], params, pad);
    }

    // Even chunks belong to AIV0, odd chunks to AIV1.
    __aicore__ inline uint32_t AivOf(uint32_t chunk) const
    {
        return chunk & 1U;
    }

    __aicore__ inline void InitState()
    {
        constexpr uint32_t nElem = 64 * 128;
        if (hasInitial_ != 0) {
            const int64_t elem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim +
                static_cast<uint64_t>(sub_) * nElem);
            DataCopy(ubD_, initGm_[elem], nElem);
            // Event 1. PrepareLoad owns event 0, and both copies are in flight.
            SetFlag<HardEvent::MTE2_V>(1);
            WaitFlag<HardEvent::MTE2_V>(1);
            NdToPanels(ubState_, ubD_);
        } else {
            Duplicate(ubState_, static_cast<bfloat16_t>(0), nElem);
        }
        SetFlag<HardEvent::V_MTE3>(1);
    }

    __aicore__ inline void LoadGate(uint32_t chunk)
    {
        const int64_t token = vTokenBase_ + static_cast<int64_t>(chunk) * kBt;
        const uint32_t rows = ChunkRows(chunk);
        if (rows == static_cast<uint32_t>(kBt)) {
            DataCopy(ubG2_, gGm_[token], kBt);
        } else {
            Duplicate(ubG2_, 0.f, static_cast<uint32_t>(kBt));
            SetFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::V_MTE2>(1);
            CopyGmToUb(ubG2_, gGm_, token, rows);
        }
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);
        CumsumExp2VF<true>(ubG2_, ubEg2_, ubSh_);
    }

    __aicore__ inline float DecayOf(LocalTensor<float> eg)
    {
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        return eg.GetValue(kBt - 1);
    }

    // GM loads only. Issued before the inverse VF so the L2 wait is not an MTE2 stall.
    __aicore__ inline void PrepareLoad(uint32_t chunk)
    {
        const int64_t qToken = qTokenBase_ + static_cast<int64_t>(chunk) * kBt;
        const int64_t vToken = vTokenBase_ + static_cast<int64_t>(chunk) * kBt;
        const uint32_t rows = ChunkRows(chunk);
        if (rows == static_cast<uint32_t>(kBt)) {
            const int32_t nRow = kBt * kDim;
            DataCopy(ubQ_, qGm_[qToken * kDim], nRow);
            DataCopy(ubK_, kGm_[qToken * kDim], nRow);
            DataCopy(ubV_, vGm_[vToken * kDim], nRow);
            DataCopy(ubG_, gGm_[vToken], kBt);
            DataCopy(ubBeta_, betaGm_[vToken], kBt);
        } else {
            Duplicate(ubQ_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt * kDim));
            Duplicate(ubK_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt * kDim));
            Duplicate(ubV_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt * kDim));
            Duplicate(ubG_, 0.f, static_cast<uint32_t>(kBt));
            Duplicate(ubBeta_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt));
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            const uint32_t nRow = rows * static_cast<uint32_t>(kDim);
            CopyGmToUb(ubQ_, qGm_, qToken * kDim, nRow);
            CopyGmToUb(ubK_, kGm_, qToken * kDim, nRow);
            CopyGmToUb(ubV_, vGm_, vToken * kDim, nRow);
            CopyGmToUb(ubG_, gGm_, vToken, rows);
            CopyGmToUb(ubBeta_, betaGm_, vToken, rows);
        }
        SetFlag<HardEvent::MTE2_V>(0);
    }

    // Varlen copies. A sequence start only has to be aligned for the 256-byte q/k/v row.
    __aicore__ inline void LoadGateVar(uint32_t chunk)
    {
        const int64_t token = vTokenBase_ + static_cast<int64_t>(chunk) * kBt;
        const uint32_t rows = ChunkRows(chunk);
        if (rows != static_cast<uint32_t>(kBt)) {
            Duplicate(ubG2_, 0.f, static_cast<uint32_t>(kBt));
            SetFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::V_MTE2>(1);
        }
        CopyGmToUbRelaxed(ubG2_, gGm_, token, rows);
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);
        CumsumExp2VF<true>(ubG2_, ubEg2_, ubSh_);
    }

    __aicore__ inline void PrepareLoadVar(uint32_t chunk)
    {
        const int64_t qToken = qTokenBase_ + static_cast<int64_t>(chunk) * kBt;
        const int64_t vToken = vTokenBase_ + static_cast<int64_t>(chunk) * kBt;
        const uint32_t rows = ChunkRows(chunk);
        if (rows != static_cast<uint32_t>(kBt)) {
            Duplicate(ubQ_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt * kDim));
            Duplicate(ubK_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt * kDim));
            Duplicate(ubV_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt * kDim));
            Duplicate(ubG_, 0.f, static_cast<uint32_t>(kBt));
            Duplicate(ubBeta_, static_cast<bfloat16_t>(0), static_cast<uint32_t>(kBt));
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        const uint32_t nRow = rows * static_cast<uint32_t>(kDim);
        CopyGmToUbRelaxed(ubQ_, qGm_, qToken * kDim, nRow);
        CopyGmToUbRelaxed(ubK_, kGm_, qToken * kDim, nRow);
        CopyGmToUbRelaxed(ubV_, vGm_, vToken * kDim, nRow);
        CopyGmToUbRelaxed(ubG_, gGm_, vToken, rows);
        CopyGmToUbRelaxed(ubBeta_, betaGm_, vToken, rows);
        SetFlag<HardEvent::MTE2_V>(0);
    }

    // K's L2 only. The L1 slot is still the live chunk's K until D has read it,
    // so the copy and kFlagQk are PublishKFlag, after ApplyD.
    __aicore__ inline void PublishK()
    {
        WaitFlag<HardEvent::MTE2_V>(0);
        // The previous publish may still be reading the NZ buffers this VF overwrites.
        if (nzArmed_ != 0) {
            WaitFlag<HardEvent::MTE3_V>(3);
            nzArmed_ = 0;
        }
        BetaSigmoidVF(ubBeta_, ubBe_);
        CumsumExp2VF<true>(ubG_, ubEg_, ubSh_);
        L2NormKAndKbgVF(ubK_, ubKnz_, ubKbg_, ubBe_, ubEg_);
    }

    __aicore__ inline void PublishKFlag(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1U;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1K_[slot], ubKnz_, kBt, kDim);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagQk);
    }

    // Q L2 and the v scale. UB only: q_exp is still the live chunk's until q@H,
    // so the L1 copies are PublishQFlag, after ApplyD.
    __aicore__ inline void PublishQ()
    {
        L2NormQAndQexpVF(ubQ_, ubQnz_, ubQexp_, ubEg_);
        ScaleRowsVF(ubV_, ubBe_, reinterpret_cast<__ubuf__ bfloat16_t *>(ubVnz_.GetPhyAddr()));
    }

    __aicore__ inline void PublishQFlag(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1U;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNzToL1(l1Q_[slot], ubQnz_, kBt, kDim);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagQ);
        CopyNzToL1(l1Vb_[slot], ubVnz_, kBt, kDim);
        CopyNzToL1(l1Kbg_[slot], ubKbg_, kBt, kDim);
        CopyNzToL1(l1Qexp_[slot], ubQexp_, kBt, kDim);
        SetFlag<HardEvent::MTE3_V>(3);
        nzArmed_ = 1;
    }

    // Prologue: both halves before the cube starts. The K slot is empty here.
    __aicore__ inline void PublishStaged(uint32_t chunk)
    {
        PublishK();
        PublishKFlag(chunk);
        PublishQ();
        PublishQFlag(chunk);
    }

    // L and the inverse stay in UB, then A goes to L1. The caller waits
    // kFlagAttn before AttnL1. A is 2KB; the cube starts A@v on kFlagInv.
    __aicore__ inline void InverseL1(uint32_t chunk)
    {
        GateAttnVF<1>(ubEg_, ubBe_, ubKkt_, ubQk_, ubL_, ubAttn_);
        InvertAVF(ubL_, ubA_, ubAbf_, ubEye_);
        const uint32_t slot = chunk & 1U;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNdToL1(l1A_[slot], ubAbf_, kBt, kBt);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagInv);
    }

    __aicore__ inline void AttnL1(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1U;
        GateAttnVF<2>(ubEg_, ubBe_, ubKkt_, ubQk_, ubL_, ubAttn_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyNdToL1(l1Attn_[slot], ubAttn_, kBt, kBt);
    }

    __aicore__ inline void CopyStatePanels(uint32_t fc0, uint32_t nFc)
    {
        constexpr uint32_t kRows = 64;
        constexpr uint32_t kFull = 128;
        constexpr uint16_t kBurst = static_cast<uint16_t>(kRows * 16 * sizeof(bfloat16_t) / 32);
        for (uint32_t fc = fc0; fc < fc0 + nFc; ++fc) {
            DataCopy(l1H_[fc * kFull * 16 + sub_ * kRows * 16], ubState_[fc * kRows * 16],
                     DataCopyParams(1, kBurst, 0, 0));
        }
    }

    // Initial state is complete before either flag, so chunk 0's two N-halves
    // both see the full H.
    __aicore__ inline void UploadH()
    {
        WaitFlag<HardEvent::V_MTE3>(1);
        CopyStatePanels(0, kDim / 16);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagH0);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagH);
    }

    // 16 tight-NZ rows into a height-32 L1 panel. The other AIV's half is the dst gap.
    __aicore__ inline void CopyHalfNz(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> nz, uint32_t row0)
    {
        constexpr uint16_t nFrac = static_cast<uint16_t>(kDim / 16);
        constexpr uint16_t burst = static_cast<uint16_t>(16 * 16 * sizeof(bfloat16_t) / 32);
        constexpr uint16_t gap = static_cast<uint16_t>((kBt - 16) * 16 * sizeof(bfloat16_t) / 32);
        DataCopy(l1[row0 * 16], nz, DataCopyParams(nFrac, burst, 0, gap));
    }

    __aicore__ inline void EmitV(LocalTensor<float> eg)
    {
        const uint32_t row0 = sub_ * 16;
        VnewHalfVF(ubU_, eg, ubVn_, ubVg_, row0, 0, 16);
        SetFlag<HardEvent::V_MTE3>(4);
        WaitFlag<HardEvent::V_MTE3>(4);
        CopyHalfNz(l1Vn_, ubVn_, row0);
        CopyHalfNz(l1Vg_, ubVg_, row0);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagV);
    }

    __aicore__ inline void StoreO(uint32_t chunk)
    {
        // The caller has waited kFlagO. The fixpipe already multiplied by scale_.
        // A partial chunk writes only the valid rows of this AIV's 16-row half.
        const uint32_t rows = ChunkRows(chunk);
        const uint32_t row0 = sub_ * 16U;
        if (row0 < rows) {
            const uint32_t n = rows - row0 < 16U ? rows - row0 : 16U;
            const int64_t token = vTokenBase_ + static_cast<int64_t>(chunk) * kBt + static_cast<int64_t>(row0);
            DataCopy(oGm_[token * kDim], ubO_, n * static_cast<uint32_t>(kDim));
        }
        CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(kFlagOd + sub_ * kAiv1));
    }

    // upload queues the first four panels before the second saxpy is issued, so
    // MTE3 copies them while that saxpy runs. The last chunk leaves both flags
    // for the epilogue.
    __aicore__ inline void ApplyD(float decay, bool upload)
    {
        SaxpyStateVF(ubState_, ubD_, decay, 0, 8);
        SetFlag<HardEvent::V_MTE3>(1);
        if (upload) {
            WaitFlag<HardEvent::V_MTE3>(1);
            CopyStatePanels(0, 4);
            // Fires when those panels land, while the second saxpy is in flight.
            CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagH0);
        }
        SaxpyStateVF(ubState_, ubD_, decay, 8, 8);
        SetFlag<HardEvent::V_MTE3>(2);
        if (upload) {
            WaitFlag<HardEvent::V_MTE3>(2);
            CopyStatePanels(4, 4);
            CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagH);
        }
    }



    __aicore__ inline LocalTensor<float> L0COf(uint8_t bank)
    {
        return bank == 0 ? l0C0_ : l0C1_;
    }

    // K@K only. kFlagKkt is the inverse's start flag.
    __aicore__ inline void Kkt(uint32_t chunk, uint8_t bank)
    {
        const uint32_t chunkAiv = AivOf(chunk);
        const uint32_t slot = chunkAiv;
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagQk + chunkAiv * kAiv1));
        WaitFlag<HardEvent::M_MTE1>(bank);
        LoadL1ToL0AB(l1K_[slot], l1K_[slot], l0A_[bank], l0B_[bank], kBt, kBt, kDim, false, false);
        SetFlag<HardEvent::MTE1_M>(bank);
        WaitFlag<HardEvent::MTE1_M>(bank);
        WaitFlag<HardEvent::FIX_M>(bank);
        MmadL0ToL0C(l0A_[bank], l0B_[bank], L0COf(bank), kBt, kBt, kDim, true);
        SetFlag<HardEvent::M_MTE1>(bank);
        SetFlag<HardEvent::M_FIX>(bank);
        WaitFlag<HardEvent::M_FIX>(bank);
        FixpipeFp32Ub(ubKkt_, L0COf(bank), kBt, kBt, static_cast<uint8_t>(chunkAiv));
        SetFlag<HardEvent::FIX_M>(bank);
        CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagKkt + chunkAiv * kAiv1));
    }

    __aicore__ inline void Qk(uint32_t chunk, uint8_t bank)
    {
        const uint32_t chunkAiv = AivOf(chunk);
        const uint32_t slot = chunkAiv;
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagQ + chunkAiv * kAiv1));
        WaitFlag<HardEvent::M_MTE1>(bank);
        LoadL1ToL0AB(l1Q_[slot], l1K_[slot], l0A_[bank], l0B_[bank], kBt, kBt, kDim, false, false);
        SetFlag<HardEvent::MTE1_M>(bank);
        WaitFlag<HardEvent::MTE1_M>(bank);
        WaitFlag<HardEvent::FIX_M>(bank);
        MmadL0ToL0C(l0A_[bank], l0B_[bank], L0COf(bank), kBt, kBt, kDim, true);
        SetFlag<HardEvent::M_MTE1>(bank);
        SetFlag<HardEvent::M_FIX>(bank);
        WaitFlag<HardEvent::M_FIX>(bank);
        FixpipeFp32Ub(ubQk_, L0COf(bank), kBt, kBt, static_cast<uint8_t>(chunkAiv));
        SetFlag<HardEvent::FIX_M>(bank);
        CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagAttn + chunkAiv * kAiv1));
    }

    // bothBanks: K@K on bank 0 and Q@K on bank 1, so the Q load overlaps the K mad.
    // Otherwise both gemms stay on bank 1 and bank 0 (the live H) is left alone.
    __aicore__ inline void KktQk(uint32_t chunk, bool bothBanks)
    {
        const uint8_t b0 = bothBanks ? static_cast<uint8_t>(0) : static_cast<uint8_t>(1);
        const uint8_t b1 = bothBanks ? static_cast<uint8_t>(1) : b0;
        Kkt(chunk, b0);
        Qk(chunk, b1);
    }

    // A@v stays in l0C0 as u. A@kbg is W: fixpipe it to L1 and load L0A before the H wait.
    __aicore__ inline void MatmulAvAndKbgLoadW(uint32_t slot)
    {
        // A@v on bank 0. u stays in l0C0, so this mad does not signal FIX.
        // Bank 1 is taken before the load and returned before the mad, so
        // A@kbg's load overlaps A@v. kbg is stored negated, so w@H accumulates -p.
        WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::M_MTE1>(0);
        LoadL1ToL0AB(l1A_[slot], l1Vb_[slot], l0A_[0], l0B_[0], kBt, kDim, kBt, false, true);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        MmadL0ToL0C(l0A_[0], l0B_[0], l0C0_, kBt, kDim, kBt, true);
        SetFlag<HardEvent::M_MTE1>(0);

        WaitFlag<HardEvent::M_MTE1>(1);
        LoadL1ToL0AB(l1A_[slot], l1Kbg_[slot], l0A_[1], l0B_[1], kBt, kDim, kBt, false, true);
        SetFlag<HardEvent::MTE1_M>(1);
        WaitFlag<HardEvent::MTE1_M>(1);
        WaitFlag<HardEvent::FIX_M>(1);
        MmadL0ToL0C(l0A_[1], l0B_[1], l0C1_, kBt, kDim, kBt, true);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::M_FIX>(1);
        // W's fixpipe and L0 reload do not need H. Queue them before the
        // scalar stall so they run during saxpy. H is the only load left
        // after the flag, and it lands in L0B while W already sits in L0A.
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_FIX>(1);
        FixpipeBf16L1(l1W_[slot], l0C1_, kBt, kDim);
        SetFlag<HardEvent::FIX_M>(1);
        SetFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::FIX_MTE1>(0);
        LoadL1ToL0A(l1W_[slot], l0A_[0], kBt, kDim, false);
        SetFlag<HardEvent::MTE1_M>(0);
    }

    // Columns 0-63. Panels 0-3 are copied before the second saxpy, so this half
    // runs while columns 64-127 are still being updated. W stays the full K.
    __aicore__ inline void MatmulWHCols0To63()
    {
        LoadL1NzHCols(l1H_, l0B_[0], 0);
        SetFlag<HardEvent::MTE1_M>(1);
        WaitFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(1);
        MmadL0ToL0C(l0A_[0], l0B_[0], l0C0_, kBt, 64, kDim, false);
    }

    // Columns 64-127, then the fixpipe of u is v_new. L0C is n-major, so those
    // columns of a [32, 128] tile start at 2048 floats.
    __aicore__ inline void MatmulWHCols64To127WriteVnew()
    {
        LoadL1NzHCols(l1H_, l0B_[1], 4);
        SetFlag<HardEvent::MTE1_M>(1);
        WaitFlag<HardEvent::MTE1_M>(1);
        MmadL0ToL0C(l0A_[0], l0B_[1], l0C0_[2048], kBt, 64, kDim, false);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeHalfBf16(ubU_, l0C0_, kBt, kDim);
        SetFlag<HardEvent::FIX_M>(0);
        CrossCoreSetFlag<0x4, PIPE_FIX>(kFlagP);
        CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagP + kAiv1));
    }

    // Full H and q_exp. The MTE1_M(0) token is set in the caller, around kFlagV,
    // so that Set and Wait stay in one function.
    __aicore__ inline void ReloadHAndQexp(uint32_t slot)
    {
        WaitFlag<HardEvent::M_MTE1>(0);
        LoadL1ToL0B(l1H_, l0B_[0], kDim, kDim, true);
        LoadL1ToL0A(l1Qexp_[slot], l0A_[0], kBt, kDim, false);
    }

    // q@H keeps bank 0 and the low 16KB of l0C0 for o. D's K and v_g load into
    // bank 1 under that mad. The second K tile reuses v_g and loads into bank 0
    // once q@H has released it. L0C for D is at 64KB.
    __aicore__ inline void MatmulQexpHAndD(uint32_t slot)
    {
        WaitFlag<HardEvent::FIX_M>(0);
        MmadL0ToL0C(l0A_[0], l0B_[0], l0C0_, kBt, kDim, kDim, true);
        SetFlag<HardEvent::M_MTE1>(0);

        constexpr int kDm = 64;
        constexpr uint32_t kL0cD = 16 * 1024;
        WaitFlag<HardEvent::M_MTE1>(1);
        LoadL1ToL0AB(l1K_[slot], l1Vg_, l0A_[1], l0B_[1], kDm, kDim, kBt, true, true);
        SetFlag<HardEvent::MTE1_M>(1);
        WaitFlag<HardEvent::M_MTE1>(0);
        auto kPart = l1K_[slot][static_cast<uint32_t>(2 * 2 * kBt * 16)];
        LoadL1ToL0A(kPart, l0A_[0], kDm, kBt, true);
        SetFlag<HardEvent::MTE1_M>(0);
        LocalTensor<float> l0cD = l0C0_[kL0cD];
        WaitFlag<HardEvent::MTE1_M>(1);
        MmadL0ToL0C(l0A_[1], l0B_[1], l0cD, kDm, kDim, kBt, true);
        // fixpipe0 starts when this mad ends, so it overlaps the second mad.
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        MmadL0ToL0C(l0A_[0], l0B_[1], l0cD[static_cast<uint32_t>(kDm * kDim)], kDm, kDim, kBt, true);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);

        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeBf16UbNz(ubD_, l0cD, kDm, kDim, 0);
        SetFlag<HardEvent::FIX_M>(0);
        CrossCoreSetFlag<0x4, PIPE_FIX>(kFlagD);

        WaitFlag<HardEvent::FIX_M>(1);
        SetFlag<HardEvent::M_FIX>(1);
        WaitFlag<HardEvent::M_FIX>(1);
        FixpipeBf16UbNz(ubD_, l0cD[static_cast<uint32_t>(kDm * kDim)], kDm, kDim, 1);
        SetFlag<HardEvent::FIX_M>(1);
        CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagD + kAiv1));
    }

    // attn@v_new accumulates into q@H's L0C. The o fixpipe waits until the
    // previous chunk has finished reading ubO_.
    __aicore__ inline void MatmulAttnVnew(uint32_t slot)
    {
        WaitFlag<HardEvent::M_MTE1>(1);
        LoadL1ToL0AB(l1Attn_[slot], l1Vn_, l0A_[1], l0B_[1], kBt, kDim, kBt, false, true);
        SetFlag<HardEvent::MTE1_M>(1);
        WaitFlag<HardEvent::MTE1_M>(1);
        MmadL0ToL0C(l0A_[1], l0B_[1], l0C0_, kBt, kDim, kBt, false);
        SetFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
    }

    __aicore__ inline void FixpipeScaledO()
    {
        FixpipeHalfBf16(ubO_, l0C0_, kBt, kDim, scale_);
        SetFlag<HardEvent::FIX_M>(0);
        CrossCoreSetFlag<0x4, PIPE_FIX>(kFlagO);
        CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagO + kAiv1));
    }

    __aicore__ inline void ProcessAiv()
    {
        if (!hasHead_) {
            return;
        }
        // sub_ is 0 or 1, so both AIVs have a prologue chunk once there are two
        // or more. A single chunk is only AIV0's; AIV1 still runs that chunk below.
        const bool moreThanOneChunk = nChunk_ > 1;
        const bool loadPrologue = sub_ == 0 || moreThanOneChunk;
        // One issue for the whole core. The first PrepareLoad then runs on MTE2
        // while this VF is still on the vector pipe. InitState uses MTE2_V(1).
        GenAuxMatrixVF(ubSh_, ubEye_);
        // prefetched: the previous task already issued this task's PrepareLoad
        // so it overlaps that task's final-state store.
        bool prefetched = false;
        // The state store reads ubD_ on MTE3. The next task's initial-state copy
        // writes that same buffer on MTE2, so the copy waits until the store is done.
        bool stateLive = false;
        for (uint32_t head = headBegin_; head < totalHeads_; head += aicNum_) {
            if (!prefetched) {
                GetCurIdx(head);
                if (loadPrologue) {
                    PrepareLoad(sub_);
                }
            }
            prefetched = false;
            if (stateLive) {
                WaitFlag<HardEvent::MTE3_MTE2>(0);
            }
            InitState();
            if (loadPrologue) {
                PublishStaged(sub_);
            }
            // Chunk 0 and chunk 1 already have K@K from the prologue, so the AIV
            // each chunk belongs to inverts once here. Chunk 0 finishes before
            // the initial H upload. Chunk 1 stays after that upload and overlaps
            // chunk 0's A@v. Later chunks are inverted one iteration early.
            if (AivOf(0) == sub_) {
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
                InverseL1(0);
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagAttn);
                AttnL1(0);
            }
            UploadH();
            if (moreThanOneChunk && AivOf(1) == sub_) {
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
                InverseL1(1);
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagAttn);
                AttnL1(1);
            }
            for (uint32_t c = 0; c < nChunk_; ++c) {
                const uint32_t chunkAiv = AivOf(c);
                const bool onThisAiv = chunkAiv == sub_;
                if (!onThisAiv) {
                    LoadGate(c);
                }
                // eg of this chunk is saved before the next publish, which reuses ubEg_.
                // L2 is after the H upload: the next chunk's w@H was waiting on it.
                const uint32_t nxt = c + 2;
                LocalTensor<float> egNow = onThisAiv ? ubEg_ : ubEg2_;
                if (onThisAiv && nxt < nChunk_) {
                    Copy64(ubScratch_, ubEg_);
                    egNow = ubScratch_;
                }
                // Next chunk's GM load overlaps v_new, o and the saxpy. It used to
                // sit inside PublishStaged, in front of the cube's K@K / V wait.
                if (onThisAiv && nxt < nChunk_) {
                    PrepareLoad(nxt);
                }
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagP);
                const float decay = DecayOf(egNow);
                EmitV(egNow);
                // K flag before o / saxpy / H. The next cube's K@K sits behind it,
                // and those three used to push the L2 past that wait.
                if (onThisAiv && nxt < nChunk_) {
                    PublishK();
                }
                // The solve does not read P, but it shares this vector with v_new.
                // Running it first holds off kFlagV and the cube waits there.
                // Chunk 1 was inverted before this loop.
                if (!onThisAiv && c > 0 && c + 1 < nChunk_) {
                    CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
                    InverseL1(c + 1);
                    CrossCoreWaitFlag<0x4, PIPE_V>(kFlagAttn);
                    AttnL1(c + 1);
                }
                // D does not read o. Saxpy and the next H are queued before o is stored.
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagD);
                ApplyD(decay, c + 1 < nChunk_);
                if (onThisAiv && nxt < nChunk_) {
                    PublishKFlag(nxt);
                    PublishQ();
                    PublishQFlag(nxt);
                }
                CrossCoreWaitFlag<0x4, PIPE_MTE3>(kFlagO);
                StoreO(c);
            }
            constexpr uint32_t nElem = 64 * 128;
            const int64_t elem = static_cast<int64_t>(
                (static_cast<uint64_t>(batchIdx_) * valueHeadNum_ + headIdx_) * kDim * kDim +
                static_cast<uint64_t>(sub_) * nElem);
            const uint32_t next = head + aicNum_;
            if (loadPrologue && next < totalHeads_) {
                GetCurIdx(next);
                PrepareLoad(sub_);
                prefetched = true;
            }
            WaitFlag<HardEvent::V_MTE3>(1);
            WaitFlag<HardEvent::V_MTE3>(2);
            if (nzArmed_ != 0) {
                WaitFlag<HardEvent::MTE3_V>(3);
                nzArmed_ = 0;
            }
            PanelsToNd(ubD_, ubState_);
            SetFlag<HardEvent::V_MTE3>(1);
            WaitFlag<HardEvent::V_MTE3>(1);
            DataCopy(stateGm_[elem], ubD_, nElem);
            if (hasInitial_ != 0 && next < totalHeads_) {
                SetFlag<HardEvent::MTE3_MTE2>(0);
                stateLive = true;
            }
        }
    }

    __aicore__ inline int64_t StateElem() const
    {
        constexpr uint32_t nElem = 64 * 128;
        return static_cast<int64_t>((static_cast<uint64_t>(stateSlot_) * valueHeadNum_ + headIdx_) * kDim * kDim +
                                    static_cast<uint64_t>(sub_) * nElem);
    }

    // One packed sequence. Token 0 of the sequence is cu[seq], not a global multiple of 32.
    __aicore__ inline void BindSequence(uint32_t seq)
    {
        const int64_t begin = cuGm_.GetValue(static_cast<int64_t>(seq));
        const int64_t end = cuGm_.GetValue(static_cast<int64_t>(seq) + 1);
        const uint32_t len = end > begin ? static_cast<uint32_t>(end - begin) : 0U;
        tailRows_ = len % static_cast<uint32_t>(kBt);
        nChunk_ = len == 0U ? 0U
                            : (tailRows_ == 0U ? len / static_cast<uint32_t>(kBt)
                                               : len / static_cast<uint32_t>(kBt) + 1U);
        stateSlot_ = seq;
        vTokenBase_ = vHeadBase_ + begin;
        qTokenBase_ = qHeadBase_ + begin;
    }

    __aicore__ inline void InitStateSeq()
    {
        constexpr uint32_t nElem = 64 * 128;
        if (hasInitial_ != 0) {
            DataCopy(ubD_, initGm_[StateElem()], nElem);
            SetFlag<HardEvent::MTE2_V>(1);
            WaitFlag<HardEvent::MTE2_V>(1);
            NdToPanels(ubState_, ubD_);
        } else {
            Duplicate(ubState_, static_cast<bfloat16_t>(0), nElem);
        }
        SetFlag<HardEvent::V_MTE3>(1);
    }

    // Issue the initial-state copy, or zero ubD, before the next sequence's PrepareLoad.
    __aicore__ inline void FillEmptyUbD()
    {
        constexpr uint32_t nElem = 64 * 128;
        if (hasInitial_ != 0) {
            DataCopy(ubD_, initGm_[StateElem()], nElem);
            SetFlag<HardEvent::MTE2_MTE3>(1);
        } else {
            // The previous sequence's store may still be reading ubD on MTE3.
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            Duplicate(ubD_, static_cast<bfloat16_t>(0), nElem);
            SetFlag<HardEvent::V_MTE3>(1);
            WaitFlag<HardEvent::V_MTE3>(1);
        }
    }

    __aicore__ inline void FinishEmptyStore()
    {
        if (hasInitial_ != 0) {
            WaitFlag<HardEvent::MTE2_MTE3>(1);
        }
    }

    // GM loads of the next sequence only. Publishing K waits until that sequence
    // starts, because D of this sequence still reads l1K_.
    __aicore__ inline bool PrefetchNext(uint32_t head, uint32_t seq)
    {
        const uint32_t nextHead = head + aicNum_;
        if (seq + 1U < seqNum_) {
            BindSequence(seq + 1U);
        } else if (nextHead < totalHeads_) {
            GetCurIdx(nextHead);
            BindSequence(0);
        } else {
            return false;
        }
        if (nChunk_ > 0U && (sub_ == 0U || nChunk_ > 1U)) {
            PrepareLoadVar(sub_);
            return true;
        }
        return false;
    }

    __aicore__ inline void StoreState(int64_t elem)
    {
        constexpr uint32_t nElem = 64 * 128;
        WaitFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(2);
        if (nzArmed_ != 0) {
            WaitFlag<HardEvent::MTE3_V>(3);
            nzArmed_ = 0;
        }
        PanelsToNd(ubD_, ubState_);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        DataCopy(stateGm_[elem], ubD_, nElem);
    }

    __aicore__ inline void RunSequence()
    {
        const bool moreThanOneChunk = nChunk_ > 1;
        const bool loadPrologue = sub_ == 0 || moreThanOneChunk;
        if (loadPrologue) {
            PublishStaged(sub_);
        }
        if (AivOf(0) == sub_) {
            CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
            InverseL1(0);
            CrossCoreWaitFlag<0x4, PIPE_V>(kFlagAttn);
            AttnL1(0);
        }
        UploadH();
        if (moreThanOneChunk && AivOf(1) == sub_) {
            CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
            InverseL1(1);
            CrossCoreWaitFlag<0x4, PIPE_V>(kFlagAttn);
            AttnL1(1);
        }
        for (uint32_t c = 0; c < nChunk_; ++c) {
            const uint32_t chunkAiv = AivOf(c);
            const bool onThisAiv = chunkAiv == sub_;
            if (!onThisAiv) {
                LoadGateVar(c);
            }
            const uint32_t nxt = c + 2;
            LocalTensor<float> egNow = onThisAiv ? ubEg_ : ubEg2_;
            if (onThisAiv && nxt < nChunk_) {
                Copy64(ubScratch_, ubEg_);
                egNow = ubScratch_;
            }
            if (onThisAiv && nxt < nChunk_) {
                PrepareLoadVar(nxt);
            }
            CrossCoreWaitFlag<0x4, PIPE_V>(kFlagP);
            const float decay = DecayOf(egNow);
            EmitV(egNow);
            if (onThisAiv && nxt < nChunk_) {
                PublishK();
            }
            if (!onThisAiv && c > 0 && c + 1 < nChunk_) {
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagKkt);
                InverseL1(c + 1);
                CrossCoreWaitFlag<0x4, PIPE_V>(kFlagAttn);
                AttnL1(c + 1);
            }
            CrossCoreWaitFlag<0x4, PIPE_V>(kFlagD);
            ApplyD(decay, c + 1 < nChunk_);
            if (onThisAiv && nxt < nChunk_) {
                PublishKFlag(nxt);
                PublishQ();
                PublishQFlag(nxt);
            }
            CrossCoreWaitFlag<0x4, PIPE_MTE3>(kFlagO);
            StoreO(c);
        }
    }

    __aicore__ inline void ProcessAivVarlen()
    {
        if (!hasHead_) {
            return;
        }
        GenAuxMatrixVF(ubSh_, ubEye_);
        bool prefetched = false;
        bool stateLive = false;
        for (uint32_t head = headBegin_; head < totalHeads_; head += aicNum_) {
            if (!prefetched) {
                GetCurIdx(head);
            }
            for (uint32_t seq = 0; seq < seqNum_; ++seq) {
                const bool prepared = prefetched;
                if (!prepared) {
                    BindSequence(seq);
                }
                prefetched = false;
                if (stateLive) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                    stateLive = false;
                }
                const bool more = seq + 1U < seqNum_ || head + aicNum_ < totalHeads_;
                if (nChunk_ == 0U) {
                    const int64_t elem = StateElem();
                    FillEmptyUbD();
                    prefetched = PrefetchNext(head, seq);
                    FinishEmptyStore();
                    DataCopy(stateGm_[elem], ubD_, 64U * 128U);
                    if (more) {
                        SetFlag<HardEvent::MTE3_MTE2>(0);
                        stateLive = true;
                    }
                    continue;
                }
                const bool loadPrologue = sub_ == 0 || nChunk_ > 1;
                if (!prepared && loadPrologue) {
                    PrepareLoadVar(sub_);
                }
                InitStateSeq();
                RunSequence();
                const int64_t elem = StateElem();
                prefetched = PrefetchNext(head, seq);
                StoreState(elem);
                if (more) {
                    SetFlag<HardEvent::MTE3_MTE2>(0);
                    stateLive = true;
                }
            }
        }
    }

    __aicore__ inline void ProcessAicVarlen()
    {
        if (!hasHead_) {
            return;
        }
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        for (uint32_t head = headBegin_; head < totalHeads_; head += aicNum_) {
            GetCurIdx(head);
            for (uint32_t seq = 0; seq < seqNum_; ++seq) {
                BindSequence(seq);
                if (nChunk_ == 0U) {
                    continue;
                }
                for (uint32_t c = 0; c < nChunk_ && c < 2; ++c) {
                    Kkt(c, static_cast<uint8_t>(c));
                }
                for (uint32_t c = 0; c < nChunk_ && c < 2; ++c) {
                    Qk(c, static_cast<uint8_t>(c));
                }
                for (uint32_t c = 0; c < nChunk_; ++c) {
                    const uint32_t slot = AivOf(c);
                    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagInv + slot * kAiv1));
                    MatmulAvAndKbgLoadW(slot);

                    CrossCoreWaitFlag<0x4, PIPE_MTE1>(kFlagH0);
                    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagH0 + kAiv1));
                    MatmulWHCols0To63();

                    CrossCoreWaitFlag<0x4, PIPE_MTE1>(kFlagH);
                    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagH + kAiv1));
                    WaitFlag<HardEvent::M_MTE1>(1);
                    MatmulWHCols64To127WriteVnew();

                    ReloadHAndQexp(slot);
                    SetFlag<HardEvent::MTE1_M>(0);
                    if (c >= 1 && c + 1 < nChunk_) {
                        KktQk(c + 1, false);
                    }
                    CrossCoreWaitFlag<0x4, PIPE_MTE1>(kFlagV);
                    CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagV + kAiv1));
                    WaitFlag<HardEvent::MTE1_M>(0);
                    MatmulQexpHAndD(slot);

                    MatmulAttnVnew(slot);
                    if (c > 0) {
                        CrossCoreWaitFlag<0x4, PIPE_FIX>(kFlagOd);
                        CrossCoreWaitFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagOd + kAiv1));
                    }
                    FixpipeScaledO();
                }
                CrossCoreWaitFlag<0x4, PIPE_FIX>(kFlagOd);
                CrossCoreWaitFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagOd + kAiv1));
            }
        }
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(1);
    }

    __aicore__ inline void ProcessAic()
    {
        if (!hasHead_) {
            return;
        }
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        for (uint32_t head = headBegin_; head < totalHeads_; head += aicNum_) {
            // Chunk 0 uses L0C bank 0 and chunk 1 uses bank 1, so the second
            // MMAD overlaps the first fixpipe. Both K@K still finish before
            // either Q@K, so AIV1's inverse is not waiting out AIV0's Q@K.
            for (uint32_t c = 0; c < nChunk_ && c < 2; ++c) {
                Kkt(c, static_cast<uint8_t>(c));
            }
            for (uint32_t c = 0; c < nChunk_ && c < 2; ++c) {
                Qk(c, static_cast<uint8_t>(c));
            }
            for (uint32_t c = 0; c < nChunk_; ++c) {
                const uint32_t slot = AivOf(c);
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagInv + slot * kAiv1));
                MatmulAvAndKbgLoadW(slot);

                CrossCoreWaitFlag<0x4, PIPE_MTE1>(kFlagH0);
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagH0 + kAiv1));
                MatmulWHCols0To63();

                CrossCoreWaitFlag<0x4, PIPE_MTE1>(kFlagH);
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagH + kAiv1));
                WaitFlag<HardEvent::M_MTE1>(1);
                MatmulWHCols64To127WriteVnew();

                ReloadHAndQexp(slot);
                SetFlag<HardEvent::MTE1_M>(0);
                if (c >= 1 && c + 1 < nChunk_) {
                    KktQk(c + 1, false);
                }
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(kFlagV);
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(static_cast<uint16_t>(kFlagV + kAiv1));
                WaitFlag<HardEvent::MTE1_M>(0);
                MatmulQexpHAndD(slot);

                MatmulAttnVnew(slot);
                if (c > 0) {
                    CrossCoreWaitFlag<0x4, PIPE_FIX>(kFlagOd);
                    CrossCoreWaitFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagOd + kAiv1));
                }
                FixpipeScaledO();
            }
            CrossCoreWaitFlag<0x4, PIPE_FIX>(kFlagOd);
            CrossCoreWaitFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagOd + kAiv1));
        }
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(1);
    }

    OnChipBuffer buf;
    GlobalTensor<bfloat16_t> qGm_, kGm_, vGm_, betaGm_, oGm_, stateGm_, initGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<int64_t> cuGm_;
    LocalTensor<bfloat16_t> ubState_, ubQ_, ubK_, ubV_, ubQnz_, ubKnz_, ubKbg_, ubQexp_, ubVnz_;
    LocalTensor<bfloat16_t> ubBeta_, ubAbf_, ubAttn_, ubU_, ubVn_, ubVg_, ubO_, ubD_;
    LocalTensor<float> ubG_, ubBe_, ubEg_, ubKkt_, ubQk_, ubL_, ubA_, ubEye_, ubScratch_;
    LocalTensor<float> ubG2_, ubEg2_;
    LocalTensor<uint32_t> ubSh_;
    LocalTensor<bfloat16_t> l1H_, l1Vn_, l1Vg_;
    LocalTensor<bfloat16_t> l1Q_[2], l1K_[2], l1Vb_[2], l1Kbg_[2], l1Qexp_[2], l1A_[2], l1Attn_[2], l1W_[2];
    LocalTensor<bfloat16_t> l0A_[2], l0B_[2];
    LocalTensor<float> l0C0_, l0C1_;

    uint32_t sub_ = 0;
    uint32_t aicNum_ = 1;
    uint32_t headBegin_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t valueHeadNum_ = 0;
    uint32_t keyHeadNum_ = 0;
    uint32_t group_ = 1;
    uint32_t seqLen_ = 0;
    uint32_t nChunk_ = 0;
    uint32_t tailRows_ = 0;
    uint32_t totalHeads_ = 0;
    uint32_t batchIdx_ = 0;
    uint32_t headIdx_ = 0;
    int64_t qTokenBase_ = 0;
    int64_t vTokenBase_ = 0;
    int64_t qHeadBase_ = 0;
    int64_t vHeadBase_ = 0;
    uint32_t seqNum_ = 0;
    uint32_t stateSlot_ = 0;
    uint32_t hasInitial_ = 0;
    uint32_t nzArmed_ = 0;
    float scale_ = 0.f;
    bool hasHead_ = false;
};

}  // namespace GdnFwd32

#endif
