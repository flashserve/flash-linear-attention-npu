/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
// Stage | AIV order | Data retained
// 1/3 | both owners, with serialized dW/Tza residual handoffs | K/G/rH, Q/beta
// 7 | both owners before any DQ_FREE, each 32-row band | D/C in L1, diag/center
// 5/9/11 | complete current band for h0 then h1 | dG, gate_state per owner
// gate | full-chunk reverse scan h0 then h1 | FP32 parameter partials
//
// Memory | KiB interval | Last consumer / next meaning
// SHARED | [0,144) | pre -> dW/Zb -> Tza -> Local -> band results/base
// HEAD0 | [144,256) | Q/K/G/dG/META, survives both bands
// HEAD1 | [256,368) | same owner-relative layout, disjoint from HEAD0
// Results | [0,32),[32,64); right [64,80),[80,96) | independent owner destinations
//
// Event | Direction | What must finish before reuse
// sharedMte2Free/sharedVFree | MTE3 -> MTE2/V | prior phase UB source reads
// ownerEgress | MTE3 -> V | dv DMA before RIGHT_FREE; dq/dk/db/gate output DMA
// stateVToMte2 | V -> MTE2 | VF3 dv_b read before dKgb overwrites shared X
// directed cross flags | AIV/AIC | both Local and DQ_FREE precede RIGHT waits
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH92_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH92_VECTOR_H
#include "chunk_kda_bwd_finalize_common.h"
#include "kernel_utils/vector/regbase.hpp"
namespace KDA::Arch92 {
#include "chunk_kda_bwd_finalize_regbase.h"
} // namespace KDA::Arch92
#include "chunk_kda_bwd_finalize_gate.h"
namespace KDA::Arch92 {
struct VectorContext {
    AscendC::GlobalTensor<float> kRstd_;
    AscendC::GlobalTensor<DTYPE_RAW_G> rawG_;
    AscendC::GlobalTensor<float> dtBias_;
    AscendC::GlobalTensor<DTYPE_D_G> dG_;
    AscendC::GlobalTensor<float> aLog_;
    AscendC::GlobalTensor<bfloat16_t> dk_;
    AscendC::GlobalTensor<DTYPE_BETA> dBeta_;
    AscendC::GlobalTensor<bfloat16_t> q_;
    AscendC::GlobalTensor<bfloat16_t> k_;
    AscendC::GlobalTensor<bfloat16_t> v_;
    AscendC::GlobalTensor<float> gk_;
    AscendC::GlobalTensor<DTYPE_BETA> beta_;
    AscendC::GlobalTensor<bfloat16_t> h_;
    AscendC::GlobalTensor<bfloat16_t> dh_;
    AscendC::GlobalTensor<float> dAqk_;
    AscendC::GlobalTensor<float> dqRaw_;
    AscendC::GlobalTensor<float> qRstd_;
    AscendC::GlobalTensor<bfloat16_t> dq_;
    AscendC::GlobalTensor<bfloat16_t> dv_;
    AscendC::GlobalTensor<uint8_t> workspace_;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::TEventID mte2ToV_[2], vToMte3_[2], ownerEgress_[2];
    AscendC::TEventID sharedMte2Free_, sharedVFree_, stateVToMte2_;
    uint32_t zBPublishCount_[2] = {0, 0};
};

__aicore__ inline void AcquireShared(VectorContext &ctx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.sharedMte2Free_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.sharedVFree_);
}
__aicore__ inline void ReleaseShared(VectorContext &ctx)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.sharedMte2Free_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.sharedVFree_);
}
__aicore__ inline void InputReady(VectorContext &ctx, uint32_t owner)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV_[owner]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV_[owner]);
}
__aicore__ inline void OutputReady(VectorContext &ctx, uint32_t owner)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3_[owner]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3_[owner]);
}

__aicore__ inline void VectorInit(VectorContext &ctx, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
                                  GM_ADDR h, GM_ADDR dh, GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR qRstd, GM_ADDR dq,
                                  GM_ADDR dv, GM_ADDR kRstd, GM_ADDR dk, GM_ADDR dBeta, GM_ADDR rawG, GM_ADDR aLog,
                                  GM_ADDR dtBias, GM_ADDR dG, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                  GM_ADDR workspace, const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
{
    ctx.q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
    ctx.rawG_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_RAW_G *>(rawG));
    ctx.aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
    ctx.dtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
    ctx.dG_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_D_G *>(dG));
    ctx.k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
    ctx.v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
    ctx.gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
    ctx.beta_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_BETA *>(beta));
    ctx.h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
    ctx.dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
    ctx.dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
    ctx.dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
    if (tiling->hasQkL2Norm != 0U) {
        ctx.qRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
    }
    ctx.dq_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dq));
    ctx.dk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dk));
    ctx.dBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_BETA *>(dBeta));
    if (tiling->hasQkL2Norm != 0U) {
        ctx.kRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
    }
    ctx.dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
    ctx.workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(workspace));

    ctx.dq_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.dk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.dv_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.dG_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.dBeta_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    ctx.rawG_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.dAqk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.dqRaw_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    ctx.q_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.k_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.beta_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    ctx.cuSeqlens_ = cuSeqlens;
    ctx.chunkIndices_ = chunkIndices;
    ctx.tiling_ = tiling;
    ctx.pipe_ = pipe;
    ctx.pipe_->InitBuffer(ctx.ubBuf_, kUbBytes);
    ctx.ub_ = ctx.ubBuf_.Get<uint8_t>();
    for (uint32_t owner = 0; owner < 2; ++owner) {
        ctx.mte2ToV_[owner] = ctx.pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        ctx.vToMte3_[owner] = ctx.pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        ctx.ownerEgress_[owner] = ctx.pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
    }
    ctx.sharedMte2Free_ = ctx.pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    ctx.sharedVFree_ = ctx.pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
    ctx.stateVToMte2_ = ctx.pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.sharedMte2Free_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.sharedVFree_);
}

template <typename T>
__aicore__ inline void LoadModelRows(VectorContext &ctx, AscendC::LocalTensor<T> dst,
                                     const AscendC::GlobalTensor<T> &src, int64_t headOffset, uint32_t elems,
                                     bool tokenMajor)
{
    if (!tokenMajor) {
        AscendC::DataCopy(dst, src[headOffset], elems);
        return;
    }
    const int64_t flat = headOffset / KDA_FINALIZE_DIM;
    const int64_t batch = flat / (ctx.tiling_->NV * ctx.tiling_->T);
    const int64_t head = (flat / ctx.tiling_->T) % ctx.tiling_->NV;
    const int64_t offset =
        ((batch * ctx.tiling_->T + flat % ctx.tiling_->T) * ctx.tiling_->NV + head) * KDA_FINALIZE_DIM;
    AscendC::DataCopyPad(
        dst, src[offset],
        AscendC::DataCopyExtParams{static_cast<uint16_t>(elems / KDA_FINALIZE_DIM), KDA_FINALIZE_DIM * sizeof(T),
                                   static_cast<uint32_t>((ctx.tiling_->NV - 1) * KDA_FINALIZE_DIM * sizeof(T)), 0, 0},
        AscendC::DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
}

__aicore__ inline AscendC::LocalTensor<uint8_t> UbBytes(VectorContext &ctx, uint32_t offset)
{
    return ctx.ub_[offset];
}

__aicore__ inline AscendC::LocalTensor<bfloat16_t> L1Bf16(VectorContext &ctx, uint32_t offset)
{
    AscendC::LocalTensor<uint8_t> l1(AscendC::TPosition::A1, 0, 512 * 1024);
    return l1[offset].ReinterpretCast<bfloat16_t>();
}

__aicore__ inline void VectorStage1(VectorContext &ctx, const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner)
{
    AcquireShared(ctx);
    auto kUb = UbBytes(ctx, HeadBase(owner) + kHeadK).ReinterpretCast<bfloat16_t>();
    auto gkUb = UbBytes(ctx, HeadBase(owner) + kHeadG).ReinterpretCast<float>();
    auto hUb = UbBytes(ctx, kPreH).ReinterpretCast<bfloat16_t>();
    auto dhUb = UbBytes(ctx, kPreDh).ReinterpretCast<bfloat16_t>();
    auto kENd = UbBytes(ctx, kPreKeHi).ReinterpretCast<bfloat16_t>();
    auto rH = UbBytes(ctx, HeadBase(owner) + kRh).ReinterpretCast<float>();
    auto lowNd = UbBytes(ctx, kPreKeLo).ReinterpretCast<bfloat16_t>();

    const int64_t token = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_DIM);
    const int64_t hState = FinalizeHOffset(*ctx.tiling_, chunk, head);
    const int64_t state = FinalizeDhOffset(*ctx.tiling_, chunk, head);
    LoadModelRows(ctx, kUb, ctx.k_, token, chunk.validRows * KDA_FINALIZE_DIM, ctx.tiling_->qkTokenMajor);
    AscendC::DataCopy(gkUb, ctx.gk_[token], chunk.validRows * KDA_FINALIZE_DIM);
    AscendC::DataCopy(hUb, ctx.h_[hState], KDA_FINALIZE_STATE_ELEMS);
    AscendC::DataCopy(dhUb, ctx.dh_[state], KDA_FINALIZE_STATE_ELEMS);
    InputReady(ctx, owner);
    FinalizeStage0VF(
        reinterpret_cast<__ubuf__ bfloat16_t *>(kENd.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(lowNd.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()), reinterpret_cast<__ubuf__ bfloat16_t *>(kUb.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(gkUb.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(hUb.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(dhUb.GetPhyAddr()), static_cast<uint16_t>(chunk.validRows));

    OutputReady(ctx, owner);
    auto kEL1 = L1Bf16(ctx, 96 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
    AscendC::DataCopy(kEL1, kENd, 8192);
    AscendC::DataCopy(L1Bf16(ctx, 128 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES), lowNd, 8192);

    ReleaseShared(ctx);
}

__aicore__ inline void VectorStage3(VectorContext &ctx, const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner)
{
    AcquireShared(ctx);
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZV_READY_BASE + owner);
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + owner);
    auto zV = UbBytes(ctx, kZv + owner * KDA_FINALIZE_MATRIX_FP32_BYTES).ReinterpretCast<float>();
    auto zW = UbBytes(ctx, kZw + owner * KDA_FINALIZE_MATRIX_FP32_BYTES).ReinterpretCast<float>();
    auto betaUb = UbBytes(ctx, HeadBase(owner) + kBeta).ReinterpretCast<DTYPE_BETA>();
    auto zBNd = UbBytes(ctx, kZb).ReinterpretCast<bfloat16_t>();

    auto qUb = UbBytes(ctx, HeadBase(owner) + kHeadQ).ReinterpretCast<bfloat16_t>();
    const int64_t betaOffset = FinalizeTokenOffset(*ctx.tiling_, chunk, head, 1);
    const int64_t token = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_DIM);

    AscendC::DataCopyExtParams betaCopy{1, static_cast<uint32_t>(chunk.validRows * sizeof(DTYPE_BETA)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DTYPE_BETA> betaPad{true, 0, 0, static_cast<DTYPE_BETA>(0)};
    AscendC::DataCopyPad(betaUb, ctx.beta_[betaOffset], betaCopy, betaPad);
    LoadModelRows(ctx, qUb, ctx.q_, token, chunk.validRows * KDA_FINALIZE_DIM, ctx.tiling_->qkTokenMajor);
    InputReady(ctx, owner);
    FinalizeStage2VF(
        reinterpret_cast<__ubuf__ bfloat16_t *>(zBNd.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(zV.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(zW.GetPhyAddr()),
        reinterpret_cast<__ubuf__ DTYPE_BETA *>(betaUb.GetPhyAddr()), static_cast<uint16_t>(chunk.validRows));
    OutputReady(ctx, owner);
    auto zBL1 = L1Bf16(ctx, 160 * 1024 + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
    if (ctx.zBPublishCount_[owner] != 0U) {
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZB_FREE_BASE + owner);
    }
    AscendC::DataCopy(zBL1, zBNd, 2 * KDA_FINALIZE_MATRIX_ELEMS);
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZB_READY_BASE + owner);
    ++ctx.zBPublishCount_[owner];
    ReleaseShared(ctx);
}

// Each FP32 NZ plane contains 64 consecutive 32-byte row blocks.
template <uint16_t PLANES>
__aicore__ inline void CopyLocalRows(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, uint16_t begin,
                                     uint16_t end)
{
    const uint16_t rows = end - begin;
    if (rows == KDA_FINALIZE_CHUNK) {
        AscendC::DataCopy(dst, src, PLANES * KDA_FINALIZE_CHUNK * 8);
    } else {
        const uint16_t skip = KDA_FINALIZE_CHUNK - rows;
        AscendC::DataCopyParams copy{PLANES, rows, skip, skip};
        AscendC::DataCopy(dst[begin * 8U], src[begin * 8U], copy);
    }
}

__aicore__ inline void VectorStage7(VectorContext &ctx, const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
                                    uint32_t rowBegin)
{
    AcquireShared(ctx);

    auto dAkkRaw = UbBytes(ctx, HeadBase(owner) + kHeadDg).ReinterpretCast<float>();

    auto dAqkNd = UbBytes(ctx, kLocalD).ReinterpretCast<FinalizeLocalType>();
    auto dAkkNd = UbBytes(ctx, kLocalC).ReinterpretCast<FinalizeLocalType>();
    auto kNegNd = UbBytes(ctx, kLocalKn).ReinterpretCast<FinalizeLocalType>();
    auto qPosNd = UbBytes(ctx, kLocalQp).ReinterpretCast<FinalizeLocalType>();
    auto bkPosNd = UbBytes(ctx, kLocalBp).ReinterpretCast<FinalizeLocalType>();
    auto q = UbBytes(ctx, HeadBase(owner) + kHeadQ).ReinterpretCast<bfloat16_t>();
    auto k = UbBytes(ctx, HeadBase(owner) + kHeadK).ReinterpretCast<bfloat16_t>();
    auto g = UbBytes(ctx, HeadBase(owner) + kHeadG).ReinterpretCast<float>();
    auto beta = UbBytes(ctx, HeadBase(owner) + kBeta).ReinterpretCast<DTYPE_BETA>();
    auto dAqkFp32 = UbBytes(ctx, kLocalRawD).ReinterpretCast<float>();

    const int64_t matrixToken = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_CHUNK);
    const uint32_t matrixElems = static_cast<uint32_t>(chunk.validRows) * KDA_FINALIZE_CHUNK;
    if (rowBegin == 0) {
        AscendC::DataCopy(dAqkFp32, ctx.dAqk_[matrixToken], matrixElems);
    }

    InputReady(ctx, owner);
    auto center = UbBytes(ctx, HeadBase(owner) + kCenter).ReinterpretCast<float>();
    const uint32_t rowEnd = FinalizeMin(rowBegin + KDA_FINALIZE_INTRA_ROWS, chunk.validRows);
    FinalizeIntraCenterVF(reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()) + rowBegin * 128,
                          reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()), rowEnd - rowBegin);
    FinalizeStage5VF(reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAkkNd.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ FinalizeLocalType *>(kNegNd.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ FinalizeLocalType *>(qPosNd.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ FinalizeLocalType *>(bkPosNd.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dAqkNd.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dAqkFp32.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(UbBytes(ctx, HeadBase(owner) + kDiagonal).GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dAkkRaw.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()), static_cast<uint16_t>(chunk.validRows),
                     rowBegin, rowEnd);
    if (chunk.validRows < KDA_FINALIZE_CHUNK) {
        FinalizeStage5TailVF(reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAqkNd.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ FinalizeLocalType *>(dAkkNd.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ FinalizeLocalType *>(kNegNd.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ FinalizeLocalType *>(qPosNd.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ FinalizeLocalType *>(bkPosNd.GetPhyAddr()), chunk.validRows);
    }

    OutputReady(ctx, owner);
    auto local =
        L1Bf16(ctx, KDA_FINALIZE_LOCAL_BASE + owner * KDA_FINALIZE_LOCAL_BYTES).ReinterpretCast<FinalizeLocalType>();
    const uint16_t paddedRows = (chunk.validRows + 7U) / 8U * 8U;
    if (rowBegin == 0) {
        CopyLocalRows<8>(local[KDA_FINALIZE_LOCAL_DAQK / sizeof(float)], dAqkNd, 0, paddedRows);
        CopyLocalRows<8>(local[KDA_FINALIZE_LOCAL_DAKK / sizeof(float)], dAkkNd, 0, paddedRows);
    }
    const uint16_t paddedEnd = (rowEnd + 7U) / 8U * 8U;
    CopyLocalRows<16>(local[KDA_FINALIZE_LOCAL_K_NEG / sizeof(float)], kNegNd, 0, paddedEnd);
    CopyLocalRows<16>(local[KDA_FINALIZE_LOCAL_Q_POS / sizeof(float)], qPosNd, rowBegin, paddedRows);
    CopyLocalRows<16>(local[KDA_FINALIZE_LOCAL_BK_POS / sizeof(float)], bkPosNd, rowBegin, paddedRows);
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_LOCAL_READY_BASE + owner);
    ReleaseShared(ctx);
}

__aicore__ inline void VectorStage5(VectorContext &ctx, const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner,
                                    int64_t coreIdx, uint64_t groupGeneration, uint32_t rowBegin)
{
    AcquireShared(ctx);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
    const uint32_t rows = FinalizeMin(KDA_FINALIZE_INTRA_ROWS, chunk.validRows - rowBegin);
    const uint32_t elems = rows * KDA_FINALIZE_DIM;
    const uint32_t offset = rowBegin * KDA_FINALIZE_DIM;
    const int64_t token = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_DIM);
    const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);

    auto dkState = UbBytes(ctx, kDkBase).ReinterpretCast<float>();
    auto dqRaw = UbBytes(ctx, kDqBase).ReinterpretCast<float>();

    auto dvb = UbBytes(ctx, kBaseX).ReinterpretCast<float>();
    auto v = UbBytes(ctx, RightBase(owner)).ReinterpretCast<bfloat16_t>();
    auto dv = UbBytes(ctx, RightBase(owner) + 8 * 1024).ReinterpretCast<bfloat16_t>();
    auto q = UbBytes(ctx, HeadBase(owner) + kHeadQ).ReinterpretCast<bfloat16_t>();
    auto k = UbBytes(ctx, HeadBase(owner) + kHeadK).ReinterpretCast<bfloat16_t>();
    auto g = UbBytes(ctx, HeadBase(owner) + kHeadG).ReinterpretCast<float>();
    auto gkLast = g[static_cast<uint32_t>(chunk.validRows - 1) * KDA_FINALIZE_DIM];
    auto beta = UbBytes(ctx, HeadBase(owner) + kBeta).ReinterpretCast<DTYPE_BETA>();
    auto rH = UbBytes(ctx, HeadBase(owner) + kRh).ReinterpretCast<float>();
    auto gateState = UbBytes(ctx, HeadBase(owner) + kGateState).ReinterpretCast<float>();
    auto dbV = UbBytes(ctx, HeadBase(owner) + kDbBase).ReinterpretCast<float>();
    auto dg = UbBytes(ctx, HeadBase(owner) + kHeadDg).ReinterpretCast<float>();

    auto wsDkState = ctx.workspace_[ws + KDA_FINALIZE_WS_DK_STATE_RAW].ReinterpretCast<float>();
    auto wsDvb = ctx.workspace_[ws + KDA_FINALIZE_WS_DVB].ReinterpretCast<float>();
    auto wsDkg = ctx.workspace_[ws + KDA_FINALIZE_WS_DKGB_RAW].ReinterpretCast<float>();
    AscendC::DataCopy(dkState, wsDkState[offset], elems);
    AscendC::DataCopy(dvb, wsDvb[offset], elems);
    AscendC::DataCopy(dqRaw, ctx.dqRaw_[token + offset], elems);
    LoadModelRows(ctx, v, ctx.v_, token + offset, elems, ctx.tiling_->vTokenMajor);
    InputReady(ctx, owner);
    FinalizeStage3VF(
        reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(dvb.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(dv.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(dqRaw.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(g[offset].GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(k[offset].GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(v.GetPhyAddr()),
        reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta[rowBegin].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(gkLast.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(rH.GetPhyAddr()),
        ctx.tiling_->scale, static_cast<uint16_t>(chunk.validRows), static_cast<uint16_t>(rowBegin),
        static_cast<uint16_t>(rows));
    OutputReady(ctx, owner);
    const bool tokenMajor = ctx.tiling_->outputTokenMajor != 0U;
    const uint64_t out = FinalizeOutputOffset(*ctx.tiling_, chunk, head, rowBegin, KDA_FINALIZE_DIM, tokenMajor);
    AscendC::DataCopyPad(
        ctx.dv_[out], dv,
        AscendC::DataCopyExtParams{
            static_cast<uint16_t>(rows), KDA_FINALIZE_DIM * sizeof(bfloat16_t), 0,
            tokenMajor ? static_cast<uint32_t>((ctx.tiling_->NV - 1) * KDA_FINALIZE_DIM * sizeof(bfloat16_t)) : 0U, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
    // VF3 must finish reading dv_b before the DMA loads dKgb into the same address.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.stateVToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.stateVToMte2_);
    AscendC::DataCopy(dvb, wsDkg[offset], elems);
    InputReady(ctx, owner);
    FinalizeStage4VF(reinterpret_cast<__ubuf__ float *>(dkState.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dqRaw.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dvb.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dg[offset].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(g[offset].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ bfloat16_t *>(q[offset].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ bfloat16_t *>(k[offset].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta[rowBegin].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(gateState.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dbV.GetPhyAddr()), static_cast<uint16_t>(chunk.validRows),
                     static_cast<uint16_t>(rowBegin), static_cast<uint16_t>(rows));
}

__aicore__ inline void VectorStage9And11(VectorContext &ctx, const FinalizeChunkInfo &chunk, int64_t head,
                                         uint32_t owner, uint32_t rowBegin)
{
    const uint32_t rows = FinalizeMin(KDA_FINALIZE_INTRA_ROWS, chunk.validRows - rowBegin);
    const uint32_t elems = rows * KDA_FINALIZE_DIM;
    const uint32_t offset = rowBegin * KDA_FINALIZE_DIM;
    const int64_t token = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_DIM) + offset;
    auto result = UbBytes(ctx, ResultBase(owner)).ReinterpretCast<float>();
    auto left = result[KDA_FINALIZE_INTRA_ROWS * KDA_FINALIZE_DIM];
    auto right = UbBytes(ctx, RightBase(owner)).ReinterpretCast<float>();

    auto dg = UbBytes(ctx, HeadBase(owner) + kHeadDg).ReinterpretCast<float>();

    auto dqBase = UbBytes(ctx, kDqBase).ReinterpretCast<float>();
    auto dkBase = UbBytes(ctx, kDkBase).ReinterpretCast<float>();

    auto dqOut = UbBytes(ctx, ResultBase(owner)).ReinterpretCast<bfloat16_t>();
    auto dkOut = UbBytes(ctx, RightBase(owner)).ReinterpretCast<bfloat16_t>();

    auto g = UbBytes(ctx, HeadBase(owner) + kHeadG).ReinterpretCast<float>()[offset];
    auto center = UbBytes(ctx, HeadBase(owner) + kCenter).ReinterpretCast<float>();
    auto bandExp = UbBytes(ctx, kBaseX).ReinterpretCast<float>();
    auto q = UbBytes(ctx, HeadBase(owner) + kHeadQ).ReinterpretCast<bfloat16_t>()[offset];
    auto k = UbBytes(ctx, HeadBase(owner) + kHeadK).ReinterpretCast<bfloat16_t>()[offset];
    auto beta = UbBytes(ctx, HeadBase(owner) + kBeta).ReinterpretCast<DTYPE_BETA>()[rowBegin];
    auto diagonal = UbBytes(ctx, HeadBase(owner) + kDiagonal).ReinterpretCast<float>()[rowBegin];
    auto dbBase = UbBytes(ctx, HeadBase(owner) + kDbBase).ReinterpretCast<float>();
    auto dbDelta = UbBytes(ctx, HeadBase(owner) + kDbDelta).ReinterpretCast<float>();
    auto qRstd = UbBytes(ctx, HeadBase(owner) + kQrstd).ReinterpretCast<float>();
    auto kRstd = UbBytes(ctx, HeadBase(owner) + kKrstd).ReinterpretCast<float>();
    auto dbOut = UbBytes(ctx, HeadBase(owner) + kDbOut).ReinterpretCast<DTYPE_BETA>();
    const AscendC::DataCopyExtParams scalarCopy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> scalarPad{false, 0, 0, 0.0f};
    if (ctx.tiling_->hasQkL2Norm != 0U) {
        AscendC::DataCopyPad(qRstd, ctx.qRstd_[token / KDA_FINALIZE_DIM], scalarCopy, scalarPad);
        AscendC::DataCopyPad(kRstd, ctx.kRstd_[token / KDA_FINALIZE_DIM], scalarCopy, scalarPad);
    }
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_DQ_LOCAL_READY_BASE + owner);
    InputReady(ctx, owner);
    FinalizeStage7VF(
        reinterpret_cast<__ubuf__ bfloat16_t *>(dqOut.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(dg[offset].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(result.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(dqBase.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(center.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(qRstd.GetPhyAddr()),
        ctx.tiling_->hasQkL2Norm, rows, reinterpret_cast<__ubuf__ float *>(diagonal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(bandExp.GetPhyAddr()));
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_KE_READY_BASE + owner);
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + owner);
    FinalizeStage9VF(reinterpret_cast<__ubuf__ float *>(left.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(right.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dbDelta.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(dg[offset].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(bandExp.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ DTYPE_BETA *>(beta.GetPhyAddr()), rows,
                     reinterpret_cast<__ubuf__ float *>(diagonal.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()));
    FinalizeStage10VF(reinterpret_cast<__ubuf__ bfloat16_t *>(dkOut.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ DTYPE_BETA *>(dbOut.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(left.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(dkBase.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(dbDelta.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(dbBase.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(kRstd.GetPhyAddr()), ctx.tiling_->hasQkL2Norm, rows);
    OutputReady(ctx, owner);
    const bool tokenMajor = ctx.tiling_->outputTokenMajor != 0U;
    const uint64_t out = FinalizeOutputOffset(*ctx.tiling_, chunk, head, rowBegin, KDA_FINALIZE_DIM, tokenMajor);
    const AscendC::DataCopyExtParams matrixCopy{
        static_cast<uint16_t>(rows), KDA_FINALIZE_DIM * sizeof(bfloat16_t), 0,
        tokenMajor ? static_cast<uint32_t>((ctx.tiling_->NV - 1) * KDA_FINALIZE_DIM * sizeof(bfloat16_t)) : 0U, 0};
    AscendC::DataCopyPad(ctx.dq_[out], dqOut, matrixCopy);
    AscendC::DataCopyPad(ctx.dk_[out], dkOut, matrixCopy);
    const bool gateTokenMajor = ctx.tiling_->gateOutputTokenMajor != 0U;
    const uint64_t betaOut = FinalizeOutputOffset(*ctx.tiling_, chunk, head, rowBegin, 1, gateTokenMajor);
    AscendC::DataCopyPad(ctx.dBeta_[betaOut], dbOut,
                         AscendC::DataCopyExtParams{
                             static_cast<uint16_t>(rows), sizeof(DTYPE_BETA), 0,
                             gateTokenMajor ? static_cast<uint32_t>((ctx.tiling_->NV - 1) * sizeof(DTYPE_BETA)) : 0U,
                             0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
    ReleaseShared(ctx);
}

__aicore__ inline void VectorGate(VectorContext &ctx, const FinalizeChunkInfo &chunk, int64_t chunkTask, int64_t head,
                                  uint32_t owner)
{
    AcquireShared(ctx);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);

    auto dg = UbBytes(ctx, HeadBase(owner) + kHeadDg).ReinterpretCast<float>();
    auto raw = UbBytes(ctx, HeadBase(owner) + kHeadG).ReinterpretCast<DTYPE_RAW_G>();
    auto bias = UbBytes(ctx, HeadBase(owner) + kBias).ReinterpretCast<float>();
    auto log = UbBytes(ctx, HeadBase(owner) + kALog).ReinterpretCast<float>();
    auto da = UbBytes(ctx, HeadBase(owner) + kPartialA).ReinterpretCast<float>();
    auto db = UbBytes(ctx, HeadBase(owner) + kPartialBias).ReinterpretCast<float>();
    const int64_t token = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_DIM);
    LoadModelRows(ctx, raw, ctx.rawG_, token, chunk.validRows * KDA_FINALIZE_DIM, ctx.tiling_->rawTokenMajor);
    AscendC::DataCopy(bias, ctx.dtBias_[head * KDA_FINALIZE_DIM], KDA_FINALIZE_DIM);
    AscendC::DataCopyPad(log, ctx.aLog_[head], AscendC::DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                         AscendC::DataCopyPadExtParams<float>{false, 0, 0, static_cast<float>(0)});
    InputReady(ctx, owner);
    FinalizeStage11VF<DTYPE_RAW_G>(
        reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()), reinterpret_cast<__ubuf__ DTYPE_RAW_G *>(raw.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(log.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(da.GetPhyAddr()), reinterpret_cast<__ubuf__ float *>(db.GetPhyAddr()),
        chunk.validRows, ctx.tiling_->lowerBound);

    auto packedG = raw.ReinterpretCast<DTYPE_D_G>();
    if constexpr (sizeof(DTYPE_D_G) == 2) {
        FinalizeGatePackVF(reinterpret_cast<__ubuf__ bfloat16_t *>(packedG.GetPhyAddr()),
                           reinterpret_cast<__ubuf__ float *>(dg.GetPhyAddr()), chunk.validRows);
    } else {
        packedG = dg.ReinterpretCast<DTYPE_D_G>();
    }
    OutputReady(ctx, owner);
    const int64_t partial = head * ctx.tiling_->totalChunkNum + chunkTask;
    const bool tokenMajor = ctx.tiling_->gateOutputTokenMajor != 0U;
    const uint64_t out = FinalizeOutputOffset(*ctx.tiling_, chunk, head, 0, KDA_FINALIZE_DIM, tokenMajor);
    AscendC::DataCopyPad(
        ctx.dG_[out], packedG,
        AscendC::DataCopyExtParams{
            static_cast<uint16_t>(chunk.validRows), KDA_FINALIZE_DIM * sizeof(DTYPE_D_G), 0,
            tokenMajor ? static_cast<uint32_t>((ctx.tiling_->NV - 1) * KDA_FINALIZE_DIM * sizeof(DTYPE_D_G)) : 0U, 0});
    AscendC::DataCopyPad(ctx.workspace_[ctx.tiling_->gatePartialOffset].ReinterpretCast<float>()[partial * 8], da,
                         AscendC::DataCopyExtParams{1, sizeof(float), 0, 0, 0});
    AscendC::DataCopy(ctx.workspace_[ctx.tiling_->dtBiasPartialOffset].ReinterpretCast<float>()[partial * 128], db,
                      128);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
    ReleaseShared(ctx);
}

__aicore__ inline void VectorDwResidual(VectorContext &ctx, uint32_t owner)
{
    AcquireShared(ctx);
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZB_READY_BASE + owner);
    auto raw = UbBytes(ctx, kDwRaw).ReinterpretCast<float>();
    auto low = UbBytes(ctx, kDwLo).ReinterpretCast<bfloat16_t>();
    FinalizeDwResidualVF((__ubuf__ bfloat16_t *)low.GetPhyAddr(), (__ubuf__ float *)raw.GetPhyAddr());
    OutputReady(ctx, owner);
    AscendC::DataCopy(L1Bf16(ctx, kL1DwLo + owner * KDA_FINALIZE_VECTOR_BF16_BYTES), low, KDA_FINALIZE_VECTOR_ELEMS);
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_KE_READY_BASE + owner);
    ReleaseShared(ctx);
}
__aicore__ inline void VectorTzaResidual(VectorContext &ctx, uint32_t owner)
{
    AcquireShared(ctx);
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_ZW_READY_BASE + owner);
    auto raw = UbBytes(ctx, kTzaRaw).ReinterpretCast<float>();
    auto low = UbBytes(ctx, kTzaLo).ReinterpretCast<bfloat16_t>();
    FinalizeResidualVF((__ubuf__ bfloat16_t *)low.GetPhyAddr(), (__ubuf__ float *)raw.GetPhyAddr(), 64);
    OutputReady(ctx, owner);
    AscendC::DataCopy(L1Bf16(ctx, kL1Tza + (2 * owner + 1) * KDA_FINALIZE_MATRIX_BF16_BYTES), low,
                      KDA_FINALIZE_MATRIX_ELEMS);
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_KE_READY_BASE + owner);
    ReleaseShared(ctx);
}

__aicore__ inline void VectorDrain(VectorContext &ctx)
{
    AcquireShared(ctx);
    for (uint32_t owner = 0; owner < 2; ++owner) {
        if (ctx.zBPublishCount_[owner] != 0) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZB_FREE_BASE + owner);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
        ctx.pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV_[owner]);
        ctx.pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(ctx.vToMte3_[owner]);
        ctx.pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(ctx.ownerEgress_[owner]);
    }
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_TASK_L1_FREE);
    ctx.pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(ctx.sharedMte2Free_);
    ctx.pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(ctx.sharedVFree_);
    ctx.pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(ctx.stateVToMte2_);
}

__aicore__ inline void VectorProcess(VectorContext &ctx)
{
    const int64_t core = AscendC::GetBlockIdx();
    const int64_t stride = AscendC::GetBlockNum();
    uint64_t generation = 0;
    for (int64_t task = core; task < ctx.tiling_->workTaskNum; task += stride, ++generation) {
        const int64_t window = task / ctx.tiling_->chunkTaskNum;
        const int64_t chunkTask = task % ctx.tiling_->chunkTaskNum;
        const int64_t firstHead = window * KDA_FINALIZE_HEADS_PER_WINDOW;
        const uint32_t active = static_cast<uint32_t>(FinalizeMin(2, ctx.tiling_->NV - firstHead));
        FinalizeChunkInfo chunk;
        ResolveFinalizeChunk(chunkTask, ctx.cuSeqlens_, ctx.chunkIndices_, *ctx.tiling_, chunk);
        if (!chunk.valid)
            continue;
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_TASK_L1_FREE);
        for (uint32_t owner = 0; owner < active; ++owner) {
            VectorStage1(ctx, chunk, firstHead + owner, owner);
        }
        // All pre-phase H/dH/kE sources must be retired before Cube may overwrite SHARED.
        for (uint32_t owner = 0; owner < active; ++owner) {
            AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZV_FREE_BASE + owner);
            AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZW_FREE_BASE + owner);
        }
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_LOCAL_READY_BASE);
        for (uint32_t owner = 0; owner < active; ++owner) {
            VectorDwResidual(ctx, owner);
            if (owner + 1 < active) {
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_LOCAL_READY_BASE + owner +
                                                                              1);
            }
        }
        for (uint32_t owner = 0; owner < active; ++owner) {
            VectorStage3(ctx, chunk, firstHead + owner, owner);
        }
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZV_FREE_BASE);
        for (uint32_t owner = 0; owner < active; ++owner) {
            VectorTzaResidual(ctx, owner);
            if (owner + 1 < active) {
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_ZV_FREE_BASE + owner + 1);
            }
        }
        for (uint32_t owner = 0; owner < active; ++owner) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_V>(KDA_FINALIZE_DAKK_READY_BASE + owner);
        }
        for (uint32_t row = 0; row < chunk.validRows; row += KDA_FINALIZE_INTRA_ROWS) {
            // Publishing only owner0 and then waiting for RIGHT0 deadlocks against Cube Stage8(owner1).
            for (uint32_t owner = 0; owner < active; ++owner) {
                VectorStage7(ctx, chunk, firstHead + owner, owner, row);
            }
            for (uint32_t owner = 0; owner < active; ++owner) {
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE3>(KDA_FINALIZE_DQ_LOCAL_FREE_BASE + owner);
            }
            for (uint32_t owner = 0; owner < active; ++owner) {
                VectorStage5(ctx, chunk, firstHead + owner, owner, core, generation, row);
                VectorStage9And11(ctx, chunk, firstHead + owner, owner, row);
            }
        }
        for (uint32_t owner = 0; owner < active; ++owner) {
            VectorGate(ctx, chunk, chunkTask, firstHead + owner, owner);
        }
    }
    VectorDrain(ctx);
}

} // namespace KDA::Arch92
#endif
