#ifndef CHUNK_SCALED_DOT_KKT_H
#define CHUNK_SCALED_DOT_KKT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

struct ChunkScaledDotKktTilingData;

namespace NsChunkScaledDotKkt {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t SCORE_WORKSPACE_BUFFER_NUM = 2;
constexpr int32_t SCORE_WORKSPACE_HEAD_BATCH = 1;
constexpr int32_t FP32_BLOCK_ELEMS = 8;
constexpr int32_t FP32_REPEAT_ELEMS = 64;
constexpr int32_t BRCB_ROWS = 8;
constexpr int32_t UB_ALIGN_BYTES = 32;
constexpr uint8_t SCORE_DONE_FLAG0 = 2;
constexpr uint8_t SCORE_DONE_FLAG1 = 3;
constexpr uint8_t SCORE_READY_FLAG0 = 4;
constexpr uint8_t SCORE_READY_FLAG1 = 5;
constexpr MatmulConfig CHUNK_SCALED_DOT_KKT_MM_CFG = GetNormalConfig(true);

using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
using KktArchTag = Catlass::Arch::AtlasA2;
using KktScoreDispatchPolicy = Catlass::Gemm::MmadPingpongTlaMulti<KktArchTag, true, false>;
using KktInt128 = tla::Int<128>;

template <typename KType, uint32_t CHUNK_KEY>
class ChunkScaledDotKkt {
    struct TaskMeta {
        int64_t b = 0;
        int64_t h = 0;
        int64_t chunk = 0;
        int64_t rowStart = 0;
        int64_t valid = 0;
    };

public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KType>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KType, true, LayoutMode::NONE, false>;

    __aicore__ inline ChunkScaledDotKkt() {}

    __aicore__ inline void Init(GM_ADDR k,
                                GM_ADDR g,
                                GM_ADDR beta,
                                GM_ADDR cuSeqlens,
                                GM_ADDR chunkIndices,
                                GM_ADDR a,
                                GM_ADDR scoreWorkspace,
                                uint64_t b,
                                uint64_t hk,
                                uint64_t hv,
                                uint64_t hvPerHk,
                                uint64_t t,
                                uint64_t kDim,
                                uint64_t bt,
                                uint64_t nt,
                                uint64_t taskNum,
                                uint64_t usedAicNum,
                                uint64_t usedAivNum,
                                uint64_t btAlign,
                                uint64_t isVarlen,
                                TPipe *pipe)
    {
        pipe_ = pipe;
        B_ = static_cast<int64_t>(b);
        Hk_ = static_cast<int64_t>(hk);
        Hv_ = static_cast<int64_t>(hv);
        hvPerHk_ = static_cast<int64_t>(hvPerHk);
        T_ = static_cast<int64_t>(t);
        K_ = static_cast<int64_t>(kDim);
        BT_ = static_cast<int64_t>(bt);
        NT_ = static_cast<int64_t>(nt);
        taskNum_ = static_cast<int64_t>(taskNum);
        usedAicNum_ = static_cast<int64_t>(usedAicNum);
        usedAivNum_ = static_cast<int64_t>(usedAivNum);
        btAlign_ = static_cast<int64_t>(btAlign);
        isVarlen_ = static_cast<int64_t>(isVarlen);

        kGm.SetGlobalBuffer((__gm__ KType *)k, B_ * Hk_ * T_ * K_);
        gGm.SetGlobalBuffer((__gm__ float *)g, B_ * Hv_ * T_);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, B_ * Hv_ * T_);
        aGm.SetGlobalBuffer((__gm__ float *)a, B_ * Hk_ * T_ * BT_);
        scoreGm.SetGlobalBuffer((__gm__ float *)scoreWorkspace,
                                usedAivNum_ * SCORE_WORKSPACE_BUFFER_NUM * SCORE_WORKSPACE_HEAD_BATCH * BT_ * BT_);
        if (isVarlen_ != 0) {
            cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
            chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunkIndices, NT_ * 2);
        }

        if ASCEND_IS_AIV {
            pipe_->InitBuffer(gQueue_, BUFFER_NUM, btAlign_ * sizeof(float));
            pipe_->InitBuffer(betaQueue_, BUFFER_NUM, btAlign_ * sizeof(float));
            pipe_->InitBuffer(scoreTileBuf_, static_cast<uint32_t>(BT_ * btAlign_ * sizeof(float)));
            pipe_->InitBuffer(outTileBuf_, static_cast<uint32_t>(BT_ * btAlign_ * sizeof(float)));
            pipe_->InitBuffer(gateBuf_, BRCB_ROWS * btAlign_ * sizeof(float));
            pipe_->InitBuffer(rowBrcbBuf_, BRCB_ROWS * FP32_BLOCK_ELEMS * sizeof(float));
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        if (UseCatlassScore()) {
            ProcessAivCatlass();
            return;
        }
        const int64_t vecIdx = static_cast<int64_t>(GetBlockIdx());
        if (vecIdx >= usedAivNum_ || usedAivNum_ <= 0) {
            return;
        }
        int64_t scoreSlot = 0;
        int64_t cachedScoreValid = -1;
        bool scoreUsed = false;
        bool pendingEpilogue = false;
        TaskMeta pendingMeta;
        int64_t pendingScoreOffset = 0;
        const int64_t chunkTaskNum = B_ * NT_;
        for (int64_t chunkTask = vecIdx; chunkTask < chunkTaskNum; chunkTask += usedAivNum_) {
            TaskMeta chunkMeta;
            DecodeChunkTask(chunkTask, chunkMeta);
            if (chunkMeta.valid <= 0) {
                continue;
            }
            for (int64_t h = 0; h < Hk_; ++h) {
                TaskMeta meta = chunkMeta;
                meta.h = h;
                const int64_t scoreOffset = GetScoreOffset(vecIdx, scoreSlot, 0);
                LaunchScoreTask(meta, scoreOffset, cachedScoreValid);
                scoreUsed = true;
                if (pendingEpilogue) {
                    ComputeEpilogueTask(pendingMeta, pendingScoreOffset);
                }
                WaitScoreTask();
                pendingMeta = meta;
                pendingScoreOffset = scoreOffset;
                pendingEpilogue = true;
                scoreSlot ^= 1;
            }
        }
        if (pendingEpilogue) {
            ComputeEpilogueTask(pendingMeta, pendingScoreOffset);
        }
        if (scoreUsed) {
            scoreMatmul.End();
        }
    }

    __aicore__ inline void ProcessAic()
    {
        if (!UseCatlassScore()) {
            return;
        }
        if constexpr (CHUNK_KEY == 0) {
            ProcessAicCatlassImpl<64>();
        } else if constexpr (CHUNK_KEY == 3) {
            ProcessAicCatlassImpl<128>();
        }
    }

    __aicore__ inline bool UseCatlassScore() const
    {
        if constexpr (CHUNK_KEY != 0 && CHUNK_KEY != 3) {
            return false;
        }
        return isVarlen_ == 0 && T_ > 0 && BT_ > 0 && K_ > 0 && (T_ % BT_) == 0 && (K_ % 16) == 0;
    }

    matmul::Matmul<AType, BType, CType, BiasType, CHUNK_SCALED_DOT_KKT_MM_CFG> scoreMatmul;

private:
    __aicore__ inline int64_t MinI64(int64_t lhs, int64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void DecodeChunkTask(int64_t chunkTask, TaskMeta &meta) const
    {
        meta.chunk = chunkTask % NT_;
        meta.h = 0;
        meta.b = chunkTask / NT_;
        if (isVarlen_ != 0) {
            const int64_t seqId = chunkIndicesGm.GetValue(meta.chunk * 2);
            const int64_t localChunk = chunkIndicesGm.GetValue(meta.chunk * 2 + 1);
            const int64_t bos = cuSeqlensGm.GetValue(seqId);
            const int64_t eos = cuSeqlensGm.GetValue(seqId + 1);
            meta.chunk = localChunk;
            meta.rowStart = bos + localChunk * BT_;
            meta.valid = MinI64(BT_, eos - meta.rowStart);
            meta.valid = MinI64(meta.valid, T_ - meta.rowStart);
        } else {
            meta.rowStart = meta.chunk * BT_;
            meta.valid = MinI64(BT_, T_ - meta.rowStart);
        }
        if (meta.valid < 0) {
            meta.valid = 0;
        }
    }

    __aicore__ inline void DecodeScoreTask(int64_t scoreTask, TaskMeta &meta) const
    {
        DecodeChunkTask(scoreTask / Hk_, meta);
        meta.h = scoreTask % Hk_;
    }

    __aicore__ inline int64_t GetScoreOffset(int64_t vecIdx, int64_t scoreSlot, int64_t headIdx) const
    {
        return (vecIdx * static_cast<int64_t>(SCORE_WORKSPACE_BUFFER_NUM) * SCORE_WORKSPACE_HEAD_BATCH +
                scoreSlot * static_cast<int64_t>(SCORE_WORKSPACE_HEAD_BATCH) + headIdx) *
               BT_ * BT_;
    }

    __aicore__ inline void LaunchScoreTask(const TaskMeta &meta, int64_t scoreOffset, int64_t &cachedScoreValid)
    {
        const int64_t kOffset = ((meta.b * Hk_ + meta.h) * T_ + meta.rowStart) * K_;
        if (cachedScoreValid != meta.valid) {
            if (cachedScoreValid < 0) {
                scoreMatmul.SetOrgShape(static_cast<int32_t>(BT_), static_cast<int32_t>(BT_), static_cast<int32_t>(K_));
            }
            scoreMatmul.SetSingleShape(static_cast<int32_t>(meta.valid), static_cast<int32_t>(meta.valid),
                                       static_cast<int32_t>(K_));
            scoreMatmul.SetTail(static_cast<int32_t>(meta.valid), static_cast<int32_t>(meta.valid),
                                static_cast<int32_t>(K_));
            cachedScoreValid = meta.valid;
        }
        scoreMatmul.SetTensorA(kGm[kOffset]);
        scoreMatmul.SetTensorB(kGm[kOffset], true);
        scoreMatmul.template IterateAll<false>(scoreGm[scoreOffset], 0, false, true);
    }

    __aicore__ inline void WaitScoreTask()
    {
        scoreMatmul.WaitIterateAll();
    }

    template <int32_t BT_VALUE>
    __aicore__ inline void ProcessAicCatlassImpl()
    {
        const int64_t cubeIdx = static_cast<int64_t>(GetBlockIdx());
        if (cubeIdx >= usedAicNum_ || usedAicNum_ <= 0) {
            return;
        }
        using L1TileShape = tla::Shape<tla::Int<BT_VALUE>, tla::Int<BT_VALUE>, KktInt128>;
        using L0TileShape = L1TileShape;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KktArchTag, KType, Catlass::layout::RowMajor, KType,
                                                                Catlass::layout::ColumnMajor, float,
                                                                Catlass::layout::RowMajor>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KktScoreDispatchPolicy, L1TileShape, L0TileShape, KType,
                                                              KType, float, void, TileCopy>;

        int64_t scoreSeq = 0;
        const int64_t scoreTaskNum = B_ * NT_ * Hk_;
        for (int64_t scoreTask = cubeIdx; scoreTask < scoreTaskNum; scoreTask += usedAicNum_) {
            TaskMeta meta;
            DecodeScoreTask(scoreTask, meta);
            if (meta.valid <= 0) {
                continue;
            }
            const int64_t scoreSlot = scoreSeq & 1;
            if (scoreSeq >= SCORE_WORKSPACE_BUFFER_NUM) {
                Catlass::Arch::CrossCoreWaitFlag(scoreDoneFlag_[scoreSlot]);
            }
            const int64_t scoreOffset = GetScoreOffset(cubeIdx, scoreSlot, 0);
            Catlass::Arch::Resource<KktArchTag> resource;
            BlockMmad blockMmad(resource);
            ComputeCatlassScoreTask(meta, scoreOffset, blockMmad);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(scoreReadyFlag_[scoreSlot]);
            ++scoreSeq;
        }
        const int64_t pendingSlots = MinI64(scoreSeq, static_cast<int64_t>(SCORE_WORKSPACE_BUFFER_NUM));
        for (int64_t slot = 0; slot < pendingSlots; ++slot) {
            Catlass::Arch::CrossCoreWaitFlag(scoreDoneFlag_[slot]);
        }
    }

    template <typename BlockMmad>
    __aicore__ inline void ComputeCatlassScoreTask(const TaskMeta &meta, int64_t scoreOffset, BlockMmad &blockMmad)
    {
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        const int64_t kOffset = ((meta.b * Hk_ + meta.h) * T_ + meta.rowStart) * K_;
        auto layoutA = tla::MakeLayout<KType, LayoutTagA>(BT_, K_);
        auto layoutB = tla::MakeLayout<KType, LayoutTagB>(K_, BT_);
        auto layoutC = tla::MakeLayout<float, LayoutTagC>(BT_, BT_);
        Catlass::GemmCoord shape{static_cast<uint32_t>(meta.valid), static_cast<uint32_t>(meta.valid),
                                 static_cast<uint32_t>(K_)};

        auto tensorA = tla::MakeTensor(kGm[kOffset], layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(kGm[kOffset], layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(scoreGm[scoreOffset], layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));

        blockMmad.preSetFlags();
        blockMmad(blockA, blockB, blockC, shape);
        blockMmad.finalWaitFlags();
    }

    __aicore__ inline void ProcessAivCatlass()
    {
        const int64_t subBlockNum = static_cast<int64_t>(GetSubBlockNum());
        if (subBlockNum <= 0) {
            return;
        }
        const int64_t subBlockIdx = static_cast<int64_t>(GetSubBlockIdx());
        const int64_t cubeIdx = static_cast<int64_t>(GetBlockIdx()) / subBlockNum;
        if (cubeIdx >= usedAicNum_ || usedAicNum_ <= 0) {
            return;
        }
        int64_t scoreSeq = 0;
        const int64_t scoreTaskNum = B_ * NT_ * Hk_;
        for (int64_t scoreTask = cubeIdx; scoreTask < scoreTaskNum; scoreTask += usedAicNum_) {
            TaskMeta meta;
            DecodeScoreTask(scoreTask, meta);
            if (meta.valid <= 0) {
                continue;
            }
            const int64_t scoreSlot = scoreSeq & 1;
            Catlass::Arch::CrossCoreWaitFlag(scoreReadyFlag_[scoreSlot]);
            const int64_t scoreOffset = GetScoreOffset(cubeIdx, scoreSlot, 0);
            ComputeEpilogueTaskRows(meta, scoreOffset, subBlockIdx, subBlockNum);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scoreDoneFlag_[scoreSlot]);
            ++scoreSeq;
        }
    }

    __aicore__ inline void ComputeEpilogueTask(const TaskMeta &meta, int64_t scoreBaseOffset)
    {
        const int64_t ghOffset = (meta.b * Hv_ + meta.h) * T_ + meta.rowStart;
        CopyTaskVector(gGm, ghOffset, gQueue_, meta.valid);
        CopyTaskVector(betaGm, ghOffset, betaQueue_, meta.valid);
        LocalTensor<float> gLocal = gQueue_.template DeQue<float>();
        LocalTensor<float> betaLocal = betaQueue_.template DeQue<float>();

        const int64_t outBaseOffset = ((meta.b * Hk_ + meta.h) * T_ + meta.rowStart) * BT_;
        const int64_t outRowStride = BT_;
        LocalTensor<float> scoreTileLocal = scoreTileBuf_.Get<float>();
        LocalTensor<float> outTileLocal = outTileBuf_.Get<float>();
        LocalTensor<float> gateLocal = gateBuf_.Get<float>();
        LocalTensor<float> rowBrcbLocal = rowBrcbBuf_.Get<float>();
        DuplicateZero(outTileLocal, BT_ * btAlign_);
        PipeBarrier<PIPE_V>();
        CopyScoreTile(scoreBaseOffset, scoreTileLocal, meta.valid);
        bool scoreReady = false;
        for (int64_t rowBase = 0; rowBase < meta.valid; rowBase += BRCB_ROWS) {
            const int64_t rows = MinI64(static_cast<int64_t>(BRCB_ROWS), meta.valid - rowBase);
            const int64_t cols = rowBase + rows;
            ComputeGateBlock(rowBase, rows, cols, gLocal, betaLocal, gateLocal, rowBrcbLocal);
            if (!scoreReady) {
                WaitMte2ToV();
                scoreReady = true;
            }
            for (int64_t lane = 0; lane < rows; ++lane) {
                const int64_t row = rowBase + lane;
                ComputeEpilogueRow(scoreTileLocal, outTileLocal, row, gateLocal[lane * btAlign_]);
            }
        }
        CopyOutTile(outBaseOffset, outRowStride, outTileLocal, meta.valid);

        gQueue_.FreeTensor(gLocal);
        betaQueue_.FreeTensor(betaLocal);
    }

    __aicore__ inline void ComputeEpilogueTaskRows(const TaskMeta &meta,
                                                   int64_t scoreBaseOffset,
                                                   int64_t subBlockIdx,
                                                   int64_t subBlockNum)
    {
        const int64_t ghOffset = (meta.b * Hv_ + meta.h) * T_ + meta.rowStart;
        CopyTaskVector(gGm, ghOffset, gQueue_, meta.valid);
        CopyTaskVector(betaGm, ghOffset, betaQueue_, meta.valid);
        LocalTensor<float> gLocal = gQueue_.template DeQue<float>();
        LocalTensor<float> betaLocal = betaQueue_.template DeQue<float>();

        const int64_t outBaseOffset = ((meta.b * Hk_ + meta.h) * T_ + meta.rowStart) * BT_;
        const int64_t outRowStride = BT_;
        LocalTensor<float> scoreTileLocal = scoreTileBuf_.Get<float>();
        LocalTensor<float> outTileLocal = outTileBuf_.Get<float>();
        LocalTensor<float> gateLocal = gateBuf_.Get<float>();
        LocalTensor<float> rowBrcbLocal = rowBrcbBuf_.Get<float>();
        DuplicateZero(outTileLocal, BT_ * btAlign_);
        PipeBarrier<PIPE_V>();
        CopyScoreTile(scoreBaseOffset, scoreTileLocal, meta.valid);
        bool scoreReady = false;
        const int64_t firstRowBase = subBlockIdx * static_cast<int64_t>(BRCB_ROWS);
        const int64_t rowStep = subBlockNum * static_cast<int64_t>(BRCB_ROWS);
        for (int64_t rowBase = firstRowBase; rowBase < meta.valid; rowBase += rowStep) {
            const int64_t rows = MinI64(static_cast<int64_t>(BRCB_ROWS), meta.valid - rowBase);
            const int64_t cols = rowBase + rows;
            ComputeGateBlock(rowBase, rows, cols, gLocal, betaLocal, gateLocal, rowBrcbLocal);
            if (!scoreReady) {
                WaitMte2ToV();
                scoreReady = true;
            }
            for (int64_t lane = 0; lane < rows; ++lane) {
                const int64_t row = rowBase + lane;
                ComputeEpilogueRow(scoreTileLocal, outTileLocal, row, gateLocal[lane * btAlign_]);
            }
        }
        CopyOutRowBlocks(outBaseOffset, outRowStride, outTileLocal, meta.valid, subBlockIdx, subBlockNum);

        gQueue_.FreeTensor(gLocal);
        betaQueue_.FreeTensor(betaLocal);
    }

    __aicore__ inline void WaitMte2ToV()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void WaitVToMte3()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void WaitMte3ToV()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void CopyScoreTile(int64_t scoreBaseOffset, LocalTensor<float> scoreTileLocal, int64_t valid)
    {
        DataCopyExtParams scoreParams;
        scoreParams.blockCount = static_cast<uint16_t>(valid);
        scoreParams.blockLen = static_cast<uint32_t>(BT_ * static_cast<int64_t>(sizeof(float)));
        scoreParams.srcStride = 0;
        scoreParams.dstStride = static_cast<uint32_t>((btAlign_ - BT_) * static_cast<int64_t>(sizeof(float)) /
                                                      UB_ALIGN_BYTES);
        scoreParams.rsv = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(scoreTileLocal, scoreGm[scoreBaseOffset], scoreParams, padParams);
    }

    __aicore__ inline void CopyOutTile(int64_t outBaseOffset,
                                       int64_t outRowStride,
                                       LocalTensor<float> outTileLocal,
                                       int64_t valid)
    {
        WaitVToMte3();
        DataCopyExtParams outParams;
        outParams.blockCount = static_cast<uint16_t>(valid);
        outParams.blockLen = static_cast<uint32_t>(BT_ * static_cast<int64_t>(sizeof(float)));
        outParams.srcStride = static_cast<uint32_t>((btAlign_ - BT_) * static_cast<int64_t>(sizeof(float)) /
                                                    UB_ALIGN_BYTES);
        outParams.dstStride = static_cast<uint32_t>((outRowStride - BT_) * static_cast<int64_t>(sizeof(float)));
        outParams.rsv = 0;
        DataCopyPad(aGm[outBaseOffset], outTileLocal, outParams);
        WaitMte3ToV();
    }

    __aicore__ inline void CopyOutRowBlocks(int64_t outBaseOffset,
                                            int64_t outRowStride,
                                            LocalTensor<float> outTileLocal,
                                            int64_t valid,
                                            int64_t subBlockIdx,
                                            int64_t subBlockNum)
    {
        WaitVToMte3();
        const int64_t firstRowBase = subBlockIdx * static_cast<int64_t>(BRCB_ROWS);
        const int64_t rowStep = subBlockNum * static_cast<int64_t>(BRCB_ROWS);
        for (int64_t rowBase = firstRowBase; rowBase < valid; rowBase += rowStep) {
            const int64_t rows = MinI64(static_cast<int64_t>(BRCB_ROWS), valid - rowBase);
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(rows);
            outParams.blockLen = static_cast<uint32_t>(BT_ * static_cast<int64_t>(sizeof(float)));
            outParams.srcStride = static_cast<uint32_t>((btAlign_ - BT_) * static_cast<int64_t>(sizeof(float)) /
                                                        UB_ALIGN_BYTES);
            outParams.dstStride = static_cast<uint32_t>((outRowStride - BT_) * static_cast<int64_t>(sizeof(float)));
            outParams.rsv = 0;
            DataCopyPad(aGm[outBaseOffset + rowBase * outRowStride], outTileLocal[rowBase * btAlign_], outParams);
        }
        WaitMte3ToV();
    }

    __aicore__ inline void CopyTaskVector(const GlobalTensor<float> &srcGm, int64_t gmOffset,
                                          TQue<QuePosition::VECIN, BUFFER_NUM> &queue, int64_t count)
    {
        LocalTensor<float> local = queue.template AllocTensor<float>();
        DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint16_t>(count * static_cast<int64_t>(sizeof(float)));
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(local, srcGm[gmOffset], params, {false, 0, 0, 0});
        queue.EnQue(local);
    }

    __aicore__ inline void DuplicateZero(const LocalTensor<float> &dstLocal, int64_t count)
    {
        constexpr int64_t maxDuplicateElems = FP32_REPEAT_ELEMS * 255;
        for (int64_t offset = 0; offset < count; offset += maxDuplicateElems) {
            const int64_t cur = MinI64(maxDuplicateElems, count - offset);
            Duplicate(dstLocal[offset], 0.0f, static_cast<int32_t>(cur));
        }
    }

    __aicore__ inline void ComputeGateBlock(int64_t rowBase,
                                            int64_t rows,
                                            int64_t cols,
                                            const LocalTensor<float> &gLocal,
                                            const LocalTensor<float> &betaLocal,
                                            const LocalTensor<float> &gateLocal,
                                            const LocalTensor<float> &rowBrcbLocal)
    {
        const uint8_t rowRepeatStride =
            static_cast<uint8_t>(btAlign_ / static_cast<int64_t>(FP32_BLOCK_ELEMS));
        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const int64_t cur = MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset);
            Copy(gateLocal[colOffset], gLocal[colOffset], static_cast<uint16_t>(cur),
                 static_cast<uint8_t>(rows), {1, 1, rowRepeatStride, 0});
        }
        PipeBarrier<PIPE_V>();

        Brcb(rowBrcbLocal, gLocal[rowBase], 1, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const int64_t cur = MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset);
            Sub(gateLocal[colOffset], rowBrcbLocal, gateLocal[colOffset], static_cast<uint64_t>(cur),
                static_cast<uint8_t>(rows), {1, 0, 1, rowRepeatStride, 1, rowRepeatStride});
        }
        PipeBarrier<PIPE_V>();

        for (int64_t lane = 0; lane < rows; ++lane) {
            LocalTensor<float> gateRow = gateLocal[lane * btAlign_];
            Maxs(gateRow, gateRow, -50.0f, static_cast<int32_t>(cols));
        }
        PipeBarrier<PIPE_V>();
        for (int64_t lane = 0; lane < rows; ++lane) {
            LocalTensor<float> gateRow = gateLocal[lane * btAlign_];
            Mins(gateRow, gateRow, 50.0f, static_cast<int32_t>(cols));
        }
        PipeBarrier<PIPE_V>();
        for (int64_t lane = 0; lane < rows; ++lane) {
            LocalTensor<float> gateRow = gateLocal[lane * btAlign_];
            Exp(gateRow, gateRow, static_cast<int32_t>(cols));
        }
        PipeBarrier<PIPE_V>();

        Brcb(rowBrcbLocal, betaLocal[rowBase], 1, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const int64_t cur = MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset);
            Mul(gateLocal[colOffset], gateLocal[colOffset], rowBrcbLocal, static_cast<uint64_t>(cur),
                static_cast<uint8_t>(rows), {1, 1, 0, rowRepeatStride, rowRepeatStride, 1});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeEpilogueRow(const LocalTensor<float> &scoreTileLocal,
                                              const LocalTensor<float> &outTileLocal,
                                              int64_t row,
                                              const LocalTensor<float> &gateRowLocal)
    {
        LocalTensor<float> scoreRowLocal = scoreTileLocal[row * btAlign_];
        LocalTensor<float> outRowLocal = outTileLocal[row * btAlign_];
        if (row > 0) {
            const int32_t prefix = static_cast<int32_t>(row);
            Mul(outRowLocal, scoreRowLocal, gateRowLocal, prefix);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> gQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> betaQueue_;
    TBuf<TPosition::VECCALC> scoreTileBuf_;
    TBuf<TPosition::VECCALC> outTileBuf_;
    TBuf<TPosition::VECCALC> gateBuf_;
    TBuf<TPosition::VECCALC> rowBrcbBuf_;

    GlobalTensor<KType> kGm;
    GlobalTensor<float> gGm;
    GlobalTensor<float> betaGm;
    GlobalTensor<float> aGm;
    GlobalTensor<float> scoreGm;
    GlobalTensor<int64_t> cuSeqlensGm;
    GlobalTensor<int64_t> chunkIndicesGm;
    Catlass::Arch::CrossCoreFlag scoreReadyFlag_[SCORE_WORKSPACE_BUFFER_NUM] = {SCORE_READY_FLAG0,
                                                                                SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlag scoreDoneFlag_[SCORE_WORKSPACE_BUFFER_NUM] = {SCORE_DONE_FLAG0, SCORE_DONE_FLAG1};

    int64_t B_ = 0;
    int64_t Hk_ = 0;
    int64_t Hv_ = 0;
    int64_t hvPerHk_ = 1;
    int64_t T_ = 0;
    int64_t K_ = 0;
    int64_t BT_ = 0;
    int64_t NT_ = 0;
    int64_t taskNum_ = 0;
    int64_t usedAicNum_ = 0;
    int64_t usedAivNum_ = 0;
    int64_t btAlign_ = 0;
    int64_t isVarlen_ = 0;
};
}  // namespace NsChunkScaledDotKkt

#endif  // CHUNK_SCALED_DOT_KKT_H
