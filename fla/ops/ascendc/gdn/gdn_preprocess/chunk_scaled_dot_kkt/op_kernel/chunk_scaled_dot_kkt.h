#ifndef CHUNK_SCALED_DOT_KKT_H
#define CHUNK_SCALED_DOT_KKT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

struct ChunkScaledDotKktTilingData;

namespace NsChunkScaledDotKkt {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t FP32_BLOCK_ELEMS = 8;
constexpr int32_t FP32_REPEAT_ELEMS = 64;
constexpr int32_t BRCB_ROWS = 8;
constexpr int32_t UB_ALIGN_BYTES = 32;
constexpr MatmulConfig CHUNK_SCALED_DOT_KKT_MM_CFG = GetNormalConfig(true);

using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <typename KType>
class ChunkScaledDotKkt {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KType>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KType, true, LayoutMode::NONE, false>;

    __aicore__ inline ChunkScaledDotKkt() {}

    __aicore__ inline void Init(GM_ADDR k,
                                GM_ADDR g,
                                GM_ADDR beta,
                                GM_ADDR a,
                                GM_ADDR scoreWorkspace,
                                uint64_t b,
                                uint64_t h,
                                uint64_t t,
                                uint64_t kDim,
                                uint64_t bt,
                                uint64_t nt,
                                uint64_t taskNum,
                                uint64_t usedAicNum,
                                uint64_t usedAivNum,
                                uint64_t btAlign,
                                TPipe *pipe)
    {
        pipe_ = pipe;
        B_ = static_cast<int64_t>(b);
        H_ = static_cast<int64_t>(h);
        T_ = static_cast<int64_t>(t);
        K_ = static_cast<int64_t>(kDim);
        BT_ = static_cast<int64_t>(bt);
        NT_ = static_cast<int64_t>(nt);
        taskNum_ = static_cast<int64_t>(taskNum);
        usedAicNum_ = static_cast<int64_t>(usedAicNum);
        usedAivNum_ = static_cast<int64_t>(usedAivNum);
        btAlign_ = static_cast<int64_t>(btAlign);

        kGm.SetGlobalBuffer((__gm__ KType *)k, B_ * H_ * T_ * K_);
        gGm.SetGlobalBuffer((__gm__ float *)g, B_ * H_ * T_);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, B_ * H_ * T_);
        aGm.SetGlobalBuffer((__gm__ float *)a, B_ * T_ * H_ * BT_);
        scoreGm.SetGlobalBuffer((__gm__ float *)scoreWorkspace, taskNum_ * BT_ * BT_);

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
        const int64_t vecIdx = static_cast<int64_t>(GetBlockIdx());
        if (vecIdx >= usedAivNum_ || usedAivNum_ <= 0) {
            return;
        }
        for (int64_t task = vecIdx; task < taskNum_; task += usedAivNum_) {
            ComputeScoreTask(task);
            ComputeEpilogueTask(task);
        }
    }

    matmul::Matmul<AType, BType, CType, BiasType, CHUNK_SCALED_DOT_KKT_MM_CFG> scoreMatmul;

private:
    __aicore__ inline int64_t MinI64(int64_t lhs, int64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void DecodeTask(int64_t task, int64_t &b, int64_t &h, int64_t &chunk, int64_t &rowStart,
                                      int64_t &valid) const
    {
        chunk = task % NT_;
        h = (task / NT_) % H_;
        b = task / (H_ * NT_);
        rowStart = chunk * BT_;
        valid = MinI64(BT_, T_ - rowStart);
        if (valid < 0) {
            valid = 0;
        }
    }

    __aicore__ inline void ComputeScoreTask(int64_t task)
    {
        int64_t b = 0;
        int64_t h = 0;
        int64_t chunk = 0;
        int64_t rowStart = 0;
        int64_t valid = 0;
        DecodeTask(task, b, h, chunk, rowStart, valid);
        if (valid <= 0) {
            return;
        }

        const int64_t kOffset = ((b * H_ + h) * T_ + rowStart) * K_;
        const int64_t scoreOffset = task * BT_ * BT_;
        scoreMatmul.SetOrgShape(static_cast<int32_t>(BT_), static_cast<int32_t>(BT_), static_cast<int32_t>(K_));
        scoreMatmul.SetSingleShape(static_cast<int32_t>(valid), static_cast<int32_t>(valid), static_cast<int32_t>(K_));
        scoreMatmul.SetTensorA(kGm[kOffset]);
        scoreMatmul.SetTensorB(kGm[kOffset], true);
        scoreMatmul.SetTail(static_cast<int32_t>(valid), static_cast<int32_t>(valid), static_cast<int32_t>(K_));
        scoreMatmul.template IterateAll<false>(scoreGm[scoreOffset], 0, false, true);
        scoreMatmul.WaitIterateAll();
        scoreMatmul.End();
    }

    __aicore__ inline void ComputeEpilogueTask(int64_t task)
    {
        int64_t b = 0;
        int64_t h = 0;
        int64_t chunk = 0;
        int64_t rowStart = 0;
        int64_t valid = 0;
        DecodeTask(task, b, h, chunk, rowStart, valid);
        if (valid <= 0) {
            return;
        }

        const int64_t ghOffset = (b * H_ + h) * T_ + rowStart;
        CopyTaskVector(gGm, ghOffset, gQueue_, valid);
        CopyTaskVector(betaGm, ghOffset, betaQueue_, valid);
        LocalTensor<float> gLocal = gQueue_.template DeQue<float>();
        LocalTensor<float> betaLocal = betaQueue_.template DeQue<float>();

        const int64_t scoreBaseOffset = task * BT_ * BT_;
        const int64_t outBaseOffset = ((b * H_ + h) * T_ + rowStart) * BT_;
        const int64_t outRowStride = BT_;
        LocalTensor<float> scoreTileLocal = scoreTileBuf_.Get<float>();
        LocalTensor<float> outTileLocal = outTileBuf_.Get<float>();
        LocalTensor<float> gateLocal = gateBuf_.Get<float>();
        LocalTensor<float> rowBrcbLocal = rowBrcbBuf_.Get<float>();
        CopyScoreTile(scoreBaseOffset, scoreTileLocal, valid);
        bool scoreReady = false;
        for (int64_t rowBase = 0; rowBase < valid; rowBase += BRCB_ROWS) {
            const int64_t rows = MinI64(static_cast<int64_t>(BRCB_ROWS), valid - rowBase);
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
        CopyOutTile(outBaseOffset, outRowStride, outTileLocal, valid);

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
        Duplicate(outRowLocal, 0.0f, static_cast<int32_t>(BT_));
        PipeBarrier<PIPE_V>();

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

    int64_t B_ = 0;
    int64_t H_ = 0;
    int64_t T_ = 0;
    int64_t K_ = 0;
    int64_t BT_ = 0;
    int64_t NT_ = 0;
    int64_t taskNum_ = 0;
    int64_t usedAicNum_ = 0;
    int64_t usedAivNum_ = 0;
    int64_t btAlign_ = 0;
};
}  // namespace NsChunkScaledDotKkt

#endif  // CHUNK_SCALED_DOT_KKT_H
