#ifndef CHUNK_GATED_DELTA_RULE_CUMSUM_KKT_H
#define CHUNK_GATED_DELTA_RULE_CUMSUM_KKT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include <type_traits>

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "chunk_gated_delta_rule_kkt_vector.h"
#endif

struct ChunkScaledDotKktTilingData;

// Phase6-private cumsum/KKT component. The public standalone
// ChunkScaledDotKkt implementation remains in its original operator directory.
namespace NsChunkScaledDotKktFusedCumsum {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t FP32_BLOCK_ELEMS = 8;
constexpr int32_t FP32_REPEAT_ELEMS = 64;
constexpr int32_t BRCB_ROWS = 8;
constexpr int32_t UB_ALIGN_BYTES = 32;
constexpr MatmulConfig CHUNK_SCALED_DOT_KKT_MM_CFG = GetNormalConfig(true);

using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <typename KType, typename OutputType = float>
class ChunkScaledDotKktFusedCumsum {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KType>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KType, true, LayoutMode::NONE, false>;

    __aicore__ inline ChunkScaledDotKktFusedCumsum() {}

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
        InitCommon(k, g, beta, cuSeqlens, chunkIndices, a, scoreWorkspace, nullptr, false, b, hk, hv, hvPerHk,
                   t, kDim, bt, nt, taskNum, usedAicNum, usedAivNum, btAlign, isVarlen, pipe);
    }

    __aicore__ inline void InitFusedCumsum(GM_ADDR k,
                                           GM_ADDR rawG,
                                           GM_ADDR beta,
                                           GM_ADDR cuSeqlens,
                                           GM_ADDR chunkIndices,
                                           GM_ADDR gCumsum,
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
        InitCommon(k, rawG, beta, cuSeqlens, chunkIndices, a, scoreWorkspace, gCumsum, true, b, hk, hv,
                   hvPerHk, t, kDim, bt, nt, taskNum, usedAicNum, usedAivNum, btAlign, isVarlen, pipe);
    }

private:
    __aicore__ inline void InitCommon(GM_ADDR k,
                                      GM_ADDR g,
                                      GM_ADDR beta,
                                      GM_ADDR cuSeqlens,
                                      GM_ADDR chunkIndices,
                                      GM_ADDR a,
                                      GM_ADDR scoreWorkspace,
                                      GM_ADDR gCumsum,
                                      bool fusedCumsum,
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
        // Phase6 expands the coefficient-generation queue to value heads so that every
        // [B, Hv, T, BT] A row is produced. Standalone KKT keeps its legacy
        // Hk task queue; infer the active task-head axis from taskNum.
        const int64_t taskDenom = B_ * NT_;
        taskHeads_ = (taskDenom > 0 && taskNum_ % taskDenom == 0)
                         ? taskNum_ / taskDenom : Hk_;
        if (taskHeads_ <= 0) {
            taskHeads_ = Hk_;
        }
        usedAicNum_ = static_cast<int64_t>(usedAicNum);
        usedAivNum_ = static_cast<int64_t>(usedAivNum);
        btAlign_ = static_cast<int64_t>(btAlign);
        isVarlen_ = static_cast<int64_t>(isVarlen);
        fusedCumsum_ = fusedCumsum;

        kGm.SetGlobalBuffer((__gm__ KType *)k, B_ * Hk_ * T_ * K_);
        gGm.SetGlobalBuffer((__gm__ float *)g, B_ * Hv_ * T_);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, B_ * Hv_ * T_);
        aGm.SetGlobalBuffer((__gm__ OutputType *)a, B_ * taskHeads_ * T_ * BT_);
        scoreGm.SetGlobalBuffer((__gm__ float *)scoreWorkspace, taskNum_ * BT_ * BT_);
        if (fusedCumsum_) {
            gCumsumGm.SetGlobalBuffer((__gm__ float *)gCumsum, B_ * Hv_ * T_);
        }
        if (isVarlen_ != 0) {
            cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
            chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunkIndices, NT_ * 2);
        }

        if ASCEND_IS_AIV {
            pipe_->InitBuffer(gQueue_, BUFFER_NUM, btAlign_ * sizeof(float));
            pipe_->InitBuffer(betaQueue_, BUFFER_NUM, btAlign_ * sizeof(float));
            pipe_->InitBuffer(scoreTileBuf_, static_cast<uint32_t>(BT_ * btAlign_ * sizeof(float)));
            pipe_->InitBuffer(outTileBuf_, static_cast<uint32_t>(BT_ * btAlign_ * sizeof(float)));
            if constexpr (!std::is_same_v<OutputType, float>) {
                pipe_->InitBuffer(typedOutTileBuf_, static_cast<uint32_t>(BT_ * btAlign_ * sizeof(OutputType)));
            }
            pipe_->InitBuffer(gateBuf_, BRCB_ROWS * btAlign_ * sizeof(float));
            pipe_->InitBuffer(rowBrcbBuf_, BRCB_ROWS * FP32_BLOCK_ELEMS * sizeof(float));
        }
    }

public:
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
        h = (task / NT_) % taskHeads_;
        b = task / (taskHeads_ * NT_);
        if (isVarlen_ != 0) {
            const int64_t seqId = chunkIndicesGm.GetValue(chunk * 2);
            const int64_t localChunk = chunkIndicesGm.GetValue(chunk * 2 + 1);
            const int64_t bos = cuSeqlensGm.GetValue(seqId);
            const int64_t eos = cuSeqlensGm.GetValue(seqId + 1);
            chunk = localChunk;
            rowStart = bos + localChunk * BT_;
            valid = MinI64(BT_, eos - rowStart);
            valid = MinI64(valid, T_ - rowStart);
        } else {
            rowStart = chunk * BT_;
            valid = MinI64(BT_, T_ - rowStart);
        }
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

        const int64_t hk = (taskHeads_ == Hv_ && hvPerHk_ > 0) ? h / hvPerHk_ : h;
        const int64_t kOffset = ((b * Hk_ + hk) * T_ + rowStart) * K_;
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

        const int64_t ghOffset = (b * Hv_ + h) * T_ + rowStart;
        if (fusedCumsum_) {
            ComputePrefixCumsumFromGm(ghOffset, valid);
            CopyTaskVector(gCumsumGm, ghOffset, gQueue_, valid);
        } else {
            CopyTaskVector(gGm, ghOffset, gQueue_, valid);
        }
        CopyTaskVector(betaGm, ghOffset, betaQueue_, valid);
        LocalTensor<float> gLocal = gQueue_.template DeQue<float>();
        LocalTensor<float> betaLocal = betaQueue_.template DeQue<float>();

        const int64_t scoreBaseOffset = task * BT_ * BT_;
        const int64_t outBaseOffset = ((b * taskHeads_ + h) * T_ + rowStart) * BT_;
        const int64_t outRowStride = BT_;
        LocalTensor<float> scoreTileLocal = scoreTileBuf_.Get<float>();
        LocalTensor<float> outTileLocal = outTileBuf_.Get<float>();
        LocalTensor<float> gateLocal = gateBuf_.Get<float>();
        LocalTensor<float> rowBrcbLocal = rowBrcbBuf_.Get<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (BT_ == static_cast<int64_t>(NsChunkGatedDeltaRuleKkt::KKT_A5_BT64) &&
            btAlign_ == static_cast<int64_t>(NsChunkGatedDeltaRuleKkt::KKT_A5_BT64) &&
            valid == static_cast<int64_t>(NsChunkGatedDeltaRuleKkt::KKT_A5_BT64)) {
            CopyScoreTile(scoreBaseOffset, scoreTileLocal, valid);
            WaitMte2ToV();
            auto scoreAddr = reinterpret_cast<uint64_t>(scoreTileLocal.GetPhyAddr());
            auto outAddr = reinterpret_cast<uint64_t>(outTileLocal.GetPhyAddr());
            auto gAddr = reinterpret_cast<uint64_t>(gLocal.GetPhyAddr());
            auto betaAddr = reinterpret_cast<uint64_t>(betaLocal.GetPhyAddr());
            asc_vf_call<NsChunkGatedDeltaRuleKkt::ProcessKktEpilogue64VF>(
                (__ubuf__ float *)outAddr, (__ubuf__ float *)scoreAddr,
                (__ubuf__ float *)gAddr, (__ubuf__ float *)betaAddr,
                0, static_cast<uint16_t>(valid), static_cast<uint16_t>(valid),
                0, 1, static_cast<uint16_t>(BRCB_ROWS));
            CopyOutTile(outBaseOffset, outRowStride, outTileLocal, valid);
            gQueue_.FreeTensor(gLocal);
            betaQueue_.FreeTensor(betaLocal);
            return;
        }
#endif
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
        if constexpr (!std::is_same_v<OutputType, float>) {
            LocalTensor<OutputType> typedOut = typedOutTileBuf_.Get<OutputType>();
            Cast(typedOut, outTileLocal, RoundMode::CAST_RINT,
                 static_cast<uint32_t>(valid * btAlign_));
            PipeBarrier<PIPE_V>();
            CopyTypedOutTile(outBaseOffset, outRowStride, typedOut, valid);
        } else {
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
    }

public:
    __aicore__ inline void ProcessScoreForSolve(int64_t tilesPerAic)
    {
        const int64_t aicIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t begin = aicIdx * tilesPerAic;
        const int64_t end = MinI64(begin + tilesPerAic, taskNum_);
        for (int64_t task = begin; task < end; ++task) {
            ComputeScoreTask(task);
        }
    }

    __aicore__ inline void ProcessAivForSolve(int64_t tilesPerAic)
    {
        const int64_t subBlockNum = static_cast<int64_t>(GetSubBlockNum());
        const int64_t subBlockIdx = static_cast<int64_t>(GetSubBlockIdx());
        const int64_t aicIdx = static_cast<int64_t>(GetBlockIdx()) / subBlockNum;
        const int64_t begin = aicIdx * tilesPerAic;
        const int64_t end = MinI64(begin + tilesPerAic, taskNum_);
        for (int64_t task = begin + subBlockIdx; task < end; task += subBlockNum) {
            ComputeScoreTask(task);
            ComputeEpilogueTask(task);
        }
    }

    __aicore__ inline void ProcessEpilogueForSolve(int64_t tilesPerAic)
    {
        const int64_t subBlockNum = static_cast<int64_t>(GetSubBlockNum());
        const int64_t subBlockIdx = static_cast<int64_t>(GetSubBlockIdx());
        const int64_t aicIdx = static_cast<int64_t>(GetBlockIdx()) / subBlockNum;
        const int64_t begin = aicIdx * tilesPerAic;
        const int64_t end = MinI64(begin + tilesPerAic, taskNum_);
        for (int64_t task = begin + subBlockIdx; task < end; task += subBlockNum) {
            ComputeEpilogueTask(task);
        }
    }

private:
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

    __aicore__ inline void CopyTypedOutTile(int64_t outBaseOffset,
                                            int64_t outRowStride,
                                            LocalTensor<OutputType> outTileLocal,
                                            int64_t valid)
    {
        WaitVToMte3();
        DataCopyExtParams outParams;
        outParams.blockCount = static_cast<uint16_t>(valid);
        outParams.blockLen = static_cast<uint32_t>(BT_ * static_cast<int64_t>(sizeof(OutputType)));
        outParams.srcStride = static_cast<uint32_t>((btAlign_ - BT_) * static_cast<int64_t>(sizeof(OutputType)) /
                                                    UB_ALIGN_BYTES);
        outParams.dstStride = static_cast<uint32_t>((outRowStride - BT_) * static_cast<int64_t>(sizeof(OutputType)));
        outParams.rsv = 0;
        DataCopyPad(aGm[outBaseOffset], outTileLocal, outParams);
        WaitMte3ToV();
    }

    __aicore__ inline void ComputePrefixCumsumFromGm(int64_t gmOffset, int64_t count)
    {
        // Match ChunkLocalCumsum's sequential FP32 vector-add order. The temporary
        // scalars live at separate 32-byte-aligned UB addresses; the public FP32
        // output is then reloaded as the KKT compute view.
        LocalTensor<float> accLocal = rowBrcbBuf_.Get<float>();
        LocalTensor<float> inputPing = rowBrcbBuf_.Get<float>()[FP32_BLOCK_ELEMS];
        LocalTensor<float> inputPong = rowBrcbBuf_.Get<float>()[2 * FP32_BLOCK_ELEMS];
        LocalTensor<float> outputPing = rowBrcbBuf_.Get<float>()[3 * FP32_BLOCK_ELEMS];
        LocalTensor<float> outputPong = rowBrcbBuf_.Get<float>()[4 * FP32_BLOCK_ELEMS];
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        event_t vToMte2Ping = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t vToMte2Pong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t mte2ToVPing = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t mte2ToVPong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t vToMte3Ping = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t vToMte3Pong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t mte3ToVPing = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        event_t mte3ToVPong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        event_t mte3ToMte2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        bool outputPingActive = false;
        bool outputPongActive = false;

        // Close the previous task's gate-vector use of the shared buffer, then keep
        // each input slot unavailable to MTE2 until its preceding vector read ends.
        SetFlag<HardEvent::V_MTE2>(vToMte2Ping);
        SetFlag<HardEvent::V_MTE2>(vToMte2Pong);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Ping);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Pong);
        DataCopyPad(inputPing, gGm[gmOffset], params, padParams);
        SetFlag<HardEvent::MTE2_V>(mte2ToVPing);
        if (count > 1) {
            SetFlag<HardEvent::V_MTE2>(vToMte2Pong);
        }
        for (int64_t row = 0; row < count; ++row) {
            const bool usePing = (row & 1) == 0;
            LocalTensor<float> inputLocal = usePing ? inputPing : inputPong;
            event_t currentMte2ToV = usePing ? mte2ToVPing : mte2ToVPong;
            WaitFlag<HardEvent::MTE2_V>(currentMte2ToV);

            if (row + 1 < count) {
                const bool nextUsesPing = !usePing;
                LocalTensor<float> nextInput = nextUsesPing ? inputPing : inputPong;
                event_t nextVToMte2 = nextUsesPing ? vToMte2Ping : vToMte2Pong;
                event_t nextMte2ToV = nextUsesPing ? mte2ToVPing : mte2ToVPong;
                WaitFlag<HardEvent::V_MTE2>(nextVToMte2);
                DataCopyPad(nextInput, gGm[gmOffset + row + 1], params, padParams);
                SetFlag<HardEvent::MTE2_V>(nextMte2ToV);
            }
            if (row == 0) {
                Adds(accLocal, inputLocal, 0.0f, 1);
            } else {
                Add(accLocal, accLocal, inputLocal, 1);
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<float> outputLocal = usePing ? outputPing : outputPong;
            event_t currentMte3ToV = usePing ? mte3ToVPing : mte3ToVPong;
            if ((usePing && outputPingActive) || (!usePing && outputPongActive)) {
                WaitFlag<HardEvent::MTE3_V>(currentMte3ToV);
            }
            Copy(outputLocal, accLocal, 1, 1, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            if (row + 2 < count) {
                event_t currentVToMte2 = usePing ? vToMte2Ping : vToMte2Pong;
                SetFlag<HardEvent::V_MTE2>(currentVToMte2);
            }
            event_t currentVToMte3 = usePing ? vToMte3Ping : vToMte3Pong;
            SetFlag<HardEvent::V_MTE3>(currentVToMte3);
            WaitFlag<HardEvent::V_MTE3>(currentVToMte3);
            DataCopyPad(gCumsumGm[gmOffset + row], outputLocal, params);
            SetFlag<HardEvent::MTE3_V>(currentMte3ToV);
            outputPingActive = outputPingActive || usePing;
            outputPongActive = outputPongActive || !usePing;
            if (row + 1 == count) {
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
            }
        }
        if (outputPingActive) {
            WaitFlag<HardEvent::MTE3_V>(mte3ToVPing);
        }
        if (outputPongActive) {
            WaitFlag<HardEvent::MTE3_V>(mte3ToVPong);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
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
    TBuf<TPosition::VECCALC> typedOutTileBuf_;
    TBuf<TPosition::VECCALC> gateBuf_;
    TBuf<TPosition::VECCALC> rowBrcbBuf_;

    GlobalTensor<KType> kGm;
    GlobalTensor<float> gGm;
    GlobalTensor<float> betaGm;
    GlobalTensor<OutputType> aGm;
    GlobalTensor<float> scoreGm;
    GlobalTensor<float> gCumsumGm;
    GlobalTensor<int64_t> cuSeqlensGm;
    GlobalTensor<int64_t> chunkIndicesGm;

    int64_t B_ = 0;
    int64_t Hk_ = 0;
    int64_t Hv_ = 0;
    int64_t hvPerHk_ = 1;
    int64_t taskHeads_ = 0;
    int64_t T_ = 0;
    int64_t K_ = 0;
    int64_t BT_ = 0;
    int64_t NT_ = 0;
    int64_t taskNum_ = 0;
    int64_t usedAicNum_ = 0;
    int64_t usedAivNum_ = 0;
    int64_t btAlign_ = 0;
    int64_t isVarlen_ = 0;
    bool fusedCumsum_ = false;
};
}  // namespace NsChunkScaledDotKktFusedCumsum

#endif  // CHUNK_GATED_DELTA_RULE_CUMSUM_KKT_H
