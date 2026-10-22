#ifndef CHUNK_SCALED_DOT_KKT_H
#define CHUNK_SCALED_DOT_KKT_H

#include "kernel_operator.h"
#include "../../common/gdn_raw_front_resources.h"
#include "../../common/front_task_order.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include <type_traits>

struct ChunkScaledDotKktTilingData;

namespace NsChunkScaledDotKkt {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t FP32_BLOCK_ELEMS = 8;
constexpr int32_t FP32_REPEAT_ELEMS = 64;
constexpr int32_t BRCB_ROWS = 32;
constexpr int32_t UB_ALIGN_BYTES = 32;
constexpr MatmulConfig CHUNK_SCALED_DOT_KKT_MM_CFG = GetNormalConfig(true);

using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

// OutputType is the CAST_RINT rounding type of the epilogue. StoreFp32SolveInput
// additionally publishes the rounded values as an FP32 GM solve input and only
// changes the GM storage type; it must not remove the low-precision rounding.
template <typename KType, typename OutputType = float, bool StoreFp32SolveInput = false,
          bool SharedKkt = false, bool kRawFront = false, int kStaticBT = 0>
class ChunkScaledDotKkt {
    static_assert(!StoreFp32SolveInput || !std::is_same_v<OutputType, float>,
                  "StoreFp32SolveInput keeps OutputType as the rounding type and needs a low-precision tile");

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
                                GdnRawFront::VectorResources *resources)
    {
        InitCommon(k, g, beta, cuSeqlens, chunkIndices, a, scoreWorkspace, nullptr, false, b, hk, hv, hvPerHk,
                   t, kDim, bt, nt, taskNum, usedAicNum, usedAivNum, btAlign, isVarlen, resources);
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
                                           GdnRawFront::VectorResources *resources)
    {
        InitCommon(k, rawG, beta, cuSeqlens, chunkIndices, a, scoreWorkspace, gCumsum, true, b, hk, hv,
                   hvPerHk, t, kDim, bt, nt, taskNum, usedAicNum, usedAivNum, btAlign, isVarlen, resources);
    }

    // Only valid with StoreFp32SolveInput and after Init/InitFusedCumsum: binds
    // the independent FP32 solve-input output. The low-precision aGm output of
    // Init is then no longer written by the epilogue.
    __aicore__ inline void InitSolveFp32Input(GM_ADDR solveFp32Input)
    {
        solveInputGm.SetGlobalBuffer((__gm__ float *)solveFp32Input,
                                     B_ * taskHeads_ * T_ * ChunkSize());
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
                                      GdnRawFront::VectorResources *resources)
    {
        resources_ = resources;
        B_ = static_cast<int64_t>(b);
        Hk_ = static_cast<int64_t>(hk);
        Hv_ = static_cast<int64_t>(hv);
        hvPerHk_ = static_cast<int64_t>(hvPerHk);
        T_ = static_cast<int64_t>(t);
        K_ = static_cast<int64_t>(kDim);
        if constexpr (kStaticBT == 0) {
            BT_Runtime_ = static_cast<int64_t>(bt);
        }
        NT_ = static_cast<int64_t>(nt);
        taskNum_ = static_cast<int64_t>(taskNum);
        // Phase6 expands the ABC work queue to value heads so that every
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
        if constexpr (kStaticBT == 0) {
            btAlign_Runtime_ = static_cast<int64_t>(btAlign);
        }
        isVarlen_ = static_cast<int64_t>(isVarlen);
        fusedCumsum_ = fusedCumsum;

        kGm.SetGlobalBuffer((__gm__ KType *)k, B_ * Hk_ * T_ * K_);
        gGm.SetGlobalBuffer((__gm__ float *)g, B_ * Hv_ * T_);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, B_ * Hv_ * T_);
        aGm.SetGlobalBuffer((__gm__ OutputType *)a, B_ * taskHeads_ * T_ * ChunkSize());
        const int64_t scoreTasks = SharedKkt ? taskNum_ / hvPerHk_ : taskNum_;
        scoreGm.SetGlobalBuffer((__gm__ float *)scoreWorkspace, scoreTasks * ChunkSize() * ChunkSize());
        if (fusedCumsum_) {
            gCumsumGm.SetGlobalBuffer((__gm__ float *)gCumsum, B_ * Hv_ * T_);
        }
        if (isVarlen_ != 0) {
            cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
            chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunkIndices, NT_ * 2);
        }

        if ASCEND_IS_AIV {
            const auto &layout = resources_->kkt;
            scoreTileBuf_.BindView(resources_, layout.score, ChunkSize() * ChunkAlign() * sizeof(float));
            outTileBuf_.BindView(resources_, layout.out, ChunkSize() * ChunkAlign() * sizeof(float));
            if constexpr (!std::is_same_v<OutputType, float>) {
                typedOutTileBuf_.BindView(resources_, layout.typed, ChunkSize() * ChunkAlign() * sizeof(OutputType));
            }
            gateBuf_.BindView(resources_, layout.gate, BRCB_ROWS * ChunkAlign() * sizeof(float));
            rowBrcbBuf_.BindView(resources_, layout.row, BRCB_ROWS * FP32_BLOCK_ELEMS * sizeof(float));
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

    __aicore__ inline int64_t DecodeTask(int64_t task, int64_t &b, int64_t &h, int64_t &chunk, int64_t &rowStart,
                                      int64_t &valid) const
    {
        int64_t legacyTask = task;
        if constexpr (kRawFront && StoreFp32SolveInput) {
            const auto parent = GdnFrontTaskOrder::Decode(task, taskHeads_, hvPerHk_, NT_);
            b = parent.batch;
            h = parent.head;
            chunk = parent.chunk;
            legacyTask = GdnFrontTaskOrder::LegacyParent(parent, taskHeads_, NT_);
        } else {
            chunk = task % NT_;
            h = (task / NT_) % taskHeads_;
            b = task / (taskHeads_ * NT_);
        }
        if (isVarlen_ != 0) {
            const int64_t seqId = chunkIndicesGm.GetValue(chunk * 2);
            const int64_t localChunk = chunkIndicesGm.GetValue(chunk * 2 + 1);
            const int64_t bos = cuSeqlensGm.GetValue(seqId);
            const int64_t eos = cuSeqlensGm.GetValue(seqId + 1);
            chunk = localChunk;
            rowStart = bos + localChunk * ChunkSize();
            valid = MinI64(ChunkSize(), eos - rowStart);
            valid = MinI64(valid, T_ - rowStart);
        } else {
            rowStart = chunk * ChunkSize();
            valid = MinI64(ChunkSize(), T_ - rowStart);
        }
        if (valid < 0) {
            valid = 0;
        }
        return legacyTask;
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
        const int64_t scoreOffset = task * ChunkSize() * ChunkSize();
        scoreMatmul.SetOrgShape(static_cast<int32_t>(ChunkSize()), static_cast<int32_t>(ChunkSize()), static_cast<int32_t>(K_));
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
        const int64_t legacyTask = DecodeTask(task, b, h, chunk, rowStart, valid);
        if (valid <= 0) {
            return;
        }

        const int64_t ghOffset = (b * Hv_ + h) * T_ + rowStart;
        LocalTensor<float> gLocal, betaLocal;
        if (fusedCumsum_) {
            ComputePrefixCumsumFromGm(ghOffset, valid);
        }
        // 前 task 的 MTE3_V wait 位于此 V_MTE2 set 之前；释放给下一 MTE2 写者。
        auto inputFree = resources_->template Event<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(inputFree);
        WaitFlag<HardEvent::V_MTE2>(inputFree);
        gLocal = resources_->template View<float>(resources_->kkt.g, ChunkAlign() * sizeof(float));
        betaLocal = resources_->template View<float>(resources_->kkt.beta, ChunkAlign() * sizeof(float));
        if (fusedCumsum_) {
            CopyTaskVectorRaw(gCumsumGm, ghOffset, gLocal, valid);
        } else {
            CopyTaskVectorRaw(gGm, ghOffset, gLocal, valid);
        }
        CopyTaskVectorRaw(betaGm, ghOffset, betaLocal, valid);
        // g/beta 同一 MTE2 前缀先就绪；后面的 score DMA 仍与 GateBlock 重叠。
        WaitMte2ToV();

        int64_t scoreTask = legacyTask;
        if constexpr (SharedKkt) {
            // score GM 仍按旧 head-major 编号；只压缩共享 Hk 轴。
            scoreTask -= (b * (taskHeads_ - Hk_) + h - h / hvPerHk_) * NT_;
        }
        const int64_t scoreBaseOffset = scoreTask * ChunkSize() * ChunkSize();
        const int64_t outBaseOffset = ((b * taskHeads_ + h) * T_ + rowStart) * ChunkSize();
        const int64_t outRowStride = ChunkSize();
        LocalTensor<float> scoreTileLocal = scoreTileBuf_.template Get<float>();
        LocalTensor<float> outTileLocal = outTileBuf_.template Get<float>();
        LocalTensor<float> gateLocal = gateBuf_.template Get<float>();
        LocalTensor<float> rowBrcbLocal = rowBrcbBuf_.template Get<float>();
        CopyScoreTile(scoreBaseOffset, scoreTileLocal, valid);
        bool scoreReady = false;
        for (int64_t rowBase = 0; rowBase < valid; rowBase += BRCB_ROWS) {
            const int64_t rows = MinI64(static_cast<int64_t>(BRCB_ROWS), valid - rowBase);
            const int64_t cols = rowBase + rows;
            Duplicate(outTileLocal[rowBase * ChunkAlign()], 0.0f,
                      static_cast<int32_t>(rows * ChunkAlign()));
            PipeBarrier<PIPE_V>();
            ComputeGateBlock(rowBase, rows, cols, gLocal, betaLocal, gateLocal, rowBrcbLocal);
            if (!scoreReady) {
                WaitMte2ToV(1);
                scoreReady = true;
            }
            for (int64_t lane = 0; lane < rows; ++lane) {
                const int64_t row = rowBase + lane;
                ComputeEpilogueRow(scoreTileLocal, outTileLocal, row, gateLocal[lane * ChunkAlign()]);
            }
            // Each row writes a disjoint output slice. Drain once before the
            // next block reuses gateLocal or CopyOutTile reads the output.
            if (cols > 1) {
                PipeBarrier<PIPE_V>();
            }
        }
        CopyOutTile(outBaseOffset, outRowStride, outTileLocal, valid);


    }

    template <HardEvent Direction> __aicore__ inline event_t Event(uint32_t slot = 0)
    {
        return resources_->template Event<Direction>(slot);
    }

    __aicore__ inline void CopyTaskVectorRaw(const GlobalTensor<float> &srcGm, int64_t gmOffset,
                                            LocalTensor<float> local, int64_t count)
    {
        DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint16_t>(count * static_cast<int64_t>(sizeof(float)));
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(local, srcGm[gmOffset], params, {false, 0, 0, 0});
    }

    __aicore__ inline void WaitMte2ToV(uint32_t slot = 0)
    {
        event_t eventId = Event<HardEvent::MTE2_V>(slot);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void WaitVToMte3()
    {
        event_t eventId = Event<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void WaitMte3ToV()
    {
        event_t eventId = Event<HardEvent::MTE3_V>();
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void CopyScoreTile(int64_t scoreBaseOffset, LocalTensor<float> scoreTileLocal, int64_t valid)
    {
        DataCopyExtParams scoreParams;
        scoreParams.blockCount = static_cast<uint16_t>(valid);
        scoreParams.blockLen = static_cast<uint32_t>(ChunkSize() * static_cast<int64_t>(sizeof(float)));
        scoreParams.srcStride = 0;
        scoreParams.dstStride = static_cast<uint32_t>((ChunkAlign() - ChunkSize()) * static_cast<int64_t>(sizeof(float)) /
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
        if constexpr (StoreFp32SolveInput) {
            LocalTensor<OutputType> typedOut = typedOutTileBuf_.template Get<OutputType>();
            Cast(typedOut, outTileLocal, RoundMode::CAST_RINT,
                 static_cast<uint32_t>(valid * ChunkAlign()));
            PipeBarrier<PIPE_V>();
            // The first cast has finished reading outTileLocal, so the second
            // cast may overwrite the same FP32 slots from typedOut.
            Cast(outTileLocal, typedOut, RoundMode::CAST_NONE,
                 static_cast<uint32_t>(valid * ChunkAlign()));
            PipeBarrier<PIPE_V>();
            CopyFp32OutTile(outBaseOffset, outRowStride, outTileLocal, valid);
        } else if constexpr (!std::is_same_v<OutputType, float>) {
            LocalTensor<OutputType> typedOut = typedOutTileBuf_.template Get<OutputType>();
            Cast(typedOut, outTileLocal, RoundMode::CAST_RINT,
                 static_cast<uint32_t>(valid * ChunkAlign()));
            PipeBarrier<PIPE_V>();
            CopyTypedOutTile(outBaseOffset, outRowStride, typedOut, valid);
        } else {
            WaitVToMte3();
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(valid);
            outParams.blockLen = static_cast<uint32_t>(ChunkSize() * static_cast<int64_t>(sizeof(float)));
            outParams.srcStride = static_cast<uint32_t>((ChunkAlign() - ChunkSize()) * static_cast<int64_t>(sizeof(float)) /
                                                        UB_ALIGN_BYTES);
            outParams.dstStride = static_cast<uint32_t>((outRowStride - ChunkSize()) * static_cast<int64_t>(sizeof(float)));
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
        const int64_t aicIdx = static_cast<int64_t>(GetBlockIdx()) / subBlockNum;
        const int64_t begin = aicIdx * tilesPerAic;
        const int64_t end = MinI64(begin + tilesPerAic, taskNum_);
        ProcessEpilogueRange(begin, end);
    }

    // begin/end是全局parent区间；仅缩小任务范围，沿用同一后处理及舍入实现。
    __aicore__ inline void ProcessEpilogueRange(int64_t begin, int64_t end)
    {
        const int64_t subBlockNum = static_cast<int64_t>(GetSubBlockNum());
        const int64_t subBlockIdx = static_cast<int64_t>(GetSubBlockIdx());
        for (int64_t task = begin + subBlockIdx; task < end; task += subBlockNum) {
            ComputeEpilogueTask(task);
        }
    }

private:
    __aicore__ inline void CopyTypedOutTile(int64_t outBaseOffset,
                                            int64_t outRowStride,
                                            LocalTensor<OutputType> outTileLocal,
                                            int64_t valid)
    {
        WaitVToMte3();
        DataCopyExtParams outParams;
        outParams.blockCount = static_cast<uint16_t>(valid);
        outParams.blockLen = static_cast<uint32_t>(ChunkSize() * static_cast<int64_t>(sizeof(OutputType)));
        outParams.srcStride = static_cast<uint32_t>((ChunkAlign() - ChunkSize()) * static_cast<int64_t>(sizeof(OutputType)) /
                                                    UB_ALIGN_BYTES);
        outParams.dstStride = static_cast<uint32_t>((outRowStride - ChunkSize()) * static_cast<int64_t>(sizeof(OutputType)));
        outParams.rsv = 0;
        DataCopyPad(aGm[outBaseOffset], outTileLocal, outParams);
        WaitMte3ToV();
    }

    __aicore__ inline void CopyFp32OutTile(int64_t outBaseOffset,
                                           int64_t outRowStride,
                                           LocalTensor<float> outTileLocal,
                                           int64_t valid)
    {
        WaitVToMte3();
        DataCopyExtParams outParams;
        outParams.blockCount = static_cast<uint16_t>(valid);
        outParams.blockLen = static_cast<uint32_t>(ChunkSize() * static_cast<int64_t>(sizeof(float)));
        outParams.srcStride = static_cast<uint32_t>((ChunkAlign() - ChunkSize()) * static_cast<int64_t>(sizeof(float)) /
                                                    UB_ALIGN_BYTES);
        outParams.dstStride = static_cast<uint32_t>((outRowStride - ChunkSize()) * static_cast<int64_t>(sizeof(float)));
        outParams.rsv = 0;
        DataCopyPad(solveInputGm[outBaseOffset], outTileLocal, outParams);
        WaitMte3ToV();
    }

    __aicore__ inline void ComputePrefixCumsumFromGm(int64_t gmOffset, int64_t count)
    {
        // Match ChunkLocalCumsum's sequential FP32 vector-add order. The temporary
        // scalars live at separate 32-byte-aligned UB addresses; the public FP32
        // output is then reloaded as the KKT compute view.
        LocalTensor<float> accLocal = rowBrcbBuf_.template Get<float>();
        LocalTensor<float> inputPing = rowBrcbBuf_.template Get<float>()[FP32_BLOCK_ELEMS];
        LocalTensor<float> inputPong = rowBrcbBuf_.template Get<float>()[2 * FP32_BLOCK_ELEMS];
        LocalTensor<float> outputPing = rowBrcbBuf_.template Get<float>()[3 * FP32_BLOCK_ELEMS];
        LocalTensor<float> outputPong = rowBrcbBuf_.template Get<float>()[4 * FP32_BLOCK_ELEMS];
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        event_t vToMte2Ping = Event<HardEvent::V_MTE2>();
        event_t vToMte2Pong = Event<HardEvent::V_MTE2>(1);
        event_t mte2ToVPing = Event<HardEvent::MTE2_V>();
        event_t mte2ToVPong = Event<HardEvent::MTE2_V>(1);
        event_t vToMte3Ping = Event<HardEvent::V_MTE3>();
        event_t vToMte3Pong = Event<HardEvent::V_MTE3>(1);
        event_t mte3ToVPing = Event<HardEvent::MTE3_V>();
        event_t mte3ToVPong = Event<HardEvent::MTE3_V>(1);
        event_t mte3ToMte2Event = Event<HardEvent::MTE3_MTE2>();
        bool outputPingActive = false;
        bool outputPongActive = false;

        // Close the previous task's gate-vector use of the shared buffer, then keep
        // each input slot unavailable to MTE2 until its preceding vector read ends.
        SetFlag<HardEvent::V_MTE2>(vToMte2Ping);

        WaitFlag<HardEvent::V_MTE2>(vToMte2Ping);

        // Raw 的 ping pair 已释放整块旧 row；pong 仅由下方 count>1 分支 seed 一次。
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
            static_cast<uint8_t>(ChunkAlign() / static_cast<int64_t>(FP32_BLOCK_ELEMS));
        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const int64_t cur = MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset);
            Copy(gateLocal[colOffset], gLocal[colOffset], static_cast<uint16_t>(cur),
                 static_cast<uint8_t>(rows), {1, 1, rowRepeatStride, 0});
        }
        PipeBarrier<PIPE_V>();

        Brcb(rowBrcbLocal, gLocal[rowBase], static_cast<uint8_t>(CeilDiv(rows, 8)),
             {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const int64_t cur = MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset);
            Sub(gateLocal[colOffset], rowBrcbLocal, gateLocal[colOffset], static_cast<uint64_t>(cur),
                static_cast<uint8_t>(rows), {1, 0, 1, rowRepeatStride, 1, rowRepeatStride});
        }
        PipeBarrier<PIPE_V>();

        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const uint64_t mask = static_cast<uint64_t>(
                MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset));
            Maxs(gateLocal[colOffset], gateLocal[colOffset], -50.0f, mask,
                 static_cast<uint8_t>(rows), {1, 1, rowRepeatStride, rowRepeatStride});
        }
        PipeBarrier<PIPE_V>();
        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const uint64_t mask = static_cast<uint64_t>(
                MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset));
            Mins(gateLocal[colOffset], gateLocal[colOffset], 50.0f, mask,
                 static_cast<uint8_t>(rows), {1, 1, rowRepeatStride, rowRepeatStride});
        }
        PipeBarrier<PIPE_V>();
        for (int64_t colOffset = 0; colOffset < cols; colOffset += FP32_REPEAT_ELEMS) {
            const uint64_t mask = static_cast<uint64_t>(
                MinI64(static_cast<int64_t>(FP32_REPEAT_ELEMS), cols - colOffset));
            Exp(gateLocal[colOffset], gateLocal[colOffset], mask,
                 static_cast<uint8_t>(rows), {1, 1, rowRepeatStride, rowRepeatStride});
        }
        PipeBarrier<PIPE_V>();

        Brcb(rowBrcbLocal, betaLocal[rowBase], static_cast<uint8_t>(CeilDiv(rows, 8)),
             {1, FP32_BLOCK_ELEMS});
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
        LocalTensor<float> scoreRowLocal = scoreTileLocal[row * ChunkAlign()];
        LocalTensor<float> outRowLocal = outTileLocal[row * ChunkAlign()];
        if (row > 0) {
            const int32_t prefix = static_cast<int32_t>(row);
            Mul(outRowLocal, scoreRowLocal, gateRowLocal, prefix);
        }
    }

private:
    GdnRawFront::VectorResources *resources_ = nullptr;
    GdnRawFront::Buffer<kRawFront> scoreTileBuf_;
    GdnRawFront::Buffer<kRawFront> outTileBuf_;
    GdnRawFront::Buffer<kRawFront> typedOutTileBuf_;
    GdnRawFront::Buffer<kRawFront> gateBuf_;
    GdnRawFront::Buffer<kRawFront> rowBrcbBuf_;

    GlobalTensor<KType> kGm;
    GlobalTensor<float> gGm;
    GlobalTensor<float> betaGm;
    GlobalTensor<OutputType> aGm;
    GlobalTensor<float> solveInputGm;
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
    __aicore__ inline int64_t ChunkSize() const
    {
        if constexpr (kStaticBT != 0) {
            return static_cast<int64_t>(kStaticBT);
        } else {
            return BT_Runtime_;
        }
    }

    int64_t BT_Runtime_ = 0;
    int64_t NT_ = 0;
    int64_t taskNum_ = 0;
    int64_t usedAicNum_ = 0;
    int64_t usedAivNum_ = 0;
    __aicore__ inline int64_t ChunkAlign() const
    {
        if constexpr (kStaticBT != 0) {
            return static_cast<int64_t>(kStaticBT);
        } else {
            return btAlign_Runtime_;
        }
    }

    int64_t btAlign_Runtime_ = 0;
    int64_t isVarlen_ = 0;
    bool fusedCumsum_ = false;
};
}  // namespace NsChunkScaledDotKkt

#endif  // CHUNK_SCALED_DOT_KKT_H
