/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_CUMSUM_PREPARE_HPP
#define GDN_CUMSUM_PREPARE_HPP

#include <cstdint>
#include "kernel_operator.h"

namespace GdnCumsumPrepare {

using namespace AscendC;

// Stage P walks the value-head axis in bounded tiles.  The local row stride
// is fixed for the worker to the aligned width of the first tile, so a final
// short tile can use right padding plus a UB stride without changing the
// Gather offset table.  The host route still limits the architecture, K/V
// dimensions and chunk size; Hv is a runtime GVA dimension.
constexpr uint64_t kHeadTile = 64;
constexpr uint64_t kHeadAlign = 8;
constexpr uint64_t kChunk64 = 64;
constexpr uint64_t kChunk128 = 128;
constexpr uint32_t kUbAlignmentBytes = 32;
constexpr uint32_t kFp32BlockElements = kUbAlignmentBytes / sizeof(float);

struct PrepareArgs {
    GM_ADDR rawG;          // [B, T, Hv], FP32
    GM_ADDR gCumsumBht;    // [B, Hv, T], FP32 workspace
    GM_ADDR gCumsumBth;    // [B, T, Hv], FP32 optional public output
    GM_ADDR cuSeqlens;     // varlen only, [sequence_count + 1], int64
    GM_ADDR chunkIndices;  // varlen only, [num_chunks, 2], int64
    uint64_t batch;
    uint64_t heads;
    uint64_t tokens;
    uint64_t chunkSize;
    uint64_t numChunks;
    uint64_t taskNum;
    uint64_t isVarlen;
    uint64_t outputG;
};

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t AlignHeadU64(uint64_t value)
{
    // kHeadAlign is a power of two, so alignment is expressed with a mask.
    return (value + kHeadAlign - 1) & ~(kHeadAlign - 1);
}

__aicore__ inline bool IsSupported(const PrepareArgs &args)
{
    constexpr uint64_t kMaxStrideHeadCount = 0xffffffffULL / sizeof(float) + 1;
    return args.rawG != nullptr && args.gCumsumBht != nullptr && args.batch > 0 &&
           args.heads > 0 && args.tokens > 0 &&
           args.heads <= kMaxStrideHeadCount &&
           (args.chunkSize == kChunk64 || args.chunkSize == kChunk128) &&
           args.numChunks > 0 && args.taskNum > 0 &&
           (args.outputG == 0 || args.gCumsumBth != nullptr) &&
           (args.isVarlen == 0 || (args.cuSeqlens != nullptr && args.chunkIndices != nullptr));
}

class Kernel {
public:
    __aicore__ inline void Init(const PrepareArgs &args)
    {
        args_ = args;
        rawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.rawG),
                               args_.batch * args_.tokens * args_.heads);
        gCumsumBhtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.gCumsumBht),
                                      args_.batch * args_.heads * args_.tokens);
        hasPublicOutput_ = args_.outputG != 0 && args_.gCumsumBth != nullptr;
        if (hasPublicOutput_) {
            gCumsumBthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.gCumsumBth),
                                          args_.batch * args_.tokens * args_.heads);
        }
        if (args_.isVarlen != 0) {
            cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(args_.cuSeqlens));
            chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(args_.chunkIndices),
                                            args_.numChunks * 2);
        }

        // Bound the largest tile at 64 heads.  The four chunk-sized FP32
        // staging buffers use the aligned width of the first tile, and the
        // accumulator uses the same width.  Thus Hv=8 keeps the original
        // 64/128 chunk footprint while a larger Hv is split into 64-head
        // tiles; a short final tile reuses that allocation with a right pad
        // of at most seven values and an explicit row stride.
        const uint64_t tileWidth = MinU64(args_.heads, kHeadTile);
        const uint64_t localStride = AlignHeadU64(tileWidth);
        const uint32_t tileBytes =
            static_cast<uint32_t>(args_.chunkSize * localStride * sizeof(float));
        pipe_.InitBuffer(inputBthBuf_, tileBytes);
        pipe_.InitBuffer(prefixBthBuf_, tileBytes);
        pipe_.InitBuffer(prefixBhtBuf_, tileBytes);
        pipe_.InitBuffer(offsetBuf_, tileBytes);
        pipe_.InitBuffer(accBuf_, localStride * sizeof(float));

        // Keep each physical dependency role explicit.  Stage P uses one
        // buffer, so reuse waits are required before the next tile overwrites
        // any input or output area.
        mte2ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        vToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        vToMte3_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        mte3ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        scalarToV_ = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        ProcessWithTopology(static_cast<uint64_t>(GetBlockIdx()),
                            static_cast<uint64_t>(GetBlockNum()));
    }

    // The formal Phase 6 entry is a mixed AIC/AIV launch.  In this kernel's
    // existing mixed topology GetBlockIdx() is already the vector-worker
    // index; GetBlockNum()*GetSubBlockNum() is the logical worker count.
    // Keep standalone Stage-P semantics above and do not remap the index.
    __aicore__ inline void ProcessMixed()
    {
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        const uint64_t physicalBlockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t logicalAivIdx = static_cast<uint64_t>(GetBlockIdx());
        ProcessWithTopology(logicalAivIdx, physicalBlockNum * subBlockNum);
    }

private:
    __aicore__ inline void ProcessWithTopology(uint64_t coreIdx, uint64_t activeAivCount)
    {
        if (args_.numChunks == 0 || args_.taskNum == 0) {
            ReleaseEvents();
            return;
        }
        if (activeAivCount == 0 || coreIdx >= activeAivCount || coreIdx >= args_.taskNum) {
            ReleaseEvents();
            return;
        }

        // Every tile of this worker uses the same local stride:
        // alignUp(min(Hv, kHeadTile), 8).  This lets the scalar side build
        // offsets once, before the first task, while a final tile still uses
        // only 0..7 values of right padding.
        const uint64_t tileWidth = MinU64(args_.heads, kHeadTile);
        const uint64_t localStride = AlignHeadU64(tileWidth);
        BuildOffsets(localStride);

        bool hasPreviousTask = false;
        for (uint64_t task = coreIdx; task < args_.taskNum; task += activeAivCount) {
            if (ProcessTask(task, hasPreviousTask, tileWidth, localStride)) {
                hasPreviousTask = true;
            }
        }

        // Close all producer edges before returning.  The following kernel in
        // the same stream consumes gCumsumBht only after this boundary.
        if (hasPreviousTask) {
            WaitForReuse();
        }
        ReleaseEvents();
    }

    __aicore__ inline void BuildOffsets(uint64_t localStride)
    {
        LocalTensor<uint32_t> offsets = offsetBuf_.Get<uint32_t>();
        // Gather writes a fully padded head-major tile.  Both head rows and
        // the chunk row length are 32B aligned:
        // dst[head * BT + row] <- src[row * localStride + head].
        // Padded source heads and tail rows are zero because prefixBth is
        // fully duplicated before each task.
        for (uint64_t head = 0; head < localStride; ++head) {
            for (uint64_t row = 0; row < args_.chunkSize; ++row) {
                offsets.SetValue(
                    static_cast<uint32_t>(head * args_.chunkSize + row),
                    static_cast<uint32_t>((row * localStride + head) * sizeof(float)));
            }
        }
        SetFlag<HardEvent::S_V>(scalarToV_);
        WaitFlag<HardEvent::S_V>(scalarToV_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeTask(uint64_t task, uint64_t &batch, uint64_t &rowStart,
                                      uint64_t &valid) const
    {
        const uint64_t chunk = task % args_.numChunks;
        batch = task / args_.numChunks;
        rowStart = chunk * args_.chunkSize;
        valid = rowStart < args_.tokens ? MinU64(args_.chunkSize, args_.tokens - rowStart) : 0;
        if (args_.isVarlen != 0) {
            // The tiling contract stores canonical (sequence, localChunk)
            // pairs in sequence-major order.  B=1 for this route.
            const int64_t sequence = chunkIndicesGm_.GetValue(chunk * 2);
            const int64_t localChunk = chunkIndicesGm_.GetValue(chunk * 2 + 1);
            const int64_t bos = cuSeqlensGm_.GetValue(sequence);
            const int64_t eos = cuSeqlensGm_.GetValue(sequence + 1);
            const int64_t start = bos + localChunk * static_cast<int64_t>(args_.chunkSize);
            const int64_t remaining = eos - start;
            rowStart = start > 0 ? static_cast<uint64_t>(start) : 0;
            valid = remaining > 0 ? MinU64(args_.chunkSize, static_cast<uint64_t>(remaining)) : 0;
            batch = 0;
        }
    }

    __aicore__ inline void WaitForReuse()
    {
        WaitFlag<HardEvent::V_MTE2>(vToMte2_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline bool ProcessTask(uint64_t task, bool hasPreviousTask,
                                       uint64_t tileWidth, uint64_t localStride)
    {
        uint64_t batch = 0;
        uint64_t rowStart = 0;
        uint64_t valid = 0;
        DecodeTask(task, batch, rowStart, valid);
        if (valid == 0 || batch >= args_.batch || rowStart >= args_.tokens) {
            return false;
        }

        bool hasPreviousTile = hasPreviousTask;
        for (uint64_t headStart = 0; headStart < args_.heads; headStart += kHeadTile) {
            if (hasPreviousTile) {
                // The next head tile reuses all four local buffers.  Wait
                // before Duplicate/DataCopy can overwrite a prior tile.
                WaitForReuse();
            }
            const uint64_t headCount = MinU64(tileWidth, args_.heads - headStart);
            ProcessHeadTile(batch, rowStart, valid, headStart, headCount, localStride);
            hasPreviousTile = true;
        }
        return true;
    }

    __aicore__ inline void ProcessHeadTile(uint64_t batch, uint64_t rowStart,
                                           uint64_t valid, uint64_t headStart,
                                           uint64_t headCount, uint64_t localStride)
    {
        LocalTensor<float> input = inputBthBuf_.Get<float>();
        LocalTensor<float> prefixBth = prefixBthBuf_.Get<float>();
        LocalTensor<float> prefixBht = prefixBhtBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();

        const uint32_t tileElements = static_cast<uint32_t>(args_.chunkSize * localStride);
        Duplicate(prefixBth, 0.0f, tileElements);
        Duplicate(acc, 0.0f, static_cast<uint32_t>(localStride));
        PipeBarrier<PIPE_V>();

        const uint64_t inputOffset =
            (batch * args_.tokens + rowStart) * args_.heads + headStart;
        const uint64_t roundedHeadCount = AlignHeadU64(headCount);
        const bool fullAlignedTile = headStart == 0 && headCount == args_.heads &&
                                     args_.heads == localStride &&
                                     (args_.heads % kHeadAlign) == 0;
        if (fullAlignedTile) {
            // Preserve the original contiguous fast path for the one-tile
            // aligned cases (Hv=8/16/32/64).
            const DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(valid * args_.heads * sizeof(float)), 0, 0, 0};
            const DataCopyPadExtParams<float> inputPad{false, 0, 0, 0.0f};
            DataCopyPad(input, rawGm_[inputOffset], inputParams, inputPad);
        } else {
            // GM is BTH with the real Hv row stride.  The local destination
            // is [valid, localStride].  DataCopyPad right-pads only to the
            // next 32B row (0..7 values); dstStride skips the remaining UB
            // row gap when this is a short final tile.
            const DataCopyExtParams inputParams{
                static_cast<uint16_t>(valid), static_cast<uint32_t>(headCount * sizeof(float)),
                static_cast<uint32_t>((args_.heads - headCount) * sizeof(float)),
                static_cast<uint32_t>((localStride - roundedHeadCount) / kHeadAlign), 0};
            const DataCopyPadExtParams<float> inputPad{
                true, 0, static_cast<uint8_t>(roundedHeadCount - headCount), 0.0f};
            DataCopyPad(input, rawGm_[inputOffset], inputParams, inputPad);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);

        // Each Add operates on this tile's independent heads.  T order and
        // the first Adds(+0.0f) operation are unchanged for every tile.
        for (uint64_t row = 0; row < valid; ++row) {
            LocalTensor<float> inputRow = input[row * localStride];
            if (row == 0) {
                Adds(acc, inputRow, 0.0f, static_cast<uint32_t>(headCount));
            } else {
                Add(acc, acc, inputRow, static_cast<uint32_t>(headCount));
            }
            PipeBarrier<PIPE_V>();
            // The first Copy argument is the element mask.  Mask=1 would
            // copy only head 0; headCount preserves all valid tile heads.
            Copy(prefixBth[row * localStride], acc, static_cast<uint64_t>(headCount), 1,
                 {1, 1, kFp32BlockElements, kFp32BlockElements});
            // Copy reads acc asynchronously; close its WAR edge before the
            // next Add overwrites the accumulator.
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(vToMte2_);

        const uint32_t elementCount = static_cast<uint32_t>(args_.chunkSize * localStride);
        Gather(prefixBht, prefixBth, offsetBuf_.Get<uint32_t>(), 0, elementCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);

        const DataCopyExtParams headParams{
            1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0};
        for (uint64_t localHead = 0; localHead < headCount; ++localHead) {
            const uint64_t head = headStart + localHead;
            const uint64_t bhtOffset = (batch * args_.heads + head) * args_.tokens + rowStart;
            DataCopyPad(gCumsumBhtGm_[bhtOffset],
                        prefixBht[localHead * args_.chunkSize], headParams);
        }
        if (hasPublicOutput_) {
            if (fullAlignedTile) {
                const DataCopyExtParams publicParams{
                    1, static_cast<uint32_t>(valid * args_.heads * sizeof(float)), 0, 0, 0};
                DataCopyPad(gCumsumBthGm_[(batch * args_.tokens + rowStart) * args_.heads],
                            prefixBth, publicParams);
            } else {
                // Reverse the same 2-D layout: source rows are localStride
                // floats (srcStride is in 32B blocks); GM rows advance by the
                // real Hv width (dstStride is in bytes).
                const DataCopyExtParams publicParams{
                    static_cast<uint16_t>(valid), static_cast<uint32_t>(headCount * sizeof(float)),
                    static_cast<uint32_t>((localStride - roundedHeadCount) / kHeadAlign),
                    static_cast<uint32_t>((args_.heads - headCount) * sizeof(float)), 0};
                DataCopyPad(gCumsumBthGm_[(batch * args_.tokens + rowStart) * args_.heads + headStart],
                            prefixBth, publicParams);
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline void ReleaseEvents()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToV_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(scalarToV_);
    }

    PrepareArgs args_{};
    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBthBuf_;
    TBuf<TPosition::VECCALC> prefixBthBuf_;
    TBuf<TPosition::VECCALC> prefixBhtBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    GlobalTensor<float> rawGm_;
    GlobalTensor<float> gCumsumBhtGm_;
    GlobalTensor<float> gCumsumBthGm_;
    GlobalTensor<int64_t> cuSeqlensGm_;
    GlobalTensor<int64_t> chunkIndicesGm_;
    TEventID mte2ToV_;
    TEventID vToMte2_;
    TEventID vToMte3_;
    TEventID mte3ToV_;
    TEventID mte3ToMte2_;
    TEventID scalarToV_;
    bool hasPublicOutput_ = false;
};

__aicore__ inline void Run(const PrepareArgs &args)
{
    if (!IsSupported(args)) {
        return;
    }
    Kernel kernel;
    kernel.Init(args);
    kernel.Process();
}

} // namespace GdnCumsumPrepare

#endif // GDN_CUMSUM_PREPARE_HPP
