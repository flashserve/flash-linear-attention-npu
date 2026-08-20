#ifndef KDA_GATE_BWD_POST_H
#define KDA_GATE_BWD_POST_H

#include "kernel_operator.h"
#include "kda_gate_bwd_post_common.h"
#include "chunk_kda_bwd_finalize_gate.h"
#if defined(__DAV_C310__) ||                                                \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) ||                    \
    (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310))
#define KDA_GATE_POST_ARCH35 1
#else
#define KDA_GATE_POST_ARCH35 0
#endif

#if KDA_GATE_POST_ARCH35
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif
#endif

namespace KDA {

#if KDA_GATE_POST_ARCH35
static __simd_vf__ inline void KdaGatePostCopyRowA5(
    __ubuf__ float *input, __ubuf__ float *acc)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t offset = 0; offset < 128; offset += kRegElements) {
        RegTensor<float> value;
        LoadAlign<float, LoadDist::DIST_NORM>(value, input + offset);
        StoreAlign(acc + offset, value, mask);
    }
}

static __simd_vf__ inline void KdaGatePostAccumulateRowA5(
    __ubuf__ float *input, __ubuf__ float *acc)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t kPairElements = 2U * kRegElements;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t offset = 0; offset < 128; offset += kPairElements) {
        RegTensor<float> input0;
        RegTensor<float> input1;
        RegTensor<float> acc0;
        RegTensor<float> acc1;
        LoadAlign<float, LoadDist::DIST_NORM>(input0, input + offset);
        LoadAlign<float, LoadDist::DIST_NORM>(
            input1, input + offset + kRegElements);
        LoadAlign<float, LoadDist::DIST_NORM>(acc0, acc + offset);
        LoadAlign<float, LoadDist::DIST_NORM>(
            acc1, acc + offset + kRegElements);
        Add(acc0, acc0, input0, mask);
        Add(acc1, acc1, input1, mask);
        StoreAlign(acc + offset, acc0, mask);
        StoreAlign(acc + offset + kRegElements, acc1, mask);
    }
}

template <uint16_t ROWS>
static __simd_vf__ inline void KdaGatePostReverseScanCarryA5(
    __ubuf__ float *data, __ubuf__ float *carry)
{
    constexpr uint32_t kCols = 128;
    constexpr uint32_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    using namespace AscendC::MicroAPI;
    RegTensor<float> valueReg;
    RegTensor<float> carryReg;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < kCols; col += kRegCols) {
        DataCopy(carryReg, carry + col);
        for (uint32_t row = ROWS; row > 0; --row) {
            const uint32_t offset = (row - 1U) * kCols + col;
            DataCopy(valueReg, data + offset);
            Add(carryReg, carryReg, valueReg, mask);
            DataCopy(data + offset, carryReg, mask);
        }
        DataCopy(carry + col, carryReg, mask);
    }
}

static __simd_vf__ inline void KdaGatePostFillA5(
    __ubuf__ float *dst, float value, uint16_t count)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> valueReg;
    Duplicate(valueReg, value);
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kRegElements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(dst + offset, valueReg, mask);
    }
}

template <uint32_t LAST_ROW>
static __simd_vf__ inline void KdaGatePostReverseScan16A5(
    __ubuf__ float *data, __ubuf__ float *carry)
{
    constexpr uint32_t kCols = 128;
    constexpr uint32_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    using namespace AscendC::MicroAPI;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < kCols; col += kRegCols) {
        RegTensor<float> valueReg;
        RegTensor<float> carryReg;
        LoadAlign<float, LoadDist::DIST_NORM>(carryReg, carry + col);
#define KDA_GATE_POST_SCAN_SEGMENT_ROW(I)                                  \
    do {                                                                     \
        constexpr uint32_t row = LAST_ROW - (I);                            \
        constexpr uint32_t offset = row * kCols;                            \
            LoadAlign<float, LoadDist::DIST_NORM>(                             \
                valueReg, data + offset + col);                                \
            Add(carryReg, carryReg, valueReg, mask);                             \
            StoreAlign(data + offset + col, carryReg, mask);                    \
    } while (0)
        KDA_GATE_POST_SCAN_SEGMENT_ROW(0U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(1U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(2U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(3U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(4U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(5U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(6U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(7U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(8U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(9U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(10U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(11U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(12U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(13U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(14U);
        KDA_GATE_POST_SCAN_SEGMENT_ROW(15U);
#undef KDA_GATE_POST_SCAN_SEGMENT_ROW
        StoreAlign(carry + col, carryReg, mask);
    }
}

static __simd_vf__ inline void KdaGatePostCopyPadRows128A5(
    __ubuf__ float *input, __ubuf__ float *output, uint16_t rows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t kCols = 2U * kRegCols;
    constexpr uint16_t kChunkRows = 64U;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * kCols;
        RegTensor<float> value0;
        RegTensor<float> value1;
        LoadAlign<float, LoadDist::DIST_NORM>(value0, input + offset);
        LoadAlign<float, LoadDist::DIST_NORM>(
            value1, input + offset + kRegCols);
        StoreAlign(output + offset, value0, mask);
        StoreAlign(output + offset + kRegCols, value1, mask);
    }
    for (uint16_t row = rows; row < kChunkRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * kCols;
        RegTensor<float> zero0;
        RegTensor<float> zero1;
        Duplicate(zero0, 0.0f, mask);
        Duplicate(zero1, 0.0f, mask);
        StoreAlign(output + offset, zero0, mask);
        StoreAlign(output + offset + kRegCols, zero1, mask);
    }
}

static __simd_vf__ inline void KdaGatePostZeroCarry128A5(
    __ubuf__ float *carry)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero0;
    RegTensor<float> zero1;
    Duplicate(zero0, 0.0f, mask);
    Duplicate(zero1, 0.0f, mask);
    StoreAlign(carry, zero0, mask);
    StoreAlign(carry + kRegCols, zero1, mask);
}

static __simd_vf__ inline void KdaGatePostReverseRows128A5(
    __ubuf__ float *input, __ubuf__ float *output, uint16_t rows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t kCols = 2U * kRegCols;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        const uint16_t sourceRow = rows - 1U - row;
        const uint32_t sourceOffset =
            static_cast<uint32_t>(sourceRow) * kCols;
        const uint32_t outputOffset = static_cast<uint32_t>(row) * kCols;
        RegTensor<float> value0;
        RegTensor<float> value1;
        LoadAlign<float, LoadDist::DIST_NORM>(
            value0, input + sourceOffset);
        LoadAlign<float, LoadDist::DIST_NORM>(
            value1, input + sourceOffset + kRegCols);
        StoreAlign(output + outputOffset, value0, mask);
        StoreAlign(output + outputOffset + kRegCols, value1, mask);
    }
}

static __simd_vf__ inline void KdaGatePostAccumulateForward128A5(
    __ubuf__ float *input, __ubuf__ float *output, uint16_t rows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t kCols = 2U * kRegCols;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> carry0;
    RegTensor<float> carry1;
    Duplicate(carry0, 0.0f, mask);
    Duplicate(carry1, 0.0f, mask);
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * kCols;
        RegTensor<float> value0;
        RegTensor<float> value1;
        LoadAlign<float, LoadDist::DIST_NORM>(value0, input + offset);
        LoadAlign<float, LoadDist::DIST_NORM>(
            value1, input + offset + kRegCols);
        Add(carry0, carry0, value0, mask);
        Add(carry1, carry1, value1, mask);
        StoreAlign(output + offset, carry0, mask);
        StoreAlign(output + offset + kRegCols, carry1, mask);
    }
}

static __aicore__ inline void KdaGatePostReverseScanStepA5(
    AscendC::LocalTensor<float> input,
    AscendC::LocalTensor<float> output,
    uint32_t rows, uint32_t stride)
{
    constexpr uint32_t kCols = 128U;
    const uint32_t addRows = rows > stride ? rows - stride : 0U;
    for (uint32_t row = 0; row < addRows; ++row) {
        const uint32_t offset = row * kCols;
        AscendC::Add(
            output[offset], input[offset],
            input[(row + stride) * kCols], kCols);
    }
    for (uint32_t row = addRows; row < rows; ++row) {
        const uint32_t offset = row * kCols;
        AscendC::Adds(output[offset], input[offset], 0.0f, kCols);
    }
}

template <bool HAS_BIAS, uint16_t ROW_BEGIN, uint16_t ROW_COUNT,
          bool RESET_CARRY,
          bool SCAN_INPUT,
          bool UNSCALE_UPSTREAM = false>
static __simd_vf__ inline void KdaGatePostSafeScannedA5(
    __ubuf__ float *dg, __ubuf__ float *dbAcc, __ubuf__ float *dAAcc,
    __ubuf__ float *rawGate, __ubuf__ float *upstream,
    __ubuf__ float *bias, __ubuf__ float *scanScratch,
    uint16_t rows, float expA, float lowerBound)
{
    constexpr uint16_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t kCols = 2U * kRegCols;
    using namespace AscendC::MicroAPI;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> one0;
    RegTensor<float> one1;
    RegTensor<float> db0;
    RegTensor<float> db1;
    RegTensor<float> dASum0;
    RegTensor<float> dASum1;
    RegTensor<float> carry0;
    RegTensor<float> carry1;
    RegTensor<float> bias0;
    RegTensor<float> bias1;
    Duplicate(one0, 1.0f, mask);
    Duplicate(one1, 1.0f, mask);
    Duplicate(dASum0, 0.0f, mask);
    Duplicate(dASum1, 0.0f, mask);
    DataCopy(db0, dbAcc);
    DataCopy(db1, dbAcc + kRegCols);
    if constexpr (SCAN_INPUT) {
        if constexpr (RESET_CARRY) {
            Duplicate(carry0, 0.0f, mask);
            Duplicate(carry1, 0.0f, mask);
        } else {
            DataCopy(carry0, scanScratch);
            DataCopy(carry1, scanScratch + kRegCols);
        }
    }
    if constexpr (HAS_BIAS) {
        DataCopy(bias0, bias);
        DataCopy(bias1, bias + kRegCols);
    }
    const float chainScale = lowerBound * expA;
    (void)rows;
#pragma unroll 64
    for (uint16_t rowIter = 0; rowIter < ROW_COUNT; ++rowIter) {
        const uint16_t sourceRow =
            ROW_BEGIN + ROW_COUNT - 1U - rowIter;
        const uint32_t offset =
            static_cast<uint32_t>(sourceRow) * kCols;
        RegTensor<float> raw0;
        RegTensor<float> raw1;
        RegTensor<float> denominator0;
        RegTensor<float> denominator1;
        RegTensor<float> sigmoid0;
        RegTensor<float> sigmoid1;
        RegTensor<float> oneMinus0;
        RegTensor<float> oneMinus1;
        RegTensor<float> upstream0;
        RegTensor<float> upstream1;
        RegTensor<float> gradient0;
        RegTensor<float> gradient1;
        RegTensor<float> dA0;
        RegTensor<float> dA1;
        DataCopy(raw0, rawGate + offset);
        DataCopy(raw1, rawGate + offset + kRegCols);
        if constexpr (HAS_BIAS) {
            Add(raw0, raw0, bias0, mask);
            Add(raw1, raw1, bias1, mask);
        }
        Muls(denominator0, raw0, -expA, mask);
        Muls(denominator1, raw1, -expA, mask);
        Exp(denominator0, denominator0, mask);
        Exp(denominator1, denominator1, mask);
        Adds(denominator0, denominator0, 1.0f, mask);
        Adds(denominator1, denominator1, 1.0f, mask);
        Div(sigmoid0, one0, denominator0, mask);
        Div(sigmoid1, one1, denominator1, mask);
        Muls(oneMinus0, sigmoid0, -1.0f, mask);
        Muls(oneMinus1, sigmoid1, -1.0f, mask);
        Adds(oneMinus0, oneMinus0, 1.0f, mask);
        Adds(oneMinus1, oneMinus1, 1.0f, mask);
        Mul(gradient0, oneMinus0, sigmoid0, mask);
        Mul(gradient1, oneMinus1, sigmoid1, mask);
        DataCopy(upstream0, upstream + offset);
        DataCopy(upstream1, upstream + offset + kRegCols);
        if constexpr (UNSCALE_UPSTREAM) {
            Muls(upstream0, upstream0, 0.6931471805599453f, mask);
            Muls(upstream1, upstream1, 0.6931471805599453f, mask);
        }
        if constexpr (SCAN_INPUT) {
            Add(carry0, carry0, upstream0, mask);
            Add(carry1, carry1, upstream1, mask);
            Mul(gradient0, gradient0, carry0, mask);
            Mul(gradient1, gradient1, carry1, mask);
        } else {
            Mul(gradient0, gradient0, upstream0, mask);
            Mul(gradient1, gradient1, upstream1, mask);
        }
        Muls(gradient0, gradient0, chainScale, mask);
        Muls(gradient1, gradient1, chainScale, mask);
        Mul(dA0, gradient0, raw0, mask);
        Mul(dA1, gradient1, raw1, mask);
        Add(db0, db0, gradient0, mask);
        Add(db1, db1, gradient1, mask);
        Add(dASum0, dASum0, dA0, mask);
        Add(dASum1, dASum1, dA1, mask);
        DataCopy(dg + offset, gradient0, mask);
        DataCopy(dg + offset + kRegCols, gradient1, mask);
    }
    if constexpr (SCAN_INPUT) {
        DataCopy(scanScratch, carry0, mask);
        DataCopy(scanScratch + kRegCols, carry1, mask);
    }
    DataCopy(dbAcc, db0, mask);
    DataCopy(dbAcc + kRegCols, db1, mask);
    RegTensor<float> dABlock0;
    RegTensor<float> dABlock1;
    RegTensor<float> dATotal;
    ReduceSum(dABlock0, dASum0, mask);
    ReduceSum(dABlock1, dASum1, mask);
    Add(dABlock0, dABlock0, dABlock1, mask);
    DataCopy<float, LoadDist::DIST_BRC_B32>(dATotal, dAAcc);
    Add(dATotal, dATotal, dABlock0, mask);
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
        dAAcc, dATotal, mask);
}
#endif

struct GatePostChunkTask {
    uint32_t batch;
    uint32_t begin;
    uint32_t end;
};

template <bool SAFE_GATE, typename RawGateT, bool VARLEN_TND>
class KdaGateBwdPostKernel {
public:
    static constexpr uint32_t kKeyDim = 128;
    static constexpr uint32_t kChunkRows = 64;
    static constexpr uint32_t kChunkBytes =
        kChunkRows * kKeyDim * sizeof(float);
    static constexpr uint32_t kRowBytes =
        kKeyDim * sizeof(float);
    static constexpr uint32_t kWorkBytes = 4 * kRowBytes;
    static_assert(4 * kChunkBytes + 4 * 1024 + 32 <= 192 * 1024,
                  "KdaGateBwdPost exceeds the 192 KiB UB budget");

    __aicore__ inline void Init(
        GM_ADDR dgAct, GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dg,
        GM_ADDR dA, GM_ADDR dBias,
        const KdaGateBwdPostTilingData &tiling,
        AscendC::TPipe *pipe)
    {
        tiling_.batch = tiling.batch;
        tiling_.seqlen = tiling.seqlen;
        tiling_.headNum = tiling.headNum;
        tiling_.keyDim = tiling.keyDim;
        tiling_.chunkSize = tiling.chunkSize;
        tiling_.chunkNum = tiling.chunkNum;
        tiling_.chunkNumPerBatch = tiling.chunkNumPerBatch;
        tiling_.usedCoreNum = tiling.usedCoreNum;
        tiling_.isVarLen = VARLEN_TND ? 1 : tiling.isVarLen;
        tiling_.hasDtBias = tiling.hasDtBias;
        // This private post kernel is launched only after Kernel C writes the
        // chunk-local (unscanned) dg_base tensor.  Keep the contract static:
        // every invocation must perform the reverse cumsum before applying
        // the raw-gate chain rule.
        tiling_.inputScanned = 0;
        tiling_.reserved1 = tiling.reserved1;
        tiling_.lowerBound = tiling.lowerBound;
        // The kernel entry owns exactly one TPipe.  Reuse it for every
        // TBuf/TQue, matching the mature pure-AIV gate-cumsum operator.
        pipe_ = pipe;
        dgActGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgAct));
        rawGGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawGateT *>(rawG));
        aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
        if (tiling_.hasDtBias != 0) {
            dtBiasGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(dtBias));
            dBiasGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(dBias));
        }
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg));
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA));
        if (tiling_.isVarLen != 0) {
            cuSeqlensGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            chunkIndicesGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }

        if constexpr (SAFE_GATE) {
            // Keep one complete [64,128] chunk in UB and materialize the
            // reverse scan before applying the gate chain.
            pipe_->InitBuffer(scanPing_, kChunkBytes);
            pipe_->InitBuffer(scanPong_, kChunkBytes);
            pipe_->InitBuffer(raw_, kChunkBytes);
            pipe_->InitBuffer(result_, kChunkBytes);
        } else
        {
            pipe_->InitBuffer(upstreamQueue_, 1, kRowBytes);
            pipe_->InitBuffer(scanPing_, kWorkBytes);
            pipe_->InitBuffer(scanPong_, kWorkBytes);
            pipe_->InitBuffer(raw_, kRowBytes);
            pipe_->InitBuffer(result_, kRowBytes);
        }
        pipe_->InitBuffer(reduce_, 4 * 1024);
        pipe_->InitBuffer(metadata_, 32);
        vToMte3_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        mte3ToV_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
        initMte2ToV_ =
            pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        rawMte2ToV_ =
            pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
        scalarVToS_ = pipe_->AllocEventID<AscendC::HardEvent::V_S>();
        scalarSToV_ = pipe_->AllocEventID<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        for (uint32_t head = core;
             head < static_cast<uint32_t>(tiling_.headNum);
             head += static_cast<uint32_t>(tiling_.usedCoreNum)) {
            ProcessHead(head);
        }
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(initMte2ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(scalarVToS_);
        pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(scalarSToV_);
    }

private:
    __aicore__ inline GatePostChunkTask DecodeTask(uint32_t taskIdx)
    {
        GatePostChunkTask task{};
        if (tiling_.inputScanned != 0) {
            task.batch = 0;
            task.begin = taskIdx *
                static_cast<uint32_t>(tiling_.chunkSize);
            if (task.begin >= static_cast<uint32_t>(tiling_.seqlen)) {
                task.end = task.begin;
                return task;
            }
            task.end = task.begin +
                static_cast<uint32_t>(tiling_.chunkSize);
            if (task.end > static_cast<uint32_t>(tiling_.seqlen)) {
                task.end = static_cast<uint32_t>(tiling_.seqlen);
            }
            return task;
        }
        if (tiling_.isVarLen != 0) {
            task.batch = 0;
            const int64_t sequence =
                ReadInt64(chunkIndicesGm_, 2U * taskIdx);
            const int64_t localChunk =
                ReadInt64(chunkIndicesGm_, 2U * taskIdx + 1U);
            const int64_t sequenceBegin =
                ReadInt64(cuSeqlensGm_, sequence);
            const int64_t sequenceEnd =
                ReadInt64(cuSeqlensGm_, sequence + 1);
            const int64_t begin =
                sequenceBegin + localChunk * kChunkRows;
            const int64_t end = begin + kChunkRows < sequenceEnd ?
                begin + kChunkRows : sequenceEnd;
            task.begin = static_cast<uint32_t>(begin);
            task.end = static_cast<uint32_t>(end);
            return task;
        }
        task.batch = taskIdx /
            static_cast<uint32_t>(tiling_.chunkNumPerBatch);
        const uint32_t localChunk = taskIdx %
            static_cast<uint32_t>(tiling_.chunkNumPerBatch);
        task.begin = localChunk *
            static_cast<uint32_t>(tiling_.chunkSize);
        task.end = task.begin +
            static_cast<uint32_t>(tiling_.chunkSize);
        if (task.end > static_cast<uint32_t>(tiling_.seqlen)) {
            task.end = static_cast<uint32_t>(tiling_.seqlen);
        }
        return task;
    }

    __aicore__ inline uint64_t TokenOffset(
        uint32_t batch, uint32_t head, uint32_t token)
    {
        if (tiling_.isVarLen != 0) {
            return (static_cast<uint64_t>(head) * tiling_.seqlen + token) *
                kKeyDim;
        }
        return ((static_cast<uint64_t>(batch) * tiling_.headNum + head) *
                    tiling_.seqlen + token) * kKeyDim;
    }

    __aicore__ inline void ProcessHead(uint32_t head)
    {
        auto dbAcc = reduce_.Get<float>();
        auto dAAcc = dbAcc[kKeyDim];
        auto scalar = reduce_.Get<float>()[136];
        auto bias = reduce_.Get<float>()[144];
        AscendC::Duplicate(dbAcc, 0.0f, kKeyDim);
        AscendC::Duplicate(dAAcc, 0.0f, 8);
        AscendC::DataCopyPad(
            scalar, aLogGm_[head],
            {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        if (tiling_.hasDtBias != 0) {
            AscendC::DataCopy(
                bias, dtBiasGm_[static_cast<uint64_t>(head) * kKeyDim],
                kKeyDim);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(initMte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(initMte2ToV_);
        AscendC::Exp(scalar, scalar, 1);
        AscendC::PipeBarrier<PIPE_V>();
        const float expA = ReadScalar(scalar);

        // For varlen, chunkNum is the number of chunk_indices entries,
        // i.e. sum_i ceil(sequence_length_i / C).  It is not generally equal
        // to ceil(total_tokens / C) when multiple sequences have tail chunks.
        const uint32_t taskCount =
            static_cast<uint32_t>(tiling_.chunkNum);
        for (uint32_t taskIdx = 0;
             taskIdx < taskCount;
             ++taskIdx) {
            const GatePostChunkTask task = DecodeTask(taskIdx);
            ProcessChunk(task, head, expA, dbAcc, dAAcc, bias);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::DataCopyPad(
            dAGm_[head], dAAcc,
            {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
        if (tiling_.hasDtBias != 0) {
            AscendC::DataCopy(
                dBiasGm_[static_cast<uint64_t>(head) * kKeyDim],
                dbAcc, kKeyDim);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void ProcessChunk(
        const GatePostChunkTask &task, uint32_t head, float expA,
        AscendC::LocalTensor<float> dbAcc,
        AscendC::LocalTensor<float> dAAcc,
        AscendC::LocalTensor<float> bias)
    {
        const uint32_t rows = task.end - task.begin;
        if (rows == 0) {
            return;
        }
        if constexpr (SAFE_GATE) {
            ProcessChunkSafeTileA5(
                task, head, rows, expA, dbAcc, dAAcc, bias);
            return;
        }
        auto accumulator = scanPing_.Get<float>();
        auto result = scanPong_.Get<float>();
        auto dAValue = result[kKeyDim];
        auto sigmoid = result[2U * kKeyDim];
        auto nextAccumulator = result[3U * kKeyDim];
        auto raw = raw_.Get<float>();
        // Proven row-stream fallback from chunk_local_cumsum: walk the chunk
        // backwards, keep the 128-column carry in UB, and consume each
        // scanned row immediately in the raw-gate chain rule.
        for (int32_t row = static_cast<int32_t>(kChunkRows) - 1;
             row >= 0; --row) {
            if (static_cast<uint32_t>(row) >= rows) {
                continue;
            }
            const uint32_t token =
                task.begin + static_cast<uint32_t>(row);
            const uint64_t rowOffset = TokenOffset(
                task.batch, head, token);
            auto upstream = upstreamQueue_.AllocTensor<float>();
            AscendC::DataCopy(
                upstream, dgActGm_[rowOffset], kKeyDim);
            upstreamQueue_.EnQue(upstream);
            if constexpr (AscendC::IsSameType<RawGateT, float>::value) {
                AscendC::DataCopyPad(
                    raw, rawGGm_[rowOffset],
                    {1, static_cast<uint32_t>(kRowBytes), 0, 0, 0},
                    {false, 0, 0, 0});
            } else {
                auto rawStage = scanPong_.Get<RawGateT>();
                AscendC::DataCopyPad(
                    rawStage, rawGGm_[rowOffset],
                    {1, static_cast<uint32_t>(
                            kKeyDim * sizeof(RawGateT)), 0, 0, 0},
                    {false, 0, 0, 0});
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
            upstream = upstreamQueue_.DeQue<float>();

            if constexpr (VARLEN_TND) {
                if (static_cast<uint32_t>(row) + 1U == rows) {
                    AscendC::Adds(
                        nextAccumulator, upstream, 0.0f, kKeyDim);
                } else {
                    AscendC::Add(
                        nextAccumulator, accumulator, upstream, kKeyDim);
                }
            } else {
                if (tiling_.inputScanned != 0 ||
                    static_cast<uint32_t>(row) + 1U == rows) {
                    AscendC::Adds(
                        nextAccumulator, upstream, 0.0f, kKeyDim);
                } else {
                    AscendC::Add(
                        nextAccumulator, accumulator, upstream, kKeyDim);
                }
            }
            // The accumulator is a loop-carried Vector dependency.  Keep the
            // barrier adjacent to the in-place Add/Copy, as in the proven
            // kda_gate_cumsum row pipeline, before issuing the raw-gate Cast.
            AscendC::PipeBarrier<PIPE_V>();
            upstreamQueue_.FreeTensor(upstream);
            if constexpr (!AscendC::IsSameType<RawGateT, float>::value) {
                auto rawStage = scanPong_.Get<RawGateT>();
                AscendC::Cast(
                    raw, rawStage, AscendC::RoundMode::CAST_NONE,
                    kKeyDim);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (tiling_.hasDtBias != 0) {
                AscendC::Add(raw, raw, bias, kKeyDim);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if constexpr (SAFE_GATE) {
                AscendC::Muls(sigmoid, raw, expA, kKeyDim);
                AscendC::PipeBarrier<PIPE_V>();
                Sigmoid(sigmoid, sigmoid, kKeyDim);
                AscendC::Muls(result, sigmoid, -1.0f, kKeyDim);
                AscendC::Adds(result, result, 1.0f, kKeyDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(result, result, sigmoid, kKeyDim);
                AscendC::Mul(result, result, nextAccumulator, kKeyDim);
                AscendC::Muls(
                    result, result, tiling_.lowerBound * expA,
                    kKeyDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(dAValue, result, raw, kKeyDim);
            } else {
                const float a = -expA;
                Sigmoid(sigmoid, raw, kKeyDim);
                AscendC::Mul(result, nextAccumulator, sigmoid, kKeyDim);
                AscendC::Muls(result, result, a, kKeyDim);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Abs(dAValue, raw, kKeyDim);
                AscendC::Muls(dAValue, dAValue, -1.0f, kKeyDim);
                AscendC::Exp(dAValue, dAValue, kKeyDim);
                AscendC::Adds(dAValue, dAValue, 1.0f, kKeyDim);
                AscendC::Ln(dAValue, dAValue, kKeyDim);
                AscendC::Maxs(raw, raw, 0.0f, kKeyDim);
                AscendC::Add(dAValue, dAValue, raw, kKeyDim);
                AscendC::Muls(dAValue, dAValue, a, kKeyDim);
                AscendC::Mul(
                    dAValue, dAValue, nextAccumulator, kKeyDim);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(dbAcc, dbAcc, result, kKeyDim);
            AccumulateScalar(dAAcc, dAValue, 1);
            AscendC::Adds(
                accumulator, nextAccumulator, 0.0f, kKeyDim);
            AscendC::PipeBarrier<PIPE_V>();
            StoreDg(rowOffset, result, kKeyDim);
        }
    }

    __aicore__ inline void ProcessChunkSafeTileA5(
        const GatePostChunkTask &task, uint32_t head, uint32_t rows,
        float expA, AscendC::LocalTensor<float> dbAcc,
        AscendC::LocalTensor<float> dAAcc,
        AscendC::LocalTensor<float> bias)
    {
        const uint32_t count = rows * kKeyDim;
        const uint64_t offset = TokenOffset(
            task.batch, head, task.begin);
        auto upstream = scanPing_.Get<float>();
        auto scanScratch = scanPong_.Get<float>();
        auto raw = raw_.Get<float>();
        auto output = result_.Get<float>();

        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t rowOffset = row * kKeyDim;
            AscendC::DataCopy(
                upstream[rowOffset], dgActGm_[offset + rowOffset],
                kKeyDim);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
        if (rows < kChunkRows) {
            AscendC::Duplicate(
                upstream[count], 0.0f,
                (kChunkRows - rows) * kKeyDim);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Kernel C defers both its Intra dg_hv reconstruction scan and the
        // raw-gate upstream scan.  Each fixed six-step pass finishes in the
        // first buffer because C is fixed at 64.
        ReverseScanChunkA5(upstream, scanScratch, kChunkRows);
        ReverseScanChunkA5(upstream, scanScratch, kChunkRows);
        auto scanned = upstream;
        auto chainScratch = scanScratch;

        if constexpr (AscendC::IsSameType<RawGateT, float>::value) {
            AscendC::DataCopyPad(
                raw, rawGGm_[offset],
                {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
        } else {
            auto rawStage = output.template ReinterpretCast<RawGateT>();
            AscendC::DataCopyPad(
                rawStage, rawGGm_[offset],
                {1, static_cast<uint32_t>(count * sizeof(RawGateT)), 0, 0, 0},
                {false, 0, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
        if constexpr (!AscendC::IsSameType<RawGateT, float>::value) {
            auto rawStage = output.template ReinterpretCast<RawGateT>();
            AscendC::Cast(
                raw, rawStage, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (tiling_.hasDtBias != 0) {
            for (uint32_t row = 0; row < rows; ++row) {
                AscendC::Add(
                    raw[row * kKeyDim], raw[row * kKeyDim],
                    bias, kKeyDim);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        // sigmoid(exp(A_log) * raw_gate)
        AscendC::Muls(output, raw, expA, count);
        AscendC::PipeBarrier<PIPE_V>();
        Sigmoid(output, output, count);

        // dg = reverse_cumsum(dg_base) * lower_bound * exp(A_log)
        //      * sigmoid * (1 - sigmoid)
        AscendC::Muls(chainScratch, output, -1.0f, count);
        AscendC::Adds(chainScratch, chainScratch, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(chainScratch, chainScratch, output, count);
        AscendC::Mul(output, chainScratch, scanned, count);
        AscendC::Muls(
            output, output, tiling_.lowerBound * expA, count);
        AscendC::PipeBarrier<PIPE_V>();
        if (rows < kChunkRows) {
            AscendC::Duplicate(
                output[count], 0.0f,
                (kChunkRows - rows) * kKeyDim);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // dA contribution can reuse the dead scan scratch plane.
        AscendC::Mul(chainScratch, output, raw, count);
        AscendC::PipeBarrier<PIPE_V>();
        AccumulateScalar(dAAcc, chainScratch, rows);

        // Fixed six-level tree reduction over 64 padded rows.  Every level
        // writes a buffer different from both inputs.
        AddBatched(raw, output, output[32U * kKeyDim], 32U * kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        AddBatched(
            chainScratch, raw, raw[16U * kKeyDim], 16U * kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        AddBatched(
            raw, chainScratch, chainScratch[8U * kKeyDim], 8U * kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        AddBatched(
            chainScratch, raw, raw[4U * kKeyDim], 4U * kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        AddBatched(
            raw, chainScratch, chainScratch[2U * kKeyDim], 2U * kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(chainScratch, raw, raw[kKeyDim], kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scanned, dbAcc, chainScratch, kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(dbAcc, scanned, 0.0f, kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();

        StoreDg(offset, output, count);
    }

    __aicore__ inline void ReverseScanChunkScalarA5(
        AscendC::LocalTensor<float> data, uint32_t rows)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        volatile __ubuf__ float *ptr =
            reinterpret_cast<volatile __ubuf__ float *>(data.GetPhyAddr());
        for (uint32_t col = 0; col < kKeyDim; ++col) {
            float carry = 0.0f;
            for (uint32_t row = rows; row > 0; --row) {
                const uint32_t offset = (row - 1U) * kKeyDim + col;
                carry += ptr[offset];
                ptr[offset] = carry;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarSToV_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarSToV_);
    }

    template <uint32_t STEP>
    __aicore__ inline void ReverseScanStep64(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src)
    {
        constexpr uint32_t kActiveRows = kChunkRows - STEP;
        AddBatched(
            dst, src, src[STEP * kKeyDim],
            kActiveRows * kKeyDim);
        AddsBatched(
            dst[kActiveRows * kKeyDim],
            src[kActiveRows * kKeyDim], 0.0f,
            STEP * kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool ReverseScanChunkA5(
        AscendC::LocalTensor<float> chunk,
        AscendC::LocalTensor<float> scratch, uint32_t rows)
    {
        (void)rows;
        ReverseScanStep64<1U>(scratch, chunk);
        ReverseScanStep64<2U>(chunk, scratch);
        ReverseScanStep64<4U>(scratch, chunk);
        ReverseScanStep64<8U>(chunk, scratch);
        ReverseScanStep64<16U>(scratch, chunk);
        ReverseScanStep64<32U>(chunk, scratch);
        return true;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessChunkSafeA5(
        const GatePostChunkTask &task, uint32_t head, uint32_t rows,
        float expA, AscendC::LocalTensor<float> dbAcc,
        AscendC::LocalTensor<float> dAAcc,
        AscendC::LocalTensor<float> bias)
    {
        const uint32_t count = rows * kKeyDim;
        const uint64_t offset = TokenOffset(task.batch, head, task.begin);
        auto upstream = upstreamQueue_.AllocTensor<float>();
        AscendC::DataCopyPad(
            upstream, dgActGm_[offset],
            {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        upstreamQueue_.EnQue(upstream);
        upstream = upstreamQueue_.DeQue<float>();

        // The regbase path is specialized for C=64. Tail rows are zeroed so
        // the fixed-size fused scan+gate loop contributes no padded gradient.
        if (rows < kChunkRows) {
            AscendC::Duplicate(
                upstream[count], 0.0f,
                (kChunkRows - rows) * kKeyDim);
        }
        AscendC::PipeBarrier<PIPE_V>();
        auto gateOutput = scanPong_.Get<float>();

        // Finish the upstream load before starting the raw-gate MTE2
        // transaction; overlapping the event chains corrupts data on A5.

        auto raw = raw_.Get<float>();
        if constexpr (AscendC::IsSameType<RawGateT, float>::value) {
            AscendC::DataCopyPad(
                raw, rawGGm_[offset],
                {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
        } else {
            auto rawStage = scanPong_.Get<RawGateT>();
            AscendC::DataCopyPad(
                rawStage, rawGGm_[offset],
                {1, static_cast<uint32_t>(count * sizeof(RawGateT)),
                 0, 0, 0},
                {false, 0, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawMte2ToV_);
        if constexpr (!AscendC::IsSameType<RawGateT, float>::value) {
            auto rawStage = scanPong_.Get<RawGateT>();
            AscendC::Cast(
                raw, rawStage, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (rows < kChunkRows) {
            AscendC::Duplicate(
                raw[count], 0.0f,
                (kChunkRows - rows) * kKeyDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // Reuse Kernel C's proven A5 safe-gate primitive verbatim.  It walks
        // the rows backwards, keeps the reverse-scan carry in registers and
        // applies the chain rule in the same pass; the input here is the
        // unscanned dg_base emitted by Kernel C.
        auto gateCarry = result_.Get<float>();
        KdaBwdCGateFillA5(
            (__ubuf__ float *)gateCarry.GetPhyAddr(), 0.0f, kKeyDim);
        if (tiling_.hasDtBias != 0) {
            KdaBwdCSafeGateBackwardA5<true>(
                (__ubuf__ float *)gateOutput.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dAAcc.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)upstream.GetPhyAddr(),
                (__ubuf__ float *)bias.GetPhyAddr(),
                (__ubuf__ float *)gateCarry.GetPhyAddr(),
                static_cast<uint16_t>(rows), expA, tiling_.lowerBound);
        } else {
            KdaBwdCSafeGateBackwardA5<false>(
                (__ubuf__ float *)gateOutput.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dAAcc.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)upstream.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)gateCarry.GetPhyAddr(),
                static_cast<uint16_t>(rows), expA, tiling_.lowerBound);
        }
        AscendC::PipeBarrier<PIPE_V>();
        StoreDg(offset, gateOutput, count);
        upstreamQueue_.FreeTensor(upstream);
    }
#endif

    __aicore__ inline void AddBatched(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src0,
        AscendC::LocalTensor<float> src1, uint32_t count)
    {
        constexpr uint32_t kMaxElements = 64U * 255U;
        for (uint32_t offset = 0; offset < count;
             offset += kMaxElements) {
            const uint32_t current =
                count - offset < kMaxElements ? count - offset : kMaxElements;
            AscendC::Add(
                dst[offset], src0[offset], src1[offset], current);
        }
    }


    __aicore__ inline void AddsBatched(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src, float scalar, uint32_t count)
    {
        constexpr uint32_t kMaxElements = 64U * 255U;
        for (uint32_t offset = 0; offset < count;
             offset += kMaxElements) {
            const uint32_t current =
                count - offset < kMaxElements ? count - offset : kMaxElements;
            AscendC::Adds(dst[offset], src[offset], scalar, current);
        }
    }

    __aicore__ inline void Sigmoid(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src, uint32_t count)
    {
        auto one = reduce_.Get<float>()[800];
        AscendC::Muls(dst, src, -1.0f, count);
        AscendC::Exp(dst, dst, count);
        AscendC::Adds(dst, dst, 1.0f, count);
        AscendC::Duplicate(one, 1.0f, kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t offset = 0; offset < count; offset += kKeyDim) {
            AscendC::Div(dst[offset], one, dst[offset], kKeyDim);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ReadScalar(
        AscendC::LocalTensor<float> value)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        return ((__ubuf__ float *)value.GetPhyAddr())[0];
    }

    __aicore__ inline int64_t ReadInt64(
        AscendC::GlobalTensor<int64_t> &tensor, uint64_t offset)
    {
        auto value = metadata_.Get<int64_t>();
        AscendC::DataCopyPad(
            value, tensor[offset],
            {1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0},
            {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(initMte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(initMte2ToV_);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalarVToS_);
        return ((__ubuf__ int64_t *)value.GetPhyAddr())[0];
    }

    __aicore__ inline void AccumulateColumns(
        AscendC::LocalTensor<float> acc,
        AscendC::LocalTensor<float> values, uint32_t rows)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Add(acc, acc, values[row * kKeyDim], kKeyDim);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateScalar(
        AscendC::LocalTensor<float> acc,
        AscendC::LocalTensor<float> values, uint32_t rows)
    {
        auto partial = reduce_.Get<float>()[280];
        auto rowSum = reduce_.Get<float>()[280 + kChunkRows * 8];
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::WholeReduceSum(
                partial[row * 8], values[row * kKeyDim],
                64, 2, 1, 1, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(
            rowSum, partial, 2, rows, 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(
            partial, rowSum, rows, 1, 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc, acc, partial, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreDg(
        uint64_t offset, AscendC::LocalTensor<float> value,
        uint32_t count)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        // Keep each transfer at one aligned 128-FP32 row. The A5 contiguous
        // UB->GM path does not reliably issue this kernel's full 32 KiB tile
        // as a single transfer, while the same row-sized form is already
        // proven by the dbias writeback below.
        const uint32_t rows = count / kKeyDim;
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::DataCopy(
                dgGm_[offset + static_cast<uint64_t>(row) * kKeyDim],
                value[row * kKeyDim], kKeyDim);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    KdaGateBwdPostScalarTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<float> dgActGm_;
    AscendC::GlobalTensor<RawGateT> rawGGm_;
    AscendC::GlobalTensor<float> aLogGm_;
    AscendC::GlobalTensor<float> dtBiasGm_;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm_;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm_;
    AscendC::GlobalTensor<float> dgGm_;
    AscendC::GlobalTensor<float> dAGm_;
    AscendC::GlobalTensor<float> dBiasGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> upstreamQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scanPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scanPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> raw_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> result_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> metadata_;
    AscendC::TEventID vToMte3_;
    AscendC::TEventID mte3ToV_;
    AscendC::TEventID initMte2ToV_;
    AscendC::TEventID rawMte2ToV_;
    AscendC::TEventID scalarVToS_;
    AscendC::TEventID scalarSToV_;
};

} // namespace KDA

#endif // KDA_GATE_BWD_POST_H
