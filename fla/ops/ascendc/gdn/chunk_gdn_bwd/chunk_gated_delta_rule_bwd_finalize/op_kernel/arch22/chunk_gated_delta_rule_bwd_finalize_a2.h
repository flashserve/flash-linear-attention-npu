#ifndef GDN_BWD_FINALIZE_A2_H
#define GDN_BWD_FINALIZE_A2_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_bwd_finalize_workspace.h"

// 私有适配结构，不复用其它算子的生成 tiling ABI。
struct ChunkBwdDqkwgTilingData {
    uint64_t B, HV, HK, T, K, V, BT, numChunks;
    float scale;
    uint32_t mul0RowNum, aicCoreNum, hkPartitions;
    uint64_t wsMm3Offset, wsMm4Offset, wsMm6Offset, wsMm5Offset;
    uint64_t wsMm7Offset, wsDsTempOffset, wsDgLastOffset, isVarLen;
};

#include "dq/chunk_bwd_dqkwg_cube.h"
#include "dq/chunk_bwd_dqkwg_vector.h"
#include "wy/prepare_wy_repr_bwd_cube.h"
#include "wy/prepare_wy_repr_bwd_vector.h"

namespace GDN {

constexpr float A2_FINALIZE_LN2 = 0.6931471805599453f;

// 沿用 07d1b942 的整数 peak 评分；显式使用已由原调度资格限定的 U32 输入域。
__aicore__ inline bool A2FinalizeDqHkPartitionPeak(
    uint64_t coreLoops, uint64_t HV, uint64_t coreNum, uint64_t partitions, uint64_t &peak)
{
    if (coreLoops == 0 || coreLoops > UINT32_MAX || coreNum == 0 ||
        partitions == 0 || partitions > UINT32_MAX || HV == 0 || HV > UINT32_MAX || HV % partitions != 0) {
        return false;
    }
    // 两个 U32 因子的乘积可由 U64 表示，因此可在乘后检查原 INT64 上界。
    const uint64_t tasks = coreLoops * partitions;
    if (tasks > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    const uint64_t waves = tasks / coreNum + static_cast<uint64_t>(tasks % coreNum != 0);
    const uint64_t heads = HV / partitions;
    // waves<=tasks，且 P 整除 HV，所以 waves*heads<=coreLoops*HV<=UINT32_MAX^2。
    peak = waves * heads;
    return true;
}

__aicore__ inline uint64_t A2FinalizeSelectDqHkPartitions(
    uint64_t B, uint64_t numChunks, uint64_t HV, uint64_t HK,
    uint64_t coreNum, uint64_t parentPartitions, bool allowBalance)
{
    if (!allowBalance || !DqkwgUseU32Schedule(B, numChunks, HV, HK, coreNum, parentPartitions)) {
        return parentPartitions;
    }
    // 原 U32 资格先除后乘界定了 B*numChunks，此处才形成 N。
    const uint64_t coreLoops = B * numChunks;
    if (coreLoops < coreNum) {
        return parentPartitions;
    }
    uint64_t parentPeak = 0;
    if (!A2FinalizeDqHkPartitionPeak(coreLoops, HV, coreNum, parentPartitions, parentPeak)) {
        return parentPartitions;
    }
    uint64_t bestPeak = parentPeak;
    uint64_t bestPartitions = parentPartitions;
    for (uint64_t divisor = 1; divisor <= HK / divisor; ++divisor) {
        if (HK % divisor != 0) {
            continue;
        }
        const uint64_t other = HK / divisor;
        for (uint32_t candidate = 0; candidate < 2; ++candidate) {
            const uint64_t partitions = candidate == 0 ? divisor : other;
            if ((candidate != 0 && other == divisor) || (HK / partitions) % 2 != 0) {
                continue;
            }
            uint64_t peak = 0;
            if (A2FinalizeDqHkPartitionPeak(coreLoops, HV, coreNum, partitions, peak) &&
                (peak < bestPeak || (peak == bestPeak && partitions < bestPartitions))) {
                bestPeak = peak;
                bestPartitions = partitions;
            }
        }
    }
    return bestPeak < parentPeak ? bestPartitions : parentPartitions;
}

__aicore__ inline ChunkBwdDqkwgTilingData A2FinalizeDqTiling(
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t)
{
    ChunkBwdDqkwgTilingData d{};
    d.B = t.B; d.HV = t.HV; d.HK = t.HK; d.T = t.T;
    d.K = t.K; d.V = t.V; d.BT = t.chunkSize;
    d.numChunks = t.isVariable != 0 ? t.totalChunkNum : t.chunkNumForT;
    d.scale = t.scale;
    d.mul0RowNum = t.V == 256 ? 16 : 32;
    d.aicCoreNum = AscendC::GetBlockNum();
    d.hkPartitions = static_cast<uint32_t>(A2FinalizeSelectDqHkPartitions(
        d.B, d.numChunks, d.HV, d.HK, d.aicCoreNum, 1,
        t.isVariable != 0 || d.T % d.BT == 0));
    d.isVarLen = t.isVariable;
    const uint64_t btk = d.aicCoreNum * d.HV * d.BT * d.K * sizeof(bfloat16_t);
    const uint64_t btb = d.aicCoreNum * d.HV * d.BT * d.BT * sizeof(bfloat16_t);
    uint64_t offset = 0;
    d.wsMm3Offset = offset; offset += btb;
    d.wsMm4Offset = offset; offset += btk;
    d.wsMm6Offset = offset; offset += btk;
    d.wsMm5Offset = offset; offset += btk;
    d.wsMm7Offset = offset; offset += btk;
    d.wsDsTempOffset = offset; offset += btb;
    d.wsDgLastOffset = offset;
    return d;
}

__aicore__ inline int64_t A2FinalizeVectorRows(int64_t bt, int64_t cols, int64_t factor)
{
    int64_t row = bt / 2;
    while (row >= 8) {
        const int64_t matrixBytes = row * cols * 2;
        const int64_t scalarBytes = (row * 4 + 31) / 32 * 32;
        const int64_t tempBytes = factor * row * cols * 4 + 3 * scalarBytes + row * 32;
        if (matrixBytes <= 16 * 1024 && tempBytes <= 128 * 1024) {
            break;
        }
        row /= 2;
    }
    return row;
}

__aicore__ inline PrepareWyReprBwdTilingData A2FinalizeWyTiling(
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t)
{
    PrepareWyReprBwdTilingData w{};
    w.B = t.B; w.HV = t.HV; w.HK = t.HK; w.groupSize = t.headRatio;
    w.T = t.T; w.K = t.K; w.V = t.V; w.chunkSize = t.chunkSize;
    w.chunkNum = t.totalChunkNum;
    w.chunkNumPerB = t.isVariable != 0 ? 0 : t.chunkNumForT;
    w.isVariable = t.isVariable;
    w.workspaceBufferCount = 4;
    w.kBytes = t.chunkSize * t.K * sizeof(bfloat16_t);
    w.vBytes = t.chunkSize * t.V * sizeof(bfloat16_t);
    w.mBytes = t.chunkSize * t.chunkSize * sizeof(bfloat16_t);
    int64_t offset = 0;
    w.kbgOffset = offset; offset += w.kBytes;
    w.vbOffset = offset; offset += w.vBytes;
    w.kbetaOffset = offset; offset += w.kBytes;
    w.dkbgOffset = offset; offset += w.kBytes;
    w.dvbOffset = offset; offset += w.vBytes;
    w.da1Offset = offset; offset += w.mBytes;
    w.da2Offset = offset; offset += w.mBytes;
    w.da4Offset = offset; offset += w.mBytes;
    w.da5Offset = offset; offset += w.mBytes;
    w.da6Offset = offset; offset += w.mBytes;
    w.dOffset = offset; offset += w.mBytes;
    w.dkbOffset = offset; offset += w.kBytes;
    w.dkOffset = offset; offset += w.kBytes;
    w.workspaceSlotSize = offset;
    w.kktOffset = 4 * offset;
    w.workspaceCoreSize = w.kktOffset + 4 * w.mBytes;
    w.kVecRow = A2FinalizeVectorRows(t.chunkSize, t.K, 2);
    w.vVecRow = A2FinalizeVectorRows(t.chunkSize, t.V, 2);
    w.mVecRow = A2FinalizeVectorRows(t.chunkSize, t.chunkSize, 4);
    w.kktVecRow = A2FinalizeVectorRows(t.chunkSize, t.chunkSize, 3);
    return w;
}

template <typename DT>
__aicore__ inline void A2FinalizeCopyIn(
    const AscendC::LocalTensor<DT> &dst, const AscendC::GlobalTensor<DT> &src,
    int64_t offset, uint32_t count)
{
    AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(DT)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src[offset], copy, pad);
}

template <typename DT>
__aicore__ inline void A2FinalizeCopyOut(
    const AscendC::GlobalTensor<DT> &dst, const AscendC::LocalTensor<DT> &src,
    int64_t offset, uint32_t count)
{
    AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(DT)), 0, 0, 0};
    AscendC::DataCopyPad(dst[offset], src, copy);
}

// S0：仅转换 g/beta 的计算 dtype 和 state_v_first 布局，不改变公开输入。
template <typename QT, typename GT, bool USE_EXP2>
__aicore__ inline void A2FinalizePrepare(
    GM_ADDR g, GM_ADDR beta, GM_ADDR h, GM_ADDR dh, GM_ADDR workspace,
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t,
    const A2FinalizeWorkspaceLayout &layout, AscendC::TPipe &pipe)
{
    constexpr uint32_t TILE = 1024;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf, floatBuf;
    pipe.InitBuffer(inputBuf, TILE * sizeof(float));
    pipe.InitBuffer(floatBuf, TILE * sizeof(float));
    auto input = inputBuf.Get<GT>();
    auto output = floatBuf.Get<float>();
    AscendC::GlobalTensor<GT> gGm, betaGm;
    AscendC::GlobalTensor<float> g32Gm, beta32Gm;
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(g));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(beta));
    g32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + layout.gOffset));
    beta32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + layout.betaOffset));
    const int64_t physicalAiv = AscendC::GetBlockIdx();
    const int64_t aivNum = AscendC::GetBlockNum() * 2;
    const int64_t scalarCount = t.B * t.HV * t.T;
    for (int64_t start = physicalAiv * TILE; start < scalarCount; start += aivNum * TILE) {
        const uint32_t count = static_cast<uint32_t>(scalarCount - start < TILE ? scalarCount - start : TILE);
        A2FinalizeCopyIn(input, gGm, start, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same<GT, float>::value) {
            AscendC::Adds(output, input, 0.0f, count);
        } else {
            AscendC::Cast(output, input, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (USE_EXP2) {
            AscendC::Muls(output, output, A2_FINALIZE_LN2, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        A2FinalizeCopyOut(g32Gm, output, start, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        A2FinalizeCopyIn(input, betaGm, start, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same<GT, float>::value) {
            AscendC::Adds(output, input, 0.0f, count);
        } else {
            AscendC::Cast(output, input, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        A2FinalizeCopyOut(beta32Gm, output, start, count);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    pipe.Reset();

    if (t.stateVFirst != 0) {
        AscendC::TBuf<AscendC::TPosition::VECCALC> stateInputBuf, stateOutputBuf;
        const uint32_t elems = static_cast<uint32_t>(t.K * t.V);
        pipe.InitBuffer(stateInputBuf, elems * sizeof(QT));
        pipe.InitBuffer(stateOutputBuf, elems * sizeof(QT));
        auto stateInput = stateInputBuf.Get<QT>();
        auto stateOutput = stateOutputBuf.Get<QT>();
        AscendC::GlobalTensor<QT> hGm, dhGm, hStandardGm, dhStandardGm;
        hGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(h));
        dhGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(dh));
        hStandardGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(workspace + layout.hOffset));
        dhStandardGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(workspace + layout.dhOffset));
        const int64_t stateCount = t.totalChunkNum * t.HV;
        for (int64_t state = physicalAiv; state < stateCount; state += aivNum) {
            for (int32_t which = 0; which < 2; ++which) {
                auto &src = which == 0 ? hGm : dhGm;
                auto &dst = which == 0 ? hStandardGm : dhStandardGm;
                AscendC::DataCopy(stateInput, src[state * elems], elems);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t ki = 0; ki < static_cast<uint32_t>(t.K); ++ki) {
                    for (uint32_t vi = 0; vi < static_cast<uint32_t>(t.V); ++vi) {
                        stateOutput.SetValue(ki * t.V + vi, stateInput.GetValue(vi * t.K + ki));
                    }
                }
                AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(dst[state * elems], stateOutput, elems);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        pipe.Reset();
    }
}

// S3：每个 (chunk, HV) 仅有一个 AIV owner；scan 严格限制在该 chunk 有效行内。
template <typename GT, bool USE_BETA_SIGMOID>
__aicore__ inline void A2FinalizeGateTail(
    GM_ADDR betaRaw, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace,
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &t,
    const A2FinalizeWorkspaceLayout &layout, AscendC::TPipe &pipe)
{
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 8 * 1024);
    auto left = buf.GetWithOffset<float>(128, 0);
    auto right = buf.GetWithOffset<float>(128, 512);
    auto values = buf.GetWithOffset<float>(128, 1024);
    auto raw = buf.GetWithOffset<GT>(128, 1536);
    auto result = buf.GetWithOffset<GT>(128, 2048);
    auto sigmoid = buf.GetWithOffset<float>(128, 2560);
    auto ones = buf.GetWithOffset<float>(128, 3072);
    auto scanBlocks = buf.GetWithOffset<float>(128 * 8, 3584);
    auto scanOffsets = buf.GetWithOffset<int32_t>(128, 7680);
    auto scanOffsetsU32 = scanOffsets.ReinterpretCast<uint32_t>();
    const uint32_t chunkSize = static_cast<uint32_t>(t.chunkSize);
    // SDK CreateVecIndex pairs its temporary S_V event while this TPipe is live.
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
    AscendC::CreateVecIndex<int32_t>(scanOffsets, int32_t(0), chunkSize);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(scanOffsets, scanOffsets, int32_t(32), chunkSize);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GlobalTensor<float> dg1Gm, dg2Gm, dbetaGm;
    AscendC::GlobalTensor<GT> dbetaOutGm, dgOutGm, betaRawGm;
    dg1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + layout.dgDqOffset));
    dg2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + layout.dgWyOffset));
    dbetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + layout.dbetaOffset));
    dgOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(dg));
    dbetaOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(dbeta));
    if constexpr (USE_BETA_SIGMOID) {
        betaRawGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(betaRaw));
    }
    const auto wyTiling = A2FinalizeWyTiling(t);
    const int64_t aivNum = AscendC::GetBlockNum() * 2;
    for (int64_t work = AscendC::GetBlockIdx(); work < t.totalChunkNum * t.HV; work += aivNum) {
        const int64_t chunk = work / t.HV;
        const int64_t hv = work % t.HV;
        PrepareWyReprBwdTaskInfo task;
        PrepareWyReprBwdGetTaskInfo(cuSeqlens, chunkIndices, wyTiling, chunk, task);
        const uint32_t rows = task.curChunkSize;
        const int64_t base = hv * t.T + task.valueBos;
        // Brcb reads ceil(rows / 8) groups, including initialized tail lanes.
        AscendC::Duplicate(values, 0.0f, chunkSize);
        A2FinalizeCopyIn(left, dg1Gm, base, rows);
        A2FinalizeCopyIn(right, dg2Gm, base, rows);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Add(values, left, right, rows);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (rows > 0) {
            AscendC::SetMaskNorm();
            AscendC::ResetMask();
            AscendC::Brcb(scanBlocks, values, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            // Merge consumed left; its first block is the independent +0 seed.
            AscendC::Duplicate(left, 0.0f, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<float>(0, 8);
            const AscendC::BinaryRepeatParams scanParams;
            for (uint32_t row = rows; row > 0; --row) {
                auto prefix = row == rows ? left : scanBlocks[row * 8];
                AscendC::Add<float, false>(
                    scanBlocks[(row - 1) * 8], prefix, scanBlocks[(row - 1) * 8],
                    AscendC::MASK_PLACEHOLDER, 1, scanParams);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetMaskNorm();
            AscendC::ResetMask();
            AscendC::Gather(values, scanBlocks, scanOffsetsU32, 0, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetMaskNorm();
            AscendC::ResetMask();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same<GT, float>::value) {
            A2FinalizeCopyOut(dgOutGm, values, base, rows);
        } else {
            AscendC::Cast(result, values, AscendC::RoundMode::CAST_RINT, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PipeBarrier<PIPE_ALL>();
            A2FinalizeCopyOut(dgOutGm, result, base, rows);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        A2FinalizeCopyIn(values, dbetaGm, base, rows);
        if constexpr (USE_BETA_SIGMOID) {
            A2FinalizeCopyIn(raw, betaRawGm, base, rows);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (USE_BETA_SIGMOID) {
            if constexpr (std::is_same<GT, float>::value) {
                AscendC::Adds(sigmoid, raw, 0.0f, rows);
            } else {
                AscendC::Cast(sigmoid, raw, AscendC::RoundMode::CAST_NONE, rows);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(sigmoid, sigmoid, -1.0f, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(sigmoid, sigmoid, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(sigmoid, sigmoid, 1.0f, rows);
            AscendC::Duplicate(ones, 1.0f, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(sigmoid, ones, sigmoid, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(ones, ones, sigmoid, rows);
            AscendC::Mul(values, values, sigmoid, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(values, values, ones, rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same<GT, float>::value) {
            if constexpr (USE_BETA_SIGMOID) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            A2FinalizeCopyOut(dbetaOutGm, values, base, rows);
        } else {
            AscendC::Cast(result, values, AscendC::RoundMode::CAST_RINT, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PipeBarrier<PIPE_ALL>();
            A2FinalizeCopyOut(dbetaOutGm, result, base, rows);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    pipe.Reset();
}

template <typename QT>
__aicore__ inline void A2FinalizeNormTile(
    const AscendC::LocalTensor<float> &grad,
    const AscendC::LocalTensor<QT> &input,
    const AscendC::LocalTensor<float> &x,
    const AscendC::LocalTensor<float> &product,
    const AscendC::LocalTensor<float> &sum,
    const AscendC::LocalTensor<float> &reduceWork,
    const AscendC::LocalTensor<float> &broadcast,
    const AscendC::GlobalTensor<float> &rstd, int64_t offset, uint32_t rows)
{
    constexpr uint32_t K = 128;
    AscendC::Cast(x, input, AscendC::RoundMode::CAST_NONE, rows * K);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(product, x, grad, rows * K);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < rows; ++row) {
        // 与 WY 的 128 列归约使用相同布局：每行两个 64 列 partial，行距 32B。
        AscendC::WholeReduceSum(reduceWork[row * 8], product[row * K], 64, 2, 1, 1, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(sum, reduceWork, 2, rows, 1, 1, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(broadcast, sum, (rows + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x, x, broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::Mul(x[64], x[64], broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(grad, grad, x, rows * K);
    AscendC::PipeBarrier<PIPE_ALL>();
    A2FinalizeCopyIn(sum, rstd, offset, rows);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Brcb(broadcast, sum, (rows + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(grad, grad, broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::Mul(grad[64], grad[64], broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::PipeBarrier<PIPE_V>();
}

// S4：独占 (chunk, HK)，合并两条 DK 支路后执行可选 Q/K norm。
template <typename QT, bool USE_QK_L2NORM>
__aicore__ inline void A2FinalizeQkTail(
    GM_ADDR q, GM_ADDR k, GM_ADDR qRstd, GM_ADDR kRstd,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dq, GM_ADDR dk,
    GM_ADDR workspace, const ChunkGatedDeltaRuleBwdFinalizeTilingData &t,
    const A2FinalizeWorkspaceLayout &layout, AscendC::TPipe &pipe)
{
    constexpr uint32_t ROWS = 32;
    constexpr uint32_t K = 128;
    constexpr uint32_t ELEMS = ROWS * K;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 112 * 1024);
    auto qInput = buf.GetWithOffset<QT>(ELEMS, 0);
    auto dqInput = buf.GetWithOffset<QT>(ELEMS, 8 * 1024);
    auto dkInput = buf.GetWithOffset<QT>(ELEMS, 16 * 1024);
    auto dkWyInput = buf.GetWithOffset<QT>(ELEMS, 24 * 1024);
    auto kInput = buf.GetWithOffset<QT>(ELEMS, 32 * 1024);
    auto gradQ = buf.GetWithOffset<float>(ELEMS, 40 * 1024);
    auto gradK = buf.GetWithOffset<float>(ELEMS, 56 * 1024);
    auto x = buf.GetWithOffset<float>(ELEMS, 72 * 1024);
    auto product = buf.GetWithOffset<float>(ELEMS, 88 * 1024);
    auto sum = buf.GetWithOffset<float>(ROWS, 104 * 1024);
    auto reduceWork = buf.GetWithOffset<float>(ROWS * 8, 105 * 1024);
    auto broadcast = buf.GetWithOffset<float>(ROWS * 8, 106 * 1024);
    AscendC::GlobalTensor<QT> qGm, kGm, dq1Gm, dk1Gm, dk2Gm, dqGm, dkGm;
    AscendC::GlobalTensor<float> qRstdGm, kRstdGm;
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(q));
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(k));
    dq1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(workspace + layout.dqOffset));
    dk1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(workspace + layout.dkOffset));
    dk2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(workspace + layout.dkWyOffset));
    dqGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(dq));
    dkGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(dk));
    if constexpr (USE_QK_L2NORM) {
        qRstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
        kRstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
    }
    const auto wyTiling = A2FinalizeWyTiling(t);
    const int64_t aivNum = AscendC::GetBlockNum() * 2;
    for (int64_t work = AscendC::GetBlockIdx(); work < t.totalChunkNum * t.HK; work += aivNum) {
        const int64_t chunk = work / t.HK;
        const int64_t hk = work % t.HK;
        PrepareWyReprBwdTaskInfo task;
        PrepareWyReprBwdGetTaskInfo(cuSeqlens, chunkIndices, wyTiling, chunk, task);
        for (uint32_t row = 0; row < task.curChunkSize; row += ROWS) {
            const uint32_t rows = task.curChunkSize - row < ROWS ? task.curChunkSize - row : ROWS;
            const uint32_t count = rows * K;
            const int64_t scalarOffset = hk * t.T + task.keyBos + row;
            const int64_t offset = scalarOffset * K;
            AscendC::DataCopy(dqInput, dq1Gm[offset], count);
            AscendC::DataCopy(dkInput, dk1Gm[offset], count);
            AscendC::DataCopy(dkWyInput, dk2Gm[offset], count);
            if constexpr (USE_QK_L2NORM) {
                AscendC::DataCopy(qInput, qGm[offset], count);
                AscendC::DataCopy(kInput, kGm[offset], count);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(gradK, dkInput, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(x, dkWyInput, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(gradK, gradK, x, count);
            AscendC::PipeBarrier<PIPE_V>();
            // 原组合链的 add 输出为 BF16；norm 输入保留该 cast 边界。
            AscendC::Cast(dkInput, gradK, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (USE_QK_L2NORM) {
                AscendC::Cast(gradQ, dqInput, AscendC::RoundMode::CAST_NONE, count);
                AscendC::Cast(gradK, dkInput, AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                A2FinalizeNormTile(gradQ, qInput, x, product, sum, reduceWork, broadcast,
                                   qRstdGm, scalarOffset, rows);
                A2FinalizeNormTile(gradK, kInput, x, product, sum, reduceWork, broadcast,
                                   kRstdGm, scalarOffset, rows);
                AscendC::Cast(dqInput, gradQ, AscendC::RoundMode::CAST_RINT, count);
                AscendC::Cast(dkInput, gradK, AscendC::RoundMode::CAST_RINT, count);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // 等待最终 Cast 写完 UB，再允许 MTE3 读取同一输出缓冲。
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(dqGm[offset], dqInput, count);
            AscendC::DataCopy(dkGm[offset], dkInput, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    pipe.Reset();
}

template <typename QT, typename GT, uint32_t V_DIM, uint32_t CHUNK_SIZE,
          bool USE_QK_L2NORM, bool USE_BETA_SIGMOID, bool USE_EXP2>
__aicore__ inline void A2FinalizeRun(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR dO, GM_ADDR du,
    GM_ADDR g, GM_ADDR beta, GM_ADDR h, GM_ADDR dh, GM_ADDR a,
    GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR betaRaw,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg,
    GM_ADDR workspace, const ChunkGatedDeltaRuleBwdFinalizeTilingData &t)
{
    const A2FinalizeWorkspaceLayout layout(t, AscendC::GetBlockNum());
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        A2FinalizePrepare<QT, GT, USE_EXP2>(g, beta, h, dh, workspace, t, layout, pipe);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<false>();
    GM_ADDR standardH = t.stateVFirst != 0 ? workspace + layout.hOffset : h;
    GM_ADDR standardDh = t.stateVFirst != 0 ? workspace + layout.dhOffset : dh;
    const auto dqTiling = A2FinalizeDqTiling(t);
    const bool dqWideSchedule = !DqkwgUseU32Schedule(
        dqTiling.B, dqTiling.numChunks, dqTiling.HV, dqTiling.HK,
        dqTiling.aicCoreNum, dqTiling.hkPartitions);
    GM_ADDR stageWorkspace = workspace + layout.stageOffset;
    if ASCEND_IS_AIC {
        ChunkBwdDqkwgCubeProcess<QT, float> cube(
            q, k, vNew, standardH, dO, standardDh, du, cuSeqlens, chunkIndices,
            workspace + layout.dqOffset, workspace + layout.dkOffset,
            workspace + layout.dwOffset, stageWorkspace);
        cube.Init(dqTiling);
        if (dqWideSchedule) {
            cube.template Process<true>();
        } else {
            cube.template Process<false>();
        }
        // 原 Cube Process 已消费末轮 3 个 clear 信用，不能在这里重复 drain。
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkBwdDqkwgVectorProcess<QT, float> vec(
            q, k, vNew, workspace + layout.gOffset, standardH, dO, standardDh, du,
            cuSeqlens, chunkIndices, workspace + layout.dqOffset,
            workspace + layout.dkOffset, workspace + layout.dwOffset,
            workspace + layout.dgDqOffset, stageWorkspace);
        vec.Init(dqTiling, &pipe);
        if (dqWideSchedule) {
            vec.template Process<true>();
        } else {
            vec.template Process<false>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe.Reset();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<false>();
    const auto wyTiling = A2FinalizeWyTiling(t);
    if ASCEND_IS_AIC {
        PrepareWyReprBwdCubeProcess<QT, float, V_DIM, CHUNK_SIZE> cube(
            k, a, workspace + layout.dwOffset, du, cuSeqlens, chunkIndices, stageWorkspace);
        cube.Init(wyTiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        PrepareWyReprBwdVectorProcess<QT, float, V_DIM, CHUNK_SIZE> vec(
            k, v, workspace + layout.betaOffset, workspace + layout.gOffset,
            cuSeqlens, chunkIndices, workspace + layout.dkWyOffset, dv,
            workspace + layout.dbetaOffset, workspace + layout.dgWyOffset, stageWorkspace);
        vec.Init(wyTiling, &pipe);
        vec.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe.Reset();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        A2FinalizeGateTail<GT, USE_BETA_SIGMOID>(
            betaRaw, cuSeqlens, chunkIndices, dbeta, dg, workspace, t, layout, pipe);
        A2FinalizeQkTail<QT, USE_QK_L2NORM>(
            q, k, qRstd, kRstd, cuSeqlens, chunkIndices, dq, dk, workspace, t, layout, pipe);
    }
}

} // namespace GDN

#endif
