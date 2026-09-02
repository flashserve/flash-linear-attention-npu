#ifndef CHUNK_KDA_BWD_FINALIZE_GATE_H
#define CHUNK_KDA_BWD_FINALIZE_GATE_H

#include "chunk_kda_bwd_finalize_wy.h"

#ifndef CHUNK_KDA_BWD_C_GATE_H
#define CHUNK_KDA_BWD_C_GATE_H
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_utils/vector/regbase.hpp"
#endif

namespace KDA {

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using namespace AscendC::MicroAPI;

constexpr CastTrait kKdaGateBf16ToFp32 = {
    RegLayout::ZERO,
    SatMode::SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

static __simd_vf__ inline void KdaBwdCCastBf16ToFp32A5(
    __ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint16_t count)
{
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<bfloat16_t> srcReg;
    RegTensor<float> dstReg;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kRegElements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(
            srcReg, src + offset);
        Cast<float, bfloat16_t, kKdaGateBf16ToFp32>(
            dstReg, srcReg, mask);
        DataCopy(dst + offset, dstReg, mask);
    }
}

static __simd_vf__ inline void KdaBwdCGateFillA5(
    __ubuf__ float *dst, float value, uint16_t count)
{
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> valueReg;
    Duplicate(valueReg, value);
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += kRegElements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(dst + offset, valueReg, mask);
    }
}

// Reverse inclusive scan over [rows, 128].  Two 64-column register blocks
// keep the running sum resident while walking rows once.  This replaces the
// six full-tile Hillis-Steele passes used by the portable path.
static __simd_vf__ inline void KdaBwdCGateReverseScanA5(
    __ubuf__ float *dst, __ubuf__ float *src, uint16_t rows)
{
    constexpr uint32_t kCols = 128;
    constexpr uint32_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> value;
    RegTensor<float> accumulator;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < kCols; col += kRegCols) {
        Duplicate(accumulator, 0.0f, mask);
        for (uint32_t row = rows; row > 0; --row) {
            const uint32_t offset = (row - 1U) * kCols + col;
            DataCopy(value, src + offset);
            Add(accumulator, accumulator, value, mask);
            DataCopy(dst + offset, accumulator, mask);
        }
    }
}

template <bool HAS_BIAS, bool ZERO_CARRY = false,
          bool SCAN_UPSTREAM = true>
static __simd_vf__ inline void KdaBwdCSafeGateBackwardA5(
    __ubuf__ float *dg, __ubuf__ float *dbAcc, __ubuf__ float *dAAcc,
    __ubuf__ float *rawGate, __ubuf__ float *upstream,
    __ubuf__ float *bias, __ubuf__ float *upstreamCarry,
    uint16_t rows, float expA, float lowerBound)
{
    constexpr uint32_t kCols = 128;
    constexpr uint32_t kRegCols =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> one;
    Duplicate(one, 1.0f, mask);
    const float chainScale = lowerBound * expA;
    for (uint32_t col = 0; col < kCols; col += kRegCols) {
        RegTensor<float> dbReg;
        RegTensor<float> dASum;
        RegTensor<float> upstreamAcc;
        if constexpr (HAS_BIAS) {
            DataCopy(dbReg, dbAcc + col);
        }
        Duplicate(dASum, 0.0f, mask);
        if constexpr (SCAN_UPSTREAM && ZERO_CARRY) {
            // Gate post resets the reverse cumsum at every KDA chunk.  Avoid
            // a UB fill followed immediately by a second SIMD helper, which
            // could cold-read stale carry contents on A5.
            Duplicate(upstreamAcc, 0.0f, mask);
        } else if constexpr (SCAN_UPSTREAM) {
            DataCopy(upstreamAcc, upstreamCarry + col);
        }
        RegTensor<float> biasReg;
        if constexpr (HAS_BIAS) {
            DataCopy(biasReg, bias + col);
        }
        for (uint32_t rowIter = rows; rowIter > 0; --rowIter) {
            const uint32_t row = rowIter - 1U;
            const uint32_t offset = row * kCols + col;
            RegTensor<float> raw;
            RegTensor<float> denominator;
            RegTensor<float> sigmoid;
            RegTensor<float> oneMinus;
            RegTensor<float> upstreamReg;
            RegTensor<float> gradient;
            RegTensor<float> dAReg;
            DataCopy(raw, rawGate + offset);
            if constexpr (HAS_BIAS) {
                Add(raw, raw, biasReg, mask);
            }
            Muls(denominator, raw, -expA, mask);
            Exp(denominator, denominator, mask);
            Adds(denominator, denominator, 1.0f, mask);
            Div(sigmoid, one, denominator, mask);
            Muls(oneMinus, sigmoid, -1.0f, mask);
            Adds(oneMinus, oneMinus, 1.0f, mask);
            Mul(gradient, oneMinus, sigmoid, mask);
            DataCopy(upstreamReg, upstream + offset);
            if constexpr (SCAN_UPSTREAM) {
                Add(upstreamAcc, upstreamAcc, upstreamReg, mask);
                Mul(gradient, gradient, upstreamAcc, mask);
            } else {
                Mul(gradient, gradient, upstreamReg, mask);
            }
            Muls(gradient, gradient, chainScale, mask);
            Mul(dAReg, gradient, raw, mask);
            if constexpr (HAS_BIAS) {
                Add(dbReg, dbReg, gradient, mask);
            }
            Add(dASum, dASum, dAReg, mask);
            DataCopy(dg + offset, gradient, mask);
        }
        if constexpr (HAS_BIAS) {
            DataCopy(dbAcc + col, dbReg, mask);
        }
        if constexpr (SCAN_UPSTREAM && !ZERO_CARRY) {
            DataCopy(upstreamCarry + col, upstreamAcc, mask);
        }
        RegTensor<float> dABlock;
        RegTensor<float> dATotal;
        ReduceSum(dABlock, dASum, mask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(dATotal, dAAcc);
        Add(dATotal, dATotal, dABlock, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dAAcc, dATotal, mask);
    }
}

#endif

template <bool SAFE_GATE, typename RawGateT>
class ChunkKdaBwdCGateProcess {
public:
    static constexpr uint32_t kChunkBytes =
        64 * 128 * sizeof(float);
    static_assert(5 * kChunkBytes + 4 * 1024 <= 192 * 1024,
                  "Kernel C raw-gate AIV buffers exceed UB capacity");
    __aicore__ ChunkKdaBwdCGateProcess(
        GM_ADDR dg, GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR dA, GM_ADDR dBias, GM_ADDR cuSeqlens,
        GM_ADDR chunkIndices)
        : dg_(dg), rawG_(rawG), aLog_(aLog), dtBias_(dtBias),
          dA_(dA), dBias_(dBias), cuSeqlens_(cuSeqlens),
          chunkIndices_(chunkIndices)
    {
    }

    __aicore__ inline void Init(
        const ChunkKdaBwdCTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg_));
        if (tiling_.useGateInKernel != 0) {
            rawGFloatGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(rawG_));
            rawGBf16Gm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t *>(rawG_));
            aLogGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(aLog_));
            if (tiling_.hasDtBias != 0) {
                dtBiasGm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float *>(dtBias_));
            }
            dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA_));
            if (tiling_.hasDtBias != 0) {
                dBiasGm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float *>(dBias_));
            }
        }

        pipe_->InitBuffer(input_, kChunkBytes);
        pipe_->InitBuffer(scan_, kChunkBytes);
        if (tiling_.useGateInKernel != 0) {
            pipe_->InitBuffer(raw_, kChunkBytes);
            pipe_->InitBuffer(tmp_, kChunkBytes);
            pipe_->InitBuffer(aux_, kChunkBytes);
            pipe_->InitBuffer(reduce_, 4 * 1024);
        }
        vToMte3_ =
            pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
        mte3ToV_ =
            pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
        mte3ToMte2_ =
            pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        if (tiling_.useGateInKernel != 0) {
            ProcessRawGate();
        } else {
            ProcessAccumulatedGate();
        }
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2_);
    }

private:
    __aicore__ inline void ProcessAccumulatedGate()
    {
        const uint32_t core =
            AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t headWindows =
            (static_cast<uint32_t>(tiling_.headNum) + 1U) / 2U;
        const uint64_t groups =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindows;
        for (uint64_t group = core; group < groups;
             group += tiling_.usedCoreNum) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(group / headWindows);
            const uint32_t headBase =
                static_cast<uint32_t>(group % headWindows) * 2U;
            const uint32_t headCount =
                headBase + 1U < static_cast<uint32_t>(tiling_.headNum) ?
                    2U : 1U;
            // The MIX launch has two AIV sub-blocks for one logical AIC
            // owner.  Gate updates dg in place, so the two AIVs must own
            // disjoint heads instead of scanning and storing both twice.
            const uint32_t laneBegin = AscendC::GetSubBlockIdx();
            const uint32_t laneEnd =
                laneBegin < headCount ? laneBegin + 1U : laneBegin;
            for (uint32_t lane = laneBegin; lane < laneEnd; ++lane) {
                ProcessChunk(
                    taskIdx, headBase + lane, false, nullptr, nullptr, 0.0f);
            }
        }
    }

    __aicore__ inline void ProcessRawGate()
    {
        const uint32_t core =
            AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t headWindows =
            (static_cast<uint32_t>(tiling_.headNum) + 1U) / 2U;
        for (uint32_t headWindow = core; headWindow < headWindows;
             headWindow += tiling_.usedCoreNum) {
          const uint32_t headBase = headWindow * 2U;
          const uint32_t headCount =
              headBase + 1U < static_cast<uint32_t>(tiling_.headNum) ?
                  2U : 1U;
          const uint32_t laneBegin = AscendC::GetSubBlockIdx();
          const uint32_t laneEnd =
              laneBegin < headCount ? laneBegin + 1U : laneBegin;
          for (uint32_t lane = laneBegin; lane < laneEnd; ++lane) {
            const uint32_t head = headBase + lane;
            auto dbAcc = reduce_.Get<float>();
            auto dAAcc = dbAcc[128];
            AscendC::Duplicate(dbAcc, 0.0f, 128);
            AscendC::Duplicate(dAAcc, 0.0f, 8);
            AscendC::PipeBarrier<PIPE_V>();

            // A_log is head-wise and is shared by every chunk.  Materialize
            // exp(A_log) once here instead of repeating a scalar GM load,
            // Exp and V->S hand-off for every chunk of a long sequence.
            auto aScalar = reduce_.Get<float>()[136];
            AscendC::DataCopyPad(
                aScalar, aLogGm_[head],
                {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Exp(aScalar, aScalar, 1);
            AscendC::PipeBarrier<PIPE_V>();
            const float expA = ReadScalar(aScalar);

            if (tiling_.isVarLen != 0) {
                for (uint32_t task = 0;
                     task < static_cast<uint32_t>(tiling_.chunkNum);
                     ++task) {
                    ProcessChunk(
                        task, head, true, &dAAcc, &dbAcc, expA);
                }
            } else {
                for (uint32_t batch = 0;
                     batch < static_cast<uint32_t>(tiling_.batch);
                     ++batch) {
                    for (uint32_t local = 0;
                         local < static_cast<uint32_t>(
                             tiling_.chunkNumPerBatch);
                         ++local) {
                        const uint32_t task =
                            batch * static_cast<uint32_t>(
                                tiling_.chunkNumPerBatch) + local;
                        ProcessChunk(
                            task, head, true, &dAAcc, &dbAcc, expA);
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::DataCopyPad(
                dAGm_[head], dAAcc,
                {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
            if (tiling_.hasDtBias != 0) {
                AscendC::DataCopy(
                    dBiasGm_[static_cast<uint64_t>(head) * 128],
                    dbAcc, 128);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
          }
        }
    }

    __aicore__ inline void ProcessChunk(
        uint32_t taskIdx, uint32_t head, bool applyRaw,
        AscendC::LocalTensor<float> *dAAcc,
        AscendC::LocalTensor<float> *dbAcc, float expA)
    {
        const WyChunkTask task = GetWyChunkTask(
            cuSeqlens_, chunkIndices_, tiling_, taskIdx);
        const uint32_t validC = task.end - task.begin;
        const uint32_t count = validC * 128;
        const uint64_t offset = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 128);
        auto src = input_.Get<float>();
        auto dst = scan_.Get<float>();
        AscendC::DataCopyPad(
            src, dgGm_[offset],
            {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE) {
            // A dense sequence made entirely of full chunks never switches
            // between the fused and split helpers, so the fast one-pass
            // register recurrence has a uniform lifecycle for every task.
            const bool denseFullChunks =
                tiling_.isVarLen == 0 &&
                static_cast<uint32_t>(tiling_.seqlen) %
                    static_cast<uint32_t>(tiling_.chunkSize) == 0;
            if (applyRaw && denseFullChunks) {
                ApplyRawGate<true>(
                    task, head, validC, src, *dAAcc, *dbAcc, expA);
                return;
            }
        }
#endif
        // Use a six-step UB scan on A5 as well.  The former single-pass
        // register recurrence carried one accumulator through a dynamic row
        // loop and could intermittently lose a dependency on cold launches.
        bool srcIsInput = true;
        for (uint32_t step = 1; step < validC; step <<= 1U) {
            auto in = srcIsInput ? src : dst;
            auto out = srcIsInput ? dst : src;
            const uint32_t activeRows = validC - step;
            AscendC::Add(
                out, in, in[step * 128], activeRows * 128);
            AscendC::Adds(
                out[activeRows * 128], in[activeRows * 128], 0.0f,
                step * 128);
            AscendC::PipeBarrier<PIPE_V>();
            srcIsInput = !srcIsInput;
        }
        auto upstream = srcIsInput ? src : dst;
        if (applyRaw) {
            ApplyRawGate<false>(
                task, head, validC, upstream, *dAAcc, *dbAcc, expA);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (SAFE_GATE) {
                // The next chunk reloads dA/db from the same UB accumulator
                // addresses in a new SIMD helper invocation.
                AscendC::PipeBarrier<PIPE_V>();
            }
#endif
        } else {
            StoreDg(offset, upstream, count);
        }
    }

    template <bool SCAN_SAFE_GATE>
    __aicore__ inline void ApplyRawGate(
        const WyChunkTask &task, uint32_t head, uint32_t rows,
        AscendC::LocalTensor<float> upstream,
        AscendC::LocalTensor<float> dAAcc,
        AscendC::LocalTensor<float> dbAcc, float expA)
    {
        const uint32_t count = rows * 128;
        const uint64_t offset = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 128);
        auto x = raw_.Get<float>();
        auto tmp = tmp_.Get<float>();
        auto aux = aux_.Get<float>();
        if (tiling_.rawGateIsBf16 == 0) {
            AscendC::DataCopyPad(
                x, rawGFloatGm_[offset],
                {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                {false, 0, 0, 0});
        } else {
            // tmp is dead until the chain-rule arithmetic below.  Use its
            // first half as the BF16 staging area and avoid another UB plane.
            auto rawStage = tmp_.Get<bfloat16_t>();
            AscendC::DataCopyPad(
                rawStage, rawGBf16Gm_[offset],
                {1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0},
                {false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            KdaBwdCCastBf16ToFp32A5(
                (__ubuf__ float *)x.GetPhyAddr(),
                (__ubuf__ bfloat16_t *)rawStage.GetPhyAddr(),
                static_cast<uint16_t>(count));
        }
        if (tiling_.hasDtBias != 0) {
            auto bias = reduce_.Get<float>()[144];
            AscendC::DataCopy(
                bias, dtBiasGm_[static_cast<uint64_t>(head) * 128], 128);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        if (tiling_.hasDtBias != 0) {
            auto bias = reduce_.Get<float>()[144];
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!SAFE_GATE) {
#endif
            for (uint32_t row = 0; row < rows; ++row) {
                AscendC::Add(x[row * 128], x[row * 128], bias, 128);
            }
            AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            }
#endif
        }

        if constexpr (SAFE_GATE) {
            const float a = expA;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (tiling_.hasDtBias != 0) {
                auto bias = reduce_.Get<float>()[144];
                KdaBwdCSafeGateBackwardA5<
                    true, SCAN_SAFE_GATE, SCAN_SAFE_GATE>(
                    (__ubuf__ float *)tmp.GetPhyAddr(),
                    (__ubuf__ float *)dbAcc.GetPhyAddr(),
                    (__ubuf__ float *)dAAcc.GetPhyAddr(),
                    (__ubuf__ float *)x.GetPhyAddr(),
                    (__ubuf__ float *)upstream.GetPhyAddr(),
                    (__ubuf__ float *)bias.GetPhyAddr(),
                    (__ubuf__ float *)aux.GetPhyAddr(),
                    static_cast<uint16_t>(rows), a, tiling_.lowerBound);
            } else {
                KdaBwdCSafeGateBackwardA5<
                    false, SCAN_SAFE_GATE, SCAN_SAFE_GATE>(
                    (__ubuf__ float *)tmp.GetPhyAddr(),
                    (__ubuf__ float *)dbAcc.GetPhyAddr(),
                    (__ubuf__ float *)dAAcc.GetPhyAddr(),
                    (__ubuf__ float *)x.GetPhyAddr(),
                    (__ubuf__ float *)upstream.GetPhyAddr(),
                    (__ubuf__ float *)x.GetPhyAddr(),
                    (__ubuf__ float *)aux.GetPhyAddr(),
                    static_cast<uint16_t>(rows), a, tiling_.lowerBound);
            }
#else
            AscendC::Muls(tmp, x, a, count);
            AscendC::PipeBarrier<PIPE_V>();
            Sigmoid(aux, tmp, count);
            AscendC::Muls(tmp, aux, -1.0f, count);
            AscendC::Adds(tmp, tmp, 1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmp, tmp, aux, count);
            AscendC::Mul(tmp, tmp, upstream, count);
            AscendC::Muls(tmp, tmp, tiling_.lowerBound * a, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(aux, tmp, x, count);
#endif
        } else {
            const float a = -expA;
            Sigmoid(aux, x, count);
            AscendC::Mul(tmp, upstream, aux, count);
            AscendC::Muls(tmp, tmp, a, count);
            AscendC::PipeBarrier<PIPE_V>();

            // Stable softplus(x) = max(x,0)+log1p(exp(-abs(x))).
            AscendC::Abs(aux, x, count);
            AscendC::Muls(aux, aux, -1.0f, count);
            AscendC::Exp(aux, aux, count);
            AscendC::Adds(aux, aux, 1.0f, count);
            AscendC::Ln(aux, aux, count);
            AscendC::Maxs(x, x, 0.0f, count);
            AscendC::Add(aux, aux, x, count);
            AscendC::Muls(aux, aux, a, count);
            AscendC::Mul(aux, aux, upstream, count);
            AscendC::PipeBarrier<PIPE_V>();
        }

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
        AccumulateColumns(dbAcc, tmp, rows);
        AccumulateScalar(dAAcc, aux, rows);
#else
        if constexpr (!SAFE_GATE) {
            AccumulateColumns(dbAcc, tmp, rows);
            AccumulateScalar(dAAcc, aux, rows);
        }
#endif
        StoreDg(offset, tmp, count);
    }

    __aicore__ inline void Sigmoid(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src, uint32_t count)
    {
        auto one = reduce_.Get<float>()[800];
        AscendC::Muls(dst, src, -1.0f, count);
        AscendC::Exp(dst, dst, count);
        AscendC::Adds(dst, dst, 1.0f, count);
        AscendC::Duplicate(one, 1.0f, 128);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t offset = 0; offset < count; offset += 128) {
            AscendC::Div(dst[offset], one, dst[offset], 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ReadScalar(
        AscendC::LocalTensor<float> value)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return ((__ubuf__ float *)value.GetPhyAddr())[0];
    }

    __aicore__ inline void AccumulateColumns(
        AscendC::LocalTensor<float> acc,
        AscendC::LocalTensor<float> values, uint32_t rows)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Add(acc, acc, values[row * 128], 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateScalar(
        AscendC::LocalTensor<float> acc,
        AscendC::LocalTensor<float> values, uint32_t rows)
    {
        auto partial = reduce_.Get<float>()[280];
        auto rowSum = reduce_.Get<float>()[280 + 64 * 8];
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::WholeReduceSum(
                partial[row * 8], values[row * 128],
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
        // A5 may silently drop a full 32 KiB UB->GM DataCopyPad transfer.
        // The aligned row-sized stores are also valid on A2/A3.
        const uint32_t rows = count / 128U;
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::DataCopy(
                dgGm_[offset + static_cast<uint64_t>(row) * 128U],
                value[row * 128U], 128U);
        }
        // The next chunk loads into the same UB plane through MTE2.  Waiting
        // on MTE3_V does not protect that overwrite; use the real
        // MTE3-to-MTE2 reuse dependency so every row reaches GM first.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    GM_ADDR dg_;
    GM_ADDR rawG_;
    GM_ADDR aLog_;
    GM_ADDR dtBias_;
    GM_ADDR dA_;
    GM_ADDR dBias_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    ChunkKdaBwdCTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<float> dgGm_;
    AscendC::GlobalTensor<float> rawGFloatGm_;
    AscendC::GlobalTensor<bfloat16_t> rawGBf16Gm_;
    AscendC::GlobalTensor<float> aLogGm_;
    AscendC::GlobalTensor<float> dtBiasGm_;
    AscendC::GlobalTensor<float> dAGm_;
    AscendC::GlobalTensor<float> dBiasGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scan_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> raw_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aux_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_;
    AscendC::TEventID vToMte3_;
    AscendC::TEventID mte3ToV_;
    AscendC::TEventID mte3ToMte2_;
};

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
__aicore__ inline void RunChunkKdaBwdCInitGateOutputsA5(
    GM_ADDR dA, GM_ADDR dBias,
    const ChunkKdaBwdCTilingData &tiling)
{
    if (AscendC::GetBlockIdx() != 0 || AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    AscendC::GlobalTensor<float> dAGm;
    AscendC::GlobalTensor<float> dBiasGm;
    dAGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA));
    if (tiling.hasDtBias != 0) {
        dBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dBias));
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuffer;
    constexpr uint32_t kZeroElements = 1024;
    pipe.InitBuffer(zeroBuffer, kZeroElements * sizeof(float));
    auto zero = zeroBuffer.Get<float>();
    AscendC::Duplicate(zero, 0.0f, kZeroElements);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

    const uint32_t headNum = static_cast<uint32_t>(tiling.headNum);
    AscendC::DataCopyPad(
        dAGm[0], zero,
        {1, headNum * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
    if (tiling.hasDtBias != 0) {
        const uint32_t total = headNum * 128U;
        for (uint32_t offset = 0; offset < total;
             offset += kZeroElements) {
            const uint32_t count =
                offset + kZeroElements <= total ?
                    kZeroElements : total - offset;
            AscendC::DataCopyPad(
                dBiasGm[offset], zero,
                {1, count * static_cast<uint32_t>(sizeof(float)),
                 0, 0, 0});
        }
    }
}
#endif

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_GATE_H


#endif // CHUNK_KDA_BWD_FINALIZE_GATE_H
