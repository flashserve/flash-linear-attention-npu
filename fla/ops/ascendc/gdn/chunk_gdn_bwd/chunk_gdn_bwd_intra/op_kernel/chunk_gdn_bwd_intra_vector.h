/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_VECTOR_H
#define CHUNK_GDN_BWD_INTRA_VECTOR_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <type_traits>
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_utils/vector/regbase.hpp"
#include "chunk_gdn_bwd_intra_common.h"

namespace GDN {

template <typename MainT, typename GateT, typename BetaT, bool USE_EXP2>
__simd_vf__ inline void ChunkGdnBwdIntraStage1VF(
    __ubuf__ MainT *aBgOut, __ubuf__ MainT *aBetaOut, __ubuf__ MainT *dOut,
    __ubuf__ MainT *aIn, __ubuf__ float *scoreIn, __ubuf__ GateT *gIn,
    __ubuf__ GateT *gRowIn, __ubuf__ BetaT *betaIn,
    uint16_t rows, uint16_t rowBegin,
    uint16_t validTokens, float scale)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ROW_ELEMENTS = 64;
    constexpr float LN2 = 0.6931471805599453f;

    RegTensor<GateT> gRawReg;
    RegTensor<BetaT> betaRawReg;
    RegTensor<MainT> aReg, aBgReg, aBetaReg, dReg;
    RegTensor<float> gZero, gOne, betaZero, betaOne, gateZero, gateOne;
    RegTensor<float> aZero, aOne, scoreZero, scoreOne;
    RegTensor<float> deltaZero, deltaOne, resultZero, resultOne;
    RegTensor<float> rowG, rowIndex, validLimit, scaleReg;
    RegTensor<half> colIndexRaw;
    RegTensor<float> colIndexZero, colIndexOne;
    MaskReg validCausalZero, validCausalOne;
    MaskReg validTailZero, validTailOne;
    MaskReg validZero, validOne;
    MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFp16 = CreateMask<half, MaskPattern::ALL>();
    uint32_t maskMainElements = ROW_ELEMENTS;
    uint32_t maskGateElements = ROW_ELEMENTS;
    uint32_t maskBetaElements = ROW_ELEMENTS;
    MaskReg maskMain = UpdateMask<MainT>(maskMainElements);
    MaskReg maskGate = UpdateMask<GateT>(maskGateElements);
    MaskReg maskBeta = UpdateMask<BetaT>(maskBetaElements);

    Duplicate(scaleReg, scale, maskFp32);
    Duplicate(rowIndex, static_cast<float>(rowBegin), maskFp32);
    Arange(colIndexRaw, 0);
    CastHalf2Float<half>(colIndexZero, colIndexOne, colIndexRaw, maskFp16);
    Duplicate(validLimit, static_cast<float>(validTokens), maskFp32);
    CompareTwoReg<float, CMPMODE::LT>(
        validTailZero, validTailOne, colIndexZero, colIndexOne,
        validLimit, validLimit, maskFp32);

    // g/beta 只搬入一次，在寄存器中统一转为 FP32 完成指数和逐元素计算。
    if constexpr (std::is_same<GateT, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(gZero, gOne, gIn);
    } else {
        LoadIn<GateT, false>(gRawReg, gIn);
        CastHalf2Float<GateT>(gZero, gOne, gRawReg, maskGate);
    }
    if constexpr (std::is_same<BetaT, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(betaZero, betaOne, betaIn);
    } else {
        LoadIn<BetaT, false>(betaRawReg, betaIn);
        CastHalf2Float<BetaT>(betaZero, betaOne, betaRawReg, maskBeta);
    }

    // exp2(x) 通过 exp(x * ln2) 实现，随后与 beta 合成 A_bg 的列缩放系数。
    Adds(gateZero, gZero, 0.0f, maskFp32);
    Adds(gateOne, gOne, 0.0f, maskFp32);
    if constexpr (USE_EXP2) {
        Muls(gateZero, gateZero, LN2, maskFp32);
        Muls(gateOne, gateOne, LN2, maskFp32);
    }
    ExpFloatTwoReg(gateZero, gateOne, gateZero, gateOne, maskFp32);
    MulFloatTwoReg(gateZero, gateOne, gateZero, gateOne,
                   betaZero, betaOne, maskFp32);

    // 逐行同时生成 A_bg、A_beta 和 D；因果区及尾块之外的 D 保持为零。
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        LoadIn<MainT, false>(aReg, aIn + rowOffset);
        CastHalf2Float<MainT>(aZero, aOne, aReg, maskMain);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            scoreZero, scoreOne, scoreIn + rowOffset);

        Mul(resultZero, aZero, gateZero, validTailZero);
        Mul(resultOne, aOne, gateOne, validTailOne);
        CastFloat2Half<MainT>(aBgReg, resultZero, resultOne, maskFp32);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(
            aBgOut + static_cast<uint32_t>(row) * 16,
            aBgReg, 32, maskMain);

        Mul(resultZero, aZero, betaZero, validTailZero);
        Mul(resultOne, aOne, betaOne, validTailOne);
        CastFloat2Half<MainT>(aBetaReg, resultZero, resultOne, maskFp32);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(
            aBetaOut + static_cast<uint32_t>(row) * 16,
            aBetaReg, 32, maskMain);

        LoadIn<GateT, true>(gRawReg, gRowIn + row);
        HalfOrFloat2Float<GateT>(rowG, gRawReg, maskFp16, maskFp32);
        CompareTwoReg<float, CMPMODE::GE>(
            validCausalZero, validCausalOne, colIndexZero, colIndexOne,
            rowIndex, rowIndex, maskFp32);
        And(validZero, validCausalZero, validTailZero, maskFp32);
        And(validOne, validCausalOne, validTailOne, maskFp32);
        Sub(deltaZero, gZero, rowG, validZero);
        Sub(deltaOne, gOne, rowG, validOne);
        if constexpr (USE_EXP2) {
            Muls(deltaZero, deltaZero, LN2, validZero);
            Muls(deltaOne, deltaOne, LN2, validOne);
        }
        Exp(deltaZero, deltaZero, validZero);
        Exp(deltaOne, deltaOne, validOne);
        Mul(resultZero, scoreZero, deltaZero, validZero);
        Mul(resultOne, scoreOne, deltaOne, validOne);
        Mul(resultZero, resultZero, scaleReg, validZero);
        Mul(resultOne, resultOne, scaleReg, validOne);
        CastFloat2Half<MainT>(dReg, resultZero, resultOne, maskFp32);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(
            dOut + static_cast<uint32_t>(row) * 16,
            dReg, 32, maskMain);
        Adds(rowIndex, rowIndex, 1.0f, maskFp32);
    }
}

template <typename MainT>
__simd_vf__ inline void ChunkGdnBwdIntraPackMatrixVF(
    __ubuf__ MainT *nzOut, __ubuf__ MainT *ndIn, uint16_t rows)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ROW_ELEMENTS = 128;
    constexpr uint32_t REG_ELEMENTS = 64;
    constexpr uint32_t COL_BLOCKS_PER_REG = 4;
    constexpr uint32_t NZ_ROWS = 32;
    RegTensor<MainT> dataReg;
    MaskReg mask = CreateMask<MainT, MaskPattern::ALL>();

    // 每个 AIV 将自己负责的 32 行从 ND 重排为 Cube 可直接读取的 NZ 半片。
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t ndRow = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        const uint32_t nzRow = static_cast<uint32_t>(row) * 16;
        for (uint32_t colPart = 0; colPart < 2; ++colPart) {
            const uint32_t ndOffset = ndRow + colPart * REG_ELEMENTS;
            const uint32_t nzOffset =
                nzRow + colPart * COL_BLOCKS_PER_REG * NZ_ROWS * 16;
            LoadIn<MainT, false>(dataReg, ndIn + ndOffset);
            StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(
                nzOut + nzOffset, dataReg, NZ_ROWS, mask);
        }
    }
}

template <typename MainT, typename GateT, typename BetaT, uint32_t G>
class ChunkGdnBwdIntraVector {
public:
    using ArchTag = Catlass::Arch::Ascend950;

    __aicore__ inline void Init(
        GM_ADDR v, GM_ADDR dO, GM_ADDR a, GM_ADDR g, GM_ADDR beta, GM_ADDR dvLocal,
        GM_ADDR workspace, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGdnBwdIntraTilingData *__restrict tiling)
    {
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(v));
        dOGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dO));
        vGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dOGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(a));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        (void)dvLocal;
        (void)workspace;
        mapper_.Init(cuSeqlens, chunkIndices, tiling);
        tiling_ = tiling;
        const int64_t subblocks = static_cast<int64_t>(AscendC::GetSubBlockNum());
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx()) / subblocks;
        part_ = static_cast<int64_t>(AscendC::GetSubBlockIdx());
    }

    __aicore__ inline void Process()
    {
        InitializePipeline();

        // 跨核同步闭环：AIC 发布 Score -> AIV 消费并归还 Score；AIV 发布共享 L1
        // record/v/d_o -> AIC 消费并归还。Mutex 只保护本 AIV 的 MTE2/V/MTE3 UB 流水。
        // 这些状态只描述共享 L1 的 v/d_o 双缓冲占用，不参与数学计算。
        bool vdoSlotUsed[INTRA_VDO_TRACK_SLOTS] = {};
        uint32_t lastVdoFlag[INTRA_VDO_TRACK_SLOTS] = {};
        const int64_t blockDim = static_cast<int64_t>(AscendC::GetBlockNum());
        const int64_t workCount = tiling_->workCount;

        // 每个 AIV 按 blockDim 步长处理多个 work；完整四 head 切片使用两两成组预取。
        if (coreIdx_ < workCount) {
            ChunkGdnBwdIntraWorkMeta meta{};
            mapper_.ResolveChunkMajor(coreIdx_, meta);
            bool firstPairPrefetched = false;
            for (int64_t work = coreIdx_; work < workCount; work += blockDim) {
                ChunkGdnBwdIntraWorkMeta nextMeta{};
                const int64_t nextWork = work + blockDim;
                const bool hasNext = nextWork < workCount;
                if (hasNext) {
                    mapper_.ResolveChunkMajor(nextWork, nextMeta);
                }
                const bool usesPairInputs =
                    meta.validTokens == tiling_->chunkSize &&
                    meta.validHeads == INTRA_CG_MAX;
                const bool prefetchNextFirstPair = usesPairInputs && hasNext &&
                    nextMeta.validTokens == tiling_->chunkSize &&
                    nextMeta.validHeads == INTRA_CG_MAX;
                ProcessWork(
                    meta, vdoSlotUsed, lastVdoFlag,
                    prefetchNextFirstPair ? &nextMeta : nullptr, firstPairPrefetched);
                firstPairPrefetched = prefetchNextFirstPair;
                if (hasNext) {
                    meta = nextMeta;
                }
            }
        }

        FinalizePipeline();
    }

private:
    // 编译期布局常量放在函数前，保证函数签名和函数体都可直接使用。
    static constexpr uint32_t INTRA_A_UB_BASE = 32 * 1024;
    static constexpr uint32_t INTRA_WORKSPACE_UB_BASE = 48 * 1024;
    static constexpr uint32_t INTRA_D_UB_OFFSET = 0;
    static constexpr uint32_t INTRA_A_BG_UB_OFFSET = 4 * 1024;
    static constexpr uint32_t INTRA_A_BETA_UB_OFFSET = 8 * 1024;
    static constexpr uint32_t INTRA_WORKSPACE_HALF_BYTES = 12 * 1024;
    static constexpr uint32_t INTRA_RAW_UB_BASE = 72 * 1024;
    static constexpr uint32_t INTRA_RAW_SLOT_BYTES = 512;
    static constexpr uint32_t INTRA_RAW_BETA_OFFSET = 256;
    static constexpr uint32_t INTRA_V_INPUT_UB_BASE = 74 * 1024;
    static constexpr uint32_t INTRA_DO_INPUT_UB_BASE = 106 * 1024;
    static constexpr uint32_t INTRA_VDO_NZ_UB_BASE = 138 * 1024;
    static constexpr uint32_t INTRA_VDO_NZ_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t INTRA_VDO_TRACK_SLOTS = 2;
    static constexpr uint32_t L1_RECORD_BASE = 128 * 1024;
    static constexpr uint32_t L1_RECORD_SLOT_BYTES = 24 * 1024;
    static constexpr uint32_t L1_MATRIX_SLOT_BYTES = 8 * 1024;
    static constexpr uint32_t L1_V_BASE = 224 * 1024;
    static constexpr uint32_t L1_DO_BASE = 256 * 1024;
    static constexpr uint32_t L1_SLOT_BYTES = 16 * 1024;

    static_assert(INTRA_A_UB_BASE + INTRA_CG_MAX * INTRA_MATRIX_HALF_BYTES ==
                      INTRA_WORKSPACE_UB_BASE,
                  "A slots must end where Stage 1 workspace begins.");
    static_assert(INTRA_WORKSPACE_UB_BASE + 2 * INTRA_WORKSPACE_HALF_BYTES ==
                      INTRA_RAW_UB_BASE,
                  "Stage 1 workspace ping-pong must end where raw inputs begin.");
    static_assert(INTRA_RAW_UB_BASE + INTRA_CG_MAX * INTRA_RAW_SLOT_BYTES ==
                      INTRA_V_INPUT_UB_BASE,
                  "Raw input slots must end where v inputs begin.");
    static_assert(INTRA_V_INPUT_UB_BASE + INTRA_CG_MAX * INTRA_VDO_INPUT_HALF_BYTES ==
                      INTRA_DO_INPUT_UB_BASE,
                  "v inputs must end where dO inputs begin.");
    static_assert(INTRA_DO_INPUT_UB_BASE + INTRA_CG_MAX * INTRA_VDO_INPUT_HALF_BYTES ==
                      INTRA_VDO_NZ_UB_BASE,
                  "dO inputs must end where packed v/dO outputs begin.");
    static_assert(INTRA_VDO_NZ_UB_BASE + 2 * INTRA_VDO_NZ_SLOT_BYTES <= 248 * 1024,
                  "General Vector UB layout must fit in 248 KiB.");

    __aicore__ inline void InitializePipeline()
    {
        // inputMutex: MTE2 写输入 UB -> Vector 读取 -> 下一轮 MTE2 覆盖。
        // outputMutex: Vector 写 ping/pong UB -> MTE3 读取 -> 下一轮 Vector 覆盖。
        for (uint32_t slot = 0; slot < INTRA_CG_MAX; ++slot) {
            inputMutex_[slot] = AscendC::AllocMutexID();
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            outputMutex_[slot] = AscendC::AllocMutexID();
        }
        rawInitMutex_ = AscendC::AllocMutexID();

        InitializeRawSlots();
        // raw 区由 Vector 清零一次；首次 MTE2 覆盖前闭合 V -> MTE2 依赖。
        AscendC::Mutex::Lock<PIPE_MTE2>(rawInitMutex_);
        AscendC::Mutex::Unlock<PIPE_MTE2>(rawInitMutex_);

    }

    __aicore__ inline void InitializeRawSlots()
    {
        constexpr uint32_t ROW_ELEMENTS = 64;
        AscendC::Mutex::Lock<PIPE_V>(rawInitMutex_);
        for (uint32_t r = 0; r < INTRA_CG_MAX; ++r) {
            auto gLocal = resource_.ubBuf.template GetBufferByByte<GateT>(
                INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES);
            auto betaLocal = resource_.ubBuf.template GetBufferByByte<BetaT>(
                INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES +
                INTRA_RAW_BETA_OFFSET);
            AscendC::Duplicate(gLocal, static_cast<GateT>(0), ROW_ELEMENTS);
            AscendC::Duplicate(betaLocal, static_cast<BetaT>(0), ROW_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mutex::Unlock<PIPE_V>(rawInitMutex_);
    }

    __aicore__ inline void FinalizePipeline()
    {
        // 等待最后一批异步消费者结束，再归还对应 Mutex ID。
        for (uint32_t slot = 0; slot < INTRA_CG_MAX; ++slot) {
            AscendC::Mutex::Lock<PIPE_V>(inputMutex_[slot]);
            AscendC::Mutex::Unlock<PIPE_V>(inputMutex_[slot]);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::Mutex::Lock<PIPE_V>(outputMutex_[slot]);
            AscendC::Mutex::Unlock<PIPE_V>(outputMutex_[slot]);
        }
        for (uint32_t slot = 0; slot < INTRA_CG_MAX; ++slot) {
            AscendC::ReleaseMutexID(inputMutex_[slot]);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::ReleaseMutexID(outputMutex_[slot]);
        }
        AscendC::ReleaseMutexID(rawInitMutex_);
    }

    __aicore__ inline void ProcessWork(
        const ChunkGdnBwdIntraWorkMeta &meta,
        bool (&vdoSlotUsed)[INTRA_VDO_TRACK_SLOTS],
        uint32_t (&lastVdoFlag)[INTRA_VDO_TRACK_SLOTS],
        const ChunkGdnBwdIntraWorkMeta *nextPairMeta,
        bool pair0Ready)
    {
        const bool pairInputs = meta.validTokens == tiling_->chunkSize &&
            meta.validHeads == INTRA_CG_MAX;
        // 先准备首份输入；循环中继续预取后续输入，使 MTE2 走在 Vector 前面。
        if (meta.validHeads > 0) {
            if (pairInputs) {
                if (!pair0Ready) {
                    PrefetchStage1InputBatch(meta, 0, 2);
                }
            } else {
                PrefetchStage1Input(meta, 0);
            }
        }
        for (int64_t r = 0; r < meta.validHeads; ++r) {
            // Queue the second pair while Vector consumes the first pair.
            if (pairInputs && r == 0) {
                PrefetchStage1InputBatch(meta, 2, 2);
            } else if (pairInputs && r == 2 && nextPairMeta != nullptr) {
                // Slots 0/1 are free now; overlap the next work with this pair.
                PrefetchStage1InputBatch(*nextPairMeta, 0, 2);
            } else if (!pairInputs && r + 1 < meta.validHeads) {
                PrefetchStage1Input(meta, r + 1);
            }
            const int64_t leader =
                ChunkGdnBwdIntraLeaderR<G>(meta.hvBegin, r);
            if (r == leader) {
                // AIC 只为每个唯一 hk 发布一次 Score；同一 hk 的后续 hv 直接复用该槽。
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    INTRA_SCORE_READY_FLAG + static_cast<uint32_t>(leader));
            }
            const uint32_t vdoSlot = static_cast<uint32_t>(r) & 1U;
            if (vdoSlotUsed[vdoSlot]) {
                // 共享 L1 的 v/d_o 只有 ping/pong 两槽，覆盖前等待 AIC 完成上一轮 MTE1 读取。
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(
                    lastVdoFlag[vdoSlot] +
                    static_cast<uint32_t>(part_) * INTRA_SUBBLOCK_FLAG_OFFSET);
            }
            lastVdoFlag[vdoSlot] =
                INTRA_WORKSPACE_READY_FLAG + static_cast<uint32_t>(r);
            vdoSlotUsed[vdoSlot] = true;
            RunStage1(meta, r, leader, pairInputs);
            const bool lastConsumer = r + 1 == meta.validHeads ||
                (meta.hvBegin + r) / G != (meta.hvBegin + r + 1) / G;
            if (lastConsumer) {
                // 当前 hk 的最后一个 hv 已读完 Score，向 AIC 归还本 AIV 的半片槽。
                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(
                    INTRA_SCORE_FREE_FLAG + static_cast<uint32_t>(leader));
            }
        }
    }

    __aicore__ inline void PrefetchStage1Input(
        const ChunkGdnBwdIntraWorkMeta &meta, int64_t r)
    {
        constexpr int64_t PART_ROWS = 32;
        constexpr int64_t ROW_ELEMENTS = 64;
        const int64_t rowBegin = part_ * PART_ROWS;
        int64_t rows = meta.validTokens - rowBegin;
        rows = rows < 0 ? 0 : (rows > PART_ROWS ? PART_ROWS : rows);
        if (rows == 0) {
            return;
        }

        const uint32_t slot = static_cast<uint32_t>(r) & 1U;
        auto aLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
            INTRA_A_UB_BASE + slot * INTRA_MATRIX_HALF_BYTES);
        auto gLocal = resource_.ubBuf.template GetBufferByByte<GateT>(
            INTRA_RAW_UB_BASE + slot * INTRA_RAW_SLOT_BYTES);
        auto betaLocal = resource_.ubBuf.template GetBufferByByte<BetaT>(
            INTRA_RAW_UB_BASE + slot * INTRA_RAW_SLOT_BYTES +
            INTRA_RAW_BETA_OFFSET);
        auto vLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
            INTRA_V_INPUT_UB_BASE + slot * INTRA_VDO_INPUT_HALF_BYTES);
        auto dOLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
            INTRA_DO_INPUT_UB_BASE + slot * INTRA_VDO_INPUT_HALF_BYTES);
        const int64_t hv = meta.hvBegin + r;
        const uint64_t matrixOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                 tiling_->seqlen + meta.tokenStart + rowBegin) * tiling_->chunkSize;
        const uint64_t vectorOffset =
            (static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                tiling_->seqlen + meta.tokenStart;
        const uint64_t vdoOffset =
            (vectorOffset + rowBegin) * tiling_->valueDim;

        // MTE2 等待 Vector 释放输入 slot；搬完后把 A/g/beta/v/d_o 一并交给 Vector。
        AscendC::Mutex::Lock<PIPE_MTE2>(inputMutex_[slot]);
        AscendC::DataCopy(aLocal, aGm_[matrixOffset],
                          static_cast<uint32_t>(rows * ROW_ELEMENTS));
        AscendC::DataCopyExtParams vectorCopy{
            1, static_cast<uint32_t>(meta.validTokens * sizeof(GateT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<GateT> gatePad{false, 0, 0, 0};
        AscendC::DataCopyPad(gLocal, gGm_[vectorOffset], vectorCopy, gatePad);
        vectorCopy.blockLen = static_cast<uint32_t>(meta.validTokens * sizeof(BetaT));
        AscendC::DataCopyPadExtParams<BetaT> betaPad{false, 0, 0, 0};
        AscendC::DataCopyPad(betaLocal, betaGm_[vectorOffset], vectorCopy, betaPad);
        const uint32_t vdoElements =
            static_cast<uint32_t>(rows * tiling_->valueDim);
        AscendC::DataCopy(vLocal, vGm_[vdoOffset], vdoElements);
        AscendC::DataCopy(dOLocal, dOGm_[vdoOffset], vdoElements);
        AscendC::Mutex::Unlock<PIPE_MTE2>(inputMutex_[slot]);
    }

    __aicore__ inline void PrefetchStage1InputBatch(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t firstSlot,
        uint32_t count)
    {
        constexpr uint32_t PART_ROWS = 32;
        constexpr uint32_t ROW_ELEMENTS = 64;
        const uint32_t rowBegin = static_cast<uint32_t>(part_) * PART_ROWS;
        const uint64_t hv = static_cast<uint64_t>(meta.hvBegin) + firstSlot;
        const uint64_t matrixOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                 tiling_->seqlen + meta.tokenStart + rowBegin) * tiling_->chunkSize;
        const uint64_t vectorOffset =
            (static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                tiling_->seqlen + meta.tokenStart;
        auto aLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
            INTRA_A_UB_BASE + firstSlot * INTRA_MATRIX_HALF_BYTES);
        auto gLocal = resource_.ubBuf.template GetBufferByByte<GateT>(
            INTRA_RAW_UB_BASE + firstSlot * INTRA_RAW_SLOT_BYTES);
        auto betaLocal = resource_.ubBuf.template GetBufferByByte<BetaT>(
            INTRA_RAW_UB_BASE + firstSlot * INTRA_RAW_SLOT_BYTES +
            INTRA_RAW_BETA_OFFSET);
        auto vLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
            INTRA_V_INPUT_UB_BASE + firstSlot * INTRA_VDO_INPUT_HALF_BYTES);
        auto dOLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
            INTRA_DO_INPUT_UB_BASE + firstSlot * INTRA_VDO_INPUT_HALF_BYTES);

        // 成组搬运同时占有连续输入 slot，防止任一 slot 尚被 Vector 使用时发生覆盖。
        for (uint32_t offset = 0; offset < count; ++offset) {
            AscendC::Mutex::Lock<PIPE_MTE2>(inputMutex_[firstSlot + offset]);
        }
        constexpr uint32_t matrixBytes = PART_ROWS * ROW_ELEMENTS * sizeof(MainT);
        AscendC::DataCopyExtParams matrixCopy{
            static_cast<uint16_t>(count), matrixBytes,
            static_cast<uint32_t>((tiling_->seqlen * ROW_ELEMENTS -
                                   PART_ROWS * ROW_ELEMENTS) * sizeof(MainT)),
            0, 0};
        AscendC::DataCopyPadExtParams<MainT> matrixPad{false, 0, 0, 0};
        AscendC::DataCopyPad(aLocal, aGm_[matrixOffset], matrixCopy, matrixPad);

        const uint32_t gateBytes = static_cast<uint32_t>(tiling_->chunkSize * sizeof(GateT));
        AscendC::DataCopyExtParams gateCopy{
            static_cast<uint16_t>(count), gateBytes,
            static_cast<uint32_t>((tiling_->seqlen - tiling_->chunkSize) * sizeof(GateT)),
            (INTRA_RAW_SLOT_BYTES - gateBytes) / 32U, 0};
        AscendC::DataCopyPadExtParams<GateT> gatePad{false, 0, 0, 0};
        AscendC::DataCopyPad(gLocal, gGm_[vectorOffset], gateCopy, gatePad);

        const uint32_t betaBytes = static_cast<uint32_t>(tiling_->chunkSize * sizeof(BetaT));
        AscendC::DataCopyExtParams betaCopy{
            static_cast<uint16_t>(count), betaBytes,
            static_cast<uint32_t>((tiling_->seqlen - tiling_->chunkSize) * sizeof(BetaT)),
            (INTRA_RAW_SLOT_BYTES - betaBytes) / 32U, 0};
        AscendC::DataCopyPadExtParams<BetaT> betaPad{false, 0, 0, 0};
        AscendC::DataCopyPad(betaLocal, betaGm_[vectorOffset], betaCopy, betaPad);
        const uint64_t vdoOffset =
            (vectorOffset + rowBegin) * tiling_->valueDim;
        const uint32_t halfElements =
            PART_ROWS * static_cast<uint32_t>(tiling_->valueDim);
        const uint64_t headStride =
            static_cast<uint64_t>(tiling_->seqlen) * tiling_->valueDim;
        AscendC::DataCopy(vLocal, vGm_[vdoOffset], halfElements);
        AscendC::DataCopy(dOLocal, dOGm_[vdoOffset], halfElements);
        if (count > 1) {
            AscendC::DataCopy(
                vLocal[halfElements], vGm_[vdoOffset + headStride], halfElements);
            AscendC::DataCopy(
                dOLocal[halfElements], dOGm_[vdoOffset + headStride], halfElements);
        }
        for (uint32_t offset = 0; offset < count; ++offset) {
            AscendC::Mutex::Unlock<PIPE_MTE2>(inputMutex_[firstSlot + offset]);
        }
    }

    __aicore__ inline void RunStage1(
        const ChunkGdnBwdIntraWorkMeta &meta, int64_t r,
        int64_t leader, bool pairInputs)
    {
        constexpr int64_t PART_ROWS = 32;
        const uint32_t inputSlot = pairInputs ? static_cast<uint32_t>(r) :
            static_cast<uint32_t>(r) & 1U;
        const uint32_t outputSlot = static_cast<uint32_t>(r & 1);
        const int64_t rowBegin = part_ * PART_ROWS;
        int64_t rows = meta.validTokens - rowBegin;
        rows = rows < 0 ? 0 : (rows > PART_ROWS ? PART_ROWS : rows);
        const uint32_t workspaceFlag = INTRA_WORKSPACE_READY_FLAG +
            static_cast<uint32_t>(r) +
            static_cast<uint32_t>(part_) * INTRA_SUBBLOCK_FLAG_OFFSET;

        if (rows > 0) {
            auto aLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
                INTRA_A_UB_BASE + inputSlot * INTRA_MATRIX_HALF_BYTES);
            auto workspaceLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
                INTRA_WORKSPACE_UB_BASE + outputSlot * INTRA_WORKSPACE_HALF_BYTES);
            auto dLocal = workspaceLocal[INTRA_D_UB_OFFSET / sizeof(MainT)];
            auto aBgLocal = workspaceLocal[INTRA_A_BG_UB_OFFSET / sizeof(MainT)]
                                .template ReinterpretCast<MainT>();
            auto aBetaLocal = workspaceLocal[INTRA_A_BETA_UB_OFFSET / sizeof(MainT)]
                                  .template ReinterpretCast<MainT>();
            auto gLocal = resource_.ubBuf.template GetBufferByByte<GateT>(
                INTRA_RAW_UB_BASE + inputSlot * INTRA_RAW_SLOT_BYTES);
            auto betaLocal = resource_.ubBuf.template GetBufferByByte<BetaT>(
                INTRA_RAW_UB_BASE + inputSlot * INTRA_RAW_SLOT_BYTES +
                INTRA_RAW_BETA_OFFSET);
            auto vLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
                INTRA_V_INPUT_UB_BASE + inputSlot * INTRA_VDO_INPUT_HALF_BYTES);
            auto dOLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
                INTRA_DO_INPUT_UB_BASE + inputSlot * INTRA_VDO_INPUT_HALF_BYTES);
            auto vdoNzLocal = resource_.ubBuf.template GetBufferByByte<MainT>(
                INTRA_VDO_NZ_UB_BASE + outputSlot * INTRA_VDO_NZ_SLOT_BYTES);
            auto vNzLocal = vdoNzLocal;
            auto dONzLocal = vdoNzLocal[
                INTRA_VDO_INPUT_HALF_BYTES / sizeof(MainT)];
            auto score = resource_.ubBuf.template GetBufferByByte<float>(
                INTRA_SCORE_UB_BASE +
                static_cast<uint32_t>(leader) * INTRA_SCORE_SLOT_BYTES);

            // Vector 同时等待 MTE2 输入 ready 和上一轮 MTE3 释放输出 ping/pong 槽。
            AscendC::Mutex::Lock<PIPE_V>(inputMutex_[inputSlot]);
            AscendC::Mutex::Lock<PIPE_V>(outputMutex_[outputSlot]);

            const uint16_t vfRows = static_cast<uint16_t>(rows);
            const uint16_t vfRowBegin = static_cast<uint16_t>(rowBegin);
            const uint16_t vfValidTokens = static_cast<uint16_t>(meta.validTokens);
            auto aBgOut = reinterpret_cast<__ubuf__ MainT *>(aBgLocal.GetPhyAddr());
            auto aBetaOut = reinterpret_cast<__ubuf__ MainT *>(aBetaLocal.GetPhyAddr());
            auto dOut = reinterpret_cast<__ubuf__ MainT *>(dLocal.GetPhyAddr());
            auto aIn = reinterpret_cast<__ubuf__ MainT *>(aLocal.GetPhyAddr());
            auto scoreIn = reinterpret_cast<__ubuf__ float *>(score.GetPhyAddr());
            auto gIn = reinterpret_cast<__ubuf__ GateT *>(gLocal.GetPhyAddr());
            auto betaIn = reinterpret_cast<__ubuf__ BetaT *>(betaLocal.GetPhyAddr());

            // use_exp2 是运行时属性；模板分派让 VF 内部保留编译期常量分支。
            if (tiling_->useExp2 != 0) {
                ChunkGdnBwdIntraStage1VF<MainT, GateT, BetaT, true>(
                    aBgOut, aBetaOut, dOut, aIn, scoreIn, gIn, gIn + vfRowBegin, betaIn,
                    vfRows, vfRowBegin, vfValidTokens, tiling_->scale);
            } else {
                ChunkGdnBwdIntraStage1VF<MainT, GateT, BetaT, false>(
                    aBgOut, aBetaOut, dOut, aIn, scoreIn, gIn, gIn + vfRowBegin, betaIn,
                    vfRows, vfRowBegin, vfValidTokens, tiling_->scale);
            }
            ChunkGdnBwdIntraPackMatrixVF<MainT>(
                reinterpret_cast<__ubuf__ MainT *>(vNzLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ MainT *>(vLocal.GetPhyAddr()),
                static_cast<uint16_t>(rows));
            ChunkGdnBwdIntraPackMatrixVF<MainT>(
                reinterpret_cast<__ubuf__ MainT *>(dONzLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ MainT *>(dOLocal.GetPhyAddr()),
                static_cast<uint16_t>(rows));
            // 输入已经读完，可让 MTE2 继续预取；输出交给 MTE3 写共享 L1。
            AscendC::Mutex::Unlock<PIPE_V>(inputMutex_[inputSlot]);
            AscendC::Mutex::Unlock<PIPE_V>(outputMutex_[outputSlot]);
            AscendC::Mutex::Lock<PIPE_MTE3>(outputMutex_[outputSlot]);

            // MTE3 写入当前 AIV 负责的半行，完成后向 AIC 发布这一半的 ready。
            StoreWorkspacePartToL1(
                static_cast<uint32_t>(r), dLocal, aBgLocal, aBetaLocal);
            StoreVdoPartToL1(
                static_cast<uint32_t>(r), vNzLocal, dONzLocal);
            AscendC::Mutex::Unlock<PIPE_MTE3>(outputMutex_[outputSlot]);
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(workspaceFlag);
        } else {
            // 尾块中本 AIV 没有有效行，也必须发布 ready，避免 AIC 永久等待。
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(workspaceFlag);
        }
    }

    __aicore__ inline void StoreWorkspacePartToL1(
        uint32_t r,
        AscendC::LocalTensor<MainT> dLocal,
        AscendC::LocalTensor<MainT> aBgLocal,
        AscendC::LocalTensor<MainT> aBetaLocal) const
    {
        AscendC::LocalTensor<uint8_t> l1Buffer(
            AscendC::TPosition::A1, 0, 512 * 1024);
        auto l1Record = l1Buffer[
            L1_RECORD_BASE + r * L1_RECORD_SLOT_BYTES]
                            .template ReinterpretCast<MainT>();
        StoreMatrixPartToL1(l1Record, dLocal, 0);
        StoreMatrixPartToL1(l1Record, aBgLocal, 1);
        StoreMatrixPartToL1(l1Record, aBetaLocal, 2);
    }

    __aicore__ inline void StoreMatrixPartToL1(
        AscendC::LocalTensor<MainT> l1Record,
        AscendC::LocalTensor<MainT> src, uint32_t matrixIndex) const
    {
        const uint32_t dstOffset =
            (matrixIndex * L1_MATRIX_SLOT_BYTES +
             static_cast<uint32_t>(part_) * INTRA_MATRIX_HALF_BYTES) /
            sizeof(MainT);
        AscendC::DataCopy(
            l1Record[dstOffset], src,
            INTRA_MATRIX_HALF_BYTES / sizeof(MainT));
    }

    __aicore__ inline void StoreVdoPartToL1(
        uint32_t r,
        AscendC::LocalTensor<MainT> vNzLocal,
        AscendC::LocalTensor<MainT> dONzLocal) const
    {
        AscendC::LocalTensor<uint8_t> l1Buffer(
            AscendC::TPosition::A1, 0, 512 * 1024);
        const uint32_t slot = r & 1U;
        const uint32_t l1VOffset = L1_V_BASE + slot * L1_SLOT_BYTES;
        const uint32_t l1DOOffset = L1_DO_BASE + slot * L1_SLOT_BYTES;
        const uint32_t partOffset =
            static_cast<uint32_t>(part_) * INTRA_VDO_INPUT_HALF_BYTES;
        auto l1V = l1Buffer[l1VOffset + partOffset]
                        .template ReinterpretCast<MainT>();
        auto l1DO = l1Buffer[l1DOOffset + partOffset]
                         .template ReinterpretCast<MainT>();
        constexpr uint32_t halfElements =
            INTRA_VDO_INPUT_HALF_BYTES / sizeof(MainT);
        AscendC::DataCopy(l1V, vNzLocal, halfElements);
        AscendC::DataCopy(l1DO, dONzLocal, halfElements);
    }

    // 运行时状态放在所有成员函数之后。
    Catlass::Arch::Resource<ArchTag> resource_;
    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;

    // GM tensor 统一使用 Gm 后缀，与函数内的 UB/L1 tensor 区分。
    AscendC::GlobalTensor<MainT> aGm_;
    AscendC::GlobalTensor<MainT> vGm_;
    AscendC::GlobalTensor<MainT> dOGm_;
    AscendC::GlobalTensor<GateT> gGm_;
    AscendC::GlobalTensor<BetaT> betaGm_;

    // inputMutex 对应最多 4 份 head 输入，outputMutex 对应 Vector ping/pong 输出。
    AscendC::MutexID inputMutex_[INTRA_CG_MAX];
    AscendC::MutexID outputMutex_[2];
    AscendC::MutexID rawInitMutex_;
    int64_t coreIdx_ = 0;
    int64_t part_ = 0;
};

} // namespace GDN

#endif // CHUNK_GDN_BWD_INTRA_VECTOR_H
