/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_causal_conv1d.cpp
 * \brief
 */

#include "../../../op_kernel/causal_conv1d.cpp"
#include "../causal_conv1d_tiling_key_test_helper.h"
#include "causal_conv1d_tiling.h"
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

#define CAUSAL_CONV1D_KERNEL_PREFILL(widthKey, fnPlanKey, activationFlag, biasFlag) \
    causal_conv1d<CAUSAL_CONV1D_TPL_RUN_MODE_FN, widthKey, fnPlanKey>
#define CAUSAL_CONV1D_KERNEL_UPDATE(activationFlag, biasFlag) \
    causal_conv1d<CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME, CAUSAL_CONV1D_TPL_FN_PLAN_INVALID>

static inline uint64_t PrefillTilingKeyForTest(FnExecutionPlan fnPlan, int32_t width, bool hasActivation, bool hasBias)
{
    return BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, fnPlan, width, hasActivation ? 1 : 0, hasBias);
}

static inline uint64_t UpdateTilingKeyForTest(bool hasActivation, bool hasBias)
{
    return BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, FN_EXECUTION_PLAN_INVALID, 4,
                                      hasActivation ? 1 : 0, hasBias);
}

static inline void PrepareFnTokenTilingForTest(CausalConv1dTilingData* tilingData)
{
    if (tilingData->tokenBlockSize <= 0) {
        tilingData->tokenBlockSize = 1;
    }
    if (tilingData->tokenBlockCnt <= 0) {
        tilingData->tokenBlockCnt = std::max<int64_t>(tilingData->cuSeqlen, 1);
    }
}

static inline void PrepareExplicitFnTokenSeqRangesForTest(CausalConv1dTilingData* tilingData, const int64_t* queryStartLoc)
{
    tilingData->hasExplicitTokenSeqRanges = 0;
    tilingData->explicitTokenSeqRangeCount = 0;
    if (tilingData->inputMode != 0 || queryStartLoc == nullptr || tilingData->tokenBlockSize <= 0 ||
        tilingData->tokenBlockCnt <= 0) {
        return;
    }

    const int64_t maxRangeCount =
        static_cast<int64_t>(sizeof(tilingData->tokenTileStartSeq) / sizeof(tilingData->tokenTileStartSeq[0]));
    if (tilingData->tokenBlockCnt > maxRangeCount) {
        return;
    }

    tilingData->hasExplicitTokenSeqRanges = 1;
    tilingData->explicitTokenSeqRangeCount = tilingData->tokenBlockCnt;
    int64_t seq = 0;
    for (int64_t tokenTileId = 0; tokenTileId < tilingData->tokenBlockCnt; ++tokenTileId) {
        const int64_t tokenStart = tokenTileId * tilingData->tokenBlockSize;
        const int64_t tokenEnd = tokenStart + tilingData->tokenBlockSize;
        while (seq < tilingData->batch && queryStartLoc[seq + 1] <= tokenStart) {
            ++seq;
        }
        int64_t endSeq = seq;
        while (endSeq < tilingData->batch && queryStartLoc[endSeq] < tokenEnd) {
            ++endSeq;
        }
        tilingData->tokenTileStartSeq[tokenTileId] = seq;
        tilingData->tokenTileEndSeq[tokenTileId] = endSeq;
    }
}

static inline void SetRuntimeFeatureFlagsForTest(CausalConv1dTilingData* tilingData, int64_t activationMode, bool hasBias)
{
    tilingData->activationMode = activationMode;
    tilingData->hasBias = hasBias ? 1 : 0;
}

template <uint32_t fnPlanKey, uint32_t activationFlag, uint32_t biasFlag,
          uint32_t widthKey = CAUSAL_CONV1D_TPL_WIDTH_4, typename... Args>
static inline void RunPrefillKernelWithKey(uint32_t gridSize, Args... args)
{
    ICPU_SET_TILING_KEY(
        PrefillTilingKeyForTest(static_cast<FnExecutionPlan>(fnPlanKey), NsCausalConv1d::DecodeWidthTplKey(widthKey),
                                activationFlag != 0, biasFlag != 0));
    ICPU_RUN_KF(CAUSAL_CONV1D_KERNEL_PREFILL(widthKey, fnPlanKey, activationFlag, biasFlag), gridSize, args...);
}

template <uint32_t activationFlag, uint32_t biasFlag, typename... Args>
static inline void RunUpdateKernelWithKey(uint32_t gridSize, Args... args)
{
    ICPU_SET_TILING_KEY(UpdateTilingKeyForTest(activationFlag != 0, biasFlag != 0));
    ICPU_RUN_KF(CAUSAL_CONV1D_KERNEL_UPDATE(activationFlag, biasFlag), gridSize, args...);
}

template <typename T, uint32_t widthKey, uint32_t fnPlanKey>
struct FnRollingFastPathTestShim : public NsCausalConv1d::CausalConv1d<T, CAUSAL_CONV1D_TPL_RUN_MODE_FN, widthKey, fnPlanKey> {
    using Base = NsCausalConv1d::CausalConv1d<T, CAUSAL_CONV1D_TPL_RUN_MODE_FN, widthKey, fnPlanKey>;
    using Base::IsFnRollingFastPathEnabled;
    using Base::ResetRuntimeState;
};

class causal_conv1d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "causal_conv1d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "causal_conv1d_test TearDown\n" << endl;
    }
};

TEST(causal_conv1d_internal, seq_task_window_mode_maps_input_mode_once)
{
    EXPECT_EQ(NsCausalConv1d::GetSeqTaskWindowMode(/*inputMode=*/0), NsCausalConv1d::SEQ_TASK_WINDOW_MODE_VARLEN);
    EXPECT_EQ(NsCausalConv1d::GetSeqTaskWindowMode(/*inputMode=*/1), NsCausalConv1d::SEQ_TASK_WINDOW_MODE_BATCH);
    EXPECT_EQ(NsCausalConv1d::GetSeqTaskWindowMode(/*inputMode=*/2), NsCausalConv1d::SEQ_TASK_WINDOW_MODE_DECODE2D);
}

TEST(causal_conv1d_internal, seq_task_window_builders_cover_varlen_batch_and_decode)
{
    const auto varlenWindow = NsCausalConv1d::BuildSeqTaskWindowVarlen(/*startVal=*/13, /*endVal=*/21);
    EXPECT_TRUE(varlenWindow.valid);
    EXPECT_EQ(varlenWindow.start, 13);
    EXPECT_EQ(varlenWindow.len, 8);

    const auto emptyVarlenWindow = NsCausalConv1d::BuildSeqTaskWindowVarlen(/*startVal=*/7, /*endVal=*/7);
    EXPECT_FALSE(emptyVarlenWindow.valid);

    const auto batchWindow = NsCausalConv1d::BuildSeqTaskWindowBatch(/*seq=*/3, /*seqLen=*/5);
    EXPECT_TRUE(batchWindow.valid);
    EXPECT_EQ(batchWindow.start, 15);
    EXPECT_EQ(batchWindow.len, 5);

    const auto invalidBatchWindow = NsCausalConv1d::BuildSeqTaskWindowBatch(/*seq=*/3, /*seqLen=*/0);
    EXPECT_FALSE(invalidBatchWindow.valid);

    const auto decodeWindow = NsCausalConv1d::BuildSeqTaskWindowDecode2D(/*seq=*/4);
    EXPECT_TRUE(decodeWindow.valid);
    EXPECT_EQ(decodeWindow.start, 4);
    EXPECT_EQ(decodeWindow.len, 1);
}

TEST(causal_conv1d_internal, runseq_ring_slot_helpers_wrap_and_track_history)
{
    EXPECT_EQ(NsCausalConv1d::RetreatRingSlot(/*slot=*/0, /*delta=*/1), 4);
    EXPECT_EQ(NsCausalConv1d::RetreatRingSlot(/*slot=*/1, /*delta=*/3), 3);
    EXPECT_EQ(NsCausalConv1d::RetreatRingSlot(/*slot=*/2, /*delta=*/1), 1);
    EXPECT_EQ(NsCausalConv1d::RetreatRingSlot(/*slot=*/4, /*delta=*/3), 1);
}

TEST(causal_conv1d_internal, tiling_key_builder_tracks_live_fn_dispatch)
{
    EXPECT_EQ(PrefillTilingKeyForTest(FN_EXECUTION_PLAN_CUTBS, /*width=*/4, /*hasActivation=*/false, /*hasBias=*/true),
              BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, FN_EXECUTION_PLAN_CUTBS, 4, 0, true));
    EXPECT_EQ(PrefillTilingKeyForTest(FN_EXECUTION_PLAN_CUTBSD, /*width=*/4, /*hasActivation=*/true, /*hasBias=*/false),
              BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, FN_EXECUTION_PLAN_CUTBSD, 4, 1, false));
    EXPECT_EQ(UpdateTilingKeyForTest(/*hasActivation=*/true, /*hasBias=*/false),
              BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, FN_EXECUTION_PLAN_INVALID, 4, 1, false));
}

TEST(causal_conv1d_internal, rolling_fast_path_rejects_bias_and_spec_tokens)
{
    CausalConv1dTilingData tilingData{};
    tilingData.hasBias = 1;
    tilingData.hasNumAcceptedTokens = 0;

    FnRollingFastPathTestShim<half, CAUSAL_CONV1D_TPL_WIDTH_4, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS> op;
    op.ResetRuntimeState(&tilingData);
    EXPECT_FALSE(op.IsFnRollingFastPathEnabled());

    tilingData.hasBias = 0;
    tilingData.hasNumAcceptedTokens = 1;
    op.ResetRuntimeState(&tilingData);
    EXPECT_FALSE(op.IsFnRollingFastPathEnabled());

    tilingData.hasBias = 0;
    tilingData.hasNumAcceptedTokens = 0;
    op.ResetRuntimeState(&tilingData);
    EXPECT_TRUE(op.IsFnRollingFastPathEnabled());
}

TEST(causal_conv1d_internal, direct_block_task_mapping_assigns_one_contiguous_task_per_block)
{
    const auto cutbsHead = NsCausalConv1d::ResolveFnDirectBlockTask(/*blockIdx=*/0, /*tokenBlockCnt=*/40,
                                                                    /*tokenBlockSize=*/103, /*cuSeqlen=*/4096,
                                                                    /*baseDimCnt=*/1, /*baseDim=*/1536, /*dim=*/1536);
    EXPECT_TRUE(cutbsHead.valid);
    EXPECT_EQ(cutbsHead.tokenTileId, 0);
    EXPECT_EQ(cutbsHead.baseDimIdx, 0);
    EXPECT_EQ(cutbsHead.tokenStart, 0);
    EXPECT_EQ(cutbsHead.tokenEnd, 103);
    EXPECT_EQ(cutbsHead.channelStart, 0);
    EXPECT_EQ(cutbsHead.baseDimSize, 1536);

    const auto cutbsTail = NsCausalConv1d::ResolveFnDirectBlockTask(/*blockIdx=*/39, /*tokenBlockCnt=*/40,
                                                                    /*tokenBlockSize=*/103, /*cuSeqlen=*/4096,
                                                                    /*baseDimCnt=*/1, /*baseDim=*/1536, /*dim=*/1536);
    EXPECT_TRUE(cutbsTail.valid);
    EXPECT_EQ(cutbsTail.tokenTileId, 39);
    EXPECT_EQ(cutbsTail.baseDimIdx, 0);
    EXPECT_EQ(cutbsTail.tokenStart, 4017);
    EXPECT_EQ(cutbsTail.tokenEnd, 4096);
    EXPECT_EQ(cutbsTail.baseDimSize, 1536);

    const auto cutbsd0 = NsCausalConv1d::ResolveFnDirectBlockTask(/*blockIdx=*/0, /*tokenBlockCnt=*/16,
                                                                  /*tokenBlockSize=*/2, /*cuSeqlen=*/32,
                                                                  /*baseDimCnt=*/2, /*baseDim=*/2048, /*dim=*/4096);
    EXPECT_TRUE(cutbsd0.valid);
    EXPECT_EQ(cutbsd0.tokenTileId, 0);
    EXPECT_EQ(cutbsd0.baseDimIdx, 0);
    EXPECT_EQ(cutbsd0.tokenStart, 0);
    EXPECT_EQ(cutbsd0.tokenEnd, 2);
    EXPECT_EQ(cutbsd0.channelStart, 0);
    EXPECT_EQ(cutbsd0.baseDimSize, 2048);

    const auto cutbsd1 = NsCausalConv1d::ResolveFnDirectBlockTask(/*blockIdx=*/1, /*tokenBlockCnt=*/16,
                                                                  /*tokenBlockSize=*/2, /*cuSeqlen=*/32,
                                                                  /*baseDimCnt=*/2, /*baseDim=*/2048, /*dim=*/4096);
    EXPECT_TRUE(cutbsd1.valid);
    EXPECT_EQ(cutbsd1.tokenTileId, 0);
    EXPECT_EQ(cutbsd1.baseDimIdx, 1);
    EXPECT_EQ(cutbsd1.channelStart, 2048);
    EXPECT_EQ(cutbsd1.baseDimSize, 2048);

    const auto invalid = NsCausalConv1d::ResolveFnDirectBlockTask(/*blockIdx=*/32, /*tokenBlockCnt=*/16,
                                                                  /*tokenBlockSize=*/2, /*cuSeqlen=*/32,
                                                                  /*baseDimCnt=*/2, /*baseDim=*/2048, /*dim=*/4096);
    EXPECT_FALSE(invalid.valid);
}

TEST(causal_conv1d_internal, direct_block_mapping_reuses_rolling_fast_path_when_allowed)
{
    CausalConv1dTilingData tilingData{};
    tilingData.inputMode = 0;
    tilingData.batch = 1;
    tilingData.hasNumAcceptedTokens = 0;
    tilingData.hasBias = 0;

    FnRollingFastPathTestShim<half, CAUSAL_CONV1D_TPL_WIDTH_4, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS> cutbsOp;
    cutbsOp.ResetRuntimeState(&tilingData);
    EXPECT_TRUE(cutbsOp.IsFnRollingFastPathEnabled());

    FnRollingFastPathTestShim<half, CAUSAL_CONV1D_TPL_WIDTH_4, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD> cutbsdOp;
    cutbsdOp.ResetRuntimeState(&tilingData);
    EXPECT_TRUE(cutbsdOp.IsFnRollingFastPathEnabled());

    tilingData.batch = 2;
    cutbsOp.ResetRuntimeState(&tilingData);
    EXPECT_TRUE(cutbsOp.IsFnRollingFastPathEnabled());

    tilingData.batch = 1;
    tilingData.inputMode = 1;
    cutbsOp.ResetRuntimeState(&tilingData);
    EXPECT_TRUE(cutbsOp.IsFnRollingFastPathEnabled());

    tilingData.inputMode = 0;
    tilingData.hasBias = 1;
    cutbsOp.ResetRuntimeState(&tilingData);
    EXPECT_FALSE(cutbsOp.IsFnRollingFastPathEnabled());

    tilingData.hasBias = 0;
    tilingData.hasNumAcceptedTokens = 1;
    cutbsOp.ResetRuntimeState(&tilingData);
    EXPECT_FALSE(cutbsOp.IsFnRollingFastPathEnabled());
}

template <typename T>
static inline T FromFloat(float v)
{
    return static_cast<T>(v);
}

template <typename T>
static inline float ToFloat(T v)
{
    return static_cast<float>(v);
}

static float SiluRef(float x)
{
    return x / (1.0f + std::exp(-x));
}

// Reference implementation (weight layout: (width, dim), dim contiguous)
// Access: weight[j, c] = weight[j * dim + c]
static void ReferenceCausalConv1dFwdBatch(const float* x, const float* weight, const float* bias, float* y,
                                         float* convStates, int32_t dim, int32_t width, int32_t stateLen,
                                         const int64_t* queryStartLoc, int32_t batch, const int64_t* cacheIndices,
                                         const int64_t* initialStateMode, int32_t activationMode, int32_t padSlotId)
{
    for (int32_t seq = 0; seq < batch; ++seq) {
        const int32_t start = static_cast<int32_t>(queryStartLoc[seq]);
        const int32_t end = static_cast<int32_t>(queryStartLoc[seq + 1]);
        const int32_t len = end - start;
        if (len <= 0) {
            continue;
        }
        const int32_t cacheIdx = static_cast<int32_t>(cacheIndices[seq]);
        if (cacheIdx == padSlotId) {
            continue;
        }

        std::vector<float> hist(width - 1, 0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            if (initialStateMode[seq] != 0) {
                for (int32_t h = 0; h < width - 1; ++h) {
                    hist[h] = convStates[(cacheIdx * stateLen + h) * dim + c];
                }
            } else {
                std::fill(hist.begin(), hist.end(), 0.0f);
            }

            for (int32_t t = 0; t < len; ++t) {
                float acc = bias != nullptr ? bias[c] : 0.0f;
                for (int32_t j = 0; j < width; ++j) {
                    // Align with vLLM/Triton (PyTorch conv1d correlation) convention:
                    //   weight[0] * X_{t-(K-1)} ... weight[K-1] * X_t
                    const int32_t srcT = t - (width - 1) + j;
                    float xval = 0.0f;
                    if (srcT >= 0) {
                        xval = x[(start + srcT) * dim + c];
                    } else {
                        const int32_t histIndex = (width - 1) + srcT; // srcT is negative
                        xval = hist[histIndex];
                    }
                    acc += weight[j * dim + c] * xval;
                }
                if (activationMode != 0) {
                    acc = SiluRef(acc);
                }
                y[(start + t) * dim + c] = acc;
            }

            // state writeback: new_state = tail_{width-1}(concat(hist, X))
            for (int32_t s = 0; s < width - 1; ++s) {
                const int32_t idxInConcat = len + s;
                float v = 0.0f;
                if (idxInConcat < (width - 1)) {
                    v = hist[idxInConcat];
                } else {
                    const int32_t tok = idxInConcat - (width - 1);
                    v = x[(start + tok) * dim + c];
                }
                convStates[(cacheIdx * stateLen + s) * dim + c] = v;
            }
        }
    }
}

TEST_F(causal_conv1d_test, fwd_bfloat16_basic)
{
    if (!std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: BF16 contract case requires a bfloat16_t-specialized DTYPE_X build";
        return;
    }

    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 4;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 0;

    const std::vector<int64_t> queryStartLoc = {0, 2, 4};
    const std::vector<int64_t> cacheIndices = {0, 1};
    const std::vector<int64_t> hasInitialState = {1, 1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(bfloat16_t);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(bfloat16_t);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(bfloat16_t);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(bfloat16_t);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(bfloat16_t);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<bfloat16_t*>(xGm);
    auto* w = reinterpret_cast<bfloat16_t*>(wGm);
    auto* b = reinterpret_cast<bfloat16_t*>(bGm);
    auto* s = reinterpret_cast<bfloat16_t*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<bfloat16_t*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<bfloat16_t>(0.01f * static_cast<float>(c) + 0.05f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<bfloat16_t>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<bfloat16_t>(0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<bfloat16_t>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<bfloat16_t>(0.2f + 0.01f * static_cast<float>(c) + 0.001f * static_cast<float>(h));
            s[(1 * stateLen + h) * dim + c] = FromFloat<bfloat16_t>(-0.2f + 0.02f * static_cast<float>(c) - 0.002f * static_cast<float>(h));
        }
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<bfloat16_t>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                 qsl, batch, idx, his, activationMode, padSlotId);

    // quantize expected output/state to BF16 (kernel writes BF16)
    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<bfloat16_t>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<bfloat16_t>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 1024;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 3;
    tilingData->tokenBlockCnt = 2;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    std::vector<bfloat16_t> sInit(numCacheLines * dim * stateLen);
    std::memcpy(sInit.data(), s, sBytes);

    std::memcpy(s, sInit.data(), sBytes);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<bfloat16_t>(12345.0f);
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, 0, 1>(
        tilingData->tokenBlockCnt, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], 5e-2) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], 5e-2) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_float16_varlen_default_basic_contract)
{
    if (!std::is_same<DTYPE_X, half>::value) {
        GTEST_SKIP() << "Skip: compiled with DTYPE_X != half";
        return;
    }

    constexpr float tol = 3e-2f;

    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 4;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 2, 4};
    const std::vector<int64_t> cacheIndices = {0, 1};
    const std::vector<int64_t> hasInitialState = {1, 1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(half);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(half);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(half);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(half);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(half);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<half*>(xGm);
    auto* w = reinterpret_cast<half*>(wGm);
    auto* b = reinterpret_cast<half*>(bGm);
    auto* s = reinterpret_cast<half*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<half*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<half>(0.03f * static_cast<float>(t) +
                                             0.002f * static_cast<float>(c % 29));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<half>(0.004f * static_cast<float>(j + 1) +
                                             0.0003f * static_cast<float>(c % 19));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<half>(0.01f * static_cast<float>((c % 7) - 3));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<half>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] =
                FromFloat<half>(0.05f + 0.001f * static_cast<float>(c % 23) + 0.002f * static_cast<float>(h));
            s[(1 * stateLen + h) * dim + c] =
                FromFloat<half>(-0.04f + 0.0015f * static_cast<float>(c % 17) - 0.001f * static_cast<float>(h));
        }
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<half>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                 qsl, batch, idx, his, activationMode, padSlotId);

    // quantize expected output/state to FP16 (kernel writes FP16)
    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<half>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<half>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 1024;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 1;
    tilingData->tokenBlockCnt = cuSeqlen;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, 1, 1>(
        tilingData->tokenBlockCnt * tilingData->baseDimCnt, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm,
        workspace, reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_float16_varlen_default_dim_tail_contract)
{
    if (!std::is_same<DTYPE_X, half>::value) {
        GTEST_SKIP() << "Skip: compiled with DTYPE_X != half";
        return;
    }

    constexpr float tol = 3e-2f;

    constexpr int32_t dim = 768;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 4;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 2, 4};
    const std::vector<int64_t> cacheIndices = {0, 1};
    const std::vector<int64_t> hasInitialState = {1, 1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(half);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(half);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(half);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(half);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(half);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<half*>(xGm);
    auto* w = reinterpret_cast<half*>(wGm);
    auto* b = reinterpret_cast<half*>(bGm);
    auto* s = reinterpret_cast<half*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<half*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<half>(0.025f * static_cast<float>(t) +
                                             0.0015f * static_cast<float>(c % 31));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<half>(0.003f * static_cast<float>(j + 1) +
                                             0.0004f * static_cast<float>(c % 13));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<half>(0.008f * static_cast<float>((c % 9) - 4));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<half>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] =
                FromFloat<half>(0.04f + 0.001f * static_cast<float>(c % 21) + 0.002f * static_cast<float>(h));
            s[(1 * stateLen + h) * dim + c] =
                FromFloat<half>(-0.03f + 0.0012f * static_cast<float>(c % 15) - 0.001f * static_cast<float>(h));
        }
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<half>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width,
                                 stateLen, qsl, batch, idx, his, activationMode, padSlotId);

    // quantize expected output/state to FP16 (kernel writes FP16)
    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<half>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<half>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 512;
    tilingData->baseDimCnt = 2;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 1;
    tilingData->tokenBlockCnt = cuSeqlen;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    const int32_t gridSize = tilingData->tokenBlockCnt * static_cast<int32_t>(tilingData->baseDimCnt);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD, 1, 1>(
        gridSize, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Reference implementation for 3D batch mode (fixed seqLen for all sequences)
static void ReferenceCausalConv1dFwd3DBatch(const float* x, const float* weight, const float* bias, float* y,
                                            float* convStates, int32_t dim, int32_t width, int32_t stateLen,
                                            int32_t seqLen, int32_t batch, const int64_t* cacheIndices,
                                            const int64_t* initialStateMode, int32_t activationMode, int32_t padSlotId)
{
    for (int32_t seq = 0; seq < batch; ++seq) {
        const int32_t start = seq * seqLen;  // Fixed offset based on seqLen
        const int32_t len = seqLen;          // Fixed length for all sequences
        const int32_t cacheIdx = static_cast<int32_t>(cacheIndices[seq]);

        if (cacheIdx == padSlotId) {
            continue;
        }

        std::vector<float> hist(width - 1, 0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            if (initialStateMode[seq] != 0) {
                for (int32_t h = 0; h < width - 1; ++h) {
                    hist[h] = convStates[(cacheIdx * stateLen + h) * dim + c];
                }
            } else {
                std::fill(hist.begin(), hist.end(), 0.0f);
            }

            for (int32_t t = 0; t < len; ++t) {
                float acc = bias != nullptr ? bias[c] : 0.0f;
                for (int32_t j = 0; j < width; ++j) {
                    const int32_t srcT = t - (width - 1) + j;
                    float xval = 0.0f;
                    if (srcT >= 0) {
                        xval = x[(start + srcT) * dim + c];
                    } else {
                        const int32_t histIndex = (width - 1) + srcT;
                        xval = hist[histIndex];
                    }
                    acc += weight[j * dim + c] * xval;
                }
                if (activationMode != 0) {
                    acc = SiluRef(acc);
                }
                y[(start + t) * dim + c] = acc;
            }

            // State writeback
            for (int32_t s = 0; s < width - 1; ++s) {
                const int32_t idxInConcat = len + s;
                float v = 0.0f;
                if (idxInConcat < (width - 1)) {
                    v = hist[idxInConcat];
                } else {
                    const int32_t tok = idxInConcat - (width - 1);
                    v = x[(start + tok) * dim + c];
                }
                convStates[(cacheIdx * stateLen + s) * dim + c] = v;
            }
        }
    }
}

// Reference implementation for decode/update mode with input x.shape=(batch, dim)
static void ReferenceCausalConv1dDecode2D(const float* x, const float* weight, const float* bias, float* y,
                                          float* convStates, int32_t dim, int32_t width, int32_t stateLen,
                                          int32_t batch, const int64_t* cacheIndices,
                                          const int64_t* initialStateMode, int32_t activationMode, int32_t padSlotId)
{
    constexpr int32_t decodeLen = 1;
    for (int32_t seq = 0; seq < batch; ++seq) {
        const int32_t start = seq;
        const int32_t cacheIdx = static_cast<int32_t>(cacheIndices[seq]);
        if (cacheIdx == padSlotId) {
            continue;
        }

        std::vector<float> hist(width - 1, 0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            if (initialStateMode[seq] != 0) {
                for (int32_t h = 0; h < width - 1; ++h) {
                    hist[h] = convStates[(cacheIdx * stateLen + h) * dim + c];
                }
            } else {
                std::fill(hist.begin(), hist.end(), 0.0f);
            }

            float acc = bias != nullptr ? bias[c] : 0.0f;
            for (int32_t j = 0; j < width; ++j) {
                const int32_t srcT = 0 - (width - 1) + j;
                float xval = 0.0f;
                if (srcT >= 0) {
                    xval = x[(start + srcT) * dim + c];
                } else {
                    const int32_t histIndex = (width - 1) + srcT;
                    xval = hist[histIndex];
                }
                acc += weight[j * dim + c] * xval;
            }
            if (activationMode != 0) {
                acc = SiluRef(acc);
            }
            y[start * dim + c] = acc;

            for (int32_t s = 0; s < width - 1; ++s) {
                const int32_t idxInConcat = decodeLen + s;
                float v = 0.0f;
                if (idxInConcat < (width - 1)) {
                    v = hist[idxInConcat];
                } else {
                    const int32_t tok = idxInConcat - (width - 1);
                    v = x[(start + tok) * dim + c];
                }
                convStates[(cacheIdx * stateLen + s) * dim + c] = v;
            }
        }
    }
}

// Reference implementation for vLLM speculative decoding semantics in decode/update mode (fixed seqLen per batch).
// - x: flattened (batch * seqLen, dim) in physical layout (token-major, dim-last contiguous).
// - convStates: (num_cache_lines, stateLen, dim)
// - numAcceptedTokens: (batch), used to select the history window offset and to update convStates in a sliding manner.
static void ReferenceCausalConv1dDecode3DSpec(const float* x, const float* weight, const float* bias, float* y,
                                              float* convStates, int32_t dim, int32_t width, int32_t stateLen,
                                              int32_t seqLen, int32_t batch, const int64_t* cacheIndices,
                                              const int64_t* initialStateMode, const int64_t* numAcceptedTokens,
                                              int32_t activationMode, int32_t padSlotId)
{
    // Only used for width=4 in current kernel implementation.
    const int32_t keep = width - 2;
    for (int32_t seq = 0; seq < batch; ++seq) {
        const int32_t start = seq * seqLen;
        const int32_t len = seqLen;
        const int32_t cacheIdx = static_cast<int32_t>(cacheIndices[seq]);
        if (cacheIdx == padSlotId) {
            continue;
        }
        int32_t offset = static_cast<int32_t>(numAcceptedTokens[seq]) - 1;
        const int32_t maxOffset = stateLen - (width - 1);
        if (offset < 0) {
            offset = 0;
        } else if (offset > maxOffset) {
            offset = maxOffset;
        }

        std::vector<float> hist(width - 1, 0.0f);
        std::vector<float> shift(keep, 0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            if (initialStateMode[seq] != 0) {
                for (int32_t h = 0; h < width - 1; ++h) {
                    hist[h] = convStates[(cacheIdx * stateLen + (offset + h)) * dim + c];
                }
                for (int32_t k = 0; k < keep; ++k) {
                    shift[k] = convStates[(cacheIdx * stateLen + (offset + 1 + k)) * dim + c];
                }
            } else {
                std::fill(hist.begin(), hist.end(), 0.0f);
                std::fill(shift.begin(), shift.end(), 0.0f);
            }

            // Forward convolution.
            for (int32_t t = 0; t < len; ++t) {
                float acc = bias != nullptr ? bias[c] : 0.0f;
                for (int32_t j = 0; j < width; ++j) {
                    const int32_t srcT = t - (width - 1) + j;
                    float xval = 0.0f;
                    if (srcT >= 0) {
                        xval = x[(start + srcT) * dim + c];
                    } else {
                        const int32_t histIndex = (width - 1) + srcT;
                        xval = hist[histIndex];
                    }
                    acc += weight[j * dim + c] * xval;
                }
                if (activationMode != 0) {
                    acc = SiluRef(acc);
                }
                y[(start + t) * dim + c] = acc;
            }

            // Spec decode state update:
            //   conv_states[0:keep] = old[offset+1 : offset+1+keep]
            //   conv_states[keep:keep+len] = x[start:start+len]
            for (int32_t k = 0; k < keep; ++k) {
                convStates[(cacheIdx * stateLen + k) * dim + c] = shift[k];
            }
            for (int32_t t = 0; t < len; ++t) {
                convStates[(cacheIdx * stateLen + (keep + t)) * dim + c] = x[(start + t) * dim + c];
            }
        }
    }
}

TEST_F(causal_conv1d_test, fwd_3d_batch_mode)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;
    // Test 3D batch mode: x.shape = (batch, seqlen, dim) with inputMode=1
    // This uses fixed seqLen for all sequences instead of queryStartLoc
    constexpr int32_t batch = 3;
    constexpr int32_t dim = 1024;
    constexpr int32_t seqLen = 4;  // Fixed sequence length for all batches
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t numCacheLines = 8;
    constexpr int32_t cuSeqlen = batch * seqLen;  // 12 total tokens
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;  // SiLU

    const std::vector<int64_t> cacheIndices = {1, 3, 5};
    const std::vector<int64_t> hasInitialState = {1, 0, 1};

    const size_t xBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);  // Still allocated but not used
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    // Initialize inputs - physical layout is (cuSeqlen, dim) with dim contiguous
    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.01f * static_cast<float>(c) + 0.1f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    // Seed cache lines with distinct history
    for (int32_t line : {1, 5}) {
        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t h = 0; h < stateLen; ++h) {
                s[(line * stateLen + h) * dim + c] = FromFloat<T>(0.5f + 0.01f * static_cast<float>(c) +
                                                                 0.001f * static_cast<float>(h) +
                                                                 0.1f * static_cast<float>(line));
            }
        }
    }
    // queryStartLoc is not used in 3D mode but still needs to be allocated
    for (int32_t i = 0; i <= batch; ++i) {
        qsl[i] = i * seqLen;  // Dummy values
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    // Compute reference using 3D batch mode function
    std::vector<float> xRef(cuSeqlen * dim);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(cuSeqlen * dim, 0.0f);
    ReferenceCausalConv1dFwd3DBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width,
                                    stateLen, seqLen, batch, idx, his, activationMode, padSlotId);

    // Quantize expected output/state to DTYPE_X
    std::vector<float> yRefQ(cuSeqlen * dim);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    // Set up tiling data for 3D batch mode
    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = seqLen;      // Non-zero for 3D batch mode
    tilingData->inputMode = 1;        // 3D batch mode
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 1024;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 4;
    tilingData->tokenBlockCnt = 3;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, 1, 1>(
        tilingData->tokenBlockCnt, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    // Check output
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }

    // Check state writeback
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}


// Test weight shape: (width, dim), dim contiguous
TEST_F(causal_conv1d_test, fwd_weight_shape_width_dim)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;
    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 6;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 3, 6};
    const std::vector<int64_t> cacheIndices = {0, 1};
    const std::vector<int64_t> hasInitialState = {1, 0};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    // Initialize x
    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.02f * static_cast<float>(c) + 0.15f * static_cast<float>(t));
        }
    }

    // Initialize weight in (width, dim) layout: w[j, c] = w[j * dim + c]
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.002f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }

    // Initialize bias
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.02f * static_cast<float>(c));
    }

    // Initialize conv states
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<T>(0.3f + 0.01f * static_cast<float>(c) +
                                                           0.001f * static_cast<float>(h));
        }
    }

    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    // Compute reference
    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                  qsl, batch, idx, his, activationMode, padSlotId);

    // Quantize expected output/state to DTYPE_X
    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 1024;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 2;
    tilingData->tokenBlockCnt = 3;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, 1, 1>(
        tilingData->tokenBlockCnt, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    // Verify output
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    // Verify state writeback
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, decode_2d_batch_mode)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;

    constexpr int32_t batch = 3;
    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t numCacheLines = 8;
    constexpr int32_t cuSeqlen = batch;  // decode 2D: x.shape=(batch, dim)
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 1, 2, 3};
    const std::vector<int64_t> cacheIndices = {1, 3, 5};
    const std::vector<int64_t> hasInitialState = {1, 1, 1};

    const size_t xBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.02f * static_cast<float>(c) + 0.2f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.002f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.02f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t line : {1, 5}) {
        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t h = 0; h < stateLen; ++h) {
                s[(line * stateLen + h) * dim + c] = FromFloat<T>(0.4f + 0.01f * static_cast<float>(c) +
                                                                  0.001f * static_cast<float>(h) +
                                                                  0.1f * static_cast<float>(line));
            }
        }
    }

    for (int32_t i = 0; i <= batch; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(cuSeqlen * dim);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(cuSeqlen * dim, 0.0f);
    ReferenceCausalConv1dDecode2D(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(),
                                  dim, width, stateLen, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(cuSeqlen * dim);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 1;
    tilingData->inputMode = 2;  // decode 2D mode
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 1024;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 0;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunUpdateKernelWithKey<1, 1>(batch, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
                                 reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, decode_2d_mode_ignores_fn_plan_when_run_mode_is_update)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;
    constexpr int32_t batch = 3;
    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t numCacheLines = 8;
    constexpr int32_t cuSeqlen = batch;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 1, 2, 3};
    const std::vector<int64_t> cacheIndices = {1, 3, 5};
    const std::vector<int64_t> updateHasHistory = {1, 1, 1};

    const size_t xBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.05f * static_cast<float>(c) + 0.3f * static_cast<float>(t + 1));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.0015f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(777.0f + static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t line : {1, 3, 5}) {
        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t h = 0; h < stateLen; ++h) {
                s[(line * stateLen + h) * dim + c] = FromFloat<T>(0.7f + 0.02f * static_cast<float>(c) +
                                                                  0.01f * static_cast<float>(h) +
                                                                  0.05f * static_cast<float>(line));
            }
        }
    }

    for (int32_t i = 0; i <= batch; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = updateHasHistory[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(cuSeqlen * dim);
    std::vector<float> wRef(width * dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(cuSeqlen * dim, 0.0f);
    ReferenceCausalConv1dDecode2D(xRef.data(), wRef.data(), nullptr, yRef.data(), sRef.data(),
                                  dim, width, stateLen, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(cuSeqlen * dim);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 1;
    tilingData->inputMode = 2;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 1024;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 0;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/false);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunUpdateKernelWithKey<1, 0>(batch, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
                                 reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_2d_varlen_unified_token_tiling_kernel)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;
    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 1;
    constexpr int32_t numCacheLines = 2;
    constexpr int32_t cuSeqlen = 5;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;
    constexpr int32_t tokenBlockSize = 2;
    constexpr int32_t tokenBlockCnt = 3;

    const std::vector<int64_t> queryStartLoc = {0, 5};
    const std::vector<int64_t> cacheIndices = {0};
    const std::vector<int64_t> hasInitialState = {1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.01f * static_cast<float>(c) + 0.12f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.02f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<T>(0.25f + 0.01f * static_cast<float>(c) +
                                                           0.002f * static_cast<float>(h));
        }
    }

    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width,
                                  stateLen, qsl, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = dim;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = tokenBlockSize;
    tilingData->tokenBlockCnt = tokenBlockCnt;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, 1, 1>(
        tokenBlockCnt, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_2d_varlen_unified_token_tiling_kernel_no_bias)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;
    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 1;
    constexpr int32_t numCacheLines = 2;
    constexpr int32_t cuSeqlen = 5;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;
    constexpr int32_t tokenBlockSize = 2;
    constexpr int32_t tokenBlockCnt = 3;

    const std::vector<int64_t> queryStartLoc = {0, 5};
    const std::vector<int64_t> cacheIndices = {0};
    const std::vector<int64_t> hasInitialState = {1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.01f * static_cast<float>(c) + 0.12f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(1.5f + 0.02f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<T>(0.25f + 0.01f * static_cast<float>(c) +
                                                           0.002f * static_cast<float>(h));
        }
    }

    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), nullptr, yRef.data(), sRef.data(), dim, width,
                                  stateLen, qsl, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = dim;
    tilingData->baseDimCnt = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = tokenBlockSize;
    tilingData->tokenBlockCnt = tokenBlockCnt;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/false);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, 1, 0, CAUSAL_CONV1D_TPL_WIDTH_4>(
        tokenBlockCnt, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_width3_basic)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;

    constexpr int32_t dim = 1024;
    constexpr int32_t width = 3;
    constexpr int32_t stateLen = width - 1;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 4;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 2, 4};
    const std::vector<int64_t> cacheIndices = {0, 1};
    const std::vector<int64_t> hasInitialState = {1, 0};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.01f * static_cast<float>(c) + 0.05f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    // Seed cache line 0 for initial state.
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<T>(0.2f + 0.01f * static_cast<float>(c) +
                                                          0.001f * static_cast<float>(h));
        }
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                  qsl, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 1024;
    tilingData->baseDimCnt = 1;
    tilingData->hasNumAcceptedTokens = 0;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 3;
    tilingData->tokenBlockCnt = 2;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, 1, 1, CAUSAL_CONV1D_TPL_WIDTH_3>(
        tilingData->tokenBlockCnt, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_width2_dim4160_default_dim_tail_contract)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 6e-2f : 3e-2f;

    constexpr int32_t dim = 4160;
    constexpr int32_t width = 2;
    constexpr int32_t stateLen = width - 1;
    constexpr int32_t batch = 1;
    constexpr int32_t numCacheLines = 2;
    constexpr int32_t cuSeqlen = 3;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 0;

    const std::vector<int64_t> queryStartLoc = {0, 3};
    const std::vector<int64_t> cacheIndices = {0};
    const std::vector<int64_t> hasInitialState = {1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.012f * static_cast<float>(t) +
                                          0.0008f * static_cast<float>(c % 37));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.0025f * static_cast<float>(j + 1) +
                                          0.00015f * static_cast<float>(c % 23));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.002f * static_cast<float>((c % 11) - 5));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        s[(0 * stateLen + 0) * dim + c] =
            FromFloat<T>(0.03f + 0.0009f * static_cast<float>(c % 29));
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                  qsl, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 4096;
    tilingData->baseDimCnt = 2;  // ceil(4160 / 4096)
    tilingData->hasNumAcceptedTokens = 0;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 1;
    tilingData->tokenBlockCnt = cuSeqlen;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    const int32_t gridSize = tilingData->tokenBlockCnt * static_cast<int32_t>(tilingData->baseDimCnt);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD, 0, 1, CAUSAL_CONV1D_TPL_WIDTH_2>(
        gridSize, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_width2_dim4160_split384_tail_with_silu)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;

    constexpr int32_t dim = 4160;
    constexpr int32_t width = 2;
    constexpr int32_t stateLen = width - 1;
    constexpr int32_t batch = 1;
    constexpr int32_t numCacheLines = 2;
    constexpr int32_t cuSeqlen = 2;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 2};
    const std::vector<int64_t> cacheIndices = {0};
    const std::vector<int64_t> hasInitialState = {1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.01f * static_cast<float>(c) + 0.03f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.01f * static_cast<float>(c % 13));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        s[(0 * stateLen + 0) * dim + c] = FromFloat<T>(0.2f + 0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                  qsl, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 384;
    tilingData->baseDimCnt = 11;  // ceil(4160 / 384), tail=320
    tilingData->hasNumAcceptedTokens = 0;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 2;
    tilingData->tokenBlockCnt = 1;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    const int32_t gridSize = tilingData->tokenBlockCnt * static_cast<int32_t>(tilingData->baseDimCnt);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD, 1, 1, CAUSAL_CONV1D_TPL_WIDTH_2>(
        gridSize, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_width2_dim4160_split384_tail_with_silu_no_bias)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;

    constexpr int32_t dim = 4160;
    constexpr int32_t width = 2;
    constexpr int32_t stateLen = width - 1;
    constexpr int32_t batch = 1;
    constexpr int32_t numCacheLines = 2;
    constexpr int32_t cuSeqlen = 2;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int64_t> queryStartLoc = {0, 2};
    const std::vector<int64_t> cacheIndices = {0};
    const std::vector<int64_t> hasInitialState = {1};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.012f * static_cast<float>(c) + 0.05f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(100.0f + 0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        s[(0 * stateLen + 0) * dim + c] = FromFloat<T>(0.2f + 0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = 0;
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), nullptr, yRef.data(), sRef.data(), dim, width, stateLen,
                                  qsl, batch, idx, his, activationMode, padSlotId);

    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = 0;
    tilingData->inputMode = 0;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 384;
    tilingData->baseDimCnt = 11;  // ceil(4160 / 384), tail=320
    tilingData->hasNumAcceptedTokens = 0;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 1;
    tilingData->tokenBlockSize = 2;
    tilingData->tokenBlockCnt = 1;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/false);
    PrepareFnTokenTilingForTest(tilingData);
    PrepareExplicitFnTokenSeqRangesForTest(tilingData, qsl);

    const int32_t gridSize = tilingData->tokenBlockCnt * static_cast<int32_t>(tilingData->baseDimCnt);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunPrefillKernelWithKey<CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD, 1, 0, CAUSAL_CONV1D_TPL_WIDTH_2>(
        gridSize, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
        reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, decode_3d_spec_numAcceptedTokens)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;

    constexpr int32_t batch = 2;
    constexpr int32_t dim = 256;
    constexpr int32_t width = 4;
    constexpr int32_t seqLen = 5;
    constexpr int32_t stateLen = (width - 1) + (seqLen - 1);  // == seqLen + 2
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = batch * seqLen;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 0;

    const std::vector<int64_t> cacheIndices = {0, 1};
    const std::vector<int64_t> hasInitialState = {1, 1};
    const std::vector<int64_t> numAcceptedTokens = {2, 4};  // offsets: 1 and 3

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t natBytes = static_cast<size_t>(batch) * sizeof(int64_t);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* natGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(natBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int64_t*>(qslGm);
    auto* idx = reinterpret_cast<int64_t*>(idxGm);
    auto* his = reinterpret_cast<int64_t*>(hisGm);
    auto* nat = reinterpret_cast<int64_t*>(natGm);
    auto* y = reinterpret_cast<T*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.01f * static_cast<float>(c) + 0.02f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.002f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.0f);
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    // Seed convStates with distinct values.
    for (int32_t line : {0, 1}) {
        for (int32_t pos = 0; pos < stateLen; ++pos) {
            for (int32_t c = 0; c < dim; ++c) {
                const float v = 0.1f * static_cast<float>(line) + 0.01f * static_cast<float>(pos) +
                                0.0001f * static_cast<float>(c);
                s[(line * stateLen + pos) * dim + c] = FromFloat<T>(v);
            }
        }
    }
    for (int32_t i = 0; i <= batch; ++i) {
        qsl[i] = i * seqLen;  // dummy, not used in 3D mode
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
        nat[i] = numAcceptedTokens[i];
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    std::vector<int64_t> natRef = numAcceptedTokens;
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dDecode3DSpec(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(),
                                      dim, width, stateLen, seqLen, batch, idx, his, natRef.data(),
                                      activationMode, padSlotId);

    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = seqLen;
    tilingData->inputMode = 1;  // 3D batch/update
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->padSlotId = padSlotId;
    tilingData->baseDim = 384;  // dim < tile, use partial
    tilingData->baseDimCnt = 1;
    tilingData->hasNumAcceptedTokens = 1;
    tilingData->hasCacheIndices = 1;
    tilingData->hasInitialStateMode = 0;
    SetRuntimeFeatureFlagsForTest(tilingData, activationMode, /*hasBias=*/true);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunUpdateKernelWithKey<0, 1>(batch, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, natGm, yGm, workspace,
                                 reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(natGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
