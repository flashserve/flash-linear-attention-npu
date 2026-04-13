/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../causal_conv1d_tiling_key_test_helper.h"
#include "../../../op_kernel/causal_conv1d_tiling_data.h"

using namespace std;
using namespace ge;

class CausalConv1dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dTiling TearDown" << std::endl;
    }
};

struct ExpectedTilingFields {
    int64_t dim = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t inputMode = 0;
    int64_t runMode = 0;
    int64_t width = 0;
    int64_t stateLen = 0;
    int64_t numCacheLines = 0;
    int64_t batch = 0;
    int64_t activationMode = 0;
    int64_t padSlotId = 0;
    int64_t hasBias = 0;
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t hasNumAcceptedTokens = 0;
    int64_t hasCacheIndices = 0;
    int64_t hasInitialStateMode = 0;
};

static ExpectedTilingFields ParseExpectedTilingFields(const std::string& base)
{
    ExpectedTilingFields fields {};
    std::istringstream iss(base);
    iss >> fields.dim >> fields.cuSeqlen >> fields.seqLen >> fields.inputMode >> fields.runMode >> fields.width >>
        fields.stateLen >> fields.numCacheLines >> fields.batch >> fields.activationMode >> fields.padSlotId >>
        fields.hasBias >> fields.baseDim >> fields.baseDimCnt >> fields.hasNumAcceptedTokens >> fields.hasCacheIndices >>
        fields.hasInitialStateMode;
    return fields;
}

static std::string AppendFnTilingData(const std::string& base, int64_t tokenBlockSize = 0, int64_t tokenBlockCnt = 0)
{
    const auto fields = ParseExpectedTilingFields(base);
    std::ostringstream oss;
    oss << fields.dim << " " << fields.cuSeqlen << " " << fields.seqLen << " " << fields.inputMode << " "
        << fields.width << " " << fields.stateLen << " " << fields.numCacheLines << " " << fields.batch << " "
        << fields.activationMode << " " << fields.padSlotId << " " << fields.hasBias << " "
        << fields.baseDim << " " << fields.baseDimCnt << " "
        << fields.hasNumAcceptedTokens << " " << fields.hasCacheIndices << " " << fields.hasInitialStateMode << " "
        << tokenBlockSize << " " << tokenBlockCnt << " "
        << 0 << " " << 0 << " ";
    for (size_t i = 0; i < sizeof(CausalConv1dTilingData{}.tokenTileStartSeq) / sizeof(int64_t); ++i) {
        oss << 0 << " ";
    }
    for (size_t i = 0; i < sizeof(CausalConv1dTilingData{}.tokenTileEndSeq) / sizeof(int64_t); ++i) {
        oss << 0 << " ";
    }
    return oss.str();
}

std::map<std::string, std::string> soc_version_infos = {
    {"Short_SoC_version", "Ascend910B"},
    {"ai_core_cnt", "40"},
    {"vector_core_cnt", "40"},
    {"cube_core_cnt", "0"},
    {"ub_size", "196608"},
};

static inline uint64_t PrefillDefaultTilingKey(int64_t unusedActivationMode = 0, bool unusedHasBias = true,
                                               FnExecutionPlan fnPlan = FN_EXECUTION_PLAN_CUTBS, int32_t width = 4)
{
    return BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, fnPlan, width, unusedActivationMode, unusedHasBias);
}

static inline uint64_t UpdateDefaultTilingKey(int64_t unusedActivationMode = 0, bool unusedHasBias = true)
{
    return BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, FN_EXECUTION_PLAN_INVALID, 4, unusedActivationMode,
                                      unusedHasBias);
}

TEST_F(CausalConv1dTiling, causal_conv1d_0) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: (num_cache_lines, state_len, dim)
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},          // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},          // cache_indices: (batch)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},          // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},          // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey();
    // dim, cuSeqlen, seqLen, inputMode, runMode, width, stateLen, numCacheLines, batch,
    // activationMode, padSlotId, hasBias, baseDim, baseDimCnt
    string expectTilingData = AppendFnTilingData("1024 5 0 0 0 4 3 8 2 0 -1 1 1024 1 0 1 1 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_initial_state_mode_optional_absent_default0) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: (num_cache_lines, state_len, dim)
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},          // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},          // cache_indices: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},          // initial_state_mode (optional, absent => default 0)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},          // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey();
    // Last flags: hasNumAcceptedTokens, hasCacheIndices, hasInitialStateMode
    string expectTilingData = AppendFnTilingData("1024 5 0 0 0 4 3 8 2 0 -1 1 1024 1 0 1 0 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_cache_indices_optional_absent_identity_mapping) {
    // cacheIndices absent => identity mapping cacheIdx=seq (requires num_cache_lines >= batch).
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: (num_cache_lines, state_len, dim)
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},          // query_start_loc: (batch+1)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},          // cache_indices (optional, absent => identity mapping)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},          // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},          // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey();
    // Last flags: hasNumAcceptedTokens, hasCacheIndices, hasInitialStateMode
    string expectTilingData = AppendFnTilingData("1024 5 0 0 0 4 3 8 2 0 -1 1 1024 1 0 0 1 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_cache_indices_optional_absent_requires_num_cache_lines_ge_batch) {
    // cacheIndices absent requires num_cache_lines >= batch for identity mapping.
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{1, 3, 1024}, {1, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: (num_cache_lines=1, state_len, dim)
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},          // query_start_loc: (batch+1) => batch=2
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},          // cache_indices (optional, absent)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},          // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},          // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_1) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},         // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},         // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},         // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},         // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey(/*activationMode=*/1);
    string expectTilingData = AppendFnTilingData("1024 5 0 0 0 4 3 8 2 1 -1 1 1024 1 0 1 1 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_1d_tiling_no_bias) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},            // query_start_loc: (batch+1)
                                                    {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},            // cache_indices: (batch)
                                                    {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},            // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},            // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey();
    string expectTilingData = AppendFnTilingData("1024 5 0 0 0 4 3 8 1 0 -1 1 1024 1 0 1 1 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_3d_batch_mode) {
    // Test 3D batch mode: x.shape = (batch, seqlen, dim)
    // This triggers inputMode=1 with fixed seqLen for all sequences
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{4, 8, 1024}, {4, 8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch=4, seqlen=8, dim=1024)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias: (dim)
                                                    {{{16, 3, 1024}, {16, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: (num_cache_lines, state_len, dim)
                                                    {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND},               // query_start_loc: (batch+1) - still required but ignored in 3D mode
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},               // cache_indices: (batch)
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},               // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},               // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{4, 8, 1024}, {4, 8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y: (batch, seqlen, dim)
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    // 3D batch mode: batch=4, dim=1024, seqLen=8, cuSeqlen=32 (4*8), inputMode=1, runMode=0
    uint64_t expectTilingKey = PrefillDefaultTilingKey(/*activationMode=*/1);
    // Expected: dim=1024, cuSeqlen=32, seqLen=8, inputMode=1, runMode=0, width=4, stateLen=3,
    //           numCacheLines=16, batch=4, activationMode=1, padSlotId=-1, hasBias=1,
    //           baseDim=512, baseDimCnt=2
    string expectTilingData = AppendFnTilingData("1024 32 8 1 0 4 3 16 4 1 -1 1 1024 1 0 1 1 ", 1, 32);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_weight_shape_width_dim) {
    // Verify (width, dim) weight shape is accepted (dim contiguous).
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},            // query_start_loc: (batch+1)
                                                    {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},            // cache_indices: (batch)
                                                    {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},            // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},            // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey();
    // dim, cuSeqlen, seqLen, inputMode, runMode, width, stateLen, numCacheLines, batch,
    // activationMode, padSlotId, hasBias, baseDim, baseDimCnt
    string expectTilingData = AppendFnTilingData("1024 5 0 0 0 4 3 8 1 0 -1 1 1024 1 0 1 1 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_decode_2d_mode) {
    // decode 2D mode: runMode=1 + x.shape=(batch, dim)
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch=4, dim=1024)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias: (dim)
                                                    {{{16, 3, 1024}, {16, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND},              // query_start_loc: (batch+1)
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},              // cache_indices: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},              // initial_state_mode: absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},              // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y: (batch, dim)
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = UpdateDefaultTilingKey(/*activationMode=*/1);
    // dim=1024, cuSeqlen=4, seqLen=1, inputMode=2(2D decode), runMode=1
    string expectTilingData = AppendFnTilingData("1024 4 1 2 1 4 3 16 4 1 -1 1 512 2 0 1 0 ");
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_decode_3d_mode_seqlen2) {
    // decode 3D mode: runMode=1 + x.shape=(batch, seqlen, dim), seqlen 全泛化（>0）
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{2, 2, 1024}, {2, 2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch=2, seqlen=2, dim=1024)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},            // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                   // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                   // cache_indices: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                   // initial_state_mode: absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                   // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{2, 2, 1024}, {2, 2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = UpdateDefaultTilingKey();
    string expectTilingData = AppendFnTilingData("1024 4 2 1 1 4 3 8 2 0 -1 1 512 2 0 1 0 ");
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_decode_3d_mode_seqlen5) {
    // runMode=1 decode supports generalized seqlen (>0)
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: seqlen=5
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                    // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // cache_indices
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                    // initial_state_mode: absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                    // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = UpdateDefaultTilingKey();
    string expectTilingData = AppendFnTilingData("1024 10 5 1 1 4 3 8 2 0 -1 1 512 2 0 1 0 ");
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_update_uses_canonical_dim_choice_for_dim384) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{4, 384}, {4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 384}, {4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{384}, {384}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias
                                                    {{{16, 3, 384}, {16, 3, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND},             // query_start_loc
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},             // cache_indices
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},             // initial_state_mode: absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},             // num_accepted_tokens
                                                },
                                                {
                                                    {{{4, 384}, {4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo,
                                                "Ascend910B",
                                                /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, UpdateDefaultTilingKey());
    ASSERT_EQ(tilingInfo.blockNum, 8U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.inputMode, 2);
    EXPECT_EQ(tilingData.batch, 4);
    EXPECT_EQ(tilingData.dim, 384);
    EXPECT_EQ(tilingData.baseDim, 192);
    EXPECT_EQ(tilingData.baseDimCnt, 2);
}

TEST_F(CausalConv1dTiling, causal_conv1d_update_prefers_exact_divisor_before_tail_choice) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias
                                                    {{{16, 3, 1024}, {16, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},             // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},             // cache_indices
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},             // initial_state_mode: absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},             // num_accepted_tokens
                                                },
                                                {
                                                    {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo,
                                                "Ascend910B",
                                                /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.blockNum, 4U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.baseDim, 512);
    EXPECT_EQ(tilingData.baseDimCnt, 2);
}

TEST_F(CausalConv1dTiling, causal_conv1d_dim_multiple_384) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 384}, {5, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 384}, {4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{384}, {384}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 3, 384}, {8, 3, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},              // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},              // cache_indices: (batch)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},              // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},              // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 384}, {5, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey();
    // Current update planner keeps the smaller tile when all candidates are underutilized.
    string expectTilingData = AppendFnTilingData("384 5 0 0 0 4 3 8 2 0 -1 1 384 1 0 1 1 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_dim_upper_bound_12288) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 12288}, {5, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 12288}, {4, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias: (dim)
                                                    {{{8, 3, 12288}, {8, 3, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                 // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                 // cache_indices: (batch)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                 // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                 // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 12288}, {5, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = PrefillDefaultTilingKey(/*activationMode=*/0, /*hasBias=*/true,
                                                       FN_EXECUTION_PLAN_CUTBSD);
    // dim=12288 -> CUTBSD path uses UB-limited baseDim, then adjusts baseDimCnt to keep token-core mapping divisible.
    string expectTilingData = AppendFnTilingData("12288 5 0 0 0 4 3 8 2 0 -1 1 3072 4 0 1 1 ", 1, 5);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_dim_tail_4160) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 4160}, {5, 4160}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 4160}, {4, 4160}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{4160}, {4160}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias: (dim)
                                                    {{{8, 3, 4160}, {8, 3, 4160}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                 // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                 // cache_indices: (batch)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                 // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                 // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 4160}, {5, 4160}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    const uint64_t expectTilingKey =
        PrefillDefaultTilingKey(/*activationMode=*/0, /*hasBias=*/true, FN_EXECUTION_PLAN_CUTBSD);
    const std::string expectTilingData =
        AppendFnTilingData("4160 5 0 0 0 4 3 8 2 0 -1 1 4096 2 0 1 1 ", 1, 5);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_dim_not_multiple_16_rejected) {
    // dim must be rejected at host validation when it is not divisible by 16.
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1000}, {5, 1000}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1000}, {4, 1000}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1000}, {1000}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias: (dim)
                                                    {{{8, 3, 1000}, {8, 3, 1000}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},              // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},              // cache_indices: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},              // initial_state_mode: absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},              // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1000}, {5, 1000}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_dim_multiple_16_minimal_aligned_case_passes) {
    // The minimum aligned dim accepted by the public contract is 16.
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 16}, {5, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 16}, {4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias: (dim)
                                                    {{{8, 3, 16}, {8, 3, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},           // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},           // cache_indices: (batch)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},           // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},           // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 16}, {5, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, PrefillDefaultTilingKey(), "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_width2_stateLen1) {
    // width=2 + stateLen=1 (minimum) should be accepted.
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width=2, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias
                                                    {{{8, 1, 1024}, {8, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: state_len=1
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},               // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},               // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},               // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},               // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, PrefillDefaultTilingKey(0, true, FN_EXECUTION_PLAN_CUTBS, 2), "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_width3_stateLen2) {
    // width=3 + stateLen=2 (minimum) should be accepted.
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{3, 1024}, {3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width=3, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias
                                                    {{{8, 2, 1024}, {8, 2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: state_len=2
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},               // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},               // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},               // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},               // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, PrefillDefaultTilingKey(0, true, FN_EXECUTION_PLAN_CUTBS, 3), "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_decode_2d_varlen_mode) {
    // decode 2D varlen mode: runMode=1 + x.shape=(cu_seqlen, dim), sliced by query_start_loc
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen=5, dim=1024)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},              // query_start_loc: (batch+1=3) => batch=2
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},              // cache_indices: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},              // initial_state_mode: absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},              // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = UpdateDefaultTilingKey();
    // dim=1024, cuSeqlen=5, seqLen=0(varlen), inputMode=0, runMode=1
    string expectTilingData = AppendFnTilingData("1024 5 0 0 1 4 3 8 2 0 -1 1 512 2 0 1 0 ");
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_decode_3d_spec_numAcceptedTokens) {
    // decode 3D mode + speculative decoding metadata: numAcceptedTokens present
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch=2, seqlen=5, dim=1024)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
                                                    {{{8, 7, 1024}, {8, 7, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: state_len=7 >= (3 + 4)
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                    // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // cache_indices
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                    // initial_state_mode: absent in runMode=1
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // num_accepted_tokens: (batch)
                                                },
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = UpdateDefaultTilingKey();
    string expectTilingData = AppendFnTilingData("1024 10 5 1 1 4 7 8 2 0 -1 1 512 2 1 1 0 ");
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_update_rejects_initial_state_mode) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                    // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                    // num_accepted_tokens
                                                },
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_prefill_rejects_numAcceptedTokens) {
    // runMode=0 keeps hasNumAcceptedTokens at the default 0 and must reject speculative metadata at host side.
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                    // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // initial_state_mode
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                    // num_accepted_tokens
                                                },
                                                {
                                                    {{{2, 5, 1024}, {2, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_query_start_loc_valuedepend_valid) {
    // queryStartLoc values should be validated when host-side data is provided (ValueDepend).
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> qslData = {0, 2, 5};  // batch=2, cuSeqlen=5
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, qslData.data()}, // query_start_loc (const host data)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, PrefillDefaultTilingKey(), "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_query_start_loc_valuedepend_invalid_first_not_zero) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> qslData = {1, 2, 5};  // invalid: first must be 0
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, qslData.data()}, // query_start_loc (const host data)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_query_start_loc_valuedepend_invalid_last_not_cuSeqlen) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> qslData = {0, 2, 4};  // invalid: last must equal cuSeqlen (5)
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, qslData.data()}, // query_start_loc (const host data)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_query_start_loc_valuedepend_invalid_decreasing) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> qslData = {0, 3, 2, 5};  // invalid: must be non-decreasing
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: cuSeqlen=5
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, qslData.data()}, // query_start_loc (batch=3)
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                  // cache_indices
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                  // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_query_start_loc_optional_absent_allowed_in_3d) {
    // queryStartLoc is optional for 3D batch mode (inputMode=1).
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{4, 8, 1024}, {4, 8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch, seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
                                                    {{{16, 3, 1024}, {16, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                     // query_start_loc absent
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                     // cache_indices
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                     // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                     // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{4, 8, 1024}, {4, 8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, PrefillDefaultTilingKey(/*activationMode=*/1), "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_query_start_loc_optional_absent_rejected_in_2d_varlen) {
    // queryStartLoc is required for 2D varlen mode (inputMode=0).
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // query_start_loc absent (invalid)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                // cache_indices
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_query_start_loc_optional_absent_allowed_in_decode_2d) {
    // queryStartLoc is optional for 2D decode mode (inputMode=2).
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // query_start_loc absent
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},                // cache_indices
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // initial_state_mode absent in runMode=1
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UpdateDefaultTilingKey(), "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_cache_indices_valuedepend_pad_slot_allowed) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> ciData = {-1, 0};
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, ciData.data()}, // cache_indices (const host data)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // initial_state_mode absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // num_accepted_tokens absent
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, PrefillDefaultTilingKey(), "", {0});
}

TEST_F(CausalConv1dTiling, causal_conv1d_cache_indices_valuedepend_out_of_range_rejected) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> ciData = {0, 8};
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: num_cache_lines=8
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, ciData.data()}, // cache_indices (const host data)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // initial_state_mode absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // num_accepted_tokens absent
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_cache_indices_valuedepend_too_large_for_int32_rejected) {
    struct CausalConv1dCompileInfo {} compileInfo;
    const int64_t tooLarge = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    std::vector<int64_t> ciData = {tooLarge, 0};
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{tooLarge + 1, 3, 1024}, {tooLarge + 1, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, ciData.data()}, // cache_indices (const host data)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // initial_state_mode absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // num_accepted_tokens absent
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_initial_state_mode_valuedepend_invalid_rejected) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> ismData = {0, 2};
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                // query_start_loc: (batch+1)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // cache_indices absent
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, ismData.data()}, // initial_state_mode (const host data)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // num_accepted_tokens absent
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_num_accepted_tokens_valuedepend_decode_2d_gt1_rejected) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> natData = {2, 1, 1, 1};
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // query_start_loc absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // cache_indices absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // initial_state_mode absent
                                                    {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, natData.data()}, // num_accepted_tokens (const host data)
                                                },
                                                {
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_num_accepted_tokens_valuedepend_decode_3d_gt_seqlen_rejected) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> natData = {4, 2};
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{2, 3, 1024}, {2, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch, seqlen=3, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias
                                                    {{{8, 7, 1024}, {8, 7, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: state_len=7
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                     // query_start_loc absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                     // cache_indices absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                     // initial_state_mode absent
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, natData.data()}, // num_accepted_tokens (const host data)
                                                },
                                                {
                                                    {{{2, 3, 1024}, {2, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_num_accepted_tokens_valuedepend_decode_varlen_gt_querylen_rejected) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> qslData = {0, 2, 5};   // lens: 2, 3
    std::vector<int64_t> natData = {3, 1};      // invalid: nat[0] > len[0]
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, qslData.data()}, // query_start_loc (const host data)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // cache_indices absent
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                // initial_state_mode absent
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, natData.data()}, // num_accepted_tokens (const host data)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                },
                                                &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CausalConv1dTiling, causal_conv1d_3d_longseq_lowbatch_uses_fn_cutbs_entry) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            {{{1, 16352, 768}, {1, 16352, 768}}, ge::DT_BF16, ge::FORMAT_ND},   // x
            {{{4, 768}, {4, 768}}, ge::DT_BF16, ge::FORMAT_ND},                  // weight
            {{{768}, {768}}, ge::DT_BF16, ge::FORMAT_ND},                        // bias
            {{{2693, 6, 768}, {2693, 6, 768}}, ge::DT_BF16, ge::FORMAT_ND},      // conv_states
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                           // query_start_loc absent
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                           // cache_indices
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                           // initial_state_mode
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                           // num_accepted_tokens absent
        },
        {
            {{{1, 16352, 768}, {1, 16352, 768}}, ge::DT_BF16, ge::FORMAT_ND},   // y
        },
        {
            {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend910B",
        /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, PrefillDefaultTilingKey(/*activationMode=*/1));

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.inputMode, 1);
    EXPECT_EQ(tilingData.batch, 1);
    EXPECT_EQ(tilingData.dim, 768);
    EXPECT_EQ(tilingData.baseDim, 768);
    EXPECT_EQ(tilingData.baseDimCnt, 1);
    EXPECT_EQ(tilingData.tokenBlockSize, 409);
    EXPECT_EQ(tilingData.tokenBlockCnt, 40);
}

TEST_F(CausalConv1dTiling, causal_conv1d_fn_dim_within_max_block_only_splits_tokens_for_3d_prefill) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            {{{2, 16, 1024}, {2, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // x
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weight
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // bias
            {{{16, 3, 1024}, {16, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // conv_states
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                         // query_start_loc absent
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                         // cache_indices
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                         // initial_state_mode
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                         // num_accepted_tokens absent
        },
        {
            {{{2, 16, 1024}, {2, 16, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // y
        },
        {
            {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend910B",
        /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, PrefillDefaultTilingKey());
    ASSERT_EQ(tilingInfo.blockNum, 32U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.inputMode, 1);
    EXPECT_EQ(tilingData.batch, 2);
    EXPECT_EQ(tilingData.dim, 1024);
    EXPECT_EQ(tilingData.baseDim, 1024);
    EXPECT_EQ(tilingData.baseDimCnt, 1);
    EXPECT_EQ(tilingData.tokenBlockSize, 1);
    EXPECT_EQ(tilingData.tokenBlockCnt, 32);
}

TEST_F(CausalConv1dTiling, causal_conv1d_fn_dim_above_max_block_splits_dim_then_tokens_for_3d_prefill) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // x
            {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // weight
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // bias
            {{{16, 3, 8192}, {16, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // conv_states
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                         // query_start_loc absent
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                         // cache_indices
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                         // initial_state_mode
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                         // num_accepted_tokens absent
        },
        {
            {{{2, 16, 8192}, {2, 16, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // y
        },
        {
            {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend910B",
        /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, PrefillDefaultTilingKey(/*activationMode=*/0, /*hasBias=*/true,
                                                            FN_EXECUTION_PLAN_CUTBSD));
    ASSERT_EQ(tilingInfo.blockNum, 32U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.inputMode, 1);
    EXPECT_EQ(tilingData.batch, 2);
    EXPECT_EQ(tilingData.dim, 8192);
    EXPECT_EQ(tilingData.baseDim, 4096);
    EXPECT_EQ(tilingData.baseDimCnt, 2);
    EXPECT_EQ(tilingData.tokenBlockSize, 2);
    EXPECT_EQ(tilingData.tokenBlockCnt, 16);
}

TEST_F(CausalConv1dTiling, causal_conv1d_fn_cutbsd_uses_divisible_dim_blocks_for_explainable_core_mapping) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            {{{5, 12288}, {5, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // x
            {{{4, 12288}, {4, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // weight
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},                // bias
            {{{8, 3, 12288}, {8, 3, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // conv_states
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},                          // query_start_loc => batch=2
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          // cache_indices
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          // initial_state_mode
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                          // num_accepted_tokens absent
        },
        {
            {{{5, 12288}, {5, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // y
        },
        {
            {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend910B",
        /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey,
              PrefillDefaultTilingKey(/*activationMode=*/0, /*hasBias=*/true, FN_EXECUTION_PLAN_CUTBSD));
    ASSERT_EQ(tilingInfo.blockNum, 20U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.dim, 12288);
    EXPECT_EQ(tilingData.baseDim, 3072);
    EXPECT_EQ(tilingData.baseDimCnt, 4);
    EXPECT_EQ(tilingData.tokenBlockSize, 1);
    EXPECT_EQ(tilingData.tokenBlockCnt, 5);
}

TEST_F(CausalConv1dTiling, causal_conv1d_varlen_prefill_dim_within_max_block_keeps_dim_intact) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{16, 3, 1024}, {16, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND},               // query_start_loc: (batch+1)=6 => batch=5
                                                    {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND},               // cache_indices: (batch)
                                                    {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND},               // initial_state_mode: (batch)
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},               // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo,
                                                "Ascend910B",
                                                /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.blockNum, 5U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.batch, 5);
    EXPECT_EQ(tilingData.dim, 1024);
    EXPECT_EQ(tilingData.baseDim, 1024);
    EXPECT_EQ(tilingData.baseDimCnt, 1);
}

TEST_F(CausalConv1dTiling, causal_conv1d_prefill_keeps_small_dim_unsplit_even_when_underutilized) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{1, 512}, {1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight
                                                    {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias
                                                    {{{16, 3, 512}, {16, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},             // query_start_loc: (batch+1)=2 => batch=1
                                                    {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},             // cache_indices
                                                    {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},             // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},             // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{1, 512}, {1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo,
                                                "Ascend910B",
                                                /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.blockNum, 1U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.batch, 1);
    EXPECT_EQ(tilingData.dim, 512);
    EXPECT_EQ(tilingData.baseDim, 512);
    EXPECT_EQ(tilingData.baseDimCnt, 1);
}

TEST_F(CausalConv1dTiling, causal_conv1d_varlen_prefill_dim_above_max_block_caps_tile_size) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{64, 8192}, {64, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
                                                    {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // weight
                                                    {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // bias
                                                    {{{16, 3, 8192}, {16, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{65}, {65}}, ge::DT_INT64, ge::FORMAT_ND},               // query_start_loc => batch=64
                                                    {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},               // cache_indices
                                                    {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},               // initial_state_mode
                                                    {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                 // num_accepted_tokens (optional)
                                                },
                                                {
                                                    {{{64, 8192}, {64, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo,
                                                "Ascend910B",
                                                /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.blockNum, 32U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.batch, 64);
    EXPECT_EQ(tilingData.dim, 8192);
    EXPECT_EQ(tilingData.baseDim, 4096);
    EXPECT_EQ(tilingData.baseDimCnt, 2);
}

TEST_F(CausalConv1dTiling, causal_conv1d_long_varlen_prefers_fn_cutbs_token_tiling) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> qslData = {0, 16352};
    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            {{{16352, 1536}, {16352, 1536}}, ge::DT_BF16, ge::FORMAT_ND},         // x
            {{{4, 1536}, {4, 1536}}, ge::DT_BF16, ge::FORMAT_ND},                 // weight
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // bias absent
            {{{1, 3, 1536}, {1, 3, 1536}}, ge::DT_BF16, ge::FORMAT_ND},           // conv_states
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, qslData.data()},      // query_start_loc
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                            // cache_indices
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                            // initial_state_mode
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                            // num_accepted_tokens absent
        },
        {
            {{{16352, 1536}, {16352, 1536}}, ge::DT_BF16, ge::FORMAT_ND},         // y
        },
        {
            {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend910B",
        /*coreNum=*/40);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, PrefillDefaultTilingKey(/*activationMode=*/1, /*hasBias=*/false));
    ASSERT_EQ(tilingInfo.blockNum, 40U);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));
    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.inputMode, 0);
    EXPECT_EQ(tilingData.batch, 1);
    EXPECT_EQ(tilingData.dim, 1536);
    EXPECT_EQ(tilingData.baseDim, 1536);
    EXPECT_EQ(tilingData.baseDimCnt, 1);
    EXPECT_EQ(tilingData.tokenBlockSize, 409);
    EXPECT_EQ(tilingData.tokenBlockCnt, 40);
    EXPECT_EQ(tilingData.hasExplicitTokenSeqRanges, 1);
    EXPECT_EQ(tilingData.explicitTokenSeqRangeCount, 40);
    EXPECT_EQ(tilingData.tokenTileStartSeq[0], 0);
    EXPECT_EQ(tilingData.tokenTileEndSeq[0], 1);
    EXPECT_EQ(tilingData.tokenTileStartSeq[39], 0);
    EXPECT_EQ(tilingData.tokenTileEndSeq[39], 1);
}

TEST_F(CausalConv1dTiling,
       causal_conv1d_long_varlen_seq_range_plan_falls_back_when_high_core_budget_exceeds_fixed_capacity) {
    struct CausalConv1dCompileInfo {} compileInfo;
    std::vector<int64_t> qslData = {0, 70000};
    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            {{{70000, 1536}, {70000, 1536}}, ge::DT_BF16, ge::FORMAT_ND},         // x
            {{{4, 1536}, {4, 1536}}, ge::DT_BF16, ge::FORMAT_ND},                 // weight
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                             // bias absent
            {{{1, 3, 1536}, {1, 3, 1536}}, ge::DT_BF16, ge::FORMAT_ND},           // conv_states
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, qslData.data()},      // query_start_loc
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                            // cache_indices
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},                            // initial_state_mode
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                            // num_accepted_tokens absent
        },
        {
            {{{70000, 1536}, {70000, 1536}}, ge::DT_BF16, ge::FORMAT_ND},         // y
        },
        {
            {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend910B",
        /*coreNum=*/160);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(CausalConv1dTilingData));

    CausalConv1dTilingData tilingData {};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(CausalConv1dTilingData));

    EXPECT_EQ(tilingData.tokenBlockSize, 438);
    EXPECT_EQ(tilingData.tokenBlockCnt, 160);
    EXPECT_EQ(tilingData.hasExplicitTokenSeqRanges, 0);
    EXPECT_EQ(tilingData.explicitTokenSeqRangeCount, 0);
}
