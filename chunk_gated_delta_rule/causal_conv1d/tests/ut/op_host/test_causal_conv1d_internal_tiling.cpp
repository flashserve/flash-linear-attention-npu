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

#include <gtest/gtest.h>

#include "../causal_conv1d_tiling_key_test_helper.h"
#include "../../../op_host/causal_conv1d_tiling_planner.h"
#include "../../../op_host/causal_conv1d_tiling_utils.h"
#include "../../../op_kernel/causal_conv1d_tiling_data.h"

using namespace optiling::causal_conv1d_host;

TEST(CausalConv1dTilingInternal, resolve_fn_execution_plan_distinguishes_cutbs_and_cutbsd)
{
    EXPECT_EQ(ResolveFnExecutionPlan(/*baseDimCnt=*/0), FN_EXECUTION_PLAN_INVALID);
    EXPECT_EQ(ResolveFnExecutionPlan(/*baseDimCnt=*/1), FN_EXECUTION_PLAN_CUTBS);
    EXPECT_EQ(ResolveFnExecutionPlan(/*baseDimCnt=*/2), FN_EXECUTION_PLAN_CUTBSD);
}

TEST(CausalConv1dTilingInternal, choose_fn_token_block_choice_uses_exact_ideal_block_size_for_cutbs)
{
    const auto tokenChoice = ChooseFnTokenBlockChoice(4096, 1, FN_EXECUTION_PLAN_CUTBS, 40);

    EXPECT_TRUE(tokenChoice.enabled);
    EXPECT_EQ(tokenChoice.tokenBlockSize, 103);
    EXPECT_EQ(tokenChoice.tokenBlockCnt, 40);
    EXPECT_EQ(tokenChoice.gridSize, 40);
}

TEST(CausalConv1dTilingInternal, choose_fn_token_block_choice_uses_exact_ideal_block_size_for_cutbsd)
{
    const auto tokenChoice = ChooseFnTokenBlockChoice(32, 2, FN_EXECUTION_PLAN_CUTBSD, 40);

    EXPECT_TRUE(tokenChoice.enabled);
    EXPECT_EQ(tokenChoice.tokenBlockSize, 2);
    EXPECT_EQ(tokenChoice.tokenBlockCnt, 16);
    EXPECT_EQ(tokenChoice.gridSize, 32);
}

TEST(CausalConv1dTilingInternal, build_fn_token_core_mapping_choice_keeps_block_to_core_mapping_explainable)
{
    const auto mapping = BuildFnTokenCoreMappingChoice(/*tokenBlockCnt=*/65, /*baseDimCnt=*/2,
                                                       FN_EXECUTION_PLAN_CUTBSD, /*coreNum=*/40);

    EXPECT_EQ(mapping.tokenCoreBudget, 20);
    EXPECT_EQ(mapping.tokenBlocksPerCore, 4);
    EXPECT_EQ(mapping.tokenCoreTailCnt, 5);
    EXPECT_EQ(mapping.blockDim, 40);
}

TEST(CausalConv1dTilingInternal, build_fn_token_seq_range_plan_maps_token_tiles_to_seq_ranges)
{
    const int64_t qslData[] = {0, 2, 5, 6};
    const auto plan = BuildFnTokenSeqRangePlan(qslData, /*batch=*/3, /*tokenBlockSize=*/2, /*tokenBlockCnt=*/3);

    ASSERT_TRUE(plan.enabled);
    ASSERT_EQ(plan.rangeCount, 3);
    EXPECT_EQ(plan.tokenTileStartSeq[0], 0);
    EXPECT_EQ(plan.tokenTileEndSeq[0], 1);
    EXPECT_EQ(plan.tokenTileStartSeq[1], 1);
    EXPECT_EQ(plan.tokenTileEndSeq[1], 2);
    EXPECT_EQ(plan.tokenTileStartSeq[2], 1);
    EXPECT_EQ(plan.tokenTileEndSeq[2], 3);
}

TEST(CausalConv1dTilingInternal, build_fn_token_seq_range_plan_disables_when_token_tile_count_exceeds_budget)
{
    const int64_t qslData[] = {0, 1024};
    const auto plan = BuildFnTokenSeqRangePlan(qslData, /*batch=*/1, /*tokenBlockSize=*/1,
                                               MAX_FN_TOKEN_SEQ_RANGE_COUNT + 1);

    EXPECT_FALSE(plan.enabled);
    EXPECT_EQ(plan.rangeCount, 0);
}

TEST(CausalConv1dTilingInternal, compute_fn_ub_limited_base_dim_respects_kernel_buffer_budget)
{
    EXPECT_EQ(ComputeFnUbLimitedBaseDim(/*ubSize=*/196608), 4096);
    EXPECT_EQ(ComputeFnUbLimitedBaseDim(/*ubSize=*/86528), 1856);
}

TEST(CausalConv1dTilingInternal, build_tiling_key_tracks_runmode_plan_and_width)
{
    const auto fnCutbsW2 = BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, FN_EXECUTION_PLAN_CUTBS, 2, 0, false);
    const auto fnCutbsW4 = BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, FN_EXECUTION_PLAN_CUTBS, 4, 0, false);
    const auto fnCutbsdW4 = BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, FN_EXECUTION_PLAN_CUTBSD, 4, 0, false);
    const auto update = BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, FN_EXECUTION_PLAN_INVALID, 4, 0,
                                                   false);

    EXPECT_NE(fnCutbsW2, fnCutbsW4);
    EXPECT_NE(fnCutbsW4, fnCutbsdW4);
    EXPECT_NE(fnCutbsdW4, update);
    EXPECT_EQ(fnCutbsW2, BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_FN, FN_EXECUTION_PLAN_CUTBS, 2, 0, false));
    EXPECT_EQ(update, BuildCausalConv1dTilingKey(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, FN_EXECUTION_PLAN_INVALID, 4, 0,
                                                 false));
}
