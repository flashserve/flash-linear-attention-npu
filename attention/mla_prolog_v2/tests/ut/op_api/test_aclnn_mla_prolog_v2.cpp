/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_mla_prolog_v2_weight_nz.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_mla_prolog_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_mla_prolog_v2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_mla_prolog_v2_test TearDown" << endl;
    }
};

#define PARAM_LIST tokenX,\
    weightDq,\
    weightUqQr,\
    weightUk,\
    weightDkvKr,\
    rmsnormGammaCq,\
    rmsnormGammaCkv,\
    ropeSin,\
    ropeCos,\
    cacheIndex,\
    kvCache,\
    krCache,\
    dequantScaleX,\
    dequantScaleWDq,\
    dequantScaleWUqQr,\
    dequantScaleWDkvKr,\
    quantScaleCkv,\
    quantScaleCkr,\
    smoothScalesCq,\
    rmsnormEpsilonCq,\
    rmsnormEpsilonCkv,\
    cacheMode
// 全量化 kvcache pertensor量化
TEST_F(l2_mla_prolog_v2_test, Ascend910B2_mla_prolog_v2_0)
{
    // auto tokenX = TensorDesc({8, 1, 7168}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokenX = TensorDesc({8, 1, 7168}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1,1);            // B,S,He
    auto weightDq = TensorDesc({7168, 1536}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1,1);          // He,Hcq
    auto weightUqQr = TensorDesc({1536, 6144}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1,1);        // Hcq,N*(D+Dr)
    auto weightUk = TensorDesc({32, 128, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1,1);        // N,D,Hckv
    auto weightDkvKr = TensorDesc({7168, 576}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1,1);        // He,Hckv+Dr
    auto rmsnormGammaCq = TensorDesc({1536}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1,1);          // Hcq
    auto rmsnormGammaCkv = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1,1);          // Hckv
    auto ropeSin = TensorDesc({8, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1,1);             // B,S,Dr
    auto ropeCos = TensorDesc({8, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1,1);             // B,S,Dr
    auto kvCache = TensorDesc({16, 128, 1, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1,1);      // BolckNum,BlockSize,Nkv,Hckv
    auto krCache = TensorDesc({16, 128, 1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1,1);       // BolckNum,BlockSize,Nkv,Dr
    auto cacheIndex = TensorDesc({8, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1,1);              // B,S
    auto dequantScaleX = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);           // B*S, 1
    auto dequantScaleWDq = TensorDesc({1, 1536}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);      // 1, Hcq
    auto dequantScaleWUqQr = TensorDesc({1, 6144}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);    // 1, N*(D+Dr)
    auto dequantScaleWDkvKr = TensorDesc({1, 576}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);    // 1, Hckv+Dr
    auto quantScaleCkv = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);              // 1
    auto quantScaleCkr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);              // 1
    auto smoothScalesCq = TensorDesc({1, 1536}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);       // 1, Hcq
    double rmsnormEpsilonCq = 1e-05f;
    double rmsnormEpsilonCkv = 1e-05f;
    char *cacheMode = "PA_BSND";
    auto queryOut = TensorDesc({8, 1, 32, 512}, ACL_INT8, ACL_FORMAT_ND);          // B,S,N,Hckv
    auto queryRopeOut = TensorDesc({8, 1, 32, 64}, ACL_BF16, ACL_FORMAT_ND);       // B,S,N,Dr
    auto dequantScaleQNopeOut = TensorDesc({8, 32, 1}, ACL_FLOAT, ACL_FORMAT_ND);   // B*S, N, 1

    auto ut = OP_API_UT(
        aclnnMlaPrologV2WeightNz,
        INPUT(PARAM_LIST),
        OUTPUT(queryOut, queryRopeOut, dequantScaleQNopeOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}