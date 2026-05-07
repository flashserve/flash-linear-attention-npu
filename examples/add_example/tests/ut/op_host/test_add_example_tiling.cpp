/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/add_example_tiling_data.h"

using namespace std;
using namespace ge;

class AddExampleTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddExampleTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddExampleTiling TearDown" << std::endl;
    }
};

std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};

TEST_F(AddExampleTiling, add_example_0) {
    struct AddExampleCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AddExample",
                                                {
                                                    {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor1
                                                    {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor2
                                                },
                                                {
                                                    {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output tensor
                                                },
                                                {
                                                    /* attrs */ 
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 8 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddExampleTiling, add_example_1) {
    struct AddExampleCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("AddExample",
                                                {
                                                    {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND}, // input tensor1
                                                    {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND}, // input tensor2
                                                },
                                                {
                                                    {{{1, 2, 8, 16}, {1, 2, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND}, // output tensor
                                                },
                                                {
                                                    /* attrs */ 
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "256 8 ";
    std::vector<size_t> expectWorkspaces = {1024 * 1024 * 16};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}