/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_example.cpp
 * \brief
 */

#include "../../../op_kernel/add_example.cpp"  
#include "add_example_tiling.h"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>  
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

class AddExampleTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "AddExampleTest SetUp\n" << endl;
        std::cout << "is_finite_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./add_example_data/");
    }
    static void TearDownTestCase()
    {
        cout << "AddExampleTest TearDown\n" << endl;
    }
private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string AddExampleTest::rootPath = "../../../../../";
const std::string AddExampleTest::dataPath = rootPath + "examples/add_example/tests/ut/op_kernel/add_example_data";

TEST_F(AddExampleTest, test_case_0)
{
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t zByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(AddExampleTilingData);
    uint32_t numBlocks = 8;

    system("cd ./add_example_data/ && python3 gen_data.py '(32, 4, 4, 4)' 'float32'");
    std::string fileName = "./add_example_data/float32_input_add_example.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(fileName, xByteSize, x, xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    ReadFile(fileName, xByteSize, y, xByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddExampleTilingData* tilingDatafromBin = reinterpret_cast<AddExampleTilingData*>(tiling);
    tilingDatafromBin->totalLength = 32 * 4 * 4 * 4;
    tilingDatafromBin->tileNum = 8;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(add_example<0>,
        numBlocks,
        x,
        y,
        z,
        workspace,
        (uint8_t *)(tilingDatafromBin));
    
    fileName = "./add_example_data/float32_output_add_example.bin";
    WriteFile(fileName, z, zByteSize);

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./add_example_data/ && python3 compare_data.py 'float32'");
    free(path_);
}
