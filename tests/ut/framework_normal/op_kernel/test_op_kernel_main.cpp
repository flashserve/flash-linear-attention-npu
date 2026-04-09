/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

class OpKernelUtEnvironment : public testing::Environment {
public:
    OpKernelUtEnvironment() {}
    virtual void SetUp() {
        cout << "Global Environment SetpUp." << endl;

        /* load libmath_op_kernel_ut_${socversion}_ut.so for init tiling funcs and infershape funcs */
        const char* buildPath = std::getenv("BUILD_PATH");
        if (buildPath == nullptr) {
            cout << "getenv BUILD_PATH failed." << endl;
            return;
        }

        string opKernelTilingSoPath = buildPath + string("/tests/ut/framework_normal/op_kernel/libtransformer_op_kernel_ut_tiling.so");
        gert::OppSoDesc oppSoDesc({ge::AscendString(opKernelTilingSoPath.c_str())}, "transformer_op_kernel_ut_so");
        shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
        if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
            cout << "add so to registry failed." << endl;
            return;
        }

        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);
    }

    virtual void TearDown() {
        cout << "Global Environment TearDown" << endl;
    }
};

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpKernelUtEnvironment());
    return RUN_ALL_TESTS();
}
