/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_all_gather_add.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "../op_host/op_api/aclnn_all_gather_add.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

constexpr int RANK_DIM = 2;
typedef struct {
    std::vector<op::fp16_t> rank0_a;
    std::vector<op::fp16_t> rank0_b;
    std::vector<op::fp16_t> rank1_a;
    std::vector<op::fp16_t> rank1_b;

    std::vector<op::fp16_t> gatherOut;
    std::vector<op::fp16_t> rank0_c;
    std::vector<op::fp16_t> rank1_c;
} TestData;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// 本示例固定shape
const std::vector<int64_t> aShape = {240, 256};
const std::vector<int64_t> bShape = {240 * RANK_DIM, 256};

const long long aShapeSize = GetShapeSize(aShape);
const long long bShapeSize = GetShapeSize(bShape);

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CompareVector(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2)
{
    const float tolerance = 0.001f; // 千分之一
    for (size_t i = 0; i < vec1.size(); ++i) {
        float a = static_cast<float>(vec1[i]);
        float b = static_cast<float>(vec2[i]);

        float diff = std::fabs(a - b);
        float maxAbs = std::max(std::fabs(a), std::fabs(b));
        if (maxAbs > 1e-6f) {
            float relativeError = diff / maxAbs;
            if (relativeError > tolerance) {
                return 1;
            }
        } else {
            if (diff > tolerance) {
                return 1;
            }
        }
    }
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
  };

int LaunchOneThreadAllGatherAdd(Args &args, TestData &testData)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    void *aDeviceAddr = nullptr;
    void *bDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *gatherOutDeviceAddr = nullptr;
    aclTensor *a = nullptr;
    aclTensor *b = nullptr;
    aclTensor *out = nullptr;
    aclTensor *gatherOut = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    std::vector<op::fp16_t> aHostData(aShapeSize, 0);
    std::vector<op::fp16_t> bHostData(bShapeSize, 0);

    // 根据随机生成的测试数据填充host侧输入
    if (args.rankId == 0) {
        std::copy(testData.rank0_a.begin(), testData.rank0_a.end(), aHostData.begin());
        std::copy(testData.rank0_b.begin(), testData.rank0_b.end(), bHostData.begin());
    } else {
        std::copy(testData.rank1_a.begin(), testData.rank1_a.end(), aHostData.begin());
        std::copy(testData.rank1_b.begin(), testData.rank1_b.end(), bHostData.begin());
    }

    std::vector<op::fp16_t> outHostData(bShapeSize, 0);
    std::vector<op::fp16_t> gatherOutHostData(bShapeSize, 0);

    ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_FLOAT16, &b);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, bShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gatherOutHostData, bShape, &gatherOutDeviceAddr,
        aclDataType::ACL_FLOAT16, &gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnAllGatherAddGetWorkspaceSize(
        a, b, hcomName, RANK_DIM, out, gatherOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAllGatherAddGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);  return ret);
    }
    // 调用第二阶段接口
    ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherAdd failed. ret = %d \n", ret); return ret);
    // 同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnAllGatherAdd execute successfully.\n", args.rankId);

    // 算子计算结果与golden数据进行对比
    std::vector<op::fp16_t> gatherOutData(bShapeSize, 0);
    // 将计算结果从device侧拷贝到host侧
    ret = aclrtMemcpy(gatherOutData.data(), bShapeSize * sizeof(gatherOutData[0]), gatherOutDeviceAddr,
                      bShapeSize * sizeof(gatherOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // 比较AllGather结果
    ret = CompareVector(gatherOutData, testData.gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] gatherOut compare failed. ret = %d \n", ret); return ret);

    std::vector<op::fp16_t> outputData(bShapeSize, 0);
    ret = aclrtMemcpy(outputData.data(), bShapeSize * sizeof(outputData[0]), outDeviceAddr,
                      bShapeSize * sizeof(outputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // 比较AllGatherAdd结果
    if (args.rankId == 0) {
        ret = CompareVector(outputData, testData.rank0_c);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] output compare failed. ret = %d \n", ret); return ret);
    } else {
        ret = CompareVector(outputData, testData.rank1_c);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] output compare failed. ret = %d \n", ret); return ret);
    }                  
    LOG_PRINT("[INFO] device_%d aclnnAllGatherAdd golden compare successfully.\n", args.rankId);

    // 释放device资源，需要根据具体API的接口定义修改
    if (a != nullptr) {
        aclDestroyTensor(a);
    }
    if (b != nullptr) {
        aclDestroyTensor(b);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (gatherOut != nullptr) {
        aclDestroyTensor(gatherOut);
    }
    if (aDeviceAddr != nullptr) {
        aclrtFree(aDeviceAddr);
    }
    if (bDeviceAddr != nullptr) {
        aclrtFree(bDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (gatherOutDeviceAddr != nullptr) {
        aclrtFree(gatherOutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

// 随机生成[-5, 5]之间的数据填充vector
int RandomVectorGenerator(std::vector<op::fp16_t> &vec, long long size)
{
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
    for (auto& elem : vec) {
        elem = static_cast<op::fp16_t>(distribution(generator));
    }
    return 0;
}

// 拼接两个大小相同的vector
int GatherVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2, std::vector<op::fp16_t> &vec3)
{
    vec3.clear();
    vec3.reserve(vec1.size() + vec2.size());
    vec3.insert(vec3.end(), vec1.begin(), vec1.end());
    vec3.insert(vec3.end(), vec2.begin(), vec2.end());
    return 0;
}

// 对两个大小相同的vector执行加法
int AddVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2, std::vector<op::fp16_t> &vec3)
{
    vec3.clear();
    vec3.resize(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        vec3[i] = vec1[i] + vec2[i];
    }
    return 0;
}

int GenerateTestData(TestData &testData)
{
    // 随机生成输入
    int ret = RandomVectorGenerator(testData.rank0_a, testData.rank0_a.size());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_a failed. ret = %d \n", ret);  return ret);
    ret = RandomVectorGenerator(testData.rank0_b, testData.rank0_b.size());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0_b failed. ret = %d \n", ret);  return ret);
    ret = RandomVectorGenerator(testData.rank1_a, testData.rank1_a.size());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_a failed. ret = %d \n", ret);  return ret);
    ret = RandomVectorGenerator(testData.rank1_b, testData.rank1_b.size());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank1_b failed. ret = %d \n", ret);  return ret);
    
    // 计算golden数据
    ret = GatherVectors(testData.rank0_a, testData.rank1_a, testData.gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate gatherOut failed. ret = %d \n", ret);  return ret);
    ret = AddVectors(testData.gatherOut, testData.rank0_b, testData.rank0_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank0 output failed. ret = %d \n", ret);  return ret);
    ret = AddVectors(testData.gatherOut, testData.rank1_b, testData.rank1_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate rank1 output failed. ret = %d \n", ret);  return ret);

    return 0;
}

int main(int argc, char *argv[])
{
    // 生成测试数据
    TestData testData = {
        std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank0_a
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank1_a
        std::vector<op::fp16_t>(aShapeSize, 0.0f), // rank0_b
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank1_b

        std::vector<op::fp16_t>(bShapeSize, 0.0f), // gatherOut
        std::vector<op::fp16_t>(bShapeSize, 0.0f), // rank0_c
        std::vector<op::fp16_t>(bShapeSize, 0.0f) // rank1_c
    };
    int ret = GenerateTestData(testData);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GenerateTestData failed. ret = %d \n", ret);  return ret);
    
    // AscendCL初始化
    ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[RANK_DIM];
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    int32_t devices[RANK_DIM];
    for (int i = 0; i < RANK_DIM; i++) {
        devices[i] = i;
    }
    // 初始化集合通信域
    HcclComm comms[RANK_DIM];
    ret = HcclCommInitAll(RANK_DIM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[RANK_DIM];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(RANK_DIM);
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadAllGatherAdd, std::ref(args [rankId]), std::ref(testData)));    
    }
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        threads[rankId]->join();
    }
    for (int i = 0; i < RANK_DIM; i++) {
        auto hcclRet = HcclCommDestroy(comms[i]);
        CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return -1);
    }
    aclFinalize();
    return 0;
}