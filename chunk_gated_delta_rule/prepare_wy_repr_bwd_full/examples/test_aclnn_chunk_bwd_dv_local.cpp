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
 * \file test_aclnn_prepare_wy_repr_bwd_full.cpp
 * \brief
 */
//testci
 
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_prepare_wy_repr_bwd_full.h"

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<short> resultData(size, 0);
  // std::cout << size << std::endl;
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
}

void PrintOutResultFp32(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  // std::cout << size << std::endl;
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

std::vector<int64_t> get_chunk_indices(
    const std::vector<int64_t>& cu_seqlens,
    int64_t chunk_size
) {
    std::vector<int64_t> result;
    
    for (size_t seq_idx = 1; seq_idx < cu_seqlens.size(); ++seq_idx) {
        int64_t seq_len = cu_seqlens[seq_idx] - cu_seqlens[seq_idx - 1];
        int64_t seq_id = static_cast<int64_t>(seq_idx - 1);
        
        int64_t num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            result.push_back(seq_id);
            result.push_back(chunk_id);
        }
    }
    
    return result;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 1;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t B = 1;
  int64_t T = 2048;
  int64_t H = 4;
  int64_t V = 128;
  int64_t K = 128;
  int64_t BT = 64;

  // std::vector<int64_t> cuSeqlensHostData = {0, 64, 128};
  // std::vector<int64_t> chunkIndicesHostData = get_chunk_indices(cuSeqlensHostData,chunk_size);
  // int64_t NT = static_cast<int64_t>(chunkIndicesHostData.size()) / 2;
  // float scale = 1.0;

  std::vector<int64_t> kShape = {B, H, T, K};
  std::vector<int64_t> vShape = {B, H, T, V};
  std::vector<int64_t> betaShape = {B, H, T};
  std::vector<int64_t> AShape = {B, H, T, BT};
  std::vector<int64_t> dAShape = {B, H, T, BT};
  std::vector<int64_t> dwShape = {B, H, T, K};
  std::vector<int64_t> duShape = {B, H, T, V};
  std::vector<int64_t> gShape = {B, H, T};

  std::vector<int64_t> dkShape = {B, H, T, K};
  std::vector<int64_t> dvShape = {B, H, T, V};
  std::vector<int64_t> dbetaShape = {B, H, T};
  std::vector<int64_t> dgShape = {B, H, T};

  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* ADeviceAddr = nullptr;
  void* dADeviceAddr = nullptr;
  void* dwDeviceAddr = nullptr;
  void* duDeviceAddr = nullptr;
  void* gDeviceAddr = nullptr;

  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* dbetaDeviceAddr = nullptr;
  void* dgDeviceAddr = nullptr;

  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* A = nullptr;
  aclTensor* dA = nullptr;
  aclTensor* dw = nullptr;
  aclTensor* du = nullptr;
  aclTensor* g = nullptr;

  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dbeta = nullptr;
  aclTensor* dg = nullptr;
  
  int64_t lenK = B * H * T * K;
  int64_t lenV = B * H * T * V;
  int64_t lenBeta = B * H * T;
  int64_t lenA = B * H * T * BT;
  std::vector<uint16_t> kHostData(lenK, 0x3C00); // 1.0 的 half 表示
  std::vector<uint16_t> vHostData(lenV, 0x3C00);
  std::vector<float> betaHostData(lenBeta, 1.0f);
  std::vector<uint16_t> AHostData(lenA, 0x3C00);
  std::vector<uint16_t> dAHostData(lenA, 0x3C00);
  std::vector<uint16_t> dwHostData(lenK, 0x3C00);
  std::vector<uint16_t> duHostData(lenV, 0x3C00);
  std::vector<float> gHostData(lenBeta, 1.0f);

  std::vector<uint16_t> dkHostData(lenK, 0x3C00);
  std::vector<uint16_t> dvHostData(lenV, 0x3C00);
  std::vector<float> dbetaHostData(lenBeta, 1.0f);
  std::vector<float> dgHostData(lenBeta, 1.0f);
  // for(int64_t Indices:chunkIndicesHostData){
  //   std::cout<< Indices << std::endl;
  // }
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(AHostData, AShape, &ADeviceAddr, aclDataType::ACL_FLOAT16, &A);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dAHostData, dAShape, &dADeviceAddr, aclDataType::ACL_FLOAT16, &dA);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dwHostData, dwShape, &dwDeviceAddr, aclDataType::ACL_FLOAT16, &dw);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(duHostData, duShape, &duDeviceAddr, aclDataType::ACL_FLOAT16, &du);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT, &g);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dbetaHostData, dbetaShape, &dbetaDeviceAddr, aclDataType::ACL_FLOAT, &dbeta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dgHostData, dgShape, &dgDeviceAddr, aclDataType::ACL_FLOAT, &dg);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnPrepareWyReprBwdFull第一段接口
  ret = aclnnPrepareWyReprBwdFullGetWorkspaceSize(k, v, beta, A, dA, dw, du, g, nullptr, nullptr, 64, dk, dv, dbeta, dg, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnPrepareWyReprBwdFullGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnPrepareWyReprBwdFull第二段接口
  ret = aclnnPrepareWyReprBwdFull(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnPrepareWyReprBwdFull failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // PrintOutResult(dkShape, &dkDeviceAddr);
  // PrintOutResultFp32(dbetaShape, &dbetaDeviceAddr);
  PrintOutResultFp32(dgShape, &dgDeviceAddr);
  // PrintOutResult(dvShape, &dvDeviceAddr);
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  // aclDestroyTensor(q);
  // aclDestroyTensor(k);
  // aclDestroyTensor(dO);
  // aclDestroyTensor(g);
  // aclDestroyTensor(cuSeqlens);
  // aclDestroyTensor(chunkIndices);
  // aclDestroyTensor(dV);

  // // 7. 释放device资源
  // aclrtFree(qDeviceAddr);
  // aclrtFree(kDeviceAddr);
  // aclrtFree(dODeviceAddr);
  // aclrtFree(gDeviceAddr);
  // aclrtFree(cuSeqlensDeviceAddr);
  // aclrtFree(chunkIndicesDeviceAddr);
  // aclrtFree(dVDeviceAddr);

  // if (workspaceSize > 0) {
  //   aclrtFree(workspaceAddr);
  // }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}