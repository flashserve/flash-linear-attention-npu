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
 * \file test_chunk_gated_delta_rule_bwd_dhu.cpp
 * \brief
 */
//testci
 
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule_bwd_dhu.h"

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
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < 10; i++) {
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

std::vector<uint8_t> createUpperTriMask(int chunk_size) {
    std::vector<uint8_t> triMask;
    triMask.reserve(chunk_size * chunk_size);
    
    // 预计算每行true的数量：第i行有(n-i)个true
    for (int i = 0; i < chunk_size; ++i) {
        // 前i个元素为false
        triMask.insert(triMask.end(), i, 0);
        // 从第i列到最后一列为true
        triMask.insert(triMask.end(), chunk_size - i, 1);
    }
    
    return triMask;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 5;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t B = 1;
  int64_t T = 1792;
  int64_t H = 2;
  int64_t V = 128;
  int64_t K = 128;
  int64_t chunk_size = 64;
  int64_t chunk_num = T/chunk_size;

  std::vector<int64_t> cuSeqlensHostData = {0, 256, 768, 1792}; // [0, 2, 4, ]
  // std::vector<int64_t> cuSeqlensHostData = {0, 128, 256};
  std::vector<int64_t> chunkIndicesHostData = get_chunk_indices(cuSeqlensHostData,chunk_size);

  int64_t NT = static_cast<int64_t>(chunkIndicesHostData.size()) / 2;
  float scale = 1.0;

  std::vector<int64_t> qShape = {B, H, T, K};
  std::vector<int64_t> kShape = {B, H, T, K};
  std::vector<int64_t> wShape = {B, H, T, K};
  std::vector<int64_t> doShape = {B, H, T, V};
  std::vector<int64_t> dvShape = {B, H, T, V};
  std::vector<int64_t> gShape = {B, H, T};
  // std::vector<int64_t> cuSeqlensShape = {static_cast<int64_t>(cuSeqlensHostData.size())};
  // std::vector<int64_t> chunkIndicesShape = {NT, 2};
  uint64_t cuSeqlensSize = cuSeqlensHostData.size();
  uint64_t chunkIndicesSize = NT*2;

  std::vector<int64_t> dv2Shape = {B, H, T, V};
  std::vector<int64_t> dhShape = {B, H, chunk_num, K, V};
  std::vector<int64_t> dh0Shape = {B, H, chunk_num, K, V};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* wDeviceAddr = nullptr;
  void* doDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* gDeviceAddr = nullptr;
  void* cuSeqlensDeviceAddr = nullptr;
  void* chunkIndicesDeviceAddr = nullptr;

  void* dv2DeviceAddr = nullptr;
  void* dhDeviceAddr = nullptr;
  void* dh0DeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* w = nullptr;
  aclTensor* d_o = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* g = nullptr;
  aclIntArray* cuSeqlens = nullptr;
  aclIntArray* chunkIndices = nullptr;

  aclTensor* dh = nullptr;
  aclTensor* dh0 = nullptr;
  aclTensor* dv2 = nullptr;
  
  int64_t lenQK = B * H * T * K;
  int64_t lenO = B * H * T * V;
  int64_t lenG = B * H * T;
  int64_t lenH = B * H * chunk_num * K * V;
  std::cout<<"lenH is " <<lenH<<std::endl;
  std::vector<uint16_t> qHostData(lenQK, 0x3C00); // 1.0 的 half 表示
  std::vector<uint16_t> kHostData(lenQK, 0x3C00);
  std::vector<uint16_t> wHostData(lenQK, 0x3C00);
  std::vector<uint16_t> doHostData(lenO, 0x3C00);
  std::vector<uint16_t> dvHostData(lenO, 0x3C00);
  std::vector<uint16_t> gHostData(lenG, 0x3C00);
  std::vector<uint16_t> dv2HostData(lenO, 0);
  std::vector<uint16_t> dhHostData(lenH, 0x3C00);
  std::vector<uint16_t> dh0HostData(lenH, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT16, &w);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(doHostData, doShape, &doDeviceAddr, aclDataType::ACL_FLOAT16, &d_o);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT16, &g);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  cuSeqlens = aclCreateIntArray(cuSeqlensHostData.data(), cuSeqlensSize);
  CHECK_RET(cuSeqlens != nullptr, return ACL_ERROR_BAD_ALLOC);
  chunkIndices = aclCreateIntArray(chunkIndicesHostData.data(), chunkIndicesSize);
  CHECK_RET(chunkIndices != nullptr, return ACL_ERROR_BAD_ALLOC);
  ret = CreateAclTensor(dv2HostData, dv2Shape, &dv2DeviceAddr, aclDataType::ACL_FLOAT16, &dv2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT16, &dh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dh0HostData, dh0Shape, &dh0DeviceAddr, aclDataType::ACL_FLOAT16, &dh0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 1024*1024*1024;
  aclOpExecutor *executor;

  // 调用aclnnChunkGatedDeltaRuleBwdDhu第一段接口
  std::cout << "aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize >>>>>" << std::endl;
  
  ret = aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(q, k, w, d_o, dv, g, nullptr, nullptr, nullptr, cuSeqlens, chunkIndices, scale, chunk_size, dh, dh0, dv2, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);
  std::cout << "aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize >>>>> end " << std::endl;

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }
  // std::cout << "allocate workspace >>>>> end " << std::endl;

  // 调用aclnnChunkGatedDeltaRuleBwdDhu第二段接口
  ret = aclnnChunkGatedDeltaRuleBwdDhu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnChunkGatedDeltaRuleBwdDhu failed. ERROR: %d\n", ret);
            return ret);
  // std::cout << "调用aclnnChunkGatedDeltaRuleBwdDhu第二段接口 end " << std::endl;

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);
  // std::cout << "aclrtSynchronizeStream end " << std::endl;

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // LOG_PRINT("d_v2 \n");
  // PrintOutResult(dv2Shape, &dv2DeviceAddr);
  LOG_PRINT("d_h \n");
  PrintOutResult(dhShape, &dhDeviceAddr);
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(w);
  aclDestroyTensor(d_o);
  aclDestroyTensor(dv);
  aclDestroyTensor(g);
  aclDestroyIntArray(cuSeqlens);
  aclDestroyIntArray(chunkIndices);
  aclDestroyTensor(dv2);
  aclDestroyTensor(dh);
  aclDestroyTensor(dh0);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(wDeviceAddr);
  aclrtFree(doDeviceAddr);
  aclrtFree(dvDeviceAddr);
  aclrtFree(gDeviceAddr);
  // aclrtFree(cuSeqlensDeviceAddr);
  // aclrtFree(chunkIndicesDeviceAddr);
  aclrtFree(dv2DeviceAddr);
  aclrtFree(dhDeviceAddr);
  aclrtFree(dh0DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}