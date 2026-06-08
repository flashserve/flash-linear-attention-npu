/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_bwd.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(float));
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7fff + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

std::vector<uint16_t> FloatVectorToBf16(const std::vector<float>& input) {
  std::vector<uint16_t> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = FloatToBf16(input[i]);
  }
  return output;
}

std::vector<float> Bf16VectorToFloat(const std::vector<uint16_t>& input) {
  std::vector<float> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = Bf16ToFloat(input[i]);
  }
  return output;
}

bool AllClose(const std::vector<float>& out, const std::vector<float>& ref, float rtol, float atol,
              const char* name) {
  if (out.size() != ref.size()) {
    LOG_PRINT("%s size mismatch: out=%zu ref=%zu\n", name, out.size(), ref.size());
    return false;
  }
  float maxAbs = 0.0f;
  float meanAbs = 0.0f;
  size_t maxIdx = 0;
  bool ok = true;
  for (size_t i = 0; i < out.size(); ++i) {
    float diff = std::fabs(out[i] - ref[i]);
    float tol = atol + rtol * std::fabs(ref[i]);
    if (diff > tol) {
      ok = false;
    }
    meanAbs += diff;
    if (diff > maxAbs) {
      maxAbs = diff;
      maxIdx = i;
    }
  }
  if (!out.empty()) {
    meanAbs /= static_cast<float>(out.size());
  }
  LOG_PRINT("%s: allclose=%s max_abs=%g mean_abs=%g idx=%zu out=%g ref=%g\n",
            name, ok ? "true" : "false", maxAbs, meanAbs, maxIdx,
            out.empty() ? 0.0f : out[maxIdx], ref.empty() ? 0.0f : ref[maxIdx]);
  return ok;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnCausalConv1dBwdTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  int64_t B = 1;
  int64_t T = 32 * 1024;
  int64_t D = 1536;
  int64_t W = 4;
  int64_t activation = 0; // 0=none, 1=silu, 2=swish
  char inputLayout[] = "BSND";

  std::vector<int64_t> xShape = {B, T, D};
  std::vector<int64_t> yShape = {B, T, D};
  std::vector<int64_t> weightShape = {W, D};
  std::vector<int64_t> dyShape = {B, T, D};
  std::vector<int64_t> initialStateShape = {B, W, D};  // [B, W, D] for conv state
  std::vector<int64_t> dhtShape = {B, W, D};  // [B, W, D] for final state gradient
  
  std::vector<int64_t> dxShape = {B, T, D};
  std::vector<int64_t> dwShape = {W, D};
  std::vector<int64_t> dbShape = {D};
  std::vector<int64_t> dh0Shape = {B, W, D};  // [B, W, D] for initial state gradient

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* dyDeviceAddr = nullptr;
  void* initialStateDeviceAddr = nullptr;
  void* dhtDeviceAddr = nullptr;

  void* dxDeviceAddr = nullptr;
  void* dwDeviceAddr = nullptr;
  void* dbDeviceAddr = nullptr;
  void* dh0DeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* y = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* initialState = nullptr;
  aclTensor* dht = nullptr;

  aclTensor* dx = nullptr;
  aclTensor* dw = nullptr;
  aclTensor* db = nullptr;
  aclTensor* dh0 = nullptr;

  // Host buffers. Try to load from .bin files under cpu_bwd_saved, fallback to sensible defaults if missing.
  std::vector<float> xHostData(GetShapeSize(xShape));
  std::vector<float> yHostData(GetShapeSize(yShape));
  std::vector<float> weightHostData(GetShapeSize(weightShape));
  std::vector<float> dyHostData(GetShapeSize(dyShape));
  std::vector<float> initialStateHostData(GetShapeSize(initialStateShape), 0.0f);
  std::vector<float> dhtHostData(GetShapeSize(dhtShape), 0.0f);

  auto ReadBinFloatFile = [&](const std::string &path, std::vector<float> &out) -> bool {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      LOG_PRINT("file not found: %s\n", path.c_str());
      return false;
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size % sizeof(float) != 0) {
      LOG_PRINT("file size is not a multiple of float: %s\n", path.c_str());
      return false;
    }
    size_t count = static_cast<size_t>(size / sizeof(float));
    if (count != out.size()) {
      LOG_PRINT("warning: %s contains %zu floats but expected %zu.\n", path.c_str(), count, out.size());
      // still attempt to read as much as fits
    }
    size_t toRead = std::min(count, out.size());
    in.read(reinterpret_cast<char *>(out.data()), toRead * sizeof(float));
    if (!in) {
      LOG_PRINT("failed to read %s\n", path.c_str());
      return false;
    }
    // If file had fewer elements than expected, pad remaining with zeros
    if (toRead < out.size()) {
      std::fill(out.begin() + toRead, out.end(), 0.0f);
    }
    LOG_PRINT("loaded %zu floats from %s\n", toRead, path.c_str());
    return true;
  };

  auto WriteBinFloatFile = [&](const std::string &path, const std::vector<float> &data) -> bool {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
      LOG_PRINT("failed to open for write: %s\n", path.c_str());
      return false;
    }
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    if (!out) {
      LOG_PRINT("failed to write %s\n", path.c_str());
      return false;
    }
    LOG_PRINT("wrote %zu floats to %s\n", data.size(), path.c_str());
    return true;
  };

  const std::string basePath = "/data/jq/github/mojo_opset/cpu_bwd_saved";
  // Try to load x, dy, weight from saved .bin files. If missing, fill with reasonable defaults.
  if (!ReadBinFloatFile(basePath + "/x.bin", xHostData)) {
    LOG_PRINT("filling x with 1.0f fallback\n");
    std::fill(xHostData.begin(), xHostData.end(), 1.0f);
  }
  if (!ReadBinFloatFile(basePath + "/dy.bin", dyHostData)) {
    LOG_PRINT("filling dy with 2.0f fallback\n");
    std::fill(dyHostData.begin(), dyHostData.end(), 2.0f);
  }
  if (!ReadBinFloatFile(basePath + "/weight.bin", weightHostData)) {
    LOG_PRINT("filling weight with 0.5f fallback\n");
    std::fill(weightHostData.begin(), weightHostData.end(), 0.5f);
  }
  // y.bin is optional; if present use it, otherwise fill with zeros to avoid all-ones default.
  if (!ReadBinFloatFile(basePath + "/y.bin", yHostData)) {
    std::fill(yHostData.begin(), yHostData.end(), 0.0f);
  }

  LOG_PRINT("running causal_conv1d_bwd BF16 precision example\n");
  std::vector<uint16_t> xHostDataBf16 = FloatVectorToBf16(xHostData);
  std::vector<uint16_t> yHostDataBf16 = FloatVectorToBf16(yHostData);
  std::vector<uint16_t> weightHostDataBf16 = FloatVectorToBf16(weightHostData);
  std::vector<uint16_t> dyHostDataBf16 = FloatVectorToBf16(dyHostData);
  std::vector<uint16_t> initialStateHostDataBf16 = FloatVectorToBf16(initialStateHostData);
  std::vector<uint16_t> dhtHostDataBf16 = FloatVectorToBf16(dhtHostData);

  std::vector<uint16_t> dxHostData(GetShapeSize(dxShape), 0);
  std::vector<uint16_t> dwHostData(GetShapeSize(dwShape), 0);
  std::vector<uint16_t> dbHostData(GetShapeSize(dbShape), 0);
  std::vector<uint16_t> dh0HostData(GetShapeSize(dh0Shape), 0);

  // 创建x aclTensor
  ret = CreateAclTensor(xHostDataBf16, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

  // 创建y aclTensor
  ret = CreateAclTensor(yHostDataBf16, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);

  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostDataBf16, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);

  // 创建dy aclTensor
  ret = CreateAclTensor(dyHostDataBf16, dyShape, &dyDeviceAddr, aclDataType::ACL_BF16, &dy);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dyTensorPtr(dy, aclDestroyTensor);

  // 创建initialState aclTensor
  ret = CreateAclTensor(initialStateHostDataBf16, initialStateShape, &initialStateDeviceAddr, aclDataType::ACL_BF16, &initialState);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> initialStateTensorPtr(initialState, aclDestroyTensor);

  // 创建dht aclTensor
  ret = CreateAclTensor(dhtHostDataBf16, dhtShape, &dhtDeviceAddr, aclDataType::ACL_BF16, &dht);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dhtTensorPtr(dht, aclDestroyTensor);

  // 创建dx aclTensor
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_BF16, &dx);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dxTensorPtr(dx, aclDestroyTensor);

  // 创建dw aclTensor
  ret = CreateAclTensor(dwHostData, dwShape, &dwDeviceAddr, aclDataType::ACL_BF16, &dw);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dwTensorPtr(dw, aclDestroyTensor);

  // 创建db aclTensor
  ret = CreateAclTensor(dbHostData, dbShape, &dbDeviceAddr, aclDataType::ACL_BF16, &db);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dbTensorPtr(db, aclDestroyTensor);

  // 创建dh0 aclTensor
  ret = CreateAclTensor(dh0HostData, dh0Shape, &dh0DeviceAddr, aclDataType::ACL_BF16, &dh0);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dh0TensorPtr(dh0, aclDestroyTensor);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  ret = aclnnCausalConv1dBwdGetWorkspaceSize(x,
                                             y,
                                             weight,
                                             dy,
                                             initialState,
                                             dht,
                                             nullptr,
                                             activation,
                                             inputLayout,
                                             dx,
                                             dw,
                                             db,
                                             dh0,
                                             &workspaceSize,
                                             &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dBwdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }

  ret = aclnnCausalConv1dBwd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dBwd failed. ERROR: %d\n", ret); return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值
  auto dxSize = GetShapeSize(dxShape);
  std::vector<uint16_t> dxDataBf16(dxSize, 0);
  ret = aclrtMemcpy(dxDataBf16.data(), dxDataBf16.size() * sizeof(uint16_t), dxDeviceAddr,
                    dxSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dx from device to host failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dxData = Bf16VectorToFloat(dxDataBf16);

  LOG_PRINT("dx output sample (first 10 elements):\n");
  for (int64_t i = 0; i < 10; i++) {
    LOG_PRINT("dx[%ld] = %f\n", i, dxData[i]);
  }

  auto dwSize = GetShapeSize(dwShape);
  std::vector<uint16_t> dwDataBf16(dwSize, 0);
  ret = aclrtMemcpy(dwDataBf16.data(), dwDataBf16.size() * sizeof(uint16_t), dwDeviceAddr,
                    dwSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dw from device to host failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dwData = Bf16VectorToFloat(dwDataBf16);

  LOG_PRINT("\ndw output sample (first 10 elements):\n");
  for (int64_t i = 0; i < 10; i++) {
    LOG_PRINT("dw[%ld] = %f\n", i, dwData[i]);
  }

  auto dbSize = GetShapeSize(dbShape);
  std::vector<uint16_t> dbDataBf16(dbSize, 0);
  ret = aclrtMemcpy(dbDataBf16.data(), dbDataBf16.size() * sizeof(uint16_t), dbDeviceAddr,
                    dbSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy db from device to host failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dbData = Bf16VectorToFloat(dbDataBf16);

  LOG_PRINT("\ndb output sample (first 10 elements):\n");
  for (int64_t i = 0; i < 10; i++) {
    LOG_PRINT("db[%ld] = %f\n", i, dbData[i]);
  }

  auto dh0Size = GetShapeSize(dh0Shape);
  std::vector<uint16_t> dh0DataBf16(dh0Size, 0);
  ret = aclrtMemcpy(dh0DataBf16.data(), dh0DataBf16.size() * sizeof(uint16_t), dh0DeviceAddr,
                    dh0Size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dh0 from device to host failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dh0Data = Bf16VectorToFloat(dh0DataBf16);

  LOG_PRINT("\ndh0 output sample (first 10 elements):\n");
  for (int64_t i = 0; i < 10 && i < dh0Size; i++) {
    LOG_PRINT("dh0[%ld] = %f\n", i, dh0Data[i]);
  }

  // Save outputs to .bin files alongside the inputs for later inspection.
  WriteBinFloatFile(basePath + "/dx_npu.bin", dxData);
  WriteBinFloatFile(basePath + "/dw_npu.bin", dwData);
  WriteBinFloatFile(basePath + "/db_npu.bin", dbData);
  WriteBinFloatFile(basePath + "/dh0_npu.bin", dh0Data);

  return ACL_SUCCESS;
}

std::vector<float> MakeQuantizedBf16Float(const std::vector<float>& input) {
  return Bf16VectorToFloat(FloatVectorToBf16(input));
}

void BuildVarlenReference(const std::vector<float>& x,
                          const std::vector<float>& weight,
                          const std::vector<float>& dy,
                          const std::vector<float>& initialState,
                          const std::vector<float>& dht,
                          const std::vector<int64_t>& queryStartLoc,
                          int64_t totalTokens,
                          int64_t D,
                          int64_t W,
                          std::vector<float>& dx,
                          std::vector<float>& dw,
                          std::vector<float>& db,
                          std::vector<float>& dh0,
                          bool quantizeStateOutputsToBf16 = true) {
  int64_t B = static_cast<int64_t>(queryStartLoc.size()) - 1;
  std::fill(dx.begin(), dx.end(), 0.0f);
  std::fill(dw.begin(), dw.end(), 0.0f);
  std::fill(db.begin(), db.end(), 0.0f);
  std::fill(dh0.begin(), dh0.end(), 0.0f);

  for (int64_t b = 0; b < B; ++b) {
    int64_t seqStart = queryStartLoc[b];
    int64_t seqEnd = queryStartLoc[b + 1];
    int64_t seqLen = seqEnd - seqStart;

    for (int64_t t = 0; t < seqLen; ++t) {
      int64_t g = seqStart + t;
      for (int64_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int64_t iw = 0; iw < W; ++iw) {
          int64_t srcT = t + iw;
          if (srcT < seqLen) {
            acc += dy[(seqStart + srcT) * D + d] * weight[(W - 1 - iw) * D + d];
          }
        }
        dx[g * D + d] = acc;
        db[d] += dy[g * D + d];
      }
    }

    int64_t tailStart = std::max<int64_t>(0, seqLen - (W - 1));
    for (int64_t t = tailStart; t < seqLen; ++t) {
      int64_t slot = 1 + t - tailStart;
      int64_t g = seqStart + t;
      if (slot >= W) {
        continue;
      }
      for (int64_t d = 0; d < D; ++d) {
        dx[g * D + d] += dht[(b * W + slot) * D + d];
      }
    }

    for (int64_t iw = 0; iw < W; ++iw) {
      int64_t wIdx = W - 1 - iw;
      for (int64_t t = 0; t + iw < seqLen; ++t) {
        int64_t xG = seqStart + t;
        int64_t dyG = seqStart + t + iw;
        for (int64_t d = 0; d < D; ++d) {
          dw[wIdx * D + d] += dy[dyG * D + d] * x[xG * D + d];
        }
      }
    }

    int64_t head = std::min(seqLen, W - 1);
    for (int64_t iw = 1; iw < W; ++iw) {
      int64_t wIdx = W - 1 - iw;
      for (int64_t row = 0; row < std::min(head, iw); ++row) {
        int64_t slot = W - iw + row;
        for (int64_t d = 0; d < D; ++d) {
          dw[wIdx * D + d] +=
              dy[(seqStart + row) * D + d] * initialState[(b * W + slot) * D + d];
        }
      }
    }

    for (int64_t slot = 1; slot < W; ++slot) {
      for (int64_t row = 0; row < std::min(seqLen, slot); ++row) {
        int64_t k = slot - 1 - row;
        for (int64_t d = 0; d < D; ++d) {
          int64_t dh0Off = (b * W + slot) * D + d;
          dh0[dh0Off] += dy[(seqStart + row) * D + d] * weight[k * D + d];
        }
      }
    }
  }

  if (quantizeStateOutputsToBf16) {
    for (float& v : dx) {
      v = Bf16ToFloat(FloatToBf16(v));
    }
    for (float& v : dh0) {
      v = Bf16ToFloat(FloatToBf16(v));
    }
  }
}

int RunVarlenFp32SingleSeqTest(aclrtStream& stream) {
  constexpr int64_t B = 1;
  constexpr int64_t T = 64;
  constexpr int64_t D = 80;
  constexpr int64_t W = 4;
  constexpr int64_t activation = 0;
  char inputLayout[] = "TND";
  std::vector<int64_t> queryStartLocHost = {0, T};

  std::vector<int64_t> xShape = {T, D};
  std::vector<int64_t> yShape = {T, D};
  std::vector<int64_t> weightShape = {W, D};
  std::vector<int64_t> dyShape = {T, D};
  std::vector<int64_t> initialStateShape = {B, W, D};
  std::vector<int64_t> dhtShape = {B, W, D};
  std::vector<int64_t> dxShape = {T, D};
  std::vector<int64_t> dwShape = {W, D};
  std::vector<int64_t> dbShape = {D};
  std::vector<int64_t> dh0Shape = {B, W, D};

  std::vector<float> xHost(GetShapeSize(xShape));
  std::vector<float> yHost(GetShapeSize(yShape), 0.0f);
  std::vector<float> weightHost(GetShapeSize(weightShape));
  std::vector<float> dyHost(GetShapeSize(dyShape));
  std::vector<float> initialStateHost(GetShapeSize(initialStateShape));
  std::vector<float> dhtHost(GetShapeSize(dhtShape));
  for (size_t i = 0; i < xHost.size(); ++i) {
    xHost[i] = (static_cast<int>((i * 3) % 31) - 15) * 0.01f;
  }
  for (size_t i = 0; i < dyHost.size(); ++i) {
    dyHost[i] = (static_cast<int>((i * 5) % 37) - 18) * 0.02f;
  }
  for (size_t i = 0; i < weightHost.size(); ++i) {
    weightHost[i] = (static_cast<int>((i * 7) % 19) - 9) * 0.03f;
  }
  for (size_t i = 0; i < initialStateHost.size(); ++i) {
    initialStateHost[i] = (static_cast<int>((i * 11) % 23) - 11) * 0.015f;
  }
  for (size_t i = 0; i < dhtHost.size(); ++i) {
    dhtHost[i] = (static_cast<int>((i * 13) % 29) - 14) * 0.0125f;
  }

  std::vector<float> dxHost(GetShapeSize(dxShape), 0.0f);
  std::vector<float> dwHost(GetShapeSize(dwShape), 0.0f);
  std::vector<float> dbHost(GetShapeSize(dbShape), 0.0f);
  std::vector<float> dh0Host(GetShapeSize(dh0Shape), 0.0f);

  void *xDeviceAddr = nullptr, *yDeviceAddr = nullptr, *weightDeviceAddr = nullptr, *dyDeviceAddr = nullptr;
  void *initialStateDeviceAddr = nullptr, *dhtDeviceAddr = nullptr;
  void *dxDeviceAddr = nullptr, *dwDeviceAddr = nullptr, *dbDeviceAddr = nullptr, *dh0DeviceAddr = nullptr;

  aclTensor *x = nullptr, *y = nullptr, *weight = nullptr, *dy = nullptr;
  aclTensor *initialState = nullptr, *dht = nullptr;
  aclIntArray *queryStartLoc = nullptr;
  aclTensor *dx = nullptr, *dw = nullptr, *db = nullptr, *dh0 = nullptr;

  auto ret = CreateAclTensor(xHost, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
  ret = CreateAclTensor(yHost, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);
  ret = CreateAclTensor(weightHost, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  ret = CreateAclTensor(dyHost, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dyTensorPtr(dy, aclDestroyTensor);
  ret = CreateAclTensor(initialStateHost, initialStateShape, &initialStateDeviceAddr, aclDataType::ACL_FLOAT, &initialState);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> initialStateTensorPtr(initialState, aclDestroyTensor);
  ret = CreateAclTensor(dhtHost, dhtShape, &dhtDeviceAddr, aclDataType::ACL_FLOAT, &dht);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dhtTensorPtr(dht, aclDestroyTensor);
  queryStartLoc = aclCreateIntArray(queryStartLocHost.data(), queryStartLocHost.size());
  CHECK_RET(queryStartLoc != nullptr, return ACL_ERROR_BAD_ALLOC);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> queryStartLocPtr(queryStartLoc, aclDestroyIntArray);

  ret = CreateAclTensor(dxHost, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dxTensorPtr(dx, aclDestroyTensor);
  ret = CreateAclTensor(dwHost, dwShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dwTensorPtr(dw, aclDestroyTensor);
  ret = CreateAclTensor(dbHost, dbShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dbTensorPtr(db, aclDestroyTensor);
  ret = CreateAclTensor(dh0Host, dh0Shape, &dh0DeviceAddr, aclDataType::ACL_FLOAT, &dh0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dh0TensorPtr(dh0, aclDestroyTensor);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnCausalConv1dBwdGetWorkspaceSize(x, y, weight, dy, initialState, dht, queryStartLoc,
                                             activation, inputLayout, dx, dw, db, dh0, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("fp32 varlen aclnnCausalConv1dBwdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("fp32 varlen allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }

  ret = aclnnCausalConv1dBwd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("fp32 varlen aclnnCausalConv1dBwd failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("fp32 varlen aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  std::vector<float> dxOut(dxHost.size(), 0.0f);
  ret = aclrtMemcpy(dxOut.data(), dxOut.size() * sizeof(float), dxDeviceAddr,
                    dxOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy fp32 varlen dx failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dwOut(dwHost.size(), 0.0f);
  ret = aclrtMemcpy(dwOut.data(), dwOut.size() * sizeof(float), dwDeviceAddr,
                    dwOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy fp32 varlen dw failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dbOut(dbHost.size(), 0.0f);
  ret = aclrtMemcpy(dbOut.data(), dbOut.size() * sizeof(float), dbDeviceAddr,
                    dbOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy fp32 varlen db failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dh0Out(dh0Host.size(), 0.0f);
  ret = aclrtMemcpy(dh0Out.data(), dh0Out.size() * sizeof(float), dh0DeviceAddr,
                    dh0Out.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy fp32 varlen dh0 failed. ERROR: %d\n", ret); return ret);

  std::vector<float> dxRef(dxOut.size(), 0.0f);
  std::vector<float> dwRef(dwOut.size(), 0.0f);
  std::vector<float> dbRef(dbOut.size(), 0.0f);
  std::vector<float> dh0Ref(dh0Out.size(), 0.0f);
  BuildVarlenReference(xHost, weightHost, dyHost, initialStateHost, dhtHost, queryStartLocHost,
                       T, D, W, dxRef, dwRef, dbRef, dh0Ref, false);

  LOG_PRINT("\nrunning causal_conv1d_bwd FP32 TND single-seq precision example\n");
  bool ok = true;
  ok = AllClose(dxOut, dxRef, 1e-4f, 1e-4f, "fp32 varlen dx") && ok;
  ok = AllClose(dwOut, dwRef, 1e-4f, 1e-4f, "fp32 varlen dw") && ok;
  ok = AllClose(dbOut, dbRef, 1e-4f, 1e-4f, "fp32 varlen db") && ok;
  ok = AllClose(dh0Out, dh0Ref, 1e-4f, 1e-4f, "fp32 varlen dh0") && ok;
  return ok ? ACL_SUCCESS : 1;
}

void LogicalToPhysicalInput(const std::vector<float>& logical,
                            const std::string& layout,
                            int64_t B,
                            int64_t T,
                            int64_t N,
                            int64_t H,
                            std::vector<float>& physical) {
  int64_t D = N * H;
  if (layout == "BNSD") {
    std::fill(physical.begin(), physical.end(), 0.0f);
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
          for (int64_t h = 0; h < H; ++h) {
            int64_t d = n * H + h;
            physical[((b * N + n) * T + t) * H + h] = logical[(b * T + t) * D + d];
          }
        }
      }
    }
    return;
  }
  if (layout == "NTD") {
    std::fill(physical.begin(), physical.end(), 0.0f);
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
          int64_t d = n * H + h;
          physical[(n * T + t) * H + h] = logical[t * D + d];
        }
      }
    }
    return;
  }
  physical = logical;
}

int RunInputLayoutFp32Test(aclrtStream& stream, const std::string& layout) {
  constexpr int64_t B = 2;
  constexpr int64_t T = 64;
  constexpr int64_t N = 2;
  constexpr int64_t H = 48;
  constexpr int64_t D = N * H;
  constexpr int64_t W = 4;
  constexpr int64_t activation = 0;
  char inputLayout[8] = {0};
  std::strncpy(inputLayout, layout.c_str(), sizeof(inputLayout) - 1);

  bool isNtD = (layout == "NTD");
  int64_t logicalB = isNtD ? 1 : B;
  int64_t logicalT = T;
  std::vector<int64_t> queryStartLocHost = {0, logicalT};
  std::vector<int64_t> xShape = isNtD ? std::vector<int64_t>{logicalT, D}
                                      : std::vector<int64_t>{logicalB, logicalT, D};
  std::vector<int64_t> yShape = isNtD ? std::vector<int64_t>{N, logicalT, H}
                                      : std::vector<int64_t>{logicalB, N, logicalT, H};
  std::vector<int64_t> dyShape = yShape;
  std::vector<int64_t> weightShape = {W, D};
  std::vector<int64_t> initialStateShape = {logicalB, W, D};
  std::vector<int64_t> dhtShape = {logicalB, W, D};
  std::vector<int64_t> dxShape = isNtD ? std::vector<int64_t>{logicalT, D}
                                       : std::vector<int64_t>{logicalB, logicalT, D};
  std::vector<int64_t> dwShape = {W, D};
  std::vector<int64_t> dbShape = {D};
  std::vector<int64_t> dh0Shape = {logicalB, W, D};

  std::vector<float> xLogical(GetShapeSize(dxShape));
  std::vector<float> yLogical(GetShapeSize(dxShape), 0.0f);
  std::vector<float> dyLogical(GetShapeSize(dxShape));
  std::vector<float> weightHost(GetShapeSize(weightShape));
  std::vector<float> initialStateHost(GetShapeSize(initialStateShape));
  std::vector<float> dhtHost(GetShapeSize(dhtShape));
  for (size_t i = 0; i < xLogical.size(); ++i) {
    xLogical[i] = (static_cast<int>((i * 3) % 31) - 15) * 0.01f;
  }
  for (size_t i = 0; i < dyLogical.size(); ++i) {
    dyLogical[i] = (static_cast<int>((i * 5) % 37) - 18) * 0.02f;
  }
  for (size_t i = 0; i < weightHost.size(); ++i) {
    weightHost[i] = (static_cast<int>((i * 7) % 19) - 9) * 0.03f;
  }
  for (size_t i = 0; i < initialStateHost.size(); ++i) {
    initialStateHost[i] = (static_cast<int>((i * 11) % 23) - 11) * 0.015f;
  }
  for (size_t i = 0; i < dhtHost.size(); ++i) {
    dhtHost[i] = (static_cast<int>((i * 13) % 29) - 14) * 0.0125f;
  }

  std::vector<float> xHost = xLogical;
  std::vector<float> yHost(GetShapeSize(yShape));
  std::vector<float> dyHost(GetShapeSize(dyShape));
  LogicalToPhysicalInput(yLogical, layout, logicalB, logicalT, N, H, yHost);
  LogicalToPhysicalInput(dyLogical, layout, logicalB, logicalT, N, H, dyHost);

  std::vector<float> dxHost(GetShapeSize(dxShape), 0.0f);
  std::vector<float> dwHost(GetShapeSize(dwShape), 0.0f);
  std::vector<float> dbHost(GetShapeSize(dbShape), 0.0f);
  std::vector<float> dh0Host(GetShapeSize(dh0Shape), 0.0f);

  void *xDeviceAddr = nullptr, *yDeviceAddr = nullptr, *weightDeviceAddr = nullptr, *dyDeviceAddr = nullptr;
  void *initialStateDeviceAddr = nullptr, *dhtDeviceAddr = nullptr;
  void *dxDeviceAddr = nullptr, *dwDeviceAddr = nullptr, *dbDeviceAddr = nullptr, *dh0DeviceAddr = nullptr;

  aclTensor *x = nullptr, *y = nullptr, *weight = nullptr, *dy = nullptr;
  aclTensor *initialState = nullptr, *dht = nullptr;
  aclIntArray *queryStartLoc = nullptr;
  aclTensor *dx = nullptr, *dw = nullptr, *db = nullptr, *dh0 = nullptr;

  auto ret = CreateAclTensor(xHost, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
  ret = CreateAclTensor(yHost, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);
  ret = CreateAclTensor(weightHost, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  ret = CreateAclTensor(dyHost, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dyTensorPtr(dy, aclDestroyTensor);
  ret = CreateAclTensor(initialStateHost, initialStateShape, &initialStateDeviceAddr, aclDataType::ACL_FLOAT, &initialState);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> initialStateTensorPtr(initialState, aclDestroyTensor);
  ret = CreateAclTensor(dhtHost, dhtShape, &dhtDeviceAddr, aclDataType::ACL_FLOAT, &dht);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dhtTensorPtr(dht, aclDestroyTensor);
  if (isNtD) {
    queryStartLoc = aclCreateIntArray(queryStartLocHost.data(), queryStartLocHost.size());
    CHECK_RET(queryStartLoc != nullptr, return ACL_ERROR_BAD_ALLOC);
  }
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> queryStartLocPtr(queryStartLoc, aclDestroyIntArray);

  ret = CreateAclTensor(dxHost, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dxTensorPtr(dx, aclDestroyTensor);
  ret = CreateAclTensor(dwHost, dwShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dwTensorPtr(dw, aclDestroyTensor);
  ret = CreateAclTensor(dbHost, dbShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dbTensorPtr(db, aclDestroyTensor);
  ret = CreateAclTensor(dh0Host, dh0Shape, &dh0DeviceAddr, aclDataType::ACL_FLOAT, &dh0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dh0TensorPtr(dh0, aclDestroyTensor);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnCausalConv1dBwdGetWorkspaceSize(x, y, weight, dy, initialState, dht, queryStartLoc,
                                             activation, inputLayout, dx, dw, db, dh0, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("%s aclnnCausalConv1dBwdGetWorkspaceSize failed. ERROR: %d\n", layout.c_str(), ret); return ret);

  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s allocate workspace failed. ERROR: %d\n", layout.c_str(), ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }

  ret = aclnnCausalConv1dBwd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnCausalConv1dBwd failed. ERROR: %d\n", layout.c_str(), ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtSynchronizeStream failed. ERROR: %d\n", layout.c_str(), ret); return ret);

  std::vector<float> dxOut(dxHost.size(), 0.0f);
  ret = aclrtMemcpy(dxOut.data(), dxOut.size() * sizeof(float), dxDeviceAddr,
                    dxOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s dx failed. ERROR: %d\n", layout.c_str(), ret); return ret);
  std::vector<float> dwOut(dwHost.size(), 0.0f);
  ret = aclrtMemcpy(dwOut.data(), dwOut.size() * sizeof(float), dwDeviceAddr,
                    dwOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s dw failed. ERROR: %d\n", layout.c_str(), ret); return ret);
  std::vector<float> dbOut(dbHost.size(), 0.0f);
  ret = aclrtMemcpy(dbOut.data(), dbOut.size() * sizeof(float), dbDeviceAddr,
                    dbOut.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s db failed. ERROR: %d\n", layout.c_str(), ret); return ret);
  std::vector<float> dh0Out(dh0Host.size(), 0.0f);
  ret = aclrtMemcpy(dh0Out.data(), dh0Out.size() * sizeof(float), dh0DeviceAddr,
                    dh0Out.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s dh0 failed. ERROR: %d\n", layout.c_str(), ret); return ret);

  std::vector<float> dxRef(dxOut.size(), 0.0f);
  std::vector<float> dwRef(dwOut.size(), 0.0f);
  std::vector<float> dbRef(dbOut.size(), 0.0f);
  std::vector<float> dh0Ref(dh0Out.size(), 0.0f);
  if (isNtD) {
    BuildVarlenReference(xLogical, weightHost, dyLogical, initialStateHost, dhtHost, queryStartLocHost,
                         logicalT, D, W, dxRef, dwRef, dbRef, dh0Ref, false);
  } else {
    std::vector<int64_t> fixedQsl = {0, logicalT};
    for (int64_t b = 0; b < logicalB; ++b) {
      std::vector<float> xOne(xLogical.begin() + b * logicalT * D, xLogical.begin() + (b + 1) * logicalT * D);
      std::vector<float> dyOne(dyLogical.begin() + b * logicalT * D, dyLogical.begin() + (b + 1) * logicalT * D);
      std::vector<float> initOne(initialStateHost.begin() + b * D * W, initialStateHost.begin() + (b + 1) * D * W);
      std::vector<float> dhtOne(dhtHost.begin() + b * D * W, dhtHost.begin() + (b + 1) * D * W);
      std::vector<float> dxOne(logicalT * D, 0.0f);
      std::vector<float> dwOne(W * D, 0.0f);
      std::vector<float> dbOne(D, 0.0f);
      std::vector<float> dh0One(D * W, 0.0f);
      BuildVarlenReference(xOne, weightHost, dyOne, initOne, dhtOne, fixedQsl,
                           logicalT, D, W, dxOne, dwOne, dbOne, dh0One, false);
      std::copy(dxOne.begin(), dxOne.end(), dxRef.begin() + b * logicalT * D);
      for (size_t i = 0; i < dwRef.size(); ++i) {
        dwRef[i] += dwOne[i];
      }
      for (size_t i = 0; i < dbRef.size(); ++i) {
        dbRef[i] += dbOne[i];
      }
      std::copy(dh0One.begin(), dh0One.end(), dh0Ref.begin() + b * D * W);
    }
  }

  LOG_PRINT("\nrunning causal_conv1d_bwd FP32 %s input-layout precision example\n", layout.c_str());
  bool ok = true;
  ok = AllClose(dxOut, dxRef, 1e-4f, 1e-4f, "layout dx") && ok;
  ok = AllClose(dwOut, dwRef, 1e-4f, 1e-4f, "layout dw") && ok;
  ok = AllClose(dbOut, dbRef, 1e-4f, 1e-4f, "layout db") && ok;
  ok = AllClose(dh0Out, dh0Ref, 1e-4f, 1e-4f, "layout dh0") && ok;
  return ok ? ACL_SUCCESS : 1;
}

int RunVarlenBf16Test(aclrtStream& stream) {
  constexpr int64_t B = 3;
  constexpr int64_t D = 80;
  constexpr int64_t W = 4;
  constexpr int64_t activation = 0;
  char inputLayout[] = "TND";
  std::vector<int64_t> queryStartLocHost = {0, 7, 72, 202};
  int64_t totalTokens = queryStartLocHost.back();

  std::vector<int64_t> xShape = {totalTokens, D};
  std::vector<int64_t> yShape = {totalTokens, D};
  std::vector<int64_t> weightShape = {W, D};
  std::vector<int64_t> dyShape = {totalTokens, D};
  std::vector<int64_t> initialStateShape = {B, W, D};
  std::vector<int64_t> dhtShape = {B, W, D};
  std::vector<int64_t> dxShape = {totalTokens, D};
  std::vector<int64_t> dwShape = {W, D};
  std::vector<int64_t> dbShape = {D};
  std::vector<int64_t> dh0Shape = {B, W, D};

  std::vector<float> xHost(GetShapeSize(xShape));
  std::vector<float> yHost(GetShapeSize(yShape), 0.0f);
  std::vector<float> weightHost(GetShapeSize(weightShape));
  std::vector<float> dyHost(GetShapeSize(dyShape));
  std::vector<float> initialStateHost(GetShapeSize(initialStateShape), 0.0f);
  std::vector<float> dhtHost(GetShapeSize(dhtShape), 0.0f);

  for (size_t i = 0; i < xHost.size(); ++i) {
    xHost[i] = (static_cast<int>(i % 23) - 11) * 0.03125f;
  }
  for (size_t i = 0; i < dyHost.size(); ++i) {
    dyHost[i] = (static_cast<int>((i * 7) % 29) - 14) * 0.015625f;
  }
  for (size_t i = 0; i < weightHost.size(); ++i) {
    weightHost[i] = (static_cast<int>((i * 5) % 17) - 8) * 0.0625f;
  }
  for (size_t i = 0; i < initialStateHost.size(); ++i) {
    initialStateHost[i] = (static_cast<int>((i * 5) % 23) - 11) * 0.03125f;
  }
  for (size_t i = 0; i < dhtHost.size(); ++i) {
    dhtHost[i] = (static_cast<int>((i * 7) % 29) - 14) * 0.015625f;
  }

  std::vector<uint16_t> xBf16 = FloatVectorToBf16(xHost);
  std::vector<uint16_t> yBf16 = FloatVectorToBf16(yHost);
  std::vector<uint16_t> weightBf16 = FloatVectorToBf16(weightHost);
  std::vector<uint16_t> dyBf16 = FloatVectorToBf16(dyHost);
  std::vector<uint16_t> initialStateBf16 = FloatVectorToBf16(initialStateHost);
  std::vector<uint16_t> dhtBf16 = FloatVectorToBf16(dhtHost);

  std::vector<uint16_t> dxHost(GetShapeSize(dxShape), 0);
  std::vector<uint16_t> dwHost(GetShapeSize(dwShape), 0);
  std::vector<uint16_t> dbHost(GetShapeSize(dbShape), 0);
  std::vector<uint16_t> dh0Host(GetShapeSize(dh0Shape), 0);

  void *xDeviceAddr = nullptr, *yDeviceAddr = nullptr, *weightDeviceAddr = nullptr, *dyDeviceAddr = nullptr;
  void *initialStateDeviceAddr = nullptr, *dhtDeviceAddr = nullptr;
  void *dxDeviceAddr = nullptr, *dwDeviceAddr = nullptr, *dbDeviceAddr = nullptr, *dh0DeviceAddr = nullptr;

  aclTensor *x = nullptr, *y = nullptr, *weight = nullptr, *dy = nullptr;
  aclTensor *initialState = nullptr, *dht = nullptr;
  aclIntArray *queryStartLoc = nullptr;
  aclTensor *dx = nullptr, *dw = nullptr, *db = nullptr, *dh0 = nullptr;

  auto ret = CreateAclTensor(xBf16, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
  ret = CreateAclTensor(yBf16, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);
  ret = CreateAclTensor(weightBf16, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  ret = CreateAclTensor(dyBf16, dyShape, &dyDeviceAddr, aclDataType::ACL_BF16, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dyTensorPtr(dy, aclDestroyTensor);
  ret = CreateAclTensor(initialStateBf16, initialStateShape, &initialStateDeviceAddr, aclDataType::ACL_BF16, &initialState);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> initialStateTensorPtr(initialState, aclDestroyTensor);
  ret = CreateAclTensor(dhtBf16, dhtShape, &dhtDeviceAddr, aclDataType::ACL_BF16, &dht);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dhtTensorPtr(dht, aclDestroyTensor);
  queryStartLoc = aclCreateIntArray(queryStartLocHost.data(), queryStartLocHost.size());
  CHECK_RET(queryStartLoc != nullptr, return ACL_ERROR_BAD_ALLOC);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> queryStartLocPtr(queryStartLoc, aclDestroyIntArray);

  ret = CreateAclTensor(dxHost, dxShape, &dxDeviceAddr, aclDataType::ACL_BF16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dxTensorPtr(dx, aclDestroyTensor);
  ret = CreateAclTensor(dwHost, dwShape, &dwDeviceAddr, aclDataType::ACL_BF16, &dw);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dwTensorPtr(dw, aclDestroyTensor);
  ret = CreateAclTensor(dbHost, dbShape, &dbDeviceAddr, aclDataType::ACL_BF16, &db);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dbTensorPtr(db, aclDestroyTensor);
  ret = CreateAclTensor(dh0Host, dh0Shape, &dh0DeviceAddr, aclDataType::ACL_BF16, &dh0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> dh0TensorPtr(dh0, aclDestroyTensor);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnCausalConv1dBwdGetWorkspaceSize(x, y, weight, dy, initialState, dht, queryStartLoc,
                                             activation, inputLayout, dx, dw, db, dh0, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("varlen aclnnCausalConv1dBwdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("varlen allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }

  ret = aclnnCausalConv1dBwd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("varlen aclnnCausalConv1dBwd failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("varlen aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  std::vector<uint16_t> dxOutBf16(dxHost.size(), 0);
  ret = aclrtMemcpy(dxOutBf16.data(), dxOutBf16.size() * sizeof(uint16_t), dxDeviceAddr,
                    dxOutBf16.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy varlen dx failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dxOut = Bf16VectorToFloat(dxOutBf16);

  std::vector<uint16_t> dwOutBf16(dwHost.size(), 0);
  ret = aclrtMemcpy(dwOutBf16.data(), dwOutBf16.size() * sizeof(uint16_t), dwDeviceAddr,
                    dwOutBf16.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy varlen dw failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dwOut = Bf16VectorToFloat(dwOutBf16);

  std::vector<uint16_t> dbOutBf16(dbHost.size(), 0);
  ret = aclrtMemcpy(dbOutBf16.data(), dbOutBf16.size() * sizeof(uint16_t), dbDeviceAddr,
                    dbOutBf16.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy varlen db failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dbOut = Bf16VectorToFloat(dbOutBf16);

  std::vector<uint16_t> dh0OutBf16(dh0Host.size(), 0);
  ret = aclrtMemcpy(dh0OutBf16.data(), dh0OutBf16.size() * sizeof(uint16_t), dh0DeviceAddr,
                    dh0OutBf16.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy varlen dh0 failed. ERROR: %d\n", ret); return ret);
  std::vector<float> dh0Out = Bf16VectorToFloat(dh0OutBf16);

  std::vector<float> dxRef(dxOut.size(), 0.0f);
  std::vector<float> dwRef(dwOut.size(), 0.0f);
  std::vector<float> dbRef(dbOut.size(), 0.0f);
  std::vector<float> dh0Ref(dh0Out.size(), 0.0f);
  BuildVarlenReference(Bf16VectorToFloat(xBf16), Bf16VectorToFloat(weightBf16), Bf16VectorToFloat(dyBf16),
                       Bf16VectorToFloat(initialStateBf16), Bf16VectorToFloat(dhtBf16),
                       queryStartLocHost, totalTokens, D, W, dxRef, dwRef, dbRef, dh0Ref);

  LOG_PRINT("\nrunning causal_conv1d_bwd BF16 varlen precision example\n");
  bool ok = true;
  ok = AllClose(dxOut, dxRef, 1e-2f, 1e-2f, "varlen dx") && ok;
  ok = AllClose(dwOut, dwRef, 1e-2f, 1e-2f, "varlen dw") && ok;
  ok = AllClose(dbOut, dbRef, 1e-2f, 1e-2f, "varlen db") && ok;
  ok = AllClose(dh0Out, dh0Ref, 1e-2f, 1e-2f, "varlen dh0") && ok;
  return ok ? ACL_SUCCESS : 1;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnCausalConv1dBwdTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dBwdTest failed. ERROR: %d\n", ret); return ret);
  ret = RunVarlenFp32SingleSeqTest(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunVarlenFp32SingleSeqTest failed. ERROR: %d\n", ret); return ret);
  ret = RunInputLayoutFp32Test(stream, "BNSD");
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunInputLayoutFp32Test(BNSD) failed. ERROR: %d\n", ret); return ret);
  ret = RunInputLayoutFp32Test(stream, "NTD");
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunInputLayoutFp32Test(NTD) failed. ERROR: %d\n", ret); return ret);
  ret = RunVarlenBf16Test(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunVarlenBf16Test failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
