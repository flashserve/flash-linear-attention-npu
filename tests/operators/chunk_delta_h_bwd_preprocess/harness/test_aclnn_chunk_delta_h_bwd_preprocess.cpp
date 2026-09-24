/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file test_aclnn_chunk_delta_h_bwd_preprocess.cpp
 * \brief aclnn 直调取数程序：输入/输出全部走文件，保证与 Python 标杆用同一份 bytes。
 *
 * 用法：
 *   run_case <dir> <dtype> <gate> <gFp32> <B> <Hk> <Hv> <T> <K> <V> <chunkSize> <scale> [varlen] [gMode]
 *     dtype : 0 = bf16, 1 = fp16
 *     gate  : 0 = 无门控, 1 = g（标量 gate）, 2 = gk（逐 K gate）, 3 = g 与 gk 同时传（反向用例）
 *     gFp32 : 1 = g 为 FP32（gate==1）；gate==2 时表示 gk 为 FP32（反向用例）
 *     varlen: 1 = 读 cu_seqlens.bin（此时 B 必须为 1）
 *     gMode : 0 = g 为 [B,Hv,T]（默认）；1 = g 为 [B,Hk,T]（反向用例，shape 不匹配）
 *
 * 输入文件（<dir> 下，按 shape / dtype 的原始字节）：q.bin k.bin w.bin do.bin dv.bin
 *   gate==1/3 时加 g.bin；gate==2/3 时加 gk.bin；varlen==1 时加 cu_seqlens.bin
 * 输出文件：<dir>/dhm_acl.bin（FP32，[Hv, K, V+K]）
 */

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_delta_h_bwd_preprocess.h"

#define CHECK_RET(cond, expr)  \
  do {                         \
    if (!(cond)) {             \
      expr;                    \
    }                          \
  } while (0)

int64_t Numel(const std::vector<int64_t> &shape) {
  int64_t size = 1;
  for (auto d : shape) {
    size *= d;
  }
  return size;
}

std::vector<char> ReadFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "open failed: " << path << std::endl;
    exit(1);
  }
  in.seekg(0, std::ios::end);
  size_t len = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<char> buf(len);
  in.read(buf.data(), static_cast<std::streamsize>(len));
  return buf;
}

void WriteFile(const std::string &path, const void *data, size_t len) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(len));
}

int CreateTensorFromFile(const std::string &path, const std::vector<int64_t> &shape, aclDataType dtype,
                         void **deviceAddr, aclTensor **tensor) {
  std::vector<char> host = ReadFile(path);
  int64_t expect = Numel(shape);
  size_t elemSize = (dtype == ACL_FLOAT || dtype == ACL_INT64) ? 4 : 2;
  if (dtype == ACL_INT64) {
    elemSize = 8;
  }
  if (static_cast<size_t>(expect) * elemSize != host.size()) {
    std::cerr << "size mismatch for " << path << ": expect " << expect * elemSize << " got " << host.size()
              << std::endl;
    return -1;
  }
  // T=0 的空张量会产生 0 字节输入（反向用例 neg_10），此时仍要给出一个合法设备指针
  const size_t allocBytes = (host.size() == 0) ? 512 : host.size();
  auto ret = aclrtMalloc(deviceAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "aclrtMalloc failed " << ret << std::endl; return ret);
  if (host.size() > 0) {
    ret = aclrtMemcpy(*deviceAddr, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
  }
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "H2D failed " << ret << std::endl; return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int CreateOutTensor(const std::vector<int64_t> &shape, aclDataType dtype, void **deviceAddr, aclTensor **tensor) {
  auto size = static_cast<size_t>(Numel(shape)) * sizeof(float);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "aclrtMalloc out failed " << ret << std::endl; return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 13) {
    std::cerr << "usage: run_case <dir> <dtype> <gate> <gFp32> <B> <Hk> <Hv> <T> <K> <V> <chunk> <scale> [varlen]"
              << std::endl;
    return 1;
  }
  std::string dir = argv[1];
  int dtypeCode = std::atoi(argv[2]);
  int gate = std::atoi(argv[3]);
  int gFp32 = std::atoi(argv[4]);
  int64_t b = std::atoll(argv[5]);
  int64_t hk = std::atoll(argv[6]);
  int64_t hv = std::atoll(argv[7]);
  int64_t t = std::atoll(argv[8]);
  int64_t kDim = std::atoll(argv[9]);
  int64_t vDim = std::atoll(argv[10]);
  int64_t chunk = std::atoll(argv[11]);
  double scale = std::atof(argv[12]);
  int varlen = (argc > 13) ? std::atoi(argv[13]) : 0;
  int gMode = (argc > 14) ? std::atoi(argv[14]) : 0;

  aclDataType modelType = (dtypeCode == 0) ? ACL_BF16 : ACL_FLOAT16;
  aclDataType gateType = (gFp32 != 0) ? ACL_FLOAT : modelType;
  // gate==2 时 gFp32 表示 gk 的 dtype（反向用例 neg_08：gk 为 FP32）
  aclDataType gkType = (gate == 2 && gFp32 != 0) ? ACL_FLOAT : modelType;

  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "aclInit failed " << ret << std::endl; return ret);
  ret = aclrtSetDevice(0);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "setDevice failed " << ret << std::endl; return ret);
  aclrtStream stream = nullptr;
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "createStream failed " << ret << std::endl; return ret);

  void *qAddr = nullptr, *kAddr = nullptr, *wAddr = nullptr, *doAddr = nullptr, *dvAddr = nullptr;
  void *gAddr = nullptr, *gkAddr = nullptr, *cuAddr = nullptr, *dhmAddr = nullptr;
  aclTensor *q = nullptr, *k = nullptr, *w = nullptr, *dO = nullptr, *dv = nullptr;
  aclTensor *g = nullptr, *gk = nullptr, *cu = nullptr, *dhm = nullptr;
  aclIntArray *cuArray = nullptr;

  std::vector<int64_t> qkShape{b, hk, t, kDim};
  std::vector<int64_t> wShape{b, hv, t, kDim};
  std::vector<int64_t> vShape{b, hv, t, vDim};
  std::vector<int64_t> gShape{b, hv, t};
  if (gMode == 1) {
    gShape = std::vector<int64_t>{b, hk, t};  // 反向用例：g 的 head 维与 Hk 对齐（与算子约束不符）
  }
  std::vector<int64_t> gkShape{b, hv, t, kDim};
  std::vector<int64_t> dhmShape{hv, kDim, vDim + kDim};

  if (CreateTensorFromFile(dir + "/q.bin", qkShape, modelType, &qAddr, &q) != 0) return 1;
  if (CreateTensorFromFile(dir + "/k.bin", qkShape, modelType, &kAddr, &k) != 0) return 1;
  if (CreateTensorFromFile(dir + "/w.bin", wShape, modelType, &wAddr, &w) != 0) return 1;
  if (CreateTensorFromFile(dir + "/do.bin", vShape, modelType, &doAddr, &dO) != 0) return 1;
  if (CreateTensorFromFile(dir + "/dv.bin", vShape, modelType, &dvAddr, &dv) != 0) return 1;
  if (gate == 1) {
    if (CreateTensorFromFile(dir + "/g.bin", gShape, gateType, &gAddr, &g) != 0) return 1;
  } else if (gate == 2) {
    if (CreateTensorFromFile(dir + "/gk.bin", gkShape, gkType, &gkAddr, &gk) != 0) return 1;
  } else if (gate == 3) {
    if (CreateTensorFromFile(dir + "/g.bin", gShape, gateType, &gAddr, &g) != 0) return 1;
    if (CreateTensorFromFile(dir + "/gk.bin", gkShape, modelType, &gkAddr, &gk) != 0) return 1;
  }
  if (varlen != 0) {
    // aclnn 侧 cu_seqlens 是 aclIntArray（不是 aclTensor）
    std::vector<char> cuHost = ReadFile(dir + "/cu_seqlens.bin");
    const size_t n = cuHost.size() / sizeof(int64_t);
    std::vector<int64_t> cuVals(n);
    std::memcpy(cuVals.data(), cuHost.data(), cuHost.size());
    cuArray = aclCreateIntArray(cuVals.data(), static_cast<uint64_t>(n));
    if (cuArray == nullptr) {
      std::cerr << "aclCreateIntArray failed" << std::endl;
      return 1;
    }
  }
  if (CreateOutTensor(dhmShape, ACL_FLOAT, &dhmAddr, &dhm) != 0) return 1;
  // 输出缓冲用非零 sentinel 初值（而不是全 0）：这样「kernel 没写」会以 sentinel 暴露出来，
  // 与「kernel 写错值」区分开；只有真正被写过的区域才会与 CPU 标杆比较。
  std::vector<float> sentinel(static_cast<size_t>(Numel(dhmShape)), -12345.0f);
  ret = aclrtMemcpy(dhmAddr, sentinel.size() * sizeof(float), sentinel.data(), sentinel.size() * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "init dhm failed " << ret << std::endl; return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  ret = aclnnChunkDeltaHBwdPreprocessGetWorkspaceSize(q, k, w, dO, dv, g, gk, cuArray, scale, chunk, dhm,
                                                      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "GetWorkspaceSize failed " << ret << std::endl; return ret);
  void *workspaceAddr = nullptr;
  ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "malloc workspace failed " << ret << std::endl; return ret);
  // workspace 预清零：kernel 只写自己规划到的子区，未写的部分必须能被区分为「未写」而不是残留数据
  ret = aclrtMemset(workspaceAddr, static_cast<size_t>(workspaceSize), 0, static_cast<size_t>(workspaceSize));
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "memset workspace failed " << ret << std::endl; return ret);
  ret = aclnnChunkDeltaHBwdPreprocess(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "aclnn execute failed " << ret << std::endl; return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "synchronize failed " << ret << std::endl; return ret);

  std::vector<float> out(static_cast<size_t>(Numel(dhmShape)));
  ret = aclrtMemcpy(out.data(), out.size() * sizeof(float), dhmAddr, out.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, std::cerr << "D2H failed " << ret << std::endl; return ret);
  WriteFile(dir + "/dhm_acl.bin", out.data(), out.size() * sizeof(float));
  // 额外 dump 整个 workspace，便于逐平面排查（偏移由 host tiling 公式可复算）
  if (workspaceSize > 0) {
    std::vector<char> ws(static_cast<size_t>(workspaceSize));
    ret = aclrtMemcpy(ws.data(), ws.size(), workspaceAddr, ws.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      WriteFile(dir + "/workspace.bin", ws.data(), ws.size());
      std::cout << "wrote " << dir << "/workspace.bin (" << ws.size() << " bytes)" << std::endl;
    }
  }
  std::cout << "dhm[0]=" << out[0] << " dhm[1]=" << out[1] << " dhm_last=" << out[out.size() - 1] << std::endl;
  std::cout << "wrote " << dir << "/dhm_acl.bin (" << out.size() << " floats)" << std::endl;

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
