/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_local_cumsum.h"

namespace {
constexpr double DEFAULT_ATOL = 2e-5;
constexpr double DEFAULT_RTOL = 2e-5;

#define CHECK_ACL(expr)                                                                                                 \
    do {                                                                                                                \
        auto _ret = (expr);                                                                                             \
        if (_ret != ACL_SUCCESS) {                                                                                      \
            std::cerr << #expr << " failed, ret=" << _ret << std::endl;                                                \
            return _ret;                                                                                                \
        }                                                                                                               \
    } while (0)

struct TestCase {
    std::string name;
    int64_t b = 0;
    int64_t t = 0;
    int64_t h = 0;
    int64_t chunkSize = 0;
    bool reverse = false;
    double scale = 1.0;
    bool isVarlen = false;
    std::string gFile;
    std::string goldenFile;
    std::string cuFile;
    std::string chunkIndicesFile;
};

struct TensorHandle {
    aclTensor *tensor = nullptr;
    void *deviceAddr = nullptr;
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::string JoinPath(const std::string &dir, const std::string &file)
{
    if (file == "-") {
        return file;
    }
    if (file.empty() || file[0] == '/') {
        return file;
    }
    if (dir.empty() || dir.back() == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

std::string DirName(const std::string &path)
{
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

template <typename T>
std::vector<T> LoadBinary(const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        throw std::runtime_error("failed to open " + path);
    }
    fs.seekg(0, std::ios::end);
    const auto bytes = fs.tellg();
    fs.seekg(0, std::ios::beg);
    if (bytes < 0 || static_cast<size_t>(bytes) % sizeof(T) != 0) {
        throw std::runtime_error("bad binary size for " + path);
    }
    std::vector<T> data(static_cast<size_t>(bytes) / sizeof(T));
    fs.read(reinterpret_cast<char *>(data.data()), bytes);
    if (!fs) {
        throw std::runtime_error("failed to read " + path);
    }
    return data;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                    TensorHandle &handle)
{
    const int64_t elementNum = GetShapeSize(shape);
    const size_t bytes = static_cast<size_t>(elementNum) * sizeof(T);
    if (static_cast<int64_t>(hostData.size()) != elementNum) {
        std::cerr << "host data size " << hostData.size() << " does not match shape element count " << elementNum
                  << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    CHECK_ACL(aclrtMalloc(&handle.deviceAddr, std::max<size_t>(bytes, sizeof(T)), ACL_MEM_MALLOC_HUGE_FIRST));
    if (bytes > 0) {
        CHECK_ACL(aclrtMemcpy(handle.deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    auto strides = MakeStrides(shape);
    handle.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), handle.deviceAddr);
    if (handle.tensor == nullptr) {
        std::cerr << "aclCreateTensor failed" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

void DestroyTensor(TensorHandle &handle)
{
    if (handle.tensor != nullptr) {
        aclDestroyTensor(handle.tensor);
        handle.tensor = nullptr;
    }
    if (handle.deviceAddr != nullptr) {
        aclrtFree(handle.deviceAddr);
        handle.deviceAddr = nullptr;
    }
}

std::vector<TestCase> LoadCases(const std::string &manifestPath)
{
    std::ifstream fs(manifestPath);
    if (!fs) {
        throw std::runtime_error("failed to open manifest " + manifestPath);
    }
    const std::string baseDir = DirName(manifestPath);
    std::vector<TestCase> cases;
    std::string line;
    while (std::getline(fs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        int reverse = 0;
        int isVarlen = 0;
        TestCase tc;
        iss >> tc.name >> tc.b >> tc.t >> tc.h >> tc.chunkSize >> reverse >> tc.scale >> isVarlen >>
            tc.gFile >> tc.goldenFile >> tc.cuFile >> tc.chunkIndicesFile;
        if (!iss) {
            throw std::runtime_error("bad manifest line: " + line);
        }
        tc.reverse = reverse != 0;
        tc.isVarlen = isVarlen != 0;
        tc.gFile = JoinPath(baseDir, tc.gFile);
        tc.goldenFile = JoinPath(baseDir, tc.goldenFile);
        tc.cuFile = JoinPath(baseDir, tc.cuFile);
        tc.chunkIndicesFile = JoinPath(baseDir, tc.chunkIndicesFile);
        cases.push_back(tc);
    }
    return cases;
}

int RunCase(const TestCase &tc, aclrtStream stream, double atol, double rtol)
{
    const std::vector<int64_t> gShape = {tc.b, tc.t, tc.h};
    const int64_t elementNum = GetShapeSize(gShape);
    auto gHost = LoadBinary<float>(tc.gFile);
    auto golden = LoadBinary<float>(tc.goldenFile);
    if (static_cast<int64_t>(gHost.size()) != elementNum || static_cast<int64_t>(golden.size()) != elementNum) {
        std::cerr << tc.name << ": data size mismatch" << std::endl;
        return 1;
    }

    TensorHandle gTensor;
    TensorHandle outTensor;
    TensorHandle cuTensor;
    TensorHandle chunkTensor;
    std::vector<float> outInit(static_cast<size_t>(elementNum), 0.0f);

    int ret = CreateAclTensor(gHost, gShape, ACL_FLOAT, gTensor);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = CreateAclTensor(outInit, gShape, ACL_FLOAT, outTensor);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    if (tc.isVarlen) {
        auto cuHost = LoadBinary<int64_t>(tc.cuFile);
        auto chunkHost = LoadBinary<int64_t>(tc.chunkIndicesFile);
        if (chunkHost.size() % 2 != 0) {
            std::cerr << tc.name << ": chunk_indices_out element count must be even" << std::endl;
            return 1;
        }
        ret = CreateAclTensor(cuHost, {static_cast<int64_t>(cuHost.size())}, ACL_INT64, cuTensor);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        ret = CreateAclTensor(chunkHost, {static_cast<int64_t>(chunkHost.size() / 2), 2}, ACL_INT64, chunkTensor);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    } else {
        std::vector<int64_t> empty;
        ret = CreateAclTensor(empty, {0}, ACL_INT64, cuTensor);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        ret = CreateAclTensor(empty, {0}, ACL_INT64, chunkTensor);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    char outputDtype[] = "float32";
    ret = aclnnChunkLocalCumsumGetWorkspaceSize(gTensor.tensor, cuTensor.tensor, chunkTensor.tensor, tc.chunkSize,
                                                tc.reverse, tc.scale, false, outputDtype, outTensor.tensor,
                                                &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::cerr << tc.name << ": aclnnChunkLocalCumsumGetWorkspaceSize failed, ret=" << ret << std::endl;
        return ret;
    }

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ret = aclnnChunkLocalCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << tc.name << ": aclnnChunkLocalCumsum failed, ret=" << ret << std::endl;
        return ret;
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> actual(static_cast<size_t>(elementNum), 0.0f);
    CHECK_ACL(aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outTensor.deviceAddr,
                          actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    double maxAbs = 0.0;
    double maxRel = 0.0;
    int64_t maxIndex = 0;
    int64_t badCount = 0;
    for (int64_t i = 0; i < elementNum; ++i) {
        const double expected = static_cast<double>(golden[static_cast<size_t>(i)]);
        const double got = static_cast<double>(actual[static_cast<size_t>(i)]);
        const double absDiff = std::abs(got - expected);
        const double relDiff = absDiff / std::max(1.0, std::abs(expected));
        if (absDiff > maxAbs) {
            maxAbs = absDiff;
            maxRel = relDiff;
            maxIndex = i;
        }
        if (!std::isfinite(got) || absDiff > (atol + rtol * std::abs(expected))) {
            ++badCount;
        }
    }

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensor(chunkTensor);
    DestroyTensor(cuTensor);
    DestroyTensor(outTensor);
    DestroyTensor(gTensor);

    std::cout << std::fixed << std::setprecision(9) << tc.name << " max_abs=" << maxAbs << " max_rel=" << maxRel
              << " max_index=" << maxIndex << " bad_count=" << badCount << " elements=" << elementNum << std::endl;
    if (badCount != 0) {
        if (std::getenv("CUMSUM_DUMP_MISMATCH") != nullptr) {
            int printed = 0;
            for (int64_t i = 0; i < elementNum && printed < 16; ++i) {
                const double expected = static_cast<double>(golden[static_cast<size_t>(i)]);
                const double got = static_cast<double>(actual[static_cast<size_t>(i)]);
                const double absDiff = std::abs(got - expected);
                if (!std::isfinite(got) || absDiff > (atol + rtol * std::abs(expected))) {
                    std::cerr << "  mismatch[" << i << "] got=" << got << " expected=" << expected
                              << " input=" << gHost[static_cast<size_t>(i)] << std::endl;
                    ++printed;
                }
            }
        }
        std::cerr << tc.name << " FAILED" << std::endl;
        return 1;
    }
    return 0;
}

double GetEnvDouble(const char *name, double defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return std::strtod(value, nullptr);
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cases.txt> [case_name ...]" << std::endl;
        return 1;
    }
    const std::string manifestPath = argv[1];
    std::set<std::string> selectedCases;
    for (int i = 2; i < argc; ++i) {
        selectedCases.insert(argv[i]);
    }
    const int32_t deviceId = std::getenv("ASCEND_DEVICE_ID") == nullptr ? 0 : std::atoi(std::getenv("ASCEND_DEVICE_ID"));
    const double atol = GetEnvDouble("CUMSUM_ATOL", DEFAULT_ATOL);
    const double rtol = GetEnvDouble("CUMSUM_RTOL", DEFAULT_RTOL);

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed, ret=" << ret << std::endl;
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed, ret=" << ret << std::endl;
        aclFinalize();
        return ret;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed, ret=" << ret << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    int failed = 0;
    try {
        auto cases = LoadCases(manifestPath);
        size_t runCount = 0;
        for (const auto &tc : cases) {
            if (selectedCases.empty() || selectedCases.count(tc.name) != 0) {
                ++runCount;
            }
        }
        if (runCount == 0) {
            std::cerr << "No selected ChunkLocalCumsum cases found." << std::endl;
            failed = 1;
        }
        std::cout << "Running " << runCount << " ChunkLocalCumsum cases, atol=" << atol << " rtol=" << rtol
                  << std::endl;
        if (failed == 0) {
            for (const auto &tc : cases) {
                if (!selectedCases.empty() && selectedCases.count(tc.name) == 0) {
                    continue;
                }
                int caseRet = RunCase(tc, stream, atol, rtol);
                if (caseRet != 0) {
                    failed = caseRet;
                    break;
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        failed = 1;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    if (failed == 0) {
        std::cout << "All ChunkLocalCumsum precision cases passed." << std::endl;
    }
    return failed;
}
