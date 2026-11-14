// Adapted from CATLASS examples/00_basic_matmul, Copyright (c) 2025 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0, provided AS IS without warranties.
// See LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt in the repository root.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include <acl/acl.h>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

// 1. Select real CATLASS device components. C = A @ B, all row-major.
// A/B are FP16, the Cube accumulator and C are FP32.
using AType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
using BType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
using CType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
using Policy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
using Tile = Catlass::GemmShape<16, 16, 16>;
using BlockMmad = Catlass::Gemm::Block::BlockMmad<Policy, Tile, Tile, AType, BType, CType>;
using Scheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;
using Kernel = Catlass::Gemm::Kernel::BasicMatmul<BlockMmad, void, Scheduler>;
using DeviceGemm = Catlass::Gemm::Device::DeviceGemm<Kernel>;

constexpr size_t kElements = 16 * 16;

void CheckAcl(aclError status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": ACL error " + std::to_string(status));
    }
}

void CheckCatlass(Catlass::Status status, const char *operation)
{
    if (status != Catlass::Status::kSuccess) {
        throw std::runtime_error(std::string(operation) + ": CATLASS error");
    }
}

template <class T>
std::array<T, kElements> ReadMatrix(const std::string &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(kElements * sizeof(T));
    if (!input || input.tellg() != bytes) {
        throw std::runtime_error(path + ": expected exactly " + std::to_string(bytes) + " bytes");
    }
    input.seekg(0);
    std::array<T, kElements> data{};
    if (!input.read(reinterpret_cast<char *>(data.data()), bytes)) {
        throw std::runtime_error(path + ": read failed");
    }
    return data;
}

// Host-side lifetime management. It is separate from the CATLASS kernel above.
struct Runtime {
    int device;
    aclrtStream stream = nullptr;
    bool initialized = false;
    bool deviceSet = false;

    explicit Runtime(int id) : device(id) {}
    void Initialize()
    {
        CheckAcl(aclInit(nullptr), "aclInit");
        initialized = true;
        CheckAcl(aclrtSetDevice(device), "aclrtSetDevice");
        deviceSet = true;
        CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Runtime()
    {
        if (stream != nullptr) { aclrtDestroyStream(stream); }
        if (deviceSet) { aclrtResetDevice(device); }
        if (initialized) { aclFinalize(); }
    }
};

struct Buffer {
    uint8_t *ptr = nullptr;
    explicit Buffer(size_t bytes)
    {
        CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&ptr), bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc");
    }
    ~Buffer() { if (ptr != nullptr) { aclrtFree(ptr); } }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

int main(int argc, char **argv)
{
    try {
        if (argc < 2 || argc > 3) {
            throw std::runtime_error("usage: matmul_cpp CASE_DIRECTORY [DEVICE_ID]");
        }
        int device = 0;
        if (argc == 3) {
            size_t parsed = 0;
            device = std::stoi(argv[2], &parsed);
            if (parsed != std::string(argv[2]).size() || device < 0) {
                throw std::runtime_error("DEVICE_ID must be a nonnegative integer");
            }
        }
        const std::string directory = argv[1];
        const auto a = ReadMatrix<uint16_t>(directory + "/a.bin");
        const auto b = ReadMatrix<uint16_t>(directory + "/b.bin");
        std::array<float, kElements> c;
        c.fill(std::numeric_limits<float>::quiet_NaN());

        Runtime runtime(device);
        runtime.Initialize();
        Buffer deviceA(sizeof(a)), deviceB(sizeof(b)), deviceC(sizeof(c));
        CheckAcl(aclrtMemcpy(deviceA.ptr, sizeof(a), a.data(), sizeof(a), ACL_MEMCPY_HOST_TO_DEVICE),
                 "copy A to device");
        CheckAcl(aclrtMemcpy(deviceB.ptr, sizeof(b), b.data(), sizeof(b), ACL_MEMCPY_HOST_TO_DEVICE),
                 "copy B to device");
        CheckAcl(aclrtMemcpy(deviceC.ptr, sizeof(c), c.data(), sizeof(c), ACL_MEMCPY_HOST_TO_DEVICE),
                 "initialize C");

        // 2. Prepare arguments and launch the CATLASS kernel on ONE Cube core.
        Kernel::Arguments args{Catlass::GemmCoord{16, 16, 16},
                               deviceA.ptr, deviceB.ptr, deviceC.ptr};
        CheckCatlass(DeviceGemm::CanImplement(args), "CanImplement");
        if (DeviceGemm::GetWorkspaceSize(args) != 0) {
            throw std::runtime_error("this teaching kernel is expected to need no GM workspace");
        }
        DeviceGemm gemm;
        CheckCatlass(gemm.Initialize(args), "Initialize");
        CheckCatlass(gemm(runtime.stream, 1), "launch");
        CheckAcl(aclrtSynchronizeStream(runtime.stream), "synchronize");

        // 3. Return actual device output. fixtures.py independently checks it on CPU.
        CheckAcl(aclrtMemcpy(c.data(), sizeof(c), deviceC.ptr, sizeof(c), ACL_MEMCPY_DEVICE_TO_HOST),
                 "copy C to host");
        std::ofstream output(directory + "/cpp_output.bin", std::ios::binary);
        if (!output.write(reinterpret_cast<const char *>(c.data()), sizeof(c))) {
            throw std::runtime_error("cannot write cpp_output.bin");
        }
        std::cout << "CATLASS C++: wrote " << directory << "/cpp_output.bin\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
