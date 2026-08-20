#include <acl/acl.h>
#include <aclnnop/aclnn_chunk_kda_bwd_a.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(aclError status, const char *what)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(status));
    }
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000U) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13U));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    mantissa += 0x1000U;
    if (mantissa & 0x800000U) {
        mantissa = 0;
        if (++exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) |
                                 (mantissa >> 13U));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1fU;
    uint32_t mantissa = value & 0x3ffU;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x3ffU;
            bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t Elements(const std::vector<int64_t> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), size_t{1},
                           [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
}

struct DeviceTensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    aclDataType dtype;
    size_t bytes;
    void *data = nullptr;
    aclTensor *tensor = nullptr;

    DeviceTensor(std::vector<int64_t> dims, aclDataType type, size_t elementBytes,
                 const void *host = nullptr)
        : shape(std::move(dims)), strides(shape.size(), 1), dtype(type),
          bytes(Elements(shape) * elementBytes)
    {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        Check(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        if (host != nullptr) {
            Check(aclrtMemcpy(data, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy H2D");
        } else {
            Check(aclrtMemset(data, bytes, 0, bytes), "aclrtMemset");
        }
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), data);
        if (tensor == nullptr) throw std::runtime_error("aclCreateTensor returned nullptr");
    }

    ~DeviceTensor()
    {
        if (tensor != nullptr) aclDestroyTensor(tensor);
        if (data != nullptr) aclrtFree(data);
    }

    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;
};

std::vector<uint16_t> MakeHalf(size_t count, uint32_t seed)
{
    std::vector<uint16_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        const int32_t q = static_cast<int32_t>((i * 17U + seed * 13U) % 29U) - 14;
        result[i] = FloatToHalf(static_cast<float>(q) / 256.0F);
    }
    return result;
}

std::vector<float> ToFloat(const std::vector<uint16_t> &input)
{
    std::vector<float> result(input.size());
    std::transform(input.begin(), input.end(), result.begin(), HalfToFloat);
    return result;
}

struct Error {
    double maxAbs = 0;
    double meanAbs = 0;
    double cosine = 0;
};

Error Compare(const std::vector<float> &actual, const std::vector<float> &expected)
{
    double sum = 0, dot = 0, aa = 0, ee = 0;
    Error result;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = std::abs(static_cast<double>(actual[i]) - expected[i]);
        result.maxAbs = std::max(result.maxAbs, d);
        sum += d;
        dot += static_cast<double>(actual[i]) * expected[i];
        aa += static_cast<double>(actual[i]) * actual[i];
        ee += static_cast<double>(expected[i]) * expected[i];
    }
    result.meanAbs = sum / actual.size();
    result.cosine = dot / std::sqrt(std::max(aa * ee, 1e-30));
    return result;
}

void Print(const char *name, const Error &error)
{
    std::cout << name << " max_abs=" << error.maxAbs
              << " mean_abs=" << error.meanAbs
              << " cosine=" << error.cosine << '\n';
}

int RunA(aclrtStream stream)
{
    constexpr int C = 64;
    constexpr int K = 128;
    constexpr int V = 128;
    constexpr float scale = 0.125F;
    const auto aqkH = MakeHalf(C * C, 1);
    const auto qgH = MakeHalf(C * K, 2);
    const auto vNewH = MakeHalf(C * V, 3);
    const auto hH = MakeHalf(K * V, 4);
    const auto doH = MakeHalf(C * V, 5);
    const auto aqkF = ToFloat(aqkH);
    const auto qgF = ToFloat(qgH);
    const auto vNewF = ToFloat(vNewH);
    const auto hF = ToFloat(hH);
    const auto doF = ToFloat(doH);

    DeviceTensor aqk({1, 1, C, C}, ACL_FLOAT16, 2, aqkH.data());
    DeviceTensor qg({1, 1, C, K}, ACL_FLOAT16, 2, qgH.data());
    DeviceTensor vNew({1, 1, C, V}, ACL_FLOAT16, 2, vNewH.data());
    DeviceTensor h({1, 1, 1, K, V}, ACL_FLOAT16, 2, hH.data());
    DeviceTensor dO({1, 1, C, V}, ACL_FLOAT16, 2, doH.data());
    DeviceTensor dv0({1, 1, C, V}, ACL_FLOAT16, 2);
    DeviceTensor q0({1, 1, 1, K, V}, ACL_FLOAT, 4);
    DeviceTensor dqRaw({1, 1, C, K}, ACL_FLOAT, 4);
    DeviceTensor dAqk({1, 1, C, C}, ACL_FLOAT, 4);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    Check(aclnnChunkKdaBwdAGetWorkspaceSize(
              aqk.tensor, qg.tensor, vNew.tensor, h.tensor, dO.tensor,
              nullptr, nullptr, scale, C, dv0.tensor, q0.tensor,
              dqRaw.tensor, dAqk.tensor, &workspaceSize, &executor),
          "Kernel A GetWorkspaceSize");
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        Check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
              "workspace malloc");
    }
    const auto start = std::chrono::steady_clock::now();
    Check(aclnnChunkKdaBwdA(workspace, workspaceSize, executor, stream),
          "Kernel A launch");
    Check(aclrtSynchronizeStream(stream), "Kernel A synchronize");
    const auto end = std::chrono::steady_clock::now();
    std::cout << "KERNEL_A_WORKSPACE_BYTES=" << workspaceSize << '\n';
    std::cout << "KERNEL_A_FIRST_RUN_US="
              << std::chrono::duration<double, std::micro>(end - start).count() << '\n';

    std::vector<uint16_t> dv0H(C * V);
    std::vector<float> q0H(K * V), dqRawH(C * K), dAqkH(C * C);
    Check(aclrtMemcpy(dv0H.data(), dv0.bytes, dv0.data, dv0.bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST), "dv0 D2H");
    Check(aclrtMemcpy(q0H.data(), q0.bytes, q0.data, q0.bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST), "q0 D2H");
    Check(aclrtMemcpy(dqRawH.data(), dqRaw.bytes, dqRaw.data, dqRaw.bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST), "dq D2H");
    Check(aclrtMemcpy(dAqkH.data(), dAqk.bytes, dAqk.data, dAqk.bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST), "dAqk D2H");
    if (workspace != nullptr) aclrtFree(workspace);

    std::vector<float> dvRef(C * V, 0), q0Ref(K * V, 0), dqRef(C * K, 0),
        daRef(C * C, 0);
    for (int i = 0; i < C; ++i) {
        for (int j = 0; j < V; ++j) {
            for (int x = 0; x < C; ++x) dvRef[i * V + j] += aqkF[x * C + i] * doF[x * V + j];
        }
        for (int j = 0; j < K; ++j) {
            for (int x = 0; x < V; ++x) dqRef[i * K + j] += doF[i * V + x] * hF[j * V + x];
        }
        for (int j = 0; j < C; ++j) {
            for (int x = 0; x < V; ++x) daRef[i * C + j] += doF[i * V + x] * vNewF[j * V + x];
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < V; ++j) {
            for (int x = 0; x < C; ++x) q0Ref[i * V + j] += qgF[x * K + i] * doF[x * V + j];
            q0Ref[i * V + j] *= scale;
        }
    }
    const Error eDv = Compare(ToFloat(dv0H), dvRef);
    const Error eQ0 = Compare(q0H, q0Ref);
    const Error eDq = Compare(dqRawH, dqRef);
    const Error eDa = Compare(dAqkH, daRef);
    Print("dv0", eDv);
    Print("Q0", eQ0);
    Print("dq_raw", eDq);
    Print("dAqk", eDa);
    const bool pass = eDv.cosine > 0.99 && eQ0.cosine > 0.99 &&
                      eDq.cosine > 0.99 && eDa.cosine > 0.99 &&
                      eDv.meanAbs < 3e-3 && eQ0.meanAbs < 3e-3 &&
                      eDq.meanAbs < 3e-3 && eDa.meanAbs < 3e-3;
    std::cout << "KERNEL_A_PRECISION=" << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? 0 : 2;
}

}  // namespace

int main(int argc, char **argv)
{
    const int device = argc > 1 ? std::stoi(argv[1]) : 0;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        const int status = RunA(stream);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device);
        aclFinalize();
        return status;
    } catch (const std::exception &error) {
        std::cerr << "CANARY_ERROR: " << error.what() << std::endl;
        return 1;
    }
}
