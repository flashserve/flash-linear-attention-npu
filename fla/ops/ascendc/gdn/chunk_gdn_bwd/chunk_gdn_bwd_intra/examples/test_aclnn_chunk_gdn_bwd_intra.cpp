/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gdn_bwd_intra.h"

namespace {

struct TensorHandle {
    void *device = nullptr;
    aclTensor *tensor = nullptr;
};

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t result = 1;
    for (int64_t dim : shape) {
        result *= dim;
    }
    return result;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        mantissa += (1U << (shift - 1U)) - 1U + ((mantissa >> shift) & 1U);
        return static_cast<uint16_t>(sign | (mantissa >> shift));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U | (mantissa == 0 ? 0U : 0x0200U));
    }
    mantissa += 0x0fffU + ((mantissa >> 13U) & 1U);
    if ((mantissa & 0x800000U) != 0) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) |
                                 (mantissa >> 13U));
}

float Fp16ToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
    int32_t exponent = static_cast<int32_t>((value >> 10U) & 0x1fU);
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign | (static_cast<uint32_t>(exponent + 112) << 23U) |
                   (mantissa << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | (static_cast<uint32_t>(exponent + 112) << 23U) |
               (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float MainToFloat(uint16_t value, bool mainFp16)
{
    return mainFp16 ? Fp16ToFloat(value) : Bf16ToFloat(value);
}

template <typename T>
bool CreateTensor(const std::vector<T> &host, const std::vector<int64_t> &shape,
                  aclDataType dtype, TensorHandle &handle)
{
    const size_t bytes = host.size() * sizeof(T);
    if (aclrtMalloc(&handle.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(handle.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return false;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    handle.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                    ACL_FORMAT_ND, shape.data(), shape.size(), handle.device);
    return handle.tensor != nullptr;
}

void DestroyTensor(TensorHandle &handle)
{
    if (handle.tensor != nullptr) {
        aclDestroyTensor(handle.tensor);
    }
    if (handle.device != nullptr) {
        aclrtFree(handle.device);
    }
}

bool CopyMainOutput(const TensorHandle &handle, std::vector<uint16_t> &host)
{
    return aclrtMemcpy(host.data(), host.size() * sizeof(uint16_t), handle.device,
                       host.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

float MaxAbsError(const std::vector<uint16_t> &actual, const std::vector<float> &expected,
                  bool mainFp16)
{
    float result = 0.0F;
    for (size_t i = 0; i < actual.size(); ++i) {
        result = std::max(result, std::abs(MainToFloat(actual[i], mainFp16) - expected[i]));
    }
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    constexpr int64_t B = 1;
    constexpr int64_t K = 128;
    constexpr int64_t V = 128;
    constexpr int64_t BT = 64;
    constexpr double SCALE = 1.0;
    const bool useExp2 = argc < 2 || std::string(argv[1]) != "false";
    const int64_t groupSize = argc < 3 ? 4 : std::atoll(argv[2]);
    const bool gateBf16 = argc >= 4 && std::string(argv[3]) == "bf16";
    const int repeats = argc < 5 ? 1 : std::atoi(argv[4]);
    const int64_t T = argc < 6 ? 64 : std::atoll(argv[5]);
    const int64_t HV = argc < 8 ? 4 : std::atoll(argv[7]);
    const bool mainFp16 = argc >= 9 && std::string(argv[8]) == "fp16";
    const bool betaFp32 = argc >= 10 && std::string(argv[9]) == "fp32";
    const bool constantGate = argc >= 11 && std::string(argv[10]) == "constant-g";
    const bool identityD = argc >= 12 && std::string(argv[11]) == "identity-d";
    if (groupSize < 1 || groupSize > 4 || repeats < 1 || T < 1 ||
        HV < 4 || HV % 4 != 0 || HV % groupSize != 0) {
        std::cerr << "Invalid G, repeats, T, or HV\n";
        return 1;
    }
    const int64_t HK = HV / groupSize;

    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "ACL initialization failed\n";
        return 1;
    }

    const std::vector<int64_t> qkShape{B, HK, T, K};
    const std::vector<int64_t> valueShape{B, HV, T, V};
    const std::vector<int64_t> gateShape{B, HV, T};
    const std::vector<int64_t> aShape{B, HV, T, BT};
    const std::vector<int64_t> wShape{B, HV, T, K};
    const aclDataType mainDtype = mainFp16 ? ACL_FLOAT16 : ACL_BF16;
    const auto encodeMain = [mainFp16](float value) {
        return mainFp16 ? FloatToFp16(value) : FloatToBf16(value);
    };

    std::vector<uint16_t> q(Numel(qkShape), encodeMain(0.0F));
    std::vector<uint16_t> k(q.size(), encodeMain(0.0F));
    std::vector<uint16_t> v(Numel(valueShape), encodeMain(0.0F));
    std::vector<uint16_t> dO(v.size(), encodeMain(0.0F));
    std::vector<float> g(Numel(gateShape), 0.0F);
    std::vector<uint16_t> gBf16(g.size(), FloatToBf16(0.0F));
    std::vector<uint16_t> betaBf16(g.size(), FloatToBf16(0.0F));
    std::vector<float> betaFp32Values(g.size(), 0.0F);
    std::vector<uint16_t> a(Numel(aShape), encodeMain(0.0F));
    std::vector<uint16_t> w(Numel(wShape), 0);
    std::vector<uint16_t> u(v.size(), 0);
    std::vector<uint16_t> dv(v.size(), 0);

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < HK; ++h) {
            q[(h * T + t) * K] = encodeMain(0.25F);
            k[(h * T + t) * K] = encodeMain(0.5F);
        }
        for (int64_t h = 0; h < HV; ++h) {
            const int64_t valueOffset = (h * T + t) * V;
            const int64_t gateOffset = h * T + t;
            v[valueOffset] = encodeMain(0.25F + static_cast<float>(h % 32) * 0.0625F);
            v[valueOffset + 1] = encodeMain(0.125F);
            dO[valueOffset] = encodeMain(0.125F);
            g[gateOffset] = constantGate
                ? 0.0F : static_cast<float>((t + h) % 8 - 4) * 0.125F;
            gBf16[gateOffset] = FloatToBf16(g[gateOffset]);
            betaFp32Values[gateOffset] =
                0.5F + static_cast<float>((t + h) % 4) * 0.125F;
            betaBf16[gateOffset] = FloatToBf16(betaFp32Values[gateOffset]);
            a[(h * T + t) * BT + t % BT] = encodeMain(1.0F);
        }
    }

    TensorHandle qTensor, kTensor, vTensor, gTensor, betaTensor, aTensor, dOTensor;
    TensorHandle wTensor, uTensor, dvTensor;
    const bool created =
        CreateTensor(q, qkShape, mainDtype, qTensor) &&
        CreateTensor(k, qkShape, mainDtype, kTensor) &&
        CreateTensor(v, valueShape, mainDtype, vTensor) &&
        (gateBf16 ? CreateTensor(gBf16, gateShape, ACL_BF16, gTensor)
                  : CreateTensor(g, gateShape, ACL_FLOAT, gTensor)) &&
        (betaFp32 ? CreateTensor(betaFp32Values, gateShape, ACL_FLOAT, betaTensor)
                  : CreateTensor(betaBf16, gateShape, ACL_BF16, betaTensor)) &&
        CreateTensor(a, aShape, mainDtype, aTensor) &&
        CreateTensor(dO, valueShape, mainDtype, dOTensor) &&
        CreateTensor(w, wShape, mainDtype, wTensor) &&
        CreateTensor(u, valueShape, mainDtype, uTensor) &&
        CreateTensor(dv, valueShape, mainDtype, dvTensor);
    if (!created) {
        std::cerr << "Tensor allocation failed\n";
        return 2;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnChunkGdnBwdIntraGetWorkspaceSize(
        qTensor.tensor, kTensor.tensor, vTensor.tensor, gTensor.tensor, betaTensor.tensor,
        aTensor.tensor, dOTensor.tensor, nullptr, nullptr, SCALE, BT, useExp2, 2,
        wTensor.tensor, uTensor.tensor, dvTensor.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::cerr << "GetWorkspaceSize failed: " << ret << '\n';
        return 3;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "Workspace allocation failed\n";
        return 4;
    }
    std::vector<float> expectedW(w.size(), 0.0F);
    std::vector<float> expectedU(u.size(), 0.0F);
    std::vector<float> expectedDv(dv.size(), 0.0F);
    for (int64_t h = 0; h < HV; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            const int64_t gateOffset = h * T + t;
            const float betaValue = betaFp32 ? betaFp32Values[gateOffset]
                                               : Bf16ToFloat(betaBf16[gateOffset]);
            const auto gate = [useExp2](float value) {
                return useExp2 ? std::exp2(value) : std::exp(value);
            };
            expectedW[(h * T + t) * K] = 0.5F * betaValue * gate(g[gateOffset]);
            expectedU[(h * T + t) * V] =
                (0.25F + static_cast<float>(h % 32) * 0.0625F) * betaValue;
            expectedU[(h * T + t) * V + 1] = 0.125F * betaValue;
            float dvValue = 0.0F;
            const int64_t chunkEnd = std::min(T, (t / BT + 1) * BT);
            for (int64_t s = t; s < chunkEnd; ++s) {
                dvValue += 0.015625F * gate(std::min(g[h * T + s] - g[gateOffset], 0.0F));
            }
            if (identityD) {
                dvValue = 0.125F;
            }
            expectedDv[(h * T + t) * V] = dvValue;
        }
    }
    float wError = 0.0F;
    float uError = 0.0F;
    float dvError = 0.0F;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (repeat > 0) {
            uint64_t currentWorkspaceSize = 0;
            executor = nullptr;
            ret = aclnnChunkGdnBwdIntraGetWorkspaceSize(
                qTensor.tensor, kTensor.tensor, vTensor.tensor, gTensor.tensor, betaTensor.tensor,
                aTensor.tensor, dOTensor.tensor, nullptr, nullptr, SCALE, BT, useExp2, 2,
                wTensor.tensor, uTensor.tensor, dvTensor.tensor, &currentWorkspaceSize, &executor);
            if (ret != ACL_SUCCESS || currentWorkspaceSize != workspaceSize) {
                std::cerr << "Repeated GetWorkspaceSize failed: " << ret << '\n';
                return 3;
            }
        }
        ret = aclnnChunkGdnBwdIntra(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            std::cerr << "Kernel execution failed at repeat " << repeat << ": " << ret << '\n';
            return 5;
        }
        if (!CopyMainOutput(wTensor, w) || !CopyMainOutput(uTensor, u) ||
            !CopyMainOutput(dvTensor, dv)) {
            std::cerr << "Output copy failed at repeat " << repeat << '\n';
            return 6;
        }
        wError = std::max(wError, MaxAbsError(w, expectedW, mainFp16));
        uError = std::max(uError, MaxAbsError(u, expectedU, mainFp16));
        dvError = std::max(dvError, MaxAbsError(dv, expectedDv, mainFp16));
    }
    std::cout << "use_exp2=" << useExp2 << " group_size=" << groupSize
              << " gate_dtype=" << (gateBf16 ? "bf16" : "fp32")
              << " beta_dtype=" << (betaFp32 ? "fp32" : "bf16")
              << " main_dtype=" << (mainFp16 ? "fp16" : "bf16")
              << " repeats=" << repeats << " T=" << T << " HV=" << HV
              << " workspace_bytes=" << workspaceSize
              << " w_max_abs=" << wError
              << " u_max_abs=" << uError << " dv_max_abs=" << dvError << '\n';

    const bool detailAll = argc >= 7 && std::string(argv[6]) == "detail-all";
    if (argc >= 7 && (std::string(argv[6]) == "detail" || detailAll)) {
        const int64_t chunkCount = (T + BT - 1) / BT;
        for (int64_t chunk = 0; chunk < chunkCount; ++chunk) {
            float chunkWError = 0.0F;
            float chunkUError = 0.0F;
            float chunkDvError = 0.0F;
            const int64_t tokenBegin = chunk * BT;
            const int64_t tokenEnd = std::min(T, tokenBegin + BT);
            for (int64_t h = 0; h < HV; ++h) {
                for (int64_t t = tokenBegin; t < tokenEnd; ++t) {
                    for (int64_t d = 0; d < K; ++d) {
                        const size_t offset = static_cast<size_t>((h * T + t) * K + d);
                        chunkWError = std::max(
                            chunkWError,
                            std::abs(MainToFloat(w[offset], mainFp16) - expectedW[offset]));
                    }
                    for (int64_t d = 0; d < V; ++d) {
                        const size_t offset = static_cast<size_t>((h * T + t) * V + d);
                        chunkUError = std::max(
                            chunkUError,
                            std::abs(MainToFloat(u[offset], mainFp16) - expectedU[offset]));
                        chunkDvError = std::max(
                            chunkDvError,
                            std::abs(MainToFloat(dv[offset], mainFp16) - expectedDv[offset]));
                    }
                }
            }
            if (chunkWError > 0.02F || chunkUError > 0.01F || chunkDvError > 0.05F) {
                std::cout << "bad_chunk=" << chunk << " w_max_abs=" << chunkWError
                          << " u_max_abs=" << chunkUError
                          << " dv_max_abs=" << chunkDvError << '\n';
                for (int64_t h = 0; h < HV; ++h) {
                    float headWError = 0.0F;
                    float headUError = 0.0F;
                    float headDvError = 0.0F;
                    float halfDvError[2] = {0.0F, 0.0F};
                    size_t headWOffset = 0;
                    size_t headUOffset = 0;
                    size_t headDvOffset = 0;
                    for (int64_t t = tokenBegin; t < tokenEnd; ++t) {
                        for (int64_t d = 0; d < K; ++d) {
                            const size_t offset = static_cast<size_t>((h * T + t) * K + d);
                            const float error =
                                std::abs(MainToFloat(w[offset], mainFp16) - expectedW[offset]);
                            if (error > headWError) {
                                headWError = error;
                                headWOffset = offset;
                            }
                        }
                        for (int64_t d = 0; d < V; ++d) {
                            const size_t offset = static_cast<size_t>((h * T + t) * V + d);
                            const float uError =
                                std::abs(MainToFloat(u[offset], mainFp16) - expectedU[offset]);
                            if (uError > headUError) {
                                headUError = uError;
                                headUOffset = offset;
                            }
                            const float error =
                                std::abs(MainToFloat(dv[offset], mainFp16) - expectedDv[offset]);
                            const int64_t half = (t - tokenBegin) < BT / 2 ? 0 : 1;
                            halfDvError[half] = std::max(halfDvError[half], error);
                            if (error > headDvError) {
                                headDvError = error;
                                headDvOffset = offset;
                            }
                        }
                    }
                    if (detailAll || headWError > 0.02F || headUError > 0.01F ||
                        headDvError > 0.05F) {
                        std::cout << "  bad_head=" << h
                                  << " w_max_abs=" << headWError
                                  << " w_actual=" << MainToFloat(w[headWOffset], mainFp16)
                                  << " w_expected=" << expectedW[headWOffset]
                                  << " u_max_abs=" << headUError
                                  << " u_actual=" << MainToFloat(u[headUOffset], mainFp16)
                                  << " u_expected=" << expectedU[headUOffset]
                                  << " dv_max_abs=" << headDvError
                                  << " dv_half0=" << halfDvError[0]
                                  << " dv_half1=" << halfDvError[1]
                                  << " dv_actual=" << MainToFloat(dv[headDvOffset], mainFp16)
                                  << " dv_expected=" << expectedDv[headDvOffset] << '\n';
                    }
                }
            }
        }
    }

    if (detailAll && T == BT && workspace != nullptr) {
        const int64_t cg = groupSize == 3 ? 3 : 4;
        const uint64_t slotCount = static_cast<uint64_t>((HV + cg - 1) / cg * cg);
        const uint64_t matrixElems = static_cast<uint64_t>(BT * BT);
        const uint64_t recordElems = 3 * matrixElems;
        const uint64_t userWorkspaceBytes = slotCount * recordElems * sizeof(uint16_t);
        if (workspaceSize >= userWorkspaceBytes) {
            const uint64_t tailBytes = workspaceSize;
            std::vector<uint16_t> workspaceTail(tailBytes / sizeof(uint16_t));
            auto *workspaceTailDevice = static_cast<uint8_t *>(workspace) +
                (workspaceSize - tailBytes);
            if (aclrtMemcpy(workspaceTail.data(), tailBytes, workspaceTailDevice,
                            tailBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
                const int64_t nominalBase = static_cast<int64_t>(tailBytes - userWorkspaceBytes);
                int64_t bestBase = nominalBase;
                float bestError = 1.0e30F;
                for (int64_t candidate = 0;
                     candidate + static_cast<int64_t>(userWorkspaceBytes) <=
                         static_cast<int64_t>(tailBytes);
                     candidate += 512) {
                    float candidateError = 0.0F;
                    const uint64_t dBase = static_cast<uint64_t>(candidate) /
                        sizeof(uint16_t) + 2 * matrixElems;
                    for (int64_t t = 0; t < BT; ++t) {
                        for (int64_t s = 0; s < BT; ++s) {
                            const float expected = t <= s
                                ? 0.125F * (useExp2
                                    ? std::exp2(std::min(g[s] - g[t], 0.0F))
                                    : std::exp(std::min(g[s] - g[t], 0.0F)))
                                : 0.0F;
                            const float actual = MainToFloat(
                                workspaceTail[dBase + t * BT + s], mainFp16);
                            candidateError = std::max(
                                candidateError, std::abs(actual - expected));
                        }
                    }
                    if (candidateError < bestError) {
                        bestError = candidateError;
                        bestBase = candidate;
                    }
                }
                std::cout << "  d_workspace_base_delta=" << bestBase - nominalBase
                          << " head0_max_abs=" << bestError << '\n';
                for (int64_t h = 0; h < HV; ++h) {
                    float dWorkspaceError = 0.0F;
                    const uint64_t dBase = static_cast<uint64_t>(bestBase) /
                        sizeof(uint16_t) + static_cast<uint64_t>(h) * recordElems +
                        2 * matrixElems;
                    for (int64_t t = 0; t < BT; ++t) {
                        for (int64_t s = 0; s < BT; ++s) {
                            const float gateColumn = gateBf16
                                ? Bf16ToFloat(gBf16[h * T + s]) : g[h * T + s];
                            const float gateRow = gateBf16
                                ? Bf16ToFloat(gBf16[h * T + t]) : g[h * T + t];
                            const float expected = t <= s
                                ? 0.125F * (useExp2
                                    ? std::exp2(std::min(gateColumn - gateRow, 0.0F))
                                    : std::exp(std::min(gateColumn - gateRow, 0.0F)))
                                : 0.0F;
                            const float actual = MainToFloat(
                                workspaceTail[dBase + t * BT + s], mainFp16);
                            dWorkspaceError = std::max(
                                dWorkspaceError, std::abs(actual - expected));
                        }
                    }
                    std::cout << "  d_workspace_head=" << h
                              << " max_abs=" << dWorkspaceError << '\n';
                }
            }
        }
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    for (TensorHandle *handle : {&qTensor, &kTensor, &vTensor, &gTensor, &betaTensor,
                                 &aTensor, &dOTensor, &wTensor, &uTensor, &dvTensor}) {
        DestroyTensor(*handle);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return (wError <= 0.02F && uError <= 0.01F && dvError <= 0.05F) ? 0 : 7;
}
