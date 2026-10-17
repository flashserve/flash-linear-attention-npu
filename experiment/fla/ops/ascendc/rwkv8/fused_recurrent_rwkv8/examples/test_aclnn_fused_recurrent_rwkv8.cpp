/*!
 * \file test_aclnn_fused_recurrent_rwkv8.cpp
 * \brief 自包含 e2e 样例：内置确定性输入 + C++ CPU golden 递推 + rel-RMSE 对拍。
 *
 * 基础 case：常规 / 带初态+scale / T 非 16 倍数 / 单 token decode（fp32，与仓外
 * 直调工程 scripts/gen_data.py 的 CASES 一致）+ bf16 常规 / fp16 带初态。
 * 训练预埋 case：s/sa 全开（T=64 整除 / T=33 floor / T=1 零快照边界）、
 * reverse=true、bf16 下 s/sa 仍 fp32。
 * fp32 case 阈值 rel-RMSE ≤ 0.002；低精度 case 的 golden 跑在"量化→反量化"的
 * 输入上，o 阈值放宽（bf16 ≤ 0.02 / fp16 ≤ 0.01），s/sa 恒 fp32
 * 仍 ≤ 0.002。任一 case FAIL 即退出码非 0。
 *
 * 可选 dump：设置环境变量 RWKV8_DUMP_DIR 时，把 fp32 case 的
 * 输入/NPU 输出落盘（raw fp32 .bin + meta.txt），供
 * check_npu_vs_reference.py 用 Python CPU golden 本体二次对拍。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_recurrent_rwkv8.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

namespace {

constexpr double REL_RMSE_TOL = 0.002;
constexpr int64_t DEFAULT_CHUNK_LEN = 16;   // 对齐官方 wkv7_cuda.cu backward 的 chunk 重建粒度

// io dtype：0=fp32（阈值 0.002），1=bf16 / 2=fp16（o 阈值按输入量化噪声放宽）
enum IoDtype : int { IO_FP32 = 0, IO_BF16 = 1, IO_FP16 = 2 };

struct Case {
    int64_t B, T, H, K, V;   // K = q/w/k/z/b 侧 head dim；V = v/o/sa 侧（K==V 为特例）
    float scale;
    bool hasInit;
    const char *desc;
    int ioDtype;
    double oTol;
    bool reverse;
    bool outputS;
    bool outputSa;
    int64_t chunkLen;        // s 快照间隔（默认 16；非 16 值与官方 backward 不兼容）
};

// (B, T, H, K, V, scale, hasInit, desc, ioDtype, oTol, reverse, outputS, outputSa, chunkLen)
// 覆盖：常规 / 带初态+scale / T非16倍数 / 单token decode / bf16 / fp16
//       / s+sa 全开（整除、floor、零快照）/ reverse / bf16+s/sa / K≠V（fp32 + bf16）
//       / 非默认 chunkLen
const Case CASES[] = {
    {2, 64, 4, 64, 64, 1.0f, false, "regular", IO_FP32, REL_RMSE_TOL, false, false, false, DEFAULT_CHUNK_LEN},
    {1, 16, 2, 64, 64, 0.125f, true, "init+scale", IO_FP32, REL_RMSE_TOL, false, false, false, DEFAULT_CHUNK_LEN},
    {2, 33, 4, 64, 64, 1.0f, true, "T=33 unaligned", IO_FP32, REL_RMSE_TOL, false, false, false, DEFAULT_CHUNK_LEN},
    {1, 1, 1, 64, 64, 1.0f, false, "decode T=1", IO_FP32, REL_RMSE_TOL, false, false, false, DEFAULT_CHUNK_LEN},
    {2, 64, 4, 64, 64, 1.0f, false, "regular bf16", IO_BF16, 0.02, false, false, false, DEFAULT_CHUNK_LEN},
    {1, 16, 2, 64, 64, 0.125f, true, "init+scale fp16", IO_FP16, 0.01, false, false, false, DEFAULT_CHUNK_LEN},
    {2, 64, 4, 64, 64, 1.0f, true, "s+sa full T=64", IO_FP32, REL_RMSE_TOL, false, true, true, DEFAULT_CHUNK_LEN},
    {2, 33, 4, 64, 64, 1.0f, true, "s+sa floor T=33", IO_FP32, REL_RMSE_TOL, false, true, true, DEFAULT_CHUNK_LEN},
    {1, 1, 1, 64, 64, 1.0f, false, "s+sa T=1 zero-chunk", IO_FP32, REL_RMSE_TOL, false, true, true, DEFAULT_CHUNK_LEN},
    {1, 16, 2, 64, 64, 0.125f, true, "reverse + s+sa", IO_FP32, REL_RMSE_TOL, true, true, true, DEFAULT_CHUNK_LEN},
    {2, 64, 4, 64, 64, 1.0f, false, "bf16 + s+sa", IO_BF16, 0.02, false, true, true, DEFAULT_CHUNK_LEN},
    {2, 64, 4, 64, 32, 1.0f, true, "K64 V32 + s+sa", IO_FP32, REL_RMSE_TOL, false, true, true, DEFAULT_CHUNK_LEN},
    {1, 16, 2, 64, 32, 0.125f, true, "K64 V32 bf16", IO_BF16, 0.02, false, false, false, DEFAULT_CHUNK_LEN},
    {2, 64, 4, 64, 64, 1.0f, true, "s+sa chunkLen=8", IO_FP32, REL_RMSE_TOL, false, true, true, 8},
    {2, 33, 4, 64, 64, 1.0f, true, "s+sa chunkLen=8 floor", IO_FP32, REL_RMSE_TOL, false, true, true, 8},
    {2, 64, 4, 64, 64, 1.0f, true, "s+sa chunkLen=T", IO_FP32, REL_RMSE_TOL, false, true, true, 64},
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// 确定性伪随机（固定 seed 的 LCG），保证样例可复现
struct Lcg {
    explicit Lcg(uint64_t seed) : state(seed) {}
    float NextUniform()   // [0,1)
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 33) & 0x7FFFFFFF) / 2147483648.0f;
    }
    float NextNormalish() // 近标准正态：12 个均匀和 - 6
    {
        float s = 0.0f;
        for (int i = 0; i < 12; i++) {
            s += NextUniform();
        }
        return s - 6.0f;
    }
    uint64_t state;
};

// ---- fp32 <-> bf16/fp16 位级转换（RNE 最近舍入），仅用于样例造数据/读回 ----
uint16_t F32ToBf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += 0x7FFFU + ((u >> 16) & 1U);   // round-to-nearest-even
    return static_cast<uint16_t>(u >> 16);
}

float Bf16ToF32(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t F32ToF16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000U;
    const int exp = static_cast<int>((u >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mant = u & 0x7FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);              // 下溢归零（测试数据不会出现）
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);    // 上溢 inf（测试数据不会出现）
    }
    uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFU;
    if (rem > 0x1000U || (rem == 0x1000U && (h & 1U))) {
        h += 1U;                                         // RNE（进位自然传播到指数位）
    }
    return static_cast<uint16_t>(h);
}

float F16ToF32(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {                                          // subnormal 规格化
            exp = 1;
            while ((mant & 0x400U) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFU;
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000U | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

aclDataType ToAclDtype(int ioDtype)
{
    if (ioDtype == IO_BF16) {
        return aclDataType::ACL_BF16;
    }
    if (ioDtype == IO_FP16) {
        return aclDataType::ACL_FLOAT16;
    }
    return aclDataType::ACL_FLOAT;
}

// 量化：fp32 → 目标 dtype 的位模式（fp32 时原样返回位拷贝）
std::vector<uint16_t> Quantize(const std::vector<float> &src, int ioDtype)
{
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        out[i] = (ioDtype == IO_BF16) ? F32ToBf16(src[i]) : F32ToF16(src[i]);
    }
    return out;
}

// 反量化：位模式 → fp32（用于 golden 输入与 NPU 输出读回比对）
std::vector<float> Dequantize(const std::vector<uint16_t> &src, int ioDtype)
{
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        out[i] = (ioDtype == IO_BF16) ? Bf16ToF32(src[i]) : F16ToF32(src[i]);
    }
    return out;
}

// 与 gen_data.py make_inputs 同分布：q/k/v ~ N(0,1)；w ∈ [-2.1,-0.1]（保 decay∈(0,1)）；
// z = normalize(randn)·randn（沿 K 维）；b 同。K 侧张量 (B,H,T,K)，v (B,H,T,V)（BHTC）
void MakeInputs(const Case &c, uint64_t seed, std::vector<float> &q, std::vector<float> &w, std::vector<float> &k,
                std::vector<float> &v, std::vector<float> &z, std::vector<float> &b,
                std::vector<float> &initialState)
{
    const int64_t kSize = c.B * c.T * c.H * c.K;
    const int64_t vSize = c.B * c.T * c.H * c.V;
    Lcg rng(seed);
    auto fillNormalK = [&](std::vector<float> &x) {
        x.resize(kSize);
        for (auto &e : x) {
            e = rng.NextNormalish();
        }
    };
    fillNormalK(q);
    fillNormalK(k);
    v.resize(vSize);
    for (auto &e : v) {
        e = rng.NextNormalish();
    }

    w.resize(kSize);
    for (auto &e : w) {
        e = -rng.NextUniform() * 2.0f - 0.1f;
    }

    // z = normalize(g1) * g2（沿 K 维）
    std::vector<float> g1(kSize), g2(kSize);
    for (auto &e : g1) {
        e = rng.NextNormalish();
    }
    for (auto &e : g2) {
        e = rng.NextNormalish();
    }
    z.resize(kSize);
    for (int64_t row = 0; row < kSize / c.K; row++) {
        double norm = 0.0;
        for (int64_t i = 0; i < c.K; i++) {
            norm += static_cast<double>(g1[row * c.K + i]) * g1[row * c.K + i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-12) {
            norm = 1e-12;
        }
        for (int64_t i = 0; i < c.K; i++) {
            z[row * c.K + i] = static_cast<float>(g1[row * c.K + i] / norm) * g2[row * c.K + i];
        }
    }
    fillNormalK(b);

    if (c.hasInit) {
        initialState.resize(c.B * c.H * c.V * c.K);   // 接口朝向 (K,V)（= 内核 Sᵀ）
        for (auto &e : initialState) {
            e = rng.NextNormalish();
        }
    } else {
        initialState.clear();
    }
}

// CPU golden：RWKV 朝向 state (N,N)（行 = v/q 侧，列 = k/z 侧），fp32 递推
//   sa    = S @ z_t
//   S     = S * decay[None,:] + sa[:,None] * b[None,:] + v[:,None] * k[None,:]
//   o_t   = S @ (q_t * scale)     decay = exp(-exp(w))
// io 布局 BHTC = (B,H,T,C)：每 (b,h) 段连续，token t 步长 = C。
// 训练预埋：goldenSa (B,H,T,N)（state 更新前的 S@z）；goldenS (B,H,T//chunkLen,N,N)
// 为官方 CUDA 转置布局（快照 [j][i] = S[i][j]，每满 chunkLen token 一拍，floor 语义）；
// reverse=true 时按 t=T-1..0 倒序递推（快照槽位仍按 token 下标 t/chunkLen）
void CpuGolden(const Case &c, const std::vector<float> &q, const std::vector<float> &w, const std::vector<float> &k,
               const std::vector<float> &v, const std::vector<float> &z, const std::vector<float> &b,
               const std::vector<float> &initialState, std::vector<float> &o,
               std::vector<float> &goldenS, std::vector<float> &goldenSa)
{
    const int64_t K = c.K;
    const int64_t V = c.V;
    o.assign(c.B * c.T * c.H * V, 0.0f);
    const int64_t numChunks = c.T / c.chunkLen;
    if (c.outputS) {
        goldenS.assign(c.B * c.H * numChunks * K * V, 0.0f);
    } else {
        goldenS.clear();
    }
    if (c.outputSa) {
        goldenSa.assign(c.B * c.T * c.H * V, 0.0f);
    } else {
        goldenSa.clear();
    }

    std::vector<float> S(V * K), sa(V), decay(K);   // S 为 RWKV 朝向 (V,K)
    for (int64_t bh = 0; bh < c.B * c.H; bh++) {
        if (c.hasInit) {
            // 接口 (K,V) → golden 内部 (V,K) 转置载入
            const float *init = initialState.data() + bh * V * K;
            for (int64_t i = 0; i < V; i++) {
                for (int64_t j = 0; j < K; j++) {
                    S[i * K + j] = init[j * V + i];
                }
            }
        } else {
            std::fill(S.begin(), S.end(), 0.0f);
        }
        for (int64_t i = 0; i < c.T; i++) {
            const int64_t t = c.reverse ? (c.T - 1 - i) : i;
            const int64_t baseK = (bh * c.T + t) * K;   // BHTC：(b,h) 段连续，步长 K
            const int64_t baseV = (bh * c.T + t) * V;
            const float *wp = w.data() + baseK;
            const float *kp = k.data() + baseK;
            const float *zp = z.data() + baseK;
            const float *bp = b.data() + baseK;
            const float *qp = q.data() + baseK;
            const float *vp = v.data() + baseV;

            for (int64_t j = 0; j < K; j++) {
                decay[j] = std::exp(-std::exp(wp[j]));
            }
            for (int64_t i = 0; i < V; i++) {
                double acc = 0.0;
                for (int64_t j = 0; j < K; j++) {
                    acc += static_cast<double>(S[i * K + j]) * zp[j];
                }
                sa[i] = static_cast<float>(acc);
            }
            if (c.outputSa) {
                std::copy(sa.begin(), sa.end(), goldenSa.begin() + baseV);
            }
            for (int64_t i = 0; i < V; i++) {
                for (int64_t j = 0; j < K; j++) {
                    S[i * K + j] = S[i * K + j] * decay[j] + sa[i] * bp[j] + vp[i] * kp[j];
                }
            }
            float *op = o.data() + baseV;
            for (int64_t i = 0; i < V; i++) {
                double acc = 0.0;
                for (int64_t j = 0; j < K; j++) {
                    acc += static_cast<double>(S[i * K + j]) * (qp[j] * c.scale);
                }
                op[i] = static_cast<float>(acc);
            }
            if (c.outputS && (t + 1) % c.chunkLen == 0) {
                // 官方转置布局：snapshot[j][i] = S[i][j]，(K,V)
                float *sp = goldenS.data() + (bh * numChunks + t / c.chunkLen) * K * V;
                for (int64_t i = 0; i < V; i++) {
                    for (int64_t j = 0; j < K; j++) {
                        sp[j * V + i] = S[i * K + j];
                    }
                }
            }
        }
    }
}

double RelRmse(const std::vector<float> &out, const std::vector<float> &golden)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < out.size(); i++) {
        const double d = static_cast<double>(out[i]) - golden[i];
        num += d * d;
        den += static_cast<double>(golden[i]) * golden[i];
    }
    if (den == 0.0) {
        return num == 0.0 ? 0.0 : 1e30;
    }
    return std::sqrt(num / den);
}

int CreateAclTensorRaw(const void *hostData, size_t bytes, const std::vector<int64_t> &shape, void **deviceAddr,
                       aclDataType dataType, aclTensor **tensor)
{
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int CreateAclTensor(const std::vector<float> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    return CreateAclTensorRaw(hostData.data(), hostData.size() * sizeof(float), shape, deviceAddr, dataType, tensor);
}

int CreateAclTensor(const std::vector<uint16_t> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    return CreateAclTensorRaw(hostData.data(), hostData.size() * sizeof(uint16_t), shape, deviceAddr, dataType,
                              tensor);
}

void ReadBack(void *deviceAddr, std::vector<float> &hostData)
{
    aclrtMemcpy(hostData.data(), hostData.size() * sizeof(float), deviceAddr, hostData.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
}

void ReadBack(void *deviceAddr, std::vector<uint16_t> &hostData)
{
    aclrtMemcpy(hostData.data(), hostData.size() * sizeof(uint16_t), deviceAddr, hostData.size() * sizeof(uint16_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
}

// ---- 可选 dump（RWKV8_DUMP_DIR 非空时生效）：raw fp32 .bin + meta，供 Python golden 对拍 ----
std::string g_dumpDir;

void DumpTensor(const std::string &path, const std::vector<float> &data)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        LOG_PRINT("dump open failed: %s\n", path.c_str());
        return;
    }
    fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
}

void DumpCase(const char *tag, const Case &c, const std::vector<float> &q, const std::vector<float> &w,
              const std::vector<float> &k, const std::vector<float> &v, const std::vector<float> &z,
              const std::vector<float> &b, const std::vector<float> &initialState, const std::vector<float> &oHost,
              const std::vector<float> &sHost, const std::vector<float> &saHost)
{
    const std::string p = g_dumpDir + "/" + tag + "_";
    DumpTensor(p + "q.bin", q);
    DumpTensor(p + "w.bin", w);
    DumpTensor(p + "k.bin", k);
    DumpTensor(p + "v.bin", v);
    DumpTensor(p + "z.bin", z);
    DumpTensor(p + "b.bin", b);
    if (c.hasInit) {
        DumpTensor(p + "initial_state.bin", initialState);
    }
    DumpTensor(p + "o_npu.bin", oHost);
    if (c.outputS) {
        // T<chunkLen 时 sHost 含 1 元素占位，只落真实快照数（可能为 0）
        const int64_t sCount = c.B * c.H * (c.T / c.chunkLen) * c.K * c.V;
        FILE *sf = fopen((p + "s_npu.bin").c_str(), "wb");
        if (sf != nullptr) {
            fwrite(sHost.data(), sizeof(float), static_cast<size_t>(sCount), sf);
            fclose(sf);
        }
    }
    if (c.outputSa) {
        DumpTensor(p + "sa_npu.bin", saHost);
    }
    FILE *f = fopen((p + "meta.txt").c_str(), "w");
    if (f != nullptr) {
        fprintf(f, "B=%ld T=%ld H=%ld K=%ld V=%ld scale=%.9g hasInit=%d reverse=%d outputS=%d outputSa=%d chunkLen=%ld desc=%s\n",
                c.B, c.T, c.H, c.K, c.V, c.scale, c.hasInit ? 1 : 0, c.reverse ? 1 : 0, c.outputS ? 1 : 0,
                c.outputSa ? 1 : 0, c.chunkLen, c.desc);
        fclose(f);
    }
}

// 跑一个 case；dumpTag 非空且 RWKV8_DUMP_DIR 已设置时，把本 case 输入/NPU 输出
// 落盘（仅 fp32 case；低精度 case 的 dump 格式未定义，跳过）
int RunCase(const Case &c, uint64_t seed, const char *dumpTag, aclrtStream stream, bool &pass)
{
    pass = false;
    std::vector<float> q, w, k, v, z, b, initialState;
    MakeInputs(c, seed, q, w, k, v, z, b, initialState);

    const bool isFp32 = (c.ioDtype == IO_FP32);
    const aclDataType ioAcl = ToAclDtype(c.ioDtype);

    // 低精度路径：量化出 device 输入；golden 输入 = 量化后反量化（与 NPU 实际看到的数值一致）
    std::vector<uint16_t> qQ, wQ, kQ, vQ, zQ, bQ;
    if (!isFp32) {
        qQ = Quantize(q, c.ioDtype);
        wQ = Quantize(w, c.ioDtype);
        kQ = Quantize(k, c.ioDtype);
        vQ = Quantize(v, c.ioDtype);
        zQ = Quantize(z, c.ioDtype);
        bQ = Quantize(b, c.ioDtype);
        q = Dequantize(qQ, c.ioDtype);
        w = Dequantize(wQ, c.ioDtype);
        k = Dequantize(kQ, c.ioDtype);
        v = Dequantize(vQ, c.ioDtype);
        z = Dequantize(zQ, c.ioDtype);
        b = Dequantize(bQ, c.ioDtype);
    }

    std::vector<float> goldenO, goldenS, goldenSa;
    CpuGolden(c, q, w, k, v, z, b, initialState, goldenO, goldenS, goldenSa);

    const std::vector<int64_t> kShape = {c.B, c.H, c.T, c.K};       // q/w/k/z/b（BHTC）
    const std::vector<int64_t> vShape = {c.B, c.H, c.T, c.V};       // v/o/sa（BHTC）
    const std::vector<int64_t> sShape = {c.B, c.H, c.T / c.chunkLen, c.K, c.V};

    // dev 下标：0-5 = q/w/k/v/z/b，6 = initialState，7 = o，8 = s，9 = sa
    void *devAddr[10] = {nullptr};
    aclTensor *tensors[10] = {nullptr};
    const std::vector<float> *inputsF[6] = {&q, &w, &k, &v, &z, &b};
    const std::vector<uint16_t> *inputsQ[6] = {&qQ, &wQ, &kQ, &vQ, &zQ, &bQ};
    int ret = ACL_SUCCESS;
    for (int i = 0; i < 6; i++) {
        const auto &shape = (i == 3) ? vShape : kShape;   // 仅 v 走 V 维
        if (isFp32) {
            ret = CreateAclTensor(*inputsF[i], shape, &devAddr[i], aclDataType::ACL_FLOAT, &tensors[i]);
        } else {
            ret = CreateAclTensor(*inputsQ[i], shape, &devAddr[i], ioAcl, &tensors[i]);
        }
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor input %d failed\n", i); return ret);
    }
    if (c.hasInit) {
        // initial_state 恒 fp32，接口朝向 (B,H,K,V)（= 内核 Sᵀ）
        ret = CreateAclTensor(initialState, {c.B, c.H, c.K, c.V}, &devAddr[6], aclDataType::ACL_FLOAT, &tensors[6]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor initialState failed\n"); return ret);
    }
    std::vector<float> oHost(GetShapeSize(vShape), 0.0f);
    std::vector<uint16_t> oQHost(GetShapeSize(vShape), 0);
    if (isFp32) {
        ret = CreateAclTensor(oHost, vShape, &devAddr[7], aclDataType::ACL_FLOAT, &tensors[7]);
    } else {
        ret = CreateAclTensor(oQHost, vShape, &devAddr[7], ioAcl, &tensors[7]);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // s/sa 恒 fp32（开关打开时才创建用户输出张量）
    // 注意 T<chunkLen 时 s 是零尺寸张量（B,H,0,N,N）：底层 buffer 给 1 元素占位，
    // 避免 aclrtMalloc(0) 的未定义行为；比对时按 numChunks>0 门控
    const int64_t numChunks = c.T / c.chunkLen;
    std::vector<float> sHost(std::max<int64_t>(GetShapeSize(sShape), 1), 0.0f);
    std::vector<float> saHost(GetShapeSize(vShape), 0.0f);
    if (c.outputS) {
        ret = CreateAclTensorRaw(sHost.data(), sHost.size() * sizeof(float), sShape, &devAddr[8],
                                 aclDataType::ACL_FLOAT, &tensors[8]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    if (c.outputSa) {
        ret = CreateAclTensor(saHost, vShape, &devAddr[9], aclDataType::ACL_FLOAT, &tensors[9]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnFusedRecurrentRwkv8GetWorkspaceSize(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                                                   tensors[5], c.hasInit ? tensors[6] : nullptr, c.scale,
                                                   c.reverse, c.outputS, c.outputSa, c.chunkLen, tensors[7],
                                                   c.outputS ? tensors[8] : nullptr,
                                                   c.outputSa ? tensors[9] : nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnFusedRecurrentRwkv8GetWorkspaceSize failed. ERROR: %d, msg: %s\n", ret,
                        aclGetRecentErrMsg()); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFusedRecurrentRwkv8(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedRecurrentRwkv8 failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (isFp32) {
        ReadBack(devAddr[7], oHost);
    } else {
        ReadBack(devAddr[7], oQHost);
        oHost = Dequantize(oQHost, c.ioDtype);
    }
    if (c.outputS) {
        ReadBack(devAddr[8], sHost);
    }
    if (c.outputSa) {
        ReadBack(devAddr[9], saHost);
    }
    if (dumpTag != nullptr && !g_dumpDir.empty() && isFp32) {
        DumpCase(dumpTag, c, q, w, k, v, z, b, initialState, oHost, sHost, saHost);
    }
    const double oErr = RelRmse(oHost, goldenO);
    double chunkErr = 0.0;
    if (c.outputS && numChunks > 0) {
        chunkErr = RelRmse(sHost, goldenS);   // 零快照时两边皆空，恒等视为 0
    }
    double saErr = 0.0;
    if (c.outputSa) {
        saErr = RelRmse(saHost, goldenSa);
    }
    pass = (oErr <= c.oTol) && (!c.outputS || chunkErr <= REL_RMSE_TOL) && (!c.outputSa || saErr <= REL_RMSE_TOL);
    LOG_PRINT("case B%ld T%ld H%ld K%ld V%ld scale=%g hasInit=%d rev=%d s=%d sa=%d (%s): o=%.3e",
              c.B, c.T, c.H, c.K, c.V, c.scale, c.hasInit ? 1 : 0, c.reverse ? 1 : 0, c.outputS ? 1 : 0,
              c.outputSa ? 1 : 0, c.desc, oErr);
    if (c.outputS) {
        LOG_PRINT(" s=%.3e", chunkErr);
    }
    if (c.outputSa) {
        LOG_PRINT(" sa=%.3e", saErr);
    }
    LOG_PRINT(" rel-RMSE, %s\n", pass ? "PASS" : "FAIL");

    for (int i = 0; i < 10; i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(tensors[i]);
        }
        if (devAddr[i] != nullptr) {
            aclrtFree(devAddr[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return pass ? 0 : 1;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    const char *dumpEnv = getenv("RWKV8_DUMP_DIR");
    if (dumpEnv != nullptr) {
        g_dumpDir = dumpEnv;
    }

    int failed = 0;
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        bool pass = false;
        const std::string tag = "case" + std::to_string(i);
        RunCase(CASES[i], 42 + i, tag.c_str(), stream, pass);
        if (!pass) {
            failed++;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failed != 0) {
        LOG_PRINT("%d case(s) FAILED\n", failed);
        return 1;
    }
    LOG_PRINT("all cases PASS\n");
    return 0;
}
