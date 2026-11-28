#pragma once
#include "projection_types.h"

namespace kda {
// Keep the MX mixed kernel's low UB scratch clear of projection tiles and live state.
constexpr uint32_t kMxUbOffset = 8 * 1024;
struct RecurrenceArgs {
    using Address = GM_ADDR;
    Address convW, conv, checkpoint, u, key, decay, bias, alphaLog, gamma;
    Address indices, accepted, prefix, outWeight, output;
    uint64_t convStride, checkpointStride, uStride, keyStride, decayStride;
    GM_ADDR plan;
    uint32_t batch, sequence, capacity, prefix64, convLayout, crosses, staged;
    float scale, lower, eps;
};

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
namespace vec {
using namespace AscendC::MicroAPI;
using F = RegTensor<float>;
constexpr CastTrait widen = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait narrow = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
__simd_callee__ inline void readBf(F &x, __ubuf__ Bf *src, MaskReg &m)
{
    RegTensor<Bf> bits;
    LoadAlign<Bf, LoadDist::DIST_UNPACK_B16>(bits, src);
    Cast<float, Bf, widen>(x, bits, m);
}
__simd_callee__ inline void logistic(F &out, F &x, MaskReg &m)
{
    F z, one;
    Muls(z, x, -1.f, m);
    Exp(z, z, m);
    Adds(z, z, 1.f, m);
    Duplicate(one, 1.f, m);
    Div(out, one, z, m);
}
__simd_vf__ inline void convolve(__ubuf__ Bf *raw, __ubuf__ Bf *history, __ubuf__ Bf *w, __ubuf__ float *out,
                                 uint16_t oldest, bool normalize, float scale)
{
    auto m = CreateMask<float, MaskPattern::ALL>();
    F result[2];
#pragma unroll
    for (uint16_t part = 0; part < 2; ++part) {
        uint16_t i = 64 * part;
        F x, weight, acc, act;
        Duplicate(acc, 0.f, m);
        for (uint16_t tap = 0; tap < 4; ++tap) {
            if (tap == 3) {
                readBf(x, raw + i, m);
            } else {
                readBf(x, history + ((oldest + tap) % 3) * 384 + i, m);
            }
            readBf(weight, w + tap * 128 + i, m);
            F term;
            Mul(term, x, weight, m);
            Add(acc, acc, term, m);
        }
        logistic(act, acc, m);
        Mul(act, acc, act, m);
        RegTensor<Bf> rounded;
        Cast<Bf, float, narrow>(rounded, act, m);
        Cast<float, Bf, widen>(result[part], rounded, m);
        RegTensor<Bf> rawBits;
        LoadAlign<Bf, LoadDist::DIST_UNPACK_B16>(rawBits, raw + i);
        StoreAlign<Bf, StoreDist::DIST_PACK_B32>(history + oldest * 384 + i, rawBits, m);
    }
    if (normalize) {
        F a, b, sum, inv;
        Mul(a, result[0], result[0], m);
        Mul(b, result[1], result[1], m);
        Add(a, a, b, m);
        ReduceSum(sum, a, m);
        Duplicate<float, HighLowPart::LOWEST>(b, sum, m);
        Adds(b, b, 1.e-6f, m);
        Sqrt(b, b, m);
        Duplicate(a, scale, m);
        Div(inv, a, b, m);
        Mul(result[0], result[0], inv, m);
        Mul(result[1], result[1], inv, m);
    }
    StoreAlign(out, result[0], m);
    StoreAlign(out + 64, result[1], m);
}
// A complete short sequence has no outgoing hidden-state dependency.
template <bool RetainHistory = false>
__simd_callee__ inline void shortConvolve(__ubuf__ Bf *raw, __ubuf__ Bf *history, __ubuf__ Bf *weights,
                                          __ubuf__ float *out, uint16_t count, bool normalize, float scale,
                                          uint16_t oldest = 0, uint16_t historyStride = 384)
{
    auto m = CreateMask<float, MaskPattern::ALL>();
    F w0[2], w1[2], w2[2], w3[2], h0[2], h1[2], h2[2];
#pragma unroll
    for (uint16_t j = 0; j < 2; ++j) {
        uint16_t i = j * 64;
        readBf(w0[j], weights + i, m);
        readBf(w1[j], weights + 128 + i, m);
        readBf(w2[j], weights + 256 + i, m);
        readBf(w3[j], weights + 384 + i, m);
        readBf(h0[j], history + (RetainHistory ? oldest * 384 : 0) + i, m);
        readBf(h1[j], history + (RetainHistory ? (oldest + 1) % 3 * 384 : historyStride) + i, m);
        readBf(h2[j], history + (RetainHistory ? (oldest + 2) % 3 * 384 : 2 * historyStride) + i, m);
    }
    for (uint16_t t = 0; t < count; ++t) {
        F result[2];
#pragma unroll
        for (uint16_t j = 0; j < 2; ++j) {
            F x, acc, term, act;
            readBf(x, raw + t * 128 + j * 64, m);
            Mul(acc, h0[j], w0[j], m);
            Mul(term, h1[j], w1[j], m);
            Add(acc, acc, term, m);
            Mul(term, h2[j], w2[j], m);
            Add(acc, acc, term, m);
            Mul(term, x, w3[j], m);
            Add(acc, acc, term, m);
            logistic(act, acc, m);
            Mul(act, acc, act, m);
            RegTensor<Bf> rounded;
            Cast<Bf, float, narrow>(rounded, act, m);
            Cast<float, Bf, widen>(result[j], rounded, m);
            h0[j] = h1[j];
            h1[j] = h2[j];
            h2[j] = x;
        }
        if (normalize) {
            F a, b, sum, inv;
            Mul(a, result[0], result[0], m);
            Mul(b, result[1], result[1], m);
            Add(a, a, b, m);
            ReduceSum(sum, a, m);
            Duplicate<float, HighLowPart::LOWEST>(b, sum, m);
            Adds(b, b, 1.e-6f, m);
            Sqrt(b, b, m);
            Duplicate(a, scale, m);
            Div(inv, a, b, m);
            Mul(result[0], result[0], inv, m);
            Mul(result[1], result[1], inv, m);
        }
        StoreAlign(out + t * 384, result[0], m);
        StoreAlign(out + t * 384 + 64, result[1], m);
    }
    if constexpr (RetainHistory) {
#pragma unroll
        for (uint16_t i = 0; i < 2; ++i) {
            RegTensor<Bf> a, b, c;
            Cast<Bf, float, narrow>(a, h0[i], m);
            Cast<Bf, float, narrow>(b, h1[i], m);
            Cast<Bf, float, narrow>(c, h2[i], m);
            StoreAlign<Bf, StoreDist::DIST_PACK_B32>(history + (oldest + count) % 3 * 384 + i * 64, a, m);
            StoreAlign<Bf, StoreDist::DIST_PACK_B32>(history + (oldest + count + 1) % 3 * 384 + i * 64, b, m);
            StoreAlign<Bf, StoreDist::DIST_PACK_B32>(history + (oldest + count + 2) % 3 * 384 + i * 64, c, m);
        }
    }
}
template <bool RetainHistory = false, bool PreviousRaw = false>
__simd_vf__ inline void shortConvolveAll(__ubuf__ Bf *raw, __ubuf__ Bf *history, __ubuf__ Bf *weights,
                                         __ubuf__ float *out, uint16_t valid, float scale, uint16_t oldest = 0)
{
    shortConvolve<RetainHistory>(raw, PreviousRaw ? raw - 384 : history, weights, out, valid, true, scale, oldest,
                                 PreviousRaw ? 128 : 384);
    shortConvolve<RetainHistory>(raw + 2048, PreviousRaw ? raw + 2048 - 384 : history + 128, weights + 512, out + 128,
                                 valid, true, 1.f, oldest, PreviousRaw ? 128 : 384);
    shortConvolve<RetainHistory>(raw + 4096, PreviousRaw ? raw + 4096 - 384 : history + 256, weights + 1024, out + 256,
                                 valid, false, 1.f, oldest, PreviousRaw ? 128 : 384);
}
__simd_vf__ inline void step(__ubuf__ float *state, __ubuf__ float *qkv, __ubuf__ Bf *dr, __ubuf__ Bf *beta,
                             __ubuf__ float *bias, float aLog, float lower, __ubuf__ float *u, __ubuf__ float *d,
                             __ubuf__ float *y)
{
    auto m = CreateMask<float, MaskPattern::ALL>();
    F q[2], k[2], dec[2], alpha, br, betaF;
    Duplicate(alpha, aLog, m);
    Exp(alpha, alpha, m);
    RegTensor<Bf> betaBits;
    LoadAlign<Bf, LoadDist::DIST_BRC_B16>(betaBits, beta);
    Cast<float, Bf, widen>(br, betaBits, m);
    logistic(betaF, br, m);
#pragma unroll
    for (uint16_t j = 0; j < 2; ++j) {
        F x, b;
        LoadAlign(q[j], qkv + j * 64);
        LoadAlign(k[j], qkv + 128 + j * 64);
        readBf(x, dr + j * 64, m);
        LoadAlign(b, bias + j * 64);
        Add(x, x, b, m);
        Mul(x, x, alpha, m);
        logistic(b, x, m);
        Muls(b, b, lower, m);
        Exp(dec[j], b, m);
        StoreAlign(d + j * 64, dec[j], m);
    }
#pragma unroll 4
    for (uint16_t row = 0; row < 128; ++row) {
        F h0, h1, p0, p1, total, dot, v, corr, bc;
        LoadAlign(h0, state + row * 128);
        LoadAlign(h1, state + row * 128 + 64);
        Mul(h0, h0, dec[0], m);
        Mul(h1, h1, dec[1], m);
        Mul(p0, h0, k[0], m);
        Mul(p1, h1, k[1], m);
        Add(total, p0, p1, m);
        ReduceSum(dot, total, m);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(v, qkv + 256 + row);
        Sub(corr, v, dot, m);
        Mul(corr, corr, betaF, m);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(u + row, corr, m);
        Duplicate<float, HighLowPart::LOWEST>(bc, corr, m);
        Mul(p0, bc, k[0], m);
        Mul(p1, bc, k[1], m);
        Add(h0, h0, p0, m);
        Add(h1, h1, p1, m);
        StoreAlign(state + row * 128, h0, m);
        StoreAlign(state + row * 128 + 64, h1, m);
        Mul(p0, h0, q[0], m);
        Mul(p1, h1, q[1], m);
        Add(total, p0, p1, m);
        ReduceSum(dot, total, m);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(y + row, dot, m);
    }
}
// Keep the FP32 state dependency between tokens. Value rows are independent within one token.
template <uint16_t ValueRows, uint16_t Count = 0>
__simd_vf__ inline void advanceWindow(__ubuf__ float *state, __ubuf__ float *qkv, __ubuf__ float *dec,
                                      __ubuf__ float *beta, __ubuf__ float *updates, __ubuf__ float *outputs,
                                      uint16_t count, uint16_t valueBegin = 0)
{
    auto m = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t t = 0; t < (Count ? Count : count); ++t) {
        F d0, d1, k0, k1, q0, q1, gain;
        auto token = qkv + t * 384;
        LoadAlign(d0, dec + t * 128);
        LoadAlign(d1, dec + t * 128 + 64);
        LoadAlign(k0, token + 128);
        LoadAlign(k1, token + 192);
        LoadAlign(q0, token);
        LoadAlign(q1, token + 64);
        LoadAlign(gain, beta + t * 64);
#pragma unroll 8
        for (uint16_t row = 0; row < ValueRows; ++row) {
            F h0, h1, a, b, dot, v, update, correction;
            LoadAlign(h0, state + row * 128);
            LoadAlign(h1, state + row * 128 + 64);
            Mul(h0, h0, d0, m);
            Mul(h1, h1, d1, m);
            Mul(a, h0, k0, m);
            Mul(b, h1, k1, m);
            Add(a, a, b, m);
            ReduceSum(dot, a, m);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v, token + 256 + valueBegin + row);
            Sub(update, v, dot, m);
            Mul(update, update, gain, m);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(updates + t * ValueRows + row, update, m);
            Duplicate<float, HighLowPart::LOWEST>(correction, update, m);
            Mul(a, correction, k0, m);
            Mul(b, correction, k1, m);
            Add(h0, h0, a, m);
            Add(h1, h1, b, m);
            StoreAlign(state + row * 128, h0, m);
            StoreAlign(state + row * 128 + 64, h1, m);
            Mul(a, h0, q0, m);
            Mul(b, h1, q1, m);
            Add(a, a, b, m);
            ReduceSum(dot, a, m);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outputs + t * ValueRows + row, dot, m);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}
// Prepare token-local decay and beta before advancing the FP32 state.
__simd_vf__ inline void shortDecay(__ubuf__ Bf *raw, __ubuf__ Bf *betaRaw, __ubuf__ float *bias, float aLog,
                                   float lower, __ubuf__ float *dec, __ubuf__ float *beta, uint16_t valid)
{
    auto m = CreateMask<float, MaskPattern::ALL>();
    F alpha, v, betaF;
    Duplicate(alpha, aLog, m);
    Exp(alpha, alpha, m);
    for (uint16_t j = 0; j < valid; ++j) {
        RegTensor<Bf> bits;
        LoadAlign<Bf, LoadDist::DIST_BRC_B16>(bits, betaRaw + j * 16);
        Cast<float, Bf, widen>(v, bits, m);
        logistic(betaF, v, m);
        StoreAlign(beta + j * 64, betaF, m);
        for (uint16_t i = 0; i < 2; ++i) {
            F x, b;
            readBf(x, raw + j * 128 + i * 64, m);
            LoadAlign(b, bias + i * 64);
            Add(x, x, b, m);
            Mul(x, x, alpha, m);
            logistic(b, x, m);
            Muls(b, b, lower, m);
            Exp(x, b, m);
            StoreAlign(dec + j * 128 + 64 * i, x, m);
        }
    }
}
__simd_vf__ inline void transposeState(__ubuf__ float *dst, __ubuf__ float *src)
{
    auto mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<int32_t> index;
    Arange(index, int32_t(0));
    Muls(index, index, int32_t(128), mask);
    for (uint16_t col = 0; col < 128; ++col) {
        F lo, hi;
        Gather(lo, src + col, reinterpret_cast<RegTensor<uint32_t> &>(index), mask);
        Gather(hi, src + 64 * 128 + col, reinterpret_cast<RegTensor<uint32_t> &>(index), mask);
        StoreAlign(dst + col * 128, lo, mask);
        StoreAlign(dst + col * 128 + 64, hi, mask);
    }
}
// Each lane owns an output row. Accumulate its dot product across the 128 contiguous state columns.
__simd_vf__ inline void stepColumnMajor(__ubuf__ float *state, __ubuf__ float *qkv, __ubuf__ float *decay,
                                        __ubuf__ float *beta, __ubuf__ float *u, __ubuf__ float *y)
{
    auto mask = CreateMask<float, MaskPattern::ALL>();
    F sumLow, sumHigh, b, updateLow, updateHigh;
    Duplicate(sumLow, 0.f, mask);
    Duplicate(sumHigh, 0.f, mask);
    // Predict the value from the decayed state.
    for (uint16_t k = 0; k < 128; ++k) {
        F stateLow, stateHigh, decayValue, keyValue, weightedKey;
        LoadAlign(stateLow, state + k * 128);
        LoadAlign(stateHigh, state + k * 128 + 64);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(decayValue, decay + k);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(keyValue, qkv + 128 + k);
        Mul(weightedKey, decayValue, keyValue, mask);
        Mul(stateLow, stateLow, weightedKey, mask);
        Mul(stateHigh, stateHigh, weightedKey, mask);
        Add(sumLow, sumLow, stateLow, mask);
        Add(sumHigh, sumHigh, stateHigh, mask);
    }
    LoadAlign(b, beta);
    LoadAlign(updateLow, qkv + 256);
    LoadAlign(updateHigh, qkv + 320);
    Sub(updateLow, updateLow, sumLow, mask);
    Sub(updateHigh, updateHigh, sumHigh, mask);
    Mul(updateLow, updateLow, b, mask);
    Mul(updateHigh, updateHigh, b, mask);
    StoreAlign(u, updateLow, mask);
    StoreAlign(u + 64, updateHigh, mask);
    // Update the state and accumulate the query projection in the same pass.
    Duplicate(sumLow, 0.f, mask);
    Duplicate(sumHigh, 0.f, mask);
    for (uint16_t k = 0; k < 128; ++k) {
        F stateLow, stateHigh, decayValue, keyValue, queryValue, productLow, productHigh;
        LoadAlign(stateLow, state + k * 128);
        LoadAlign(stateHigh, state + k * 128 + 64);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(decayValue, decay + k);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(keyValue, qkv + 128 + k);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(queryValue, qkv + k);
        Mul(stateLow, stateLow, decayValue, mask);
        Mul(stateHigh, stateHigh, decayValue, mask);
        Mul(productLow, updateLow, keyValue, mask);
        Mul(productHigh, updateHigh, keyValue, mask);
        Add(stateLow, stateLow, productLow, mask);
        Add(stateHigh, stateHigh, productHigh, mask);
        StoreAlign(state + k * 128, stateLow, mask);
        StoreAlign(state + k * 128 + 64, stateHigh, mask);
        Mul(productLow, stateLow, queryValue, mask);
        Mul(productHigh, stateHigh, queryValue, mask);
        Add(sumLow, sumLow, productLow, mask);
        Add(sumHigh, sumHigh, productHigh, mask);
    }
    StoreAlign(y, sumLow, mask);
    StoreAlign(y + 64, sumHigh, mask);
}
__simd_vf__ inline void finish(__ubuf__ float *y, __ubuf__ Bf *gate, __ubuf__ float *gamma, float eps, __ubuf__ Bf *out)
{
    auto m = CreateMask<float, MaskPattern::ALL>();
    F a, b, aa, bb, s, n, one;
    LoadAlign(a, y);
    LoadAlign(b, y + 64);
    Mul(aa, a, a, m);
    Mul(bb, b, b, m);
    Add(aa, aa, bb, m);
    ReduceSum(s, aa, m);
    Duplicate<float, HighLowPart::LOWEST>(n, s, m);
    Muls(n, n, 1.f / 128.f, m);
    Adds(n, n, eps, m);
    Sqrt(n, n, m);
    Duplicate(one, 1.f, m);
    Div(n, one, n, m);
    for (uint16_t i = 0; i < 2; ++i) {
        F x, g, weight, act;
        LoadAlign(x, y + 64 * i);
        LoadAlign(weight, gamma + 64 * i);
        readBf(g, gate + 64 * i, m);
        logistic(act, g, m);
        Mul(x, x, n, m);
        Mul(x, x, weight, m);
        Mul(x, x, act, m);
        RegTensor<Bf> rounded;
        Cast<Bf, float, narrow>(rounded, x, m);
        StoreAlign<Bf, StoreDist::DIST_PACK_B32>(out + 64 * i, rounded, m);
    }
}
} // namespace vec
#endif

class Recurrence {
    using Resource = Catlass::Arch::Resource<Arch>;
    template <typename T>
    CATLASS_DEVICE AscendC::GlobalTensor<T> gm(GM_ADDR ptr, uint64_t offset = 0)
    {
        AscendC::GlobalTensor<T> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr) + offset);
        return g;
    }

public:
    CATLASS_DEVICE uint32_t prefix(const RecurrenceArgs &a, uint32_t b)
    {
        if (a.sequence) {
            return b * a.sequence;
        }
        if (a.prefix64) {
            return static_cast<uint32_t>(gm<int64_t>(a.prefix).GetValue(b));
        }
        return gm<int32_t>(a.prefix).GetValue(b);
    }
    CATLASS_DEVICE uint32_t owner(const RecurrenceArgs &a, uint32_t row)
    {
        if (a.sequence) {
            return row / a.sequence;
        }
        uint32_t lo = 0, hi = a.batch;
        if (a.plan) {
            lo = reinterpret_cast<__gm__ const int32_t *>(a.plan)[2 + row / 16];
            if (row % 16 == 0) {
                return lo;
            }
            hi = lo + (a.batch - lo < 16 ? a.batch - lo : 16);
        }
        while (lo + 1 < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (prefix(a, mid) <= row) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

public:
    template <bool AllowColumnMajor = true, bool MxQkv = false, uint16_t ValueRows = 128>
    CATLASS_DEVICE void consume(Resource &resource, const RecurrenceArgs &a, GM_ADDR gated, GM_ADDR carry,
                                uint32_t heads, uint32_t head, uint32_t global, uint32_t ringRow, uint32_t valid,
                                AscendC::LocalTensor<Bf> slot, uint32_t wholeSeq = UINT32_MAX, uint16_t valueBegin = 0,
                                GM_ADDR historySnapshot = nullptr, bool reuseHead = false)
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        constexpr uint32_t base = 41984 + (MxQkv ? kMxUbOffset : 0);
        auto state = resource.ubBuf.template GetBufferByByte<float>(base);
        auto hist = resource.ubBuf.template GetBufferByByte<Bf>(base + 65536);
        auto weights = resource.ubBuf.template GetBufferByByte<Bf>(base + 65536 + 2304);
        constexpr uint32_t tmp = base + 65536 + 2304 + 3072;
        auto f = resource.ubBuf.template GetBufferByByte<float>(tmp);
        auto gatedRow = resource.ubBuf.template GetBufferByByte<Bf>(tmp + 8 * 512);
        auto keyRow = resource.ubBuf.template GetBufferByByte<half>(tmp + 8 * 512 + 256);
        uint32_t width = heads * 128;
        auto qkv = f, u = f[384], dec = f[512], y = f[640], bias = f[768], gamma = f[896];
        if (!reuseHead) {
            for (uint32_t kind = 0; kind < 3; ++kind) {
                if (a.convLayout == 0) {
                    AscendC::DataCopyPad(weights[kind * 512], gm<Bf>(a.convW, uint64_t(kind) * width + head * 128),
                                         AscendC::DataCopyExtParams{4, 256, (3 * width - 128) * 2, 0, 0},
                                         AscendC::DataCopyPadExtParams<Bf>{});
                } else {
                    uint64_t group = a.convLayout == 1 ? uint64_t(kind) * heads + head : kind;
                    AscendC::DataCopy(weights[kind * 512], gm<Bf>(a.convW, group * 512), 512);
                }
            }
            AscendC::DataCopy(bias, gm<float>(a.bias, head * 128), 128);
            AscendC::DataCopy(gamma, gm<float>(a.gamma), 128);
        }
        float aLog = gm<float>(a.alphaLog).GetValue(head);
        uint32_t b = wholeSeq == UINT32_MAX ? owner(a, global) : wholeSeq;
        auto chunk = resource.ubBuf.template GetBufferByByte<float>(tmp + 10 * 512);
        constexpr uint32_t window = MxQkv && ValueRows == 128 ? 8 : 4;
        auto chunkDec = chunk[window * 384], chunkBeta = chunkDec[window * 128];
        auto chunkU = chunkBeta[window * 64], chunkY = chunkU[window * 128];
        const bool completeWindow =
            (MxQkv || (a.staged >= 2 && valid < 16)) && valid <= 16 &&
            (!carry || global / 64 == (global + valid - 1) / 64) &&
            (wholeSeq != UINT32_MAX || (prefix(a, b) == global && prefix(a, b + 1) == global + valid));
        const bool shortSequence = !MxQkv && a.staged >= 2 && valid < 16 && completeWindow;
        const bool knownSpan = MxQkv ? completeWindow : shortSequence;
        uint32_t shortIndex = knownSpan ? gm<int32_t>(a.indices).GetValue(b) : 0;
        constexpr uint32_t shortOut = tmp + 10 * 512 + (window == 8 ? 7 : 5) * 4096;
        auto gatedRows = resource.ubBuf.template GetBufferByByte<Bf>(shortOut);
        auto keyRows = resource.ubBuf.template GetBufferByByte<half>(shortOut + window * 256);
        for (uint32_t r = 0; r < valid; ++r) {
            uint32_t row = global + r, start = knownSpan ? global : prefix(a, b),
                     end = knownSpan ? global + valid : prefix(a, b + 1), t = row - start;
            const bool useColumnMajor = !MxQkv && ValueRows == 128 && AllowColumnMajor && (end - start) >= 16;
            uint32_t idx = knownSpan ? shortIndex : gm<int32_t>(a.indices).GetValue(b);
            uint64_t convBase = uint64_t(idx) * a.convStride;
            if (t == 0 || (carry && row % 64 == 0)) {
                auto src = t == 0 ? gm<float>(a.checkpoint, uint64_t(idx) * a.checkpointStride + head * 16384) :
                                    gm<float>(carry, head * 16384);
                // Checkpoints are row-major. A carry keeps the on-chip layout between 64-row tiles.
                AscendC::DataCopy(useColumnMajor && t != 0 ? chunk : state, src[valueBegin * 128], ValueRows * 128);
                uint32_t accepted = gm<int32_t>(a.accepted).GetValue(b);
                uint32_t hs = t ? t - 1 : (accepted ? accepted - 1 : 0);
                const uint64_t convStart = (ValueRows == 128 ? convBase + uint64_t(hs) * 3 * width : 0) + head * 128;
                GM_ADDR convHistory = ValueRows == 128 ? a.conv : historySnapshot;
                // Gather the three history rows into the cyclic UB buffer, splitting only at its wrap.
                const uint32_t wrap = t % 3;
                const uint32_t first = (3 - wrap) * 3;
                AscendC::DataCopyPad(
                    hist[wrap * 384], gm<Bf>(convHistory, convStart),
                    AscendC::DataCopyExtParams{static_cast<uint16_t>(first), 256, (width - 128) * 2, 0, 0},
                    AscendC::DataCopyPadExtParams<Bf>{});
                if (wrap) {
                    AscendC::DataCopyPad(
                        hist, gm<Bf>(convHistory, convStart + uint64_t(first) * width),
                        AscendC::DataCopyExtParams{static_cast<uint16_t>(9 - first), 256, (width - 128) * 2, 0, 0},
                        AscendC::DataCopyPadExtParams<Bf>{});
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                if (useColumnMajor && t == 0) {
                    vec::transposeState((__ubuf__ float *)chunk.GetPhyAddr(), (__ubuf__ float *)state.GetPhyAddr());
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if (t == 0 && valueBegin == 0) {
                    AscendC::DataCopyPad(gm<Bf>(a.conv, convBase + head * 128), hist[384],
                                         AscendC::DataCopyExtParams{6, 256, 0, (width - 128) * 2, 0});
                }
            }
            if (MxQkv || shortSequence) {
                uint32_t count = valid - r < window ? valid - r : window;
                if (count > end - row) {
                    count = end - row;
                }
                // A complete staged request can cross a global 64-row boundary without a state handoff.
                if (carry && count > 64 - row % 64) {
                    count = 64 - row % 64;
                }
                auto raw = (__ubuf__ Bf *)slot[r * 128].GetPhyAddr();
                auto history = (__ubuf__ Bf *)hist.GetPhyAddr();
                auto filter = (__ubuf__ Bf *)weights.GetPhyAddr();
                auto prepared = (__ubuf__ float *)chunk.GetPhyAddr();
                if (completeWindow && r >= 3) {
                    vec::shortConvolveAll<false, true>(raw, history, filter, prepared, count, a.scale);
                } else if (completeWindow && r == 0) {
                    vec::shortConvolveAll(raw, history, filter, prepared, count, a.scale);
                } else {
                    // The history copy at t=0 and the previous group must be complete before updating the ring.
                    AscendC::PipeBarrier<PIPE_ALL>();
                    vec::shortConvolveAll<true>(raw, history, filter, prepared, count, a.scale, t % 3);
                }
                if (!valueBegin) {
                    for (uint32_t i = 0; i < 3; ++i) {
                        AscendC::DataCopyPad(
                            gm<Bf>(a.conv, convBase + uint64_t(2 + t) * 3 * width + i * width + head * 128),
                            slot[i * 2048 + r * 128],
                            AscendC::DataCopyExtParams{static_cast<uint16_t>(count), 256, 0, (3 * width - 128) * 2, 0});
                    }
                }
                vec::shortDecay(
                    (__ubuf__ Bf *)slot[3 * 2048 + r * 128].GetPhyAddr(),
                    (__ubuf__ Bf *)slot[5 * 2048 + r * 16].GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(), aLog,
                    a.lower, (__ubuf__ float *)chunkDec.GetPhyAddr(), (__ubuf__ float *)chunkBeta.GetPhyAddr(), count);
                AscendC::PipeBarrier<PIPE_V>();
                auto statePtr = (__ubuf__ float *)state.GetPhyAddr();
                auto qkvPtr = (__ubuf__ float *)chunk.GetPhyAddr();
                auto decayPtr = (__ubuf__ float *)chunkDec.GetPhyAddr();
                auto betaPtr = (__ubuf__ float *)chunkBeta.GetPhyAddr();
                auto updatePtr = (__ubuf__ float *)chunkU.GetPhyAddr();
                auto outputPtr = (__ubuf__ float *)chunkY.GetPhyAddr();
                if constexpr (ValueRows == 128) {
                    vec::advanceWindow<128>(statePtr, qkvPtr, decayPtr, betaPtr, updatePtr, outputPtr, count, 0);
                } else {
                    // Keep the token count explicit for each value-row stripe.
                    switch (count) {
                        case 1:
                            vec::advanceWindow<ValueRows, 1>(statePtr, qkvPtr, decayPtr, betaPtr, updatePtr, outputPtr,
                                                             1, valueBegin);
                            break;
                        case 2:
                            vec::advanceWindow<ValueRows, 2>(statePtr, qkvPtr, decayPtr, betaPtr, updatePtr, outputPtr,
                                                             2, valueBegin);
                            break;
                        case 3:
                            vec::advanceWindow<ValueRows, 3>(statePtr, qkvPtr, decayPtr, betaPtr, updatePtr, outputPtr,
                                                             3, valueBegin);
                            break;
                        default:
                            vec::advanceWindow<ValueRows, 4>(statePtr, qkvPtr, decayPtr, betaPtr, updatePtr, outputPtr,
                                                             4, valueBegin);
                            break;
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (!valueBegin) {
                    for (uint32_t j = 0; j < count; ++j) {
                        if constexpr (ValueRows == 128) {
                            vec::finish((__ubuf__ float *)chunkY[j * 128].GetPhyAddr(),
                                        (__ubuf__ Bf *)slot[4 * 2048 + (r + j) * 128].GetPhyAddr(),
                                        (__ubuf__ float *)gamma.GetPhyAddr(), a.eps,
                                        (__ubuf__ Bf *)gatedRows[j * 128].GetPhyAddr());
                        }
                        AscendC::Cast(keyRows[j * 128], chunk[j * 384 + 128], AscendC::RoundMode::CAST_RINT, 128);
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                const uint64_t at = (uint64_t(head) * a.capacity + t) * 128;
                if (!valueBegin) {
                    AscendC::DataCopy(gm<float>(a.decay, idx * a.decayStride + at), chunkDec, count * 128);
                    AscendC::DataCopy(gm<half>(a.key, idx * a.keyStride + at), keyRows, count * 128);
                }
                if constexpr (ValueRows == 128) {
                    AscendC::DataCopy(gm<float>(a.u, idx * a.uStride + at), chunkU, count * 128);
                } else {
                    AscendC::DataCopyPad(gm<float>(a.u, idx * a.uStride + at + valueBegin), chunkU,
                                         AscendC::DataCopyExtParams{static_cast<uint16_t>(count), ValueRows * 4, 0,
                                                                    (128 - ValueRows) * 4, 0});
                }
                if constexpr (ValueRows == 128) {
                    AscendC::DataCopyPad(
                        gm<Bf>(gated, uint64_t(ringRow + r) * width + head * 128), gatedRows,
                        AscendC::DataCopyExtParams{static_cast<uint16_t>(count), 256, 0, (width - 128) * 2, 0});
                } else {
                    AscendC::DataCopyPad(gm<float>(gated, uint64_t(ringRow + r) * width + head * 128 + valueBegin),
                                         chunkY,
                                         AscendC::DataCopyExtParams{static_cast<uint16_t>(count), ValueRows * 4, 0,
                                                                    (width - ValueRows) * 4, 0});
                }
                if (carry && (row + count) % 64 == 0 && row + count < end) {
                    AscendC::DataCopy(gm<float>(carry, head * 16384 + valueBegin * 128), state, ValueRows * 128);
                }
                if (row + count == end)
                    ++b;
                r += count - 1;
                AscendC::PipeBarrier<PIPE_ALL>();
                continue;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < 3; ++i) {
                auto raw = slot[i * 2048 + r * 128];
                vec::convolve((__ubuf__ Bf *)raw.GetPhyAddr(), (__ubuf__ Bf *)hist[i * 128].GetPhyAddr(),
                              (__ubuf__ Bf *)weights[i * 512].GetPhyAddr(), (__ubuf__ float *)qkv[i * 128].GetPhyAddr(),
                              t % 3, i < 2, i == 0 ? a.scale : 1.f);
                AscendC::DataCopy(gm<Bf>(a.conv, convBase + uint64_t(2 + t) * 3 * width + i * width + head * 128), raw,
                                  128);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (useColumnMajor) {
                vec::shortDecay((__ubuf__ Bf *)slot[3 * 2048 + r * 128].GetPhyAddr(),
                                (__ubuf__ Bf *)slot[5 * 2048 + r * 16].GetPhyAddr(),
                                (__ubuf__ float *)bias.GetPhyAddr(), aLog, a.lower, (__ubuf__ float *)dec.GetPhyAddr(),
                                (__ubuf__ float *)state.GetPhyAddr(), 1);
                AscendC::PipeBarrier<PIPE_V>();
                vec::stepColumnMajor((__ubuf__ float *)chunk.GetPhyAddr(), (__ubuf__ float *)qkv.GetPhyAddr(),
                                     (__ubuf__ float *)dec.GetPhyAddr(), (__ubuf__ float *)state.GetPhyAddr(),
                                     (__ubuf__ float *)u.GetPhyAddr(), (__ubuf__ float *)y.GetPhyAddr());
            } else {
                vec::step((__ubuf__ float *)state.GetPhyAddr(), (__ubuf__ float *)qkv.GetPhyAddr(),
                          (__ubuf__ Bf *)slot[3 * 2048 + r * 128].GetPhyAddr(),
                          (__ubuf__ Bf *)slot[5 * 2048 + r * 16].GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(),
                          aLog, a.lower, (__ubuf__ float *)u.GetPhyAddr(), (__ubuf__ float *)dec.GetPhyAddr(),
                          (__ubuf__ float *)y.GetPhyAddr());
            }
            AscendC::PipeBarrier<PIPE_V>();
            vec::finish((__ubuf__ float *)y.GetPhyAddr(), (__ubuf__ Bf *)slot[4 * 2048 + r * 128].GetPhyAddr(),
                        (__ubuf__ float *)gamma.GetPhyAddr(), a.eps, (__ubuf__ Bf *)gatedRow.GetPhyAddr());
            AscendC::Cast(keyRow, qkv[128], AscendC::RoundMode::CAST_RINT, 128);
            AscendC::PipeBarrier<PIPE_ALL>();
            uint64_t at = (uint64_t(head) * a.capacity + t) * 128;
            AscendC::DataCopy(gm<float>(a.u, idx * a.uStride + at), u, 128);
            AscendC::DataCopy(gm<float>(a.decay, idx * a.decayStride + at), dec, 128);
            AscendC::DataCopy(gm<half>(a.key, idx * a.keyStride + at), keyRow, 128);
            AscendC::DataCopy(gm<Bf>(gated, uint64_t(ringRow + r) * width + head * 128), gatedRow, 128);
            if (carry && (row + 1) % 64 == 0 && row + 1 < end) {
                AscendC::DataCopy(gm<float>(carry, head * 16384), useColumnMajor ? chunk : state, 16384);
            }
            if (row + 1 == end) {
                ++b;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
#endif
    }
};
} // namespace kda
