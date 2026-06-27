/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float LN2 = 0.69314718055994530942f;
constexpr uint32_t EXP2_UB_ELEMENTS = 256;
constexpr uint32_t EXP2_EVENT_ID = 0;

__aicore__ inline uint32_t FloatToBits(float value)
{
    union Bits {
        __aicore__ Bits() {}
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline float BitsToFloat(uint32_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint32_t u;
        float f;
    } bits;
    bits.u = value;
    return bits.f;
}

__aicore__ inline uint16_t Bf16ToBits(bfloat16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        bfloat16_t f;
        uint16_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline bfloat16_t BitsToBf16(uint16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint16_t u;
        bfloat16_t f;
    } bits;
    bits.u = value;
    return bits.f;
}

template <typename T>
__aicore__ inline float ReadAsFloat(const GlobalTensor<T> &tensor, uint64_t offset)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        return BitsToFloat(static_cast<uint32_t>(Bf16ToBits(tensor.GetValue(offset))) << 16);
    }
    return static_cast<float>(tensor.GetValue(offset));
}

template <typename T>
__aicore__ inline void WriteFromFloat(GlobalTensor<T> &tensor, uint64_t offset, float value)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        uint32_t bits = FloatToBits(value);
        uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
        tensor.SetValue(offset, BitsToBf16(static_cast<uint16_t>((bits + bias) >> 16)));
        return;
    }
    tensor.SetValue(offset, static_cast<T>(value));
}

template <typename T>
class ChunkKdaFwdKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR finalState,
                                GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg,
                                GM_ADDR vNew, GM_ADDR h, const ChunkKdaFwdTilingData &tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ T *)q);
        k_.SetGlobalBuffer((__gm__ T *)k);
        v_.SetGlobalBuffer((__gm__ T *)v);
        gk_.SetGlobalBuffer((__gm__ float *)gk);
        beta_.SetGlobalBuffer((__gm__ float *)beta);
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer((__gm__ T *)initialState);
        }
        if (cuSeqlens != nullptr) {
            cuSeqlens_.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
        }
        if (chunkIndices != nullptr) {
            chunkIndices_.SetGlobalBuffer((__gm__ int64_t *)chunkIndices);
        }
        o_.SetGlobalBuffer((__gm__ T *)o);
        finalState_.SetGlobalBuffer((__gm__ T *)finalState);
        aqk_.SetGlobalBuffer((__gm__ T *)aqk);
        akk_.SetGlobalBuffer((__gm__ T *)akk);
        w_.SetGlobalBuffer((__gm__ T *)w);
        u_.SetGlobalBuffer((__gm__ T *)u);
        qg_.SetGlobalBuffer((__gm__ T *)qg);
        kg_.SetGlobalBuffer((__gm__ T *)kg);
        vNew_.SetGlobalBuffer((__gm__ T *)vNew);
        h_.SetGlobalBuffer((__gm__ T *)h);

        B_ = tiling.batch;
        N_ = tiling.seqNum;
        H_ = tiling.qHeadNum;
        HV_ = tiling.vHeadNum;
        T_ = tiling.seqlen;
        K_ = tiling.kHeadDim;
        V_ = tiling.vHeadDim;
        BT_ = tiling.chunkSize;
        NT_ = tiling.totalChunks;
        scale_ = tiling.scale;
        hasInitial_ = tiling.hasInitialState;
        isVarLen_ = tiling.isVarLen;

        pipe_->InitBuffer(exp2Buf_, EXP2_UB_ELEMENTS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t taskNum = static_cast<uint64_t>(N_ * HV_);
        uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += blockNum) {
            uint64_t seq = task / HV_;
            uint64_t hv = task % HV_;
            ProcessSeqHead(seq, hv);
        }
    }

private:
    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        return ((b * H_ + h) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t KVOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d, uint64_t dim) const
    {
        return ((b * HV_ + hv) * T_ + t) * dim + d;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * HV_ + hv) * T_ + t;
    }

    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t j) const
    {
        return ((b * HV_ + hv) * T_ + t) * BT_ + j;
    }

    __aicore__ inline uint64_t HOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t d, uint64_t r) const
    {
        return (((b * HV_ + hv) * NT_ + chunkIdx) * K_ + d) * V_ + r;
    }

    __aicore__ inline uint64_t StateOffset(uint64_t seq, uint64_t hv, uint64_t d, uint64_t r) const
    {
        return ((seq * HV_ + hv) * K_ + d) * V_ + r;
    }

    __aicore__ inline uint64_t ChunkCountBefore(uint64_t seq) const
    {
        if (!isVarLen_) {
            return 0;
        }
        uint64_t count = 0;
        for (uint64_t s = 0; s < seq; ++s) {
            uint64_t start = static_cast<uint64_t>(cuSeqlens_.GetValue(s));
            uint64_t end = static_cast<uint64_t>(cuSeqlens_.GetValue(s + 1));
            count += (end - start + BT_ - 1) / BT_;
        }
        return count;
    }

    __aicore__ inline void RunExp2(LocalTensor<float> &tensor, uint32_t count)
    {
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        Exp(tensor, tensor, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
    }

    __aicore__ inline LocalTensor<float> Exp2G(uint64_t b, uint64_t hv, uint64_t t)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        for (uint64_t d = 0; d < K_; ++d) {
            exp2Local.SetValue(d, gk_.GetValue(KVOffset(b, hv, t, d, K_)) * LN2);
        }
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline LocalTensor<float> Exp2GDiff(uint64_t b, uint64_t hv, uint64_t lhs, uint64_t rhs)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        for (uint64_t d = 0; d < K_; ++d) {
            float diff = gk_.GetValue(KVOffset(b, hv, lhs, d, K_)) - gk_.GetValue(KVOffset(b, hv, rhs, d, K_));
            exp2Local.SetValue(d, diff * LN2);
        }
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline float DotQK(uint64_t b, uint64_t h, uint64_t hv, uint64_t ti, uint64_t tj)
    {
        float acc = 0.0f;
        LocalTensor<float> relGate = Exp2GDiff(b, hv, ti, tj);
        for (uint64_t d = 0; d < K_; ++d) {
            float qi = ReadAsFloat(q_, QOffset(b, h, ti, d));
            float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
            acc += qi * kj * relGate.GetValue(d);
        }
        return acc;
    }

    __aicore__ inline float DotKK(uint64_t b, uint64_t h, uint64_t hv, uint64_t ti, uint64_t tj)
    {
        float acc = 0.0f;
        LocalTensor<float> relGate = Exp2GDiff(b, hv, ti, tj);
        for (uint64_t d = 0; d < K_; ++d) {
            float ki = ReadAsFloat(k_, QOffset(b, h, ti, d));
            float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
            acc += ki * kj * relGate.GetValue(d);
        }
        return acc;
    }

    __aicore__ inline void InitState(uint64_t seq, uint64_t hv)
    {
        for (uint64_t d = 0; d < K_; ++d) {
            for (uint64_t r = 0; r < V_; ++r) {
                float value = 0.0f;
                if (hasInitial_) {
                    value = ReadAsFloat(initialState_, StateOffset(seq, hv, d, r));
                }
                WriteFromFloat(finalState_, StateOffset(seq, hv, d, r), value);
            }
        }
    }

    __aicore__ inline void StoreCurrentState(uint64_t b, uint64_t hv, uint64_t seq, uint64_t chunkIdx)
    {
        for (uint64_t d = 0; d < K_; ++d) {
            for (uint64_t r = 0; r < V_; ++r) {
                float value = ReadAsFloat(finalState_, StateOffset(seq, hv, d, r));
                WriteFromFloat(h_, HOffset(b, hv, chunkIdx, d, r), value);
            }
        }
    }

    __aicore__ inline void ProcessChunk(uint64_t b, uint64_t seq, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                        uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            for (uint64_t j = 0; j < BT_; ++j) {
                float aqkValue = 0.0f;
                float akkValue = 0.0f;
                if (j < curT && j <= i) {
                    uint64_t tj = start + j;
                    aqkValue = DotQK(b, h, hv, ti, tj) * scale_;
                    if (j < i) {
                        akkValue = DotKK(b, h, hv, ti, tj) * beta_.GetValue(BetaOffset(b, hv, ti));
                    }
                }
                WriteFromFloat(aqk_, AOffset(b, hv, ti, j), aqkValue);
                WriteFromFloat(akk_, AOffset(b, hv, ti, j), akkValue);
            }

            for (uint64_t j = 0; j < i; ++j) {
                float sum = 0.0f;
                for (uint64_t m = j; m < i; ++m) {
                    float lim = ReadAsFloat(akk_, AOffset(b, hv, ti, m));
                    float ymj = ReadAsFloat(akk_, AOffset(b, hv, start + m, j));
                    sum += lim * ymj;
                }
                WriteFromFloat(akk_, AOffset(b, hv, ti, j), -sum);
            }
            WriteFromFloat(akk_, AOffset(b, hv, ti, i), 1.0f);
        }

        uint64_t last = end - 1;
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            LocalTensor<float> expG = Exp2G(b, hv, ti);
            for (uint64_t d = 0; d < K_; ++d) {
                float qv = ReadAsFloat(q_, QOffset(b, h, ti, d));
                WriteFromFloat(qg_, KVOffset(b, hv, ti, d, K_), qv * expG.GetValue(d));
            }

            LocalTensor<float> expLastMinusG = Exp2GDiff(b, hv, last, ti);
            for (uint64_t d = 0; d < K_; ++d) {
                float kv = ReadAsFloat(k_, QOffset(b, h, ti, d));
                WriteFromFloat(kg_, KVOffset(b, hv, ti, d, K_), kv * expLastMinusG.GetValue(d));
            }

            float wSum[EXP2_UB_ELEMENTS];
            for (uint64_t d = 0; d < K_; ++d) {
                wSum[d] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                LocalTensor<float> expGj = Exp2G(b, hv, tj);
                float betaJ = beta_.GetValue(BetaOffset(b, hv, tj));
                float a = ReadAsFloat(akk_, AOffset(b, hv, ti, j));
                for (uint64_t d = 0; d < K_; ++d) {
                    float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
                    wSum[d] += a * kj * betaJ * expGj.GetValue(d);
                }
            }
            for (uint64_t d = 0; d < K_; ++d) {
                WriteFromFloat(w_, KVOffset(b, hv, ti, d, K_), wSum[d]);
            }
            for (uint64_t r = 0; r < V_; ++r) {
                float sum = 0.0f;
                for (uint64_t j = 0; j < curT; ++j) {
                    uint64_t tj = start + j;
                    float a = ReadAsFloat(akk_, AOffset(b, hv, ti, j));
                    float vj = ReadAsFloat(v_, KVOffset(b, hv, tj, r, V_));
                    float betaJ = beta_.GetValue(BetaOffset(b, hv, tj));
                    sum += a * vj * betaJ;
                }
                WriteFromFloat(u_, KVOffset(b, hv, ti, r, V_), sum);
            }
        }

        StoreCurrentState(b, hv, seq, chunkIdx);

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            for (uint64_t r = 0; r < V_; ++r) {
                float value = ReadAsFloat(u_, KVOffset(b, hv, ti, r, V_));
                for (uint64_t d = 0; d < K_; ++d) {
                    float wi = ReadAsFloat(w_, KVOffset(b, hv, ti, d, K_));
                    float hprev = ReadAsFloat(h_, HOffset(b, hv, chunkIdx, d, r));
                    value -= wi * hprev;
                }
                WriteFromFloat(vNew_, KVOffset(b, hv, ti, r, V_), value);
            }
        }

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            for (uint64_t r = 0; r < V_; ++r) {
                float out = 0.0f;
                for (uint64_t d = 0; d < K_; ++d) {
                    out += ReadAsFloat(qg_, KVOffset(b, hv, ti, d, K_)) *
                           ReadAsFloat(h_, HOffset(b, hv, chunkIdx, d, r)) * scale_;
                }
                for (uint64_t j = 0; j < curT; ++j) {
                    uint64_t tj = start + j;
                    out += ReadAsFloat(aqk_, AOffset(b, hv, ti, j)) *
                           ReadAsFloat(vNew_, KVOffset(b, hv, tj, r, V_));
                }
                WriteFromFloat(o_, KVOffset(b, hv, ti, r, V_), out);
            }
        }

        LocalTensor<float> decayGate = Exp2G(b, hv, last);
        for (uint64_t d = 0; d < K_; ++d) {
            float decay = decayGate.GetValue(d);
            for (uint64_t r = 0; r < V_; ++r) {
                float next = decay * ReadAsFloat(h_, HOffset(b, hv, chunkIdx, d, r));
                for (uint64_t i = 0; i < curT; ++i) {
                    uint64_t ti = start + i;
                    next += ReadAsFloat(kg_, KVOffset(b, hv, ti, d, K_)) *
                            ReadAsFloat(vNew_, KVOffset(b, hv, ti, r, V_));
                }
                WriteFromFloat(finalState_, StateOffset(seq, hv, d, r), next);
            }
        }
    }

    __aicore__ inline void ProcessSeqHead(uint64_t seq, uint64_t hv)
    {
        uint64_t b = isVarLen_ ? 0 : seq;
        uint64_t seqStart = 0;
        uint64_t seqEnd = T_;
        if (isVarLen_) {
            seqStart = static_cast<uint64_t>(cuSeqlens_.GetValue(seq));
            seqEnd = static_cast<uint64_t>(cuSeqlens_.GetValue(seq + 1));
        }
        uint64_t h = hv / (HV_ / H_);
        uint64_t chunkBase = isVarLen_ ? ChunkCountBefore(seq) : 0;
        InitState(seq, hv);
        uint64_t localChunk = 0;
        for (uint64_t start = seqStart; start < seqEnd; start += BT_) {
            uint64_t end = start + BT_;
            if (end > seqEnd) {
                end = seqEnd;
            }
            ProcessChunk(b, seq, h, hv, chunkBase + localChunk, start, end);
            ++localChunk;
        }
    }

private:
    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<T> v_;
    GlobalTensor<float> gk_;
    GlobalTensor<float> beta_;
    GlobalTensor<T> initialState_;
    GlobalTensor<int64_t> cuSeqlens_;
    GlobalTensor<int64_t> chunkIndices_;
    GlobalTensor<T> o_;
    GlobalTensor<T> finalState_;
    GlobalTensor<T> aqk_;
    GlobalTensor<T> akk_;
    GlobalTensor<T> w_;
    GlobalTensor<T> u_;
    GlobalTensor<T> qg_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<T> h_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> exp2Buf_;

    uint64_t B_ = 0;
    uint64_t N_ = 0;
    uint64_t H_ = 0;
    uint64_t HV_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t BT_ = 0;
    uint64_t NT_ = 0;
    float scale_ = 1.0f;
    bool hasInitial_ = false;
    bool isVarLen_ = false;
};
} // namespace

extern "C" __global__ __aicore__ void chunk_kda_fwd(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
                                                      GM_ADDR initial_state, GM_ADDR cu_seqlens,
                                                      GM_ADDR chunk_indices, GM_ADDR o, GM_ADDR final_state,
                                                      GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
                                                      GM_ADDR kg, GM_ADDR v_new, GM_ADDR h, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (tilingData.dataType == 2) {
        ChunkKdaFwdKernel<float> op;
        op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, o, final_state, aqk, akk, w, u, qg, kg,
                v_new, h, tilingData, &pipe);
        op.Process();
    } else if (tilingData.dataType == 1) {
        ChunkKdaFwdKernel<bfloat16_t> op;
        op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, o, final_state, aqk, akk, w, u, qg, kg,
                v_new, h, tilingData, &pipe);
        op.Process();
    } else {
        ChunkKdaFwdKernel<half> op;
        op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, o, final_state, aqk, akk, w, u, qg, kg,
                v_new, h, tilingData, &pipe);
        op.Process();
    }
}
