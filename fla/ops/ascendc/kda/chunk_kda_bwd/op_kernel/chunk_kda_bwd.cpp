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
constexpr uint64_t MAX_K_DIM = 256;
constexpr uint32_t EXP2_EVENT_ID = 0;
constexpr uint32_t KDA_MTE2_V_EVENT_ID = 1;

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
__aicore__ inline float ReadAsFloat(GlobalTensor<T> &tensor, uint64_t offset)
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
__aicore__ inline void AddToTensor(GlobalTensor<T> &tensor, uint64_t offset, float delta)
{
    float value = ReadAsFloat(tensor, offset) + delta;
    WriteFromFloat(tensor, offset, value);
}

__aicore__ inline void AddToFloatTensor(GlobalTensor<float> &tensor, uint64_t offset, float delta)
{
    tensor.SetValue(offset, tensor.GetValue(offset) + delta);
}

template <typename T>
class ChunkKdaBwdKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR aqk,
                                GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew,
                                GM_ADDR h, GM_ADDR dO, GM_ADDR initialState, GM_ADDR dht, GM_ADDR cuSeqlens,
                                GM_ADDR chunkIndices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
                                GM_ADDR dgk, GM_ADDR dh0, const ChunkKdaBwdTilingData &tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ T *)q);
        k_.SetGlobalBuffer((__gm__ T *)k);
        v_.SetGlobalBuffer((__gm__ T *)v);
        gk_.SetGlobalBuffer((__gm__ float *)gk);
        beta_.SetGlobalBuffer((__gm__ float *)beta);
        aqk_.SetGlobalBuffer((__gm__ T *)aqk);
        akk_.SetGlobalBuffer((__gm__ T *)akk);
        w_.SetGlobalBuffer((__gm__ T *)w);
        u_.SetGlobalBuffer((__gm__ T *)u);
        qg_.SetGlobalBuffer((__gm__ T *)qg);
        kg_.SetGlobalBuffer((__gm__ T *)kg);
        vNew_.SetGlobalBuffer((__gm__ T *)vNew);
        h_.SetGlobalBuffer((__gm__ T *)h);
        dO_.SetGlobalBuffer((__gm__ T *)dO);
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer((__gm__ T *)initialState);
        }
        if (dht != nullptr) {
            dht_.SetGlobalBuffer((__gm__ T *)dht);
        }
        if (cuSeqlens != nullptr) {
            cuSeqlens_.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
        }
        if (chunkIndices != nullptr) {
            chunkIndices_.SetGlobalBuffer((__gm__ int64_t *)chunkIndices);
        }
        dq_.SetGlobalBuffer((__gm__ T *)dq);
        dk_.SetGlobalBuffer((__gm__ T *)dk);
        dv_.SetGlobalBuffer((__gm__ T *)dv);
        dbeta_.SetGlobalBuffer((__gm__ float *)dbeta);
        dgk_.SetGlobalBuffer((__gm__ float *)dgk);
        dh0_.SetGlobalBuffer((__gm__ T *)dh0);

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
        hasDht_ = tiling.hasDht;
        isVarLen_ = tiling.isVarLen;

        pipe_->InitBuffer(exp2Buf_, MAX_K_DIM * sizeof(float));
        pipe_->InitBuffer(vecBuf_, MAX_K_DIM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t taskNum = static_cast<uint64_t>(N_ * H_);
        uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += blockNum) {
            uint64_t seq = task / H_;
            uint64_t h = task % H_;
            ProcessSeqHead(seq, h);
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

    __aicore__ inline float DState(uint64_t seq, uint64_t hv, uint64_t d, uint64_t r)
    {
        return ReadAsFloat(dh0_, StateOffset(seq, hv, d, r));
    }

    __aicore__ inline float HPrev(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t d, uint64_t r)
    {
        return ReadAsFloat(h_, HOffset(b, hv, chunkIdx, d, r));
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

    template <typename CopyT>
    __aicore__ inline void CopyRowIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset)
    {
        uint64_t rowBytes = K_ * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(K_));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    __aicore__ inline LocalTensor<float> Exp2G(uint64_t b, uint64_t hv, uint64_t t)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        CopyRowIn(exp2Local, gk_, KVOffset(b, hv, t, 0, K_));
        SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        Muls(exp2Local, exp2Local, LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline LocalTensor<float> Exp2GDiff(uint64_t b, uint64_t hv, uint64_t lhs, uint64_t rhs)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        LocalTensor<float> rhsLocal = vecBuf_.Get<float>();
        CopyRowIn(exp2Local, gk_, KVOffset(b, hv, lhs, 0, K_));
        CopyRowIn(rhsLocal, gk_, KVOffset(b, hv, rhs, 0, K_));
        SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        Sub(exp2Local, exp2Local, rhsLocal, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        Muls(exp2Local, exp2Local, LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline void ZeroSeqHead(uint64_t b, uint64_t seqStart, uint64_t seqEnd, uint64_t h,
                                       uint64_t hvBegin, uint64_t hvEnd, uint64_t seq)
    {
        for (uint64_t t = seqStart; t < seqEnd; ++t) {
            for (uint64_t d = 0; d < K_; ++d) {
                WriteFromFloat(dq_, QOffset(b, h, t, d), 0.0f);
                WriteFromFloat(dk_, QOffset(b, h, t, d), 0.0f);
            }
        }
        for (uint64_t hv = hvBegin; hv < hvEnd; ++hv) {
            for (uint64_t t = seqStart; t < seqEnd; ++t) {
                dbeta_.SetValue(BetaOffset(b, hv, t), 0.0f);
                for (uint64_t d = 0; d < K_; ++d) {
                    dgk_.SetValue(KVOffset(b, hv, t, d, K_), 0.0f);
                }
                for (uint64_t r = 0; r < V_; ++r) {
                    WriteFromFloat(dv_, KVOffset(b, hv, t, r, V_), 0.0f);
                }
            }
            for (uint64_t d = 0; d < K_; ++d) {
                for (uint64_t r = 0; r < V_; ++r) {
                    float value = 0.0f;
                    if (hasDht_) {
                        value = ReadAsFloat(dht_, StateOffset(seq, hv, d, r));
                    }
                    WriteFromFloat(dh0_, StateOffset(seq, hv, d, r), value);
                }
            }
        }
    }

    __aicore__ inline float DVNewGradValue(uint64_t b, uint64_t seq, uint64_t hv, uint64_t start, uint64_t end,
                                           uint64_t token, uint64_t r)
    {
        uint64_t curT = end - start;
        uint64_t j = token - start;
        float value = 0.0f;
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            value += ReadAsFloat(aqk_, AOffset(b, hv, ti, j)) *
                     ReadAsFloat(dO_, KVOffset(b, hv, ti, r, V_));
        }
        for (uint64_t d = 0; d < K_; ++d) {
            value += ReadAsFloat(kg_, KVOffset(b, hv, token, d, K_)) * DState(seq, hv, d, r);
        }
        return value;
    }

    __aicore__ inline void AddQGKGDecayGrad(uint64_t b, uint64_t seq, uint64_t qh, uint64_t hv,
                                            uint64_t chunkIdx, uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        uint64_t last = end - 1;
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            LocalTensor<float> expG = Exp2G(b, hv, ti);
            for (uint64_t d = 0; d < K_; ++d) {
                float dQG = 0.0f;
                for (uint64_t r = 0; r < V_; ++r) {
                    dQG += ReadAsFloat(dO_, KVOffset(b, hv, ti, r, V_)) * HPrev(b, hv, chunkIdx, d, r);
                }
                dQG *= scale_;

                float qv = ReadAsFloat(q_, QOffset(b, qh, ti, d));
                float expg = expG.GetValue(d);
                AddToTensor(dq_, QOffset(b, qh, ti, d), dQG * expg);
                AddToFloatTensor(dgk_, KVOffset(b, hv, ti, d, K_), dQG * qv * expg * LN2);
            }

            LocalTensor<float> expLastMinusG = Exp2GDiff(b, hv, last, ti);
            for (uint64_t d = 0; d < K_; ++d) {
                float dKG = 0.0f;
                for (uint64_t r = 0; r < V_; ++r) {
                    dKG += DState(seq, hv, d, r) * ReadAsFloat(vNew_, KVOffset(b, hv, ti, r, V_));
                }
                float kv = ReadAsFloat(k_, QOffset(b, qh, ti, d));
                float factor = expLastMinusG.GetValue(d);
                float kgGateGrad = dKG * kv * factor * LN2;
                AddToTensor(dk_, QOffset(b, qh, ti, d), dKG * factor);
                AddToFloatTensor(dgk_, KVOffset(b, hv, last, d, K_), kgGateGrad);
                AddToFloatTensor(dgk_, KVOffset(b, hv, ti, d, K_), -kgGateGrad);
            }
        }

        LocalTensor<float> decayGate = Exp2G(b, hv, last);
        for (uint64_t d = 0; d < K_; ++d) {
            float decay = decayGate.GetValue(d);
            float dDecay = 0.0f;
            for (uint64_t r = 0; r < V_; ++r) {
                dDecay += DState(seq, hv, d, r) * HPrev(b, hv, chunkIdx, d, r);
            }
            AddToFloatTensor(dgk_, KVOffset(b, hv, last, d, K_), dDecay * decay * LN2);
        }
    }

    __aicore__ inline float DWValue(uint64_t b, uint64_t seq, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                    uint64_t end, uint64_t token, uint64_t d)
    {
        float value = 0.0f;
        for (uint64_t r = 0; r < V_; ++r) {
            value -= DVNewGradValue(b, seq, hv, start, end, token, r) * HPrev(b, hv, chunkIdx, d, r);
        }
        return value;
    }

    __aicore__ inline float DUSum(uint64_t b, uint64_t seq, uint64_t hv, uint64_t start, uint64_t end,
                                  uint64_t token, uint64_t r)
    {
        return DVNewGradValue(b, seq, hv, start, end, token, r);
    }

    __aicore__ inline float DYValue(uint64_t b, uint64_t seq, uint64_t qh, uint64_t hv, uint64_t start,
                                    uint64_t end,
                                    uint64_t rowToken, uint64_t colToken, uint64_t chunkIdx)
    {
        float value = 0.0f;
        LocalTensor<float> expG = Exp2G(b, hv, colToken);
        for (uint64_t d = 0; d < K_; ++d) {
            float dw = DWValue(b, seq, hv, chunkIdx, start, end, rowToken, d);
            float kj = ReadAsFloat(k_, QOffset(b, qh, colToken, d));
            float betaJ = beta_.GetValue(BetaOffset(b, hv, colToken));
            value += dw * kj * betaJ * expG.GetValue(d);
        }
        for (uint64_t r = 0; r < V_; ++r) {
            float du = DUSum(b, seq, hv, start, end, rowToken, r);
            float vj = ReadAsFloat(v_, KVOffset(b, hv, colToken, r, V_));
            float betaJ = beta_.GetValue(BetaOffset(b, hv, colToken));
            value += du * vj * betaJ;
        }
        return value;
    }

    __aicore__ inline float DRawAkk(uint64_t b, uint64_t seq, uint64_t qh, uint64_t hv, uint64_t start,
                                   uint64_t end, uint64_t row, uint64_t col, uint64_t curT, uint64_t chunkIdx)
    {
        float value = 0.0f;
        for (uint64_t m = 0; m < curT; ++m) {
            float ymi = ReadAsFloat(akk_, AOffset(b, hv, start + m, row));
            if (ymi == 0.0f) {
                continue;
            }
            for (uint64_t n = 0; n < curT; ++n) {
                float yjn = ReadAsFloat(akk_, AOffset(b, hv, start + col, n));
                if (yjn == 0.0f) {
                    continue;
                }
                value -= ymi * DYValue(b, seq, qh, hv, start, end, start + m, start + n, chunkIdx) * yjn;
            }
        }
        return value;
    }

    __aicore__ inline void AddWUGrad(uint64_t b, uint64_t seq, uint64_t qh, uint64_t hv, uint64_t chunkIdx,
                                     uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        for (uint64_t j = 0; j < curT; ++j) {
            uint64_t tj = start + j;
            float betaJ = beta_.GetValue(BetaOffset(b, hv, tj));
            LocalTensor<float> expG = Exp2G(b, hv, tj);
            for (uint64_t d = 0; d < K_; ++d) {
                float dKbg = 0.0f;
                for (uint64_t i = 0; i < curT; ++i) {
                    uint64_t ti = start + i;
                    dKbg += ReadAsFloat(akk_, AOffset(b, hv, ti, j)) *
                             DWValue(b, seq, hv, chunkIdx, start, end, ti, d);
                }
                float kv = ReadAsFloat(k_, QOffset(b, qh, tj, d));
                float eg = expG.GetValue(d);
                AddToTensor(dk_, QOffset(b, qh, tj, d), dKbg * betaJ * eg);
                AddToFloatTensor(dbeta_, BetaOffset(b, hv, tj), dKbg * kv * eg);
                AddToFloatTensor(dgk_, KVOffset(b, hv, tj, d, K_), dKbg * kv * betaJ * eg * LN2);
            }
            for (uint64_t r = 0; r < V_; ++r) {
                float dVb = 0.0f;
                for (uint64_t i = 0; i < curT; ++i) {
                    uint64_t ti = start + i;
                    dVb += ReadAsFloat(akk_, AOffset(b, hv, ti, j)) * DUSum(b, seq, hv, start, end, ti, r);
                }
                float vv = ReadAsFloat(v_, KVOffset(b, hv, tj, r, V_));
                AddToTensor(dv_, KVOffset(b, hv, tj, r, V_), dVb * betaJ);
                AddToFloatTensor(dbeta_, BetaOffset(b, hv, tj), dVb * vv);
            }
        }

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            float betaI = beta_.GetValue(BetaOffset(b, hv, ti));
            for (uint64_t j = 0; j < i; ++j) {
                uint64_t tj = start + j;
                float dRaw = DRawAkk(b, seq, qh, hv, start, end, i, j, curT, chunkIdx);
                float sumKK = 0.0f;
                LocalTensor<float> relGate = Exp2GDiff(b, hv, ti, tj);
                for (uint64_t d = 0; d < K_; ++d) {
                    float ki = ReadAsFloat(k_, QOffset(b, qh, ti, d));
                    float kj = ReadAsFloat(k_, QOffset(b, qh, tj, d));
                    float er = relGate.GetValue(d);
                    sumKK += ki * kj * er;
                    float common = dRaw * betaI * er;
                    AddToTensor(dk_, QOffset(b, qh, ti, d), common * kj);
                    AddToTensor(dk_, QOffset(b, qh, tj, d), common * ki);
                    float gateGrad = common * ki * kj * LN2;
                    AddToFloatTensor(dgk_, KVOffset(b, hv, ti, d, K_), gateGrad);
                    AddToFloatTensor(dgk_, KVOffset(b, hv, tj, d, K_), -gateGrad);
                }
                AddToFloatTensor(dbeta_, BetaOffset(b, hv, ti), dRaw * sumKK);
            }
        }
        (void)seq;
    }

    __aicore__ inline void AddAqkGrad(uint64_t b, uint64_t qh, uint64_t hv, uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            for (uint64_t j = 0; j <= i; ++j) {
                uint64_t tj = start + j;
                float dAqk = 0.0f;
                for (uint64_t r = 0; r < V_; ++r) {
                    dAqk += ReadAsFloat(dO_, KVOffset(b, hv, ti, r, V_)) *
                            ReadAsFloat(vNew_, KVOffset(b, hv, tj, r, V_));
                }
                LocalTensor<float> relGate = Exp2GDiff(b, hv, ti, tj);
                for (uint64_t d = 0; d < K_; ++d) {
                    float qi = ReadAsFloat(q_, QOffset(b, qh, ti, d));
                    float kj = ReadAsFloat(k_, QOffset(b, qh, tj, d));
                    float common = dAqk * scale_ * relGate.GetValue(d);
                    AddToTensor(dq_, QOffset(b, qh, ti, d), common * kj);
                    AddToTensor(dk_, QOffset(b, qh, tj, d), common * qi);
                    float gateGrad = common * qi * kj * LN2;
                    AddToFloatTensor(dgk_, KVOffset(b, hv, ti, d, K_), gateGrad);
                    AddToFloatTensor(dgk_, KVOffset(b, hv, tj, d, K_), -gateGrad);
                }
            }
        }
    }

    __aicore__ inline void UpdateDStatePrev(uint64_t b, uint64_t seq, uint64_t hv, uint64_t chunkIdx,
                                            uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        uint64_t last = end - 1;
        float newState[MAX_K_DIM];
        LocalTensor<float> decayGate = Exp2G(b, hv, last);
        for (uint64_t r = 0; r < V_; ++r) {
            for (uint64_t d = 0; d < K_; ++d) {
                float decay = decayGate.GetValue(d);
                float value = DState(seq, hv, d, r) * decay;
                for (uint64_t i = 0; i < curT; ++i) {
                    uint64_t ti = start + i;
                    value += scale_ * ReadAsFloat(qg_, KVOffset(b, hv, ti, d, K_)) *
                             ReadAsFloat(dO_, KVOffset(b, hv, ti, r, V_));
                    value -= ReadAsFloat(w_, KVOffset(b, hv, ti, d, K_)) *
                             DVNewGradValue(b, seq, hv, start, end, ti, r);
                }
                if (d < MAX_K_DIM) {
                    newState[d] = value;
                }
            }
            for (uint64_t d = 0; d < K_ && d < MAX_K_DIM; ++d) {
                WriteFromFloat(dh0_, StateOffset(seq, hv, d, r), newState[d]);
            }
        }
    }

    __aicore__ inline void ProcessChunk(uint64_t b, uint64_t seq, uint64_t qh, uint64_t hv, uint64_t chunkIdx,
                                        uint64_t start, uint64_t end)
    {
        if (end <= start) {
            return;
        }
        AddQGKGDecayGrad(b, seq, qh, hv, chunkIdx, start, end);
        AddWUGrad(b, seq, qh, hv, chunkIdx, start, end);
        AddAqkGrad(b, qh, hv, start, end);
        UpdateDStatePrev(b, seq, hv, chunkIdx, start, end);
    }

    __aicore__ inline void ProcessSeqHead(uint64_t seq, uint64_t qh)
    {
        uint64_t b = isVarLen_ ? 0 : seq;
        uint64_t seqStart = 0;
        uint64_t seqEnd = T_;
        if (isVarLen_) {
            seqStart = static_cast<uint64_t>(cuSeqlens_.GetValue(seq));
            seqEnd = static_cast<uint64_t>(cuSeqlens_.GetValue(seq + 1));
        }
        uint64_t group = HV_ / H_;
        uint64_t hvBegin = qh * group;
        uint64_t hvEnd = hvBegin + group;
        uint64_t chunkBase = isVarLen_ ? ChunkCountBefore(seq) : 0;
        uint64_t chunkCount = (seqEnd - seqStart + BT_ - 1) / BT_;

        ZeroSeqHead(b, seqStart, seqEnd, qh, hvBegin, hvEnd, seq);
        for (uint64_t hv = hvBegin; hv < hvEnd; ++hv) {
            for (uint64_t c = chunkCount; c > 0; --c) {
                uint64_t localChunk = c - 1;
                uint64_t start = seqStart + localChunk * BT_;
                uint64_t end = start + BT_;
                if (end > seqEnd) {
                    end = seqEnd;
                }
                ProcessChunk(b, seq, qh, hv, chunkBase + localChunk, start, end);
            }
        }
    }

private:
    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<T> v_;
    GlobalTensor<float> gk_;
    GlobalTensor<float> beta_;
    GlobalTensor<T> aqk_;
    GlobalTensor<T> akk_;
    GlobalTensor<T> w_;
    GlobalTensor<T> u_;
    GlobalTensor<T> qg_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<T> h_;
    GlobalTensor<T> dO_;
    GlobalTensor<T> initialState_;
    GlobalTensor<T> dht_;
    GlobalTensor<int64_t> cuSeqlens_;
    GlobalTensor<int64_t> chunkIndices_;
    GlobalTensor<T> dq_;
    GlobalTensor<T> dk_;
    GlobalTensor<T> dv_;
    GlobalTensor<float> dbeta_;
    GlobalTensor<float> dgk_;
    GlobalTensor<T> dh0_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> exp2Buf_;
    TBuf<TPosition::VECCALC> vecBuf_;

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
    bool hasDht_ = false;
    bool isVarLen_ = false;
};
} // namespace

extern "C" __global__ __aicore__ void chunk_kda_bwd(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
                                                      GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
                                                      GM_ADDR kg, GM_ADDR v_new, GM_ADDR h, GM_ADDR d_o,
                                                      GM_ADDR initial_state, GM_ADDR dht, GM_ADDR cu_seqlens,
                                                      GM_ADDR chunk_indices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                                      GM_ADDR dbeta, GM_ADDR dgk, GM_ADDR dh0, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (tilingData.dataType == 2) {
        ChunkKdaBwdKernel<float> op;
        op.Init(q, k, v, gk, beta, aqk, akk, w, u, qg, kg, v_new, h, d_o, initial_state, dht, cu_seqlens,
                chunk_indices, dq, dk, dv, dbeta, dgk, dh0, tilingData, &pipe);
        op.Process();
    } else if (tilingData.dataType == 1) {
        ChunkKdaBwdKernel<bfloat16_t> op;
        op.Init(q, k, v, gk, beta, aqk, akk, w, u, qg, kg, v_new, h, d_o, initial_state, dht, cu_seqlens,
                chunk_indices, dq, dk, dv, dbeta, dgk, dh0, tilingData, &pipe);
        op.Process();
    } else {
        ChunkKdaBwdKernel<half> op;
        op.Init(q, k, v, gk, beta, aqk, akk, w, u, qg, kg, v_new, h, d_o, initial_state, dht, cu_seqlens,
                chunk_indices, dq, dk, dv, dbeta, dgk, dh0, tilingData, &pipe);
        op.Process();
    }
}
