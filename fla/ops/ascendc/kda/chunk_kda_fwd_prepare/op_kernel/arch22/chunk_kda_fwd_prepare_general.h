// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"

namespace KdaPrepare::Arch22 {

// One owner processes one (chunk, head). Offsets and strides are in elements.
struct GeneralPrepareChunk {
    uint32_t rows, keyDim, valueDim;
    uint64_t keyOffset, valueOffset, betaOffset, keyStride, valueStride, betaStride;
    uint64_t cacheKeyOffset, cacheValueOffset, cacheScalarOffset, scoreOffset;
    float scale, epsilon, lowerBound;
    bool normalize, useExp2;
    uint32_t gateMode, betaMode;
};

struct GeneralPrepareBuffers {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR aLog;
    GM_ADDR bias;
    GM_ADDR gk;
    GM_ADDR aqk;
    GM_ADDR akk;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR qg;
    GM_ADDR kg;
    GM_ADDR qgScaled;
    GM_ADDR qHat;
    GM_ADDR kHat;
    GM_ADDR qRstd;
    GM_ADDR kRstd;
    GM_ADDR betaEff;
};

// FP32 cache path: all cache pointers are owned, non-aliasing storage supplied by
// the caller, including scratch-backed caches when a public output is omitted.
template<class InputT, class GateT, class BetaT>
class GeneralPrepare {
    using F = AscendC::LocalTensor<float>;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrix_, vectors_;
    F lower_, inverse_, x_, y_, z_, gi_, gj_, accum_, temp_, reduction_, scalar_, last_;
    GeneralPrepareBuffers b_;
    GeneralPrepareChunk c_;

    __aicore__ inline void Barrier() { AscendC::PipeBarrier<PIPE_V>(); }
    __aicore__ inline void ToScalar() {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline void FromScalar() {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
    template<class T>
    __aicore__ inline void Read(F dst, GM_ADDR ptr, uint64_t offset, uint32_t count) {
        AscendC::GlobalTensor<T> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));
        auto transfer = input_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count*sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{true, 0, 0, 0};
        AscendC::DataCopyPad(transfer, gm[offset], copy, pad);
        input_.EnQue(transfer);
        transfer = input_.template DeQue<T>();
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(dst, transfer, 0.0F, count);
        } else {
            AscendC::Cast(dst, transfer, AscendC::RoundMode::CAST_NONE, count);
        }
        Barrier();
        input_.FreeTensor(transfer);
    }
    __aicore__ inline void Write(GM_ADDR ptr, uint64_t offset, F src, uint32_t count) {
        AscendC::GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ptr));
        auto transfer = output_.template AllocTensor<float>();
        AscendC::Adds(transfer, src, 0.0F, count);
        output_.EnQue(transfer);
        transfer = output_.template DeQue<float>();
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count*sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(gm[offset], transfer, copy);
        output_.FreeTensor(transfer);
    }
    __aicore__ inline float Sum(F src, uint32_t count) {
        AscendC::ReduceSum(scalar_, src, reduction_, count);
        ToScalar();
        return scalar_.GetValue(0);
    }
    __aicore__ inline void Exp(F dst, F src, uint32_t count) {
        if (c_.useExp2) {
            AscendC::Muls(dst, src, 0.6931471805599453F, count);
            Barrier();
            AscendC::Exp(dst, dst, count);
        } else {
            AscendC::Exp(dst, src, count);
        }
        Barrier();
    }
    __aicore__ inline void Sigmoid(F dst, uint32_t count) {
        // exp(min(x,0)) / (1+exp(-abs(x))) avoids overflow on either sign.
        AscendC::Abs(temp_, dst, count); Barrier();
        AscendC::Muls(temp_, temp_, -1.0F, count); Barrier();
        AscendC::Exp(temp_, temp_, count); Barrier();
        AscendC::Adds(temp_, temp_, 1.0F, count); Barrier();
        AscendC::Mins(dst, dst, 0.0F, count); Barrier();
        AscendC::Exp(dst, dst, count); Barrier();
        AscendC::Div(dst, dst, temp_, count); Barrier();
    }
    __aicore__ inline void Normalize(F row, GM_ADDR rstd, uint32_t i) {
        float factor = 1.0F;
        if (c_.normalize) {
            AscendC::Mul(temp_, row, row, c_.keyDim); Barrier();
            float square = Sum(temp_, c_.keyDim);
            AscendC::Duplicate(scalar_, square+c_.epsilon, 8); Barrier();
            AscendC::Rsqrt(scalar_, scalar_, 8); ToScalar();
            factor = scalar_.GetValue(0);
            AscendC::Muls(row, row, factor, c_.keyDim); Barrier();
        }
        AscendC::Duplicate(scalar_, factor, 8); Barrier();
        Write(rstd, c_.cacheScalarOffset+i, scalar_, 1);
    }
    template<bool RECOMPUTE = false>
    __aicore__ inline void Inputs() {
        const uint32_t k = c_.keyDim;
        AscendC::Duplicate(accum_, 0.0F, k); Barrier();
        for (uint32_t i=0; i<c_.rows; ++i) {
            if constexpr (!RECOMPUTE) {
                Read<InputT>(x_, b_.q, c_.keyOffset+i*c_.keyStride, k);
                Normalize(x_, b_.qRstd, i);
                Write(b_.qHat, c_.cacheKeyOffset+i*k, x_, k);
                Read<InputT>(x_, b_.k, c_.keyOffset+i*c_.keyStride, k);
                Normalize(x_, b_.kRstd, i);
                Write(b_.kHat, c_.cacheKeyOffset+i*k, x_, k);
            }
            Read<GateT>(x_, b_.gate, c_.keyOffset+i*c_.keyStride, k);
            if (c_.gateMode == 1 || c_.gateMode == 2) {
                if (b_.bias != nullptr) {
                    Read<float>(y_, b_.bias, 0, k);
                    AscendC::Add(x_, x_, y_, k); Barrier();
                }
                float a = 1.0F;
                if (b_.aLog != nullptr) {
                    Read<float>(scalar_, b_.aLog, 0, 1);
                    AscendC::Exp(scalar_, scalar_, 1); ToScalar();
                    a = scalar_.GetValue(0);
                }
                if (c_.gateMode == 2) {
                    AscendC::Muls(x_, x_, a, k); Barrier();
                    Sigmoid(x_, k);
                    AscendC::Muls(x_, x_, c_.lowerBound, k); Barrier();
                } else {
                    // softplus(x) = max(x,0) + log(1+exp(-abs(x))).
                    AscendC::Abs(y_, x_, k); Barrier();
                    AscendC::Muls(y_, y_, -1.0F, k); Barrier();
                    AscendC::Exp(y_, y_, k); Barrier();
                    AscendC::Adds(y_, y_, 1.0F, k); Barrier();
                    AscendC::Ln(y_, y_, k); Barrier();
                    AscendC::Maxs(x_, x_, 0.0F, k); Barrier();
                    AscendC::Add(x_, x_, y_, k); Barrier();
                    AscendC::Muls(x_, x_, -a, k); Barrier();
                }
            }
            if (c_.gateMode == 3) {
                // Recompute without raw gate consumes the saved cumulative G.
                Write(b_.gk, c_.cacheKeyOffset+i*k, x_, k);
            } else {
                if (c_.useExp2) { AscendC::Muls(x_, x_, 1.4426950408889634F, k); Barrier(); }
                AscendC::Add(accum_, accum_, x_, k); Barrier();
                Write(b_.gk, c_.cacheKeyOffset+i*k, accum_, k);
            }
            if constexpr (!RECOMPUTE) {
                Read<BetaT>(x_, b_.beta, c_.betaOffset+i*c_.betaStride, 1);
                if (c_.betaMode != 0) Sigmoid(x_, 1);
                if (c_.betaMode == 2) { AscendC::Muls(x_, x_, 2.0F, 1); Barrier(); }
                Write(b_.betaEff, c_.cacheScalarOffset+i, x_, 1);
            }
        }
        // Subsequent stages consume this owner's completed GM cache writes.
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Scores() {
        const uint32_t k=c_.keyDim;
        AscendC::Duplicate(lower_, 0.0F, 4096);
        AscendC::Duplicate(inverse_, 0.0F, 4096); Barrier();
        for (uint32_t i=0; i<c_.rows; ++i) {
            Read<float>(gi_, b_.gk, c_.cacheKeyOffset+i*k, k);
            Read<float>(x_, b_.qHat, c_.cacheKeyOffset+i*k, k);
            Read<float>(y_, b_.kHat, c_.cacheKeyOffset+i*k, k);
            Read<float>(scalar_, b_.betaEff, c_.cacheScalarOffset+i, 1);
            ToScalar(); float beta=scalar_.GetValue(0);
            AscendC::Duplicate(accum_, 0.0F, 64); Barrier();
            for (uint32_t j=0; j<=i; ++j) {
                Read<float>(gj_, b_.gk, c_.cacheKeyOffset+j*k, k);
                Read<float>(z_, b_.kHat, c_.cacheKeyOffset+j*k, k);
                AscendC::Sub(temp_, gi_, gj_, k); Barrier();
                Exp(temp_, temp_, k);
                AscendC::Mul(z_, z_, temp_, k); Barrier();
                AscendC::Mul(temp_, x_, z_, k); Barrier();
                float score=Sum(temp_, k)*c_.scale;
                accum_.SetValue(j, score);
                if (j<i) {
                    AscendC::Mul(temp_, y_, z_, k); Barrier();
                    float lower=Sum(temp_, k)*beta;
                    lower_.SetValue(i*64+j, lower);
                }
            }
            FromScalar();
            Write(b_.aqk, c_.scoreOffset+i*64, accum_, 64);
        }
        ToScalar();
        for (uint32_t i=0; i<64; ++i) {
            lower_.SetValue(i*64+i, 1.0F);
            inverse_.SetValue(i*64+i, 1.0F);
        }
        FromScalar();
        for (uint32_t i=0; i<c_.rows; ++i) {
            for (uint32_t r=0; r<i; ++r) {
                ToScalar(); float coefficient=-lower_.GetValue(i*64+r);
                AscendC::Muls(temp_, inverse_[r*64], coefficient, 64); Barrier();
                AscendC::Add(inverse_[i*64], inverse_[i*64], temp_, 64); Barrier();
            }
            Write(b_.akk, c_.scoreOffset+i*64, inverse_[i*64], 64);
        }
    }
    __aicore__ inline void Derived() {
        const uint32_t k=c_.keyDim, v=c_.valueDim;
        Read<float>(last_, b_.gk, c_.cacheKeyOffset+(c_.rows-1)*k, k);
        for (uint32_t i=0; i<c_.rows; ++i) {
            Read<float>(gi_, b_.gk, c_.cacheKeyOffset+i*k, k);
            Exp(gi_, gi_, k);
            Read<float>(x_, b_.qHat, c_.cacheKeyOffset+i*k, k);
            AscendC::Mul(x_, x_, gi_, k); Barrier();
            Write(b_.qg, c_.cacheKeyOffset+i*k, x_, k);
            AscendC::Muls(x_, x_, c_.scale, k); Barrier();
            if (b_.qgScaled != nullptr) Write(b_.qgScaled, c_.cacheKeyOffset+i*k, x_, k);
            Read<float>(gi_, b_.gk, c_.cacheKeyOffset+i*k, k);
            AscendC::Sub(gi_, last_, gi_, k); Barrier(); Exp(gi_, gi_, k);
            Read<float>(x_, b_.kHat, c_.cacheKeyOffset+i*k, k);
            AscendC::Mul(x_, x_, gi_, k); Barrier();
            Write(b_.kg, c_.cacheKeyOffset+i*k, x_, k);
            AscendC::Duplicate(accum_, 0.0F, k);
            AscendC::Duplicate(y_, 0.0F, v); Barrier();
            for (uint32_t j=0; j<=i; ++j) {
                Read<float>(scalar_, b_.betaEff, c_.cacheScalarOffset+j, 1);
                ToScalar(); float beta=scalar_.GetValue(0), a=inverse_.GetValue(i*64+j);
                Read<float>(gi_, b_.gk, c_.cacheKeyOffset+j*k, k); Exp(gi_, gi_, k);
                Read<float>(x_, b_.kHat, c_.cacheKeyOffset+j*k, k);
                AscendC::Mul(x_, x_, gi_, k); Barrier();
                AscendC::Muls(x_, x_, beta, k); Barrier();
                AscendC::Muls(x_, x_, a, k); Barrier();
                AscendC::Add(accum_, accum_, x_, k); Barrier();
                Read<InputT>(x_, b_.v, c_.valueOffset+j*c_.valueStride, v);
                AscendC::Muls(x_, x_, beta, v); Barrier();
                AscendC::Muls(x_, x_, a, v); Barrier();
                AscendC::Add(y_, y_, x_, v); Barrier();
            }
            Write(b_.w, c_.cacheKeyOffset+i*k, accum_, k);
            Write(b_.u, c_.cacheValueOffset+i*v, y_, v);
        }
    }
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe) {
        pipe.InitBuffer(input_, 1, 1024);
        pipe.InitBuffer(output_, 1, 1024);
        pipe.InitBuffer(matrix_, 2*64*64*sizeof(float));
        pipe.InitBuffer(vectors_, 10*256*sizeof(float));
        lower_=matrix_.Get<float>(); inverse_=lower_[4096];
        x_=vectors_.Get<float>(); y_=x_[256]; z_=x_[512]; gi_=x_[768]; gj_=x_[1024];
        accum_=x_[1280]; temp_=x_[1536]; reduction_=x_[1792]; scalar_=x_[2048]; last_=x_[2304];
    }
    __aicore__ inline void Process(const GeneralPrepareBuffers &buffers, const GeneralPrepareChunk &chunk) {
        b_=buffers; c_=chunk;
        Inputs<false>(); Scores(); Derived();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    // qHat/kHat and betaEff are saved FP32 inputs; akk is the forward inverse.
    // Never normalize, sigmoid or solve the triangular system again here.
    __aicore__ inline void Recompute(const GeneralPrepareBuffers &buffers, const GeneralPrepareChunk &chunk) {
        b_=buffers; c_=chunk;
        Inputs<true>();
        AscendC::Duplicate(inverse_, 0.0F, 4096); Barrier();
        for (uint32_t i=0; i<c_.rows; ++i)
            Read<float>(inverse_[i*64], b_.akk, c_.scoreOffset+i*64, 64);
        Derived();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace KdaPrepare::Arch22
