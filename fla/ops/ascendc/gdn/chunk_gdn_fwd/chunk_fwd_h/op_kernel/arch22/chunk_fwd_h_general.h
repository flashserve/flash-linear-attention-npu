// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"

namespace GDN::Arch22 {
struct GeneralFwdHSequence {
    uint32_t tokens, keyDim, valueDim, valueBegin, valueCount;
    uint64_t keyOffset, valueOffset, stateOffset, snapshotOffset, snapshotStride;
    bool useExp2, stateVFirst;
};
struct GeneralFwdHBuffers {
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR kg;
    GM_ADDR gk;
    GM_ADDR initialState;
    GM_ADDR h;
    GM_ADDR vNew;
    GM_ADDR finalState;
};

// Each owner holds every K row of one 32-column state slice. A chunk's Vnew
// completes before any state row is updated; h records the FP32 entrance state.
template<class StateT>
class GeneralFwdH {
    using F = AscendC::LocalTensor<float>;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf_, kgBuf_, valueBuf_, rowBuf_;
    F state_, keys_, values_, row_, product_, decay_, accumulator_;
    GeneralFwdHBuffers b_;
    GeneralFwdHSequence c_;
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
        auto transfer=input_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copy{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};
        AscendC::DataCopyPadExtParams<T> pad{true,0,0,0};
        AscendC::DataCopyPad(transfer,gm[offset],copy,pad);
        input_.EnQue(transfer); transfer=input_.template DeQue<T>();
        if constexpr (sizeof(T)==sizeof(float)) AscendC::Adds(dst,transfer,0.0F,count);
        else AscendC::Cast(dst,transfer,AscendC::RoundMode::CAST_NONE,count);
        Barrier(); input_.FreeTensor(transfer);
    }
    template<class T>
    __aicore__ inline void Write(GM_ADDR ptr, uint64_t offset, F src, uint32_t count) {
        AscendC::GlobalTensor<T> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));
        auto transfer=output_.template AllocTensor<T>();
        if constexpr (sizeof(T)==sizeof(float)) AscendC::Adds(transfer,src,0.0F,count);
        else AscendC::Cast(transfer,src,AscendC::RoundMode::CAST_RINT,count);
        output_.EnQue(transfer); transfer=output_.template DeQue<T>();
        AscendC::DataCopyExtParams copy{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};
        AscendC::DataCopyPad(gm[offset],transfer,copy);
        output_.FreeTensor(transfer);
    }
    __aicore__ inline void LoadState() {
        AscendC::Duplicate(state_,0.0F,c_.keyDim*32); Barrier();
        if (b_.initialState==nullptr) return;
        if (!c_.stateVFirst) {
            for (uint32_t k=0;k<c_.keyDim;++k)
                Read<StateT>(state_[k*32],b_.initialState,
                    c_.stateOffset+k*c_.valueDim+c_.valueBegin,c_.valueCount);
        } else {
            for (uint32_t v=0;v<c_.valueCount;++v) {
                Read<StateT>(row_,b_.initialState,
                    c_.stateOffset+(c_.valueBegin+v)*c_.keyDim,c_.keyDim);
                ToScalar();
                for (uint32_t k=0;k<c_.keyDim;++k) state_.SetValue(k*32+v,row_.GetValue(k));
            }
            FromScalar();
        }
    }
    template<class T>
    __aicore__ inline void StoreState(GM_ADDR dst,uint64_t offset) {
        if (!c_.stateVFirst) {
            for (uint32_t k=0;k<c_.keyDim;++k)
                Write<T>(dst,offset+k*c_.valueDim+c_.valueBegin,state_[k*32],c_.valueCount);
        } else {
            ToScalar();
            for (uint32_t v=0;v<c_.valueCount;++v) {
                for (uint32_t k=0;k<c_.keyDim;++k) row_.SetValue(k,state_.GetValue(k*32+v));
                FromScalar();
                Write<T>(dst,offset+(c_.valueBegin+v)*c_.keyDim,row_,c_.keyDim);
                // The output queue copies row_ with Vector before scalar reuse.
                ToScalar();
            }
        }
    }
    __aicore__ inline void Chunk(uint32_t begin,uint32_t rows,uint32_t chunk) {
        StoreState<float>(b_.h,c_.snapshotOffset+chunk*c_.snapshotStride);
        for (uint32_t i=0;i<rows;++i) {
            Read<float>(row_,b_.w,c_.keyOffset+(begin+i)*c_.keyDim,c_.keyDim);
            AscendC::Duplicate(accumulator_,0.0F,c_.valueCount); Barrier();
            for (uint32_t k=0;k<c_.keyDim;++k) {
                ToScalar(); float weight=row_.GetValue(k);
                AscendC::Muls(product_,state_[k*32],weight,c_.valueCount); Barrier();
                AscendC::Add(accumulator_,accumulator_,product_,c_.valueCount); Barrier();
            }
            Read<float>(row_,b_.u,c_.valueOffset+(begin+i)*c_.valueDim+c_.valueBegin,c_.valueCount);
            AscendC::Sub(accumulator_,row_,accumulator_,c_.valueCount); Barrier();
            AscendC::Adds(values_[i*32],accumulator_,0.0F,c_.valueCount); Barrier();
            Write<float>(b_.vNew,c_.valueOffset+(begin+i)*c_.valueDim+c_.valueBegin,accumulator_,c_.valueCount);
            Read<float>(keys_[i*256],b_.kg,c_.keyOffset+(begin+i)*c_.keyDim,c_.keyDim);
        }
        Read<float>(decay_,b_.gk,c_.keyOffset+(begin+rows-1)*c_.keyDim,c_.keyDim);
        if (c_.useExp2) { AscendC::Muls(decay_,decay_,0.6931471805599453F,c_.keyDim); Barrier(); }
        AscendC::Exp(decay_,decay_,c_.keyDim); Barrier();
        for (uint32_t k=0;k<c_.keyDim;++k) {
            ToScalar(); float decay=decay_.GetValue(k);
            AscendC::Duplicate(accumulator_,0.0F,c_.valueCount); Barrier();
            for (uint32_t i=0;i<rows;++i) {
                ToScalar(); float key=keys_.GetValue(i*256+k);
                AscendC::Muls(product_,values_[i*32],key,c_.valueCount); Barrier();
                AscendC::Add(accumulator_,accumulator_,product_,c_.valueCount); Barrier();
            }
            AscendC::Muls(product_,state_[k*32],decay,c_.valueCount); Barrier();
            AscendC::Add(state_[k*32],product_,accumulator_,c_.valueCount); Barrier();
        }
    }
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe) {
        pipe.InitBuffer(input_,1,1024); pipe.InitBuffer(output_,1,1024);
        pipe.InitBuffer(stateBuf_,256*32*sizeof(float));
        pipe.InitBuffer(kgBuf_,64*256*sizeof(float));
        pipe.InitBuffer(valueBuf_,64*32*sizeof(float));
        pipe.InitBuffer(rowBuf_,4*256*sizeof(float));
        state_=stateBuf_.Get<float>(); keys_=kgBuf_.Get<float>(); values_=valueBuf_.Get<float>();
        row_=rowBuf_.Get<float>(); product_=row_[256]; decay_=row_[512]; accumulator_=row_[768];
    }
    __aicore__ inline void Process(const GeneralFwdHBuffers &buffers,const GeneralFwdHSequence &sequence) {
        b_=buffers; c_=sequence; LoadState();
        for (uint32_t begin=0,chunk=0;begin<c_.tokens;begin+=64,++chunk)
            Chunk(begin,(c_.tokens-begin<64 ? c_.tokens-begin : 64U),chunk);
        if (b_.finalState!=nullptr) StoreState<StateT>(b_.finalState,c_.stateOffset);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace GDN::Arch22
