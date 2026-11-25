// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
namespace GDN::Arch22 {
struct GeneralDhuSequence {
    uint32_t tokens,keyDim,valueDim,valueBegin,valueCount;
    uint64_t keyOffset,valueOffset,doOffset,doStride,stateOffset,snapshotOffset,snapshotStride;
    float scale;
    bool useExp2,stateVFirst;
};
struct GeneralDhuBuffers {
    GM_ADDR qg;
    GM_ADDR kg;
    GM_ADDR w;
    GM_ADDR gk;
    GM_ADDR dO;
    GM_ADDR dvLocal;
    GM_ADDR dht;
    GM_ADDR dh;
    GM_ADDR dv;
    GM_ADDR dh0;
};
// One (sequence,head,V-slice) owner keeps every K row. No cross-owner
// reductions or shared result slots. UB: 3*256*32*4+6*256*4+2048=106496 B.
template<class GradT>
class GeneralDhu {
    using F=AscendC::LocalTensor<float>;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> states_,rows_;
    F state_,termQ_,termW_,row_,product_,decay_,dv_,grad_,weights_;
    GeneralDhuBuffers b_;
    GeneralDhuSequence c_;
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
        if(b_.dht==nullptr)return;
        if(!c_.stateVFirst){
            for(uint32_t k=0;k<c_.keyDim;++k)
                Read<float>(state_[k*32],b_.dht,c_.stateOffset+k*c_.valueDim+c_.valueBegin,c_.valueCount);
        }else{
            for(uint32_t v=0;v<c_.valueCount;++v){
                Read<float>(row_,b_.dht,c_.stateOffset+(c_.valueBegin+v)*c_.keyDim,c_.keyDim);
                ToScalar();
                for(uint32_t k=0;k<c_.keyDim;++k)state_.SetValue(k*32+v,row_.GetValue(k));
            }
            FromScalar();
        }
    }
    __aicore__ inline void StoreState(GM_ADDR dst,uint64_t offset){
        if(!c_.stateVFirst){
            for(uint32_t k=0;k<c_.keyDim;++k)
                Write<float>(dst,offset+k*c_.valueDim+c_.valueBegin,state_[k*32],c_.valueCount);
        }else{
            ToScalar();
            for(uint32_t v=0;v<c_.valueCount;++v){
                for(uint32_t k=0;k<c_.keyDim;++k)row_.SetValue(k,state_.GetValue(k*32+v));
                FromScalar();Write<float>(dst,offset+(c_.valueBegin+v)*c_.keyDim,row_,c_.keyDim);
                ToScalar();
            }
        }
    }
    __aicore__ inline void Chunk(uint32_t begin,uint32_t rows,uint32_t chunk){
        // Finalize consumes Dnext, not the updated entrance adjoint Dprev.
        StoreState(b_.dh,c_.snapshotOffset+chunk*c_.snapshotStride);
        AscendC::Duplicate(termQ_,0.0F,c_.keyDim*32);
        AscendC::Duplicate(termW_,0.0F,c_.keyDim*32);Barrier();
        for(uint32_t i=0;i<rows;++i){
            const uint64_t ko=c_.keyOffset+(begin+i)*c_.keyDim;
            const uint64_t vo=c_.valueOffset+(begin+i)*c_.valueDim+c_.valueBegin;
            Read<float>(row_,b_.kg,ko,c_.keyDim);
            AscendC::Duplicate(dv_,0.0F,c_.valueCount);Barrier();
            for(uint32_t k=0;k<c_.keyDim;++k){
                ToScalar();float key=row_.GetValue(k);
                AscendC::Muls(product_,state_[k*32],key,c_.valueCount);Barrier();
                AscendC::Add(dv_,dv_,product_,c_.valueCount);Barrier();
            }
            Read<float>(product_,b_.dvLocal,vo,c_.valueCount);
            AscendC::Add(dv_,dv_,product_,c_.valueCount);Barrier();
            Write<float>(b_.dv,vo,dv_,c_.valueCount);
            Read<GradT>(grad_,b_.dO,c_.doOffset+(begin+i)*c_.doStride+c_.valueBegin,c_.valueCount);
            Read<float>(row_,b_.qg,ko,c_.keyDim);
            Read<float>(weights_,b_.w,ko,c_.keyDim);
            for(uint32_t k=0;k<c_.keyDim;++k){
                ToScalar();float q=row_.GetValue(k),w=weights_.GetValue(k);
                AscendC::Muls(product_,grad_,q,c_.valueCount);Barrier();
                AscendC::Add(termQ_[k*32],termQ_[k*32],product_,c_.valueCount);Barrier();
                AscendC::Muls(product_,dv_,w,c_.valueCount);Barrier();
                AscendC::Add(termW_[k*32],termW_[k*32],product_,c_.valueCount);Barrier();
            }
        }
        Read<float>(decay_,b_.gk,c_.keyOffset+(begin+rows-1)*c_.keyDim,c_.keyDim);
        if(c_.useExp2){AscendC::Muls(decay_,decay_,0.6931471805599453F,c_.keyDim);Barrier();}
        AscendC::Exp(decay_,decay_,c_.keyDim);Barrier();
        for(uint32_t k=0;k<c_.keyDim;++k){
            ToScalar();float decay=decay_.GetValue(k);
            AscendC::Muls(product_,state_[k*32],decay,c_.valueCount);Barrier();
            // Scale belongs after the complete token reduction of qg.T @ dO.
            AscendC::Muls(grad_,termQ_[k*32],c_.scale,c_.valueCount);Barrier();
            AscendC::Add(product_,product_,grad_,c_.valueCount);Barrier();
            AscendC::Sub(state_[k*32],product_,termW_[k*32],c_.valueCount);Barrier();
        }
    }
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe){
        pipe.InitBuffer(input_,1,1024);pipe.InitBuffer(output_,1,1024);
        pipe.InitBuffer(states_,3*256*32*sizeof(float));pipe.InitBuffer(rows_,6*256*sizeof(float));
        state_=states_.Get<float>();termQ_=state_[8192];termW_=state_[16384];
        row_=rows_.Get<float>();product_=row_[256];decay_=row_[512];dv_=row_[768];grad_=row_[1024];weights_=row_[1280];
    }
    __aicore__ inline void Process(const GeneralDhuBuffers &buffers,const GeneralDhuSequence &sequence){
        b_=buffers;c_=sequence;LoadState();
        uint32_t chunks=(c_.tokens+63)/64;
        for(uint32_t n=chunks;n>0;--n){
            uint32_t begin=(n-1)*64;
            Chunk(begin,c_.tokens-begin<64?c_.tokens-begin:64,n-1);
        }
        if(b_.dh0!=nullptr)StoreState(b_.dh0,c_.stateOffset);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace GDN::Arch22
