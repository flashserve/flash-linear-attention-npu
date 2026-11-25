// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
namespace KDA::Arch22 {
struct GeneralBwdPrepareChunk {
    uint32_t rows,keyDim,valueDim;
    uint64_t scoreOffset,valueOffset,doOffset,doStride,stateOffset,keyOffset;
    float scale;
    bool stateVFirst;
};
struct GeneralBwdPrepareBuffers {
    GM_ADDR aqk;
    GM_ADDR vNew;
    GM_ADDR dO;
    GM_ADDR h;
    GM_ADDR dAqk;
    GM_ADDR dvLocal;
    GM_ADDR dqRaw;
};
template<class GradT> class GeneralBwdPrepare {
    using F=AscendC::LocalTensor<float>;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    F grad_,value_,product_,sum_,row_,reduce_,scalar_,dq_;
    GeneralBwdPrepareBuffers b_;
    GeneralBwdPrepareChunk c_;
    __aicore__ inline void Barrier(){AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void ToScalar(){
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline void FromScalar(){
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
    template<class T> __aicore__ inline void Read(F dst,GM_ADDR ptr,uint64_t offset,uint32_t count){
        AscendC::GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));
        auto transfer=input_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copy{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};
        AscendC::DataCopyPadExtParams<T> pad{true,0,0,0};
        AscendC::DataCopyPad(transfer,gm[offset],copy,pad);
        input_.EnQue(transfer);transfer=input_.template DeQue<T>();
        if constexpr(sizeof(T)==4)AscendC::Adds(dst,transfer,0.0F,count);
        else AscendC::Cast(dst,transfer,AscendC::RoundMode::CAST_NONE,count);
        Barrier();input_.FreeTensor(transfer);
    }
    __aicore__ inline void Write(GM_ADDR ptr,uint64_t offset,F src,uint32_t count){
        AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ptr));
        auto transfer=output_.template AllocTensor<float>();
        AscendC::Adds(transfer,src,0.0F,count);
        output_.EnQue(transfer);transfer=output_.template DeQue<float>();
        AscendC::DataCopyExtParams copy{1,static_cast<uint32_t>(count*sizeof(float)),0,0,0};
        AscendC::DataCopyPad(gm[offset],transfer,copy);output_.FreeTensor(transfer);
    }
    __aicore__ inline float Dot(F a,F b,uint32_t count){
        AscendC::Mul(product_,a,b,count);Barrier();
        AscendC::ReduceSum(scalar_,product_,reduce_,count);ToScalar();
        return scalar_.GetValue(0);
    }
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe){
        pipe.InitBuffer(input_,1,1024);pipe.InitBuffer(output_,1,1024);
        pipe.InitBuffer(buffer_,8*256*sizeof(float));
        grad_=buffer_.Get<float>();value_=grad_[256];product_=grad_[512];sum_=grad_[768];
        row_=grad_[1024];reduce_=grad_[1280];scalar_=grad_[1536];dq_=grad_[1792];
    }
    __aicore__ inline void Process(const GeneralBwdPrepareBuffers &buffers,const GeneralBwdPrepareChunk &chunk){
        b_=buffers;c_=chunk;
        for(uint32_t i=0;i<c_.rows;++i){
            Read<GradT>(grad_,b_.dO,c_.doOffset+i*c_.doStride,c_.valueDim);
            AscendC::Duplicate(row_,0.0F,64);Barrier();
            for(uint32_t j=0;j<=i;++j){
                Read<float>(value_,b_.vNew,c_.valueOffset+j*c_.valueDim,c_.valueDim);
                float score=Dot(grad_,value_,c_.valueDim)*c_.scale;
                row_.SetValue(j,score);
            }
            FromScalar();Write(b_.dAqk,c_.scoreOffset+i*64,row_,64);
            if(!c_.stateVFirst){
                for(uint32_t k=0;k<c_.keyDim;++k){
                    Read<float>(value_,b_.h,c_.stateOffset+k*c_.valueDim,c_.valueDim);
                    float q=Dot(grad_,value_,c_.valueDim);dq_.SetValue(k,q);
                }
                FromScalar();
            }else{
                AscendC::Duplicate(dq_,0.0F,c_.keyDim);Barrier();
                for(uint32_t v=0;v<c_.valueDim;++v){
                    Read<float>(value_,b_.h,c_.stateOffset+v*c_.keyDim,c_.keyDim);
                    ToScalar();float d=grad_.GetValue(v);
                    AscendC::Muls(product_,value_,d,c_.keyDim);Barrier();
                    AscendC::Add(dq_,dq_,product_,c_.keyDim);Barrier();
                }
            }
            Write(b_.dqRaw,c_.keyOffset+i*c_.keyDim,dq_,c_.keyDim);
            AscendC::Duplicate(sum_,0.0F,c_.valueDim);Barrier();
            for(uint32_t j=i;j<c_.rows;++j){
                Read<float>(row_,b_.aqk,c_.scoreOffset+j*64,64);
                ToScalar();float a=row_.GetValue(i);
                Read<GradT>(value_,b_.dO,c_.doOffset+j*c_.doStride,c_.valueDim);
                AscendC::Muls(product_,value_,a,c_.valueDim);Barrier();
                AscendC::Add(sum_,sum_,product_,c_.valueDim);Barrier();
            }
            Write(b_.dvLocal,c_.valueOffset+i*c_.valueDim,sum_,c_.valueDim);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace KDA::Arch22
