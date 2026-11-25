// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
namespace KDA::Arch22 {
struct GeneralFinalizeChunk {
    uint32_t rows, keyDim, valueDim, valueBegin, valueCount;
    uint64_t keyOffset, valueOffset, scoreOffset, stateOffset, outputOffset, outputStride;
    bool stateVFirst;
};
struct GeneralFinalizeBuffers {
    GM_ADDR qgScaled;
    GM_ADDR aqk;
    GM_ADDR vNew;
    GM_ADDR h;
    GM_ADDR o;
};
template<class OutputT> class GeneralFwdFinalize {
    using F=AscendC::LocalTensor<float>;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf_,rowBuf_;
    F state_,row_,intra_,inter_,product_,value_;
    GeneralFinalizeBuffers b_;
    GeneralFinalizeChunk c_;
    __aicore__ inline void Barrier(){AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void ToScalar(){
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline void FromScalar(){
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
    __aicore__ inline void Read(F dst,GM_ADDR ptr,uint64_t offset,uint32_t count){
        AscendC::GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ptr));
        auto transfer=input_.template AllocTensor<float>();
        AscendC::DataCopyExtParams copy{1,static_cast<uint32_t>(count*sizeof(float)),0,0,0};
        AscendC::DataCopyPadExtParams<float> pad{true,0,0,0};
        AscendC::DataCopyPad(transfer,gm[offset],copy,pad);
        input_.EnQue(transfer);transfer=input_.template DeQue<float>();
        AscendC::Adds(dst,transfer,0.0F,count);Barrier();input_.FreeTensor(transfer);
    }
    __aicore__ inline void Write(uint64_t offset,F src,uint32_t count){
        AscendC::GlobalTensor<OutputT> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT *>(b_.o));
        auto transfer=output_.template AllocTensor<OutputT>();
        AscendC::Cast(transfer,src,AscendC::RoundMode::CAST_RINT,count);
        output_.EnQue(transfer);transfer=output_.template DeQue<OutputT>();
        AscendC::DataCopyExtParams copy{1,static_cast<uint32_t>(count*sizeof(OutputT)),0,0,0};
        AscendC::DataCopyPad(gm[offset],transfer,copy);output_.FreeTensor(transfer);
    }
    __aicore__ inline void LoadState(){
        if(!c_.stateVFirst){
            for(uint32_t k=0;k<c_.keyDim;++k)
                Read(state_[k*32],b_.h,c_.stateOffset+k*c_.valueDim+c_.valueBegin,c_.valueCount);
        }else{
            for(uint32_t v=0;v<c_.valueCount;++v){
                Read(row_,b_.h,c_.stateOffset+(c_.valueBegin+v)*c_.keyDim,c_.keyDim);
                ToScalar();
                for(uint32_t k=0;k<c_.keyDim;++k)state_.SetValue(k*32+v,row_.GetValue(k));
            }
            FromScalar();
        }
    }
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe){
        pipe.InitBuffer(input_,1,1024);pipe.InitBuffer(output_,1,1024);
        pipe.InitBuffer(stateBuf_,256*32*sizeof(float));pipe.InitBuffer(rowBuf_,5*256*sizeof(float));
        state_=stateBuf_.Get<float>();row_=rowBuf_.Get<float>();
        intra_=row_[256];inter_=row_[512];product_=row_[768];value_=row_[1024];
    }
    __aicore__ inline void Process(const GeneralFinalizeBuffers &buffers,const GeneralFinalizeChunk &chunk){
        b_=buffers;c_=chunk;LoadState();
        for(uint32_t i=0;i<c_.rows;++i){
            AscendC::Duplicate(inter_,0.0F,c_.valueCount);
            AscendC::Duplicate(intra_,0.0F,c_.valueCount);Barrier();
            Read(row_,b_.qgScaled,c_.keyOffset+i*c_.keyDim,c_.keyDim);
            for(uint32_t k=0;k<c_.keyDim;++k){
                ToScalar();float q=row_.GetValue(k);
                AscendC::Muls(product_,state_[k*32],q,c_.valueCount);Barrier();
                AscendC::Add(inter_,inter_,product_,c_.valueCount);Barrier();
            }
            Read(row_,b_.aqk,c_.scoreOffset+i*64,64);
            for(uint32_t j=0;j<=i;++j){
                ToScalar();float score=row_.GetValue(j);
                Read(value_,b_.vNew,c_.valueOffset+j*c_.valueDim+c_.valueBegin,c_.valueCount);
                AscendC::Muls(product_,value_,score,c_.valueCount);Barrier();
                AscendC::Add(intra_,intra_,product_,c_.valueCount);Barrier();
            }
            AscendC::Add(inter_,inter_,intra_,c_.valueCount);Barrier();
            Write(c_.outputOffset+i*c_.outputStride+c_.valueBegin,inter_,c_.valueCount);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace KDA::Arch22
