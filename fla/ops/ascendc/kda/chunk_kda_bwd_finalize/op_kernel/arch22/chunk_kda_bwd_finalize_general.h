// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
namespace KDA::Arch22 {
struct GeneralBwdFinalizeChunk {
    uint32_t rows,keyDim,valueDim;
    uint64_t keyOffset,valueOffset,scalarOffset,scoreOffset,stateOffset;
    uint64_t inputKeyOffset,inputValueOffset,inputScalarOffset,keyStride,valueStride,scalarStride;
    float scale,lowerBound;
    bool normalize,useExp2,stateVFirst;
    uint32_t gateMode,betaMode;
};
struct GeneralBwdFinalizeBuffers {
    GM_ADDR qHat;
    GM_ADDR kHat;
    GM_ADDR v;
    GM_ADDR gk;
    GM_ADDR betaEff;
    GM_ADDR akk;
    GM_ADDR vNew;
    GM_ADDR h;
    GM_ADDR dh;
    GM_ADDR dvScan;
    GM_ADDR dAqk;
    GM_ADDR dqRaw;
    GM_ADDR qRstd;
    GM_ADDR kRstd;
    GM_ADDR rawGate;
    GM_ADDR aLog;
    GM_ADDR bias;
    GM_ADDR scratchW;
    GM_ADDR scratchQ;
    GM_ADDR scratchK;
    GM_ADDR scratchG;
    GM_ADDR scratchBeta;
    GM_ADDR dq;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR dg;
    GM_ADDR dbeta;
    GM_ADDR partialA;
    GM_ADDR partialBias;
};
// Each (chunk,head) owns its scratch and all input gradients. Parameter
// partials are private; a separate deterministic reduction combines chunks.
// UB = 4*64*64*4 + 24*256*4 + 2*1024 = 92160 bytes.
template<class InputT,class GateT,class BetaT>
class GeneralBwdFinalize {
    using F=AscendC::LocalTensor<float>;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrices_,vectors_;
    F inverse_,dA_,middle_,dL_;
    F x_,y_,z_,gi_,gj_,ex_,product_,reduce_,scalar_,acc_,dq_,dk_,dg_,dw_,du_,last_,lastGrad_,biasGrad_,stateRow_,transferRow_,aq_,ki_,qi_,suffix_;
    GeneralBwdFinalizeBuffers b_;
    GeneralBwdFinalizeChunk c_;
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

    __aicore__ inline float Sum(F a,uint32_t n){
        AscendC::ReduceSum(scalar_,a,reduce_,n);ToScalar();return scalar_.GetValue(0);
    }
    __aicore__ inline float Dot(F a,F b,uint32_t n){
        AscendC::Mul(product_,a,b,n);Barrier();return Sum(product_,n);
    }
    __aicore__ inline void Exp(F dst,F src,uint32_t n){
        if(c_.useExp2){AscendC::Muls(dst,src,0.6931471805599453F,n);Barrier();AscendC::Exp(dst,dst,n);}
        else AscendC::Exp(dst,src,n);
        Barrier();
    }
    __aicore__ inline float Beta(uint32_t i){
        Read<float>(scalar_,b_.betaEff,c_.scalarOffset+i,1);ToScalar();return scalar_.GetValue(0);
    }
    __aicore__ inline void StateRow(GM_ADDR ptr,uint32_t k){
        if(!c_.stateVFirst)Read<float>(stateRow_,ptr,c_.stateOffset+k*c_.valueDim,c_.valueDim);
        else{
            for(uint32_t v=0;v<c_.valueDim;++v){
                Read<float>(transferRow_,ptr,c_.stateOffset+v*c_.keyDim,c_.keyDim);
                ToScalar();stateRow_.SetValue(v,transferRow_.GetValue(k));
            }
            FromScalar();
        }
    }
    __aicore__ inline void AddScratch(GM_ADDR ptr,uint32_t i,F delta){
        const uint64_t offset=c_.keyOffset+i*c_.keyDim;
        AscendC::PipeBarrier<PIPE_ALL>();
        Read<float>(acc_,ptr,offset,c_.keyDim);
        AscendC::Add(acc_,acc_,delta,c_.keyDim);Barrier();Write<float>(ptr,offset,acc_,c_.keyDim);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void MatrixGradients(){
        const uint32_t K=c_.keyDim,V=c_.valueDim,M=c_.rows;
        AscendC::Duplicate(inverse_,0.0F,4096);AscendC::Duplicate(dA_,0.0F,4096);
        AscendC::Duplicate(middle_,0.0F,4096);AscendC::Duplicate(dL_,0.0F,4096);Barrier();
        for(uint32_t i=0;i<M;++i){
            Read<float>(inverse_[i*64],b_.akk,c_.scoreOffset+i*64,64);
            Read<float>(du_,b_.dvScan,c_.valueOffset+i*V,V);
            for(uint32_t k=0;k<K;++k){
                StateRow(b_.h,k);float d=-Dot(du_,stateRow_,V);dw_.SetValue(k,d);
            }
            FromScalar();Write<float>(b_.scratchW,c_.keyOffset+i*K,dw_,K);
            for(uint32_t j=0;j<M;++j){
                float beta=Beta(j);
                Read<float>(gi_,b_.gk,c_.keyOffset+j*K,K);Exp(ex_,gi_,K);
                Read<float>(x_,b_.kHat,c_.keyOffset+j*K,K);
                AscendC::Mul(x_,x_,ex_,K);Barrier();AscendC::Muls(x_,x_,beta,K);Barrier();
                float da=Dot(dw_,x_,K);
                Read<InputT>(x_,b_.v,c_.inputValueOffset+j*c_.valueStride,V);
                AscendC::Muls(x_,x_,beta,V);Barrier();da+=Dot(du_,x_,V);
                dA_.SetValue(i*64+j,da);
            }
        }
        FromScalar();
        for(uint32_t i=0;i<M;++i){
            for(uint32_t r=i;r<M;++r){
                ToScalar();float a=inverse_.GetValue(r*64+i);
                AscendC::Muls(x_,dA_[r*64],a,64);Barrier();
                AscendC::Add(middle_[i*64],middle_[i*64],x_,64);Barrier();
            }
            for(uint32_t j=0;j<i;++j){
                float dl=-Dot(middle_[i*64],inverse_[j*64],M);
                dL_.SetValue(i*64+j,dl);
            }
        }
        FromScalar();AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void DirectGradients(){
        const uint32_t K=c_.keyDim,V=c_.valueDim,M=c_.rows;
        const float base=c_.useExp2?0.6931471805599453F:1.0F;
        Read<float>(last_,b_.gk,c_.keyOffset+(M-1)*K,K);
        AscendC::Duplicate(lastGrad_,0.0F,K);Barrier();
        for(uint32_t i=0;i<M;++i){
            AscendC::Duplicate(dw_,0.0F,K);AscendC::Duplicate(du_,0.0F,V);Barrier();
            for(uint32_t r=i;r<M;++r){
                ToScalar();float a=inverse_.GetValue(r*64+i);
                Read<float>(x_,b_.scratchW,c_.keyOffset+r*K,K);
                AscendC::Muls(x_,x_,a,K);Barrier();AscendC::Add(dw_,dw_,x_,K);Barrier();
                Read<float>(x_,b_.dvScan,c_.valueOffset+r*V,V);
                AscendC::Muls(x_,x_,a,V);Barrier();AscendC::Add(du_,du_,x_,V);Barrier();
            }
            float beta=Beta(i);
            AscendC::Muls(x_,du_,beta,V);Barrier();Write<InputT>(b_.dv,c_.inputValueOffset+i*c_.valueStride,x_,V);
            Read<InputT>(x_,b_.v,c_.inputValueOffset+i*c_.valueStride,V);
            float db=Dot(du_,x_,V);
            Read<float>(gi_,b_.gk,c_.keyOffset+i*K,K);Exp(ex_,gi_,K);
            Read<float>(ki_,b_.kHat,c_.keyOffset+i*K,K);
            AscendC::Mul(x_,ki_,ex_,K);Barrier();db+=Dot(dw_,x_,K);
            AscendC::Duplicate(scalar_,db,8);Barrier();Write<float>(b_.scratchBeta,c_.scalarOffset+i,scalar_,1);
            AscendC::Mul(dk_,dw_,ex_,K);Barrier();AscendC::Muls(dk_,dk_,beta,K);Barrier();
            Read<float>(dq_,b_.dqRaw,c_.keyOffset+i*K,K);
            AscendC::Muls(dq_,dq_,c_.scale,K);Barrier();AscendC::Mul(dq_,dq_,ex_,K);Barrier();
            Read<float>(qi_,b_.qHat,c_.keyOffset+i*K,K);
            AscendC::Mul(dg_,dq_,qi_,K);Barrier();AscendC::Mul(x_,dk_,ki_,K);Barrier();
            AscendC::Add(dg_,dg_,x_,K);Barrier();AscendC::Muls(dg_,dg_,base,K);Barrier();
            Read<float>(du_,b_.vNew,c_.valueOffset+i*V,V);
            for(uint32_t k=0;k<K;++k){
                StateRow(b_.dh,k);float d=Dot(du_,stateRow_,V);dw_.SetValue(k,d);
            }
            FromScalar();AscendC::Sub(ex_,last_,gi_,K);Barrier();Exp(ex_,ex_,K);
            AscendC::Mul(x_,dw_,ex_,K);Barrier();AscendC::Add(dk_,dk_,x_,K);Barrier();
            if(i+1<M){
                AscendC::Mul(x_,x_,ki_,K);Barrier();AscendC::Muls(x_,x_,base,K);Barrier();
                AscendC::Sub(dg_,dg_,x_,K);Barrier();AscendC::Add(lastGrad_,lastGrad_,x_,K);Barrier();
            }
            Write<float>(b_.scratchQ,c_.keyOffset+i*K,dq_,K);
            Write<float>(b_.scratchK,c_.keyOffset+i*K,dk_,K);
            Write<float>(b_.scratchG,c_.keyOffset+i*K,dg_,K);
        }
        Exp(ex_,last_,K);
        for(uint32_t k=0;k<K;++k){
            StateRow(b_.h,k);AscendC::Adds(du_,stateRow_,0.0F,V);Barrier();
            StateRow(b_.dh,k);float d=Dot(du_,stateRow_,V);
            ToScalar();lastGrad_.SetValue(k,lastGrad_.GetValue(k)+base*ex_.GetValue(k)*d);
        }
        FromScalar();AddScratch(b_.scratchG,M-1,lastGrad_);
    }
    __aicore__ inline void ScoreGradients(){
        const uint32_t K=c_.keyDim,M=c_.rows;
        const float base=c_.useExp2?0.6931471805599453F:1.0F;
        for(uint32_t i=0;i<M;++i){
            float beta=Beta(i);
            Read<float>(qi_,b_.qHat,c_.keyOffset+i*K,K);
            Read<float>(ki_,b_.kHat,c_.keyOffset+i*K,K);
            Read<float>(gi_,b_.gk,c_.keyOffset+i*K,K);
            Read<float>(aq_,b_.dAqk,c_.scoreOffset+i*64,64);
            float db=0.0F;
            for(uint32_t j=0;j<=i;++j){
                ToScalar();float a=aq_.GetValue(j),l=dL_.GetValue(i*64+j);
                Read<float>(z_,b_.kHat,c_.keyOffset+j*K,K);
                Read<float>(gj_,b_.gk,c_.keyOffset+j*K,K);
                AscendC::Sub(ex_,gi_,gj_,K);Barrier();Exp(ex_,ex_,K);
                AscendC::Mul(y_,z_,ex_,K);Barrier();
                AscendC::Muls(dq_,y_,a,K);Barrier();AddScratch(b_.scratchQ,i,dq_);
                AscendC::Muls(dk_,qi_,a,K);Barrier();
                if(j<i){
                    db+=l*Dot(ki_,y_,K);
                    AscendC::Muls(x_,ki_,l*beta,K);Barrier();AscendC::Add(dk_,dk_,x_,K);Barrier();
                    AscendC::Muls(x_,y_,l*beta,K);Barrier();AddScratch(b_.scratchK,i,x_);
                }
                AscendC::Mul(dk_,dk_,ex_,K);Barrier();AddScratch(b_.scratchK,j,dk_);
                if(j<i){
                    AscendC::Mul(dg_,dk_,z_,K);Barrier();AscendC::Muls(dg_,dg_,base,K);Barrier();
                    AddScratch(b_.scratchG,i,dg_);
                    AscendC::Muls(dg_,dg_,-1.0F,K);Barrier();AddScratch(b_.scratchG,j,dg_);
                }
            }
            Read<float>(scalar_,b_.scratchBeta,c_.scalarOffset+i,1);ToScalar();db+=scalar_.GetValue(0);
            if(c_.betaMode==1)db*=beta*(1.0F-beta);
            else if(c_.betaMode==2)db*=beta*(1.0F-0.5F*beta);
            AscendC::Duplicate(scalar_,db,8);Barrier();Write<BetaT>(b_.dbeta,c_.inputScalarOffset+i*c_.scalarStride,scalar_,1);
        }
    }
    __aicore__ inline void NormGradient(GM_ADDR saved,GM_ADDR scratch,GM_ADDR rstd,GM_ADDR out,uint32_t i){
        const uint32_t K=c_.keyDim;
        Read<float>(x_,scratch,c_.keyOffset+i*K,K);
        if(c_.normalize){
            Read<float>(y_,saved,c_.keyOffset+i*K,K);float projection=Dot(x_,y_,K);
            AscendC::Muls(y_,y_,projection,K);Barrier();AscendC::Sub(x_,x_,y_,K);Barrier();
            Read<float>(scalar_,rstd,c_.scalarOffset+i,1);ToScalar();float inv=scalar_.GetValue(0);
            AscendC::Muls(x_,x_,inv,K);Barrier();
        }
        Write<InputT>(out,c_.inputKeyOffset+i*c_.keyStride,x_,K);
    }
    __aicore__ inline void GateAndNorm(){
        const uint32_t K=c_.keyDim;
        AscendC::Duplicate(suffix_,0.0F,K);AscendC::Duplicate(biasGrad_,0.0F,K);Barrier();
        float da=0.0F,a=1.0F;
        if(c_.gateMode!=0 && b_.aLog!=nullptr){
            Read<float>(scalar_,b_.aLog,0,1);AscendC::Exp(scalar_,scalar_,1);ToScalar();a=scalar_.GetValue(0);
        }
        for(uint32_t n=c_.rows;n>0;--n){
            uint32_t i=n-1;
            Read<float>(dg_,b_.scratchG,c_.keyOffset+i*K,K);
            AscendC::Add(suffix_,suffix_,dg_,K);Barrier();
            AscendC::Muls(dg_,suffix_,c_.useExp2?1.4426950408889634F:1.0F,K);Barrier();
            if(c_.gateMode!=0){
                Read<GateT>(x_,b_.rawGate,c_.inputKeyOffset+i*c_.keyStride,K);
                if(b_.bias!=nullptr){Read<float>(y_,b_.bias,0,K);AscendC::Add(x_,x_,y_,K);Barrier();}
                if(c_.gateMode==2){
                    AscendC::Muls(y_,x_,a,K);Barrier();AscendC::Abs(y_,y_,K);Barrier();
                    AscendC::Muls(y_,y_,-1.0F,K);Barrier();AscendC::Exp(y_,y_,K);Barrier();
                    AscendC::Adds(z_,y_,1.0F,K);Barrier();AscendC::Mul(z_,z_,z_,K);Barrier();
                    AscendC::Div(y_,y_,z_,K);Barrier();AscendC::Muls(y_,y_,c_.lowerBound*a,K);Barrier();
                    AscendC::Mul(dg_,dg_,y_,K);Barrier();da+=Dot(dg_,x_,K);
                }else{
                    // Stable softplus and sigmoid; retained for standalone parity.
                    AscendC::Abs(y_,x_,K);Barrier();AscendC::Muls(y_,y_,-1.0F,K);Barrier();AscendC::Exp(y_,y_,K);Barrier();
                    AscendC::Adds(z_,y_,1.0F,K);Barrier();AscendC::Ln(y_,z_,K);Barrier();
                    AscendC::Maxs(ex_,x_,0.0F,K);Barrier();AscendC::Add(y_,y_,ex_,K);Barrier();
                    AscendC::Muls(y_,y_,-a,K);Barrier();da+=Dot(dg_,y_,K);
                    AscendC::Mins(y_,x_,0.0F,K);Barrier();AscendC::Exp(y_,y_,K);Barrier();AscendC::Div(y_,y_,z_,K);Barrier();
                    AscendC::Muls(y_,y_,-a,K);Barrier();AscendC::Mul(dg_,dg_,y_,K);Barrier();
                }
                AscendC::Add(biasGrad_,biasGrad_,dg_,K);Barrier();
            }
            Write<GateT>(b_.dg,c_.inputKeyOffset+i*c_.keyStride,dg_,K);
            NormGradient(b_.qHat,b_.scratchQ,b_.qRstd,b_.dq,i);
            NormGradient(b_.kHat,b_.scratchK,b_.kRstd,b_.dk,i);
        }
        if(b_.partialA!=nullptr){AscendC::Duplicate(scalar_,da,8);Barrier();Write<float>(b_.partialA,0,scalar_,1);}
        if(b_.partialBias!=nullptr)Write<float>(b_.partialBias,0,biasGrad_,K);
    }
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe){
        pipe.InitBuffer(input_,1,1024);pipe.InitBuffer(output_,1,1024);
        pipe.InitBuffer(matrices_,4*4096*sizeof(float));pipe.InitBuffer(vectors_,24*256*sizeof(float));
        inverse_=matrices_.Get<float>();dA_=inverse_[4096];middle_=inverse_[8192];dL_=inverse_[12288];
        x_=vectors_.Get<float>();y_=x_[256];z_=x_[512];gi_=x_[768];gj_=x_[1024];ex_=x_[1280];
        product_=x_[1536];reduce_=x_[1792];scalar_=x_[2048];acc_=x_[2304];dq_=x_[2560];dk_=x_[2816];
        dg_=x_[3072];dw_=x_[3328];du_=x_[3584];last_=x_[3840];lastGrad_=x_[4096];biasGrad_=x_[4352];
        stateRow_=x_[4608];transferRow_=x_[4864];aq_=x_[5120];ki_=x_[5376];qi_=x_[5632];suffix_=x_[5888];
    }
    __aicore__ inline void ReduceParameters(uint32_t count,uint32_t heads,uint32_t head,uint32_t keys,
        GM_ADDR partialA,GM_ADDR partialBias,GM_ADDR outA,GM_ADDR outBias){
        float da=0.0F;AscendC::Duplicate(acc_,0.0F,keys);Barrier();
        for(uint32_t i=0;i<count;++i){
            if(outA!=nullptr){Read<float>(scalar_,partialA,i*heads+head,1);ToScalar();da+=scalar_.GetValue(0);}
            if(outBias!=nullptr){Read<float>(x_,partialBias,(i*heads+head)*keys,keys);AscendC::Add(acc_,acc_,x_,keys);Barrier();}
        }
        if(outA!=nullptr){AscendC::Duplicate(scalar_,da,8);Barrier();Write<float>(outA,head,scalar_,1);}
        if(outBias!=nullptr)Write<float>(outBias,head*keys,acc_,keys);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Process(const GeneralBwdFinalizeBuffers &buffers,const GeneralBwdFinalizeChunk &chunk){
        b_=buffers;c_=chunk;MatrixGradients();DirectGradients();ScoreGradients();
        AscendC::PipeBarrier<PIPE_ALL>();GateAndNorm();AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace KDA::Arch22
