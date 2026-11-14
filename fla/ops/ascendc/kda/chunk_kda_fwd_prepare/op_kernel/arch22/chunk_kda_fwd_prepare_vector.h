// Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_kda_fwd_prepare_common.h"

namespace KdaPrepare::Arch22 {
template<uint32_t Norm,uint32_t Gate,class GateT,class BetaT> class Vector {
    const PrepareKernelArgs &a_; TPipe *pipe_; uint32_t core_,lane_;
    TBuf<TPosition::VECCALC> ub_;
    LocalTensor<bfloat16_t> q_,k_,v_;
    LocalTensor<uint8_t> triangle_;
    LocalTensor<float> g_,work_,small_,qr_,kr_,beta_,bias_,blocks_;
    Work w_; uint64_t base_;
    bool aLogCached_=false;
    template<class T> __aicore__ inline void BindTensor(LocalTensor<T> &dst,
        const LocalTensor<uint8_t> &allocation,uint32_t offset,uint32_t bytes) {
        TBuffAddr address{};
        address.logicPos=static_cast<uint8_t>(TPosition::VECCALC);
        address.bufferHandle=allocation.GetBufferHandle();
        address.bufferAddr=static_cast<uint32_t>(allocation.GetPhyAddr())+offset;
        address.dataLen=bytes;
        dst.SetAddr(address);
        // Materialize all descriptor fields before cross-stage scalar readers.
        volatile TBuffAddr &stored=dst.address_;
        stored.dataLen=address.dataLen;stored.bufferAddr=address.bufferAddr;
        stored.bufferHandle=address.bufferHandle;stored.logicPos=address.logicPos;
        dst.SetShapeInfo(ShapeInfo{});
    }
    template<class T> __aicore__ inline void Read(LocalTensor<T> dst,GM_ADDR ptr,
        uint64_t offset,uint32_t rows,uint32_t width,uint32_t heads,bool tokenMajor=false,bool streaming=false) {
        // A DMA must not reuse UB while an earlier vector producer still writes it.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams p{static_cast<uint16_t>(rows),static_cast<uint32_t>(width*sizeof(T)),
            static_cast<uint32_t>(tokenMajor?(heads-1)*width*sizeof(T):0),0,0};
        DataCopyPadExtParams<T> pad{false,0,0,0};
        auto source=Gm<T>(ptr,offset);
        if(streaming)source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad(dst,source,p,pad);
    }
    __aicore__ inline void DrainStores() {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    template<class T> __aicore__ inline void Store(GM_ADDR ptr,uint64_t offset,
        LocalTensor<T> src,uint32_t count) {
        DataCopyExtParams p{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};
        DataCopyPad(Gm<T>(ptr,offset),src,p);
    }
    template<class T> __aicore__ inline void Write(GM_ADDR ptr,uint64_t offset,
        LocalTensor<T> src,uint32_t count,bool drain=true) {
        VToMte3();Store(ptr,offset,src,count);if(drain)DrainStores();
    }
    __aicore__ inline void RowMultiply(LocalTensor<float> dst,LocalTensor<float> src,
        LocalTensor<float> broadcast,uint32_t rows,uint32_t width) {
        BinaryRepeatParams p{1,1,0,static_cast<uint8_t>(width/8),
            static_cast<uint8_t>(width/8),1};
        Mul(dst,src,broadcast,64,rows,p);
        if(width==128) Mul(dst[64],src[64],broadcast,64,rows,p);
        VBarrier();
    }
    __aicore__ inline void Normalize(LocalTensor<bfloat16_t> src,LocalTensor<float> rstd,bool onesReady=false) {
        if constexpr(Norm!=1) {Duplicate(rstd,1.0F,64);VBarrier();}
        else if(w_.rows<64) {Duplicate(rstd,1.0F,64);VBarrier();}
        if constexpr(Norm==1) {
            auto values=work_,squares=work_[8192],ones=small_[1536];
            uint32_t rows=w_.rows,count=rows*128;
            Cast(values,src,RoundMode::CAST_NONE,count);VBarrier();
            Mul(squares,values,values,count);VBarrier();
            BinaryRepeatParams pairs{1,1,1,16,16,16};
            Add(squares,squares,squares[64],64,rows,pairs);VBarrier();
            WholeReduceSum(rstd,squares,64,rows,1,1,16);VBarrier();
            Adds(rstd,rstd,a_.tiling.epsilon,rows);VBarrier();
            Sqrt(rstd,rstd,rows);VBarrier();
            if(!onesReady) {Duplicate(ones,1.0F,64);VBarrier();}
            Div(rstd,ones,rstd,rows);VBarrier();
            Brcb(blocks_,rstd,8,{1,8});VBarrier();
            RowMultiply(values,values,blocks_,rows,128);
            Cast(src,values,RoundMode::CAST_RINT,count);VBarrier();
        }
    }
    __aicore__ inline void Prepare() {
        // Full-row DMA/Cast overwrites every consumed input element.
        // Partial chunks retain the original complete zero padding.
        if(w_.rows<64) {
            Duplicate(q_,bfloat16_t(0),8192);Duplicate(k_,bfloat16_t(0),8192);
            Duplicate(g_,0.0F,8192);Duplicate(beta_,0.0F,64);
        }
        VBarrier();
        const auto &t=a_.tiling;uint32_t h=w_.head[lane_],qh=w_.qh[lane_];
        Read(q_,a_.q,QkvInputOffset(t,w_,qh,t.qkHeadNum,128),w_.rows,128,t.qkHeadNum,t.inputSequenceMajor,t.qkHeadNum==t.valueHeadNum);
        Read(k_,a_.k,QkvInputOffset(t,w_,qh,t.qkHeadNum,128),w_.rows,128,t.qkHeadNum,t.inputSequenceMajor,t.qkHeadNum==t.valueHeadNum);
        // Input types are selected by the package's existing dtype objects,
        // independently of the six Norm/Gate tiling keys.
        auto gateRaw=work_.template ReinterpretCast<GateT>();
        if constexpr(sizeof(GateT)==sizeof(float)) {
            Read(g_,a_.rawGate,GateInputOffset(t,w_,h,128),w_.rows,128,t.valueHeadNum,t.inputSequenceMajor,true);
        } else {
            // work_ is free until Normalize/GatePrefix. Do not expand in-place.
            Read(gateRaw,a_.rawGate,GateInputOffset(t,w_,h,128),w_.rows,128,t.valueHeadNum,t.inputSequenceMajor,true);
        }
        auto raw=small_[2048].template ReinterpretCast<BetaT>();
        if constexpr(sizeof(BetaT)==2) {
            // Define complete raw/packed conversion blocks, including inactive lanes.
            Duplicate(small_[2048],0.0F,544);VBarrier();
        }
        if(!t.inputSequenceMajor) {
            // Head-major beta is contiguous: FP32 DMA writes beta_ directly;
            // BF16 retains the existing raw scratch and conversion boundary.
            if constexpr(sizeof(BetaT)==sizeof(float)) {
                Read(beta_,a_.beta,BetaInputOffset(t,w_,h),1,w_.rows,1,false);
            } else {
                Read(raw,a_.beta,BetaInputOffset(t,w_,h),1,w_.rows,1,false);
            }
        } else {
            // beta alone has scalar-width rows. Read a bounded head tile,
            // then select this lane's head; never materialize the full tensor.
            constexpr uint32_t BetaTileHeads=8;
            const uint32_t heads=t.valueHeadNum;
            if(heads<=BetaTileHeads) {
                Read(raw,a_.beta,BetaInputOffset(t,w_,0),1,w_.rows*heads,heads,false);
            } else {
                const uint32_t first=(h/BetaTileHeads)*BetaTileHeads;
                const uint32_t columns=heads-first<BetaTileHeads?heads-first:BetaTileHeads;
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopyExtParams p{static_cast<uint16_t>(w_.rows),static_cast<uint32_t>(columns*sizeof(BetaT)),
                    static_cast<uint32_t>((heads-columns)*sizeof(BetaT)),0,0};
                DataCopyPadExtParams<BetaT> pad{false,0,0,0};
                DataCopyPad(raw,Gm<BetaT>(a_.beta,BetaInputOffset(t,w_,first)),p,pad);
            }
        }
        if constexpr(Gate!=0) {
            if(!aLogCached_)Read(small_[384],a_.aLog,h,1,1,1,false);
            if(t.hasDtBias) Read(bias_,a_.dtBias,uint64_t(h)*128,1,128,1,false);
        }
        Mte2ToV();
        if constexpr(sizeof(GateT)!=sizeof(float)) {
            Cast(g_,gateRaw,RoundMode::CAST_NONE,w_.rows*128);VBarrier();
        }
        if(!t.inputSequenceMajor) {
            if constexpr(sizeof(BetaT)!=sizeof(float)) {
                Cast(beta_,raw,RoundMode::CAST_NONE,w_.rows);
            }
        } else {
            // Scalar rows are padded to 32B by DMA. Gather row starts in
            // Vector, preserving BF16 bits and the original FP32 Cast order.
            auto offsets=small_[BetaGatherIndexSmall].template ReinterpretCast<uint32_t>();
            const uint32_t headOffset=(t.valueHeadNum<=8?h:h%8)*sizeof(BetaT);
            if constexpr(sizeof(BetaT)==sizeof(float)) {
                Gather(beta_,raw,offsets,headOffset,w_.rows);VBarrier();
            } else {
                auto packed=small_[2560].template ReinterpretCast<BetaT>();
                Gather(packed.template ReinterpretCast<uint16_t>(),
                       raw.template ReinterpretCast<uint16_t>(),offsets,headOffset,w_.rows);
                VBarrier();Cast(beta_,packed,RoundMode::CAST_NONE,w_.rows);
            }
        }
        VBarrier();
        if(t.betaMode!=0) {
            Muls(beta_,beta_,-1.0F,w_.rows);VBarrier();
            Exp(beta_,beta_,w_.rows);VBarrier();
            Adds(beta_,beta_,1.0F,w_.rows);VBarrier();
            Duplicate(small_[1536],1.0F,w_.rows);VBarrier();
            Div(beta_,small_[1536],beta_,w_.rows);VBarrier();
            if(t.betaMode==2){Muls(beta_,beta_,2.0F,w_.rows);VBarrier();}
        }
        Normalize(q_,qr_);Normalize(k_,kr_,true);
        if(t.outputMode==1||t.outputMode==2) {
            VToMte3(); // Checkpoints remain read-only until the score batch is drained.
            if(w_.owner[lane_]) {
                Store(a_.qHat,Out(t,w_,qh,t.qkHeadNum,128),q_,w_.rows*128);
                Store(a_.kHat,Out(t,w_,qh,t.qkHeadNum,128),k_,w_.rows*128);
                Store(a_.qRstd,Out(t,w_,qh,t.qkHeadNum,1),qr_,w_.rows);
                Store(a_.kRstd,Out(t,w_,qh,t.qkHeadNum,1),kr_,w_.rows);
            }
            Store(a_.betaEff,Out(t,w_,h,t.valueHeadNum,1),beta_,w_.rows);
        }
        GatePrefix();
        Write(a_.gk,Out(t,w_,h,t.valueHeadNum,128),g_,w_.rows*128,false);
    }
    __aicore__ inline void GatePrefix() {
        uint32_t count=w_.rows*128;
        if constexpr(Gate!=0) {
            float a;
            if(aLogCached_)a=small_.GetValue(ALogCacheSmall+w_.head[lane_]);
            else {
                Exp(small_[384],small_[384],1);VBarrier();VToS();
                a=small_.GetValue(384);
                // Retire this scalar UB reader before the next vector/DMA reuse.
                SToV();
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            }
            if(a_.tiling.hasDtBias) {
                BinaryRepeatParams p{1,1,1,16,16,0};
                Add(g_,g_,bias_,64,w_.rows,p);
                Add(g_[64],g_[64],bias_[64],64,w_.rows,p);VBarrier();
            }
            if constexpr(Gate==2) {
                Muls(work_,g_,-a,count);VBarrier();Exp(work_,work_,count);VBarrier();
                Adds(work_,work_,1.0F,count);VBarrier();Duplicate(g_,a_.tiling.lowerBound,count);VBarrier();
                Div(g_,g_,work_,count);VBarrier();
            } else {
                // Compensated log1p(exp(-abs(x))); preserve small Softplus steps.
                for(uint32_t begin=0;begin<w_.rows;begin+=32) {
                    uint32_t n=(w_.rows-begin<32?w_.rows-begin:32)*128;
                    auto x=g_[begin*128],u=work_,sum=work_[4096];
                    auto residual=q_[8192].template ReinterpretCast<float>();
                    Abs(u,x,n);VBarrier();Muls(u,u,-1.0F,n);VBarrier();Exp(u,u,n);VBarrier();
                    Adds(sum,u,1.0F,n);VBarrier();Adds(residual,sum,-1.0F,n);VBarrier();
                    Sub(residual,u,residual,n);VBarrier();Div(residual,residual,sum,n);VBarrier();
                    Ln(u,sum,n);VBarrier();Add(u,u,residual,n);VBarrier();
                    Maxs(x,x,0.0F,n);VBarrier();Add(u,u,x,n);VBarrier();Muls(x,u,-a,n);VBarrier();
                }
            }
        }
        if constexpr(Gate==0 && sizeof(GateT)==sizeof(float)) {
            // Signed precomputed steps need the low rounding residual when
            // prefixes cancel. Keep the same 8-row hierarchy and FP32 output.
            Duplicate(work_,0.0F,8192);VBarrier();
            for(uint32_t row=1;row<8;++row)
                PrefixPair<128,128,128,8>(row*128,row*128,(row-1)*128);
            for(uint32_t block=1;block<8;++block)
                PrefixPair<16,16,16,1>(((block+1)*8-1)*128,
                    ((block+1)*8-1)*128,(block*8-1)*128);
            for(uint32_t block=1;block<8;++block)
                PrefixPair<16,16,0,7>(block*8*128,block*8*128,(block*8-1)*128);
            Add(g_,g_,work_,8192);VBarrier();
        } else {
            // Padded input rows are zero. All scans stop at this work's 64-row chunk.
            // Association differs from sequential FP32: this is separately native-tested.
            BinaryRepeatParams local{1,1,1,128,128,128};
            for(uint32_t r=1;r<8;++r) {
                Add(g_[r*128],g_[r*128],g_[(r-1)*128],64,8,local);
                Add(g_[r*128+64],g_[r*128+64],g_[(r-1)*128+64],64,8,local);VBarrier();
            }
            for(uint32_t block=1;block<8;++block) {
                uint32_t last=(block+1)*8-1,previous=block*8-1;
                Add(g_[last*128],g_[last*128],g_[previous*128],128);VBarrier();
            }
            BinaryRepeatParams carry{1,1,1,16,16,0};
            for(uint32_t block=1;block<8;++block) {
                uint32_t first=block*8,previous=first-1;
                Add(g_[first*128],g_[first*128],g_[previous*128],64,7,carry);
                Add(g_[first*128+64],g_[first*128+64],g_[previous*128+64],64,7,carry);
            }
            VBarrier();
        }
        if(a_.tiling.useExp2){Muls(g_,g_,1.4426950408889634F,count);VBarrier();}
    }
    template<uint8_t D,uint8_t A,uint8_t B,uint8_t Repeats>
    __aicore__ inline void PrefixPair(uint32_t dst,uint32_t lhs,uint32_t rhs) {
        // Work[0,8192): low prefix residuals. Three compact 8x128 FP32
        // temporaries occupy [8192,11264), within the existing 64KiB Work.
        auto sum=work_[8192],virtualB=work_[9216],error=work_[10240];
        for(uint32_t half=0;half<128;half+=64)
            Add(sum[half],g_[lhs+half],g_[rhs+half],64,Repeats,{1,1,1,16,A,B});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Sub(virtualB[half],sum[half],g_[lhs+half],64,Repeats,{1,1,1,16,16,A});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Sub(error[half],sum[half],virtualB[half],64,Repeats,{1,1,1,16,16,16});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Sub(error[half],g_[lhs+half],error[half],64,Repeats,{1,1,1,16,A,16});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Sub(virtualB[half],g_[rhs+half],virtualB[half],64,Repeats,{1,1,1,16,B,16});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Add(error[half],error[half],virtualB[half],64,Repeats,{1,1,1,16,16,16});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Add(work_[dst+half],work_[lhs+half],work_[rhs+half],64,Repeats,{1,1,1,D,A,B});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Add(work_[dst+half],work_[dst+half],error[half],64,Repeats,{1,1,1,D,D,16});
        VBarrier();
        for(uint32_t half=0;half<128;half+=64)
            Adds(g_[dst+half],sum[half],0.0F,64,Repeats,{1,1,D,16});
        VBarrier();
    }
    template<bool Exp2> __aicore__ inline void Factor(LocalTensor<float> f,
        uint32_t count,float lo,float hi) {
        float scale=Exp2?1.0F:0.6931471805599453F;
        Maxs(f,f,lo*scale,count);VBarrier();Mins(f,f,hi*scale,count);VBarrier();
        if constexpr(Exp2){Muls(f,f,0.6931471805599453F,count);VBarrier();}
        Exp(f,f,count);VBarrier();
    }
    __aicore__ inline void Difference(LocalTensor<float> dst,uint32_t begin,
        uint32_t reference,uint32_t rows,bool negative,bool fence=true) {
        BinaryRepeatParams p{1,1,1,16,static_cast<uint8_t>(negative?0:16),
            static_cast<uint8_t>(negative?16:0)};
        for(uint32_t half=0;half<2;++half) {
            auto x=g_[begin*128+64*half],ref=g_[reference*128+64*half];
            if(negative)Sub(dst[64*half],ref,x,64,rows,p);
            else Sub(dst[64*half],x,ref,64,rows,p);
        }
        if(fence)VBarrier();
    }
    __aicore__ inline LocalTensor<bfloat16_t> MinusBuffer(uint32_t band) {
        if(band<2)return v_[band*2048];
        if(band==2)return q_[8192];
        return k_[8192];
    }
    template<bool Exp2> __aicore__ inline void ScorePayload() {
        auto factor=work_,values=work_[8192];auto qOut=q_[8192],kOut=k_[8192];
        uint32_t bands=(w_.rows+15)/16,count=bands*2048;
        for(uint32_t band=0;band<bands;++band) {
            uint32_t begin=band*16,end=w_.rows<begin+16?w_.rows:begin+16;
            Difference(factor[begin*128],begin,(begin+end)/2,16,false,false);
        }
        VBarrier();Factor<Exp2>(factor,count,-126,120);
        Cast(values,q_,RoundMode::CAST_NONE,count);VBarrier();
        Mul(values,values,factor,count);VBarrier();Cast(qOut,values,RoundMode::CAST_RINT,count);VBarrier();
        Cast(values,k_,RoundMode::CAST_NONE,count);VBarrier();
        Mul(values,values,factor,count);VBarrier();Cast(kOut,values,RoundMode::CAST_RINT,count);VBarrier();
        VToMte3();
        Store(a_.workspace+base_+Payload,0,qOut,count);
        Store(a_.workspace+base_+Payload+0x4000,0,kOut,count);
        DrainStores(); // Upper Q/K may now become K- sources, preserving their lifetime.
        for(uint32_t band=0;band<bands;++band) {
            uint32_t begin=band*16,end=w_.rows<begin+16?w_.rows:begin+16;
            uint32_t n=16*(band+1);
            Difference(factor,0,(begin+end)/2,n,true);Factor<Exp2>(factor,n*128,-126,120);
            Cast(values,k_,RoundMode::CAST_NONE,n*128);VBarrier();
            Mul(values,values,factor,n*128);VBarrier();
            Cast(MinusBuffer(band),values,RoundMode::CAST_RINT,n*128);VBarrier();
        }
        VToMte3();
        for(uint32_t band=0;band<bands;++band)
            Store(a_.workspace+base_+Payload+MinusOffset(band),0,MinusBuffer(band),16*(band+1)*128);
        DrainStores(); // All three BF16 source buffers are now available to current RHS.
    }
    template<bool Exp2> __aicore__ inline void RhsAndOutputs() {
        const auto &t=a_.tiling;uint32_t count=w_.rows*128,h=w_.head[lane_];
        auto factor=work_,values=work_[8192];auto rhs=q_[8192],qg=k_[8192];
        Brcb(blocks_,beta_,8,{1,8});VBarrier();
        Adds(factor,g_,0.0F,count);VBarrier();Factor<Exp2>(factor,count,-80,80);
        Cast(values,k_,RoundMode::CAST_NONE,count);VBarrier();Mul(values,values,factor,count);VBarrier();
        Cast(rhs,values,RoundMode::CAST_RINT,count);VBarrier();Cast(values,rhs,RoundMode::CAST_NONE,count);VBarrier();
        RowMultiply(values,values,blocks_,w_.rows,128);
        if(w_.rows<64) {Duplicate(rhs,bfloat16_t(0),8192);VBarrier();}
        Cast(rhs,values,RoundMode::CAST_RINT,count);VBarrier();
        Write(a_.workspace+base_+RhsK,0,rhs,8192,false);
        Cast(values,q_,RoundMode::CAST_NONE,count);VBarrier();Mul(values,values,factor,count);VBarrier();
        Cast(qg,values,RoundMode::CAST_RINT,count);VBarrier();
        if(t.outputMode==2)Write(a_.qg,Out(t,w_,h,t.valueHeadNum,128),qg,count,false);
        Cast(values,qg,RoundMode::CAST_NONE,count);VBarrier();Muls(values,values,t.scale,count);VBarrier();
        Cast(v_,values,RoundMode::CAST_RINT,count);VBarrier();
        Write(a_.qgScaled,Out(t,w_,h,t.valueHeadNum,128),v_,count,false);
        DrainStores(); // Release RHS/qg/scaled sources before using the three buffers again.
        Difference(factor,0,w_.rows-1,w_.rows,true);Factor<Exp2>(factor,count,-80,80);
        Cast(values,k_,RoundMode::CAST_NONE,count);VBarrier();Mul(values,values,factor,count);VBarrier();
        Cast(v_,values,RoundMode::CAST_RINT,count);VBarrier();
        Write(a_.kg,Out(t,w_,h,t.valueHeadNum,128),v_,count,false);
        if(w_.rows<64) {Duplicate(qg,bfloat16_t(0),8192);VBarrier();}
        Read(qg,a_.v,QkvInputOffset(t,w_,h,t.valueHeadNum,128),w_.rows,128,t.valueHeadNum,t.inputSequenceMajor,true);Mte2ToV();
        Cast(values,qg,RoundMode::CAST_NONE,count);VBarrier();RowMultiply(values,values,blocks_,w_.rows,128);
        if(w_.rows<64) {Duplicate(rhs,bfloat16_t(0),8192);VBarrier();}
        Cast(rhs,values,RoundMode::CAST_RINT,count);VBarrier();
        Write(a_.workspace+base_+RhsV,0,rhs,8192,false);
    }
    __aicore__ inline void LeafInverse16() {
        if(w_.rows>16) {
            // P is ready before the inverse. Its private upper-G buffer is
            // disjoint from every forward-solve read/write and stays live
            // until the original final MTE3 drain.
            auto cross=g_[4096];
            const uint32_t count=w_.rows>32?2048:1024;
            Duplicate(cross,0.0F,count);VBarrier();
            UnaryRepeatParams copy{1,1,2,8};
            Adds(cross,g_[16*64],0.0F,16,16,copy);
            Adds(cross[768],g_[48*64+32],0.0F,16,16,copy);
            if(w_.rows>32) {
                Adds(cross[1024],g_[32*64],0.0F,16,16,copy);
                Adds(cross[1280],g_[32*64+16],0.0F,16,16,copy);
                Adds(cross[1536],g_[48*64],0.0F,16,16,copy);
                Adds(cross[1792],g_[48*64+16],0.0F,16,16,copy);
            }
            VBarrier();Write(a_.workspace+base_+CrossFp32,0,cross,count,false);
        }
        // Logical inverse[row,column,leaf], physical row pitch72float.
        // A diagonal contains four adjacent ones. Even/odd rows are two
        // disjoint stride19-block streams, avoiding a seed expansion.
        auto inverse=work_;
        Duplicate(inverse,0.0F,1152);VBarrier();
        uint64_t oddMask[2]={0xf0,0};
        Duplicate(inverse,1.0F,static_cast<uint64_t>(4),8,1,19);
        Duplicate(inverse[72],1.0F,oddMask,8,1,19);
        Duplicate(v_,bfloat16_t(0),4096);VBarrier();
        auto indices=small_[InverseGatherIndexSmall].template ReinterpretCast<int32_t>();
        auto batchIndices=work_[2048].template ReinterpretCast<int32_t>();
        auto gathered=work_[4096];
        const uint32_t steps=w_.rows<16?w_.rows-1:15;
        if(steps) {
            auto offsets=triangle_[InverseStepMask].template ReinterpretCast<int32_t>();
            BinaryRepeatParams make{1,1,0,16,0,1};
            Add(batchIndices,indices,offsets,64,steps,make);
            Add(batchIndices[64],indices[64],offsets,64,steps,make);VBarrier();
            Gather(gathered,g_,batchIndices.template ReinterpretCast<uint32_t>(),0,128*steps);VBarrier();
        }
        for(uint32_t k=0;k<steps;++k) {
            BinaryRepeatParams update{1,1,0,9,0,1};
            const uint32_t first=k+1;
            MulAddDst(inverse[first*72],inverse[k*72],gathered[k*128+first*8],64,15-k,update);VBarrier();
        }
        // The original index buffer is dead. Compress each leaf directly
        // into its final ND packet slot; no separate solved-matrix relay.
        auto packed=work_[2048];
        Duplicate(packed,0.0F,2048);VBarrier();
        uint64_t selected=0;
        GatherMask(packed,inverse,static_cast<uint8_t>(3),false,0,{1,16,9,0},selected);
        GatherMask(packed[768],inverse,static_cast<uint8_t>(4),false,0,{1,16,9,0},selected);
        GatherMask(packed[1024],inverse,static_cast<uint8_t>(5),false,0,{1,16,9,0},selected);
        GatherMask(packed[1792],inverse,static_cast<uint8_t>(6),false,0,{1,16,9,0},selected);VBarrier();
        UnaryRepeatParams store{1,1,4,2};
        Cast(v_,packed,RoundMode::CAST_RINT,16,16,store);
        Cast(v_[1040],packed[768],RoundMode::CAST_RINT,16,16,store);
        Cast(v_[2080],packed[1024],RoundMode::CAST_RINT,16,16,store);
        Cast(v_[3120],packed[1792],RoundMode::CAST_RINT,16,16,store);VBarrier();
        if(w_.rows>16)
            Write(a_.workspace+base_+DiagFp32,0,packed,2048,false);
        Write(a_.workspace+base_+Payload,0,v_,4096,false);
        if(a_.tiling.outputMode!=0)
            Write(a_.akk,Out(a_.tiling,w_,w_.head[lane_],a_.tiling.valueHeadNum,64),v_,w_.rows*64,false);
        DrainStores();
    }
    __aicore__ inline void LeafInverse32() {
        Read(v_,a_.workspace+base_+Eye,0,1,4096,1,false);Mte2ToV();
        // Two independent 32x32 FP32 matrices, packed as 64 rows x32 columns.
        auto inverse=work_;
        UnaryRepeatParams pack{1,1,4,4};
        Cast(inverse,v_,RoundMode::CAST_NONE,32,32,pack);
        Cast(inverse[1024],v_[32*64+32],RoundMode::CAST_NONE,32,32,pack);VBarrier();
        auto indices=small_[InverseGatherIndexSmall].template ReinterpretCast<int32_t>();
        auto batchIndices=work_[2048].template ReinterpretCast<int32_t>();
        auto gathered=work_[4096],rowCopies=small_[1536];
        uint32_t steps=w_.rows<32?w_.rows-1:31;
        // Q/K are retired; only their GM Lower copy is needed by Cube.
        // Reserve a bounded alias across their combined 64KiB UB region.
        DrainStores();
        auto coefficients=ub_.template GetWithOffset<float>((UbG-UbQ)/4,UbQ);
        if(steps) {
            for(uint32_t k=0;k<steps;++k)
                Adds(batchIndices[k*64],indices,static_cast<int32_t>(k*4),64);
            VBarrier();
            Gather(gathered,g_,batchIndices.template ReinterpretCast<uint32_t>(),0,64*steps);
            VBarrier();
            Brcb(coefficients,gathered,static_cast<uint8_t>(8*steps),{1,8});VBarrier();
        }
        for(uint32_t k=0;k<steps;++k) {
            auto current=coefficients[k*512];
            UnaryRepeatParams rows{1,1,4,128};
            Adds(rowCopies,inverse[k*32],0.0F,32,2,rows);VBarrier();
            BinaryRepeatParams update{1,1,0,4,0,1};
            const uint32_t first=k+1,remaining=31-k;
            MulAddDst(inverse[first*32],rowCopies,current[first*8],32,remaining,update);
            if(w_.rows>32)MulAddDst(inverse[1024+first*32],rowCopies[32],current[256+first*8],32,remaining,update);
            VBarrier();
        }
        // v_ still holds the immutable zero off-diagonal/identity padding.
        Cast(v_,inverse,RoundMode::CAST_RINT,32,32,pack);
        Cast(v_[32*64+32],inverse[1024],RoundMode::CAST_RINT,32,32,pack);VBarrier();
        Write(a_.workspace+base_+Payload,0,v_,4096,false);
        if(a_.tiling.outputMode!=0)
            Write(a_.akk,Out(a_.tiling,w_,w_.head[lane_],a_.tiling.valueHeadNum,64),v_,w_.rows*64,false);
        DrainStores();
    }
    __aicore__ inline void LeafInverse64() {
        Read(v_,a_.workspace+base_+Eye,0,1,4096,1,false);Mte2ToV();
        // Keep the local solve in FP32; only the completed inverse is relayed.
        // Work: inverse[0,4096), gather indices[4096,5120), values[5120,6144).
        auto inverse=work_;
        Cast(inverse,v_,RoundMode::CAST_NONE,4096);VBarrier();
        auto indices=small_[InverseGatherIndexSmall].template ReinterpretCast<int32_t>();
        auto batchIndices=work_[4096].template ReinterpretCast<int32_t>();
        auto gathered=work_[5120],rowCopy=small_[1536];
        DrainStores(); // Retire Lower/Q/K stores before the Q alias is overwritten.
        auto coefficients=ub_.template GetWithOffset<float>((UbK-UbQ)/4,UbQ);
        const uint32_t steps=w_.rows-1;
        for(uint32_t begin=0;begin<steps;begin+=16) {
            const uint32_t count=steps-begin<16?steps-begin:16;
            for(uint32_t k=0;k<count;++k)
                Adds(batchIndices[k*64],indices,static_cast<int32_t>((begin+k)*4),64);
            VBarrier();
            Gather(gathered,g_,batchIndices.template ReinterpretCast<uint32_t>(),0,64*count);
            VBarrier();
            Brcb(coefficients,gathered,static_cast<uint8_t>(8*count),{1,8});VBarrier();
            for(uint32_t step=0;step<count;++step) {
                const uint32_t k=begin+step,first=k+1;
                Adds(rowCopy,inverse[k*64],0.0F,64);VBarrier();
                BinaryRepeatParams update{1,1,0,8,0,1};
                MulAddDst(inverse[first*64],rowCopy,coefficients[step*512+first*8],
                          64,63-k,update);VBarrier();
            }
        }
        Cast(v_,inverse,RoundMode::CAST_RINT,4096);VBarrier();
        Write(a_.workspace+base_+Payload,0,v_,4096,false);
        if(a_.tiling.outputMode!=0)
            Write(a_.akk,Out(a_.tiling,w_,w_.head[lane_],a_.tiling.valueHeadNum,64),v_,w_.rows*64,false);
        DrainStores();
    }
    __aicore__ inline void KeepTriangle(LocalTensor<float> values,LocalTensor<uint8_t> mask,uint32_t count) {
        if ASCEND_IS_AIV {
            // Same Arch22 mode1 instruction as Select(...,0), with a shared
            // zero CMPMASK. The eight calls do not modify CMPMASK.
            SetMaskCount();SetVectorMask<float>(0,count);
            vsel(reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr()),
                 1,1,1,1,8,8,8,static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
            ResetMask();SetMaskNorm();
        }
    }
    __aicore__ inline void LowerAndAqk(uint32_t slot) {
        // Cross-core raw-ready orders GM production, not this AIV's preceding
        // RHS vector consumer of work_. Retire that UB reader before raw DMA.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        auto raw=Gm<float>(a_.workspace+base_+Raw);PublishCache(raw);
        uint32_t bands=(w_.rows+15)/16;
        for(uint32_t s=0;s<bands;++s) {
            uint32_t n=16*(s+1),offset=256*s*(s+1);
            DataCopyExtParams p{32,n*4,(64-n)*4,0,0};
            DataCopyPadExtParams<float> pad{false,0,0,0};
            DataCopyPad(work_[offset],raw[s*2048],p,pad);
        }
        Mte2ToV();DrainStores(); // kg uses v_, which becomes the Aqk buffer below.
        if constexpr(Norm==0 || Gate==0)Duplicate(q_,bfloat16_t(0),4096);
        Duplicate(v_,bfloat16_t(0),8192);
        if constexpr(Norm==0 || Gate==0)Duplicate(g_,0.0F,4096);
        else if(bands<4)Duplicate(g_[bands*1024],0.0F,4096-bands*1024);
        if constexpr(Norm!=0 && Gate!=0)Duplicate(small_,0.0F,64);
        VBarrier();
        // Four independent 16-row bands replace the scalar row/beta dispatch.
        Muls(small_[256],beta_,-1.0F,64);VBarrier();
        Brcb(blocks_,small_[256],8,{1,8});VBarrier();
        SetCmpMask(triangle_[ZeroSelectMask].template ReinterpretCast<uint16_t>());VBarrier();
        for(uint32_t band=0;band<bands;++band) {
            uint32_t n=16*(band+1),pack=256*band*(band+1),begin=band*16;
            auto qraw=work_[pack],kraw=work_[pack+16*n];
            auto qtmp=work_[8192],ptmp=work_[10240];
            UnaryRepeatParams cast{1,1,4,static_cast<uint8_t>(n/8)};
            BinaryRepeatParams mul{1,1,0,static_cast<uint8_t>(n/8),static_cast<uint8_t>(n/8),1};
            // Aqk and P have separate producers, consumers and temporary ranges.
            Muls(qtmp,qraw,a_.tiling.scale,16*n);
            Mul(ptmp,kraw,blocks_[begin*8],n,16,mul);VBarrier();
            KeepTriangle(qtmp,triangle_[320+16*band*(band+1)],16*n);
            KeepTriangle(ptmp,triangle_[16*band*(band+1)],16*n);VBarrier();
            Cast(v_[begin*64],qtmp,RoundMode::CAST_RINT,n,16,cast);
            if constexpr(Norm!=0 && Gate!=0) {
                auto magnitudes=work_[12288+pack/2];
                Abs(magnitudes,ptmp,16*n);
            }
            UnaryRepeatParams keep{1,1,8,static_cast<uint8_t>(n/8)};
            if constexpr(Norm!=0 && Gate!=0) {
                Adds(g_.template ReinterpretCast<int32_t>()[begin*64],
                     ptmp.template ReinterpretCast<int32_t>(),int32_t(0),n,16,keep);
            } else {
                Adds(g_[begin*64],ptmp,0.0F,n,16,keep);
                Cast(q_[begin*64],ptmp,RoundMode::CAST_RINT,n,16,cast);
            }
            VBarrier();
        }
        if constexpr(Norm!=0 && Gate!=0) {
            WholeReduceMax(small_,work_[12288],64,2*bands*(bands+1),1,1,8,
                           ReduceOrder::ORDER_ONLY_VALUE);
            // The following drained Aqk Write has V->MTE3->V ordering, which
            // also retires this reduction before the second-stage reduction.
        }
        // Packed masks were initialized once. The selected FP32 zeros cast
        // to the same +0 BF16 bits as the former per-row Duplicate operations.
        // Adaptive classification reads only Small/Work. Its existing
        // choice-store drain also retires Aqk before a risky Eye read.
        if constexpr(Norm!=0 && Gate!=0)
            Write(a_.aqk,Out(a_.tiling,w_,w_.head[lane_],a_.tiling.valueHeadNum,64),v_,w_.rows*64,false);
        else Write(a_.aqk,Out(a_.tiling,w_,w_.head[lane_],a_.tiling.valueHeadNum,64),v_,w_.rows*64);
        if constexpr(Gate==0) {
            Write(a_.workspace+base_+Lower,0,q_,4096,false);LeafInverse64();
        }
        else if constexpr(Norm!=0) {
            WholeReduceMax(small_[64],small_,64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);VBarrier();VToS();
            const float magnitude=small_.GetValue(64);
            bool fallback=magnitude>FastInverseMaxCoefficient && magnitude<=MaxFiniteFp32;
            if(fallback) {
                // Preserve the existing special-value evaluation order. Max
                // alone may overlook a NaN mixed with finite large entries.
                // A sum of nonnegative finite magnitudes cannot become NaN;
                // overflow to +Inf still keeps the finite-input FP32 fallback.
                Duplicate(small_[128],0.0F,64);VBarrier();
                WholeReduceSum(small_[128],work_[12288],64,2*bands*(bands+1),1,1,8);VBarrier();
                WholeReduceSum(small_[192],small_[128],64,1,1,1,8);VBarrier();VToS();
                const float sum=small_.GetValue(192);
                fallback=sum==sum;
            }
            if(!fallback) {
                Duplicate(q_,bfloat16_t(0),4096);VBarrier();
                for(uint32_t band=0;band<bands;++band) {
                    uint32_t n=16*(band+1),begin=band*16;
                    UnaryRepeatParams cast{1,1,4,8};
                    Cast(q_[begin*64],g_[begin*64],RoundMode::CAST_RINT,n,16,cast);
                }
                VBarrier();
                Write(a_.workspace+base_+Lower,0,q_,4096,false);
            }
            if(!fallback && a_.tiling.outputMode!=0) {
                // Fast Save reuses v_ before the shared choice drain.
                DrainStores();
                Duplicate(v_,bfloat16_t(0),4096);VBarrier();
                Write(a_.akk,Out(a_.tiling,w_,w_.head[lane_],a_.tiling.valueHeadNum,64),v_,w_.rows*64,false);
            }
            small_.SetValue(64,fallback?1.0F:0.0F);SToV();
            Write(a_.workspace+base_+InverseChoice,0,small_[64],1,false);
            DrainStores();
            // Fast Cube work can overlap the other AIV's FP32 solve. All
            // public fast-path zeroing must retire before Cube writes Akk.
            CrossCoreSetFlag<0x2,PIPE_MTE3>(LowerChoiceReadyBase+slot);
            if(fallback)LeafInverse16();
        }
        else {
            Write(a_.workspace+base_+Lower,0,q_,4096,false);LeafInverse32();
        }

    }
    __aicore__ inline void TriangleOnce() {
        // 4 band-packed masks: 32+64+96+128 bytes per strict/inclusive set.
        // UbMask's existing 2 KiB holds both immutable 320-byte sets.
        auto words=triangle_.template ReinterpretCast<uint16_t>();
        for(uint32_t band=0;band<4;++band) {
            uint32_t offset=8*band*(band+1),width=band+1;
            for(uint32_t row=0;row<16;++row) {
                for(uint32_t col=0;col<width;++col) {
                    uint32_t count=16*(band-col)+row;
                    uint16_t strict=count>=16?0xffff:static_cast<uint16_t>((1U<<count)-1);
                    uint16_t inclusive=count>=15?0xffff:static_cast<uint16_t>((1U<<(count+1))-1);
                    words.SetValue(offset+row*width+col,strict);
                    words.SetValue(160+offset+row*width+col,inclusive);
                }
            }
        }
        SToV();
        if constexpr(Norm!=0 && Gate!=0) {
            auto offsets=triangle_[InverseStepMask].template ReinterpretCast<int32_t>();
            for(uint32_t k=0;k<15;++k)Duplicate(offsets[k*8],static_cast<int32_t>(k*4),8);
            VBarrier();
        }
        Duplicate(triangle_[ZeroSelectMask].template ReinterpretCast<uint32_t>(),uint32_t(0),8);
        VBarrier();
    }
    __aicore__ inline void IdentityOnce() {
        Duplicate(k_,bfloat16_t(0),4096);VBarrier();VToS();
        for(uint32_t r=0;r<64;++r)k_.SetValue(r*64+r,bfloat16_t(1));
        SToV();
        // Private immutable GM regions. Both complete before the first score-ready.
        for(uint32_t slot=0;slot<2;++slot)
            Write(a_.workspace+GmLane(core_,slot,lane_)+Eye,0,k_,4096);
    }

    __aicore__ inline void InitALogCache() {
        if constexpr(Gate!=0) {
            uint32_t heads=a_.tiling.valueHeadNum;
            if(heads<=ALogCacheHeads) {
                Read(small_[ALogCacheSmall],a_.aLog,0,1,heads,1,false);
                Mte2ToV();Exp(small_[ALogCacheSmall],small_[ALogCacheSmall],heads);
                VBarrier();VToS();aLogCached_=true;
            }
        }
    }

public:
    __aicore__ inline Vector(const PrepareKernelArgs &a):a_(a) {}
    __aicore__ inline void Init(const PrepareKernelArgs &a,TPipe *pipe) {
        pipe_=pipe;uint32_t sub=GetSubBlockNum();core_=GetBlockIdx()/sub;lane_=GetSubBlockIdx();
        pipe_->InitBuffer(ub_,UbBytes);
        // TBuf still owns the same single UB allocation. Explicitly initialize
        // every descriptor field before it persists across stages.
        auto bytes=ub_.template Get<uint8_t>();
        BindTensor(q_,bytes,UbQ,UbK-UbQ);BindTensor(k_,bytes,UbK,UbG-UbK);
        BindTensor(g_,bytes,UbG,UbWork-UbG);BindTensor(work_,bytes,UbWork,UbV-UbWork);
        BindTensor(v_,bytes,UbV,UbSmall-UbV);BindTensor(small_,bytes,UbSmall,UbMask-UbSmall);
        BindTensor(triangle_,bytes,UbMask,UbBytes-UbMask);
        BindTensor(qr_,bytes,UbSmall,256);BindTensor(kr_,bytes,UbSmall+256,256);
        BindTensor(beta_,bytes,UbSmall+512,256);BindTensor(bias_,bytes,UbSmall+1024,512);
        BindTensor(blocks_,bytes,UbSmall+4096,2048);
        if(core_<a_.tiling.usedCoreNum){
            {
                auto offsets=small_[InverseGatherIndexSmall].template ReinterpretCast<uint32_t>();
                if constexpr(Norm!=0 && Gate!=0) {
                    for(uint32_t i=0;i<128;++i) {
                        const uint32_t row=i/8,leaf=i%4;
                        offsets.SetValue(i,((leaf*16+row)*64+leaf*16)*4);
                    }
                } else for(uint32_t row=0;row<64;++row)
                    offsets.SetValue(row,(row*64+(Gate==0?0:(row/32)*32))*4);
                SToV();
            }
            if(a_.tiling.inputSequenceMajor) { // Sequence-major beta tile indices only.
                auto offsets=small_[BetaGatherIndexSmall].template ReinterpretCast<uint32_t>();
                // BF16 Gather may consume a full 128-entry offset register pair.
                // Upper masked lanes have neutral indices in the existing spare UB.
                Duplicate(offsets,uint32_t(0),128);VBarrier();VToS();
                const uint32_t stride=a_.tiling.valueHeadNum<=8?
                    a_.tiling.valueHeadNum*sizeof(BetaT):32;
                for(uint32_t row=0;row<64;++row)offsets.SetValue(row,row*stride);
                SToV();
            }
            InitALogCache();TriangleOnce();IdentityOnce();
        }
    }
    __aicore__ inline void Process() {
        if(core_>=a_.tiling.usedCoreNum)return;
        uint32_t generation=0;
        for(uint64_t job=core_;job<WorkCount(a_.tiling);job+=a_.tiling.usedCoreNum,++generation) {
            uint64_t workJob=job;
            if constexpr(Norm!=0 && Gate!=0)workJob=RotateWork(a_.tiling,job,core_,generation);
            uint32_t slot=generation&1;w_=Describe(a_,workJob);base_=GmLane(core_,slot,lane_);
            CrossCoreWaitFlag<0x2,PIPE_MTE3>(6+slot);
            if(w_.active[lane_]&&w_.rows) {
                Prepare();
                if(a_.tiling.useExp2)ScorePayload<true>();else ScorePayload<false>();
            }
            PublishCache(Gm<uint8_t>(a_.workspace+base_));
            CrossCoreSetFlag<0x2,PIPE_MTE3>(slot);
            if(w_.active[lane_]&&w_.rows) {
                if(a_.tiling.useExp2)RhsAndOutputs<true>();else RhsAndOutputs<false>();
            }
            CrossCoreSetFlag<0x2,PIPE_MTE3>(RhsReadyBase+slot);
            CrossCoreWaitFlag<0x2,PIPE_MTE2>(2+slot);
            if(w_.active[lane_]&&w_.rows)LowerAndAqk(slot);
            else if constexpr(Norm!=0 && Gate!=0)
                CrossCoreSetFlag<0x2,PIPE_MTE3>(LowerChoiceReadyBase+slot);
            PublishCache(Gm<uint8_t>(a_.workspace+base_));
            CrossCoreSetFlag<0x2,PIPE_MTE3>(4+slot);
        }
        CrossCoreWaitFlag<0x2,PIPE_MTE3>(6);
        CrossCoreWaitFlag<0x2,PIPE_MTE3>(7);
    }
};
}
