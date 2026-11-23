#pragma once
#include "projection_types.h"

namespace kda {
struct RecurrenceArgs {
    using Address = GM_ADDR;
    Address convW, conv, checkpoint, u, key, decay, bias, alphaLog, gamma;
    Address indices, accepted, prefix, outWeight, output;
    uint64_t convStride, checkpointStride, uStride, keyStride, decayStride;
    uint32_t batch, sequence, capacity, prefix64, convLayout, crosses, staged;
    float scale, lower, eps;
};

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
namespace vec {
using namespace AscendC::MicroAPI;
using F = RegTensor<float>;
constexpr CastTrait widen = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr CastTrait narrow = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
__simd_callee__ inline void readBf(F &x, __ubuf__ Bf *src, MaskReg &m) {
    RegTensor<Bf> bits;
    LoadAlign<Bf, LoadDist::DIST_UNPACK_B16>(bits, src);
    Cast<float,Bf,widen>(x,bits,m);
}
__simd_callee__ inline void logistic(F &out, F &x, MaskReg &m) {
    F z,one;
    Muls(z,x,-1.f,m); Exp(z,z,m); Adds(z,z,1.f,m);
    Duplicate(one,1.f,m); Div(out,one,z,m);
}
__simd_vf__ inline void convolve(__ubuf__ Bf *raw, __ubuf__ Bf *history,
        __ubuf__ Bf *w, __ubuf__ float *out, uint16_t oldest, bool normalize, float scale) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    F result[2];
    #pragma unroll
    for (uint16_t part=0;part<2;++part) {
        uint16_t i=64*part;
        F x,weight,acc,act;
        Duplicate(acc,0.f,m);
        for (uint16_t tap=0;tap<4;++tap) {
            if (tap==3) {readBf(x,raw+i,m);}
            else {readBf(x,history+((oldest+tap)%3)*384+i,m);}
            readBf(weight,w+tap*128+i,m);
            F term; Mul(term,x,weight,m); Add(acc,acc,term,m);
        }
        logistic(act,acc,m); Mul(act,acc,act,m);
        RegTensor<Bf> rounded;
        Cast<Bf,float,narrow>(rounded,act,m);
        Cast<float,Bf,widen>(result[part],rounded,m);
        RegTensor<Bf> rawBits;
        LoadAlign<Bf,LoadDist::DIST_UNPACK_B16>(rawBits,raw+i);
        StoreAlign<Bf,StoreDist::DIST_PACK_B32>(history+oldest*384+i,rawBits,m);
    }
    if (normalize) {
        F a,b,sum,inv;
        Mul(a,result[0],result[0],m); Mul(b,result[1],result[1],m);
        Add(a,a,b,m); ReduceSum(sum,a,m); Duplicate<float,HighLowPart::LOWEST>(b,sum,m);
        Adds(b,b,1.e-6f,m); Sqrt(b,b,m); Duplicate(a,scale,m); Div(inv,a,b,m);
        Mul(result[0],result[0],inv,m); Mul(result[1],result[1],inv,m);
    }
    StoreAlign(out,result[0],m); StoreAlign(out+64,result[1],m);
}
// A complete short sequence has no outgoing hidden-state dependency.
__simd_vf__ inline void shortConvolve(__ubuf__ Bf *raw,__ubuf__ Bf *history,
        __ubuf__ Bf *weights,__ubuf__ float *out,uint16_t count,bool normalize,float scale) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    F w0[2],w1[2],w2[2],w3[2],h0[2],h1[2],h2[2];
    #pragma unroll
    for(uint16_t j=0;j<2;++j) {
        uint16_t i=j*64;
        readBf(w0[j],weights+i,m); readBf(w1[j],weights+128+i,m);
        readBf(w2[j],weights+256+i,m); readBf(w3[j],weights+384+i,m);
        readBf(h0[j],history+i,m); readBf(h1[j],history+384+i,m); readBf(h2[j],history+768+i,m);
    }
    for(uint16_t t=0;t<count;++t) {
        F result[2];
        #pragma unroll
        for(uint16_t j=0;j<2;++j) {
            F x,acc,term,act;
            readBf(x,raw+t*128+j*64,m);
            Mul(acc,h0[j],w0[j],m); Mul(term,h1[j],w1[j],m); Add(acc,acc,term,m);
            Mul(term,h2[j],w2[j],m); Add(acc,acc,term,m); Mul(term,x,w3[j],m); Add(acc,acc,term,m);
            logistic(act,acc,m); Mul(act,acc,act,m);
            RegTensor<Bf> rounded;
            Cast<Bf,float,narrow>(rounded,act,m); Cast<float,Bf,widen>(result[j],rounded,m);
            h0[j]=h1[j]; h1[j]=h2[j]; h2[j]=x;
        }
        if(normalize) {
            F a,b,sum,inv;
            Mul(a,result[0],result[0],m); Mul(b,result[1],result[1],m); Add(a,a,b,m); ReduceSum(sum,a,m);
            Duplicate<float,HighLowPart::LOWEST>(b,sum,m);
            Adds(b,b,1.e-6f,m); Sqrt(b,b,m); Duplicate(a,scale,m); Div(inv,a,b,m);
            Mul(result[0],result[0],inv,m); Mul(result[1],result[1],inv,m);
        }
        StoreAlign(out+t*384,result[0],m); StoreAlign(out+t*384+64,result[1],m);
    }
}
__simd_vf__ inline void step(__ubuf__ float *state, __ubuf__ float *qkv, __ubuf__ Bf *dr,
        __ubuf__ Bf *beta, __ubuf__ float *bias, float aLog, float lower,
        __ubuf__ float *u, __ubuf__ float *d, __ubuf__ float *y) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    F q[2],k[2],dec[2],alpha,br,betaF;
    Duplicate(alpha,aLog,m); Exp(alpha,alpha,m);
    RegTensor<Bf> betaBits;
    LoadAlign<Bf,LoadDist::DIST_BRC_B16>(betaBits,beta);
    Cast<float,Bf,widen>(br,betaBits,m); logistic(betaF,br,m);
    #pragma unroll
    for(uint16_t j=0;j<2;++j) {
        F x,b;
        LoadAlign(q[j],qkv+j*64); LoadAlign(k[j],qkv+128+j*64);
        readBf(x,dr+j*64,m); LoadAlign(b,bias+j*64); Add(x,x,b,m); Mul(x,x,alpha,m);
        logistic(b,x,m); Muls(b,b,lower,m); Exp(dec[j],b,m); StoreAlign(d+j*64,dec[j],m);
    }
    for(uint16_t row=0;row<128;++row) {
        F h0,h1,p0,p1,total,dot,v,corr,bc;
        LoadAlign(h0,state+row*128); LoadAlign(h1,state+row*128+64);
        Mul(h0,h0,dec[0],m); Mul(h1,h1,dec[1],m);
        Mul(p0,h0,k[0],m); Mul(p1,h1,k[1],m); Add(total,p0,p1,m); ReduceSum(dot,total,m);
        LoadAlign<float,LoadDist::DIST_BRC_B32>(v,qkv+256+row);
        Sub(corr,v,dot,m); Mul(corr,corr,betaF,m);
        StoreAlign<float,StoreDist::DIST_FIRST_ELEMENT_B32>(u+row,corr,m);
        Duplicate<float,HighLowPart::LOWEST>(bc,corr,m);
        Mul(p0,bc,k[0],m); Mul(p1,bc,k[1],m); Add(h0,h0,p0,m); Add(h1,h1,p1,m);
        StoreAlign(state+row*128,h0,m); StoreAlign(state+row*128+64,h1,m);
        Mul(p0,h0,q[0],m); Mul(p1,h1,q[1],m); Add(total,p0,p1,m); ReduceSum(dot,total,m);
        StoreAlign<float,StoreDist::DIST_FIRST_ELEMENT_B32>(y+row,dot,m);
    }
}
// Expand a short, complete recurrence from its read-only checkpoint. The
// coefficient products never divide by decay, including when it underflows.
__simd_vf__ inline void shortDecay(__ubuf__ Bf *raw,__ubuf__ Bf *betaRaw,
        __ubuf__ float *bias,float aLog,float lower,__ubuf__ float *dec,__ubuf__ float *beta) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    F alpha,v,betaF;
    Duplicate(alpha,aLog,m); Exp(alpha,alpha,m);
    RegTensor<Bf> bits; LoadAlign<Bf,LoadDist::DIST_BRC_B16>(bits,betaRaw);
    Cast<float,Bf,widen>(v,bits,m); logistic(betaF,v,m); StoreAlign(beta,betaF,m);
    for(uint16_t i=0;i<2;++i) {
        F x,b; readBf(x,raw+i*64,m); LoadAlign(b,bias+i*64);
        Add(x,x,b,m); Mul(x,x,alpha,m); logistic(b,x,m); Muls(b,b,lower,m);
        Exp(x,b,m); StoreAlign(dec+64*i,x,m);
    }
}
__simd_vf__ inline void shortPrepare(__ubuf__ float *qkv,__ubuf__ float *dec,
        __ubuf__ float *weighted,__ubuf__ float *coeff,uint16_t count) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    for(uint16_t t=0;t<count;++t) {
        F q0,q1,k0,k1,p0,p1;
        LoadAlign(q0,qkv+t*384); LoadAlign(q1,qkv+t*384+64);
        LoadAlign(k0,qkv+t*384+128); LoadAlign(k1,qkv+t*384+192);
        Duplicate(p0,1.f,m); Duplicate(p1,1.f,m);
        for(uint16_t j=0;j<=t;++j) {
            uint16_t s=t-j;
            F w0,w1,a,b,sum;
            LoadAlign(w0,qkv+s*384+128); LoadAlign(w1,qkv+s*384+192);
            Mul(w0,w0,p0,m); Mul(w1,w1,p1,m);
            Mul(a,w0,k0,m); Mul(b,w1,k1,m); Add(a,a,b,m); ReduceSum(sum,a,m);
            StoreAlign<float,StoreDist::DIST_FIRST_ELEMENT_B32>(coeff+(t*4+s)*2,sum,m);
            Mul(a,w0,q0,m); Mul(b,w1,q1,m); Add(a,a,b,m); ReduceSum(sum,a,m);
            StoreAlign<float,StoreDist::DIST_FIRST_ELEMENT_B32>(coeff+(t*4+s)*2+1,sum,m);
            LoadAlign(a,dec+s*128); LoadAlign(b,dec+s*128+64);
            Mul(p0,p0,a,m); Mul(p1,p1,b,m);
        }
        Mul(k0,k0,p0,m); Mul(k1,k1,p1,m); Mul(q0,q0,p0,m); Mul(q1,q1,p1,m);
        StoreAlign(weighted+t*256,k0,m); StoreAlign(weighted+t*256+64,k1,m);
        StoreAlign(weighted+t*256+128,q0,m); StoreAlign(weighted+t*256+192,q1,m);
    }
}
__simd_vf__ inline void shortProject(__ubuf__ float *state,__ubuf__ float *weighted,
        __ubuf__ float *u,__ubuf__ float *y,uint16_t count) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    for(uint16_t row=0;row<128;++row) {
        F h0,h1; LoadAlign(h0,state+128*row); LoadAlign(h1,state+128*row+64);
        for(uint16_t t=0;t<count;++t) {
            F a,b,sum;
            LoadAlign(a,weighted+t*256); LoadAlign(b,weighted+t*256+64);
            Mul(a,a,h0,m); Mul(b,b,h1,m); Add(a,a,b,m); ReduceSum(sum,a,m);
            StoreAlign<float,StoreDist::DIST_FIRST_ELEMENT_B32>(u+t*128+row,sum,m);
            LoadAlign(a,weighted+t*256+128); LoadAlign(b,weighted+t*256+192);
            Mul(a,a,h0,m); Mul(b,b,h1,m); Add(a,a,b,m); ReduceSum(sum,a,m);
            StoreAlign<float,StoreDist::DIST_FIRST_ELEMENT_B32>(y+t*128+row,sum,m);
        }
    }
}
__simd_vf__ inline void shortCorrect(__ubuf__ float *qkv,__ubuf__ float *beta,__ubuf__ float *coeff,
        __ubuf__ float *predicted,__ubuf__ float *y,__ubuf__ float *u,uint16_t t) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    #pragma unroll
    for(uint16_t part=0;part<2;++part) {
        uint16_t i=64*part; F pred,out,v,b,term,uk,c;
        LoadAlign(pred,predicted+t*128+i); LoadAlign(out,y+t*128+i);
        for(uint16_t s=0;s<t;++s) {
            LoadAlign(uk,u+s*128+i);
            LoadAlign<float,LoadDist::DIST_BRC_B32>(c,coeff+(t*4+s)*2);
            Mul(term,c,uk,m); Add(pred,pred,term,m);
            LoadAlign<float,LoadDist::DIST_BRC_B32>(c,coeff+(t*4+s)*2+1);
            Mul(term,c,uk,m); Add(out,out,term,m);
        }
        LoadAlign(v,qkv+t*384+256+i); LoadAlign(b,beta+t*64);
        Sub(uk,v,pred,m); Mul(uk,uk,b,m);
        LoadAlign<float,LoadDist::DIST_BRC_B32>(c,coeff+(t*4+t)*2+1);
        Mul(term,uk,c,m); Add(out,out,term,m);
        StoreAlign(u+t*128+i,uk,m); StoreAlign(y+t*128+i,out,m);
    }
}
__simd_vf__ inline void finish(__ubuf__ float *y,__ubuf__ Bf *gate,__ubuf__ float *gamma,
        float eps,__ubuf__ Bf *out) {
    auto m=CreateMask<float,MaskPattern::ALL>();
    F a,b,aa,bb,s,n,one;
    LoadAlign(a,y); LoadAlign(b,y+64);
    Mul(aa,a,a,m); Mul(bb,b,b,m); Add(aa,aa,bb,m); ReduceSum(s,aa,m);
    Duplicate<float,HighLowPart::LOWEST>(n,s,m); Muls(n,n,1.f/128.f,m); Adds(n,n,eps,m);
    Sqrt(n,n,m); Duplicate(one,1.f,m); Div(n,one,n,m);
    for(uint16_t i=0;i<2;++i) {
        F x,g,weight,act;
        LoadAlign(x,y+64*i); LoadAlign(weight,gamma+64*i); readBf(g,gate+64*i,m); logistic(act,g,m);
        Mul(x,x,n,m); Mul(x,x,weight,m); Mul(x,x,act,m);
        RegTensor<Bf> rounded; Cast<Bf,float,narrow>(rounded,x,m);
        StoreAlign<Bf,StoreDist::DIST_PACK_B32>(out+64*i,rounded,m);
    }
}
} // namespace vec
#endif

class Recurrence {
    using Resource = Catlass::Arch::Resource<Arch>;
    template<typename T> CATLASS_DEVICE AscendC::GlobalTensor<T> gm(GM_ADDR ptr, uint64_t offset=0) {
        AscendC::GlobalTensor<T> g; g.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr)+offset); return g;
    }
public:
    CATLASS_DEVICE uint32_t prefix(const RecurrenceArgs &a,uint32_t b) {
        if(a.sequence) return b*a.sequence;
        if(a.prefix64) return static_cast<uint32_t>(gm<int64_t>(a.prefix).GetValue(b));
        return gm<int32_t>(a.prefix).GetValue(b);
    }
    CATLASS_DEVICE uint32_t owner(const RecurrenceArgs &a,uint32_t row) {
        if(a.sequence) return row/a.sequence;
        uint32_t lo=0,hi=a.batch;
        while(lo+1<hi) {uint32_t mid=lo+(hi-lo)/2; if(prefix(a,mid)<=row) lo=mid; else hi=mid;}
        return lo;
    }
public:
    CATLASS_DEVICE void consume(Resource &resource,const RecurrenceArgs &a, GM_ADDR gated, GM_ADDR carry,
        uint32_t heads,uint32_t head,uint32_t global,uint32_t ringRow,uint32_t valid,
        AscendC::LocalTensor<Bf> slot) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        constexpr uint32_t base=41984;
        auto state=resource.ubBuf.template GetBufferByByte<float>(base);
        auto hist=resource.ubBuf.template GetBufferByByte<Bf>(base+65536);
        auto weights=resource.ubBuf.template GetBufferByByte<Bf>(base+65536+2304);
        constexpr uint32_t tmp=base+65536+2304+3072;
        auto f=resource.ubBuf.template GetBufferByByte<float>(tmp);
        auto gatedRow=resource.ubBuf.template GetBufferByByte<Bf>(tmp+8*512);
        auto keyRow=resource.ubBuf.template GetBufferByByte<half>(tmp+8*512+256);
        uint32_t width=heads*128;
        auto qkv=f, u=f[384], dec=f[512], y=f[640], bias=f[768], gamma=f[896];
        for(uint32_t kind=0;kind<3;++kind) {
            if(a.convLayout==0) {
                AscendC::DataCopyPad(weights[kind*512],gm<Bf>(a.convW,uint64_t(kind)*width+head*128),
                    AscendC::DataCopyExtParams{4,256,(3*width-128)*2,0,0},AscendC::DataCopyPadExtParams<Bf>{});
            } else {
                uint64_t group=a.convLayout==1?uint64_t(kind)*heads+head:kind;
                AscendC::DataCopy(weights[kind*512],gm<Bf>(a.convW,group*512),512);
            }
        }
        AscendC::DataCopy(bias,gm<float>(a.bias,head*128),128);
        AscendC::DataCopy(gamma,gm<float>(a.gamma),128);
        float aLog=gm<float>(a.alphaLog).GetValue(head);
        uint32_t b=owner(a,global);
        auto chunk=resource.ubBuf.template GetBufferByByte<float>(tmp+10*512);
        auto chunkDec=chunk[4*384],chunkBeta=chunkDec[4*128],weighted=chunkBeta[4*64];
        auto coeff=weighted[4*256],chunkU=coeff[64],chunkY=chunkU[4*128],predicted=chunkY[4*128];
        bool shortSequence=a.staged>=2 && valid<=4 && prefix(a,b)==global && prefix(a,b+1)==global+valid;
        uint32_t shortIndex=shortSequence?gm<int32_t>(a.indices).GetValue(b):0;
        constexpr uint32_t shortOut=tmp+10*512+5*4096;
        auto gatedRows=resource.ubBuf.template GetBufferByByte<Bf>(shortOut);
        auto keyRows=resource.ubBuf.template GetBufferByByte<half>(shortOut+1024);
        for(uint32_t r=0;r<valid;++r) {
            uint32_t row=global+r,start=shortSequence?global:prefix(a,b),end=shortSequence?global+valid:prefix(a,b+1),t=row-start;
            uint32_t idx=shortSequence?shortIndex:gm<int32_t>(a.indices).GetValue(b);
            uint64_t convBase=uint64_t(idx)*a.convStride;
            if(t==0 || (carry && row%64==0)) {
                auto src=t==0?gm<float>(a.checkpoint,uint64_t(idx)*a.checkpointStride+head*16384):gm<float>(carry,head*16384);
                AscendC::DataCopy(state,src,16384);
                uint32_t accepted=gm<int32_t>(a.accepted).GetValue(b);
                uint32_t hs=t ? t-1 : (accepted?accepted-1:0);
                for(uint32_t h=0;h<3;++h) for(uint32_t i=0;i<3;++i) {
                    uint64_t c=convBase+uint64_t(hs+h)*3*width+i*width+head*128;
                    AscendC::DataCopy(hist[(((t+h)%3)*3+i)*128],gm<Bf>(a.conv,c),128);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                if(t==0) for(uint32_t h=0;h<2;++h) for(uint32_t i=0;i<3;++i)
                    AscendC::DataCopy(gm<Bf>(a.conv,convBase+uint64_t(h)*3*width+i*width+head*128),
                                      hist[((h+1)*3+i)*128],128);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if(shortSequence) {
                for(uint32_t i=0;i<3;++i) {
                    vec::shortConvolve((__ubuf__ Bf*)slot[i*2048].GetPhyAddr(),(__ubuf__ Bf*)hist[i*128].GetPhyAddr(),
                        (__ubuf__ Bf*)weights[i*512].GetPhyAddr(),(__ubuf__ float*)chunk[i*128].GetPhyAddr(),
                        valid,i<2,i==0?a.scale:1.f);
                    AscendC::DataCopyPad(gm<Bf>(a.conv,convBase+uint64_t(2)*3*width+i*width+head*128),
                        slot[i*2048],AscendC::DataCopyExtParams{static_cast<uint16_t>(valid),256,0,(3*width-128)*2,0});
                }
                for(uint32_t j=0;j<valid;++j) {
                    vec::shortDecay((__ubuf__ Bf*)slot[3*2048+j*128].GetPhyAddr(),
                        (__ubuf__ Bf*)slot[5*2048+j*16].GetPhyAddr(),(__ubuf__ float*)bias.GetPhyAddr(),aLog,a.lower,
                        (__ubuf__ float*)chunkDec[j*128].GetPhyAddr(),(__ubuf__ float*)chunkBeta[j*64].GetPhyAddr());
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                vec::shortPrepare((__ubuf__ float*)chunk.GetPhyAddr(),(__ubuf__ float*)chunkDec.GetPhyAddr(),
                    (__ubuf__ float*)weighted.GetPhyAddr(),(__ubuf__ float*)coeff.GetPhyAddr(),valid);
                AscendC::PipeBarrier<PIPE_ALL>();
                vec::shortProject((__ubuf__ float*)state.GetPhyAddr(),(__ubuf__ float*)weighted.GetPhyAddr(),
                    (__ubuf__ float*)predicted.GetPhyAddr(),(__ubuf__ float*)chunkY.GetPhyAddr(),valid);
                AscendC::PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<valid;++j) {
                    vec::shortCorrect((__ubuf__ float*)chunk.GetPhyAddr(),(__ubuf__ float*)chunkBeta.GetPhyAddr(),
                        (__ubuf__ float*)coeff.GetPhyAddr(),(__ubuf__ float*)predicted.GetPhyAddr(),
                        (__ubuf__ float*)chunkY.GetPhyAddr(),(__ubuf__ float*)chunkU.GetPhyAddr(),j);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    vec::finish((__ubuf__ float*)chunkY[j*128].GetPhyAddr(),(__ubuf__ Bf*)slot[4*2048+j*128].GetPhyAddr(),
                        (__ubuf__ float*)gamma.GetPhyAddr(),a.eps,(__ubuf__ Bf*)gatedRows[j*128].GetPhyAddr());
                    AscendC::Cast(keyRows[j*128],chunk[j*384+128],AscendC::RoundMode::CAST_RINT,128);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                uint64_t at=uint64_t(head)*a.capacity*128;
                AscendC::DataCopy(gm<float>(a.u,idx*a.uStride+at),chunkU,valid*128);
                AscendC::DataCopy(gm<float>(a.decay,idx*a.decayStride+at),chunkDec,valid*128);
                AscendC::DataCopy(gm<half>(a.key,idx*a.keyStride+at),keyRows,valid*128);
                AscendC::DataCopyPad(gm<Bf>(gated,uint64_t(ringRow)*width+head*128),gatedRows,
                    AscendC::DataCopyExtParams{static_cast<uint16_t>(valid),256,0,(width-128)*2,0});
                AscendC::PipeBarrier<PIPE_ALL>();
                return;
            }
            for(uint32_t i=0;i<3;++i) {
                auto raw=slot[i*2048+r*128];
                vec::convolve((__ubuf__ Bf*)raw.GetPhyAddr(),(__ubuf__ Bf*)hist[i*128].GetPhyAddr(),
                    (__ubuf__ Bf*)weights[i*512].GetPhyAddr(),(__ubuf__ float*)qkv[i*128].GetPhyAddr(),
                    t%3,i<2,i==0?a.scale:1.f);
                AscendC::DataCopy(gm<Bf>(a.conv,convBase+uint64_t(2+t)*3*width+i*width+head*128),raw,128);
            }
            AscendC::PipeBarrier<PIPE_V>();
            vec::step((__ubuf__ float*)state.GetPhyAddr(),(__ubuf__ float*)qkv.GetPhyAddr(),
                (__ubuf__ Bf*)slot[3*2048+r*128].GetPhyAddr(),(__ubuf__ Bf*)slot[5*2048+r*16].GetPhyAddr(),
                (__ubuf__ float*)bias.GetPhyAddr(),aLog,a.lower,(__ubuf__ float*)u.GetPhyAddr(),
                (__ubuf__ float*)dec.GetPhyAddr(),(__ubuf__ float*)y.GetPhyAddr());
            AscendC::PipeBarrier<PIPE_V>();
            vec::finish((__ubuf__ float*)y.GetPhyAddr(),(__ubuf__ Bf*)slot[4*2048+r*128].GetPhyAddr(),
                (__ubuf__ float*)gamma.GetPhyAddr(),a.eps,(__ubuf__ Bf*)gatedRow.GetPhyAddr());
            AscendC::Cast(keyRow,qkv[128],AscendC::RoundMode::CAST_RINT,128);
            AscendC::PipeBarrier<PIPE_ALL>();
            uint64_t at=(uint64_t(head)*a.capacity+t)*128;
            AscendC::DataCopy(gm<float>(a.u,idx*a.uStride+at),u,128);
            AscendC::DataCopy(gm<float>(a.decay,idx*a.decayStride+at),dec,128);
            AscendC::DataCopy(gm<half>(a.key,idx*a.keyStride+at),keyRow,128);
            AscendC::DataCopy(gm<Bf>(gated,uint64_t(ringRow+r)*width+head*128),gatedRow,128);
            if(carry && (row+1)%64==0 && row+1<end) AscendC::DataCopy(gm<float>(carry,head*16384),state,16384);
            if(row+1==end) ++b;
            AscendC::PipeBarrier<PIPE_ALL>();
        }
#endif
    }
};
} // namespace kda
