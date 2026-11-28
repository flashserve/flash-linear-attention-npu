#pragma once
#include <prelude_native.hpp>
namespace Prelude {
using namespace Catlass;
using namespace AscendC;
// Convert directly into video scratch. These operands are dead before text or video Gram A starts.
__aicore__ inline void CastAlphaInputs(Addresses& g) {
    if(!g.alphaBf16) return;
    const int sizes[5]={g.f*g.width,128*g.width,g.h*128*128,g.h*128,g.h};
    int64_t offset=0;
    for(int i=0;i<5;++i){
        if(!(g.alphaBf16 & (1<<i))) continue;
        GM_ADDR dst=g.p[K32]+offset*sizeof(float);
#if defined(__DAV_VEC__)
        constexpr int chunk=2048;
        LocalTensor<bfloat16_t> buffer(TPosition::VECCALC,0,chunk);
        LocalTensor<float> result(TPosition::VECCALC,chunk*2,chunk);
        GlobalTensor<bfloat16_t> source; GlobalTensor<float> converted;
        source.SetGlobalBuffer((__gm__ bfloat16_t*)g.extra[i]);
        converted.SetGlobalBuffer((__gm__ float*)dst);
        for(int first=GetBlockIdx()*chunk;first<sizes[i];first+=GetBlockNum()*GetSubBlockNum()*chunk){
            const uint32_t count=sizes[i]-first<chunk?sizes[i]-first:chunk;
            DataCopyPad(buffer,source[first],{1,count*2,0,0,0},{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            Cast(result,buffer,RoundMode::CAST_NONE,count);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(converted[first],result,{1,count*4,0,0,0});
            PipeBarrier<PIPE_ALL>();
        }
#endif
        g.extra[i]=dst;
        offset+=(sizes[i]+7)/8*8;
    }
    SyncAll<false>();
}
#if defined(__DAV_CUBE__)
template<typename T,bool leftT,bool rightT>
__aicore__ inline void Matrix(GM_ADDR ap,GM_ADDR bp,GM_ADDR cp,int batch,int m,int n,int k) {
    using LA=typename std::conditional<leftT,layout::ColumnMajor,layout::RowMajor>::type;
    using LB=typename std::conditional<rightT,layout::ColumnMajor,layout::RowMajor>::type;
    using A=Gemm::GemmType<T,LA>;using B=Gemm::GemmType<T,LB>;using C=Gemm::GemmType<T,layout::RowMajor>;
    using Copy=Gemm::Tile::TileCopy<Arch::AtlasA2,A,B,C>;
    using AM=typename Copy::CopyL1ToL0A::LayoutSrc;using BM=typename Copy::CopyL1ToL0B::LayoutSrc;
    using AL=typename Copy::CopyL1ToL0A::LayoutDst;using BL=typename Copy::CopyL1ToL0B::LayoutDst;
    Arch::Resource<Arch::AtlasA2> resource;
    auto a=resource.l1Buf.template GetBufferByByte<T>(0);
    auto b=resource.l1Buf.template GetBufferByByte<T>(65536);
    auto a0=resource.l0ABuf.template GetBufferByByte<T>(0);auto b0=resource.l0BBuf.template GetBufferByByte<T>(0);
    auto c0=resource.l0CBuf.template GetBufferByByte<float>(0);
    typename Copy::CopyGmToL1A inA;typename Copy::CopyGmToL1B inB;
    typename Copy::CopyL1ToL0A loadA;typename Copy::CopyL1ToL0B loadB;
    typename Copy::CopyL0CToGm save;Gemm::Tile::TileMmad<Arch::AtlasA2,A,B,void> mm;
    GlobalTensor<T> aG,bG,cG;
    aG.SetGlobalBuffer((__gm__ T*)ap);bG.SetGlobalBuffer((__gm__ T*)bp);cG.SetGlobalBuffer((__gm__ T*)cp);
    SetHF32Mode(HF32Mode::DISABLE);
    const int mt=(m+31)/32,nt=(n+63)/64;
    for(int job=GetBlockIdx();job<batch*mt*nt;job+=GetBlockNum()){
        const int j=job/(mt*nt),r=(job/nt%mt)*32,col=(job%nt)*64;
        const uint32_t rows=m-r<32?m-r:32,cols=n-col<64?n-col:64;
        for(int off=0;off<k;off+=32){
            const uint32_t count=k-off<32?k-off:32;
            auto la=AM::template MakeLayout<T>(rows,count);auto lb=BM::template MakeLayout<T>(count,cols);
            auto aoff=int64_t(j)*m*k+(leftT?off*m+r:r*k+off);
            auto boff=int64_t(j)*n*k+(rightT?col*k+off:off*n+col);
            inA(a,aG[aoff],la,LA{rows,count,leftT?m:k});
            inB(b,bG[boff],lb,LB{count,cols,rightT?k:n});PipeBarrier<PIPE_ALL>();
            loadA(a0,a,AL::template MakeLayout<T>(rows,count),la);
            loadB(b0,b,BL::template MakeLayout<T>(count,cols),lb);PipeBarrier<PIPE_ALL>();
            mm(c0,a0,b0,rows,cols,count,off==0,0);PipeBarrier<PIPE_ALL>();
        }
        auto lc=layout::zN::MakeLayoutInL0C(MakeCoord(rows,cols));
        save(cG[(int64_t(j)*m+r)*n+col],c0,layout::RowMajor{rows,cols,n},lc);
        PipeBarrier<PIPE_ALL>();
    }
}
#endif
#if defined(__DAV_VEC__)
__aicore__ inline GM_ADDR PackText(Addresses g,int input,int stride,GM_ADDR workspace) {
    const int rowStride=g.textStride[stride], headStride=g.textStride[stride+1];
    if(rowStride==g.h*128 && headStride==128) return g.extra[input];
    GlobalTensor<uint16_t> source,dest;
    source.SetGlobalBuffer((__gm__ uint16_t*)g.extra[input]);
    dest.SetGlobalBuffer((__gm__ uint16_t*)workspace);
    LocalTensor<uint16_t> ub(TPosition::VECCALC,0,32*128);
    const int tiles=(g.length+31)/32;
    for(int job=GetBlockIdx();job<g.h*tiles;job+=GetBlockNum()*GetSubBlockNum()){
        const int head=job/tiles,first=(job%tiles)*32;
        const uint16_t rows=g.length-first<32?g.length-first:32;
        DataCopyExtParams read{rows,256,uint32_t(rowStride-128)*2,0,0};
        DataCopyPad(ub,source[int64_t(first)*rowStride+head*headStride],read,{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams write{rows,256,0,uint32_t(g.h-1)*256,0};
        DataCopyPad(dest[(int64_t(first)*g.h+head)*128],ub,write);PipeBarrier<PIPE_ALL>();
    }
    return workspace;
}
__aicore__ inline GM_ADDR PackBeta(Addresses g) {
    const int headStride=g.textStride[4],rowStride=g.textStride[5];
    if(headStride==g.length && rowStride==1) return g.extra[7];
    GlobalTensor<uint16_t> src,dst;
    src.SetGlobalBuffer((__gm__ uint16_t*)g.extra[7]);dst.SetGlobalBuffer((__gm__ uint16_t*)g.p[Y]);
    LocalTensor<uint16_t> ub(TPosition::VECCALC,0,64*16);
    const int tiles=(g.length+63)/64;
    for(int job=GetBlockIdx();job<g.h*tiles;job+=GetBlockNum()*GetSubBlockNum()){
        const int head=job/tiles,first=(job%tiles)*64;
        const uint16_t rows=g.length-first<64?g.length-first:64;
        DataCopyExtParams read{rows,2,uint32_t(rowStride-1)*2,0,0};
        DataCopyPad(ub,src[int64_t(head)*headStride+int64_t(first)*rowStride],read,{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
        for(int i=1;i<rows;++i) ub.SetValue(i,ub.GetValue(16*i));
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[int64_t(head)*g.length+first],ub,{1,uint32_t(rows)*2,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
    return g.p[Y];
}
__aicore__ inline void Gate(Addresses g, GM_ADDR delta) {
    GlobalTensor<float> x,bias,scale,result;
    x.SetGlobalBuffer((__gm__ float*)delta);bias.SetGlobalBuffer((__gm__ float*)g.extra[3]);
    scale.SetGlobalBuffer((__gm__ float*)g.extra[4]);result.SetGlobalBuffer((__gm__ float*)g.p[Alpha]);
    LocalTensor<float> v(TPosition::VECCALC,0,128*5);
    LocalTensor<uint8_t> roundedOne(TPosition::VECCALC,4096,32);
    LocalTensor<uint8_t> aboveThreshold(TPosition::VECCALC,4128,32);
    for(int j=GetBlockIdx();j<g.f*g.h;j+=GetBlockNum()*GetSubBlockNum()){
        int h=j%g.h;DataCopy(v,x[j*128],128);DataCopy(v[128],bias[h*128],128);
        Duplicate(v[256],scale.GetValue(h),128);PipeBarrier<PIPE_ALL>();
        Add(v,v,v[128],128);Exp(v[256],v[256],128);PipeBarrier<PIPE_V>();
        Exp(v[512],v,128);PipeBarrier<PIPE_V>();Adds(v[384],v[512],1.f,128);PipeBarrier<PIPE_V>();
        CompareScalar(roundedOne,v[384],1.f,CMPMODE::EQ,128);
        CompareScalar(aboveThreshold,v,20.f,CMPMODE::GT,128);
        PipeBarrier<PIPE_V>();
        Adds(v[128],v[384],-1.f,128);PipeBarrier<PIPE_V>();Div(v[128],v[512],v[128],128);PipeBarrier<PIPE_V>();
        Ln(v[384],v[384],128);PipeBarrier<PIPE_V>();Mul(v[384],v[384],v[128],128);PipeBarrier<PIPE_ALL>();
        Select(v[384],roundedOne,v[512],v[384],SELMODE::VSEL_TENSOR_TENSOR_MODE,128);
        PipeBarrier<PIPE_V>();
        Select(v[384],aboveThreshold,v,v[384],SELMODE::VSEL_TENSOR_TENSOR_MODE,128);
        PipeBarrier<PIPE_V>();
        Mul(v,v[384],v[256],128);PipeBarrier<PIPE_V>();Muls(v,v,-1.f,128);PipeBarrier<PIPE_V>();
        Exp(v,v,128);PipeBarrier<PIPE_ALL>();DataCopy(result[j*128],v,128);PipeBarrier<PIPE_ALL>();
    }
}
#endif
#if defined(__DAV_CUBE__)
__aicore__ inline void Down(Addresses g) {
    Projection::Run(g.extra[0],g.extra[1],g.p[Trans],g.extra[8]);
}
__aicore__ inline void Up(Addresses g) {
    Projection::Run(g.p[Trans],g.extra[2],g.p[First],g.extra[9]);
}
#endif
__aicore__ inline void Run(Addresses g) {
    auto delta=g.p[First];
#if defined(__DAV_CUBE__)
    Down(g);
    PipeBarrier<PIPE_ALL>();
#endif
    SyncAll<false>();
#if defined(__DAV_CUBE__)
    Up(g);
#endif
    SyncAll<false>();
#if defined(__DAV_VEC__)
    Gate(g,delta);
#endif
    if(g.length){
#if defined(__DAV_VEC__)
        const auto k=PackText(g,5,0,g.p[Query]);
        const auto v=PackText(g,6,2,g.p[WK32]);
        g.extra[7]=PackBeta(g);
        SyncAll<true>();
        JyKV(0,nullptr,nullptr,k,v,g.extra[7],g.p[Key],g.p[Value],1,g.h,g.length,
             GetBlockNum()*GetSubBlockNum(),1,1,GetBlockIdx());
#endif
    }
    SyncAll<false>();
    if(!g.length) return;
    Addresses t=g;t.f=1;t.p[Beta]=g.extra[7];
#if defined(__DAV_VEC__)
    Front::CastKeys(t,g.length);
#elif defined(__DAV_CUBE__)
    TextB::Run(g.p[Value],g.p[Key],g.p[B16],g.extra[10]+256);
#endif
    SyncAll<false>();
#if defined(__DAV_CUBE__)
    TextA::Run(g.p[WK32],g.p[K32],g.p[GramA],g.extra[10]);
#else
    Front::CastB(t);
    t.seed=0; Front::SeedBanks(t);
#endif
    SyncAll<false>();
#if defined(__DAV_VEC__)
    JyInit(0,nullptr,nullptr,g.p[GramA],g.p[NegativeMatrix],g.p[First],g.h,GetBlockNum()*GetSubBlockNum(),1,1,GetBlockIdx());
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> ub(TPosition::VECCALC,0,16384);GlobalTensor<float> eye,x;
    eye.SetGlobalBuffer((__gm__ float*)g.p[Eye]);x.SetGlobalBuffer((__gm__ float*)g.p[First]);
#endif
    SyncAll<false>();
#if defined(__DAV_VEC__)
    for(int j=GetBlockIdx();j<g.h;j+=GetBlockNum()*GetSubBlockNum()){
        DataCopy(ub,eye,16384);PipeBarrier<PIPE_ALL>();Muls(ub,ub,g.textScale,16384);PipeBarrier<PIPE_ALL>();
        DataCopy(x[j*16384],ub,16384);PipeBarrier<PIPE_ALL>();
    }
#endif
    SyncAll<false>();
    JyShallowNewton::Params p{g.p[NegativeMatrix],g.p[First],g.p[TwoEye],g.p[Work],g.p[Inv],nullptr,nullptr,(uint32_t)g.h,(uint32_t)g.textSteps,0.f};
    JyShallowNewton::RunFour(p);SyncAll<false>();
#if defined(__DAV_CUBE__)
    Matrix<float,false,false>(g.p[B32],g.p[Inv],g.p[Trans],g.h,128,128,128);
#endif
    SyncAll<false>();
#if defined(__DAV_VEC__)
    GlobalTensor<float> inj,seed;inj.SetGlobalBuffer((__gm__ float*)g.p[Trans]);seed.SetGlobalBuffer((__gm__ float*)g.p[Seed]);
    for(int j=GetBlockIdx();j<g.h;j+=GetBlockNum()*GetSubBlockNum()){
        DataCopy(ub,inj[j*16384],16384);PipeBarrier<PIPE_ALL>();Muls(ub,ub,0.5f,16384);PipeBarrier<PIPE_ALL>();
        DataCopy(seed[j*16384],ub,16384);PipeBarrier<PIPE_ALL>();
    }
#endif
    SyncAll<false>();
}
}
