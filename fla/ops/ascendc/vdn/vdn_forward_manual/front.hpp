#pragma once
#include "params.hpp"
#include "conv_body.hpp"
#include "triton.hpp"
#include "gram_body.hpp"
struct Addresses {GM_ADDR p[BufferCount];int f,h,bridgePanel,steps,conv,seed; GM_ADDR clocks; int debug;};
namespace Front {
using namespace AscendC;
constexpr int S=405,D=128;
#if defined(__DAV_VEC__)
__aicore__ inline void Pack(GM_ADDR src,GM_ADDR packed,GM_ADDR tables,int frames,int heads) {
    constexpr int tokens=32,groups=16,lanes=16,elems=tokens*groups*lanes;
    GlobalTensor<uint16_t> input,output;GlobalTensor<uint32_t> offsets;
    input.SetGlobalBuffer((__gm__ uint16_t*)src);output.SetGlobalBuffer((__gm__ uint16_t*)packed);
    offsets.SetGlobalBuffer((__gm__ uint32_t*)(tables+256+6480*4));
    LocalTensor<uint16_t> u(TPosition::VECCALC,0,2*elems);
    LocalTensor<uint32_t> index(TPosition::VECCALC,2*elems*sizeof(uint16_t),elems);
    DataCopy(index,offsets,elems);PipeBarrier<PIPE_ALL>();
    int c1=heads*8,panels=(c1+groups-1)/groups,rows=(S+tokens-1)/tokens;
    for(int job=GetBlockIdx();job<frames*rows*panels;job+=GetBlockNum()*GetSubBlockNum()) {
        int f=job/(rows*panels),t=job/panels%rows*tokens,c=job%panels*groups;
        uint16_t nr=S-t<tokens?S-t:tokens,nc=c1-c<groups?c1-c:groups;
        DataCopy(u,input[((int64_t(f)*S+t)*c1+c)*lanes],{nr,nc,uint16_t(c1-nc),uint16_t(groups-nc)});
        PipeBarrier<PIPE_ALL>();
        Gather(u[elems],u,index,0,nc*tokens*lanes);PipeBarrier<PIPE_ALL>();
        for(int z=0;z<nc;++z)
            DataCopy(output[((int64_t(f)*c1+c+z)*S+t)*lanes],u[elems+z*tokens*lanes],nr*lanes);
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void Unpack(GM_ADDR src,GM_ADDR dst,int f,int h) {
    GlobalTensor<bfloat16_t> a,b;a.SetGlobalBuffer((__gm__ bfloat16_t*)src);b.SetGlobalBuffer((__gm__ bfloat16_t*)dst);
    LocalTensor<bfloat16_t> ub(TPosition::VECCALC,0,S*D);
    for(int j=GetBlockIdx();j<f*h;j+=GetBlockNum()*GetSubBlockNum()){
        for(int z=0;z<8;++z) DataCopy(ub[z*16],a[(int64_t(j)*8+z)*S*16],{S,1,0,7});
        PipeBarrier<PIPE_ALL>();DataCopy(b[int64_t(j)*S*D],ub,S*D);PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void PackWeights(GM_ADDR sp,GM_ADDR tm,GM_ADDR filter,GM_ADDR weights,GM_ADDR lut,int h){
    GlobalTensor<uint16_t> a,t,b,u;a.SetGlobalBuffer((__gm__ uint16_t*)sp);t.SetGlobalBuffer((__gm__ uint16_t*)tm);
    b.SetGlobalBuffer((__gm__ uint16_t*)filter);u.SetGlobalBuffer((__gm__ uint16_t*)weights);
    LocalTensor<uint16_t> buf(TPosition::VECCALC,0,8000);
    auto m=buf[6400],x=buf[6816],o=buf[6896];
    LocalTensor<uint32_t> indices(TPosition::VECCALC,32768,6480);
    GlobalTensor<uint32_t> offsets;offsets.SetGlobalBuffer((__gm__ uint32_t*)(lut+256));
    DataCopy(indices,offsets,6480);Duplicate(m[400],uint16_t(0),16);PipeBarrier<PIPE_ALL>();
    for(int j=GetBlockIdx();j<h*8;j+=GetBlockNum()*GetSubBlockNum()){

        DataCopy(m,a[j*400],400);DataCopy(x,t[j*80],80);PipeBarrier<PIPE_ALL>();
        Gather(buf,m,indices,0,6400);Gather(o,x,indices[6400],0,80);
        PipeBarrier<PIPE_ALL>();DataCopy(b[j*6400],buf,6400);
        for(int k=0;k<5;++k) DataCopy(u[k*h*128+j*16],o[k*16],16);
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void SeedBanks(Addresses g) {
    LocalTensor<float> ub(TPosition::VECCALC,0,16384);
    GlobalTensor<float> eye,p,s,init;
    eye.SetGlobalBuffer((__gm__ float*)g.p[Eye]);p.SetGlobalBuffer((__gm__ float*)g.p[Prefix]);s.SetGlobalBuffer((__gm__ float*)g.p[Suffix]);
    init.SetGlobalBuffer((__gm__ float*)g.p[Seed]);
    if(GetBlockIdx()==0){
        Duplicate(ub,0.f,16384);PipeBarrier<PIPE_ALL>();
        for(int d=0;d<128;++d) ub.SetValue(d*129,1.f);
        PipeBarrier<PIPE_ALL>();DataCopy(eye,ub,16384);PipeBarrier<PIPE_ALL>();
    }
    for(int j=GetBlockIdx();j<g.h;j+=GetBlockNum()*GetSubBlockNum()){
        if(g.seed) DataCopy(ub,init[j*16384],16384);else Duplicate(ub,0.f,16384);
        PipeBarrier<PIPE_ALL>();DataCopy(p[j*16384],ub,16384);DataCopy(s[(int64_t(g.f)*g.h+j)*16384],ub,16384);PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void CastKeys(Addresses g) {
    LocalTensor<float> vals(TPosition::VECCALC,0,4096);auto factors=vals[4096],scales=vals[8192];
    auto bf=vals[8256].ReinterpretCast<bfloat16_t>();auto bt=bf[4096];
    GlobalTensor<bfloat16_t> k,b;k.SetGlobalBuffer((__gm__ bfloat16_t*)g.p[Key]);b.SetGlobalBuffer((__gm__ bfloat16_t*)g.p[Beta]);
    GlobalTensor<float> x,w;x.SetGlobalBuffer((__gm__ float*)g.p[K32]);w.SetGlobalBuffer((__gm__ float*)g.p[WK32]);
    int rows=g.f*g.h*S;
    for(int j=GetBlockIdx();j<(rows+31)/32;j+=GetBlockNum()*GetSubBlockNum()){
        int r=32*j;int n=rows-r<32?rows-r:32;
        DataCopy(bf,k[int64_t(r)*128],n*128);
        DataCopyPad(bt,b[r],{1,uint32_t(2*n),0,0,0},{true,0,0,0});PipeBarrier<PIPE_ALL>();
        Cast(vals,bf,RoundMode::CAST_NONE,n*128);Cast(scales,bt,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
        for(int c=0;c<16;++c) Brcb(factors[c*8],scales,(n+7)/8,{16,128});
        PipeBarrier<PIPE_ALL>();DataCopy(x[int64_t(r)*128],vals,n*128);Mul(factors,factors,vals,n*128);PipeBarrier<PIPE_ALL>();
        DataCopy(w[int64_t(r)*128],factors,n*128);PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void CastB(Addresses g){
    GlobalTensor<bfloat16_t> src;GlobalTensor<float> dst;src.SetGlobalBuffer((__gm__ bfloat16_t*)g.p[B16]);dst.SetGlobalBuffer((__gm__ float*)g.p[B32]);
    LocalTensor<float> v(TPosition::VECCALC,0,8192);auto b=v[8192].ReinterpretCast<bfloat16_t>();
    for(int j=GetBlockIdx();j<g.f*g.h*2;j+=GetBlockNum()*GetSubBlockNum()){
        int64_t off=int64_t(j)*8192;DataCopy(b,src[off],8192);PipeBarrier<PIPE_ALL>();Cast(v,b,RoundMode::CAST_NONE,8192);PipeBarrier<PIPE_ALL>();DataCopy(dst[off],v,8192);PipeBarrier<PIPE_ALL>();
    }
}
#endif
// Device counter rows contain begin/end for each physical worker and each operation.
__aicore__ inline void Mark(Addresses g,int task,int end) {
    if(g.debug){
        PipeBarrier<PIPE_ALL>();
#if defined(__DAV_CUBE__)
        int lane=GetBlockIdx();
#else
        int lane=20+GetBlockIdx();
#endif
        ((__gm__ int64_t*)g.clocks)[(task*60+lane)*16+end]=GetSystemCycle();
    }
}
#if defined(__DAV_VEC__)
__aicore__ inline void PackProjection(Addresses g,bool value,GM_ADDR packed,GM_ADDR filter) {
    PackWeights(g.p[value?Vsp:Ksp],g.p[value?Vtm:Ktm],filter,g.p[value?VTime:KTime],g.p[ConvT],g.h);
    Pack(g.p[value?V:K],packed,g.p[ConvT],g.f,g.h);
}
__aicore__ inline void Temporal(Addresses g,bool value){
    for(int id=GetBlockIdx();id<40;id+=GetBlockNum()*GetSubBlockNum()){
        if(value) JyValue(0,nullptr,nullptr,g.p[Panel],g.p[VTime],g.p[Value],g.p[Beta],g.h,S,0,g.f,g.f,40,1,1,id);
        else JyKey(0,nullptr,nullptr,g.p[Panel],g.p[KTime],g.p[Key],g.p[Key],g.h,S,0,g.f,g.f,40,1,1,id);
    }
}
__aicore__ inline void QueryFeatures(Addresses g){
    Mark(g,4,0);
    for(int id=GetBlockIdx();id<40;id+=GetBlockNum()*GetSubBlockNum()) JyQuery(0,nullptr,nullptr,g.p[Q],g.p[Query],g.f,g.h,S,40,1,1,id);
    Mark(g,4,1);
}
#endif
#if defined(__DAV_CUBE__)
__aicore__ inline void Spatial(Addresses g,GM_ADDR packed,GM_ADDR filter){
    OriginalConv::Spatial2097177((__gm__ bfloat16_t*)packed,(__gm__ bfloat16_t*)filter,(__gm__ bfloat16_t*)g.p[ConvOut],g.p[ConvT]);
}
#endif
template<int Version> __aicore__ inline void Prepare(Addresses g){
#if defined(__DAV_VEC__)
    SeedBanks(g);
#endif
    SyncAll<false>();
    if(g.conv) {
        // K32 first becomes live after both spatial convolutions and temporal features.
        GM_ADDR packedV=Version>=5?g.p[K32]:g.p[Packed];
        GM_ADDR filterV=Version>=5?packedV+int64_t(g.f)*g.h*S*D*sizeof(bfloat16_t):g.p[Filter];
#if defined(__DAV_VEC__)
        PackProjection(g,false,g.p[Packed],g.p[Filter]);
#endif
        SyncAll<false>();
#if defined(__DAV_CUBE__)
        Mark(g,9,0);Spatial(g,g.p[Packed],g.p[Filter]);Mark(g,9,1);
#elif defined(__DAV_VEC__)
        if constexpr(Version>=5){Mark(g,10,0);PackProjection(g,true,packedV,filterV);Mark(g,10,1);}
#endif
        SyncAll<false>();
#if defined(__DAV_VEC__)
        Unpack(g.p[ConvOut],g.p[Panel],g.f,g.h);
#endif
        SyncAll<false>();
        if constexpr(Version==0){
#if defined(__DAV_VEC__)
            Mark(g,0,0);Temporal(g,false);Mark(g,0,1);
#endif
            SyncAll<false>();
        }
#if defined(__DAV_VEC__)
        if constexpr(Version<5){Mark(g,10,0);PackProjection(g,true,packedV,filterV);Mark(g,10,1);}
#endif
        SyncAll<false>();
#if defined(__DAV_CUBE__)
        Mark(g,1,0);Spatial(g,packedV,filterV);Mark(g,1,1);
#elif defined(__DAV_VEC__)
        if constexpr(Version>=1){Mark(g,0,0);Temporal(g,false);Mark(g,0,1);}
#endif
        SyncAll<false>();
#if defined(__DAV_VEC__)
        Unpack(g.p[ConvOut],g.p[Panel],g.f,g.h);
#endif
        SyncAll<false>();
#if defined(__DAV_VEC__)
        Temporal(g,true);
#endif
        SyncAll<false>();
    }else{
#if defined(__DAV_VEC__)
        for(int id=GetBlockIdx();id<40;id+=GetBlockNum()*GetSubBlockNum()){
            if(g.h==7) JyKV7(0,nullptr,nullptr,g.p[K],g.p[V],g.p[Beta],g.p[Key],g.p[Value],g.f,g.h,S,40,1,1,id);
            else JyKV14(0,nullptr,nullptr,g.p[K],g.p[V],g.p[Beta],g.p[Key],g.p[Value],g.f,g.h,S,40,1,1,id);
        }
#endif
        SyncAll<false>();
    }
#if defined(__DAV_VEC__)
    if constexpr(Version<3) QueryFeatures(g);
    Mark(g,2,0);CastKeys(g);Mark(g,2,1);
#elif defined(__DAV_CUBE__)
    if constexpr(Version>=2){Mark(g,3,0);OriginalGramB::Run(g.p[Value],g.p[Key],g.p[B16],g.p[GramBT]);Mark(g,3,1);}
#endif
    SyncAll<false>();
#if defined(__DAV_CUBE__)
    if constexpr(Version<2){Mark(g,3,0);OriginalGramB::Run(g.p[Value],g.p[Key],g.p[B16],g.p[GramBT]);Mark(g,3,1);PipeBarrier<PIPE_ALL>();}
    Mark(g,5,0);OriginalGramA::Run(g.p[WK32],g.p[K32],g.p[GramA],g.p[GramAT]);Mark(g,5,1);
#elif defined(__DAV_VEC__)
    if constexpr(Version>=3){QueryFeatures(g);Mark(g,6,0);CastB(g);Mark(g,6,1);}
#endif
    SyncAll<false>();
#if defined(__DAV_VEC__)
    if constexpr(Version<3){Mark(g,6,0);CastB(g);Mark(g,6,1);}
    for(int id=GetBlockIdx();id<40;id+=GetBlockNum()*GetSubBlockNum()) JyInit(0,nullptr,nullptr,g.p[GramA],g.p[NegativeMatrix],g.p[First],g.f*g.h,40,1,1,id);
#endif
    SyncAll<false>();
}
}
