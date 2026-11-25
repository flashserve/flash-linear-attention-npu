// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_kda_bwd_recompute_general.h"
#include "../../../chunk_kda_fwd_finalize/op_kernel/arch22/chunk_kda_fwd_finalize_general_launch.h"
namespace KDA::Arch22 {
template<class ValueT,class GateT>
__aicore__ inline void RunGeneralRecompute(GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR g,GM_ADDR beta,GM_ADDR a,
    GM_ADDR aLog,GM_ADDR bias,GM_ADDR cu,GM_ADDR indices,GM_ADDR w,GM_ADDR u,GM_ADDR qg,GM_ADDR kg,GM_ADDR gk,
    const KDA::ChunkKdaBwdRecomputeTilingData &t){
    AscendC::TPipe pipe;GeneralRecompute<ValueT,GateT> op;op.Init(pipe);
    AscendC::TQue<AscendC::QuePosition::VECIN,1> metadata;pipe.InitBuffer(metadata,1,32);
    for(uint64_t task=AscendC::GetBlockIdx();task<static_cast<uint64_t>(t.chunkNum*t.Hv);task+=AscendC::GetBlockNum()){
        uint64_t chunk=task/t.Hv,head=task%t.Hv,batch=t.isVariable?0:chunk/((t.T+63)/64);
        uint64_t begin=(t.isVariable?chunk:chunk%((t.T+63)/64))*64,end=t.T;
        if(t.isVariable){
            int64_t seq=0,local=chunk,start=0,stop=0;
            if(indices){ReadFinalizePair(metadata,indices,chunk*2,seq,local);ReadFinalizePair(metadata,cu,seq,start,stop);}
            else for(;seq<t.vecRow;++seq){ReadFinalizePair(metadata,cu,seq,start,stop);auto n=(stop-start+63)/64;if(local<n)break;local-=n;}
            begin=start+local*64;end=stop;
        }
        uint64_t token=(batch*t.Hv+head)*t.T+begin,tm=(batch*t.T+begin)*t.Hv+head;
        const bool valueTm=t.reserved&4,gateTm=(t.reserved&1) && t.useGateInKernel;
        KdaPrepare::Arch22::GeneralPrepareChunk c{};
        c.rows=end-begin<64?end-begin:64;c.keyDim=t.K;c.valueDim=t.V;
        c.keyOffset=(gateTm?tm:token)*t.K;c.keyStride=(gateTm?t.Hv:1)*t.K;
        c.valueOffset=(valueTm?tm:token)*t.V;c.valueStride=(valueTm?t.Hv:1)*t.V;
        c.cacheKeyOffset=token*t.K;c.cacheValueOffset=token*t.V;c.cacheScalarOffset=token;c.scoreOffset=token*64;
        c.scale=1.0F;c.useExp2=t.useExp2;c.gateMode=t.useGateInKernel?2:3;
        uint32_t bits=t.lowerBoundBits;c.lowerBound=*reinterpret_cast<float *>(&bits);
        KdaPrepare::Arch22::GeneralPrepareBuffers b{};
        b.v=v;b.gate=g;b.aLog=t.hasALog?aLog+head*4:nullptr;b.bias=t.hasDtBias?bias+head*t.K*4:nullptr;
        b.qHat=q;b.kHat=k;b.betaEff=beta;b.akk=a;b.gk=gk;b.w=w;b.u=u;b.qg=qg;b.kg=kg;
        op.Recompute(b,c);
    }
}
}
