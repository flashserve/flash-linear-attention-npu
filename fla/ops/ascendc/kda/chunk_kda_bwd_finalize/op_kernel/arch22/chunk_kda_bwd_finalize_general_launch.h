// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_kda_bwd_finalize_general.h"
#include "../../../chunk_kda_fwd_finalize/op_kernel/arch22/chunk_kda_fwd_finalize_general_launch.h"
namespace KDA::Arch22 {
template<class InputT,class GateT,class BetaT>
__aicore__ inline void RunGeneralBwdFinalize(GeneralBwdFinalizeBuffers b,GM_ADDR cu,GM_ADDR indices,
    GM_ADDR pa,GM_ADDR pb,GM_ADDR paOut,GM_ADDR pbOut,GM_ADDR da,GM_ADDR db,GM_ADDR scratch,
    const FinalizeTilingData &t,bool reducing){
    AscendC::TPipe pipe;GeneralBwdFinalize<InputT,GateT,BetaT> op;op.Init(pipe);
    if(reducing){
        for(uint32_t head=AscendC::GetBlockIdx();head<t.heads;head+=AscendC::GetBlockNum())
            op.ReduceParameters(t.chunks,t.heads,head,t.keyDim,pa,pb,da,db);
        return;
    }
    AscendC::TQue<AscendC::QuePosition::VECIN,1> metadata;pipe.InitBuffer(metadata,1,32);
    const uint64_t keyBytes=static_cast<uint64_t>(t.batch)*t.heads*t.tokens*t.keyDim*4;
    b.scratchW=scratch;b.scratchQ=scratch+keyBytes;b.scratchK=scratch+2*keyBytes;
    b.scratchG=scratch+3*keyBytes;b.scratchBeta=scratch+4*keyBytes;
    GM_ADDR alog=b.aLog;GM_ADDR bias=b.bias;
    for(uint64_t task=AscendC::GetBlockIdx();task<static_cast<uint64_t>(t.chunks*t.heads);task+=AscendC::GetBlockNum()){
        uint64_t chunk=task/t.heads,head=task%t.heads,batch=t.isVariable?0:chunk/((t.tokens+63)/64);
        uint64_t begin=(t.isVariable?chunk:chunk%((t.tokens+63)/64))*64,end=t.tokens;
        if(t.isVariable){int64_t seq,local,start,stop;ReadFinalizePair(metadata,indices,chunk*2,seq,local);ReadFinalizePair(metadata,cu,seq,start,stop);begin=start+local*64;end=stop;}
        uint64_t token=(batch*t.heads+head)*t.tokens+begin;
        GeneralBwdFinalizeChunk c{};c.rows=end-begin<64?end-begin:64;c.keyDim=t.keyDim;c.valueDim=t.valueDim;
        c.keyOffset=token*t.keyDim;c.valueOffset=token*t.valueDim;c.scalarOffset=token;c.scoreOffset=token*64;
        c.stateOffset=task*t.keyDim*t.valueDim;c.inputKeyOffset=c.keyOffset;c.inputValueOffset=c.valueOffset;c.inputScalarOffset=token;
        c.keyStride=t.keyDim;c.valueStride=t.valueDim;c.scalarStride=1;c.scale=t.scale;c.lowerBound=t.lowerBound;
        c.normalize=t.hasQkL2Norm;c.useExp2=t.useExp2;c.stateVFirst=t.stateVFirst;c.gateMode=t.gateMode;c.betaMode=t.betaMode;
        b.aLog=alog?alog+head*4:nullptr;b.bias=bias?bias+head*t.keyDim*4:nullptr;
        b.partialA=paOut+task*4;b.partialBias=pbOut+task*t.keyDim*4;
        op.Process(b,c);
    }
}
}
