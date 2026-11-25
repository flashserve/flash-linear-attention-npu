// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_kda_fwd_prepare_general.h"
#include "../chunk_kda_fwd_prepare_struct.h"
namespace KdaPrepare::Arch22 {
__aicore__ inline void ReadMetadataPair(
    AscendC::TQue<AscendC::QuePosition::VECIN,1> &metadata,
    GM_ADDR ptr,uint64_t offset,int64_t &a,int64_t &b){
        AscendC::GlobalTensor<int64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ptr));
        auto values=metadata.AllocTensor<int64_t>();
        AscendC::DataCopyExtParams copy{1,16,0,0,0};
        AscendC::DataCopyPadExtParams<int64_t> pad{true,0,0,0};
        AscendC::DataCopyPad(values,gm[offset],copy,pad);
        metadata.EnQue(values);values=metadata.DeQue<int64_t>();
        AscendC::PipeBarrier<PIPE_ALL>();
        a=values.GetValue(0);b=values.GetValue(1);
        metadata.FreeTensor(values);
}

template<class InputT,class GateT,class BetaT>
__aicore__ inline void RunGeneral(GeneralPrepareBuffers buffers,GM_ADDR cu,GM_ADDR indices,
                                  const ChunkKdaFwdPrepareTilingData &t){
    AscendC::TPipe pipe;
    GeneralPrepare<InputT,GateT,BetaT> op;
    op.Init(pipe);
    AscendC::TQue<AscendC::QuePosition::VECIN,1> metadata;
    pipe.InitBuffer(metadata,1,32);
    const uint64_t count=static_cast<uint64_t>(t.totalChunks)*(t.isVarLen?1:t.batch)*t.valueHeadNum;
    GM_ADDR aLog=buffers.aLog;
    GM_ADDR bias=buffers.bias;
    for(uint64_t work=AscendC::GetBlockIdx();work<count;work+=t.usedCoreNum){
        const uint32_t head=work%t.valueHeadNum;
        const uint32_t chunk=work/t.valueHeadNum;
        uint32_t batch=0,begin=0,end=t.seqLen;
        if(t.isVarLen){
            int64_t sequence=0,local=chunk,start=0,stop=0;
            if(indices!=nullptr) {
                ReadMetadataPair(metadata,indices,static_cast<uint64_t>(chunk)*2,sequence,local);
                ReadMetadataPair(metadata,cu,sequence,start,stop);
            } else {
                for(;sequence<t.seqNum;++sequence) {
                    ReadMetadataPair(metadata,cu,sequence,start,stop);
                    const int64_t chunks=(stop-start+63)/64;
                    if(local<chunks)break;
                    local-=chunks;
                }
            }
            begin=static_cast<uint32_t>(start+local*64);end=static_cast<uint32_t>(stop);
        }else{
            batch=chunk/t.totalChunks;begin=(chunk%t.totalChunks)*64;
        }
        const uint64_t inputToken=t.inputSequenceMajor?
            (static_cast<uint64_t>(batch)*t.seqLen+begin)*t.valueHeadNum+head:
            (static_cast<uint64_t>(batch)*t.valueHeadNum+head)*t.seqLen+begin;
        const uint64_t cacheToken=(static_cast<uint64_t>(batch)*t.valueHeadNum+head)*t.seqLen+begin;
        GeneralPrepareChunk c{};
        c.rows=end-begin<64?end-begin:64;c.keyDim=t.keyDim;c.valueDim=t.valueDim;
        c.keyOffset=inputToken*t.keyDim;c.valueOffset=inputToken*t.valueDim;c.betaOffset=inputToken;
        c.keyStride=t.keyDim*(t.inputSequenceMajor?t.valueHeadNum:1);
        c.valueStride=t.valueDim*(t.inputSequenceMajor?t.valueHeadNum:1);
        c.betaStride=t.inputSequenceMajor?t.valueHeadNum:1;
        c.cacheKeyOffset=cacheToken*t.keyDim;c.cacheValueOffset=cacheToken*t.valueDim;
        c.cacheScalarOffset=cacheToken;c.scoreOffset=cacheToken*64;
        c.scale=t.scale;c.epsilon=t.epsilon;c.lowerBound=t.lowerBound;
        c.normalize=t.normMode!=0;c.useExp2=t.useExp2;c.gateMode=t.gateMode;c.betaMode=t.betaMode;
        buffers.aLog=aLog==nullptr?nullptr:aLog+head*sizeof(float);
        buffers.bias=bias==nullptr?nullptr:bias+static_cast<uint64_t>(head)*t.keyDim*sizeof(float);
        op.Process(buffers,c);
    }
}
} // namespace KdaPrepare::Arch22
