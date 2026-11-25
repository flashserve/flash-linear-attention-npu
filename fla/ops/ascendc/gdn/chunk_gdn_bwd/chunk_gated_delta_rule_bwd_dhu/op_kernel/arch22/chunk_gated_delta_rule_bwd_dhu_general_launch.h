// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_gated_delta_rule_bwd_dhu_general.h"

namespace GDN::Arch22 {
__aicore__ inline void ReadDhuPair(AscendC::TQue<AscendC::QuePosition::VECIN,1> &queue,
    GM_ADDR ptr,uint64_t offset,int64_t &a,int64_t &b) {
    AscendC::GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ptr));
    auto local=queue.AllocTensor<int64_t>();
    AscendC::DataCopyExtParams copy{1,16,0,0,0};
    AscendC::DataCopyPadExtParams<int64_t> pad{true,0,0,0};
    AscendC::DataCopyPad(local,gm[offset],copy,pad);
    queue.EnQue(local);local=queue.DeQue<int64_t>();
    AscendC::PipeBarrier<PIPE_ALL>();
    a=local.GetValue(0);b=local.GetValue(1);queue.FreeTensor(local);
}

template<class GradT>
__aicore__ inline void RunGeneralDhu(const GeneralDhuBuffers &b,GM_ADDR cu,const GDN::ChunkGatedDeltaRuleBwdDhuTilingData &t){
    AscendC::TPipe pipe;GeneralDhu<GradT> op;op.Init(pipe);
    AscendC::TQue<AscendC::QuePosition::VECIN,1> metadata;pipe.InitBuffer(metadata,1,32);
    const uint64_t tiles=(t.V+31)/32;
    for(uint64_t task=AscendC::GetBlockIdx();task<static_cast<uint64_t>(t.taskNum);task+=AscendC::GetBlockNum()){
        uint64_t tile=task%tiles,head=(task/tiles)%t.HV,seq=task/(tiles*t.HV);
        uint64_t batch=t.isVariable?0:seq,begin=0,end=t.T,first=seq*t.chunkNumForT;
        if(t.isVariable){
            int64_t start,stop;ReadDhuPair(metadata,cu,seq,start,stop);begin=start;end=stop;first=0;
            for(uint64_t s=0;s<seq;++s){ReadDhuPair(metadata,cu,s,start,stop);first+=(stop-start+63)/64;}
        }
        uint64_t token=(batch*t.HV+head)*t.T+begin;
        GeneralDhuSequence c{};c.tokens=end-begin;c.keyDim=t.K;c.valueDim=t.V;
        c.valueBegin=tile*32;c.valueCount=t.V-c.valueBegin<32?t.V-c.valueBegin:32;
        c.keyOffset=token*t.K;c.valueOffset=token*t.V;c.doOffset=token*t.V;c.doStride=t.V;
        c.stateOffset=(seq*t.HV+head)*t.K*t.V;c.snapshotOffset=(first*t.HV+head)*t.K*t.V;c.snapshotStride=t.HV*t.K*t.V;
        c.scale=t.scale;c.useExp2=t.useExp2;c.stateVFirst=t.stateVFirst;op.Process(b,c);
    }
}
}
