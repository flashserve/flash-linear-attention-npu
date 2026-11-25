// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_fwd_h_general.h"
#include "../chunk_fwd_h_struct.h"
namespace GDN::Arch22 {
__aicore__ inline void ReadStateBounds(AscendC::TQue<AscendC::QuePosition::VECIN,1> &queue,
    GM_ADDR ptr,uint64_t offset,int64_t &a,int64_t &b) {
    AscendC::GlobalTensor<int64_t> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ptr));
    auto local=queue.AllocTensor<int64_t>();
    AscendC::DataCopyExtParams copy{1,16,0,0,0};
    AscendC::DataCopyPadExtParams<int64_t> pad{true,0,0,0};
    AscendC::DataCopyPad(local,gm[offset],copy,pad);queue.EnQue(local);local=queue.DeQue<int64_t>();
    AscendC::PipeBarrier<PIPE_ALL>();a=local.GetValue(0);b=local.GetValue(1);queue.FreeTensor(local);
}
template<class StateT,bool USE_EXP2,bool STATE_V_FIRST>
__aicore__ inline void RunGeneralState(const GeneralFwdHBuffers &buffers,GM_ADDR cu,
    const __gm__ ChunkFwdHTilingData *t) {
    AscendC::TPipe pipe;GeneralFwdH<StateT> op;op.Init(pipe);
    AscendC::TQue<AscendC::QuePosition::VECIN,1> metadata;pipe.InitBuffer(metadata,1,32);
    const uint32_t heads=t->vNumHead,key=t->kHeadDim,value=t->vHeadDim,tokens=t->seqlen;
    const uint32_t tiles=(value+31)/32;
    const uint64_t count=static_cast<uint64_t>(t->batch)*heads*tiles;
    for(uint64_t work=AscendC::GetBlockIdx();work<count;work+=AscendC::GetBlockNum()) {
        const uint32_t tile=work%tiles,head=(work/tiles)%heads,seq=work/(tiles*heads);
        uint64_t prefix=0;int64_t start=0,stop=tokens;
        const uint32_t batch=t->isVariedLen?0:seq;
        if(t->isVariedLen) {
            for(uint32_t previous=0;previous<=seq;++previous) {
                ReadStateBounds(metadata,cu,previous,start,stop);
                if(previous<seq)prefix+=(stop-start+63)/64;
            }
        } else prefix=static_cast<uint64_t>(batch)*((tokens+63)/64);
        const uint64_t token=(static_cast<uint64_t>(batch)*heads+head)*tokens+start;
        GeneralFwdHSequence c{};
        c.tokens=stop-start;c.keyDim=key;c.valueDim=value;c.valueBegin=tile*32;
        c.valueCount=value-c.valueBegin<32?value-c.valueBegin:32;
        c.keyOffset=token*key;c.valueOffset=token*value;
        c.stateOffset=(static_cast<uint64_t>(seq)*heads+head)*key*value;
        c.snapshotOffset=(prefix*heads+head)*key*value;
        c.snapshotStride=static_cast<uint64_t>(heads)*key*value;
        c.useExp2=USE_EXP2;c.stateVFirst=STATE_V_FIRST;
        op.Process(buffers,c);
    }
}
} // namespace GDN::Arch22
