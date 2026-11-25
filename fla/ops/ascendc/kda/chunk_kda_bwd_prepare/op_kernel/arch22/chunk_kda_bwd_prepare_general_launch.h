// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_kda_bwd_prepare_general.h"
#include "../../../chunk_kda_fwd_finalize/op_kernel/arch22/chunk_kda_fwd_finalize_general_launch.h"
#include "../chunk_kda_bwd_prepare_struct.h"
namespace KDA::Arch22 {
template<class GradT>
__aicore__ inline void RunGeneralBwdPrepare(const GeneralBwdPrepareBuffers &b,GM_ADDR cu,GM_ADDR indices,
    const KDA::ChunkKdaBwdPrepareTilingData &t){
    AscendC::TPipe pipe;GeneralBwdPrepare<GradT> op;op.Init(pipe);
    AscendC::TQue<AscendC::QuePosition::VECIN,1> metadata;pipe.InitBuffer(metadata,1,32);
    for(uint64_t task=AscendC::GetBlockIdx();task<static_cast<uint64_t>(t.totalChunkNum*t.NV);task+=AscendC::GetBlockNum()){
        const uint64_t chunk=task/t.NV,head=task%t.NV;
        uint64_t batch=t.isVariable?0:chunk/t.denseChunkNum;
        uint64_t begin=(t.isVariable?chunk:chunk%t.denseChunkNum)*64,end=t.T;
        if(t.isVariable){
            int64_t seq,local,start,stop;ReadFinalizePair(metadata,indices,chunk*2,seq,local);
            ReadFinalizePair(metadata,cu,seq,start,stop);begin=start+local*64;end=stop;
        }
        uint64_t token=(batch*t.NV+head)*t.T+begin;
        GeneralBwdPrepareChunk c{};c.rows=end-begin<64?end-begin:64;c.keyDim=t.K;c.valueDim=t.V;
        c.scoreOffset=token*64;c.valueOffset=token*t.V;c.doOffset=token*t.V;c.doStride=t.V;
        c.stateOffset=(chunk*t.NV+head)*t.K*t.V;c.keyOffset=token*t.K;c.scale=t.scale;c.stateVFirst=t.stateVFirst;
        op.Process(b,c);
    }
}
}
