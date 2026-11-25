// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_kda_fwd_finalize_general.h"
#include "../chunk_kda_fwd_finalize_struct.h"
namespace KDA::Arch22 {
__aicore__ inline void ReadFinalizePair(AscendC::TQue<AscendC::QuePosition::VECIN,1> &queue,
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
template<class OutputT>
__aicore__ inline void RunGeneralFinalize(const GeneralFinalizeBuffers &buffers, GM_ADDR cu, GM_ADDR indices,
    const KdaFinalize::ChunkKdaFwdFinalizeTilingData &t) {
    AscendC::TPipe pipe;
    GeneralFwdFinalize<OutputT> op;op.Init(pipe);
    AscendC::TQue<AscendC::QuePosition::VECIN,1> metadata;pipe.InitBuffer(metadata,1,32);
    const uint32_t tiles=(t.valueDim+31)/32;
    const uint64_t count=static_cast<uint64_t>(t.totalChunks)*(t.isVarLen?1:t.batch)*t.valueHeadNum*tiles;
    for(uint64_t work=AscendC::GetBlockIdx();work<count;work+=t.usedCoreNum) {
        const uint32_t tile=work%tiles;
        const uint32_t head=(work/tiles)%t.valueHeadNum;
        const uint32_t flatChunk=work/(tiles*t.valueHeadNum);
        const uint32_t batch=t.isVarLen?0:flatChunk/t.totalChunks;
        const uint32_t chunk=flatChunk%t.totalChunks;
        uint32_t begin=chunk*64,end=t.seqLen;
        if(t.isVarLen) {
            int64_t seq=0,local=chunk,start=0,stop=0;
            if(indices!=nullptr) {
                ReadFinalizePair(metadata,indices,static_cast<uint64_t>(chunk)*2,seq,local);
                ReadFinalizePair(metadata,cu,seq,start,stop);
            } else {
                for(;seq<t.seqNum;++seq) {
                    ReadFinalizePair(metadata,cu,seq,start,stop);
                    const int64_t chunks=(stop-start+63)/64;
                    if(local<chunks)break;
                    local-=chunks;
                }
            }
            begin=static_cast<uint32_t>(start+local*64);end=static_cast<uint32_t>(stop);
        }
        const uint64_t token=(static_cast<uint64_t>(batch)*t.valueHeadNum+head)*t.seqLen+begin;
        const uint64_t outToken=t.outputSequenceMajor?
            (static_cast<uint64_t>(batch)*t.seqLen+begin)*t.valueHeadNum+head:token;
        GeneralFinalizeChunk c{};
        c.rows=end-begin<64?end-begin:64;c.keyDim=t.keyDim;c.valueDim=t.valueDim;
        c.valueBegin=tile*32;c.valueCount=t.valueDim-c.valueBegin<32?t.valueDim-c.valueBegin:32;
        c.keyOffset=token*t.keyDim;c.valueOffset=token*t.valueDim;c.scoreOffset=token*64;
        c.stateOffset=((static_cast<uint64_t>(batch)*t.totalChunks+chunk)*t.valueHeadNum+head)*t.keyDim*t.valueDim;
        c.outputOffset=outToken*t.valueDim;c.outputStride=static_cast<uint64_t>(t.valueDim)*(t.outputSequenceMajor?t.valueHeadNum:1);
        c.stateVFirst=t.stateVFirst;op.Process(buffers,c);
    }
}
} // namespace KDA::Arch22
