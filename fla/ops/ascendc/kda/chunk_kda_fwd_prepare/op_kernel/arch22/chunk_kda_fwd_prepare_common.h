// Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "chunk_kda_fwd_prepare_resource.h"

namespace KdaPrepare::Arch22 {
using namespace AscendC;
__aicore__ inline uint32_t MinusOffset(uint32_t band) {
    return 0x8000+band*(band+1)*0x800;
}
struct Work {
    uint32_t batch=0, token=0, rows=0, head[2]={0,0}, qh[2]={0,0};
    bool active[2]={false,false}, owner[2]={false,false};
};
__aicore__ inline uint64_t WorkCount(const PrepareRuntimeTiling &t) {
    return (t.isVarLen?uint64_t(t.totalChunks):uint64_t(t.batch)*t.totalChunks)*((t.valueHeadNum+1)/2);
}
__aicore__ inline Work Describe(const PrepareKernelArgs &a,uint64_t job) {
    const auto &t=a.tiling; Work w;
    uint32_t pairs=(t.valueHeadNum+1)/2, chunk=job/pairs, pair=job%pairs;
    if(t.isVarLen) {
        GlobalTensor<int64_t> ids,cu;
        ids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(a.chunkIndices));
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(a.cuSeqlens));
        uint32_t seq=ids.GetValue(2*chunk), local=ids.GetValue(2*chunk+1);
        uint32_t end=cu.GetValue(seq+1); w.token=cu.GetValue(seq)+64*local;
        w.rows=w.token<end?(end-w.token<64?end-w.token:64):0;
    } else {
        w.batch=chunk/t.totalChunks; w.token=(chunk%t.totalChunks)*64;
        w.rows=t.seqLen-w.token<64?t.seqLen-w.token:64;
    }
    const uint32_t ratio=t.valueHeadNum/t.qkHeadNum;
    for(uint32_t l=0;l<2;++l) {
        w.head[l]=2*pair+l; w.active[l]=w.head[l]<t.valueHeadNum;
        w.qh[l]=w.active[l]?w.head[l]/ratio:0;
        w.owner[l]=w.active[l] && w.head[l]%ratio==0;
    }
    return w;
}
__aicore__ inline uint64_t Out(const PrepareRuntimeTiling &t,const Work &w,
                              uint32_t h,uint32_t heads,uint32_t width) {
    return ((uint64_t(w.batch)*heads+h)*t.seqLen+w.token)*width;
}
__aicore__ inline uint64_t QkvInputOffset(const PrepareRuntimeTiling &t,const Work &w,
                             uint32_t h,uint32_t heads,uint32_t width) {
    return t.inputSequenceMajor?((uint64_t(w.batch)*t.seqLen+w.token)*heads+h)*width:
                                Out(t,w,h,heads,width);
}
// Raw g/beta follow the same public source layout as q/k/v.
__aicore__ inline uint64_t GateInputOffset(const PrepareRuntimeTiling &t,const Work &w,
                                         uint32_t head,uint32_t width) {
    return QkvInputOffset(t,w,head,t.valueHeadNum,width);
}
__aicore__ inline uint64_t BetaInputOffset(const PrepareRuntimeTiling &t,const Work &w,
                                         uint32_t head) {
    return GateInputOffset(t,w,head,1);
}
__aicore__ inline uint64_t GmLane(uint32_t core,uint32_t slot,uint32_t lane) {
    return uint64_t(core)*GroupBytes+slot*SlotBytes+lane*LaneBytes;
}
template<class T> __aicore__ inline GlobalTensor<T> Gm(GM_ADDR p,uint64_t elements=0) {
    GlobalTensor<T> g;g.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(p)+elements);return g;
}
__aicore__ inline void VBarrier(){PipeBarrier<PIPE_V>();}
__aicore__ inline void VToMte3(){SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);}
__aicore__ inline void Mte2ToV(){SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);}
__aicore__ inline void VToS(){SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);}
__aicore__ inline void SToV(){SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);}
template<class T> __aicore__ inline void PublishCache(GlobalTensor<T> g) {
    // Work payload/raw/Lower/Eye/RHS are transferred exclusively by DMA/FIX.
    // No scalar load/store touches these regions. Producer PIPE_MTE3/PIPE_FIX
    // ready flags and consumer PIPE_MTE2 waits are the release/acquire protocol;
    // scalar D-cache writeback/invalidation cannot make these DMA regions safer.
    // Immutable chunk metadata scalar reads are outside this function's regions.
    (void)g;
}
}
