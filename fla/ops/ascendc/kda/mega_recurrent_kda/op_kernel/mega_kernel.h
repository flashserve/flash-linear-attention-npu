#pragma once
#include "recurrence.h"
#include "catlass/arch/cross_core_sync.hpp"

namespace kda {
// One BF16 tile in the paired AIV's UB. Beta has one value per row and a padded stride of 16.
constexpr uint32_t kSlotBytes = 20992, kSliceRows = 16, kHeadDim = 128, kGlobalRows = 64;
enum class Plane : uint32_t { Q, K, V, Decay, Gate, Beta };

struct MegaParams {
    GM_ADDR x;
    GM_ADDR qkv;
    GM_ADDR wa;
    GM_ADDR wb;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR decayRing;
    uint32_t rows, heads;
    RecurrenceArgs recurrence{};
};

template<bool SpansTile>
class MegaKernel {
public:
    using Params=MegaParams;
    CATLASS_DEVICE void operator()(const Params &p)
    {
        if constexpr(!SpansTile) {
            doubleBuffered(p);
            return;
        }
        uint32_t group = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            if(AscendC::GetSubBlockIdx()!=0) return;
            group /= AscendC::GetTaskRation();
            publishFree();
        }
        uint32_t groups = AscendC::GetBlockNum();
        uint32_t tiles = (p.rows - 1) / kGlobalRows + 1;
        uint32_t phase = 0;
        for(uint32_t wave=0;wave<tiles;++wave) {
            uint32_t first=wave*kGlobalRows,rows=p.rows-first<kGlobalRows?p.rows-first:kGlobalRows;
            sharedDecay(p,first,rows,group,groups,phase);
            barrier();
            constexpr uint32_t cuts=4;
            for(uint64_t work=group;work<uint64_t(cuts)*p.heads;work+=groups) {
                uint32_t part=work/p.heads,head=work%p.heads;
                uint32_t offset=part?sequenceCut(p,first,rows,part*16):0;
                uint32_t end=part+1<cuts?sequenceCut(p,first,rows,(part+1)*16):rows;
                for (uint32_t r=offset; r<end; r+=kSliceRows) {
                    uint32_t valid = end-r < kSliceRows ? end-r : kSliceRows;
                    if ASCEND_IS_AIC {
                        waitFree(phase);
                        uint32_t col = head*kHeadDim;
                        uint32_t width = p.heads*kHeadDim;
                        for (uint32_t i=0; i<2; ++i) {
                            matmul(p.x,p.qkv,first+r,col+i*width,valid,kHeadDim,7168,
                                   localTile(phase,static_cast<Plane>(i)));
                        }
                    }
                    uint32_t col=head*kHeadDim;
                    if ASCEND_IS_AIC {
                        matmul(p.x,p.qkv,first+r,col+2*p.heads*kHeadDim,valid,kHeadDim,7168,localTile(phase,Plane::V));
                        matmul(p.decayRing,p.wb,r,col,valid,kHeadDim,128,localTile(phase,Plane::Decay));
                        matmul(p.x,p.gate,first+r,col,valid,kHeadDim,7168,localTile(phase,Plane::Gate));
                        matmul(p.x,p.beta,first+r,head,valid,1,7168,localTile(phase,Plane::Beta));
                        notifyReady(phase);
                    }
                    if ASCEND_IS_AIV {
                        waitReady(phase);
                        GM_ADDR gated=p.decayRing+32768;
                        GM_ADDR carry=gated+uint64_t(p.heads)*32768;
                        Recurrence{}.consume(resource_,p.recurrence,gated,carry,p.heads,head,
                            first+r,r,valid,localTile(phase,Plane::Q));
                        notifyFree(phase);
                    }
                    phase ^= 1;
                }
            }
            barrier();
            outputTiles(p,first,rows,group,groups,phase);
            barrier();
        }
        if ASCEND_IS_AIC { waitFree(0); waitFree(1); }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Catlass::Arch::Resource<Arch> resource_;
    CATLASS_DEVICE AscendC::LocalTensor<Bf> localTile(uint32_t phase, Plane plane)
    {
        uint32_t offset = phase*kSlotBytes + static_cast<uint32_t>(plane)*kSliceRows*kHeadDim*sizeof(Bf);
        return resource_.ubBuf.template GetBufferByByte<Bf>(offset);
    }
    template<uint32_t TileM=16>
    CATLASS_DEVICE void matmul(GM_ADDR aPtr, GM_ADDR wPtr, uint32_t row,
            uint32_t col, uint32_t m, uint32_t n, uint32_t k, AscendC::LocalTensor<Bf> buffer)
    {
        AscendC::GlobalTensor<Bf> a,w;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(aPtr) + uint64_t(row)*k);
        w.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(wPtr) + uint64_t(col)*k);
        auto x = tla::MakeTensor(a,tla::MakeLayout<Bf,Row>(m,k),Catlass::Arch::PositionGM{});
        auto weight = tla::MakeTensor(w,tla::MakeLayout<Bf,Col>(k,n),Catlass::Arch::PositionGM{});
        uint32_t stride = (n+15)/16*16;
        auto ub = tla::MakeTensor(buffer,tla::MakeLayout<Bf,Row>(m,stride),Catlass::Arch::PositionUB{});
        using WideProjection = Catlass::Gemm::Block::BlockMmadTla<Policy,
            tla::Shape<tla::Int<TileM>,tla::Int<128>,tla::Int<512>>,
            tla::Shape<tla::Int<TileM>,tla::Int<128>,tla::Int<128>>,
            Bf,Bf,Bf,void,Copy,TileMmad>;
        WideProjection op(resource_);
        op(x,weight,ub,Catlass::GemmCoord(m,n,k));
    }
    // Lead with 64 rows when the final 128-row wave would contain at most 64.
    // This releases the first group of sequences to Vector one projection earlier.
    struct Waves {
        uint32_t firstRows, firstCapacity, storageRows, count, span, decayRows;
        CATLASS_DEVICE uint32_t offset(uint32_t i) const { return i==0?0:firstRows+(i-1)*span; }
    };
    CATLASS_DEVICE Waves waves(uint32_t rows) {
        uint32_t span=rows>128 && rows<=256?256:128;
        uint32_t tail=rows%span;
        uint32_t first=rows<=span?rows:((tail && tail<=64)?64:span);
        uint32_t count=1+(rows-first+span-1)/span;
        uint32_t firstCap=count>2?span:(first+15)/16*16;
        uint32_t left=rows-first;
        uint32_t secondCap=left>=span?span:(left+15)/16*16;
        return {first,firstCap,firstCap+secondCap,count,span,span==256?firstCap:128};
    }
    CATLASS_DEVICE void stagedConsumer(const Params &p,uint32_t first,uint32_t rows,uint32_t worker,uint32_t workers,
                                      uint32_t firstHead=0,uint32_t count=0)
    {
        uint32_t width=p.heads*kHeadDim;
        uint32_t span=p.recurrence.staged==3?128:64;
        uint32_t slot=p.recurrence.staged>=2 ? (first/span)%2 : 0;
        uint32_t betaStride=p.recurrence.staged>=2 ? (p.heads+15)/16*16 : p.heads;
        GM_ADDR gatedBase=p.decayRing+(p.recurrence.staged>=2?16384:32768);
        uint32_t planeRows=64,totalGatedRows=128;
        if(p.recurrence.staged==3) {
            auto shape=waves(p.rows);
            slot=first?(1+(first-shape.firstRows)/shape.span)%2:0;
            span=shape.firstCapacity;
            totalGatedRows=shape.storageRows;
            planeRows=slot?shape.storageRows-span:span;
            gatedBase=p.decayRing+shape.decayRows*256;
        }
        GM_ADDR gated=gatedBase+uint64_t(slot)*span*width*2;
        GM_ADDR planes=gatedBase+uint64_t(totalGatedRows)*width*2+uint64_t(slot)*span*(5*uint64_t(width)+betaStride)*2;
        Recurrence rec;
        uint32_t firstSeq=rec.owner(p.recurrence,first);
        uint32_t lastSeq=rec.owner(p.recurrence,first+rows-1);
        if(!count) count=p.heads;
        for(uint64_t task=worker;task<uint64_t(lastSeq-firstSeq+1)*count;task+=workers) {
            uint32_t head=firstHead+task%count,b=firstSeq+task/count;
            uint32_t start=rec.prefix(p.recurrence,b),end=rec.prefix(p.recurrence,b+1);
            for(uint32_t row=start;row<end;row+=kSliceRows) {
                uint32_t m=end-row<kSliceRows?end-row:kSliceRows;
                for(uint32_t i=0;i<6;++i) {
                    AscendC::GlobalTensor<Bf> src;
                    src.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(planes)+uint64_t(i)*planeRows*width);
                    auto tile=localTile(0,static_cast<Plane>(i));
                    uint64_t row0=row-first;
                    if(i==5) {
                        AscendC::DataCopyPad(tile,src[row0*betaStride+head],
                            AscendC::DataCopyExtParams{static_cast<uint16_t>(m),2,(betaStride-1)*2,0,0},
                            AscendC::DataCopyPadExtParams<Bf>{});
                    } else {
                        AscendC::DataCopyPad(tile,src[row0*width+head*128],
                            AscendC::DataCopyExtParams{static_cast<uint16_t>(m),256,(width-128)*2,0,0},
                            AscendC::DataCopyPadExtParams<Bf>{});
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                rec.consume(resource_,p.recurrence,gated,nullptr,p.heads,head,row,row-first,m,localTile(0,Plane::Q));
            }
        }
    }
    CATLASS_DEVICE void sharedDecay(const Params &p,uint32_t first,uint32_t rows,uint32_t group,uint32_t groups,uint32_t &phase)
    {
        for (uint32_t start = group*kSliceRows; start < rows; start += groups*kSliceRows) {
            uint32_t valid = rows-start < kSliceRows ? rows-start : kSliceRows;
            auto local = localTile(phase, Plane::Q);
            if ASCEND_IS_AIC {
                waitFree(phase);
            }
            if ASCEND_IS_AIC {matmul(p.x,p.wa,first+start,0,valid,128,7168,local);}
            if ASCEND_IS_AIC { notifyReady(phase); }
            if ASCEND_IS_AIV {
                waitReady(phase);
                store(p.decayRing, start, 0, kHeadDim, valid, kHeadDim, local);
                notifyFree(phase);
            }
            phase ^= 1;
        }
    }
    CATLASS_DEVICE void outputTiles(const Params &p,uint32_t first,uint32_t rows,uint32_t group,uint32_t groups,uint32_t &phase)
    {
        constexpr uint32_t cols=7168/kHeadDim;
        constexpr uint32_t outputRows=64;
        uint32_t slices=(rows+outputRows-1)/outputRows;
        for(uint32_t task=group;task<slices*cols;task+=groups) {
            uint32_t r=task/cols*outputRows, col=task%cols*kHeadDim;
            uint32_t valid=rows-r<outputRows?rows-r:outputRows;
            auto tile=localTile(phase,Plane::Q);
            if ASCEND_IS_AIC {
                waitFree(phase);
                matmul<64>(p.decayRing+32768,p.recurrence.outWeight,r,col,valid,kHeadDim,p.heads*kHeadDim,tile);
                notifyReady(phase);
            }
            if ASCEND_IS_AIV {
                waitReady(phase);
                store(p.recurrence.output,first+r,col,7168,valid,kHeadDim,tile);
                notifyFree(phase);
            }
            phase^=1;
        }
    }
    CATLASS_DEVICE void headPipeline(const Params &p,const Waves &shape,uint32_t id,uint32_t groups) {
        const uint32_t width=p.heads*128,betaStride=(p.heads+15)/16*16,planeRows=shape.firstCapacity;
        GM_ADDR gated=p.decayRing+shape.decayRows*256;
        GM_ADDR planes=gated+uint64_t(shape.storageRows)*width*2;
        constexpr uint32_t waveLimit=2;
        const uint32_t waves=p.heads<waveLimit?p.heads:waveLimit;
        for(uint32_t tick=0;tick<waves+2;++tick) {
            if ASCEND_IS_AIC {
                if(tick<waves) {
                    uint32_t firstHead=uint64_t(p.heads)*tick/waves;
                    uint32_t count=uint64_t(p.heads)*(tick+1)/waves-firstHead;
                    uint64_t part=uint64_t(count)*128,betaBase=4*part+128;
                    uint64_t total=tick?4*part:betaBase+p.heads,blocks=(total+15)/16;
                    uint64_t cursor=(blocks*id/groups)*16,stop=(blocks*(id+1)/groups)*16;
                    if(stop>total) stop=total;
                    // Keep each Cube core on one projection to avoid a second matrix pipeline setup.
                    const uint32_t kinds=tick?4:6;
                    if(groups>=kinds) {
                        uint32_t groupStart=0;
                        uint64_t columnStart=0;
                        for(uint32_t kind=0;kind<kinds;++kind) {
                            const uint64_t columns=kind<4?part:(kind==4?128:p.heads);
                            const uint64_t end=columnStart+columns;
                            uint32_t groupEnd=static_cast<uint32_t>((end*groups+total/2)/total);
                            const uint32_t limit=groups-(kinds-kind-1);
                            if(groupEnd>limit) groupEnd=limit;
                            if(groupEnd<=groupStart) groupEnd=groupStart+1;
                            if(id<groupEnd) {
                                const uint32_t rank=id-groupStart, team=groupEnd-groupStart;
                                const uint64_t units=(columns+15)/16;
                                cursor=columnStart+16*(units*rank/team);
                                stop=columnStart+16*(units*(rank+1)/team);
                                if(stop>end) stop=end;
                                break;
                            }
                            groupStart=groupEnd;
                            columnStart=end;
                        }
                    }
                    while(cursor<stop) {
                        uint32_t kind=cursor<4*part?cursor/part:(cursor<betaBase?4:5);
                        uint64_t start=kind<4?uint64_t(kind)*part:(kind==4?4*part:betaBase);
                        uint32_t stride=kind<4?width:(kind==4?128:betaStride);
                        uint32_t local=cursor-start,extent=kind<4?part:(kind==4?128:p.heads);
                        uint32_t col=local+(kind<4?firstHead*128:0);
                        uint64_t n=stop-cursor;
                        if(n>extent-local) n=extent-local;
                        if(n>128) n=128;
                        GM_ADDR weight=kind<3?p.qkv:(kind==3?p.gate:(kind==4?p.wa:p.beta));
                        uint32_t weightCol=kind<3?kind*width+col:col;
                        GM_ADDR dest=kind==4?p.decayRing:planes+uint64_t(kind==3?4:kind)*planeRows*width*2;
                        if(p.rows<=160 && n<=96)
                            matmulToGm<96,512,64,160>(p.x,weight,dest,0,weightCol,0,col,stride,p.rows,7168,n);
                        else matmulToGm<128,256,64>(p.x,weight,dest,0,weightCol,0,col,stride,p.rows,7168,n);
                        cursor+=n;
                    }
                    if(!tick) {
                        AscendC::CrossCoreSetFlag<0,PIPE_FIX>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                        AscendC::CrossCoreWaitFlag<0,PIPE_MTE2>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                        for(uint32_t h=id;h<p.heads;h+=groups)
                            matmulToGm(p.decayRing,p.wb,planes+uint64_t(3)*planeRows*width*2,0,h*128,0,h*128,width,p.rows,128);
                    }
                } else if(tick==waves+1) {
                    for(uint32_t panel=id;panel<7168/256;panel+=groups)
                        matmulToGm<256,256,64>(gated,p.recurrence.outWeight,p.recurrence.output,0,panel*256,
                                               0,panel*256,7168,p.rows,width,256);
                }
            }
            if ASCEND_IS_AIV {
                if(tick>=1 && tick<=waves) {
                    uint32_t left=uint64_t(p.heads)*(tick-1)/waves;
                    uint32_t count=uint64_t(p.heads)*tick/waves-left;
                    stagedConsumer(p,0,p.rows,id,2*groups,left,count);
                }
            }
            AscendC::SyncAll<false>();
        }
    }
    CATLASS_DEVICE void doubleBuffered(const Params &p)
    {
        uint32_t id=AscendC::GetBlockIdx(),groups=AscendC::GetBlockNum();
        uint32_t span=p.recurrence.staged==3?128:64;
        auto shape=waves(p.rows);
        if(p.recurrence.staged==3 && shape.count==1 && p.heads>=2) {
            headPipeline(p,shape,id,groups);
            return;
        }
        uint32_t tiles=span==128?shape.count:(p.rows-1)/span+1,width=p.heads*128,betaStride=(p.heads+15)/16*16;
        uint32_t storageRows=span==128?shape.storageRows:128;
        uint32_t firstCap=span==128?shape.firstCapacity:64;
        GM_ADDR gated=p.decayRing+(span==128?shape.decayRows:span)*256;
        GM_ADDR scratch=gated+uint64_t(storageRows)*width*2;
        uint64_t slotBytes=uint64_t(firstCap)*2*(5*uint64_t(width)+betaStride);
        // The two projection slabs and the two gated slabs have independent consumers.
        for(uint32_t tick=0;tick<tiles+2;++tick) {
            if ASCEND_IS_AIC {
                if(tick<tiles) {
                    uint32_t first=span==128?shape.offset(tick):span*tick;
                    uint32_t limit=span==128 && !tick?shape.firstRows:span,m=p.rows-first<limit?p.rows-first:limit;
                    uint32_t planeRows=tick%2?storageRows-firstCap:firstCap;
                    GM_ADDR planes=scratch+(tick%2)*slotBytes;
                    // Shared decay-A and the independent input projections share one dispatch.
                    // Assign equal numbers of aligned output columns to resident Cube cores.
                    const uint64_t betaBase=4*uint64_t(width)+128;
                    const uint64_t total=betaBase+p.heads,blocks=(total+15)/16;
                    uint64_t cursor=(blocks*id/groups)*16,stop=(blocks*(id+1)/groups)*16;
                    if(stop>total) stop=total;
                    while(cursor<stop) {
                        uint32_t kind=cursor<4*uint64_t(width)?cursor/width:(cursor<betaBase?4:5);
                        uint64_t start=kind<4?uint64_t(kind)*width:(kind==4?4*uint64_t(width):betaBase);
                        uint32_t stride=kind<4?width:(kind==4?128:betaStride);
                        uint32_t physical=kind==5?p.heads:stride;
                        uint32_t col=cursor-start;
                        uint64_t n=stop-cursor;
                        if(n>physical-col) n=physical-col;
                        if(n>160) n=160;
                        GM_ADDR weight=kind<3?p.qkv:(kind==3?p.gate:(kind==4?p.wa:p.beta));
                        uint32_t weightCol=kind<3?kind*width+col:col;
                        GM_ADDR dest=kind==4?p.decayRing:planes+uint64_t(kind==3?4:kind)*planeRows*width*2;
                        matmulToGm<160,256,64>(p.x,weight,dest,first,weightCol,0,col,stride,m,7168,n);
                        cursor+=n;
                    }
                    // Only Cube workers consume decay-A. The following loads are issued by MTE2.
                    AscendC::CrossCoreSetFlag<0,PIPE_FIX>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                    AscendC::CrossCoreWaitFlag<0,PIPE_MTE2>(Catlass::Arch::AIC_INTER_BLOCK_BARRIER);
                    for(uint32_t h=id;h<p.heads;h+=groups)
                        matmulToGm(p.decayRing,p.wb,planes+uint64_t(3)*planeRows*width*2,0,h*128,0,h*128,width,m,128);
                }
                if(tick>=2) {
                    uint32_t tile=tick-2,first=span==128?shape.offset(tile):tile*span;
                    uint32_t limit=span==128 && !tile?shape.firstRows:span,m=p.rows-first<limit?p.rows-first:limit;
                    GM_ADDR ready=gated+uint64_t(tile%2)*firstCap*width*2;
                    for(uint32_t panel=id;panel<7168/256;panel+=groups)
                        matmulToGm<256,256,64>(ready,p.recurrence.outWeight,p.recurrence.output,0,panel*256,
                                   first,panel*256,7168,m,width,256);
                }
            }
            if ASCEND_IS_AIV {
                if(tick>=1 && tick<=tiles) {
                    uint32_t tile=tick-1,first=span==128?shape.offset(tile):tile*span;
                    uint32_t limit=span==128 && !tile?shape.firstRows:span,m=p.rows-first<limit?p.rows-first:limit;
                    stagedConsumer(p,first,m,id,2*groups);
                }
            }
            AscendC::SyncAll<false>();
        }
    }
    template<int TileN=128,int L1K=256,int L0K=64,int TileM=256>
    CATLASS_DEVICE void matmulToGm(GM_ADDR input,GM_ADDR weight,GM_ADDR out,uint32_t xRow,uint32_t wCol,
                                   uint32_t outRow,uint32_t outCol,uint32_t stride,uint32_t m,uint32_t k,uint32_t n=128)
    {
        AscendC::GlobalTensor<Bf> a,w,c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(input)+uint64_t(xRow)*k);
        w.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(weight)+uint64_t(wCol)*k);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(out)+uint64_t(outRow)*stride);
        auto x=tla::MakeTensor(a,tla::MakeLayout<Bf,Row>(m,k),Catlass::Arch::PositionGM{});
        auto wt=tla::MakeTensor(w,tla::MakeLayout<Bf,Col>(k,n),Catlass::Arch::PositionGM{});
        auto base=tla::MakeTensor(c,tla::MakeLayout<Bf,Row>(m,stride),Catlass::Arch::PositionGM{});
        auto dst=GetTile(base,tla::MakeCoord(0,outCol),tla::MakeShape(m,n));
        using GmCopy=Catlass::Gemm::Tile::PackedTileCopyTla<Arch,Bf,Row,Bf,Col,Bf,Row>;
        using Mmad=Catlass::Gemm::Tile::TileMmadTla<Arch,Bf,GmCopy::LayoutTagL1A>;
        using Op=Catlass::Gemm::Block::BlockMmadTla<Policy,
            tla::Shape<tla::Int<TileM>,tla::Int<TileN>,tla::Int<L1K>>,
            tla::Shape<tla::Int<TileM>,tla::Int<TileN>,tla::Int<L0K>>,Bf,Bf,Bf,void,GmCopy,Mmad>;
        Op op(resource_);
        op(x,wt,dst,Catlass::GemmCoord(m,n,k));
    }
    CATLASS_DEVICE uint32_t sequenceCut(Params const &p,uint32_t first,uint32_t tileEnd,uint32_t cut)
    {
        if(cut>=tileEnd) return tileEnd;
        const auto &a=p.recurrence;
        uint32_t target=first+cut;
        if(a.sequence) {
            uint64_t boundary=(uint64_t(target)+a.sequence-1)/a.sequence*a.sequence;
            return boundary-first<tileEnd?boundary-first:tileEnd;
        }
        uint32_t lo=0,hi=a.batch;
        while(lo<hi) {
            uint32_t mid=lo+(hi-lo)/2;
            uint64_t v=a.prefix64?reinterpret_cast<__gm__ int64_t*>(a.prefix)[mid]:
                reinterpret_cast<__gm__ int32_t*>(a.prefix)[mid];
            if(v<target) lo=mid+1; else hi=mid;
        }
        uint64_t v=a.prefix64?reinterpret_cast<__gm__ int64_t*>(a.prefix)[lo]:
            reinterpret_cast<__gm__ int32_t*>(a.prefix)[lo];
        return v-first<tileEnd?v-first:tileEnd;
    }
    CATLASS_DEVICE void store(GM_ADDR output,uint32_t row,uint32_t col,uint32_t stride,
                              uint32_t m,uint32_t n,AscendC::LocalTensor<Bf> buffer)
    {
        AscendC::GlobalTensor<Bf> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(output) + uint64_t(row)*stride);
        auto full = tla::MakeTensor(gm,tla::MakeLayout<Bf,Row>(m,stride),Catlass::Arch::PositionGM{});
        auto dest = GetTile(full,tla::MakeCoord(0,col),tla::MakeShape(m,n));
        auto ub = tla::MakeTensor(buffer,tla::MakeLayout<Bf,Row>(m,(n+15)/16*16),Catlass::Arch::PositionUB{});
        ObserveProjection observe;
        observe(dest,ub);
    }
    CATLASS_DEVICE void notifyFree(uint32_t i) { AscendC::CrossCoreSetFlag<4,PIPE_MTE3>(2+i); }
    CATLASS_DEVICE void waitFree(uint32_t i) { AscendC::CrossCoreWaitFlag<4,PIPE_FIX>(2+i); }
    CATLASS_DEVICE void notifyReady(uint32_t i) { AscendC::CrossCoreSetFlag<4,PIPE_FIX>(4+i); }
    CATLASS_DEVICE void waitReady(uint32_t i) { AscendC::CrossCoreWaitFlag<4,PIPE_MTE3>(4+i); }
    CATLASS_DEVICE void publishFree() { notifyFree(0); notifyFree(1); }
    CATLASS_DEVICE void barrier()
    {
        if ASCEND_IS_AIC {
            waitFree(0); waitFree(1);
            // Every resident Cube participates after its paired AIV completed its GM writes.
            Catlass::Arch::CrossCoreBarrier<0, PIPE_FIX>();
            AscendC::CrossCoreSetFlag<4,PIPE_FIX>(6);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag<4,PIPE_MTE3>(6);
            publishFree();
        }
    }
};
} // namespace kda
