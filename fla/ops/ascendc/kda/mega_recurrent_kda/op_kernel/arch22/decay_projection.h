#pragma once
#include "mega_recurrent_kda_v2_cube.h"

namespace MegaRecurrentKdaV2 {

// The input projection is reduced in FP32. Near a BF16 rounding midpoint,
// replay the dot product in input order before rounding the intermediate.
template <class Output>
CATLASS_DEVICE void projectPanel(Catlass::Arch::Resource<KdaGemm::ArchTag>& resource,
    GM_ADDR x, GM_ADDR weight, GM_ADDR output, uint32_t rows, uint32_t k,
    uint32_t cols, uint32_t stride)
{
    using Row = KdaGemm::LayoutTagA;
    using Col = KdaGemm::LayoutTagB;
    using Bf = bfloat16_t;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaGemm::ArchTag, Bf, Row, Bf, Col, Output, Row>;
    using Op = Catlass::Gemm::Block::BlockMmadTla<KdaGemm::DispatchPolicy, KdaGemm::L1TileShape,
        KdaGemm::L0TileShape, Bf, Bf, Output, void, Copy>;
    AscendC::GlobalTensor<Bf> a,b;
    AscendC::GlobalTensor<Output> c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(x));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ Bf*>(weight));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ Output*>(output));
    auto aa=tla::MakeTensor(a,tla::MakeLayout<Bf,Row>(rows,k),Catlass::Arch::PositionGM{});
    auto bb=tla::MakeTensor(b,tla::MakeLayout<Bf,Col>(k,cols),Catlass::Arch::PositionGM{});
    auto cc=tla::MakeTensor(c,tla::MakeLayout<Output,Row>(rows,stride),Catlass::Arch::PositionGM{});
    auto dst=GetTile(cc,tla::MakeCoord(0,0),tla::MakeShape(rows,cols));
    Op op(resource);
    op(aa,bb,dst,Catlass::GemmCoord(rows,cols,k));
}

CATLASS_DEVICE void roundDecay(const V2Params& p)
{
    constexpr uint32_t hidden=kHidden, cols=kHeadDim;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer,hidden*12+1024);
    auto ub=buffer.Get<uint8_t>();
    auto xb=ub.ReinterpretCast<bfloat16_t>();
    auto wb=ub[hidden*2].ReinterpretCast<bfloat16_t>();
    auto xf=ub[hidden*4].ReinterpretCast<float>();
    auto wf=ub[hidden*8].ReinterpretCast<float>();
    auto accum=ub[hidden*12].ReinterpretCast<float>();
    auto packed=ub[hidden*12+cols*4].ReinterpretCast<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> x,w,dst;
    AscendC::GlobalTensor<float> src;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.hiddenStates));
    w.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.decayAWeight));
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.wQkv));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.wGateRaw));
    for(uint32_t row=AscendC::GetBlockIdx();row<p.tokens;row+=p.aivNum){
        AscendC::DataCopy(accum,src[uint64_t(row)*cols],cols);
        AscendC::PipeBarrier<PIPE_ALL>();
        bool loaded=false;
        for(uint32_t col=0;col<cols;++col){
            const uint32_t bits=accum.ReinterpretCast<uint32_t>().GetValue(col);
            // A BF16 midpoint ends in 0x8000 in the FP32 significand.
            const uint32_t low=bits&0xffffu;
            const uint32_t distance=low>0x8000u?low-0x8000u:0x8000u-low;
            // Re-evaluate a band of 32 FP32 ULPs on either side of a midpoint.
            if (distance > 32u) continue;
            if(!loaded){
                AscendC::DataCopy(xb,x[uint64_t(row)*hidden],hidden);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast(xf,xb,AscendC::RoundMode::CAST_NONE,hidden);
                loaded=true;
            }
            AscendC::DataCopy(wb,w[uint64_t(col)*hidden],hidden);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(wf,wb,AscendC::RoundMode::CAST_NONE,hidden);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(wf,wf,xf,hidden);
            AscendC::PipeBarrier<PIPE_ALL>();
            float ordered=0.f;
            const auto* products = reinterpret_cast<__ubuf__ float*>(wf.GetPhyAddr());
            #pragma unroll 16
            for (uint32_t k = 0; k < hidden; ++k) ordered += products[k];
            accum.SetValue(col,ordered);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(packed,accum,AscendC::RoundMode::CAST_RINT,cols);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(dst[uint64_t(row)*cols],packed,cols);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

CATLASS_DEVICE void projectDecay(const V2Params& p)
{
    constexpr uint32_t rowsPerTile=32,cols=kHeadDim;
    const uint32_t tiles=CeilDiv(p.tokens,rowsPerTile);
    if ASCEND_IS_AIC {
        Catlass::Arch::Resource<KdaGemm::ArchTag> resource;
        for(uint32_t tile=AscendC::GetBlockIdx();tile<tiles;tile+=p.aicNum){
            uint32_t first=tile*rowsPerTile,rows=p.tokens-first;
            if(rows>rowsPerTile) rows=rowsPerTile;
            projectPanel<float>(resource,p.hiddenStates+uint64_t(first)*p.hidden*2,p.decayAWeight,
                p.wQkv+uint64_t(first)*cols*4,rows,p.hidden,cols,cols);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV { roundDecay(p); }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIC {
        Catlass::Arch::Resource<KdaGemm::ArchTag> resource;
        for(uint32_t task=AscendC::GetBlockIdx();task<tiles*p.n;task+=p.aicNum){
            uint32_t first=(task/p.n)*rowsPerTile,col=(task%p.n)*cols,rows=p.tokens-first;
            if(rows>rowsPerTile) rows=rowsPerTile;
            projectPanel<bfloat16_t>(resource,p.wGateRaw+uint64_t(first)*cols*2,
                p.decayBWeight+uint64_t(col)*cols*2,p.wDecayRaw+(uint64_t(first)*p.p+col)*2,
                rows,cols,cols,p.p);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<false>();
}
} // namespace MegaRecurrentKdaV2
