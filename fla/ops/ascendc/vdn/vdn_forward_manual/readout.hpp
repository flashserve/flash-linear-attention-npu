#pragma once
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
namespace JyReadout {
using namespace Catlass;
template<typename T>
CATLASS_DEVICE void Project(GM_ADDR q,GM_ADDR state,GM_ADDR out,int matrices,int tokens) {
    constexpr uint32_t D=128;
    using A=Gemm::GemmType<T,layout::RowMajor>;
    using B=Gemm::GemmType<T,layout::ColumnMajor>;
    using C=Gemm::GemmType<T,layout::RowMajor>;
    using Copy=Gemm::Tile::TileCopy<Arch::AtlasA2,A,B,C>;
    using Am=typename Copy::CopyL1ToL0A::LayoutSrc;
    using Bm=typename Copy::CopyL1ToL0B::LayoutSrc;
    using Al=typename Copy::CopyL1ToL0A::LayoutDst;
    using Bl=typename Copy::CopyL1ToL0B::LayoutDst;
    Arch::Resource<Arch::AtlasA2> resource;
    auto a=resource.l1Buf.template GetBufferByByte<T>(0);
    auto b=resource.l1Buf.template GetBufferByByte<T>(D*D*sizeof(T));
    auto left=resource.l0ABuf.template GetBufferByByte<T>(0);
    auto right=resource.l0BBuf.template GetBufferByByte<T>(0);
    auto acc=resource.l0CBuf.template GetBufferByByte<float>(0);
    typename Copy::CopyGmToL1A inA; typename Copy::CopyGmToL1B inB;
    typename Copy::CopyL1ToL0A loadA;typename Copy::CopyL1ToL0B loadB;typename Copy::CopyL0CToGm save;
    Gemm::Tile::TileMmad<Arch::AtlasA2,A,B,void> mm;
    AscendC::GlobalTensor<T> Q,S,Y;
    Q.SetGlobalBuffer((__gm__ T*)q); S.SetGlobalBuffer((__gm__ T*)state);Y.SetGlobalBuffer((__gm__ T*)out);
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const auto lb=Bm::template MakeLayout<T>(D,D);
    for(int job=AscendC::GetBlockIdx();job<matrices;job+=AscendC::GetBlockNum()) {
        inB(b,S[int64_t(job)*D*D],lb,layout::ColumnMajor{D,D});
        AscendC::PipeBarrier<PIPE_ALL>();
        loadB(right,b,Bl::template MakeLayout<T>(D,D),lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        for(int start=0;start<tokens;start+=D) {
            const uint32_t m=tokens-start<D?tokens-start:D;
            const auto la=Am::template MakeLayout<T>(m,D);
            const auto lc=layout::zN::MakeLayoutInL0C(MakeCoord(m,D));
            const auto offset=(int64_t(job)*tokens+start)*D;
            inA(a,Q[offset],la,layout::RowMajor{m,D}); AscendC::PipeBarrier<PIPE_ALL>();
            loadA(left,a,Al::template MakeLayout<T>(m,D),la); AscendC::PipeBarrier<PIPE_ALL>();
            mm(acc,left,right,m,D,D,true,0); AscendC::PipeBarrier<PIPE_ALL>();
            save(Y[offset],acc,layout::RowMajor{m,D},lc); AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}
}
