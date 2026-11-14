// Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "chunk_kda_fwd_prepare_common.h"
namespace KdaPrepare::Arch22 {
constexpr FixpipeConfig NzOutput={CO2Layout::NZ,false};
template<uint32_t Norm,uint32_t Gate> class Cube {
    const PrepareKernelArgs &a_;TPipe *pipe_;uint32_t core_,slot_;bool identityReady_[2],leafFast_[2];Work w_;
    TBuf<TPosition::A1> l1_;TBuf<TPosition::A2> la_;
    TBuf<TPosition::B2> lb_;TBuf<TPosition::CO1> lc_;
    TEventID copyReady_,loadReady_,abFree_[2],matrixReady_,cFree_,l1Ready_,laneFree_[2],scoreReleased_;
    __aicore__ inline LocalTensor<bfloat16_t> L1(uint32_t lane,uint32_t bytes=0) {
        return l1_.Get<uint8_t>()[lane*L1LaneBytes+bytes].ReinterpretCast<bfloat16_t>();
    }
    __aicore__ inline void CopyNz(LocalTensor<bfloat16_t> dst,GlobalTensor<bfloat16_t> src,
        uint32_t rows,uint32_t columns,uint32_t pitch=64) {
        Nd2NzParams p{};p.ndNum=1;p.nValue=rows;p.dValue=columns;p.srcDValue=columns;
        p.dstNzNStride=1;p.dstNzC0Stride=pitch;DataCopy(dst,src,p);
    }
    __aicore__ inline void CopiesReady() {
        SetFlag<HardEvent::MTE2_MTE1>(copyReady_);WaitFlag<HardEvent::MTE2_MTE1>(copyReady_);
    }
    __aicore__ inline void LoadsReady() {
        SetFlag<HardEvent::MTE1_M>(loadReady_);WaitFlag<HardEvent::MTE1_M>(loadReady_);
    }
    __aicore__ inline void MatrixReady() {
        SetFlag<HardEvent::M_FIX>(matrixReady_);WaitFlag<HardEvent::M_FIX>(matrixReady_);
    }
    __aicore__ inline void LoadA(LocalTensor<bfloat16_t> dst,LocalTensor<bfloat16_t> src,
        uint32_t rows,uint32_t depth,uint32_t row=0,uint32_t column=0) {
        LoadData2DParams p{};p.repeatTimes=depth/16;p.srcStride=4;
        for(uint32_t r=0;r<rows/16;++r)
            LoadData(dst[r*(depth/16)*256],src[column*64+(row+16*r)*16],p);
    }
    __aicore__ inline void LoadB(LocalTensor<bfloat16_t> dst,LocalTensor<bfloat16_t> src,
        uint32_t columns,uint32_t depth,uint32_t sourceWidth,
        uint32_t row=0,uint32_t column=0) {
        LoadData3DParamsV2<bfloat16_t> p{};
        p.l1H=1;p.l1W=64;p.channelSize=sourceWidth;
        p.kExtension=columns;p.mExtension=depth;p.kStartPt=column;p.mStartPt=row;
        p.strideW=1;p.strideH=1;p.filterW=1;p.filterH=1;
        p.dilationFilterW=1;p.dilationFilterH=1;p.enTranspose=true;
        LoadData(dst,src,p);
    }
    __aicore__ inline void Scores(uint32_t lane) {
        WaitFlag<HardEvent::MTE1_MTE2>(laneFree_[lane]);
        uint64_t base=GmLane(core_,slot_,lane);auto payload=Gm<bfloat16_t>(a_.workspace+base);
        PublishCache(payload);uint32_t bands=(w_.rows+15)/16,padded=bands*16;
        CopyNz(L1(lane),payload,padded,128);
        CopyNz(L1(lane,0x4000),payload[8192],padded,128);
        for(uint32_t s=0;s<bands;++s)
            CopyNz(L1(lane,MinusOffset(s)),payload[MinusOffset(s)/2],16*(s+1),128,16*(s+1));
        CopiesReady();auto A=la_.Get<bfloat16_t>(),B=lb_.Get<bfloat16_t>();auto C=lc_.Get<float>();
        for(uint32_t s=0;s<bands;++s) {
            WaitFlag<HardEvent::M_MTE1>(abFree_[0]);
            LoadData2DParams q{};q.repeatTimes=8;q.srcStride=4;
            LoadData(A,L1(lane)[s*256],q);LoadData(A[2048],L1(lane,0x4000)[s*256],q);
            LoadData2DParams k{};k.repeatTimes=8*(s+1);k.srcStride=1;
            LoadData(B,L1(lane,MinusOffset(s)),k);LoadsReady();
            WaitFlag<HardEvent::FIX_M>(cFree_);
            MmadParams mm{};mm.m=32;mm.n=16*(s+1);mm.k=128;mm.cmatrixInitVal=true;
            Mmad(C,A,B,mm);SetFlag<HardEvent::M_MTE1>(abFree_[0]);MatrixReady();
            auto p=FixpipeParamsV220(16*(s+1),32,32,64,false);p.quantPre=QuantMode_t::NoQuant;
            Fixpipe<float,float,CFG_ROW_MAJOR>(Gm<float>(a_.workspace+base+Raw,s*2048),C,p);
            SetFlag<HardEvent::FIX_M>(cFree_);
        }
    }
    template<uint32_t Block,bool FastOnly=false> __aicore__ inline void GemmPair(
        uint32_t left,uint32_t right,uint32_t dst,
        uint32_t ar,uint32_t ac,uint32_t br,uint32_t bc,uint32_t cr,uint32_t cc,bool publish) {
        WaitFlag<HardEvent::FIX_M>(cFree_);
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane])continue;
            if constexpr(FastOnly)if(!leafFast_[lane])continue;
            auto A=la_.Get<bfloat16_t>()[lane*8192],B=lb_.Get<bfloat16_t>()[lane*16384];
            auto C=lc_.Get<float>()[lane*4096];
            WaitFlag<HardEvent::M_MTE1>(abFree_[lane]);
            LoadA(A,L1(lane,left),Block,Block,ar,ac);
            LoadB(B,L1(lane,right),Block,Block,64,br,bc);LoadsReady();
            MmadParams mm{};mm.m=Block;mm.n=Block;mm.k=Block;mm.cmatrixInitVal=true;
            Mmad(C,A,B,mm);SetFlag<HardEvent::M_MTE1>(abFree_[lane]);
        }
        MatrixReady();
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane])continue;
            if constexpr(FastOnly)if(!leafFast_[lane])continue;
            auto C=lc_.Get<float>()[lane*4096];
            auto f=FixpipeParamsV220(Block,Block,Block,64,false);f.quantPre=QuantMode_t::F322BF16;
            Fixpipe<bfloat16_t,float,NzOutput>(L1(lane,dst)[cc*64+cr*16],C,f);
            if(publish&&a_.tiling.outputMode!=0&&cr<w_.rows) {
                uint32_t rows=w_.rows-cr<Block?w_.rows-cr:Block;
                auto g=FixpipeParamsV220(Block,rows,Block,64,false);g.quantPre=QuantMode_t::F322BF16;
                uint64_t off=Out(a_.tiling,w_,w_.head[lane],a_.tiling.valueHeadNum,64)+cr*64+cc;
                Fixpipe<bfloat16_t,float,CFG_ROW_MAJOR>(Gm<bfloat16_t>(a_.akk,off),C,g);
            }
        }
        SetFlag<HardEvent::FIX_M>(cFree_);
        SetFlag<HardEvent::FIX_MTE1>(l1Ready_);WaitFlag<HardEvent::FIX_MTE1>(l1Ready_);
    }
    __aicore__ inline void LoadLeafB(LocalTensor<bfloat16_t> dst,LocalTensor<bfloat16_t> src) {
        LoadData2DParams p{};p.repeatTimes=1;p.srcStride=1;p.ifTranspose=true;
        LoadData(dst,src,p);
    }
    __aicore__ inline void PackedLeavesPair(bool square,bool publish) {
        uint32_t depth=square?16:32;
        WaitFlag<HardEvent::FIX_M>(cFree_);
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane] || !leafFast_[lane])continue;
            auto A=la_.Get<bfloat16_t>()[lane*8192],B=lb_.Get<bfloat16_t>()[lane*16384];
            auto C=lc_.Get<float>()[lane*4096];
            WaitFlag<HardEvent::M_MTE1>(abFree_[lane]);
            for(uint32_t leaf=0;leaf<4;++leaf) {
                uint32_t b=leaf*16;
                LoadData2DParams copy{};copy.repeatTimes=square?1:2;copy.srcStride=0;
                LoadData(A[leaf*depth*16],L1(lane,square?Power:X)[b*80],copy);
                if(square)LoadLeafB(B[leaf*256],L1(lane,Power)[b*80]);
                else {
                    LoadLeafB(B[leaf*256],L1(lane,I)[b*80]);
                    LoadLeafB(B[1024+leaf*256],L1(lane,Power)[b*80]);
                }
            }
            LoadsReady();
            MmadParams mm{};mm.m=64;mm.n=64;mm.k=depth;mm.cmatrixInitVal=true;
            Mmad(C,A,B,mm);SetFlag<HardEvent::M_MTE1>(abFree_[lane]);
        }
        // A/B load for lane1 overlaps lane0 MAC; both C banks retire before reuse.
        MatrixReady();
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane] || !leafFast_[lane])continue;
            auto C=lc_.Get<float>()[lane*4096];
            for(uint32_t leaf=0;leaf<4;++leaf) {
                uint32_t b=leaf*16;
                auto f=FixpipeParamsV220(16,16,64,64,false);f.quantPre=QuantMode_t::F322BF16;
                Fixpipe<bfloat16_t,float,NzOutput>(L1(lane,square?Power:X)[b*80],C[b*80],f);
                if(publish&&a_.tiling.outputMode!=0&&b<w_.rows) {
                    uint32_t rows=w_.rows-b<16?w_.rows-b:16;
                    auto g=FixpipeParamsV220(16,rows,64,64,false);g.quantPre=QuantMode_t::F322BF16;
                    uint64_t off=Out(a_.tiling,w_,w_.head[lane],a_.tiling.valueHeadNum,64)+b*65;
                    Fixpipe<bfloat16_t,float,CFG_ROW_MAJOR>(Gm<bfloat16_t>(a_.akk,off),C[b*80],g);
                }
            }
        }
        SetFlag<HardEvent::FIX_M>(cFree_);
        SetFlag<HardEvent::FIX_MTE1>(l1Ready_);WaitFlag<HardEvent::FIX_MTE1>(l1Ready_);
    }
    template<uint32_t Block,bool FastOnly=false> __aicore__ inline void MergePair(uint32_t base) {
        uint32_t lower=base+Block;
        GemmPair<Block,FastOnly>(X,P,Tmp,lower,lower,lower,base,lower,base,false);
        GemmPair<Block,FastOnly>(Tmp,X,X,lower,base,base,base,lower,base,true);
    }
    __aicore__ inline LocalTensor<float> L1Fp32(uint32_t lane,uint32_t offset) {
        return L1(lane,offset).template ReinterpretCast<float>();
    }
    __aicore__ inline void CopySquareFp32(uint32_t lane,uint32_t offset,GlobalTensor<float> src) {
        // Each 16x32 slice is four consecutive 16x8 FP32 fractals, exactly
        // the same Zz order as the former four separate 16x16 squares.
        Nd2NzParams p{};p.ndNum=2;p.nValue=16;p.dValue=32;p.srcDValue=32;
        p.srcNdMatrixStride=512;p.dstNzMatrixStride=512;
        p.dstNzNStride=1;p.dstNzC0Stride=16;
        DataCopy(L1Fp32(lane,offset),src,p);
    }
    __aicore__ inline void LoadSquareAFp32(LocalTensor<float> dst,LocalTensor<float> src) {
        LoadData2DParams p{};p.repeatTimes=8;p.srcStride=1;LoadData(dst,src,p);
    }
    __aicore__ inline void LoadSquareBFp32(LocalTensor<float> dst,LocalTensor<float> src) {
        LoadData2dTransposeParams p{};p.repeatTimes=1;p.srcStride=1;p.dstFracGap=1;
        for(uint32_t row=0;row<2;++row)
            for(uint32_t col=0;col<2;++col)
                LoadDataWithTranspose(dst[row*512+col*128],src[(row*2+col)*256],p);
    }
    template<bool Leaves> __aicore__ inline void MergeFp32Pair() {
        // Each lane keeps distinct A/B/C, L1 and GM banks. Only phase boundaries
        // are shared; the two products and FP32 relays within a head are unchanged.
        MmadParams mm{};mm.m=Leaves?16:32;mm.n=32;mm.k=32;mm.cmatrixInitVal=true;
        WaitFlag<HardEvent::FIX_M>(cFree_);
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane] || leafFast_[lane])continue;
            auto A=la_.Get<float>()[lane*4096],B=lb_.Get<float>()[lane*8192];
            auto C=lc_.Get<float>()[lane*4096];
            WaitFlag<HardEvent::M_MTE1>(abFree_[lane]);
            if constexpr(Leaves) {
                LoadData2DParams p{};p.repeatTimes=2;p.srcStride=1;
                LoadData(A,L1Fp32(lane,FpX0)[768],p);
                LoadData(A[256],L1Fp32(lane,FpX1)[768],p);
            } else LoadSquareAFp32(A,L1Fp32(lane,FpX1));
            LoadSquareBFp32(B,L1Fp32(lane,Leaves?FpP:FpFinalP));LoadsReady();
            Mmad(C,A,B,mm);SetFlag<HardEvent::M_MTE1>(abFree_[lane]);
        }
        MatrixReady();
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane] || leafFast_[lane])continue;
            auto C=lc_.Get<float>()[lane*4096];
            auto f=FixpipeParamsV220(32,Leaves?16:32,Leaves?16:32,32,false);
            f.quantPre=QuantMode_t::NoQuant;
            auto relay=Gm<float>(a_.workspace+GmLane(core_,slot_,lane)+MergeFp32);
            Fixpipe<float,float,CFG_ROW_MAJOR>(relay,C,f);
        }
        // Both FP32 GM producers retire before either DMA consumer. A2/A3
        // cannot directly relay FP32 L0C to L1; do not quantize this boundary.
        PipeBarrier<PIPE_ALL>();
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane] || leafFast_[lane])continue;
            auto relay=Gm<float>(a_.workspace+GmLane(core_,slot_,lane)+MergeFp32);
            if constexpr(Leaves) {
                Nd2NzParams p{};p.ndNum=1;p.nValue=16;p.dValue=32;p.srcDValue=32;
                p.dstNzNStride=1;p.dstNzC0Stride=16;
                DataCopy(L1Fp32(lane,FpT),relay,p);
            } else CopySquareFp32(lane,FpT,relay);
        }
        CopiesReady();
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane] || leafFast_[lane])continue;
            auto A=la_.Get<float>()[lane*4096],B=lb_.Get<float>()[lane*8192];
            auto C=lc_.Get<float>()[lane*4096];
            WaitFlag<HardEvent::M_MTE1>(abFree_[lane]);
            if constexpr(Leaves) {
                LoadData2DParams p{};p.repeatTimes=4;p.srcStride=1;
                LoadData(A,L1Fp32(lane,FpT),p);
                LoadData2dTransposeParams t{};t.repeatTimes=1;t.srcStride=1;t.dstFracGap=1;
                LoadDataWithTranspose(B,L1Fp32(lane,FpX0),t);
                LoadDataWithTranspose(B[128],L1Fp32(lane,FpX0)[256],t);
                LoadDataWithTranspose(B[512],L1Fp32(lane,FpX0)[256],t);
                LoadDataWithTranspose(B[640],L1Fp32(lane,FpX1),t);
            } else {
                LoadSquareAFp32(A,L1Fp32(lane,FpT));
                LoadSquareBFp32(B,L1Fp32(lane,FpX0));
            }
            LoadsReady();Mmad(C,A,B,mm);SetFlag<HardEvent::M_MTE1>(abFree_[lane]);
        }
        MatrixReady();
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane] || leafFast_[lane])continue;
            auto C=lc_.Get<float>()[lane*4096];
            if constexpr(Leaves) {
                auto leaves=Gm<float>(a_.workspace+GmLane(core_,slot_,lane)+DiagFp32);
                for(uint32_t pair=0;pair<2;++pair) {
                    const uint32_t row=pair*32+16,col=pair*32;
                    auto fp=FixpipeParamsV220(16,16,16,16,false);fp.quantPre=QuantMode_t::NoQuant;
                    Fixpipe<float,float,CFG_ROW_MAJOR>(leaves[pair*1024+512],C[pair*256],fp);
                    auto bf=FixpipeParamsV220(16,16,16,64,false);bf.quantPre=QuantMode_t::F322BF16;
                    Fixpipe<bfloat16_t,float,NzOutput>(L1(lane,X)[col*64+row*16],C[pair*256],bf);
                    if(a_.tiling.outputMode!=0 && row<w_.rows) {
                        const uint32_t rows=w_.rows-row<16?w_.rows-row:16;
                        auto g=FixpipeParamsV220(16,rows,16,64,false);g.quantPre=QuantMode_t::F322BF16;
                        const uint64_t off=Out(a_.tiling,w_,w_.head[lane],a_.tiling.valueHeadNum,64)+row*64+col;
                        Fixpipe<bfloat16_t,float,CFG_ROW_MAJOR>(Gm<bfloat16_t>(a_.akk,off),C[pair*256],g);
                    }
                }
            } else {
                auto bf=FixpipeParamsV220(32,32,32,64,false);bf.quantPre=QuantMode_t::F322BF16;
                Fixpipe<bfloat16_t,float,NzOutput>(L1(lane,X)[32*16],C,bf);
                if(a_.tiling.outputMode!=0) {
                    auto g=FixpipeParamsV220(32,w_.rows-32,32,64,false);g.quantPre=QuantMode_t::F322BF16;
                    const uint64_t off=Out(a_.tiling,w_,w_.head[lane],a_.tiling.valueHeadNum,64)+32*64;
                    Fixpipe<bfloat16_t,float,CFG_ROW_MAJOR>(Gm<bfloat16_t>(a_.akk,off),C,g);
                }
            }
        }
        if constexpr(Leaves) {
            PipeBarrier<PIPE_ALL>();
            if(w_.rows>32) {
                for(uint32_t lane=0;lane<2;++lane) {
                    if(!w_.active[lane] || leafFast_[lane])continue;
                    const uint64_t base=GmLane(core_,slot_,lane);
                    auto leaves=Gm<float>(a_.workspace+base+DiagFp32);
                    Nd2NzParams cross{};cross.ndNum=2;cross.nValue=16;cross.dValue=16;cross.srcDValue=16;
                    cross.srcNdMatrixStride=1024;cross.dstNzMatrixStride=1024;
                    cross.dstNzNStride=1;cross.dstNzC0Stride=16;
                    DataCopy(L1Fp32(lane,FpX0)[512],leaves[512],cross);
                }
                CopiesReady();
            }
        }
        SetFlag<HardEvent::FIX_M>(cFree_);
        SetFlag<HardEvent::FIX_MTE1>(l1Ready_);WaitFlag<HardEvent::FIX_MTE1>(l1Ready_);
    }
    __aicore__ inline void PreloadRhs() {
        WaitFlag<HardEvent::MTE1_MTE2>(scoreReleased_);
        // The explicit score-release event retires the final L1 reader. RHS targets overlap old
        // score payload, never resident identity or live inverse matrices.
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane])continue;
            auto source=Gm<bfloat16_t>(a_.workspace+GmLane(core_,slot_,lane));
            CopyNz(L1(lane,L1RhsK),source[RhsK/2],64,128);
            CopyNz(L1(lane,L1RhsV),source[RhsV/2],64,128);
        }
        // MTE2 is ordered. InversePair's CopiesReady covers these copies too;
        // issue them before waiting for Vector's Lower/Aqk publication.
    }
    __aicore__ inline void InversePair() {
        if constexpr(Norm==0 || Gate==0) {
            for(uint32_t lane=0;lane<2;++lane) {
                if(!w_.active[lane])continue;
                auto source=Gm<bfloat16_t>(a_.workspace+GmLane(core_,slot_,lane));
                if constexpr(Gate!=0)CopyNz(L1(lane,P),source[Lower/2],64,64);
                CopyNz(L1(lane,X),source[Payload/2],64,64);
            }
            CopiesReady();
        } else {
            // Arch22 CrossCoreWaitFlag blocks all subsequent instructions.
            // Lower-ready already acquires the producer's drained choice store;
            // reading that flag does not require retiring independent RHS DMA.
            for(uint32_t lane=0;lane<2;++lane) {
                leafFast_[lane]=false;
                if(!w_.active[lane])continue;
                const uint64_t base=GmLane(core_,slot_,lane);
                auto choice=Gm<float>(a_.workspace+base+InverseChoice);
                DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(choice);
                leafFast_[lane]=choice.GetValue(0)==0.0F;
                auto source=Gm<bfloat16_t>(a_.workspace+base);
                if(leafFast_[lane]) {
                    CopyNz(L1(lane,X),source[Eye/2],64,64);
                    CopyNz(L1(lane,P),source[Lower/2],64,64);
                    CopyNz(L1(lane,Power),source[Lower/2],64,64);
                    if(!identityReady_[lane]) {
                        CopyNz(L1(lane,I),source[Eye/2],64,64);identityReady_[lane]=true;
                    }
                }
            }
            CopiesReady();
            if(leafFast_[0] || leafFast_[1]) {
                for(uint32_t iteration=0;iteration<4;++iteration) {
                    PackedLeavesPair(false,iteration==3);
                    if(iteration!=3)PackedLeavesPair(true,false);
                }
                if(w_.rows>16)MergePair<16,true>(0);
                if(w_.rows>48)MergePair<16,true>(32);
            }
        }
        if constexpr(Gate!=0)if(w_.rows>32) {
            if constexpr(Norm!=0) {
                if(leafFast_[0] || leafFast_[1])MergePair<32,true>(0);
            } else MergePair<32>(0);
        }
        if constexpr(Norm!=0 && Gate!=0) {
            // Consume the final publication even for an all-fast pair so flags
            // cannot leak across slot generations. Risk relays are read only now.
            CrossCoreWaitFlag<0x2,PIPE_MTE2>(4+slot_);
            bool risk=false;
            for(uint32_t lane=0;lane<2;++lane) {
                if(!w_.active[lane] || leafFast_[lane])continue;
                risk=true;
                const uint64_t base=GmLane(core_,slot_,lane);
                CopyNz(L1(lane,X),Gm<bfloat16_t>(a_.workspace+base+Payload),64,64);
                if(w_.rows>16) {
                    auto squares=Gm<float>(a_.workspace+base+DiagFp32);
                    Nd2NzParams p{};p.ndNum=w_.rows>32?16:12;p.nValue=16;p.dValue=16;p.srcDValue=16;
                    p.srcNdMatrixStride=256;p.dstNzMatrixStride=256;
                    p.dstNzNStride=1;p.dstNzC0Stride=16;
                    DataCopy(L1Fp32(lane,FpX0),squares,p);
                }
            }
            if(risk)CopiesReady();
            if(risk && w_.rows>16)MergeFp32Pair<true>();
            if(risk && w_.rows>32)MergeFp32Pair<false>();
        }
    }
    __aicore__ inline void PreloadWu(uint32_t lane) {
        uint32_t m=((w_.rows+15)/16)*16;
        auto A=la_.Get<bfloat16_t>()[lane*8192];
        auto B=lb_.Get<bfloat16_t>()[lane*16384];
        WaitFlag<HardEvent::M_MTE1>(abFree_[lane]);
        LoadA(A,L1(lane,X),m,m);
        LoadB(B,L1(lane,L1RhsK),128,m,128);
        LoadB(B[8192],L1(lane,L1RhsV),128,m,128);
        SetFlag<HardEvent::MTE1_MTE2>(laneFree_[lane]);
        LoadsReady();
    }
    __aicore__ inline void Wu() {
        uint32_t m=((w_.rows+15)/16)*16;PreloadWu(0);
        for(uint32_t lane=0;lane<2;++lane) {
            if(!w_.active[lane])continue;
            auto A=la_.Get<bfloat16_t>()[lane*8192];
            auto B=lb_.Get<bfloat16_t>()[lane*16384];auto C=lc_.Get<float>();
            WaitFlag<HardEvent::FIX_M>(cFree_);
            MmadParams mm{};mm.m=m;mm.n=128;mm.k=m;mm.cmatrixInitVal=true;
            Mmad(C,A,B,mm);Mmad(C[8192],A,B[8192],mm);
            SetFlag<HardEvent::M_MTE1>(abFree_[lane]);
            // Next head uses distinct A/B banks while this head's MAC is active.
            if(lane==0&&w_.active[1])PreloadWu(1);
            MatrixReady();auto p=FixpipeParamsV220(128,w_.rows,m,128,false);p.quantPre=QuantMode_t::F322BF16;
            uint64_t off=Out(a_.tiling,w_,w_.head[lane],a_.tiling.valueHeadNum,128);
            Fixpipe<bfloat16_t,float,CFG_ROW_MAJOR>(Gm<bfloat16_t>(a_.w,off),C,p);
            Fixpipe<bfloat16_t,float,CFG_ROW_MAJOR>(Gm<bfloat16_t>(a_.u,off),C[8192],p);
            SetFlag<HardEvent::FIX_M>(cFree_);
        }
    }
public:
    __aicore__ inline Cube(const PrepareKernelArgs &a):a_(a) {}
    __aicore__ inline void Init(const PrepareKernelArgs &a,TPipe *pipe) {
        pipe_=pipe;core_=GetBlockIdx();
        if constexpr(Norm!=0 && Gate!=0)SetHF32Mode(HF32Mode::DISABLE);
        pipe_->InitBuffer(l1_,L1Bytes);pipe_->InitBuffer(la_,0x10000);
        pipe_->InitBuffer(lb_,0x10000);pipe_->InitBuffer(lc_,0x10000);
        copyReady_=pipe_->AllocEventID<HardEvent::MTE2_MTE1>();
        loadReady_=pipe_->AllocEventID<HardEvent::MTE1_M>();
        matrixReady_=pipe_->AllocEventID<HardEvent::M_FIX>();
        cFree_=pipe_->AllocEventID<HardEvent::FIX_M>();
        l1Ready_=pipe_->AllocEventID<HardEvent::FIX_MTE1>();
        scoreReleased_=pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
        for(uint32_t i=0;i<2;++i) {
            identityReady_[i]=false;
            abFree_[i]=pipe_->AllocEventID<HardEvent::M_MTE1>();
            laneFree_[i]=pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
            SetFlag<HardEvent::M_MTE1>(abFree_[i]);SetFlag<HardEvent::MTE1_MTE2>(laneFree_[i]);
        }
        SetFlag<HardEvent::FIX_M>(cFree_);
    }
    __aicore__ inline void Process() {
        if(core_>=a_.tiling.usedCoreNum)return;
        CrossCoreSetFlag<0x2,PIPE_FIX>(6);CrossCoreSetFlag<0x2,PIPE_FIX>(7);
        uint32_t generation=0;
        for(uint64_t job=core_;job<WorkCount(a_.tiling);job+=a_.tiling.usedCoreNum,++generation) {
            uint64_t workJob=job;
            if constexpr(Norm!=0 && Gate!=0)workJob=RotateWork(a_.tiling,job,core_,generation);
            slot_=generation&1;w_=Describe(a_,workJob);
            CrossCoreWaitFlag<0x2,PIPE_MTE2>(slot_);
            if(w_.rows) {
                for(uint32_t lane=0;lane<2;++lane)if(w_.active[lane])Scores(lane);
                // Both lane score leases end before RHS reuses their L1 bytes.
                SetFlag<HardEvent::MTE1_MTE2>(scoreReleased_);
            }
            CrossCoreSetFlag<0x2,PIPE_FIX>(2+slot_);
            CrossCoreWaitFlag<0x2,PIPE_MTE2>(RhsReadyBase+slot_);
            if(w_.rows)PreloadRhs();
            if constexpr(Norm!=0 && Gate!=0)
                CrossCoreWaitFlag<0x2,PIPE_MTE2>(LowerChoiceReadyBase+slot_);
            else CrossCoreWaitFlag<0x2,PIPE_MTE2>(4+slot_);
            if(w_.rows)InversePair();
            else if constexpr(Norm!=0 && Gate!=0)CrossCoreWaitFlag<0x2,PIPE_MTE2>(4+slot_);
            if(w_.rows)Wu();
            CrossCoreSetFlag<0x2,PIPE_FIX>(6+slot_);
        }
        WaitFlag<HardEvent::FIX_M>(cFree_);
        for(uint32_t i=0;i<2;++i) {
            WaitFlag<HardEvent::M_MTE1>(abFree_[i]);WaitFlag<HardEvent::MTE1_MTE2>(laneFree_[i]);
            pipe_->ReleaseEventID<HardEvent::M_MTE1>(abFree_[i]);
            pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(laneFree_[i]);
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(copyReady_);
        pipe_->ReleaseEventID<HardEvent::MTE1_M>(loadReady_);
        pipe_->ReleaseEventID<HardEvent::M_FIX>(matrixReady_);
        pipe_->ReleaseEventID<HardEvent::FIX_M>(cFree_);
        pipe_->ReleaseEventID<HardEvent::FIX_MTE1>(l1Ready_);
        pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(scoreReleased_);
    }
};
template<uint32_t Norm,uint32_t Gate,class GateT,class BetaT> __aicore__ inline void Run(const PrepareKernelArgs &a) {
    TPipe pipe;
    if ASCEND_IS_AIV {Vector<Norm,Gate,GateT,BetaT> v(a);v.Init(a,&pipe);v.Process();}
    if ASCEND_IS_AIC {Cube<Norm,Gate> c(a);c.Init(a,&pipe);c.Process();}
}
}
