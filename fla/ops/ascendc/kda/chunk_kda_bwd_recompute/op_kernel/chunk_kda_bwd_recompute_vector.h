/** Copyright (c) 2026 Tianjin University, Ltd. Licensed under BSD-3-Clause. */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_VECTOR_H
#define CHUNK_KDA_BWD_RECOMPUTE_VECTOR_H
#include "chunk_kda_bwd_recompute_struct.h"
#include "chunk_kda_bwd_recompute_common.h"
using namespace AscendC;
namespace KDA {
template <typename QkType, typename GateType, typename BetaType, int HEAD_MODE=0>
class ChunkKdaBwdRecomputeVectorProcess {
public:
    __aicore__ inline ChunkKdaBwdRecomputeVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR cu, GM_ADDR indices, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR gk,
        GM_ADDR workspace, const ChunkKdaBwdRecomputeTilingData *t)
        : cu_(cu), indices_(indices), workspace_(workspace), t_(t)
    {
        q_.SetGlobalBuffer((__gm__ QkType*)q); k_.SetGlobalBuffer((__gm__ QkType*)k);
        v_.SetGlobalBuffer((__gm__ QkType*)v); g_.SetGlobalBuffer((__gm__ GateType*)g);
        beta_.SetGlobalBuffer((__gm__ BetaType*)beta); alog_.SetGlobalBuffer((__gm__ float*)aLog);
        bias_.SetGlobalBuffer((__gm__ float*)dtBias); qg_.SetGlobalBuffer((__gm__ QkType*)qg);
        kg_.SetGlobalBuffer((__gm__ QkType*)kg); gk_.SetGlobalBuffer((__gm__ float*)gk);
    }
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe_=pipe;
        pipe->InitBuffer(qBuf_,8192); pipe->InitBuffer(kBuf_,8192);
        pipe->InitBuffer(vBuf_,8192); pipe->InitBuffer(outBuf_,8192);
        pipe->InitBuffer(gBuf_,16384);
        pipe->InitBuffer(gFp32Buf_,16384); pipe->InitBuffer(tmpFp32Buf_,16384);
        pipe->InitBuffer(kRawBuf_,16384); pipe->InitBuffer(workBuf_,16384); pipe->InitBuffer(qFpBuf_,16384);
        pipe->InitBuffer(betaRaw_,256); pipe->InitBuffer(betaFp_,256); pipe->InitBuffer(betaBc_,2048);
        pipe->InitBuffer(dtBiasBuf_,256); pipe->InitBuffer(expABuf_,256); pipe->InitBuffer(accBuf_,256);
        useGate_=t_->useGateInKernel; useExp2_=t_->useExp2;
        hasALog_=t_->hasALog; hasDtBias_=t_->hasDtBias;
        gateScale_=KdaBwdRecomputeBitsToFloat(t_->lowerBoundBits)*(useExp2_?KDA_BWD_RECOMPUTE_RCP_LN2:1.f);
        m2v_=pipe->AllocEventID<HardEvent::MTE2_V>();
        v2m2_=pipe->AllocEventID<HardEvent::V_MTE2>();
        v2m3_=pipe->AllocEventID<HardEvent::V_MTE3>();
        m3v_=pipe->AllocEventID<HardEvent::MTE3_V>();
    }
    __aicore__ inline void Process()
    {
        const uint64_t core=GetBlockIdx()/GetSubBlockNum(), cores=t_->usedAic;
        if (core>=cores || cores==0) return;
        const uint64_t cg=t_->headGroup, nh=(t_->Hv+cg-1)/cg, count=t_->chunkNum*nh;
        const uint64_t channel=GetSubBlockIdx()*64;
        for (uint64_t work=core, ordinal=0;work<count;work+=cores,++ordinal) {
            const uint32_t page=ordinal%2;
            if (ordinal>=2) CrossCoreWaitFlag<0x2,PIPE_MTE3>(KDA_BWD_RECOMPUTE_PAGE_FREE_BASE+page);
            uint64_t bos,eos;
            KdaBwdRecomputeGetChunkOffset(cu_,indices_,t_->B,t_->Hv,t_->T,64,work/nh,bos,eos,t_->isVariable);
            const uint32_t rows=eos-bos;
            const uint64_t batch=t_->isVariable?0:(work/nh)/((t_->T+63)/64);
            const uint64_t bosK=bos-batch*(t_->Hv-t_->Hk)*t_->T;
            auto pageBase=(__gm__ QkType*)workspace_+KdaRecomputePageOffset(core,page,16384,cg);
            GlobalTensor<QkType> vb,kbg; vb.SetGlobalBuffer(pageBase);kbg.SetGlobalBuffer(pageBase+8192);
            uint64_t cached=~uint64_t(0);
            for (uint64_t slot=0;slot<cg && (work%nh)*cg+slot<uint64_t(t_->Hv);++slot) {
                const uint64_t h=(work%nh)*cg+slot;
                uint64_t hk;
                if constexpr (HEAD_MODE==0) hk=h;
                else hk=h/t_->hvPerHk;
                const uint64_t base=(h*t_->T+bos)*128+channel;
                const uint64_t qbase=(hk*t_->T+bosK)*128+channel;
                auto qr=qBuf_.Get<QkType>(),kr=kBuf_.Get<QkType>(),vr=vBuf_.Get<QkType>();
                auto gr=gBuf_.Get<GateType>();
                SetFlag<HardEvent::V_MTE2>(v2m2_);WaitFlag<HardEvent::V_MTE2>(v2m2_);
                if (cached!=hk) {
                    Load(qr,q_,qbase,rows,t_->Hk,2);
                    Load(kr,k_,qbase,rows,t_->Hk,2);
                }
                Load(gr,g_,base,rows,t_->Hv,1); Load(vr,v_,base,rows,t_->Hv,4);
                auto br=betaRaw_.Get<BetaType>();
                DataCopyPad(br,beta_[h*t_->T+bos],{1,rows*uint32_t(sizeof(BetaType)),0,0,0},{false,0,0,0});
                if (hasDtBias_) DataCopy(dtBiasBuf_.Get<float>(),bias_[h*128+channel],64);
                if (hasALog_) {
                    // Scalar read is only for this head's scalar, never tensor elements.
                    Duplicate(expABuf_.Get<float>(),alog_.GetValue(h),64);
                    PipeBarrier<PIPE_V>(); Exp(expABuf_.Get<float>(),expABuf_.Get<float>(),64);
                }
                SetFlag<HardEvent::MTE2_V>(m2v_);WaitFlag<HardEvent::MTE2_V>(m2v_);
                auto gf=gFp32Buf_.Get<float>(), raw=kRawBuf_.Get<float>();
                auto bf=betaFp_.Get<float>(), bc=betaBc_.Get<float>();
                Duplicate(bf,0.0f,64);PipeBarrier<PIPE_V>();
                Convert(gf,gr,rows*64); Convert(bf,br,rows);
                if (cached!=hk) { Cast(raw,kr,RoundMode::CAST_NONE,rows*64);cached=hk; }
                PipeBarrier<PIPE_V>();
                if (useGate_) ApplySafeGate(gf,rows);
                else if(useExp2_) { Muls(gf,gf,KDA_BWD_RECOMPUTE_RCP_LN2,rows*64);PipeBarrier<PIPE_V>(); }
                auto acc=accBuf_.Get<float>(); Duplicate(acc,0.f,64);PipeBarrier<PIPE_V>();
                CumsumRows(gf,acc,rows);
                if(useGate_) Store(gk_,gf,base,rows);
                auto exp=tmpFp32Buf_.Get<float>(),qf=qFpBuf_.Get<float>(),scratch=workBuf_.Get<float>();
                ExpGate(gf,exp,rows); Cast(qf,qr,RoundMode::CAST_NONE,rows*64);PipeBarrier<PIPE_V>();
                Mul(qf,qf,exp,rows*64);PipeBarrier<PIPE_V>();
                CastStore(qg_,qf,base,rows,false);
                Brcb(bc,bf,static_cast<uint8_t>(((rows+7)/8)),{1,8});PipeBarrier<PIPE_V>();
                MulKByBetaExp(scratch,raw,bc,exp,rows);
                CastStore(kbg,scratch,slot*16384+channel,rows,true);
                Cast(qf,vr,RoundMode::CAST_NONE,rows*64);PipeBarrier<PIPE_V>();
                Mul(qf,qf,bc,64,rows,{1,1,0,8,8,1});PipeBarrier<PIPE_V>();
                CastStore(vb,qf,slot*16384+channel,rows,true);
                ComputeKg(scratch,raw,gf,acc,rows);PipeBarrier<PIPE_V>();
                CastStore(kg_,scratch,base,rows,false);
            }
            CrossCoreSetFlag<0x2,PIPE_MTE3>(KDA_BWD_RECOMPUTE_PAGE_READY_BASE+page);
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(m2v_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(v2m2_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(v2m3_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(m3v_);
    }
private:
    template<typename T> __aicore__ inline void Load(LocalTensor<T> dst,GlobalTensor<T> src,
        uint64_t offset,uint32_t rows,uint64_t heads,int64_t flag)
    {
        uint64_t stride=128;
        if(t_->reserved & flag) {
            const uint64_t token=offset/128, bh=token/t_->T,pos=token%t_->T;
            offset=((bh/heads*t_->T+pos)*heads+bh%heads)*128+offset%128;
            stride=heads*128;
        }
        DataCopyPad(dst,src[offset],{uint16_t(rows),uint32_t(64*sizeof(T)),uint32_t((stride-64)*sizeof(T)),0,0},{false,0,0,T(0)});
    }
    template<typename T> __aicore__ inline void Convert(LocalTensor<float> dst,LocalTensor<T> src,uint32_t n)
    {
        if constexpr(std::is_same<T,float>::value) DataCopy(dst,src,((n+7)/8)*8U);
        else Cast(dst,src,RoundMode::CAST_NONE,n);
    }
    template<typename T> __aicore__ inline void Store(GlobalTensor<T> dst,LocalTensor<T> src,uint64_t offset,uint32_t rows)
    {
        SetFlag<HardEvent::V_MTE3>(v2m3_);WaitFlag<HardEvent::V_MTE3>(v2m3_);
        DataCopyPad(dst[offset],src,{uint16_t(rows),uint32_t(64*sizeof(T)),0,uint32_t(64*sizeof(T)),0});
        SetFlag<HardEvent::MTE3_V>(m3v_);WaitFlag<HardEvent::MTE3_V>(m3v_);
    }
    __aicore__ inline void CastStore(GlobalTensor<QkType> dst,LocalTensor<float> src,uint64_t offset,uint32_t rows,bool pad)
    {
        auto out=outBuf_.Get<QkType>();
        if(pad) {Duplicate(out,QkType(0),4096);PipeBarrier<PIPE_V>();}
        Cast(out,src,RoundMode::CAST_RINT,rows*64);PipeBarrier<PIPE_V>();
        Store(dst,out,offset,pad?64:rows);
    }
    __aicore__ inline void ApplySafeGate(LocalTensor<float> &gate, uint32_t rows)
    {
        const uint32_t n = rows * 64;
        const uint8_t stride = 8;
        if (hasDtBias_) {
            auto bias = dtBiasBuf_.Get<float>();
            Add(gate, gate, bias, 64, rows, {1,1,1,stride,stride,0});
            PipeBarrier<PIPE_V>();
        }
        if (hasALog_) {
            auto expA = expABuf_.Get<float>();
            Mul(gate, gate, expA, 64, rows, {1,1,1,stride,stride,0});
            PipeBarrier<PIPE_V>();
        }
        auto tmp = tmpFp32Buf_.Get<float>();
        Abs(tmp, gate, n);
        PipeBarrier<PIPE_V>();
        Muls(tmp, tmp, -1.0f, n);
        Mins(gate, gate, 0.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(tmp, tmp, n);
        Exp(gate, gate, n);
        PipeBarrier<PIPE_V>();
        Adds(tmp, tmp, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Div(gate, gate, tmp, n);
        PipeBarrier<PIPE_V>();
        Muls(gate, gate, gateScale_, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CumsumRows(LocalTensor<float> &gate, LocalTensor<float> &acc, uint32_t rows)
    {
        Add(gate, gate, acc, K_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> src = gate;
        LocalTensor<float> dst = tmpFp32Buf_.Get<float>();
        bool inScratch = false;
        for (uint32_t step = 1; step < rows; step <<= 1) {
            DataCopy(dst, src, step * K_);
            Add(dst[step * K_], src[step * K_], src, (rows - step) * K_);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> swap = src;
            src = dst;
            dst = swap;
            inScratch = !inScratch;
        }
        if (inScratch) {
            DataCopy(gate, src, rows * K_);
            PipeBarrier<PIPE_V>();
        }
        DataCopy(acc, gate[(rows - 1) * K_], K_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExpGate(LocalTensor<float> &gk, LocalTensor<float> &out, uint32_t rows)
    {
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        if (useExp2_) {
            Muls(out, gk, KDA_BWD_RECOMPUTE_LN2, n);
            PipeBarrier<PIPE_V>();
            Exp(out, out, n);
        } else {
            Exp(out, gk, n);
        }
    }

    __aicore__ inline void MulKByBetaExp(
        LocalTensor<float> &out, LocalTensor<float> &kVal, LocalTensor<float> &betaBrcb,
        LocalTensor<float> &expGk, uint32_t rows)
    {
        uint64_t offset = 0;
        uint8_t repeatStride = K_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
        while (offset < K_) {
            Mul(out[offset], kVal[offset], expGk[offset], KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64, rows,
                {1, 1, 1, repeatStride, repeatStride, repeatStride});
            offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64;
        }
        PipeBarrier<PIPE_V>();
        offset = 0;
        while (offset < K_) {
            Mul(out[offset], out[offset], betaBrcb, KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64, rows,
                {1, 1, 0, repeatStride, repeatStride, 1});
            offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeKg(
        LocalTensor<float> &kgOut, LocalTensor<float> &kRaw, LocalTensor<float> &gk,
        LocalTensor<float> &gkLast, uint32_t rows)
    {
        LocalTensor<float> delta = tmpFp32Buf_.Get<float>();
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        const uint8_t rowStride = K_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
        for (uint32_t offset = 0; offset < K_; offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64) {
            Sub(delta[offset], gkLast[offset], gk[offset], KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64,
                rows, {1, 1, 1, rowStride, 0, rowStride});
        }
        PipeBarrier<PIPE_V>();
        if (useExp2_) {
            Muls(delta, delta, KDA_BWD_RECOMPUTE_LN2, n);
            PipeBarrier<PIPE_V>();
        }
        Exp(delta, delta, n);
        PipeBarrier<PIPE_V>();
        Mul(kgOut, kRaw, delta, n);
    }


    GM_ADDR cu_; GM_ADDR indices_; GM_ADDR workspace_;
    const ChunkKdaBwdRecomputeTilingData *t_;
    TPipe *pipe_;
    GlobalTensor<QkType> q_,k_,v_,qg_,kg_;
    GlobalTensor<GateType> g_;
    GlobalTensor<BetaType> beta_;
    GlobalTensor<float> alog_,bias_,gk_;
    TBuf<TPosition::VECCALC> qBuf_,kBuf_,vBuf_,outBuf_,gBuf_;
    TBuf<TPosition::VECCALC> gFp32Buf_,tmpFp32Buf_,kRawBuf_,workBuf_,qFpBuf_;
    TBuf<TPosition::VECCALC> betaRaw_,betaFp_,betaBc_,dtBiasBuf_,expABuf_,accBuf_;
    TEventID m2v_,v2m3_,m3v_,v2m2_;
    static constexpr uint64_t K_=64;
    bool useGate_,useExp2_,hasALog_,hasDtBias_;
    float gateScale_;
};
}
#endif
