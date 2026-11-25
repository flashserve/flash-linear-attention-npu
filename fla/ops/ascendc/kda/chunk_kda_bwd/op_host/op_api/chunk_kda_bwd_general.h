// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
static aclnnStatus GeneralChunkKdaBwdGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *wOptional, const aclTensor *qgOptional,
    const aclTensor *kgOptional, const aclTensor *vNewOptional,
    const aclTensor *hOptional, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, double lowerBound,
    bool disableRecompute, bool useExp2, bool stateVFirst,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaEffOptional, const char *layout,
    bool useBetaSigmoidInKernel, bool allowNegEigval,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dh0OutOptional,
    const aclTensor *dAOutOptional, const aclTensor *dBiasOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto invalid=[&](const char *name,const char *why){
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME,name,why);return ACLNN_ERR_PARAM_INVALID;
    };
    if(GetCurrentPlatformInfo().GetCurNpuArch()!=NpuArch::DAV_2201)return invalid("soc","FP32 cache backward requires A2/A3");
    if(chunkSize!=64 || stateVFirst || initialStateOptional || dhtOptional || dh0OutOptional)
        return invalid("chunk/state","requires C64, K-first; combined h0/dht/dh0 remain outside this scope");
    if((useGateInKernel && !safeGate) || (allowNegEigval && !useBetaSigmoidInKernel))
        return invalid("flags","raw training gate requires safe_gate; negative eigval requires sigmoid");
    if(!std::isfinite(static_cast<float>(scale)) || !std::isfinite(static_cast<float>(lowerBound)))return invalid("scale/lower_bound","must be finite");
    if(!layout)return invalid("layout","missing");
    const bool packed=std::strcmp(layout,"NTD")==0 || std::strcmp(layout,"TND")==0;
    const bool tm=std::strcmp(layout,"BSND")==0 || std::strcmp(layout,"TND")==0;
    if(!packed && std::strcmp(layout,"BNSD") && std::strcmp(layout,"BSND"))return invalid("layout","unsupported");
    const auto &vs=v->GetViewShape();const size_t rank=packed?3:4,base=packed?0:1;
    if(vs.GetDimNum()!=rank || q->GetViewShape().GetDimNum()!=rank || (packed && !cuSeqlensOptional))return invalid("rank","layout mismatch");
    const int64_t B=packed?1:vs.GetDim(0),H=vs.GetDim(base+(tm?1:0)),T=vs.GetDim(base+(tm?0:1));
    const int64_t K=q->GetViewShape().GetDim(rank-1),V=vs.GetDim(rank-1);
    if(B<=0 || H<=0 || T<=0 || !((K==V && (K==60 || K==64 || K==100 || K==128 || K==256)) || (K==64 && V==32)))
        return invalid("dimensions","requires equal heads and one of the six general K/V pairs");
    if(T>INT64_MAX/B/H/256/16)return invalid("size","workspace size overflow");
    const auto data=v->GetDataType();if(data!=DataType::DT_FLOAT16 && data!=DataType::DT_BF16)return invalid("v","requires FP16/BF16");
    const auto key=packed?MakeShape({H,T,K}):MakeShape({B,H,T,K});
    const auto val=packed?MakeShape({H,T,V}):MakeShape({B,H,T,V});
    const auto scalar=packed?MakeShape({H,T}):MakeShape({B,H,T});
    const auto matrix=packed?MakeShape({H,T,64}):MakeShape({B,H,T,64});
    const auto keyTM=packed?MakeShape({T,H,K}):MakeShape({B,T,H,K});
    const auto valTM=packed?MakeShape({T,H,V}):MakeShape({B,T,H,V});
    const auto scalarTM=packed?MakeShape({T,H}):MakeShape({B,T,H});
    const bool savedQ=q->GetDataType()==DataType::DT_FLOAT;
    if(!MatchesTensor(q,savedQ?key:(tm?keyTM:key),savedQ?DataType::DT_FLOAT:data) ||
       !MatchesTensor(k,savedQ?key:(tm?keyTM:key),savedQ?DataType::DT_FLOAT:data) ||
       !MatchesTensor(v,tm?valTM:val,data) || !MatchesTensor(dO,valTM,data) ||
       !MatchesTensor(aqk,matrix,DataType::DT_FLOAT) || !MatchesTensor(akk,matrix,DataType::DT_FLOAT))
        return invalid("inputs","q/k saved FP32 or original data, v/dO original dtype, Aqk/Akk head-major FP32");
    if(!MatchesTensor(beta,scalarTM,data) && !MatchesTensor(beta,scalarTM,DataType::DT_FLOAT))return invalid("beta","token-major original dtype/FP32 required");
    if(qRstdOptional && (!savedQ || !MatchesTensor(qRstdOptional,scalar,DataType::DT_FLOAT) || !MatchesTensor(kRstdOptional,scalar,DataType::DT_FLOAT)))
        return invalid("rstd","normalization requires saved FP32 qhat/khat and matching FP32 rstd");
    if(useBetaSigmoidInKernel && !MatchesTensor(betaEffOptional,scalar,DataType::DT_FLOAT))return invalid("beta_eff","sigmoid requires saved FP32 effective beta");
    if(betaEffOptional && !MatchesTensor(betaEffOptional,scalar,DataType::DT_FLOAT))return invalid("beta_eff","must match head-major FP32 scalar shape");
    const auto gateType=rawGOptional?rawGOptional->GetDataType():DataType::DT_FLOAT;
    if(rawGOptional && (!(gateType==data || gateType==DataType::DT_FLOAT) || !MatchesTensor(rawGOptional,keyTM,gateType)))return invalid("raw_g","token-major gate shape/dtype mismatch");
    if(useGateInKernel && (!rawGOptional || !MatchesTensor(aLogOptional,MakeShape({H}),DataType::DT_FLOAT)))return invalid("raw_g/a_log","raw safe gate requires both inputs");
    if(dtBiasOptional && (dtBiasOptional->GetDataType()!=DataType::DT_FLOAT || !IsContiguous(dtBiasOptional) || dtBiasOptional->GetViewShape().GetShapeSize()!=H*K))return invalid("dt_bias","requires contiguous FP32 H*K");
    if(!MatchesTensor(dqOut,tm?keyTM:key,data) || !MatchesTensor(dkOut,tm?keyTM:key,data) || !MatchesTensor(dvOut,tm?valTM:val,data) ||
       !MatchesTensor(dbOut,scalarTM,beta->GetDataType()) || !MatchesTensor(dgOut,keyTM,gateType))return invalid("outputs","output shape/dtype mismatch");
    if(dAOutOptional && (!aLogOptional || !MatchesTensor(dAOutOptional,MakeShape({H}),DataType::DT_FLOAT)))return invalid("dA","requires FP32 H and a_log");
    if(dBiasOutOptional && (!dtBiasOptional || dBiasOutOptional->GetDataType()!=DataType::DT_FLOAT || !IsContiguous(dBiasOutOptional) || dBiasOutOptional->GetViewShape().GetShapeSize()!=H*K))return invalid("dbias","requires FP32 H*K and dt_bias");
    int64_t nc=(T+63)/64;std::vector<int64_t> generated;
    if(cuSeqlensOptional){
        const auto *cu=cuSeqlensOptional;const size_t n=cu->Size();
        if(B!=1 || n<2 || (*cu)[0]!=0 || (*cu)[n-1]!=T)return invalid("cu_seqlens","requires B=1 and boundaries from 0 to T");
        nc=0;
        for(size_t seq=0;seq+1<n;++seq){
            if((*cu)[seq]<0 || (*cu)[seq+1]<(*cu)[seq] || (*cu)[seq+1]>T)return invalid("cu_seqlens","must be nondecreasing within T");
            for(int64_t c=0;c<((*cu)[seq+1]-(*cu)[seq]+63)/64;++c,++nc){generated.push_back(seq);generated.push_back(c);}
        }
        if(chunkIndicesOptional){
            if(chunkIndicesOptional->Size()!=generated.size())return invalid("chunk_indices","incorrect count");
            for(size_t i=0;i<generated.size();++i)if((*chunkIndicesOptional)[i]!=generated[i])return invalid("chunk_indices","must be canonical sequence-major pairs");
        }
    }else if(chunkIndicesOptional)return invalid("chunk_indices","requires cu_seqlens");
    const auto state=packed?MakeShape({nc,H,K,V}):MakeShape({B,nc,H,K,V});
    if(disableRecompute){
        for(auto *t:{wOptional,qgOptional,kgOptional})if(!MatchesTensor(t,key,DataType::DT_FLOAT))return invalid("saved cache","w/qg/kg must be FP32 head-major");
        if(!MatchesTensor(vNewOptional,val,DataType::DT_FLOAT) || !MatchesTensor(hOptional,state,DataType::DT_FLOAT) || !MatchesTensor(gk,key,DataType::DT_FLOAT))return invalid("saved state","v_new/h/gk FP32 shapes mismatch");
    }else if(!useGateInKernel && !MatchesTensor(gk,key,DataType::DT_FLOAT))return invalid("gk","preactivated recompute requires saved cumulative G");
    auto uniqueExecutor=CREATE_EXECUTOR();if(!uniqueExecutor.get())return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    auto *ex=uniqueExecutor.get();
    NormalizeOriginalShapes({q,k,v,beta,gk,aqk,akk,wOptional,qgOptional,kgOptional,vNewOptional,hOptional,dO,rawGOptional,aLogOptional,dtBiasOptional,qRstdOptional,kRstdOptional,betaEffOptional,dqOut,dkOut,dvOut,dbOut,dgOut,dAOutOptional,dBiasOutOptional});
    auto alloc=[&](const op::Shape &s,DataType type){return ex->AllocTensor(s,type,Format::FORMAT_ND);};
    auto fp32=[&](const aclTensor *x){return x && x->GetDataType()!=DataType::DT_FLOAT?l0op::Cast(x,DataType::DT_FLOAT,ex):x;};
    if(cuSeqlensOptional && !chunkIndicesOptional)chunkIndicesOptional=ex->AllocIntArray(generated.data(),generated.size());
    const auto *qh=fp32(savedQ?q:(tm?SwapHeadToken(q,packed,ex):q));
    const auto *kh=fp32(savedQ?k:(tm?SwapHeadToken(k,packed,ex):k));
    const auto *vh=tm?SwapHeadToken(v,packed,ex):v;
    const auto *doh=SwapHeadToken(dO,packed,ex);
    const auto *bh=useBetaSigmoidInKernel?betaEffOptional:fp32(SwapHeadToken(beta,packed,ex));
    const auto *raw=rawGOptional?SwapHeadToken(rawGOptional,packed,ex):gk;
    const aclTensor *log=aLogOptional,*bias=dtBiasOptional;
    if(!log){std::vector<float> z(H,0);log=ex->ConvertToTensor(z.data(),z.size(),DataType::DT_FLOAT);}
    if(!bias){std::vector<float> z(H*K,0);bias=ex->ConvertToTensor(z.data(),z.size(),DataType::DT_FLOAT);}
    bias=l0op::Reshape(bias,MakeShape({H,K}),ex);
    if(!qh || !kh || !vh || !doh || !bh || !log || !bias)return ACLNN_ERR_INNER_NULLPTR;
    const aclTensor *w=wOptional,*qg=qgOptional,*kg=kgOptional,*vn=vNewOptional,*h=hOptional,*G=gk;
    if(!disableRecompute){
        w=alloc(key,DataType::DT_FLOAT);qg=alloc(key,DataType::DT_FLOAT);kg=alloc(key,DataType::DT_FLOAT);
        vn=alloc(val,DataType::DT_FLOAT);h=alloc(state,DataType::DT_FLOAT);G=alloc(key,DataType::DT_FLOAT);
        const auto *u=alloc(val,DataType::DT_FLOAT);
        if(!w || !qg || !kg || !vn || !h || !G || !u)return ACLNN_ERR_INNER_NULLPTR;
        const auto recompute=l0op::ChunkKdaBwdRecompute(BatchView(qh,packed,ex),BatchView(kh,packed,ex),BatchView(vh,packed,ex),
            BatchView(useGateInKernel?raw:gk,packed,ex),BatchView(bh,packed,ex),BatchView(akk,packed,ex),log,bias,
            cuSeqlensOptional,chunkIndicesOptional,64,useGateInKernel,useExp2,lowerBound,"BNSD",
            BatchView(w,packed,ex),BatchView(u,packed,ex),BatchView(qg,packed,ex),BatchView(kg,packed,ex),BatchView(G,packed,ex),ex,"",true);
        for(auto *x:recompute)if(!x)return ACLNN_ERR_INNER_NULLPTR;
        const auto fwd=l0op::ChunkFwdH(BatchView(kg,packed,ex),BatchView(w,packed,ex),BatchView(u,packed,ex),nullptr,
            BatchView(G,packed,ex),nullptr,cuSeqlensOptional,chunkIndicesOptional,false,64,true,useExp2,false,
            BatchView(h,packed,ex),BatchView(vn,packed,ex),nullptr,ex);
        if(!fwd[0] || !fwd[1])return ACLNN_ERR_INNER_NULLPTR;
    }
    if(!raw)raw=G;
    const auto *da=alloc(matrix,DataType::DT_FLOAT),*dv0=alloc(val,DataType::DT_FLOAT),*dqraw=alloc(key,DataType::DT_FLOAT);
    const auto *dh=alloc(state,DataType::DT_FLOAT),*dvscan=alloc(val,DataType::DT_FLOAT);
    if(!da || !dv0 || !dqraw || !dh || !dvscan)return ACLNN_ERR_INNER_NULLPTR;
    auto prepare=l0op::ChunkKdaBwdPrepare(aqk,vn,doh,h,cuSeqlensOptional,chunkIndicesOptional,packed?"NTD":"BNSD",scale,64,false,da,dv0,dqraw,ex);
    for(auto *x:prepare)if(!x)return ACLNN_ERR_INNER_NULLPTR;
    auto scan=l0op::ChunkGatedDeltaRuleBwdDhu(BatchView(qg,packed,ex),BatchView(kg,packed,ex),BatchView(w,packed,ex),
        BatchView(doh,packed,ex),BatchView(dv0,packed,ex),nullptr,BatchView(G,packed,ex),nullptr,nullptr,
        cuSeqlensOptional,chunkIndicesOptional,scale,64,useExp2,false,BatchView(dh,packed,ex),nullptr,BatchView(dvscan,packed,ex),ex);
    if(!scan[0] || !scan[2])return ACLNN_ERR_INNER_NULLPTR;
    const auto *dq=alloc(key,data),*dk=alloc(key,data),*dv=alloc(val,data),*db=alloc(scalar,DataType::DT_FLOAT),*dg=alloc(key,gateType);
    const auto *dA=alloc(MakeShape({H}),DataType::DT_FLOAT),*dbias=alloc(MakeShape({H,K}),DataType::DT_FLOAT);
    if(!dq || !dk || !dv || !db || !dg || !dA || !dbias)return ACLNN_ERR_INNER_NULLPTR;
    const auto result=l0op::ChunkKdaBwdFinalize(qh,kh,vh,G,raw,bh,log,bias,akk,vn,h,dh,dvscan,da,dqraw,qRstdOptional,kRstdOptional,
        cuSeqlensOptional,chunkIndicesOptional,scale,lowerBound,64,safeGate,useGateInKernel,useExp2,false,
        useBetaSigmoidInKernel,allowNegEigval,packed?"NTD":"BNSD",dq,dk,dv,db,dg,dA,dbias,ex,"");
    for(auto *x:result)if(!x)return ACLNN_ERR_INNER_NULLPTR;
    if(!CopyGradient(tm?SwapHeadToken(dq,packed,ex):dq,dqOut,ex) ||
       !CopyGradient(tm?SwapHeadToken(dk,packed,ex):dk,dkOut,ex) ||
       !CopyGradient(tm?SwapHeadToken(dv,packed,ex):dv,dvOut,ex) ||
       !CopyGradient(SwapHeadToken(db,packed,ex),dbOut,ex) || !CopyGradient(SwapHeadToken(dg,packed,ex),dgOut,ex) ||
       (dAOutOptional && !CopyGradient(dA,dAOutOptional,ex)) || (dBiasOutOptional && !CopyGradient(dbias,dBiasOutOptional,ex)))return ACLNN_ERR_INNER_NULLPTR;
    *workspaceSize=uniqueExecutor->GetWorkspaceSize();uniqueExecutor.ReleaseTo(executor);return ACLNN_SUCCESS;
}
