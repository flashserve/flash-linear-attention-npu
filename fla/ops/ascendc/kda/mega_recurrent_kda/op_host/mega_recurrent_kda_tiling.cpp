#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "err/ops_err.h"
#include "kda_workspace.h"
#include "../op_kernel/mega_recurrent_kda_tiling_data.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace {
using Dims=std::vector<int64_t>;
Dims dimensions(gert::TilingContext *ctx,size_t input) {
    auto tensor=input==18?ctx->GetOptionalInputShape(input):ctx->GetInputShape(input);
    if(!tensor) return {};
    const auto &origin=tensor->GetOriginShape();
    const auto &s=origin.GetDimNum()?origin:tensor->GetStorageShape();
    Dims dims(s.GetDimNum());
    for(size_t i=0;i<dims.size();++i) {
        dims[i]=s.GetDim(i);
        if(dims[i]<=0) throw std::invalid_argument("expected positive input dimensions");
    }
    return dims;
}
uint32_t count(int64_t value) {
    if(value<=0 || uint64_t(value)>UINT32_MAX) throw std::invalid_argument("unsupported index span");
    return static_cast<uint32_t>(value);
}
void expect(const Dims &actual,const Dims &required,const char *name) {
    if(actual!=required) throw std::invalid_argument(name);
}
ge::graphStatus tile(gert::TilingContext *ctx) {
    if(!ctx) return ge::GRAPH_FAILED;
    try {
        auto attrs=ctx->GetAttrs();
        auto metadata=attrs?attrs->GetListInt(3):nullptr;
        if(!metadata || metadata->GetSize()!=11) throw std::invalid_argument("missing logical state geometry");
        const auto *info=metadata->GetData();
        Dims x=dimensions(ctx,0);
        if((x.size()!=2 && x.size()!=3) || x.back()!=7168)
            throw std::invalid_argument("hidden_states must be BSH or TH with H=7168");
        Dims batchDim=dimensions(ctx,14);
        if(batchDim.size()!=1) throw std::invalid_argument("state_indices must be a vector");
        const int64_t batch=batchDim[0],slots=info[0],n=info[1],length=info[2],capacity=info[3];
        if(n>UINT32_MAX/(6*128)) throw std::invalid_argument("head count exceeds device stride range");
        const int64_t width=n*128;
        expect(dimensions(ctx,15),{batch},"accepted-token shape");
        expect(dimensions(ctx,1),{3*width,7168},"QKV projection shape");
        expect(dimensions(ctx,2),{128,7168},"decay-A shape"); expect(dimensions(ctx,3),{width,128},"decay-B shape");
        expect(dimensions(ctx,4),{n,7168},"beta shape"); expect(dimensions(ctx,5),{width,7168},"gate shape");
        expect(dimensions(ctx,6),{128},"norm weight shape"); expect(dimensions(ctx,7),{7168,width},"output weight shape");
        expect(dimensions(ctx,16),{n},"A_log shape"); expect(dimensions(ctx,17),{n,128},"dt_bias shape");
        Dims weights=dimensions(ctx,8);
        uint32_t layout=weights==Dims{4,3*width}?0:(weights==Dims{3,n,4,128}?1:2);
        if(layout==2) expect(weights,{3,4,128},"conv1d weight shape");
        const bool packed=x.size()==2;
        uint32_t seq=packed?0:count(x[1]);
        if(!packed && x[0]!=batch) throw std::invalid_argument("BSH batch mismatch");
        if(packed) expect(dimensions(ctx,18),{batch+1},"cu_seqlens shape");
        uint32_t rows=count(packed?x[0]:kda::checkedProduct(batch,seq));
        if(info[9]<=0 || (info[10]!=0 && info[10]!=1)) throw std::invalid_argument("invalid workspace plan");
        const auto bytes=static_cast<uint64_t>(info[9]);
        auto scale=attrs?attrs->GetAttrPointer<float>(0):nullptr;
        auto lower=attrs?attrs->GetAttrPointer<float>(1):nullptr;
        auto eps=attrs?attrs->GetAttrPointer<float>(2):nullptr;
        if(!scale || !lower || !eps || !std::isfinite(*scale) || !std::isfinite(*lower) ||
            *lower < -5 || *lower>0 || !std::isfinite(*eps) || *eps<std::numeric_limits<float>::min())
            throw std::invalid_argument("invalid scale, lower_bound or rms_norm_eps");
        MegaRecurrentKdaTilingData t{uint64_t(info[4]),uint64_t(info[5]),uint64_t(info[6]),
            uint64_t(info[7]),uint64_t(info[8]),count(batch),seq,rows,
            count(n),count(capacity),packed,layout,bool(info[10]),*scale,*lower,*eps};
        auto raw=ctx->GetRawTilingData(); auto sizes=ctx->GetWorkspaceSizes(1);
        auto platform=platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
        uint32_t cores=platform.GetCoreNumAic();
        if(!cores || !raw || raw->GetCapacity()<sizeof(t) || !sizes) return ge::GRAPH_FAILED;
        auto system=platform.GetLibApiWorkSpaceSize();
        if(bytes>UINT64_MAX-system) return ge::GRAPH_FAILED;
        sizes[0]=bytes+system;
        std::memcpy(raw->GetData(),&t,sizeof(t)); raw->SetDataSize(sizeof(t));
        ctx->SetBlockDim(cores); ctx->SetTilingKey(t.crosses?2:1);
        return ge::GRAPH_SUCCESS;
    } catch(const std::exception &e) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ctx->GetNodeName(),"MegaRecurrentKda",e.what());
        return ge::GRAPH_FAILED;
    }
}
}
namespace {
struct CompileInfo {};
ge::graphStatus prepare(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
}
IMPL_OP_OPTILING(MegaRecurrentKda).Tiling(tile).TilingParse<CompileInfo>(prepare);
