#include "aclnn_mega_recurrent_kda.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <cmath>
#include <array>
#include <algorithm>
#include <vector>
#include "../kda_workspace.h"
#include <initializer_list>
#include <limits>

using namespace op;
namespace {
// All supported views are contiguous inside a physical slot. Check an output
// interval against the first slot it can intersect, including strided pools.
bool disjointOutput(const aclTensor *input, uintptr_t outBegin, uintptr_t outEnd) {
    const auto &shape=input->GetViewShape();
    if(!shape.GetDimNum() || shape.GetDim(0)<=0 || input->Numel()<=0) return false;
    const uint64_t slots=shape.GetDim(0);
    const uint64_t bytesPerElement=op::TypeSize(input->GetDataType());
    const uint64_t inner=static_cast<uint64_t>(input->Numel())/slots;
    const auto &strides=input->GetViewStrides();
    if(!bytesPerElement || strides.size()!=shape.GetDimNum() || strides[0]<0 ||
       inner>UINTPTR_MAX/bytesPerElement || uint64_t(strides[0])>UINTPTR_MAX/bytesPerElement) return false;
    const uintptr_t begin=reinterpret_cast<uintptr_t>(input->GetData());
    const uint64_t width=inner*bytesPerElement, step=uint64_t(strides[0])*bytesPerElement;
    if(!begin || !width || width>UINTPTR_MAX-begin) return false;
    if(outEnd<=begin) return true;
    uint64_t slot=0;
    if(outBegin>=begin+width) {
        if(slots==1) return true;
        if(!step) return false;
        slot=(outBegin-begin-width)/step+1;
    }
    return slot>=slots || (slot<=(UINTPTR_MAX-begin)/std::max<uint64_t>(step,1) && begin+slot*step>=outEnd);
}
}
namespace l0op {
OP_TYPE_REGISTER(MegaRecurrentKda);

aclnnStatus LaunchMegaRecurrentKda(
    const aclTensor *hidden_states,
    const aclTensor *qkv_projection_weight,
    const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight,
    const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight,
    const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight,
    const aclTensor *conv1d_weight,
    aclTensor *conv_state,
    const aclTensor *checkpoint_state,
    aclTensor *replay_u,
    aclTensor *replay_k,
    aclTensor *replay_decay,
    const aclIntArray *state_indices,
    const aclIntArray *conv_num_accepted_tokens,
    const aclTensor *a_log,
    const aclTensor *dt_bias,
    const aclIntArray *cu_seqlens, float scale, float lower_bound, float rms_norm_eps,
    const aclTensor *out, aclOpExecutor *executor) {
    if(conv_state->GetViewShape().GetDimNum()!=3 || replay_u->GetViewShape().GetDimNum()!=4)
        return ACLNN_ERR_PARAM_INVALID;
    const int64_t slots=conv_state->GetViewShape().GetDim(0), length=conv_state->GetViewShape().GetDim(1);
    const int64_t heads=replay_u->GetViewShape().GetDim(1), capacity=replay_u->GetViewShape().GetDim(2);
    if(heads<=0 || heads>UINT32_MAX/(6*128) || slots<=0 || length<=0 || capacity<=0)
        return ACLNN_ERR_PARAM_INVALID;
    std::array<int64_t,11> geometry{slots,heads,length,capacity};
    const auto stateGeometry=[&](const aclTensor *tensor,std::initializer_list<int64_t> dims,int64_t &stride) {
        const auto &view=tensor->GetViewShape();
        const auto &strides=tensor->GetViewStrides();
        if(view.GetDimNum()!=dims.size() || strides.size()!=dims.size()) return false;
        int64_t packed=1;
        for(size_t j=dims.size();j-->0;) {
            int64_t n=dims.begin()[j];
            if(n!=view.GetDim(j) || n<=0) return false;
            if(j) {
                if(strides[j]!=packed) return false;
                if(packed>INT64_MAX/n) return false;
                packed*=n;
            }
        }
        stride=strides[0];
        return stride>=packed;
    };
    if(!stateGeometry(conv_state,{slots,length,3*heads*128},geometry[4]) ||
       !stateGeometry(checkpoint_state,{slots,heads,128,128},geometry[5]) ||
       !stateGeometry(replay_u,{slots,heads,capacity,128},geometry[6]) ||
       !stateGeometry(replay_k,{slots,heads,capacity,128},geometry[7]) ||
       !stateGeometry(replay_decay,{slots,heads,capacity,128},geometry[8])) return ACLNN_ERR_PARAM_INVALID;
    const auto &hidden=hidden_states->GetViewShape();
    const int64_t rank=hidden.GetDimNum();
    if ((rank!=2 && rank!=3) || hidden.GetDim(rank-1)!=7168 || !state_indices || !conv_num_accepted_tokens)
        return ACLNN_ERR_PARAM_INVALID;
    const auto batch=state_indices->Size();
    if (!batch || batch>UINT32_MAX || conv_num_accepted_tokens->Size()!=batch ||
        (rank==3 && (hidden.GetDim(0)!=static_cast<int64_t>(batch) || cu_seqlens)) ||
        (rank==2 && (!cu_seqlens || cu_seqlens->Size()!=batch+1))) return ACLNN_ERR_PARAM_INVALID;
    std::vector<int32_t> indices(batch),accepted(batch);
    for(size_t i=0;i<batch;++i) {
        const int64_t id=(*state_indices)[i], count=(*conv_num_accepted_tokens)[i];
        if(id<0 || id>INT32_MAX || count<0 || count>INT32_MAX) return ACLNN_ERR_PARAM_INVALID;
        indices[i]=static_cast<int32_t>(id);
        accepted[i]=static_cast<int32_t>(count);
    }
    const int64_t sequence=rank==3?hidden.GetDim(1):0;
    try {
        const uint64_t tokens=rank==3?kda::checkedProduct(batch,sequence):hidden.GetDim(0);
        if(!tokens || tokens>UINT32_MAX) return ACLNN_ERR_PARAM_INVALID;
        std::vector<int64_t> prefix;
        if(cu_seqlens) {
            prefix.reserve(batch+1);
            for(size_t i=0;i<=batch;++i) prefix.push_back((*cu_seqlens)[i]);
        }
        kda::Shape shape{static_cast<int64_t>(batch),sequence,static_cast<int64_t>(tokens),heads,slots,length,capacity};
        auto plan=kda::makeTiling(shape,prefix.empty()?nullptr:prefix.data(),indices.data(),accepted.data(),3);
        if(plan.bytes>INT64_MAX) return ACLNN_ERR_PARAM_INVALID;
        geometry[9]=static_cast<int64_t>(plan.bytes);
        geometry[10]=plan.kernel==kda::TemplateId::T02;
    } catch(const std::exception &) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto *ids=executor->ConvertToTensor(indices.data(),indices.size(),op::DataType::DT_INT32);
    const auto *counts=executor->ConvertToTensor(accepted.data(),accepted.size(),op::DataType::DT_INT32);
    const auto *prefixTensor=cu_seqlens?executor->ConvertToTensor(cu_seqlens,op::DataType::DT_INT64):nullptr;
    if(!ids || !counts || (cu_seqlens && !prefixTensor)) return ACLNN_ERR_INNER_NULLPTR;
    const aclIntArray *stateInfo=executor->AllocIntArray(geometry.data(),geometry.size());
    if(!stateInfo) return ACLNN_ERR_INNER_NULLPTR;
    return ADD_TO_LAUNCHER_LIST_AICORE(MegaRecurrentKda,
        OP_INPUT(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight, output_norm_weight, output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k, replay_decay, ids, counts, a_log, dt_bias, prefixTensor),
        OP_OUTPUT(out, conv_state, replay_u, replay_k, replay_decay),
        OP_ATTR(scale, lower_bound, rms_norm_eps, stateInfo));
}
} // namespace l0op

extern "C" aclnnStatus aclnnMegaRecurrentKdaGetWorkspaceSize(
    const aclTensor *hidden_states,
    const aclTensor *qkv_projection_weight,
    const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight,
    const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight,
    const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight,
    const aclTensor *conv1d_weight,
    aclTensor *conv_state,
    const aclTensor *checkpoint_state,
    aclTensor *replay_u,
    aclTensor *replay_k,
    aclTensor *replay_decay,
    const aclIntArray *state_indices,
    const aclIntArray *conv_num_accepted_tokens,
    const aclTensor *a_log,
    const aclTensor *dt_bias,
    const aclIntArray *cu_seqlens,
    double scale,
    double lower_bound,
    double rms_norm_eps,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnMegaRecurrentKda, DFX_IN(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight, output_norm_weight, output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k, replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, scale, lower_bound, rms_norm_eps),
                   DFX_OUT(out, conv_state, replay_u, replay_k, replay_decay));
    if(!workspaceSize || !executor || !out) return ACLNN_ERR_PARAM_NULLPTR;
    const aclTensor *required[]={hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight, output_norm_weight, output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k, replay_decay, a_log, dt_bias,out};
    const op::DataType types[]={op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT,op::DataType::DT_BF16};
    for(size_t i=0;i<sizeof(required)/sizeof(required[0]);++i) {
        if(!required[i]) return ACLNN_ERR_PARAM_NULLPTR;
        if(required[i]->GetDataType()!=types[i]) return ACLNN_ERR_PARAM_INVALID;
        const bool state=i>=9 && i<=13;
        if(!state && !IsContiguous(required[i])) return ACLNN_ERR_PARAM_INVALID;
    }
    if(out->GetViewShape()!=hidden_states->GetViewShape() || !std::isfinite(static_cast<float>(scale)) ||
       !std::isfinite(lower_bound) || lower_bound < -5 || lower_bound>0 ||
       !std::isfinite(static_cast<float>(rms_norm_eps)) || rms_norm_eps<std::numeric_limits<float>::min())
        return ACLNN_ERR_PARAM_INVALID;
    const uintptr_t outBegin=reinterpret_cast<uintptr_t>(out->GetData());
    const int64_t elements=out->Numel();
    if(!outBegin || elements<=0 || uint64_t(elements)>(UINTPTR_MAX-outBegin)/sizeof(uint16_t))
        return ACLNN_ERR_PARAM_INVALID;
    const uintptr_t outEnd=outBegin+uint64_t(elements)*sizeof(uint16_t);
    for(size_t i=0;i+1<sizeof(required)/sizeof(required[0]);++i) {
        if(!disjointOutput(required[i],outBegin,outEnd)) return ACLNN_ERR_PARAM_INVALID;
    }
    auto holder=CREATE_EXECUTOR();
    if(!holder.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    auto status=l0op::LaunchMegaRecurrentKda(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight, output_norm_weight, output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k, replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens,
        static_cast<float>(scale),static_cast<float>(lower_bound),static_cast<float>(rms_norm_eps),out,holder.get());
    if(status!=ACLNN_SUCCESS) return status;
    *workspaceSize=holder->GetWorkspaceSize();
    holder.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnMegaRecurrentKda(void *workspace,uint64_t workspaceSize,
                                              aclOpExecutor *executor,aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnMegaRecurrentKda);
    return CommonOpExecutorRun(workspace,workspaceSize,executor,stream);
}
