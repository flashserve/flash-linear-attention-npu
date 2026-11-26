#include "aclnn_mega_recurrent_kda.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/platform.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include "../kda_workspace.h"
#include "../kda_metadata.h"
#include <initializer_list>
#include <iterator>
#include <limits>

using namespace op;
namespace {
namespace state = kda::geometry;

struct TensorRequirement {
    const aclTensor *tensor;
    op::DataType dtype;
    bool allowsSlotStride = false;
};

// All supported views are contiguous inside a physical slot. Check an output
// interval against the first slot it can intersect, including strided pools.
bool IsDisjointFromOutput(const aclTensor *input, uintptr_t outBegin, uintptr_t outEnd)
{
    const auto &shape = input->GetViewShape();
    if (!shape.GetDimNum() || shape.GetDim(0) <= 0 || input->Numel() <= 0) {
        return false;
    }
    const uint64_t slots = shape.GetDim(0);
    const uint64_t bytesPerElement = op::TypeSize(input->GetDataType());
    const uint64_t inner = static_cast<uint64_t>(input->Numel()) / slots;
    const auto &strides = input->GetViewStrides();
    if (!bytesPerElement || strides.size() != shape.GetDimNum() || strides[0] < 0 ||
        inner > UINTPTR_MAX / bytesPerElement || uint64_t(strides[0]) > UINTPTR_MAX / bytesPerElement) {
        return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(input->GetData());
    const uint64_t width = inner * bytesPerElement, step = uint64_t(strides[0]) * bytesPerElement;
    if (!begin || !width || width > UINTPTR_MAX - begin) {
        return false;
    }
    if (outEnd <= begin) {
        return true;
    }
    uint64_t slot = 0;
    if (outBegin >= begin + width) {
        if (slots == 1) {
            return true;
        }
        if (!step) {
            return false;
        }
        slot = (outBegin - begin - width) / step + 1;
    }
    return slot >= slots ||
           (slot <= (UINTPTR_MAX - begin) / std::max<uint64_t>(step, 1) && begin + slot * step >= outEnd);
}
bool ReadSlotStride(const aclTensor *tensor, std::initializer_list<int64_t> dims, int64_t &stride)
{
    const auto &view = tensor->GetViewShape();
    const auto &strides = tensor->GetViewStrides();
    if (view.GetDimNum() != dims.size() || strides.size() != dims.size()) {
        return false;
    }
    int64_t packed = 1;
    for (size_t j = dims.size(); j-- > 0;) {
        int64_t n = dims.begin()[j];
        if (n != view.GetDim(j) || n <= 0) {
            return false;
        }
        if (j) {
            if (strides[j] != packed) {
                return false;
            }
            if (packed > INT64_MAX / n) {
                return false;
            }
            packed *= n;
        }
    }
    stride = strides[0];
    return stride >= packed;
}

aclnnStatus ReadStateGeometry(const aclTensor *conv_state, const aclTensor *checkpoint_state, const aclTensor *replay_u,
                              const aclTensor *replay_k, const aclTensor *replay_decay, state::Values &geometry)
{
    if (conv_state->GetViewShape().GetDimNum() != 3 || replay_u->GetViewShape().GetDimNum() != 4) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t slots = conv_state->GetViewShape().GetDim(0), length = conv_state->GetViewShape().GetDim(1);
    const int64_t heads = replay_u->GetViewShape().GetDim(1), capacity = replay_u->GetViewShape().GetDim(2);
    if (heads <= 0 || heads > UINT32_MAX / (6 * 128) || slots <= 0 || length <= 0 || capacity <= 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    geometry[state::PoolSlots] = slots;
    geometry[state::Heads] = heads;
    geometry[state::ConvLength] = length;
    geometry[state::ReplayCapacity] = capacity;
    if (!ReadSlotStride(conv_state, {slots, length, 3 * heads * 128}, geometry[state::ConvStride]) ||
        !ReadSlotStride(checkpoint_state, {slots, heads, 128, 128}, geometry[state::CheckpointStride]) ||
        !ReadSlotStride(replay_u, {slots, heads, capacity, 128}, geometry[state::UStride]) ||
        !ReadSlotStride(replay_k, {slots, heads, capacity, 128}, geometry[state::KStride]) ||
        !ReadSlotStride(replay_decay, {slots, heads, capacity, 128}, geometry[state::DecayStride])) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus BuildSequencePlan(const aclTensor *hidden_states, const aclIntArray *state_indices,
                              const aclIntArray *conv_num_accepted_tokens, const aclIntArray *cu_seqlens,
                              state::Values &geometry, std::vector<int32_t> &indices, std::vector<int32_t> &accepted)
{
    const auto &hidden = hidden_states->GetViewShape();
    const int64_t rank = hidden.GetDimNum();
    if ((rank != 2 && rank != 3) || hidden.GetDim(rank - 1) <= 0 || !state_indices || !conv_num_accepted_tokens) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto batch = state_indices->Size();
    if (!batch || batch > UINT32_MAX || conv_num_accepted_tokens->Size() != batch ||
        (rank == 3 && (hidden.GetDim(0) != static_cast<int64_t>(batch) || cu_seqlens)) ||
        (rank == 2 && (!cu_seqlens || cu_seqlens->Size() != batch + 1))) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    indices.resize(batch);
    accepted.resize(batch);
    for (size_t i = 0; i < batch; ++i) {
        const int64_t id = (*state_indices)[i], count = (*conv_num_accepted_tokens)[i];
        if (id < 0 || id > INT32_MAX || count < 0 || count > INT32_MAX) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        indices[i] = static_cast<int32_t>(id);
        accepted[i] = static_cast<int32_t>(count);
    }
    const int64_t sequence = rank == 3 ? hidden.GetDim(1) : 0;
    try {
        const uint64_t tokens = rank == 3 ? kda::checkedProduct(batch, sequence) : hidden.GetDim(0);
        if (!tokens || tokens > UINT32_MAX) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        std::vector<int64_t> prefix;
        if (cu_seqlens) {
            prefix.reserve(batch + 1);
            for (size_t i = 0; i <= batch; ++i) {
                prefix.push_back((*cu_seqlens)[i]);
            }
        }
        kda::Shape shape{static_cast<int64_t>(batch),    sequence,
                         static_cast<int64_t>(tokens),   geometry[state::Heads],
                         geometry[state::PoolSlots],     geometry[state::ConvLength],
                         geometry[state::ReplayCapacity]};
        auto plan =
            kda::makeTiling(shape, prefix.empty() ? nullptr : prefix.data(), indices.data(), accepted.data(), 3);
        if (plan.bytes > INT64_MAX) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        geometry[state::WorkspaceBytes] = static_cast<int64_t>(plan.bytes);
        geometry[state::CrossesTile] = plan.kernel == kda::TemplateId::T02;
    } catch (const std::exception &) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
bool IsVector(const aclTensor *tensor, op::DataType dtype, int64_t length)
{
    return tensor && tensor->GetDataType() == dtype && tensor->GetViewShape().GetDimNum() == 1 &&
           tensor->Numel() == length && tensor->GetViewStrides()[0] == 1;
}

aclnnStatus BuildDevicePlan(const aclTensor *hidden, const aclTensor *ids, const aclTensor *counts,
                            const aclTensor *prefix, const aclTensor *plan, state::Values &geometry)
{
    const auto soc = GetCurrentPlatformInfo().GetSocVersion();
    if ((soc != SocVersion::ASCEND950 && soc != SocVersion::ASCEND910B && soc != SocVersion::ASCEND910_93)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto &dims = hidden->GetViewShape();
    const auto rank = dims.GetDimNum();
    if ((rank != 2 && rank != 3) || dims.GetDim(rank - 1) <= 0 || !ids) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t batch = ids->Numel();
    const bool packed = rank == 2;
    const int64_t seq = packed ? 0 : dims.GetDim(1);
    if (batch <= 0 || batch > INT32_MAX || !IsVector(ids, DataType::DT_INT32, batch) ||
        !IsVector(counts, DataType::DT_INT32, batch) ||
        (!packed && (dims.GetDim(0) != batch || seq <= 0 || seq > INT32_MAX / batch || prefix || plan))) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t tokens = packed ? dims.GetDim(0) : batch * seq;
    if (tokens < batch || tokens > INT32_MAX ||
        (packed &&
         ((!IsVector(prefix, DataType::DT_INT32, batch + 1) && !IsVector(prefix, DataType::DT_INT64, batch + 1)) ||
          !IsVector(plan, DataType::DT_INT32, 2 + (tokens + 15) / 16)))) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    kda::Shape shape{batch,
                     seq,
                     tokens,
                     geometry[state::Heads],
                     geometry[state::PoolSlots],
                     geometry[state::ConvLength],
                     geometry[state::ReplayCapacity]};
    try {
        const bool crosses = !packed && tokens > 64 && 64 % seq != 0;
        auto bytes = kda::workspaceFor(shape, crosses, 3).bytes;
        if (packed && tokens > 64) {
            bytes = std::max(bytes, kda::workspaceFor(shape, true, 3).bytes);
        }
        if (bytes > INT64_MAX) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        geometry[state::WorkspaceBytes] = bytes;
        geometry[state::CrossesTile] = packed ? 2 : crosses;
    } catch (const std::exception &) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

namespace l0op {
OP_TYPE_REGISTER(MegaRecurrentKda);

template <typename IndexInput>
aclnnStatus LaunchMegaRecurrentKda(const aclTensor *hidden_states, const aclTensor *qkv_projection_weight,
                                   const aclTensor *decay_projection_a_weight,
                                   const aclTensor *decay_projection_b_weight, const aclTensor *beta_projection_weight,
                                   const aclTensor *output_gate_projection_weight, const aclTensor *output_norm_weight,
                                   const aclTensor *output_projection_weight, const aclTensor *conv1d_weight,
                                   aclTensor *conv_state, const aclTensor *checkpoint_state, aclTensor *replay_u,
                                   aclTensor *replay_k, aclTensor *replay_decay, const IndexInput *state_indices,
                                   const IndexInput *conv_num_accepted_tokens, const aclTensor *a_log,
                                   const aclTensor *dt_bias, const IndexInput *cu_seqlens, const aclTensor *plan,
                                   float scale, float lower_bound, float rms_norm_eps, const aclTensor *out,
                                   aclOpExecutor *executor, const aclTensor *weightScale, int64_t mxAlg)
{
    state::Values geometry{};
    auto status = ReadStateGeometry(conv_state, checkpoint_state, replay_u, replay_k, replay_decay, geometry);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    const aclTensor *ids = nullptr, *counts = nullptr, *prefixTensor = nullptr;
    std::vector<int32_t> indices, accepted;
    if constexpr (std::is_same_v<IndexInput, aclIntArray>) {
        status = BuildSequencePlan(hidden_states, state_indices, conv_num_accepted_tokens, cu_seqlens, geometry,
                                   indices, accepted);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
        ids = executor->ConvertToTensor(indices.data(), indices.size(), op::DataType::DT_INT32);
        counts = executor->ConvertToTensor(accepted.data(), accepted.size(), op::DataType::DT_INT32);
        prefixTensor = cu_seqlens ? executor->ConvertToTensor(cu_seqlens, op::DataType::DT_INT64) : nullptr;
        if (!ids || !counts || (cu_seqlens && !prefixTensor)) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    } else {
        ids = state_indices;
        counts = conv_num_accepted_tokens;
        prefixTensor = cu_seqlens;
        status = BuildDevicePlan(hidden_states, ids, counts, prefixTensor, plan, geometry);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    const aclIntArray *stateInfo = executor->AllocIntArray(geometry.data(), geometry.size());
    if (!stateInfo) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ADD_TO_LAUNCHER_LIST_AICORE(
        MegaRecurrentKda,
        OP_INPUT(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                 beta_projection_weight, output_gate_projection_weight, output_norm_weight, output_projection_weight,
                 conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k, replay_decay, ids, counts, a_log,
                 dt_bias, prefixTensor, plan, weightScale),
        OP_OUTPUT(out, conv_state, replay_u, replay_k, replay_decay),
        OP_ATTR(scale, lower_bound, rms_norm_eps, stateInfo, mxAlg));
}
} // namespace l0op

template <typename IndexInput>
static aclnnStatus
GetWorkspaceSize(const aclTensor *hidden_states, const aclTensor *qkv_projection_weight,
                 const aclTensor *decay_projection_a_weight, const aclTensor *decay_projection_b_weight,
                 const aclTensor *beta_projection_weight, const aclTensor *output_gate_projection_weight,
                 const aclTensor *output_norm_weight, const aclTensor *output_projection_weight,
                 const aclTensor *conv1d_weight, aclTensor *conv_state, const aclTensor *checkpoint_state,
                 aclTensor *replay_u, aclTensor *replay_k, aclTensor *replay_decay, const IndexInput *state_indices,
                 const IndexInput *conv_num_accepted_tokens, const aclTensor *a_log, const aclTensor *dt_bias,
                 const IndexInput *cu_seqlens, const aclTensor *plan, double scale, double lower_bound,
                 double rms_norm_eps, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor,
                 const aclTensor *weightScale = nullptr, int64_t mxAlg = 0, bool int8Qkv = false)
{
    if (!workspaceSize || !executor || !out) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const TensorRequirement required[] = {
        {hidden_states, op::DataType::DT_BF16},
        {qkv_projection_weight, int8Qkv ? op::DataType::DT_INT8 :
                                (weightScale ? op::DataType::DT_FLOAT8_E4M3FN : op::DataType::DT_BF16)},
        {decay_projection_a_weight, op::DataType::DT_BF16},
        {decay_projection_b_weight, op::DataType::DT_BF16},
        {beta_projection_weight, op::DataType::DT_BF16},
        {output_gate_projection_weight, op::DataType::DT_BF16},
        {output_norm_weight, op::DataType::DT_FLOAT},
        {output_projection_weight, op::DataType::DT_BF16},
        {conv1d_weight, op::DataType::DT_BF16},
        {conv_state, op::DataType::DT_BF16, true},
        {checkpoint_state, op::DataType::DT_FLOAT, true},
        {replay_u, op::DataType::DT_FLOAT, true},
        {replay_k, op::DataType::DT_FLOAT16, true},
        {replay_decay, op::DataType::DT_FLOAT, true},
        {a_log, op::DataType::DT_FLOAT},
        {dt_bias, op::DataType::DT_FLOAT},
        {out, op::DataType::DT_BF16},
    };
    for (const auto &requirement : required) {
        if (!requirement.tensor) {
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (requirement.tensor->GetDataType() != requirement.dtype ||
            (!requirement.allowsSlotStride && !IsContiguous(requirement.tensor))) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (out->GetViewShape() != hidden_states->GetViewShape() || !std::isfinite(static_cast<float>(scale)) ||
        !std::isfinite(lower_bound) || lower_bound < -5 || lower_bound > 0 ||
        !std::isfinite(static_cast<float>(rms_norm_eps)) || rms_norm_eps < std::numeric_limits<float>::min()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const uintptr_t outBegin = reinterpret_cast<uintptr_t>(out->GetData());
    const int64_t elements = out->Numel();
    if (!outBegin || elements <= 0 || uint64_t(elements) > (UINTPTR_MAX - outBegin) / sizeof(uint16_t)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const uintptr_t outEnd = outBegin + uint64_t(elements) * sizeof(uint16_t);
    for (size_t i = 0; i + 1 < std::size(required); ++i) {
        if (!IsDisjointFromOutput(required[i].tensor, outBegin, outEnd)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if constexpr (std::is_same_v<IndexInput, aclTensor>) {
        for (const auto *tensor : {state_indices, conv_num_accepted_tokens, cu_seqlens, plan}) {
            if (tensor && !IsDisjointFromOutput(tensor, outBegin, outEnd)) return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const auto &inputShape = hidden_states->GetViewShape();
    const auto rank = inputShape.GetDimNum();
    if (rank != 2 && rank != 3) return ACLNN_ERR_PARAM_INVALID;
    const int64_t hidden = inputShape.GetDim(rank - 1);
    if (hidden <= 0 || hidden > INT32_MAX || (int8Qkv && hidden % 32) || (weightScale && !int8Qkv && hidden % 64)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (int8Qkv) {
        const auto soc = GetCurrentPlatformInfo().GetSocVersion();
        const auto &view = qkv_projection_weight->GetViewShape();
        const auto &storage = qkv_projection_weight->GetStorageShape();
        const auto &replay = replay_u->GetViewShape();
        if ((soc != SocVersion::ASCEND910B && soc != SocVersion::ASCEND910_93) ||
            replay.GetDimNum() != 4 || replay.GetDim(1) <= 0 || replay.GetDim(1) > INT32_MAX / 384) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t cols = replay.GetDim(1) * 384;
        if (view.GetDimNum() != 2 || view.GetDim(0) != hidden || view.GetDim(1) != cols ||
            storage.GetDimNum() != 4 || storage.GetDim(0) != cols / 32 || storage.GetDim(1) != hidden / 16 ||
            storage.GetDim(2) != 16 || storage.GetDim(3) != 32 ||
            qkv_projection_weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ ||
            !IsVector(weightScale, DataType::DT_BF16, cols) || weightScale->GetStorageFormat() != op::Format::FORMAT_ND ||
            !IsDisjointFromOutput(weightScale, outBegin, outEnd)) return ACLNN_ERR_PARAM_INVALID;
    } else if (weightScale) {
        const auto &view = qkv_projection_weight->GetViewShape();
        if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND950 || (mxAlg != 0 && mxAlg != 1) ||
            view.GetDimNum() != 3 || view.GetDim(0) != 1 || view.GetDim(1) != hidden ||
            replay_u->GetViewShape().GetDimNum() != 4 ||
            view.GetDim(2) / 384 != replay_u->GetViewShape().GetDim(1) || view.GetDim(2) % 384 ||
            qkv_projection_weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ ||
            weightScale->GetDataType() != op::DataType::DT_FLOAT8_E8M0 ||
            weightScale->GetStorageFormat() != op::Format::FORMAT_ND || !IsContiguous(weightScale) ||
            !IsDisjointFromOutput(weightScale, outBegin, outEnd)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto &storage = qkv_projection_weight->GetStorageShape();
        const auto &groups = weightScale->GetViewShape();
        const int64_t cols = view.GetDim(2);
        if (storage.GetDimNum() != 5 || storage.GetDim(0) != 1 || storage.GetDim(1) != cols / 32 ||
            storage.GetDim(2) != hidden / 16 || storage.GetDim(3) != 16 || storage.GetDim(4) != 32 ||
            groups.GetDimNum() != 4 || groups.GetDim(0) != 1 || groups.GetDim(1) != hidden / 64 ||
            groups.GetDim(2) != cols || groups.GetDim(3) != 2) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    auto holder = CREATE_EXECUTOR();
    if (!holder.get()) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto status = l0op::LaunchMegaRecurrentKda(
        hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
        beta_projection_weight, output_gate_projection_weight, output_norm_weight, output_projection_weight,
        conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k, replay_decay, state_indices,
        conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, plan, static_cast<float>(scale),
        static_cast<float>(lower_bound), static_cast<float>(rms_norm_eps), out, holder.get(), weightScale, mxAlg);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    *workspaceSize = holder->GetWorkspaceSize();
    holder.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnMegaRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMegaRecurrentKda);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnMegaRecurrentKdaGetWorkspaceSize(
    const aclTensor *hidden_states, const aclTensor *qkv_projection_weight, const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight, const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight, const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight, const aclTensor *conv1d_weight, aclTensor *conv_state,
    const aclTensor *checkpoint_state, aclTensor *replay_u, aclTensor *replay_k, aclTensor *replay_decay,
    const aclIntArray *state_indices, const aclIntArray *conv_num_accepted_tokens, const aclTensor *a_log,
    const aclTensor *dt_bias, const aclIntArray *cu_seqlens, double scale, double lower_bound, double rms_norm_eps,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMegaRecurrentKda,
                   DFX_IN(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                          beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                          output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                          replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, scale,
                          lower_bound, rms_norm_eps),
                   DFX_OUT(out, conv_state, replay_u, replay_k, replay_decay));
    return GetWorkspaceSize(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                            beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                            output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                            replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, nullptr,
                            scale, lower_bound, rms_norm_eps, out, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnMegaRecurrentKdaV2GetWorkspaceSize(
    const aclTensor *hidden_states, const aclTensor *qkv_projection_weight, const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight, const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight, const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight, const aclTensor *conv1d_weight, aclTensor *conv_state,
    const aclTensor *checkpoint_state, aclTensor *replay_u, aclTensor *replay_k, aclTensor *replay_decay,
    const aclTensor *state_indices, const aclTensor *conv_num_accepted_tokens, const aclTensor *a_log,
    const aclTensor *dt_bias, const aclTensor *cu_seqlens, const aclTensor *plan, double scale, double lower_bound,
    double rms_norm_eps, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMegaRecurrentKdaV2,
                   DFX_IN(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                          beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                          output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                          replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, plan,
                          scale, lower_bound, rms_norm_eps),
                   DFX_OUT(out, conv_state, replay_u, replay_k, replay_decay));
    return GetWorkspaceSize(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                            beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                            output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                            replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, plan,
                            scale, lower_bound, rms_norm_eps, out, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnMegaRecurrentKdaV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnMegaRecurrentKdaMxGetWorkspaceSize(
    const aclTensor *hidden_states, const aclTensor *qkv_projection_weight, const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight, const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight, const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight, const aclTensor *conv1d_weight, aclTensor *conv_state,
    const aclTensor *checkpoint_state, aclTensor *replay_u, aclTensor *replay_k, aclTensor *replay_decay,
    const aclTensor *state_indices, const aclTensor *conv_num_accepted_tokens, const aclTensor *a_log,
    const aclTensor *dt_bias, const aclTensor *cu_seqlens, const aclTensor *plan, const aclTensor *weightScale,
    int64_t mxAlg, double scale, double lower_bound,
    double rms_norm_eps, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (!weightScale) return ACLNN_ERR_PARAM_NULLPTR;
    L2_DFX_PHASE_1(aclnnMegaRecurrentKdaMx,
                   DFX_IN(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                          beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                          output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                          replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, plan, weightScale, mxAlg,
                          scale, lower_bound, rms_norm_eps),
                   DFX_OUT(out, conv_state, replay_u, replay_k, replay_decay));
    return GetWorkspaceSize(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                            beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                            output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                            replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, plan,
                            scale, lower_bound, rms_norm_eps, out, workspaceSize, executor, weightScale, mxAlg);
}

extern "C" aclnnStatus aclnnMegaRecurrentKdaMx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
extern "C" aclnnStatus aclnnMegaRecurrentKdaInt8GetWorkspaceSize(
    const aclTensor *hidden_states, const aclTensor *qkv_projection_weight, const aclTensor *decay_projection_a_weight,
    const aclTensor *decay_projection_b_weight, const aclTensor *beta_projection_weight,
    const aclTensor *output_gate_projection_weight, const aclTensor *output_norm_weight,
    const aclTensor *output_projection_weight, const aclTensor *conv1d_weight, aclTensor *conv_state,
    const aclTensor *checkpoint_state, aclTensor *replay_u, aclTensor *replay_k, aclTensor *replay_decay,
    const aclTensor *state_indices, const aclTensor *conv_num_accepted_tokens, const aclTensor *a_log,
    const aclTensor *dt_bias, const aclTensor *cu_seqlens, const aclTensor *plan, const aclTensor *weightScale,
    double scale, double lower_bound, double rms_norm_eps, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (!weightScale) return ACLNN_ERR_PARAM_NULLPTR;
    L2_DFX_PHASE_1(aclnnMegaRecurrentKdaInt8,
                   DFX_IN(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                          beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                          output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                          replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, plan, weightScale,
                          scale, lower_bound, rms_norm_eps),
                   DFX_OUT(out, conv_state, replay_u, replay_k, replay_decay));
    return GetWorkspaceSize(hidden_states, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
                            beta_projection_weight, output_gate_projection_weight, output_norm_weight,
                            output_projection_weight, conv1d_weight, conv_state, checkpoint_state, replay_u, replay_k,
                            replay_decay, state_indices, conv_num_accepted_tokens, a_log, dt_bias, cu_seqlens, plan,
                            scale, lower_bound, rms_norm_eps, out, workspaceSize, executor, weightScale, 0, true);
}

extern "C" aclnnStatus aclnnMegaRecurrentKdaInt8(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
