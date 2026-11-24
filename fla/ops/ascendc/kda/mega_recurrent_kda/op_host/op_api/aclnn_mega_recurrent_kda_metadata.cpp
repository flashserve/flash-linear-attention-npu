#include "aclnn_mega_recurrent_kda_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/platform.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(MegaRecurrentKdaMetadata);

aclnnStatus LaunchSequencePlan(const aclTensor *prefix, const aclTensor *plan, aclOpExecutor *executor)
{
    static internal::AicpuTaskSpace space("MegaRecurrentKdaMetadata");
    return ADD_TO_LAUNCHER_LIST_AICPU(MegaRecurrentKdaMetadata, OP_ATTR_NAMES({}), OP_INPUT(prefix), OP_OUTPUT(plan),
                                      OP_ATTR());
}
} // namespace l0op

extern "C" aclnnStatus aclnnMegaRecurrentKdaMetadataGetWorkspaceSize(const aclTensor *cuSeqlens, int64_t tokens,
                                                                     const aclTensor *plan, uint64_t *workspaceSize,
                                                                     aclOpExecutor **executor)
{
    if (!cuSeqlens || !plan || !workspaceSize || !executor) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const auto dtype = cuSeqlens->GetDataType();
    const auto soc = GetCurrentPlatformInfo().GetSocVersion();
    if ((soc != SocVersion::ASCEND950 && soc != SocVersion::ASCEND910B && soc != SocVersion::ASCEND910_93) || tokens < 1 || tokens > INT32_MAX ||
        cuSeqlens->GetViewShape().GetDimNum() != 1 || cuSeqlens->Numel() < 2 || cuSeqlens->Numel() - 1 > tokens ||
        cuSeqlens->GetViewStrides()[0] != 1 || (dtype != DataType::DT_INT32 && dtype != DataType::DT_INT64) ||
        plan->GetViewShape().GetDimNum() != 1 || plan->Numel() != 2 + (tokens + 15) / 16 ||
        plan->GetDataType() != DataType::DT_INT32 || plan->GetViewStrides()[0] != 1) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto holder = CREATE_EXECUTOR();
    if (!holder.get()) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto status = l0op::LaunchSequencePlan(cuSeqlens, plan, holder.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    *workspaceSize = holder->GetWorkspaceSize();
    holder.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnMegaRecurrentKdaMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
