/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "aclnn_kda_layout_swap12.h"
#include "kda_layout_swap12.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

bool HasSwap12Shape(const aclTensor *x, const aclTensor *y)
{
    const size_t rank = Rank(x);
    if (rank != Rank(y) || rank < 3) {
        return false;
    }
    if (rank == 3) {
        return Dim(y, 0) == Dim(x, 1) && Dim(y, 1) == Dim(x, 0) && Dim(y, 2) == Dim(x, 2);
    }
    if (Dim(y, 0) != Dim(x, 0) || Dim(y, 1) != Dim(x, 2) || Dim(y, 2) != Dim(x, 1)) {
        return false;
    }
    for (size_t index = 3; index < rank; ++index) {
        if (Dim(y, index) != Dim(x, index)) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *y)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
    CHECK_COND(Rank(x) >= 3, ACLNN_ERR_PARAM_INVALID, "x rank must be at least 3.");
    CHECK_COND(x->GetDataType() == DataType::DT_FLOAT || x->GetDataType() == DataType::DT_FLOAT16 ||
                   x->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "x must be float32, float16 or bfloat16.");
    for (size_t index = 0; index < Rank(x); ++index) {
        CHECK_COND(Dim(x, index) > 0, ACLNN_ERR_PARAM_INVALID, "all x dimensions must be positive.");
    }
    CHECK_COND(x->GetDataType() == y->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "x and y must have the same dtype.");
    CHECK_COND(HasSwap12Shape(x, y), ACLNN_ERR_PARAM_INVALID,
               "y shape must equal x with logical dimensions 1 and 2 swapped; rank3 swaps dimensions 0 and 1.");
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnKdaLayoutSwap12GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *dependencyOptional,
    const aclTensor *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnKdaLayoutSwap12, DFX_IN(x, dependencyOptional), DFX_OUT(y));
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    CHECK_RET(CheckParams(x, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::KdaLayoutSwap12(xContiguous, dependencyOptional, y, uniqueExecutor.get());
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKdaLayoutSwap12(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKdaLayoutSwap12);
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
