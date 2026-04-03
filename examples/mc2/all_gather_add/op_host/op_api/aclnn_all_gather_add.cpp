/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "aclnn_all_gather_add.h"
#include "securec.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" { 
#endif
static constexpr size_t DIMS_NUM = 2;
enum NnopbaseHcclServerType : uint32_t {
  NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
  NNOPBASE_HCCL_SERVER_TYPE_MTE,
  NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                                          int64_t rankSize, const aclTensor *cOut,
                                                          const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor);
extern aclnnStatus aclnnInnerAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

static bool CheckNotNull(const aclTensor *a, const aclTensor *b, 
                         const aclTensor *gatherout, const aclTensor *output)	
{	
    OP_CHECK_NULL(a, return false);	
    OP_CHECK_NULL(b, return false);	
    OP_CHECK_NULL(gatherout, return false);	
    OP_CHECK_NULL(output, return false);	
    return true;	
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16
};

static bool CheckDtypeValid(const aclTensor* a, const aclTensor* b, const aclTensor* gatherout, const aclTensor* output)	
{	
    OP_CHECK_DTYPE_NOT_SUPPORT(a, DTYPE_SUPPORT_LIST, return false);	
    OP_CHECK_DTYPE_NOT_SUPPORT(b, DTYPE_SUPPORT_LIST, return false);	
    OP_CHECK_DTYPE_NOT_SUPPORT(gatherout, DTYPE_SUPPORT_LIST, return false);	
    OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);	
    return true;
}

static bool CheckShape(const aclTensor *a, const aclTensor *b, const aclTensor *gatherOut, const aclTensor *output, int64_t rankSize)
{
    OP_CHECK_WRONG_DIMENSION(a, DIMS_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(b, DIMS_NUM, return false);

    auto aLen = a->GetViewShape().GetDim(0);
    auto bLen = b->GetViewShape().GetDim(0);
    auto gatherOutLen = gatherOut->GetViewShape().GetDim(0);
    auto outPutLen = output->GetViewShape().GetDim(0);

    OP_API_CHECK((aLen * rankSize != bLen || bLen != gatherOutLen || gatherOutLen != outPutLen), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
        "The operand's shape should satisfy: aDim0 * rankSize = bDim0 = cDim0 = gatherOutDim0.");
        return false;
    });

    return true;
}

static aclnnStatus CheckParams(const aclTensor *a, const aclTensor *b, const aclTensor *gatherout, const aclTensor *output, int64_t rankSize)
{
    CHECK_RET(CheckNotNull(a, b, gatherout, output), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(a, b, gatherout, output), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(a, b, output, gatherout, rankSize), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                              int64_t rankSize, const aclTensor *cOut,
                                              const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                              aclOpExecutor **executor) 
{
    auto retParam = CheckParams(a, b, gatherOutOut, cOut, rankSize);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    OP_LOGD("A is %s, B is %s.", a->ToString().GetString(), b->ToString().GetString());
    OP_LOGD("Output is %s, gatherOut is %s.", cOut->ToString().GetString(), gatherOutOut->ToString().GetString());

    aclnnStatus ret = aclnnInnerAllGatherAddGetWorkspaceSize(a, b, group, rankSize,
                                                             cOut, gatherOutOut, workspaceSize, executor);
    OP_LOGD("AllGatherAdd, aclnnInnerGetWorkspaceSize ret = %d.", ret);

    return ret;
}

aclnnStatus aclnnAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                              aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    }
    auto ret = aclnnInnerAllGatherAdd(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore,ret = %d.", ret);
        return ret;
    }
    return ret;
}

#ifdef __cplusplus
}
#endif