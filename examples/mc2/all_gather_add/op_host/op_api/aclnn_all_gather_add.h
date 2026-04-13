/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OP_API_INC_ALL_GATHER_ADD
#define OP_API_INC_ALL_GATHER_ADD

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现allGather + add 融合计算
 * @brief aclnnAllGatherAdd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] a: add左操作数，数据类型支持：float16。
 * @param [in] b: add右操作数，数据类型支持：float16。
 * @param [in] group: 标识列组的字符串。
 * @param [in] rankSize: 标识通信域内参与AllGather通信的npu卡数。
 * @param [out] cOut: 计算+通信的结果，数据类型：同输入。
 * @param [out] gatherOut: 仅gather通信操作的结果，数据类型：同输入。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                                        int64_t rankSize, const aclTensor *cOut, const aclTensor *gatherOutOut,
                                                        uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnAllGatherAdd的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAbsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ALL_GATHER_ADD