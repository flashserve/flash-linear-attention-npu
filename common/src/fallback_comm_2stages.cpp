/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fallback_comm_2stages.cpp
 * \brief
 */

#include "fallback/fallback_comm_2stages.h"

#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "aclnn/aclnn_base.h"
#include "runtime/base.h"
#include "log/log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace std;
using namespace gert;
using namespace ge;

ge::graphStatus ExecuteOpLaunch(gert::OpExecuteLaunchContext *context) {
  auto params = reinterpret_cast<OpApiParams *>(context->GetOpApiParams());
  auto workspace_sizes = context->GetWorkspaceSizes();
  auto workspace_addrs = context->GetWorkspaceAddrs();
  OP_CHECK_IF((workspace_sizes->GetSize() == 0) || (workspace_addrs->GetSize() == 0), 
    OP_LOGE("aclnnfallback", "no workspace addrs"), return ge::GRAPH_FAILED);
  auto workspace_size = workspace_sizes->GetData()[0];
  auto workspace_addr = workspace_addrs->GetData()[0]->GetAddr();

  auto acl_stream = context->GetStream();
  auto opApiFunc = params->op_api_func;
  OP_CHECK_IF(opApiFunc == nullptr, 
    OP_LOGE("aclnnfallback", "opApiFunc nullptr"), return ge::GRAPH_FAILED);
  auto op_api_ret = opApiFunc(workspace_addr, workspace_size, params->executor, acl_stream);
  for (auto &av : params->converted_params) {
    if (av.deleter != nullptr) {
      av.deleter(av.pointer);
    }
  }
  params->converted_params.clear();
  if (op_api_ret != 0) {
    OP_LOGE("aclnnfallback", "call %s allocate workspace failed op_api_ret: %d", context->GetNodeName(), op_api_ret);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace fallback

#ifdef __cplusplus
}
#endif