#pragma once
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnMegaRecurrentKdaMetadataGetWorkspaceSize(const aclTensor *cuSeqlens, int64_t tokens,
                                                                    const aclTensor *plan, uint64_t *workspaceSize,
                                                                    aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnMegaRecurrentKdaMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream);
#ifdef __cplusplus
}
#endif
