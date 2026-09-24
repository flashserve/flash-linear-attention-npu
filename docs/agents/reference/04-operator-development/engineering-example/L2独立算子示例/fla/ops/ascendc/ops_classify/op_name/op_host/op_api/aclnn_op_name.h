/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_host/op_api/aclnn_op_name.h
 *
 * 注意事项：
 *   1. 这是公开 C 接口（L2）：函数名、形参顺序、形参类型、返回码一旦发布不得修改；
 *      新能力走 aclnn_<算子>_v2.h（见工程结构规范 §5.2）。
 *   2. 输出指针的"可选性"只在这里表达：stateOut/xNormOut 传 nullptr 表示本次不导出；
 *      def 里对应输出仍是 REQUIRED。
 *   3. 必选输出（yOut）不靠"形参能不能传空"区分，由 L2 显式校验非空并返回
 *      ACLNN_ERR_PARAM_NULLPTR。
 *   4. GetWorkspaceSize 与 Launch 必须成对出现，且都要有 __attribute__((visibility("default")))。
 *   5. 头文件顶部注释写清：与 V1 的关系（V2 时）、支持范围、每个可空输出的缺席语义。
 *   6. 不在本层解释 autograd 重计算策略；disable_recompute 之类只存在于 Python/legacy 包装层。
 *   7. 原地形态的形参约定：被写回的 state 用**两个槽位**表达——输入 `initialStateRef` 与输出
 *      `finalState`；`inplaceFinalState=true` 时调用方把同一张张量传给两个槽位，false 时 finalState 指向
 *      scratch。头文件注释要写明该开关的默认值与含义（参考 aclnn_recurrent_kda.h 的 inplaceFinalState）。
 */

#ifndef OP_API_INC_ACLNN_OP_NAME_H
#define OP_API_INC_ACLNN_OP_NAME_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 可选输出语义：
 *   stateOut/xNormOut 为 nullptr 时不导出该结果；两者要么同时给出（save 档），
 *   要么同时为空（none 档），其它组合返回 ACLNN_ERR_PARAM_INVALID。
 *   两种档位下 y 的计算结果逐位一致，只有是否落公开 GM 的区别。 */
__attribute__((visibility("default")))
aclnnStatus aclnnOpNameGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *stateOut,
    const aclTensor *xNormOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnOpName(void *workspace, uint64_t workspaceSize,
                            aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_OP_NAME_H
