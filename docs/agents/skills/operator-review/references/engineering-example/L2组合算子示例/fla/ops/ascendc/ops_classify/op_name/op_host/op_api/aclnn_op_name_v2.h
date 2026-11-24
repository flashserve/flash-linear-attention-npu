/**
 * 示例文件（形态 B）：fla/ops/ascendc/ops_classify/op_name/op_host/op_api/aclnn_op_name_v2.h
 *
 * 注意事项：
 *   1. V1（aclnn_op_name.h）一旦发布就不能改：函数名、形参数量与顺序、类型冻结。
 *      需要新开关/新可选输出时新开 V2，声明写在本文件。
 *   2. 实现可以追加到 V1 的 aclnn_op_name.cpp 尾部（此时 V1 的公开头完全不动），
 *      也可以单独建 aclnn_op_name_v2.cpp；两种都合法，参考 chunk_kda_fwd 用的是前者。
 *   3. 新增的可选输出只加在 V2 形参尾部；def 不新增输出（见工程结构规范 §3.2/§5.2）。
 *   4. 新增开关的默认值必须等于 V1 的历史语义（示例：tail_mode=false、epsilon=1e-6）。
 *   5. 头文件顶部注释写清四件事：与 V1 的关系、V2 支持范围、每个新增开关的语义与默认值、
 *      不满足条件时的返回码。
 *   6. 场景选择（走 V1 还是 V2）在 Stable 适配层 `run_` 完成，调用方只面对一套 Python 签名；
 *      Stable 不做支持范围拦截，两条入口都不支持时仍由 aclnn 返回结构化错误。
 *   7. V1 与 V2 共用参数校验、输出指针语义、公开输出布局；同一输入下公开输出逐位一致。
 */

#ifndef OP_API_INC_ACLNN_OP_NAME_V2_H
#define OP_API_INC_ACLNN_OP_NAME_V2_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* OpName V2 入口：在同一个 executor 内按
 *   OpName（主实现）-> OpNameTail（尾部归一化）
 * 组合两个已交付算子，并接受额外的归一化开关与尾部保存输出。
 *
 * 与 aclnnOpNameGetWorkspaceSize 的关系：
 *   - V1 签名与 ABI 不变，仍使用自身实现；
 *   - V2 新增 `tailMode`、`tailScale` 两个开关与尾部保存输出 `tailOut`；
 *   - 场景选择由 Stable `run_op_name` 完成：只有 V1 能表达且支持默认语义时才选择 V1；V2 开关即使
 *     因 shape/layout 不满足 V2 支持范围，也必须选择 V2 让 aclnn 返回结构化参数错误，不能丢给 V1。
 *
 * 支持范围：x 为 BF16、D=128、chunk_size=64、公开输出连续。不满足时返回
 * ACLNN_ERR_PARAM_INVALID；Stable 只负责选择能表达完整参数的入口，不在下发前拦截。
 *
 * 开关语义（默认值即 V1 行为）：
 *   tailMode  = false：不做尾部归一化（等价 V1）
 *   tailScale = 1.0：tailMode=true 时参与尾部缩放
 *
 * 新增可选输出：tailOut 传 nullptr 表示本次不导出；传与不传的计算结果逐位一致。
 */
__attribute__((visibility("default")))
aclnnStatus aclnnOpNameV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *g, const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout, double scale,
    int64_t chunkSize, double epsilon, bool tailMode, double tailScale,
    const aclTensor *yOut, const aclTensor *stateOut, const aclTensor *xNormOut,
    const aclTensor *tailOut, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnOpNameV2(void *workspace, uint64_t workspaceSize,
                              aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_OP_NAME_V2_H
