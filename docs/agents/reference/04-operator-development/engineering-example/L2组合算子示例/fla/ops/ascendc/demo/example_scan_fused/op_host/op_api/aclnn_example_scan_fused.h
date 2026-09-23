/**
 * 示例文件：.../example_scan_fused/op_host/op_api/aclnn_example_scan_fused.h
 *
 * 注意事项：
 *   1. 形态 C 的公开接口只有这一份 aclnn（没有 def 生成的接口），它是本算子唯一的对外契约。
 *   2. 可选输出仍然只在 L2 用可空描述符表达；本算子没有 def，也就不存在"def 里写 OPTIONAL"的问题，
 *      但语义与形态 A 保持一致：nullptr = 本次不导出。
 *   3. 头文件要写清组合关系、支持范围、与回落入口的关系（哪条路径走本入口，其余去哪）。
 *   4. 可见性宏跟随相邻算子：本仓既有 `__attribute__((visibility("default")))`（新增算子常用）
 *      也有 `ACLNN_API`（部分组合入口沿用）；照抄同目录/同类算子的写法，不要混用同一份文件两种写法。
 *   5. 公开形参顺序一旦交付不能改；需要新能力时新开 V2（见形态 B 的示例）。
 */

#ifndef OP_API_INC_ACLNN_EXAMPLE_SCAN_FUSED_H
#define OP_API_INC_ACLNN_EXAMPLE_SCAN_FUSED_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ExampleScanFused：在同一 executor 内组合 ExampleScan 与 ExampleScanTail。
 *
 * 组合关系与支持范围见算子 README；不满足支持范围时返回 ACLNN_ERR_PARAM_INVALID，
 * 并提示改用单算子入口 example_scan。
 *
 * 可选输出：state/xNorm 必须同时给出或同时为空；tailOut 独立可选（nullptr = 不导出）。
 */
ACLNN_API aclnnStatus aclnnExampleScanFusedGetWorkspaceSize(
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
    const aclTensor *tailOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnExampleScanFused(void *workspace, uint64_t workspaceSize,
                                           aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_EXAMPLE_SCAN_FUSED_H
