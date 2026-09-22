/**
 * 示例文件（形态 B）：fla/ops/ascendc/demo/example_scan/op_host/op_api/aclnn_example_scan_v2.h
 *
 * 注意事项：
 *   1. V1（aclnn_example_scan.h）一旦发布就不能改：函数名、形参顺序、类型、返回码全部冻结。
 *      需要新开关/新可选输出时新开 V2，声明写在本文件。
 *   2. 实现可以追加到 V1 的 aclnn_example_scan.cpp 尾部（此时 V1 的公开头完全不动），
 *      也可以单独建 aclnn_example_scan_v2.cpp；两种都合法，参考 chunk_kda_fwd 用的是前者。
 *   3. 新增的可选输出只加在 V2 形参尾部；def 不新增输出（见工程结构规范 §3.2/§5.2）。
 *   4. 新增开关的默认值必须等于 V1 的历史语义（示例：tail_mode=false、epsilon=1e-6）。
 *   5. 头文件顶部注释写清四件事：与 V1 的关系、V2 支持范围、每个新增开关的语义与默认值、
 *      不满足条件时的返回码。
 *   6. 场景选择（走 V1 还是 V2）在上层 Python 入口完成，调用方只面对一套 Python 签名。
 *   7. V1 与 V2 共用参数校验、输出指针语义、公开输出布局；同一输入下公开输出逐位一致。
 */

#ifndef OP_API_INC_ACLNN_EXAMPLE_SCAN_V2_H
#define OP_API_INC_ACLNN_EXAMPLE_SCAN_V2_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ExampleScan V2 入口：在同一个 executor 内按
 *   ExampleScan（主实现）-> ExampleScanTail（尾部归一化）
 * 组合两个已交付算子，并接受额外的归一化开关与尾部保存输出。
 *
 * 与 aclnnExampleScanGetWorkspaceSize 的关系：
 *   - V1 签名与 ABI 不变，仍使用自身实现；
 *   - V2 新增 `tailMode`、`tailScale` 两个开关与尾部保存输出 `tailOut`；
 *   - 场景选择由 fla_npu.ops.ascendc.example_scan 完成：满足组合场景时优先 V2，否则回落 V1。
 *
 * 支持范围：x 为 BF16、D=128、chunk_size=64、公开输出连续。不满足时返回
 * ACLNN_ERR_PARAM_INVALID（提示改用 V1）。
 *
 * 开关语义（默认值即 V1 行为）：
 *   tailMode  = false：不做尾部归一化（等价 V1）
 *   tailScale = 1.0：tailMode=true 时参与尾部缩放
 *
 * 新增可选输出：tailOut 传 nullptr 表示本次不导出；传与不传的计算结果逐位一致。
 */
__attribute__((visibility("default")))
aclnnStatus aclnnExampleScanV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *g, const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout, double scale,
    int64_t chunkSize, double epsilon, bool tailMode, double tailScale,
    const aclTensor *yOut, const aclTensor *stateOut, const aclTensor *xNormOut,
    const aclTensor *tailOut, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnExampleScanV2(void *workspace, uint64_t workspaceSize,
                              aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_EXAMPLE_SCAN_V2_H
