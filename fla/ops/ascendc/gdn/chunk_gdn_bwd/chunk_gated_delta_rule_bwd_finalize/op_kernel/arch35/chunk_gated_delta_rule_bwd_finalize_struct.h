#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

#define TPL_BF16 10
#define TPL_FP32 30

#ifndef TORCH_MODE
// Stage 0 主张量只支持 BF16，g/beta 使用相同的 BF16 或 FP32 dtype。
// backward 开关进入 TilingKey，使无效输入搬运和 VF 指令在编译期裁掉。
ASCENDC_TPL_ARGS_DECL(ChunkGatedDeltaRuleBwdFinalize,
    ASCENDC_TPL_DTYPE_DECL(D_T_Q, TPL_BF16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, TPL_BF16, TPL_FP32),
    ASCENDC_TPL_BOOL_DECL(USE_QK_L2NORM, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_BETA_SIGMOID, 0, 1),
    ASCENDC_TPL_BOOL_DECL(ALLOW_NEG_EIGVAL, 0, 1),
);

#define TPL_SEL_ONE(Q, G, USE_QK_L2NORM_VALUE, USE_BETA_SIGMOID_VALUE, ALLOW_NEG_EIGVAL_VALUE) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DTYPE_SEL(D_T_Q, Q), \
        ASCENDC_TPL_DTYPE_SEL(D_T_G, G), \
        ASCENDC_TPL_BOOL_SEL(USE_QK_L2NORM, USE_QK_L2NORM_VALUE), \
        ASCENDC_TPL_BOOL_SEL(USE_BETA_SIGMOID, USE_BETA_SIGMOID_VALUE), \
        ASCENDC_TPL_BOOL_SEL(ALLOW_NEG_EIGVAL, ALLOW_NEG_EIGVAL_VALUE), \
    )

#define TPL_SEL_FOR_DTYPE(Q, G) \
    TPL_SEL_ONE(Q, G, 0, 0, 0), \
    TPL_SEL_ONE(Q, G, 0, 1, 0), \
    TPL_SEL_ONE(Q, G, 0, 1, 1), \
    TPL_SEL_ONE(Q, G, 1, 0, 0), \
    TPL_SEL_ONE(Q, G, 1, 1, 0), \
    TPL_SEL_ONE(Q, G, 1, 1, 1)

ASCENDC_TPL_SEL(
    // g/beta 共用 dtype 模板参数；allow_neg_eigval 仅在 beta sigmoid 开启时可达，共 12 个实例。
    TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_BF16),
    TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_FP32),
);

#undef TPL_SEL_FOR_DTYPE
#undef TPL_SEL_ONE
#endif

struct ChunkGatedDeltaRuleBwdFinalizeTilingData {
    // 输入逻辑 shape。K、V、chunkSize 分别从 q、v、A 的 shape 推导；
    // 即使当前实现只支持 K=128、V=128、chunkSize=64，也必须作为运行时 tiling 数据传入。
    int64_t B;
    int64_t HK;
    int64_t HV;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkSize;

    // chunk 调度信息。逻辑任务是 (chunk, headGroup)，不按行拆分 head。
    // isVariable 在运行时选择定长或变长 offset 计算，不进入 TilingKey。
    int64_t chunkNumForT;
    int64_t totalChunkNum;
    int64_t taskNum;
    int64_t seqNum;
    int64_t isVariable;
    int64_t stateVFirst;

    // taskGroupSize 是不超过 4 的 GVA 整数倍，且选择时保证无尾组；
    // headGroupNum 是每个 chunk 的 HV 方向逻辑任务数。
    // 两个 AIV 按任务序号交错承包完整 HV：AIV0 处理偶数任务，AIV1 处理奇数任务。
    int64_t headRatio;
    int64_t taskGroupSize;
    int64_t headGroupNum;

    float scale;
};

} // namespace GDN

#endif
