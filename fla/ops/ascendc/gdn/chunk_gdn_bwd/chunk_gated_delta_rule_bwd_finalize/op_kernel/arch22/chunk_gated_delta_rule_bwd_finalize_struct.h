#ifndef GDN_BWD_FINALIZE_A2_STRUCT_H
#define GDN_BWD_FINALIZE_A2_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

#define TPL_BF16 10
#define TPL_FP32 30

#ifndef TORCH_MODE
// arch22 私有模板：保留原五字段顺序，末位追加 MHA owner 编译开关。
// false 的 16 项先注册，true 的 16 项随后；非 220 和 Host 继续使用原五字段声明。
ASCENDC_TPL_ARGS_DECL(ChunkGatedDeltaRuleBwdFinalize,
    ASCENDC_TPL_DTYPE_DECL(D_T_Q, TPL_BF16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, TPL_BF16, TPL_FP32),
    ASCENDC_TPL_BOOL_DECL(USE_QK_L2NORM, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_BETA_SIGMOID, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1),
    ASCENDC_TPL_BOOL_DECL(ENABLE_MHA_OWNER, 0, 1),
);

#define GDN_A2_TPL_SEL_ONE(Q, G, NORM, SIGMOID, EXP2, MHA_OWNER) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DTYPE_SEL(D_T_Q, Q), \
        ASCENDC_TPL_DTYPE_SEL(D_T_G, G), \
        ASCENDC_TPL_BOOL_SEL(USE_QK_L2NORM, NORM), \
        ASCENDC_TPL_BOOL_SEL(USE_BETA_SIGMOID, SIGMOID), \
        ASCENDC_TPL_BOOL_SEL(USE_EXP2, EXP2), \
        ASCENDC_TPL_BOOL_SEL(ENABLE_MHA_OWNER, MHA_OWNER), \
    )

#define GDN_A2_TPL_SEL_FOR_DTYPE(Q, G, MHA_OWNER) \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 0, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 0, 1, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 1, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 1, 1, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 0, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 0, 1, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 1, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 1, 1, MHA_OWNER)

ASCENDC_TPL_SEL(
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_BF16, 0),
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_FP32, 0),
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_BF16, 1),
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_FP32, 1),
);

#undef GDN_A2_TPL_SEL_FOR_DTYPE
#undef GDN_A2_TPL_SEL_ONE
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
