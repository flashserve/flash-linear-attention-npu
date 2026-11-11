#ifndef GDN_BWD_FINALIZE_A2_STRUCT_H
#define GDN_BWD_FINALIZE_A2_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

#define TPL_BF16 10
#define TPL_FP32 30


// Runtime layout codes shared by this operator's host and kernels.
namespace FinalizeLayout {
enum Layout : int64_t {
    BNSD = 0,
    BSND = 1,
    NTD = 2,
    TND = 3,
};
} // namespace FinalizeLayout

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
    int64_t layout;
};

} // namespace GDN

#endif
