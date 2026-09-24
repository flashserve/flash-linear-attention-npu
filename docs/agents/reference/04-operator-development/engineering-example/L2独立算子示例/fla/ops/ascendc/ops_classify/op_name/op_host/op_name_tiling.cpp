/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_host/op_name_tiling.cpp
 *
 * 注意事项：
 *   1. 入口只做三件事：校验 -> 计算调度与 TilingData -> SetTilingKey/SetBlockDim/workspace。
 *      校验结论必须与算子 README「已知限制」逐条对应；报错打实际值。
 *   2. TilingKey 必须用 GET_TPL_TILING_KEY(...) 生成，实参顺序与 op_kernel/<算子>_tiling_key.h
 *      的 ASCENDC_TPL_ARGS_DECL 声明顺序一致；禁止手写常量或按平台编码。
 *   3. host 侧档位枚举与 kernel 侧 TPL token 是两套名字、同一组数值：用 static_assert 在编译期钉住
 *      （示例见下），不要靠注释对齐。
 *   4. 平台判定用 platform.GetCurNpuArch()（A5 = NpuArch::DAV_3510）；平台 tile 常量从
 *      op_host/op_tiling/arch22/、op_host/op_tiling/arch35/ 下的 <算子>_tiling_impl.h 取，
 *      kernel 侧同名常量必须与之一致。
 *   5. workspace 必须计入 platform.GetLibApiWorkSpaceSize()；总大小按变量算，不写死数字。
 *   6. 本层不判断"输出指针是否为空"：L2 已经把档位算成 output_mode 属性传进来。
 *   7. 校验失败必须 OP_LOGE + return ge::GRAPH_FAILED，不允许带着默认值继续。
 *   8. SaveToBuffer 与 SetDataSize 必须成对；漏 SetDataSize 会让设备侧读到 0 字节 tiling。
 *   9. IMPL_OP_OPTILING 的 op 名要与 *_def.cpp 的 OP_ADD 一致，TilingParse 必须注册。
 */

#include "op_name_tiling.h"

#include <cstdint>
#include <cstring>
#include <limits>

#include "../op_kernel/op_name_tiling_key.h"
#include "op_tiling/arch22/op_name_tiling_impl.h"
#include "op_tiling/arch35/op_name_tiling_impl.h"
#include "op_name_tiling_processor.h"
#include "platform/soc_spec.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t OP_NAME_SUPPORTED_DIM = 128;
constexpr uint32_t OP_NAME_MIX_BATCH_MODE = 1;

// host 档位枚举（op_name_output_mask.h）与 kernel 侧 TPL token（op_name_tiling_key.h）
// 必须是同一组数值，否则 L2 算出的档位与 kernel 实例不匹配。
static_assert(OP_NAME_OUTPUT_MODE_NONE == OpNameNs::OP_NAME_TPL_OUTPUT_NONE,
              "host 与 kernel 的 none 档位取值必须一致。");
static_assert(OP_NAME_OUTPUT_MODE_SAVE == OpNameNs::OP_NAME_TPL_OUTPUT_SAVE,
              "host 与 kernel 的 save 档位取值必须一致。");

struct OpNameShapeInfo {
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t headNum = 0;
    int64_t dim = 0;
    bool seqMajor = false;
    bool varLen = false;
};

bool ParseLayout(const char *layout, bool &seqMajor)
{
    if (layout == nullptr || std::strcmp(layout, "BSND") == 0) {
        seqMajor = true;
        return true;
    }
    if (std::strcmp(layout, "BNSD") == 0) {
        seqMajor = false;
        return true;
    }
    // TND/NTD：打包 token。只解释输入，输出布局固定（见 docs/api.md）。
    if (std::strcmp(layout, "TND") == 0 || std::strcmp(layout, "NTD") == 0) {
        seqMajor = true;
        return true;
    }
    return false;
}

bool ResolveShape(gert::TilingContext *context, OpNameShapeInfo &info)
{
    const auto xShape = context->GetInputShape(OP_NAME_INPUT_X);
    if (xShape == nullptr) {
        return false;
    }
    // BSND=[B,T,H,D]、BNSD=[B,H,T,D]、变长=[totalTokens,H,D]；
    // 变长标记与 seqNum 由 cu_seqlens/chunk_indices 推导。
    // ...
    info.dim = xShape->GetStorageShape().GetDim(3);
    return true;
}

bool CheckShapeAndAttrs(gert::TilingContext *context, const OpNameShapeInfo &info,
                        float &scale, float &epsilon, uint32_t &chunkSize, uint32_t &outputMode)
{
    const auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "属性表为空。");
        return false;
    }
    const float *scalePtr = attrs->GetAttrPointer<float>(OP_NAME_ATTR_SCALE);
    const int64_t *chunkSizePtr = attrs->GetAttrPointer<int64_t>(OP_NAME_ATTR_CHUNK_SIZE);
    const float *epsilonPtr = attrs->GetAttrPointer<float>(OP_NAME_ATTR_EPSILON);
    const int64_t *outputModePtr = attrs->GetAttrPointer<int64_t>(OP_NAME_ATTR_OUTPUT_MODE);
    if (scalePtr == nullptr || chunkSizePtr == nullptr || epsilonPtr == nullptr ||
        outputModePtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "缺少必需属性 scale/chunk_size/epsilon/output_mode。");
        return false;
    }
    if (info.dim != OP_NAME_SUPPORTED_DIM) {
        OP_LOGE(context->GetNodeName(), "D 只支持 %ld，当前 Ddim=%ld。",
                static_cast<long>(OP_NAME_SUPPORTED_DIM), static_cast<long>(info.dim));
        return false;
    }
    if (*chunkSizePtr != 64 && *chunkSizePtr != 128) {
        OP_LOGE(context->GetNodeName(), "chunk_size 只支持 64/128，当前 chunk_size=%ld。",
                static_cast<long>(*chunkSizePtr));
        return false;
    }
    if (*epsilonPtr <= 0.0F || *scalePtr <= 0.0F) {
        OP_LOGE(context->GetNodeName(), "epsilon/scale 必须为正数，当前 epsilon=%f scale=%f。",
                *epsilonPtr, *scalePtr);
        return false;
    }
    if (*outputModePtr != OP_NAME_OUTPUT_MODE_NONE &&
        *outputModePtr != OP_NAME_OUTPUT_MODE_SAVE) {
        OP_LOGE(context->GetNodeName(),
                "output_mode 只支持 0（none）/1（save），当前 output_mode=%ld。",
                static_cast<long>(*outputModePtr));
        return false;
    }
    scale = *scalePtr;
    epsilon = *epsilonPtr;
    chunkSize = static_cast<uint32_t>(*chunkSizePtr);
    outputMode = static_cast<uint32_t>(*outputModePtr);
    // 变长元数据：cu_seqlens 首元素为 0、单调不减、末元素等于总 token 数；
    // chunk_indices 必须与 cu_seqlens 成对出现（否则报 ACLNN_ERR_PARAM_INVALID 语义的错）。
    // ...
    return true;
}

bool DtypeToTemplateToken(ge::DataType dtype, uint64_t &token)
{
    switch (dtype) {
        case ge::DT_BF16:
            token = OpNameNs::OP_NAME_TPL_BF16;
            return true;
        case ge::DT_FLOAT16:
            token = OpNameNs::OP_NAME_TPL_FP16;
            return true;
        case ge::DT_FLOAT:
            token = OpNameNs::OP_NAME_TPL_FP32;
            return true;
        default:
            return false;
    }
}

} // namespace

ge::graphStatus Tiling4OpName(gert::TilingContext *context)
{
    const auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "属性表为空。");
        return ge::GRAPH_FAILED;
    }
    const char *layout = attrs->GetAttrPointer<char>(OP_NAME_ATTR_LAYOUT);

    OpNameShapeInfo shape;
    if (!ParseLayout(layout, shape.seqMajor) || !ResolveShape(context, shape)) {
        OP_LOGE(context->GetNodeName(), "layout=%s 或输入 shape 无法解析。",
                layout == nullptr ? "nullptr" : layout);
        return ge::GRAPH_FAILED;
    }

    float scale = 1.0F;
    float epsilon = 1.0e-6F;
    uint32_t chunkSize = 64;
    uint32_t outputMode = OP_NAME_OUTPUT_MODE_NONE;
    if (!CheckShapeAndAttrs(context, shape, scale, epsilon, chunkSize, outputMode)) {
        return ge::GRAPH_FAILED;
    }

    const auto xDesc = context->GetInputDesc(OP_NAME_INPUT_X);
    const auto gDesc = context->GetInputDesc(OP_NAME_INPUT_G);
    const auto stateDesc = context->GetOptionalInputDesc(OP_NAME_INPUT_INITIAL_STATE);
    if (xDesc == nullptr || gDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "输入 x/g 的描述符为空。");
        return ge::GRAPH_FAILED;
    }
    uint64_t xToken = 0;
    uint64_t gToken = 0;
    if (!DtypeToTemplateToken(xDesc->GetDataType(), xToken) ||
        !DtypeToTemplateToken(gDesc->GetDataType(), gToken)) {
        OP_LOGE(context->GetNodeName(), "输入 dtype 不在支持范围内。");
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t coreNum = platform.GetCoreNumAic();
    const bool isAscend950 = platform.GetCurNpuArch() == NpuArch::DAV_3510;
    const uint32_t archTileT =
        isAscend950 ? OP_NAME_ARCH35_TILE_T : OP_NAME_ARCH22_TILE_T;

    OpNameScheduleContext scheduleContext;
    scheduleContext.batch = static_cast<uint64_t>(shape.batch);
    scheduleContext.headNum = static_cast<uint64_t>(shape.headNum);
    scheduleContext.chunksPerSequence =
        static_cast<uint64_t>((shape.seqLen + chunkSize - 1) / chunkSize);
    scheduleContext.dim = static_cast<uint64_t>(shape.dim);
    scheduleContext.chunkSize = chunkSize;
    scheduleContext.coreNum = coreNum;
    scheduleContext.libApiWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
    scheduleContext.outputMode = outputMode;
    OpNameSchedule schedule;
    if (!OpNameTilingProcessor(scheduleContext).Process(schedule)) {
        OP_LOGE(context->GetNodeName(),
                "无法生成合法调度：headNum=%lu chunksPerSequence=%lu coreNum=%lu。",
                static_cast<unsigned long>(scheduleContext.headNum),
                static_cast<unsigned long>(scheduleContext.chunksPerSequence),
                static_cast<unsigned long>(scheduleContext.coreNum));
        return ge::GRAPH_FAILED;
    }

    // 模板参数顺序必须与 op_kernel/op_name_tiling_key.h 的 ASCENDC_TPL_ARGS_DECL 一致。
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        xToken, gToken, OpNameNs::OP_NAME_TPL_NORM_L2,
        static_cast<uint64_t>(stateDesc != nullptr), static_cast<uint64_t>(outputMode));

    OpNameTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(scheduleContext.batch));
    tiling.set_seqLen(static_cast<uint32_t>(shape.seqLen));
    tiling.set_headNum(static_cast<uint32_t>(scheduleContext.headNum));
    tiling.set_dim(static_cast<uint32_t>(shape.dim));
    tiling.set_chunkSize(chunkSize);
    tiling.set_chunksPerSequence(static_cast<uint32_t>(scheduleContext.chunksPerSequence));
    tiling.set_usedCoreNum(schedule.usedCoreNum);
    tiling.set_headsPerCore(schedule.headsPerCore);
    tiling.set_outputMode(outputMode);
    tiling.set_scale(scale);
    tiling.set_epsilon(epsilon);
    tiling.set_isVarLen(shape.varLen);
    tiling.set_hasInitialState(stateDesc != nullptr);

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(schedule.usedCoreNum);
    if (context->SetScheduleMode(OP_NAME_MIX_BATCH_MODE) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "设置 MIX AIC/AIV batch 调度模式失败。");
        return ge::GRAPH_FAILED;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr || schedule.workspaceBytes > std::numeric_limits<size_t>::max()) {
        OP_LOGE(context->GetNodeName(), "workspace 大小超出平台 size_t 范围。");
        return ge::GRAPH_FAILED;
    }
    // 平台预留（LibApi）已计入 workspaceBytes；kernel 侧只从 GetUserWorkspace 取用户区，不重复预留。
    workspace[0] = static_cast<size_t>(schedule.workspaceBytes);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    OP_LOGI(context->GetNodeName(),
            "OpName layout=%s tilingKey=%lu usedCoreNum=%u headsPerCore=%u outputMode=%u "
            "tileT=%u workspace=%lu is950=%d",
            layout == nullptr ? "nullptr" : layout, static_cast<unsigned long>(tilingKey),
            schedule.usedCoreNum, schedule.headsPerCore, outputMode, archTileT,
            static_cast<unsigned long>(schedule.workspaceBytes), static_cast<int>(isAscend950));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForOpName(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(OpName)
    .Tiling(Tiling4OpName)
    .TilingParse<OpNameCompileInfo>(TilingPrepareForOpName);

} // namespace optiling
