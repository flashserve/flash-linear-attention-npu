// -----------------------------------------------------------------------------------------------------------
// chunk_kda_fwd_fused 算子 tiling(chunk32 大融合,issue #829)
// 输入按 [H,T,K] 头连续布局,dense batch 由 aclnn 拍平成 [B*H,T,K]。
// K=V=128,q/k/v 使用 bf16;尾块(T%128≠0)由 kernel 侧按 seqLen 钳制有效行;
// blockDim = min(B*HV, aicCoreNum),kernel 内按任务(task = batch*HV + value_head)
// 静态步进循环;tiling key 恒为 0。
// -----------------------------------------------------------------------------------------------------------
#include "chunk_kda_fwd_fused_tiling.h"

#include <cstdint>
#include <cstring>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_G_IDX = 3;             // g:bf16(组合 0)或 fp32(组合 1)
constexpr size_t INPUT_BETA_IDX = 4;          // beta:同上
constexpr size_t INPUT_A_LOG_IDX = 5;          // a_log fp32 [HV](真实 HV,不随 batch)
constexpr size_t INPUT_INITIAL_STATE_IDX = 7;   // optional 输入:initial_state fp32 [B*HV,K,V](batch 拍平)
constexpr size_t OUTPUT_O_IDX = 0;
constexpr size_t OUTPUT_FINAL_STATE_IDX = 1;    // optional 输出:final_state fp32 [H,K,V]
constexpr size_t ATTR_SCALE_IDX = 0;
constexpr size_t ATTR_LOWER_BOUND_IDX = 1;
constexpr size_t ATTR_LAYOUT_IDX = 2;
constexpr size_t ATTR_CU_SEQLENS_IDX = 3;   // OPTIONAL ListInt:非空即 varlen 单启动
constexpr size_t ATTR_EPSILON_IDX = 4;      // OPTIONAL Float(默认 1e-6):q/k L2 归一化 epsilon
constexpr uint64_t KDA_FD32_MAX_VARLEN_SEQ = 1024;
constexpr uint64_t KDA_FD32_CHUNK_TOKENS = 128;  // kernel 每组处理 128 token(4 个 32 行 chunk)
constexpr uint64_t KDA_FD32_HEAD_DIM = 128;

uint64_t Fp32BitsToBf16Pattern(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint64_t>(bits >> 16);   // host 契约:截断 bf16
}

uint64_t Fp32Bits(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint64_t>(bits);
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwdFused(gert::TilingContext *context)
{
    const auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    const auto attrs = context->GetAttrs();
    if (qDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float scale = static_cast<float>(*attrs->GetAttrPointer<double>(ATTR_SCALE_IDX));
    const float lowerBound = *attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_IDX);
    // OPTIONAL 带默认值的 attr 由框架补全(同 lower_bound),无需按 GetAttrNum 截住
    const float epsilon = *attrs->GetAttrPointer<float>(ATTR_EPSILON_IDX);

    // cu_seqlens attr 非空即 varlen:packed token-major [T_total, H/HV, *],
    // 任务 = seq*HV + value_head,序列边界打包进 tiling cuData(kernel 从 tiling GM 直读)。
    // 注意:dense 侧 attr 可能整个缺位(OP_ATTR 传 nullptr),GetListInt 越界读的是
    // 野指针而非 nullptr——必须先按 GetAttrNum 截住;"存在但空"再按长度兜底。
    const auto cuVec = (attrs->GetAttrNum() > ATTR_CU_SEQLENS_IDX)
                           ? attrs->GetListInt(ATTR_CU_SEQLENS_IDX)
                           : nullptr;
    const bool isVarlen = (cuVec != nullptr && cuVec->GetSize() >= 2);
    // 布局契约:dense 为 [H, T, K] 头连续(batch 拍平进头维);varlen 为 [T, H, K] token 主序
    const auto qShape = context->GetInputShape(INPUT_Q_IDX)->GetStorageShape();
    const auto vShape = context->GetInputShape(2)->GetStorageShape();
    const auto aLogShape = context->GetInputShape(INPUT_A_LOG_IDX)->GetStorageShape();
    if (qShape.GetDimNum() != 3 || vShape.GetDimNum() != 3 || aLogShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const char *layout = attrs->GetAttrPointer<char>(ATTR_LAYOUT_IDX);
    const bool headMajorVarlen = isVarlen && layout != nullptr && std::strcmp(layout, "NTD") == 0;
    const uint64_t headNum = headMajorVarlen ? static_cast<uint64_t>(qShape.GetDim(0))
                                             : isVarlen ? static_cast<uint64_t>(qShape.GetDim(1))
                                                        : static_cast<uint64_t>(qShape.GetDim(0));
    const uint64_t seqLen = headMajorVarlen ? static_cast<uint64_t>(qShape.GetDim(1))
                                            : isVarlen ? static_cast<uint64_t>(qShape.GetDim(0))
                                                       : static_cast<uint64_t>(qShape.GetDim(1));
    const uint64_t kDim = static_cast<uint64_t>(qShape.GetDim(2));
    const uint64_t hvNum = headMajorVarlen ? static_cast<uint64_t>(vShape.GetDim(0))
                                           : isVarlen ? static_cast<uint64_t>(vShape.GetDim(1))
                                                      : static_cast<uint64_t>(vShape.GetDim(0));
    const uint64_t hvReal = static_cast<uint64_t>(aLogShape.GetDim(0));   // 真实 HV(共享参数寻址用)
    if (kDim != KDA_FD32_HEAD_DIM || seqLen == 0 || headNum == 0 || hvReal == 0 ||
        hvNum < headNum || hvNum % headNum != 0 || hvNum % hvReal != 0) {
        return ge::GRAPH_FAILED;
    }
    // varlen:校验并打包 cu_seqlens(0 起、单调不降、终点 = T_total;nSeq ∈ [1,1024])
    uint64_t nSeq = 0;
    uint64_t cuData[KDA_FD32_MAX_VARLEN_SEQ + 1] = {};
    if (isVarlen) {
        const int64_t *cu = reinterpret_cast<const int64_t *>(cuVec->GetData());
        nSeq = static_cast<uint64_t>(cuVec->GetSize()) - 1;
        if (cuVec->GetSize() < 2 || nSeq > KDA_FD32_MAX_VARLEN_SEQ || cu[0] != 0 ||
            static_cast<uint64_t>(cu[cuVec->GetSize() - 1]) != seqLen) {
            return ge::GRAPH_FAILED;
        }
        for (uint64_t i = 0; i <= nSeq; ++i) {
            if (i > 0 && cu[i] < cu[i - 1]) {
                return ge::GRAPH_FAILED;
            }
            cuData[i] = static_cast<uint64_t>(cu[i]);
        }
    }
    const uint64_t stateHeads = isVarlen ? nSeq * hvNum : hvNum;   // dense: B*HV;varlen: nSeq*HV
    const uint64_t taskCount = stateHeads;

    // optional initial_state:提供时必须 fp32 [HV,K,V](state_v_first=false 唯一形态)
    const bool hasInitialState = context->GetOptionalInputDesc(INPUT_INITIAL_STATE_IDX) != nullptr;
    if (hasInitialState) {
        const auto isShape = context->GetOptionalInputShape(INPUT_INITIAL_STATE_IDX)->GetStorageShape();
        if (isShape.GetDimNum() != 3 ||
            static_cast<uint64_t>(isShape.GetDim(0)) != stateHeads ||
            static_cast<uint64_t>(isShape.GetDim(1)) != KDA_FD32_HEAD_DIM ||
            static_cast<uint64_t>(isShape.GetDim(2)) != KDA_FD32_HEAD_DIM) {
            return ge::GRAPH_FAILED;
        }
    }
    // optional final_state:placeholder 输出按"未请求"处理(shape 元素数 <=1)
    bool outputFinalState = false;
    const auto fsInfo = context->GetIrOutputInstanceInfo(OUTPUT_FINAL_STATE_IDX);
    if (fsInfo != nullptr && fsInfo->GetInstanceNum() != 0) {
        const auto fsShape = context->GetOutputShape(fsInfo->GetInstanceStart());
        if (fsShape != nullptr && fsShape->GetStorageShape().GetShapeSize() != 1) {
            const auto &fs = fsShape->GetStorageShape();
            if (fs.GetDimNum() != 3 ||
                static_cast<uint64_t>(fs.GetDim(0)) != stateHeads ||
                static_cast<uint64_t>(fs.GetDim(1)) != KDA_FD32_HEAD_DIM ||
                static_cast<uint64_t>(fs.GetDim(2)) != KDA_FD32_HEAD_DIM) {
                return ge::GRAPH_FAILED;
            }
            outputFinalState = true;
        }
    }

    // 任务拍平共 taskCount 个(dense: batch*HV+head;varlen: seq*HV+head);
    // blockDim 收在 AIC 核组数内,kernel 内静态步进循环领取(task = blockIdx + n*blockDim)。
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t aicCoreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAic());
    const uint64_t blockNum = taskCount < aicCoreNum ? taskCount : aicCoreNum;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    context->SetTilingKey(0);  // 融合 kernel 只编出默认 key 0(KERNEL_TASK_TYPE_DEFAULT)
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;  // tmp 在生产路径不写,无 workspace
    }

    ChunkKdaFwdFusedTilingData tiling;
    tiling.set_nChunks((seqLen + KDA_FD32_CHUNK_TOKENS - 1) / KDA_FD32_CHUNK_TOKENS);
    tiling.set_scaleBf16(Fp32BitsToBf16Pattern(scale));   // bf16-only:fp32 高 16 位(host 契约)
    tiling.set_gateScaleBits(Fp32Bits(lowerBound));
    tiling.set_headNum(headNum);
    tiling.set_seqLen(seqLen);
    tiling.set_hasInitialState(hasInitialState ? 1 : 0);
    tiling.set_outputFinalState(outputFinalState ? 1 : 0);
    tiling.set_qkGroup(hvNum / headNum);
    tiling.set_taskCount(taskCount);
    tiling.set_hvNumReal(hvReal);
    tiling.set_blockNum(blockNum);
    tiling.set_varlenMode(headMajorVarlen ? 2 : (isVarlen ? 1 : 0));
    tiling.set_epsilonBits(Fp32Bits(epsilon));
    // g/beta 任一为 fp32 即使用 fp32 组合(外层已保证两者同 dtype)
    const uint64_t gateFlags = (context->GetInputDesc(INPUT_G_IDX)->GetDataType() == ge::DT_FLOAT ? 1 : 0) |
                               (context->GetInputDesc(INPUT_BETA_IDX)->GetDataType() == ge::DT_FLOAT ? 2 : 0);
    tiling.set_gateFlags(gateFlags);
    tiling.set_cuData(cuData);   // dense 时全零(kernel 不看);varlen 为打包的边界
    // varlen 下 nChunks 不由 host 定(逐序列解析),置 0 仅供 sanity
    if (isVarlen) {
        tiling.set_nChunks(0);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwdFused(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdFused)
    .Tiling(Tiling4ChunkKdaFwdFused)
    .TilingParse<ChunkKdaFwdFusedCompileInfo>(TilingPrepare4ChunkKdaFwdFused);
} // namespace optiling
