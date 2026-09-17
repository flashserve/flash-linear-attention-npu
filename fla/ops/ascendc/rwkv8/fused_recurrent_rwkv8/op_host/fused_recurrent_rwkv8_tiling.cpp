/*!
 * \file fused_recurrent_rwkv8_tiling.cpp
 * \brief
 */
#include "fused_recurrent_rwkv8_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(FusedRecurrentRwkv8, FusedRecurrentRwkv8Tiling, 0);

const size_t Q_INDEX = 0;
const size_t W_INDEX = 1;
const size_t K_INDEX = 2;
const size_t V_INDEX = 3;
const size_t Z_INDEX = 4;
const size_t B_INDEX = 5;
const size_t INITIAL_STATE_INDEX = 6;

const size_t ATTR_SCALE_INDEX = 0;
const size_t ATTR_OUTPUT_CHUNK_STATE_INDEX = 1;
const size_t ATTR_OUTPUT_SA_INDEX = 2;
const size_t ATTR_REVERSE_INDEX = 3;
const size_t ATTR_CHUNK_LEN_INDEX = 4;

const size_t IO_RANK = 4;

const uint32_t MAX_HEAD_DIM = 128;    // K 侧上界：UB 预算（state K×V fp32 + K 侧向量）决定
const uint32_t MAX_V_DIM = 128;       // V 侧上界：V 不切分，K×V state 须整体放入单核 UB
const uint32_t DEFAULT_CHUNK_LEN = 16;   // 对齐官方 wkv7_cuda.cu backward 的 chunk 重建粒度

void FusedRecurrentRwkv8Tiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
}

ge::graphStatus FusedRecurrentRwkv8Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus FusedRecurrentRwkv8Tiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetFlagAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetFlagAttrs."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::DoOpTiling()
{
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::DoLibApiTiling()
{
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
};

uint64_t FusedRecurrentRwkv8Tiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus FusedRecurrentRwkv8Tiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;   // 本算子无 workspace
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus FusedRecurrentRwkv8Tiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.B * tilingData_.H);   // 一核一个 (b,h) 账本
    auto tilingDataSize = sizeof(FusedRecurrentRwkv8TilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(Q_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(Q_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(W_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(W_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(K_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(K_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(V_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(V_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(Z_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(Z_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(B_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(B_INDEX));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::AnalyzeDtype()
{
    const size_t requiredIndices[] = {Q_INDEX, W_INDEX, K_INDEX, V_INDEX, Z_INDEX, B_INDEX};
    const char *requiredNames[] = {"q", "w", "k", "v", "z", "b"};
    const auto ioDtype = context_->GetInputDesc(Q_INDEX)->GetDataType();
    OP_CHECK_IF(ioDtype != ge::DT_FLOAT16 && ioDtype != ge::DT_BF16 && ioDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "q dtype should be float16/bfloat16/float32"),
                return ge::GRAPH_FAILED);
    for (size_t i = 1; i < 6; i++) {
        auto dtype = context_->GetInputDesc(requiredIndices[i])->GetDataType();
        OP_CHECK_IF(dtype != ioDtype,
                    OP_LOGE(context_->GetNodeName(), "%s dtype must equal q dtype", requiredNames[i]),
                    return ge::GRAPH_FAILED);
    }

    // state 张量恒 fp32（递推累加不降精度）
    if (context_->GetOptionalInputDesc(INITIAL_STATE_INDEX) != nullptr) {
        auto initDtype = context_->GetOptionalInputDesc(INITIAL_STATE_INDEX)->GetDataType();
        OP_CHECK_IF(initDtype != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "initial_state dtype should be float32"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::AnalyzeShapes()
{
    auto &qShape = context_->GetInputShape(Q_INDEX)->GetOriginShape();
    OP_CHECK_IF(qShape.GetDimNum() != IO_RANK,
                OP_LOGE(context_->GetNodeName(), "q should be rank-4 (B,H,T,K), got %zu", qShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // K 侧（w/k/z/b）必须与 q 全等；V 侧（v）前 3 维与 q 一致、末维独立
    const size_t kSideIndices[] = {W_INDEX, K_INDEX, Z_INDEX, B_INDEX};
    const char *kSideNames[] = {"w", "k", "z", "b"};
    for (size_t i = 0; i < 4; i++) {
        auto &shape = context_->GetInputShape(kSideIndices[i])->GetOriginShape();
        OP_CHECK_IF(shape.GetDimNum() != IO_RANK,
                    OP_LOGE(context_->GetNodeName(), "%s should be rank-4 (B,H,T,K)", kSideNames[i]),
                    return ge::GRAPH_FAILED);
        for (size_t d = 0; d < IO_RANK; d++) {
            OP_CHECK_IF(shape.GetDim(d) != qShape.GetDim(d),
                        OP_LOGE(context_->GetNodeName(), "%s shape must equal q shape", kSideNames[i]),
                        return ge::GRAPH_FAILED);
        }
    }
    auto &vShape = context_->GetInputShape(V_INDEX)->GetOriginShape();
    OP_CHECK_IF(vShape.GetDimNum() != IO_RANK,
                OP_LOGE(context_->GetNodeName(), "v should be rank-4 (B,H,T,V)"), return ge::GRAPH_FAILED);
    for (size_t d = 0; d < 3; d++) {
        OP_CHECK_IF(vShape.GetDim(d) != qShape.GetDim(d),
                    OP_LOGE(context_->GetNodeName(), "v dims B/H/T must equal q"), return ge::GRAPH_FAILED);
    }

    int64_t dimB = qShape.GetDim(0);
    int64_t dimH = qShape.GetDim(1);
    int64_t dimT = qShape.GetDim(2);
    int64_t dimK = qShape.GetDim(3);
    int64_t dimV = vShape.GetDim(3);
    OP_CHECK_IF(dimB < 1 || dimT < 1 || dimH < 1,
                OP_LOGE(context_->GetNodeName(), "B/H/T should be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimK <= 0 || dimK % 8 != 0 || dimK > MAX_HEAD_DIM,
                OP_LOGE(context_->GetNodeName(), "K should be a positive multiple of 8 and <= %u, got %ld",
                        MAX_HEAD_DIM, dimK),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimV <= 0 || dimV % 8 != 0 || dimV > MAX_V_DIM,
                OP_LOGE(context_->GetNodeName(), "V should be a positive multiple of 8 and <= %u, got %ld",
                        MAX_V_DIM, dimV),
                return ge::GRAPH_FAILED);

    // UB 预算：state 单个 K×V fp32 buffer + K 侧 8 个 + V 侧 3 个 fp32 向量 buffer
    // + 低精度路径 staging（K 侧 5 个 + V 侧 2 个，按最坏 2 字节/dtype 估算；fp32 路径无 staging）
    uint64_t ubNeed = 1UL * dimK * dimV * sizeof(float) +
                      (8UL * dimK + 3UL * dimV) * sizeof(float) +
                      (5UL * dimK + 2UL * dimV) * sizeof(uint16_t);
    OP_CHECK_IF(compileInfo_.ubSize > 0 && ubNeed > compileInfo_.ubSize,
                OP_LOGE(context_->GetNodeName(), "UB not enough: need %lu bytes, have %lu bytes",
                        ubNeed, compileInfo_.ubSize),
                return ge::GRAPH_FAILED);

    if (context_->GetOptionalInputShape(INITIAL_STATE_INDEX) != nullptr) {
        auto &initShape = context_->GetOptionalInputShape(INITIAL_STATE_INDEX)->GetOriginShape();
        OP_CHECK_IF(initShape.GetDimNum() != IO_RANK || initShape.GetDim(0) != dimB || initShape.GetDim(1) != dimH ||
                        initShape.GetDim(2) != dimK || initShape.GetDim(3) != dimV,
                    OP_LOGE(context_->GetNodeName(), "initial_state shape should be (B,H,K,V)"),
                    return ge::GRAPH_FAILED);
    }

    tilingData_.B = static_cast<uint32_t>(dimB);
    tilingData_.T = static_cast<uint32_t>(dimT);
    tilingData_.H = static_cast<uint32_t>(dimH);
    tilingData_.K = static_cast<uint32_t>(dimK);
    tilingData_.V = static_cast<uint32_t>(dimV);

    return ge::GRAPH_SUCCESS;
}

bool FusedRecurrentRwkv8Tiling::CheckFormat(ge::Format format, const std::string &desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(context_->GetNodeName(), "%s format not support NZ", desc.c_str());
        return false;
    }
    return true;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::AnalyzeFormat()
{
    const size_t requiredIndices[] = {Q_INDEX, W_INDEX, K_INDEX, V_INDEX, Z_INDEX, B_INDEX};
    const char *requiredNames[] = {"q", "w", "k", "v", "z", "b"};
    for (size_t i = 0; i < 6; i++) {
        if (!CheckFormat(context_->GetInputDesc(requiredIndices[i])->GetStorageFormat(), requiredNames[i])) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->GetOptionalInputDesc(INITIAL_STATE_INDEX) != nullptr) {
        auto initFormat = context_->GetOptionalInputDesc(INITIAL_STATE_INDEX)->GetStorageFormat();
        OP_CHECK_IF(initFormat == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(context_->GetNodeName(), "initial_state format not support NZ"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    auto scalePtr = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    OP_CHECK_IF(scalePtr == nullptr, OP_LOGE(context_->GetNodeName(), "scale attr is null"), return ge::GRAPH_FAILED);
    tilingData_.scale = *scalePtr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::GetFlagAttrs()
{
    // 训练预埋三个 bool 开关，缺省一律 false（0）；chunk_len 缺省 16
    tilingData_.reverse = 0;
    tilingData_.outputChunkState = 0;
    tilingData_.outputSa = 0;
    tilingData_.chunkLen = DEFAULT_CHUNK_LEN;
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const bool *ptr = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_CHUNK_STATE_INDEX);
    if (ptr != nullptr) {
        tilingData_.outputChunkState = *ptr ? 1 : 0;
    }
    ptr = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_SA_INDEX);
    if (ptr != nullptr) {
        tilingData_.outputSa = *ptr ? 1 : 0;
    }
    ptr = attrs->GetAttrPointer<bool>(ATTR_REVERSE_INDEX);
    if (ptr != nullptr) {
        tilingData_.reverse = *ptr ? 1 : 0;
    }
    const int64_t *chunkLenPtr = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_LEN_INDEX);
    if (chunkLenPtr != nullptr) {
        OP_CHECK_IF(*chunkLenPtr < 1,
                    OP_LOGE(context_->GetNodeName(), "chunk_len must be >= 1, got %ld", *chunkLenPtr),
                    return ge::GRAPH_FAILED);
        tilingData_.chunkLen = static_cast<uint32_t>(*chunkLenPtr);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedRecurrentRwkv8Tiling::GetOptionalInput()
{
    tilingData_.hasInitialState = (context_->GetOptionalInputDesc(INITIAL_STATE_INDEX) != nullptr) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

void FusedRecurrentRwkv8Tiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "B: [%u]", tilingData_.B);
    OP_LOGD(context_->GetNodeName(), "T: [%u]", tilingData_.T);
    OP_LOGD(context_->GetNodeName(), "H: [%u]", tilingData_.H);
    OP_LOGD(context_->GetNodeName(), "K: [%u]", tilingData_.K);
    OP_LOGD(context_->GetNodeName(), "V: [%u]", tilingData_.V);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "hasInitialState: [%u]", tilingData_.hasInitialState);
    OP_LOGD(context_->GetNodeName(), "reverse: [%u]", tilingData_.reverse);
    OP_LOGD(context_->GetNodeName(), "outputChunkState: [%u]", tilingData_.outputChunkState);
    OP_LOGD(context_->GetNodeName(), "outputSa: [%u]", tilingData_.outputSa);
    OP_LOGD(context_->GetNodeName(), "chunkLen: [%u]", tilingData_.chunkLen);
}

static ge::graphStatus FusedRecurrentRwkv8TilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("FusedRecurrentRwkv8", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForFusedRecurrentRwkv8(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("FusedRecurrentRwkv8", "context is null"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<FusedRecurrentRwkv8CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedRecurrentRwkv8)
    .Tiling(FusedRecurrentRwkv8TilingFunc)
    .TilingParse<FusedRecurrentRwkv8CompileInfo>(TilingPrepareForFusedRecurrentRwkv8);
} // namespace optiling
