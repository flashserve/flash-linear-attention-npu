/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "../../chunk_gated_delta_rule_fwd_tiling.h"

#include "../../../op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../../../op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.h"
#include "../../../op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#include "../../../op_kernel/internal/arch22/operators/chunk_recompute_wu_fwd_ho/op_kernel/chunk_recompute_wu_fwd_ho_struct.h"
#include "../../../op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22_struct.h"

#include "securec.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>
#include <register/op_impl_registry.h>

namespace optiling {

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdArch22StateOutput(gert::TilingContext *context,
                                                               bool separateHoWorkspace);

namespace {

constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_K = 1;
constexpr size_t INPUT_V = 2;
constexpr size_t INPUT_BETA = 3;
constexpr size_t INPUT_A_STORAGE = 4;
constexpr size_t INPUT_RAW_G = 5;
constexpr size_t INPUT_GK = 6;
constexpr size_t INPUT_INITIAL_STATE = 7;
constexpr size_t INPUT_CU_SEQLENS = 8;
constexpr size_t INPUT_CHUNK_INDICES = 9;

constexpr size_t ATTR_OUTPUT_FINAL_STATE = 0;
constexpr size_t ATTR_CHUNK_SIZE = 1;
constexpr size_t ATTR_OUTPUT_G_CUMSUM = 3;
constexpr size_t ATTR_RAW_G_LAYOUT = 4;
constexpr size_t ATTR_QKV_LAYOUT = 5;
constexpr size_t ATTR_O_LAYOUT = 6;
constexpr size_t ATTR_OUTPUT_H = 7;

constexpr int64_t SUPPORTED_K_DIM = 128;
constexpr int64_t SUPPORTED_V_DIM_128 = 128;
constexpr int64_t SUPPORTED_V_DIM_256 = 256;
constexpr int64_t CHUNK_64 = 64;
constexpr int64_t CHUNK_128 = 128;
constexpr uint32_t TILING_KEY_V128 = 1;
constexpr uint32_t TILING_KEY_V256 = 2;
constexpr uint32_t TILING_KEY_PREPARED_BTH = 3;
constexpr uint32_t TILING_KEY_PREPARED_BTH_V256 = 4;
// Keys 1–4 and their workspace data flow remain the no-H path.
constexpr uint32_t TILING_KEY_EXPORT_H_OFFSET = 4;
constexpr uint64_t WORKSPACE_ALIGNMENT = 512;
constexpr uint64_t TILING_ALIGNMENT = 8;
constexpr uint64_t FP32_BLOCK_ELEMS = 8;
// Stage P encodes the GM row gap in DataCopyExtParams::srcStride/dstStride
// (uint32 bytes).  This is a transport-field bound, not a supported-head
// whitelist; larger tensors are routed through the BHT layout by the host.
constexpr uint64_t MAX_PREPARE_HEADS = 0xffffffffULL / sizeof(float) + 1;
constexpr uint64_t LOW_PRECISION_SOLVE_WORKSPACE_SLOTS = 5;
constexpr uint64_t FP32_SOLVE_WORKSPACE_SLOTS = 4;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : CeilDiv(value, alignment) * alignment;
}

constexpr uint64_t HO_READY_SLOT_BYTES = 32;  // 每 bank 2*C 个 32B 槽，与协议 kSlotInt32 一致

// 仅用于新增 HO 预留算术的无回绕核查：结果可表则写入 *out 并返回 false。
bool AddOverflow(uint64_t a, uint64_t b, uint64_t *out)
{
    const uint64_t sum = a + b;
    const bool overflow = sum < a;
    *out = sum;
    return overflow;
}

bool MulOverflow(uint64_t a, uint64_t b, uint64_t *out)
{
    const uint64_t product = a * b;
    const bool overflow = a != 0 && product / a != b;
    *out = product;
    return overflow;
}

// AlignUp 的无回绕版本。checked add 得到 bumped=value+alignment-1 后必须直接
// 截断对齐（bumped/alignment*alignment <= bumped，不会二次溢出）；不能对
// bumped 再做 CeilDiv——那会二次取整（0 对齐 512 错成 512），且 bumped 取
// UINT64_MAX 时 CeilDiv 内部加法回绕会误报成功并输出 0。value 超过最后可
// 对齐输入（bumped 回绕）或 alignment 为 0 时返回 true，*out 不再可用。
bool AlignUpChecked(uint64_t value, uint64_t alignment, uint64_t *out)
{
    if (alignment == 0) {
        return true;
    }
    uint64_t bumped = 0;
    if (AddOverflow(value, alignment - 1, &bumped)) {
        return true;
    }
    *out = bumped / alignment * alignment;
    return false;
}

bool IsShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (shape->GetStorageShape().GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

bool GetChunkCount(const gert::StorageShape *shape, uint64_t *count)
{
    if (shape == nullptr || count == nullptr) {
        return false;
    }
    const auto &storage = shape->GetStorageShape();
    if (storage.GetDimNum() == 1 && storage.GetDim(0) > 0 && storage.GetDim(0) % 2 == 0) {
        *count = static_cast<uint64_t>(storage.GetDim(0) / 2);
        return true;
    }
    if (storage.GetDimNum() == 2 && storage.GetDim(0) > 0 && storage.GetDim(1) == 2) {
        *count = static_cast<uint64_t>(storage.GetDim(0));
        return true;
    }
    return false;
}

ge::graphStatus BuildAbcCubeTiling(uint64_t bt, uint64_t k, ge::DataType dtype,
                                   AscendC::tiling::TCubeTiling &tiling)
{
    matmul_tiling::MatmulApiTiling mm;
    const auto inputType = dtype == ge::DT_BF16
                               ? matmul_tiling::DataType::DT_BF16
                               : matmul_tiling::DataType::DT_FLOAT16;
    if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    inputType, false) != 0 ||
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    inputType, true) != 0 ||
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT) != 0 ||
        mm.EnableBias(false) != 0) {
        return ge::GRAPH_FAILED;
    }
    const int32_t btI32 = static_cast<int32_t>(bt);
    const int32_t kI32 = static_cast<int32_t>(k);
    if (mm.SetShape(btI32, btI32, kI32) != 0 ||
        mm.SetOrgShape(btI32, btI32, kI32) != 0 ||
        mm.SetFixSplit(btI32, btI32, -1) != 0 ||
        mm.SetBufferSpace(-1, -1, -1, -1) != 0) {
        return ge::GRAPH_FAILED;
    }
    return mm.GetTiling(tiling) == -1 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdArch22(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr || context->GetAttrs() == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid tiling context."),
                return ge::GRAPH_FAILED);
    const auto *qShape = context->GetOptionalInputShape(INPUT_Q);
    const auto *kShape = context->GetOptionalInputShape(INPUT_K);
    const auto *vShape = context->GetOptionalInputShape(INPUT_V);
    const auto *betaShape = context->GetOptionalInputShape(INPUT_BETA);
    const auto *aShape = context->GetOptionalInputShape(INPUT_A_STORAGE);
    const auto *gShape = context->GetOptionalInputShape(INPUT_RAW_G);
    OP_CHECK_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr || betaShape == nullptr ||
                    aShape == nullptr || gShape == nullptr ||
                    qShape->GetStorageShape().GetDimNum() != 4 ||
                    kShape->GetStorageShape().GetDimNum() != 4 ||
                    vShape->GetStorageShape().GetDimNum() != 4 ||
                    betaShape->GetStorageShape().GetDimNum() != 3 ||
                    aShape->GetStorageShape().GetDimNum() != 4 ||
                    gShape->GetStorageShape().GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "Phase 6 requires rank-4 q/k/v/A and rank-3 beta/raw_g."),
                return ge::GRAPH_FAILED);
    const auto *qDesc = context->GetInputDesc(INPUT_Q);
    const auto *kDesc = context->GetInputDesc(INPUT_K);
    const auto *vDesc = context->GetInputDesc(INPUT_V);
    const auto *betaDesc = context->GetInputDesc(INPUT_BETA);
    const auto *aDesc = context->GetInputDesc(INPUT_A_STORAGE);
    const auto *gDesc = context->GetInputDesc(INPUT_RAW_G);
    OP_CHECK_IF(qDesc == nullptr || kDesc == nullptr || vDesc == nullptr || betaDesc == nullptr ||
                    aDesc == nullptr || gDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "Phase 6 requires valid input descriptors."),
                return ge::GRAPH_FAILED);
    const ge::DataType inputDtype = qDesc->GetDataType();
    const bool isFp16 = inputDtype == ge::DT_FLOAT16;
    const bool isBf16 = inputDtype == ge::DT_BF16;
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t *rawGLayoutAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_RAW_G_LAYOUT);
    const int64_t rawGLayout = rawGLayoutAttr == nullptr ? 0 : *rawGLayoutAttr;
    OP_CHECK_IF(rawGLayout != 0 && rawGLayout != 1,
                OP_LOGE(context->GetNodeName(), "raw_g_layout must be 0 (BHT) or 1 (BTH)."),
                return ge::GRAPH_FAILED);
    const int64_t *qkvLayoutAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_QKV_LAYOUT);
    const int64_t qkvLayout = qkvLayoutAttr == nullptr ? 0 : *qkvLayoutAttr;
    OP_CHECK_IF(qkvLayout != 0 && qkvLayout != 1,
                OP_LOGE(context->GetNodeName(), "qkv_layout must be 0 (BHTD) or 1 (BTHD)."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qkvLayout == 1 && platform.GetCurNpuArch() != NpuArch::DAV_2201,
                OP_LOGE(context->GetNodeName(), "Native token-major QKV requires DAV_2201."),
                return ge::GRAPH_FAILED);
    const int64_t *oLayoutAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_O_LAYOUT);
    const int64_t oLayout = oLayoutAttr == nullptr ? 0 : *oLayoutAttr;
    OP_CHECK_IF(oLayout != 0 && oLayout != 1,
                OP_LOGE(context->GetNodeName(), "o_layout must be 0 (BHTV) or 1 (BTHV)."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oLayout == 1 && platform.GetCurNpuArch() != NpuArch::DAV_2201,
                OP_LOGE(context->GetNodeName(), "Native token-major O requires DAV_2201."),
                return ge::GRAPH_FAILED);
    const size_t headAxis = qkvLayout == 1 ? 2 : 1;
    const size_t tokenAxis = qkvLayout == 1 ? 1 : 2;
    const gert::Shape qStorage = qShape->GetStorageShape();
    const int64_t batch = qStorage.GetDim(0);
    const int64_t heads = qStorage.GetDim(headAxis);
    const int64_t tokens = qStorage.GetDim(tokenAxis);
    const int64_t kDim = qStorage.GetDim(3);
    const gert::Shape vStorage = vShape->GetStorageShape();
    const int64_t valueHeads = vStorage.GetDim(headAxis);
    const int64_t vDim = vStorage.GetDim(3);
    const auto *cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
    const auto *chunkShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
    const auto *cuDesc = context->GetOptionalInputDesc(INPUT_CU_SEQLENS);
    const auto *chunkDesc = context->GetOptionalInputDesc(INPUT_CHUNK_INDICES);
    const bool hasCu = cuDesc != nullptr && cuShape != nullptr;
    const bool hasChunks = chunkDesc != nullptr && chunkShape != nullptr;
    const bool isVarlen = hasCu || hasChunks;
    OP_CHECK_IF((!isFp16 && !isBf16) ||
                    batch <= 0 || heads <= 0 || tokens <= 0 || kDim != SUPPORTED_K_DIM ||
                    (vDim != SUPPORTED_V_DIM_128 && vDim != SUPPORTED_V_DIM_256) ||
                    valueHeads <= 0 || (valueHeads % heads) != 0,
                OP_LOGE(context->GetNodeName(),
                        "Phase 6 requires positive B/Hk/T, Hk divides Hv, K=128, and V=128/256; dense T may be arbitrary."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(qkvLayout == 1 ? IsShape(kShape, {batch, tokens, heads, kDim}) :
                                      IsShape(kShape, {batch, heads, tokens, kDim})),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires k to match q in [B,H,T,K]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(qkvLayout == 1 ? IsShape(vShape, {batch, tokens, valueHeads, vDim}) :
                                      IsShape(vShape, {batch, valueHeads, tokens, vDim})),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires v=[B,Hv,T,V] with Hv divisible by Hk."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape(betaShape, {batch, valueHeads, tokens}),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires beta=[B,Hv,T]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((rawGLayout == 0 && !IsShape(gShape, {batch, valueHeads, tokens})) ||
                    (rawGLayout == 1 && !IsShape(gShape, {batch, tokens, valueHeads})),
                OP_LOGE(context->GetNodeName(),
                        "Phase 6 raw_g layout contract is 0:[B,Hv,T] or 1:[B,T,Hv]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDesc->GetDataType() != inputDtype ||
                    vDesc->GetDataType() != inputDtype ||
                    aDesc->GetDataType() != inputDtype ||
                    betaDesc->GetDataType() != ge::DT_FLOAT ||
                    gDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(),
                        "Phase 6 requires matching FP16/BF16 inputs/A and FP32 beta/g."),
                return ge::GRAPH_FAILED);

    const bool *outputFinalState =
        context->GetAttrs()->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE);
    const int64_t *chunkSize = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const bool *outputGCumsum = context->GetAttrs()->GetAttrPointer<bool>(ATTR_OUTPUT_G_CUMSUM);
    const bool *outputH = context->GetAttrs()->GetAttrPointer<bool>(ATTR_OUTPUT_H);
    uint64_t varlenChunks = 0;
    OP_CHECK_IF(outputFinalState == nullptr || chunkSize == nullptr ||
                    (*chunkSize != CHUNK_64 && *chunkSize != CHUNK_128) ||
                    context->GetOptionalInputDesc(INPUT_GK) != nullptr || hasCu != hasChunks ||
                    (isVarlen && (cuDesc->GetDataType() != ge::DT_INT64 ||
                                  chunkDesc->GetDataType() != ge::DT_INT64 ||
                                  cuShape->GetStorageShape().GetDimNum() != 1 ||
                                  cuShape->GetStorageShape().GetDim(0) < 2 ||
                                  !GetChunkCount(chunkShape, &varlenChunks))),
                OP_LOGE(context->GetNodeName(),
                        "Phase 6 requires chunk_size=64/128 and paired valid varlen metadata."),
                return ge::GRAPH_FAILED);
    const bool exportH = outputH != nullptr && *outputH;
    const auto *hDesc = context->GetOutputDesc(4);
    const auto *hShape = context->GetOutputShape(4);
    OP_CHECK_IF(hDesc == nullptr || hShape == nullptr,
                OP_LOGE(context->GetNodeName(), "The low-level H output descriptor is required."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(exportH && (hDesc->GetDataType() != inputDtype ||
                    !IsShape(hShape, {batch, valueHeads,
                        static_cast<int64_t>(isVarlen ? varlenChunks :
                            CeilDiv(static_cast<uint64_t>(tokens), static_cast<uint64_t>(*chunkSize))),
                        kDim, vDim})),
                OP_LOGE(context->GetNodeName(), "H must be [B,Hv,NT,K,V] with Q dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawGLayout == 1 &&
                    (platform.GetCurNpuArch() != NpuArch::DAV_2201 ||
                     kDim != SUPPORTED_K_DIM ||
                     (vDim != SUPPORTED_V_DIM_128 && vDim != SUPPORTED_V_DIM_256) ||
                     (*chunkSize != CHUNK_64 && *chunkSize != CHUNK_128) ||
                     (isVarlen && batch != 1)),
                OP_LOGE(context->GetNodeName(),
                        "raw_g_layout=1 requires DAV_2201, K=128, V=128/256, BT64/128, and varlen B=1; Hv uses tiled Stage P."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawGLayout == 1 && static_cast<uint64_t>(valueHeads) > MAX_PREPARE_HEADS,
                OP_LOGE(context->GetNodeName(),
                        "raw_g_layout=1 GM row gap exceeds DataCopyExtParams uint32 bytes; use BHT fallback."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape(aShape, {batch, valueHeads, tokens, *chunkSize}),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires a_storage=[B,Hv,T,chunk_size]."),
                return ge::GRAPH_FAILED);
    // A2 判定与实际 AIC/AIV 核数只计算一次：HO 预留判定与后续 ABC 布局复用。
    const bool useFp32Solve = platform.GetCurNpuArch() == NpuArch::DAV_2201;
    const uint64_t aicCoreNum = std::max<uint64_t>(1, platform.GetCoreNumAic());
    const uint64_t aivCoreNum = std::max<uint64_t>(1, platform.GetCoreNumAiv());
    const uint64_t systemWorkspace = platform.GetLibApiWorkSpaceSize();

    // HO 空闲流水：host 仅做“可能命中”的 GM ready 预留判定，真实 0<P<C、
    // P<=C-P 与至少一条多 chunk 序列由 device 统一判断。必要条件：DAV_2201、
    // 无 gk、K=128、V=128/256、BT=64/128、tokens>BT；dense 用完整必要条件
    // P=batch*Hv*ceil(V/128)<C 且 P<=C-P，varlen 只能用 S>=1 下界套同样两条件
    // （host 不读设备 cu_seqlens 值，也不把 tokenBatch 当非空 S）。ready 区
    // AlignUp(bankCount*2*C*32B, 512) 与 P 的乘法全部无回绕，bankCount 须在
    // 协议 uint32 参数范围内；任一不满足则不预留（三个新字段清零，
    // StateOutput 保持原串行布局）。
    const uint64_t hoVBlockNum = CeilDiv(static_cast<uint64_t>(vDim), SUPPORTED_V_DIM_128);
    const uint64_t hoReadyBankCount = isVarlen
                                          ? varlenChunks
                                          : CeilDiv(static_cast<uint64_t>(tokens),
                                                    static_cast<uint64_t>(*chunkSize));
    uint64_t hoReadyRegionBytes = 0;
    bool hoPipelineReserved = false;
    if (useFp32Solve && context->GetOptionalInputDesc(INPUT_GK) == nullptr &&
        kDim == SUPPORTED_K_DIM &&
        (vDim == SUPPORTED_V_DIM_128 || vDim == SUPPORTED_V_DIM_256) &&
        (*chunkSize == CHUNK_64 || *chunkSize == CHUNK_128) &&
        static_cast<uint64_t>(tokens) > static_cast<uint64_t>(*chunkSize) &&
        hoReadyBankCount <= 0xffffffffULL) {
        // P 下界（varlen 按 S>=1）或精确值（dense S=batch）；乘法回绕即天文
        // 尺寸，视作 P>=C 不预留。追加与设备侧一致的比例条件 P<=C-P：varlen
        // 用下界估计故可能多预留，真实 S 仍由设备复核；前一条件已保证
        // P<C 成立，aicCoreNum - hoProducerTasks 不会无符号下溢。
        uint64_t hoProducerTasks = 0;
        const bool hoProducerOk =
            !MulOverflow(static_cast<uint64_t>(valueHeads), hoVBlockNum, &hoProducerTasks) &&
            (isVarlen ||
             !MulOverflow(static_cast<uint64_t>(batch), hoProducerTasks, &hoProducerTasks)) &&
            hoProducerTasks < aicCoreNum &&
            hoProducerTasks <= aicCoreNum - hoProducerTasks;
        // aicCoreNum 源自 uint32 核数，2*C*32B 不回绕；再与 bankCount 相乘核查。
        uint64_t hoBankBytes = 0;
        uint64_t hoReadyBytes = 0;
        hoPipelineReserved =
            hoProducerOk && !MulOverflow(aicCoreNum, 2 * HO_READY_SLOT_BYTES, &hoBankBytes) &&
            !MulOverflow(hoReadyBankCount, hoBankBytes, &hoReadyBytes) &&
            !AlignUpChecked(hoReadyBytes, WORKSPACE_ALIGNMENT, &hoReadyRegionBytes);
        if (!hoPipelineReserved) {
            hoReadyRegionBytes = 0;
        }
    }
    OP_CHECK_IF(Tiling4ChunkGatedDeltaRuleFwdArch22StateOutput(context, hoPipelineReserved) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Reuse of the accepted Phase 5 suffix tiling failed."),
                return ge::GRAPH_FAILED);

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    OP_CHECK_IF(workspaceSizes[0] < systemWorkspace,
                OP_LOGE(context->GetNodeName(), "Phase 5 workspace is smaller than system workspace."),
                return ge::GRAPH_FAILED);

    GDN::Arch22ChunkGatedDeltaRuleFwdTrailer trailer{};
    auto &abc = trailer.abc;
    abc.qkvLayout = static_cast<uint64_t>(qkvLayout);
    abc.oLayout = static_cast<uint64_t>(oLayout);
    abc.B = static_cast<uint64_t>(batch);
    abc.Hk = static_cast<uint64_t>(heads);
    abc.Hv = static_cast<uint64_t>(valueHeads);
    abc.hvPerHk = static_cast<uint64_t>(valueHeads / heads);
    abc.T = static_cast<uint64_t>(tokens);
    abc.K = SUPPORTED_K_DIM;
    abc.BT = static_cast<uint64_t>(*chunkSize);
    abc.NT = isVarlen ? varlenChunks : CeilDiv(abc.T, abc.BT);
    // ABC produces one KKT/solve tile per value head. K is shared by the
    // contiguous group of hvPerHk value heads mapped to one logical K head.
    abc.taskNum = abc.B * abc.Hv * abc.NT;
    abc.usedAicNum = aicCoreNum;
    abc.usedAivNum = std::min<uint64_t>(aivCoreNum, aicCoreNum * 2);
    abc.btAlign = AlignUp(abc.BT, FP32_BLOCK_ELEMS);
    abc.isVarlen = isVarlen ? 1 : 0;
    abc.scoreWorkspaceBytes =
        AlignUp(abc.taskNum * abc.BT * abc.BT * sizeof(float), WORKSPACE_ALIGNMENT);
    abc.aWorkspaceBytes = AlignUp(
        abc.B * abc.Hv * abc.T * abc.BT * sizeof(uint16_t), WORKSPACE_ALIGNMENT);
    const uint64_t solveWorkspaceBytes = abc.BT == CHUNK_64
        ? FP32_SOLVE_WORKSPACE_SLOTS * abc.BT * abc.BT * sizeof(float)
        : LOW_PRECISION_SOLVE_WORKSPACE_SLOTS * abc.BT * abc.BT * sizeof(uint16_t);
    abc.solveWorkspacePerCoreBytes = AlignUp(solveWorkspaceBytes, WORKSPACE_ALIGNMENT);
    if (useFp32Solve) {
        // 每物理核组的固定 arena 覆盖所有层，temp/result 分界不随阶段改变。
        const uint64_t arenaElements = abc.BT == CHUNK_128
            ? GDN::FP32_SOLVE_ARENA128_ELEMENTS : GDN::FP32_SOLVE_ARENA64_ELEMENTS;
        abc.solveWorkspacePerCoreBytes = AlignUp(
            arenaElements * sizeof(float), WORKSPACE_ALIGNMENT);
        trailer.solveSequenceCount = isVarlen ? cuShape->GetStorageShape().GetDim(0) - 1 : 0;
    }
    abc.totalTiles = static_cast<int64_t>(abc.taskNum);
    abc.matrixSize = *chunkSize;
    abc.numHeads = valueHeads;
    abc.seqLen = tokens;
    abc.batchSize = batch;
    abc.isLower = 1;
    abc.hasCuSeqlens = isVarlen ? 1 : 0;
    abc.tilesPerCore = static_cast<int64_t>(CeilDiv(abc.taskNum, aicCoreNum));
    abc.chunkSize = *chunkSize;
    abc.numChunks = isVarlen ? 0 : static_cast<int64_t>(abc.NT);
    abc.lastChunkValidSize = isVarlen ? 0 :
        (tokens % *chunkSize == 0 ? *chunkSize : tokens % *chunkSize);
    abc.totalChunks = static_cast<int64_t>(abc.NT);
    // Public A is always BNSD [B, Hv, T, BT]. The fused KKT producer also
    // enumerates B -> Hv -> chunk, so varlen uses the private SolveTri mode 4
    // (BNSD + cu_seqlens/chunk_indices). Mode 3 is reserved for standalone
    // SolveTri's public NTD layout and must not be reused here.
    abc.layoutMode = isVarlen ? 4 : 0;
    abc.dtypeMode = isBf16 ? 1 : 0;
    abc.totalTokens = isVarlen ? tokens : 0;
    OP_CHECK_IF(BuildAbcCubeTiling(abc.BT, abc.K, inputDtype,
                                   abc.cubeTilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(),
                        "Failed to build the Phase 6 KKT Matmul tiling."),
                return ge::GRAPH_FAILED);
    uint64_t workspaceOffset = AlignUp(workspaceSizes[0] - systemWorkspace, WORKSPACE_ALIGNMENT);
    trailer.scoreWorkspaceOffset = workspaceOffset;
    workspaceOffset += abc.scoreWorkspaceBytes;
    trailer.aWorkspaceOffset = workspaceOffset;
    workspaceOffset += abc.aWorkspaceBytes;
    trailer.solveWorkspaceOffset = workspaceOffset;
    workspaceOffset += aicCoreNum * abc.solveWorkspacePerCoreBytes;
    trailer.gCumsumBhtOffset = workspaceOffset;
    workspaceOffset += AlignUp(abc.B * abc.Hv * abc.T * sizeof(float), WORKSPACE_ALIGNMENT);
    trailer.outputGCumsum = (outputGCumsum == nullptr || *outputGCumsum) ? 1 : 0;
    if (useFp32Solve) {
        // 独立保存 FP32 数据版本；只复用已 drain 的跨层 scratch。
        const uint64_t rows = abc.B * abc.Hv * abc.T;
        trailer.solveFp32InputOffset = workspaceOffset;
        workspaceOffset += AlignUp(rows * abc.BT * sizeof(float), WORKSPACE_ALIGNMENT);
        trailer.solveD16Offset = workspaceOffset;
        workspaceOffset += AlignUp(rows * 16 * sizeof(float), WORKSPACE_ALIGNMENT);
        trailer.solveD32Offset = workspaceOffset;
        workspaceOffset += AlignUp(rows * 32 * sizeof(float), WORKSPACE_ALIGNMENT);
        trailer.solveD64Offset = trailer.solveD32Offset;
        if (abc.BT == CHUNK_128) {
            trailer.solveD64Offset = workspaceOffset;
            workspaceOffset += AlignUp(rows * 64 * sizeof(float), WORKSPACE_ALIGNMENT);
        }
    }
    if (useFp32Solve) {
        const uint64_t capacity = std::min<uint64_t>(
            GDN::FRONT_PARENT_BATCH_SIZE, static_cast<uint64_t>(abc.tilesPerCore));
        trailer.frontWuWorkspaceOffset = workspaceOffset;
        workspaceOffset += AlignUp(aicCoreNum * capacity * abc.BT *
            (static_cast<uint64_t>(vDim) + abc.K) * sizeof(uint16_t), WORKSPACE_ALIGNMENT);
    }
    if (hoPipelineReserved) {
        // GM ready 区追加在 Phase6 全部 scratch/中间区之后，offset 与上方字段同
        // 以 userWorkspace 为基址；abc.NT 与前置判定的 hoReadyBankCount 同式
        // （dense CeilDiv(T,BT)、varlen 总 chunk 数上界），uint32 范围已核查。
        // 追加算术无回绕；理论溢出时按不可用处理（字段保持 0，不分配 ready）。
        uint64_t hoReadyOffset = 0;
        uint64_t hoReadyEnd = 0;
        if (!AlignUpChecked(workspaceOffset, WORKSPACE_ALIGNMENT, &hoReadyOffset) &&
            !AddOverflow(hoReadyOffset, hoReadyRegionBytes, &hoReadyEnd)) {
            trailer.hoPipelineAvailable = 1;
            trailer.hoReadyWorkspaceOffset = hoReadyOffset;
            trailer.hoReadyBankCount = abc.NT;
            workspaceOffset = hoReadyEnd;
        }
    }
    // ready 追加后的最终总量 systemWorkspace+workspaceOffset 也需无回绕核查；
    // 回绕只可能来自天文尺寸输入，此时分配不可实现，直接返回失败而不是把
    // 回绕后的值当成功分配上报。
    uint64_t totalWorkspaceSize = 0;
    OP_CHECK_IF(AddOverflow(systemWorkspace, workspaceOffset, &totalWorkspaceSize),
                OP_LOGE(context->GetNodeName(), "Phase 6 total workspace size wraps uint64."),
                return ge::GRAPH_FAILED);
    workspaceSizes[0] = totalWorkspaceSize;

    GdnMegaArch22FwdHTilingData hTiling;
    const uint64_t hTilingSize = hTiling.GetDataSize();
    OP_CHECK_IF(hTilingSize != sizeof(::GdnMegaArch22FwdHTilingData),
                OP_LOGE(context->GetNodeName(),
                        "FwdH host/kernel tiling size mismatch: host=%lu, kernel=%zu.",
                        hTilingSize, sizeof(::GdnMegaArch22FwdHTilingData)),
                return ge::GRAPH_FAILED);
    const uint64_t oTilingOffset = AlignUp(hTilingSize, TILING_ALIGNMENT);
    const uint64_t phase5TrailerEnd = oTilingOffset + sizeof(GDN::GdnMegaArch22FwdOTilingData) +
                                      sizeof(GDN::ChunkRecomputeWUFwdHOTrailer);
    const uint64_t phase6TrailerOffset = AlignUp(phase5TrailerEnd, TILING_ALIGNMENT);
    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    const uint64_t rawTilingSize = phase6TrailerOffset + sizeof(trailer);
    OP_CHECK_IF(rawTilingSize > rawTiling->GetCapacity(),
                OP_LOGE(context->GetNodeName(), "Phase 6 combined tiling exceeds raw tiling capacity."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(static_cast<uint8_t *>(rawTiling->GetData()) + phase6TrailerOffset,
                         rawTiling->GetCapacity() - phase6TrailerOffset,
                         &trailer, sizeof(trailer)) != EOK,
                OP_LOGE(context->GetNodeName(), "Serialize Phase 6 ABC trailer failed."),
                return ge::GRAPH_FAILED);
    rawTiling->SetDataSize(rawTilingSize);
    const uint32_t baseKey = rawGLayout == 1 ?
        (vDim == SUPPORTED_V_DIM_256 ? TILING_KEY_PREPARED_BTH_V256 : TILING_KEY_PREPARED_BTH) :
        (vDim == SUPPORTED_V_DIM_256 ? TILING_KEY_V256 : TILING_KEY_V128);
    context->SetTilingKey(baseKey + (exportH ? TILING_KEY_EXPORT_H_OFFSET : 0));
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(),
            "Phase 6 tiling: B=%ld, Hk=%ld, Hv=%ld, T=%ld, K=%ld, V=%ld, blocks=%lu, tasks=%lu, "
            "suffix=%zu, total=%zu, hoReady=%lu, hoBanks=%lu, hoReadyOffset=%lu.",
            batch, heads, valueHeads, tokens, kDim, vDim,
            aicCoreNum, abc.taskNum, workspaceSizes[0] - systemWorkspace,
            workspaceSizes[0], trailer.hoPipelineAvailable, trailer.hoReadyBankCount,
            trailer.hoReadyWorkspaceOffset);
    return ge::GRAPH_SUCCESS;
}


} // namespace optiling
