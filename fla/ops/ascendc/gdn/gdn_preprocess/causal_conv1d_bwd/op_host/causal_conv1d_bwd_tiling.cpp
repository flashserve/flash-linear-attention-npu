/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd_tiling.cpp
 * \brief Causal Conv1D backward tiling implementation
 *
 * Tiling keys:
 *   key=0 : Main kernel. Activation is selected by tiling data at runtime.
 */

#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "platform/soc_spec.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "causal_conv1d_bwd_tiling.h"
#include <algorithm>
#include <cstring>

namespace {
    constexpr uint32_t BYTE_BLOCK = 32;
    constexpr uint32_t X_INPUT_INDEX = 0;
    constexpr uint32_t Y_INPUT_INDEX = 1;
    constexpr uint32_t WEIGHT_INPUT_INDEX = 2;
    constexpr uint32_t DY_INPUT_INDEX = 3;
    constexpr uint32_t INITIAL_STATE_INPUT_INDEX = 4;
    constexpr uint32_t DHT_INPUT_INDEX = 5;
    constexpr uint32_t QUERY_START_LOC_INPUT_INDEX = 6;
    constexpr uint32_t DIM_INDEX0 = 0;
    constexpr uint32_t DIM_INDEX1 = 1;
    constexpr uint32_t DIM_INDEX2 = 2;
    constexpr uint32_t DIM_INDEX3 = 3;
    constexpr uint32_t ATTR_ACTIVATION_INDEX = 0;
    constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 1;
    constexpr uint32_t RESERVED_UB = 16 * 1024;
    constexpr uint32_t B16_DTYPE_SIZE = 2U;
    constexpr uint32_t FP32_DTYPE_SIZE = 4U;
    constexpr uint32_t BLOCK_ALIGN_NUM = 8U;
    constexpr uint32_t DIRECT_FAST_BT_MAX = 224U;
    constexpr uint32_t DIRECT_FAST_BT_MIN = 64U;
    constexpr uint32_t DIRECT_FAST_BT_STEP = 32U;
    constexpr uint32_t DIRECT_910B_BT_BD64 = 104U;
    constexpr uint32_t DIRECT_910B_BT_BD128 = 55U;
    // 必须与 kernel 侧 NsCausalConv1dBwd::BWD_DIRECT_W_MAX 保持一致（arch35 regbase 快路径）。
    constexpr uint32_t ARCH35_DIRECT_W_MAX = 8U;
    constexpr uint32_t ACTIVATION_NONE = 0;
    constexpr uint32_t ACTIVATION_SILU = 1;
    constexpr uint32_t ACTIVATION_SWISH = 2;
    constexpr uint32_t INPUT_LAYOUT_BSND = 0; // y/dy: [B, T, N*D].
    constexpr uint32_t INPUT_LAYOUT_TND = 1;  // y/dy: [total_tokens, N*D].
    constexpr uint32_t INPUT_LAYOUT_BNSD = 2; // y/dy: [B, N, T, D].
    constexpr uint32_t INPUT_LAYOUT_NTD = 3;  // y/dy: [N, total_tokens, D].
}

namespace optiling {
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using namespace Ops::Transformer::OpTiling;

static ge::graphStatus CausalConv1dBwdTilingFunc(gert::TilingContext *context)
{
    CausalConv1dBwdTilingData *tiling = context->GetTilingData<CausalConv1dBwdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CausalConv1dBwdTilingData), 0, sizeof(CausalConv1dBwdTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(X_INPUT_INDEX)->GetStorageShape();
    auto dyShape = context->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    auto wShape = context->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto xDesc = context->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    bool isB16Input = (xDtype == ge::DT_BF16 || xDtype == ge::DT_FLOAT16);
    bool isFp32Input = (xDtype == ge::DT_FLOAT);
    uint32_t B = 0;
    uint32_t T = 0;
    uint32_t D = 0;
    uint32_t totalTokens = 0;
    uint32_t inputMode = 1;
    uint32_t inputLayout = INPUT_LAYOUT_BSND;
    uint32_t inputN = 1;
    uint32_t inputHeadDim = 0;

    uint32_t activation = ACTIVATION_NONE;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > ATTR_ACTIVATION_INDEX) {
        activation = *(attrs->GetAttrPointer<uint32_t>(ATTR_ACTIVATION_INDEX));
    }
    const char *inputLayoutStr = "BSND";
    if (attrs != nullptr && attrs->GetAttrNum() > ATTR_INPUT_LAYOUT_INDEX) {
        const char *attrLayout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
        if (attrLayout != nullptr) {
            inputLayoutStr = attrLayout;
        }
    }
    if (std::strcmp(inputLayoutStr, "BNSD") == 0) {
        inputLayout = INPUT_LAYOUT_BNSD;
    } else if (std::strcmp(inputLayoutStr, "NTD") == 0) {
        inputLayout = INPUT_LAYOUT_NTD;
    } else if (std::strcmp(inputLayoutStr, "TND") == 0) {
        inputLayout = INPUT_LAYOUT_TND;
    } else if (std::strcmp(inputLayoutStr, "BSND") == 0 || std::strcmp(inputLayoutStr, "BSH") == 0) {
        inputLayout = INPUT_LAYOUT_BSND;
    } else {
        OP_LOGE(context, "unsupported inputLayout %s, expect BSND/TND/BNSD/NTD", inputLayoutStr);
        return ge::GRAPH_FAILED;
    }

    if (inputLayout == INPUT_LAYOUT_BNSD) {
        OP_CHECK_IF(xShape.GetDimNum() != 3,
                    OP_LOGE(context, "x must be logical 3D [B, T, D] when inputLayout is BNSD"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dyShape.GetDimNum() != 4,
                    OP_LOGE(context, "dy must be 4D [B, N, T, Dh] when inputLayout is BNSD"),
                    return ge::GRAPH_FAILED);
        inputMode = 1;
        B = xShape.GetDim(DIM_INDEX0);
        T = xShape.GetDim(DIM_INDEX1);
        D = xShape.GetDim(DIM_INDEX2);
        inputN = dyShape.GetDim(DIM_INDEX1);
        inputHeadDim = dyShape.GetDim(DIM_INDEX3);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != B ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX2)) != T ||
                inputN * inputHeadDim != D,
            OP_LOGE(context, "dy must be [B, N, T, Dh] with D=N*Dh for logical x [B, T, D]"),
            return ge::GRAPH_FAILED);
        totalTokens = B * T;
    } else if (inputLayout == INPUT_LAYOUT_NTD) {
        OP_CHECK_IF(xShape.GetDimNum() != 2,
                    OP_LOGE(context, "x must be logical 2D [total_tokens, D] when inputLayout is NTD"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dyShape.GetDimNum() != 3,
                    OP_LOGE(context, "dy must be 3D [N, total_tokens, Dh] when inputLayout is NTD"),
                    return ge::GRAPH_FAILED);
        inputMode = 0;
        totalTokens = xShape.GetDim(DIM_INDEX0);
        D = xShape.GetDim(DIM_INDEX1);
        inputN = dyShape.GetDim(DIM_INDEX0);
        inputHeadDim = dyShape.GetDim(DIM_INDEX2);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != totalTokens ||
                inputN * inputHeadDim != D,
            OP_LOGE(context, "dy must be [N, total_tokens, Dh] with D=N*Dh for logical x [total_tokens, D]"),
            return ge::GRAPH_FAILED);
    } else if (inputLayout == INPUT_LAYOUT_TND) {
        OP_CHECK_IF(xShape.GetDimNum() != 2 || dyShape.GetDimNum() != 2,
                    OP_LOGE(context, "x and dy must be [total_tokens, D] when inputLayout is TND"),
                    return ge::GRAPH_FAILED);
        inputMode = 0;
        totalTokens = xShape.GetDim(DIM_INDEX0);
        D = xShape.GetDim(DIM_INDEX1);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != totalTokens ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != D,
            OP_LOGE(context, "dy shape must match x shape when inputLayout is TND"),
            return ge::GRAPH_FAILED);
        inputN = 1;
        inputHeadDim = D;
    } else if (inputLayout == INPUT_LAYOUT_BSND) {
        OP_CHECK_IF(xShape.GetDimNum() != 3 || dyShape.GetDimNum() != 3,
                    OP_LOGE(context, "x and dy must be [B, T, D] when inputLayout is BSND/BSH"),
                    return ge::GRAPH_FAILED);
        inputMode = 1;
        B = xShape.GetDim(DIM_INDEX0);
        T = xShape.GetDim(DIM_INDEX1);
        D = xShape.GetDim(DIM_INDEX2);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != B ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != T ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX2)) != D,
            OP_LOGE(context, "dy shape must match x shape when inputLayout is BSND/BSH"),
            return ge::GRAPH_FAILED);
        totalTokens = B * T;
        inputN = 1;
        inputHeadDim = D;
    } else {
        OP_LOGE(context, "x shape does not match inputLayout %s", inputLayoutStr);
        return ge::GRAPH_FAILED;
    }

    auto yShapePtr = context->GetOptionalInputShape(Y_INPUT_INDEX);
    if (yShapePtr != nullptr && yShapePtr->GetStorageShape().GetDimNum() > 0) {
        auto yShape = yShapePtr->GetStorageShape();
        OP_CHECK_IF(yShape.GetDimNum() != dyShape.GetDimNum(),
                    OP_LOGE(context, "y and dy must have the same shape"),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < dyShape.GetDimNum(); i++) {
            OP_CHECK_IF(yShape.GetDim(i) != dyShape.GetDim(i),
                        OP_LOGE(context, "y and dy must have the same shape"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(activation != ACTIVATION_NONE,
                    OP_LOGE(context, "y is required when activation is enabled"),
                    return ge::GRAPH_FAILED);
    }
    uint32_t W = wShape.GetDim(DIM_INDEX0);
    OP_CHECK_IF(wShape.GetDimNum() != 2 || static_cast<uint32_t>(wShape.GetDim(DIM_INDEX1)) != D,
                OP_LOGE(context, "weight must be [W, D_total], got dim1 %ld expect %u",
                        wShape.GetDimNum() > DIM_INDEX1 ? wShape.GetDim(DIM_INDEX1) : 0, D),
                return ge::GRAPH_FAILED);

    uint32_t numChunks = 0;
    if (inputMode == 0) {
        auto qslShapePtr = context->GetOptionalInputShape(QUERY_START_LOC_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, qslShapePtr);
        auto qslShape = qslShapePtr->GetStorageShape();
        OP_CHECK_IF(qslShape.GetDimNum() != 1 || qslShape.GetDim(0) < 1,
                    OP_LOGE(context, "queryStartLoc must be 1D and non-empty in varlen mode"),
                    return ge::GRAPH_FAILED);
        B = qslShape.GetDim(0) - 1;
        const gert::CompileTimeTensorDesc *qslDesc = context->GetOptionalInputDesc(QUERY_START_LOC_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
        OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(context, "queryStartLoc dtype must be int64"),
                    return ge::GRAPH_FAILED);
        const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INPUT_INDEX);
        const int64_t *qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
        OP_CHECK_NULL_WITH_CONTEXT(context, qslData);
        OP_CHECK_IF(qslData[0] != 0 || qslData[B] != static_cast<int64_t>(totalTokens),
                    OP_LOGE(context, "queryStartLoc must start with 0 and end with total_tokens"),
                    return ge::GRAPH_FAILED);
        uint32_t maxSeqLen = 0;
        for (uint32_t i = 0; i < B; i++) {
            OP_CHECK_IF(qslData[i + 1] < qslData[i],
                        OP_LOGE(context, "queryStartLoc must be non-decreasing"),
                        return ge::GRAPH_FAILED);
            uint32_t seqLen = static_cast<uint32_t>(qslData[i + 1] - qslData[i]);
            maxSeqLen = std::max(maxSeqLen, seqLen);
            numChunks += CeilDiv(seqLen, static_cast<uint32_t>(64));
        }
        T = maxSeqLen;
    }

    bool useInitialState = false;
    bool useFinalState = false;
    {
        auto initStateShape = context->GetInputShape(INITIAL_STATE_INPUT_INDEX);
        if (initStateShape != nullptr && initStateShape->GetStorageShape().GetDimNum() > 0) {
            auto shape = initStateShape->GetStorageShape();
            OP_CHECK_IF(
                shape.GetDimNum() != 3 ||
                    static_cast<uint32_t>(shape.GetDim(0)) != B ||
                    static_cast<uint32_t>(shape.GetDim(1)) != W ||
                    static_cast<uint32_t>(shape.GetDim(2)) != D,
                OP_LOGE(context, "initial_state must be [B, W, D]=[%u, %u, %u]", B, W, D),
                return ge::GRAPH_FAILED);
            useInitialState = true;
        }
        auto dhtShape = context->GetInputShape(DHT_INPUT_INDEX);
        if (dhtShape != nullptr && dhtShape->GetStorageShape().GetDimNum() > 0) {
            auto shape = dhtShape->GetStorageShape();
            OP_CHECK_IF(
                shape.GetDimNum() != 3 ||
                    static_cast<uint32_t>(shape.GetDim(0)) != B ||
                    static_cast<uint32_t>(shape.GetDim(1)) != W ||
                    static_cast<uint32_t>(shape.GetDim(2)) != D,
                OP_LOGE(context, "dht must be [B, W, D]=[%u, %u, %u]", B, W, D),
                return ge::GRAPH_FAILED);
            useFinalState = true;
        }
    }

    uint64_t ubSize = 0;
    uint64_t sysWorkspaceSize = 0;
    uint32_t coreNum = 0;
    bool isAscend910B = false;
    // A5(DAV_3510) 与 A6(DAV_9201) 共用 arch35 系 regbase kernel（kernel 侧由编译期架构宏分派），
    // 但 host tiling 策略按平台分别给出：每个平台在下方各自的分支里声明自己的 UB 预算口径、
    // 切分开关、BT 上限与输入 staging 能力。不使用跨架构并集布尔值驱动业务策略，
    // 避免后续任一平台的调整错误影响另一个平台。
    struct ArchHostPolicy {
        bool useRegbaseVfBudget; // 普通路径 UB 预算是否计入 regbase 的 dx/db 输出 staging
        bool allowIdSplit;       // 是否允许 i_d 独占切分
        bool allowBtFill;        // 是否允许小 shape 的 BT 台阶回填
        bool allow2DSplit;       // 是否允许 chunk × i_d 二维切分
        bool allowTripleBuffer;  // 快路径输入 staging 是否允许三缓冲
        uint32_t fastBtMax;      // regbase 快路径 BT 上限
        uint32_t fastBtStep;     // regbase 快路径 BT 回退步距
    };
    ArchHostPolicy archPolicy = {}; // 默认全关：910B/A2/A3 保持原有路径
    {
        fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        const auto npuArch = ascendcPlatform.GetCurNpuArch();
        isAscend910B = (npuArch == NpuArch::DAV_2201);
        if (npuArch == NpuArch::DAV_3510) {
            // A5(arch35)：PR #1045 已验证的策略，BT 224/32，仅双缓冲。
            archPolicy = {true, true, true, true, false, DIRECT_FAST_BT_MAX, DIRECT_FAST_BT_STEP};
        } else if (npuArch == NpuArch::DAV_9201) {
            // A6(Ascend960DT)：UB 384KB，BT 上限 240、回退步距 16，允许三缓冲输入 staging。
            archPolicy = {true, true, true, true, true, 240U, 16U};
        }
        coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    const bool useRegbaseVfBudget = archPolicy.useRegbaseVfBudget;

    ubSize = (ubSize - RESERVED_UB) / BYTE_BLOCK * BYTE_BLOCK;

    uint32_t alignDim = (inputLayout == INPUT_LAYOUT_BNSD || inputLayout == INPUT_LAYOUT_NTD) ? inputHeadDim : D;
    OP_CHECK_IF(alignDim % 16 != 0,
                OP_LOGE(context, "%s D %u must be divisible by 16", inputLayoutStr, alignDim),
                return ge::GRAPH_FAILED);
    uint32_t BD = (D % 64 == 0) ? 64 : 16;
    bool wideBlockLayoutSupported =
        (inputLayout != INPUT_LAYOUT_BNSD && inputLayout != INPUT_LAYOUT_NTD) ||
        (inputHeadDim >= 128 && inputHeadDim % 128 == 0);
    bool use910BWideBlock = isAscend910B && isB16Input &&
                            activation == ACTIVATION_SILU &&
                            !useInitialState && !useFinalState &&
                            W == 4 && D % 128 == 0 && wideBlockLayoutSupported;
    if (use910BWideBlock) {
        BD = 128;
    }
    uint32_t numBlksD = D / BD;

    // i_d 切分（useIdSplit）：每个核独占若干 i_d 的全部 chunk，核间无 dw/db 重叠。
    bool useIdSplit = archPolicy.allowIdSplit && numBlksD >= coreNum;

    // A5/A6 各自的 regbase 快路径：b16、BD∈{64,16}、W∈[1,8]、无 initial_state/dht。
    // 是否启用由各平台策略的 useRegbaseVfBudget 决定，A5/A6 互不继承判定结果。
    // fp32 仅支持 BD=16（D 非 64 整数倍）且 useIdSplit 成立：BD=64 的 fp32 未验证，仍走普通路径。
    // activation 支持 SILU/SWISH（同一套 SiLU 反向公式）与 NONE（dy 直读）。
    // BD=64 时行宽已满一个向量寄存器；BD=16 仅当 useIdSplit 成立（核数按 i_d 铺满）才启用：
    // 非 idSplit 下核间并行度只来自 chunk 数，而快路径 BT 最大 224 会把 chunk 数压得很低
    // （例如 T=512、BT=224 只剩 3 个 chunk ⇒ 仅 3 个核），反而比普通路径 BT=64 更慢。
    // BD=16 且 numBlksD==1（即 D==16）时，i_d 方向本来就只有 1 个块，chunk 并行度与普通路径
    // 完全一致（T<=64 时两条路径都只产生 1 个 chunk、1 个核）；不存在上面担心的“快路径 BT 过大
    // 反而压掉 chunk 并行度”的问题，因此可以无 i_d 切分直接启用快路径。
    bool useArch35DirectFast = archPolicy.useRegbaseVfBudget &&
                               ((isB16Input && (BD == 64 ||
                                                (BD == 16 && (useIdSplit || (numBlksD == 1 && T <= 64))))) ||
                                (isFp32Input && BD == 16 && useIdSplit)) &&
                               (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH ||
                                activation == ACTIVATION_NONE) &&
                               !useInitialState && !useFinalState &&
                               W >= 1 && W <= ARCH35_DIRECT_W_MAX;
    bool use910BDirectFast = isAscend910B && isB16Input &&
                             activation == ACTIVATION_SILU &&
                             !useInitialState && !useFinalState &&
                             W == 4 && (BD == 64 || BD == 128);
    uint32_t rawDtypeSize = isB16Input ? B16_DTYPE_SIZE : FP32_DTYPE_SIZE;
    const uint32_t fastBtMax = archPolicy.fastBtMax;
    const uint32_t fastBtStep = archPolicy.fastBtStep;
    uint32_t inputStages = 2;
    auto CalcArch35DirectNeed = [BD, W, rawDtypeSize, isB16Input](uint32_t bt, uint32_t stages) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        // BD=16 时行宽不足一个向量寄存器，VF 内无掩码读会越过行末；kernel 侧对 tempBuf/weightBuf/
        // dwBuf 各预留 64 元素、dbBuf 至少 64 元素（见 InitBuffer）。此处同步计入以保持预算一致。
        uint32_t regPad = (BD < 64U) ? 64U : 0U;
        uint32_t dbAl = (BD < 64U) ? 64U : bdAl;
        uint32_t need = 0;
        need += (stages * btBdAl + 2 * stages * dyBdAl + wBdAl + regPad) * rawDtypeSize;
        if (isB16Input) {
            // b16 额外 3 份原始位宽缓冲：castBuf_（x 的转换 staging、dw 的输出 staging，容量取
            // max(BT*BD, W*BD)）、dxOutBuf_（dx 输出 staging）、dbOutBuf_（db 输出 staging，
            // hasBias 恒为 1）。与 kernel InitBuffer 逐项对齐，漏算会让 BT 选到实际分配已超过
            // UB 的取值。
            need += (std::max(btBdAl, wBdAl) + btBdAl + bdAl) * rawDtypeSize;
        }
        // fp32 计算/输出缓冲：dx + dyAct(temp) + dw + db。
        need += (btBdAl + dyBdAl + wBdAl + 2 * regPad + dbAl) * FP32_DTYPE_SIZE;
        return need;
    };
    auto Calc910BDirectNeed = [BD, W](uint32_t bt) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        uint32_t xRawAl = std::max(btBdAl, wBdAl);
        uint32_t need = (btBdAl + 3 * dyBdAl + 2 * wBdAl + 2 * bdAl) * FP32_DTYPE_SIZE;
        need += (xRawAl + 2 * dyBdAl + 2 * btBdAl) * B16_DTYPE_SIZE;
        return need;
    };
    // 普通路径（chunk 切分 / i_d 切分）的 UB 预算，不含两级归约 stage buffer（该部分在预算内按需追加）。
    // 提取成函数是为了在 BT 选择阶段做「预算超额则减半 BT」的兜底：BT 越小 UB 需求越小。
    auto CalcNormalNeed = [BD, W, rawDtypeSize, isB16Input, useRegbaseVfBudget, activation,
                           useInitialState, useFinalState](uint32_t bt) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        uint32_t calcBdAl = std::max(std::max(btBdAl, wBdAl), dyBdAl);
        uint32_t wBtBdAl = CeilAlign(W * bt * BD, BLOCK_ALIGN_NUM);
        const bool hasWeight = true;
        const bool hasBias = true;
        uint32_t need = (3 * btBdAl + 2 * calcBdAl) * FP32_DTYPE_SIZE;
        need += (dyBdAl - btBdAl) * FP32_DTYPE_SIZE;
        if (hasWeight) need += 2 * wBdAl * FP32_DTYPE_SIZE;
        if (hasWeight && activation == ACTIVATION_NONE) need += wBtBdAl * FP32_DTYPE_SIZE;
        if (hasBias)   need += bdAl * FP32_DTYPE_SIZE;
        if (hasBias && activation == ACTIVATION_NONE) need += btBdAl * FP32_DTYPE_SIZE;
        if (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH)
            need += 2 * dyBdAl * FP32_DTYPE_SIZE;
        if (useInitialState || useFinalState) {
            need += bdAl * FP32_DTYPE_SIZE;
        }
        // b16 的原始位宽缓冲：castBuf_（x/dw 的转换 staging）恒分配；dxOutBuf_/dbOutBuf_（dx/db
        // 输出 staging）只在 regbase 构建中存在（A5/A6 各自由编译期架构宏选择），910B 不分配。
        // 按平台策略 useRegbaseVfBudget 判定，与 kernel InitBuffer 逐项对齐，避免预算少于实际分配。
        if (isB16Input) {
            need += calcBdAl * rawDtypeSize;
            if (useRegbaseVfBudget) {
                need += (btBdAl + bdAl) * rawDtypeSize;
            }
        }
        return need;
    };

    auto CalcNumChunks = [&](uint32_t bt) -> uint32_t {
        if (bt == 0) {
            return 0;
        }
        if (inputMode == 0) {
            const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INPUT_INDEX);
            const int64_t *qslData = qslTensor->GetData<int64_t>();
            uint32_t n = 0;
            for (uint32_t i = 0; i < B; i++) {
                n += CeilDiv(static_cast<uint32_t>(qslData[i + 1] - qslData[i]), bt);
            }
            return n;
        }
        return CeilDiv(T, bt) * B;
    };

    uint32_t BT = 64;
    if (useArch35DirectFast) {
        if (archPolicy.allowTripleBuffer) {
            // 三缓冲（仅 A6）：按「三缓冲预算放得下」且「核内可流水 tile 数 >= 3」确定性选择，
            // 不读取任何环境变量，避免运行期行为随外部环境漂移。
            auto PerCoreTilesAt = [&](uint32_t bt) -> uint32_t {
                uint32_t chunks = CalcNumChunks(bt);
                if (useIdSplit) {
                    return chunks;
                }
                uint32_t coresAtBt = std::min(coreNum, std::max(chunks, 1U));
                return (chunks + coresAtBt - 1) / coresAtBt;
            };
            uint32_t btProbe = std::min(T, fastBtMax);
            while (btProbe > DIRECT_FAST_BT_MIN && CalcArch35DirectNeed(btProbe, 3U) > ubSize) {
                btProbe = (btProbe > DIRECT_FAST_BT_MIN + fastBtStep) ? (btProbe - fastBtStep) : DIRECT_FAST_BT_MIN;
            }
            if (btProbe >= DIRECT_FAST_BT_MIN && PerCoreTilesAt(btProbe) >= 3) {
                inputStages = 3;
            }
        }
        BT = std::min(T, fastBtMax);
        while (BT > DIRECT_FAST_BT_MIN && CalcArch35DirectNeed(BT, inputStages) > ubSize) {
            BT = (BT > DIRECT_FAST_BT_MIN + fastBtStep) ? (BT - fastBtStep) : DIRECT_FAST_BT_MIN;
        }
        if (inputStages == 3 && CalcArch35DirectNeed(BT, inputStages) > ubSize) {
            inputStages = 2;
            BT = std::min(T, fastBtMax);
            while (BT > DIRECT_FAST_BT_MIN && CalcArch35DirectNeed(BT, inputStages) > ubSize) {
                BT = (BT > DIRECT_FAST_BT_MIN + fastBtStep) ? (BT - fastBtStep) : DIRECT_FAST_BT_MIN;
            }
        }
    } else if (use910BDirectFast) {
        uint32_t maxBt = (BD == 128) ? DIRECT_910B_BT_BD128 : DIRECT_910B_BT_BD64;
        BT = std::min(T, maxBt);
        while (BT > 1 && Calc910BDirectNeed(BT) > ubSize) {
            BT--;
        }
    } else if (T < BT) {
        BT = T;
    }
    OP_CHECK_IF(BT == 0 || totalTokens == 0,
                OP_LOGE(context, "empty sequence is not supported: T %u totalTokens %u", T, totalTokens),
                return ge::GRAPH_FAILED);
    numChunks = CalcNumChunks(BT);

    // 小 shape 核数填充：!useIdSplit 时核间并行度只来自 chunk 数，D 方向再多 i_d 也只能每核串行遍历。
    // T 较小时 BT 取到接近 T，numChunks 很小（如 B1T128D256 只有 1 个 chunk ⇒ 仅 1 个核在跑）。
    // 此处仅当 !useIdSplit 且 numBlksD >= 2 且 numChunks < target 时，按 BT_FILL_STEP 台阶减小 BT
    // 以增加 chunk 数，直到达到 target 或触到 BT_FILL_FLOOR。
    // 已满足 numChunks >= target 的 shape 完全不改 BT；useIdSplit 的大 shape（D 方向已铺满核）亦不受影响。
    // numBlksD == 1（D=16/64）时单个 i_d 的 chunk 工作量本就极小，再切分只会增加每 chunk 的固定开销
    // （partial 归约/同步），实测在 ATK 与 msprof 上均出现回归，故跳过。
    constexpr uint32_t BT_FILL_STEP = 32U;
    constexpr uint32_t BT_FILL_FLOOR = 16U;
    constexpr uint32_t BT_FILL_CAP = 8U;
    // 工作量感知的核数上限：目标核数默认仍取 CAP=8，但在 regbase VF「快路径」上再按总元素量
    // totalTokens*D 收紧——快路径单位元素效率高，每核约 BT_FILL_ELEMS_PER_CORE 元素即可吃满并行收益，
    // 工作量小的快路径 shape 不需要更多核。可避免"总工作量不大但 baseline 恰有 4 核"的快路径 shape
    // 被推到 8 核而过切分（例：B4·T64·D128 32768 元素、快路径 BT=64 ⇒ 4 chunk；固定 CAP=8 会推到
    // 8 核，kernel 中位由 ~8.14us 变慢到 ~8.79us/-8%，ATK 侧复现 -2.7%）。
    // 普通路径是 latency-bound（每 chunk 固定开销大），更多核可直接摊薄单 chunk 延迟，其 CAP=8 行为
    // 已由上一轮 ATK/探针验证无回归，故保持不变；下限 BT_FILL_MIN_TARGET 用以保留 1→4 核的最大收益。
    constexpr uint32_t BT_FILL_ELEMS_PER_CORE = 8192U;
    constexpr uint32_t BT_FILL_MIN_TARGET = 4U;
    if (archPolicy.allowBtFill && !useIdSplit && numBlksD >= 2) {
        uint32_t target = std::min(coreNum, BT_FILL_CAP);
        if (useArch35DirectFast) {
            uint64_t totalElems = static_cast<uint64_t>(totalTokens) * D;
            uint64_t byWork = (totalElems >= BT_FILL_ELEMS_PER_CORE)
                                  ? (totalElems / BT_FILL_ELEMS_PER_CORE) : 1ULL;
            target = static_cast<uint32_t>(
                std::min<uint64_t>(target, std::max<uint64_t>(BT_FILL_MIN_TARGET, byWork)));
        }
        while (BT > BT_FILL_FLOOR && numChunks < target) {
            BT = (BT > BT_FILL_FLOOR + BT_FILL_STEP) ? (BT - BT_FILL_STEP) : BT_FILL_FLOOR;
            numChunks = CalcNumChunks(BT);
        }
    }

    // UB 预算兜底：普通路径的 BT 上限固定为 64，极端 shape（超大卷积核宽度 W、或 b16 的 cast/
    // 输出 staging 叠加）即使 BT=64 也放不下。BT 越小每份 buffer 越小，故逐级减半直到预算满足，
    // 保证算子仍能正常执行而不是在 tiling 阶段直接报错。BT 减半只会增加 chunk 数，需同步刷新
    // numChunks（后续 2D 切分与 chunkPerCore 均依赖它）。快路径与 910B 快路径已有各自的 BT 回退。
    if (!useArch35DirectFast && !use910BDirectFast) {
        while (BT > 1U && CalcNormalNeed(BT) > ubSize) {
            BT >>= 1U;
        }
        numChunks = CalcNumChunks(BT);
    }

    // chunk × i_d 二维切分：把 numBlksD 个 i_d 与 numChunks 个 chunk 一起铺到 AIV 核上
    // （blockNum = numBlksD * numChunks），每个核只处理 1 个 i_d 的 1 个 chunk 组，
    // 消除「每核串行遍历全部 i_d」的固定开销（每 i_d 的 CopyInWeight/Duplicate/partial 回写/栅栏）。
    // 小 shape 下该固定开销是主瓶颈：B2T63D80W2 的 aiv_scalar+mte2 合计 ~12us，与 vec 近似串行。
    // 仅当全部 (i_d, chunk) 工作单元数严格小于 AIV 核数时启用：恰好在等于核数（=coreNum）的边界
    // 配置上（如 B8·T256·D32·W2 ⇒ numBlksD=2、numChunks=32、乘积=64）1D 路径本就用满 chunk 维
    // 并行度，2D 再把 partial 写回/归约规模按 numBlksD 放大，反而相对原始基线回归（单算子 TD
    // 实测 -4.4%/-6.1%）；严格小于可排除该边界，同时保留小 shape（乘积远小于核数）的收益。
    bool use2DSplit = archPolicy.allow2DSplit && !useIdSplit && !useArch35DirectFast &&
                      numBlksD >= 2 && numChunks >= 1 &&
                      static_cast<uint64_t>(numBlksD) * numChunks < coreNum;

    uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
    uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
    uint32_t need = 0;
    uint32_t dwStageRows = 0;
    if (useArch35DirectFast) {
        need = CalcArch35DirectNeed(BT, inputStages);
    } else if (use910BDirectFast) {
        need = Calc910BDirectNeed(BT);
    } else {
        need = CalcNormalNeed(BT);

        // 两级归约：SILU/SWISH 分支下把 dw/db 的行归约从"每 chunk 全量归约到 1 行"改为
        // "chunk 内只归约到 R 行并累积进 stage buffer，i_d 末再做 R→1"。
        // 元素工作量基本不变，收益来自每 chunk 归约依赖链/屏障数由 log2(BT) 降到 log2(BT/R)。
        // R 取 2 的幂、≤BT，在 UB 预算内取最大者；放不下则 R=0 退化为逐 chunk 全量归约。
        if (useIdSplit && (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH)) {
            uint32_t rMax = 1;
            while ((rMax << 1) <= BT) rMax <<= 1;
            for (uint32_t r = rMax; r >= 2; r >>= 1) {
                uint32_t dwStageAl = CeilAlign(W * r * BD, BLOCK_ALIGN_NUM);
                uint32_t dbStageAl = CeilAlign(r * BD, BLOCK_ALIGN_NUM);
                uint32_t stageBytes = (dwStageAl + dbStageAl) * FP32_DTYPE_SIZE;
                if (need + stageBytes <= ubSize) {
                    dwStageRows = r;
                    need += stageBytes;
                    break;
                }
            }
        }
    }

    OP_CHECK_IF(need > ubSize,
                OP_LOGE(context, "UB overflow: need %u > available %lu", need, ubSize),
                return ge::GRAPH_FAILED);

    // The kernel entry is not template-specialized by activation. Keep one
    // function entry and let tiling->activation select the runtime path.
    uint64_t tilingKey = CAUSAL_CONV1D_BWD_TPL_ACTIVATION_NONE;

    // Compute block dimension
    //
    // A5(arch35) 上有两个核间切分方案：
    //  - i_d 切分（useIdSplit）：每个核独占若干 i_d 的全部 chunk。由于每个 i_d 的 dw/db
    //    只与自身通道相关，核间无重叠，可直接写回 GM，无需 partial 缓冲与第二趟归约。
    //    该方案对 regbase 快路径与普通路径（含 fp32、W>4、存在 initial_state/dht 等）
    //    均适用，要求 D 方向 block 数足以铺满 AIV 核。
    //  - chunk 切分（原方案）：每个核负责部分 chunk、遍历全部 i_d，需要 partial 归约。
    uint32_t chunkPerCore = 0;
    uint32_t tailChunk = 0;
    uint32_t totalChunks = numChunks;
    uint32_t mainCoreNum = useIdSplit ? coreNum
                          : (use2DSplit ? (numBlksD * numChunks)
                                        : std::min(coreNum, totalChunks));
    if (useIdSplit || use2DSplit) {
        chunkPerCore = 0;
        tailChunk = 0;
    } else if (totalChunks <= mainCoreNum) {
        chunkPerCore = 1;
        tailChunk = 0;
    } else {
        chunkPerCore = totalChunks / mainCoreNum;
        tailChunk = totalChunks % mainCoreNum;
    }

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(mainCoreNum);
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetScheduleMode(1) error"), return ge::GRAPH_FAILED);

    tiling->B = B;
    tiling->T = T;
    tiling->D = D;
    tiling->W = W;
    tiling->activation = activation;
    tiling->hasWeight = 1;
    tiling->hasBias = 1;
    tiling->useInitialState = useInitialState ? 1 : 0;
    tiling->useFinalState = useFinalState ? 1 : 0;
    tiling->inputMode = inputMode;
    tiling->inputLayout = inputLayout;
    tiling->inputN = inputN;
    tiling->inputHeadDim = inputHeadDim;
    tiling->totalTokens = totalTokens;
    tiling->blockNum = mainCoreNum;
    tiling->BT = BT;
    tiling->BD = BD;
    tiling->numBlksD = numBlksD;
    tiling->numChunks = numChunks;
    tiling->batchPerCore = 1;
    tiling->tailBatch = 0;
    tiling->chunkPerCore = chunkPerCore;
    tiling->tailChunk = tailChunk;
    tiling->useIdSplit = useIdSplit ? 1 : 0;
    tiling->use2DSplit = use2DSplit ? 1 : 0;
    tiling->dwStageRows = dwStageRows;
    tiling->inputStages = useArch35DirectFast ? static_cast<int64_t>(inputStages) : 2;

    // useIdSplit 时每个核独占 i_d 并直接写回 dw/db，不再需要 partial 归约缓冲。
    uint64_t partialDwBytes = useIdSplit ? 0ULL
        : static_cast<uint64_t>(mainCoreNum) * numBlksD * wBdAl * FP32_DTYPE_SIZE;
    uint64_t partialDbBytes = useIdSplit ? 0ULL
        : static_cast<uint64_t>(mainCoreNum) * numBlksD * bdAl * FP32_DTYPE_SIZE;
    uint64_t wsSize = sysWorkspaceSize + partialDwBytes + partialDbBytes;
    tiling->workspaceSize = sysWorkspaceSize;

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = wsSize;

    OP_LOGD(context,
            "Tiling result: B[%ld] T[%ld] totalTokens[%ld] D[%ld] W[%ld] inputMode[%ld] activation[%ld] "
            "inputLayout[%ld] inputN[%ld] inputHeadDim[%ld] "
            "BT[%ld] BD[%ld] numBlksD[%ld] numChunks[%ld] "
            "mainCore[%d] chunkPerCore[%ld] tail[%ld] sysWs[%ld] wsSize[%ld] idSplit[%d] split2D[%d] "
            "dwStageRows[%u] inputStages[%ld]",
            B, T, totalTokens, D, W, inputMode, activation,
            inputLayout, inputN, inputHeadDim,
            BT, BD, numBlksD, numChunks,
            mainCoreNum, chunkPerCore, tailChunk, sysWorkspaceSize, wsSize,
            useIdSplit ? 1 : 0, use2DSplit ? 1 : 0, dwStageRows,
            static_cast<int64_t>(useArch35DirectFast ? inputStages : 2));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCausalConv1dBwd(gert::TilingParseContext *context)
{
    OP_LOGD(context, "Enter TilingParseForCausalConv1dBwd.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1dBwd)
    .Tiling(CausalConv1dBwdTilingFunc)
    .TilingParse<CausalConv1dBwdCompileInfo>(TilingParseForCausalConv1dBwd);

} // namespace optiling
