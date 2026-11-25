/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd_tiling_arch35.cpp
 * \brief Causal Conv1D backward A5(arch35/DAV_3510) dedicated tiling implementation
 *
 * 本文件为 A5 独立 tiling 实现，与共享 tiling 入口解耦：共享入口负责解析/校验并填充
 * CausalConv1dBwdHostTilingArgs，然后在 DAV_3510 上调用本文件实现。
 */

#include "op_common/log/log.h"
#include "util/math_util.h"
#include "causal_conv1d_bwd_tiling_arch35.h"
#include <algorithm>
#include <string>

namespace {
    constexpr const char *OP_NAME = "CausalConv1dBwd";
    constexpr uint32_t RESERVED_UB = 16 * 1024;
    constexpr uint32_t B16_DTYPE_SIZE = 2U;
    constexpr uint32_t FP32_DTYPE_SIZE = 4U;
    constexpr uint32_t BLOCK_ALIGN_NUM = 8U;
    constexpr uint32_t DIRECT_FAST_BT_MAX = 224U;
    constexpr uint32_t DIRECT_FAST_BT_MIN = 64U;
    constexpr uint32_t DIRECT_FAST_BT_STEP = 32U;
    constexpr uint32_t ARCH35_DIRECT_W_MAX = 8U;
    constexpr uint32_t ACTIVATION_NONE = 0;
    constexpr uint32_t ACTIVATION_SILU = 1;
    constexpr uint32_t ACTIVATION_SWISH = 2;
    constexpr uint32_t QUERY_START_LOC_INPUT_INDEX = 6;
}

namespace optiling {
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

ge::graphStatus CausalConv1dBwdTilingA5(const CausalConv1dBwdHostTilingArgs &args,
                                        CausalConv1dBwdTilingData *tiling)
{
    const uint32_t B = args.B;
    const uint32_t T = args.T;
    const uint32_t D = args.D;
    const uint32_t W = args.W;
    const uint32_t activation = args.activation;
    const uint32_t inputMode = args.inputMode;
    const uint32_t inputLayout = args.inputLayout;
    const uint32_t inputN = args.inputN;
    const uint32_t inputHeadDim = args.inputHeadDim;
    const uint32_t totalTokens = args.totalTokens;
    const bool isB16Input = args.isB16Input;
    const bool isFp32Input = args.isFp32Input;
    const bool useInitialState = args.useInitialState;
    const bool useFinalState = args.useFinalState;
    const uint32_t coreNum = args.coreNum;
    const uint64_t ubSize = args.ubSize;
    const uint64_t sysWorkspaceSize = args.sysWorkspaceSize;
    gert::TilingContext *context = args.context;

    // A5(arch35/DAV_3510) 的架构策略固定为：regbase 预算开启、允许 id 切分/BT 填充/2D 切分，
    // 不支持三缓冲（inputStages 恒为 2），直通 BT 上限/步长分别为 224/32。

    uint32_t BD = (D % 64 == 0) ? 64 : 16;
    uint32_t numBlksD = D / BD;

    // i_d 切分（useIdSplit）：每个核独占若干 i_d 的全部 chunk，核间无 dw/db 重叠。
    bool useIdSplit = numBlksD >= coreNum;

    bool arch35DirectPathCompatible = (W >= 1 && W <= ARCH35_DIRECT_W_MAX);
    if (activation != ACTIVATION_NONE) {
        arch35DirectPathCompatible = arch35DirectPathCompatible && (W == 4 || numBlksD <= 4 || useIdSplit);
    }
    bool useArch35DirectFast =
        ((isB16Input && (BD == 64 || (BD == 16 && (useIdSplit || (numBlksD == 1 && T <= 64))))) ||
         (isFp32Input && (BD == 64 || (BD == 16 && useIdSplit)))) &&
        (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH || activation == ACTIVATION_NONE) &&
        !useInitialState && !useFinalState && arch35DirectPathCompatible;
    uint32_t rawDtypeSize = isB16Input ? B16_DTYPE_SIZE : FP32_DTYPE_SIZE;

    auto CalcArch35DirectNeed = [BD, W, rawDtypeSize, isB16Input](uint32_t bt, uint32_t stages) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        uint32_t regPad = (BD < 64U) ? 64U : 0U;
        uint32_t dbAl = (BD < 64U) ? 64U : bdAl;
        uint32_t need = 0;
        need += (stages * btBdAl + 2 * stages * dyBdAl + wBdAl + regPad) * rawDtypeSize;
        if (isB16Input) {
            need += (std::max(btBdAl, wBdAl) + btBdAl + bdAl) * rawDtypeSize;
        }
        need += (btBdAl + dyBdAl + wBdAl + 2 * regPad + dbAl) * FP32_DTYPE_SIZE;
        return need;
    };
    // 普通路径（chunk 切分 / i_d 切分）的 UB 预算，不含两级归约 stage buffer（该部分在预算内按需追加）。
    auto CalcNormalNeed = [BD, W, rawDtypeSize, isB16Input, activation,
                           useInitialState, useFinalState](uint32_t bt) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        uint32_t calcBdAl = std::max(std::max(btBdAl, wBdAl), dyBdAl);
        uint32_t wBtBdAl = CeilAlign(W * bt * BD, BLOCK_ALIGN_NUM);
        uint32_t need = (3 * btBdAl + 2 * calcBdAl) * FP32_DTYPE_SIZE;
        need += (dyBdAl - btBdAl) * FP32_DTYPE_SIZE;
        need += 2 * wBdAl * FP32_DTYPE_SIZE;
        if (activation == ACTIVATION_NONE) need += wBtBdAl * FP32_DTYPE_SIZE;
        need += bdAl * FP32_DTYPE_SIZE;
        if (activation == ACTIVATION_NONE) need += btBdAl * FP32_DTYPE_SIZE;
        if (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH)
            need += 2 * dyBdAl * FP32_DTYPE_SIZE;
        if (useInitialState || useFinalState) {
            need += bdAl * FP32_DTYPE_SIZE;
        }
        if (isB16Input) {
            need += calcBdAl * rawDtypeSize;
            need += (btBdAl + bdAl) * rawDtypeSize; // useRegbaseVfBudget = true
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
        // A5 不支持三缓冲：inputStages 恒为 2，无需三缓冲探测。
        BT = std::min(T, DIRECT_FAST_BT_MAX);
        while (BT > DIRECT_FAST_BT_MIN && CalcArch35DirectNeed(BT, 2U) > ubSize) {
            BT = (BT > DIRECT_FAST_BT_MIN + DIRECT_FAST_BT_STEP) ? (BT - DIRECT_FAST_BT_STEP)
                                                                 : DIRECT_FAST_BT_MIN;
        }
    } else if (T < BT) {
        BT = T;
    }
    OP_CHECK_IF(
        BT == 0 || totalTokens == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "sequence length",
                                              "T=" + std::to_string(T) + "; BT=" + std::to_string(BT) +
                                                  "; totalTokens=" + std::to_string(totalTokens),
                                              "expected BT > 0 and totalTokens > 0; empty sequence is not supported"),
        return ge::GRAPH_FAILED);
    uint32_t numChunks = CalcNumChunks(BT);

    constexpr uint64_t ID_SPLIT_ELEM_BUDGET = 16384ULL;
    if (useArch35DirectFast && !useIdSplit && numBlksD >= 2U && numChunks >= 1U &&
        (numChunks <= numBlksD ||
         (numChunks <= 4U * numBlksD &&
          static_cast<uint64_t>(totalTokens) * D <= ID_SPLIT_ELEM_BUDGET))) {
        useIdSplit = true;
    }

    constexpr uint32_t BT_FILL_STEP = 32U;
    constexpr uint32_t BT_FILL_FLOOR = 16U;
    constexpr uint32_t BT_FILL_CAP = 8U;
    constexpr uint32_t BT_FILL_CAP_FAST = 16U;
    constexpr uint32_t BT_FILL_ELEMS_PER_CORE = 8192U;
    constexpr uint32_t BT_FILL_MIN_TARGET = 4U;
    if (!useIdSplit && numBlksD >= 2) {
        uint32_t target = std::min(coreNum, useArch35DirectFast ? BT_FILL_CAP_FAST : BT_FILL_CAP);
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

    if (!useIdSplit && !useArch35DirectFast && numBlksD >= 2 && BT == 64U && numChunks < coreNum) {
        uint32_t numChunksHalf = CalcNumChunks(BT >> 1U);
        if (numChunksHalf == coreNum) {
            BT >>= 1U;
            numChunks = numChunksHalf;
        }
    }

    if (!useArch35DirectFast) {
        while (BT > 1U && CalcNormalNeed(BT) > ubSize) {
            BT >>= 1U;
        }
        numChunks = CalcNumChunks(BT);
    }

    // chunk × i_d 二维切分：把 numBlksD 个 i_d 与 numChunks 个 chunk 一起铺到 AIV 核上
    const bool twoDByUnit = static_cast<uint64_t>(numBlksD) * numChunks < coreNum;
    uint32_t splitG = 0;
    bool use2DSplit = false;
    if (!useIdSplit && !useArch35DirectFast && numBlksD >= 2 && numChunks >= 1) {
        if (twoDByUnit) {
            splitG = numChunks; // 每核恰好 1 个 (i_d, chunk) 单元，沿用历史行为
            use2DSplit = true;
        } else if (numBlksD >= 4 && numChunks < coreNum) {
            splitG = std::min(numChunks, coreNum / numBlksD);
            use2DSplit = (splitG >= 2); // splitG<2 时 2D 无额外并行度，退回 1D
        }
    }

    uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
    uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
    uint32_t need = 0;
    uint32_t dwStageRows = 0;
    if (useArch35DirectFast) {
        need = CalcArch35DirectNeed(BT, 2U);
    } else {
        need = CalcNormalNeed(BT);

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
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s",
                        std::string("UB capacity: requiredBytes=" + std::to_string(need) +
                                    "; availableBytes=" + std::to_string(ubSize) +
                                    "; expected requiredBytes <= availableBytes for selected tile")
                            .c_str()),
                return ge::GRAPH_FAILED);

    // The kernel entry is not template-specialized by activation. Keep one
    // function entry and let tiling->activation select the runtime path.
    uint64_t tilingKey = CAUSAL_CONV1D_BWD_TPL_ACTIVATION_NONE;

    uint32_t chunkPerCore = 0;
    uint32_t tailChunk = 0;
    uint32_t totalChunks = numChunks;
    uint32_t mainCoreNum = useIdSplit ? std::min(coreNum, numBlksD)
                          : (use2DSplit ? (numBlksD * splitG)
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

    constexpr uint32_t RED_STAGE_ELEMS = 2048U; // 2048 个 fp32 = 8KB，与 kernel 侧同一上限
    uint32_t redStageRows = 0;
    if (mainCoreNum > 1U && !useIdSplit && !use2DSplit) {
        uint32_t rowsCap = RED_STAGE_ELEMS / (W * BD);
        if (rowsCap < 1U) {
            rowsCap = 1U;
        }
        redStageRows = (mainCoreNum < rowsCap) ? mainCoreNum : rowsCap;
        if (redStageRows < 2U) {
            redStageRows = 0U;
        }
    }
    if (redStageRows >= 2U) {
        uint64_t redBytes =
            static_cast<uint64_t>(CeilAlign(redStageRows * W * BD, BLOCK_ALIGN_NUM)) * FP32_DTYPE_SIZE;
        OP_CHECK_IF(redBytes > RESERVED_UB,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "ubSize", std::to_string(redBytes).c_str(),
                        "the partial-reduce staging buffer exceeds the reserved UB region"),
                    return ge::GRAPH_FAILED);
    }

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(mainCoreNum);
    const auto scheduleRc = context->SetScheduleMode(1);
    OP_CHECK_IF(scheduleRc != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s",
                        std::string("SetScheduleMode: mode=1; rc=" + std::to_string(scheduleRc) +
                                    "; expected rc=0 (GRAPH_SUCCESS)")
                            .c_str()),
                return ge::GRAPH_FAILED);

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
    tiling->redStageRows = redStageRows;
    tiling->inputStages = 2;

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
            "Tiling(A5) result: B[%ld] T[%ld] totalTokens[%ld] D[%ld] W[%ld] inputMode[%ld] activation[%ld] "
            "inputLayout[%ld] inputN[%ld] inputHeadDim[%ld] "
            "BT[%ld] BD[%ld] numBlksD[%ld] numChunks[%ld] "
            "mainCore[%d] chunkPerCore[%ld] tail[%ld] sysWs[%ld] wsSize[%ld] idSplit[%d] split2D[%d] "
            "dwStageRows[%u] redStageRows[%u] inputStages[%ld]",
            B, T, totalTokens, D, W, inputMode, activation,
            inputLayout, inputN, inputHeadDim,
            BT, BD, numBlksD, numChunks,
            mainCoreNum, chunkPerCore, tailChunk, sysWorkspaceSize, wsSize,
            useIdSplit ? 1 : 0, use2DSplit ? 1 : 0, dwStageRows, redStageRows,
            static_cast<int64_t>(2));

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
