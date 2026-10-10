/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file pre_process_fwd_kernel_merged_tiling_processor.h
 * \brief tiling 计算主体（header-only，§2）：由 op_host/..._tiling.cpp 的注册入口调用。
 */

#ifndef PRE_PROCESS_FWD_KERNEL_MERGED_TILING_PROCESSOR_H
#define PRE_PROCESS_FWD_KERNEL_MERGED_TILING_PROCESSOR_H

#include "pre_process_fwd_kernel_merged_tiling.h"

// 注意：调用方 op_host/*_tiling.cpp 本身已在 namespace optiling 内 ⇒ 本头**不再**包 namespace，
//       否则会解析成 optiling::optiling::PreProcessFwdTilingProcessor。
inline ge::graphStatus PreProcessFwdTilingProcessor(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<PreProcessFwdKernelMergedTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto kShapePtr = context->GetInputShape(INPUT_K_IDX);
    auto wShapePtr = context->GetInputShape(INPUT_W_IDX);
    auto uShapePtr = context->GetInputShape(INPUT_U_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, uShapePtr);
    const gert::Shape kShape = kShapePtr->GetStorageShape();
    const gert::Shape uShape = uShapePtr->GetStorageShape();

    const int64_t B = kShape.GetDim(0);
    const int64_t Hk = kShape.GetDim(1);
    const int64_t T = kShape.GetDim(2);
    const int64_t K = kShape.GetDim(3);
    const int64_t Hv = uShape.GetDim(1);
    const int64_t V = uShape.GetDim(3);
    const int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));

    OP_CHECK_IF(B != 1, OP_LOGE(context->GetNodeName(), "B must be 1 (varlen packed), got %ld", B),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(K != PPFM_FIXED_K || V != PPFM_FIXED_V,
                OP_LOGE(context->GetNodeName(), "K/V must be 128, got %ld/%ld", K, V),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(chunkSize != PPFM_FIXED_CHUNK,
                OP_LOGE(context->GetNodeName(), "chunk_size must be 64, got %ld", chunkSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Hk <= 0 || Hv <= 0 || Hv % Hk != 0,
                OP_LOGE(context->GetNodeName(), "GVA requires Hv %% Hk == 0, got Hk=%ld Hv=%ld", Hk, Hv),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(wShapePtr->GetStorageShape().GetDim(1) != Hv,
                OP_LOGE(context->GetNodeName(), "w head dim must equal Hv"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(uShape.GetDim(0) != 1 || uShape.GetDim(2) != T || uShape.GetDim(3) != V,
                OP_LOGE(context->GetNodeName(), "u must be [1,Hv,T,V] and match k's T"), return ge::GRAPH_FAILED);

    // gate 二选一；DPLR（bg / v）本版本不支持：显式拒绝，保证 TilingKey 3（USE_BG）永不被下发
    auto gTensor = context->GetOptionalInputTensor(INPUT_G_IDX);
    auto gkTensor = context->GetOptionalInputTensor(INPUT_GK_IDX);
    auto bgTensor = context->GetOptionalInputTensor(INPUT_BG_IDX);
    const bool hasG = gTensor != nullptr;
    const bool hasGk = gkTensor != nullptr;
    const bool hasBg = bgTensor != nullptr;
    const bool hasV = context->GetOptionalInputTensor(INPUT_V_IDX) != nullptr;
    OP_CHECK_IF(hasG == hasGk, OP_LOGE(context->GetNodeName(), "exactly one of g / gk must be given"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hasBg,
                OP_LOGE(context->GetNodeName(),
                        "bg is not supported: DPLR is not implemented in this release (GDN/KDA only)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hasV,
                OP_LOGE(context->GetNodeName(),
                        "v is not supported: it is DPLR-only; GDN/KDA takes the values from u"),
                return ge::GRAPH_FAILED);
    const int64_t gateMode = hasGk ? GDN::PPFM_GATE_USE_GK : GDN::PPFM_GATE_USE_G;
    const ge::DataType gateDtype = hasG ? gTensor->GetDataType() : gkTensor->GetDataType();

    // g / gk 的 shape 必须与 docs/api.md §2.1 的契约一致：
    //   g  [1, Hv, T]      gk [1, Hv, T, K]     （gate 按 value head）
    if (hasG) {
        const gert::StorageShape *gShapePtr = context->GetOptionalInputShape(INPUT_G_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, gShapePtr);
        const gert::Shape gShape = gShapePtr->GetStorageShape();
        OP_CHECK_IF(gShape.GetDimNum() != 3,
                    OP_LOGE(context->GetNodeName(), "g must be [1,Hv,T], got dim num %ld",
                            static_cast<int64_t>(gShape.GetDimNum())),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gShape.GetDim(0) != 1 || gShape.GetDim(1) != Hv || gShape.GetDim(2) != T,
                    OP_LOGE(context->GetNodeName(), "g must be [1,Hv,T] matching u's Hv and k's T"),
                    return ge::GRAPH_FAILED);
    } else {
        const gert::StorageShape *gkShapePtr = context->GetOptionalInputShape(INPUT_GK_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, gkShapePtr);
        const gert::Shape gkShape = gkShapePtr->GetStorageShape();
        OP_CHECK_IF(gkShape.GetDimNum() != 4,
                    OP_LOGE(context->GetNodeName(), "gk must be [1,Hv,T,K], got dim num %ld",
                            static_cast<int64_t>(gkShape.GetDimNum())),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gkShape.GetDim(0) != 1 || gkShape.GetDim(1) != Hv || gkShape.GetDim(2) != T ||
                        gkShape.GetDim(3) != K,
                    OP_LOGE(context->GetNodeName(), "gk must be [1,Hv,T,K] matching u's Hv and k's T/K"),
                    return ge::GRAPH_FAILED);
    }

    // cu_seqlens：host int 数组，必给；校验 0 <= cu[0] < ... < cu[-1] <= T
    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    OP_CHECK_IF(cuSeqlensTensor == nullptr,
                OP_LOGE(context->GetNodeName(), "cu_seqlens is required (varlen only)"), return ge::GRAPH_FAILED);
    const int64_t cuNumel = static_cast<int64_t>(cuSeqlensTensor->GetShapeSize());
    OP_CHECK_IF(cuNumel < 2, OP_LOGE(context->GetNodeName(), "cu_seqlens must have >= 2 elements"),
                return ge::GRAPH_FAILED);
    const int64_t *cuData = cuSeqlensTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, cuData);
    for (int64_t i = 0; i < cuNumel; ++i) {
        OP_CHECK_IF(cuData[i] < 0 || cuData[i] > T,
                    OP_LOGE(context->GetNodeName(), "cu_seqlens[%ld]=%ld out of range [0,%ld]", i, cuData[i], T),
                    return ge::GRAPH_FAILED);
        if (i > 0) {
            OP_CHECK_IF(cuData[i] <= cuData[i - 1],
                        OP_LOGE(context->GetNodeName(), "cu_seqlens must be strictly increasing at %ld", i),
                        return ge::GRAPH_FAILED);
        }
    }
    const int64_t nSeq = cuNumel - 1;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t aicNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
    // ---- P5 列块切分：把"链"按列切成 colSplit 份（h 切 V 列、m 切 K 列，两侧独立） ----
    // 实测（950，msprof op Task Duration）：
    //   * Nwork=8 / aic=28：切 2 份仍是 1 波，16 个核干活 ⇒ 783 → 653 µs（−16.7%）✅
    //   * Nwork=32 / aic=28：切 2 份把波数从 2 抬到 3，而每项成本**并不减半**
    //     （staging/left/decay 被两半重复、状态更新的逐行回路条数不变）
    //     ⇒ 4027 → 4976 µs（+23.6%）❌
    // 因此只在一波放得下时切：2·Nwork ≤ aicNum。cube 的 N 维此时是 64（仍够宽）。
    const int64_t hwItems = nSeq * Hv;
    int64_t colSplit = 1;
    if (aicNum > 0 && hwItems * 2 <= aicNum) {
        colSplit = 2;
    }
    // 测试/调试钩子：环境变量可强制列块切分因子（1 或 2），用于位级 A/B 与回归。
    if (const char *forceSplit = std::getenv("PPFM_FORCE_COLSPLIT")) {
        const int64_t v = std::atoll(forceSplit);
        if (v == 1 || v == 2) {
            colSplit = v;
        }
    }
    // ---- R21 混合调度 ----
    // 链数 > 核数 时 round-robin 会让 (hwItems % aicNum) 个核跑 2 条链 ⇒ 关键路径 = 2 波。
    // 把余数链按列切成 S 片（成本模型 c(s)=α+(1-α)/s，实测 α≈0.72），
    // S 片分给 S*r 个核的第二个任务 ⇒ 尾巴从「整宽」变「1/S 宽」。约束 r*S <= aicNum。
    // 实测：模型 case（hwItems=32/A=28）S=2 −7.2%、S=4 −9.84%（vs colSplit=1）。
    // ⚠ 平台门控（2026-09-30）：hybridS>1 的分片任务路径在 A2/910B 上有低频偶发，
    //   实测只在 **S=2** 的组合上被观察到（gdn-hy30：hwItems=30/A=20 ⇒ S=2；
    //   见 outputs/PPFM_910B_A2FIX_OPS_20260930.md §6.3）。
    //   而 S=4 在 A2 上已有两组证据：
    //     ① `kda T=256/HK=HV=64` 强制 S=4 × 30 个独立进程 vs CPU 标杆：0 失败、单一取值；
    //     ② `dump_hm` 的 `gdn-hy64`（A2 上 rem=4 ⇒ S=4）与 S=1 的结果**逐位相同**
    //        （列切分不改任何 MMAD 的 K 累加顺序）。
    //   ⇒ A5/ASCEND950 全开；A2/A3 **只放行 S=4**，S=2 仍禁用（回收 KDA 档约 5%）。
    //   复现/诊断：环境变量 PPFM_HYBRID_S 在门控之后生效，可强制打开任意 S。
    const bool hybridAll =
        ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    int64_t hybridS = 1;
    int64_t hybridBase = 0;
    if (colSplit == 1 && aicNum > 1 && hwItems > aicNum) {
        const int64_t base = (hwItems / aicNum) * aicNum;
        const int64_t rem = hwItems - base;
        int64_t s = 1;
        if (rem > 0 && rem < aicNum) {
            // 片数 = V=128 的可整除因子（1/2/4/8），且受「第 2 波核数」限制 r*s <= aicNum。
            // 成本模型 c(s) = α + (1-α)/s，实测 α≈0.72 ⇒ s 越大越好，但 cb_=16 风险大，封顶 4。
            const int64_t sMax = aicNum / rem;
            s = (sMax >= 4) ? 4 : ((sMax >= 2) ? 2 : 1);
            // A2/A3：S=2 的分片任务路径有低频偶发（gdn-hy30），S=4 已验证 ⇒ 只放行 S=4。
            if (!hybridAll && s == 2) {
                s = 1;
            }
        }
        if (s >= 2) {
            hybridS = s;
            hybridBase = base;
        }
    }
    // 调试钩子：PPFM_HYBRID_S=0 关掉、=2/4 强制指定（用于上板 A/B）。
    if (const char *forceHyb = std::getenv("PPFM_HYBRID_S")) {
        const int64_t v = std::atoll(forceHyb);
        if (v <= 1) {
            hybridS = 1;
            hybridBase = 0;
        } else if ((v == 2 || v == 4) && aicNum > 1 && hwItems > aicNum) {
            const int64_t base = (hwItems / aicNum) * aicNum;
            const int64_t rem = hwItems - base;
            if (rem > 0 && rem * v <= aicNum) {
                hybridS = v;
                hybridBase = base;
            }
        }
    }
    const int64_t taskNum = (hybridS > 1)
        ? (hybridBase + (hwItems - hybridBase) * hybridS)
        : (hwItems * colSplit);

    tiling->B = B;
    tiling->Hk = Hk;
    tiling->Hv = Hv;
    tiling->hvPerHk = Hv / Hk;
    tiling->T = T;
    tiling->K = K;
    tiling->V = V;
    tiling->chunkSize = chunkSize;
    tiling->nSeq = nSeq;
    tiling->gateMode = gateMode;
    tiling->gateDtype = DtypeToEnum(gateDtype);
    tiling->isVariedLen = 1;
    tiling->usedAicNum = (taskNum < aicNum) ? taskNum : aicNum;
    tiling->taskNum = taskNum;
    tiling->colSplit = colSplit;
    tiling->hybridS = hybridS;
    tiling->hybridBase = hybridBase;

    OP_CHECK_IF(context->GetRawTilingData() == nullptr ||
                    context->GetRawTilingData()->GetCapacity() <
                        sizeof(GDN::PreProcessFwdKernelMergedTilingData),
                OP_LOGE(context->GetNodeName(), "tiling buffer too small for tcube tiling"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!BuildCubeTiling(ascendcPlatform, tiling),
                OP_LOGE(context->GetNodeName(), "build cube(TCubeTiling) tiling failed"),
                return ge::GRAPH_FAILED);

    context->SetTilingKey(static_cast<uint32_t>(gateMode) + 1U);
    context->SetBlockDim(static_cast<uint32_t>(tiling->usedAicNum));
    context->SetScheduleMode(1);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // 系统 workspace + 每个工作项一块 GM 临时区（kernel 侧口径见 op_kernel/..._struct.h）
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() +
                          static_cast<size_t>(tiling->usedAicNum) *
                              static_cast<size_t>(GDN::PPFM_CORE_WS_BYTES);
    PrintTiling(context, *tiling);
    return ge::GRAPH_SUCCESS;
}


#endif  // PRE_PROCESS_FWD_KERNEL_MERGED_TILING_PROCESSOR_H
