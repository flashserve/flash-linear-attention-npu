/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_bwd_kernel_merged_tiling_processor.h
 * \brief
 * Tiling processor for pre_process_bwd_kernel_merged, decoupled from gert::TilingContext so that the same
 * validation / core partition / workspace rules can be reused by the aclnn entry.
 */

#ifndef PRE_PROCESS_BWD_KERNEL_MERGED_TILING_PROCESSOR_H
#define PRE_PROCESS_BWD_KERNEL_MERGED_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "tiling_base/tiling_base.h"
#include "../../op_kernel/pre_process_bwd_kernel_merged_struct.h"

using CP::PreProcessBwdKernelMergedTilingData;

namespace optiling {

static constexpr size_t CDHP_INPUT_Q_IDX = 0;
static constexpr size_t CDHP_INPUT_K_IDX = 1;
static constexpr size_t CDHP_INPUT_W_IDX = 2;
static constexpr size_t CDHP_INPUT_DO_IDX = 3;
static constexpr size_t CDHP_INPUT_DV_IDX = 4;
static constexpr size_t CDHP_INPUT_G_IDX = 5;
static constexpr size_t CDHP_INPUT_GK_IDX = 6;
static constexpr size_t CDHP_INPUT_CU_SEQLENS_IDX = 7;

static constexpr size_t CDHP_ATTR_SCALE_IDX = 0;
static constexpr size_t CDHP_ATTR_CHUNK_SIZE_IDX = 1;

static constexpr size_t CDHP_DIM_0 = 0;
static constexpr size_t CDHP_DIM_1 = 1;
static constexpr size_t CDHP_DIM_2 = 2;
static constexpr size_t CDHP_DIM_3 = 3;

// workspace 子区按 512 B 对齐，避免各子区首地址跨 cache line
static constexpr uint64_t CDHP_WS_ALIGN = 512;
// V0 chunk slot 的流水深度；仅影响 workspace 大小与 ping-pong 度，不影响语义
static constexpr uint64_t CDHP_SLOT_NUM = 2;
// v2.1 双 window 流水：与 kernel 侧 pre_process_bwd_kernel_merged_policy.h 的 CDHP_WINDOW_COUNT 一致
//（op_host 不包含 kernel 头文件，因此这里单独声明）。
static constexpr uint64_t CDHP_TILING_WINDOW_COUNT = 2;
static constexpr uint64_t CDHP_META_WS_BYTES = 512;
static constexpr uint64_t CDHP_HALF_DTYPE_SIZE = 2;
static constexpr uint64_t CDHP_FP32_DTYPE_SIZE = 4;
// 模型 dtype（BF16/FP16）统一按 2 B 规划；dhm 与链式累加固定 FP32
static constexpr uint64_t CDHP_MODEL_DTYPE_SIZE = 2;

struct PreProcessBwdKernelMergedTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *wShape;
    const gert::StorageShape *doShape;
    const gert::StorageShape *dvShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *gkShape;
    const gert::StorageShape *cuSeqlensShape;
    ge::DataType qDtype;
    ge::DataType gDtype;
    ge::DataType gkDtype;  // gk 的 dtype 必须单独取：g 缺失时 gDtype 会落到缺省值，不能用它校验 gk
    bool hasG;
    bool hasGk;
    bool hasScaleAttr;
    double scaleAttr;
    int32_t chunkSize;
    uint32_t totalCoreNum;
    // 一个 AI Core block 内配对几个 AIV：A5（arch35）走 MIX_AIC_1_2 ⇒ 2，A2/A3（arch22）⇒ 1。
    // 每个 AIV 独立承包一个 head 的整条逆序链，因此 workspace 平面要按 blockDim * aivPerBlock 切片。
    uint32_t aivPerBlock;
    size_t sysWorkspaceSize;
};

class PreProcessBwdKernelMergedTilingProcessor {
public:
    explicit PreProcessBwdKernelMergedTilingProcessor(PreProcessBwdKernelMergedTilingContext &ctx,
                                                     PreProcessBwdKernelMergedTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    uint64_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

    uint64_t GetBlockDim() const
    {
        return blockDim_;
    }

    // gate 模式（模板参数之一，见 op_kernel/<op>_tiling_key.h 的 ASCENDC_TPL_ARGS_DECL）
    uint32_t GetGateMode() const
    {
        return gateMode_;
    }

    template <typename T>
    static T CeilDiv(T a, T b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }

    static uint64_t AlignUp(uint64_t value, uint64_t align)
    {
        return CeilDiv(value, align) * align;
    }

    ge::graphStatus Init()
    {
        const gert::Shape qShape = ctx_.qShape->GetStorageShape();
        const gert::Shape kShape = ctx_.kShape->GetStorageShape();
        const gert::Shape wShape = ctx_.wShape->GetStorageShape();
        const gert::Shape doShape = ctx_.doShape->GetStorageShape();
        const gert::Shape dvShape = ctx_.dvShape->GetStorageShape();

        B_ = static_cast<uint64_t>(qShape.GetDim(CDHP_DIM_0));
        Hk_ = static_cast<uint64_t>(qShape.GetDim(CDHP_DIM_1));
        T_ = static_cast<uint64_t>(qShape.GetDim(CDHP_DIM_2));
        K_ = static_cast<uint64_t>(qShape.GetDim(CDHP_DIM_3));
        Hv_ = static_cast<uint64_t>(doShape.GetDim(CDHP_DIM_1));
        V_ = static_cast<uint64_t>(doShape.GetDim(CDHP_DIM_3));

        OP_CHECK_IF(kShape.GetDim(CDHP_DIM_0) != static_cast<int64_t>(B_) ||
                        kShape.GetDim(CDHP_DIM_1) != static_cast<int64_t>(Hk_) ||
                        kShape.GetDim(CDHP_DIM_2) != static_cast<int64_t>(T_) ||
                        kShape.GetDim(CDHP_DIM_3) != static_cast<int64_t>(K_),
                    OP_LOGE(ctx_.nodeName, "k must match q as [B,Hk,T,K]."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(wShape.GetDim(CDHP_DIM_0) != static_cast<int64_t>(B_) ||
                        wShape.GetDim(CDHP_DIM_1) != static_cast<int64_t>(Hv_) ||
                        wShape.GetDim(CDHP_DIM_2) != static_cast<int64_t>(T_) ||
                        wShape.GetDim(CDHP_DIM_3) != static_cast<int64_t>(K_),
                    OP_LOGE(ctx_.nodeName,
                            "w must be [B,Hv,T,K]; expect [%lu,%lu,%lu,%lu], got [%ld,%ld,%ld,%ld].", B_, Hv_, T_, K_,
                            wShape.GetDim(CDHP_DIM_0), wShape.GetDim(CDHP_DIM_1), wShape.GetDim(CDHP_DIM_2),
                            wShape.GetDim(CDHP_DIM_3)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(doShape.GetDim(CDHP_DIM_0) != static_cast<int64_t>(B_) ||
                        doShape.GetDim(CDHP_DIM_2) != static_cast<int64_t>(T_),
                    OP_LOGE(ctx_.nodeName, "d_o batch/time must match q."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dvShape.GetDim(CDHP_DIM_0) != doShape.GetDim(CDHP_DIM_0) ||
                        dvShape.GetDim(CDHP_DIM_1) != static_cast<int64_t>(Hv_) ||
                        dvShape.GetDim(CDHP_DIM_2) != doShape.GetDim(CDHP_DIM_2) ||
                        dvShape.GetDim(CDHP_DIM_3) != static_cast<int64_t>(V_),
                    OP_LOGE(ctx_.nodeName, "dv must have the same [B,Hv,T,V] shape as d_o."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(B_ == 0 || T_ == 0 || Hk_ == 0 || Hv_ == 0 || K_ == 0 || V_ == 0,
                    OP_LOGE(ctx_.nodeName, "empty tensor is not supported; got B=%lu T=%lu Hk=%lu Hv=%lu K=%lu V=%lu.",
                            B_, T_, Hk_, Hv_, K_, V_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((Hv_ % Hk_) != 0,
                    OP_LOGE(ctx_.nodeName,
                            "GVA: Hv (value/state heads) must be an integer multiple of Hk (q/k heads); got Hk=%lu "
                            "Hv=%lu.",
                            Hk_, Hv_),
                    return ge::GRAPH_FAILED);
        // 本版只支持 K = V = 128：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 侧逐行的
        // 向量/寄存器读写也依赖该行宽；K/V 取其他值（64/96/72/256 均已实测）会读到未对齐或未初始化
        // 区域，表现为设备报错或结果错误，所以这里只放开目标场景，其余取值直接拦截。
        OP_CHECK_IF(K_ != CP::PRE_PROCESS_BWD_KERNEL_MERGED_K_DIM ||
                        V_ != CP::PRE_PROCESS_BWD_KERNEL_MERGED_V_DIM,
                    OP_LOGE(ctx_.nodeName,
                            "this version only supports K=%u, V=%u, chunk_size=%u; got K=%lu V=%lu. "
                            "Other state dims need the tiling/workspace/tile layout to be generalized.",
                            CP::PRE_PROCESS_BWD_KERNEL_MERGED_K_DIM, CP::PRE_PROCESS_BWD_KERNEL_MERGED_V_DIM,
                            CP::PRE_PROCESS_BWD_KERNEL_MERGED_CHUNK_SIZE, K_, V_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint64_t>(ctx_.chunkSize) != CP::PRE_PROCESS_BWD_KERNEL_MERGED_CHUNK_SIZE,
                    OP_LOGE(ctx_.nodeName, "chunk_size must be %u, but got %d.",
                            CP::PRE_PROCESS_BWD_KERNEL_MERGED_CHUNK_SIZE, ctx_.chunkSize),
                    return ge::GRAPH_FAILED);

        chunkSize_ = static_cast<uint64_t>(ctx_.chunkSize);
        // 列 tile 宽度与上游一致：K <= 64 用 32，否则用 64
        blockSize_ = (K_ <= 64) ? 32 : 64;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ResolveGateMode()
    {
        OP_CHECK_IF(ctx_.hasG && ctx_.hasGk,
                    OP_LOGE(ctx_.nodeName,
                            "g and gk are mutually exclusive: GDN passes scalar g, KDA/GDN2 pass per-K gk. "
                            "Both were provided."),
                    return ge::GRAPH_FAILED);

        if (ctx_.hasGk) {
            const gert::Shape gkShape = ctx_.gkShape->GetStorageShape();
            OP_CHECK_IF(gkShape.GetDim(CDHP_DIM_0) != static_cast<int64_t>(B_) ||
                            gkShape.GetDim(CDHP_DIM_1) != static_cast<int64_t>(Hv_) ||
                            gkShape.GetDim(CDHP_DIM_2) != static_cast<int64_t>(T_) ||
                            gkShape.GetDim(CDHP_DIM_3) != static_cast<int64_t>(K_),
                        OP_LOGE(ctx_.nodeName, "gk must be [B,Hv,T,K]."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(ctx_.gkDtype != ctx_.qDtype,
                        OP_LOGE(ctx_.nodeName,
                                "gk dtype must be the same as q/k dtype; got q/k=%d, gk=%d.",
                                static_cast<int32_t>(ctx_.qDtype), static_cast<int32_t>(ctx_.gkDtype)),
                        return ge::GRAPH_FAILED);
            gateMode_ = PRE_PROCESS_BWD_KERNEL_MERGED_GATE_MODE_PER_K;
            tiling_.useGateGk = 1;
            tiling_.useGateG = 0;
            return ge::GRAPH_SUCCESS;
        }
        if (ctx_.hasG) {
            const gert::Shape gShape = ctx_.gShape->GetStorageShape();
            OP_CHECK_IF(gShape.GetDim(CDHP_DIM_0) != static_cast<int64_t>(B_) ||
                            gShape.GetDim(CDHP_DIM_1) != static_cast<int64_t>(Hv_) ||
                            gShape.GetDim(CDHP_DIM_2) != static_cast<int64_t>(T_),
                        OP_LOGE(ctx_.nodeName, "g must be [B,Hv,T]."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(ctx_.gDtype != ctx_.qDtype && ctx_.gDtype != ge::DT_FLOAT,
                        OP_LOGE(ctx_.nodeName, "g dtype must be FP32 or the same as q/k dtype."),
                        return ge::GRAPH_FAILED);
            gateMode_ = PRE_PROCESS_BWD_KERNEL_MERGED_GATE_MODE_SCALAR;
            tiling_.useGateG = 1;
            tiling_.useGateGk = 0;
            return ge::GRAPH_SUCCESS;
        }
        gateMode_ = PRE_PROCESS_BWD_KERNEL_MERGED_GATE_MODE_NONE;
        tiling_.useGateG = 0;
        tiling_.useGateGk = 0;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ResolveSegment()
    {
        isVarLen_ = ctx_.cuSeqlensShape != nullptr;
        if (isVarLen_) {
            const gert::Shape cuSeqlensShape = ctx_.cuSeqlensShape->GetStorageShape();
            const int64_t seqNum = cuSeqlensShape.GetDim(CDHP_DIM_0) - 1;
            OP_CHECK_IF(seqNum < 1,
                        OP_LOGE(ctx_.nodeName, "cu_seqlens must contain at least 2 entries, but got shape dim0=%ld.",
                                cuSeqlensShape.GetDim(CDHP_DIM_0)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(B_ != 1,
                        OP_LOGE(ctx_.nodeName,
                                "when cu_seqlens is provided B must be 1 (one packed segment per launch), but got "
                                "B=%lu.",
                                B_),
                        return ge::GRAPH_FAILED);
            // 本 kernel 一次只处理一个 [bos,eos)：varlen 时由 kernel 从 GM 读取 cu_seqlens[0:2] 覆盖，
            // host 侧只给出 [0,T) 作为缺省值，不读取输入数值。
            tiling_.bos = 0;
            tiling_.eos = T_;
        } else {
            OP_CHECK_IF(B_ != 1,
                        OP_LOGE(ctx_.nodeName,
                                "dense path requires B=1 because one launch handles a single [0,T) segment; got "
                                "B=%lu. Pass cu_seqlens for packed input.",
                                B_),
                        return ge::GRAPH_FAILED);
            tiling_.bos = 0;
            tiling_.eos = T_;
        }
        tiling_.isVarLen = isVarLen_ ? 1 : 0;
        tiling_.seqNum = 1;
        tiling_.chunkNum = CeilDiv(T_, chunkSize_);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanCorePartition()
    {
        const uint64_t tileV = CeilDiv(V_, blockSize_);
        const uint64_t tileK = CeilDiv(K_, blockSize_);
        const uint64_t tileNum = tileV + tileK;
        const uint64_t totalCore = static_cast<uint64_t>(ctx_.totalCoreNum);

        OP_CHECK_IF(totalCore == 0, OP_LOGE(ctx_.nodeName, "platform reported 0 AIC cores."), return ge::GRAPH_FAILED);

        // 本版 kernel 在核内按"整头"处理（不做列 tile 拆分），因此统一走仅按 head 连续分核：
        // 同一 head 的全部 chunk 留在同一 AIV，禁止按 chunk 分核。
        // (hv, 列 tile) 展平分核留待 kernel 支持列 tile 后放开。
        //
        // A5（1 AIC : 2 AIV）下一个 block 的两个 AIV 各自承包一个 head，因此 blockDim 按
        // ceil(Hv/2) 规划：Hv=32 时用 16 个 block × 2 AIV 覆盖 32 个 head，而不像 1:1 那样
        // 只能开 28 个 block、每核串两个 head。A2/A3 仍是 1:1，保持原规则。
        // 1:2 核型：每个 block 的 2 个 AIV 各承包一个 head，因此按 ceil(Hv/2) 开 block，
        // groupHeads = ceil(Hv/blockDim)（Hv=32/核数 20 → 16 个 block × 2 head；Hv=64 → 20 个
        // block × 4 head = 2 轮，每轮两个 AIV 各有各的 head）。
        // kernel 侧把每 block 的 head 数补齐到 2 的倍数，只有当一个 block 的 head 数是奇数时才会
        // 出现"重复算最后一个 head"的填充槽（幂等，仅浪费一点算力）。
        splitMode_ = CP::PRE_PROCESS_BWD_KERNEL_MERGED_SPLIT_BY_HEAD;
        const uint64_t aivPerBlock = std::max<uint64_t>(1, static_cast<uint64_t>(ctx_.aivPerBlock));
        // v23（按 head 数与核数的关系选映射）：
        //   * head 数**不超过** AIC 核数（h8 这类小 head 场景）：每个 head 独占一个 block，blockDim = Hv，
        //     一个 block 的两个 AIV 合干这一条链的两半（halfSplit）。这样不会退化成"4 个 block 各串 2 个
        //     head、20 个 AIC 里 16 个空闲"（h8 实测 12.0 ms / 2.36× 就是这个退化造成的）。
        //   * head 数超过核数（gva 32 / kda 64 / long 32）：保持 1:2 的"每 block 两个 AIV 各一个 head"，
        //     blockDim = ceil(Hv/aivPerBlock)，否则链会被拉到更多轮、每轮的手握次数不变而总轮数变多。
        const bool halfSplit = (aivPerBlock == 2) && (Hv_ <= totalCore);
        blockDim_ = halfSplit ? std::min(Hv_, totalCore) : std::min(CeilDiv(Hv_, aivPerBlock), totalCore);
        groupHeads_ = CeilDiv(Hv_, blockDim_);
        tiling_.aivPerBlock = aivPerBlock;

        tiling_.usedCoreNum = blockDim_;
        tiling_.blockDim = blockDim_;
        tiling_.splitMode = splitMode_;
        tiling_.groupHeads = groupHeads_;
        // v19/v23：head 内并行（两个 AIV 合干同一条链）。打开条件见上面的 halfSplit；
        // 打开时 workspace 的"份"数按 block 算（同一个 head 的两半共享同一份平面），kernel 侧另有一套
        // 对应实现（见 struct.h / common.h 的注释）。
        tiling_.halfSplit = halfSplit ? 1 : 0;
        // v20：把 ZP 并进"AB + Z"的同一次 GEMM，只在 arch35（A5）打开——两份拼接操作数的布局与
        // 消费它的 cube 都在 arch35 侧，arch22（A2/A3）仍是六 Stage 的独立 ZP 平面。
        // 实测（design.md §29）：head 内并行不划算，但"每 chunk 链上的 MMAD 次数"是本算子的真实瓶颈，
        // 因此这条合并单独打开。
        tiling_.mergeZp = (aivPerBlock == 2) ? 1 : 0;
        tiling_.tileV = tileV;
        tiling_.tileK = tileK;
        tiling_.tileNum = tileNum;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanWorkspace()
    {
        // V0 的 chunk slot：Q̄s[M,K] + K̄[M,K] + W[M,K] + do[M,V]（模型 dtype，零填充）+ decayK[K]（FP32）。
        // 每个工作组 CDHP_WINDOW_COUNT 份 slot（v2.1 双 window 流水）：同一 chunk 下不同 head 的操作数
        // 不能共用同一块 GM；同一工作组的两份 slot 按 window = chunkIdx & 1 使用，AIV 才能领先 AIC
        // 一个 chunk 准备操作数而不覆盖 C1 正在读的那份。
        const uint64_t slotModelBytes =
            chunkSize_ * (3 * K_ + 2 * V_) * CDHP_MODEL_DTYPE_SIZE + K_ * CDHP_FP32_DTYPE_SIZE;
        const uint64_t slotBytes = AlignUp(slotModelBytes, CDHP_WS_ALIGN);
        // 以下平面都是"每个工作组各一份"：同一 chunk 下不同 head 的中间量不能共用同一块 GM。
        // 每个平面按 blockDim_ 复制 blockDim_ 份，kernel 用 BlockIdx() 选自己的那一份。
        // dh / p 是 ping-pong 状态：每个工作组占 2 份（parity 0/1），平面总份数为 2 * blockDim_。
        // 约定：单份大小必须是 512 B 的整数倍，这样 AlignUp(wg * n * 单份) / (wg * n) == 单份，
        // kernel 侧 planeBytes / sliceCount 才能精确还原单份大小（否则切片错位会写到相邻平面）。
        // K, V >= 64 时 K*V*4、K*K*4 等均为 512 B 的整数倍，见 Init() 的平台/形状校验。
        // wg：workspace 里"每份平面"的份数 = blockDim * aivPerBlock（A5 下每个 block 的两个 AIV
        // 各有自己的一份中间量/slot，不能共用）。
        const uint64_t wg = blockDim_ * std::max<uint64_t>(1, tiling_.aivPerBlock);
        // v20：mergeZp 打开时 bOper 的列数从 V 扩到 V+K（追加 P_bf 段），AB/Z 的输出平面从 [K,V]
        // 扩到 [K, V+K]（inc | ZP），同时独立的 ZP 平面与 P_bf 平面不再分配。
        const uint64_t mz = (tiling_.mergeZp != 0) ? 1U : 0U;
        // dH：跨 chunk 状态，ping-pong 双缓冲，FP32；dHBf：供下一轮 C1 当矩阵操作数的模型 dtype 拷贝
        // 平面份数：dh/p 每个工作组 2 份（parity，与 window 对齐）；其余中间量 v2.1 起也按 window 分 2 份。
        const uint64_t win = CDHP_TILING_WINDOW_COUNT;
        const uint64_t dhWsBytes = AlignUp(wg * 2 * K_ * V_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t dhBfWsBytes = AlignUp(wg * win * K_ * V_ * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        // v3 平面（详见 op_kernel/<op>_common.h 的 Slot*/At 注释）：
        //   dvPreWs → AB [K,V] FP32      （链外：Q̄sᵀ@do + Wᵀ@(-dv)；mergeZp 打开时是 [K, V+K] 的
        //                                 inc | ZP 平面）
        //   dvHatWs → Z  [K,V] FP32      （链上：(-T1)@dH_prev）
        //   qtermWs → ZP [K,K] FP32      （P 链上：(-T1)@P_prev；mergeZp 打开时不分配，ZP 落在 dvPreWs 右侧）
        //   wtermWs → T1 的模型 dtype 副本 [K,K]（Cube 当矩阵操作数）
        //   t1Ws    → T1 的 FP32 平面 [K,K]
        const uint64_t dvPreWsBytes =
            AlignUp(wg * win * K_ * (V_ + mz * K_) * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t dvHatWsBytes = AlignUp(wg * win * K_ * V_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t qtermWsBytes =
            mz ? 0 : AlignUp(wg * win * K_ * K_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t wtermWsBytes = AlignUp(wg * win * K_ * K_ * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t t1WsBytes = AlignUp(wg * win * K_ * K_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        // v13：AB 与 Z 合并成一次 GEMM 的两份拼接操作数（arch35 用；arch22 不使用，但一起分配，
        // 保证 tiling 只有一套布局）：
        //   aOper = [Q̄s(M,K) | W(M,K) | (-T1)ᵀ(K,K)]，模型 dtype；Cube 的 A 操作数（列主序 [K, 2M+K]）
        //   bOper = [do(M,V) | -dv(M,V) | dH_bf(K,V)]，模型 dtype；Cube 的 B 操作数（行主序 [2M+K, V]）
        //   mergeZp 打开时 bOper = [do | -dv | dH_bf | P_bf]，行主序 [2M+K, V+K]，行 0..2M 的右侧 K 列补零
        const uint64_t aOperWsBytes = AlignUp(
            wg * win * (2 * chunkSize_ * K_ + K_ * K_) * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t bOperWsBytes = AlignUp(
            wg * win * (2 * chunkSize_ + mz * K_) * (V_ + mz * K_) * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t pcWsBytes = 0;  // v3 起不再需要独立的 P_c 平面
        const uint64_t pWsBytes = AlignUp(wg * 2 * K_ * K_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t pBfWsBytes =
            mz ? 0 : AlignUp(wg * win * K_ * K_ * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t metaWsBytes = CDHP_META_WS_BYTES;

        uint64_t offset = 0;
        tiling_.metaWsOffset = offset;
        offset += metaWsBytes;
        tiling_.slotWsOffset = offset;
        offset += wg * win * slotBytes;
        tiling_.dhWsOffset = offset;
        offset += dhWsBytes;
        tiling_.dhBfWsOffset = offset;
        offset += dhBfWsBytes;
        tiling_.dvPreWsOffset = offset;
        offset += dvPreWsBytes;
        tiling_.dvHatWsOffset = offset;
        offset += dvHatWsBytes;
        tiling_.qtermWsOffset = offset;
        offset += qtermWsBytes;
        tiling_.wtermWsOffset = offset;
        offset += wtermWsBytes;
        tiling_.t1WsOffset = offset;
        offset += t1WsBytes;
        tiling_.aOperWsOffset = offset;
        offset += aOperWsBytes;
        tiling_.bOperWsOffset = offset;
        offset += bOperWsBytes;
        tiling_.pcWsOffset = offset;
        offset += pcWsBytes;
        tiling_.pWsOffset = offset;
        offset += pWsBytes;
        tiling_.pBfWsOffset = offset;
        offset += pBfWsBytes;

        tiling_.slotNum = CDHP_TILING_WINDOW_COUNT;  // 每个工作组 2 个 slot（v2.1 双 window）
        tiling_.slotBytes = slotBytes;
        tiling_.dhWsBytes = dhWsBytes;
        tiling_.dhBfWsBytes = dhBfWsBytes;
        tiling_.dvPreWsBytes = dvPreWsBytes;
        tiling_.dvHatWsBytes = dvHatWsBytes;
        tiling_.qtermWsBytes = qtermWsBytes;
        tiling_.wtermWsBytes = wtermWsBytes;
        tiling_.t1WsBytes = t1WsBytes;
        tiling_.aOperWsBytes = aOperWsBytes;
        tiling_.bOperWsBytes = bOperWsBytes;
        tiling_.pcWsBytes = pcWsBytes;
        tiling_.pWsBytes = pWsBytes;
        tiling_.pBfWsBytes = pBfWsBytes;
        tiling_.metaWsBytes = metaWsBytes;
        tiling_.totalWsBytes = offset;

        workspaceSize_ = static_cast<uint64_t>(ctx_.sysWorkspaceSize) + offset;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(Init() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ResolveGateMode() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ResolveSegment() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(PlanCorePartition() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(PlanWorkspace() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

        const bool isScale = ctx_.hasScaleAttr;
        tiling_.B = B_;
        tiling_.T = T_;
        tiling_.Hk = Hk_;
        tiling_.Hv = Hv_;
        tiling_.K = K_;
        tiling_.V = V_;
        tiling_.chunkSize = chunkSize_;
        tiling_.blockSize = blockSize_;
        tiling_.kGroupNum = CeilDiv(K_, static_cast<uint64_t>(CP::PRE_PROCESS_BWD_KERNEL_MERGED_K_GROUP_ROWS));
        tiling_.isScale = isScale ? 1 : 0;
        tiling_.scale = isScale ? static_cast<float>(ctx_.scaleAttr) : 1.0f;
        return ge::GRAPH_SUCCESS;
    }

private:
    PreProcessBwdKernelMergedTilingContext &ctx_;
    PreProcessBwdKernelMergedTilingData &tiling_;

    uint64_t B_ = 0;
    uint64_t T_ = 0;
    uint64_t Hk_ = 0;
    uint64_t Hv_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t chunkSize_ = CP::PRE_PROCESS_BWD_KERNEL_MERGED_CHUNK_SIZE;
    uint64_t blockSize_ = 64;
    bool isVarLen_ = false;
    uint64_t splitMode_ = CP::PRE_PROCESS_BWD_KERNEL_MERGED_SPLIT_BY_HEAD;
    uint64_t blockDim_ = 0;
    uint64_t groupHeads_ = 0;
    uint64_t workspaceSize_ = 0;
    uint32_t gateMode_ = PRE_PROCESS_BWD_KERNEL_MERGED_GATE_MODE_NONE;
};

} // namespace optiling

#endif // PRE_PROCESS_BWD_KERNEL_MERGED_TILING_PROCESSOR_H
