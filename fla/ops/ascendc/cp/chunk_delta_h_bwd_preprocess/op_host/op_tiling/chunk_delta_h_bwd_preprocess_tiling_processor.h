/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_tiling_processor.h
 * \brief
 * Tiling processor for chunk_delta_h_bwd_preprocess, decoupled from gert::TilingContext so that the same
 * validation / core partition / workspace rules can be reused by aclnn and by the <<<>>> direct launch path.
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_TILING_PROCESSOR_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "tiling_base/tiling_base.h"
#include "../../op_kernel/chunk_delta_h_bwd_preprocess_struct.h"

using CP::ChunkDeltaHBwdPreprocessTilingData;

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
static constexpr uint64_t CDHP_META_WS_BYTES = 512;
static constexpr uint64_t CDHP_HALF_DTYPE_SIZE = 2;
static constexpr uint64_t CDHP_FP32_DTYPE_SIZE = 4;
// 模型 dtype（BF16/FP16）统一按 2 B 规划；dhm 与链式累加固定 FP32
static constexpr uint64_t CDHP_MODEL_DTYPE_SIZE = 2;

struct ChunkDeltaHBwdPreprocessTilingContext {
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
    size_t sysWorkspaceSize;
};

class ChunkDeltaHBwdPreprocessTilingProcessor {
public:
    explicit ChunkDeltaHBwdPreprocessTilingProcessor(ChunkDeltaHBwdPreprocessTilingContext &ctx,
                                                     ChunkDeltaHBwdPreprocessTilingData &tiling)
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

    uint32_t GetTilingKey() const
    {
        return tilingKey_;
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
        OP_CHECK_IF(K_ > CP::CHUNK_DELTA_H_BWD_PREPROCESS_MAX_K,
                    OP_LOGE(ctx_.nodeName, "K must be <= %u, but got %lu.",
                            CP::CHUNK_DELTA_H_BWD_PREPROCESS_MAX_K, K_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint64_t>(ctx_.chunkSize) != CP::CHUNK_DELTA_H_BWD_PREPROCESS_CHUNK_SIZE,
                    OP_LOGE(ctx_.nodeName, "chunk_size must be %u, but got %d.",
                            CP::CHUNK_DELTA_H_BWD_PREPROCESS_CHUNK_SIZE, ctx_.chunkSize),
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
            tilingKey_ = CP::CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_GK;
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
            tilingKey_ = (ctx_.gDtype == ge::DT_FLOAT) ? CP::CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_G_FP32
                                                       : CP::CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_G;
            tiling_.useGateG = 1;
            tiling_.useGateGk = 0;
            return ge::GRAPH_SUCCESS;
        }
        tilingKey_ = CP::CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_NONE;
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
        // 同一 head 的全部 chunk 留在同一工作组，禁止按 chunk 分核。
        // (hv, 列 tile) 展平分核留待 kernel 支持列 tile 后放开。
        splitMode_ = CP::CHUNK_DELTA_H_BWD_PREPROCESS_SPLIT_BY_HEAD;
        blockDim_ = std::min(Hv_, totalCore);
        groupHeads_ = CeilDiv(Hv_, blockDim_);

        tiling_.usedCoreNum = blockDim_;
        tiling_.blockDim = blockDim_;
        tiling_.splitMode = splitMode_;
        tiling_.groupHeads = groupHeads_;
        tiling_.tileV = tileV;
        tiling_.tileK = tileK;
        tiling_.tileNum = tileNum;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanWorkspace()
    {
        // V0 的 chunk slot：Q̄s[M,K] + K̄[M,K] + W[M,K] + do[M,V]（模型 dtype，零填充）+ decayK[K]（FP32）。
        // 每个工作组一个 slot：同一 chunk 下不同 head 的操作数不能共用同一块 GM，否则会互相覆盖；
        // 同 chunk 内 AIV/AIC 严格交替，一个 slot 足够（下一轮 V0 必然在上一轮 C5 之后）。
        const uint64_t slotModelBytes =
            chunkSize_ * (3 * K_ + V_) * CDHP_MODEL_DTYPE_SIZE + K_ * CDHP_FP32_DTYPE_SIZE;
        const uint64_t slotBytes = AlignUp(slotModelBytes, CDHP_WS_ALIGN);
        // 以下平面都是"每个工作组各一份"：同一 chunk 下不同 head 的中间量不能共用同一块 GM。
        // 每个平面按 blockDim_ 复制 blockDim_ 份，kernel 用 BlockIdx() 选自己的那一份。
        // dh / p 是 ping-pong 状态：每个工作组占 2 份（parity 0/1），平面总份数为 2 * blockDim_。
        // 约定：单份大小必须是 512 B 的整数倍，这样 AlignUp(wg * n * 单份) / (wg * n) == 单份，
        // kernel 侧 planeBytes / sliceCount 才能精确还原单份大小（否则切片错位会写到相邻平面）。
        // K, V >= 64 时 K*V*4、K*K*4 等均为 512 B 的整数倍，见 Init() 的平台/形状校验。
        const uint64_t wg = blockDim_;
        // dH：跨 chunk 状态，ping-pong 双缓冲，FP32；dHBf：供下一轮 C1 当矩阵操作数的模型 dtype 拷贝
        const uint64_t dhWsBytes = AlignUp(wg * 2 * K_ * V_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t dhBfWsBytes = AlignUp(wg * K_ * V_ * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        // E 分支中间量
        const uint64_t dvPreWsBytes = AlignUp(wg * chunkSize_ * V_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t dvHatWsBytes = AlignUp(wg * chunkSize_ * V_ * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t qtermWsBytes = AlignUp(wg * K_ * V_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t wtermWsBytes = AlignUp(wg * K_ * V_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        // P 分支：[K,K] 平面按模型 dtype 作为矩阵操作数；FP32 只用于链式累加与最终输出
        const uint64_t t1WsBytes = AlignUp(wg * K_ * K_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t pcWsBytes = AlignUp(wg * K_ * K_ * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t pWsBytes = AlignUp(wg * 2 * K_ * K_ * CDHP_FP32_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t pBfWsBytes = AlignUp(wg * K_ * K_ * CDHP_MODEL_DTYPE_SIZE, CDHP_WS_ALIGN);
        const uint64_t metaWsBytes = CDHP_META_WS_BYTES;

        uint64_t offset = 0;
        tiling_.metaWsOffset = offset;
        offset += metaWsBytes;
        tiling_.slotWsOffset = offset;
        offset += blockDim_ * slotBytes;
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
        tiling_.pcWsOffset = offset;
        offset += pcWsBytes;
        tiling_.pWsOffset = offset;
        offset += pWsBytes;
        tiling_.pBfWsOffset = offset;
        offset += pBfWsBytes;

        tiling_.slotNum = 1;  // 每个工作组 1 个 slot
        tiling_.slotBytes = slotBytes;
        tiling_.dhWsBytes = dhWsBytes;
        tiling_.dhBfWsBytes = dhBfWsBytes;
        tiling_.dvPreWsBytes = dvPreWsBytes;
        tiling_.dvHatWsBytes = dvHatWsBytes;
        tiling_.qtermWsBytes = qtermWsBytes;
        tiling_.wtermWsBytes = wtermWsBytes;
        tiling_.t1WsBytes = t1WsBytes;
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
        tiling_.kGroupNum = CeilDiv(K_, static_cast<uint64_t>(CP::CHUNK_DELTA_H_BWD_PREPROCESS_K_GROUP_ROWS));
        tiling_.isScale = isScale ? 1 : 0;
        tiling_.scale = isScale ? static_cast<float>(ctx_.scaleAttr) : 1.0f;
        return ge::GRAPH_SUCCESS;
    }

private:
    ChunkDeltaHBwdPreprocessTilingContext &ctx_;
    ChunkDeltaHBwdPreprocessTilingData &tiling_;

    uint64_t B_ = 0;
    uint64_t T_ = 0;
    uint64_t Hk_ = 0;
    uint64_t Hv_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t chunkSize_ = CP::CHUNK_DELTA_H_BWD_PREPROCESS_CHUNK_SIZE;
    uint64_t blockSize_ = 64;
    bool isVarLen_ = false;
    uint64_t splitMode_ = CP::CHUNK_DELTA_H_BWD_PREPROCESS_SPLIT_BY_HEAD;
    uint64_t blockDim_ = 0;
    uint64_t groupHeads_ = 0;
    uint64_t workspaceSize_ = 0;
    uint32_t tilingKey_ = CP::CHUNK_DELTA_H_BWD_PREPROCESS_TILING_KEY_NONE;
};

} // namespace optiling

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_TILING_PROCESSOR_H
