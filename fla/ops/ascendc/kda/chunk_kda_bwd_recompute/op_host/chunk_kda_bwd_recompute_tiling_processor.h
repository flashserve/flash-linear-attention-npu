/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_RECOMPUTE_TILING_PROCESSOR_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/chunk_kda_bwd_recompute_struct.h"
#include "../op_kernel/chunk_kda_bwd_recompute_tiling_key.h"

namespace optiling {

using KDA::ChunkKdaBwdRecomputeTilingData;

constexpr size_t KDA_RECOMPUTE_Q_IDX = 0;
constexpr size_t KDA_RECOMPUTE_K_IDX = 1;
constexpr size_t KDA_RECOMPUTE_V_IDX = 2;
constexpr size_t KDA_RECOMPUTE_G_IDX = 3;
constexpr size_t KDA_RECOMPUTE_BETA_IDX = 4;
constexpr size_t KDA_RECOMPUTE_A_IDX = 5;
constexpr size_t KDA_RECOMPUTE_A_LOG_IDX = 6;
constexpr size_t KDA_RECOMPUTE_DT_BIAS_IDX = 7;
constexpr size_t KDA_RECOMPUTE_CU_SEQLENS_IDX = 8;
constexpr size_t KDA_RECOMPUTE_CHUNK_INDICES_IDX = 9;
constexpr size_t KDA_RECOMPUTE_CHUNK_SIZE_ATTR = 0;
constexpr size_t KDA_RECOMPUTE_USE_GATE_ATTR = 1;
constexpr size_t KDA_RECOMPUTE_USE_EXP2_ATTR = 2;
constexpr size_t KDA_RECOMPUTE_LOWER_BOUND_ATTR = 3;

struct ChunkKdaBwdRecomputeTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *aShape;
    const gert::StorageShape *aLogShape;
    const gert::StorageShape *dtBiasShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    const int64_t *cuSeqlensData;
    const int64_t *chunkIndicesData;
    ge::DataType gateDtype;
    ge::DataType betaDtype;
    int64_t chunkSize;
    bool useGateInKernel;
    bool useExp2;
    float lowerBound;
    uint64_t ubSize;
    size_t sysWorkspaceSize;
};

class ChunkKdaBwdRecomputeTilingProcessor {
public:
    ChunkKdaBwdRecomputeTilingProcessor(
        ChunkKdaBwdRecomputeTilingContext &ctx, ChunkKdaBwdRecomputeTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(CheckSpec() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ComputeChunkNum() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetVecRow() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        workspaceSize_ = ctx_.sysWorkspaceSize + userWorkspaceSize_;
        tilingKey_ = ComputeTilingKey();
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const
    {
        return tilingKey_;
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

private:
    ge::graphStatus CheckSpec()
    {
        OP_CHECK_IF(ctx_.chunkSize != 64,
                    OP_LOGE(ctx_.nodeName, "chunk_size must be 64."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.qShape->GetStorageShape().GetDimNum() != 4,
                    OP_LOGE(ctx_.nodeName, "q must be rank-4 BNSD."),
                    return ge::GRAPH_FAILED);
        const gert::Shape q = ctx_.qShape->GetStorageShape();
        const gert::Shape k = ctx_.kShape->GetStorageShape();
        const gert::Shape v = ctx_.vShape->GetStorageShape();
        const gert::Shape g = ctx_.gShape->GetStorageShape();
        const gert::Shape beta = ctx_.betaShape->GetStorageShape();
        const gert::Shape a = ctx_.aShape->GetStorageShape();
        OP_CHECK_IF(q.GetDim(3) != 128 || k.GetDim(3) != 128 || v.GetDim(3) != 128,
                    OP_LOGE(ctx_.nodeName, "K and V must be 128."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(g.GetDim(3) != 128,
                    OP_LOGE(ctx_.nodeName, "g head dim must be 128."),
                    return ge::GRAPH_FAILED);
        tiling_.B = q.GetDim(0);
        tiling_.Hk = k.GetDim(1);
        tiling_.Hv = v.GetDim(1);
        OP_CHECK_IF(tiling_.Hv <= 0 || tiling_.Hk <= 0 || tiling_.Hv % tiling_.Hk != 0,
                    OP_LOGE(ctx_.nodeName, "Invalid GQA head configuration."),
                    return ge::GRAPH_FAILED);
        tiling_.hvPerHk = tiling_.Hv / tiling_.Hk;
        tiling_.T = q.GetDim(2);
        tiling_.K = 128;
        tiling_.V = 128;
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.useGateInKernel = ctx_.useGateInKernel ? 1 : 0;
        tiling_.useExp2 = ctx_.useExp2 ? 1 : 0;
        tiling_.hasALog = ctx_.aLogShape != nullptr ? 1 : 0;
        tiling_.hasDtBias = ctx_.dtBiasShape != nullptr ? 1 : 0;
        uint32_t lowerBoundBits = 0;
        float lowerBound = ctx_.lowerBound;
        memcpy(&lowerBoundBits, &lowerBound, sizeof(lowerBoundBits));
        tiling_.lowerBoundBits = static_cast<int64_t>(lowerBoundBits);
        tiling_.isVariable = ctx_.cuSeqlensShape != nullptr ? 1 : 0;
        chunkSize_ = static_cast<uint64_t>(ctx_.chunkSize);
        if (ctx_.useGateInKernel && ctx_.aLogShape != nullptr) {
            const float *aLogData = reinterpret_cast<const float *>(
                ctx_.aLogShape != nullptr ? ctx_.aLogShape : nullptr);
            (void)aLogData;
        }
        OP_CHECK_IF(v.GetDim(0) != q.GetDim(0) || v.GetDim(2) != q.GetDim(2),
                    OP_LOGE(ctx_.nodeName, "q/k/v batch and T must match."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(g.GetDim(0) != v.GetDim(0) || g.GetDim(1) != v.GetDim(1) || g.GetDim(2) != v.GetDim(2),
                    OP_LOGE(ctx_.nodeName, "g shape must match v on B/H/T."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(beta.GetDimNum() != 3,
                    OP_LOGE(ctx_.nodeName, "beta must be rank-3 BHT."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(a.GetDim(1) != tiling_.Hv,
                    OP_LOGE(ctx_.nodeName, "A head dim must equal Hv."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ComputeChunkNum()
    {
        if (ctx_.cuSeqlensShape != nullptr) {
            OP_CHECK_IF(ctx_.chunkIndicesShape == nullptr,
                        OP_LOGE(ctx_.nodeName, "chunk_indices required for varlen."),
                        return ge::GRAPH_FAILED);
            tiling_.chunkNum = ctx_.chunkIndicesShape->GetStorageShape().GetDim(0) / 2;
            tiling_.B = 1;
            return ge::GRAPH_SUCCESS;
        }
        const int64_t chunksPerSeq = (tiling_.T + tiling_.chunkSize - 1) / tiling_.chunkSize;
        tiling_.chunkNum = tiling_.B * chunksPerSeq;
        return ge::GRAPH_SUCCESS;
    }

    static uint64_t Align32(uint64_t bytes)
    {
        return (bytes + 31ULL) & ~31ULL;
    }

    ge::graphStatus SetVecRow()
    {
        constexpr uint64_t sizeofQk = 2;
        uint64_t sizeofGate = ctx_.gateDtype == ge::DT_FLOAT ? 4 : 2;
        uint64_t sizeofBeta = ctx_.betaDtype == ge::DT_FLOAT ? 4 : 2;
        const uint64_t kDim = static_cast<uint64_t>(tiling_.K);
        // A5 AIV UB is 256KiB. ProcessGateAndK keeps many TQue/TBuf live at once,
        // so cap well below the advertised size to leave TPipe metadata room.
        constexpr uint64_t kUbCap = 192 * 1024;
        const uint64_t ubLimit = ctx_.ubSize == 0 ? kUbCap : std::min(ctx_.ubSize, kUbCap);
        uint64_t rowNum = chunkSize_;

        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += Align32(2 * rowNum * kDim * sizeofGate);
            useUbSize += Align32(2 * rowNum * kDim * sizeofQk);
            useUbSize += Align32(2 * rowNum * kDim * sizeofQk);
            useUbSize += Align32(2 * rowNum * sizeofBeta);
            useUbSize += Align32(2 * rowNum * kDim * sizeof(float));
            useUbSize += Align32(2 * rowNum * kDim * sizeof(float));
            useUbSize += Align32(2 * rowNum * kDim * sizeofQk);
            useUbSize += Align32(2 * rowNum * kDim * sizeofQk);
            useUbSize += Align32(2 * rowNum * kDim * sizeofQk);
            useUbSize += Align32(rowNum * kDim * sizeof(float));
            useUbSize += Align32(rowNum * kDim * sizeof(float));
            useUbSize += Align32(kDim * sizeof(float));
            useUbSize += Align32(kDim * sizeof(float));
            useUbSize += Align32(rowNum * kDim * sizeof(float));
            useUbSize += Align32(rowNum * kDim * sizeof(float));
            useUbSize += Align32(rowNum * kDim * sizeof(float));
            useUbSize += Align32(rowNum * kDim * sizeof(float));
            useUbSize += Align32(rowNum * sizeof(float));
            useUbSize += Align32(rowNum * 32);
            useUbSize += Align32(kDim * sizeof(float));
            useUbSize += Align32(kDim * sizeof(float));
            useUbSize += Align32(kDim * sizeof(float));
            useUbSize += Align32(32);
            useUbSize += 4096;
            if (useUbSize <= ubLimit) {
                break;
            }
            rowNum /= 2;
        }

        tiling_.vecRow = static_cast<int64_t>(rowNum);
        // A5 L1 path keeps kbg/vb on-chip (design §6.1). GM workspace is only the sys buffer.
        userWorkspaceSize_ = 0;
        workspaceSize_ = ctx_.sysWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    uint64_t ComputeTilingKey() const
    {
        uint64_t key = 1;
        if (ctx_.betaDtype == ge::DT_FLOAT) {
            key = 2;
        }
        if (ctx_.gateDtype == ge::DT_FLOAT) {
            key += 2;
        }
        return key;
    }

    ChunkKdaBwdRecomputeTilingContext &ctx_;
    ChunkKdaBwdRecomputeTilingData &tiling_;
    size_t workspaceSize_ = 0;
    size_t userWorkspaceSize_ = 0;
    uint64_t tilingKey_ = 1;
    uint64_t chunkSize_ = 64;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_RECOMPUTE_TILING_PROCESSOR_H
