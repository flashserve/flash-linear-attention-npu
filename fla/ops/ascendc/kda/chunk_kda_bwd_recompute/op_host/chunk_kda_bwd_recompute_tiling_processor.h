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
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/chunk_kda_bwd_recompute_struct.h"
#include "../op_kernel/chunk_kda_bwd_recompute_tiling_key.h"
#include <string>

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
constexpr size_t KDA_RECOMPUTE_GK_OUTPUT_IDX = 4;
constexpr size_t KDA_RECOMPUTE_CHUNK_SIZE_ATTR = 0;
constexpr size_t KDA_RECOMPUTE_USE_GATE_ATTR = 1;
constexpr size_t KDA_RECOMPUTE_USE_EXP2_ATTR = 2;
constexpr size_t KDA_RECOMPUTE_LOWER_BOUND_ATTR = 3;
constexpr size_t KDA_RECOMPUTE_INPUT_LAYOUT_ATTR = 4;

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
    bool enableCompactWorkspace;
    bool isAscend960 = false;
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
        if (CheckSpec() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (ComputeChunkNum() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (SetVecRow() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        tilingKey_ = ComputeTilingKey();
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const
    {
        return tilingKey_;
    }

    size_t GetWorkspaceSize(uint32_t usedAic) const
    {
        return ctx_.sysWorkspaceSize + usedAic * workspaceBytesPerCore_;
    }

private:
    ge::graphStatus CheckSpec()
    {
        if (ctx_.chunkSize != 64) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName, "chunk_size", "invalid", "chunk_size must be 64.");
        return ge::GRAPH_FAILED;
    }
        for (const auto *shape : {ctx_.qShape,ctx_.kShape,ctx_.vShape,ctx_.gShape,ctx_.aShape}) {
            if (shape == nullptr || shape->GetStorageShape().GetDimNum() != 4) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"rank","invalid","q/k/v/g/A must have rank 4.");
                return ge::GRAPH_FAILED;
            }
        }
        const gert::Shape q = ctx_.qShape->GetStorageShape();
        const gert::Shape k = ctx_.kShape->GetStorageShape();
        const gert::Shape v = ctx_.vShape->GetStorageShape();
        const gert::Shape g = ctx_.gShape->GetStorageShape();
        const gert::Shape beta = ctx_.betaShape->GetStorageShape();
        const gert::Shape a = ctx_.aShape->GetStorageShape();
        if (q.GetDim(3) != 128 || k.GetDim(3) != 128 || v.GetDim(3) != 128) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName, "K", "invalid", "K and V must be 128.");
        return ge::GRAPH_FAILED;
    }
        if (g.GetDim(3) != 128) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName, "g", std::to_string(g.GetDim(3)), "g head dim must be 128.");
        return ge::GRAPH_FAILED;
    }
        if (ctx_.gateDtype != ge::DT_BF16 && ctx_.gateDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx_.nodeName, "dtype", "invalid", "g dtype must be BF16 or FP32.");
        return ge::GRAPH_FAILED;
    }
        if (ctx_.betaDtype != ge::DT_BF16 && ctx_.betaDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx_.nodeName, "dtype", "invalid", "beta dtype must be BF16 or FP32.");
        return ge::GRAPH_FAILED;
    }
        if (q != k || q.GetDim(0) < 1 || q.GetDim(2) < 0 || a.GetDim(0) != v.GetDim(0) ||
            a.GetDim(2) != v.GetDim(2) || a.GetDim(3) != 64 || beta.GetDimNum() != 3 ||
            beta.GetDim(0) != v.GetDim(0) || beta.GetDim(1) != v.GetDim(1) || beta.GetDim(2) != v.GetDim(2)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"shape","invalid","q/k, A and beta dimensions must match their declared head-major shapes.");
            return ge::GRAPH_FAILED;
        }
        tiling_.B = q.GetDim(0);
        tiling_.Hk = k.GetDim(1);
        tiling_.Hv = v.GetDim(1);
        if (tiling_.Hv <= 0 || tiling_.Hk <= 0 || tiling_.Hv % tiling_.Hk != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName, "Hv/Hk", std::to_string(tiling_.Hv) + "/" + std::to_string(tiling_.Hk),
            "Hv must be positive and divisible by Hk");
        return ge::GRAPH_FAILED;
    }
        if (ctx_.useGateInKernel && (!ctx_.aLogShape || !std::isfinite(ctx_.lowerBound) ||
            ctx_.lowerBound < -5.0f || ctx_.lowerBound >= 0.0f)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"A_log/lower_bound",
                std::string(ctx_.aLogShape ? "present/" : "nullptr/") + std::to_string(ctx_.lowerBound),
                "Gate requires A_log and finite lower_bound in [-5,0).");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.aLogShape && (ctx_.aLogShape->GetStorageShape().GetDimNum()!=1 ||
            ctx_.aLogShape->GetStorageShape().GetDim(0)!=tiling_.Hv)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx_.nodeName,"A_log","mismatch","A_log must have shape [HV].");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.dtBiasShape) {
            const auto &bias=ctx_.dtBiasShape->GetStorageShape();
            const bool flat=bias.GetDimNum()==1 && bias.GetDim(0)==tiling_.Hv*128;
            const bool matrix=bias.GetDimNum()==2 && bias.GetDim(0)==tiling_.Hv && bias.GetDim(1)==128;
            if(!flat && !matrix) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx_.nodeName,"dt_bias","mismatch","dt_bias must be [HV,128] or flat HV*128.");
                return ge::GRAPH_FAILED;
            }
        }
        tiling_.hvPerHk = tiling_.Hv / tiling_.Hk;
        tiling_.headGroup = tiling_.hvPerHk == 3 ? 3 : std::min<int64_t>(4,tiling_.Hv);
        tiling_.reservedV2 = 0;
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
        if (v.GetDim(0) != q.GetDim(0) || v.GetDim(2) != q.GetDim(2)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx_.nodeName, "q", "invalid", "q/k/v batch and T must match.");
        return ge::GRAPH_FAILED;
    }
        if (g.GetDim(0) != v.GetDim(0) || g.GetDim(1) != v.GetDim(1) || g.GetDim(2) != v.GetDim(2)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx_.nodeName, "g", "invalid", "g shape must match v on B/H/T.");
        return ge::GRAPH_FAILED;
    }
        if (beta.GetDimNum() != 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ctx_.nodeName, "beta", std::to_string(beta.GetDimNum()), "beta must be rank-3 BHT.");
        return ge::GRAPH_FAILED;
    }
        if (a.GetDim(1) != tiling_.Hv) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName, "Hv", "invalid", "A head dim must equal Hv.");
        return ge::GRAPH_FAILED;
    }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ComputeChunkNum()
    {
        // Stable ABI represents an empty int array as nullptr. A packed empty task
        // has no chunk descriptors to access, so validate only its zero boundaries.
        if (tiling_.T == 0 && ctx_.cuSeqlensShape != nullptr && ctx_.chunkIndicesShape == nullptr) {
            const auto &shape=ctx_.cuSeqlensShape->GetStorageShape();
            if(shape.GetDimNum()!=1 || shape.GetDim(0)<2 || !ctx_.cuSeqlensData || tiling_.B!=1) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"cu_seqlens",std::to_string(tiling_.B),"Empty packed input requires B=1 and rank-1 host boundaries.");
                return ge::GRAPH_FAILED;
            }
            for(int64_t i=0;i<shape.GetDim(0);++i) if(ctx_.cuSeqlensData[i]!=0) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"cu_seqlens",std::to_string(ctx_.cuSeqlensData[i]),"All boundaries must be zero for T=0.");
                return ge::GRAPH_FAILED;
            }
            tiling_.chunkNum=0;return ge::GRAPH_SUCCESS;
        }
        if ((ctx_.cuSeqlensShape != nullptr) != (ctx_.chunkIndicesShape != nullptr)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"metadata","invalid","cu_seqlens and chunk_indices must be paired.");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.cuSeqlensShape != nullptr) {
            const auto &cuShape=ctx_.cuSeqlensShape->GetStorageShape();
            const auto &ciShape=ctx_.chunkIndicesShape->GetStorageShape();
            if (tiling_.B != 1 || cuShape.GetDimNum()!=1 || ciShape.GetDimNum()!=1 ||
                cuShape.GetDim(0)<2 || ciShape.GetDim(0)%2 || !ctx_.cuSeqlensData || (ciShape.GetDim(0)!=0 && !ctx_.chunkIndicesData)) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"metadata","invalid","packed B=1 requires rank-1 host metadata and complete pairs.");
                return ge::GRAPH_FAILED;
            }
            const auto *cu=ctx_.cuSeqlensData,*ci=ctx_.chunkIndicesData;
            const int64_t n=cuShape.GetDim(0), pairs=ciShape.GetDim(0)/2;
            if (cu[0]!=0 || cu[n-1]!=tiling_.T) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"cu_seqlens",
                    std::to_string(cu[0])+"/"+std::to_string(cu[n-1])+"/T="+std::to_string(tiling_.T),"First boundary must be 0 and final boundary must equal T.");
                return ge::GRAPH_FAILED;
            }
            int64_t index=0;
            for(int64_t seq=0;seq<n-1;++seq) {
                if(cu[seq]<0 || cu[seq+1]<cu[seq]) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"cu_seqlens",std::to_string(cu[seq])+"/"+std::to_string(cu[seq+1]),"Boundaries must be nonnegative and nondecreasing.");
                    return ge::GRAPH_FAILED;
                }
                const int64_t count=(cu[seq+1]-cu[seq]+63)/64;
                for(int64_t chunk=0;chunk<count;++chunk,++index) {
                    if(index>=pairs || ci[2*index]!=seq || ci[2*index+1]!=chunk) {
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"chunk_indices","invalid","metadata must contain each sequence-major chunk once.");
                        return ge::GRAPH_FAILED;
                    }
                }
            }
            if(index!=pairs) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"chunk_indices",std::to_string(pairs)+"/expected="+std::to_string(index),"Chunk descriptor count must exactly match sequence boundaries.");
                return ge::GRAPH_FAILED;
            }
            tiling_.chunkNum=pairs;return ge::GRAPH_SUCCESS;
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
        if (ctx_.isAscend960) {
            using namespace KdaBwdRecomputeArch92;
            tiling_.vecRow = kRows;
            workspaceBytesPerCore_ = 0;
            return ge::GRAPH_SUCCESS;
        }
        if (!ctx_.enableCompactWorkspace) {
            constexpr uint64_t requiredUb = 134400;
            if (ctx_.ubSize != 0 && ctx_.ubSize < requiredUb) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName,"UB",std::to_string(ctx_.ubSize),"Recompute requires 134400 UB bytes per AIV.");
                return ge::GRAPH_FAILED;
            }
            tiling_.vecRow = 64;
            workspaceBytesPerCore_ = uint64_t(2) * tiling_.headGroup * 32768;
            return ge::GRAPH_SUCCESS;
        }
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
            useUbSize += Align32(rowNum * kDim * sizeofGate);
            useUbSize += 2 * Align32(rowNum * kDim * sizeofQk);
            useUbSize += Align32(rowNum * sizeofBeta);
            useUbSize += Align32(rowNum * kDim * sizeof(float));
            useUbSize += 3 * Align32(rowNum * kDim * sizeofQk);
            useUbSize += 6 * Align32(rowNum * kDim * sizeof(float));
            useUbSize += Align32(rowNum * sizeof(float));
            useUbSize += Align32(rowNum * 32);
            useUbSize += 3 * Align32(kDim * sizeof(float));
            useUbSize += Align32(32);
            useUbSize += 4096;
            if (useUbSize <= ubLimit) {
                break;
            }
            rowNum /= 2;
        }

        tiling_.vecRow = static_cast<int64_t>(rowNum);
        if (!ctx_.enableCompactWorkspace) {
            workspaceBytesPerCore_ = KDA::RECOMPUTE_PAGE_COUNT * KDA::RECOMPUTE_HEADS_PER_GROUP * chunkSize_ *
                (kDim + static_cast<uint64_t>(tiling_.V)) * sizeofQk;
        }
        return ge::GRAPH_SUCCESS;
    }

    static uint64_t DtypeToTemplateToken(ge::DataType dtype)
    {
        return dtype == ge::DT_FLOAT ? CHUNK_KDA_BWD_RECOMPUTE_TPL_FP32 :
                                      CHUNK_KDA_BWD_RECOMPUTE_TPL_BF16;
    }

    uint64_t ComputeTilingKey() const
    {
        using namespace KDA;
        return GET_TPL_TILING_KEY(
            DtypeToTemplateToken(ctx_.gateDtype),
            DtypeToTemplateToken(ctx_.betaDtype),
            static_cast<uint64_t>(tiling_.K),
            static_cast<uint64_t>(tiling_.hvPerHk > 1));
    }

    ChunkKdaBwdRecomputeTilingContext &ctx_;
    ChunkKdaBwdRecomputeTilingData &tiling_;
    size_t workspaceBytesPerCore_ = 0;
    uint64_t tilingKey_ = 1;
    uint64_t chunkSize_ = 64;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_RECOMPUTE_TILING_PROCESSOR_H
