#ifndef CHUNK_KDA_BWD_C_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_C_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/chunk_kda_bwd_common.h"

namespace optiling {

enum ChunkKdaBwdCInputIndex : size_t {
    KDA_C_Q = 0,
    KDA_C_K,
    KDA_C_V,
    KDA_C_V_NEW,
    KDA_C_GK,
    KDA_C_BETA,
    KDA_C_AKK,
    KDA_C_H,
    KDA_C_DH,
    KDA_C_DV_SCAN,
    KDA_C_DQ_RAW,
    KDA_C_DAQK,
    KDA_C_RAW_G,
    KDA_C_A_LOG,
    KDA_C_DT_BIAS,
    KDA_C_CU_SEQLENS,
    KDA_C_CHUNK_INDICES,
    KDA_C_INPUT_COUNT
};
constexpr size_t KDA_C_REQUIRED_INPUT_COUNT = KDA_C_RAW_G;
constexpr size_t KDA_C_SCALE_ATTR = 0;
constexpr size_t KDA_C_CHUNK_SIZE_ATTR = 1;
constexpr size_t KDA_C_SAFE_GATE_ATTR = 2;
constexpr size_t KDA_C_USE_GATE_ATTR = 3;
constexpr size_t KDA_C_LOWER_BOUND_ATTR = 4;
constexpr size_t KDA_C_DH_HEAD_MAJOR_ATTR = 5;
constexpr int64_t KDA_C_SLOT_BYTES = 256 * 1024;
constexpr int64_t KDA_C_SLOT_COUNT = 4;

struct ChunkKdaBwdCTilingContext {
    const char *nodeName;
    const gert::StorageShape *shapes[KDA_C_INPUT_COUNT];
    ge::DataType types[KDA_C_INPUT_COUNT];
    float scale;
    int64_t chunkSize;
    bool safeGate;
    bool useGateInKernel;
    float lowerBound;
    bool dhHeadMajor;
    bool validateIntermediateShapes;
    uint32_t aicCoreNum;
    size_t systemWorkspaceSize;
};

class ChunkKdaBwdCTilingProcessor {
public:
    ChunkKdaBwdCTilingProcessor(
        ChunkKdaBwdCTilingContext &ctx, KDA::ChunkKdaBwdCTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    ge::graphStatus Process()
    {
        for (size_t i = 0; i < KDA_C_REQUIRED_INPUT_COUNT; ++i) {
            OP_CHECK_IF(ctx_.shapes[i] == nullptr,
                        OP_LOGE(ctx_.nodeName,
                                "all non-metadata Kernel C inputs are required"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(ctx_.chunkSize != 64,
                    OP_LOGE(ctx_.nodeName, "Kernel C requires chunk_size=64"),
                    return ge::GRAPH_FAILED);
        const bool hasCu = ctx_.shapes[KDA_C_CU_SEQLENS] != nullptr;
        const bool hasChunks = ctx_.shapes[KDA_C_CHUNK_INDICES] != nullptr;
        OP_CHECK_IF(hasCu != hasChunks,
                    OP_LOGE(ctx_.nodeName,
                            "cu_seqlens and chunk_indices must be both present or absent"),
                    return ge::GRAPH_FAILED);
        tiling_.isVarLen = hasCu ? 1 : 0;
        tiling_.dhHeadMajor = ctx_.dhHeadMajor ? 1 : 0;
        tiling_.useGateInKernel = ctx_.useGateInKernel ? 1 : 0;
        tiling_.lowerBound = ctx_.lowerBound;
        tiling_.hasDtBias =
            ctx_.shapes[KDA_C_DT_BIAS] != nullptr &&
            ctx_.shapes[KDA_C_DT_BIAS]->GetStorageShape().GetShapeSize() > 0 ?
                1 : 0;
        tiling_.rawGateIsBf16 =
            ctx_.useGateInKernel &&
            ctx_.types[KDA_C_RAW_G] == ge::DT_BF16 ? 1 : 0;

        const ge::DataType dataType = ctx_.types[KDA_C_Q];
        OP_CHECK_IF(dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16,
                    OP_LOGE(ctx_.nodeName,
                            "Kernel C data inputs must be FP16 or BF16"),
                    return ge::GRAPH_FAILED);
        for (size_t i : {KDA_C_K, KDA_C_V, KDA_C_V_NEW,
                         KDA_C_AKK, KDA_C_H, KDA_C_DH, KDA_C_DV_SCAN}) {
            OP_CHECK_IF(ctx_.types[i] != dataType,
                        OP_LOGE(ctx_.nodeName,
                                "Kernel C matrix inputs must share q dtype"),
                        return ge::GRAPH_FAILED);
        }
        for (size_t i : {KDA_C_GK, KDA_C_DQ_RAW, KDA_C_DAQK}) {
            OP_CHECK_IF(ctx_.types[i] != ge::DT_FLOAT,
                        OP_LOGE(ctx_.nodeName,
                                "gk/dq_raw/dAqk must be FP32"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(ctx_.types[KDA_C_BETA] != ge::DT_BF16 &&
                        ctx_.types[KDA_C_BETA] != ge::DT_FLOAT,
                    OP_LOGE(ctx_.nodeName, "beta must be BF16 or FP32"),
                    return ge::GRAPH_FAILED);
        if (ctx_.useGateInKernel) {
            OP_CHECK_IF(ctx_.shapes[KDA_C_RAW_G] == nullptr ||
                            ctx_.shapes[KDA_C_A_LOG] == nullptr,
                        OP_LOGE(ctx_.nodeName,
                                "raw_g and a_log are required when use_gate_in_kernel=true"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((ctx_.types[KDA_C_RAW_G] != ge::DT_FLOAT &&
                         ctx_.types[KDA_C_RAW_G] != ge::DT_BF16) ||
                            ctx_.types[KDA_C_A_LOG] != ge::DT_FLOAT ||
                            (tiling_.hasDtBias != 0 &&
                             ctx_.types[KDA_C_DT_BIAS] != ge::DT_FLOAT),
                        OP_LOGE(ctx_.nodeName,
                                "raw_g must be BF16/FP32; a_log/dt_bias must be FP32"),
                        return ge::GRAPH_FAILED);
        }

        const gert::Shape q = Shape(KDA_C_Q);
        const gert::Shape k = Shape(KDA_C_K);
        const gert::Shape v = Shape(KDA_C_V);
        const gert::Shape vNew = Shape(KDA_C_V_NEW);
        const gert::Shape gk = Shape(KDA_C_GK);
        const gert::Shape beta = Shape(KDA_C_BETA);
        const gert::Shape akk = Shape(KDA_C_AKK);
        const gert::Shape h = Shape(KDA_C_H);
        const gert::Shape dh = Shape(KDA_C_DH);
        const gert::Shape dvScan = Shape(KDA_C_DV_SCAN);
        const gert::Shape dqRaw = Shape(KDA_C_DQ_RAW);
        const gert::Shape dAqk = Shape(KDA_C_DAQK);

        const size_t tokenRank = hasCu ? 3 : 4;
        const size_t stateRank = hasCu ? 4 : 5;
        OP_CHECK_IF(q.GetDimNum() != tokenRank ||
                        k.GetDimNum() != tokenRank ||
                        v.GetDimNum() != tokenRank ||
                        vNew.GetDimNum() != tokenRank ||
                        gk.GetDimNum() != tokenRank ||
                        akk.GetDimNum() != tokenRank ||
                        dvScan.GetDimNum() != tokenRank ||
                        dqRaw.GetDimNum() != tokenRank ||
                        dAqk.GetDimNum() != tokenRank ||
                        h.GetDimNum() != stateRank,
                    OP_LOGE(ctx_.nodeName,
                            "Kernel C expects dense head-major rank-4/5 or varlen rank-3/4"),
                    return ge::GRAPH_FAILED);

        const size_t headAxis = hasCu ? 0 : 1;
        const size_t tokenAxis = hasCu ? 1 : 2;
        const size_t dimAxis = hasCu ? 2 : 3;
        tiling_.batch = hasCu ? 1 : static_cast<int64_t>(q.GetDim(0));
        tiling_.headNum = static_cast<int64_t>(q.GetDim(headAxis));
        tiling_.seqlen = static_cast<int64_t>(q.GetDim(tokenAxis));
        tiling_.keyDim = static_cast<int64_t>(q.GetDim(dimAxis));
        tiling_.valueDim = static_cast<int64_t>(v.GetDim(dimAxis));
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.scale = ctx_.scale;
        OP_CHECK_IF(tiling_.keyDim != 128 || tiling_.valueDim != 128,
                    OP_LOGE(ctx_.nodeName,
                            "Kernel C P0 requires K=128 and V=128"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(k.GetDim(dimAxis) != 128 ||
                        gk.GetDim(dimAxis) != 128 ||
                        dqRaw.GetDim(dimAxis) != 128 ||
                        vNew.GetDim(dimAxis) !=
                            static_cast<size_t>(tiling_.valueDim) ||
                        dvScan.GetDim(dimAxis) !=
                            static_cast<size_t>(tiling_.valueDim) ||
                        akk.GetDim(dimAxis) != 64 ||
                        dAqk.GetDim(dimAxis) != 64,
                    OP_LOGE(ctx_.nodeName,
                            "K/gk/dq_raw and V tensors or C matrices have invalid width"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!SameLeading(q, k, dimAxis) ||
                        !SameLeading(q, gk, dimAxis) ||
                        !SameLeading(q, dqRaw, dimAxis) ||
                        !SameLeading(q, akk, dimAxis) ||
                        !SameLeading(q, dAqk, dimAxis) ||
                        !SameLeading(q, v, dimAxis) ||
                        !SameLeading(q, vNew, dimAxis) ||
                        !SameLeading(q, dvScan, dimAxis),
                    OP_LOGE(ctx_.nodeName,
                            "all token tensors must share B/H/T or H/T"),
                    return ge::GRAPH_FAILED);
        if (ctx_.useGateInKernel) {
            const gert::Shape rawG = Shape(KDA_C_RAW_G);
            const gert::Shape aLog = Shape(KDA_C_A_LOG);
            OP_CHECK_IF(rawG.GetDim(dimAxis) != 128 ||
                            !SameLeading(q, rawG, dimAxis) ||
                            aLog.GetDimNum() != 1 ||
                            aLog.GetDim(0) !=
                                static_cast<size_t>(tiling_.headNum),
                        OP_LOGE(ctx_.nodeName,
                                "raw_g must match q and a_log must be [H]"),
                        return ge::GRAPH_FAILED);
            if (tiling_.hasDtBias != 0) {
                const gert::Shape bias = Shape(KDA_C_DT_BIAS);
                OP_CHECK_IF(bias.GetDimNum() != 2 ||
                                bias.GetDim(0) !=
                                    static_cast<size_t>(tiling_.headNum) ||
                                bias.GetDim(1) != 128,
                            OP_LOGE(ctx_.nodeName,
                                    "dt_bias must be [H,K]"),
                            return ge::GRAPH_FAILED);
            }
        }

        const size_t betaRank = hasCu ? 2 : 3;
        OP_CHECK_IF(beta.GetDimNum() != betaRank ||
                        (!hasCu && beta.GetDim(0) !=
                            static_cast<size_t>(tiling_.batch)) ||
                        beta.GetDim(hasCu ? 0 : 1) !=
                            static_cast<size_t>(tiling_.headNum) ||
                        beta.GetDim(hasCu ? 1 : 2) !=
                            static_cast<size_t>(tiling_.seqlen),
                    OP_LOGE(ctx_.nodeName,
                            "beta must be dense [B,H,T] or varlen [H,T]"),
                    return ge::GRAPH_FAILED);

        if (!hasCu) {
            tiling_.chunkNumPerBatch =
                (tiling_.seqlen + tiling_.chunkSize - 1) / tiling_.chunkSize;
            tiling_.chunkNum = tiling_.batch * tiling_.chunkNumPerBatch;
            OP_CHECK_IF(!CheckDenseState(h),
                        OP_LOGE(ctx_.nodeName,
                                "h must be [B,chunkNum,H,K,V]"),
                        return ge::GRAPH_FAILED);
            const bool validDh = !ctx_.validateIntermediateShapes ||
                (dh.GetDimNum() == 5 &&
                (ctx_.dhHeadMajor ? CheckDenseDhHeadMajor(dh) :
                                    CheckDenseState(dh)));
            if (!validDh) {
                OP_LOGE(ctx_.nodeName,
                        "dh must be [B,chunkNum,H,K,V] or PR291 [B,H,chunkNum,K,V]");
                return ge::GRAPH_FAILED;
            }
        } else {
            const gert::Shape cu = Shape(KDA_C_CU_SEQLENS);
            const gert::Shape chunks = Shape(KDA_C_CHUNK_INDICES);
            OP_CHECK_IF(cu.GetDimNum() != 1 || cu.GetDim(0) < 2 ||
                            chunks.GetDimNum() != 1 ||
                            chunks.GetDim(0) == 0 ||
                            (chunks.GetDim(0) & 1U) != 0,
                        OP_LOGE(ctx_.nodeName,
                                "varlen metadata requires cu_seqlens[N+1] and chunk_indices[2*Nc]"),
                        return ge::GRAPH_FAILED);
            tiling_.chunkNumPerBatch = 0;
            tiling_.chunkNum = static_cast<int64_t>(chunks.GetDim(0) / 2);
            OP_CHECK_IF(!CheckVarlenState(h),
                        OP_LOGE(ctx_.nodeName,
                                "varlen h must be [totalChunks,H,K,V]"),
                        return ge::GRAPH_FAILED);
            const bool validDh = !ctx_.validateIntermediateShapes ||
                (ctx_.dhHeadMajor ?
                (dh.GetDimNum() == 5 && CheckVarlenDhHeadMajor(dh)) :
                (dh.GetDimNum() == 4 && CheckVarlenState(dh)));
            if (!validDh) {
                OP_LOGE(ctx_.nodeName,
                        "varlen dh must be [totalChunks,H,K,V] or PR291 [1,H,totalChunks,K,V]");
                return ge::GRAPH_FAILED;
            }
        }

        const uint64_t headWindows =
            (static_cast<uint64_t>(tiling_.headNum) + 1U) / 2U;
        const uint64_t taskGroups =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindows;
        blockDim_ = static_cast<uint32_t>(std::min<uint64_t>(
            taskGroups, static_cast<uint64_t>(ctx_.aicCoreNum)));
        blockDim_ = blockDim_ == 0 ? 1 : blockDim_;
        tiling_.usedCoreNum = blockDim_;

        OP_CHECK_IF(BuildWorkspace() != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        workspaceSize_ = ctx_.systemWorkspaceSize +
            static_cast<size_t>(blockDim_) *
                static_cast<size_t>(tiling_.workspaceCoreSize);
        tilingKey_ = (ctx_.safeGate ? 1U : 3U) + (hasCu ? 1U : 0U) +
            (tiling_.valueDim == 256 ? 4U : 0U);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetBlockDim() const { return blockDim_; }
    size_t GetWorkspaceSize() const { return workspaceSize_; }
    uint64_t GetTilingKey() const { return tilingKey_; }

private:
    gert::Shape Shape(size_t index) const
    {
        return ctx_.shapes[index]->GetOriginShape();
    }

    bool SameLeading(
        const gert::Shape &lhs, const gert::Shape &rhs, size_t dimAxis) const
    {
        if (lhs.GetDimNum() != rhs.GetDimNum()) {
            return false;
        }
        for (size_t i = 0; i < dimAxis; ++i) {
            if (lhs.GetDim(i) != rhs.GetDim(i)) {
                return false;
            }
        }
        return true;
    }

    bool CheckDenseState(const gert::Shape &shape) const
    {
        return shape.GetDim(0) == static_cast<size_t>(tiling_.batch) &&
               shape.GetDim(1) ==
                   static_cast<size_t>(tiling_.chunkNumPerBatch) &&
               shape.GetDim(2) == static_cast<size_t>(tiling_.headNum) &&
               shape.GetDim(3) == 128 &&
               shape.GetDim(4) == static_cast<size_t>(tiling_.valueDim);
    }

    bool CheckVarlenState(const gert::Shape &shape) const
    {
        return shape.GetDim(0) == static_cast<size_t>(tiling_.chunkNum) &&
               shape.GetDim(1) == static_cast<size_t>(tiling_.headNum) &&
               shape.GetDim(2) == 128 &&
               shape.GetDim(3) == static_cast<size_t>(tiling_.valueDim);
    }

    bool CheckDenseDhHeadMajor(const gert::Shape &shape) const
    {
        return shape.GetDim(0) == static_cast<size_t>(tiling_.batch) &&
               shape.GetDim(1) == static_cast<size_t>(tiling_.headNum) &&
               shape.GetDim(2) ==
                   static_cast<size_t>(tiling_.chunkNumPerBatch) &&
               shape.GetDim(3) == 128 &&
               shape.GetDim(4) == static_cast<size_t>(tiling_.valueDim);
    }

    bool CheckVarlenDhHeadMajor(const gert::Shape &shape) const
    {
        return shape.GetDim(0) == 1 &&
               shape.GetDim(1) == static_cast<size_t>(tiling_.headNum) &&
               shape.GetDim(2) == static_cast<size_t>(tiling_.chunkNum) &&
               shape.GetDim(3) == 128 &&
               shape.GetDim(4) == static_cast<size_t>(tiling_.valueDim);
    }

    uint64_t Align512(uint64_t value) const
    {
        return (value + 511U) / 512U * 512U;
    }

    ge::graphStatus BuildWorkspace()
    {
        tiling_.workspaceSlotSize = KDA_C_SLOT_BYTES;
        tiling_.workspaceSlotCount = KDA_C_SLOT_COUNT;
        tiling_.workspaceCoreSize =
            tiling_.workspaceSlotCount * tiling_.workspaceSlotSize;

        tiling_.kEOffset = 0;
        tiling_.dqRawOffset = 16 * 1024;
        tiling_.dkRawOffset = 48 * 1024;
        tiling_.dWOffset = 80 * 1024;
        tiling_.zVOffset = 96 * 1024;
        tiling_.dVbOffset = 112 * 1024;
        if (tiling_.valueDim == 128) {
            tiling_.zWOffset = 144 * 1024;
            tiling_.dKgbOffset = 160 * 1024;
            tiling_.zaInputOffset = 192 * 1024;
            tiling_.zaOutputOffset = 224 * 1024;
        } else {
            // dVb is [64,256] FP32 and occupies [112,176) KiB.  Move the
            // later WY temporaries behind it; the slot remains 256 KiB.
            tiling_.zWOffset = 176 * 1024;
            tiling_.dKgbOffset = 184 * 1024;
            tiling_.zaInputOffset = 216 * 1024;
            tiling_.zaOutputOffset = 224 * 1024;
        }
        const uint64_t wyEnd = std::max<uint64_t>(
            std::max<uint64_t>(
                static_cast<uint64_t>(tiling_.zaInputOffset) +
                    64U * 64U * sizeof(float),
                static_cast<uint64_t>(tiling_.zaOutputOffset) +
                    64U * 64U * sizeof(uint16_t)),
            240U * 1024U + 2U * 128U * sizeof(float));
        OP_CHECK_IF(wyEnd > static_cast<uint64_t>(KDA_C_SLOT_BYTES),
                    OP_LOGE(ctx_.nodeName,
                            "derived WY slot exceeds the shared 256 KiB slot"),
                    return ge::GRAPH_FAILED);

        const uint64_t c = 64;
        const uint64_t k = 128;
        // Kernel C uses the mature 16-row Intra tile on every architecture.
        // Keep Host offsets derived from that same physical row count; using
        // the retired A5 row32 layout moves dK_lower to a region that the
        // row16 Cube result never writes.
        const uint64_t bc = 16;
        tiling_.intraALowerOffset = 0;
        tiling_.intraBLowerOffset = static_cast<int64_t>(
            Align512(2 * bc * c * sizeof(float)));
        tiling_.intraAUpperOffset = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(tiling_.intraBLowerOffset) +
                     c * k * sizeof(float)));
        tiling_.intraBUpperOffset = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(tiling_.intraAUpperOffset) +
                     2 * c * bc * sizeof(float)));
        const uint64_t inputs = Align512(
            static_cast<uint64_t>(tiling_.intraBUpperOffset) +
            2 * c * k * sizeof(float));
        tiling_.intraResultRegionOffset = static_cast<int64_t>(inputs);
        tiling_.intraResultDqOffset = 0;
        tiling_.intraResultDkLowerOffset = static_cast<int64_t>(
            Align512(bc * k * sizeof(float)));
        tiling_.intraResultDkUpperOffset = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(
                         tiling_.intraResultDkLowerOffset) +
                     bc * k * sizeof(float)));
        const uint64_t intraEnd =
            static_cast<uint64_t>(tiling_.intraResultRegionOffset) +
            Align512(static_cast<uint64_t>(
                         tiling_.intraResultDkUpperOffset) +
                     bc * k * sizeof(float));
        OP_CHECK_IF(intraEnd > static_cast<uint64_t>(KDA_C_SLOT_BYTES),
                    OP_LOGE(ctx_.nodeName,
                            "derived Intra slot exceeds the shared 256 KiB slot"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ChunkKdaBwdCTilingContext &ctx_;
    KDA::ChunkKdaBwdCTilingData &tiling_;
    uint32_t blockDim_ = 1;
    size_t workspaceSize_ = 0;
    uint64_t tilingKey_ = 1;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_C_TILING_PROCESSOR_H
