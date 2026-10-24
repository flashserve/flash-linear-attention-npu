#ifndef CHUNK_KDA_BWD_FINALIZE_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_FINALIZE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../../op_kernel/chunk_kda_bwd_finalize_struct.h"

namespace optiling {

struct ChunkKdaBwdFinalizeTilingContext {
    const char *nodeName;
    const gert::StorageShape *q;
    const gert::StorageShape *k;
    const gert::StorageShape *v;
    const gert::StorageShape *gk;
    const gert::StorageShape *beta;
    const gert::StorageShape *akk;
    const gert::StorageShape *h;
    const gert::StorageShape *dh;
    const gert::StorageShape *dvScan;
    const gert::StorageShape *qRstd;
    const gert::StorageShape *kRstd;
    const gert::StorageShape *cuSeqlens;
    const gert::StorageShape *chunkIndices;
    double scale;
    double lowerBound;
    int64_t chunkSize;
    bool safeGate;
    bool useGateInKernel;
    bool useExp2;
    bool stateVFirst;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class ChunkKdaBwdFinalizeTilingProcessor {
public:
    ChunkKdaBwdFinalizeTilingProcessor(
        ChunkKdaBwdFinalizeTilingContext &ctx,
        KDA::ChunkKdaBwdFinalizeTilingData &tiling)
        : ctx_(ctx), tiling_(tiling) {}

    ge::graphStatus Process()
    {
        if (ctx_.q == nullptr || ctx_.k == nullptr || ctx_.v == nullptr ||
            ctx_.gk == nullptr || ctx_.beta == nullptr || ctx_.akk == nullptr ||
            ctx_.h == nullptr || ctx_.dh == nullptr || ctx_.dvScan == nullptr) {
            OP_LOGE(ctx_.nodeName, "required Stage0--2 input shape is null");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.chunkSize != 64 || !ctx_.safeGate || !ctx_.useGateInKernel ||
            !ctx_.useExp2 || ctx_.stateVFirst) {
            OP_LOGE(ctx_.nodeName,
                    "A5 v1 requires chunk_size=64, safe_gate/use_gate/use_exp2=true, state_v_first=false");
            return ge::GRAPH_FAILED;
        }
        const bool hasRstd = ctx_.qRstd != nullptr || ctx_.kRstd != nullptr;
        if ((ctx_.qRstd == nullptr) != (ctx_.kRstd == nullptr)) {
            OP_LOGE(ctx_.nodeName, "q_rstd and k_rstd must appear together");
            return ge::GRAPH_FAILED;
        }
        const bool variable = ctx_.cuSeqlens != nullptr || ctx_.chunkIndices != nullptr;
        if ((ctx_.cuSeqlens == nullptr) != (ctx_.chunkIndices == nullptr)) {
            OP_LOGE(ctx_.nodeName, "cu_seqlens and chunk_indices must appear together");
            return ge::GRAPH_FAILED;
        }

        const gert::Shape q = ctx_.q->GetStorageShape();
        const gert::Shape k = ctx_.k->GetStorageShape();
        const gert::Shape v = ctx_.v->GetStorageShape();
        const gert::Shape gk = ctx_.gk->GetStorageShape();
        const gert::Shape beta = ctx_.beta->GetStorageShape();
        const gert::Shape akk = ctx_.akk->GetStorageShape();
        const gert::Shape h = ctx_.h->GetStorageShape();
        const gert::Shape dh = ctx_.dh->GetStorageShape();
        const size_t tokenRank = q.GetDimNum();
        const size_t scalarRank = tokenRank - 1;
        const size_t stateRank = tokenRank + 1;
        if ((tokenRank != 3 && tokenRank != 4) || k.GetDimNum() != tokenRank ||
            v.GetDimNum() != tokenRank || gk.GetDimNum() != tokenRank ||
            akk.GetDimNum() != tokenRank || beta.GetDimNum() != scalarRank ||
            h.GetDimNum() != stateRank || dh.GetDimNum() != stateRank) {
            OP_LOGE(ctx_.nodeName, "dense/varlen rank does not match KernelC contract");
            return ge::GRAPH_FAILED;
        }

        const size_t base = tokenRank == 3 ? 0 : 1;
        tiling_.B = base == 0 ? 1 : k.GetDim(0);
        tiling_.NQ = q.GetDim(base);
        tiling_.NV = k.GetDim(base);
        tiling_.T = k.GetDim(base + 1);
        tiling_.K = k.GetDim(base + 2);
        tiling_.V = v.GetDim(base + 2);
        if (variable) {
            if (tiling_.B != 1) return ge::GRAPH_FAILED;
            tiling_.denseChunkNum = 0;
            tiling_.totalChunkNum = h.GetDim(base);
            const gert::Shape cu = ctx_.cuSeqlens->GetStorageShape();
            const gert::Shape indices = ctx_.chunkIndices->GetStorageShape();
            if (cu.GetDimNum() != 1 || indices.GetDimNum() != 1 || cu.GetDim(0) < 2 ||
                indices.GetDim(0) != 2 * tiling_.totalChunkNum) {
                OP_LOGE(ctx_.nodeName, "invalid varlen metadata shape");
                return ge::GRAPH_FAILED;
            }
            tiling_.seqNum = cu.GetDim(0) - 1;
        } else {
            tiling_.denseChunkNum = h.GetDim(base);
            tiling_.totalChunkNum = tiling_.B * tiling_.denseChunkNum;
            tiling_.seqNum = tiling_.B;
        }
        if (tiling_.B <= 0 || tiling_.NQ != tiling_.NV || tiling_.NV <= 0 ||
            tiling_.T <= 0 || tiling_.K != 128 || tiling_.V != 128 ||
            tiling_.totalChunkNum <= 0) {
            OP_LOGE(ctx_.nodeName, "A5 v1 requires NQ=NV, K=V=128 and nonempty tensors");
            return ge::GRAPH_FAILED;
        }
        const int64_t chunks = variable ? tiling_.totalChunkNum : tiling_.denseChunkNum;
        const bool validStates = (base == 0 || (h.GetDim(0) == tiling_.B && dh.GetDim(0) == tiling_.B)) &&
            h.GetDim(base + 1) == tiling_.NV && h.GetDim(base + 2) == tiling_.K &&
            h.GetDim(base + 3) == tiling_.V && dh.GetDim(base) == chunks &&
            dh.GetDim(base + 1) == tiling_.NV && dh.GetDim(base + 2) == tiling_.K &&
            dh.GetDim(base + 3) == tiling_.V &&
            (variable || chunks == (tiling_.T + ctx_.chunkSize - 1) / ctx_.chunkSize);
        if (!validStates) {
            OP_LOGE(ctx_.nodeName, "expected NT-first h and dh");
            return ge::GRAPH_FAILED;
        }
        tiling_.chunkTaskNum = tiling_.totalChunkNum;
        tiling_.headWindowNum = (tiling_.NV + 1) / 2;
        tiling_.workTaskNum = tiling_.headWindowNum * tiling_.chunkTaskNum;
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.isVariable = variable ? 1U : 0U;
        tiling_.hasQkL2Norm = hasRstd ? 1U : 0U;
        tiling_.safeGate = ctx_.safeGate ? 1U : 0U;
        tiling_.useGateInKernel = ctx_.useGateInKernel ? 1U : 0U;
        tiling_.useExp2 = ctx_.useExp2 ? 1U : 0U;
        tiling_.stateVFirst = ctx_.stateVFirst ? 1U : 0U;
        tiling_.scale = static_cast<float>(ctx_.scale);
        tiling_.lowerBound = static_cast<float>(ctx_.lowerBound);
        blockDim_ = static_cast<uint32_t>(std::min<int64_t>(
            ctx_.aicCoreNum == 0 ? 1U : ctx_.aicCoreNum, tiling_.workTaskNum));
        const size_t slotBytes = static_cast<size_t>(blockDim_) * 4U * 160U * 1024U;
        const size_t taskHeads = static_cast<size_t>(tiling_.totalChunkNum * tiling_.NV);
        // Give each concurrently written scalar partial a complete DMA block.
        const size_t gateBytes = Align512(taskHeads * 32U);
        const size_t dtBytes = Align512(taskHeads * 128U * sizeof(float));
        tiling_.slotWorkspaceOffset = 0;
        tiling_.gatePartialOffset = slotBytes;
        tiling_.dtBiasPartialOffset = slotBytes + gateBytes;
        workspaceSize_ = ctx_.sysWorkspaceSize + slotBytes + gateBytes + dtBytes;
        tilingKey_ = (variable ? 2U : 1U) + (hasRstd ? 2U : 0U);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetBlockDim() const { return blockDim_; }
    uint32_t GetTilingKey() const { return tilingKey_; }
    size_t GetWorkspaceSize() const { return workspaceSize_; }

private:
    static size_t Align512(size_t value) { return (value + 511U) / 512U * 512U; }
    ChunkKdaBwdFinalizeTilingContext &ctx_;
    KDA::ChunkKdaBwdFinalizeTilingData &tiling_;
    uint32_t blockDim_ = 1;
    uint32_t tilingKey_ = 1;
    size_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_FINALIZE_TILING_PROCESSOR_H
