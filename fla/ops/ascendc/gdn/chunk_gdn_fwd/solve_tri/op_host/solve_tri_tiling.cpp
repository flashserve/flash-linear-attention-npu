/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "solve_tri_tiling.h"
 #include "register/op_impl_registry.h"
 #include "log/log.h"
 #include "tiling/platform/platform_ascendc.h"
 #include <string>
 namespace optiling {
 
 constexpr uint32_t INPUT_X_IDX = 0;
 constexpr uint32_t INPUT_CU_SEQLENS_IDX = 1;
 constexpr uint32_t INPUT_CHUNK_INDICES_IDX = 2;
 constexpr uint32_t OUTPUT_X_OUT_IDX = 0;
constexpr uint32_t ATTR_LAYOUT_IDX = 0;

static ge::graphStatus ValidateVarlenMetadata(gert::TilingContext *context, int64_t totalTokens,
                                              int64_t chunkSize, int64_t &totalChunks)
{
    const auto *cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    const auto *indexShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    if (cuShape == nullptr || indexShape == nullptr) {
        OP_LOGE(context->GetNodeName(),
                "cu_seqlens and chunk_indices must both be provided for tnd/ntd layout.");
        return ge::GRAPH_FAILED;
    }
    if (cuShape->GetStorageShape().GetDimNum() != 1 ||
        indexShape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must be rank-1 tensors.");
        return ge::GRAPH_FAILED;
    }
    const int64_t cuCount = cuShape->GetStorageShape().GetDim(0);
    const int64_t seqCount = cuCount - 1;
    const int64_t indexCount = indexShape->GetStorageShape().GetDim(0);
    if (seqCount <= 0 || indexCount <= 0 || indexCount % 2 != 0) {
        OP_LOGE(context->GetNodeName(),
                "varlen metadata requires cu_seqlens=[seq_count+1] and an even chunk_indices, "
                "but got cu_count=%ld and chunk_indices_count=%ld.",
                cuCount, indexCount);
        return ge::GRAPH_FAILED;
    }
    const auto *cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
    const auto *indexTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
    if (cuTensor == nullptr || indexTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "varlen metadata tensors are unavailable during tiling.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *cu = cuTensor->GetData<int64_t>();
    const int64_t *indices = indexTensor->GetData<int64_t>();
    if (cu == nullptr || indices == nullptr) {
        OP_LOGE(context->GetNodeName(), "varlen metadata value-depend data is null.");
        return ge::GRAPH_FAILED;
    }
    if (cu[0] != 0 || cu[seqCount] != totalTokens) {
        OP_LOGE(context->GetNodeName(),
                "cu_seqlens must start at 0 and end at total_T=%ld, but got [%ld, %ld].",
                totalTokens, cu[0], cu[seqCount]);
        return ge::GRAPH_FAILED;
    }
    int64_t expectedChunks = 0;
    for (int64_t seq = 0; seq < seqCount; ++seq) {
        const int64_t begin = cu[seq];
        const int64_t end = cu[seq + 1];
        if (begin < 0 || end <= begin || end > totalTokens) {
            OP_LOGE(context->GetNodeName(),
                    "cu_seqlens sequence %ld must satisfy 0 <= begin < end <= %ld, got [%ld, %ld].",
                    seq, totalTokens, begin, end);
            return ge::GRAPH_FAILED;
        }
        const int64_t length = end - begin;
        expectedChunks += (length - 1) / chunkSize + 1;
    }
    if (indexCount != expectedChunks * 2) {
        OP_LOGE(context->GetNodeName(),
                "chunk_indices count=%ld does not match expected pairs=%ld.",
                indexCount, expectedChunks * 2);
        return ge::GRAPH_FAILED;
    }
    int64_t pair = 0;
    for (int64_t seq = 0; seq < seqCount; ++seq) {
        const int64_t chunks = (cu[seq + 1] - cu[seq] - 1) / chunkSize + 1;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if (indices[pair * 2] != seq || indices[pair * 2 + 1] != chunk) {
                OP_LOGE(context->GetNodeName(),
                        "chunk_indices[%ld] must be [seq_id=%ld, chunk_id=%ld], but got [%ld, %ld].",
                        pair, seq, chunk, indices[pair * 2], indices[pair * 2 + 1]);
                return ge::GRAPH_FAILED;
            }
            ++pair;
        }
    }
    totalChunks = expectedChunks;
    return ge::GRAPH_SUCCESS;
}
 
 static ge::graphStatus SolveTriTilingFunc(gert::TilingContext* context)
 {
     auto platformInfo = context->GetPlatformInfo();
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
     int64_t coreNum = ascendcPlatform.GetCoreNumAic();
     if (coreNum == 0) {
         OP_LOGE(context->GetNodeName(), "SolveTri requires at least one AIC core.");
         return ge::GRAPH_FAILED;
     }
 
     // Get input shape:
     //   BNSD (mode 0): [B, H, S, BT]            (4D, chunks contiguous)
     //   BSND (mode 1): [B, S, H, BT]            (4D, chunks non-contiguous)
     //   TND  (mode 2): [total_T, H, BT]         (3D, varlen, chunks non-contiguous)
     //   NTD  (mode 3): [H, total_T, BT]         (3D, varlen, chunks contiguous; transpose of TND)
     auto inputShape = context->GetInputShape(INPUT_X_IDX);
     if (inputShape == nullptr) {
         OP_LOGE(context->GetNodeName(), "SolveTri input x shape is unavailable.");
         return ge::GRAPH_FAILED;
     }
     auto shape = inputShape->GetStorageShape();
     int64_t ndim = shape.GetDimNum();
     if (ndim != 3 && ndim != 4) {
         OP_LOGE(context->GetNodeName(), "SolveTri input x must have rank 3 or 4, got %ld.", ndim);
         return ge::GRAPH_FAILED;
     }
 
     // Get layout attribute to determine shape parsing
     auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "SolveTri layout attributes are unavailable.");
        return ge::GRAPH_FAILED;
    }
     const char *layoutStr = attrs->GetStr(ATTR_LAYOUT_IDX);
     std::string layout = layoutStr ? layoutStr : "bsnd";
 
     // layoutMode: 0=BNSD, 1=BSND, 2=TND, 3=NTD
     // "bhtd" 保留为 mode 0 的兼容别名（等价于 bnsd）
     int64_t layoutMode = 1;  // default to BSND
     if (layout == "bnsd" || layout == "bhtd") {
         layoutMode = 0;
     } else if (layout == "bsnd") {
         layoutMode = 1;
     } else if (layout == "tnd") {
         layoutMode = 2;
     } else if (layout == "ntd") {
         layoutMode = 3;
    } else {
        OP_LOGE(context->GetNodeName(),
                "layout must be one of bsnd, bnsd, bhtd, tnd, ntd, but got %s.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((layoutMode == 0 || layoutMode == 1) != (ndim == 4)) {
        OP_LOGE(context->GetNodeName(),
                "layout %s requires rank %s, but got rank %ld.", layout.c_str(),
                (layoutMode == 0 || layoutMode == 1) ? "4" : "3", ndim);
        return ge::GRAPH_FAILED;
     }
 
     int64_t B, H, T, BT;
     if (ndim == 4) {
         if (layoutMode == 0) {
             // BNSD: [B, H, S, BT]
             B = shape.GetDim(0);
             H = shape.GetDim(1);
             T = shape.GetDim(2);
             BT = shape.GetDim(3);
         } else {
             // BSND: [B, S, H, BT]
             B = shape.GetDim(0);
             T = shape.GetDim(1);
             H = shape.GetDim(2);
             BT = shape.GetDim(3);
         }
     } else {
         // 3D varlen
         if (layoutMode == 3) {
             // NTD: [H, total_T, BT]
             B = 1;
             H = shape.GetDim(0);
             T = shape.GetDim(1);
             BT = shape.GetDim(2);
         } else {
             // TND: [total_T, H, BT]
             B = 1;
             T = shape.GetDim(0);
             H = shape.GetDim(1);
             BT = shape.GetDim(2);
         }
     }
 
     int64_t chunkSize = BT;
    if (B <= 0 || H <= 0 || T <= 0 ||
        !(chunkSize == 16 || chunkSize == 32 || chunkSize == 64 || chunkSize == 128)) {
        OP_LOGE(context->GetNodeName(),
                "SolveTri requires positive B/H/T and chunkSize in {16,32,64,128}, "
                "but got B=%ld, H=%ld, T=%ld, chunkSize=%ld.",
                B, H, T, chunkSize);
        return ge::GRAPH_FAILED;
    }
 
    // isVarlen only for TND/NTD mode
    int64_t isVarlen = (layoutMode == 2 || layoutMode == 3) ? 1 : 0;
    int64_t hasCuSeqlens = isVarlen;

    // totalTokens: NTD 偏移计算需要（= T，即 total_T）；其余模式置 0
    int64_t totalTokens = (layoutMode == 3) ? T : 0;

    int64_t totalChunks = 0;
    int64_t numChunks = 0;
    int64_t totalTiles = 0;
    int64_t lastChunkValidSize = 0;

    if (isVarlen) {
        if (ValidateVarlenMetadata(context, T, chunkSize, totalChunks) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        totalTiles = totalChunks * H;
        numChunks = 0;
        lastChunkValidSize = 0;
    } else {
        totalChunks = 0;
        numChunks = (T + chunkSize - 1) / chunkSize;
        totalTiles = B * numChunks * H;
        int64_t remainder = T % chunkSize;
        lastChunkValidSize = (remainder == 0) ? chunkSize : remainder;
    }

    // chunk=16：每个 AIC 一轮处理 8 个 tile（2 Vector × 4 leaves）
    // chunk=32：每个 AIC 一轮处理 2 个 tile（单 Vector × 4 个 16×16 叶子）
    constexpr int64_t kTilesPerAicBatch16 = 8;
    constexpr int64_t kTilesPerAicBatch32 = 2;
    int64_t tilesPerCore = 0;
    int64_t usedCoreNum = 0;
    if (chunkSize == 16 || chunkSize == 32) {
        int64_t tilesPerBatch = (chunkSize == 16) ? kTilesPerAicBatch16 : kTilesPerAicBatch32;
        int64_t totalBatches = (totalTiles + tilesPerBatch - 1) / tilesPerBatch;
        if (totalBatches < 1) {
            totalBatches = 1;
        }
        int64_t batchesPerCore = (totalBatches + coreNum - 1) / coreNum;
        if (batchesPerCore < 1) {
            batchesPerCore = 1;
        }
        tilesPerCore = batchesPerCore * tilesPerBatch;
        usedCoreNum = (totalBatches + batchesPerCore - 1) / batchesPerCore;
    } else {
        tilesPerCore = (totalTiles + coreNum - 1) / coreNum;
        usedCoreNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // Get input dtype: 0=fp16, 1=bf16
    const auto *inputDesc = context->GetInputDesc(INPUT_X_IDX);
    const auto *outputDesc = context->GetOutputDesc(OUTPUT_X_OUT_IDX);
    if (inputDesc == nullptr || outputDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "SolveTri input or output descriptor is null.");
        return ge::GRAPH_FAILED;
    }
    auto inputDtype = inputDesc->GetDataType();
    int64_t dtypeMode = 0;  // default fp16
    if (inputDtype == ge::DT_BF16) {
        dtypeMode = 1;
    } else if (inputDtype != ge::DT_FLOAT16) {
        OP_LOGE(context->GetNodeName(), "SolveTri input dtype must be FP16 or BF16, got %d.",
                static_cast<int>(inputDtype));
        return ge::GRAPH_FAILED;
    }
    if (outputDesc->GetDataType() != inputDtype) {
        OP_LOGE(context->GetNodeName(), "SolveTri input and output dtype must match.");
        return ge::GRAPH_FAILED;
    }

    // Set tiling data
    SolveTriTilingData tiling;
    tiling.set_totalTiles(totalTiles);
    tiling.set_matrixSize(chunkSize);
    tiling.set_numHeads(H);
    tiling.set_seqLen(T);
    tiling.set_batchSize(B);
    tiling.set_isLower(1);
    tiling.set_hasCuSeqlens(hasCuSeqlens);
    tiling.set_tilesPerCore(tilesPerCore);
    tiling.set_chunkSize(chunkSize);
    tiling.set_numChunks(numChunks);
    tiling.set_lastChunkValidSize(lastChunkValidSize);
    tiling.set_isVarlen(isVarlen);
    tiling.set_totalChunks(totalChunks);
    tiling.set_layoutMode(layoutMode);
    tiling.set_dtypeMode(dtypeMode);
    tiling.set_totalTokens(totalTokens);
 
     // tilingKey = chunkSize（16/32/64/128）；ascend950 按 key 分发到不同 kernel 类
     // 910b 路径同样按 chunkSize 设 key，kernel 入口按 key 进入统一实现
     uint64_t tilingKey = static_cast<uint64_t>(chunkSize);
     context->SetTilingKey(tilingKey);
     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                         context->GetRawTilingData()->GetCapacity());
     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
 
     context->SetBlockDim(usedCoreNum);
 
     // Workspace: ascend950 全程片上缓存，仅预留系统 workspace；910b 需 GM 辅助矩阵中转区
     uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
     size_t* ws = context->GetWorkspaceSizes(1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // FP32 MBH：每 AIC 核一块 side×side FP32 NZ 中转（Fixpipe ChannelSplit L0C→GM→L1）
    // chunk16 纯 Vector，不需要中转；chunk32 把两个 32×32 打进 64×64 工作区
    size_t mbhSide = static_cast<size_t>(chunkSize);
    if (chunkSize == 16) {
        mbhSide = 0;
    } else if (chunkSize == 32) {
        mbhSide = 64;
    }
    size_t userWorkspaceSize = static_cast<size_t>(usedCoreNum) * mbhSide * mbhSide * sizeof(float);
    userWorkspaceSize = ((userWorkspaceSize + 511) / 512) * 512;
    ws[0] = userWorkspaceSize + sysWorkspaceSize;
#else
     size_t userWorkspaceSize;
     if (chunkSize == 64) {
         constexpr size_t fp32WorkspaceSlots = 4;
         constexpr size_t fp32TileBatch = 16;
         constexpr size_t fp32PipeBatches = 4;
         constexpr size_t fp32WorkspaceStride = 64;
         userWorkspaceSize =
             usedCoreNum * fp32PipeBatches * fp32TileBatch * fp32WorkspaceSlots * chunkSize *
             fp32WorkspaceStride * sizeof(float);
     } else {
         // FP32 Vector substitution keeps A and its inverse in UB.
         userWorkspaceSize = 0;
     }
     userWorkspaceSize = ((userWorkspaceSize + 511) / 512) * 512;
     ws[0] = userWorkspaceSize + sysWorkspaceSize;
#endif
     return ge::GRAPH_SUCCESS;
 }
 
 static ge::graphStatus SolveTriTilingParse(gert::TilingParseContext* context)
 {
     return ge::GRAPH_SUCCESS;
 }
 
 struct SolveTriCompileInfo {};
 
 IMPL_OP_OPTILING(SolveTri)
     .Tiling(SolveTriTilingFunc)
     .TilingParse<SolveTriCompileInfo>(SolveTriTilingParse);
 
 }  // namespace optiling
