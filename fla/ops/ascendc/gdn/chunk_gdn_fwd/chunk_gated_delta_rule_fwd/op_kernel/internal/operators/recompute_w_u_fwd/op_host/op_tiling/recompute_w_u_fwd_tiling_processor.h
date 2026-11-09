/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef GDN_MEGA_ARCH35_RECOMPUTE_W_U_FWD_TILING_PROCESSOR_H
#define GDN_MEGA_ARCH35_RECOMPUTE_W_U_FWD_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../../op_kernel/recompute_w_u_fwd_struct.h"
#include "op_common/log/log.h"
#include "../../../../../../../../../common/host_utils/log.h"

using GDN::GdnMegaArch35RecomputeWUTilingData;

namespace optiling {
namespace {

// Keep Phase6 helpers local to this translation unit. The standalone WU
// processor has the same names but a different tiling layout and algorithm.
// External inline symbols would let the linker substitute either definition.

static constexpr int64_t RECOMPUTE_W_U_FWD_V_DIM_128 = 128;
static constexpr int64_t RECOMPUTE_W_U_FWD_V_DIM_256 = 256;

static constexpr size_t RECOMPUTE_W_U_FWD_INPUT_K_IDX = 0;
static constexpr size_t RECOMPUTE_W_U_FWD_INPUT_V_IDX = 1;
static constexpr size_t RECOMPUTE_W_U_FWD_INPUT_BETA_IDX = 2;
static constexpr size_t RECOMPUTE_W_U_FWD_INPUT_A_IDX = 3;
static constexpr size_t RECOMPUTE_W_U_FWD_INPUT_G_IDX = 4;
static constexpr size_t RECOMPUTE_W_U_FWD_INPUT_SEQLENS_IDX = 6;
static constexpr size_t RECOMPUTE_W_U_FWD_INPUT_CHUNK_INDICES_IDX = 7;

static constexpr size_t RECOMPUTE_W_U_FWD_DIM_NUM_3 = 3;
static constexpr size_t RECOMPUTE_W_U_FWD_DIM_NUM_4 = 4;

static constexpr size_t RECOMPUTE_W_U_FWD_DIM_0 = 0;
static constexpr size_t RECOMPUTE_W_U_FWD_DIM_1 = 1;
static constexpr size_t RECOMPUTE_W_U_FWD_DIM_2 = 2;
static constexpr size_t RECOMPUTE_W_U_FWD_DIM_3 = 3;

static constexpr int64_t RECOMPUTE_W_U_FWD_CHUNK_SIZE_64 = 64;
static constexpr int64_t RECOMPUTE_W_U_FWD_CHUNK_SIZE_128 = 128;
static constexpr int64_t RECOMPUTE_W_U_FWD_VAR_LEN_B_DIM_1 = 1;

static constexpr const char *const RECOMPUTE_W_U_FWD_INPUT_K_NAME = "k";
static constexpr const char *const RECOMPUTE_W_U_FWD_INPUT_V_NAME = "v";
static constexpr const char *const RECOMPUTE_W_U_FWD_INPUT_BETA_NAME = "beta";
static constexpr const char *const RECOMPUTE_W_U_FWD_INPUT_A_NAME = "A";
static constexpr const char *const RECOMPUTE_W_U_FWD_INPUT_G_NAME = "g";
static constexpr const char *const RECOMPUTE_W_U_FWD_INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const RECOMPUTE_W_U_FWD_INPUT_SEQLENS_NAME = "cu_seqlens";

static constexpr uint64_t RECOMPUTE_W_U_FWD_SIZE_HALF = 2;
static constexpr uint64_t RECOMPUTE_W_U_FWD_SIZE_FP32 = 4;
static constexpr uint64_t RECOMPUTE_W_U_FWD_ONE_BLOCK_32 = 32;

static constexpr uint64_t RECOMPUTE_W_U_FWD_INTERLEAVED_UB_RESERVED_BYTES = 16 * 1024;
static constexpr uint64_t RECOMPUTE_W_U_FWD_INTERLEAVED_MIN_VEC_ROW = 8;
static constexpr uint32_t RECOMPUTE_W_U_FWD_A5_GM_RING_DEPTH = 8;

struct RecomputeWUFwdTilingContext {
    const char *nodeName;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *aShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    const int64_t *cuSeqlensData;
    const int64_t *chunkIndicesData;
    int32_t chunkSize;
    ge::DataType kDtype;
    ge::DataType betaDtype;
    uint64_t ubSize;
    size_t sysWorkspaceSize;
    uint64_t aicCoreNum;
    bool enableA5CompactWorkspace;
};

class RecomputeWUFwdTilingProcessor {
    RecomputeWUFwdTilingContext &ctx_;
    GdnMegaArch35RecomputeWUTilingData &tiling_;
    size_t workspaceSize_ = 0;
    int64_t B = 0;
    int64_t Hk = 0;
    int64_t Hv = 0;
    int64_t hvPerHk = 1;
    int64_t K = 0;
    int64_t V = 0;
    int64_t T = 0;
    int64_t chunkSize = 0;

public:
    explicit RecomputeWUFwdTilingProcessor(RecomputeWUFwdTilingContext &ctx, GdnMegaArch35RecomputeWUTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr;
    }

    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                             const char *inputName)
    {
        if (curShape == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(ctx_.nodeName, inputName);
            return ge::GRAPH_FAILED;
        }
        const gert::Shape storageShape = curShape->GetStorageShape();
        size_t dimNum = storageShape.GetDimNum();
        if (dimNum != validDimNum) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(ctx_.nodeName, inputName,
                "rank=" + std::to_string(dimNum), "rank=" + std::to_string(validDimNum));
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, RECOMPUTE_W_U_FWD_DIM_NUM_4, RECOMPUTE_W_U_FWD_INPUT_K_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, RECOMPUTE_W_U_FWD_DIM_NUM_4, RECOMPUTE_W_U_FWD_INPUT_V_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.betaShape, RECOMPUTE_W_U_FWD_DIM_NUM_3,
                                             RECOMPUTE_W_U_FWD_INPUT_BETA_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.aShape, RECOMPUTE_W_U_FWD_DIM_NUM_4, RECOMPUTE_W_U_FWD_INPUT_A_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, RECOMPUTE_W_U_FWD_DIM_NUM_3, RECOMPUTE_W_U_FWD_INPUT_G_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                 const char *inputName2, size_t compareDimNum)
    {
        size_t shapeDim1 = 0;
        size_t shapeDim2 = 0;
        for (size_t dimIndex = 0; dimIndex < compareDimNum; dimIndex++) {
            shapeDim1 = shape1.GetDim(dimIndex);
            shapeDim2 = shape2.GetDim(dimIndex);
            if (shapeDim1 != shapeDim2) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(ctx_.nodeName,
                    (std::string(inputName1) + "," + inputName2).c_str(), "dim mismatch", "same shape");
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetKbgExpVecRow(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
    {
        uint64_t rowNum = static_cast<uint64_t>(chunkSize);
        uint64_t sizeofKType = RECOMPUTE_W_U_FWD_SIZE_FP32;
        uint64_t sizeofBetaType = RECOMPUTE_W_U_FWD_SIZE_FP32;
        if (kType != ge::DataType::DT_FLOAT) {
            sizeofKType = RECOMPUTE_W_U_FWD_SIZE_HALF;
        }
        if (betaType != ge::DataType::DT_FLOAT) {
            sizeofBetaType = RECOMPUTE_W_U_FWD_SIZE_HALF;
        }
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(K) * sizeofKType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(K) * sizeofKType;
            useUbSize += rowNum * static_cast<uint64_t>(K) * sizeof(float);
            useUbSize += rowNum * sizeof(float);
            useUbSize += rowNum * sizeof(float);
            useUbSize += rowNum * RECOMPUTE_W_U_FWD_ONE_BLOCK_32;

            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.kbgExpVecRow = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetVbVecRow(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
    {
        uint64_t rowNum = static_cast<uint64_t>(chunkSize);
        uint64_t sizeofKType = RECOMPUTE_W_U_FWD_SIZE_FP32;
        uint64_t sizeofBetaType = RECOMPUTE_W_U_FWD_SIZE_FP32;
        if (kType != ge::DataType::DT_FLOAT) {
            sizeofKType = RECOMPUTE_W_U_FWD_SIZE_HALF;
        }
        if (betaType != ge::DataType::DT_FLOAT) {
            sizeofBetaType = RECOMPUTE_W_U_FWD_SIZE_HALF;
        }
        while (rowNum >= 8) {
            uint64_t useUbSize = 0;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(V) * sizeofKType;
            useUbSize += 2 * rowNum * sizeofBetaType;
            useUbSize += 2 * rowNum * static_cast<uint64_t>(V) * sizeofKType;
            useUbSize += rowNum * static_cast<uint64_t>(V) * sizeof(float);
            useUbSize += rowNum * sizeof(float);
            useUbSize += rowNum * RECOMPUTE_W_U_FWD_ONE_BLOCK_32;

            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        tiling_.vbVecRow = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetInterleavedVecRow(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
    {
        const uint64_t sizeofKType =
            kType == ge::DataType::DT_FLOAT ? RECOMPUTE_W_U_FWD_SIZE_FP32 : RECOMPUTE_W_U_FWD_SIZE_HALF;
        const uint64_t sizeofBetaType =
            betaType == ge::DataType::DT_FLOAT ? RECOMPUTE_W_U_FWD_SIZE_FP32 : RECOMPUTE_W_U_FWD_SIZE_HALF;
        if (ubSize <= RECOMPUTE_W_U_FWD_INTERLEAVED_UB_RESERVED_BYTES) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "ubSize", std::to_string(ubSize),
                ">" + std::to_string(RECOMPUTE_W_U_FWD_INTERLEAVED_UB_RESERVED_BYTES));
            return ge::GRAPH_FAILED;
        }

        const uint64_t usableUbSize = ubSize - RECOMPUTE_W_U_FWD_INTERLEAVED_UB_RESERVED_BYTES;
        const uint64_t dataDim = static_cast<uint64_t>(K > V ? K : V);
        // data-in, data-out, beta and g queues are all double buffered. The two
        // data queues use max(K, V) and are reused between Vb and KbgExp.
        const uint64_t bytesPerRow = 4 * (dataDim * sizeofKType + sizeofBetaType);
        uint64_t rowNum = static_cast<uint64_t>(chunkSize);
        uint64_t useUbSize = rowNum * bytesPerRow;
        while (rowNum > RECOMPUTE_W_U_FWD_INTERLEAVED_MIN_VEC_ROW && useUbSize > usableUbSize) {
            rowNum /= 2;
            useUbSize = rowNum * bytesPerRow;
        }
        if (rowNum < RECOMPUTE_W_U_FWD_INTERLEAVED_MIN_VEC_ROW || useUbSize > usableUbSize) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "ubSize", std::to_string(ubSize), "sufficient for A5 interleaved path");
            return ge::GRAPH_FAILED;
        }

        // Keep the fused trailer layout unchanged: the A5 vector path consumes
        // vbVecRow as its shared interleaved row count; the legacy path retains
        // the independently calculated values on non-A5 devices.
        tiling_.vbVecRow = static_cast<int64_t>(rowNum);
        tiling_.kbgExpVecRow = static_cast<int64_t>(rowNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape kStorageShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vStorageShape = ctx_.vShape->GetStorageShape();
        const gert::Shape betaStorageShape = ctx_.betaShape->GetStorageShape();
        const gert::Shape AStorageShape = ctx_.aShape->GetStorageShape();
        const gert::Shape gStorageShape = ctx_.gShape->GetStorageShape();
        B = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0));
        Hk = static_cast<int64_t>(kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_1));
        Hv = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_1));
        if (Hk <= 0 || Hv <= 0) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "Hk,Hv", "non-positive", "positive");
            return ge::GRAPH_FAILED;
        }
        if (Hv % Hk != 0) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "Hv,Hk", "not divisible", "Hv % Hk == 0");
            return ge::GRAPH_FAILED;
        }
        hvPerHk = Hv / Hk;
        if (vStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0) != kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(ctx_.nodeName, "v,k", "B mismatch", "same B");
            return ge::GRAPH_FAILED;
        }
        if (vStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_2) != kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_2)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(ctx_.nodeName, "v,k", "T mismatch", "same T");
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, RECOMPUTE_W_U_FWD_INPUT_BETA_NAME,
                                 RECOMPUTE_W_U_FWD_INPUT_G_NAME, RECOMPUTE_W_U_FWD_DIM_NUM_3) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        if (kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0) != gStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(ctx_.nodeName, "k,g", "B mismatch", "same B");
            return ge::GRAPH_FAILED;
        }
        if (kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_2) != gStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_2)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(ctx_.nodeName, "k,g", "T mismatch", "same T");
            return ge::GRAPH_FAILED;
        }
        if (kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0) != AStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(ctx_.nodeName, "k,A", "B mismatch", "same B");
            return ge::GRAPH_FAILED;
        }
        if (kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_2) != AStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_2)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(ctx_.nodeName, "k,A", "T mismatch", "same T");
            return ge::GRAPH_FAILED;
        }
        if (AStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_1) != Hv) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(ctx_.nodeName, "A", "head mismatch", "A H == Hv");
            return ge::GRAPH_FAILED;
        }
        if (betaStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_1) != Hv) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(ctx_.nodeName, "beta", "head mismatch", "beta H == Hv");
            return ge::GRAPH_FAILED;
        }
        if (gStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_1) != Hv) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(ctx_.nodeName, "g", "head mismatch", "g H == Hv");
            return ge::GRAPH_FAILED;
        }
        T = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_2));
        K = static_cast<int64_t>(kStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_3));
        V = static_cast<int64_t>(vStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_3));
        tiling_.B = B;
        tiling_.Hk = Hk;
        tiling_.Hv = Hv;
        tiling_.hvPerHk = hvPerHk;
        tiling_.T = T;
        tiling_.K = K;
        tiling_.V = V;
        if (V != RECOMPUTE_W_U_FWD_V_DIM_128 && V != RECOMPUTE_W_U_FWD_V_DIM_256) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "V", std::to_string(V), "128 or 256");
            return ge::GRAPH_FAILED;
        }
        chunkSize = static_cast<int64_t>(ctx_.chunkSize);
        if (chunkSize != RECOMPUTE_W_U_FWD_CHUNK_SIZE_64 && chunkSize != RECOMPUTE_W_U_FWD_CHUNK_SIZE_128) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "chunkSize", std::to_string(chunkSize), "64 or 128");
            return ge::GRAPH_FAILED;
        }
        tiling_.chunkSize = chunkSize;

        OP_CHECK_IF(SetVbVecRow(ctx_.ubSize, ctx_.kDtype, ctx_.betaDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetVbVecRow Failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetKbgExpVecRow(ctx_.ubSize, ctx_.kDtype, ctx_.betaDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetKbgExpVecRow Failed."), return ge::GRAPH_FAILED);
        if (ctx_.enableA5CompactWorkspace) {
            OP_CHECK_IF(SetInterleavedVecRow(ctx_.ubSize, ctx_.kDtype, ctx_.betaDtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(ctx_.nodeName, "SetInterleavedVecRow Failed."), return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    static int64_t CeilDiv(int64_t a, int64_t b)
    {
        if (unlikely(b == 0)) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    ge::graphStatus FixLenTiling()
    {
        tiling_.chunkNum = tiling_.B * CeilDiv(tiling_.T, tiling_.chunkSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        if (ctx_.cuSeqlensShape == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(ctx_.nodeName, RECOMPUTE_W_U_FWD_INPUT_SEQLENS_NAME);
            return ge::GRAPH_FAILED;
        }
        if (ctx_.chunkIndicesShape == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(ctx_.nodeName, RECOMPUTE_W_U_FWD_INPUT_CHUNK_INDICES_NAME);
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.chunkIndicesShape, 1, RECOMPUTE_W_U_FWD_INPUT_CHUNK_INDICES_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, 1, RECOMPUTE_W_U_FWD_INPUT_SEQLENS_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);

        const gert::Shape seqlensStorageShape = ctx_.cuSeqlensShape->GetStorageShape();
        int64_t seqlensDim0 = seqlensStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0);
        if (seqlensDim0 < 2) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(ctx_.nodeName, "seqlens", "dim0 < 2", "dim0 >= 2");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.cuSeqlensData == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(ctx_.nodeName, RECOMPUTE_W_U_FWD_INPUT_SEQLENS_NAME);
            return ge::GRAPH_FAILED;
        }
        const int64_t *cuSeqlens = ctx_.cuSeqlensData;
        if (cuSeqlens[0] != 0) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "seqlens[0]", std::to_string(cuSeqlens[0]), "0");
            return ge::GRAPH_FAILED;
        }
        std::vector<int64_t> expectChunkIndices;
        for (int64_t i = 1; i < seqlensDim0; i++) {
            int64_t curSeqLen = cuSeqlens[i] - cuSeqlens[i - 1];
            if (curSeqLen <= 0) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "seqlens", "non-increasing",
                    "strictly increasing seqlens");
                return ge::GRAPH_FAILED;
            }
            for (int64_t j = 0; j < curSeqLen; j += chunkSize) {
                expectChunkIndices.push_back(i - 1);
                expectChunkIndices.push_back(j / chunkSize);
            }
        }

        const gert::Shape chunkIndicesStorageShape = ctx_.chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0);
        if (chunkIndicesDim0 != static_cast<int64_t>(expectChunkIndices.size())) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(ctx_.nodeName, "chunk_indices", "length mismatch",
                "len == " + std::to_string(expectChunkIndices.size()));
            return ge::GRAPH_FAILED;
        }
        if (ctx_.chunkIndicesData == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(ctx_.nodeName, RECOMPUTE_W_U_FWD_INPUT_CHUNK_INDICES_NAME);
            return ge::GRAPH_FAILED;
        }
        const int64_t *chunkIndices = ctx_.chunkIndicesData;
        for (size_t i = 0; i < expectChunkIndices.size(); i++) {
            if (expectChunkIndices[i] != chunkIndices[i]) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "chunk_indices", "non-canonical", "canonical (seq,chunk) pairs");
                return ge::GRAPH_FAILED;
            }
        }
        tiling_.chunkNum = chunkIndicesStorageShape.GetDim(RECOMPUTE_W_U_FWD_DIM_0) / 2;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        const uint64_t elementSize = ctx_.kDtype == ge::DataType::DT_FLOAT ?
            RECOMPUTE_W_U_FWD_SIZE_FP32 : RECOMPUTE_W_U_FWD_SIZE_HALF;
        uint64_t userWorkspaceSize = elementSize * static_cast<uint64_t>(tiling_.B) *
            static_cast<uint64_t>(tiling_.Hv) * static_cast<uint64_t>(tiling_.T) *
            static_cast<uint64_t>(tiling_.V);
        if (ctx_.enableA5CompactWorkspace) {
            OP_CHECK_IF(ctx_.aicCoreNum == 0,
                        OP_LOGE(ctx_.nodeName, "AIC core count must be positive for compact recompute workspace."),
                        return ge::GRAPH_FAILED);
            userWorkspaceSize = elementSize * ctx_.aicCoreNum *
                static_cast<uint64_t>(RECOMPUTE_W_U_FWD_A5_GM_RING_DEPTH) *
                static_cast<uint64_t>(tiling_.chunkSize) *
                static_cast<uint64_t>(tiling_.V + tiling_.K);
        }
        workspaceSize_ = ctx_.sysWorkspaceSize + static_cast<size_t>(userWorkspaceSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        if (IsVariableLength()) {
            if (tiling_.B != RECOMPUTE_W_U_FWD_VAR_LEN_B_DIM_1) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(ctx_.nodeName, "B", std::to_string(tiling_.B),
                    "1 (varlen)");
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 1;
        } else {
            OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 0;
        }
        OP_CHECK_IF(WorkspaceTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace
} // namespace optiling

#endif // GDN_MEGA_ARCH35_RECOMPUTE_W_U_FWD_TILING_PROCESSOR_H
