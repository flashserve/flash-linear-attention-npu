/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_COMMIT_ARCH35_H
#define GDN_RECOVERSSM_COMMIT_ARCH35_H

#include <type_traits>

#include "kernel_operator.h"
#include "../gdn_recoverssm_commit_struct.h"

namespace GDN {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t GRSC_BUFFERS_2 = 2U;
constexpr uint8_t GRSC_RECORD_MUTEX[GRSC_BUFFERS_2] = {3, 4};
constexpr uint8_t GRSC_STATE_MUTEX[GRSC_BUFFERS_2] = {5, 6};
constexpr uint8_t GRSC_BOUNDARY_MUTEX[GRSC_BUFFERS_2] = {7, 8};
constexpr uint32_t GRSC_FP32_PER_REG_64 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t GRSC_REQUIRED_DIM_128 = 2U * GRSC_FP32_PER_REG_64;
constexpr uint32_t GRSC_MAX_SPEC_QUERY_LEN_32 = 32U;
constexpr uint32_t GRSC_TILE_ELEMENTS_8192 = GRSC_VALUE_TILE_64 * GRSC_REQUIRED_DIM_128;

constexpr CastTrait GRSC_FP32_TO_B16_ZERO = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING,
                                             RoundMode::CAST_RINT};
constexpr CastTrait GRSC_FP32_TO_B16_ONE = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                            RoundMode::CAST_RINT};
constexpr CastTrait GRSC_B16_TO_FP32_ZERO = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                             RoundMode::CAST_NONE};
constexpr CastTrait GRSC_B16_TO_FP32_ONE = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

struct GrscTaskMeta {
    uint32_t head;
    uint32_t layer;
    uint32_t valueTile;
    uint32_t effectiveLen;
    uint32_t recoveryLen;
    int64_t sourcePage;
    int64_t finalPage;
    int64_t boundaryPage;
    int64_t stateBase;
    int64_t stateBlockStride;
    int64_t replayBase;
    int64_t replayBlockStride;
    uint64_t replayRecordOffset;
    bool needBoundary;
};

template <typename stateType, bool ALIGN_MODE, bool NEED_BOUNDARY>
__simd_vf__ inline void ProcessCommitValueTileVf(__ubuf__ float *records, uint32_t valueRowBase, uint32_t recordCount,
                                                 __ubuf__ stateType *sourceState, __ubuf__ stateType *finalState,
                                                 __ubuf__ stateType *boundaryState, uint32_t boundaryLen)
{
    MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    RegTensor<float> sourceA0;
    RegTensor<float> sourceA1;
    RegTensor<float> sourceB0;
    RegTensor<float> sourceB1;
    RegTensor<float> finalCorrectionA0;
    RegTensor<float> finalCorrectionA1;
    RegTensor<float> finalCorrectionB0;
    RegTensor<float> finalCorrectionB1;
    RegTensor<float> finalDecayA0;
    RegTensor<float> finalDecayA1;
    RegTensor<float> finalDecayB0;
    RegTensor<float> finalDecayB1;
    RegTensor<float> boundaryCorrectionA0;
    RegTensor<float> boundaryCorrectionA1;
    RegTensor<float> boundaryCorrectionB0;
    RegTensor<float> boundaryCorrectionB1;
    RegTensor<float> boundaryDecayA0;
    RegTensor<float> boundaryDecayA1;
    RegTensor<float> boundaryDecayB0;
    RegTensor<float> boundaryDecayB1;
    RegTensor<float> key0;
    RegTensor<float> key1;
    RegTensor<float> weightA0;
    RegTensor<float> weightA1;
    RegTensor<float> weightB0;
    RegTensor<float> weightB1;
    RegTensor<float> scalarA;
    RegTensor<float> scalarB;

    for (uint32_t row = 0U; row < GRSC_VALUE_TILE_64; row += 2U) {
        const uint32_t offsetA = row * GRSC_REQUIRED_DIM_128;
        const uint32_t offsetB = offsetA + GRSC_REQUIRED_DIM_128;
        if constexpr (std::is_same<stateType, float>::value) {
            LoadAlign<float, LoadDist::DIST_NORM>(sourceA0, sourceState + offsetA);
            LoadAlign<float, LoadDist::DIST_NORM>(sourceA1, sourceState + offsetA + GRSC_FP32_PER_REG_64);
            LoadAlign<float, LoadDist::DIST_NORM>(sourceB0, sourceState + offsetB);
            LoadAlign<float, LoadDist::DIST_NORM>(sourceB1, sourceState + offsetB + GRSC_FP32_PER_REG_64);
        } else {
            RegTensor<stateType> statePackedA;
            RegTensor<stateType> statePackedB;
            LoadAlign<stateType, LoadDist::DIST_NORM>(statePackedA, sourceState + offsetA);
            LoadAlign<stateType, LoadDist::DIST_NORM>(statePackedB, sourceState + offsetB);
            Cast<float, stateType, GRSC_B16_TO_FP32_ZERO>(sourceA0, statePackedA, maskB16);
            Cast<float, stateType, GRSC_B16_TO_FP32_ONE>(sourceA1, statePackedA, maskB16);
            Cast<float, stateType, GRSC_B16_TO_FP32_ZERO>(sourceB0, statePackedB, maskB16);
            Cast<float, stateType, GRSC_B16_TO_FP32_ONE>(sourceB1, statePackedB, maskB16);
        }

        Duplicate(finalCorrectionA0, 0.0f, maskFp32);
        Duplicate(finalCorrectionA1, 0.0f, maskFp32);
        Duplicate(finalCorrectionB0, 0.0f, maskFp32);
        Duplicate(finalCorrectionB1, 0.0f, maskFp32);
        Duplicate(finalDecayA0, 1.0f, maskFp32);
        Duplicate(finalDecayA1, 1.0f, maskFp32);
        Duplicate(finalDecayB0, 1.0f, maskFp32);
        Duplicate(finalDecayB1, 1.0f, maskFp32);
        if constexpr (ALIGN_MODE && NEED_BOUNDARY) {
            Duplicate(boundaryCorrectionA0, 0.0f, maskFp32);
            Duplicate(boundaryCorrectionA1, 0.0f, maskFp32);
            Duplicate(boundaryCorrectionB0, 0.0f, maskFp32);
            Duplicate(boundaryCorrectionB1, 0.0f, maskFp32);
            Duplicate(boundaryDecayA0, 1.0f, maskFp32);
            Duplicate(boundaryDecayA1, 1.0f, maskFp32);
            Duplicate(boundaryDecayB0, 1.0f, maskFp32);
            Duplicate(boundaryDecayB1, 1.0f, maskFp32);
        }

        for (uint32_t position = recordCount; position > boundaryLen; --position) {
            __ubuf__ float *record = records + static_cast<uint32_t>(position - 1U) * GRSC_RECORD_STRIDE_264;
            if constexpr (std::is_same<stateType, float>::value) {
                LoadAlign<float, LoadDist::DIST_NORM>(key0, record + GRSC_REQUIRED_DIM_128);
                LoadAlign<float, LoadDist::DIST_NORM>(key1, record + GRSC_REQUIRED_DIM_128 + GRSC_FP32_PER_REG_64);
            } else {
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(key0, key1, record + GRSC_REQUIRED_DIM_128);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarA, record + valueRowBase + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarB, record + valueRowBase + row + 1U);
            Mul(weightA0, key0, finalDecayA0, maskFp32);
            Mul(weightB0, key0, finalDecayB0, maskFp32);
            Mul(weightA1, key1, finalDecayA1, maskFp32);
            Mul(weightB1, key1, finalDecayB1, maskFp32);
            MulAddDst(finalCorrectionA0, scalarA, weightA0, maskFp32);
            MulAddDst(finalCorrectionB0, scalarB, weightB0, maskFp32);
            MulAddDst(finalCorrectionA1, scalarA, weightA1, maskFp32);
            MulAddDst(finalCorrectionB1, scalarB, weightB1, maskFp32);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarA, record + 2U * GRSC_REQUIRED_DIM_128);
            Mul(finalDecayA0, finalDecayA0, scalarA, maskFp32);
            Mul(finalDecayB0, finalDecayB0, scalarA, maskFp32);
            Mul(finalDecayA1, finalDecayA1, scalarA, maskFp32);
            Mul(finalDecayB1, finalDecayB1, scalarA, maskFp32);
        }

        for (uint32_t position = boundaryLen; position > 0U; --position) {
            __ubuf__ float *record = records + static_cast<uint32_t>(position - 1U) * GRSC_RECORD_STRIDE_264;
            if constexpr (std::is_same<stateType, float>::value) {
                LoadAlign<float, LoadDist::DIST_NORM>(key0, record + GRSC_REQUIRED_DIM_128);
                LoadAlign<float, LoadDist::DIST_NORM>(key1, record + GRSC_REQUIRED_DIM_128 + GRSC_FP32_PER_REG_64);
            } else {
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(key0, key1, record + GRSC_REQUIRED_DIM_128);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarA, record + valueRowBase + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarB, record + valueRowBase + row + 1U);
            Mul(weightA0, key0, finalDecayA0, maskFp32);
            Mul(weightB0, key0, finalDecayB0, maskFp32);
            Mul(weightA1, key1, finalDecayA1, maskFp32);
            Mul(weightB1, key1, finalDecayB1, maskFp32);
            MulAddDst(finalCorrectionA0, scalarA, weightA0, maskFp32);
            MulAddDst(finalCorrectionB0, scalarB, weightB0, maskFp32);
            MulAddDst(finalCorrectionA1, scalarA, weightA1, maskFp32);
            MulAddDst(finalCorrectionB1, scalarB, weightB1, maskFp32);
            if constexpr (ALIGN_MODE && NEED_BOUNDARY) {
                Mul(weightA0, key0, boundaryDecayA0, maskFp32);
                Mul(weightB0, key0, boundaryDecayB0, maskFp32);
                Mul(weightA1, key1, boundaryDecayA1, maskFp32);
                Mul(weightB1, key1, boundaryDecayB1, maskFp32);
                MulAddDst(boundaryCorrectionA0, scalarA, weightA0, maskFp32);
                MulAddDst(boundaryCorrectionB0, scalarB, weightB0, maskFp32);
                MulAddDst(boundaryCorrectionA1, scalarA, weightA1, maskFp32);
                MulAddDst(boundaryCorrectionB1, scalarB, weightB1, maskFp32);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scalarA, record + 2U * GRSC_REQUIRED_DIM_128);
            Mul(finalDecayA0, finalDecayA0, scalarA, maskFp32);
            Mul(finalDecayB0, finalDecayB0, scalarA, maskFp32);
            Mul(finalDecayA1, finalDecayA1, scalarA, maskFp32);
            Mul(finalDecayB1, finalDecayB1, scalarA, maskFp32);
            if constexpr (ALIGN_MODE && NEED_BOUNDARY) {
                Mul(boundaryDecayA0, boundaryDecayA0, scalarA, maskFp32);
                Mul(boundaryDecayB0, boundaryDecayB0, scalarA, maskFp32);
                Mul(boundaryDecayA1, boundaryDecayA1, scalarA, maskFp32);
                Mul(boundaryDecayB1, boundaryDecayB1, scalarA, maskFp32);
            }
        }

        if constexpr (ALIGN_MODE && NEED_BOUNDARY) {
            Mul(weightA0, sourceA0, boundaryDecayA0, maskFp32);
            Mul(weightB0, sourceB0, boundaryDecayB0, maskFp32);
            Mul(weightA1, sourceA1, boundaryDecayA1, maskFp32);
            Mul(weightB1, sourceB1, boundaryDecayB1, maskFp32);
            Add(weightA0, weightA0, boundaryCorrectionA0, maskFp32);
            Add(weightB0, weightB0, boundaryCorrectionB0, maskFp32);
            Add(weightA1, weightA1, boundaryCorrectionA1, maskFp32);
            Add(weightB1, weightB1, boundaryCorrectionB1, maskFp32);
            if constexpr (std::is_same<stateType, float>::value) {
                StoreAlign<float, StoreDist::DIST_NORM>(boundaryState + offsetA, weightA0, maskFp32);
                StoreAlign<float, StoreDist::DIST_NORM>(boundaryState + offsetA + GRSC_FP32_PER_REG_64, weightA1,
                                                        maskFp32);
                StoreAlign<float, StoreDist::DIST_NORM>(boundaryState + offsetB, weightB0, maskFp32);
                StoreAlign<float, StoreDist::DIST_NORM>(boundaryState + offsetB + GRSC_FP32_PER_REG_64, weightB1,
                                                        maskFp32);
            } else {
                RegTensor<stateType> packedA;
                RegTensor<stateType> packedB;
                Cast<stateType, float, GRSC_FP32_TO_B16_ONE>(packedA, weightA1, maskFp32);
                Cast<stateType, float, GRSC_FP32_TO_B16_ONE>(packedB, weightB1, maskFp32);
                Cast<stateType, float, GRSC_FP32_TO_B16_ZERO>(packedA, weightA0, maskFp32);
                Cast<stateType, float, GRSC_FP32_TO_B16_ZERO>(packedB, weightB0, maskFp32);
                StoreAlign(boundaryState + offsetA, packedA, maskB16);
                StoreAlign(boundaryState + offsetB, packedB, maskB16);
            }
        }

        Mul(sourceA0, sourceA0, finalDecayA0, maskFp32);
        Mul(sourceB0, sourceB0, finalDecayB0, maskFp32);
        Mul(sourceA1, sourceA1, finalDecayA1, maskFp32);
        Mul(sourceB1, sourceB1, finalDecayB1, maskFp32);
        Add(sourceA0, sourceA0, finalCorrectionA0, maskFp32);
        Add(sourceB0, sourceB0, finalCorrectionB0, maskFp32);
        Add(sourceA1, sourceA1, finalCorrectionA1, maskFp32);
        Add(sourceB1, sourceB1, finalCorrectionB1, maskFp32);
        if constexpr (std::is_same<stateType, float>::value) {
            StoreAlign<float, StoreDist::DIST_NORM>(finalState + offsetA, sourceA0, maskFp32);
            StoreAlign<float, StoreDist::DIST_NORM>(finalState + offsetA + GRSC_FP32_PER_REG_64, sourceA1, maskFp32);
            StoreAlign<float, StoreDist::DIST_NORM>(finalState + offsetB, sourceB0, maskFp32);
            StoreAlign<float, StoreDist::DIST_NORM>(finalState + offsetB + GRSC_FP32_PER_REG_64, sourceB1, maskFp32);
        } else {
            RegTensor<stateType> packedA;
            RegTensor<stateType> packedB;
            Cast<stateType, float, GRSC_FP32_TO_B16_ONE>(packedA, sourceA1, maskFp32);
            Cast<stateType, float, GRSC_FP32_TO_B16_ONE>(packedB, sourceB1, maskFp32);
            Cast<stateType, float, GRSC_FP32_TO_B16_ZERO>(packedA, sourceA0, maskFp32);
            Cast<stateType, float, GRSC_FP32_TO_B16_ZERO>(packedB, sourceB0, maskFp32);
            StoreAlign(finalState + offsetA, packedA, maskB16);
            StoreAlign(finalState + offsetB, packedB, maskB16);
        }
    }
}

template <typename stateType, typename planType, typename stateIndexType, bool ALIGN_MODE>
class GdnRecoverssmCommitVector {
public:
    __aicore__ inline void Init(const GdnRecoverssmCommitTilingData *tiling, GM_ADDR state, GM_ADDR replay,
                                GM_ADDR table, GM_ADDR stateIndices, GM_ADDR commitLens, GM_ADDR finalStateIndices,
                                GM_ADDR boundaryStateIndices, GM_ADDR boundaryRecoveryLens)
    {
        tiling_ = tiling;
        stateAnchorGm_.SetGlobalBuffer((__gm__ stateType *)state);
        stateOutGm_.SetGlobalBuffer((__gm__ stateType *)state);
        replayAnchorGm_.SetGlobalBuffer((__gm__ float *)replay);
        tableGm_.SetGlobalBuffer((__gm__ int64_t *)table);
        stateIndicesGm_.SetGlobalBuffer((__gm__ stateIndexType *)stateIndices);
        commitLensGm_.SetGlobalBuffer((__gm__ planType *)commitLens);
        finalIndicesGm_.SetGlobalBuffer((__gm__ planType *)finalStateIndices);
        boundaryIndicesGm_.SetGlobalBuffer((__gm__ planType *)boundaryStateIndices);
        boundaryLensGm_.SetGlobalBuffer((__gm__ planType *)boundaryRecoveryLens);
        pipe_.InitBuffer(recordBuf_,
                         GRSC_BUFFERS_2 * GRSC_MAX_SPEC_QUERY_LEN_32 * GRSC_RECORD_STRIDE_264 * sizeof(float));
        pipe_.InitBuffer(stateRawBuf_, GRSC_BUFFERS_2 * GRSC_TILE_ELEMENTS_8192 * sizeof(stateType));
        if constexpr (ALIGN_MODE) {
            pipe_.InitBuffer(boundaryRawBuf_, GRSC_BUFFERS_2 * GRSC_TILE_ELEMENTS_8192 * sizeof(stateType));
        }
        LocalTensor<float> recordBase = recordBuf_.Get<float>();
        record_[0] = recordBase;
        record_[1] = recordBase[GRSC_MAX_SPEC_QUERY_LEN_32 * GRSC_RECORD_STRIDE_264];
        LocalTensor<stateType> stateBase = stateRawBuf_.Get<stateType>();
        state_[0] = stateBase;
        state_[1] = stateBase[GRSC_TILE_ELEMENTS_8192];
        if constexpr (ALIGN_MODE) {
            LocalTensor<stateType> boundaryBase = boundaryRawBuf_.Get<stateType>();
            boundary_[0] = boundaryBase;
            boundary_[1] = boundaryBase[GRSC_TILE_ELEMENTS_8192];
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        uint32_t task = core;
        uint32_t buffer = 0U;
        GrscTaskMeta taskMeta{};

        for (; task < tiling_->taskCount; task += coreNum) {
            PrepareTaskMeta(task, taskMeta);
            if (taskMeta.effectiveLen == 0U || taskMeta.sourcePage <= 0 || taskMeta.finalPage <= 0) {
                continue;
            }
            CopyIn(buffer, taskMeta);
            Compute(buffer, taskMeta);
            CopyOut(buffer, taskMeta);
            buffer ^= 1U;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t buffer, const GrscTaskMeta &meta)
    {
        replayGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(static_cast<uint64_t>(meta.replayBase)));
        DataCopyExtParams recordParams{
            static_cast<uint16_t>(meta.effectiveLen), static_cast<uint32_t>(GRSC_PAYLOAD_257 * sizeof(float)),
            static_cast<int64_t>((tiling_->replayPosStride - GRSC_PAYLOAD_257) * sizeof(float)), 0U, 0U};
        const DataCopyPadExtParams<float> recordPad{false, 0U, 0U, 0.0f};
        Mutex::Lock<PIPE_MTE2>(GRSC_RECORD_MUTEX[buffer]);
        DataCopyPad(record_[buffer], replayGm_[meta.replayRecordOffset], recordParams, recordPad);
        Mutex::Unlock<PIPE_MTE2>(GRSC_RECORD_MUTEX[buffer]);

        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(static_cast<uint64_t>(meta.stateBase)));
        uint64_t sourceOffset = meta.sourcePage * meta.stateBlockStride +
                                static_cast<uint64_t>(meta.head) * tiling_->stateHeadStride +
                                static_cast<uint64_t>(meta.valueTile) * GRSC_VALUE_TILE_64 * tiling_->stateValueStride;
        DataCopyExtParams stateInParams{
            static_cast<uint16_t>(GRSC_VALUE_TILE_64), static_cast<uint32_t>(GRSC_REQUIRED_DIM_128 * sizeof(stateType)),
            static_cast<int64_t>((tiling_->stateValueStride - GRSC_REQUIRED_DIM_128) * sizeof(stateType)), 0U, 0U};
        const DataCopyPadExtParams<stateType> statePad{false, 0U, 0U, static_cast<stateType>(0)};
        Mutex::Lock<PIPE_MTE2>(GRSC_STATE_MUTEX[buffer]);
        DataCopyPad(state_[buffer], stateGm_[sourceOffset], stateInParams, statePad);
        Mutex::Unlock<PIPE_MTE2>(GRSC_STATE_MUTEX[buffer]);
    }

    __aicore__ inline void Compute(uint32_t buffer, const GrscTaskMeta &meta)
    {
        __ubuf__ stateType *boundaryState = nullptr;
        Mutex::Lock<PIPE_V>(GRSC_RECORD_MUTEX[buffer]);
        Mutex::Lock<PIPE_V>(GRSC_STATE_MUTEX[buffer]);
        if constexpr (ALIGN_MODE) {
            Mutex::Lock<PIPE_V>(GRSC_BOUNDARY_MUTEX[buffer]);
            boundaryState = (__ubuf__ stateType *)boundary_[buffer].GetPhyAddr();
        }
        if constexpr (ALIGN_MODE) {
            if (meta.needBoundary) {
                ProcessCommitValueTileVf<stateType, ALIGN_MODE, true>(
                    (__ubuf__ float *)record_[buffer].GetPhyAddr(), meta.valueTile * GRSC_VALUE_TILE_64,
                    meta.effectiveLen, (__ubuf__ stateType *)state_[buffer].GetPhyAddr(),
                    (__ubuf__ stateType *)state_[buffer].GetPhyAddr(), boundaryState, meta.recoveryLen);
            } else {
                ProcessCommitValueTileVf<stateType, ALIGN_MODE, false>(
                    (__ubuf__ float *)record_[buffer].GetPhyAddr(), meta.valueTile * GRSC_VALUE_TILE_64,
                    meta.effectiveLen, (__ubuf__ stateType *)state_[buffer].GetPhyAddr(),
                    (__ubuf__ stateType *)state_[buffer].GetPhyAddr(), boundaryState, 0U);
            }
        } else {
            ProcessCommitValueTileVf<stateType, ALIGN_MODE, false>(
                (__ubuf__ float *)record_[buffer].GetPhyAddr(), meta.valueTile * GRSC_VALUE_TILE_64, meta.effectiveLen,
                (__ubuf__ stateType *)state_[buffer].GetPhyAddr(), (__ubuf__ stateType *)state_[buffer].GetPhyAddr(),
                boundaryState, 0U);
        }
        Mutex::Unlock<PIPE_V>(GRSC_STATE_MUTEX[buffer]);
        Mutex::Unlock<PIPE_V>(GRSC_RECORD_MUTEX[buffer]);
        if constexpr (ALIGN_MODE) {
            Mutex::Unlock<PIPE_V>(GRSC_BOUNDARY_MUTEX[buffer]);
        }
    }

    __aicore__ inline void CopyOut(uint32_t buffer, const GrscTaskMeta &meta)
    {
        stateOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(static_cast<uint64_t>(meta.stateBase)));
        const DataCopyExtParams stateOutParams{
            static_cast<uint16_t>(GRSC_VALUE_TILE_64), static_cast<uint32_t>(GRSC_REQUIRED_DIM_128 * sizeof(stateType)),
            0U, static_cast<uint32_t>((tiling_->stateValueStride - GRSC_REQUIRED_DIM_128) * sizeof(stateType)), 0U};
        uint64_t outputOffset = 0U;
        if constexpr (ALIGN_MODE) {
            if (meta.needBoundary) {
                outputOffset = meta.boundaryPage * meta.stateBlockStride +
                               static_cast<uint64_t>(meta.head) * tiling_->stateHeadStride +
                               static_cast<uint64_t>(meta.valueTile) * GRSC_VALUE_TILE_64 * tiling_->stateValueStride;
                Mutex::Lock<PIPE_MTE3>(GRSC_BOUNDARY_MUTEX[buffer]);
                DataCopyPad(stateOutGm_[outputOffset], boundary_[buffer], stateOutParams);
                Mutex::Unlock<PIPE_MTE3>(GRSC_BOUNDARY_MUTEX[buffer]);
            }
        }
        outputOffset = meta.finalPage * meta.stateBlockStride +
                       static_cast<uint64_t>(meta.head) * tiling_->stateHeadStride +
                       static_cast<uint64_t>(meta.valueTile) * GRSC_VALUE_TILE_64 * tiling_->stateValueStride;
        Mutex::Lock<PIPE_MTE3>(GRSC_STATE_MUTEX[buffer]);
        DataCopyPad(stateOutGm_[outputOffset], state_[buffer], stateOutParams);
        Mutex::Unlock<PIPE_MTE3>(GRSC_STATE_MUTEX[buffer]);
    }

    __aicore__ inline void PrepareTaskMeta(uint32_t task, GrscTaskMeta &meta)
    {
        const uint32_t heads = tiling_->valueHeads;
        const uint32_t layers = tiling_->layerCount;
        const uint32_t request = task / (layers * heads * GRSC_VALUE_TILES_2);
        uint32_t taskRemainder = task - request * (layers * heads * GRSC_VALUE_TILES_2);
        meta.valueTile = taskRemainder % GRSC_VALUE_TILES_2;
        taskRemainder /= GRSC_VALUE_TILES_2;
        meta.layer = taskRemainder / heads;
        meta.head = taskRemainder - meta.layer * heads;
        meta.effectiveLen = static_cast<uint32_t>(commitLensGm_.GetValue(request));
        meta.sourcePage = static_cast<int64_t>(stateIndicesGm_.GetValue(request));
        meta.finalPage = static_cast<int64_t>(finalIndicesGm_.GetValue(request));
        meta.boundaryPage = 0;
        meta.recoveryLen = 0U;
        meta.needBoundary = false;
        if constexpr (ALIGN_MODE) {
            meta.boundaryPage = static_cast<int64_t>(boundaryIndicesGm_.GetValue(request));
            meta.recoveryLen = static_cast<uint32_t>(boundaryLensGm_.GetValue(request));
            meta.needBoundary = meta.boundaryPage > 0 &&
                                !(meta.boundaryPage == meta.finalPage && meta.recoveryLen == meta.effectiveLen);
            meta.recoveryLen = meta.needBoundary ? meta.recoveryLen : 0U;
        }
        meta.stateBase = tableGm_.GetValue(static_cast<uint64_t>(meta.layer));
        meta.stateBlockStride = tableGm_.GetValue(layers + meta.layer);
        meta.replayBase = tableGm_.GetValue(2U * layers + meta.layer);
        meta.replayBlockStride = tableGm_.GetValue(3U * layers + meta.layer);
        meta.replayRecordOffset =
            meta.sourcePage * meta.replayBlockStride + static_cast<uint64_t>(meta.head) * tiling_->replayHeadStride;
    }

    const GdnRecoverssmCommitTilingData *tiling_{nullptr};
    TPipe pipe_;
    TBuf<TPosition::VECCALC> recordBuf_;
    TBuf<TPosition::VECCALC> stateRawBuf_;
    TBuf<TPosition::VECCALC> boundaryRawBuf_;

    GlobalTensor<stateType> stateAnchorGm_;
    GlobalTensor<stateType> stateGm_;
    GlobalTensor<stateType> stateOutGm_;
    GlobalTensor<float> replayAnchorGm_;
    GlobalTensor<float> replayGm_;
    GlobalTensor<int64_t> tableGm_;
    GlobalTensor<stateIndexType> stateIndicesGm_;
    GlobalTensor<planType> commitLensGm_;
    GlobalTensor<planType> finalIndicesGm_;
    GlobalTensor<planType> boundaryIndicesGm_;
    GlobalTensor<planType> boundaryLensGm_;

    LocalTensor<float> record_[GRSC_BUFFERS_2];
    LocalTensor<stateType> state_[GRSC_BUFFERS_2];
    LocalTensor<stateType> boundary_[GRSC_BUFFERS_2];
};

} // namespace GDN

#endif // GDN_RECOVERSSM_COMMIT_ARCH35_H
