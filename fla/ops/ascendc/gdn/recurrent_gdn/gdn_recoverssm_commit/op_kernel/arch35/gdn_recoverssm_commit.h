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

template <typename stateType, bool ALIGN_MODE>
class GdnRecoverssmCommitVector {
public:
    __aicore__ inline explicit GdnRecoverssmCommitVector(
        const GdnRecoverssmCommitTilingData *tiling)
        : tiling_(tiling)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR state, GM_ADDR replay, GM_ADDR table, GM_ADDR stateIndices,
        GM_ADDR commitLens, GM_ADDR finalStateIndices,
        GM_ADDR boundaryStateIndices, GM_ADDR boundaryRecoveryLens)
    {
        stateAnchorGm_.SetGlobalBuffer((__gm__ stateType *)state);
        replayAnchorGm_.SetGlobalBuffer((__gm__ float *)replay);
        tableGm_.SetGlobalBuffer((__gm__ int64_t *)table);
        stateIndices32Gm_.SetGlobalBuffer((__gm__ int32_t *)stateIndices);
        stateIndices64Gm_.SetGlobalBuffer((__gm__ int64_t *)stateIndices);
        commitLens32Gm_.SetGlobalBuffer((__gm__ int32_t *)commitLens);
        commitLens64Gm_.SetGlobalBuffer((__gm__ int64_t *)commitLens);
        finalIndices32Gm_.SetGlobalBuffer((__gm__ int32_t *)finalStateIndices);
        finalIndices64Gm_.SetGlobalBuffer((__gm__ int64_t *)finalStateIndices);
        boundaryIndices32Gm_.SetGlobalBuffer((__gm__ int32_t *)boundaryStateIndices);
        boundaryIndices64Gm_.SetGlobalBuffer((__gm__ int64_t *)boundaryStateIndices);
        boundaryLens32Gm_.SetGlobalBuffer((__gm__ int32_t *)boundaryRecoveryLens);
        boundaryLens64Gm_.SetGlobalBuffer((__gm__ int64_t *)boundaryRecoveryLens);
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        for (uint32_t task = core; task < tiling_->taskCount; task += coreNum) {
            ProcessTask(task);
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t matrixElements = GRSC_DIM_128 * GRSC_DIM_128;
        constexpr uint32_t vectorElements = GRSC_DIM_128;
        pipe_.InitBuffer(finalCorrectionBuf_, matrixElements * sizeof(float));
        if constexpr (ALIGN_MODE) {
            pipe_.InitBuffer(boundaryCorrectionBuf_, matrixElements * sizeof(float));
            pipe_.InitBuffer(boundaryDecayBuf_, vectorElements * sizeof(float));
            pipe_.InitBuffer(boundaryKeyBuf_, vectorElements * sizeof(float));
            pipe_.InitBuffer(boundaryRowBuf_, vectorElements * sizeof(float));
            pipe_.InitBuffer(boundaryRawBuf_, vectorElements * sizeof(stateType));
        }
        pipe_.InitBuffer(recordBuf_, GRSC_PAYLOAD * sizeof(float));
        pipe_.InitBuffer(finalDecayBuf_, vectorElements * sizeof(float));
        pipe_.InitBuffer(finalKeyBuf_, vectorElements * sizeof(float));
        pipe_.InitBuffer(stateRowBuf_, vectorElements * sizeof(float));
        pipe_.InitBuffer(finalRowBuf_, vectorElements * sizeof(float));
        pipe_.InitBuffer(stateRawBuf_, vectorElements * sizeof(stateType));
        finalCorrection_ = finalCorrectionBuf_.Get<float>();
        if constexpr (ALIGN_MODE) {
            boundaryCorrection_ = boundaryCorrectionBuf_.Get<float>();
            boundaryDecay_ = boundaryDecayBuf_.Get<float>();
            boundaryKey_ = boundaryKeyBuf_.Get<float>();
            boundaryRow_ = boundaryRowBuf_.Get<float>();
            boundaryRaw_ = boundaryRawBuf_.Get<stateType>();
        }
        record_ = recordBuf_.Get<float>();
        finalDecay_ = finalDecayBuf_.Get<float>();
        finalKey_ = finalKeyBuf_.Get<float>();
        stateRow_ = stateRowBuf_.Get<float>();
        finalRow_ = finalRowBuf_.Get<float>();
        stateRaw_ = stateRawBuf_.Get<stateType>();
    }

    __aicore__ inline int64_t LoadStateIndex(uint64_t offset) const
    {
        return tiling_->stateIndicesDtype == 32U ?
                   static_cast<int64_t>(stateIndices32Gm_.GetValue(offset)) :
                   stateIndices64Gm_.GetValue(offset);
    }

    __aicore__ inline int64_t LoadPlan(
        const GlobalTensor<int32_t> &plan32,
        const GlobalTensor<int64_t> &plan64, uint64_t offset) const
    {
        return tiling_->planDtype == 32U ?
                   static_cast<int64_t>(plan32.GetValue(offset)) :
                   plan64.GetValue(offset);
    }

    __aicore__ inline void ProcessTask(uint32_t task)
    {
        // The record buffer is reused across tasks; finish the previous
        // Vector consumers before the next MTE2 overwrites it.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const uint32_t heads = tiling_->valueHeads;
        const uint32_t layers = tiling_->layerCount;
        const uint32_t request = task / (layers * heads);
        uint32_t taskRemainder = task - request * (layers * heads);
        const uint32_t layer = taskRemainder / heads;
        const uint32_t head = taskRemainder - layer * heads;

        const int64_t commitLen =
            LoadPlan(commitLens32Gm_, commitLens64Gm_,
                     static_cast<uint64_t>(request) *
                         tiling_->commitLensStride);
        if (commitLen <= 0) {
            return;
        }
        const int64_t windowLen = static_cast<int64_t>(tiling_->specQueryLen);
        const uint32_t effectiveLen =
            static_cast<uint32_t>(commitLen < windowLen ? commitLen : windowLen);
        if (effectiveLen == 0U) {
            return;
        }

        const int64_t sourcePage =
            LoadStateIndex(static_cast<uint64_t>(request) *
                           tiling_->stateIndicesStride);
        const int64_t finalPage =
            LoadPlan(finalIndices32Gm_, finalIndices64Gm_,
                     static_cast<uint64_t>(request) *
                         tiling_->finalStateIndicesStride);
        if (sourcePage <= 0 || finalPage <= 0) {
            return;
        }

        int64_t boundaryPage = 0;
        uint32_t recoveryLen = 0U;
        bool needBoundary = false;
        if constexpr (ALIGN_MODE) {
            boundaryPage =
                LoadPlan(boundaryIndices32Gm_, boundaryIndices64Gm_,
                         static_cast<uint64_t>(request) *
                             tiling_->boundaryStateIndicesStride);
            const int64_t rawRecovery =
                LoadPlan(boundaryLens32Gm_, boundaryLens64Gm_,
                         static_cast<uint64_t>(request) *
                             tiling_->boundaryRecoveryLensStride);
            int64_t boundedRecovery = rawRecovery > 0 ? rawRecovery : 0;
            if (boundedRecovery > static_cast<int64_t>(effectiveLen)) {
                boundedRecovery = static_cast<int64_t>(effectiveLen);
            }
            recoveryLen = static_cast<uint32_t>(boundedRecovery);
            needBoundary =
                boundaryPage > 0 &&
                !(boundaryPage == finalPage && recoveryLen == effectiveLen);
        }

        const uint32_t tableLayerStride = layers;
        const int64_t stateBase =
            tableGm_.GetValue(static_cast<uint64_t>(layer));
        const int64_t stateBlockStride =
            tableGm_.GetValue(tableLayerStride + layer);
        const int64_t replayBase =
            tableGm_.GetValue(2U * tableLayerStride + layer);
        const int64_t replayBlockStride =
            tableGm_.GetValue(3U * tableLayerStride + layer);
        if (stateBase <= 0 || stateBlockStride <= 0 || replayBase <= 0 ||
            replayBlockStride <= 0) {
            return;
        }

        stateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ stateType *>(static_cast<uint64_t>(stateBase)));
        replayGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(static_cast<uint64_t>(replayBase)));

        Duplicate(finalCorrection_, 0.0f, GRSC_DIM_128 * GRSC_DIM_128);
        Duplicate(finalDecay_, 1.0f, GRSC_DIM_128);
        if constexpr (ALIGN_MODE) {
            Duplicate(boundaryCorrection_, 0.0f, GRSC_DIM_128 * GRSC_DIM_128);
            Duplicate(boundaryDecay_, 1.0f, GRSC_DIM_128);
        }

        const uint64_t replayHeadOffset =
            static_cast<uint64_t>(head) * tiling_->replayHeadStride;
        for (uint32_t tokenOffset = 0; tokenOffset < effectiveLen; ++tokenOffset) {
            if (tokenOffset != 0U) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            const uint32_t position = effectiveLen - tokenOffset - 1U;
            const uint64_t recordOffset =
                sourcePage * replayBlockStride + replayHeadOffset +
                static_cast<uint64_t>(position) * tiling_->replayPosStride;
            const DataCopyExtParams recordParams{
                1U, static_cast<uint32_t>(GRSC_PAYLOAD * sizeof(float)), 0U,
                0U, 0U};
            const DataCopyPadExtParams<float> recordPad{false, 0U, 0U, 0.0f};
            DataCopyPad(record_, replayGm_[recordOffset], recordParams, recordPad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Mul(finalKey_, record_[GRSC_DIM_128], finalDecay_, GRSC_DIM_128);
            for (uint32_t row = 0U; row < GRSC_DIM_128; ++row) {
                Axpy(finalCorrection_[row * GRSC_DIM_128], finalKey_,
                     record_.GetValue(row), GRSC_DIM_128);
            }
            Muls(finalDecay_, finalDecay_, record_.GetValue(2U * GRSC_DIM_128),
                 GRSC_DIM_128);

            if constexpr (ALIGN_MODE) {
                if (needBoundary && position < recoveryLen) {
                    Mul(boundaryKey_, record_[GRSC_DIM_128], boundaryDecay_,
                        GRSC_DIM_128);
                    for (uint32_t row = 0U; row < GRSC_DIM_128; ++row) {
                        Axpy(boundaryCorrection_[row * GRSC_DIM_128],
                             boundaryKey_, record_.GetValue(row), GRSC_DIM_128);
                    }
                    Muls(boundaryDecay_, boundaryDecay_,
                         record_.GetValue(2U * GRSC_DIM_128), GRSC_DIM_128);
                }
            }
        }

        const uint64_t stateHeadOffset =
            static_cast<uint64_t>(head) * tiling_->stateHeadStride;
        const DataCopyPadExtParams<stateType> statePad{false, 0U, 0U,
                                                       static_cast<stateType>(0)};
        const DataCopyExtParams stateParams{
            1U, static_cast<uint32_t>(GRSC_DIM_128 * sizeof(stateType)), 0U,
            0U, 0U};
        for (uint32_t row = 0U; row < GRSC_DIM_128; ++row) {
            if (row != 0U) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            const uint64_t sourceOffset =
                sourcePage * stateBlockStride + stateHeadOffset +
                static_cast<uint64_t>(row) * tiling_->stateValueStride;
            if constexpr (std::is_same<stateType, float>::value) {
                DataCopyPad(stateRow_, stateGm_[sourceOffset], stateParams,
                            statePad);
            } else {
                DataCopyPad(stateRaw_, stateGm_[sourceOffset], stateParams,
                            statePad);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (!std::is_same<stateType, float>::value) {
                Cast(stateRow_, stateRaw_, RoundMode::CAST_NONE, GRSC_DIM_128);
            }

            Mul(finalRow_, stateRow_, finalDecay_, GRSC_DIM_128);
            Add(finalRow_, finalRow_,
                finalCorrection_[row * GRSC_DIM_128], GRSC_DIM_128);
            if constexpr (ALIGN_MODE) {
                if (needBoundary) {
                    Mul(boundaryRow_, stateRow_, boundaryDecay_, GRSC_DIM_128);
                    Add(boundaryRow_, boundaryRow_,
                        boundaryCorrection_[row * GRSC_DIM_128], GRSC_DIM_128);
                }
            }

            if constexpr (!std::is_same<stateType, float>::value) {
                Cast(stateRaw_, finalRow_, RoundMode::CAST_RINT, GRSC_DIM_128);
                if constexpr (ALIGN_MODE) {
                    if (needBoundary) {
                        Cast(boundaryRaw_, boundaryRow_, RoundMode::CAST_RINT,
                             GRSC_DIM_128);
                    }
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr (ALIGN_MODE) {
                if (needBoundary) {
                    const uint64_t boundaryOffset =
                        boundaryPage * stateBlockStride + stateHeadOffset +
                        static_cast<uint64_t>(row) * tiling_->stateValueStride;
                    if constexpr (std::is_same<stateType, float>::value) {
                        DataCopyPad(stateGm_[boundaryOffset], boundaryRow_,
                                    stateParams);
                    } else {
                        DataCopyPad(stateGm_[boundaryOffset], boundaryRaw_,
                                    stateParams);
                    }
                }
            }
            const uint64_t finalOffset =
                finalPage * stateBlockStride + stateHeadOffset +
                static_cast<uint64_t>(row) * tiling_->stateValueStride;
            if constexpr (std::is_same<stateType, float>::value) {
                DataCopyPad(stateGm_[finalOffset], finalRow_, stateParams);
            } else {
                DataCopyPad(stateGm_[finalOffset], stateRaw_, stateParams);
            }
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    const GdnRecoverssmCommitTilingData *tiling_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> finalCorrectionBuf_;
    TBuf<TPosition::VECCALC> boundaryCorrectionBuf_;
    TBuf<TPosition::VECCALC> recordBuf_;
    TBuf<TPosition::VECCALC> finalDecayBuf_;
    TBuf<TPosition::VECCALC> boundaryDecayBuf_;
    TBuf<TPosition::VECCALC> finalKeyBuf_;
    TBuf<TPosition::VECCALC> boundaryKeyBuf_;
    TBuf<TPosition::VECCALC> stateRowBuf_;
    TBuf<TPosition::VECCALC> finalRowBuf_;
    TBuf<TPosition::VECCALC> boundaryRowBuf_;
    TBuf<TPosition::VECCALC> stateRawBuf_;
    TBuf<TPosition::VECCALC> boundaryRawBuf_;

    GlobalTensor<stateType> stateAnchorGm_;
    GlobalTensor<stateType> stateGm_;
    GlobalTensor<float> replayAnchorGm_;
    GlobalTensor<float> replayGm_;
    GlobalTensor<int64_t> tableGm_;
    GlobalTensor<int32_t> stateIndices32Gm_;
    GlobalTensor<int64_t> stateIndices64Gm_;
    GlobalTensor<int32_t> commitLens32Gm_;
    GlobalTensor<int64_t> commitLens64Gm_;
    GlobalTensor<int32_t> finalIndices32Gm_;
    GlobalTensor<int64_t> finalIndices64Gm_;
    GlobalTensor<int32_t> boundaryIndices32Gm_;
    GlobalTensor<int64_t> boundaryIndices64Gm_;
    GlobalTensor<int32_t> boundaryLens32Gm_;
    GlobalTensor<int64_t> boundaryLens64Gm_;

    LocalTensor<float> finalCorrection_;
    LocalTensor<float> boundaryCorrection_;
    LocalTensor<float> record_;
    LocalTensor<float> finalDecay_;
    LocalTensor<float> boundaryDecay_;
    LocalTensor<float> finalKey_;
    LocalTensor<float> boundaryKey_;
    LocalTensor<float> stateRow_;
    LocalTensor<float> finalRow_;
    LocalTensor<float> boundaryRow_;
    LocalTensor<stateType> stateRaw_;
    LocalTensor<stateType> boundaryRaw_;
};

} // namespace GDN

#endif // GDN_RECOVERSSM_COMMIT_ARCH35_H
