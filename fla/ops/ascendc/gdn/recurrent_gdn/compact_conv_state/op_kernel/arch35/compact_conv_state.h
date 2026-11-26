/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef COMPACT_CONV_STATE_ARCH35_H
#define COMPACT_CONV_STATE_ARCH35_H

#include "kernel_operator.h"
#include "../compact_conv_state_struct.h"

namespace GDN {

using namespace AscendC;

constexpr uint8_t CCS_MUTEX[CCS_BUFFERS_2] = {3, 4};
constexpr uint32_t CCS_LOCAL_BUFFERS_4 = CCS_BUFFERS_2 * CCS_ROLE_BUFFERS_2;

struct CompactConvStateTaskMeta {
    uint32_t channelOffset;
    uint32_t channelCount;
    uint32_t effectiveLen;
    uint32_t recoveryLen;
    int64_t sourcePage;
    int64_t finalPage;
    int64_t boundaryPage;
    int64_t baseAddr;
    int64_t blockStride;
    int64_t dimStride;
    int64_t tokenStride;
    uint16_t copyBlockCount;
    uint32_t copyBlockLen;
    int64_t copyBlockGap;
    int64_t copyLocalGap;
    bool active;
    bool needBoundary;
};

template <typename stateType, typename planType, typename stateIndexType, bool ALIGN_MODE, bool SHARED_DIM_STRIDE>
class CompactConvStateVector {
public:
    __aicore__ inline void Init(const CompactConvStateTilingData *tiling, GM_ADDR convState, GM_ADDR baseAddrs,
                                GM_ADDR blockStrides, GM_ADDR dimStrides, GM_ADDR tokenStrides, GM_ADDR stateIndices,
                                GM_ADDR commitLens, GM_ADDR finalStateIndices, GM_ADDR boundaryStateIndices,
                                GM_ADDR boundaryRecoveryLens)
    {
        (void)convState;
        tiling_ = tiling;
        baseAddrsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(baseAddrs));
        blockStridesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(blockStrides));
        dimStridesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(dimStrides));
        tokenStridesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tokenStrides));
        stateIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ stateIndexType *>(stateIndices));
        commitLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ planType *>(commitLens));
        finalIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ planType *>(finalStateIndices));
        boundaryIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ planType *>(boundaryStateIndices));
        boundaryLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ planType *>(boundaryRecoveryLens));

        const uint32_t elements = static_cast<uint32_t>(tiling_->bufferElements);
        pipe_.InitBuffer(bufferBuf_, CCS_LOCAL_BUFFERS_4 * elements * sizeof(stateType));
        LocalTensor<stateType> base = bufferBuf_.Get<stateType>();
        for (uint32_t index = 0U; index < CCS_LOCAL_BUFFERS_4; ++index) {
            buffer_[index] = base[index * elements];
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreNum = GetBlockNum();
        uint32_t task = GetBlockIdx();
        uint32_t slot = 0U;
        CompactConvStateTaskMeta meta{};
        for (; task < tiling_->taskCount; task += coreNum) {
            PrepareTaskMeta(task, meta);
            if (meta.active) {
                CopyIn(slot, meta);
                CopyOut(slot, meta);
            }
            slot ^= 1U;
        }
    }

private:
    __aicore__ inline void PrepareTaskMeta(uint32_t task, CompactConvStateTaskMeta &meta)
    {
        const uint32_t layers = tiling_->layerCount;
        const uint32_t tilesPerRowLayer = tiling_->tilesPerRowLayer;
        const uint32_t request = task / (layers * tilesPerRowLayer);
        uint32_t remainder = task - request * layers * tilesPerRowLayer;
        const uint32_t layer = remainder / tilesPerRowLayer;
        const uint32_t tile = remainder - layer * tilesPerRowLayer;
        const uint32_t channelOffset = tile * CCS_CHANNEL_TILE_256;

        const uint32_t remainingChannels = tiling_->channelCount - channelOffset;
        meta.channelOffset = channelOffset;
        meta.channelCount = remainingChannels < CCS_CHANNEL_TILE_256 ? remainingChannels : CCS_CHANNEL_TILE_256;
        meta.effectiveLen = static_cast<uint32_t>(commitLensGm_.GetValue(request));
        meta.sourcePage = static_cast<int64_t>(
            stateIndicesGm_.GetValue(static_cast<uint64_t>(request) * tiling_->stateIndicesStride));
        meta.finalPage = static_cast<int64_t>(finalIndicesGm_.GetValue(request));
        meta.boundaryPage = 0;
        meta.recoveryLen = 0U;
        meta.baseAddr = baseAddrsGm_.GetValue(layer);
        meta.blockStride = blockStridesGm_.GetValue(layer);
        meta.tokenStride = tokenStridesGm_.GetValue(layer);
        if constexpr (SHARED_DIM_STRIDE) {
            meta.dimStride = static_cast<int64_t>(tiling_->sharedDimStride);
        } else {
            meta.dimStride = dimStridesGm_.GetValue(layer);
        }
        if constexpr (ALIGN_MODE) {
            meta.boundaryPage = static_cast<int64_t>(boundaryIndicesGm_.GetValue(request));
            meta.recoveryLen = static_cast<uint32_t>(boundaryLensGm_.GetValue(request));
        }
        meta.active =
            meta.sourcePage > tiling_->nullBlockId && meta.finalPage > tiling_->nullBlockId && meta.effectiveLen > 0U;
        if (meta.dimStride == 1) {
            meta.copyBlockCount = static_cast<uint16_t>(tiling_->historyLen);
            meta.copyBlockLen = static_cast<uint32_t>(meta.channelCount * sizeof(stateType));
            meta.copyBlockGap = static_cast<int64_t>((meta.tokenStride - meta.channelCount) * sizeof(stateType));
        } else {
            meta.copyBlockCount = static_cast<uint16_t>(meta.channelCount);
            meta.copyBlockLen = static_cast<uint32_t>(tiling_->historyLen * sizeof(stateType));
            meta.copyBlockGap = static_cast<int64_t>((meta.dimStride - tiling_->historyLen) * sizeof(stateType));
        }
        meta.copyLocalGap = 0;
        if constexpr (ALIGN_MODE) {
            meta.needBoundary = meta.boundaryPage > tiling_->nullBlockId && meta.recoveryLen > 0U &&
                                !(meta.boundaryPage == meta.finalPage && meta.recoveryLen == meta.effectiveLen);
        } else {
            meta.needBoundary = false;
        }
    }

    __aicore__ inline void CopyIn(uint32_t slot, const CompactConvStateTaskMeta &meta)
    {
        Mutex::Lock<PIPE_MTE2>(CCS_MUTEX[slot]);
        convStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(static_cast<uint64_t>(meta.baseAddr)));
        const DataCopyExtParams params{meta.copyBlockCount, meta.copyBlockLen, meta.copyBlockGap, meta.copyLocalGap,
                                       0U};
        const DataCopyPadExtParams<stateType> pad{false, 0U, 0U, static_cast<stateType>(0)};
        const uint64_t finalSourceOffset = meta.sourcePage * meta.blockStride + meta.channelOffset * meta.dimStride +
                                           static_cast<uint64_t>(meta.effectiveLen - 1U) * meta.tokenStride;
        DataCopyPad(buffer_[slot * 2U], convStateGm_[finalSourceOffset], params, pad);
        if (meta.needBoundary) {
            const uint64_t boundarySourceOffset = meta.sourcePage * meta.blockStride +
                                                  meta.channelOffset * meta.dimStride +
                                                  static_cast<uint64_t>(meta.recoveryLen - 1U) * meta.tokenStride;
            DataCopyPad(buffer_[slot * 2U + 1U], convStateGm_[boundarySourceOffset], params, pad);
        }
        Mutex::Unlock<PIPE_MTE2>(CCS_MUTEX[slot]);
    }

    __aicore__ inline void CopyOut(uint32_t slot, const CompactConvStateTaskMeta &meta)
    {
        Mutex::Lock<PIPE_MTE3>(CCS_MUTEX[slot]);
        convStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(static_cast<uint64_t>(meta.baseAddr)));
        const DataCopyExtParams params{meta.copyBlockCount, meta.copyBlockLen, meta.copyLocalGap, meta.copyBlockGap,
                                       0U};
        if (meta.needBoundary) {
            const uint64_t boundaryDestinationOffset =
                meta.boundaryPage * meta.blockStride + meta.channelOffset * meta.dimStride;
            DataCopyPad(convStateGm_[boundaryDestinationOffset], buffer_[slot * 2U + 1U], params);
        }
        const uint64_t finalDestinationOffset = meta.finalPage * meta.blockStride + meta.channelOffset * meta.dimStride;
        DataCopyPad(convStateGm_[finalDestinationOffset], buffer_[slot * 2U], params);
        Mutex::Unlock<PIPE_MTE3>(CCS_MUTEX[slot]);
    }

    const CompactConvStateTilingData *tiling_{nullptr};
    TPipe pipe_;
    TBuf<TPosition::VECCALC> bufferBuf_;
    GlobalTensor<stateType> convStateGm_;
    GlobalTensor<int64_t> baseAddrsGm_;
    GlobalTensor<int64_t> blockStridesGm_;
    GlobalTensor<int64_t> dimStridesGm_;
    GlobalTensor<int64_t> tokenStridesGm_;
    GlobalTensor<stateIndexType> stateIndicesGm_;
    GlobalTensor<planType> commitLensGm_;
    GlobalTensor<planType> finalIndicesGm_;
    GlobalTensor<planType> boundaryIndicesGm_;
    GlobalTensor<planType> boundaryLensGm_;
    LocalTensor<stateType> buffer_[CCS_LOCAL_BUFFERS_4];
};

} // namespace GDN

#endif // COMPACT_CONV_STATE_ARCH35_H
