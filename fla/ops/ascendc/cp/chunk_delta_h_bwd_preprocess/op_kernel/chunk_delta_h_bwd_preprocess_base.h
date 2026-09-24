/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_base.h
 * \brief
 * 与计算引擎无关的公共部分：tiling 解码、GVA 映射、本核任务范围、workspace 寻址。
 *
 * 注意：本目录当前是蓝图，未接入构建。接入前必须按目标 CANN 版本核对本文件中标记为 PROPOSED 的
 * MicroAPI 调用（DataCopy / Mmad / Fixpipe / CrossCoreSetFlag 等），并完成设备精度与 sanitizer 验证。
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_BASE_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_BASE_H

#include "kernel_operator.h"
#include "chunk_delta_h_bwd_preprocess_policy.h"
#include "chunk_delta_h_bwd_preprocess_struct.h"

using namespace AscendC;

namespace CP {

// 本核（工作组）在一次 launch 内负责的任务集合。
// 分核规则来自 stage 合同 §5：
//   - splitMode == BY_HEAD：仅按 head 连续分核，本核遍历该 head 的全部列 tile；
//   - splitMode == BY_TILE：Hv 不足时把 (hv, 列 tile) 展平后按 balanced half-open range 分配。
// 任何情况下都禁止按 chunk 分核：dH 与 P 都是跨 chunk 状态。
struct ChunkDeltaHBwdPreTaskRange {
    uint32_t taskBegin;
    uint32_t taskCount;
    bool tileSplit;  // true：task = (hv, tile) 展平后的序号；false：task = hv
};

template <typename DT, typename GT>
class ChunkDeltaHBwdPreBase {
public:
    __aicore__ inline void InitTilingData(const ChunkDeltaHBwdPreprocessTilingData &tiling)
    {
        tiling_ = tiling;
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        coreNum_ = static_cast<uint32_t>(tiling.blockDim);
        hvPerHk_ = static_cast<uint32_t>(tiling.Hv / tiling.Hk);
    }

    // 计算本核的任务范围（host 只下发 splitMode / groupHeads / tileNum，kernel 用同一公式重算）
    __aicore__ inline ChunkDeltaHBwdPreTaskRange ResolveTaskRange() const
    {
        ChunkDeltaHBwdPreTaskRange range{0, 0, false};
        if (tiling_.splitMode == CP::CHUNK_DELTA_H_BWD_PREPROCESS_SPLIT_BY_HEAD) {
            const uint32_t groupHeads = static_cast<uint32_t>(tiling_.groupHeads);
            range.taskBegin = blockIdx_ * groupHeads;
            const uint32_t headNum = static_cast<uint32_t>(tiling_.Hv);
            const uint32_t begin = range.taskBegin;
            range.taskCount = (begin >= headNum) ? 0 : Min(groupHeads, headNum - begin);
            range.tileSplit = false;
            return range;
        }
        const uint32_t taskNum = static_cast<uint32_t>(tiling_.Hv * tiling_.tileNum);
        const uint32_t base = taskNum / coreNum_;
        const uint32_t extra = taskNum % coreNum_;
        range.taskBegin = blockIdx_ * base + Min(blockIdx_, extra);
        range.taskCount = base + ((blockIdx_ < extra) ? 1U : 0U);
        range.tileSplit = true;
        return range;
    }

    // hv → hk 的 GVA 映射：不要求物理 repeat q/k
    __aicore__ inline uint32_t HkOfHv(uint32_t hv) const
    {
        return hv / hvPerHk_;
    }

    // 列 tile 语义：tile < tileV 属于 E 分支（V 方向），否则属于 P 分支（K 方向）
    __aicore__ inline bool IsVTile(uint32_t tileIdx) const
    {
        return tileIdx < static_cast<uint32_t>(tiling_.tileV);
    }

    __aicore__ inline uint32_t TileOffset(uint32_t tileIdx) const
    {
        const uint32_t bs = static_cast<uint32_t>(tiling_.blockSize);
        return IsVTile(tileIdx) ? (tileIdx * bs) : ((tileIdx - static_cast<uint32_t>(tiling_.tileV)) * bs);
    }

    __aicore__ inline uint32_t TileWidth(uint32_t tileIdx, uint32_t total) const
    {
        const uint32_t bs = static_cast<uint32_t>(tiling_.blockSize);
        const uint32_t remain = (total > TileOffset(tileIdx)) ? (total - TileOffset(tileIdx)) : 0;
        return Min(bs, remain);
    }

    // 本 segment 的 chunk 数与逆序遍历起点
    __aicore__ inline uint32_t ChunkNum(uint32_t bos, uint32_t eos) const
    {
        const uint32_t cs = static_cast<uint32_t>(tiling_.chunkSize);
        return (eos > bos) ? ((eos - bos + cs - 1) / cs) : 0;
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b) const
    {
        return a < b ? a : b;
    }

    // user workspace 子区基址（偏移由 host tiling 规划，单位字节）
    __aicore__ inline GM_ADDR WsAt(uint64_t offset) const
    {
        return userWs_ + offset;
    }

    // ---- 平面寻址（偏移与大小来自 host tiling，单位字节） ----
    __aicore__ inline GM_ADDR SlotAt(uint32_t slotIdx, uint64_t inSlotOffset) const
    {
        return userWs_ + tiling_.slotWsOffset + static_cast<uint64_t>(slotIdx) * tiling_.slotBytes + inSlotOffset;
    }

    // slot 内部布局（与 host 的 slotBytes 规划一致）：
    //   Q̄s[chunkSize,K] | K̄[chunkSize,K] | W[chunkSize,K] | do[chunkSize,V]（均模型 dtype，零填充）
    //   | decayK[K]（FP32）
    __aicore__ inline uint64_t SlotQBytes() const
    {
        return tiling_.chunkSize * tiling_.K * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotKBytes() const
    {
        return tiling_.chunkSize * tiling_.K * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotWBytes() const
    {
        return tiling_.chunkSize * tiling_.K * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotDoBytes() const
    {
        return tiling_.chunkSize * tiling_.V * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotDecayOffset() const
    {
        return SlotQBytes() + SlotKBytes() + SlotWBytes() + SlotDoBytes();
    }

    // 平面都按工作组切分：tiling 里每个平面的大小 = blockDim * 单份大小（dh/p 每个工作组含 2 份 parity），
    // 本工作组取第 BlockIdx() 份。sliceCount 必须是该平面里"份"的总数，sliceIndex 是份序号；
    // 两者必须一起给出，否则 sliceBytes 会被算错并越界写到相邻平面。
    __aicore__ inline GM_ADDR PlaneAt(uint64_t offset, uint64_t planeBytes, uint64_t sliceCount,
                                      uint64_t sliceIndex) const
    {
        return userWs_ + offset + sliceIndex * (planeBytes / sliceCount);
    }

    // 每工作组一份的平面：份数 == blockDim
    __aicore__ inline GM_ADDR WgPlaneAt(uint64_t offset, uint64_t planeBytes) const
    {
        return PlaneAt(offset, planeBytes, coreNum_, blockIdx_);
    }

    // 每工作组两份（parity 0/1）的平面：份数 == 2 * blockDim
    __aicore__ inline GM_ADDR WgParityPlaneAt(uint64_t offset, uint64_t planeBytes, uint32_t parity) const
    {
        return PlaneAt(offset, planeBytes, 2 * coreNum_, static_cast<uint64_t>(blockIdx_) * 2 + parity);
    }

    __aicore__ inline GM_ADDR DhAt(uint32_t parity) const
    {
        return WgParityPlaneAt(tiling_.dhWsOffset, tiling_.dhWsBytes, parity);
    }

    __aicore__ inline GM_ADDR DhBfAt() const
    {
        return WgPlaneAt(tiling_.dhBfWsOffset, tiling_.dhBfWsBytes);
    }

    __aicore__ inline GM_ADDR DvPreAt() const
    {
        return WgPlaneAt(tiling_.dvPreWsOffset, tiling_.dvPreWsBytes);
    }

    __aicore__ inline GM_ADDR DvHatAt() const
    {
        return WgPlaneAt(tiling_.dvHatWsOffset, tiling_.dvHatWsBytes);
    }

    __aicore__ inline GM_ADDR QtermAt() const
    {
        return WgPlaneAt(tiling_.qtermWsOffset, tiling_.qtermWsBytes);
    }

    __aicore__ inline GM_ADDR WtermAt() const
    {
        return WgPlaneAt(tiling_.wtermWsOffset, tiling_.wtermWsBytes);
    }

    __aicore__ inline GM_ADDR T1At() const
    {
        return WgPlaneAt(tiling_.t1WsOffset, tiling_.t1WsBytes);
    }

    __aicore__ inline GM_ADDR PcAt() const
    {
        return WgPlaneAt(tiling_.pcWsOffset, tiling_.pcWsBytes);
    }

    __aicore__ inline GM_ADDR PAt(uint32_t parity) const
    {
        return WgParityPlaneAt(tiling_.pWsOffset, tiling_.pWsBytes, parity);
    }

    __aicore__ inline GM_ADDR PBfAt() const
    {
        return WgPlaneAt(tiling_.pBfWsOffset, tiling_.pBfWsBytes);
    }

    // 本 segment 的 [bos, eos)：varlen 由 cu_seqlens[0:2] 覆盖 host 缺省值
    __aicore__ inline void ResolveSegment(GM_ADDR cuSeqlens)
    {
        if (tiling_.isVarLen != 0 && cuSeqlens != nullptr) {
            GlobalTensor<int64_t> cuSeqlensGm;
            cuSeqlensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            bos_ = static_cast<uint32_t>(cuSeqlensGm.GetValue(0));
            eos_ = static_cast<uint32_t>(cuSeqlensGm.GetValue(1));
        } else {
            bos_ = static_cast<uint32_t>(tiling_.bos);
            eos_ = static_cast<uint32_t>(tiling_.eos);
        }
    }

    __aicore__ inline uint32_t Bos() const
    {
        return bos_;
    }

    __aicore__ inline uint32_t Eos() const
    {
        return eos_;
    }

    // 第 chunkIdx 个 chunk 的起始 token 与有效行数
    __aicore__ inline uint32_t ChunkStart(uint32_t chunkIdx) const
    {
        return bos_ + chunkIdx * static_cast<uint32_t>(tiling_.chunkSize);
    }

    __aicore__ inline uint32_t ChunkRows(uint32_t chunkIdx) const
    {
        const uint32_t start = ChunkStart(chunkIdx);
        const uint32_t remain = (eos_ > start) ? (eos_ - start) : 0;
        return Min(static_cast<uint32_t>(tiling_.chunkSize), remain);
    }

    __aicore__ inline void SetUserWorkspace(GM_ADDR userWs)
    {
        userWs_ = userWs;
    }

    __aicore__ inline uint32_t BlockIdx() const
    {
        return blockIdx_;
    }

protected:
    ChunkDeltaHBwdPreprocessTilingData tiling_;
    GM_ADDR userWs_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t hvPerHk_ = 1;
    uint32_t bos_ = 0;
    uint32_t eos_ = 0;
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_BASE_H
