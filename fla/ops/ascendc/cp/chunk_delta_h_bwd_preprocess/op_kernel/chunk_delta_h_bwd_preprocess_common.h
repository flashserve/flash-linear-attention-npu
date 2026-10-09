/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_common.h
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
struct ChunkDeltaHBwdPreprocessTaskRange {
    uint32_t taskBegin;
    uint32_t taskCount;
    bool tileSplit;  // true：task = (hv, tile) 展平后的序号；false：task = hv
};

template <typename DT, typename GT>
class ChunkDeltaHBwdPreprocessBase {
public:
    __aicore__ inline void InitTilingData(const ChunkDeltaHBwdPreprocessTilingData &tiling)
    {
        tiling_ = tiling;
        coreNum_ = static_cast<uint32_t>(tiling.blockDim);
        // A5：一个 block 内配对 2 个 AIV，两个 AIV 各承包一个 head；A2/A3：1 个 AIV。
        // sliceIdx_ 是本 AIV 在 workspace 里的"份"序号，所有平面/slot 寻址都换成它。
        aivPerBlock_ = (tiling.aivPerBlock == 0) ? 1U : static_cast<uint32_t>(tiling.aivPerBlock);
        subBlockIdx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        // 1:2 核型下 AIV 侧 GetBlockIdx() 返回的是 AIV 的展平序号（0..2*blockDim-1），
        // 除以 subBlockNum 才是所属 block；AIC 侧直接就是 block 序号。
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx()) / static_cast<uint32_t>(AscendC::GetSubBlockNum());
        // v19：head 内并行时两个 AIV 合干同一条链（各切一半），操作数/中间量平面**按 block 共享**
        // （每个 AIV 只写自己那一半的行/列），因此 slice 不再按 sub 分。
        halfSplit_ = (tiling.halfSplit != 0) ? 1U : 0U;
        sliceIdx_ = (halfSplit_ != 0) ? blockIdx_ : (blockIdx_ * aivPerBlock_ + subBlockIdx_);
        sliceNum_ = (halfSplit_ != 0) ? coreNum_ : (coreNum_ * aivPerBlock_);
        hvPerHk_ = static_cast<uint32_t>(tiling.Hv / tiling.Hk);
    }

    // 本 AIV 负责的 head 集合（同一 block 内第 s 个 AIV 取 taskBegin + s、+aivPerBlock …）
    __aicore__ inline uint32_t AivHeadBegin(const ChunkDeltaHBwdPreprocessTaskRange &range) const
    {
        return range.taskBegin + subBlockIdx_;
    }

    // 同一个 block 里两个 AIV 必须步调完全一致：A2/A3 的 0x2 是"AIC:2*AIV 集合同步"，任一 flag 都要
    // 两端 AIV 同时 set/wait。因此这里把本 block 的 head 数**向上取整到 aivPerBlock 的倍数**，
    // 多出来的槽用"本 block 最后一个 head"重复填满（幂等：重复算一遍、写同样的结果）。
    // 注意不要改成"不补齐 + 把 blockDim 开到 Hv"去让小 Hv 场景每 head 一个 AIC：
    // 实测那个配置会让 2-chunk 用例的 P 面重新出错（另一处跨核 handoff 的时序依赖被暴露出来），
    // 要启用得先把 PBf→Cube 这条跨核链也做成 fwd_h 那种双向 credit。
    __aicore__ inline uint32_t AivRounds(const ChunkDeltaHBwdPreprocessTaskRange &range) const
    {
        if (halfSplit_ != 0) {
            // v19：一轮 = 一个 head（两个 AIV 各做它的一半），故轮数就是本 block 的 head 数
            return range.taskCount;
        }
        return (range.taskCount + aivPerBlock_ - 1) / aivPerBlock_;
    }

    __aicore__ inline uint32_t AivHeadCount(const ChunkDeltaHBwdPreprocessTaskRange &range) const
    {
        return AivRounds(range);
    }

    __aicore__ inline uint32_t AivHeadAt(const ChunkDeltaHBwdPreprocessTaskRange &range, uint32_t i) const
    {
        if (halfSplit_ != 0) {
            // v19：两个 AIV 处理**同一个** head（半个链），不再按 subBlockIdx 偏移
            const uint32_t hv = range.taskBegin + i;
            const uint32_t headNum = static_cast<uint32_t>(tiling_.Hv);
            return (hv < headNum) ? hv : (headNum - 1);
        }
        const uint32_t hv = range.taskBegin + subBlockIdx_ + i * aivPerBlock_;
        const uint32_t headNum = static_cast<uint32_t>(tiling_.Hv);
        return (hv < headNum) ? hv : (headNum - 1);  // 填充槽：重复最后一个 head
    }

    // v19：本 AIV 在 head 内并行里的"半"序号（0/1）：
    //   0 → dH 的前一半 V 列 + P 的前一半 K 行 + V0 的前一半行；1 → 另一半
    __aicore__ inline uint32_t HalfOf() const
    {
        return subBlockIdx_;
    }

    __aicore__ inline bool HalfSplit() const
    {
        return halfSplit_ != 0;
    }

    // Cube 侧用：本 block 参与服务的"槽"数（已按 aivPerBlock 补齐）
    __aicore__ inline uint32_t AivSlotCount(const ChunkDeltaHBwdPreprocessTaskRange &range) const
    {
        return AivRounds(range) * aivPerBlock_;
    }

    __aicore__ inline uint32_t AivPerBlock() const
    {
        return aivPerBlock_;
    }

    __aicore__ inline uint32_t SubBlockIdx() const
    {
        return subBlockIdx_;
    }

    // 计算本核的任务范围（host 只下发 splitMode / groupHeads / tileNum，kernel 用同一公式重算）
    __aicore__ inline ChunkDeltaHBwdPreprocessTaskRange ResolveTaskRange() const
    {
        ChunkDeltaHBwdPreprocessTaskRange range{0, 0, false};
        if (tiling_.splitMode == CP::CHUNK_DELTA_H_BWD_PREPROCESS_SPLIT_BY_HEAD) {
            // 每核连续 groupHeads 个 head。注意：这里**不要**换成 base/extra 的均衡分配——
            // 实测均衡分配（28 核全忙）会把 P 面（PAt/ZpAt/PBf 这条链）的一个**时序竞态**暴露出来
            // （E 面始终正确、只有 P 面出错，且出错的 head 每次不同）。该竞态是既有的、由 GM 往返
            // 可见性时序决定，必须先正面修掉（方向见 design.md §13.5 的 L0C→UB / 常驻 P），
            // 否则任何"让核更忙"的分核改动都会踩到它。
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
    //   Q̄s[M,K] | W[M,K] | K̄[M,K] | do[M,V] | negdv[M,V]（均模型 dtype，零填充）| decayK[K]（FP32）
    // 说明：v3 起 Q̄s 与 W 相邻、do 与 negdv 相邻，便于把
    //   AB = Q̄sᵀ@do + Wᵀ@(-dv) 合并成一次 GEMM（A = [Q̄s|W]ᵀ [K,2M]，B = [do;-dv] [2M,V]）。
    __aicore__ inline uint64_t SlotQBytes() const
    {
        return tiling_.chunkSize * tiling_.K * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotWBytes() const
    {
        return tiling_.chunkSize * tiling_.K * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotKBytes() const
    {
        return tiling_.chunkSize * tiling_.K * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotDoBytes() const
    {
        return tiling_.chunkSize * tiling_.V * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotNegDvBytes() const
    {
        return tiling_.chunkSize * tiling_.V * sizeof(uint16_t);
    }

    __aicore__ inline uint64_t SlotDecayOffset() const
    {
        return SlotQBytes() + SlotWBytes() + SlotKBytes() + SlotDoBytes() + SlotNegDvBytes();
    }

    // 合并 GEMM 的 A 操作数：slot 起始处 [Q̄s|W]（2M 行 x K）
    __aicore__ inline GM_ADDR SlotAbA(uint32_t slotIdx) const
    {
        return SlotAt(slotIdx, 0);
    }

    // 合并 GEMM 的 B 操作数：do 起始处 [do;-dv]（2M 行 x V）
    __aicore__ inline GM_ADDR SlotAbB(uint32_t slotIdx) const
    {
        return SlotAt(slotIdx, SlotQBytes() + SlotWBytes() + SlotKBytes());
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
        return PlaneAt(offset, planeBytes, sliceNum_, sliceIdx_);
    }

    // 每工作组两份（parity 0/1）的平面：份数 == 2 * blockDim
    __aicore__ inline GM_ADDR WgParityPlaneAt(uint64_t offset, uint64_t planeBytes, uint32_t parity) const
    {
        return PlaneAt(offset, planeBytes, 2 * sliceNum_, static_cast<uint64_t>(sliceIdx_) * 2 + parity);
    }

    // v2.1：每工作组按 window（0/1）分份的平面：份数 == 2 * blockDim。
    // window 由 chunkIdx & 1 给出，让 AIV 可以领先 AIC 一个 chunk 而不覆盖正在被消费的操作数。
    __aicore__ inline GM_ADDR WgWindowPlaneAt(uint64_t offset, uint64_t planeBytes, uint32_t window) const
    {
        return PlaneAt(offset, planeBytes, sliceNum_ * CDHP_WINDOW_COUNT,
                       static_cast<uint64_t>(sliceIdx_) * CDHP_WINDOW_COUNT + (window & 1u));
    }

    // 本工作组第 window 个 slot 的起始地址
    __aicore__ inline GM_ADDR SlotAtWindow(uint32_t window, uint64_t inSlotOffset) const
    {
        return SlotAt(SliceOfWindow(window), inSlotOffset);
    }

    // 本 AIV 在 workspace 里的"份"序号与该 block 的 window 组合出的 slot 序号
    __aicore__ inline uint32_t SliceOfWindow(uint32_t window) const
    {
        return sliceIdx_ * CDHP_WINDOW_COUNT + (window & 1u);
    }

    __aicore__ inline uint32_t SliceIdx() const
    {
        return sliceIdx_;
    }

    // ---- AIC 侧：指定配对 AIV（sub = 0/1）的寻址 ----
    // A5 下一个 block 里两个 AIV 各有自己的一份 slot/平面；AIC 逐个 head 服务时必须显式给 sub。
    // A2/A3 的 aivPerBlock_ == 1，sub 只能是 0，公式退化成原来的形式。
    __aicore__ inline uint32_t SliceIdxOfAiv(uint32_t sub) const
    {
        // v19：head 内并行（两个 AIV 合干同一条链）时，操作数/中间量平面按 **block** 共享一份，
        // AIC 侧必须取这同一份（sub 只用来选 flag 的 peer AIV）；若仍按 blockIdx*aivPerBlock + sub
        // 取，AIC 会落到"另一份"上：slot 读到别的 window、window 平面直接越过平面边界，
        // 实测表现为 synchronize failed 507015（部分档位）或中间量整片错值。
        if (halfSplit_ != 0) {
            (void)sub;
            return blockIdx_;
        }
        return blockIdx_ * aivPerBlock_ + sub;
    }

    __aicore__ inline GM_ADDR WgWindowPlaneAtAiv(uint64_t offset, uint64_t planeBytes, uint32_t sub,
                                                 uint32_t window) const
    {
        const uint64_t slice = SliceIdxOfAiv(sub);
        return PlaneAt(offset, planeBytes, sliceNum_ * CDHP_WINDOW_COUNT,
                       slice * CDHP_WINDOW_COUNT + (window & 1u));
    }

    __aicore__ inline GM_ADDR WgParityPlaneAtAiv(uint64_t offset, uint64_t planeBytes, uint32_t sub,
                                                 uint32_t parity) const
    {
        return PlaneAt(offset, planeBytes, 2 * sliceNum_,
                       static_cast<uint64_t>(SliceIdxOfAiv(sub)) * 2 + parity);
    }

    __aicore__ inline uint32_t SlotOfWindowAiv(uint32_t sub, uint32_t window) const
    {
        return SliceIdxOfAiv(sub) * CDHP_WINDOW_COUNT + (window & 1u);
    }

    __aicore__ inline GM_ADDR DhBfAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.dhBfWsOffset, tiling_.dhBfWsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR AbAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.dvPreWsOffset, tiling_.dvPreWsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR ZAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.dvHatWsOffset, tiling_.dvHatWsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR ZpAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.qtermWsOffset, tiling_.qtermWsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR T1AtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.t1WsOffset, tiling_.t1WsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR T1BfAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.wtermWsOffset, tiling_.wtermWsBytes, sub, window);
    }

    // ---- v13：AB 与 Z 合并成一次 GEMM 的两份拼接操作数（arch35 使用）----
    // aOper：[Q̄s(M,K) | W(M,K) | (-T1)ᵀ(K,K)]，Cube 的 A 操作数（按列主序 [K, 2M+K] 读）
    // bOper：[do(M,V) | -dv(M,V) | dH_bf(K,V)]，Cube 的 B 操作数（按行主序 [2M+K, V] 读）
    __aicore__ inline GM_ADDR AOperAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.aOperWsOffset, tiling_.aOperWsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR BOperAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.bOperWsOffset, tiling_.bOperWsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR AOperAt(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.aOperWsOffset, tiling_.aOperWsBytes, window);
    }

    __aicore__ inline GM_ADDR BOperAt(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.bOperWsOffset, tiling_.bOperWsBytes, window);
    }

    __aicore__ inline GM_ADDR PBfAtAiv(uint32_t sub, uint32_t window = 0) const
    {
        return WgWindowPlaneAtAiv(tiling_.pBfWsOffset, tiling_.pBfWsBytes, sub, window);
    }

    __aicore__ inline GM_ADDR DhAt(uint32_t parity) const
    {
        return WgParityPlaneAt(tiling_.dhWsOffset, tiling_.dhWsBytes, parity);
    }

    __aicore__ inline GM_ADDR DhBfAt(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.dhBfWsOffset, tiling_.dhBfWsBytes, window);
    }

    // v20：把 ZP = (-T1)@P_bf 并进"AB + Z"的同一次 GEMM（仅 A5/arch35，见 struct.h 的 mergeZp）。
    // 打开后：
    //   B 操作数 bOper = [do(M,V) | -dv(M,V) | dH_bf(K,V) | P_bf(K,K)]，行主序 [2M+K, V+K]；
    //   其中行 0..2M 的 [V, V+K) 列必须为 0（do/-dv 不参与 ZP 列）；
    //   C 平面 = [K, V+K]（列 0..V 是 inc = AB + Z，列 V..V+K 是 ZP）；
    //   P_bf 由 AIV 写在 bOper 的第四段，不再单独分配 pBf 平面；
    //   独立的 ZP 平面（qtermWs）随之不再分配。
    __aicore__ inline bool MergeZp() const
    {
        return tiling_.mergeZp != 0;
    }

    // inc 所在平面的行距：合并后是 V+K（与 ZP 同一块平面），否则仍是 V
    __aicore__ inline uint32_t AbRowStride() const
    {
        return MergeZp() ? (static_cast<uint32_t>(tiling_.V) + static_cast<uint32_t>(tiling_.K))
                         : static_cast<uint32_t>(tiling_.V);
    }

    // ZP 在平面内的列偏移与行距（合并后在 inc 右侧；否则是独立的 [K,K] 平面）
    __aicore__ inline uint32_t ZpCol0() const
    {
        return MergeZp() ? static_cast<uint32_t>(tiling_.V) : 0U;
    }

    __aicore__ inline uint32_t ZpRowStride() const
    {
        return MergeZp() ? (static_cast<uint32_t>(tiling_.V) + static_cast<uint32_t>(tiling_.K))
                         : static_cast<uint32_t>(tiling_.K);
    }

    // P_bf（模型 dtype 的 [K,K] 矩阵操作数）的行距
    __aicore__ inline uint32_t PBfRowStride() const
    {
        return MergeZp() ? (static_cast<uint32_t>(tiling_.V) + static_cast<uint32_t>(tiling_.K))
                         : static_cast<uint32_t>(tiling_.K);
    }

    // AB + Z 的输出平面（Cube 写、Vector 读；不依赖状态，属链外工作）：
    //   非合并：[K,V]；合并（v20）：[K, V+K]，列 0..V 是 inc、列 V..V+K 是 ZP。
    __aicore__ inline GM_ADDR AbAt(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.dvPreWsOffset, tiling_.dvPreWsBytes, window);
    }

    // Z = T1ᵀ... 即 (-T1)@dH_prev  [K,V]（Cube 写、Vector 读；在状态链上）
    __aicore__ inline GM_ADDR ZAt(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.dvHatWsOffset, tiling_.dvHatWsBytes, window);
    }

    // ZP = (-T1)@P_prev  [K,K]（Cube 写、Vector 读；P 链上）
    // v20 合并后 ZP 与 inc 共平面，基址与 AbAt 相同，列偏移/行距见 ZpCol0()/ZpRowStride()。
    __aicore__ inline GM_ADDR ZpAt(uint32_t window = 0) const
    {
        if (MergeZp()) {
            return AbAt(window);
        }
        return WgWindowPlaneAt(tiling_.qtermWsOffset, tiling_.qtermWsBytes, window);
    }

    // T1 = Wᵀ@(-K̄) 的模型 dtype 副本 [K,K]（Vector 写、Cube 当矩阵操作数读）
    __aicore__ inline GM_ADDR T1BfAt(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.wtermWsOffset, tiling_.wtermWsBytes, window);
    }

    // T1 = Wᵀ@(-K̄) 的 FP32 平面 [K,K]（Cube 写、Vector 读后转成模型 dtype）
    __aicore__ inline GM_ADDR T1At(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.t1WsOffset, tiling_.t1WsBytes, window);
    }

    __aicore__ inline GM_ADDR PAt(uint32_t parity) const
    {
        return WgParityPlaneAt(tiling_.pWsOffset, tiling_.pWsBytes, parity);
    }

    __aicore__ inline GM_ADDR PBfAt(uint32_t window = 0) const
    {
        if (MergeZp()) {
            // B 操作数第四段：行 2M..2M+K、列 V..V+K（行距 V+K）
            const uint64_t rows = static_cast<uint64_t>(tiling_.chunkSize);
            const uint64_t vDim = static_cast<uint64_t>(tiling_.V);
            return BOperAt(window) +
                   (2U * rows * (vDim + static_cast<uint64_t>(tiling_.K)) + vDim) * sizeof(uint16_t);
        }
        return WgWindowPlaneAt(tiling_.pBfWsOffset, tiling_.pBfWsBytes, window);
    }

    // ---- arch22（A2/A3）兼容别名 ----
    // arch22 仍是 v1/v2 的六 Stage 结构，读写按同一套访问器成对出现，因此沿用旧名字即可保持正确；
    // 语义对应见上面的 v3 注释（v2 的 dV_pre 平面在 v3 里被 AB 取代等）。
    __aicore__ inline GM_ADDR DvPreAt(uint32_t window = 0) const { return AbAt(window); }
    __aicore__ inline GM_ADDR DvHatAt(uint32_t window = 0) const { return ZAt(window); }
    __aicore__ inline GM_ADDR QtermAt(uint32_t window = 0) const { return ZpAt(window); }
    __aicore__ inline GM_ADDR WtermAt(uint32_t window = 0) const { return T1BfAt(window); }
    __aicore__ inline GM_ADDR PcAt(uint32_t window = 0) const
    {
        return WgWindowPlaneAt(tiling_.pcWsOffset, tiling_.wtermWsBytes, window);
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
    uint32_t aivPerBlock_ = 1;   // 1：A2/A3（1:1）；2：A5（1:2）
    uint32_t subBlockIdx_ = 0;   // 本 AIV 在 block 内的序号（A2/A3 恒为 0）
    uint32_t halfSplit_ = 0;     // v19：head 内并行（两个 AIV 合干同一条链）开关
    uint32_t sliceIdx_ = 0;      // workspace 平面/slot 的"份"序号 = blockIdx * aivPerBlock + subBlockIdx
    uint32_t sliceNum_ = 1;      // "份"总数 = blockDim * aivPerBlock
    uint32_t hvPerHk_ = 1;
    uint32_t bos_ = 0;
    uint32_t eos_ = 0;
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_BASE_H
