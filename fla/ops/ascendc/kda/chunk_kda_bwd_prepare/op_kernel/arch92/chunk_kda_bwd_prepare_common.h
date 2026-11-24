/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH92_COMMON_H
#define CHUNK_KDA_BWD_PREPARE_ARCH92_COMMON_H

#include "../chunk_kda_bwd_prepare_utils.h"

namespace KDA::Arch92 {

constexpr uint32_t kRows = 64; // 每个 chunk 最多 64 个 token
constexpr uint32_t kDim = 128; // K、V 通道数均为 128
constexpr uint32_t kSlots = 4; // 每核组的 L1 输入 owner 数和 UB 结果组数，每份对应一个 task
constexpr uint32_t kL0Slots = 2; // L0A、L0B、L0C 各自的槽位数
constexpr uint32_t kL0SlotBytes = 0x8000; // 每个 L0A/B/C 槽 32 KiB，各层两槽共 64 KiB

// L1：每个 owner 保存一个 task 的全部输入；以下 offset 均为 owner 内的字节偏移。
constexpr uint32_t kOwnerBytes = 0x12000; // 单份 L1 输入区 72 KiB
constexpr uint32_t kDoOffset = 0; // 0 KiB：dO，BF16 [64,128]，占 16 KiB
constexpr uint32_t kVNewOffset = 0x4000; // 16 KiB：vNew，BF16 [64,128]，占 16 KiB
constexpr uint32_t kHOffset = 0x8000; // 32 KiB：h，BF16 [128,128]，占 32 KiB
constexpr uint32_t kAqkOffset = 0x10000; // 64 KiB：aqk，BF16 [64,64]，占 8 KiB
constexpr uint32_t kL1Bytes = kSlots * kOwnerBytes; // 四份 L1 输入区总计 288 KiB

// UB：每组保存一个 task 的全部结果；以下 raw/out/Q/D offset 均为组内字节偏移。
constexpr uint32_t kGroupBytes = 0x14000; // 单组 UB 结果区 80 KiB
constexpr uint32_t kRawOffset = 0; // 0 KiB：A_raw，FP32 [64,64]，占 16 KiB，Cube 原始结果
constexpr uint32_t kOutOffset = 0x4000; // 16 KiB：A_out，FP32 [64,64]，占 16 KiB，VF 后写 d_aqk
constexpr uint32_t kQOffset = 0x8000; // 32 KiB：Q，FP32 [64,128]，占 32 KiB，最终写 dq_raw
constexpr uint32_t kDOffset = 0x10000; // 64 KiB：D，BF16 [64,128]，占 16 KiB，最终写 dv
constexpr uint32_t kMaskOffset = kSlots * kGroupBytes; // UB 绝对字节偏移 320 KiB：共享 mask，占 16 KiB
constexpr uint32_t kUbBytes = kMaskOffset + kRows * kRows * sizeof(float); // 四组结果加共享 mask，共 336 KiB

// 跨核 flag：实际 ID 为起始编号加 slot（0..3），单 AIV 共使用 0..15。
constexpr uint8_t kCrossMode = 0x4; // AIC/AIV 跨核 SetFlag/WaitFlag 使用的同步模式
constexpr uint16_t kUbFree = 0; // ID 0..3，AIV MTE3 → AIC FIX：整组结果已写 GM，UB 可复用
constexpr uint16_t kAReady = 4; // ID 4..7，AIC FIX → AIV V：A_raw 已写入 UB，可执行 VF
constexpr uint16_t kQReady = 8; // ID 8..11，AIC FIX → AIV MTE3：Q 已写入 UB，可写 dq_raw
constexpr uint16_t kDReady = 12; // ID 12..15，AIC FIX → AIV MTE3：D 已写入 UB，可写 dv

__aicore__ inline uint32_t Align16(uint32_t n) { return (n + 15) / 16 * 16; }

struct WorkInfo {
    ChunkInfo chunk;
    int64_t head = 0;
    uint32_t slot = 0;
};

// Both engines use this iterator. Slots follow the LOCAL valid-work ordinal,
// never head%4 or global work%4 (the grid stride may itself be a multiple of 4).
class WorkIterator {
public:
    __aicore__ inline WorkIterator(const ChunkKdaBwdPrepareTilingData *tiling,
                                  GM_ADDR cu, GM_ADDR indices)
        : tiling_(tiling), cu_(cu), indices_(indices),
          work_(AscendC::GetBlockIdx()), stride_(AscendC::GetBlockNum()) {}

    __aicore__ inline bool Next(WorkInfo &result)
    {
        while (stride_ > 0 && work_ < tiling_->workTaskNum) {
            const int64_t id = work_;
            work_ += stride_;
            ResolveChunk(id % tiling_->chunkTaskNum, cu_, indices_, *tiling_, result.chunk);
            if (!result.chunk.valid) continue;
            result.head = id / tiling_->chunkTaskNum;
            result.slot = static_cast<uint32_t>(ordinal_++ % kSlots);
            return true;
        }
        result.chunk.valid = false;
        return false;
    }

private:
    const ChunkKdaBwdPrepareTilingData *tiling_;
    GM_ADDR cu_;
    GM_ADDR indices_;
    int64_t work_, stride_;
    uint64_t ordinal_ = 0;
};

template <AscendC::HardEvent E>
__aicore__ inline void AllocateEvents(AscendC::TPipe *pipe, AscendC::TEventID *ids,
                                    uint32_t count, bool credit)
{
    for (uint32_t i = 0; i < count; ++i) {
        ids[i] = pipe->AllocEventID<E>();
        if (credit) AscendC::SetFlag<E>(ids[i]);
    }
}

template <AscendC::HardEvent E>
__aicore__ inline void ReleaseEvents(AscendC::TPipe *pipe, AscendC::TEventID *ids,
                                   uint32_t count, bool credit)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (credit) AscendC::WaitFlag<E>(ids[i]);
        pipe->ReleaseEventID<E>(ids[i]);
    }
}

} // namespace KDA::Arch92
#endif
