// Finalize A2 private implementation; arithmetic/scheduling source: 07d1b942.
// Kept local so --ops=chunk_gated_delta_rule_bwd_finalize builds independently.
/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_BWD_DQKWG_SCHEDULE_H
#define CHUNK_BWD_DQKWG_SCHEDULE_H

#include <cstdint>

// 原合法、已分配输入域内的整数调度资格；不扩展旧地址或 metadata 支持范围。
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline constexpr bool DqkwgUseU32Schedule(
    uint64_t B, uint64_t numChunks, uint64_t HV, uint64_t HK, uint64_t coreNum, uint64_t hkPartitions)
{
    constexpr uint64_t limitU32 = UINT32_MAX;
    if (hkPartitions != 1 || coreNum == 0 || coreNum > limitU32 ||
        B > limitU32 || numChunks > limitU32 || HK == 0 || HK > HV || HV > limitU32 || HV % HK != 0) {
        return false;
    }
    // 先做除法界定再形成 B*N；末次 taskIdx += C 也必须可表示。
    const uint64_t taskLimit = limitU32 - (coreNum - 1);
    return numChunks == 0 || B <= taskLimit / numChunks;
}

template <bool WIDE_SCHEDULE>
struct DqkwgScheduleIndex {
    using Type = uint64_t;
};

template <>
struct DqkwgScheduleIndex<false> {
    using Type = uint32_t;
};

#endif // CHUNK_BWD_DQKWG_SCHEDULE_H
