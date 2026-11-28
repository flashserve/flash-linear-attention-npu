#pragma once

#include <stdint.h>

#include "kernel_operator.h"

// 与 host 侧 tiling 布局共享的 cu_seqlens 容量。
constexpr uint32_t cu_data_capacity = 1025;

// 与 op_host/chunk_kda_fwd_fused_tiling.h 的 ChunkKdaFwdFusedTilingData 一致。
struct KdaFd32TilingData {
    uint64_t nChunks;
    uint64_t scaleBf16;
    uint64_t gateScaleBits;
    uint64_t headNum;
    uint64_t seqLen;
    uint64_t hasInitialState;
    uint64_t outputFinalState;
    uint64_t qkGroup;
    uint64_t taskCount;
    uint64_t hvNumReal;
    uint64_t blockNum;
    uint64_t varlenMode;
    uint64_t epsilonBits;
    uint64_t gateFlags;
    uint64_t cuData[cu_data_capacity];
};

// C/V 共享形状和地址单位；各侧独有常量仍留在对应实现头。
constexpr uint32_t key_dim = 128;
constexpr uint32_t value_dim = 128;
constexpr uint32_t chunk_size = 32;
constexpr uint32_t chunk_half_rows = chunk_size / 2;
constexpr uint32_t tokens_per_group = 4 * chunk_size;
constexpr int32_t fractal_len = 16;
constexpr int32_t fractal_elems = fractal_len * fractal_len;
constexpr uint32_t state_k_half = key_dim / 2;
constexpr uint32_t value_tile = value_dim / 4;
constexpr uint32_t l1_window_bytes = 512 * 1024;
constexpr uint32_t ub_addr_unit = 512;
constexpr uint32_t l1_addr_unit = 1024;
constexpr uint64_t default_epsilon_bits = 0x3F50624Du; // FP32 位模式：0.814。

// CrossCore event ID；保留现有名称，避免同步协议两端出现不对称的重命名。
constexpr uint32_t FIXtoMTE3_L1Release = 0;
constexpr uint32_t FIXtoMTE3_UBReady = 2;
constexpr uint32_t FIXtoV_UBReady = 3;
constexpr uint32_t MTE1toV_UBReady = 5;
constexpr uint32_t VtoFIX_UBRelease = 6;
constexpr uint32_t VtoMTE1_UBRelease = 8;
constexpr uint32_t MTE3toMTE1_L1Ready = 9;
constexpr uint32_t MTE3toMTE1_UBRelease = 11;
constexpr uint32_t MTE1toMTE3_L1Release = 12;
constexpr uint32_t MTE3toFIX_UBRelease = 14;

constexpr uint32_t sync_type_count = 16;
constexpr uint32_t sync_queue_depth = 30;
constexpr uint32_t sync_init_flag_count = 8;
constexpr uint32_t cv_sync_mode = 4;             // AIC 与指定 AIV 之间的 mode-4 同步。
constexpr uint32_t sync_phase_period = 15;       // 同一 flagId 每 15 次切换一次滚动相位。
constexpr uint32_t peer_vector_flag_offset = 16; // 另一 AIV 使用的 CrossCore event ID 偏移。

#ifdef __CCE_KT_TEST__
#undef __aicore__
#define __aicore__
#else
#undef __aicore__
#define __aicore__ [aicore]
#endif

#if defined(__DAV_C310_CUBE__)
class SyncCV
{
public:
    uint32_t setState[sync_type_count] = {};
    uint32_t setQueue[sync_type_count][sync_queue_depth] = {};
    uint32_t queueHead[sync_type_count] = {};
    uint32_t queueTail[sync_type_count] = {};
    uint32_t flagCount[sync_type_count] = {};

    __aicore__ inline void Init()
    {
        // 每个任务开始时清零软件伪 Set 队列。
        for (uint32_t i = 0; i < sync_type_count; ++i) {
            setState[i] = 0;
            queueHead[i] = 0;
            queueTail[i] = 0;
        }
        // flagCount 记录硬件 CrossCore flag 的滚动相位，必须跨 task 保留。
        for (uint32_t i = 0; i < sync_init_flag_count; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(i);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(i);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
    }

    template <uint32_t eventType>
    __aicore__ inline void SetIntraBlock(uint32_t bufferId)
    {
        if constexpr (eventType == FIXtoMTE3_L1Release || eventType == FIXtoV_UBReady) {
            const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_FIX>(eventType + phase);
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_FIX>(eventType + phase + peer_vector_flag_offset);
            flagCount[eventType]++;
        } else if constexpr (eventType == FIXtoMTE3_UBReady) {
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_FIX>(eventType);
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_FIX>(eventType + peer_vector_flag_offset);
        } else if constexpr (eventType == MTE1toMTE3_L1Release) {
            const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_MTE1>(eventType + phase);
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_MTE1>(eventType + phase + peer_vector_flag_offset);
            flagCount[eventType]++;
        } else if constexpr (eventType == MTE1toV_UBReady) {
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_MTE1>(eventType);
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_MTE1>(eventType + peer_vector_flag_offset);
        } else {
            setState[eventType] |= uint32_t{1} << bufferId;
            setQueue[eventType][queueHead[eventType]] = bufferId;
            queueHead[eventType]++;
            queueHead[eventType] -= sync_queue_depth * (queueHead[eventType] >= sync_queue_depth);
        }
    }

    template <uint32_t eventType>
    __aicore__ inline void WaitIntraBlock(uint32_t bufferId)
    {
        while ((setState[eventType] & (uint32_t{1} << bufferId)) != 0U) {
            const uint32_t queuedBufferId = setQueue[eventType][queueTail[eventType]];
            queueTail[eventType]++;
            queueTail[eventType] -= sync_queue_depth * (queueTail[eventType] >= sync_queue_depth);
            setState[eventType] &= ~(uint32_t{1} << queuedBufferId);
            if constexpr (eventType == VtoFIX_UBRelease || eventType == MTE3toFIX_UBRelease) {
                const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_FIX>(eventType + phase);
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_FIX>(eventType + phase + peer_vector_flag_offset);
                flagCount[eventType]++;
            }
            if constexpr (eventType == MTE3toMTE1_L1Ready) {
                const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_MTE1>(eventType + phase);
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_MTE1>(eventType + phase + peer_vector_flag_offset);
                flagCount[eventType]++;
            }
            if constexpr (eventType == VtoMTE1_UBRelease || eventType == MTE3toMTE1_UBRelease) {
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_MTE1>(eventType);
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_MTE1>(eventType + peer_vector_flag_offset);
            }
        }
    }

    __aicore__ inline void Finalize()
    {
        for (uint32_t i = 0; i < sync_init_flag_count; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(i);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(i);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
    }
};
#elif defined(__DAV_C310_VEC__)
class SyncCV
{
public:
    uint32_t setState[sync_type_count] = {};
    uint32_t setQueue[sync_type_count][sync_queue_depth] = {};
    uint32_t queueHead[sync_type_count] = {};
    uint32_t queueTail[sync_type_count] = {};
    uint32_t flagCount[sync_type_count] = {};

    __aicore__ inline void Init()
    {
        // 每个任务开始时清零软件伪 Set 队列，使簿记从任务起点重新开始。
        for (uint32_t i = 0; i < sync_type_count; ++i) {
            setState[i] = 0;
            queueHead[i] = 0;
            queueTail[i] = 0;
        }
        // flagCount 记录硬件 CrossCore flag 的滚动相位，必须跨 task 保留。
        for (uint32_t i = 0; i < sync_init_flag_count; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(i);
        }
    }

    template <uint32_t eventType>
    __aicore__ inline void SetIntraBlock(uint32_t bufferId)
    {
        if constexpr (eventType == VtoFIX_UBRelease) {
            const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_V>(eventType + phase);
            flagCount[eventType]++;
        } else if constexpr (eventType == VtoMTE1_UBRelease) {
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_V>(eventType);
        } else if constexpr (eventType == MTE3toFIX_UBRelease || eventType == MTE3toMTE1_L1Ready) {
            const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_MTE3>(eventType + phase);
            flagCount[eventType]++;
        } else if constexpr (eventType == MTE3toMTE1_UBRelease) {
            AscendC::CrossCoreSetFlag<cv_sync_mode, PIPE_MTE3>(eventType);
        } else {
            setState[eventType] |= uint32_t{1} << bufferId;
            setQueue[eventType][queueHead[eventType]] = bufferId;
            queueHead[eventType]++;
            queueHead[eventType] -= sync_queue_depth * (queueHead[eventType] >= sync_queue_depth);
        }
    }

    template <uint32_t eventType>
    __aicore__ inline void WaitIntraBlock(uint32_t bufferId)
    {
        while ((setState[eventType] & (uint32_t{1} << bufferId)) != 0U) {
            const uint32_t queuedBufferId = setQueue[eventType][queueTail[eventType]];
            queueTail[eventType]++;
            queueTail[eventType] -= sync_queue_depth * (queueTail[eventType] >= sync_queue_depth);
            setState[eventType] &= ~(uint32_t{1} << queuedBufferId);
            if constexpr (eventType == FIXtoV_UBReady) {
                const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_V>(eventType + phase);
            }
            if constexpr (eventType == MTE1toV_UBReady) {
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_V>(eventType);
            }
            if constexpr (eventType == FIXtoMTE3_L1Release || eventType == MTE1toMTE3_L1Release) {
                const uint32_t phase = (flagCount[eventType] / sync_phase_period) & 1U;
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_MTE3>(eventType + phase);
            }
            if constexpr (eventType == FIXtoMTE3_UBReady) {
                AscendC::CrossCoreWaitFlag<cv_sync_mode, PIPE_MTE3>(eventType);
            }
            flagCount[eventType]++;
        }
    }

    __aicore__ inline void Finalize()
    {
        for (uint32_t i = 0; i < sync_init_flag_count; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(i);
        }
    }
};
#endif
