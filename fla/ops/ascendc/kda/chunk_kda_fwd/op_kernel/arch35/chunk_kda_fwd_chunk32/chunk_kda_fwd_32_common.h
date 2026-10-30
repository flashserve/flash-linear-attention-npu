#pragma once

#include <stdint.h>

#include "kernel_operator.h"

// 与 op_host/chunk_kda_fwd_tiling.h 的 ChunkKdaFwd32TilingData 一致。
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
    uint64_t cuData[1025];
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

#define FIXtoMTE3_L1Release  0
#define FIXtoMTE3_UBReady  2
#define FIXtoV_UBReady 3
#define MTE1toV_UBReady 5
#define VtoFIX_UBRelease 6
#define VtoMTE1_UBRelease 8
#define MTE3toMTE1_L1Ready 9
#define MTE3toMTE1_UBRelease 11
#define MTE1toMTE3_L1Release 12
#define MTE3toFIX_UBRelease 14

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
    uint32_t vecSetState[16] = {};
    uint32_t vecSetQue1[16][30] = {};
    uint32_t hSQ[16] = {};
    uint32_t tSQ[16] = {};
    uint32_t cnt[16] = {};
    //uint32_t vecSetQue2[16][8] = {};

    __aicore__ inline void Init() {
        // 每个任务开始时清零软件伪 Set 队列。
        for(int i = 0; i < 16; i ++) {
            vecSetState[i] = 0;
            hSQ[i] = 0;
            tSQ[i] = 0;
            cnt[i] = 0;
        }
        for(int i = 0; i < 8; i ++) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(i);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(i);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
    }

    template <uint32_t type>
    __aicore__ inline void SetIntraBlock(uint32_t buffid) {
        if constexpr (type == FIXtoMTE3_L1Release || type == FIXtoV_UBReady) {
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(type + ((cnt[type]/15)&1));
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(type + ((cnt[type]/15)&1) + 16);
            cnt[type] ++;
        }
        else if constexpr (type == FIXtoMTE3_UBReady) {
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(type);
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(type + 16);
        }
        else if constexpr (type == MTE1toMTE3_L1Release) {
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(type + ((cnt[type]/15)&1));
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(type + ((cnt[type]/15)&1) + 16);
            cnt[type] ++;
        }
        else if constexpr (type == MTE1toV_UBReady) {
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(type);
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(type + 16);
        }
        else {
            vecSetState[type] = (vecSetState[type]) | ((uint32_t)1 << buffid);
            vecSetQue1[type][hSQ[type]] = buffid;
            // uint32_t t = vecSetQue2[type][(pSQ[type]) /4];
            // mask = ~mask;
            // t = t & mask;
            // vecSetQue2[type][(pSQ[type]) /4] = t;
            hSQ[type] ++;
            hSQ[type] -= 30*(hSQ[type] >= 30);
        }
    }

    template <uint32_t type>
    __aicore__ inline void WaitIntraBlock(uint32_t buffid) {
        while(((vecSetState[type]) & ((uint32_t)1 << buffid))) {
            uint32_t id1 = vecSetQue1[type][tSQ[type]];
            tSQ[type] ++;
            tSQ[type] -= 30*(tSQ[type] >= 30);
            vecSetState[type] = (vecSetState[type]) & (~((uint32_t)1 << id1));
            // uint32_t t = vecSetQue2[type][(pSQ[type]) /4];
            // uint32_t id2 = t & (uint32_t)0xFF;
            if constexpr (type == VtoFIX_UBRelease || type == MTE3toFIX_UBRelease) {
                AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(type + ((cnt[type]/15)&1));
                AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(type + ((cnt[type]/15)&1) + 16);
                cnt[type] ++;
            }
            if constexpr (type == MTE3toMTE1_L1Ready) {
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(type + ((cnt[type]/15)&1));
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(type + ((cnt[type]/15)&1) + 16);
                cnt[type] ++;
            }
            if constexpr (type == VtoMTE1_UBRelease || type == MTE3toMTE1_UBRelease) {
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(type);
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(type + 16);
            }
        }
    }

    __aicore__ inline void Finalize() {
        for(int i = 0; i < 8; i ++) {
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
    uint32_t vecSetState[16] = {};
    uint32_t vecSetQue1[16][30] = {};
    uint32_t hSQ[16] = {};
    uint32_t tSQ[16] = {};
    uint32_t cnt[16] = {};
    //uint32_t vecSetQue2[16][8] = {};

    __aicore__ inline void Init() {
        // 每个任务开始时清零软件伪 Set 队列,使簿记从任务起点重新开始。
        for(int i = 0; i < 16; i ++) {
            vecSetState[i] = 0;
            hSQ[i] = 0;
            tSQ[i] = 0;
            cnt[i] = 0;
        }
        for(int i = 0; i < 8; i ++) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(i);
        }
    }

    template <uint32_t type>
    __aicore__ inline void SetIntraBlock(uint32_t buffid) {
        if constexpr (type == VtoFIX_UBRelease) {
            AscendC::CrossCoreSetFlag<4, PIPE_V>(type + ((cnt[type]/15)&1));
            cnt[type] ++;
        }
        else if constexpr (type == VtoMTE1_UBRelease) {
            AscendC::CrossCoreSetFlag<4, PIPE_V>(type);
        }
        else if constexpr (type == MTE3toFIX_UBRelease || type == MTE3toMTE1_L1Ready) {
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(type + ((cnt[type]/15)&1));
            cnt[type] ++;
        }
        else if constexpr (type == MTE3toMTE1_UBRelease) {
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(type);
        }
        else {
            vecSetState[type] = (vecSetState[type]) | ((uint32_t)1 << buffid);
            vecSetQue1[type][hSQ[type]] = buffid;
            // uint32_t t = vecSetQue2[type][(pSQ[type]) >> 2];
            // mask = ~mask;
            // t = t & mask;
            // vecSetQue2[type][(pSQ[type]) >> 2] = t;
            hSQ[type] ++;
            hSQ[type] -= 30*(hSQ[type] >= 30);
        }
    }

    template <uint32_t type>
    __aicore__ inline void WaitIntraBlock(uint32_t buffid) {
         while(((vecSetState[type]) & ((uint32_t)1 << buffid))) {
            uint32_t id1 = vecSetQue1[type][tSQ[type]];
            tSQ[type] ++;
            tSQ[type] -= 30*(tSQ[type] >= 30);
            vecSetState[type] = (vecSetState[type]) & (~((uint32_t)1 << id1));
            // uint32_t t = vecSetQue2[type][(pSQ[type]) >> 2];
            // uint32_t id2 = t & (uint32_t)0xFF;
            if constexpr (type == FIXtoV_UBReady)
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(type + ((cnt[type]/15)&1));
            if constexpr (type == MTE1toV_UBReady)
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(type);
            if constexpr (type == FIXtoMTE3_L1Release || type == MTE1toMTE3_L1Release)
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(type + ((cnt[type]/15)&1));
            if constexpr (type == FIXtoMTE3_UBReady)
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(type);
            cnt[type] ++;
        }
    }

    __aicore__ inline void Finalize() {
        for(int i = 0; i < 8; i ++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(i);
        }
    }
};
#endif
