
// Implement Init for set/wait flag mechanism in Cube
// Implement Cube-side set/wait_intra_block with the "pseudo-set method"

#include "kernel_operator.h"  // SetFlag/WaitFlag/CrossCoreSetFlag/CrossCoreWaitFlag(AscendC 基础 API)

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
