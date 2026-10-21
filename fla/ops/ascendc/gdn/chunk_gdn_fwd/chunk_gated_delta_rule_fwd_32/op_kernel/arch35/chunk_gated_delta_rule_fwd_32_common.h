/**
 * On-chip map for the fused GDN forward. Addresses match the prepare kernel:
 * one LocalTensor per memory level, byte 0 is the hardware base.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_COMMON_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_COMMON_H

#include "kernel_operator.h"

struct HardwareInfo {
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 256 * 1024;
    static uint32_t const ubSize = 248 * 1024;
};

enum class BufferType : uint32_t {
    ASCEND_UB,
    ASCEND_CB,
    ASCEND_L0A,
    ASCEND_L0B,
    ASCEND_L0C,
    ASCEND_MAX
};

struct OnChipBuffer {
    template <typename T>
    using Tensor = AscendC::LocalTensor<T>;

public:
    __aicore__ inline OnChipBuffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
            HardwareInfo::ubSize, HardwareInfo::l1Size, HardwareInfo::l0ASize, HardwareInfo::l0BSize,
            HardwareInfo::l0CSize};
        buffer_[(uint32_t)BufferType::ASCEND_UB] =
            Tensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        buffer_[(uint32_t)BufferType::ASCEND_CB] =
            Tensor<uint8_t>(AscendC::TPosition::A1, 0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        buffer_[(uint32_t)BufferType::ASCEND_L0A] =
            Tensor<uint8_t>(AscendC::TPosition::A2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        buffer_[(uint32_t)BufferType::ASCEND_L0B] =
            Tensor<uint8_t>(AscendC::TPosition::B2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        buffer_[(uint32_t)BufferType::ASCEND_L0C] =
            Tensor<uint8_t>(AscendC::TPosition::CO1, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
    }

    template <BufferType bufferType, typename Dtype = float>
    __aicore__ inline Tensor<Dtype> GetBuffer(const uint32_t offset) const
    {
        return buffer_[(uint32_t)bufferType][offset].template ReinterpretCast<Dtype>();
    }

private:
    AscendC::LocalTensor<uint8_t> buffer_[(uint32_t)BufferType::ASCEND_MAX];
};

namespace GdnFwd32 {
using namespace AscendC;

constexpr int kBt = 32;
constexpr int kDim = 128;
constexpr float kEps = 1e-6f;
constexpr float kRcpLn2 = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr int32_t kVl = 64;

constexpr uint32_t UB_R = 0;
constexpr uint32_t UB_Q = 32 * 1024;
constexpr uint32_t UB_K = 40 * 1024;
constexpr uint32_t UB_V = 48 * 1024;
constexpr uint32_t UB_KBG = 56 * 1024;
constexpr uint32_t UB_QEXP = 64 * 1024;
constexpr uint32_t UB_G = 72 * 1024;
constexpr uint32_t UB_BETA = 73 * 1024;
constexpr uint32_t UB_BE = 74 * 1024;
constexpr uint32_t UB_GP = 75 * 1024;
constexpr uint32_t UB_EG = 76 * 1024;
constexpr uint32_t UB_KKT = 80 * 1024;
constexpr uint32_t UB_QK = 84 * 1024;
constexpr uint32_t UB_L = 88 * 1024;
constexpr uint32_t UB_X00 = 92 * 1024;
constexpr uint32_t UB_X11 = 93 * 1024;
constexpr uint32_t UB_AFP = 94 * 1024;
constexpr uint32_t UB_ABF = 98 * 1024;
constexpr uint32_t UB_ATTN = 100 * 1024;
constexpr uint32_t UB_U = 104 * 1024;
constexpr uint32_t UB_P = 120 * 1024;
constexpr uint32_t UB_OACC = 136 * 1024;
constexpr uint32_t UB_D = 152 * 1024;

constexpr uint32_t L1_K = 0;
constexpr uint32_t L1_Q = 8 * 1024;
constexpr uint32_t L1_VB = 16 * 1024;
constexpr uint32_t L1_KBG = 24 * 1024;
constexpr uint32_t L1_QEXP = 32 * 1024;
constexpr uint32_t L1_A = 40 * 1024;
constexpr uint32_t L1_H = 48 * 1024;
constexpr uint32_t L1_W = 80 * 1024;
constexpr uint32_t L1_VNEW = 88 * 1024;
constexpr uint32_t L1_VNEWG = 96 * 1024;
constexpr uint32_t L1_ATTN = 104 * 1024;

constexpr AscendC::FixpipeConfig kFixUbNz2Nd = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig kFixUbNz = {AscendC::CO2Layout::NZ, true};

// Stage1/2 ping-pong. One AIV owns two chunks (db 0/1). A round is 4 chunks:
//   task 0 AIV0 db0 [0,32)   task 1 AIV1 db0 [32,64)
//   task 2 AIV0 db1 [64,96)  task 3 AIV1 db1 [96,128)
// UB slots below are per AIV. L1 slots are shared, one per task.
constexpr uint32_t kS1Q = 0;
constexpr uint32_t kS1K = 16 * 1024;
constexpr uint32_t kS1V = 32 * 1024;
constexpr uint32_t kS1Kbg = 48 * 1024;
constexpr uint32_t kS1Qexp = 64 * 1024;
constexpr uint32_t kS1G = 80 * 1024;
constexpr uint32_t kS1Beta = 81 * 1024;
constexpr uint32_t kS1Be = 82 * 1024;
constexpr uint32_t kS1Gp = 83 * 1024;
constexpr uint32_t kS1Eg = 84 * 1024;
constexpr uint32_t kS1Kkt = 88 * 1024;
constexpr uint32_t kS1Qk = 96 * 1024;
constexpr uint32_t kS1SlotBf16 = 8 * 1024;
constexpr uint32_t kS1SlotG = 512;
constexpr uint32_t kS1SlotBeta = 256;
constexpr uint32_t kS1SlotBe = 256;
constexpr uint32_t kS1SlotGp = 512;
constexpr uint32_t kS1SlotEg = 256;
constexpr uint32_t kS1SlotKkt = 4 * 1024;
constexpr uint32_t kL1TaskStride = 16 * 1024;
constexpr uint32_t kL1Vb0 = 64 * 1024;
constexpr uint32_t kL1Kbg0 = 96 * 1024;
constexpr uint32_t kL1A0 = 128 * 1024;
constexpr uint32_t kL1H0 = 136 * 1024;
constexpr uint32_t kL1W0 = 264 * 1024;
constexpr uint32_t kUbL = 104 * 1024;
constexpr uint32_t kUbAttn = 112 * 1024;
constexpr uint32_t kUbX00 = 116 * 1024;
constexpr uint32_t kUbX11 = 117 * 1024;
constexpr uint32_t kUbAfp = 120 * 1024;
constexpr uint32_t kUbAbf = 124 * 1024;
constexpr uint32_t kUbU = 128 * 1024;
constexpr uint32_t kUbP = 160 * 1024;
// Half of H (64x128 fp32 delta, 64x128 bf16 state). Each AIV keeps its K rows
// for the whole head, same idea as kda_32_32 slot 23. The low UB is the
// ping-pong prepare area, so these sit above ubP.
constexpr uint32_t kUbD = 192 * 1024;
constexpr uint32_t kUbR = 224 * 1024;
// Side copies of the chunk whose L2 runs under the current step's Stage6.
// ubSh_ ends at 253184; UB is 248KB. be 32, gp 32, eg+rcp 64 floats.
constexpr uint32_t kUbFut = 253184;
constexpr uint32_t kUbFutBe = kUbFut;
constexpr uint32_t kUbFutGp = kUbFut + 128;
constexpr uint32_t kUbFutEg = kUbFut + 256;
constexpr uint32_t kL0AbBank = 32 * 1024;
constexpr uint32_t kL0CBank = 64 * 1024;
constexpr int kTasksPerRound = 4;
// AIC sees an AIV1 CrossCore id as id + 16.
// 0..3: Stage1 L1 ready, then Stage2 UB ready.
// 4..7: Stage3 L1 ready, then Stage4 UB ready. Set as soon as that chunk's
// Stage3 finishes; the other chunk keeps running.
// Serial tail, one id each direction so an AIV never waits its own Set.
// 8 H on L1, 9 P in UB, 10 v_new on L1, 11 D and o_acc in UB.
constexpr uint16_t kAiv1FlagOff = 16;
constexpr uint16_t kFlagH = 8;
constexpr uint16_t kFlagP = 9;
constexpr uint16_t kFlagV = 10;
constexpr uint16_t kFlagO = 11;
// 12: owner has published the next chunk's H. 13: AIC tells the next owner to start.
constexpr uint16_t kFlagNext = 12;
constexpr uint16_t kFlagGo = 13;
// 14: both AIVs finished this head (including the last chunk's GM store).
// 15: AIC releases both AIVs into the next head. Without this, the even AIV
// starts the next head's Stage1 while the odd AIV's last chunk still reads L1.
constexpr uint16_t kFlagDone = 14;
constexpr uint16_t kFlagHead = 15;

// AIC sees an AIV1 Set(id) as id + 16, and an AIV1 Wait(id) hears AIC's Set(id + 16).
__aicore__ inline uint16_t AicSeen(uint16_t id, uint32_t task)
{
    if ((task & 1U) != 0) {
        return static_cast<uint16_t>(id + kAiv1FlagOff);
    }
    return id;
}

__aicore__ inline int32_t PingPongDb(uint32_t task)
{
    return static_cast<int32_t>(task >> 1);
}

__aicore__ inline uint16_t CubeWaitFlag(uint32_t task)
{
    uint16_t flag = static_cast<uint16_t>(task);
    if ((task & 1U) != 0) {
        flag = static_cast<uint16_t>(flag + kAiv1FlagOff);
    }
    return flag;
}

__aicore__ inline void SyncAll()
{
    PipeBarrier<PIPE_ALL>();
}

}  // namespace GdnFwd32

#endif
