/**
 * Hardware buffer bases for the fused GDN forward. One LocalTensor per memory
 * level, byte 0 is the hardware base. Tile addresses live in the kernel.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_COMMON_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_COMMON_H

#include "kernel_operator.h"

namespace GdnFwd32 {

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

using namespace AscendC;

constexpr int kBt = 32;
// vsstb splits a 256 B register into 8 datablocks. A block stride divisible by 8
// keeps every write in one UB bank group (8 cycles). +1 puts them in 8 groups.
constexpr int kNzBlkStride = kBt + 1;
constexpr int kHalfNzBlkStride = 17;
constexpr int kDim = 128;
constexpr float kEps = 1e-6f;
constexpr int32_t kVl = 64;

constexpr AscendC::FixpipeConfig kFixUbNz2Nd = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig kFixUbNz = {AscendC::CO2Layout::NZ, true};

}  // namespace GdnFwd32

#endif
