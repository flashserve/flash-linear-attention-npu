#pragma once
#include "recurrence.h"

namespace kda {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
// A short M leaves most Cube rows unused. Partition K across FP32 vector lanes so
// a single long, serial accumulator cannot move decay-A across a BF16 midpoint.
__simd_vf__ inline void decayDot(__ubuf__ Bf *input, __ubuf__ Bf *weights, __ubuf__ float *output, uint32_t hidden)
{
    using namespace AscendC::MicroAPI;
    auto mask = CreateMask<float, MaskPattern::ALL>();
    vec::F sums[4];
#pragma unroll
    for (uint16_t i = 0; i < 4; ++i) {
        Duplicate(sums[i], 0.f, mask);
    }
    for (uint16_t k = 0; k < hidden; k += 64) {
        vec::F x;
        vec::readBf(x, input + k, mask);
#pragma unroll
        for (uint16_t i = 0; i < 4; ++i) {
            vec::F weight;
            vec::readBf(weight, weights + uint32_t(i) * hidden + k, mask);
            MulAddDst(sums[i], x, weight, mask);
        }
    }
#pragma unroll
    for (uint16_t i = 0; i < 4; ++i) {
        vec::F total;
        ReduceSum(total, sums[i], mask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(output + i, total, mask);
    }
}
#endif

// Below four rows the reference GEMV reduces along K; larger M uses the matrix kernel.
CATLASS_DEVICE bool useDecayGemv(uint32_t rows, uint32_t hidden)
{
    return rows < 4 && hidden % 64 == 0 && uint64_t(hidden) * 10 + 512 <= Arch::UB_SIZE;
}

CATLASS_DEVICE void shortDecayA(Catlass::Arch::Resource<Arch> &resource, GM_ADDR hiddenInput, GM_ADDR weights,
                                GM_ADDR decayA, uint32_t rows, uint32_t worker, uint32_t workers, uint32_t hidden)
{
    constexpr uint32_t columns = 16;
    auto input = resource.ubBuf.template GetBufferByByte<Bf>(0);
    auto weight = resource.ubBuf.template GetBufferByByte<Bf>(hidden * sizeof(Bf));
    auto sums = resource.ubBuf.template GetBufferByByte<float>(5 * hidden * sizeof(Bf));
    auto output = resource.ubBuf.template GetBufferByByte<Bf>(5 * hidden * sizeof(Bf) + 256);
    AscendC::GlobalTensor<Bf> x, w, a;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(hiddenInput));
    w.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(weights));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ Bf *>(decayA));
    for (uint32_t job = worker; job < rows * (128 / columns); job += workers) {
        uint32_t row = job / (128 / columns), col = job % (128 / columns) * columns;
        AscendC::DataCopy(input, x[uint64_t(row) * hidden], hidden);
        for (uint32_t i = 0; i < columns; i += 4) {
            AscendC::DataCopy(weight, w[uint64_t(col + i) * hidden], 4 * hidden);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
            decayDot((__ubuf__ Bf *)input.GetPhyAddr(), (__ubuf__ Bf *)weight.GetPhyAddr(),
                     (__ubuf__ float *)sums[i].GetPhyAddr(), hidden);
#endif
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(output, sums, AscendC::RoundMode::CAST_RINT, columns);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(a[row * 128 + col], output, columns);
        if (job + workers < rows * (128 / columns)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}
} // namespace kda
