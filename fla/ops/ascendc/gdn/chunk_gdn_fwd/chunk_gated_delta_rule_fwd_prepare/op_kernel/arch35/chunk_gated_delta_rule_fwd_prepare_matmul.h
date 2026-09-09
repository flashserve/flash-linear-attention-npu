/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Cube MMAD: C[m, n] = A[m, k] @ B[k, n] into L0C. L1 is cube NZ.
 *   fp32 C0=8, bf16 C0=16.
 *   transposeB=false: B NZ is [n, k] (kkt: same k' tile as A, n=m).
 *   transposeB=true:  B NZ is [k, n] (I / -L / Y). Leaf tiles from 8-col
 *   ND->NZ are NZ(inv.T): pass transposeA / !transposeB for those Loads.
 * fp32 L1 slots are 64x64, so A/B srcStride is kNumMFracs64.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_MATMUL_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_MATMUL_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_fwd_prepare_common.h"

namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;

__aicore__ inline void FillLoad2D(AscendC::LoadData2DParamsV2 &p, uint16_t mStep, uint16_t kStep,
                                  uint16_t srcStride, uint16_t dstStride, bool transpose)
{
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = mStep;
    p.kStep = kStep;
    p.srcStride = srcStride;
    p.dstStride = dstStride;
    p.ifTranspose = transpose;
    p.sid = 0;
}

template <typename InDtype>
__aicore__ inline void MatmulToL0C(AscendC::LocalTensor<InDtype> l1A,
                                   AscendC::LocalTensor<InDtype> l1B,
                                   AscendC::LocalTensor<InDtype> l0A,
                                   AscendC::LocalTensor<InDtype> l0B,
                                   AscendC::LocalTensor<float> l0C,
                                   int32_t m, int32_t n, int32_t k,
                                   bool initC, bool transposeB, bool transposeA = false,
                                   uint8_t evt = 0)
{
    constexpr uint32_t c0 = 32 / sizeof(InDtype);
    constexpr bool kFp32 = sizeof(InDtype) == sizeof(float);
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t aKFracs = static_cast<uint16_t>(k / c0);
    const uint16_t aSrc = kFp32 ? static_cast<uint16_t>(kNumMFracs64) : mFracs;

    AscendC::LoadData2DParamsV2 loadA;
    FillLoad2D(loadA, mFracs, aKFracs, aSrc, mFracs, transposeA);
    AscendC::LoadData(l0A, l1A, loadA);

    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16)
                                        : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / c0)
                                        : static_cast<uint16_t>(k / c0);
    const uint16_t bSrc = kFp32 ? static_cast<uint16_t>(kNumMFracs64) : bMFracs;

    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, bMFracs, bKFracs, bSrc, bMFracs, transposeB);
    AscendC::LoadData(l0B, l1B, loadB);

    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = k;
    mmad.cmatrixInitVal = initC;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

// C = A[m,k] @ B[k,n]. One MMAD. A is 64x64 (kStep=4). B is 64x128
// (isTranspose, kStep=8, dstStride=8). MMAD k stays kDim.
template <typename InDtype>
__aicore__ inline void WuMatmulToL0C(AscendC::LocalTensor<InDtype> l1A,
                                     AscendC::LocalTensor<InDtype> l1B,
                                     AscendC::LocalTensor<InDtype> l0A,
                                     AscendC::LocalTensor<InDtype> l0B,
                                     AscendC::LocalTensor<float> l0C,
                                     int32_t m, int32_t n, int32_t kDim,
                                     uint8_t evt)
{
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t nFracs = static_cast<uint16_t>(n / 16);
    const uint16_t kFracs = static_cast<uint16_t>(kDim / 16);

    AscendC::LoadData2DParamsV2 loadA;
    FillLoad2D(loadA, mFracs, kFracs, mFracs, mFracs, false);
    AscendC::LoadData(l0A, l1A, loadA);

    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, kFracs, nFracs, kFracs, nFracs, true);
    AscendC::LoadData(l0B, l1B, loadB);

    SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = kDim;
    mmad.cmatrixInitVal = true;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif
