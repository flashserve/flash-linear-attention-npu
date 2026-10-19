/**
 * Cube path copied from chunk_gated_delta_rule_fwd_prepare:
 * ND column-fraction upload, LoadData2D, Mmad, Fixpipe NZ2ND back to UB.
 * transposeB false: B NZ is [n, k] (kkt). true: B NZ is [k, n] (WuMatmul).
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_MATMUL_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_MATMUL_H

#include "chunk_gated_delta_rule_fwd_32_common.h"

namespace GdnFwd32 {

__aicore__ inline void FillLoad2D(AscendC::LoadData2DParamsV2 &p, uint16_t mStep, uint16_t kStep, uint16_t srcStride,
                                  uint16_t dstStride, bool transpose)
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

__aicore__ inline void LoadL1NzToL0AB(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l1B,
                                      LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B, int32_t m, int32_t n,
                                      int32_t k, bool transposeB, uint16_t kStart = 0, uint8_t evt = 0,
                                      bool transposeA = false)
{
    constexpr uint32_t c0 = 16;
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t aKFracs = static_cast<uint16_t>(k / c0);
    AscendC::LoadData2DParamsV2 loadA;
    if (transposeA) {
        // Source NZ is the matmul A transposed: stored [k, m], same rule as transposeB.
        FillLoad2D(loadA, aKFracs, mFracs, aKFracs, mFracs, true);
    } else {
        FillLoad2D(loadA, mFracs, aKFracs, mFracs, mFracs, false);
        loadA.kStartPosition = kStart;
    }
    AscendC::LoadData(l0A, l1A, loadA);

    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16) : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / c0) : static_cast<uint16_t>(k / c0);
    const uint16_t bDst = transposeB ? static_cast<uint16_t>(n / 16) : bMFracs;
    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, bMFracs, bKFracs, bMFracs, bDst, transposeB);
    AscendC::LoadData(l0B, l1B, loadB);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
}

__aicore__ inline void MmadL0ToL0C(LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B, LocalTensor<float> l0C,
                                   int32_t m, int32_t n, int32_t k, bool initC, uint8_t evt = 0)
{
    if (initC) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    }
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = k;
    mmad.cmatrixInitVal = initC;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

__aicore__ inline void UploadNdToL1(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> ubNd, uint32_t rows,
                                    uint32_t cols)
{
    const uint32_t nFrac = cols / 16;
    const uint16_t srcGap = nFrac == 0 ? 0 : static_cast<uint16_t>(nFrac - 1);
    for (uint32_t fc = 0; fc < nFrac; ++fc) {
        DataCopy(l1[fc * 16 * rows], ubNd[fc * 16], DataCopyParams(static_cast<uint16_t>(rows), 1, srcGap, 0));
    }
}

__aicore__ inline void DownloadL1NzToUb(LocalTensor<bfloat16_t> ubNd, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                       uint32_t cols)
{
    const uint32_t nFrac = cols / 16;
    const uint16_t dstGap = nFrac == 0 ? 0 : static_cast<uint16_t>(nFrac - 1);
    for (uint32_t fc = 0; fc < nFrac; ++fc) {
        DataCopy(ubNd[fc * 16], l1[fc * 16 * rows], DataCopyParams(static_cast<uint16_t>(rows), 1, 0, dstGap));
    }
}

// L0C is fp32 NZ. ROW_MAJOR Fixpipe writes row-major ND into the selected AIV UB.
__aicore__ inline void FixpipeFp32Ub(LocalTensor<float> ub, LocalTensor<float> l0C, int m, int n, uint8_t subBlk = 0)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p;
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = static_cast<uint32_t>(n);
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.subBlockId = (subBlk != 0);
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    AscendC::Fixpipe<float, float, kFixUbNz2Nd>(ub, l0C, p);
}

constexpr AscendC::FixpipeConfig kFixNzL1 = {AscendC::CO2Layout::NZ, false};

// L0C fp32 NZ -> L1 bf16 NZ, C0=16. Column-fractal stride is m * 16 elements.
__aicore__ inline void FixpipeBf16L1(LocalTensor<bfloat16_t> l1, LocalTensor<float> l0C, int m, int n)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = static_cast<uint32_t>(m) * 16;
    p.quantPre = QuantMode_t::F322BF16;
    p.isChannelSplit = false;
    p.unitFlag = 0;
    AscendC::Fixpipe<bfloat16_t, float, kFixNzL1>(l1, l0C, p);
}

}  // namespace GdnFwd32

#endif
