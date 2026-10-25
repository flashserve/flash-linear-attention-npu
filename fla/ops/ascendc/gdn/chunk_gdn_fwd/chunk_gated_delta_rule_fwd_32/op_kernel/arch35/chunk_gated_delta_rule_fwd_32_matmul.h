/**
 * Cube path copied from chunk_gated_delta_rule_fwd_prepare:
 * ND column-fraction upload, LoadData2D, Mmad, Fixpipe NZ2ND back to UB.
 * transposeB false: B NZ is [n, k] (kkt). true: B NZ is [k, n] (WuMatmul).
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_MATMUL_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_MATMUL_H

#include "chunk_gated_delta_rule_fwd_32_common.h"

namespace GdnFwd32 {

// L1 NZ to one L0 side. transposeA stores the source as [k, m]. No flags.
__aicore__ inline void LoadL1ToL0A(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l0A, int32_t m, int32_t k,
                                   bool transposeA)
{
    constexpr uint32_t c0 = 16;
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t aKFracs = static_cast<uint16_t>(k / c0);
    AscendC::LoadData2DParamsV2 loadA;
    loadA.mStartPosition = 0;
    loadA.kStartPosition = 0;
    loadA.sid = 0;
    if (transposeA) {
        loadA.mStep = aKFracs;
        loadA.kStep = mFracs;
        loadA.srcStride = aKFracs;
        loadA.dstStride = mFracs;
        loadA.ifTranspose = true;
    } else {
        loadA.mStep = mFracs;
        loadA.kStep = aKFracs;
        loadA.srcStride = mFracs;
        loadA.dstStride = mFracs;
        loadA.ifTranspose = false;
    }
    AscendC::LoadData(l0A, l1A, loadA);
}

// transposeB stores the source as [k, n]. No flags.
__aicore__ inline void LoadL1ToL0B(LocalTensor<bfloat16_t> l1B, LocalTensor<bfloat16_t> l0B, int32_t n, int32_t k,
                                   bool transposeB)
{
    constexpr uint32_t c0 = 16;
    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16) : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / c0) : static_cast<uint16_t>(k / c0);
    const uint16_t bDst = transposeB ? static_cast<uint16_t>(n / 16) : bMFracs;
    AscendC::LoadData2DParamsV2 loadB;
    loadB.mStartPosition = 0;
    loadB.kStartPosition = 0;
    loadB.mStep = bMFracs;
    loadB.kStep = bKFracs;
    loadB.srcStride = bMFracs;
    loadB.dstStride = bDst;
    loadB.ifTranspose = transposeB;
    loadB.sid = 0;
    AscendC::LoadData(l0B, l1B, loadB);
}

// Both sides, A first. No flags: the caller sets MTE1_M, and D waits it later.
__aicore__ inline void LoadL1ToL0AB(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l1B,
                                    LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B, int32_t m, int32_t n,
                                    int32_t k, bool transposeA, bool transposeB)
{
    constexpr uint32_t c0 = 16;
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t aKFracs = static_cast<uint16_t>(k / c0);
    AscendC::LoadData2DParamsV2 loadA;
    loadA.mStartPosition = 0;
    loadA.kStartPosition = 0;
    loadA.sid = 0;
    if (transposeA) {
        loadA.mStep = aKFracs;
        loadA.kStep = mFracs;
        loadA.srcStride = aKFracs;
        loadA.dstStride = mFracs;
        loadA.ifTranspose = true;
    } else {
        loadA.mStep = mFracs;
        loadA.kStep = aKFracs;
        loadA.srcStride = mFracs;
        loadA.dstStride = mFracs;
        loadA.ifTranspose = false;
    }
    AscendC::LoadData(l0A, l1A, loadA);

    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16) : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / c0) : static_cast<uint16_t>(k / c0);
    const uint16_t bDst = transposeB ? static_cast<uint16_t>(n / 16) : bMFracs;
    AscendC::LoadData2DParamsV2 loadB;
    loadB.mStartPosition = 0;
    loadB.kStartPosition = 0;
    loadB.mStep = bMFracs;
    loadB.kStep = bKFracs;
    loadB.srcStride = bMFracs;
    loadB.dstStride = bDst;
    loadB.ifTranspose = transposeB;
    loadB.sid = 0;
    AscendC::LoadData(l0B, l1B, loadB);
}

// 64 columns of the full 128-row H. colFrac is 0 or 4. srcStride stays 8
// because each column panel in L1 is still 128 rows tall. dstStride matches
// a packed N=64 tile.
__aicore__ inline void LoadL1NzHCols(LocalTensor<bfloat16_t> l1H, LocalTensor<bfloat16_t> l0B, uint32_t colFrac)
{
    AscendC::LoadData2DParamsV2 loadB;
    loadB.mStartPosition = 0;
    loadB.kStartPosition = colFrac;
    loadB.mStep = 8;
    loadB.kStep = 4;
    loadB.srcStride = 8;
    loadB.dstStride = 4;
    loadB.ifTranspose = true;
    loadB.sid = 0;
    AscendC::LoadData(l0B, l1H, loadB);
}

// BF16 x BF16 into FP32. initC clears L0C. No flags.
__aicore__ inline void MmadL0ToL0C(LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B, LocalTensor<float> l0C,
                                   int32_t m, int32_t n, int32_t k, bool initC)
{
    AscendC::MmadParams mmad;
    mmad.m = m;
    mmad.n = n;
    mmad.k = k;
    mmad.cmatrixInitVal = initC;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    AscendC::Mmad(l0C, l0A, l0B, mmad);
}

// Row-major ND [rows, cols] -> tight NZ in L1. One DataCopy per 16-column panel.
// cols is a multiple of 16. The vector pipe does not touch the layout.
__aicore__ inline void CopyNdToL1(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> nd, uint32_t rows, uint32_t cols)
{
    const uint32_t nFrac = cols / 16;
    const uint16_t srcGap = static_cast<uint16_t>(nFrac - 1);
    for (uint32_t fc = 0; fc < nFrac; ++fc) {
        DataCopy(l1[fc * 16 * rows], nd[fc * 16],
                 DataCopyParams(static_cast<uint16_t>(rows), 1, srcGap, 0));
    }
}


// Tight NZ already sitting in UB (produced by the VF) -> one UB->L1 copy.
// Caller orders the vector store with V_MTE3 before this.
__aicore__ inline void CopyNzToL1(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> nz, uint32_t rows, uint32_t cols)
{
    const uint32_t bytes = rows * cols * sizeof(bfloat16_t);
    DataCopy(l1, nz, DataCopyParams(1, static_cast<uint16_t>(bytes / 32), 0, 0));
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


// L0C fp32 -> UB bf16 NZ. One column panel is [m, 16] contiguous, same image
// Panels are contiguous, so the state saxpy stays elementwise.
__aicore__ inline void FixpipeBf16UbNz(LocalTensor<bfloat16_t> ub, LocalTensor<float> l0C, int m, int n,
                                       uint8_t subBlk = 0)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = static_cast<uint32_t>(m) * 16;
    p.quantPre = QuantMode_t::F322BF16;
    p.dualDstCtl = 0;
    p.subBlockId = (subBlk != 0);
    p.isChannelSplit = false;
    AscendC::Fixpipe<bfloat16_t, float, kFixUbNz>(ub, l0C, p);
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


// [m, n] L0C, m is a multiple of 32. Each call writes m/2 rows to one AIV.
// dualDstCtl 0b01 would split M in one instruction, but it rejects F322BF16
// (aicore 169: fixpipe parameter invalid). The cast stays on each call.
// scale == 0 keeps a plain cast. A nonzero scale is applied in the fixpipe
// (fp32 multiply, then bf16), so the vector core does not scale o again.
__aicore__ inline void FixpipeHalfBf16(LocalTensor<bfloat16_t> ub, LocalTensor<float> l0C, int m, int n,
                                       float scale = 0.f)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m / 2);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = static_cast<uint32_t>(n);
    if (scale == 0.f) {
        p.quantPre = QuantMode_t::F322BF16;
    } else {
        p.quantPre = QuantMode_t::QF322BF16_PRE;
        union {
            float f;
            uint32_t u;
        } bits;
        bits.f = scale;
        p.deqScalar = static_cast<uint64_t>(bits.u);
    }
    p.dualDstCtl = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    p.subBlockId = false;
    Fixpipe<bfloat16_t, float, kFixUbNz2Nd>(ub, l0C, p);
    p.subBlockId = true;
    Fixpipe<bfloat16_t, float, kFixUbNz2Nd>(ub, l0C[16 * 16], p);
}

}  // namespace GdnFwd32

#endif
