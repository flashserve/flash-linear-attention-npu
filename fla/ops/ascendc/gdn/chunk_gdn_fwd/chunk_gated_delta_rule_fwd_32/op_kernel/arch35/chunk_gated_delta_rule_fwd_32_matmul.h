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
                                      bool transposeA = false, bool waitReady = true)
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
    if (waitReady) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    }
}

// One K-half of H. L1 is a 128-row NZ image; srcStride stays 8 so the next
// column panel is 128 rows away. rowFrac is 0 or 4 (rows 0 or 64).
__aicore__ inline void LoadL1NzHHalf(LocalTensor<bfloat16_t> l1H, LocalTensor<bfloat16_t> l0B, uint32_t rowFrac)
{
    AscendC::LoadData2DParamsV2 loadB;
    loadB.mStartPosition = rowFrac;
    loadB.kStartPosition = 0;
    loadB.mStep = 4;
    loadB.kStep = 8;
    loadB.srcStride = 8;
    loadB.dstStride = 8;
    loadB.ifTranspose = true;
    loadB.sid = 0;
    AscendC::LoadData(l0B, l1H, loadB);
}

// Reload only L0B. Caller already waited M_MTE1. The matching L0A load and
// MTE1_M flag come later, so this load can sit under the previous mad.
__aicore__ inline void LoadL1NzToL0B(LocalTensor<bfloat16_t> l1B, LocalTensor<bfloat16_t> l0B, int32_t n, int32_t k,
                                     bool transposeB)
{
    constexpr uint32_t c0 = 16;
    const uint16_t bMFracs = transposeB ? static_cast<uint16_t>(k / 16) : static_cast<uint16_t>(n / 16);
    const uint16_t bKFracs = transposeB ? static_cast<uint16_t>(n / c0) : static_cast<uint16_t>(k / c0);
    const uint16_t bDst = transposeB ? static_cast<uint16_t>(n / 16) : bMFracs;
    AscendC::LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, bMFracs, bKFracs, bMFracs, bDst, transposeB);
    AscendC::LoadData(l0B, l1B, loadB);
}

// Reload only L0A. L0B stays, so the four D slices share one v_new_g load.
// Caller waited M_MTE1 and will set/wait MTE1_M around the following mad.
__aicore__ inline void LoadL1NzToL0A(LocalTensor<bfloat16_t> l1A, LocalTensor<bfloat16_t> l0A, int32_t m, int32_t k,
                                     bool transposeA)
{
    constexpr uint32_t c0 = 16;
    const uint16_t mFracs = static_cast<uint16_t>(m / 16);
    const uint16_t aKFracs = static_cast<uint16_t>(k / c0);
    AscendC::LoadData2DParamsV2 loadA;
    if (transposeA) {
        FillLoad2D(loadA, aKFracs, mFracs, aKFracs, mFracs, true);
    } else {
        FillLoad2D(loadA, mFracs, aKFracs, mFracs, mFracs, false);
    }
    AscendC::LoadData(l0A, l1A, loadA);
}

// waitFix is false when this tile sits above a live L0C result. The caller
// still has to wait that FIX_M once before setting it again.
__aicore__ inline void MmadL0ToL0C(LocalTensor<bfloat16_t> l0A, LocalTensor<bfloat16_t> l0B, LocalTensor<float> l0C,
                                   int32_t m, int32_t n, int32_t k, bool initC, uint8_t evt = 0,
                                   bool waitFix = true)
{
    if (initC && waitFix) {
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

// Same ND rows, placed at row0 of one NZ whose panel height is fullRows.
// H is one [128,128] so w@H and q@H are a single K=128 load, not four K=32 tiles.
__aicore__ inline void CopyNdRowsToFullNz(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> nd, uint32_t row0,
                                         uint32_t rows, uint32_t cols, uint32_t fullRows)
{
    const uint32_t nFrac = cols / 16;
    const uint16_t srcGap = static_cast<uint16_t>(nFrac - 1);
    for (uint32_t fc = 0; fc < nFrac; ++fc) {
        DataCopy(l1[fc * fullRows * 16 + row0 * 16], nd[fc * 16],
                 DataCopyParams(static_cast<uint16_t>(rows), 1, srcGap, 0));
    }
}

// Tight NZ already sitting in UB (produced by the VF) -> one UB->L1 copy.
// Caller orders the vector store with V_MTE3 before this, and pairs MTE3_V(7)
// around the scratch this copy reads.
__aicore__ inline void CopyNzToL1(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> nz, uint32_t rows, uint32_t cols)
{
    const uint32_t bytes = rows * cols * sizeof(bfloat16_t);
    DataCopy(l1, nz, DataCopyParams(1, static_cast<uint16_t>(bytes / 32), 0, 0));
}

// ND [rows, cols] -> tight NZ in scratch, then one UB->L1 copy.
// 128-wide matrices that the producer already stored as NZ use CopyNzToL1.
// 32-wide A/attn stay a 2-burst column copy: the vector pack is more work
// than two DataCopies. Wider callers still pack here (H, q_exp).
__aicore__ inline void UploadNdToL1(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> ubNd,
                                    LocalTensor<bfloat16_t> scratch, uint32_t rows, uint32_t cols)
{
    using namespace AscendC::MicroAPI;
    // Previous copy may still be reading this scratch.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
    if (cols <= 32) {
        const uint32_t nFrac = cols / 16;
        const uint16_t srcGap = nFrac == 0 ? 0 : static_cast<uint16_t>(nFrac - 1);
        for (uint32_t fc = 0; fc < nFrac; ++fc) {
            DataCopy(l1[fc * 16 * rows], ubNd[fc * 16],
                     DataCopyParams(static_cast<uint16_t>(rows), 1, srcGap, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
        return;
    }
    __ubuf__ bfloat16_t *src = (__ubuf__ bfloat16_t *)ubNd.GetPhyAddr();
    __ubuf__ bfloat16_t *dst = (__ubuf__ bfloat16_t *)scratch.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t n = cols;
        MaskReg preg = UpdateMask<bfloat16_t>(n);
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            RegTensor<bfloat16_t> row;
            LoadAlign(row, src + static_cast<uint32_t>(r) * cols);
            // Block j of this row lands at row r of panel j. Stride is the panel
            // height in 32B blocks, so the panels sit back to back.
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(dst + static_cast<uint32_t>(r) * 16, row, rows,
                                                                  preg);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
    CopyNzToL1(l1, scratch, rows, cols);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
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

// L0C fp32 -> UB bf16 row-major. u and P are consumed only as bf16, so the
// vector subtract never widens them back to two fp32 registers.
__aicore__ inline void FixpipeBf16Ub(LocalTensor<bfloat16_t> ub, LocalTensor<float> l0C, int m, int n,
                                     uint8_t subBlk = 0)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p;
    p.nSize = static_cast<uint16_t>(n);
    p.mSize = static_cast<uint16_t>(m);
    p.srcStride = static_cast<uint16_t>(m);
    p.dstStride = static_cast<uint32_t>(n);
    p.quantPre = QuantMode_t::F322BF16;
    p.dualDstCtl = 0;
    p.subBlockId = (subBlk != 0);
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    AscendC::Fixpipe<bfloat16_t, float, kFixUbNz2Nd>(ub, l0C, p);
}

// L0C fp32 -> UB bf16 NZ. One column panel is [m, 16] contiguous, same image
// CopyNdRowsToFullNz writes, so the state saxpy can stay elementwise.
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

}  // namespace GdnFwd32

#endif
