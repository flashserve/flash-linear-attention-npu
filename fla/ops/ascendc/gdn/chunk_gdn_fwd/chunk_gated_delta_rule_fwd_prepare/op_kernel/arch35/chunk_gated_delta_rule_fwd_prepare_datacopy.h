/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Data movement: UB/L1/GM copies, NZ/ND, Fixpipe, Stage0 fills,
 * Stage1/6 bf16 ND→NZ, Stage3 leaf pack and L1 upload.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_DATACOPY_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_DATACOPY_H

#include <type_traits>

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_fwd_prepare_common.h"
#include "chunk_gated_delta_rule_fwd_prepare_vf.h"

namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;

// GM -> UB, any length. Uses DataCopyPad so tail 1-D (g/beta/rstd) need not
// be 32 B aligned. Aligned 2-D tiles (M*K bf16) also take this path.
template <typename T>
__aicore__ inline void CopyGmToUbElems(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t nElem)
{
    if (nElem == 0) {
        return;
    }
    DataCopyExtParams p{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(nElem * sizeof(T)),
        0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(dst, src, p, pad);
}

// UB -> GM, any length.
template <typename T>
__aicore__ inline void CopyUbToGmElems(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t nElem)
{
    if (nElem == 0) {
        return;
    }
    DataCopyExtParams p{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(nElem * sizeof(T)),
        0, 0, 0};
    DataCopyPad(dst, src, p);
}

// Fill I_vcs [32,64] and scatter idx {0,32}.
__aicore__ inline void FillVcsIdentity(AscendC::LocalTensor<float> ubVcsI,
                                       AscendC::LocalTensor<uint32_t> ubIdxB32)
{
    VcsIdentityVF(ubVcsI);
    AscendC::Duplicate(ubIdxB32, (uint32_t)0, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    ubIdxB32.SetValue(0, (uint32_t)0);
    ubIdxB32.SetValue(1, (uint32_t)kVcs32);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}

// Contiguous UB→L1. Zeros are identical in ND and cube-NZ, so no format convert.
// fp32 block is 8 elements; burst = n*n/8. (bf16 ub_to_l1 uses n*n/16.)
__aicore__ inline void UbToL1Fp32(AscendC::LocalTensor<float> l1Tensor,
                                  AscendC::LocalTensor<float> ubTensor, uint32_t n)
{
    AscendC::DataCopy(l1Tensor, ubTensor,
                      AscendC::DataCopyParams(1, n * n / 8, 0, 0));
}

// ND 64x64 fp32 -> L1 cube NZ C0=8. Eight 8-col copies (same idea as k'
// bf16 (64,1,7,0)). No UB TransDataTo5HD / Blk16ToBlk8.
__aicore__ inline void UbNd64ToL1Nz8(AscendC::LocalTensor<float> l1,
                                     AscendC::LocalTensor<float> nd)
{
    for (uint16_t fc = 0; fc < 8; ++fc) {
        AscendC::DataCopy(l1[static_cast<int32_t>(fc) * 8 * kChunk64],
                          nd[static_cast<int32_t>(fc) * 8],
                          AscendC::DataCopyParams(kChunk64, 1, 7, 0));
    }
}

// Packed 32x64 ND (two leaves side by side) -> one 32x32 in a 64x64 NZ
// quadrant. packed row is 64 so srcStride=7. rowQuad/colQuad in {0,1}.
__aicore__ inline void UbPackedLeafToL1(AscendC::LocalTensor<float> l1,
                                        AscendC::LocalTensor<float> packed,
                                        int32_t rowQuad, int32_t colQuad, int32_t srcCol)
{
    for (int32_t fj = 0; fj < 4; ++fj) {
        const int32_t fc = colQuad * 4 + fj;
        const int32_t fr = rowQuad * 2;
        const int32_t l1Off = (fc * static_cast<int32_t>(kNumMFracs64) + fr) * kFracLen8;
        const int32_t srcOff = srcCol + fj * 8;
        AscendC::DataCopy(l1[l1Off], packed[srcOff],
                          AscendC::DataCopyParams(static_cast<uint16_t>(kVcs32), 1, 7, 0));
    }
}

// L0C NZ -> one Vector UB as row-major ND. CFG_ROW_MAJOR enables NZ2ND on
// the Fixpipe path. TransformParams<ROW_MAJOR> is Nz2NdParams: one 64x64,
// ndNum=1 so src/dst NdStride are unused. dualDstCtl=0 writes the whole
// MxN to the UB selected by subBlockId (0=AIV0, 1=AIV1). dualDstCtl 0b01/0b10
// splits M or N across the two AIVs; it does not broadcast a full tile.
__aicore__ inline void FixpipeL0cToUbFp32Nd(AscendC::LocalTensor<float> ubTensor,
                                            AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize,
                                            uint8_t subBlk)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p;
    p.nSize = chunkSize;
    p.mSize = chunkSize;
    p.srcStride = chunkSize;
    p.dstStride = chunkSize;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.subBlockId = (subBlk != 0);
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_UB>(ubTensor, l0CTensor, p);
}

// Arch3510 L0C fp32 NZ -> GM ND (bf16 via F322BF16). Official scenario 2
// plus quantPre; ndNum=1. srcStride is aligned M (C0_Size units), never N.
template <typename OutDtype>
__aicore__ inline void FixpipeL0cToGmNd(AscendC::GlobalTensor<OutDtype> gmTensor,
                                        AscendC::LocalTensor<float> l0CTensor,
                                        uint32_t validRows, uint32_t curSize, uint32_t dstStride)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p;
    p.nSize = static_cast<uint16_t>(curSize);
    p.mSize = static_cast<uint16_t>(validRows);
    p.srcStride = kChunk64;
    p.dstStride = dstStride;
    p.quantPre = QuantMode_t::F322BF16;
    if constexpr (std::is_same_v<OutDtype, float>) {
        p.quantPre = QuantMode_t::NoQuant;
    } else if constexpr (std::is_same_v<OutDtype, half>) {
        p.quantPre = QuantMode_t::F322F16;
    }
    p.dualDstCtl = 0;
    p.subBlockId = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    AscendC::Fixpipe<OutDtype, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0CTensor, p);
}

// L0C fp32 NZ (C0=16 in CO1) -> GM cube NZ C0=8. Arch3510 isChannelSplit:
// nSize must be a multiple of 8; dstStride is in elements (adjacent Z).
// 64x64: srcStride=64 (C0_Size units), dstStride=64*8. Matches solve_tri.
__aicore__ inline void FixpipeL0cToGmNzCs(AscendC::GlobalTensor<float> gmTensor,
                                          AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
    p.nSize = chunkSize;
    p.mSize = chunkSize;
    p.srcStride = chunkSize;
    p.dstStride = chunkSize * 8;
    p.quantPre = QuantMode_t::NoQuant;
    p.isChannelSplit = true;
    AscendC::Fixpipe<float, float, CFG_NZ_L1>(gmTensor, l0CTensor, p);
}

// GM row-major ND -> L1 cube NZ. AIC MTE2, Nd2NzParams (GM -> A1):
// dstNzC0Stride / dstNzNStride are in C0_SIZE (32B).
template <typename T>
__aicore__ inline void CopyGmNdToL1Nz(AscendC::LocalTensor<T> l1Tensor,
                                      AscendC::GlobalTensor<T> gmTensor,
                                      uint32_t rows, uint32_t cols, uint32_t srcD = 0)
{
    AscendC::Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = cols;
    p.srcDValue = (srcD == 0) ? cols : srcD;
    p.srcNdMatrixStride = 0;
    p.dstNzC0Stride = static_cast<uint16_t>(rows);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmTensor, p);
}

// GM cube NZ C0=8 (Fixpipe isChannelSplit) -> L1, same layout. One burst,
// same params as UbToL1Fp32. n=64 is 512 blocks (16 KiB).
__aicore__ inline void CopyGmNzToL1Fp32(AscendC::LocalTensor<float> l1Tensor,
                                        AscendC::GlobalTensor<float> gmTensor, uint32_t n)
{
    AscendC::DataCopy(l1Tensor, gmTensor, AscendC::DataCopyParams(1, n * n / 8, 0, 0));
}

// Stage1 k' / Stage6 vb,kbg: ND row-major -> L1 cube NZ C0=16.
// Dest N-frac fc at fc*16*64, src C0 at fc*16, src row gap = cols/16-1.
// Caller owns V_MTE3 before and MTE3_V after.
template <typename T>
__aicore__ inline void UploadBf16NdToL1(LocalTensor<T> l1, LocalTensor<T> ubNd, uint32_t cols)
{
    const uint32_t nFrac = cols / 16;
    const uint16_t srcGap = static_cast<uint16_t>(nFrac - 1);
    for (uint32_t fc = 0; fc < nFrac; ++fc) {
        DataCopy(l1[static_cast<int32_t>(fc) * 16 * static_cast<int32_t>(kChunk64)],
                 ubNd[static_cast<int32_t>(fc * 16)], DataCopyParams(kChunk64, 1, srcGap, 0));
    }
}

// Packed -L00 | -L11 from ubLFull (VF already wrote -L). VCS consumes this as-is.
__aicore__ inline void PackDiagLeavesFromUb(LocalTensor<float> packed, LocalTensor<float> L)
{
    const uint16_t burst = 4;
    const uint16_t gap = 4;
    DataCopy(packed, L, DataCopyParams(32, burst, gap, gap));
    DataCopy(packed[32], L[32 * 64 + 32], DataCopyParams(32, burst, gap, gap));
}

// Packed VCS ND -> L1 NZ quadrants (Right=TL L00, Left=BR L11), and
// -L ND64 -> L1 NZ. 8-col DataCopy, no UB TransDataTo5HD.
// Off-diagonal leaves stay Stage0 zero.
__aicore__ inline void UploadDiagLeavesAndFullAToL1(LocalTensor<float> l1LeafRight, LocalTensor<float> l1LeafLeft,
                                                    LocalTensor<float> l1NegL, LocalTensor<float> packedNd32x64,
                                                    LocalTensor<float> negLNd64)
{
    UbPackedLeafToL1(l1LeafRight, packedNd32x64, 0, 0, 0);
    UbPackedLeafToL1(l1LeafLeft, packedNd32x64, 1, 1, static_cast<int32_t>(kVcs32));
    UbNd64ToL1Nz8(l1NegL, negLNd64);
}

// -L from kkt/g/β, pack diag leaves, VCS (I+Lii)^{-1} into ubResVcs.
__aicore__ inline void Stage3_ConstructLAndVcs(LocalTensor<float> kkt, LocalTensor<float> g, LocalTensor<float> beta,
                                               LocalTensor<float> mask, LocalTensor<float> ubL,
                                               LocalTensor<float> packed, LocalTensor<float> ubIVcs,
                                               LocalTensor<float> ubResVcs, LocalTensor<uint32_t> ubVcsIdx)
{
    NegLowerLVF(kkt, g, beta, mask, ubL);
    PackDiagLeavesFromUb(packed, ubL);
    DataCopy(ubResVcs, ubIVcs, static_cast<int32_t>(kVcsPackedElems32));
    MulReduceScatterVF32(ubResVcs, packed, ubResVcs, ubVcsIdx);
}

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif
