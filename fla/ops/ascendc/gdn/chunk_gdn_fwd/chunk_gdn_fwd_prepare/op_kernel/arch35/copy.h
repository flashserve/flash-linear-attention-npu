/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Data movement: UB/L1/GM copies, NZ/ND, Fixpipe, Stage0 resident fills.
 */

#ifndef CHUNK_GDN_FWD_PREPARE_COPY_H
#define CHUNK_GDN_FWD_PREPARE_COPY_H

#include <type_traits>

#include "kernel_operator.h"
#include "common.h"

namespace GdnStage {
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

template <typename T>
__aicore__ inline void TransposeB32(LocalTensor<T> dst, LocalTensor<T> src, uint32_t curTileALen)
{
    int32_t rRepeartTimes = 16 / 8;
    for (int32_t i = 0; i < rRepeartTimes; i++) {
        TransDataTo5HDParams params;
        LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
        LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];

        uint32_t aRepeartTimes = (curTileALen + static_cast<uint32_t>(DATA_BLOCK_COUNT) - 1) /
                                 static_cast<uint32_t>(DATA_BLOCK_COUNT);
        params.repeatTimes = aRepeartTimes;
        params.srcRepStride = aRepeartTimes == 1 ? 0 : CONST_TWO;
        params.dstRepStride = aRepeartTimes == 1 ? 0 : DATA_BLOCK_COUNT * rRepeartTimes;
        for (int32_t j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
            uint32_t offset = DATA_BLOCK_COUNT_HALF * curTileALen * i + curTileALen * j;
            srcLocalList[j] = src[offset];
            srcLocalList[j + DATA_BLOCK_COUNT_HALF] = src[offset + DATA_BLOCK_COUNT_HALF];
        }
        for (int32_t j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
            uint32_t offset = DATA_BLOCK_COUNT_HALF * i + DATA_BLOCK_COUNT_HALF * rRepeartTimes * j;
            dstLocalList[j * CONST_TWO] = dst[offset];
            dstLocalList[j * CONST_TWO + 1] = dst[offset + DATA_BLOCK_COUNT_HALF * DATA_BLOCK_COUNT_HALF * rRepeartTimes];
        }

        AscendC::TransDataTo5HD(dstLocalList, srcLocalList, params);
    }
}

template <typename T>
__aicore__ inline void NzFp32Blk16ToBlk8(LocalTensor<T> dst, LocalTensor<T> src, uint32_t n)
{
    const uint32_t numFracs = n / 16;
    for (uint32_t i = 0; i < numFracs; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            AscendC::DataCopy(dst[(i * 2 + j) * 8 * n], src[i * 16 * n + j * 8],
                              AscendC::DataCopyParams(static_cast<uint16_t>(n), 1, 1, 0));
        }
    }
}

namespace Prepare {

// I_vcs 32x64 ND + scatter idx {0,32} on every AIV. Cube I_64 / leaf zeros
// are painted ND and 8-col copied in Stage0_GenerateResidentAux.
__aicore__ inline void Stage0_GenIdentity(AscendC::LocalTensor<float> ubVcsI,
                                          AscendC::LocalTensor<uint32_t> ubIdxB32)
{
    Stage0_PaintVcsIdentity(ubVcsI);
    AscendC::Duplicate(ubIdxB32, (uint32_t)0, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    ubIdxB32.SetValue(0, (uint32_t)0);
    ubIdxB32.SetValue(1, (uint32_t)kVcs32);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}

} // namespace Prepare

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

__aicore__ inline void UbToL1Fp32Nz(AscendC::LocalTensor<float> l1Tensor,
                                    AscendC::LocalTensor<float> ubTensor, uint32_t n)
{
    // 64x64 fp32 is 512 blocks. Fused MIX MTE3 has dropped a single 512-block
    // burst at ~8KB (left NZ half zeros, right half NaN). Split into 128-block
    // chunks (4KB), under the common 255-block cap.
    const uint16_t blk128 = 128;
    const int32_t elems128 = 1024;
    const int32_t nCopy = static_cast<int32_t>((n * n) / elems128);
    for (int32_t i = 0; i < nCopy; ++i) {
        AscendC::DataCopy(l1Tensor[i * elems128], ubTensor[i * elems128],
                          AscendC::DataCopyParams(1, blk128, 0, 0));
    }
}

template <typename T>
__aicore__ inline void UbToL1Elems(AscendC::LocalTensor<T> l1Tensor,
                                   AscendC::LocalTensor<T> ubTensor, uint32_t elems)
{
    constexpr uint32_t kBlk = 32 / sizeof(T);
    const uint32_t nBlk = elems / kBlk;
    constexpr uint16_t kMaxBlk = 128;
    uint32_t done = 0;
    while (done < nBlk) {
        uint16_t thisBlk = kMaxBlk;
        if (done + kMaxBlk > nBlk) {
            thisBlk = static_cast<uint16_t>(nBlk - done);
        }
        AscendC::DataCopy(l1Tensor[done * kBlk], ubTensor[done * kBlk],
                          AscendC::DataCopyParams(1, thisBlk, 0, 0));
        done += thisBlk;
    }
}

__aicore__ inline void FixpipeL0cToUbFp32Nz(AscendC::LocalTensor<float> ubTensor,
                                            AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize,
                                            uint8_t dualDstCtl = 0)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
    p.nSize = chunkSize;
    p.mSize = chunkSize;
    p.srcStride = chunkSize;
    p.dstStride = chunkSize * 16;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = dualDstCtl;
    p.subBlockId = 0;
    AscendC::Fixpipe<float, float, CFG_NZ_UB>(ubTensor, l0CTensor, p);
}

// NZ Fixpipe to one Vector, selected by subBlk (0=AIV0, 1=AIV1).
__aicore__ inline void FixpipeL0cToUbFp32NzAiv(AscendC::LocalTensor<float> ubTensor,
                                               AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize,
                                               uint8_t subBlk)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
    p.nSize = chunkSize;
    p.mSize = chunkSize;
    p.srcStride = chunkSize;
    p.dstStride = chunkSize * 16;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.subBlockId = (subBlk != 0);
    AscendC::Fixpipe<float, float, CFG_NZ_UB>(ubTensor, l0CTensor, p);
}

// L0C NZ -> one Vector UB as row-major ND. CFG_ROW_MAJOR enables NZ2ND on
// the Fixpipe path. TransformParams<ROW_MAJOR> is Nz2NdParams: one 64x64,
// ndNum=1 so src/dst NdStride are unused. dualDstCtl=0 writes the whole
// MxN to the UB selected by subBlockId (0=AIV0, 1=AIV1).
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

// Official fixpipe_l0c2gm scenario 2 (dav-3510 CFG_ROW_MAJOR):
//   srcStride = CeilAlign(baseM, CUBE_BLOCK=16), nSize = N, dstStride = N.
// Stage5 64x64 hid a srcStride=N bug because M==N. This op's L0C tiles
// are always BT=64 rows (MMAD m=chunkSize), so srcStride is kChunk64.

// solve_tri FixpipeL0cToGM: L0C -> GM bf16 ND. Proven on this SoC for (I+L)^{-1}.
template <typename OutDtype>
__aicore__ inline void FixpipeL0cToGmNdV220(AscendC::GlobalTensor<OutDtype> gmTensor,
                                            AscendC::LocalTensor<float> l0CTensor,
                                            uint32_t validRows, uint32_t curSize, uint32_t dstStride)
{
    auto p = AscendC::FixpipeParamsV220(curSize, validRows, kChunk64, dstStride, false);
    p.quantPre = QuantMode_t::F322BF16;
    if constexpr (std::is_same_v<OutDtype, float>) {
        p.quantPre = QuantMode_t::NoQuant;
    } else if constexpr (std::is_same_v<OutDtype, half>) {
        p.quantPre = QuantMode_t::F322F16;
    }
    AscendC::Fixpipe<OutDtype, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0CTensor, p);
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

// L0C fp32 NZ -> L1 bf16 (or fp16) cube NZ C0=16. No NZ2ND, no channel-split.
// dstStride unit is elements; 64x64 uses chunkSize*16. Matches solve_tri FixpipeL0cToL1.
template <typename OutDtype>
__aicore__ inline void FixpipeL0cToL1Bf16Nz(AscendC::LocalTensor<OutDtype> l1Tensor,
                                            AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
    p.nSize = chunkSize;
    p.mSize = chunkSize;
    p.srcStride = chunkSize;
    p.dstStride = chunkSize * 16;
    p.quantPre = QuantMode_t::F322BF16;
    p.isChannelSplit = false;
    if constexpr (std::is_same_v<OutDtype, half>) {
        p.quantPre = QuantMode_t::F322F16;
    } else if constexpr (std::is_same_v<OutDtype, float>) {
        p.quantPre = QuantMode_t::NoQuant;
    }
    AscendC::Fixpipe<OutDtype, float, CFG_NZ_L1>(l1Tensor, l0CTensor, p);
}

template <typename T>
__aicore__ inline void CopyGmToL1Elems(AscendC::LocalTensor<T> l1Tensor,
                                       AscendC::GlobalTensor<T> gmTensor, uint32_t elems)
{
    constexpr uint32_t kBlk = 32 / sizeof(T);
    const uint32_t nBlk = elems / kBlk;
    constexpr uint16_t kMaxBlk = 128;
    uint32_t done = 0;
    while (done < nBlk) {
        uint16_t thisBlk = kMaxBlk;
        if (done + kMaxBlk > nBlk) {
            thisBlk = static_cast<uint16_t>(nBlk - done);
        }
        AscendC::DataCopy(l1Tensor[done * kBlk], gmTensor[done * kBlk],
                          AscendC::DataCopyParams(1, thisBlk, 0, 0));
        done += thisBlk;
    }
}

// GM row-major ND -> L1 cube NZ. AIC MTE2, Nd2NzParams (GM -> A1):
// https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_00127.html
// dstNzC0Stride / dstNzNStride are in C0_SIZE (32B). Matches solve_tri
// cube PrepareConstants / cube_paths CopyGmNdToL1NzBf16.
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

// Cube NZ C0=8 (channel-split 64x64) -> row-major ND. Each N-fractal is 64x8.
__aicore__ inline void NzC08ToNd64Fp32(AscendC::LocalTensor<float> nd,
                                       AscendC::LocalTensor<float> nz)
{
    for (uint32_t fc = 0; fc < 8; ++fc) {
        AscendC::DataCopy(nd[fc * 8], nz[fc * 8 * kChunk64],
                          AscendC::DataCopyParams(kChunk64, 1, 0, 7));
    }
}

// NZ C0=16 (Fixpipe CFG_NZ_UB) -> row-major ND. 64x64 fp32.
__aicore__ inline void NzC016ToNd64Fp32(AscendC::LocalTensor<float> nd,
                                        AscendC::LocalTensor<float> nz)
{
    for (uint32_t fc = 0; fc < 4; ++fc) {
        AscendC::DataCopy(nd[fc * 16], nz[fc * 16 * kChunk64],
                          AscendC::DataCopyParams(kChunk64, 2, 0, 6));
    }
}

// Inverse of NzC016ToNd64Fp32. Avoids UB Nd2NzParams (illegal GM decode on this op).
__aicore__ inline void Nd64ToNzC016Fp32(AscendC::LocalTensor<float> nz,
                                        AscendC::LocalTensor<float> nd)
{
    for (uint32_t fc = 0; fc < 4; ++fc) {
        AscendC::DataCopy(nz[fc * 16 * kChunk64], nd[fc * 16],
                          AscendC::DataCopyParams(kChunk64, 2, 6, 0));
    }
}

// 64x64 fp32 ND -> L1 cube NZ (C0=8). Manual ND->C0=16, then Blk16->Blk8.
__aicore__ inline void UploadFp32Nd64ToL1Nz8(AscendC::LocalTensor<float> l1,
                                             AscendC::LocalTensor<float> ubNd,
                                             AscendC::LocalTensor<float> ubNz16,
                                             AscendC::LocalTensor<float> ubNz8)
{
    Nd64ToNzC016Fp32(ubNz16, ubNd);
    NzFp32Blk16ToBlk8(ubNz8, ubNz16, kChunk64);
    SetFlag<AscendC::HardEvent::V_MTE3>(5);
    WaitFlag<AscendC::HardEvent::V_MTE3>(5);
    UbToL1Fp32Nz(l1, ubNz8, kChunk64);
    SetFlag<AscendC::HardEvent::MTE3_V>(5);
    WaitFlag<AscendC::HardEvent::MTE3_V>(5);
}

// row-major ND 64 x cols bf16 -> NZ C0=16. cols is 64, 128, or 256.
__aicore__ inline void Nd64ToNzC016Bf16Wide(AscendC::LocalTensor<bfloat16_t> nz,
                                            AscendC::LocalTensor<bfloat16_t> nd,
                                            uint32_t cols)
{
    if (cols == 128) {
        for (uint32_t fc = 0; fc < 8; ++fc) {
            AscendC::DataCopy(nz[fc * 16 * kChunk64], nd[fc * 16],
                              AscendC::DataCopyParams(kChunk64, 1, 7, 0));
        }
    } else if (cols == 256) {
        for (uint32_t fc = 0; fc < 16; ++fc) {
            AscendC::DataCopy(nz[fc * 16 * kChunk64], nd[fc * 16],
                              AscendC::DataCopyParams(kChunk64, 1, 15, 0));
        }
    } else {
        for (uint32_t fc = 0; fc < 4; ++fc) {
            AscendC::DataCopy(nz[fc * 16 * kChunk64], nd[fc * 16],
                              AscendC::DataCopyParams(kChunk64, 1, 3, 0));
        }
    }
}

// Place one 32x32 cube-NZ leaf (C0=8) into a 64x64 L1 slot at quadrant (rowQuad, colQuad).
// Same addressing as solve_tri AivScatterLeavesToL1. rowQuad/colQuad in {0,1}.
__aicore__ inline void ScatterNz8Leaf32ToL1(AscendC::LocalTensor<float> l1,
                                            AscendC::LocalTensor<float> leafNz8,
                                            int32_t rowQuad, int32_t colQuad)
{
    constexpr int32_t mFracsLeaf = static_cast<int32_t>(kVcs32 / 16);
    constexpr int32_t nFracsLeaf = static_cast<int32_t>(kVcs32 / 8);
    for (int32_t fi = 0; fi < mFracsLeaf; ++fi) {
        for (int32_t fj = 0; fj < nFracsLeaf; ++fj) {
            const int32_t fr = rowQuad * mFracsLeaf + fi;
            const int32_t fc = colQuad * nFracsLeaf + fj;
            const int32_t l1Off = (fc * kNumMFracs64 + fr) * kFracLen8;
            const int32_t srcOff = (fj * mFracsLeaf + fi) * kFracLen8;
            AscendC::DataCopy(l1[l1Off], leafNz8[srcOff],
                              AscendC::DataCopyParams(1, static_cast<uint16_t>(kFracLen8 / 8), 0, 0));
        }
    }
}

// TransposeB32Vcs32 uses HardEvent 2, which aliases fused CrossCore 0x2
// (AIV->AIC k' handshake). Use a free event so TransDataTo5HD cannot skip.
template <uint8_t Evt>
__aicore__ inline void TransposeB32Vcs32Evt(LocalTensor<float> dstTwoNz32,
                                            LocalTensor<float> srcNd32x64,
                                            LocalTensor<float> tmp64x32)
{
    constexpr uint32_t kBlk = static_cast<uint32_t>(kFracLen);
    TransposeB32(tmp64x32, srcNd32x64, kVcsPack32);
    TransposeB32(tmp64x32[16 * kVcsPack32], srcNd32x64[16 * kVcsPack32], kVcsPack32);
    SetFlag<AscendC::HardEvent::V_MTE3>(Evt);
    WaitFlag<AscendC::HardEvent::V_MTE3>(Evt);
    AscendC::DataCopy(dstTwoNz32[0], tmp64x32[0], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[kBlk], tmp64x32[kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[2 * kBlk], tmp64x32[4 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[3 * kBlk], tmp64x32[5 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[4 * kBlk], tmp64x32[2 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[5 * kBlk], tmp64x32[3 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[6 * kBlk], tmp64x32[6 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[7 * kBlk], tmp64x32[7 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    SetFlag<AscendC::HardEvent::MTE3_V>(Evt);
    WaitFlag<AscendC::HardEvent::MTE3_V>(Evt);
}

// 32x64 ND (two 32x32 leaves side by side) -> two C0=8 NZ leaves in nz8Two.
// Event 4 does not alias fused CrossCore 0x0/0x2/0x6/0x8.
__aicore__ inline void Nd32x64ToTwoNz8Leaves(AscendC::LocalTensor<float> nz8Two,
                                             AscendC::LocalTensor<float> nd32x64,
                                             AscendC::LocalTensor<float> dstTwoNz16,
                                             AscendC::LocalTensor<float> tmpNz16)
{
    TransposeB32Vcs32Evt<5>(dstTwoNz16, nd32x64, tmpNz16);
    NzFp32Blk16ToBlk8(nz8Two, dstTwoNz16, kVcs32);
    NzFp32Blk16ToBlk8(nz8Two[kVcs32NzElems], dstTwoNz16[kVcs32NzElems], kVcs32);
    SetFlag<AscendC::HardEvent::V_MTE3>(5);
    WaitFlag<AscendC::HardEvent::V_MTE3>(5);
}

} // namespace GdnStage

#endif
