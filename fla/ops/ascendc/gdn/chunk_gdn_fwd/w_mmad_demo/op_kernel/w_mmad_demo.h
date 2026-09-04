/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Isolated MIX demo of fused Stage3 leaf upload + Stage4/5 MBH:
 *   input  L  [64,64] fp32 strictly lower (host-built)
 *   output A  [64,64] fp32 ~ (I+L)^{-1}
 *
 * Leaves stay ND on UB (no TransDataTo5HD). UB->L1 is 8-col DataCopy to
 * cube NZ C0=8, scattered into 64x64 blkdiag slots. Because that NZ is
 * NZ(inv.T) rather than the fused 5HD layout, LoadData ifTranspose is
 * flipped whenever A or B is a leaf tile. I / -L / Y keep the fused
 * convention (A false, B true). See:
 * https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/03_basic_api/03_matrix_compute/load_data_2dv2_l12l0/README.md
 *
 *   Y = I + LeafLeft @ (-L)
 *   A = LeafLeft + Y @ LeafRight
 * LeafRight = blkdiag(inv00, 0), LeafLeft = blkdiag(0, inv11).
 */

#ifndef W_MMAD_DEMO_H
#define W_MMAD_DEMO_H

#include "kernel_operator.h"

namespace WMmadDemoOp {
using namespace AscendC;

constexpr int32_t kN = 64;
constexpr int32_t kLeaf = 32;
constexpr uint16_t kNumMFracs64 = 4;
constexpr int32_t kFracLen8 = 16 * 8;
constexpr int32_t kElems64 = kN * kN;
constexpr int32_t kPackedElems = kLeaf * kN;
constexpr uint16_t kFlagAivDone = 0;
constexpr uint32_t kUbSize = 248 * 1024;
constexpr uint32_t kL1I = 0;
constexpr uint32_t kL1NegL = 16 * 1024;
constexpr uint32_t kL1LeafR = 32 * 1024;
constexpr uint32_t kL1LeafL = 48 * 1024;
constexpr uint32_t kL1Y = 64 * 1024;

constexpr FixpipeConfig kCfgNzGm = {CO2Layout::NZ, false};
constexpr FixpipeConfig kCfgNdGm = {CO2Layout::ROW_MAJOR, false};

struct OnChipBuffer {
    template <typename T>
    using Tensor = LocalTensor<T>;

    __aicore__ inline OnChipBuffer()
    {
        buffer_[0] = Tensor<uint8_t>(TPosition::VECIN, 0, kUbSize);
        buffer_[1] = Tensor<uint8_t>(TPosition::A1, 0, 512 * 1024);
        buffer_[2] = Tensor<uint8_t>(TPosition::A2, 0, 64 * 1024);
        buffer_[3] = Tensor<uint8_t>(TPosition::B2, 0, 64 * 1024);
        buffer_[4] = Tensor<uint8_t>(TPosition::CO1, 0, 256 * 1024);
    }

    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> Ub(uint32_t off) const
    {
        return buffer_[0][off].template ReinterpretCast<Dtype>();
    }
    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L1(uint32_t off) const
    {
        return buffer_[1][off].template ReinterpretCast<Dtype>();
    }
    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L0A(uint32_t off) const
    {
        return buffer_[2][off].template ReinterpretCast<Dtype>();
    }
    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L0B(uint32_t off) const
    {
        return buffer_[3][off].template ReinterpretCast<Dtype>();
    }
    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L0C(uint32_t off) const
    {
        return buffer_[4][off].template ReinterpretCast<Dtype>();
    }

    LocalTensor<uint8_t> buffer_[5];
};

__aicore__ inline void FillLoad2D(LoadData2DParamsV2 &p, uint16_t mStep, uint16_t kStep,
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

// 64x64 fp32: fractal [16,8], fractalNum=2. Square m=k=n=64 so mStep/kStep
// stay 4/8 for both trans and non-trans (Load2Dv2 README B32 5.2.3 / 5.4.3).
// Fused default: A false, B true. Leaf tiles flip that bit.
__aicore__ inline void MatmulFp32(LocalTensor<float> l1A, LocalTensor<float> l1B,
                                      LocalTensor<float> l0A, LocalTensor<float> l0B,
                                      LocalTensor<float> l0C, bool initC,
                                      bool aIsLeaf, bool bIsLeaf)
{
    const bool transA = aIsLeaf;
    const bool transB = !bIsLeaf;
    LoadData2DParamsV2 loadA;
    FillLoad2D(loadA, kNumMFracs64, static_cast<uint16_t>(kN / 8),
               kNumMFracs64, kNumMFracs64, transA);
    LoadData(l0A, l1A, loadA);
    LoadData2DParamsV2 loadB;
    FillLoad2D(loadB, kNumMFracs64, static_cast<uint16_t>(kN / 8),
               kNumMFracs64, kNumMFracs64, transB);
    LoadData(l0B, l1B, loadB);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    MmadParams mmad;
    mmad.m = kN;
    mmad.n = kN;
    mmad.k = kN;
    mmad.cmatrixInitVal = initC;
    mmad.cmatrixSource = false;
    mmad.unitFlag = 0;
    Mmad(l0C, l0A, l0B, mmad);
}

__aicore__ inline void UbToL1Fp32Split(LocalTensor<float> l1, LocalTensor<float> ub)
{
    const uint16_t blk128 = 128;
    const int32_t elems128 = 1024;
    const int32_t nCopy = kElems64 / elems128;
    for (int32_t i = 0; i < nCopy; ++i) {
        DataCopy(l1[i * elems128], ub[i * elems128], DataCopyParams(1, blk128, 0, 0));
    }
}

__aicore__ inline void CopyGmNzToL1(LocalTensor<float> l1, GlobalTensor<float> gm)
{
    const uint16_t blk128 = 128;
    const int32_t elems128 = 1024;
    const int32_t nCopy = kElems64 / elems128;
    for (int32_t i = 0; i < nCopy; ++i) {
        DataCopy(l1[i * elems128], gm[i * elems128], DataCopyParams(1, blk128, 0, 0));
    }
}

// ND 64x64 -> L1 cube NZ C0=8. Same idea as k' (64,1,7,0) but 8 fp32 cols.
__aicore__ inline void UbNd64ToL1Nz8(LocalTensor<float> l1, LocalTensor<float> nd)
{
    for (uint16_t fc = 0; fc < 8; ++fc) {
        DataCopy(l1[static_cast<int32_t>(fc) * 8 * kN], nd[static_cast<int32_t>(fc) * 8],
                 DataCopyParams(kN, 1, 7, 0));
    }
}

// 32x32 ND leaf inside packed 32x64 -> one quadrant of a 64x64 NZ slot.
__aicore__ inline void UbPackedLeafToL1(LocalTensor<float> l1, LocalTensor<float> packed,
                                        int32_t rowQuad, int32_t colQuad, int32_t srcCol)
{
    for (int32_t fj = 0; fj < 4; ++fj) {
        const int32_t fc = colQuad * 4 + fj;
        const int32_t fr = rowQuad * 2;
        const int32_t l1Off = (fc * kNumMFracs64 + fr) * kFracLen8;
        const int32_t srcOff = srcCol + fj * 8;
        DataCopy(l1[l1Off], packed[srcOff], DataCopyParams(kLeaf, 1, 7, 0));
    }
}

__aicore__ inline void PackDiagLeaves(LocalTensor<float> packed, LocalTensor<float> L)
{
    DataCopy(packed, L, DataCopyParams(32, 4, 4, 4));
    DataCopy(packed[32], L[32 * 64 + 32], DataCopyParams(32, 4, 4, 4));
}

__aicore__ inline void PaintIdentity(LocalTensor<float> ub)
{
    Duplicate(ub, 0.0f, kElems64);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    for (int32_t i = 0; i < kN; ++i) {
        ub.SetValue(i * kN + i, 1.0f);
    }
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
}

__aicore__ inline void PaintVcsI(LocalTensor<float> ub)
{
    Duplicate(ub, 0.0f, kPackedElems);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    for (int32_t i = 0; i < kLeaf; ++i) {
        ub.SetValue(i * kN + i, 1.0f);
        ub.SetValue(i * kN + i + kLeaf, 1.0f);
    }
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
}

template <typename T, typename U>
__simd_vf__ inline void MulReduceScatterVF32(__ubuf__ T *dstAddr, __ubuf__ T *src0Addr, __ubuf__ T *src1Addr,
                                             __ubuf__ U *idxAddr, uint32_t scatterCount, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1a;
    AscendC::Reg::RegTensor<T> srcReg1b;
    AscendC::Reg::RegTensor<T> mulA;
    AscendC::Reg::RegTensor<T> mulB;
    AscendC::Reg::RegTensor<T> blkA;
    AscendC::Reg::RegTensor<T> blkB;
    AscendC::Reg::RegTensor<T> pairA;
    AscendC::Reg::RegTensor<T> pairB;
    AscendC::Reg::RegTensor<T> pair2A;
    AscendC::Reg::RegTensor<T> pair2B;
    AscendC::Reg::RegTensor<U> idxBase;
    AscendC::Reg::RegTensor<U> scatterIdxReg;

    uint32_t maskCount = oneRepeatSize;
    uint32_t pairMaskCount = scatterCount * 2;
    AscendC::Reg::MaskReg inputMask;
    AscendC::Reg::MaskReg pairMask;
    AscendC::Reg::MaskReg scatterMask;
    inputMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(maskCount);
    pairMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(pairMaskCount);
    scatterMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scatterCount);
    AscendC::Reg::LoadAlign(idxBase, idxAddr);
    for (uint16_t iterIdx = 1; iterIdx < 32; iterIdx++) {
        AscendC::Reg::Adds(scatterIdxReg, idxBase, (uint32_t)iterIdx, scatterMask);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr + iterIdx * oneRepeatSize);
        const uint16_t nPair = static_cast<uint16_t>(iterIdx >> 1);
        const uint16_t nTail = static_cast<uint16_t>(iterIdx & 1);
        for (uint16_t p = 0; p < nPair; p++) {
            const uint32_t i0 = static_cast<uint32_t>(p) * 2;
            const uint32_t i1 = i0 + 1;
            AscendC::Reg::LoadAlign(srcReg1a, src1Addr + i0 * oneRepeatSize);
            AscendC::Reg::LoadAlign(srcReg1b, src1Addr + i1 * oneRepeatSize);
            AscendC::Reg::Mul(mulA, srcReg0, srcReg1a, inputMask);
            AscendC::Reg::Mul(mulB, srcReg0, srcReg1b, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(blkA, mulA, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(blkB, mulB, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pairA, blkA, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pairB, blkB, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pair2A, pairA, pairMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pair2B, pairB, pairMask);
            AscendC::Reg::Scatter(dstAddr + i0 * oneRepeatSize, pair2A, scatterIdxReg, scatterMask);
            AscendC::Reg::Scatter(dstAddr + i1 * oneRepeatSize, pair2B, scatterIdxReg, scatterMask);
        }
        for (uint16_t t = 0; t < nTail; t++) {
            const uint32_t i = static_cast<uint32_t>(nPair) * 2;
            AscendC::Reg::LoadAlign(srcReg1a, src1Addr + i * oneRepeatSize);
            AscendC::Reg::Mul(mulA, srcReg0, srcReg1a, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(blkA, mulA, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pairA, blkA, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(pair2A, pairA, pairMask);
            AscendC::Reg::Scatter(dstAddr + i * oneRepeatSize, pair2A, scatterIdxReg, scatterMask);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

__aicore__ inline void RunVcs(LocalTensor<float> packed, LocalTensor<float> vcs,
                              LocalTensor<uint32_t> idx)
{
    __ubuf__ float *dstAddr = (__ubuf__ float *)vcs.GetPhyAddr();
    __ubuf__ float *src0Addr = (__ubuf__ float *)packed.GetPhyAddr();
    __ubuf__ float *src1Addr = dstAddr;
    __ubuf__ uint32_t *idxAddr = (__ubuf__ uint32_t *)idx.GetPhyAddr();
    MulReduceScatterVF32(dstAddr, src0Addr, src1Addr, idxAddr, 2u, 64u);
}

__aicore__ inline void FixpipeYNzCs(GlobalTensor<float> gm, LocalTensor<float> l0C)
{
    FixpipeParamsArch3510<CO2Layout::NZ> p;
    p.nSize = kN;
    p.mSize = kN;
    p.srcStride = kN;
    p.dstStride = kN * 8;
    p.quantPre = QuantMode_t::NoQuant;
    p.isChannelSplit = true;
    Fixpipe<float, float, kCfgNzGm>(gm, l0C, p);
}

__aicore__ inline void FixpipeInvNd(GlobalTensor<float> gm, LocalTensor<float> l0C)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    p.nSize = kN;
    p.mSize = kN;
    p.srcStride = kN;
    p.dstStride = kN;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0;
    p.subBlockId = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, kCfgNdGm>(gm, l0C, p);
}

class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR pre, GM_ADDR c, GM_ADDR workspace)
    {
        (void)b;
        (void)pre;
        gmL.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        gmInv.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
        GM_ADDR userWs = GetUserWorkspace(workspace);
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs));
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            ProcessAiv();
        } else {
            ProcessAic();
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        if (GetSubBlockIdx() != 0) {
            return;
        }
        OnChipBuffer buf;
        LocalTensor<float> ubL = buf.Ub<float>(0);
        LocalTensor<float> ubPacked = buf.Ub<float>(16 * 1024);
        LocalTensor<float> ubVcs = buf.Ub<float>(24 * 1024);
        LocalTensor<float> ubI = buf.Ub<float>(32 * 1024);
        LocalTensor<uint32_t> ubIdx = buf.Ub<uint32_t>(48 * 1024);
        LocalTensor<float> l1I = buf.L1<float>(kL1I);
        LocalTensor<float> l1NegL = buf.L1<float>(kL1NegL);
        LocalTensor<float> l1LeafR = buf.L1<float>(kL1LeafR);
        LocalTensor<float> l1LeafL = buf.L1<float>(kL1LeafL);

        DataCopy(ubL, gmL, kElems64);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Muls(ubL, ubL, -1.0f, kElems64);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        PackDiagLeaves(ubPacked, ubL);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        PaintVcsI(ubVcs);
        Duplicate(ubIdx, (uint32_t)0, 8);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        ubIdx.SetValue(0, (uint32_t)0);
        ubIdx.SetValue(1, (uint32_t)kLeaf);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        PipeBarrier<PIPE_V>();
        RunVcs(ubPacked, ubVcs, ubIdx);
        PipeBarrier<PIPE_V>();

        Duplicate(ubI, 0.0f, kElems64);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UbToL1Fp32Split(l1LeafR, ubI);
        UbToL1Fp32Split(l1LeafL, ubI);
        UbPackedLeafToL1(l1LeafR, ubVcs, 0, 0, 0);
        UbPackedLeafToL1(l1LeafL, ubVcs, 1, 1, kLeaf);
        UbNd64ToL1Nz8(l1NegL, ubL);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);

        PaintIdentity(ubI);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UbNd64ToL1Nz8(l1I, ubI);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(kFlagAivDone);
    }

    __aicore__ inline void ProcessAic()
    {
        OnChipBuffer buf;
        LocalTensor<float> l1I = buf.L1<float>(kL1I);
        LocalTensor<float> l1NegL = buf.L1<float>(kL1NegL);
        LocalTensor<float> l1LeafR = buf.L1<float>(kL1LeafR);
        LocalTensor<float> l1LeafL = buf.L1<float>(kL1LeafL);
        LocalTensor<float> l1Y = buf.L1<float>(kL1Y);
        LocalTensor<float> l0A = buf.L0A<float>(0);
        LocalTensor<float> l0B = buf.L0B<float>(0);
        LocalTensor<float> l0C = buf.L0C<float>(0);

        CrossCoreWaitFlag<0x4, PIPE_MTE1>(kFlagAivDone);
        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);

        // Stage4: Y = I + LeafLeft @ (-L)
        MatmulFp32(l1I, l1I, l0A, l0B, l0C, true, false, false);
        SetFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(0);
        MatmulFp32(l1LeafL, l1NegL, l0A, l0B, l0C, false, true, false);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeYNzCs(gmY, l0C);
        SetFlag<HardEvent::FIX_MTE2>(0);
        WaitFlag<HardEvent::FIX_MTE2>(0);
        CopyGmNzToL1(l1Y, gmY);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);

        // Stage5: A = LeafLeft + Y @ LeafRight
        MatmulFp32(l1I, l1LeafL, l0A, l0B, l0C, true, false, true);
        SetFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(0);
        MatmulFp32(l1Y, l1LeafR, l0A, l0B, l0C, false, false, true);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeInvNd(gmInv, l0C);
        SetFlag<HardEvent::FIX_M>(0);
    }

private:
    GlobalTensor<float> gmL, gmInv, gmY;
};

} // namespace WMmadDemoOp

#endif
