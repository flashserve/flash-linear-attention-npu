/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_o_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

// regbase.hpp 自身无 include guard，本算子 kernel 会同时包含 qkmask 与 output 两个
// epilogue 头，此处用算子私有宏防止同一编译单元内重复包含（重定义 constexpr/inline 符号）
#ifndef FLA_FWD_O_REGBASE_HPP_SEEN
#define FLA_FWD_O_REGBASE_HPP_SEEN
#include "kernel_utils/vector/regbase.hpp"
#endif

// RegBase VF 函数：Mins(0) → Exp → Mul(maskUbTensor) + 置零，全融合
// 按对角线所在 64 列 tile 分两个循环处理：
//   absRow < 64:  列 [0,64) 乘 maskUbTensor[absRow]，列 [64,128) 置零
//   absRow >= 64: 列 [0,64) 保留 Exp，列 [64,128) 乘 maskUbTensor[absRow-64]
// maskUbTensor 为 64×64 下三角矩阵，行 r 的前 r+1 个元素为 1.0，其余为 0.0。
__simd_vf__ inline void QkmaskCausalMaskVf(
    __ubuf__ float* __restrict__ upAddr,        // gbrcUpUbTensor 起始地址（Sub 结果，也是输出）
    __ubuf__ float* __restrict__ maskBase,      // maskUbTensor[0] 起始地址（64×64 矩阵起点）
    uint32_t mActualThisStage,                  // 本 stage 行数
    uint32_t alignedNActual,                    // 对齐列数（64 或 128）
    uint32_t gbrcStart,                         // 本 stage 起始绝对行号
    uint32_t gbrcEffStart,                      // mask 行起始偏移（gbrcStart - gbrcRealStart，0-7）
    uint32_t gbrcRealStart,                     // 列偏移起点（8 对齐，与 MemBase gbrcRealStart 一致）
    uint32_t gbrcRealEnd)                       // 列有效上界（8 对齐，与 MemBase gbrcRealEnd 一致）
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);  // 64

    (void)gbrcEffStart;  // mask 行号直接用 absRow 计算，gbrcEffStart 仅供外部 Sub 偏移使用

    RegTensor<float> vregUp;
    RegTensor<float> vregMask;
    RegTensor<float> vregZero;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();

    Duplicate(vregZero, 0.0f, maskFull);

    uint32_t gbrcEnd = gbrcStart + mActualThisStage;

    // Hardware Loop 规范（asc-devkit vf_loop_optimization）：边界/基址外提，三元与
    // clamp 只出现在循环外；归纳变量 uint16_t、零基、步长 1。段 2 的 up 行基址
    // upBase2 = upAddr + (seg2Start - gbrcStart) * alignedNActual 恒非负——原实现
    // 以 absRow(从 64 起) - gbrcStart 计行号，gbrcStart > 64 时（chunk128 的
    // subBlock1/stage1）前段迭代无符号下溢，靠地址回绕落在死缓冲才未出错。
    uint32_t seg1End = gbrcEnd > VL ? VL : gbrcEnd;
    uint32_t loop1Cnt = seg1End > gbrcStart ? seg1End - gbrcStart : 0;
    uint32_t seg2Start = gbrcStart > VL ? gbrcStart : VL;
    uint32_t loop2Cnt = gbrcEnd > seg2Start ? gbrcEnd - seg2Start : 0;
    __ubuf__ float* maskBase1 = maskBase + gbrcStart * VL;
    __ubuf__ float* maskBase2 = maskBase + (seg2Start - VL) * VL;
    __ubuf__ float* upBase2 = upAddr + (seg2Start - gbrcStart) * alignedNActual;

    // 第一个循环：absRow < 64，对角线在第一个 64 列 tile 内
    //   列 [0, 64):   Mins(0)→Exp→Mul(maskUbTensor[absRow]) 因果掩码
    //   列 [64, 128): 置零（远未来区域）
    for (uint16_t row = 0; row < static_cast<uint16_t>(loop1Cnt); ++row) {
        __ubuf__ float* rowAddr = upAddr + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase1 + row * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mins(vregUp, vregUp, 0.0f, maskFull);
        Exp(vregUp, vregUp, maskFull);
        Mul(vregUp, vregUp, vregMask, maskFull);
        StoreAlign(rowAddr, vregUp, maskFull);

        StoreAlign(rowAddr + VL, vregZero, maskFull);
    }
    // 第二个循环：absRow >= 64，对角线在第二个 64 列 tile 内
    //   列 [0, 64):   Mins(0)→Exp 保留（全部 j < 64 <= absRow，无需 mask）
    //   列 [64, 128): Mins(0)→Exp→Mul(maskUbTensor[absRow-64]) 因果掩码
    for (uint16_t row = 0; row < static_cast<uint16_t>(loop2Cnt); ++row) {
        __ubuf__ float* rowAddr = upBase2 + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase2 + row * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        Mins(vregUp, vregUp, 0.0f, maskFull);
        Exp(vregUp, vregUp, maskFull);
        StoreAlign(rowAddr, vregUp, maskFull);

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr + VL);
        Mins(vregUp, vregUp, 0.0f, maskFull);
        Exp(vregUp, vregUp, maskFull);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mul(vregUp, vregUp, vregMask, maskFull);
        StoreAlign(rowAddr + VL, vregUp, maskFull);
    }
}

// 尾块 VF 函数：alignedNActual ∈ (64, 128)，如 80, 96, 112
// 与满块 VF 的差异：第二段宽度 = alignedNActual - 64 (< 64)，用 UpdateMask 限制有效元素数
//   absRow < 64:  列 [0,64) 乘 mask[absRow]，列 [64,N) 置零 (tailWidth 个)
//   absRow >= 64: 列 [0,64) 保留 Exp，列 [64,N) 乘 mask[absRow-64] (tailWidth 个)
__simd_vf__ inline void QkmaskCausalMaskVfTail(
    __ubuf__ float* __restrict__ upAddr,
    __ubuf__ float* __restrict__ maskBase,
    uint32_t mActualThisStage,
    uint32_t alignedNActual,
    uint32_t gbrcStart,
    uint32_t gbrcEffStart,
    uint32_t gbrcRealStart,
    uint32_t gbrcRealEnd)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);  // 64

    (void)gbrcEffStart;
    (void)gbrcRealStart;
    (void)gbrcRealEnd;

    RegTensor<float> vregUp;
    RegTensor<float> vregMask;
    RegTensor<float> vregZero;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();

    Duplicate(vregZero, 0.0f, maskFull);

    uint32_t gbrcEnd = gbrcStart + mActualThisStage;
    uint32_t tailWidth = alignedNActual - VL;
    uint32_t tailWidth1 = alignedNActual > VL ? VL : alignedNActual;
    uint32_t tailWidth2 = alignedNActual > tailWidth1 ? alignedNActual - tailWidth1 : 0;
    MaskReg maskTail1 = UpdateMask<float>(tailWidth1);
    MaskReg maskTail2 = UpdateMask<float>(tailWidth2);

    // 同满块版：边界/基址外提 + 零基 uint16_t 归纳变量（Hardware Loop 规范）
    uint32_t seg1End = gbrcEnd > VL ? VL : gbrcEnd;
    uint32_t loop1Cnt = seg1End > gbrcStart ? seg1End - gbrcStart : 0;
    uint32_t seg2Start = gbrcStart > VL ? gbrcStart : VL;
    uint32_t loop2Cnt = gbrcEnd > seg2Start ? gbrcEnd - seg2Start : 0;
    __ubuf__ float* maskBase1 = maskBase + gbrcStart * VL;
    __ubuf__ float* maskBase2 = maskBase + (seg2Start - VL) * VL;
    __ubuf__ float* upBase2 = upAddr + (seg2Start - gbrcStart) * alignedNActual;
    (void)tailWidth;

    // 第一个循环：absRow < 64，对角线在第一个 64 列 tile 内
    //   列 [0, 64):      Mins(0)→Exp→Mul(mask[absRow])
    //   列 [64, N):       置零（tailWidth 个）
    for (uint16_t row = 0; row < static_cast<uint16_t>(loop1Cnt); ++row) {
        __ubuf__ float* rowAddr = upAddr + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase1 + row * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mins(vregUp, vregUp, 0.0f, maskTail1);
        Exp(vregUp, vregUp, maskTail1);
        Mul(vregUp, vregUp, vregMask, maskTail1);
        StoreAlign(rowAddr, vregUp, maskTail1);
        StoreAlign(rowAddr + VL, vregZero, maskTail2);
    }
    // 第二个循环：absRow >= 64，对角线在第二个 64 列 tile 内
    //   列 [0, 64):      Mins(0)→Exp 保留
    //   列 [64, N):       Mins(0)→Exp→Mul(mask[absRow-64])（tailWidth 个）
    for (uint16_t row = 0; row < static_cast<uint16_t>(loop2Cnt); ++row) {
        __ubuf__ float* rowAddr = upBase2 + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase2 + row * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        Mins(vregUp, vregUp, 0.0f, maskTail1);
        Exp(vregUp, vregUp, maskTail1);
        StoreAlign(rowAddr, vregUp, maskTail1);

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr + VL);
        Mins(vregUp, vregUp, 0.0f, maskTail2);
        Exp(vregUp, vregUp, maskTail2);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mul(vregUp, vregUp, vregMask, maskTail2);
        StoreAlign(rowAddr + VL, vregUp, maskTail2);
    }
}

namespace Catlass::Epilogue::Block {

template <
    class AOutputType_,
    class GInputType_,
    class AInputType_,
    class MaskInputType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdOQkmask,
    AOutputType_,
    GInputType_,
    AInputType_,
    MaskInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdOQkmask;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using AElementOutput = typename AOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    using MaskElementInput = typename MaskInputType_::Element;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_ELENUM_PER_BLK = 8;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t UB_TILE_SIZE = 16384;  // 64 * 128 * 2B
    static constexpr uint32_t UB_LINE_SIZE = 512;   // 128 * 2 * 2B
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;    // 128 * 2
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;   // 128
    static constexpr uint32_t MULTIPLIER = 2;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        // UB layout (ub2l1, shared contract with the kernel and output epilogue —
        // see docs/agents/chunk_fwd_o_l0c2ub_ub2l1_design.md §4.1):
        //   [0, 71K)    base region — maskUbTensor (kernel-lifetime tril mask, read
        //              every round) + gbrc/gcomp/share/g; offsets identical to the
        //              output epilogue.
        //   [71K, 199K) Cube work slots (4 x 32KB) — written by Cube-side Fixpipe
        //              while Vec1 runs; all Vec1 buffers must stay out of this range.
        //   [199K, 248K) Vec1 scratch (a fp32 + out fp32 + out half), time-shared
        //              with Vec2's out scratch (Vec1/Vec2 never run concurrently
        //              within one AIV subBlock).
        constexpr uint32_t BASE = 0;
        constexpr uint32_t MASK_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_SIZE = 40 * UB_LINE_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 16 * UB_LINE_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t G_FLOAT_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t UB_WORK_SLOT_BYTES = 64 * 128 * sizeof(float);
        constexpr uint32_t UB_SLOT_REGION_END = 71 * 1024 + 4 * UB_WORK_SLOT_BYTES;

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_OFFSET = MASK_UB_TENSOR_OFFSET + MASK_UB_TENSOR_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_OFFSET = GBRCLEFTCAST_UB_TENSOR_OFFSET + GBRCLEFTCAST_UB_TENSOR_SIZE;
        constexpr uint32_t GCOMP_UB_TENSOR_OFFSET = GBRCUP_UB_TENSOR_OFFSET + GBRCUP_UB_TENSOR_SIZE;
        constexpr uint32_t SHARE_UB_TENSOR_OFFSET = GCOMP_UB_TENSOR_OFFSET + G_FLOAT_UB_TENSOR_SIZE;

        maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(MASK_UB_TENSOR_OFFSET);
        gbrcLeftcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCLEFTCAST_UB_TENSOR_OFFSET);
        gbrcUpUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCUP_UB_TENSOR_OFFSET);
        gcompUbTensor = resource.ubBuf.template GetBufferByByte<float>(GCOMP_UB_TENSOR_OFFSET);
        shareUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_UB_TENSOR_OFFSET);

        constexpr uint32_t G_UB_TENSOR_OFFSET = SHARE_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + G_FLOAT_UB_TENSOR_SIZE;

        constexpr uint32_t A_UB_TENSOR_OFFSET = UB_SLOT_REGION_END;
        constexpr uint32_t OUT_UB_TENSOR_OFFSET = A_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_HALF_UB_TENSOR_OFFSET = OUT_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        static_assert(OUT_HALF_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE <= 248 * 1024, "UB overflow");

        gUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(G_UB_TENSOR_OFFSET);
        gUbFPTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET);
        gUbBFTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET);
        aUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(A_UB_TENSOR_OFFSET);
        outUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(OUT_UB_TENSOR_OFFSET);
        outUbFPTensorPing = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_HALF_UB_TENSOR_OFFSET);
        outUbBFTensorPing = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_HALF_UB_TENSOR_OFFSET);

        // Ping == Pong for Vec1's a/out tiles: consumed within one Vec1 call before
        // reuse (V event handoffs); the Cube slots carry the real ping-pong semantics.
        gUbTensorPong = gUbTensorPing;
        gUbFPTensorPong = gUbFPTensorPing;
        gUbBFTensorPong = gUbBFTensorPing;
        aUbTensorPong = aUbTensorPing;
        outUbTensorPong = outUbTensorPing;
        outUbFPTensorPong = outUbFPTensorPing;
        outUbBFTensorPong = outUbBFTensorPing;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    /// ub2l1: direct the aftermask output to an L1 slot (zN layout) instead of GM.
    CATLASS_DEVICE
    void EnableL1Output(AscendC::LocalTensor<AElementOutput> l1OutputTensor)
    {
        l1Output = l1OutputTensor;
        enableL1Output = true;
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<AElementOutput> maskOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<AElementInput> attnInput,
        AscendC::GlobalTensor<MaskElementInput> boolInput,
        uint32_t fullChunkSize,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        uint32_t &pingpongFlag
        , uint32_t batchIdx, uint32_t headIdx, uint32_t chunkIdx,
        Arch::CrossCoreFlag* waitFlag = nullptr
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nActual = chunkSize;
        uint32_t alignedNActual = CeilDiv(nActual, 16) * 16;
        bool isContiguousFullTile = chunkSize == fullChunkSize && nActual == alignedNActual;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetA = mOffset * nActual + nOffset;
        uint16_t aInputDstStride;
        if((nActual - 1) % 16 <= 7) aInputDstStride = 1;
        else aInputDstStride = 0;

        uint32_t gbrcStart, gbrcRealStart, gbrcRealEnd, gbrcRealProcess, gbrcEffStart, gbrcEffEnd, mulsRemain, mulsRemainIdx;
        if(mActualThisSubBlock <= 32)
        {   if(subBlockIdx == 0)
            {
                gbrcStart = 0;
                gbrcRealStart = 0;
                gbrcRealProcess = mActualThisSubBlock;
            }
            else
            {
                gbrcStart = mActualPerSubBlock;
                gbrcRealStart = gbrcStart & ~7;
                gbrcRealProcess = mActual - gbrcRealStart;
            }

            gbrcEffStart = gbrcStart - gbrcRealStart;
            gbrcEffEnd = gbrcEffStart + mActualThisSubBlock;

            uint32_t dstUpShape_[2] = {mActualThisSubBlock, alignedNActual};
            uint32_t srcUpShape_[2] = {1, alignedNActual};
            uint32_t dstLeftShape_[2] = {gbrcRealProcess, alignedNActual};
            uint32_t srcLeftShape_[2] = {gbrcRealProcess, 1};

            AscendC::ResetMask();
            AscendC::GlobalTensor<AElementOutput> maskOutputThisSubBlock = maskOutput[gbrcStart * nActual];
            AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[gbrcStart * nActual];
            AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;


            AscendC::DataCopyParams aInputUbParams{(uint16_t)mActualThisSubBlock, (uint16_t)(nActual*sizeof(float)), 0, aInputDstStride};
            AscendC::DataCopyPadParams aInputUbPadParams{false, 0, 0, 0};
            // UB->GM DataCopyPad advances the UB source by AlignUp(blockLen, 32B).
            // For fp16/bf16 qk-mask rows this matches alignedNActual, so srcStride stays 0.
            AscendC::DataCopyExtParams aOutputUbParams{(uint16_t)mActualThisSubBlock, (uint32_t)(nActual*sizeof(half)), 0, 0, 0};

            AscendC::DataCopyParams gfloatUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
            AscendC::DataCopyParams ghalfUbParams{1, (uint16_t)(mActual*sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

            AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
            AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
            AscendC::LocalTensor<AElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
            AscendC::LocalTensor<AElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;
            AscendC::LocalTensor<float> gUbTensor = (pingpongFlag == 0) ? gUbTensorPing : gUbTensorPong;
            AscendC::LocalTensor<GElementInput> gUbFPTensor = (pingpongFlag == 0) ? gUbFPTensorPing : gUbFPTensorPong;
            AscendC::LocalTensor<GElementInput> gUbBFTensor = (pingpongFlag == 0) ? gUbBFTensorPing : gUbBFTensorPong;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            if constexpr(std::is_same<GElementInput, float>::value) {
                AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gfloatUbParams, gUbPadParams);
            } else {
                AscendC::DataCopyPad(gUbFPTensor, gInputThisSubBlock, ghalfUbParams, gUbPadParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            if constexpr(!std::is_same<GElementInput, float>::value) {
                AscendC::Cast(gUbTensor, gUbFPTensor, AscendC::RoundMode::CAST_NONE, mActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Copy(gcompUbTensor, gUbTensor, 64, 2, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Broadcast<float, 2, 0>(gbrcUpUbTensor, gcompUbTensor, dstUpShape_, srcUpShape_, shareUbTensor);
            AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gcompUbTensor[gbrcRealStart], dstLeftShape_, srcLeftShape_, shareUbTensor);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(gbrcUpUbTensor, gbrcLeftcastUbTensor[gbrcEffStart*alignedNActual], gbrcUpUbTensor, mActualThisSubBlock * alignedNActual);
            AscendC::PipeBarrier<PIPE_V>();
            gbrcRealEnd = CeilDiv(gbrcStart + mActualThisSubBlock, 8) * 8;
            // [RegBase] Mins(0) → Exp → Mul(maskUbTensor) + 置零 全融合为单个 VF 函数。
            // 本分支 mActualThisSubBlock<=32 ⇒ chunkSize<=65 ⇒ alignedNActual∈{64,80}，
            // 永远 < 128，直接走 QkmaskCausalMaskVfTail（alignedNActual==64 时 tailWidth2=0，第二段为 no-op）。
            {
                auto upAddr = reinterpret_cast<uint64_t>(gbrcUpUbTensor.GetPhyAddr());
                auto maskBase = reinterpret_cast<uint64_t>(maskUbTensor.GetPhyAddr());
                QkmaskCausalMaskVfTail((__ubuf__ float*)upAddr,
                                       (__ubuf__ float*)maskBase,
                                       mActualThisSubBlock, alignedNActual,
                                       gbrcStart, gbrcEffStart, gbrcRealStart, gbrcRealEnd);
                AscendC::PipeBarrier<PIPE_V>();
            }
            (void)gbrcRealEnd;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            if (waitFlag) Arch::CrossCoreWaitFlag(*waitFlag);
            if(isContiguousFullTile) AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mActualThisSubBlock*nActual);
            else AscendC::DataCopyPad(aUbTensor, attnInputThisSubBlock, aInputUbParams, aInputUbPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::Mul(outUbTensor, aUbTensor, gbrcUpUbTensor, mActualThisSubBlock * alignedNActual);
            AscendC::PipeBarrier<PIPE_V>();

            if(std::is_same<AElementOutput, half>::value)
            {
                AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                if (enableL1Output) {
                    CopyOutputToL1(outUbFPTensor, gbrcStart, mActualThisSubBlock, nActual, alignedNActual);
                } else if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbFPTensor, mActualThisSubBlock*nActual);
                else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbFPTensor, aOutputUbParams);
            }
            else
            {
                AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                if (enableL1Output) {
                    CopyOutputToL1(outUbBFTensor, gbrcStart, mActualThisSubBlock, nActual, alignedNActual);
                } else if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbBFTensor, mActualThisSubBlock*nActual);
                else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbBFTensor, aOutputUbParams);
            }
            pingpongFlag = 1 - pingpongFlag;
        }
        else // mActualThisSubBlock  > 32 ; <=64
        {
            AscendC::ResetMask();
            AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

            AscendC::DataCopyParams gfloatUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
            AscendC::DataCopyParams ghalfUbParams{1, (uint16_t)(mActual*sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

            AscendC::LocalTensor<float> gUbTensor = (pingpongFlag == 0) ? gUbTensorPing : gUbTensorPong;
            AscendC::LocalTensor<GElementInput> gUbFPTensor = (pingpongFlag == 0) ? gUbFPTensorPing : gUbFPTensorPong;
            AscendC::LocalTensor<GElementInput> gUbBFTensor = (pingpongFlag == 0) ? gUbBFTensorPing : gUbBFTensorPong;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            if constexpr(std::is_same<GElementInput, float>::value) {
                AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gfloatUbParams, gUbPadParams);
            } else {
                AscendC::DataCopyPad(gUbFPTensor, gInputThisSubBlock, ghalfUbParams, gUbPadParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            if constexpr(!std::is_same<GElementInput, float>::value) {
                AscendC::Cast(gUbTensor, gUbFPTensor, AscendC::RoundMode::CAST_NONE, mActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Copy(gcompUbTensor, gUbTensor, 64, 2, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t mActualPerStage = CeilDiv(mActualThisSubBlock, 2);
            uint32_t mActualThisStage = 0;
            for(uint32_t stage = 0; stage < 2; ++stage)
            {
                if(stage==0) mActualThisStage = mActualPerStage;
                else mActualThisStage = mActualThisSubBlock - mActualPerStage;

                if(subBlockIdx == 0 && stage == 0)
                {
                    gbrcStart = 0;
                    gbrcRealStart = 0;
                    gbrcRealProcess = mActualThisStage;
                }
                else if(subBlockIdx == 0 && stage == 1)
                {
                    gbrcStart = mActualPerStage;
                    gbrcRealStart = gbrcStart & ~7;
                    gbrcRealProcess = mActualThisSubBlock - gbrcRealStart;
                }
                else if(subBlockIdx == 1 && stage == 0)
                {
                    gbrcStart = mActualPerSubBlock;
                    gbrcRealStart = gbrcStart & ~7;
                    gbrcRealProcess = mActualPerSubBlock + mActualThisStage - gbrcRealStart;
                }
                else if(subBlockIdx == 1 && stage == 1)
                {
                    gbrcStart = mActualPerSubBlock + mActualPerStage;
                    gbrcRealStart = gbrcStart & ~7;
                    gbrcRealProcess = mActual - gbrcRealStart;
                }

                gbrcEffStart = gbrcStart - gbrcRealStart;

                AscendC::GlobalTensor<AElementOutput> maskOutputThisSubBlock = maskOutput[gbrcStart * nActual];
                AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[gbrcStart * nActual];

                AscendC::DataCopyParams aInputUbParams{(uint16_t)mActualThisStage, (uint16_t)(nActual*sizeof(float)), 0, aInputDstStride};
                AscendC::DataCopyPadParams aInputUbPadParams{false, 0, 0, 0};
                // UB->GM DataCopyPad advances the UB source by AlignUp(blockLen, 32B).
                // For fp16/bf16 qk-mask rows this matches alignedNActual, so srcStride stays 0.
                AscendC::DataCopyExtParams aOutputUbParams{(uint16_t)mActualThisStage, (uint32_t)(nActual*sizeof(half)), 0, 0, 0};

                AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
                AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
                AscendC::LocalTensor<AElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
                AscendC::LocalTensor<AElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                if (waitFlag && stage == 0) Arch::CrossCoreWaitFlag(*waitFlag);
                if(isContiguousFullTile) AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mActualThisStage*nActual);
                else AscendC::DataCopyPad(aUbTensor, attnInputThisSubBlock, aInputUbParams, aInputUbPadParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);

                uint32_t dstUpShape_[2] = {mActualThisStage, alignedNActual};
                uint32_t srcUpShape_[2] = {1, alignedNActual};
                uint32_t dstLeftShape_[2] = {gbrcRealProcess, alignedNActual};
                uint32_t srcLeftShape_[2] = {gbrcRealProcess, 1};

                AscendC::Broadcast<float, 2, 0>(gbrcUpUbTensor, gcompUbTensor, dstUpShape_, srcUpShape_, shareUbTensor);
                AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gcompUbTensor[gbrcRealStart], dstLeftShape_, srcLeftShape_, shareUbTensor);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(gbrcUpUbTensor, gbrcLeftcastUbTensor[gbrcEffStart*alignedNActual], gbrcUpUbTensor, mActualThisStage * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();

                gbrcRealEnd = CeilDiv(gbrcStart + mActualThisStage, 8) * 8;
                // [RegBase] Mins(0) → Exp → Mul(maskUbTensor) + 置零 全融合为单个 VF 函数。
                // 满块 (alignedNActual==128) 走 QkmaskCausalMaskVf，尾块 (>64,<128) 走 QkmaskCausalMaskVfTail。
                {
                    auto upAddr = reinterpret_cast<uint64_t>(gbrcUpUbTensor.GetPhyAddr());
                    auto maskBase = reinterpret_cast<uint64_t>(maskUbTensor.GetPhyAddr());
                    if (alignedNActual == 128) {
                        QkmaskCausalMaskVf((__ubuf__ float*)upAddr,
                                           (__ubuf__ float*)maskBase,
                                           mActualThisStage, alignedNActual,
                                           gbrcStart, gbrcEffStart, gbrcRealStart, gbrcRealEnd);
                    } else {
                        QkmaskCausalMaskVfTail((__ubuf__ float*)upAddr,
                                               (__ubuf__ float*)maskBase,
                                               mActualThisStage, alignedNActual,
                                               gbrcStart, gbrcEffStart, gbrcRealStart, gbrcRealEnd);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                (void)gbrcRealEnd;
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
                AscendC::Mul(outUbTensor, aUbTensor, gbrcUpUbTensor, mActualThisStage * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();
                if(std::is_same<AElementOutput, half>::value)
                {
                    AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisStage * alignedNActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    if (enableL1Output) {
                        CopyOutputToL1(outUbFPTensor, gbrcStart, mActualThisStage, nActual, alignedNActual);
                    } else if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbFPTensor, mActualThisStage*nActual);
                    else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbFPTensor, aOutputUbParams);
                }
                else
                {
                    AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisStage * alignedNActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    if (enableL1Output) {
                        CopyOutputToL1(outUbBFTensor, gbrcStart, mActualThisStage, nActual, alignedNActual);
                    } else if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbBFTensor, mActualThisStage*nActual);
                    else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbBFTensor, aOutputUbParams);
                }
                pingpongFlag = 1 - pingpongFlag;
            }
        }

    }

private:
    /// ub2l1: copy a rows x cols half/bf16 tile from UB (row-major, stride
    /// srcStride) into the L1 aftermask slot at zN row offset rowOffset.
    /// Faithful port of the catlass CopyUb2L1Tla reference (Ascend950,
    /// RowMajor -> zN, see copy_ub_to_l1_tla.hpp): stride-derived addressing on a
    /// GetTile sub-tile, NOT hand-derived algebra — the zN sub-tile stride units
    /// only stay consistent with Cube3's L1A reader when derived from the tile.
    CATLASS_DEVICE
    void CopyOutputToL1(
        AscendC::LocalTensor<AElementOutput> ubTensor,
        uint32_t rowOffset,
        uint32_t rows,
        uint32_t cols,
        uint32_t srcStride)
    {
        if (!enableL1Output || rows == 0) {
            return;
        }

        auto ubLayout = tla::MakeLayout(
            tla::MakeShape(rows, cols), tla::MakeStride(srcStride, tla::Int<1>{}), tla::MakeShape(rows, cols));
        auto l1Layout = tla::MakeLayout<AElementOutput, Catlass::layout::zN>(tla::Int<128>{}, tla::Int<128>{});
        auto tensorUb = tla::MakeTensor(ubTensor, ubLayout, Catlass::Arch::PositionUB{});
        auto tensorL1 = tla::MakeTensor(l1Output, l1Layout, Catlass::Arch::PositionL1{});
        auto tensorL1Tile = GetTile(tensorL1, tla::MakeCoord(rowOffset, 0), tla::MakeShape(rows, cols));

        constexpr uint32_t BYTE_PER_BLK = 32;
        constexpr uint32_t BYTE_PER_C0 = 32;  // zN fractal C0 size (16 x 2B half/bf16), same as catlass
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(AElementOutput);
        const uint32_t srcDValue = tla::get<0>(tensorUb.stride());
        const uint32_t dstNzNStride = tla::get<0, 0>(tensorL1Tile.stride()) / ELE_NUM_PER_C0;
        const uint32_t dstNzC0Stride = tla::get<1, 1>(tensorL1Tile.stride()) / ELE_NUM_PER_C0;

        auto dstOffset = tensorL1Tile.layout()(tensorL1Tile.coord());
        auto srcOffset = tensorUb.layout()(tensorUb.coord());

        uint32_t nRows = rows;
        uint32_t nCols = RoundUp<BYTE_PER_BLK>(cols * sizeof(AElementOutput)) / BYTE_PER_BLK;
        if (nCols > nRows) {
            for (uint32_t i = 0; i < nRows; i++) {
                uint32_t dst = i * dstNzNStride * BYTE_PER_BLK / sizeof(AElementOutput);
                uint32_t src = i * srcDValue;
                AscendC::DataCopyParams dataCopyParams(nCols, 1, 0, dstNzC0Stride - 1);
                AscendC::DataCopy(
                    tensorL1Tile.data()[dstOffset + dst], tensorUb.data()[srcOffset + src], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < nCols; i++) {
                uint32_t dst = i * dstNzC0Stride * BYTE_PER_BLK / sizeof(AElementOutput);
                uint32_t src = i * BYTE_PER_BLK / sizeof(AElementOutput);
                AscendC::DataCopyParams dataCopyParams(
                    nRows, 1, srcDValue * sizeof(AElementOutput) / BYTE_PER_BLK - 1, dstNzNStride - 1);
                AscendC::DataCopy(
                    tensorL1Tile.data()[dstOffset + dst], tensorUb.data()[srcOffset + src], dataCopyParams);
            }
        }
        // SSBuffer write-completion fence: block the Vector pipeline until the
        // MTE3 queue has issued these L1 writes, so the caller's cross-core
        // completion flag (vec1Done) cannot overtake the data in flight.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    AscendC::LocalTensor<float> maskUbTensor;
    AscendC::LocalTensor<float> gbrcLeftcastUbTensor;
    AscendC::LocalTensor<float> gbrcUpUbTensor;
    AscendC::LocalTensor<float> gcompUbTensor;
    AscendC::LocalTensor<uint8_t> shareUbTensor;

    AscendC::LocalTensor<float> gUbTensorPing;
    AscendC::LocalTensor<GElementInput> gUbFPTensorPing;
    AscendC::LocalTensor<GElementInput> gUbBFTensorPing;
    AscendC::LocalTensor<float> aUbTensorPing;
    AscendC::LocalTensor<float> outUbTensorPing;
    AscendC::LocalTensor<AElementOutput> outUbFPTensorPing;
    AscendC::LocalTensor<AElementOutput> outUbBFTensorPing;

    AscendC::LocalTensor<float> gUbTensorPong;
    AscendC::LocalTensor<GElementInput> gUbFPTensorPong;
    AscendC::LocalTensor<GElementInput> gUbBFTensorPong;
    AscendC::LocalTensor<float> aUbTensorPong;
    AscendC::LocalTensor<float> outUbTensorPong;
    AscendC::LocalTensor<AElementOutput> outUbFPTensorPong;
    AscendC::LocalTensor<AElementOutput> outUbBFTensorPong;

    // ub2l1 state (set via EnableL1Output before operator()).
    bool enableL1Output = false;
    AscendC::LocalTensor<AElementOutput> l1Output;

};
}

#endif
