/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP

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

// RegBase VF 函数：Mul(H, exp(g)) → Add(attn) → Muls(scale) 全融合，逐行寄存器处理
// nActual=128 时分两段各 64 列处理，与 qkmask 的 VF 结构对齐
// exp(g) 不物化到 UB：每行一条 DIST_BRC_B32 标量广播加载（地址仅需 4 字节对齐），
// 两个列段共用同一广播寄存器（替代原 Broadcast 物化 + 逐行 DIST_NORM 读回）
__simd_vf__ inline void OutputFusedVf(
    __ubuf__ float* __restrict__ outAddr,       // outUbTensor 起始地址（输出）
    __ubuf__ float* __restrict__ hAddr,          // hUbTensor 起始地址（H 数据）
    __ubuf__ float* __restrict__ aAddr,          // aUbTensor 起始地址（attn 数据）
    __ubuf__ float* __restrict__ gExpScalarAddr, // exp(g) 标量数组地址（每行一个 fp32）
    uint32_t mActualThisStage,                   // 本 stage 行数
    uint32_t nActual,                            // 列数（128）
    float scale)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);  // 64

    // 使用两组寄存器，同类型指令连续执行，利用 RegBase 指令流水优化
    RegTensor<float> vregH1, vregH2;
    RegTensor<float> vregA1, vregA2;
    RegTensor<float> vregExp;
    RegTensor<float> vregOut1, vregOut2;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();

    // Hardware Loop 规范：uint16_t 零基归纳变量 + 边界外提（mActualThisStage ≤ 64）
    const uint16_t rowCnt = static_cast<uint16_t>(mActualThisStage);
    for (uint16_t row = 0; row < rowCnt; ++row) {
        __ubuf__ float* rowH = hAddr + row * nActual;
        __ubuf__ float* rowA = aAddr + row * nActual;
        __ubuf__ float* rowOut = outAddr + row * nActual;

        // 连续 Load；exp(g) 逐行标量广播（BRC_B32），两段共用同一寄存器
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vregExp, gExpScalarAddr + row);
        LoadAlign<float, LoadDist::DIST_NORM>(vregH1, rowH);
        LoadAlign<float, LoadDist::DIST_NORM>(vregH2, rowH + VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregA1, rowA);
        LoadAlign<float, LoadDist::DIST_NORM>(vregA2, rowA + VL);

        // 连续 Mul: H * exp(g)
        Mul(vregOut1, vregH1, vregExp, maskFull);
        Mul(vregOut2, vregH2, vregExp, maskFull);

        // 连续 Add: + attn
        Add(vregOut1, vregOut1, vregA1, maskFull);
        Add(vregOut2, vregOut2, vregA2, maskFull);

        // 连续 Muls: * scale
        Muls(vregOut1, vregOut1, scale, maskFull);
        Muls(vregOut2, vregOut2, scale, maskFull);

        // 连续 Store
        StoreAlign(rowOut, vregOut1, maskFull);
        StoreAlign(rowOut + VL, vregOut2, maskFull);
    }
}

// RegBase VF 函数（Wide 版本）：Mul(H, exp(g)) → Add(attn) → Muls(scale) 全融合
// nActual=256 时分四段各 64 列处理，使用四组寄存器最大化指令流水
// V256(Wide/GM) 路径保留 Broadcast 物化：其发出位置与 h/a 的 GM→UB MTE2 传输
// 重叠（免费），VF 直接读物化结果；V128 路径见 OutputFusedVf 的 BRC_B32 说明
__simd_vf__ inline void OutputFusedVfWide(
    __ubuf__ float* __restrict__ outAddr,       // outUbTensor 起始地址（输出）
    __ubuf__ float* __restrict__ hAddr,          // hUbTensor 起始地址（H 数据）
    __ubuf__ float* __restrict__ aAddr,          // aUbTensor 起始地址（attn 数据）
    __ubuf__ float* __restrict__ brcExpAddr,     // gbrcLeftcastUbTensor[gbrcEffStart*nActual] 地址
    uint32_t mActualThisStage,                   // 本 tile 行数
    uint32_t nActual,                            // 列数（256）
    float scale)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);  // 64

    // 使用四组寄存器，同类型指令连续执行，利用 RegBase 指令流水优化
    RegTensor<float> vregH1, vregH2, vregH3, vregH4;
    RegTensor<float> vregA1, vregA2, vregA3, vregA4;
    RegTensor<float> vregExp1, vregExp2, vregExp3, vregExp4;
    RegTensor<float> vregOut1, vregOut2, vregOut3, vregOut4;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();

    // Hardware Loop 规范：uint16_t 零基归纳变量 + 边界外提（mActualThisStage ≤ 16）
    const uint16_t rowCnt = static_cast<uint16_t>(mActualThisStage);
    for (uint16_t row = 0; row < rowCnt; ++row) {
        __ubuf__ float* rowH = hAddr + row * nActual;
        __ubuf__ float* rowA = aAddr + row * nActual;
        __ubuf__ float* rowOut = outAddr + row * nActual;
        __ubuf__ float* rowExp = brcExpAddr + row * nActual;

        // 连续 Load（四段）
        LoadAlign<float, LoadDist::DIST_NORM>(vregH1, rowH);
        LoadAlign<float, LoadDist::DIST_NORM>(vregH2, rowH + VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregH3, rowH + 2 * VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregH4, rowH + 3 * VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregA1, rowA);
        LoadAlign<float, LoadDist::DIST_NORM>(vregA2, rowA + VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregA3, rowA + 2 * VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregA4, rowA + 3 * VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregExp1, rowExp);
        LoadAlign<float, LoadDist::DIST_NORM>(vregExp2, rowExp + VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregExp3, rowExp + 2 * VL);
        LoadAlign<float, LoadDist::DIST_NORM>(vregExp4, rowExp + 3 * VL);

        // 连续 Mul: H * exp(g)
        Mul(vregOut1, vregH1, vregExp1, maskFull);
        Mul(vregOut2, vregH2, vregExp2, maskFull);
        Mul(vregOut3, vregH3, vregExp3, maskFull);
        Mul(vregOut4, vregH4, vregExp4, maskFull);

        // 连续 Add: + attn
        Add(vregOut1, vregOut1, vregA1, maskFull);
        Add(vregOut2, vregOut2, vregA2, maskFull);
        Add(vregOut3, vregOut3, vregA3, maskFull);
        Add(vregOut4, vregOut4, vregA4, maskFull);

        // 连续 Muls: * scale
        Muls(vregOut1, vregOut1, scale, maskFull);
        Muls(vregOut2, vregOut2, scale, maskFull);
        Muls(vregOut3, vregOut3, scale, maskFull);
        Muls(vregOut4, vregOut4, scale, maskFull);

        // 连续 Store（四段）
        StoreAlign(rowOut, vregOut1, maskFull);
        StoreAlign(rowOut + VL, vregOut2, maskFull);
        StoreAlign(rowOut + 2 * VL, vregOut3, maskFull);
        StoreAlign(rowOut + 3 * VL, vregOut4, maskFull);
    }
}

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class AInputType_,
    class HInputType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdOOutput,
    HOutputType_,
    GInputType_,
    AInputType_,
    HInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdOOutput;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    using HElementInput = typename HInputType_::Element;

    // using CopyGmToUbInput = Tile::CopyGm2Ub<ArchTag, InputType_>;
    // using CopyUbToGmOutput = Tile::CopyUb2Gm<ArchTag, OutputType_>;

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
        // UB layout (l0c2ub, shared contract with the kernel and qkmask epilogue —
        // see docs/agents/chunk_fwd_o_l0c2ub_ub2l1_design.md §4.1):
        //   [0, 71K)    base region (mask/gbrc/gcomp/share/g) — offsets identical to
        //              the qkmask epilogue; maskUbTensor/gbrcUpUbTensor here are the
        //              same physical bytes qkmask actively uses (do NOT reclaim).
        //   [71K, 199K) Cube work slots: V/H ping + V/H pong (4 x 32KB), written by
        //              Cube-side Fixpipe SPLIT_M while Vec1/Vec2 run — must stay
        //              disjoint from every Vec buffer below.
        //   [199K, 248K) Vec1/Vec2 time-shared scratch (out fp32 + out half).
        constexpr uint32_t BASE = 0;
        constexpr uint32_t MASK_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_SIZE = 40 * UB_LINE_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 16 * UB_LINE_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t G_FLOAT_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t UB_WORK_SLOT_BYTES = 64 * 128 * sizeof(float);

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_OFFSET = MASK_UB_TENSOR_OFFSET + MASK_UB_TENSOR_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_OFFSET = GBRCLEFTCAST_UB_TENSOR_OFFSET + GBRCLEFTCAST_UB_TENSOR_SIZE;
        constexpr uint32_t GCOMP_UB_TENSOR_OFFSET = GBRCUP_UB_TENSOR_OFFSET + GBRCUP_UB_TENSOR_SIZE;
        constexpr uint32_t SHARE_UB_TENSOR_OFFSET = GCOMP_UB_TENSOR_OFFSET + G_FLOAT_UB_TENSOR_SIZE;

        maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(MASK_UB_TENSOR_OFFSET);
        gbrcLeftcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCLEFTCAST_UB_TENSOR_OFFSET);
        gbrcUpUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCUP_UB_TENSOR_OFFSET);
        shareUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_UB_TENSOR_OFFSET);
        // gcomp 区域仍由 qkmask epilogue 持有（同物理 UB、同偏移）；本 epilogue 的
        // V128 路径经 BRC_B32 直读 g buffer，V256(Wide) 路径经 gbrcLeftcast 物化。

        // Cube2/Cube3 Fixpipe work slots (kernel owns the same constants).
        constexpr uint32_t UB_V_WORK_PING_OFFSET = 71 * 1024;
        constexpr uint32_t UB_H_WORK_PING_OFFSET = UB_V_WORK_PING_OFFSET + UB_WORK_SLOT_BYTES;
        constexpr uint32_t UB_V_WORK_PONG_OFFSET = UB_H_WORK_PING_OFFSET + UB_WORK_SLOT_BYTES;
        constexpr uint32_t UB_H_WORK_PONG_OFFSET = UB_V_WORK_PONG_OFFSET + UB_WORK_SLOT_BYTES;

        // g buffers: single copy (read-only, no ping/pong needed).
        constexpr uint32_t G_UB_TENSOR_OFFSET = SHARE_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + G_FLOAT_UB_TENSOR_SIZE;

        // Vec scratch above the slot region, time-shared with Vec1's a/out tiles
        // (Vec1 and Vec2 never run concurrently within one AIV subBlock).
        // out fp32 + out half need REAL ping/pong copies: the MTE3 DataCopy that
        // drains outUbB* is asynchronous, and the next stage's (or next task's)
        // VF/Cast would overwrite the same bytes while the copy's tail rows are
        // still in flight — observed as the last rows of one output segment
        // carrying the next segment's data. g stays single-copy (its only
        // consumer is V itself, serialized by the MTE2_V handshakes).
        constexpr uint32_t OUT_UB_TENSOR_OFFSET = UB_H_WORK_PONG_OFFSET + UB_WORK_SLOT_BYTES;
        constexpr uint32_t OUT_HALF_UB_TENSOR_OFFSET = OUT_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_UB_TENSOR_PONG_OFFSET = OUT_HALF_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_HALF_UB_TENSOR_PONG_OFFSET = OUT_UB_TENSOR_PONG_OFFSET + FLOAT_UB_TENSOR_SIZE;
        static_assert(OUT_HALF_UB_TENSOR_PONG_OFFSET + HALF_UB_TENSOR_SIZE <= 248 * 1024, "UB overflow");

        gUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(G_UB_TENSOR_OFFSET);
        gUbFPTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET);
        gUbBFTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET);
        outUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(OUT_UB_TENSOR_OFFSET);
        outUbFPTensorPing = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_OFFSET);
        outUbBFTensorPing = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_OFFSET);
        outUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(OUT_UB_TENSOR_PONG_OFFSET);
        outUbFPTensorPong = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_PONG_OFFSET);
        outUbBFTensorPong = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_PONG_OFFSET);

        // g buffers: single copy (read-only into V, no ping/pong needed).
        gUbTensorPong = gUbTensorPing;
        gUbFPTensorPong = gUbFPTensorPing;
        gUbBFTensorPong = gUbBFTensorPing;

        // Work-slot mirrors (only meaningful on the l0c2ub path where the kernel
        // passes LocalTensor inputs; kept for the GM fallback path's DataCopy form).
        aUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(UB_V_WORK_PING_OFFSET);
        aUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(UB_V_WORK_PONG_OFFSET);
        hUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(UB_H_WORK_PING_OFFSET);
        hUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(UB_H_WORK_PONG_OFFSET);
    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void CopyOutputToGm(
        AscendC::GlobalTensor<HElementOutput> output,
        AscendC::LocalTensor<HElementOutput> outputUb,
        uint32_t rows,
        uint32_t cols,
        uint32_t outputStride)
    {
        if (cols == outputStride) {
            AscendC::DataCopy(output, outputUb, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(HElementOutput)),
            0,
            static_cast<uint32_t>((outputStride - cols) * sizeof(HElementOutput)),
            0};
        AscendC::DataCopyPad(output, outputUb, outputParams);
    }

    /// V256 wide path — always fed by the GM workspace path (nActual > 128 cannot
    /// fit the 128-column UB slots), so the work inputs are GlobalTensor.
    template <class WorkInputA, class WorkInputH>
    CATLASS_DEVICE
    void ProcessWideOutput(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        WorkInputA attnInput,
        WorkInputH hInput,
        float scale,
        uint32_t mActual,
        uint32_t nActual,
        uint32_t outputStride,
        uint32_t &pingpongFlag,
        Arch::CrossCoreFlag* waitFlag = nullptr,
        Arch::CrossCoreFlag* setFlag = nullptr
        )
    {
        static constexpr uint32_t ROW_TILE = 16;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = rowBegin + rowsPerSubBlock;
        if (rowEnd > mActual) {
            rowEnd = mActual;
        }
        if (rowBegin >= mActual) {
            if (waitFlag) Arch::CrossCoreWaitFlag(*waitFlag);
            if (setFlag) Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(*setFlag);
            return;
        }

        AscendC::ResetMask();
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::DataCopyParams gfloatUbParams{1, static_cast<uint16_t>(mActual * sizeof(float)), 0, 0};
        AscendC::DataCopyParams gInputUbParams{1, static_cast<uint16_t>(mActual * sizeof(GElementInput)), 0, 0};
        AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

        AscendC::LocalTensor<float> gUbTensor = (pingpongFlag == 0) ? gUbTensorPing : gUbTensorPong;
        AscendC::LocalTensor<GElementInput> gUbFPTensor = (pingpongFlag == 0) ? gUbFPTensorPing : gUbFPTensorPong;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        if constexpr(std::is_same<GElementInput, float>::value) {
            AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gfloatUbParams, gUbPadParams);
        } else {
            AscendC::DataCopyPad(gUbFPTensor, gInputThisSubBlock, gInputUbParams, gUbPadParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        if constexpr(!std::is_same<GElementInput, float>::value) {
            AscendC::Cast(gUbTensor, gUbFPTensor, AscendC::RoundMode::CAST_NONE, mActual);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t rowStart = rowBegin;
        while (rowStart < rowEnd) {
            uint32_t alignExtra = rowStart & 7;
            uint32_t maxRowsThisTile = ROW_TILE - alignExtra;
            uint32_t rowsThisTile = rowEnd - rowStart;
            if (rowsThisTile > maxRowsThisTile) {
                rowsThisTile = maxRowsThisTile;
            }

            // Broadcast 物化 exp(g)：其发出位置在 h/a 的 MTE2 传输期间（免费），
            // V256(Wide/GM) 路径保留此结构（V128 路径已改为 VF 内 BRC_B32）
            uint32_t gbrcRealStart = rowStart & ~7;
            uint32_t gbrcRealProcess = alignExtra + rowsThisTile;
            uint32_t gbrcEffStart = alignExtra;
            uint32_t dstShape_[2] = {gbrcRealProcess, nActual};
            uint32_t srcShape_[2] = {gbrcRealProcess, 1};

            AscendC::GlobalTensor<HElementOutput> hOutputThisTile = hOutput[rowStart * outputStride];
            auto attnInputThisTile = attnInput[rowStart * nActual];
            auto hInputThisTile = hInput[rowStart * nActual];

            AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
            AscendC::LocalTensor<float> hUbTensor = (pingpongFlag == 0) ? hUbTensorPing : hUbTensorPong;
            AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
            AscendC::LocalTensor<HElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
            AscendC::LocalTensor<HElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            if (waitFlag && rowStart == rowBegin) Arch::CrossCoreWaitFlag(*waitFlag);
            AscendC::DataCopy(hUbTensor, hInputThisTile, rowsThisTile * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
            AscendC::DataCopy(aUbTensor, attnInputThisTile, rowsThisTile * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
            if (setFlag && rowStart + rowsThisTile >= rowEnd) {
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(*setFlag);
            }

            AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gUbTensor[gbrcRealStart], dstShape_, srcShape_, shareUbTensor);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
            {
                auto outAddr = reinterpret_cast<uint64_t>(outUbTensor.GetPhyAddr());
                auto hAddr = reinterpret_cast<uint64_t>(hUbTensor.GetPhyAddr());
                auto aAddr = reinterpret_cast<uint64_t>(aUbTensor.GetPhyAddr());
                auto expAddr = reinterpret_cast<uint64_t>(gbrcLeftcastUbTensor.GetPhyAddr()) + gbrcEffStart * nActual * sizeof(float);
                OutputFusedVfWide((__ubuf__ float*)outAddr,
                                  (__ubuf__ float*)hAddr,
                                  (__ubuf__ float*)aAddr,
                                  (__ubuf__ float*)expAddr,
                                  rowsThisTile, nActual, scale);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
            if(std::is_same<HElementOutput, half>::value)
            {
                AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, rowsThisTile * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                CopyOutputToGm(hOutputThisTile, outUbFPTensor, rowsThisTile, nActual, outputStride);
            }
            else
            {
                AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, rowsThisTile * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                CopyOutputToGm(hOutputThisTile, outUbBFTensor, rowsThisTile, nActual, outputStride);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
            pingpongFlag = 1 - pingpongFlag;
            rowStart += rowsThisTile;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    /// Entry point, parameterized over the work-input storage class:
    /// GlobalTensor (GM workspace path, V256 + fallback) or LocalTensor
    /// (l0c2ub UB-slot path, V128). Reading a slot is only legal after
    /// waitFlag (cube3Done) fires.
    template <class WorkInputA, class WorkInputH>
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        WorkInputA attnInput,
        WorkInputH hInput,
        float scale,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vBlockDim,
        uint32_t vHeadDim,
        uint32_t &pingpongFlag
        , uint32_t batchIdx, uint32_t headIdx, uint32_t chunkIdx,
        Arch::CrossCoreFlag* waitFlag = nullptr,
        Arch::CrossCoreFlag* setFlag = nullptr
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nActual = vBlockDim;
        if (nActual > 128) {
            ProcessWideOutput(hOutput, gInput, attnInput, hInput, scale, mActual, nActual, vHeadDim, pingpongFlag, waitFlag, setFlag);
            return;
        }
        ProcessOutput(hOutput, gInput, attnInput, hInput, scale, mActual, nActual, vHeadDim, pingpongFlag, waitFlag, setFlag);
    }

    /// V128 narrow path, parameterized over the work-input storage class:
    /// GlobalTensor (GM workspace path) or LocalTensor (l0c2ub UB-slot path).
    /// Reading the slot is only legal after waitFlag (cube3Done) fires.
    template <class WorkInputA, class WorkInputH>
    CATLASS_DEVICE
    void ProcessOutput(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        WorkInputA attnInput,
        WorkInputH hInput,
        float scale,
        uint32_t mActual,
        uint32_t nActual,
        uint32_t outputStride,
        uint32_t &pingpongFlag,
        Arch::CrossCoreFlag* waitFlag = nullptr,
        Arch::CrossCoreFlag* setFlag = nullptr
        )
    {
        // l0c2ub: with LocalTensor work inputs the slot IS the a/h buffer — no GM->UB
        // MTE2 copy (Fixpipe wrote the slot directly on the Cube side).
        constexpr bool kWorkFromUb = std::is_same_v<WorkInputA, AscendC::LocalTensor<AElementInput>>;
        uint32_t alignedM = CeilDiv(nActual, 8) * 8;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetA = mOffset * nActual + nOffset;

        uint32_t gbrcStart;
        if(mActualThisSubBlock <= 32)
        {
            if(subBlockIdx == 0)
            {
                gbrcStart = 0;
            }
            else
            {
                gbrcStart = mActualPerSubBlock;
            }

            AscendC::ResetMask();
            // l0c2ub slot inputs: Fixpipe SPLIT_M delivers each subBlock's own row
            // half at ITS OWN UB slot base (subBlock0 rows -> UB[0:], subBlock1 rows
            // -> UB[0:] of subBlock1's view). So the slot needs NO row offset here;
            // attnInput/hInput must be taken at offset 0 on the UB path.
            auto attnInputThisSubBlock = kWorkFromUb ? attnInput[0] : attnInput[gbrcStart * nActual];
            auto hInputThisSubBlock = kWorkFromUb ? hInput[0] : hInput[gbrcStart * nActual];
            AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
            AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[gbrcStart * nActual];

            AscendC::DataCopyParams gfloatUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
            AscendC::DataCopyParams ghalfUbParams{1, (uint16_t)(mActual*sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

            // With slot inputs, the ping/pong pair carries the same stage slot the Cube
            // side just filled; subBlock rows live at the SPLIT_M half offset.
            AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
            AscendC::LocalTensor<float> hUbTensor = (pingpongFlag == 0) ? hUbTensorPing : hUbTensorPong;
            AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
            AscendC::LocalTensor<HElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
            AscendC::LocalTensor<HElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;
            AscendC::LocalTensor<float> gUbTensor = (pingpongFlag == 0) ? gUbTensorPing : gUbTensorPong;
            AscendC::LocalTensor<GElementInput> gUbFPTensor = (pingpongFlag == 0) ? gUbFPTensorPing : gUbFPTensorPong;
            AscendC::LocalTensor<GElementInput> gUbBFTensor = (pingpongFlag == 0) ? gUbBFTensorPing : gUbBFTensorPong;
            if constexpr (kWorkFromUb) {
                // SPLIT_M wrote subBlock0's rows to slot[0:rows) and subBlock1's rows to
                // slot[rows:2*rows) of this subBlock's own UB — no offset needed here.
                aUbTensor = attnInputThisSubBlock;
                hUbTensor = hInputThisSubBlock;
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

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

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            if (waitFlag) Arch::CrossCoreWaitFlag(*waitFlag);
            if constexpr (!kWorkFromUb) {
                AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            } else {
                // Slot data is already in UB (Fixpipe wrote it); satisfy the tail's
                // MTE2_V wait without an actual MTE2 transfer.
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            }

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
            if constexpr (!kWorkFromUb) {
                AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                // GM path: release the slot-ownership flag right after the copy is
                // issued (early release, pre-existing behavior).
                if (setFlag) Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(*setFlag);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                // l0c2ub path: do NOT release early — the Cube's next-round Fixpipe
                // writes this very UB slot, racing the VF reads below (UB write vs
                // vector read has no implicit ordering). Release after consumption.
            }

            AscendC::Exp(gUbTensor, gUbTensor, mActual);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
            {
                auto outAddr = reinterpret_cast<uint64_t>(outUbTensor.GetPhyAddr());
                auto hAddr = reinterpret_cast<uint64_t>(hUbTensor.GetPhyAddr());
                auto aAddr = reinterpret_cast<uint64_t>(aUbTensor.GetPhyAddr());
                auto gExpAddr = reinterpret_cast<uint64_t>(gUbTensor.GetPhyAddr()) + gbrcStart * sizeof(float);
                OutputFusedVf((__ubuf__ float*)outAddr,
                               (__ubuf__ float*)hAddr,
                               (__ubuf__ float*)aAddr,
                               (__ubuf__ float*)gExpAddr,
                               mActualThisSubBlock, nActual, scale);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (kWorkFromUb) {
                // l0c2ub: slot fully consumed — release slot ownership now. The slot
                // is consumed by VECTOR reads (OutputFusedVf), so gate the notify on
                // PIPE_V (a PipeBarrier<PIPE_V> precedes this point either way, making
                // the pipe choice functionally equivalent — see
                // docs/agents/chunk_fwd_o_intermittent_accuracy_fix.md §4.2).
                if (setFlag) Arch::CrossCoreSetFlag<0x2, PIPE_V>(*setFlag);
            }
            if(std::is_same<HElementOutput, half>::value)
            {
                AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::DataCopy(hOutputThisSubBlock, outUbFPTensor, mActualThisSubBlock * nActual);
            }
            else
            {
                AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::DataCopy(hOutputThisSubBlock, outUbBFTensor, mActualThisSubBlock * nActual);
            }
            pingpongFlag = 1 - pingpongFlag;
        }
        else
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
            AscendC::Exp(gUbTensor, gUbTensor, mActual);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t mActualPerStage = CeilDiv(mActualThisSubBlock, 2);
            uint32_t mActualThisStage = 0;
            for(uint32_t stage = 0; stage < 2; stage++)
            {
                if(stage == 0) mActualThisStage = mActualPerStage;
                else mActualThisStage = mActualThisSubBlock - mActualPerStage;

                if(subBlockIdx == 0 && stage == 0)
                {
                    gbrcStart = 0;
                }
                else if(subBlockIdx == 0 && stage == 1)
                {
                    gbrcStart = mActualPerStage;
                }
                else if(subBlockIdx == 1 && stage == 0)
                {
                    gbrcStart = mActualPerSubBlock;
                }
                else if(subBlockIdx == 1 && stage == 1)
                {
                    gbrcStart = mActualPerSubBlock + mActualPerStage;
                }

                AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[gbrcStart * nActual];
                // l0c2ub: SPLIT_M already delivered this subBlock's row half at its own
                // UB slot base — take the slot at offset 0; the stage loop below walks
                // within the half via ubStageOffset.
                auto attnInputThisSubBlock = kWorkFromUb ? attnInput[0] : attnInput[gbrcStart * nActual];
                auto hInputThisSubBlock = kWorkFromUb ? hInput[0] : hInput[gbrcStart * nActual];

                AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
                AscendC::LocalTensor<float> hUbTensor = (pingpongFlag == 0) ? hUbTensorPing : hUbTensorPong;
                AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
                AscendC::LocalTensor<HElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
                AscendC::LocalTensor<HElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;
                if constexpr (kWorkFromUb) {
                    // Slot holds this subBlock's SPLIT_M half at [0, mActualThisSubBlock);
                    // stage 0 reads rows [0, mActualPerStage), stage 1 the rest.
                    uint32_t ubRowOffset = stage * mActualPerStage * nActual;
                    aUbTensor = attnInputThisSubBlock[ubRowOffset];
                    hUbTensor = hInputThisSubBlock[ubRowOffset];
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                if (waitFlag && stage == 0) Arch::CrossCoreWaitFlag(*waitFlag);
                if constexpr (!kWorkFromUb) {
                    AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisStage * nActual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
                }

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
                if constexpr (!kWorkFromUb) {
                    AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mActualThisStage * nActual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                    // GM path: keep the pre-existing early release.
                    if (setFlag && stage == 1) {
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(*setFlag);
                    }
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                    // l0c2ub: release after the final stage's VF consumption instead
                    // (Cube's next-round Fixpipe would race the slot reads).
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                if (stage == 1) {
                    // Stage 1 overwrites outUb* while stage 0's MTE3 DataCopy may still
                    // be draining its tail rows — wait for stage 0's copy to complete
                    // (signalled right after its DataCopy below).
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
                }
                {
                    auto outAddr = reinterpret_cast<uint64_t>(outUbTensor.GetPhyAddr());
                    auto hAddr = reinterpret_cast<uint64_t>(hUbTensor.GetPhyAddr());
                    auto aAddr = reinterpret_cast<uint64_t>(aUbTensor.GetPhyAddr());
                    auto gExpAddr = reinterpret_cast<uint64_t>(gUbTensor.GetPhyAddr()) + gbrcStart * sizeof(float);
                    OutputFusedVf((__ubuf__ float*)outAddr,
                                   (__ubuf__ float*)hAddr,
                                   (__ubuf__ float*)aAddr,
                                   (__ubuf__ float*)gExpAddr,
                                   mActualThisStage, nActual, scale);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if constexpr (kWorkFromUb) {
                    // l0c2ub: slot fully consumed at the end of the last stage; gate
                    // on PIPE_V like the <=32 branch (Vector reads consume the slot).
                    if (setFlag && stage == 1) {
                        Arch::CrossCoreSetFlag<0x2, PIPE_V>(*setFlag);
                    }
                }

                if(std::is_same<HElementOutput, half>::value)
                {
                    AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisStage * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::DataCopy(hOutputThisSubBlock, outUbFPTensor, mActualThisStage * nActual);
                }
                else
                {
                    AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisStage * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::DataCopy(hOutputThisSubBlock, outUbBFTensor, mActualThisStage * nActual);
                }
                if (stage == 0) {
                    // MTE3-V reverse sync: let stage 0's DataCopy finish before stage 1's
                    // VF overwrites the (formerly shared) outUb* scratch rows.
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
                }
                pingpongFlag = 1 - pingpongFlag;
            }
        }
    }

private:
    AscendC::LocalTensor<float> maskUbTensor;
    AscendC::LocalTensor<float> gbrcLeftcastUbTensor;
    AscendC::LocalTensor<float> gbrcUpUbTensor;
    AscendC::LocalTensor<uint8_t> shareUbTensor;

    AscendC::LocalTensor<float> gUbTensorPing;
    AscendC::LocalTensor<GElementInput> gUbFPTensorPing;
    AscendC::LocalTensor<GElementInput> gUbBFTensorPing;
    AscendC::LocalTensor<float> aUbTensorPing;
    AscendC::LocalTensor<float> hUbTensorPing;
    AscendC::LocalTensor<float> outUbTensorPing;
    AscendC::LocalTensor<HElementOutput> outUbFPTensorPing;
    AscendC::LocalTensor<HElementOutput> outUbBFTensorPing;

    AscendC::LocalTensor<float> gUbTensorPong;
    AscendC::LocalTensor<GElementInput> gUbFPTensorPong;
    AscendC::LocalTensor<GElementInput> gUbBFTensorPong;
    AscendC::LocalTensor<float> aUbTensorPong;
    AscendC::LocalTensor<float> hUbTensorPong;
    AscendC::LocalTensor<float> outUbTensorPong;
    AscendC::LocalTensor<HElementOutput> outUbFPTensorPong;
    AscendC::LocalTensor<HElementOutput> outUbBFTensorPong;


};
}

#endif
