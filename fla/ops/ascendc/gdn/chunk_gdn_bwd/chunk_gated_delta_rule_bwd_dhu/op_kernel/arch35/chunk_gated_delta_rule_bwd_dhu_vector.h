/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_vector.h
 * \brief A5 vector path for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "adv_api/utils/init_global_memory.h"
#include "kernel_utils/vector/regbase.hpp"
#include "chunk_gated_delta_rule_bwd_dhu_common.h"
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"

namespace GDN {

using namespace AscendC::MicroAPI;

template <typename CopyType>
__simd_vf__ inline void CastLocalToFloatRegbase(__ubuf__ float *dst, __ubuf__ CopyType *src, uint16_t elements)
{
    const uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(CopyType);
    const uint16_t loopCnt = static_cast<uint16_t>((elements + eleNumPerVf - 1) / eleNumPerVf);
    const uint16_t pairLoopCnt = loopCnt / 2;
    const uint16_t hasSingleLoop = loopCnt % 2;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    if constexpr (std::is_same<CopyType, float>::value) {
        RegTensor<float> srcReg;
        for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
            const uint32_t elemOffset = loopIdx * eleNumPerVf;
            LoadAlign(srcReg, src + elemOffset);
            StoreAlign(dst + elemOffset, srcReg, maskFull32);
        }
    } else {
        RegTensor<CopyType> srcReg0;
        RegTensor<CopyType> srcReg1;
        RegTensor<float> srcZeroReg0;
        RegTensor<float> srcOneReg0;
        RegTensor<float> srcZeroReg1;
        RegTensor<float> srcOneReg1;
        for (uint16_t pairIdx = 0; pairIdx < pairLoopCnt; ++pairIdx) {
            const uint32_t elemOffset0 = pairIdx * 2 * eleNumPerVf;
            const uint32_t elemOffset1 = elemOffset0 + eleNumPerVf;
            LoadIn<CopyType, false>(srcReg0, src + elemOffset0);
            LoadIn<CopyType, false>(srcReg1, src + elemOffset1);
            CastHalf2Float<CopyType>(srcZeroReg0, srcOneReg0, srcReg0, maskFull16);
            CastHalf2Float<CopyType>(srcZeroReg1, srcOneReg1, srcReg1, maskFull16);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst + elemOffset0, srcZeroReg0, srcOneReg0, maskFull32);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst + elemOffset1, srcZeroReg1, srcOneReg1, maskFull32);
        }
        for (uint16_t singleIdx = 0; singleIdx < hasSingleLoop; ++singleIdx) {
            const uint32_t elemOffset = pairLoopCnt * 2 * eleNumPerVf;
            LoadIn<CopyType, false>(srcReg0, src + elemOffset);
            CastHalf2Float<CopyType>(srcZeroReg0, srcOneReg0, srcReg0, maskFull16);
            StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst + elemOffset, srcZeroReg0, srcOneReg0, maskFull32);
        }
    }
}

__simd_vf__ inline void FillFloatRegbase(__ubuf__ float *dst, float value, uint16_t elements)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loopCnt = static_cast<uint16_t>((elements + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> dstReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    Duplicate(dstReg, value, maskFull);
    for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        const uint32_t elemOffset = loopIdx * ELEMS_PER_VF;
        StoreAlign(dst + elemOffset, dstReg, maskFull);
    }
}

__simd_vf__ inline void GateDualFactorFuseRegbase(__ubuf__ float *gateFactor, __ubuf__ float *dvGateFactor,
                                                  __ubuf__ float *gateRaw, uint16_t elements,
                                                  __ubuf__ float *gateLast, bool useExp2)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loopCnt = static_cast<uint16_t>((elements + ELEMS_PER_VF - 1) / ELEMS_PER_VF);
    const float ln2 = 0.69314718055994530942f;

    RegTensor<float> srcReg;
    RegTensor<float> lastReg;
    RegTensor<float> factorReg;
    RegTensor<float> dvReg;
    MaskReg maskLoop;
    LoadIn<float, true>(lastReg, gateLast);
    for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        const uint32_t elemOffset = loopIdx * ELEMS_PER_VF;
        uint32_t curElems = elements - elemOffset > ELEMS_PER_VF ? ELEMS_PER_VF : elements - elemOffset;
        maskLoop = UpdateMask<float>(curElems);
        LoadAlign(srcReg, gateRaw + elemOffset);
        Sub(dvReg, lastReg, srcReg, maskLoop);
        if (useExp2) {
            Muls(dvReg, dvReg, ln2, maskLoop);
            Muls(factorReg, srcReg, ln2, maskLoop);
        } else {
            factorReg = srcReg;  // useExp2=false：gateFactor = exp(g)，srcReg 即操作数
        }
        Exp(factorReg, factorReg, maskLoop);
        Exp(dvReg, dvReg, maskLoop);
        StoreAlign(gateFactor + elemOffset, factorReg, maskLoop);
        StoreAlign(dvGateFactor + elemOffset, dvReg, maskLoop);
    }
}

__simd_vf__ inline void AddFloatTwoRegbase(__ubuf__ float *dst, __ubuf__ float *src1,
                                           __ubuf__ float *src2, uint16_t elements)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loopCnt = static_cast<uint16_t>((elements + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> dstReg;
    RegTensor<float> srcReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    #pragma unroll 2
    for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        const uint32_t elemOffset = loopIdx * ELEMS_PER_VF;
        LoadAlign(dstReg, dst + elemOffset);
        LoadAlign(srcReg, src2 + elemOffset);
        Add(dstReg, dstReg, srcReg, maskFull);
        StoreAlign(dst + elemOffset, dstReg, maskFull);
    }
}

__simd_vf__ inline void MulScalarPtrRegbase(__ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *factor,
                                            uint16_t elements)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loopCnt = static_cast<uint16_t>((elements + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> srcReg;
    RegTensor<float> factorReg;
    RegTensor<float> dstReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    LoadIn<float, true>(factorReg, factor);
    #pragma unroll 2
    for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        const uint32_t elemOffset = loopIdx * ELEMS_PER_VF;
        LoadAlign(srcReg, src + elemOffset);
        Mul(dstReg, srcReg, factorReg, maskFull);
        StoreAlign(dst + elemOffset, dstReg, maskFull);
    }
}

__simd_vf__ inline void MulRowsByFactorsRegbase(__ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *factors,
                                                uint16_t rowCount, uint16_t colCount)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t colLoop = static_cast<uint16_t>((colCount + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> srcReg;
    RegTensor<float> factorReg;
    RegTensor<float> dstReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    #pragma unroll 2
    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(factorReg, factors + row);
        for (uint16_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            const uint32_t colOffset = colIdx * ELEMS_PER_VF;
            const uint32_t elemOffset = row * colCount + colOffset;
            LoadAlign(srcReg, src + elemOffset);
            Mul(dstReg, srcReg, factorReg, maskFull);
            StoreAlign(dst + elemOffset, dstReg, maskFull);
        }
    }
}

__simd_vf__ inline void MulRowsByFactorsAddRegbase(__ubuf__ float *dst, __ubuf__ float *src,
                                                   __ubuf__ float *factors, __ubuf__ float *add,
                                                   uint16_t rowCount, uint16_t colCount)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t colLoop = static_cast<uint16_t>((colCount + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> srcReg;
    RegTensor<float> factorReg;
    RegTensor<float> addReg;
    RegTensor<float> dstReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    #pragma unroll 2
    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(factorReg, factors + row);
        for (uint16_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            const uint32_t colOffset = colIdx * ELEMS_PER_VF;
            const uint32_t elemOffset = row * colCount + colOffset;
            LoadAlign(srcReg, src + elemOffset);
            LoadAlign(addReg, add + elemOffset);
            Mul(dstReg, srcReg, factorReg, maskFull);
            Add(dstReg, dstReg, addReg, maskFull);
            StoreAlign(dst + elemOffset, dstReg, maskFull);
        }
    }
}

__simd_vf__ inline void StateUpdateFuseRegbase(__ubuf__ float *state, __ubuf__ float *termQ,
                                               __ubuf__ float *termW, float scale, uint16_t elements)
{
    // 单 pass 完成 state += termQ * scale - termW（替换 Muls/Sub/Add 三个标量 API
    // 与其间的 4 个 PipeBarrier：state/termQ/termW 只经寄存器一趟）
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loopCnt = static_cast<uint16_t>((elements + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> stateReg;
    RegTensor<float> termQReg;
    RegTensor<float> termWReg;
    RegTensor<float> tmpReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    #pragma unroll 2
    for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        const uint32_t elemOffset = loopIdx * ELEMS_PER_VF;
        LoadAlign(stateReg, state + elemOffset);
        LoadAlign(termQReg, termQ + elemOffset);
        LoadAlign(termWReg, termW + elemOffset);
        Muls(tmpReg, termQReg, scale, maskFull);
        Sub(tmpReg, tmpReg, termWReg, maskFull);
        Add(stateReg, stateReg, tmpReg, maskFull);
        StoreAlign(state + elemOffset, stateReg, maskFull);
    }
}

// fp32 → DT 输出 cast 的 CastTrait：与原 AscendC::Cast(CAST_RINT) 舍入语义一致
template <typename DT>
constexpr CastTrait BWD_DHU_FP32_TO_DT_PACK = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename DT>
__simd_vf__ inline void CastFp32ToOutputRegbase(__ubuf__ DT *dst, __ubuf__ float *src, uint16_t elements)
{
    // fp32 → DT 寄存器内 cast 直写输出缓冲（替换 CopyOutFp32Rows 的标量
    // AscendC::Cast；fp32 读一趟、DT 写一趟，无中间 API 边界）
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loopCnt = static_cast<uint16_t>((elements + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> srcReg;
    RegTensor<DT> dstReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    #pragma unroll 2
    for (uint16_t loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
        const uint32_t elemOffset = loopIdx * ELEMS_PER_VF;
        LoadAlign(srcReg, src + elemOffset);
        Cast<DT, float, BWD_DHU_FP32_TO_DT_PACK<DT>>(dstReg, srcReg, maskFull);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dst + elemOffset, dstReg, maskFull);
    }
}

template <typename DT>
__simd_vf__ inline void MulRowsByFactorsAddCastOutRegbase(__ubuf__ DT *dst, __ubuf__ float *src,
                                                          __ubuf__ float *factors, __ubuf__ float *add,
                                                          uint16_t rowCount, uint16_t colCount)
{
    // dv2 尾部融合单 pass：×gate 因子 + 加 dv + fp32→DT cast 直写 outputBuf。
    // 替换 MulRowsByFactorsAddRegbase（fp32 落 UB）+ PipeBarrier +
    // CopyOutFp32Rows 的 cast 段——fp32 中间量只经寄存器不落 UB。
    // RINT 舍入与原 AscendC::Cast(CAST_RINT) 一致（trait 同 opt2）。
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t colLoop = static_cast<uint16_t>((colCount + ELEMS_PER_VF - 1) / ELEMS_PER_VF);

    RegTensor<float> srcReg;
    RegTensor<float> factorReg;
    RegTensor<float> addReg;
    RegTensor<float> tmpReg;
    RegTensor<DT> dstReg;
    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    #pragma unroll 2
    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadIn<float, true>(factorReg, factors + row);
        for (uint16_t colIdx = 0; colIdx < colLoop; ++colIdx) {
            const uint32_t colOffset = colIdx * ELEMS_PER_VF;
            const uint32_t elemOffset = row * colCount + colOffset;
            LoadAlign(srcReg, src + elemOffset);
            LoadAlign(addReg, add + elemOffset);
            Mul(tmpReg, srcReg, factorReg, maskFull32);
            Add(tmpReg, tmpReg, addReg, maskFull32);
            Cast<DT, float, BWD_DHU_FP32_TO_DT_PACK<DT>>(dstReg, tmpReg, maskFull32);
            StoreAlign<DT, StoreDist::DIST_PACK_B32>(dst + elemOffset, dstReg, maskFull32);
        }
    }
}

template <typename DT, typename GT, int USE_GK>
class ChunkGatedDeltaRuleBwdDhuVector {
public:
    __aicore__ inline ChunkGatedDeltaRuleBwdDhuVector() = default;

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR gate, GM_ADDR dv, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dh,
                                GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace,
                                const ChunkGatedDeltaRuleBwdDhuTilingData *__restrict tilingData,
                                AscendC::TPipe *pipe)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(q));
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(gate));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dv));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh));
        if (tilingData->hasDh0 != 0 && dh0 != nullptr) {
            dh0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh0));
            dh0Addr_ = dh0;
            hasDh0_ = true;
        }
        dv2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dv2));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace));
        workspaceStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        tiling_ = tilingData;
        pipe_ = pipe;
        B_ = tiling_->B;
        HK_ = tiling_->HK;
        HV_ = tiling_->HV;
        T_ = tiling_->T;
        K_ = tiling_->K;
        V_ = tiling_->V;
        HRatio_ = tiling_->HRatio;
        chunkSize_ = tiling_->chunkSize;
        totalChunkNum_ = tiling_->totalChunkNum;
        headsPerTask_ = tiling_->headsPerTask;
        headWindowNum_ = tiling_->headWindowNum;
        taskNum_ = tiling_->taskNum;
        isVariable_ = tiling_->isVariable;
        stateVFirst_ = tiling_->stateVFirst != 0;
        scale_ = tiling_->scale;
        stateWorkspaceOffset_ = tiling_->stateWorkspaceOffset;
        dvStateWorkspaceOffset_ = tiling_->dvStateWorkspaceOffset;
        termQWorkspaceOffset_ = tiling_->termQWorkspaceOffset;
        termWWorkspaceOffset_ = tiling_->termWWorkspaceOffset;
        workspaceElemsPerSubBlock_ = tiling_->workspaceElemsPerSubBlock;
        dh0ClearCoreNum_ = tiling_->dh0ClearCoreNum;
        dh0ClearElemsPerCore_ = tiling_->dh0ClearElemsPerCore;
        dh0ClearTailElems_ = tiling_->dh0ClearTailElems;
        vecRow_ = tiling_->vecRow > 0 ? tiling_->vecRow : 8;
        gateElems_ = K_ > chunkSize_ ? K_ : chunkSize_;
        subBlockNum_ = static_cast<int64_t>(AscendC::GetSubBlockNum());
        if (subBlockNum_ <= 0) {
            subBlockNum_ = 1;
        }
        subBlockIdx_ = static_cast<int64_t>(AscendC::GetSubBlockIdx());
        if (subBlockIdx_ < 0 || subBlockIdx_ >= subBlockNum_) {
            subBlockIdx_ = 0;
        }
        // state UB 驻留：tiling 预算允许时启用；kernel 侧防御性复核（subBlockNum<2
        // 时 owned head 可能超 2 个槽位、K*V 超 FillFloatRegbase 的 uint16 上限、
        // stateVFirst 时驻留 dh0 分支未实现 CopyOutDh0VFirst 转置布局），
        // 复核不过则回退 GM 往返路径（tile 态 buffer 更小，预算必然够）
        stateResident_ = tiling_->stateResident != 0 && subBlockNum_ >= 2 && K_ * V_ <= 65535 &&
            !stateVFirst_;

        const int64_t inputElems = vecRow_ * (K_ > V_ ? K_ : V_);
        const int64_t outputRows = vecRow_ > 16 ? vecRow_ : 16;
        const int64_t outputElems = outputRows * (K_ > V_ ? K_ : V_);
        if constexpr (std::is_same<DT, bfloat16_t>::value) {
            pipe_->InitBuffer(matrixCvPing_, vecRow_ * V_ * static_cast<int64_t>(sizeof(DT)));
            pipe_->InitBuffer(matrixCvPong_, vecRow_ * V_ * static_cast<int64_t>(sizeof(DT)));
        }
        pipe_->InitBuffer(qInputPing_, inputElems * static_cast<int64_t>(sizeof(DT)));
        pipe_->InitBuffer(qInputPong_, inputElems * static_cast<int64_t>(sizeof(DT)));
        pipe_->InitBuffer(gInputPing_, gateElems_ * static_cast<int64_t>(sizeof(GT)));
        pipe_->InitBuffer(gInputPong_, gateElems_ * static_cast<int64_t>(sizeof(GT)));
        pipe_->InitBuffer(outputPing_, outputElems * static_cast<int64_t>(sizeof(DT)));
        pipe_->InitBuffer(outputPong_, outputElems * static_cast<int64_t>(sizeof(DT)));
        // 驻留态按全量 [K,V] fp32 分配（ping/pong 复用为 owned head 槽位）；
        // GM 往返态按行 tile
        const int64_t stateBufElems = stateResident_ ? K_ * V_ : vecRow_ * V_;
        pipe_->InitBuffer(statePing_, stateBufElems * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(statePong_, stateBufElems * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(qFp32Buf_, inputElems * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(gateFactorAllFp32_, HEADS_PER_TASK * gateElems_ * static_cast<int64_t>(sizeof(float)));
        if constexpr (USE_GK == 0) {
            pipe_->InitBuffer(gRawAllFp32_, HEADS_PER_TASK * gateElems_ * static_cast<int64_t>(sizeof(float)));
            pipe_->InitBuffer(dvGateFactorAllFp32_,
                              HEADS_PER_TASK * gateElems_ * static_cast<int64_t>(sizeof(float)));
        }
        pipe_->InitBuffer(outFp32Buf_, inputElems * static_cast<int64_t>(sizeof(float)));

        if constexpr (std::is_same<DT, bfloat16_t>::value) {
            matrixCvBuf_[0] = matrixCvPing_.template Get<DT>();
            matrixCvBuf_[1] = matrixCvPong_.template Get<DT>();
        }
        qInputBuf_[0] = qInputPing_.template Get<DT>();
        qInputBuf_[1] = qInputPong_.template Get<DT>();
        gateInputBuf_[0] = gInputPing_.template Get<GT>();
        gateInputBuf_[1] = gInputPong_.template Get<GT>();
        outputBuf_[0] = outputPing_.template Get<DT>();
        outputBuf_[1] = outputPong_.template Get<DT>();
        stateBuf_[0] = statePing_.template Get<float>();
        stateBuf_[1] = statePong_.template Get<float>();

        InitVectorEvents();
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t qgL1PaddedRows = 128;
        constexpr uint32_t matrixTileBytes = qgL1PaddedRows * 128 * sizeof(DT);
        constexpr uint32_t qgL1ScratchOffset = 4 * matrixTileBytes;
        AscendC::LocalTensor<uint8_t> l1Buffer(AscendC::TPosition::A1, 0, 512 * 1024);
        AscendC::LocalTensor<DT> qgL1Scratch[HEADS_PER_TASK] = {
            l1Buffer[qgL1ScratchOffset].template ReinterpretCast<DT>(),
            l1Buffer[qgL1ScratchOffset + matrixTileBytes].template ReinterpretCast<DT>(),
            l1Buffer[qgL1ScratchOffset + 2 * matrixTileBytes].template ReinterpretCast<DT>(),
            l1Buffer[qgL1ScratchOffset + 3 * matrixTileBytes].template ReinterpretCast<DT>()};

        if (hasDh0_) {
            const int64_t vecBlockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
            if (vecBlockIdx >= 0 && vecBlockIdx < dh0ClearCoreNum_) {
                int64_t clearOffset = vecBlockIdx * dh0ClearElemsPerCore_;
                int64_t clearElems = dh0ClearElemsPerCore_;
                if (vecBlockIdx + 1 == dh0ClearCoreNum_) {
                    clearOffset = (dh0ClearCoreNum_ - 1) * dh0ClearElemsPerCore_;
                    clearElems = dh0ClearTailElems_;
                }
                if (clearElems > 0) {
                    if constexpr (sizeof(DT) == sizeof(uint16_t)) {
                        AscendC::GlobalTensor<uint16_t> dh0ClearGm;
                        dh0ClearGm.SetGlobalBuffer(
                            reinterpret_cast<__gm__ uint16_t *>(dh0Addr_) + clearOffset);
                        AscendC::Fill(dh0ClearGm, static_cast<uint64_t>(clearElems),
                                      static_cast<uint16_t>(0));
                    } else {
                        AscendC::GlobalTensor<uint32_t> dh0ClearGm;
                        dh0ClearGm.SetGlobalBuffer(
                            reinterpret_cast<__gm__ uint32_t *>(dh0Addr_) + clearOffset);
                        AscendC::Fill(dh0ClearGm, static_cast<uint64_t>(clearElems),
                                      static_cast<uint32_t>(0));
                    }
                }
            }
            AscendC::SyncAll<true>();
        }

        const int64_t coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx() / subBlockNum_);
        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());

        for (int64_t taskIdx = coreIdx; taskIdx < taskNum_; taskIdx += blockNum) {
            const int64_t seqIdx = taskIdx / headWindowNum_;
            const int64_t headWindowIdx = taskIdx - seqIdx * headWindowNum_;
            const int64_t hvBase = headWindowIdx * headsPerTask_;
            const int64_t headCnt = Min(headsPerTask_, HV_ - hvBase);
            const int64_t taskRound = (taskIdx - coreIdx) / blockNum;
            const int64_t windowStartSlot = (taskRound & 1) * HEADS_PER_TASK;
            if (headCnt <= 0) {
                continue;
            }

            SeqInfo seqInfo;
            GetSeqInfo(cuSeqlens_, *tiling_, seqIdx, seqInfo);
            if (!seqInfo.valid) {
                continue;
            }

            for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                if (headOffset % subBlockNum_ != subBlockIdx_) {
                    continue;
                }
                if (stateResident_) {
                    // 驻留态：直接清 UB 槽位（workspace state 段不再读写）
                    AscendC::LocalTensor<float> stateFp32 = stateBuf_[StateResidentSlot(headOffset)];
                    FillFloatRegbase((__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                     0.0f, static_cast<uint16_t>(K_ * V_));
                    AscendC::PipeBarrier<PIPE_V>();
                    continue;
                }
                const int64_t workspaceBase = WorkspaceBase(coreIdx, windowStartSlot + headOffset);
                const int64_t stateBase = StateWorkspaceFloatOffset(workspaceBase, 0);
                for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                    const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                    const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                    const uint32_t stateIdx = curStatePingPong_;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[stateIdx]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[stateIdx]);
                    AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                    FillFloatRegbase((__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()), 0.0f,
                                     static_cast<uint16_t>(elems));
                    AscendC::PipeBarrier<PIPE_V>();
                    CopyOutStateRows(stateIdx, stateFp32, stateBase + rowOffset * V_, elems);
                    curStatePingPong_ ^= 1U;
                }
            }

            for (int64_t chunkIdx = seqInfo.chunkCnt - 1; chunkIdx >= 0; --chunkIdx) {
                ChunkInfo chunkInfo;
                GetChunkInfoBySeqChunk(chunkIndices_, *tiling_, seqInfo, chunkIdx, chunkInfo);
                if (!chunkInfo.valid) {
                    continue;
                }

                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    const int64_t hv = hvBase + headOffset;
                    const int64_t hq = hv / HRatio_;
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, workspaceSlot);
                    const int64_t stateBase = StateWorkspaceFloatOffset(workspaceBase, 0);
                    const int64_t qBase =
                        ((chunkInfo.bIdx * HK_ + hq) * T_ + chunkInfo.tokenStart) * K_;
                    const int64_t dhBase = DhOffset(chunkInfo.bIdx, hv, chunkInfo.outputChunkIdx);
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        // C3 配平：非归属 subblock 同点空转 set dhReady（cube 侧
                        // 两个 mode-2 wait 各需两个 AIV 的计数）
                        Catlass::Arch::CrossCoreFlag dhReadyFlag{DH_READY_FLAG};
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(dhReadyFlag);
                        continue;
                    }
                    AscendC::LocalTensor<float> gateFactor =
                        gateFactorAllFp32_.template Get<float>()[headOffset * gateElems_];
                    if constexpr (USE_GK == 0) {
                        AscendC::LocalTensor<float> gateRaw =
                            gRawAllFp32_.template Get<float>()[headOffset * gateElems_];
                        AscendC::LocalTensor<float> dvGateFactor =
                            dvGateFactorAllFp32_.template Get<float>()[headOffset * gateElems_];
                        const int64_t gateBase = (chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart;
                        const uint32_t gateIdx = CopyInGateRows(
                            gateGm_, gateInputBuf_[curGateInputPingPong_], gateBase,
                            static_cast<uint32_t>(chunkInfo.chunkLen));
                        CastGateInputRows(gateRaw, gateInputBuf_[gateIdx],
                                          static_cast<uint32_t>(chunkInfo.chunkLen), gateIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        const int64_t lastRow = chunkInfo.chunkLen - 1;
                        GateDualFactorFuseRegbase(
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(gateFactor.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(dvGateFactor.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(gateRaw.GetPhyAddr()),
                            static_cast<uint16_t>(chunkInfo.chunkLen),
                            ((__ubuf__ float *)reinterpret_cast<uint64_t>(gateRaw.GetPhyAddr())) + lastRow,
                            tiling_->useExp2 != 0);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        const int64_t lastToken = chunkInfo.tokenStart + chunkInfo.chunkLen - 1;
                        const int64_t gateBase = ((chunkInfo.bIdx * HV_ + hv) * T_ + lastToken) * K_;
                        const uint32_t gateIdx = CopyInGateRows(
                            gateGm_, gateInputBuf_[curGateInputPingPong_], gateBase,
                            static_cast<uint32_t>(K_));
                        CastGateInputRows(gateFactor, gateInputBuf_[gateIdx], static_cast<uint32_t>(K_), gateIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(gateFactor, gateFactor, LN2, static_cast<uint32_t>(K_));
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(gateFactor, gateFactor, static_cast<uint32_t>(K_));
                        AscendC::PipeBarrier<PIPE_V>();
                    }

                    for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                        const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                        if (stateResident_) {
                            // 驻留态：state 常驻 UB 槽位，行 tile 原位读改（无 GM 往返）。
                            // dh 输出 decay 前状态、随后原位 decay，均由 V pipe 顺序保证
                            AscendC::LocalTensor<float> stateFp32 =
                                stateBuf_[StateResidentSlot(headOffset)][rowOffset * V_];
                            CopyOutFp32Rows(dhGm_, stateFp32, dhBase + rowOffset * V_, elems);
                            if constexpr (USE_GK == 0) {
                                const int64_t lastRow = chunkInfo.chunkLen - 1;
                                MulScalarPtrRegbase(
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                    ((__ubuf__ float *)reinterpret_cast<uint64_t>(gateFactor.GetPhyAddr())) + lastRow,
                                    static_cast<uint16_t>(elems));
                            } else {
                                MulRowsByFactorsRegbase(
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                    ((__ubuf__ float *)reinterpret_cast<uint64_t>(gateFactor.GetPhyAddr())) + rowOffset,
                                    static_cast<uint16_t>(curRows), static_cast<uint16_t>(V_));
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            continue;
                        }
                        const uint32_t stateIdx = CopyInStateRows(
                            stateBuf_[curStatePingPong_], stateBase + rowOffset * V_, elems);
                        AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[stateIdx]);
                        CopyOutFp32Rows(dhGm_, stateFp32, dhBase + rowOffset * V_, elems);
                        if constexpr (USE_GK == 0) {
                            const int64_t lastRow = chunkInfo.chunkLen - 1;
                            MulScalarPtrRegbase(
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                ((__ubuf__ float *)reinterpret_cast<uint64_t>(gateFactor.GetPhyAddr())) + lastRow,
                                static_cast<uint16_t>(elems));
                        } else {
                            MulRowsByFactorsRegbase(
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                ((__ubuf__ float *)reinterpret_cast<uint64_t>(gateFactor.GetPhyAddr())) + rowOffset,
                                static_cast<uint16_t>(curRows), static_cast<uint16_t>(V_));
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        CopyOutStateRows(stateIdx, stateFp32, stateBase + rowOffset * V_, elems);
                    }

                    // C3：dh 已全部写 GM（上面 state 行循环的 CopyOutFp32Rows 完成），
                    // GEMM0 唯一跨核输入就绪——先于 qg 生成/dvGateFactor 发射 dhReady，
                    // 让 cube 的 GEMM0 与本 AIV 的 qg 生成重叠。PIPE_MTE3 与 dh 的
                    // MTE3 写同 pipe 保序。
                    Catlass::Arch::CrossCoreFlag dhReadyFlag{DH_READY_FLAG};
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(dhReadyFlag);

                    constexpr uint32_t c0Elems = 32 / sizeof(DT);
                    AscendC::DataCopyEnhancedParams qgCopyEnhanced;
                    qgCopyEnhanced.blockMode = AscendC::BlockMode::BLOCK_MODE_VECTOR;
                    uint32_t qgScratchSlot = static_cast<uint32_t>(headOffset);
                    bool produceQG = true;
                    if constexpr (USE_GK != 0) {
                        const int64_t groupStartHv = hq * HRatio_;
                        qgScratchSlot = static_cast<uint32_t>(groupStartHv > hvBase ? groupStartHv - hvBase : 0);
                        produceQG = headOffset == 0 || hq != (hv - 1) / HRatio_;
                    }
                    if (produceQG) {
                        AscendC::LocalTensor<DT> qgL1 = qgL1Scratch[qgScratchSlot];
                        for (int64_t rowOffset = 0; rowOffset < chunkInfo.chunkLen; rowOffset += vecRow_) {
                            const int64_t curRows = Min(vecRow_, chunkInfo.chunkLen - rowOffset);
                            const uint32_t qIdx = CopyInRows(
                                qGm_, qInputBuf_[curQInputPingPong_], qBase + rowOffset * K_,
                                static_cast<uint32_t>(curRows * K_));
                            AscendC::LocalTensor<float> qFp32 = qFp32Buf_.template Get<float>();
                            CastInputRows(qFp32, qInputBuf_[qIdx], static_cast<uint32_t>(curRows * K_), qIdx);
                            AscendC::PipeBarrier<PIPE_V>();
                            if constexpr (USE_GK == 0) {
                                MulRowsByFactorsRegbase(
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(qFp32.GetPhyAddr()),
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(qFp32.GetPhyAddr()),
                                    ((__ubuf__ float *)reinterpret_cast<uint64_t>(gateFactor.GetPhyAddr())) + rowOffset,
                                    static_cast<uint16_t>(curRows), static_cast<uint16_t>(K_));
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                            const uint32_t outputIdx = curOutputPingPong_;
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
                            AscendC::Cast(outputBuf_[outputIdx], qFp32, AscendC::RoundMode::CAST_RINT,
                                          static_cast<uint32_t>(curRows * K_));
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
                            const AscendC::DataCopyParams qgCopyParams{
                                static_cast<uint16_t>(curRows), 1,
                                static_cast<uint16_t>(K_ / c0Elems - 1), 0};
                            for (int64_t colOffset = 0; colOffset < K_; colOffset += c0Elems) {
                                const int64_t l1Offset =
                                    (colOffset / c0Elems) * qgL1PaddedRows * c0Elems + rowOffset * c0Elems;
                                AscendC::DataCopy(qgL1[l1Offset], outputBuf_[outputIdx][colOffset],
                                                  qgCopyParams, qgCopyEnhanced);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
                            curOutputPingPong_ ^= 1U;
                        }
                    }
                    // dvGateFactor 已在 stage1 头部与 gateFactor 一次 Load 合并产出
                    // （GateDualFactorFuseRegbase），此处无需再算
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                }
                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        continue;
                    }
                    const int64_t hv = hvBase + headOffset;
                    const int64_t dvBase =
                        ((chunkInfo.bIdx * HV_ + hv) * T_ + chunkInfo.tokenStart) * V_;
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, windowStartSlot + headOffset);
                    const int64_t dvStateBase = workspaceBase + dvStateWorkspaceOffset_;
                    uint32_t cvListId = 0;
                    for (int64_t rowOffset = 0; rowOffset < chunkInfo.chunkLen; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, chunkInfo.chunkLen - rowOffset);
                        const int64_t rowElems = rowOffset * V_;
                        const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                        AscendC::LocalTensor<float> outFp32 = outFp32Buf_.template Get<float>();
                        uint32_t dvIdx = 0;
                        if constexpr (std::is_same<DT, bfloat16_t>::value) {
                            const bool useGmDvState = V_ == 256 && chunkInfo.chunkLen > 64;
                            if (useGmDvState) {
                                const uint32_t dvStateIdx = CopyInRows(
                                    workspaceGm_, qInputBuf_[curQInputPingPong_], dvStateBase + rowElems, elems);
                                CastInputRows(outFp32, qInputBuf_[dvStateIdx], elems, dvStateIdx);
                                AscendC::PipeBarrier<PIPE_V>();
                            } else {
                                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                                    MATRIX_CV_AIC_TO_AIV_FLAG_BEGIN + cvListId);
                                CastLocalToFloatRegbase<DT>(
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(outFp32.GetPhyAddr()),
                                    (__ubuf__ DT *)reinterpret_cast<uint64_t>(matrixCvBuf_[cvListId].GetPhyAddr()),
                                    static_cast<uint16_t>(elems));
                                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(
                                    MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN + cvListId);
                                cvListId ^= 1U;
                            }
                            dvIdx = CopyInRows(
                                dvGm_, qInputBuf_[curQInputPingPong_], dvBase + rowElems, elems);
                        } else {
                            const uint32_t dvStateIdx = CopyInRows(
                                workspaceGm_, qInputBuf_[curQInputPingPong_], dvStateBase + rowElems, elems);
                            CastInputRows(outFp32, qInputBuf_[dvStateIdx], elems, dvStateIdx);
                            AscendC::PipeBarrier<PIPE_V>();
                            dvIdx = CopyInRows(
                                dvGm_, qInputBuf_[curQInputPingPong_], dvBase + rowElems, elems);
                        }
                        AscendC::LocalTensor<float> dvFp32 = qFp32Buf_.template Get<float>();
                        CastInputRows(dvFp32, qInputBuf_[dvIdx], elems, dvIdx);
                        AscendC::PipeBarrier<PIPE_V>();
                        if constexpr (USE_GK == 0) {
                            // dv2 尾部融合：×gate + 加 dv + cast 直写 outputBuf 单 pass
                            // （消 MulRowsByFactorsAdd 的 fp32 落 UB、barrier 与
                            // CopyOutFp32Rows 的独立 cast 段）；MTE3_V 等待时序与原
                            // CopyOutFp32Rows 一致（写 outputBuf 前等上次 DataCopy 完成）
                            AscendC::LocalTensor<float> dvGateFactor =
                                dvGateFactorAllFp32_.template Get<float>()[headOffset * gateElems_];
                            const uint32_t outputIdx = curOutputPingPong_;
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
                            MulRowsByFactorsAddCastOutRegbase<DT>(
                                (__ubuf__ DT *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx].GetPhyAddr()),
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(outFp32.GetPhyAddr()),
                                ((__ubuf__ float *)reinterpret_cast<uint64_t>(dvGateFactor.GetPhyAddr())) + rowOffset,
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(dvFp32.GetPhyAddr()),
                                static_cast<uint16_t>(curRows), static_cast<uint16_t>(V_));
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
                            AscendC::DataCopy(dv2Gm_[dvBase + rowElems], outputBuf_[outputIdx], elems);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
                            curOutputPingPong_ ^= 1U;
                        } else {
                            // USE_GK=1 dv2 合成 VF 化：标量 Add → 单 pass 寄存器加
                            AddFloatTwoRegbase(
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(outFp32.GetPhyAddr()),
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(outFp32.GetPhyAddr()),
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(dvFp32.GetPhyAddr()),
                                static_cast<uint16_t>(elems));
                            AscendC::PipeBarrier<PIPE_V>();
                            CopyOutFp32Rows(dv2Gm_, outFp32, dvBase + rowElems, elems);
                        }
                    }
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                }

                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        continue;
                    }
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, workspaceSlot);
                    const int64_t stateBase = StateWorkspaceFloatOffset(workspaceBase, 0);
                    const int64_t termQBase = workspaceBase + termQWorkspaceOffset_;
                    AscendC::LocalTensor<float> termQFp32 = qFp32Buf_.template Get<float>();
                    AscendC::LocalTensor<float> outFp32 = outFp32Buf_.template Get<float>();
                    uint32_t cvListId = 0;

                    for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                        const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                        const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                        const int64_t rowElems = rowOffset * V_;
                        const uint32_t termQIdx = CopyInRows(
                            workspaceGm_, qInputBuf_[curQInputPingPong_], termQBase + rowElems, elems);
                        CastInputRows(termQFp32, qInputBuf_[termQIdx], elems, termQIdx);
                        // termW 填充（CV 握手 / GM 读）所有路径必须执行：
                        // 驻留态跳过它会导致 AIV 不消费 CV tile、mode-4 握手失衡
                        uint32_t stateIdx = 0;
                        if (!stateResident_) {
                            stateIdx = CopyInStateRows(
                                stateBuf_[curStatePingPong_], stateBase + rowElems, elems);
                        }
                        if constexpr (std::is_same<DT, bfloat16_t>::value) {
                            const bool useGmTermW = V_ == 256 && chunkInfo.chunkLen > 64;
                            if (useGmTermW) {
                                const int64_t termWBase = workspaceBase + termWWorkspaceOffset_;
                                const uint32_t termWIdx = CopyInRows(
                                    workspaceGm_, qInputBuf_[curQInputPingPong_], termWBase + rowElems, elems);
                                CastInputRows(outFp32, qInputBuf_[termWIdx], elems, termWIdx);
                            } else {
                                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                                    MATRIX_CV_AIC_TO_AIV_FLAG_BEGIN + cvListId);
                                CastLocalToFloatRegbase<DT>(
                                    (__ubuf__ float *)reinterpret_cast<uint64_t>(outFp32.GetPhyAddr()),
                                    (__ubuf__ DT *)reinterpret_cast<uint64_t>(matrixCvBuf_[cvListId].GetPhyAddr()),
                                    static_cast<uint16_t>(elems));
                                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(
                                    MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN + cvListId);
                                cvListId ^= 1U;
                            }
                        } else {
                            const int64_t termWBase = workspaceBase + termWWorkspaceOffset_;
                            const uint32_t termWIdx = CopyInRows(
                                workspaceGm_, qInputBuf_[curQInputPingPong_], termWBase + rowElems, elems);
                            CastInputRows(outFp32, qInputBuf_[termWIdx], elems, termWIdx);
                        }
                        if (stateResident_) {
                            // 驻留态：state 常驻 UB 槽位，行 tile 原位累加（无 GM 往返）。
                            // termQ/outFp32 均由 V pipe 先序填充（CastInputRows/CV 消费），
                            // 同 pipe 保序；state 不经 MTE，无需 state 事件
                            AscendC::LocalTensor<float> stateFp32 =
                                stateBuf_[StateResidentSlot(headOffset)][rowElems];
                            AscendC::PipeBarrier<PIPE_V>();
                            const float scaleLocal = scale_;
                            StateUpdateFuseRegbase(
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(termQFp32.GetPhyAddr()),
                                (__ubuf__ float *)reinterpret_cast<uint64_t>(outFp32.GetPhyAddr()),
                                scaleLocal, static_cast<uint16_t>(elems));
                            AscendC::PipeBarrier<PIPE_V>();
                            continue;
                        }
                        AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[stateIdx]);
                        AscendC::PipeBarrier<PIPE_V>();
                        // 单 pass VF 融合：state += termQ*scale - termW（消 3 个标量 API 与
                        // 中间 3 个 PipeBarrier；scale 先提局部变量避免 VF 内成员访问破坏融合）
                        const float scaleLocal = scale_;
                        StateUpdateFuseRegbase(
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(stateFp32.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(termQFp32.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(outFp32.GetPhyAddr()),
                            scaleLocal, static_cast<uint16_t>(elems));
                        AscendC::PipeBarrier<PIPE_V>();
                        CopyOutStateRows(stateIdx, stateFp32, stateBase + rowElems, elems);
                    }
                }
            }

            if (hasDh0_) {
                for (int64_t headOffset = 0; headOffset < headCnt; ++headOffset) {
                    if (headOffset % subBlockNum_ != subBlockIdx_) {
                        continue;
                    }
                    const int64_t workspaceSlot = windowStartSlot + headOffset;
                    const int64_t workspaceBase = WorkspaceBase(coreIdx, workspaceSlot);
                    const int64_t hv = hvBase + headOffset;
                    // dh0 偏移按 main 侧 cd357a66 融合接口约定简化（不再按变长
                    // outputChunkIdx 查找）
                    const int64_t dh0Base = (seqIdx * HV_ + hv) * K_ * V_;
                    if (stateResident_) {
                        // 驻留态：state 常驻 UB 槽位，直接行 tile 输出（无 GM 读回）
                        for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                            const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                            const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                            AscendC::LocalTensor<float> stateFp32 =
                                stateBuf_[StateResidentSlot(headOffset)][rowOffset * V_];
                            CopyOutFp32Rows(dh0Gm_, stateFp32, dh0Base + rowOffset * V_, elems);
                        }
                        continue;
                    }
                    const int64_t stateBase = StateWorkspaceFloatOffset(workspaceBase, 0);
                    if (!stateVFirst_) {
                        for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
                            const int64_t curRows = Min(vecRow_, K_ - rowOffset);
                            const uint32_t elems = static_cast<uint32_t>(curRows * V_);
                            const uint32_t stateIdx = CopyInStateRows(
                                stateBuf_[curStatePingPong_], stateBase + rowOffset * V_, elems);
                            AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[stateIdx]);
                            CopyOutFp32Rows(dh0Gm_, stateFp32, dh0Base + rowOffset * V_, elems);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[stateIdx]);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[stateIdx]);
                        }
                    } else {
                        CopyOutDh0VFirst(stateBase, dh0Base);
                    }
                }
            }

        }

        ReleaseVectorEvents();
    }

private:
    static constexpr uint32_t BUFFER_COUNT = 2;
    static constexpr float LN2 = 0.69314718055994530942f;

    __aicore__ inline void InitVectorEvents()
    {
        for (uint32_t eventIdx = 0; eventIdx < BUFFER_COUNT; ++eventIdx) {
            qMte2ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            qVToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            gateMte2ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            gateVToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            vToMte3Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            stateMte2ToVEvent_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            stateVToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            stateVToMte3Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            stateMte3ToMte2Event_[eventIdx] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[eventIdx]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[eventIdx]);
        }
        for (uint32_t cvIdx = 0; cvIdx < CV_BUFFER_COUNT; ++cvIdx) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_V>(MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN + cvIdx);
        }
    }

    __aicore__ inline void ReleaseVectorEvents()
    {
        for (uint32_t eventIdx = 0; eventIdx < BUFFER_COUNT; ++eventIdx) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[eventIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(qMte2ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(gateMte2ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(stateVToMte3Event_[eventIdx]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[eventIdx]);
        }
    }

    template <typename CopyType>
    __aicore__ inline uint32_t CopyInRows(AscendC::GlobalTensor<CopyType> &inputTensor,
                                          AscendC::LocalTensor<CopyType> dstTensor, int64_t inputOffset,
                                          uint32_t elements)
    {
        const uint32_t inputIdx = curQInputPingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[inputIdx]);
        AscendC::DataCopy(dstTensor, inputTensor[inputOffset], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(qMte2ToVEvent_[inputIdx]);
        curQInputPingPong_ ^= 1U;
        return inputIdx;
    }

    __aicore__ inline void CastInputRows(AscendC::LocalTensor<float> dstTensor, AscendC::LocalTensor<DT> srcTensor,
                                         uint32_t elements, uint32_t inputIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(qMte2ToVEvent_[inputIdx]);
        CastLocalToFloatRegbase<DT>((__ubuf__ float *)reinterpret_cast<uint64_t>(dstTensor.GetPhyAddr()),
                                    (__ubuf__ DT *)reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr()),
                                    static_cast<uint16_t>(elements));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qVToMte2Event_[inputIdx]);
    }

    template <typename CopyType>
    __aicore__ inline uint32_t CopyInGateRows(AscendC::GlobalTensor<CopyType> &inputTensor,
                                              AscendC::LocalTensor<CopyType> dstTensor, int64_t inputOffset,
                                              uint32_t elements)
    {
        const uint32_t inputIdx = curGateInputPingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[inputIdx]);
        AscendC::DataCopyPad(dstTensor, inputTensor[inputOffset],
                             {1, elements * static_cast<uint32_t>(sizeof(CopyType)), 0, 0, 0},
                             {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(gateMte2ToVEvent_[inputIdx]);
        curGateInputPingPong_ ^= 1U;
        return inputIdx;
    }

    template <typename CopyType>
    __aicore__ inline void CastGateInputRows(AscendC::LocalTensor<float> dstTensor,
                                             AscendC::LocalTensor<CopyType> srcTensor, uint32_t elements,
                                             uint32_t inputIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(gateMte2ToVEvent_[inputIdx]);
        CastLocalToFloatRegbase<CopyType>((__ubuf__ float *)reinterpret_cast<uint64_t>(dstTensor.GetPhyAddr()),
                                          (__ubuf__ CopyType *)reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr()),
                                          static_cast<uint16_t>(elements));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(gateVToMte2Event_[inputIdx]);
    }

    __aicore__ inline void CopyOutFp32Rows(AscendC::GlobalTensor<DT> &outTensor,
                                           AscendC::LocalTensor<float> srcTensor, int64_t outOffset,
                                           uint32_t elements)
    {
        const uint32_t outputIdx = curOutputPingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
        CastFp32ToOutputRegbase<DT>(
            (__ubuf__ DT *)reinterpret_cast<uint64_t>(outputBuf_[outputIdx].GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr()),
            static_cast<uint16_t>(elements));
        // Set/Wait 配对不可拆：DataCopy 在 MTE3 流水发射前必须等 V 写完 outputBuf
        // （删除该 Wait 会导致 MTE3 先于 VF cast 读缓冲的数据竞争）
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[outputIdx]);
        AscendC::DataCopy(outTensor[outOffset], outputBuf_[outputIdx], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[outputIdx]);
        curOutputPingPong_ ^= 1U;
    }

    __aicore__ inline void TransposeStateTile(AscendC::LocalTensor<DT> dstTensor,
                                              AscendC::LocalTensor<DT> srcTensor) const
    {
        constexpr uint32_t TRANSPOSE_ROWS = 16;
        constexpr uint32_t DATA_BLOCK_BYTES = 32;
        constexpr uint32_t ELEMS_PER_BLOCK = DATA_BLOCK_BYTES / sizeof(uint16_t);
        uint64_t dstList[TRANSPOSE_ROWS];
        uint64_t srcList[TRANSPOSE_ROWS];
        const uint64_t dstAddr = reinterpret_cast<uint64_t>(dstTensor.GetPhyAddr());
        const uint64_t srcAddr = reinterpret_cast<uint64_t>(srcTensor.GetPhyAddr());
        const uint16_t repeatTimes = static_cast<uint16_t>(V_ / ELEMS_PER_BLOCK);
        AscendC::TransDataTo5HDParams transposeParams{
            false, false, static_cast<uint8_t>(repeatTimes),
            static_cast<uint16_t>(repeatTimes > 1 ? TRANSPOSE_ROWS : 0),
            static_cast<uint16_t>(repeatTimes > 1 ? 1 : 0)};
        for (uint32_t row = 0; row < TRANSPOSE_ROWS; ++row) {
            srcList[row] = srcAddr + row * V_ * sizeof(uint16_t);
            dstList[row] = dstAddr + row * TRANSPOSE_ROWS * sizeof(uint16_t);
        }
        AscendC::TransDataTo5HD<uint16_t>(dstList, srcList, transposeParams);
    }

    __aicore__ inline void CopyOutFp32RowsVFirst(AscendC::GlobalTensor<DT> &outTensor,
                                                  AscendC::LocalTensor<float> srcTensor,
                                                  int64_t outBase, int64_t rowOffset,
                                                  int64_t rowCount)
    {
        constexpr int64_t TRANSPOSE_ROWS = 16;
        constexpr uint32_t SRC_OUTPUT_IDX = 0;
        constexpr uint32_t DST_OUTPUT_IDX = 1;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[SRC_OUTPUT_IDX]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[DST_OUTPUT_IDX]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[SRC_OUTPUT_IDX]);
        for (int64_t tileRow = 0; tileRow < rowCount; tileRow += TRANSPOSE_ROWS) {
            const int64_t curRows = Min(TRANSPOSE_ROWS, rowCount - tileRow);
            const uint32_t elems = static_cast<uint32_t>(curRows * V_);
            AscendC::Cast(outputBuf_[SRC_OUTPUT_IDX], srcTensor[tileRow * V_],
                          AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
            TransposeStateTile(outputBuf_[DST_OUTPUT_IDX], outputBuf_[SRC_OUTPUT_IDX]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[DST_OUTPUT_IDX]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_[DST_OUTPUT_IDX]);
            const AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(V_), static_cast<uint32_t>(curRows * sizeof(DT)),
                static_cast<uint32_t>((TRANSPOSE_ROWS - curRows) * sizeof(DT)),
                static_cast<uint32_t>((K_ - curRows) * sizeof(DT)), 0};
            AscendC::DataCopyPad(outTensor[outBase + rowOffset + tileRow],
                                 outputBuf_[DST_OUTPUT_IDX], copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[DST_OUTPUT_IDX]);
            if (tileRow + TRANSPOSE_ROWS < rowCount) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_[DST_OUTPUT_IDX]);
            }
        }
    }

    __aicore__ inline void CopyOutDh0VFirst(int64_t stateBase, int64_t dh0Base)
    {
        for (int64_t rowOffset = 0; rowOffset < K_; rowOffset += vecRow_) {
            const int64_t curRows = Min(vecRow_, K_ - rowOffset);
            const uint32_t elems = static_cast<uint32_t>(curRows * V_);
            const uint32_t stateIdx = CopyInStateRows(
                stateBuf_[curStatePingPong_], stateBase + rowOffset * V_, elems);
            AscendC::LocalTensor<float> stateFp32 = stateBuf_[stateIdx];
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[stateIdx]);
            CopyOutFp32RowsVFirst(dh0Gm_, stateFp32, dh0Base, rowOffset, curRows);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[stateIdx]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[stateIdx]);
        }
    }

    __aicore__ inline uint32_t CopyInStateRows(AscendC::LocalTensor<float> dstTensor, int64_t inputOffset,
                                               uint32_t elements)
    {
        const uint32_t inputIdx = curStatePingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[inputIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[inputIdx]);
        AscendC::DataCopy(dstTensor, workspaceStateGm_[inputOffset], elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(stateMte2ToVEvent_[inputIdx]);
        curStatePingPong_ ^= 1U;
        return inputIdx;
    }

    __aicore__ inline uint32_t StateResidentSlot(int64_t headOffset) const
    {
        // 驻留态槽位 = 本 subblock 拥有的 head 序号（headsPerTask<=HEADS_PER_TASK=4、
        // subBlockNum>=2 时最多 2 个 owned head，ping/pong 恰好够用）
        return static_cast<uint32_t>((headOffset / subBlockNum_) & 1);
    }

    __aicore__ inline void CopyOutStateRows(uint32_t stateIdx, AscendC::LocalTensor<float> srcTensor,
                                            int64_t outOffset, uint32_t elements)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(stateVToMte3Event_[stateIdx]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2Event_[stateIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(stateVToMte3Event_[stateIdx]);
        AscendC::DataCopy(workspaceStateGm_[outOffset], srcTensor, elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stateMte3ToMte2Event_[stateIdx]);
    }

    __aicore__ inline int64_t DhOffset(int64_t b, int64_t hv, int64_t chunkIdx) const
    {
        return ((b * totalChunkNum_ + chunkIdx) * HV_ + hv) * K_ * V_;
    }

    __aicore__ inline int64_t WorkspaceBase(int64_t coreIdx, int64_t workspaceSlot) const
    {
        return (coreIdx * WORKSPACE_BUFFER_COUNT + workspaceSlot) * workspaceElemsPerSubBlock_;
    }

    __aicore__ inline int64_t StateWorkspaceFloatOffset(int64_t workspaceBase, int64_t rowOffset) const
    {
        return ((workspaceBase + stateWorkspaceOffset_) * static_cast<int64_t>(sizeof(DT))) /
                   static_cast<int64_t>(sizeof(float)) +
               rowOffset * V_;
    }

    AscendC::GlobalTensor<DT> qGm_;
    AscendC::GlobalTensor<GT> gateGm_;
    AscendC::GlobalTensor<DT> dvGm_;
    AscendC::GlobalTensor<DT> dhGm_;
    AscendC::GlobalTensor<DT> dh0Gm_;
    AscendC::GlobalTensor<DT> dv2Gm_;
    AscendC::GlobalTensor<DT> workspaceGm_;
    AscendC::GlobalTensor<float> workspaceStateGm_;

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixCvPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixCvPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qInputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qInputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gInputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gInputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> statePing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> statePong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gRawAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateFactorAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dvGateFactorAllFp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outFp32Buf_;

    AscendC::LocalTensor<DT> matrixCvBuf_[CV_BUFFER_COUNT];
    AscendC::LocalTensor<DT> qInputBuf_[BUFFER_COUNT];
    AscendC::LocalTensor<GT> gateInputBuf_[BUFFER_COUNT];
    AscendC::LocalTensor<DT> outputBuf_[BUFFER_COUNT];
    AscendC::LocalTensor<float> stateBuf_[BUFFER_COUNT];
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{VEC_TO_CUBE_FLAG_READY};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CUBE_TO_VEC_FLAG_READY};

    AscendC::TEventID qMte2ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID qVToMte2Event_[BUFFER_COUNT];
    AscendC::TEventID gateMte2ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID gateVToMte2Event_[BUFFER_COUNT];
    AscendC::TEventID vToMte3Event_[BUFFER_COUNT];
    AscendC::TEventID mte3ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID stateMte2ToVEvent_[BUFFER_COUNT];
    AscendC::TEventID stateVToMte2Event_[BUFFER_COUNT];
    AscendC::TEventID stateVToMte3Event_[BUFFER_COUNT];
    AscendC::TEventID stateMte3ToMte2Event_[BUFFER_COUNT];
    uint32_t curQInputPingPong_ = 0;
    uint32_t curGateInputPingPong_ = 0;
    uint32_t curOutputPingPong_ = 0;
    uint32_t curStatePingPong_ = 0;

    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR dh0Addr_ = nullptr;
    const ChunkGatedDeltaRuleBwdDhuTilingData *tiling_ = nullptr;
    int64_t B_ = 0;
    int64_t HK_ = 0;
    int64_t HV_ = 0;
    int64_t T_ = 0;
    int64_t K_ = 0;
    int64_t V_ = 0;
    int64_t HRatio_ = 0;
    int64_t chunkSize_ = 0;
    int64_t vecRow_ = 8;
    int64_t gateElems_ = 0;
    int64_t totalChunkNum_ = 0;
    int64_t headsPerTask_ = 0;
    int64_t headWindowNum_ = 0;
    int64_t taskNum_ = 0;
    int64_t subBlockNum_ = 1;
    int64_t subBlockIdx_ = 0;
    int64_t isVariable_ = 0;
    float scale_ = 1.0f;
    bool hasDh0_ = false;
    bool stateVFirst_ = false;
    bool stateResident_ = false;
    int64_t dh0ClearCoreNum_ = 0;
    int64_t dh0ClearElemsPerCore_ = 0;
    int64_t dh0ClearTailElems_ = 0;
    int64_t workspaceElemsPerSubBlock_ = 0;
    int64_t stateWorkspaceOffset_ = 0;
    int64_t dvStateWorkspaceOffset_ = 0;
    int64_t termQWorkspaceOffset_ = 0;
    int64_t termWWorkspaceOffset_ = 0;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H
