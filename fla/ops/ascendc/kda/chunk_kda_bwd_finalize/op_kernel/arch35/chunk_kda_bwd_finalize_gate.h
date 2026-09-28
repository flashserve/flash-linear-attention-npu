/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_GATE_H
#define CHUNK_KDA_BWD_FINALIZE_GATE_H

#include "chunk_kda_bwd_finalize_common.h"
#include "kernel_utils/vector/regbase.hpp"

namespace KDA {
using namespace AscendC::MicroAPI;

template <typename LogT>
__simd_vf__ inline void FinalizeStage11VF(
    __ubuf__ float *dg, __ubuf__ float *raw, __ubuf__ float *bias,
    __ubuf__ LogT *aLog, __ubuf__ float *da, __ubuf__ float *db,
    uint16_t rows, float lowerBound)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> e;
    if constexpr (AscendC::IsSameType<LogT, float>::value) {
        LoadIn<float, true>(e, aLog);
    } else {
        RegTensor<bfloat16_t> packed;
        LoadIn<bfloat16_t, true>(packed, aLog);
        Cast<float, bfloat16_t, ctHalf2Fp32Zero>(e, packed, mask);
    }
    Exp(e, e, mask);
    RegTensor<float> one, sum0, sum1, da0, da1, db0, db1, bias0, bias1;
    Duplicate(one, 1.0f, mask);
    Duplicate(sum0, 0.0f, mask);
    Duplicate(sum1, 0.0f, mask);
    Duplicate(da0, 0.0f, mask);
    Duplicate(da1, 0.0f, mask);
    Duplicate(db0, 0.0f, mask);
    Duplicate(db1, 0.0f, mask);
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(bias0, bias1, bias);
    // Reverse scan and both parameter partials remain in registers.
    for (uint16_t reverse = 0; reverse < rows; ++reverse) {
        const uint32_t offset = (rows - 1U - reverse) * KDA_FINALIZE_DIM;
        RegTensor<float> g0, g1, x0, x1, s0, s1, tmp0, tmp1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(g0, g1, dg + offset);
        Add(sum0, sum0, g0, mask);
        Add(sum1, sum1, g1, mask);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(x0, x1, raw + offset);
        Add(x0, x0, bias0, mask);
        Add(x1, x1, bias1, mask);
        Mul(s0, x0, e, mask);
        Mul(s1, x1, e, mask);
        Muls(s0, s0, -1.0f, mask);
        Muls(s1, s1, -1.0f, mask);
        Exp(s0, s0, mask);
        Exp(s1, s1, mask);
        Adds(s0, s0, 1.0f, mask);
        Adds(s1, s1, 1.0f, mask);
        Div(s0, one, s0, mask);
        Div(s1, one, s1, mask);
        Muls(g0, sum0, lowerBound, mask);
        Muls(g1, sum1, lowerBound, mask);
        Mul(g0, g0, e, mask);
        Mul(g1, g1, e, mask);
        Mul(g0, g0, s0, mask);
        Mul(g1, g1, s1, mask);
        Sub(tmp0, one, s0, mask);
        Sub(tmp1, one, s1, mask);
        Mul(g0, g0, tmp0, mask);
        Mul(g1, g1, tmp1, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(dg + offset, g0, g1, mask);
        Add(db0, db0, g0, mask);
        Add(db1, db1, g1, mask);
        Mul(tmp0, g0, x0, mask);
        Mul(tmp1, g1, x1, mask);
        Add(da0, da0, tmp0, mask);
        Add(da1, da1, tmp1, mask);
    }
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(db, db0, db1, mask);
    RegTensor<float> total;
    Add(da0, da0, da1, mask);
    ReduceSum(total, da0, mask);
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(da, total, mask);
}

__simd_vf__ inline void FinalizeStage12VF(
    __ubuf__ float *dbOut, __ubuf__ float *daOut,
    __ubuf__ float *dbIn, __ubuf__ float *daIn, uint16_t rows, bool first)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> sum0, sum1, sumA;
    if (first) {
        Duplicate(sum0, 0.0f, mask);
        Duplicate(sum1, 0.0f, mask);
        Duplicate(sumA, 0.0f, mask);
    } else {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(sum0, sum1, dbOut);
        LoadIn<float, true>(sumA, daOut);
    }
    for (uint16_t row = 0; row < rows; ++row) {
        RegTensor<float> value0, value1, valueA;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(value0, value1, dbIn + row * 128U);
        LoadIn<float, true>(valueA, daIn + row * 8U);
        Add(sum0, sum0, value0, mask);
        Add(sum1, sum1, value1, mask);
        Add(sumA, sumA, valueA, mask);
    }
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dbOut, sum0, sum1, mask);
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(daOut, sumA, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__aicore__ inline void FinalizeStage12(
    GM_ADDR workspace, GM_ADDR dALog, GM_ADDR dDtBias,
    const ChunkKdaBwdFinalizeTilingData &tiling)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 70 * 1024);
    auto dbIn = buffer.Get<float>();
    auto daIn = dbIn[16384];
    auto dbOut = daIn[1024];
    auto daOut = dbOut[128];
    AscendC::GlobalTensor<float> partialA, partialB, outA, outB;
    partialA.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + tiling.gatePartialOffset));
    partialB.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + tiling.dtBiasPartialOffset));
    outA.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dALog));
    outB.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dDtBias));
    const auto mte2V = pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
    const auto vMte2 = pipe.AllocEventID<AscendC::HardEvent::V_MTE2>();
    const auto vMte3 = pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
    const auto mte3V = pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
    // Own complete 32-byte output blocks, including a possible final tail.
    // No two cores write adjacent scalar lanes of the same DMA block.
    for (int64_t group = AscendC::GetBlockIdx() * 8; group < tiling.NV;
         group += AscendC::GetBlockNum() * AscendC::GetSubBlockNum() * 8) {
        const int64_t end = FinalizeMin(group + 8, tiling.NV);
        for (int64_t head = group; head < end; ++head) {
            for (int64_t start = 0; start < tiling.totalChunkNum; start += 128) {
                const uint16_t rows = static_cast<uint16_t>(FinalizeMin(128, tiling.totalChunkNum - start));
                const int64_t index = head * tiling.totalChunkNum + start;
                AscendC::DataCopy(dbIn, partialB[index * 128], rows * 128);
                AscendC::DataCopy(daIn, partialA[index * 8], rows * 8);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                FinalizeStage12VF(
                    reinterpret_cast<__ubuf__ float *>(dbOut.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(daOut.GetPhyAddr()) + head - group,
                    reinterpret_cast<__ubuf__ float *>(dbIn.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(daIn.GetPhyAddr()), rows, start == 0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vMte2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vMte2);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::DataCopy(outB[head * 128], dbOut, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
        AscendC::DataCopyPad(outA[group], daOut,
            AscendC::DataCopyExtParams{1, static_cast<uint32_t>((end - group) * sizeof(float)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
    }
}

} // namespace KDA
#endif
