// Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
namespace KdaPrepare::Arch22 {
constexpr uint32_t Rows=64, Dim=128;
constexpr uint32_t RhsReadyBase=8; // Two generation flags, shared AIC/AIV.
constexpr uint32_t LowerChoiceReadyBase=10; // Early classification, before FP32 leaf solve.
// Two independent value-head lanes in each generation; two GM generations.
constexpr uint32_t Payload=0, Raw=0x12000, Lower=0x1A000, Eye=0x1C000;
// Score payload is retired before Lower. BF16 X and the scoped FP32
// merge relays fit before the per-lane selection cache line.
constexpr uint32_t DiagFp32=0x2000, CrossFp32=0x4000, FinalCrossFp32=0x5000;
constexpr uint32_t MergeFp32=0x6000; // All input squares retire before this relay.
constexpr uint32_t InverseChoice=0x10000;
constexpr float FastInverseMaxCoefficient=1.0F/32.0F;
constexpr float MaxFiniteFp32=3.402823466e38F;
static_assert(DiagFp32>=64*64*sizeof(uint16_t));
static_assert(DiagFp32+2*32*32*sizeof(float)==CrossFp32);
static_assert(CrossFp32+32*32*sizeof(float)==FinalCrossFp32);
static_assert(FinalCrossFp32+32*32*sizeof(float)==MergeFp32);
static_assert(MergeFp32+32*32*sizeof(float)<=InverseChoice && InverseChoice+32<Raw);
static_assert(FinalCrossFp32+32*32*sizeof(float)<=InverseChoice);
constexpr uint32_t RhsK=0x1E000, RhsV=0x22000, LaneBytes=0x26000;
constexpr uint32_t SlotBytes=2*LaneBytes, GroupBytes=2*SlotBytes;
// L1 lane changes meaning only after score has drained. Matrix offsets in bytes.
constexpr uint32_t L1LaneBytes=0x18000, L1Bytes=2*L1LaneBytes;
constexpr uint32_t P=0, X=0x2000, Tmp=0x4000, I=0x14000, Power=0x8000;
constexpr uint32_t L1RhsK=0xC000, L1RhsV=0x10000;
constexpr uint32_t FpX0=0x4000,FpX1=0x5000,FpP=0x6000,FpFinalP=0x7000,FpT=0x8000;
static_assert(FpX0>=X+64*64*sizeof(uint16_t) && FpT+32*32*sizeof(float)<=L1RhsK);
static_assert(FpX0+32*32*sizeof(float)==FpX1 && FpX1+32*32*sizeof(float)==FpP);
static_assert(FpP+32*32*sizeof(float)==FpFinalP);
static_assert(FpFinalP+32*32*sizeof(float)==FpT);
constexpr uint32_t UbBytes=0x2F800; // 190 KiB, per AIV.
constexpr uint32_t UbQ=0, UbK=0x8000, UbG=0x10000, UbWork=0x18000;
constexpr uint32_t UbV=0x28000, UbSmall=0x2C000, UbMask=0x2F000;
constexpr uint32_t ZeroSelectMask=640;
static_assert(ZeroSelectMask>=640 && ZeroSelectMask+32<=1024);
constexpr uint32_t InverseStepMask=1024;
static_assert(InverseStepMask>=1024 && InverseStepMask+15*32<=UbBytes-UbMask);
static_assert((12288+2560)*sizeof(float)<=UbV-UbWork);
// FP32 indices in the existing Small UB allocation. FP32 beta DMA ends at
// index 2560 and its BF16 packed output ends at 2592.
constexpr uint32_t ALogCacheSmall=2688, ALogCacheHeads=128;
constexpr uint32_t BetaGatherIndexSmall=2816;
constexpr uint32_t InverseGatherIndexSmall=2944;
static_assert(InverseGatherIndexSmall>=BetaGatherIndexSmall+128);
static_assert(UbSmall+(InverseGatherIndexSmall+128)*sizeof(uint32_t)<=UbMask);
static_assert(31*64*8*sizeof(float)<=UbG-UbQ);
static_assert(BetaGatherIndexSmall>=ALogCacheSmall+ALogCacheHeads);
static_assert(UbSmall+(BetaGatherIndexSmall+128)*sizeof(uint32_t)<=UbMask);
static_assert(ALogCacheSmall>=2592);
static_assert(UbSmall+(ALogCacheSmall+ALogCacheHeads)*sizeof(float)<=UbMask);
constexpr uint32_t MinusOffsetHost(uint32_t band) {
    return 0x8000+band*(band+1)*0x800;
}
static_assert(MinusOffsetHost(3)+64*128*2==0x12000);
static_assert(RhsV+64*128*2==LaneBytes);
static_assert(L1RhsV+64*128*2<=I && I+64*64*2<=L1LaneBytes);
static_assert(UbMask+0x800==UbBytes && UbBytes<=192*1024);
}
