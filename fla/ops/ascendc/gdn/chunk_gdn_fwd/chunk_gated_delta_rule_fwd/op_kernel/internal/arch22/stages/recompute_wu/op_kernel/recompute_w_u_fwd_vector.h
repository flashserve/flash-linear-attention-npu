/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd.h
 * \brief
 */


#ifndef RECOMPUTE_W_U_FWD_VECTOR_H
#define RECOMPUTE_W_U_FWD_VECTOR_H

#include "../../../common/qkv_input_layout.h"
#include "../../../common/gdn_raw_front_resources.h"
#include "recompute_w_u_fwd_struct.h"
#include "catlass/arch/cross_core_sync.hpp"
using namespace AscendC;

using GDN::GdnMegaArch22RecomputeWUTilingData;

template <typename kType, typename betaType, bool kFlattenHeadTasks = false,
          bool kAbcTaskOrder = false, bool kRawFront = false,
          int kStaticV = 0, int kStaticBT = 0, int kQkvLayout = -1>
class RecomputeWUFwdVectorProcess {
public:
    __aicore__ inline bool InputSequenceMajor() const
    {
        if constexpr (kQkvLayout >= 0) {
            return kQkvLayout == 1;
        } else {
            return inputSequenceMajorRuntime_;
        }
    }

    bool inputSequenceMajorRuntime_{false};
    __aicore__ inline void ConfigureInputLayout(bool sequenceMajor)
    {
        if constexpr (kQkvLayout < 0) {
            inputSequenceMajorRuntime_ = sequenceMajor;
        }
    }

    /** @brief constructor */
    __aicore__ inline RecomputeWUFwdVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
                                                        GM_ADDR workspace_);

    __aicore__ inline void Process(const GDN::RecomputeTaskRange *range = nullptr);
    __aicore__ inline void ProcessInterleaved();
    __aicore__ inline void BeginPreparation(const GDN::RecomputeTaskRange *range);
    __aicore__ inline void PrepareRange(uint64_t localBegin, uint64_t localEnd);
    __aicore__ inline void EndPreparation();
    __aicore__ inline void Init(const GdnMegaArch22RecomputeWUTilingData &tiling,
                                GdnRawFront::VectorResources *resources);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t Hv = 1;
    uint64_t Hk = 1;
    uint64_t hvPerHk = 1;
    uint64_t K = 0;
    __aicore__ inline uint64_t ValueDim() const
    {
        if constexpr (kStaticV != 0) {
            return static_cast<uint64_t>(kStaticV);
        } else {
            return VRuntime_;
        }
    }

    uint64_t VRuntime_ = 0;
    __aicore__ inline uint64_t ChunkSize() const
    {
        if constexpr (kStaticBT != 0) {
            return static_cast<uint64_t>(kStaticBT);
        } else {
            return chunkSizeRuntime_;
        }
    }

    uint64_t chunkSizeRuntime_ = 0;
    uint64_t chunkNum = 0;
    uint64_t interleavedMembaseVecRow = 0;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR workspace;
    GdnRawFront::VectorResources *frontResources = nullptr;
    const GDN::RecomputeTaskRange *taskRange = nullptr;
    GdnRawFront::WuLayout interleavedLayout;
    uint32_t rawBlock = 0;

    template <bool kPrepareOnly>
    __aicore__ inline void BeginInterleaved();
    template <bool kPrepareOnly>
    __aicore__ inline void ProcessInterleavedRange(uint32_t loopBegin, uint32_t loopEnd);
    template <bool kPrepareOnly>
    __aicore__ inline void EndInterleaved();

private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<kType> workSpaceTensor;

    GdnRawFront::Buffer<kRawFront> kFp32Buf;
    GdnRawFront::Buffer<kRawFront> vFp32Buf;
    GdnRawFront::Buffer<kRawFront> betaFp32Buf;
    GdnRawFront::Buffer<kRawFront> betaFp32BrcbBuf;
    GdnRawFront::Buffer<kRawFront> gFp32Buf;
    // GdnRawFront::Buffer<kRawFront> gFp32BrcbBuf;

};

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
__aicore__ inline RecomputeWUFwdVectorProcess<kType, betaType,
                                               kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::RecomputeWUFwdVectorProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType, kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::Init(
    const GdnMegaArch22RecomputeWUTilingData &tiling, GdnRawFront::VectorResources *resources)
{
    frontResources = resources;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    B = tiling.B;
    T = tiling.T;
    Hv = tiling.Hv;
    Hk = tiling.Hk;
    hvPerHk = tiling.hvPerHk;
    K = tiling.K;
    if constexpr (kStaticV == 0) {
        VRuntime_ = tiling.V;
    }
    if constexpr (kStaticBT == 0) {
        chunkSizeRuntime_ = tiling.chunkSize;
    }
    chunkNum = tiling.chunkNum;
    interleavedMembaseVecRow = tiling.interleavedMembaseVecRow;
    return;
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType, kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::Process(const GDN::RecomputeTaskRange *range)
{
    taskRange = range;
    ProcessInterleaved();
    return;
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
__aicore__ inline void RecomputeWUFwdVectorProcess<kType, betaType,
    kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::ProcessInterleaved()
{
    if (taskRange->fullyPrepared) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        // 整批 RHS 独占 GM；保留一次 MTE3 发布，末 done7 保护下一批复用。
        if (taskRange->begin < taskRange->end) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
        }
#else
        // 所有 RHS 已由最终 Solve 逐窗生产；只闭合原 Cube 的四任务通知协议。
        // 第二 Solve 退休 ACK 已消费，才允许复用 ready3/free5。
        for (uint64_t begin = taskRange->begin; begin < taskRange->end;
             begin += GDN::RECOMPUTE_WU_SYNC_WINDOW_TASKS) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            if (begin != taskRange->begin) {
                CrossCoreWaitFlag<0x2>(SYNC_AIC_AIV_FLAG_5);
            }
        }
#endif
        return;
    }
    BeginInterleaved<false>();
    ProcessInterleavedRange<false>(static_cast<uint32_t>(taskRange->begin),
                                   static_cast<uint32_t>(taskRange->end));
    EndInterleaved<false>();
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
__aicore__ inline void RecomputeWUFwdVectorProcess<kType, betaType,
    kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::BeginPreparation(
        const GDN::RecomputeTaskRange *range)
{
    static_assert(sizeof(kType) == 2, "A2 WU preparation preserves two-byte input rounding");
    // 最终 Solve 低区：BT64 成对为24KiB，BT128 单任务为48KiB；后者仍为最大值。
    static_assert(3 * 128 * 128 + 64 * (12 * 128 + 56) <= Catlass::Arch::AtlasA2::UB_SIZE,
                  "A2 WU preparation and final Solve views exceed UB");
    taskRange = range;
    BeginInterleaved<true>();
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
__aicore__ inline void RecomputeWUFwdVectorProcess<kType, betaType,
    kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::PrepareRange(
        uint64_t localBegin, uint64_t localEnd)
{
    ProcessInterleavedRange<true>(static_cast<uint32_t>(taskRange->begin + localBegin),
                                  static_cast<uint32_t>(taskRange->begin + localEnd));
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
__aicore__ inline void RecomputeWUFwdVectorProcess<kType, betaType,
    kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::EndPreparation()
{
    EndInterleaved<true>();
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
template <bool kPrepareOnly>
__aicore__ inline void RecomputeWUFwdVectorProcess<kType, betaType,
    kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::BeginInterleaved()
{
    const uint32_t rowNum = interleavedMembaseVecRow;
    const uint32_t width = static_cast<uint32_t>(K > ValueDim() ? K : ValueDim());
    const uint32_t ubOffset = kPrepareOnly
        ? static_cast<uint32_t>(ChunkSize() * 64 * (sizeof(float32_t) + sizeof(kType))) : 0;
    constexpr uint32_t eventOffset = kPrepareOnly ? 2 : 0;
    interleavedLayout.Init(rowNum, width, sizeof(kType), true);
    const auto &layout = interleavedLayout;
    rawBlock = 0;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        SetFlag<HardEvent::V_MTE2>(frontResources->template Event<HardEvent::V_MTE2>(slot + eventOffset));
        SetFlag<HardEvent::MTE3_V>(frontResources->template Event<HardEvent::MTE3_V>(slot + eventOffset));
    }
    vFp32Buf.BindView(frontResources, ubOffset + layout.fp32, rowNum * ValueDim() * sizeof(float32_t));
    kFp32Buf.BindView(frontResources, ubOffset + layout.fp32, rowNum * K * sizeof(float32_t));
    betaFp32Buf.BindView(frontResources, ubOffset + layout.betaFp32, rowNum * sizeof(float32_t));
    betaFp32BrcbBuf.BindView(frontResources, ubOffset + layout.brcb, rowNum * ONE_BLOCK_32);
    gFp32Buf.BindView(frontResources, ubOffset + layout.gFp32, rowNum * sizeof(float32_t));
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
template <bool kPrepareOnly>
__aicore__ inline void RecomputeWUFwdVectorProcess<kType, betaType,
    kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::ProcessInterleavedRange(
        uint32_t loopBegin, uint32_t loopEnd)
{
    // Only the owned A2 parent-batch caller instantiates this non-null-range path.
    // This static host path uses R64: Vb leaves the same head's complete beta in betaFP32.
    constexpr bool kReuseVbBeta = kPrepareOnly && kRawFront && kFlattenHeadTasks && kAbcTaskOrder &&
        kStaticBT == 64 && kStaticV == 128 && std::is_same<betaType, float32_t>();
    const uint32_t rowNum = interleavedMembaseVecRow;
    const uint32_t width = static_cast<uint32_t>(K > ValueDim() ? K : ValueDim());
    // BT64 成对 f/o 占24KiB，BT128 最终单任务占48KiB；准备区与其分离，保护跨窗异步读写。
    const uint32_t ubOffset = kPrepareOnly
        ? static_cast<uint32_t>(ChunkSize() * 64 * (sizeof(float32_t) + sizeof(kType))) : 0;
    constexpr uint32_t eventOffset = kPrepareOnly ? 2 : 0;
    const uint64_t coreLoopsInB = (T + ChunkSize() - 1) / ChunkSize();
    const auto &layout = interleavedLayout;
    const uint32_t inputSlotBytes = GdnRawFront::Align(rowNum * width * sizeof(kType));
    const uint32_t betaSlotBytes = GdnRawFront::Align(rowNum * sizeof(float32_t));
    uint32_t vecTaskIdx = loopBegin - static_cast<uint32_t>(taskRange->begin);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;
    uint32_t subBlockIdx = GetSubBlockIdx();
    auto tensorVFp32 = vFp32Buf.template Get<float32_t>();
    auto tensorKFp32 = kFp32Buf.template Get<float32_t>();
    auto tensorGFP32 = gFp32Buf.template Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.template Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.template Get<float32_t>();
    for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; ++loopIdx) {
        uint32_t chunkIdx = 0;
        uint32_t hBegin = 0;
        uint32_t hEnd = 0;
        DecodeRecomputeTask<kFlattenHeadTasks, kAbcTaskOrder>(
            loopIdx, cu_seqlens, Hv, T, ChunkSize(), chunkNum, hvPerHk, chunkIdx, hBegin, hEnd);
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, ChunkSize(), chunkIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (uint32_t h = hBegin; h < hEnd; ++h) {
            ++vecTaskIdx;
            const bool isLast = loopIdx + 1 == loopEnd && h + 1 == hEnd;
            if ((vecTaskIdx & 0b01) != subBlockIdx) {
                if constexpr (!kPrepareOnly) {
                    if ((vecTaskIdx & 0b11) == 0 || isLast) {
                        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
                        if (vecTaskIdx > GDN::RECOMPUTE_WU_SYNC_WINDOW_TASKS) {
                            CrossCoreWaitFlag<0x2>(SYNC_AIC_AIV_FLAG_5);
                        }
                    }
                }
                continue;
            }
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * ValueDim();
                auto betaOffset = h * T + bos + rowOffset;
                uint32_t rawSlot = 0;
                LocalTensor<kType> tensorVin;
                LocalTensor<betaType> tensorBetain;
                LocalTensor<kType> tensorVbOut;
                rawSlot = rawBlock++ & 1;
                WaitFlag<HardEvent::V_MTE2>(frontResources->template Event<HardEvent::V_MTE2>(rawSlot + eventOffset));
                tensorVin = frontResources->template View<kType>(ubOffset + layout.input + rawSlot * inputSlotBytes, rowNum * ValueDim() * sizeof(kType));
                tensorBetain = frontResources->template View<betaType>(ubOffset + layout.beta + rawSlot * betaSlotBytes, rowNum * sizeof(betaType));
                tensorVbOut = frontResources->template View<kType>(ubOffset + layout.out + rawSlot * inputSlotBytes, rowNum * ValueDim() * sizeof(kType));
                // copyin
                {
                    if (InputSequenceMajor()) {
                        const uint64_t inputOffset = GDN::QkvSequenceMajorOffset(vOffset, T, Hv, ValueDim());
                        const uint64_t rowGap = static_cast<uint64_t>(Hv - 1) * ValueDim() * sizeof(kType);
                        if (rowGap <= 0xffffffffULL) {
                            DataCopyPad(tensorVin, vTensor[inputOffset],
                                {static_cast<uint16_t>(curRowNum), static_cast<uint32_t>(ValueDim() * sizeof(kType)),
                                 static_cast<uint32_t>(rowGap), 0, 0}, {false, 0, 0, 0});
                        } else {
                            // The DMA gap field is uint32 bytes; large strides remain correct.
                            for (uint32_t row = 0; row < curRowNum; ++row) {
                                DataCopy(tensorVin[row * ValueDim()],
                                    vTensor[inputOffset + static_cast<uint64_t>(row) * Hv * ValueDim()], ValueDim());
                            }
                        }
                    } else {
                        DataCopy(tensorVin, vTensor[vOffset], ValueDim() * curRowNum);
                    }
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});

                    SetFlag<HardEvent::MTE2_V>(frontResources->template Event<HardEvent::MTE2_V>(rawSlot + eventOffset));
                }
                // compute
                {
                    WaitFlag<HardEvent::MTE2_V>(frontResources->template Event<HardEvent::MTE2_V>(rawSlot + eventOffset));
                    WaitFlag<HardEvent::MTE3_V>(frontResources->template Event<HardEvent::MTE3_V>(rawSlot + eventOffset));

                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    Cast(tensorVFp32, tensorVin, RoundMode::CAST_NONE, ValueDim() * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = ValueDim() * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < ValueDim()) {
                        Mul(tensorVFp32[perchannelResOffset], tensorVFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorVbOut, tensorVFp32, RoundMode::CAST_RINT, ValueDim() * curRowNum);

                    SetFlag<HardEvent::V_MTE2>(frontResources->template Event<HardEvent::V_MTE2>(rawSlot + eventOffset));
                    SetFlag<HardEvent::V_MTE3>(frontResources->template Event<HardEvent::V_MTE3>(rawSlot + eventOffset));

                }
                // copyout
                {
                    WaitFlag<HardEvent::V_MTE3>(frontResources->template Event<HardEvent::V_MTE3>(rawSlot + eventOffset));
                    const uint64_t dstOffset = ((loopIdx - taskRange->begin) * ChunkSize() + rowOffset) * ValueDim();
                    DataCopy(workSpaceTensor[dstOffset], tensorVbOut, ValueDim() * curRowNum);
                    SetFlag<HardEvent::MTE3_V>(frontResources->template Event<HardEvent::MTE3_V>(rawSlot + eventOffset));
                }
            }

            uint64_t hk = h / hvPerHk;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                // 注意：定长场景下 GetChunkOffset 返回的 bos 已含按 Hv 计的批次偏移 bIdx*Hv*T
                // （见 recompute_w_u_fwd_common.h GetChunkOffset 的 bos += bIdx*H*T，此处 H 传入的是 Hv）。
                // k 只有 Hk 个 head，需把批次偏移换算成 bIdx*Hk*T，即 bos - bIdx*(Hv-Hk)*T。
                // 此换算强耦合于 GetChunkOffset 的批次偏移实现，若后者修改需同步更新此处。
                // coreLoopsInB 必须与 GetChunkOffset 内保持一致的算法。
                uint64_t bIdx = cu_seqlens ? 0 : (chunkIdx / coreLoopsInB);
                uint64_t bosK = cu_seqlens ? bos : (bos - bIdx * (Hv - Hk) * T);
                auto kSrcOffset = (hk * T + bosK + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
                uint32_t rawSlot = 0;
                LocalTensor<kType> tensorKin;
                LocalTensor<betaType> tensorBetain;
                LocalTensor<betaType> tensorGin;
                LocalTensor<kType> tensorOut;
                rawSlot = rawBlock++ & 1;
                WaitFlag<HardEvent::V_MTE2>(frontResources->template Event<HardEvent::V_MTE2>(rawSlot + eventOffset));
                tensorKin = frontResources->template View<kType>(ubOffset + layout.input + rawSlot * inputSlotBytes, rowNum * K * sizeof(kType));
                tensorBetain = frontResources->template View<betaType>(ubOffset + layout.beta + rawSlot * betaSlotBytes, rowNum * sizeof(betaType));
                tensorGin = frontResources->template View<betaType>(ubOffset + layout.g + rawSlot * betaSlotBytes, rowNum * sizeof(betaType));
                tensorOut = frontResources->template View<kType>(ubOffset + layout.out + rawSlot * inputSlotBytes, rowNum * K * sizeof(kType));
                // copyin
                {
                    if (InputSequenceMajor()) {
                        const uint64_t inputOffset = GDN::QkvSequenceMajorOffset(kSrcOffset, T, Hk, K);
                        const uint64_t rowGap = static_cast<uint64_t>(Hk - 1) * K * sizeof(kType);
                        if (rowGap <= 0xffffffffULL) {
                            DataCopyPad(tensorKin, kTensor[inputOffset],
                                {static_cast<uint16_t>(curRowNum), static_cast<uint32_t>(K * sizeof(kType)),
                                 static_cast<uint32_t>(rowGap), 0, 0}, {false, 0, 0, 0});
                        } else {
                            // The DMA gap field is uint32 bytes; large strides remain correct.
                            for (uint32_t row = 0; row < curRowNum; ++row) {
                                DataCopy(tensorKin[row * K],
                                    kTensor[inputOffset + static_cast<uint64_t>(row) * Hk * K], K);
                            }
                        }
                    } else {
                        DataCopy(tensorKin, kTensor[kSrcOffset], K * curRowNum);
                    }
                    if constexpr (!kReuseVbBeta) {
                        DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    }
                    DataCopyPad(tensorGin, gTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    SetFlag<HardEvent::MTE2_V>(frontResources->template Event<HardEvent::MTE2_V>(rawSlot + eventOffset));
                }
                // compute
                {
                    WaitFlag<HardEvent::MTE2_V>(frontResources->template Event<HardEvent::MTE2_V>(rawSlot + eventOffset));
                    WaitFlag<HardEvent::MTE3_V>(frontResources->template Event<HardEvent::MTE3_V>(rawSlot + eventOffset));
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorGFP32, tensorGin, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        if constexpr (!kReuseVbBeta) {
                            DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                        }
                        DataCopy(tensorGFP32, tensorGin, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    Exp(tensorGFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    Mul(tensorBetaFP32, tensorBetaFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorKFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    Cast(tensorOut, tensorKFp32, RoundMode::CAST_RINT, K * curRowNum);
                    SetFlag<HardEvent::V_MTE2>(frontResources->template Event<HardEvent::V_MTE2>(rawSlot + eventOffset));
                    SetFlag<HardEvent::V_MTE3>(frontResources->template Event<HardEvent::V_MTE3>(rawSlot + eventOffset));
                }
                // copyout
                {
                    WaitFlag<HardEvent::V_MTE3>(frontResources->template Event<HardEvent::V_MTE3>(rawSlot + eventOffset));
                    const uint64_t dstOffset = taskRange->capacity * ChunkSize() * ValueDim() +
                        ((loopIdx - taskRange->begin) * ChunkSize() + rowOffset) * K;
                    DataCopy(workSpaceTensor[dstOffset], tensorOut, K * curRowNum);
                    SetFlag<HardEvent::MTE3_V>(frontResources->template Event<HardEvent::MTE3_V>(rawSlot + eventOffset));
                }
            }
            if constexpr (!kPrepareOnly) {
                if ((vecTaskIdx & 0b11) == 0 || isLast) {
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
                    if (vecTaskIdx > GDN::RECOMPUTE_WU_SYNC_WINDOW_TASKS) {
                        CrossCoreWaitFlag<0x2>(SYNC_AIC_AIV_FLAG_5);
                    }
                }
            }
        }
    }
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kAbcTaskOrder, bool kRawFront,
          int kStaticV, int kStaticBT, int kQkvLayout>
template <bool kPrepareOnly>
__aicore__ inline void RecomputeWUFwdVectorProcess<kType, betaType,
    kFlattenHeadTasks, kAbcTaskOrder, kRawFront, kStaticV, kStaticBT, kQkvLayout>::EndInterleaved()
{
    constexpr uint32_t eventOffset = kPrepareOnly ? 2 : 0;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        WaitFlag<HardEvent::V_MTE2>(frontResources->template Event<HardEvent::V_MTE2>(slot + eventOffset));
        WaitFlag<HardEvent::MTE3_V>(frontResources->template Event<HardEvent::MTE3_V>(slot + eventOffset));
    }
}

#endif // RECOMPUTE_W_U_FWD_VECTOR_H
