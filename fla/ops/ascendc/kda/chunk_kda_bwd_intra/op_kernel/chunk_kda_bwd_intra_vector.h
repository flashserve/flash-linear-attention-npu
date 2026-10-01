/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_VECTOR_H
#define CHUNK_KDA_BWD_INTRA_VECTOR_H

#include "kernel_operator.h"
#include "chunk_kda_bwd_intra_struct.h"
#include "chunk_kda_bwd_intra_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "catlass/arch/cross_core_sync.hpp"
#include "arch35/chunk_kda_bwd_intra_regbase.h"
#endif

namespace KDA {

template <
    uint32_t K_DIM, uint32_t CHUNK_SIZE, bool SAFE_GATE, bool VARLEN_TND,
    typename BetaT>
class ChunkKdaBwdIntraVectorProcess {
public:
    __aicore__ ChunkKdaBwdIntraVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR gk, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg, GM_ADDR dqOut, GM_ADDR dkOut,
        GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR chunkMetadata, GM_ADDR workspace)
        : q_(q), k_(k), gk_(gk), beta_(beta), dAqk_(dAqk), dAkk_(dAkk),
          dq_(dq), dk_(dk), db_(db), dg_(dg), dqOut_(dqOut), dkOut_(dkOut),
          dbOut_(dbOut), dgOut_(dgOut), chunkMetadata_(chunkMetadata),
          workspace_(workspace)
    {
        static_assert(SAFE_GATE, "The unsafe gate branch is reserved but not instantiated in v1.");
        static_assert(!VARLEN_TND || K_DIM == 128, "Varlen P0 only supports K=128.");
    }

    __aicore__ inline void Init(const ChunkKdaBwdIntraTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q_));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k_));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk_));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta_));
        dAqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk_));
        dAkkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk_));
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq_));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk_));
        dbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db_));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg_));
        dqOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqOut_));
        dkOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dkOut_));
        dbOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbOut_));
        dgOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgOut_));
        if constexpr (VARLEN_TND) {
            chunkMetadataGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t *>(chunkMetadata_));
        }
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_));

        pipe_->InitBuffer(inputQueue_, 2, kIoBufferBytes);
        pipe_->InitBuffer(outputQueue_, 2, kIoBufferBytes);
        pipe_->InitBuffer(matrixInputPing_, kIoBufferBytes);
        pipe_->InitBuffer(matrixInputPong_, kIoBufferBytes);
        pipe_->InitBuffer(arena_, kArenaBytes);
        pipe_->InitBuffer(reduceTmp_, kReduceTmpBytes);
        InitMatrixInputEvents();
        if constexpr (VARLEN_TND) {
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
            auto offsets = ScalarGatherOffsets();
            for (uint32_t row = 0; row < kProcessRowBlock; ++row) {
                offsets.SetValue(
                    row, row * (32 / sizeof(float)) * sizeof(float));
                offsets.SetValue(
                    kProcessRowBlock + row,
                    row * (32 / sizeof(bfloat16_t)) * sizeof(float));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
#endif
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kHeadsPerWindow - 1) / kHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
        uint32_t taskIdx = coreIdx / headWindowCount;
        uint32_t headWindowIdx = coreIdx % headWindowCount;
        const uint32_t taskStride = coreNum / headWindowCount;
        const uint32_t headWindowStride = coreNum % headWindowCount;
        uint64_t windowIdx = 0;

        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum) {
            const uint32_t headBase = headWindowIdx * kHeadsPerWindow;
            const uint32_t headCount = headBase + 1 < headNum ? 2 : 1;
            const ChunkTask task =
                GetChunkTask<VARLEN_TND>(tiling_, chunkMetadataGm_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            const uint32_t rowBlockCount =
                (validLen + kProcessRowBlock - 1) / kProcessRowBlock;
            for (uint32_t rowBlock = 0; rowBlock < rowBlockCount; ++rowBlock) {
                const uint32_t rowStart = rowBlock * kProcessRowBlock;
                const uint32_t validRows =
                    rowStart + kProcessRowBlock <= validLen ?
                        kProcessRowBlock : validLen - rowStart;

                // PR190-style two-head stage ordering: finish Vector-Pre for
                // head0 then head1 before entering Vector-Post.  This lets
                // Cube(head0) overlap Vector-Pre(head1).
                for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
                    const uint32_t slot = WorkspaceSlot(windowIdx, headInWindow);
                    PrepareHead(task, headBase + headInWindow, rowStart, validRows, slot);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeReadyFlag_);
#else
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kVecToCubeReadyFlag);
#endif
                }
                for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecReadyFlag_);
#else
                    AscendC::CrossCoreWaitFlag(kCubeToVecReadyFlag);
#endif
                    const uint32_t slot = WorkspaceSlot(windowIdx, headInWindow);
                    FinishHead(task, headBase + headInWindow, rowStart, validRows, slot);
                }
                ++windowIdx;
            }
            taskIdx += taskStride;
            headWindowIdx += headWindowStride;
            if (headWindowIdx >= headWindowCount) {
                headWindowIdx -= headWindowCount;
                ++taskIdx;
            }
        }
        ReleaseMatrixInputEvents();
    }

private:
    static constexpr uint32_t kProcessRowBlock =
        ProcessRowBlock<K_DIM, VARLEN_TND>::value;
    static constexpr uint32_t kPlaneElements = 8 * 128;
    // Each AIV owns half of K in PackLowerB.  For the target K=128 path,
    // 16 FP32 rows x 64 columns exactly fill one 4 KiB arena plane.
    static constexpr uint32_t kLowerPackRows = K_DIM == 128 ? 16 : 8;
    // PackUpperB owns all 128 columns.  Its K=128 hot path uses two
    // consecutive 4 KiB arena planes and one 8 KiB IO tile per operand.
    static constexpr uint32_t kUpperPackRows = K_DIM == 128 ? 16 : 8;
    static constexpr uint32_t kUpperMatrixPlanes = 2;
    static constexpr uint32_t kIoBufferBytes = 8 * 1024;
    static constexpr uint32_t kArenaBytes = 96 * 1024;
    static constexpr uint32_t kReduceTmpBytes = 32 * 1024;
    static constexpr uint32_t kUbBudgetBytes = 192 * 1024;
    static constexpr uint32_t kFp32BlockElements = 32 / sizeof(float);
    static constexpr uint32_t kMatrixInputBufferCount = 2;
    static_assert(
        6 * kIoBufferBytes + kArenaBytes + kReduceTmpBytes <= kUbBudgetBytes,
        "Vector UB buffers exceed the A2/A3 192 KiB budget.");
    static_assert(kProcessRowBlock * CHUNK_SIZE <= 2 * kPlaneElements,
                  "Packed A-matrix tile exceeds its two-plane arena group.");
    static_assert(kProcessRowBlock * CHUNK_SIZE * sizeof(float) <= kIoBufferBytes,
                  "Packed A-matrix FP32 tile exceeds the IO queue buffer.");
    static_assert(kLowerPackRows * (K_DIM / 2) <= kPlaneElements,
                  "PackLowerB row tile exceeds one arena plane.");
    static_assert(kLowerPackRows * (K_DIM / 2) * sizeof(float) <= kIoBufferBytes,
                  "PackLowerB FP32 row tile exceeds the IO queue buffer.");
    static_assert(kUpperPackRows * 128 <= kUpperMatrixPlanes * kPlaneElements,
                  "PackUpperB row tile exceeds its arena plane group.");
    static_assert(kUpperPackRows * 128 * sizeof(float) <= kIoBufferBytes,
                  "PackUpperB FP32 row tile exceeds the IO queue buffer.");
    static_assert(24 * kPlaneElements * sizeof(float) <= kArenaBytes,
                  "Vector scratch layout exceeds the arena buffer.");

    __aicore__ inline AscendC::LocalTensor<float> Plane(uint32_t index)
    {
        return arena_.Get<float>()[index * kPlaneElements];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> ScalarGatherOffsets()
    {
        return arena_.Get<uint32_t>()[22 * kPlaneElements];
    }

    __aicore__ inline AscendC::LocalTensor<float> ScalarStage()
    {
        return Plane(23);
    }

    __aicore__ inline void InitMatrixInputEvents()
    {
        for (uint32_t slot = 0; slot < kMatrixInputBufferCount; ++slot) {
            matrixMte2ToVEvent_[slot] =
                static_cast<event_t>(
                    pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
            matrixVToMte2Event_[slot] =
                static_cast<event_t>(
                    pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
        }
    }

    __aicore__ inline void ReleaseMatrixInputEvents()
    {
        for (uint32_t slot = 0; slot < kMatrixInputBufferCount; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(
                matrixMte2ToVEvent_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
        }
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> MatrixInput(uint32_t slot)
    {
        return slot == 0 ? matrixInputPing_.Get<T>() : matrixInputPong_.Get<T>();
    }

    template <typename T>
    __aicore__ inline uint32_t CopyInMatrixRows(
        AscendC::GlobalTensor<T> src, uint32_t rows, uint32_t cols,
        uint32_t srcRowElements)
    {
        const uint32_t slot = currentMatrixInputSlot_;
        currentMatrixInputSlot_ ^= 1U;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((srcRowElements - cols) * sizeof(T)),
            0,
            0
        };
        AscendC::DataCopyPad(
            MatrixInput<T>(slot), src, copyParams, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
        return slot;
    }

    __aicore__ inline void ConsumeMatrixRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t slot, uint32_t count)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            count);
#else
        AscendC::Adds(dst, src, 0.0f, count);
#endif
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
    }

    __aicore__ inline void ConsumeMatrixRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<bfloat16_t> src,
        uint32_t slot, uint32_t count)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCastBf16ToFp32(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)src.GetPhyAddr(),
            count);
#else
        AscendC::Cast(
            dst, src, AscendC::RoundMode::CAST_NONE, count);
#endif
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
    }

    template <typename T0, typename T1>
    __aicore__ inline void LoadMatrixRowsPair(
        AscendC::LocalTensor<float> dst0, AscendC::GlobalTensor<T0> src0,
        uint32_t rows0, uint32_t cols0, uint32_t srcRowElements0,
        AscendC::LocalTensor<float> dst1, AscendC::GlobalTensor<T1> src1,
        uint32_t rows1, uint32_t cols1, uint32_t srcRowElements1)
    {
        // Issue both GM reads before consuming either input.  The second
        // MTE2 transfer can overlap the first input's Vector conversion.
        const uint32_t slot0 =
            CopyInMatrixRows(src0, rows0, cols0, srcRowElements0);
        const uint32_t slot1 =
            CopyInMatrixRows(src1, rows1, cols1, srcRowElements1);
        ConsumeMatrixRows(
            dst0, MatrixInput<T0>(slot0), slot0, rows0 * cols0);
        ConsumeMatrixRows(
            dst1, MatrixInput<T1>(slot1), slot1, rows1 * cols1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, uint32_t count)
    {
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyPad(
            input, src, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)ready.GetPhyAddr(),
            count);
#else
        AscendC::Adds(dst, ready, 0.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
#endif
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<bfloat16_t> src,
        uint32_t count)
    {
        auto input = inputQueue_.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(
            input, src, {1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0},
            {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<bfloat16_t>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCastBf16ToFp32(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)ready.GetPhyAddr(),
            count);
#else
        AscendC::Cast(dst, ready, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
#endif
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Store(
        AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t count)
    {
        AscendC::PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            count);
#else
        AscendC::Adds(output, src, 0.0f, count);
#endif
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<float>();
        AscendC::DataCopyPad(
            dst, ready, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
        uint32_t rows, uint32_t cols, uint32_t srcRowElements)
    {
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            static_cast<uint32_t>((srcRowElements - cols) * sizeof(float)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)ready.GetPhyAddr(),
            rows * cols);
#else
        AscendC::Adds(dst, ready, 0.0f, rows * cols);
        AscendC::PipeBarrier<PIPE_V>();
#endif
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<bfloat16_t> src,
        uint32_t rows, uint32_t cols, uint32_t srcRowElements)
    {
        auto input = inputQueue_.AllocTensor<bfloat16_t>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
            static_cast<uint32_t>((srcRowElements - cols) * sizeof(bfloat16_t)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<bfloat16_t>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCastBf16ToFp32(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)ready.GetPhyAddr(),
            rows * cols);
#else
        AscendC::Cast(dst, ready, AscendC::RoundMode::CAST_NONE, rows * cols);
        AscendC::PipeBarrier<PIPE_V>();
#endif
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void StoreRows(
        AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows, uint32_t cols, uint32_t dstRowElements)
    {
        AscendC::PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            rows * cols);
#else
        AscendC::Adds(output, src, 0.0f, rows * cols);
#endif
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            0,
            static_cast<uint32_t>((dstRowElements - cols) * sizeof(float)),
            0
        };
        AscendC::DataCopyPad(dst, ready, copyParams);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline uint32_t TensorRowElements() const
    {
        if constexpr (VARLEN_TND) {
            return static_cast<uint32_t>(tiling_.headNum) * K_DIM;
        }
        return K_DIM;
    }

    __aicore__ inline uint32_t MatrixRowElements() const
    {
        if constexpr (VARLEN_TND) {
            return static_cast<uint32_t>(tiling_.headNum) * CHUNK_SIZE;
        }
        return CHUNK_SIZE;
    }

    __aicore__ inline uint32_t ScalarRowElements() const
    {
        if constexpr (VARLEN_TND) {
            return static_cast<uint32_t>(tiling_.headNum);
        }
        return 1;
    }

    __aicore__ inline void LoadStridedScalars(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src,
        uint32_t rows, uint32_t srcRowElements)
    {
        constexpr uint32_t kLocalRowElements = 32 / sizeof(float);
        const uint32_t stagedElements = rows * kLocalRowElements;
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(float)),
            static_cast<uint32_t>((srcRowElements - 1) * sizeof(float)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<float>();
        auto stage = ScalarStage();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)stage.GetPhyAddr(),
            (__ubuf__ float *)ready.GetPhyAddr(),
            stagedElements);
        inputQueue_.FreeTensor(ready);
        KdaRegbaseGatherScalars(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)stage.GetPhyAddr(),
            rows, kLocalRowElements);
#else
        AscendC::Adds(stage, ready, 0.0f, stagedElements);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
        AscendC::Gather(
            dst, stage, ScalarGatherOffsets(), static_cast<uint32_t>(0), rows);
        AscendC::PipeBarrier<PIPE_V>();
#endif
    }

    __aicore__ inline void LoadStridedScalars(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<bfloat16_t> src,
        uint32_t rows, uint32_t srcRowElements)
    {
        constexpr uint32_t kLocalRowElements = 32 / sizeof(bfloat16_t);
        const uint32_t stagedElements = rows * kLocalRowElements;
        auto input = inputQueue_.AllocTensor<bfloat16_t>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(bfloat16_t)),
            static_cast<uint32_t>((srcRowElements - 1) * sizeof(bfloat16_t)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<bfloat16_t>();
        auto stage = ScalarStage();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCastBf16ToFp32(
            (__ubuf__ float *)stage.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)ready.GetPhyAddr(),
            stagedElements);
        inputQueue_.FreeTensor(ready);
        KdaRegbaseGatherScalars(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)stage.GetPhyAddr(),
            rows, kLocalRowElements);
#else
        AscendC::Cast(stage, ready, AscendC::RoundMode::CAST_NONE, stagedElements);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
        AscendC::Gather(
            dst, stage, ScalarGatherOffsets()[kProcessRowBlock],
            static_cast<uint32_t>(0), rows);
        AscendC::PipeBarrier<PIPE_V>();
#endif
    }

    __aicore__ inline void LoadScalarRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src, uint32_t rows)
    {
        if constexpr (VARLEN_TND) {
            LoadStridedScalars(dst, src, rows, ScalarRowElements());
        } else {
            Load(dst, src, rows);
        }
    }

    __aicore__ inline void LoadScalarRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<bfloat16_t> src, uint32_t rows)
    {
        if constexpr (VARLEN_TND) {
            LoadStridedScalars(dst, src, rows, ScalarRowElements());
        } else {
            Load(dst, src, rows);
        }
    }

    __aicore__ inline void StoreScalarRows(
        AscendC::GlobalTensor<float> dst,
        AscendC::LocalTensor<float> src, uint32_t rows)
    {
        if constexpr (!VARLEN_TND) {
            Store(dst, src, rows);
            return;
        }
        constexpr uint32_t kLocalRowElements = 32 / sizeof(float);
        AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        auto output = outputQueue_.AllocTensor<float>();
        KdaRegbaseScatterScalars(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            rows, kLocalRowElements);
#else
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        auto output = outputQueue_.AllocTensor<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Duplicate(
                output[row * kLocalRowElements], src.GetValue(row), 1);
        }
#endif
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<float>();
        // For VECOUT->GM DataCopyPad, a non-aligned local block is rounded to
        // one 32-byte data block.  Adjacent staged rows therefore use
        // srcStride=0; GM dstStride remains byte-based.
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(float)),
            0,
            static_cast<uint32_t>((ScalarRowElements() - 1) * sizeof(float)),
            0
        };
        AscendC::DataCopyPad(dst, ready, copyParams);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline uint64_t SlotBase(uint32_t slot) const
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        return static_cast<uint64_t>(coreIdx) * tiling_.workspaceCoreSize +
               static_cast<uint64_t>(slot) * tiling_.workspaceSlotSize;
    }

    __aicore__ inline void Exp2(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t count)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseExp2(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            count);
#else
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(dst, src, kLn2, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dst, dst, count);
#endif
    }

    __aicore__ inline void PrepareHead(
        const ChunkTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t slot)
    {
        const uint64_t slotBase = SlotBase(slot);
        const uint32_t prefix = rowStart + validRows;
        const uint32_t future = task.end - task.begin - rowStart;
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        PackLowerA(task, head, rowStart, validRows, prefix, subBlock, slotBase);
        PackUpperA(task, head, rowStart, validRows, future, subBlock, slotBase);
        PackLowerB(task, head, rowStart, validRows, prefix, subBlock, slotBase);
        PackUpperB(task, head, rowStart, future, subBlock, slotBase);
    }

    __aicore__ inline void PackLowerA(
        const ChunkTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t prefix, uint32_t subBlock, uint64_t slotBase)
    {
        auto work = Plane(0);
        // The A5 row32 tile occupies two consecutive 4 KiB planes.  Keep the
        // masked destination in a disjoint two-plane group.
        auto masked = Plane(2);
        auto &source = subBlock == 0 ? dAqkGm_ : dAkkGm_;
        const uint32_t rowBase = subBlock * kProcessRowBlock;

        // DataCopyPad lays out each UB row at a 32-byte boundary.  Full chunks
        // therefore use one 16-row transfer for the aligned 16/32/48/64
        // prefixes.  Preserve the row-wise path for non-aligned tail chunks.
        if (prefix % kFp32BlockElements == 0) {
            const uint64_t srcOffset =
                MatrixOffset<VARLEN_TND>(
                    tiling_, task.batchIdx, head, task.begin + rowStart);
            LoadRows(work, source[srcOffset], validRows, prefix, MatrixRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseMaskLowerA(
                (__ubuf__ float *)masked.GetPhyAddr(),
                (__ubuf__ float *)work.GetPhyAddr(),
                validRows, rowStart, prefix, kProcessRowBlock);
#else
            AscendC::Duplicate(masked, 0.0f, kProcessRowBlock * prefix);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < validRows; ++row) {
                const uint32_t validCols = rowStart + row + 1;
                AscendC::Adds(
                    masked[row * prefix], work[row * prefix], 0.0f, validCols);
            }
#endif
            const uint64_t dstOffset =
                slotBase / sizeof(float) + tiling_.aLowerOffset / sizeof(float) +
                static_cast<uint64_t>(rowBase) * prefix;
            StoreRows(
                workspaceGm_[dstOffset], masked, kProcessRowBlock, prefix, prefix);
            return;
        }

        for (uint32_t row = 0; row < kProcessRowBlock; ++row) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseFill(
                (__ubuf__ float *)work.GetPhyAddr(), 0.0f, prefix);
#else
            AscendC::Duplicate(work, 0.0f, prefix);
#endif
            if (row < validRows) {
                const uint32_t validCols = rowStart + row + 1;
                const uint64_t srcOffset =
                    MatrixOffset<VARLEN_TND>(
                        tiling_, task.batchIdx, head, task.begin + rowStart + row);
                Load(work, source[srcOffset], validCols);
            }
            const uint64_t dstOffset =
                slotBase / sizeof(float) + tiling_.aLowerOffset / sizeof(float) +
                static_cast<uint64_t>(rowBase + row) * prefix;
            Store(workspaceGm_[dstOffset], work, prefix);
        }
    }

    __aicore__ inline void PackUpperA(
        const ChunkTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t future, uint32_t subBlock, uint64_t slotBase)
    {
        auto work = Plane(0);
        auto masked = Plane(2);
        auto &source = subBlock == 0 ? dAqkGm_ : dAkkGm_;
        const uint32_t physicalRowBase = subBlock * future;

        // The largest Upper-A tile is 64 rows x 16 FP32 values, exactly one
        // 4 KiB IO buffer.  Move the complete future range in one MTE2/MTE3
        // batch instead of issuing one pair of transfers per 8-row tile.
        const uint64_t srcOffset =
            MatrixOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, task.begin + rowStart, rowStart);
        LoadRows(
            work, source[srcOffset], future, kProcessRowBlock,
            MatrixRowElements());

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseMaskUpperA(
            (__ubuf__ float *)masked.GetPhyAddr(),
            (__ubuf__ float *)work.GetPhyAddr(),
            future, validRows, kProcessRowBlock);
#else
        AscendC::Duplicate(masked, 0.0f, future * kProcessRowBlock);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t maskedRows = future < validRows ? future : validRows;
        for (uint32_t row = 0; row < maskedRows; ++row) {
            AscendC::Adds(
                masked[row * kProcessRowBlock], work[row * kProcessRowBlock],
                0.0f, row + 1);
        }
        if (future > maskedRows) {
            const uint32_t tailElements =
                (future - maskedRows) * kProcessRowBlock;
            AscendC::Adds(
                masked[maskedRows * kProcessRowBlock],
                work[maskedRows * kProcessRowBlock], 0.0f, tailElements);
        }
#endif

        const uint64_t dstOffset =
            slotBase / sizeof(float) + tiling_.aUpperOffset / sizeof(float) +
            static_cast<uint64_t>(physicalRowBase) * kProcessRowBlock;
        StoreRows(
            workspaceGm_[dstOffset], masked, future, kProcessRowBlock,
            kProcessRowBlock);
    }

    __aicore__ inline void PackLowerB(
        const ChunkTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t prefix, uint32_t subBlock, uint64_t slotBase)
    {
        auto data = Plane(0);
        auto gate = Plane(1);
        auto anchor = Plane(2);
        auto exponent = Plane(3);
        const uint32_t cols = K_DIM / 2;
        const uint32_t col = subBlock * cols;
        const uint32_t anchorRow =
            task.begin + rowStart + (validRows > 8 ? 8 : validRows - 1);
        Load(anchor,
             gkGm_[TensorOffset<VARLEN_TND>(
                 tiling_, task.batchIdx, head, anchorRow, col)],
             cols);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        for (uint32_t row = 1; row < kLowerPackRows; ++row) {
            AscendC::Adds(anchor[row * cols], anchor, 0.0f, cols);
        }
        AscendC::PipeBarrier<PIPE_V>();
#endif
        for (uint32_t sourceRow = 0; sourceRow < prefix; sourceRow += kLowerPackRows) {
            const uint32_t rows =
                sourceRow + kLowerPackRows <= prefix ? kLowerPackRows : prefix - sourceRow;
            const uint32_t count = rows * cols;
            const uint32_t token = task.begin + sourceRow;
            LoadMatrixRowsPair(
                data,
                kGm_[TensorOffset<VARLEN_TND>(
                    tiling_, task.batchIdx, head, token, col)],
                rows, cols, TensorRowElements(),
                gate,
                gkGm_[TensorOffset<VARLEN_TND>(
                    tiling_, task.batchIdx, head, token, col)],
                rows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseGateScale<true, false>(
                (__ubuf__ float *)data.GetPhyAddr(),
                (__ubuf__ float *)gate.GetPhyAddr(),
                (__ubuf__ float *)anchor.GetPhyAddr(),
                (__ubuf__ float *)0, rows, cols);
#else
            AscendC::Sub(exponent, anchor, gate, count);
            Exp2(exponent, exponent, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(data, data, exponent, count);
#endif
            const uint64_t dstOffset =
                slotBase / sizeof(float) + tiling_.bLowerOffset / sizeof(float) +
                static_cast<uint64_t>(sourceRow) * K_DIM + col;
            StoreRows(workspaceGm_[dstOffset], data, rows, cols, K_DIM);
        }
    }

    __aicore__ inline void PackUpperB(
        const ChunkTask &task, uint32_t head, uint32_t rowStart, uint32_t future,
        uint32_t subBlock, uint64_t slotBase)
    {
        // data/gate/anchor/exponent may each contain 16 x 128 FP32 values.
        // Give every matrix two adjacent 4 KiB planes; beta temporaries start
        // after those non-overlapping 8 KiB regions.
        auto data = Plane(0);
        auto gate = Plane(2);
        auto anchor = Plane(4);
        auto exponent = Plane(6);
        auto beta = Plane(8);
        auto betaBroadcast = Plane(9);
        const uint32_t anchorLocal = rowStart + 8 < task.end - task.begin ?
                                     rowStart + 8 : task.end - task.begin - 1;
        const uint32_t anchorRow = task.begin + anchorLocal;
        for (uint32_t col = 0; col < K_DIM; col += 128) {
            const uint32_t cols = col + 128 <= K_DIM ? 128 : K_DIM - col;
            Load(anchor,
                 gkGm_[TensorOffset<VARLEN_TND>(
                     tiling_, task.batchIdx, head, anchorRow, col)],
                 cols);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
            for (uint32_t row = 1; row < kUpperPackRows; ++row) {
                AscendC::Adds(anchor[row * cols], anchor, 0.0f, cols);
            }
            AscendC::PipeBarrier<PIPE_V>();
#endif
            for (uint32_t sourceRow = 0; sourceRow < future; sourceRow += kUpperPackRows) {
                const uint32_t rows =
                    sourceRow + kUpperPackRows <= future ? kUpperPackRows : future - sourceRow;
                const uint32_t count = rows * cols;
                const uint32_t token = task.begin + rowStart + sourceRow;
                auto &source = subBlock == 0 ? qGm_ : kGm_;
                LoadMatrixRowsPair(
                    data,
                    source[TensorOffset<VARLEN_TND>(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements(),
                    gate,
                    gkGm_[TensorOffset<VARLEN_TND>(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if (subBlock != 0) {
                    LoadScalarRows(
                        beta,
                        betaGm_[ScalarOffset<VARLEN_TND>(
                            tiling_, task.batchIdx, head, token)],
                        rows);
                    KdaRegbaseGateScale<false, true>(
                        (__ubuf__ float *)data.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)anchor.GetPhyAddr(),
                        (__ubuf__ float *)beta.GetPhyAddr(),
                        rows, cols);
                } else {
                    KdaRegbaseGateScale<false, false>(
                        (__ubuf__ float *)data.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)anchor.GetPhyAddr(),
                        (__ubuf__ float *)0, rows, cols);
                }
#else
                AscendC::Sub(exponent, gate, anchor, count);
                Exp2(exponent, exponent, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(data, data, exponent, count);
                if (subBlock != 0) {
                    LoadScalarRows(
                        beta,
                        betaGm_[ScalarOffset<VARLEN_TND>(
                            tiling_, task.batchIdx, head, token)],
                        rows);
                    AscendC::Brcb(
                        betaBroadcast, beta, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint8_t rowStride =
                        static_cast<uint8_t>(cols * sizeof(float) / 32);
                    for (uint32_t offset = 0; offset < cols; offset += 64) {
                        const uint32_t mask = offset + 64 <= cols ? 64 : cols - offset;
                        AscendC::Mul(data[offset], data[offset], betaBroadcast, mask, rows,
                                   {1, 1, 0, rowStride, rowStride, 1});
                    }
                }
#endif
                const uint32_t physicalRow = subBlock * future + sourceRow;
                const uint64_t dstOffset =
                    slotBase / sizeof(float) + tiling_.bUpperOffset / sizeof(float) +
                    static_cast<uint64_t>(physicalRow) * K_DIM + col;
                StoreRows(workspaceGm_[dstOffset], data, rows, cols, K_DIM);
            }
        }
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void FinishHeadA5Dense16(
        const ChunkTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t slot)
    {
        static_assert(K_DIM == 128,
                      "The A5 16-row Vector-Post path is specialized for K=128.");
        static_assert(!VARLEN_TND,
                      "The A5 16-row Vector-Post path is dense-only.");
        static_assert(kProcessRowBlock == 32,
                      "The A5 16-row Vector-Post path requires the proven row32 outer tile.");

        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t ownedBegin = subBlock * 16;
        if (ownedBegin >= validRows) {
            return;
        }
        const uint32_t ownedRows =
            ownedBegin + 16 <= validRows ? 16 : validRows - ownedBegin;
        const uint32_t tokenBegin = task.begin + rowStart + ownedBegin;
        const uint64_t slotBase = SlotBase(slot);

        // One 16 x 128 FP32 matrix occupies two consecutive 4 KiB planes.
        // Keep only the values that remain live across the scale and output
        // phases.  Once scale finishes, gate/anchor/beta are dead and their
        // planes are reused by inputGrad/output, so the existing 96 KiB arena
        // is sufficient and no additional UB is reserved.
        auto rawDq = Plane(0);       // planes 0-1
        auto rawDkLower = Plane(2);  // planes 2-3
        auto rawDkUpper = Plane(4);  // planes 4-5
        auto k = Plane(6);           // planes 6-7
        auto gate = Plane(8);        // planes 8-9; reused by inputGrad
        auto q = Plane(10);          // planes 10-11
        auto anchor = Plane(12);     // plane 12; reused by output
        auto beta = Plane(13);       // plane 13; reused by output
        auto dbAcc = Plane(14);      // plane 14

        KdaRegbaseFill(
            (__ubuf__ float *)dbAcc.GetPhyAddr(), 0.0f, ownedRows);

        const uint32_t anchorLocal = rowStart + 8 < task.end - task.begin ?
                                     rowStart + 8 : task.end - task.begin - 1;
        const uint32_t anchorRow = task.begin + anchorLocal;
        LoadScalarRows(
            beta,
            betaGm_[ScalarOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);

        const uint32_t cols = 128;
        const uint32_t count = ownedRows * cols;
        Load(anchor,
             gkGm_[TensorOffset<VARLEN_TND>(
                 tiling_, task.batchIdx, head, anchorRow, 0)],
             cols);
        const uint64_t resultBase =
            slotBase / sizeof(float) + tiling_.resultRegionOffset / sizeof(float);
        LoadMatrixRowsPair(
            q,
            qGm_[TensorOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements(),
            k,
            kGm_[TensorOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        LoadMatrixRowsPair(
            gate,
            gkGm_[TensorOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements(),
            rawDq,
            workspaceGm_[resultBase + tiling_.resultDqOffset / sizeof(float) +
                         static_cast<uint64_t>(ownedBegin) * K_DIM],
            ownedRows, cols, K_DIM);
        LoadMatrixRowsPair(
            rawDkLower,
            workspaceGm_[resultBase + tiling_.resultDkLowerOffset / sizeof(float) +
                         static_cast<uint64_t>(ownedBegin) * K_DIM],
            ownedRows, cols, K_DIM,
            rawDkUpper,
            workspaceGm_[resultBase + tiling_.resultDkUpperOffset / sizeof(float) +
                         static_cast<uint64_t>(ownedBegin) * K_DIM],
            ownedRows, cols, K_DIM);

        KdaRegbaseFinishScale(
            (__ubuf__ float *)rawDq.GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
            (__ubuf__ float *)k.GetPhyAddr(),
            (__ubuf__ float *)gate.GetPhyAddr(),
            (__ubuf__ float *)anchor.GetPhyAddr(),
            (__ubuf__ float *)beta.GetPhyAddr(),
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            ownedRows, cols);

        auto output = Plane(12);  // planes 12-13, after anchor/beta's last use

        // The original gradients are consumed once and do not need to remain
        // in the arena.  Reuse the existing 8 KiB matrix ping/pong buffers as
        // their final RegBase operands: issue dq/dk together, then refill the
        // released dq slot with dg while dq is being copied out.  This removes
        // three 16 x 128 FP32 UB-to-UB copies and overlaps the mandatory MTE2
        // reads with the neighbouring Vector/MTE3 work without increasing UB.
        const uint32_t dqSlot = CopyInMatrixRows(
            dqGm_[TensorOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        const uint32_t dkSlot = CopyInMatrixRows(
            dkGm_[TensorOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[dqSlot]);
        KdaRegbaseAdd2(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)MatrixInput<float>(dqSlot).GetPhyAddr(),
            (__ubuf__ float *)rawDq.GetPhyAddr(),
            count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[dqSlot]);

        const uint32_t dgSlot = CopyInMatrixRows(
            dgGm_[TensorOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        StoreRows(dqOutGm_[TensorOffset<VARLEN_TND>(
                      tiling_, task.batchIdx, head, tokenBegin, 0)],
                  output, ownedRows, cols, TensorRowElements());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[dkSlot]);
        KdaRegbaseAdd3(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)MatrixInput<float>(dkSlot).GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
            count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[dkSlot]);
        StoreRows(dkOutGm_[TensorOffset<VARLEN_TND>(
                      tiling_, task.batchIdx, head, tokenBegin, 0)],
                  output, ownedRows, cols, TensorRowElements());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[dgSlot]);
        KdaRegbaseDg(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)MatrixInput<float>(dgSlot).GetPhyAddr(),
            (__ubuf__ float *)q.GetPhyAddr(),
            (__ubuf__ float *)rawDq.GetPhyAddr(),
            (__ubuf__ float *)k.GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
            count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[dgSlot]);
        StoreRows(dgOutGm_[TensorOffset<VARLEN_TND>(
                      tiling_, task.batchIdx, head, tokenBegin, 0)],
                  output, ownedRows, cols, TensorRowElements());

        // output is no longer live; plane 13 can hold the final db input.
        LoadScalarRows(
            beta,
            dbGm_[ScalarOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);
        KdaRegbaseAdd2(
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)beta.GetPhyAddr(),
            ownedRows);
        StoreScalarRows(
            dbOutGm_[ScalarOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin)],
            dbAcc, ownedRows);
    }
#endif

    __aicore__ inline void FinishHead(
        const ChunkTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t slot)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!VARLEN_TND && K_DIM == 128) {
            FinishHeadA5Dense16(task, head, rowStart, validRows, slot);
            return;
        }
#endif
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock = kProcessRowBlock / 2;
        // Preserve the proven eight-row path for A2/A3, A5 varlen and all
        // non-K128 specializations.
        for (uint32_t ownedOffset = 0; ownedOffset < rowsPerSubBlock;
             ownedOffset += 8) {
            const uint32_t ownedBegin =
                subBlock * rowsPerSubBlock + ownedOffset;
            if (ownedBegin >= validRows) {
                continue;
            }
            const uint32_t ownedRows =
                ownedBegin + 8 <= validRows ? 8 : validRows - ownedBegin;
            const uint32_t tokenBegin = task.begin + rowStart + ownedBegin;
            const uint64_t slotBase = SlotBase(slot);

        auto rawDq = Plane(0);
        auto rawDkLower = Plane(1);
        auto rawDkUpper = Plane(2);
        auto q = Plane(3);
        auto k = Plane(4);
        auto gate = Plane(5);
        auto anchor = Plane(6);
        auto posScale = Plane(7);
        auto negScale = Plane(8);
        auto beta = Plane(9);
        auto betaBroadcast = Plane(10);
        auto inputGrad = Plane(11);
        auto temp = Plane(12);
        auto output = Plane(13);
        auto product = Plane(14);
        auto rowReduce = Plane(15);
        auto dbAcc = Plane(16);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseFill(
            (__ubuf__ float *)dbAcc.GetPhyAddr(), 0.0f, ownedRows);
#else
        AscendC::Duplicate(dbAcc, 0.0f, ownedRows);
#endif

        const uint32_t anchorLocal = rowStart + 8 < task.end - task.begin ?
                                     rowStart + 8 : task.end - task.begin - 1;
        const uint32_t anchorRow = task.begin + anchorLocal;
        LoadScalarRows(
            beta,
            betaGm_[ScalarOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        AscendC::Brcb(betaBroadcast, beta, static_cast<uint8_t>((ownedRows + 7) / 8), {1, 8});
#endif

        for (uint32_t col = 0; col < K_DIM; col += 128) {
            const uint32_t cols = col + 128 <= K_DIM ? 128 : K_DIM - col;
            const uint32_t count = ownedRows * cols;
            Load(anchor,
                 gkGm_[TensorOffset<VARLEN_TND>(
                     tiling_, task.batchIdx, head, anchorRow, col)],
                 cols);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
            for (uint32_t row = 1; row < ownedRows; ++row) {
                AscendC::Adds(anchor[row * cols], anchor, 0.0f, cols);
            }
#endif
            const uint64_t resultBase =
                slotBase / sizeof(float) + tiling_.resultRegionOffset / sizeof(float);
            LoadMatrixRowsPair(
                q,
                qGm_[TensorOffset<VARLEN_TND>(
                    tiling_, task.batchIdx, head, tokenBegin, col)],
                ownedRows, cols, TensorRowElements(),
                k,
                kGm_[TensorOffset<VARLEN_TND>(
                    tiling_, task.batchIdx, head, tokenBegin, col)],
                ownedRows, cols, TensorRowElements());
            LoadMatrixRowsPair(
                gate,
                gkGm_[TensorOffset<VARLEN_TND>(
                    tiling_, task.batchIdx, head, tokenBegin, col)],
                ownedRows, cols, TensorRowElements(),
                rawDq,
                workspaceGm_[resultBase + tiling_.resultDqOffset / sizeof(float) +
                             static_cast<uint64_t>(ownedBegin) * K_DIM + col],
                ownedRows, cols, K_DIM);
            LoadMatrixRowsPair(
                rawDkLower,
                workspaceGm_[resultBase + tiling_.resultDkLowerOffset / sizeof(float) +
                             static_cast<uint64_t>(ownedBegin) * K_DIM + col],
                ownedRows, cols, K_DIM,
                rawDkUpper,
                workspaceGm_[resultBase + tiling_.resultDkUpperOffset / sizeof(float) +
                             static_cast<uint64_t>(ownedBegin) * K_DIM + col],
                ownedRows, cols, K_DIM);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseFinishScale(
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                (__ubuf__ float *)k.GetPhyAddr(),
                (__ubuf__ float *)gate.GetPhyAddr(),
                (__ubuf__ float *)anchor.GetPhyAddr(),
                (__ubuf__ float *)beta.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                ownedRows, cols);
#else
            AscendC::Sub(posScale, gate, anchor, count);
            AscendC::Sub(negScale, anchor, gate, count);
            Exp2(posScale, posScale, count);
            Exp2(negScale, negScale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(rawDq, rawDq, posScale, count);
            AscendC::Mul(rawDkLower, rawDkLower, posScale, count);
            AscendC::Mul(rawDkUpper, rawDkUpper, negScale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, rawDkLower, k, count);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t reduceShape[2] = {ownedRows, cols};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                rowReduce, product, reduceTmp_.Get<uint8_t>(), reduceShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(dbAcc, dbAcc, rowReduce, ownedRows);

            const uint8_t rowStride = static_cast<uint8_t>(cols * sizeof(float) / 32);
            for (uint32_t offset = 0; offset < cols; offset += 64) {
                const uint32_t mask = offset + 64 <= cols ? 64 : cols - offset;
                AscendC::Mul(rawDkLower[offset], rawDkLower[offset], betaBroadcast,
                             mask, ownedRows, {1, 1, 0, rowStride, rowStride, 1});
            }
            AscendC::PipeBarrier<PIPE_V>();
#endif

            LoadRows(inputGrad,
                     dqGm_[TensorOffset<VARLEN_TND>(
                         tiling_, task.batchIdx, head, tokenBegin, col)],
                     ownedRows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseAdd2(
                (__ubuf__ float *)output.GetPhyAddr(),
                (__ubuf__ float *)inputGrad.GetPhyAddr(),
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                count);
#else
            AscendC::Add(output, inputGrad, rawDq, count);
#endif
            StoreRows(dqOutGm_[TensorOffset<VARLEN_TND>(
                          tiling_, task.batchIdx, head, tokenBegin, col)],
                      output, ownedRows, cols, TensorRowElements());

            LoadRows(inputGrad,
                     dkGm_[TensorOffset<VARLEN_TND>(
                         tiling_, task.batchIdx, head, tokenBegin, col)],
                     ownedRows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseAdd3(
                (__ubuf__ float *)output.GetPhyAddr(),
                (__ubuf__ float *)inputGrad.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                count);
#else
            AscendC::Add(temp, rawDkLower, rawDkUpper, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(output, inputGrad, temp, count);
#endif
            StoreRows(dkOutGm_[TensorOffset<VARLEN_TND>(
                          tiling_, task.batchIdx, head, tokenBegin, col)],
                      output, ownedRows, cols, TensorRowElements());

            LoadRows(inputGrad,
                     dgGm_[TensorOffset<VARLEN_TND>(
                         tiling_, task.batchIdx, head, tokenBegin, col)],
                     ownedRows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseDg(
                (__ubuf__ float *)output.GetPhyAddr(),
                (__ubuf__ float *)inputGrad.GetPhyAddr(),
                (__ubuf__ float *)q.GetPhyAddr(),
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                (__ubuf__ float *)k.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                count);
#else
            AscendC::Mul(temp, q, rawDq, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(output, inputGrad, temp, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(temp, rawDkLower, rawDkUpper, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(temp, k, temp, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(output, output, temp, count);
#endif
            StoreRows(dgOutGm_[TensorOffset<VARLEN_TND>(
                          tiling_, task.batchIdx, head, tokenBegin, col)],
                      output, ownedRows, cols, TensorRowElements());
        }

        LoadScalarRows(
            beta,
            dbGm_[ScalarOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseAdd2(
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)beta.GetPhyAddr(),
            ownedRows);
#else
        AscendC::Add(dbAcc, dbAcc, beta, ownedRows);
#endif
        StoreScalarRows(
            dbOutGm_[ScalarOffset<VARLEN_TND>(
                tiling_, task.batchIdx, head, tokenBegin)],
            dbAcc, ownedRows);
        }
    }

    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR gk_;
    GM_ADDR beta_;
    GM_ADDR dAqk_;
    GM_ADDR dAkk_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR db_;
    GM_ADDR dg_;
    GM_ADDR dqOut_;
    GM_ADDR dkOut_;
    GM_ADDR dbOut_;
    GM_ADDR dgOut_;
    GM_ADDR chunkMetadata_;
    GM_ADDR workspace_;

    ChunkKdaBwdIntraTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outputQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> matrixInputPing_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> matrixInputPong_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> arena_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceTmp_;
    uint32_t currentMatrixInputSlot_ = 0;
    event_t matrixMte2ToVEvent_[kMatrixInputBufferCount]{};
    event_t matrixVToMte2Event_[kMatrixInputBufferCount]{};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    Catlass::Arch::CrossCoreFlag vecToCubeReadyFlag_{kVecToCubeReadyFlag};
    Catlass::Arch::CrossCoreFlag cubeToVecReadyFlag_{kCubeToVecReadyFlag};
#endif
    AscendC::GlobalTensor<bfloat16_t> qGm_;
    AscendC::GlobalTensor<bfloat16_t> kGm_;
    AscendC::GlobalTensor<float> gkGm_;
    AscendC::GlobalTensor<BetaT> betaGm_;
    AscendC::GlobalTensor<float> dAqkGm_;
    AscendC::GlobalTensor<float> dAkkGm_;
    AscendC::GlobalTensor<float> dqGm_;
    AscendC::GlobalTensor<float> dkGm_;
    AscendC::GlobalTensor<float> dbGm_;
    AscendC::GlobalTensor<float> dgGm_;
    AscendC::GlobalTensor<float> dqOutGm_;
    AscendC::GlobalTensor<float> dkOutGm_;
    AscendC::GlobalTensor<float> dbOutGm_;
    AscendC::GlobalTensor<float> dgOutGm_;
    AscendC::GlobalTensor<int64_t> chunkMetadataGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_INTRA_VECTOR_H
