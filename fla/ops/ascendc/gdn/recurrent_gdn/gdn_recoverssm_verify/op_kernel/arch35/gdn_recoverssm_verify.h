/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_VERIFY_ARCH35_H
#define GDN_RECOVERSSM_VERIFY_ARCH35_H

#include <type_traits>

#include "kernel_operator.h"
#include "../gdn_recoverssm_verify_struct.h"

namespace GDN {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint8_t GRSV_INPUT_MUTEX[GRSV_INPUT_BUFFERS_2] = {3, 4};
constexpr uint8_t GRSV_OUTPUT_MUTEX[GRSV_INPUT_BUFFERS_2] = {5, 6};
constexpr uint8_t GRSV_STATE_MUTEX[GRSV_STATE_BANKS_3] = {7, 8, 9};

constexpr uint32_t GRSV_DATA_BLOCK_BYTES_32 = 32U;
constexpr uint32_t GRSV_BF16_PER_BLOCK_16 = GRSV_DATA_BLOCK_BYTES_32 / sizeof(bfloat16_t);
constexpr uint32_t GRSV_FP32_PER_BLOCK_8 = GRSV_DATA_BLOCK_BYTES_32 / sizeof(float);
constexpr uint32_t GRSV_FP32_PER_REG_64 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t GRSV_REQUIRED_DIM = 2U * GRSV_FP32_PER_REG_64;

constexpr CastTrait GRSV_FP32_TO_B16_ZERO = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING,
                                             RoundMode::CAST_RINT};
constexpr CastTrait GRSV_FP32_TO_B16_ONE = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                            RoundMode::CAST_RINT};
constexpr CastTrait GRSV_B16_TO_FP32_ZERO = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                             RoundMode::CAST_NONE};
constexpr CastTrait GRSV_B16_TO_FP32_ONE = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

template <typename inType, typename outType, typename stateType, bool NORMALIZE_QK>
__simd_vf__ inline void VerifyTokenVf(__ubuf__ float *stateDst, __ubuf__ float *stateSrc, __ubuf__ stateType *stateRaw,
                                      bool loadInitialState, __ubuf__ inType *keyRaw, __ubuf__ inType *queryRaw,
                                      __ubuf__ inType *valueRaw, __ubuf__ inType *aRaw, __ubuf__ inType *bRaw,
                                      __ubuf__ outType *output, __ubuf__ float *correctionRecord,
                                      __ubuf__ float *keyRecord, __ubuf__ float *decayRecord, float scale, float aLog,
                                      float dtBias, float softplusBeta, float softplusThreshold)
{
    MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskFirst = CreateMask<float, MaskPattern::VL1>();

    RegTensor<inType> keyPacked;
    RegTensor<inType> queryPacked;
    RegTensor<inType> valuePacked;
    RegTensor<inType> valuePackedNext;
    RegTensor<float> key0;
    RegTensor<float> key1;
    RegTensor<float> query0;
    RegTensor<float> query1;
    RegTensor<float> state0;
    RegTensor<float> state1;
    RegTensor<float> state2;
    RegTensor<float> state3;
    RegTensor<stateType> statePacked;
    RegTensor<stateType> statePackedNext;
    RegTensor<float> value;
    RegTensor<float> valueNext;
    RegTensor<float> rawA;
    RegTensor<float> rawB;
    RegTensor<float> scalarGate;
    RegTensor<float> scalarBeta;
    RegTensor<float> one;
    RegTensor<float> work0;
    RegTensor<float> work1;
    RegTensor<float> work2;
    RegTensor<float> dot;
    RegTensor<float> dotNext;
    RegTensor<float> correction;
    RegTensor<float> correctionNext;
    RegTensor<float> updateKey0;
    RegTensor<float> updateKey1;
    RegTensor<float> out0;
    RegTensor<float> out1;
    RegTensor<float> outSum;
    RegTensor<float> outSumNext;
    RegTensor<float> norm;
    RegTensor<float> invNorm;
    MaskReg aboveThreshold;
    UnalignRegForStore outputStore;
    UnalignRegForStore correctionStore;

    LoadAlign<inType, LoadDist::DIST_NORM>(keyPacked, keyRaw);
    LoadAlign<inType, LoadDist::DIST_NORM>(queryPacked, queryRaw);
    Cast<float, inType, GRSV_B16_TO_FP32_ZERO>(key0, keyPacked, maskB16);
    Cast<float, inType, GRSV_B16_TO_FP32_ZERO>(query0, queryPacked, maskB16);
    Cast<float, inType, GRSV_B16_TO_FP32_ONE>(key1, keyPacked, maskB16);
    Cast<float, inType, GRSV_B16_TO_FP32_ONE>(query1, queryPacked, maskB16);

    if constexpr (NORMALIZE_QK) {
        Mul(work0, query0, query0, maskFp32);
        Mul(work1, query1, query1, maskFp32);
        Add(work0, work0, work1, maskFp32);
        ReduceSum(norm, work0, maskFp32);
        Adds(norm, norm, 1.0e-6f, maskFirst);
        Sqrt(norm, norm, maskFirst);
        Duplicate(one, 1.0f, maskFp32);
        Div(invNorm, one, norm, maskFirst);
        Duplicate(work2, invNorm, maskFp32);
        Mul(query0, query0, work2, maskFp32);
        Mul(query1, query1, work2, maskFp32);

        Mul(work0, key0, key0, maskFp32);
        Mul(work1, key1, key1, maskFp32);
        Add(work0, work0, work1, maskFp32);
        ReduceSum(norm, work0, maskFp32);
        Adds(norm, norm, 1.0e-6f, maskFirst);
        Sqrt(norm, norm, maskFirst);
        Div(invNorm, one, norm, maskFirst);
        Duplicate(work2, invNorm, maskFp32);
        Mul(key0, key0, work2, maskFp32);
        Mul(key1, key1, work2, maskFp32);
    }
    Muls(query0, query0, scale, maskFp32);
    Muls(query1, query1, scale, maskFp32);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(keyRecord, key0, key1, maskFp32);

    RegTensor<inType> aPacked;
    RegTensor<inType> bPacked;
    LoadAlign<inType, LoadDist::DIST_BRC_B16>(aPacked, aRaw);
    LoadAlign<inType, LoadDist::DIST_BRC_B16>(bPacked, bRaw);
    Cast<float, inType, GRSV_B16_TO_FP32_ZERO>(rawA, aPacked, maskB16);
    Cast<float, inType, GRSV_B16_TO_FP32_ZERO>(rawB, bPacked, maskB16);
    Duplicate(one, 1.0f, maskFp32);

    Adds(scalarGate, rawA, dtBias, maskFp32);
    Muls(work0, scalarGate, softplusBeta, maskFp32);
    CompareScalar<float, CMPMODE::GT>(aboveThreshold, work0, softplusThreshold, maskFp32);
    Mins(work1, work0, softplusThreshold, maskFp32);
    Exp(work1, work1, maskFp32);
    Adds(work1, work1, 1.0f, maskFp32);
    Ln(work1, work1, maskFp32);
    Muls(work1, work1, 1.0f / softplusBeta, maskFp32);
    Select(work1, scalarGate, work1, aboveThreshold);
    Duplicate(work2, aLog, maskFp32);
    Exp(work2, work2, maskFp32);
    Mul(work0, work1, work2, maskFp32);
    Muls(work0, work0, -1.0f, maskFp32);
    Exp(scalarGate, work0, maskFp32);
    StoreAlign<float, StoreDist::DIST_NORM>(decayRecord, scalarGate, maskFirst);

    Muls(scalarBeta, rawB, -1.0f, maskFp32);
    Exp(scalarBeta, scalarBeta, maskFp32);
    Adds(scalarBeta, scalarBeta, 1.0f, maskFp32);
    Div(scalarBeta, one, scalarBeta, maskFp32);
    Mul(updateKey0, key0, scalarBeta, maskFp32);
    Mul(updateKey1, key1, scalarBeta, maskFp32);

    for (uint32_t row = 0; row < GRSV_REQUIRED_DIM; row += 2U) {
        const uint32_t offset = row * GRSV_REQUIRED_DIM;
        if (loadInitialState) {
            if constexpr (std::is_same<stateType, float>::value) {
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state0, state1, (__ubuf__ float *)stateRaw + offset);
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state2, state3,
                                                            (__ubuf__ float *)stateRaw + offset + GRSV_REQUIRED_DIM);
            } else {
                LoadAlign<stateType, LoadDist::DIST_NORM>(statePacked, stateRaw + offset);
                LoadAlign<stateType, LoadDist::DIST_NORM>(statePackedNext, stateRaw + offset + GRSV_REQUIRED_DIM);
                Cast<float, stateType, GRSV_B16_TO_FP32_ZERO>(state0, statePacked, maskB16);
                Cast<float, stateType, GRSV_B16_TO_FP32_ONE>(state1, statePacked, maskB16);
                Cast<float, stateType, GRSV_B16_TO_FP32_ZERO>(state2, statePackedNext, maskB16);
                Cast<float, stateType, GRSV_B16_TO_FP32_ONE>(state3, statePackedNext, maskB16);
            }
        } else {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state0, state1, stateSrc + offset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state2, state3, stateSrc + offset + GRSV_REQUIRED_DIM);
        }
        LoadAlign<inType, LoadDist::DIST_BRC_B16>(valuePacked, valueRaw + row);
        LoadAlign<inType, LoadDist::DIST_BRC_B16>(valuePackedNext, valueRaw + row + 1U);
        Cast<float, inType, GRSV_B16_TO_FP32_ZERO>(value, valuePacked, maskB16);
        Cast<float, inType, GRSV_B16_TO_FP32_ZERO>(valueNext, valuePackedNext, maskB16);

        Mul(state0, state0, scalarGate, maskFp32);
        Mul(state2, state2, scalarGate, maskFp32);
        Mul(state1, state1, scalarGate, maskFp32);
        Mul(state3, state3, scalarGate, maskFp32);

        Mul(dot, state0, key0, maskFp32);
        Mul(dotNext, state2, key0, maskFp32);
        MulAddDst(dot, state1, key1, maskFp32);
        MulAddDst(dotNext, state3, key1, maskFp32);
        ReduceSum(dot, dot, maskFp32);
        ReduceSum(dotNext, dotNext, maskFp32);
        Duplicate(dot, dot, maskFp32);
        Duplicate(dotNext, dotNext, maskFp32);
        Sub(correction, value, dot, maskFp32);
        Sub(correctionNext, valueNext, dotNext, maskFp32);
        Mul(correction, correction, scalarBeta, maskFp32);
        Mul(correctionNext, correctionNext, scalarBeta, maskFp32);
        MulAddDst(state0, correction, key0, maskFp32);
        MulAddDst(state2, correctionNext, key0, maskFp32);
        MulAddDst(state1, correction, key1, maskFp32);
        MulAddDst(state3, correctionNext, key1, maskFp32);

        StoreAlign<float, StoreDist::DIST_INTLV_B32>(stateDst + offset, state0, state1, maskFp32);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(stateDst + offset + GRSV_REQUIRED_DIM, state2, state3, maskFp32);

        Mul(out0, state0, query0, maskFp32);
        Mul(out1, state2, query0, maskFp32);
        MulAddDst(out0, state1, query1, maskFp32);
        MulAddDst(out1, state3, query1, maskFp32);
        ReduceSum(outSum, out0, maskFp32);
        ReduceSum(outSumNext, out1, maskFp32);
        Cast<outType, float, GRSV_FP32_TO_B16_ZERO>(valuePacked, outSum, maskFirst);
        Cast<outType, float, GRSV_FP32_TO_B16_ZERO>(valuePackedNext, outSumNext, maskFirst);
        StoreUnAlign(output, valuePacked, outputStore, 1);
        StoreUnAlign(output, valuePackedNext, outputStore, 1);

        StoreUnAlign(correctionRecord, correction, correctionStore, 1);
        StoreUnAlign(correctionRecord, correctionNext, correctionStore, 1);
    }
    StoreUnAlignPost(output, outputStore, 0);
    StoreUnAlignPost(correctionRecord, correctionStore, 0);
}

template <typename inType, typename stateType, bool USE_QK_L2NORM>
class GdnRecoverssmVerifyVector {
public:
    __aicore__ inline explicit GdnRecoverssmVerifyVector(const GdnRecoverssmVerifyTilingData *tiling) : tiling_(tiling)
    {
    }

    __aicore__ inline void Init(GM_ADDR aLog, GM_ADDR a, GM_ADDR b, GM_ADDR dtBias, GM_ADDR query, GM_ADDR key,
                                GM_ADDR value, GM_ADDR checkpoint, GM_ADDR replay, GM_ADDR queryStartLoc,
                                GM_ADDR stateIndices, GM_ADDR out)
    {
        aLogGm_.SetGlobalBuffer((__gm__ float *)aLog);
        aGm_.SetGlobalBuffer((__gm__ inType *)a);
        bGm_.SetGlobalBuffer((__gm__ inType *)b);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)dtBias);
        queryGm_.SetGlobalBuffer((__gm__ inType *)query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)value);
        checkpointGm_.SetGlobalBuffer((__gm__ stateType *)checkpoint);
        replayGm_.SetGlobalBuffer((__gm__ float *)replay);
        queryStartGm_.SetGlobalBuffer((__gm__ int32_t *)queryStartLoc);
        queryStart64Gm_.SetGlobalBuffer((__gm__ int64_t *)queryStartLoc);
        stateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)stateIndices);
        stateIndices64Gm_.SetGlobalBuffer((__gm__ int64_t *)stateIndices);
        outGm_.SetGlobalBuffer((__gm__ bfloat16_t *)out);
        InitBuffers();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        uint32_t currentBank = 0;
        uint32_t tileIndex = 0;
        for (uint32_t task = core; task < tiling_->taskCount; task += coreNum) {
            const uint32_t row = task / tiling_->valueHeads;
            const uint32_t hv = task - row * tiling_->valueHeads;
            int64_t page = 0;
            if (tiling_->metadataDtype == 64U) {
                page = stateIndices64Gm_.GetValue(row * tiling_->stateIndexStride);
            } else {
                page = stateIndicesGm_.GetValue(row * tiling_->stateIndexStride);
            }
            int64_t bos = 0;
            int64_t eos = 0;
            if (tiling_->metadataDtype == 64U) {
                bos = queryStart64Gm_.GetValue(row * tiling_->queryStartStride);
                eos = queryStart64Gm_.GetValue((row + 1U) * tiling_->queryStartStride);
            } else {
                bos = queryStartGm_.GetValue(row * tiling_->queryStartStride);
                eos = queryStartGm_.GetValue((row + 1U) * tiling_->queryStartStride);
            }

            const int64_t taskTokens = eos - bos;
            if (taskTokens == 0) {
                continue;
            }
            const bool hasNextTask = task + coreNum < tiling_->taskCount;
            currentBank = (currentBank + 1U) % GRSV_STATE_BANKS_3;
            const uint32_t initialBank = currentBank;
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            LoadInitialState(static_cast<uint64_t>(page), hv, currentBank);
            if (taskTokens == 1 && hasNextTask) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            uint32_t srcBank = currentBank;
            const uint32_t qkHead = hv / (tiling_->valueHeads / tiling_->keyHeads);
            const float aLog = aLogGm_.GetValue(hv);
            const float dtBias = dtBiasGm_.GetValue(hv);
            bool firstToken = true;
            for (int64_t start = bos; start < eos; start += GRSV_TOKEN_TILE_4, ++tileIndex) {
                const int64_t remaining = eos - start;
                const uint32_t rows = static_cast<uint32_t>(
                    remaining < static_cast<int64_t>(GRSV_TOKEN_TILE_4) ? remaining : GRSV_TOKEN_TILE_4);
                const uint32_t tile = tileIndex & 1U;
                CopyInputTile(tile, static_cast<uint64_t>(start), rows, qkHead, hv);
                Mutex::Lock<PIPE_V>(GRSV_INPUT_MUTEX[tile]);
                Mutex::Lock<PIPE_V>(GRSV_OUTPUT_MUTEX[tile]);
                for (uint32_t token = 0; token < rows; ++token) {
                    const uint32_t dstBank = (srcBank + 1U) % GRSV_STATE_BANKS_3;
                    Mutex::Lock<PIPE_V>(GRSV_STATE_MUTEX[dstBank]);
                    VerifyTokenVf<inType, bfloat16_t, stateType, USE_QK_L2NORM>(
                        (__ubuf__ float *)stateBanks_[dstBank].GetPhyAddr(),
                        (__ubuf__ float *)stateBanks_[srcBank].GetPhyAddr(),
                        (__ubuf__ stateType *)stateBanks_[initialBank].GetPhyAddr(), firstToken,
                        (__ubuf__ inType *)kTile_[tile][token * GRSV_REQUIRED_DIM].GetPhyAddr(),
                        (__ubuf__ inType *)qTile_[tile][token * GRSV_REQUIRED_DIM].GetPhyAddr(),
                        (__ubuf__ inType *)vTile_[tile][token * GRSV_REQUIRED_DIM].GetPhyAddr(),
                        (__ubuf__ inType *)aTile_[tile][token * GRSV_BF16_PER_BLOCK_16].GetPhyAddr(),
                        (__ubuf__ inType *)bTile_[tile][token * GRSV_BF16_PER_BLOCK_16].GetPhyAddr(),
                        (__ubuf__ bfloat16_t *)outTile_[tile][token * GRSV_REQUIRED_DIM].GetPhyAddr(),
                        (__ubuf__ float *)correctionTile_[tile][token * GRSV_REQUIRED_DIM].GetPhyAddr(),
                        (__ubuf__ float *)keyTile_[tile][token * GRSV_REQUIRED_DIM].GetPhyAddr(),
                        (__ubuf__ float *)decayTile_[tile][token * GRSV_FP32_PER_BLOCK_8].GetPhyAddr(), tiling_->scale,
                        aLog, dtBias, tiling_->softplusBeta, tiling_->softplusThreshold);
                    PipeBarrier<PIPE_V>();
                    Mutex::Unlock<PIPE_V>(GRSV_STATE_MUTEX[dstBank]);
                    srcBank = dstBank;
                    currentBank = srcBank;
                    firstToken = false;
                    if (taskTokens > 1 && hasNextTask && start + token - bos == taskTokens - 2) {
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    }
                }
                Mutex::Unlock<PIPE_V>(GRSV_OUTPUT_MUTEX[tile]);
                Mutex::Unlock<PIPE_V>(GRSV_INPUT_MUTEX[tile]);
                WriteOutputTile(tile, static_cast<uint64_t>(start), static_cast<uint64_t>(page), hv, row, rows);
            }
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        uint32_t offset = 0;
        constexpr uint32_t stateElements = GRSV_REQUIRED_DIM * GRSV_REQUIRED_DIM;
        for (uint32_t bank = 0; bank < GRSV_STATE_BANKS_3; ++bank) {
            stateBanks_[bank] = LocalTensor<float>(TPosition::VECCALC, offset, stateElements);
            offset += stateElements * sizeof(float);
        }
        constexpr uint32_t vectorElements = GRSV_TOKEN_TILE_4 * GRSV_REQUIRED_DIM;
        for (uint32_t tile = 0; tile < GRSV_INPUT_BUFFERS_2; ++tile) {
            qTile_[tile] = LocalTensor<inType>(TPosition::VECCALC, offset, vectorElements);
            offset += vectorElements * sizeof(inType);
            kTile_[tile] = LocalTensor<inType>(TPosition::VECCALC, offset, vectorElements);
            offset += vectorElements * sizeof(inType);
            vTile_[tile] = LocalTensor<inType>(TPosition::VECCALC, offset, vectorElements);
            offset += vectorElements * sizeof(inType);
            aTile_[tile] = LocalTensor<inType>(TPosition::VECCALC, offset, GRSV_TOKEN_TILE_4 * GRSV_BF16_PER_BLOCK_16);
            offset += GRSV_TOKEN_TILE_4 * GRSV_DATA_BLOCK_BYTES_32;
            bTile_[tile] = LocalTensor<inType>(TPosition::VECCALC, offset, GRSV_TOKEN_TILE_4 * GRSV_BF16_PER_BLOCK_16);
            offset += GRSV_TOKEN_TILE_4 * GRSV_DATA_BLOCK_BYTES_32;
            outTile_[tile] = LocalTensor<bfloat16_t>(TPosition::VECCALC, offset, vectorElements);
            offset += vectorElements * sizeof(bfloat16_t);
            correctionTile_[tile] = LocalTensor<float>(TPosition::VECCALC, offset, vectorElements);
            offset += vectorElements * sizeof(float);
            keyTile_[tile] = LocalTensor<float>(TPosition::VECCALC, offset, vectorElements);
            offset += vectorElements * sizeof(float);
            decayTile_[tile] =
                LocalTensor<float>(TPosition::VECCALC, offset, GRSV_TOKEN_TILE_4 * GRSV_FP32_PER_BLOCK_8);
            offset += GRSV_TOKEN_TILE_4 * GRSV_FP32_PER_BLOCK_8 * sizeof(float);
        }
    }

    __aicore__ inline void LoadInitialState(uint64_t page, uint32_t hv, uint32_t bank)
    {
        const uint64_t stateOffset = page * tiling_->checkpointStrides[0] + hv * tiling_->checkpointStrides[1];
        LocalTensor<stateType> rawState = stateBanks_[bank].template ReinterpretCast<stateType>();
        Mutex::Lock<PIPE_MTE2>(GRSV_STATE_MUTEX[bank]);
        DataCopyExtParams params{
            static_cast<uint16_t>(tiling_->valueDim), static_cast<uint32_t>(tiling_->keyDim * sizeof(stateType)),
            static_cast<uint32_t>((tiling_->checkpointStrides[2] - tiling_->keyDim) * sizeof(stateType)),
            static_cast<uint32_t>((GRSV_REQUIRED_DIM - tiling_->keyDim) * sizeof(stateType)), 0};
        DataCopyPad(rawState, checkpointGm_[stateOffset], params,
                    DataCopyPadExtParams<stateType>{false, 0, 0, static_cast<stateType>(0)});
        Mutex::Unlock<PIPE_MTE2>(GRSV_STATE_MUTEX[bank]);
    }

    __aicore__ inline void CopyInputTile(uint32_t tile, uint64_t start, uint32_t rows, uint32_t qkHead, uint32_t hv)
    {
        const uint64_t qOffset = start * tiling_->queryStrides[1] + qkHead * tiling_->queryStrides[2];
        const uint64_t kOffset = start * tiling_->keyStrides[1] + qkHead * tiling_->keyStrides[2];
        const uint64_t vOffset = start * tiling_->valueStrides[1] + hv * tiling_->valueStrides[2];
        const uint64_t aOffset = start * tiling_->aStrides[0] + hv * tiling_->aStrides[1];
        const uint64_t bOffset = start * tiling_->bStrides[0] + hv * tiling_->bStrides[1];
        DataCopyExtParams qParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(tiling_->keyDim * sizeof(inType)),
                                  static_cast<uint32_t>((tiling_->queryStrides[1] - tiling_->keyDim) * sizeof(inType)),
                                  static_cast<uint32_t>((GRSV_REQUIRED_DIM - tiling_->keyDim) * sizeof(inType)), 0};
        DataCopyExtParams kParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(tiling_->keyDim * sizeof(inType)),
                                  static_cast<uint32_t>((tiling_->keyStrides[1] - tiling_->keyDim) * sizeof(inType)),
                                  static_cast<uint32_t>((GRSV_REQUIRED_DIM - tiling_->keyDim) * sizeof(inType)), 0};
        DataCopyExtParams vParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(tiling_->valueDim * sizeof(inType)),
            static_cast<uint32_t>((tiling_->valueStrides[1] - tiling_->valueDim) * sizeof(inType)),
            static_cast<uint32_t>((GRSV_REQUIRED_DIM - tiling_->valueDim) * sizeof(inType)), 0};
        const DataCopyPadExtParams<inType> pad{false, 0, 0, static_cast<inType>(0)};
        const DataCopyPadExtParams<inType> scalarPad{false, 0, GRSV_BF16_PER_BLOCK_16 - 1U, static_cast<inType>(0)};
        const DataCopyExtParams aParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(inType)),
                                        static_cast<uint32_t>((tiling_->aStrides[0] - 1U) * sizeof(inType)), 0, 0};
        const DataCopyExtParams bParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(inType)),
                                        static_cast<uint32_t>((tiling_->bStrides[0] - 1U) * sizeof(inType)), 0, 0};
        Mutex::Lock<PIPE_MTE2>(GRSV_INPUT_MUTEX[tile]);
        DataCopyPad(qTile_[tile], queryGm_[qOffset], qParams, pad);
        DataCopyPad(kTile_[tile], keyGm_[kOffset], kParams, pad);
        DataCopyPad(vTile_[tile], valueGm_[vOffset], vParams, pad);
        DataCopyPad(aTile_[tile], aGm_[aOffset], aParams, scalarPad);
        DataCopyPad(bTile_[tile], bGm_[bOffset], bParams, scalarPad);
        Mutex::Unlock<PIPE_MTE2>(GRSV_INPUT_MUTEX[tile]);
    }

    __aicore__ inline void WriteOutputTile(uint32_t tile, uint64_t start, uint64_t page, uint32_t hv,
                                           uint64_t requestIndex, uint32_t rows)
    {
        Mutex::Lock<PIPE_MTE3>(GRSV_OUTPUT_MUTEX[tile]);
        const uint64_t outOffset = start * tiling_->outStrides[1] + hv * tiling_->outStrides[2];
        DataCopyExtParams outParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(tiling_->valueDim * sizeof(bfloat16_t)),
            static_cast<uint32_t>((GRSV_REQUIRED_DIM - tiling_->valueDim) * sizeof(bfloat16_t)),
            static_cast<uint32_t>((tiling_->outStrides[1] - tiling_->valueDim) * sizeof(bfloat16_t)), 0};
        const uint64_t replayBase = page * tiling_->replayStrides[0] + hv * tiling_->replayStrides[1];
        int64_t bos = 0;
        if (tiling_->metadataDtype == 64U) {
            bos = queryStart64Gm_.GetValue(requestIndex * tiling_->queryStartStride);
        } else {
            bos = queryStartGm_.GetValue(requestIndex * tiling_->queryStartStride);
        }
        const uint64_t replayOffset = replayBase + (start - static_cast<uint64_t>(bos)) * tiling_->replayStrides[2];
        DataCopyPad(outGm_[outOffset], outTile_[tile], outParams);
        DataCopyExtParams correctionParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(tiling_->valueDim * sizeof(float)),
            static_cast<uint32_t>((GRSV_REQUIRED_DIM - tiling_->valueDim) * sizeof(float)),
            static_cast<uint32_t>((tiling_->replayStrides[2] - tiling_->valueDim) * sizeof(float)), 0};
        DataCopyPad(replayGm_[replayOffset], correctionTile_[tile], correctionParams);

        DataCopyExtParams keyParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(tiling_->keyDim * sizeof(float)),
            static_cast<uint32_t>((GRSV_REQUIRED_DIM - tiling_->keyDim) * sizeof(float)),
            static_cast<uint32_t>((tiling_->replayStrides[2] - tiling_->keyDim) * sizeof(float)), 0};
        DataCopyPad(replayGm_[replayOffset + tiling_->valueDim], keyTile_[tile], keyParams);

        const DataCopyExtParams decayParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(float)), 0,
                                            static_cast<uint32_t>((tiling_->replayStrides[2] - 1U) * sizeof(float)), 0};
        DataCopyPad(replayGm_[replayOffset + tiling_->valueDim + tiling_->keyDim], decayTile_[tile], decayParams);
        Mutex::Unlock<PIPE_MTE3>(GRSV_OUTPUT_MUTEX[tile]);
    }

private:
    const GdnRecoverssmVerifyTilingData *tiling_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<inType> aGm_;
    GlobalTensor<inType> bGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<stateType> checkpointGm_;
    GlobalTensor<float> replayGm_;
    GlobalTensor<int32_t> queryStartGm_;
    GlobalTensor<int64_t> queryStart64Gm_;
    GlobalTensor<int32_t> stateIndicesGm_;
    GlobalTensor<int64_t> stateIndices64Gm_;
    GlobalTensor<bfloat16_t> outGm_;

    LocalTensor<float> stateBanks_[GRSV_STATE_BANKS_3];
    LocalTensor<inType> qTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<inType> kTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<inType> vTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<inType> aTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<inType> bTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<bfloat16_t> outTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<float> correctionTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<float> keyTile_[GRSV_INPUT_BUFFERS_2];
    LocalTensor<float> decayTile_[GRSV_INPUT_BUFFERS_2];
};

} // namespace GDN

#endif // GDN_RECOVERSSM_VERIFY_ARCH35_H
