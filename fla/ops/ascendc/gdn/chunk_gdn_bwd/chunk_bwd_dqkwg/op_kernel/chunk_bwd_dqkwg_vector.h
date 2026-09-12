/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_vector.h
 *
 * CV 深融合 (chunk-interleaved A/B/C/D) 的 AIV (Vector) 端。
 *
 * 与 cube 端镜像: 每核任务流 A(c0..cM-1), B(...), C(...), D(...)。
 * 每个 (stage, chunk) task: 两个 AIV sub-block 各 WaitCubeReady 一次,
 * 处理完该 chunk 全部 head (sub-block 内部按 head / row 切分) 后各 SetVectorDone 一次。
 * cube 落后处理: vector 处理 task[i] 时 cube 已在做 task[i+1]。
 *
 * stage 映射 (与 cube 对应):
 *   A_vector = 原 Part1 vector (dw 取负 + dg_last) + 原 Part2 vector (mul1)
 *   B_vector = 原 Part3 vector (ds_temp + dg 部分)
 *   C_vector = 原 Part4 + Part6 vector (dq 最终 + dg)
 *   D_vector = 原 Part5 + Part7 vector (dk 最终 + dg 最终, 含 dg_last 重算)
 *
 * 同步计数平衡 (区别于 cv_merge 死锁版本): 无 preseed, 无 per-head flag, 无 stage drain;
 * 每个 AIV sub-block 每 task 恰好 1 次 WaitCubeReady + 1 次 SetVectorDone。
 */

#ifndef CHUNK_BWD_DQKWG_VECTOR_H
#define CHUNK_BWD_DQKWG_VECTOR_H

#include "chunk_bwd_dqkwg_common.h"
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_bwd_dqkwg_regbase.h"
#endif

using namespace AscendC;

template <typename DataType, typename GType>
class ChunkBwdDqkwgVectorProcess {
public:
    __aicore__ inline ChunkBwdDqkwgVectorProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_,
                                                 GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices,
                                                 GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace);

    __aicore__ inline void Init(const ChunkBwdDqkwgTilingData &tiling, TPipe *pipe_);
    __aicore__ inline void Process();

private:
    // 4 个 stage 的处理函数 (CV 深融合)
    // bos/real_BT 由 Process() 统一算一次后传入, 避免每个 stage 重复 GetChunkOffset
    __aicore__ inline void ProcessAVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT); // 原 Part1 (dw 取负 + dg_last)
    __aicore__ inline void ProcessBVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT); // 原 Part2 (mul1) + 原 Part3 (ds_temp + dg 部分)
    __aicore__ inline void ProcessCVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT); // 原 Part4 + Part6 (dq 最终)
    __aicore__ inline void ProcessCVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT); // 原 Part4 + Part6 (dq 最终) (GVA)
    __aicore__ inline void ProcessDVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT); // 原 Part5 + Part7 (dk 最终 + dg 最终)
    __aicore__ inline void ProcessDVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT); // 原 Part5 + Part7 (dk 最终 + dg 最终) (GVA)

    // mul1 一个 row-half 的计算 (= A 的 Part2 per-head 内核, 输出 fp32 到 outFp32[half] )。
    // A (小 case) 调它后 cast+写 GM; B (大 case) 调它两次 (两个 row-half) 后直接乘 ds, 省掉 mul1 GM 往返。
    __aicore__ inline void ComputeMul1HalfFp32(const LocalTensor<float> &outFp32, const LocalTensor<float> &tensorMaskA,
                                               const LocalTensor<float> &tensorGTmpFp32, const LocalTensor<float> &tensorGFp32Left, 
                                               const LocalTensor<float> &tensorGFp32Right, uint32_t BT_sub_start, uint32_t real_BT);

    __aicore__ inline void CopyGateWithPad(const LocalTensor<GType> &dst, const GlobalTensor<GType> &src, uint64_t offset,
                                           uint32_t validLen, uint32_t totalLen);
    __aicore__ inline void ComputeDqState(const LocalTensor<float> &tensorDqInFp32, const LocalTensor<float> &tensorGFp32, const LocalTensor<float> &tensorGTmpFp32, uint32_t real_BT);

private:
    // 输入输出指针
    GM_ADDR ptrQ;
    GM_ADDR ptrK;
    GM_ADDR ptrV;
    GM_ADDR ptrG;
    GM_ADDR ptrH;
    GM_ADDR ptrDo;
    GM_ADDR ptrDh;
    GM_ADDR ptrDv;
    GM_ADDR ptrCuSeqLen;
    GM_ADDR ptrChunkIndices;
    GM_ADDR ptrDq;
    GM_ADDR ptrDk;
    GM_ADDR ptrDw;
    GM_ADDR ptrDg;
    GM_ADDR ptrWorkspace;

    // Tiling 参数 (GVA: H 拆为 HV/HK, HV = n_ratio * HK)
    uint64_t B;
    uint64_t HV; // value 侧 head 数 (== 原 H)
    uint64_t HK; // key/query 侧 head 数, HV = n_ratio * HK
    uint64_t T;
    uint64_t K;
    uint64_t V;
    uint64_t BT;
    uint64_t half_BT;
    uint64_t numChunks;
    float scale;
    bool isVarLen;
    bool gvaMode;
    uint32_t mul0RowNum = 0;
    uint64_t n_ratio = 1;    // GVA: HV / HK
    uint32_t coreNum = 0; // CV 深融合 blockDim (cube/vector 共用)
    uint32_t coreIdx = 0;
    uint32_t subBlockIdx = 0;

    // Workspace 偏移
    uint64_t wsMm3Offset;
    uint64_t wsMm4Offset;
    uint64_t wsMm6Offset;
    uint64_t wsMm5Offset;
    uint64_t wsMm7Offset;
    uint64_t wsDsTempOffset;
    uint64_t wsDgLastOffset;
    int BUFFER_NUM = 1;

    // Pipeline
    TPipe *pipe = nullptr;

    // Global Tensors
    GlobalTensor<DataType> gmQ, gmK, gmV, gmDo, gmH, gmDh, gmDv;
    GlobalTensor<DataType> gmDq, gmDk, gmDw;
    GlobalTensor<GType> gmG, gmDg;
    GlobalTensor<float> gmDgLast;
    GlobalTensor<DataType> gmMm3, gmMm4, gmMm6, gmMm5, gmMm7, gmDsTemp;
    GlobalTensor<uint64_t> cuSeqlensTensor, chunkIndicesTensor;

    // Queues (用于流水)
    TQue<TPosition::VECIN, 2> inQue1;
    TQue<TPosition::VECIN, 2> inQue2;
    TQue<TPosition::VECIN, 2> inQue3;
    TQue<TPosition::VECIN, 2> inQue4;
    TQue<TPosition::VECOUT, 2> outQue1;
    TQue<TPosition::VECOUT, 2> outQue2;


    // Calc Buffers (UB 空间)
    TBuf<TPosition::VECCALC> calcBuf1; // 主计算缓冲区 (fp32)
    TBuf<TPosition::VECCALC> calcBuf2; // 辅助计算缓冲区 (fp32)
    TBuf<TPosition::VECCALC> maskBuf;  // A_vector mask
    TBuf<TPosition::VECCALC> dgBuf;    // dg 值缓冲区

    // Local Tensors
    LocalTensor<float> tensorGTmpFp32;
    LocalTensor<float> tensorDgTmpFp32;
    LocalTensor<float> tensorDgFinal;
    LocalTensor<float> tensorMaskA;

    // UB 空间常量
    static constexpr uint32_t UB_BLOCK_SIZE = 32;
};

// ============== 构造函数 ==============
template <typename DataType, typename GType>
__aicore__ inline ChunkBwdDqkwgVectorProcess<DataType, GType>::ChunkBwdDqkwgVectorProcess(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen,
    GM_ADDR chunk_indices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace)
    : ptrQ(q), ptrK(k), ptrV(v), ptrG(g), ptrH(h), ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLen(cu_seqlen),
      ptrChunkIndices(chunk_indices), ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrDg(dg), ptrWorkspace(workspace)
{
}

// ============== 初始化 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Init(const ChunkBwdDqkwgTilingData &tiling,
                                                                         TPipe *pipe_)
{
    pipe = pipe_;

    scale = tiling.scale;
    B = tiling.B;
    HV = tiling.HV;
    HK = tiling.HK;
    n_ratio = (HK > 0) ? (HV / HK) : 1;
    T = tiling.T;
    K = tiling.K;
    V = tiling.V;
    BT = tiling.BT;
    half_BT = BT / 2;
    numChunks = tiling.numChunks;
    coreNum = tiling.aicCoreNum;
    coreIdx = GetBlockIdx() / GetSubBlockNum();
    subBlockIdx = GetSubBlockIdx();
    wsMm3Offset = tiling.wsMm3Offset;
    wsMm4Offset = tiling.wsMm4Offset;
    wsMm6Offset = tiling.wsMm6Offset;
    wsMm5Offset = tiling.wsMm5Offset;
    wsMm7Offset = tiling.wsMm7Offset;
    wsDsTempOffset = tiling.wsDsTempOffset;
    wsDgLastOffset = tiling.wsDgLastOffset;
    isVarLen = (tiling.isVarLen == 1);
    gvaMode = (HV != HK);
    mul0RowNum = tiling.mul0RowNum;

    if (BT == 64) {
        BUFFER_NUM = 2;
    } else {
        BUFFER_NUM = 1;
    }

    gmQ.SetGlobalBuffer((__gm__ DataType *)ptrQ);
    gmK.SetGlobalBuffer((__gm__ DataType *)ptrK);
    gmV.SetGlobalBuffer((__gm__ DataType *)ptrV);
    gmG.SetGlobalBuffer((__gm__ GType *)ptrG);
    gmH.SetGlobalBuffer((__gm__ DataType *)ptrH);
    gmDo.SetGlobalBuffer((__gm__ DataType *)ptrDo);
    gmDh.SetGlobalBuffer((__gm__ DataType *)ptrDh);
    gmDv.SetGlobalBuffer((__gm__ DataType *)ptrDv);

    gmDq.SetGlobalBuffer((__gm__ DataType *)ptrDq);
    gmDk.SetGlobalBuffer((__gm__ DataType *)ptrDk);
    gmDw.SetGlobalBuffer((__gm__ DataType *)ptrDw);
    gmDg.SetGlobalBuffer((__gm__ GType *)ptrDg);

    gmMm3.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm3Offset));
    gmMm4.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm4Offset));
    gmMm6.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm6Offset));
    gmMm5.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm5Offset));
    gmMm7.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm7Offset));
    gmDsTemp.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsDsTempOffset));
    gmDgLast.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)ptrWorkspace + wsDgLastOffset)); // 中间结果使用float

    if (isVarLen) {
        cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)ptrCuSeqLen);
        chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)ptrChunkIndices);
    }

    pipe->InitBuffer(inQue1, BUFFER_NUM, BT * K * sizeof(float));     // 64K
    pipe->InitBuffer(inQue2, BUFFER_NUM, BT * K * sizeof(float));     // 64K
    pipe->InitBuffer(inQue3, 2, BT * sizeof(float));                  // 1K
    pipe->InitBuffer(inQue4, 2, BT * sizeof(float));                  // 1K
    pipe->InitBuffer(outQue1, BUFFER_NUM, BT * K * sizeof(DataType)); // 32K
    pipe->InitBuffer(outQue2, BUFFER_NUM, BT * sizeof(float));        // 512B

    pipe->InitBuffer(calcBuf1, BT * 8 * sizeof(float));               // 4K
    pipe->InitBuffer(calcBuf2, BT * 8 * sizeof(float));               // 4K
    pipe->InitBuffer(maskBuf, 64 * 64 * sizeof(float));               // 16K
    pipe->InitBuffer(dgBuf, BT * 8 * sizeof(float));                  // 4K

    tensorGTmpFp32 = calcBuf1.Get<float>();
    tensorDgTmpFp32 = calcBuf2.Get<float>();
    tensorDgFinal = dgBuf.Get<float>();
    tensorMaskA = maskBuf.Get<float>();
    Duplicate(tensorMaskA, 0.0f, 64 * 64);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < 64; i++) {
        Duplicate(tensorMaskA[i * 64], scale, i + 1);
    }
    PipeBarrier<PIPE_V>();
}

// ============== 主处理函数 (CV 深融合: A -> B -> C -> D, 每 stage 内 chunk 级与 cube 流水) ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Process()
{
    // 预置三个信用，ds、mm3、mm4、mm5为空，cube会根据信用来生产
    SetVecClear();
    SetVecClear();
    SetVecClear();
    uint32_t coreLoops = B * numChunks;
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        // 统一算一次 bos/real_BT, 传给 4 个 stage 复用 (替代每 stage 内部 GetChunkOffset)
        uint32_t bos = 0;
        uint32_t eos = 0;
        GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, B, HV, T, BT, loopIdx, bos, eos, isVarLen);
        uint32_t real_BT = eos - bos;
        ProcessAVector(coreIdx, loopIdx, bos, real_BT);
        ProcessBVector(coreIdx, loopIdx, bos, real_BT);
        if (gvaMode) {
            ProcessCVectorForGva(coreIdx, loopIdx, bos, real_BT);
            ProcessDVectorForGva(coreIdx, loopIdx, bos, real_BT);
        } else {
            ProcessCVector(coreIdx, loopIdx, bos, real_BT);
            ProcessDVector(coreIdx, loopIdx, bos, real_BT);
        }
    }
}

template <typename DataType, typename GType>
__aicore__ inline void
ChunkBwdDqkwgVectorProcess<DataType, GType>::CopyGateWithPad(const LocalTensor<GType> &dst, const GlobalTensor<GType> &src,
                                                             uint64_t offset, uint32_t validLen, uint32_t totalLen)
{
    // 先注释掉，等待泛化验证
    // if (validLen == totalLen) {
    //     DataCopy(dst, src[offset], totalLen);
    //     return;
    // }
    // Duplicate(dst, static_cast<GType>(0), totalLen);
    // TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
    // SetFlag<HardEvent::V_MTE2>(eventId);
    // WaitFlag<HardEvent::V_MTE2>(eventId);
    // if (validLen == 0) {
    //     return;
    // }

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLen * sizeof(GType)), 0, 0, 0};
    // constexpr uint32_t elemsPerBlock = UB_BLOCK_SIZE / sizeof(GType);
    // uint8_t rightPadding = static_cast<uint8_t>((elemsPerBlock - (validLen % elemsPerBlock)) % elemsPerBlock);
    DataCopyPadExtParams<GType> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src[offset], copyParams, padParams);
}

// mul1 一个 row-half 的计算: 输出 fp32 因果掩码 exp 到 outFp32 (该 half 的 [real_BT, BT], 行优先)。
// 内核与 ProcessAVector Part2 完全一致 (同 buffer: inQue3 g / calcBuf1 brcb / calcBuf2,3 g±)。
// 调用方需先 InitBuffer 这些 buffer 并构建 tensorMaskA。outFp32 必须指向该 half 的行起点。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ComputeMul1HalfFp32(
    const LocalTensor<float> &outFp32, const LocalTensor<float> &tensorMaskA, 
    const LocalTensor<float> &tensorGTmpFp32, const LocalTensor<float> &tensorGFp32Left, 
    const LocalTensor<float> &tensorGFp32Right, uint32_t BT_sub_start, uint32_t real_BT)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (BT == 64) {
        Mul1Half<64, 0>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                        (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
    } else {
        if (BT_sub_start == 0) {
            Mul1Half<128, 0>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                             (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        } else {
            Mul1Half<128, 64>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                              (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        }
    }
#else
    Muls(tensorGFp32Right, tensorGFp32Left, static_cast<float>(-1), BT);
    Brcb(tensorGTmpFp32, tensorGFp32Left[BT_sub_start], CEIL_DIV(real_BT, 8), {1, 8});
    PipeBarrier<PIPE_V>();
    if (BT == 64) {
        AscendC::Add(outFp32, tensorGFp32Right, tensorGTmpFp32, CAL_NUM_FLOAT, real_BT, {1, 1, 0, 8, 0, 1});
        PipeBarrier<PIPE_V>();
        Mins(outFp32, outFp32, static_cast<float>(0.0), real_BT * BT);
        PipeBarrier<PIPE_V>();
        Exp(outFp32, outFp32, real_BT * BT);
        PipeBarrier<PIPE_V>();
        Mul(outFp32, outFp32, tensorMaskA, real_BT * BT);
        PipeBarrier<PIPE_V>();
    } else {
        AscendC::Copy(outFp32, tensorGFp32Right, CAL_NUM_FLOAT, real_BT, {1, 1, 16, 0});
        PipeBarrier<PIPE_V>();
        AscendC::Copy(outFp32[CAL_NUM_FLOAT], tensorGFp32Right[CAL_NUM_FLOAT], CAL_NUM_FLOAT, real_BT, {1, 1, 16, 0});
        PipeBarrier<PIPE_V>();
        AscendC::Add(outFp32, outFp32, tensorGTmpFp32, CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
        AscendC::Add(outFp32[CAL_NUM_FLOAT], outFp32[CAL_NUM_FLOAT], tensorGTmpFp32, CAL_NUM_FLOAT, real_BT,
                     {1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
        Mins(outFp32, outFp32, static_cast<float>(0.0), real_BT * BT);
        PipeBarrier<PIPE_V>();
        Exp(outFp32, outFp32, real_BT * BT);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams binaryRepeatParams{1, 1, 1, 16, 16, 8};
        UnaryRepeatParams unaryRepeatParams{1, 1, 16, 16};
        if (BT_sub_start == 0) {
            Mul(outFp32, outFp32, tensorMaskA, 64, real_BT, binaryRepeatParams);
            PipeBarrier<PIPE_V>();
            Muls(outFp32[64], outFp32[64], static_cast<float>(0), 64, real_BT, unaryRepeatParams);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(outFp32, outFp32, scale, 64, real_BT, unaryRepeatParams);
            PipeBarrier<PIPE_V>();
            Mul(outFp32[64], outFp32[64], tensorMaskA, 64, real_BT, binaryRepeatParams);
            PipeBarrier<PIPE_V>();
        }
    }
#endif
}

// ComputeDqState: dq_state = dq_inner * exp(g)[:,None] * scale
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ComputeDqState(
    const LocalTensor<float> &tensorDqInFp32, const LocalTensor<float> &tensorGFp32, const LocalTensor<float> &tensorGTmpFp32, uint32_t real_BT)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    DqState((__ubuf__ float *)tensorDqInFp32.GetPhyAddr(), (__ubuf__ float *)tensorDqInFp32.GetPhyAddr(), 
            (__ubuf__ float *)tensorGFp32.GetPhyAddr(), (__ubuf__ float *)tensorGTmpFp32.GetPhyAddr(), real_BT, scale);
#else
    Exp(tensorGFp32, tensorGFp32, real_BT);
    PipeBarrier<PIPE_V>();
    Muls(tensorGFp32, tensorGFp32, scale, real_BT);
    PipeBarrier<PIPE_V>();
    Brcb(tensorGTmpFp32, tensorGFp32, CEIL_DIV(real_BT, 8), {1, 8});
    PipeBarrier<PIPE_V>();
    AscendC::Mul(tensorDqInFp32, tensorDqInFp32, tensorGTmpFp32, CAL_NUM_FLOAT, real_BT,
                    {1, 1, 0, 16, 16, 1});
    AscendC::Mul(tensorDqInFp32[CAL_NUM_FLOAT], tensorDqInFp32[CAL_NUM_FLOAT], tensorGTmpFp32,
                    CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
    PipeBarrier<PIPE_V>();
#endif
}


// ============== A_vector: Part1 (dw 取负 + dg_last) ==============
// 每 chunk: WaitCubeReady 一次 -> Part1 (head-split) -> SetVectorDone 一次。
// 依赖: Part1 读 dw(由 A_cube 产出); mul1 改由 B_vector 内联计算, 不再预写 GM。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessAVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    const uint32_t hDhSize = mul0RowNum * V; // 16 * 256 or 32 * 128 = 4k

    // ----- Part1 buffers (h/dh, dw) -----
    uint32_t bIdx = loopIdx / numChunks;
    uint32_t chunkIdx = loopIdx % numChunks;
    uint32_t actual_dwSize = real_BT * K;
    WaitCubeReady();

    // 同步AIV子核，等待dg_last空间释放
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);

    // ---------- Part1: dg_last = sum(h*dh), dw = -dw (head-split) ----------
    for (uint32_t h = 0; h < HV; h++) {
        if ((h & 1) != subBlockIdx) {
            continue;
        }
        uint64_t hOffset = ((bIdx * HV + h) * numChunks + chunkIdx) * K * V;
        uint64_t dwOffset = (h * T + bos) * K; // 最终输出 ptrDw 仍全局寻址
        uint64_t dgLastOffset = DqkwgScalarElemOffset(coreIdx, h, HV);

        // ===== dg_last = sum(h * dh) =====
        // CV 融合优化: 在 cube-bound 的 stage A 算 (被 cube 的 2H 个 matmul 藏住), 写 wsDgLast;
        //   vector-bound 的 D_vector 改为读 wsDgLast, 省掉 D 的 h/dh 读 + K*V 归约 (给瓶颈减负)。
        auto tensorSumFp32 = inQue1.AllocTensor<float>();
        for (uint32_t row = 0; row < K; row += mul0RowNum) {
            {
                auto tensorHFp32 = inQue2.AllocTensor<float>();
                // 拷入后半部分，给后面Cast预留空间
                auto tensorHIn = tensorHFp32[hDhSize].template ReinterpretCast<DataType>();
                DataCopy(tensorHIn, gmH[hOffset + row * V], hDhSize);
                DataCopy(tensorHIn[hDhSize], gmDh[hOffset + row * V], hDhSize);
                inQue2.EnQue(tensorHFp32);
            }
            {
                auto tensorHFp32 = inQue2.DeQue<float>();
                auto tensorDhFp32 = tensorHFp32[hDhSize];
                auto tensorHIn = tensorDhFp32.template ReinterpretCast<DataType>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if (row == 0) {
                    DgLastMulAccum<DataType, false>(
                        (__ubuf__ float *)tensorSumFp32.GetPhyAddr(), (__ubuf__ DataType *)tensorHIn.GetPhyAddr(),
                        (__ubuf__ DataType *)tensorHIn[hDhSize].GetPhyAddr(), hDhSize);
                } else {
                    DgLastMulAccum<DataType, true>(
                        (__ubuf__ float *)tensorSumFp32.GetPhyAddr(), (__ubuf__ DataType *)tensorHIn.GetPhyAddr(),
                        (__ubuf__ DataType *)tensorHIn[hDhSize].GetPhyAddr(), hDhSize);
                }
#else
                Cast(tensorHFp32, tensorHIn, RoundMode::CAST_NONE, 2 * hDhSize);
                PipeBarrier<PIPE_V>();
                if (row == 0) {
                    Mul(tensorSumFp32, tensorHFp32, tensorDhFp32, hDhSize);
                } else {
                    Mul(tensorHFp32, tensorHFp32, tensorDhFp32, hDhSize);
                    PipeBarrier<PIPE_V>();
                    Add(tensorSumFp32, tensorSumFp32, tensorHFp32, hDhSize);
                }
                PipeBarrier<PIPE_V>();
#endif
                inQue2.FreeTensor(tensorHFp32);
            }
        }
        {
            auto tensorDgLastOut = outQue2.AllocTensor<float>();
            ReduceSumCustom(tensorDgLastOut, tensorSumFp32, hDhSize);
            PipeBarrier<PIPE_V>();
            outQue2.EnQue(tensorDgLastOut);
            inQue1.FreeTensor(tensorSumFp32);
        }
        {
            auto tensorDgLastOut = outQue2.DeQue<float>();
            DataCopyPad(gmDgLast[dgLastOffset], tensorDgLastOut, {1, sizeof(float), 0, 0});
            outQue2.FreeTensor(tensorDgLastOut);
        }

        // ===== dw = -dw, then vector-repair row-0 first block =====
        {
            auto tensorDwIn = inQue1.AllocTensor<DataType>();
            DataCopy(tensorDwIn[actual_dwSize], gmDw[dwOffset], actual_dwSize);
            inQue1.EnQue(tensorDwIn);
        }
        {
            auto tensorDwIn = inQue1.DeQue<DataType>();
            auto tensorDwOut = outQue1.AllocTensor<DataType>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            DwNegate<DataType>((__ubuf__ DataType *)tensorDwOut.GetPhyAddr(),
                                (__ubuf__ DataType *)tensorDwIn[actual_dwSize].GetPhyAddr(),
                                actual_dwSize);
#else
            auto tensorDwInFp32 = tensorDwIn.template ReinterpretCast<float>();
            Cast(tensorDwInFp32, tensorDwIn[actual_dwSize], RoundMode::CAST_NONE, actual_dwSize);
            PipeBarrier<PIPE_V>();
            Muls(tensorDwInFp32, tensorDwInFp32, -1.0f, actual_dwSize);
            PipeBarrier<PIPE_V>();
            Cast(tensorDwOut, tensorDwInFp32, RoundMode::CAST_RINT, actual_dwSize);
            PipeBarrier<PIPE_V>();
#endif
            outQue1.EnQue(tensorDwOut);
            inQue1.FreeTensor(tensorDwIn);
        }
        {
            auto tensorDwOut = outQue1.DeQue<DataType>();
            DataCopy(gmDw[dwOffset], tensorDwOut, actual_dwSize);
            outQue1.FreeTensor(tensorDwOut);
        }
    }
}

// ============== B_vector: 原 Part2 (mul1) + 原 Part3 (ds_temp + dg 部分) ==============
// 每 chunk: WaitCubeReady 一次 -> head-split 处理 (读 ds(B_cube), 内联算 mul1, 读 mm5(A_cube)) -> SetVectorDone 一次。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessBVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t actual_dsSize = real_BT * BT;
    WaitCubeReady();

    for (uint32_t h = 0; h < HV; h++) {
        if ((h & 1) != subBlockIdx) {
            continue;
        }
        uint64_t gOffset = (h * T + bos);
        uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, HV, BT);
        uint64_t mm3Offset = dsOffset;
        uint64_t dgOffset = gOffset;

        {
            auto tensorDsIn = inQue1.AllocTensor<DataType>();
            auto tensorGIn = inQue3.AllocTensor<GType>();
            DataCopy(tensorDsIn[BT * BT], gmDsTemp[dsOffset], actual_dsSize);
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
            }
            inQue1.EnQue(tensorDsIn);
            inQue3.EnQue(tensorGIn);
        }
        {
            auto tensorDsInFp16 = inQue1.DeQue<DataType>();
            auto tensorDsInFp32 = tensorDsInFp16.template ReinterpretCast<float>();
            auto tensorGIn = inQue3.DeQue<GType>();
            auto tensorGFp32Left = tensorGIn.template ReinterpretCast<float>();
            auto tensorDsTempOut = outQue1.AllocTensor<DataType>();
            auto tensorDgOut = outQue2.AllocTensor<float>();

            Cast(tensorDsInFp32, tensorDsInFp16[BT * BT], RoundMode::CAST_NONE, actual_dsSize);
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorGFp32Left, tensorGIn[BT], RoundMode::CAST_NONE, BT);
            }
            PipeBarrier<PIPE_V>();

            // mul1 从 g 内联算两个 row-half, 写入 inQue2 buffer (fp32)。
            auto tensorMul1InFp32 = inQue2.AllocTensor<float>();
            uint32_t vec0 = (real_BT >= 64) ? 64 : real_BT;
            uint32_t vec1 = (real_BT > 64) ? (real_BT - 64) : 0;
            ComputeMul1HalfFp32(tensorMul1InFp32, tensorMaskA, tensorGTmpFp32, tensorGFp32Left, tensorDgTmpFp32, 0, vec0);
            if (vec1 > 0) {
                ComputeMul1HalfFp32(tensorMul1InFp32[vec0 * BT], tensorMaskA, tensorGTmpFp32, tensorGFp32Left, tensorDgTmpFp32, vec0, vec1);
            }
            PipeBarrier<PIPE_V>();
            // b_ds_temp = b_ds * mul1 (已应用掩码)
            Mul(tensorDsInFp32, tensorDsInFp32, tensorMul1InFp32, actual_dsSize);
            PipeBarrier<PIPE_V>();
            inQue3.FreeTensor(tensorGIn);
            inQue2.FreeTensor(tensorMul1InFp32);

            // 搬入 mm3, 复用 mul1 空间
            auto tensorMm3InFp16Tmp = inQue2.AllocTensor<DataType>();
            DataCopy(tensorMm3InFp16Tmp[BT * BT], gmMm3[mm3Offset], actual_dsSize);
            inQue2.EnQue(tensorMm3InFp16Tmp);
            auto tensorMm3InFp16 = inQue2.DeQue<DataType>();
            auto tensorMm3InFp32 = tensorMm3InFp16.template ReinterpretCast<float>();
            Cast(tensorMm3InFp32, tensorMm3InFp16[BT * BT], RoundMode::CAST_NONE, actual_dsSize);
            PipeBarrier<PIPE_V>();

            Mul(tensorMm3InFp32, tensorDsInFp32, tensorMm3InFp32, actual_dsSize);       // b_ds2 = b_ds_temp * mm3
            Cast(tensorDsTempOut, tensorDsInFp32, RoundMode::CAST_RINT, actual_dsSize); // ds_temp -> fp16
            PipeBarrier<PIPE_V>();

            // 行求和 -> [BT] (+Add0.C)
            // 列求和 -> [BT] (-Add0.D)
            if (real_BT > FP32_PER_REPEAT) {
                DataCopy(tensorDsInFp32, tensorMm3InFp32, {static_cast<uint16_t>(real_BT), 8, 8, 0});
                PipeBarrier<PIPE_V>();
                Add(tensorDsInFp32, tensorDsInFp32, tensorMm3InFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT, real_BT, {1, 1, 1, 8, 8, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut, tensorDsInFp32, FP32_PER_REPEAT, real_BT, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                Add(tensorMm3InFp32, tensorMm3InFp32[BT], tensorMm3InFp32, FP32_PER_REPEAT, real_BT - 1, {1, 1, 1, 0, static_cast<uint8_t>(BT / 8), 0});
                PipeBarrier<PIPE_V>();
                Add(tensorMm3InFp32[FP32_PER_REPEAT], tensorMm3InFp32[BT + FP32_PER_REPEAT], tensorMm3InFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT, real_BT - 1, {1, 1, 1, 0, static_cast<uint8_t>(BT / 8), 0});
                PipeBarrier<PIPE_V>();
            } else {
                WholeReduceSum(tensorDgOut, tensorMm3InFp32, real_BT, real_BT, 1, 1, BT / 8);
                PipeBarrier<PIPE_V>();
                if (real_BT > 1) {
                    Add(tensorMm3InFp32, tensorMm3InFp32[BT], tensorMm3InFp32, real_BT, real_BT - 1, {1, 1, 1, 0, static_cast<uint8_t>(BT / 8), 0});
                    PipeBarrier<PIPE_V>();
                }
            }

            Sub(tensorDgOut, tensorDgOut, tensorMm3InFp32, real_BT);
            PipeBarrier<PIPE_V>();
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, real_BT);
            }

            outQue1.EnQue(tensorDsTempOut);
            outQue2.EnQue(tensorDgOut);
            inQue1.FreeTensor(tensorDsInFp16);
            inQue2.FreeTensor(tensorMm3InFp16);
        }
        {
            auto tensorDsTempOut = outQue1.DeQue<DataType>();
            auto tensorDgOut = outQue2.DeQue<GType>();
            DataCopy(gmDsTemp[dsOffset], tensorDsTempOut, actual_dsSize);
            DataCopyPad(gmDg[dgOffset], tensorDgOut, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
            outQue1.FreeTensor(tensorDsTempOut);
            outQue2.FreeTensor(tensorDgOut);
        }
    }
    SetVecCredit();
    SetVecClear();
}

// ============== C_vector: 原 Part4 + Part6 (dq 最终 + dg) ==============
// C_cube 在一次 SetCubeReady 前已产出 dq_inner(ptrDq) 与 mm6(wsMm6); C_vector 一次 WaitCubeReady 后
// 同时读取二者 (无 per-head phase 握手), 计算 dq_state/dg 并 dq += mm6。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessCVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t actual_dqSize = real_BT * K;
    WaitCubeReady();

    for (uint32_t h = 0; h < HV; h++) {
        if ((h & 1) != subBlockIdx) {
            continue;
        }
        uint64_t qkOffset = (h * T + bos) * K;    // q (HK)
        uint64_t dqOutOffset = (h * T + bos) * K; // dq (HK)
        uint64_t gOffset = (h * T + bos);
        uint64_t mm4Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        // CopyIn: dq_inner, q, g, dg
        {
            auto tensorDqIn = inQue1.AllocTensor<DataType>();
            auto tensorQIn = inQue2.AllocTensor<DataType>();
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            DataCopy(tensorDqIn[actual_dqSize], gmMm4[mm4Offset], actual_dqSize);
            DataCopy(tensorQIn[actual_dqSize], gmQ[qkOffset], actual_dqSize);
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue1.EnQue(tensorDqIn);
            inQue2.EnQue(tensorQIn);
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        {
            auto tensorDqInFp16 = inQue1.DeQue<DataType>();
            auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
            auto tensorQInFp16 = inQue2.DeQue<DataType>();
            auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();
            auto tensorGIn = inQue3.DeQue<GType>();
            auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
            auto tensorDgIn = inQue4.DeQue<GType>();
            auto tensorDgAdd = tensorDgIn.template ReinterpretCast<float>();
            auto tensorDgOut = outQue2.AllocTensor<float>();

            Cast(tensorDqInFp32, tensorDqInFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
            Cast(tensorQInFp32, tensorQInFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, BT);
                Cast(tensorDgAdd, tensorDgIn[BT], RoundMode::CAST_NONE, BT);
            }
            PipeBarrier<PIPE_V>();

            // dq_state = dq_inner * exp(g)[:,None] * scale
            ComputeDqState(tensorDqInFp32, tensorGFp32, tensorGTmpFp32, real_BT);

            // dg_C = row_sum(dq_state * q)
            Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, actual_dqSize);
            PipeBarrier<PIPE_V>();
            Add(tensorQInFp32, tensorQInFp32, tensorQInFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, real_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorDgOut, tensorQInFp32, FP32_PER_REPEAT, real_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            Add(tensorDgOut, tensorDgAdd, tensorDgOut, real_BT);
            PipeBarrier<PIPE_V>();
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, BT);
            }

            outQue2.EnQue(tensorDgOut);
            inQue2.FreeTensor(tensorQInFp16);
            inQue3.FreeTensor(tensorGIn);
            inQue4.FreeTensor(tensorDgIn);

            // dg 写回
            {
                auto tensorDgOutDeq = outQue2.DeQue<GType>();
                DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
                outQue2.FreeTensor(tensorDgOutDeq);
            }

            // dq += mm6 (从 wsMm6 环形区读取, dq_state 仍在 UB)
            {
                auto tensorMm6In = inQue2.AllocTensor<DataType>();
                DataCopy(tensorMm6In[actual_dqSize], gmMm6[mm4Offset], actual_dqSize); // mm6 compact ring
                inQue2.EnQue(tensorMm6In);
            }
            {
                auto tensorMm6InFp16 = inQue2.DeQue<DataType>();
                auto tensorMm6Fp32 = tensorMm6InFp16.template ReinterpretCast<float>();
                auto tensorDqOut = outQue1.AllocTensor<DataType>();
                Cast(tensorMm6Fp32, tensorMm6InFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
                PipeBarrier<PIPE_V>();
                Add(tensorDqInFp32, tensorDqInFp32, tensorMm6Fp32, actual_dqSize);
                PipeBarrier<PIPE_V>();
                Cast(tensorDqOut, tensorDqInFp32, RoundMode::CAST_RINT, actual_dqSize);
                outQue1.EnQue(tensorDqOut);
                inQue1.FreeTensor(tensorDqInFp16);
                inQue2.FreeTensor(tensorMm6InFp16);
            }
            {
                auto tensorDqOut = outQue1.DeQue<DataType>();
                DataCopy(gmDq[dqOutOffset], tensorDqOut, actual_dqSize);
                outQue1.FreeTensor(tensorDqOut);
            }
        }
    }
    SetVecClear();
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessCVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t half_dqSize = half_BT * K;
    uint32_t actual_dqSize = real_BT * K;
    uint32_t first_BT = real_BT > half_BT ? half_BT : real_BT;
    uint32_t second_BT = real_BT - first_BT;
    uint32_t first_dqSize = first_BT * K;
    uint32_t second_dqSize = second_BT * K;
    uint32_t bIdx = loopIdx / numChunks;
    uint64_t bos_hk = bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(HV - HK) * T;
    WaitCubeReady();

    LocalTensor<float> tensorDqSum;
    for (uint32_t h = 0; h < HV; h++) {
        uint32_t hk_idx = h / n_ratio;
        if ((hk_idx & 1) != subBlockIdx) {
            continue;
        }
        uint64_t qkOffset = (hk_idx * T + bos_hk) * K;    // q (HK)
        uint64_t dqOutOffset = (hk_idx * T + bos_hk) * K; // dq (HK)
        uint64_t gOffset = (h * T + bos);
        uint64_t mm4Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        // CopyIn: g, dg
        {
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        auto tensorGIn = inQue3.DeQue<GType>();
        auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
        auto tensorDgIn = inQue4.DeQue<GType>();
        auto tensorDgAdd = tensorDgIn.template ReinterpretCast<float>();
        auto tensorDgOut = outQue2.AllocTensor<float>();
        if constexpr (!std::is_same<GType, float>::value) {
            Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, BT);
            Cast(tensorDgAdd, tensorDgIn[BT], RoundMode::CAST_NONE, BT);
        }

        // gva模式下，dq在UB上做累加，由于UB大小限制，dq和q分两次拷入计算
        // CopyIn First half: dq_inner, q
        {
            auto tensorDqIn = inQue2.AllocTensor<DataType>();
            auto tensorQIn = tensorDqIn[half_dqSize * 2];
            DataCopy(tensorDqIn[half_dqSize], gmMm4[mm4Offset], first_dqSize);
            DataCopy(tensorQIn[half_dqSize], gmQ[qkOffset], first_dqSize);
            inQue2.EnQue(tensorDqIn);
        }
        {
            auto tensorDqInFp16 = inQue2.DeQue<DataType>();
            auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
            auto tensorQInFp16 = tensorDqInFp16[half_dqSize * 2];
            auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();

            Cast(tensorDqInFp32, tensorDqInFp16[half_dqSize], RoundMode::CAST_NONE, first_dqSize);
            Cast(tensorQInFp32, tensorQInFp16[half_dqSize], RoundMode::CAST_NONE, first_dqSize);
            PipeBarrier<PIPE_V>();

            // dq_state = dq_inner * exp(g)[:,None] * scale
            ComputeDqState(tensorDqInFp32, tensorGFp32, tensorGTmpFp32, first_BT);

            // dg_C = row_sum(dq_state * q)
            Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, first_dqSize);
            PipeBarrier<PIPE_V>();
            Add(tensorQInFp32, tensorQInFp32, tensorQInFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, first_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorDgOut, tensorQInFp32, FP32_PER_REPEAT, first_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();

            if (h % n_ratio == 0) {
                tensorDqSum = inQue1.AllocTensor<float>();
                DataCopy(tensorDqSum, tensorDqInFp32, first_dqSize);
                PipeBarrier<PIPE_V>();
            } else {
                Add(tensorDqSum, tensorDqSum, tensorDqInFp32, first_dqSize);
                PipeBarrier<PIPE_V>();
            }
            inQue2.FreeTensor(tensorDqInFp16);
        }
        if (second_BT > 0) {
            // CopyIn Second half: dq_inner, q
            {
                auto tensorDqIn = inQue2.AllocTensor<DataType>();
                auto tensorQIn = tensorDqIn[half_dqSize * 2];
                DataCopy(tensorDqIn[half_dqSize], gmMm4[mm4Offset + half_dqSize], second_dqSize);
                DataCopy(tensorQIn[half_dqSize], gmQ[qkOffset + half_dqSize], second_dqSize);
                inQue2.EnQue(tensorDqIn);
            }
            {
                auto tensorDqInFp16 = inQue2.DeQue<DataType>();
                auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
                auto tensorQInFp16 = tensorDqInFp16[half_dqSize * 2];
                auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();

                Cast(tensorDqInFp32, tensorDqInFp16[half_dqSize], RoundMode::CAST_NONE, second_dqSize);
                Cast(tensorQInFp32, tensorQInFp16[half_dqSize], RoundMode::CAST_NONE, second_dqSize);
                PipeBarrier<PIPE_V>();

                // dq_state = dq_inner * exp(g)[:,None] * scale
                ComputeDqState(tensorDqInFp32, tensorGFp32[half_BT], tensorGTmpFp32, second_BT);

                // dg_C = row_sum(dq_state * q)
                Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, second_dqSize);
                PipeBarrier<PIPE_V>();
                Add(tensorQInFp32, tensorQInFp32, tensorQInFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, second_BT, {1, 1, 1, 16, 16, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut[half_BT], tensorQInFp32, FP32_PER_REPEAT, second_BT, 1, 1, 16);
                PipeBarrier<PIPE_V>();

                if (h % n_ratio == 0) {
                    DataCopy(tensorDqSum[half_dqSize], tensorDqInFp32, second_dqSize);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(tensorDqSum[half_dqSize], tensorDqSum[half_dqSize], tensorDqInFp32, second_dqSize);
                    PipeBarrier<PIPE_V>();
                }
                inQue2.FreeTensor(tensorDqInFp16);
            }
        }

        Add(tensorDgOut, tensorDgAdd, tensorDgOut, real_BT);
        PipeBarrier<PIPE_V>();
        if constexpr (!std::is_same<GType, float>::value) {
            Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, BT);
        }
        outQue2.EnQue(tensorDgOut);
        inQue3.FreeTensor(tensorGIn);
        inQue4.FreeTensor(tensorDgIn);

        // dg 写回
        {
            auto tensorDgOutDeq = outQue2.DeQue<GType>();
            DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
            outQue2.FreeTensor(tensorDgOutDeq);
        }

        // dq += mm6 (从 wsMm6 环形区读取, dq_state 仍在 UB)
        {
            auto tensorMm6In = inQue2.AllocTensor<DataType>();
            DataCopy(tensorMm6In[actual_dqSize], gmMm6[mm4Offset], actual_dqSize); // mm6 compact ring
            inQue2.EnQue(tensorMm6In);
        }
        {
            auto tensorMm6InFp16 = inQue2.DeQue<DataType>();
            auto tensorMm6Fp32 = tensorMm6InFp16.template ReinterpretCast<float>();
            Cast(tensorMm6Fp32, tensorMm6InFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
            PipeBarrier<PIPE_V>();
            Add(tensorDqSum, tensorDqSum, tensorMm6Fp32, actual_dqSize);
            PipeBarrier<PIPE_V>();
            inQue2.FreeTensor(tensorMm6InFp16);
        }
        if (h % n_ratio == n_ratio - 1) {
            auto tensorDqOut = outQue1.AllocTensor<DataType>();
            Cast(tensorDqOut, tensorDqSum, RoundMode::CAST_RINT, actual_dqSize);
            outQue1.EnQue(tensorDqOut);
            inQue1.FreeTensor(tensorDqSum);

            tensorDqOut = outQue1.DeQue<DataType>();
            DataCopy(gmDq[dqOutOffset], tensorDqOut, actual_dqSize);
            outQue1.FreeTensor(tensorDqOut);
        }
    }
    SetVecClear();
}

// ============== D_vector: 原 Part5 + Part7 (dk 最终 + dg 最终) ==============
// D_cube 在一次 SetCubeReady 前已产出 dk_inner(ptrDk) 与 mm7(wsMm7); D_vector 一次 WaitCubeReady 后
// 读取二者, 完成 dk_state / dg 最终 / dk += mm7。dg_last 本地重算 (与原 Part5 一致)。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessDVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t real_BT_aligned = (real_BT + 15) / 16 * 16;
    uint32_t actual_dkSize = real_BT * K;
    WaitCubeReady();

    // wait dg out
    TEventID eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    for (uint32_t h = 0; h < HV; h++) {
        if ((h & 1) != subBlockIdx) {
            continue;
        }
        uint64_t kOffset = (h * T + bos) * K;     // k (HK)
        uint64_t dkOutOffset = (h * T + bos) * K; // dk (HK)
        uint64_t gOffset = (h * T + bos);
        uint64_t mm5Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        // CV 融合优化: 读 A_vector 算好的 dg_last = sum(h*dh) (替代本地重算, 省 D 的 h/dh 读 + K*V 归约)。
        // 跨 stage 可见性由 Process() 中 A->B->C->D 的 PipeBarrier<PIPE_ALL> 保证; A(c,h)/D(c,h) 同 sub-block。
        // 输出格式与原重算一致: tensorGTmpFp32[16..23] = 8 份 dg_last。
        {
            uint64_t dgLastOffset = DqkwgScalarElemOffset(coreIdx, h, HV);
            DataCopyExtParams copyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 7, 0};
            TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
            DataCopyPad(tensorDgTmpFp32[8], gmDgLast[dgLastOffset], copyParams, padParams);
            TEventID eDg = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eDg);
            WaitFlag<HardEvent::MTE2_V>(eDg);
            Brcb(tensorGTmpFp32[16], tensorDgTmpFp32[8], 1, {1, 8}); // 广播到 [16..23] 8 份
            PipeBarrier<PIPE_V>();
        }

        // CopyIn: dk_inner, k, g, dg
        {
            auto tensorDkIn = inQue1.AllocTensor<DataType>();
            auto tensorKIn = inQue2.AllocTensor<DataType>();
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            DataCopy(tensorDkIn[actual_dkSize], gmMm5[mm5Offset], actual_dkSize);
            DataCopy(tensorKIn[actual_dkSize], gmK[kOffset], actual_dkSize);
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue1.EnQue(tensorDkIn);
            inQue2.EnQue(tensorKIn);
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        {
            auto tensorDkIn = inQue1.DeQue<DataType>();
            auto tensorDkFp32 = tensorDkIn.template ReinterpretCast<float>();
            auto tensorKIn = inQue2.DeQue<DataType>();
            auto tensorKFp32 = tensorKIn.template ReinterpretCast<float>();
            auto tensorGIn = inQue3.DeQue<GType>();
            auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
            auto tensorDgIn = inQue4.DeQue<GType>();
            auto tensorDgTmp = tensorDgIn.template ReinterpretCast<float>();
            auto tensorDgOut = outQue2.AllocTensor<GType>();

            Cast(tensorDkFp32, tensorDkIn[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
            Cast(tensorKFp32, tensorKIn[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
                Cast(tensorDgTmp, tensorDgIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
            }
            PipeBarrier<PIPE_V>();

            // MUL2: dk_state = dk_inner * exp(-g + g_last)[:,None]
            uint32_t last_line_no = (real_BT - 1) / 8 * 8;
            uint32_t last_line_idx = real_BT - 1 - last_line_no;
            Brcb(tensorDgFinal, tensorGFp32[last_line_no], 1, {1, 8}); // [8,8] 第 last_line_idx 行 = gLast
            PipeBarrier<PIPE_V>();
            Muls(tensorGFp32, tensorGFp32, -1.0f, real_BT_aligned);
            DataCopy(tensorGTmpFp32, tensorDgFinal[last_line_idx * 8], 8);
            PipeBarrier<PIPE_V>();
            AscendC::Add(tensorGFp32, tensorGFp32, tensorDgFinal[last_line_idx * 8], CAL_NUM_FLOAT,
                            BT / CAL_NUM_FLOAT, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
            Exp(tensorGFp32, tensorGFp32, real_BT_aligned);
            PipeBarrier<PIPE_V>();

            Brcb(tensorDgFinal, tensorGFp32, CEIL_DIV(real_BT, 8), {1, 8});
            PipeBarrier<PIPE_V>();
            AscendC::Mul(tensorDkFp32, tensorDkFp32, tensorDgFinal, CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(tensorDkFp32[CAL_NUM_FLOAT], tensorDkFp32[CAL_NUM_FLOAT], tensorDgFinal, CAL_NUM_FLOAT,
                            real_BT, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();

            Mul(tensorKFp32, tensorKFp32, tensorDkFp32, actual_dkSize); // mul8 = dk_state * k
            PipeBarrier<PIPE_V>();

            // Add0.B = row_sum(dk_state * k)
            Add(tensorKFp32, tensorKFp32, tensorKFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, real_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorGFp32, tensorKFp32, FP32_PER_REPEAT, real_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();

            Sub(tensorDgTmp, tensorDgTmp, tensorGFp32, BT); // Add.0 最终结果 (dg_B+dg_C+dg_D)
            PipeBarrier<PIPE_V>();

            // Sum0: [real_BT] -> [1]
            if (real_BT > FP32_PER_REPEAT) {
                Add(tensorGFp32, tensorGFp32, tensorGFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT);
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgFinal, tensorGFp32, FP32_PER_REPEAT, 1, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                WholeReduceSum(tensorDgFinal, tensorGFp32, real_BT, 1, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            }
            Brcb(tensorDgTmpFp32, tensorDgFinal, 1, {1, 8});
            PipeBarrier<PIPE_V>();
            Exp(tensorGTmpFp32, tensorGTmpFp32, 8);
            PipeBarrier<PIPE_V>();
            Mul(tensorDgTmpFp32[16], tensorGTmpFp32[16], tensorGTmpFp32, 8);
            PipeBarrier<PIPE_V>();
            Add(tensorDgTmpFp32[16], tensorDgTmpFp32[16], tensorDgTmpFp32, 8); // add4 = dg_last_term
            PipeBarrier<PIPE_V>();
            Brcb(tensorDgFinal, tensorDgTmpFp32[16], 1, {1, 8});
            PipeBarrier<PIPE_V>();
            uint64_t offset = (real_BT - 1) / 8 * 8;
            uint64_t mask[1] = {0};
            mask[0] = 1ULL << (real_BT - 1 - offset); // 仅最后一个位置加 dg_last_term
            Add(tensorDgTmp[offset], tensorDgTmp[offset], tensorDgFinal, mask, 1, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<GType, float>::value) {
                DataCopy(tensorDgOut, tensorDgTmp, BT);
            } else {
                Cast(tensorDgOut, tensorDgTmp, RoundMode::CAST_RINT, BT);
            }
            PipeBarrier<PIPE_V>();

            outQue2.EnQue(tensorDgOut);
            inQue2.FreeTensor(tensorKIn);
            inQue3.FreeTensor(tensorGIn);
            inQue4.FreeTensor(tensorDgIn);
            {
                auto tensorDgOutDeq = outQue2.DeQue<GType>();
                DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
                outQue2.FreeTensor(tensorDgOutDeq);
            }

            // dk += mm7 (从 wsMm7 读取, dk_state 仍在 UB)
            {
                auto tensorMm7In = inQue2.AllocTensor<DataType>();
                DataCopy(tensorMm7In[actual_dkSize], gmMm7[mm5Offset], actual_dkSize);
                inQue2.EnQue(tensorMm7In);
            }
            {
                auto tensorMm7In = inQue2.DeQue<DataType>();
                auto tensorMm7Fp32 = tensorMm7In.template ReinterpretCast<float>();
                auto tensorDkOut = outQue1.AllocTensor<DataType>();
                Cast(tensorMm7Fp32, tensorMm7In[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
                PipeBarrier<PIPE_V>();
                Add(tensorDkFp32, tensorDkFp32, tensorMm7Fp32, actual_dkSize);
                PipeBarrier<PIPE_V>();
                Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, actual_dkSize);
                outQue1.EnQue(tensorDkOut);
                inQue1.FreeTensor(tensorDkIn);
                inQue2.FreeTensor(tensorMm7In);
            }
            {
                auto tensorDkOut = outQue1.DeQue<DataType>();
                DataCopy(gmDk[dkOutOffset], tensorDkOut, actual_dkSize);
                outQue1.FreeTensor(tensorDkOut);
            }
        }
    }
    SetVecClear();
}

// ============== D_vector GVA: 原 Part5 + Part7 (dk 最终 + dg 最终) (GVA) ==============
// 与 ProcessDVector 的区别 (镜像 ProcessCVectorForGva 的 GVA 适配):
//  1. 按 hk_idx = h / n_ratio 切分 sub-block, 而非 h & 1;
//  2. k/dk 用 HK 侧偏移 (bos_hk), g/dg 用 HV 侧偏移 (h);
//  3. dk/k 因 UB 限制分两半拷入计算, dk_state 在 UB 上跨 n_ratio 个 HV head 累加;
//  4. dg 仍逐 HV head 写回; dk 仅在 h % n_ratio == n_ratio - 1 时写回 (HK head 粒度);
//  5. mm7 逐 HV head 累加到 dkSum。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessDVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t half_dkSize = half_BT * K;
    uint32_t real_BT_aligned = (real_BT + 15) / 16 * 16;
    uint32_t actual_dkSize = real_BT * K;
    uint32_t first_BT = real_BT > half_BT ? half_BT : real_BT;
    uint32_t second_BT = real_BT - first_BT;
    uint32_t first_dkSize = first_BT * K;
    uint32_t second_dkSize = second_BT * K;
    uint32_t bIdx = loopIdx / numChunks;
    uint64_t bos_hk = bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(HV - HK) * T;
    WaitCubeReady();

    // wait dg out
    TEventID eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    LocalTensor<float> tensorDkSum;
    for (uint32_t h = 0; h < HV; h++) {
        uint32_t hk_idx = h / n_ratio;
        if ((hk_idx & 1) != subBlockIdx) {
            continue;
        }
        uint64_t kOffset = (hk_idx * T + bos_hk) * K;     // k (HK)
        uint64_t dkOutOffset = (hk_idx * T + bos_hk) * K; // dk (HK)
        uint64_t gOffset = (h * T + bos);
        uint64_t mm5Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        // CV 融合优化: 读 A_vector 算好的 dg_last = sum(h*dh)
        {
            uint64_t dgLastOffset = DqkwgScalarElemOffset(coreIdx, h, HV);
            TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
            DataCopyExtParams copyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 7, 0};
            DataCopyPad(tensorDgTmpFp32[8], gmDgLast[dgLastOffset], copyParams, padParams);
            TEventID eDg = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eDg);
            WaitFlag<HardEvent::MTE2_V>(eDg);
            Brcb(tensorGTmpFp32[16], tensorDgTmpFp32[8], 1, {1, 8}); // 广播到 [16..23] 8 份
            PipeBarrier<PIPE_V>();
        }

        // CopyIn: g, dg
        {
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        auto tensorGIn = inQue3.DeQue<GType>();
        auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
        auto tensorDgIn = inQue4.DeQue<GType>();
        auto tensorDgTmp = tensorDgIn.template ReinterpretCast<float>();
        auto tensorDgOut = outQue2.AllocTensor<GType>();
        if constexpr (!std::is_same<GType, float>::value) {
            Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
            Cast(tensorDgTmp, tensorDgIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
        }
        PipeBarrier<PIPE_V>();

        // MUL2: dk_state = dk_inner * exp(-g + g_last)[:,None] -- 全量 real_BT 计算 exp 因子
        uint32_t last_line_no = (real_BT - 1) / 8 * 8;
        uint32_t last_line_idx = real_BT - 1 - last_line_no;
        Brcb(tensorDgFinal, tensorGFp32[last_line_no], 1, {1, 8}); // [8,8] 第 last_line_idx 行 = gLast
        PipeBarrier<PIPE_V>();
        Muls(tensorGFp32, tensorGFp32, -1.0f, real_BT_aligned);
        DataCopy(tensorGTmpFp32, tensorDgFinal[last_line_idx * 8], 8);
        PipeBarrier<PIPE_V>();
        AscendC::Add(tensorGFp32, tensorGFp32, tensorDgFinal[last_line_idx * 8], CAL_NUM_FLOAT,
                        BT / CAL_NUM_FLOAT, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Exp(tensorGFp32, tensorGFp32, real_BT_aligned);
        PipeBarrier<PIPE_V>();

        // gva模式下，dk在UB上做累加，由于UB大小限制，dk和k分两次拷入计算
        // First half: dk_inner, k
        {
            auto tensorDkIn = inQue2.AllocTensor<DataType>();
            auto tensorKIn = tensorDkIn[half_dkSize * 2];
            DataCopy(tensorDkIn[half_dkSize], gmMm5[mm5Offset], first_dkSize);
            DataCopy(tensorKIn[half_dkSize], gmK[kOffset], first_dkSize);
            inQue2.EnQue(tensorDkIn);
        }
        {
            auto tensorDkInFp16 = inQue2.DeQue<DataType>();
            auto tensorDkFp32 = tensorDkInFp16.template ReinterpretCast<float>();
            auto tensorKInFp16 = tensorDkInFp16[half_dkSize * 2];
            auto tensorKFp32 = tensorKInFp16.template ReinterpretCast<float>();

            Cast(tensorDkFp32, tensorDkInFp16[half_dkSize], RoundMode::CAST_NONE, first_dkSize);
            Cast(tensorKFp32, tensorKInFp16[half_dkSize], RoundMode::CAST_NONE, first_dkSize);
            PipeBarrier<PIPE_V>();

            // dk_state = dk_inner * exp(-g + g_last)[:,None]
            Brcb(tensorDgFinal, tensorGFp32, CEIL_DIV(first_BT, 8), {1, 8});
            PipeBarrier<PIPE_V>();
            AscendC::Mul(tensorDkFp32, tensorDkFp32, tensorDgFinal, CAL_NUM_FLOAT, first_BT, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(tensorDkFp32[CAL_NUM_FLOAT], tensorDkFp32[CAL_NUM_FLOAT], tensorDgFinal, CAL_NUM_FLOAT,
                            first_BT, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();

            if (h % n_ratio == 0) {
                tensorDkSum = inQue1.AllocTensor<float>();
                DataCopy(tensorDkSum, tensorDkFp32, first_dkSize);
                PipeBarrier<PIPE_V>();
            } else {
                Add(tensorDkSum, tensorDkSum, tensorDkFp32, first_dkSize);
                PipeBarrier<PIPE_V>();
            }

            // mul8 = dk_state * k
            Mul(tensorKFp32, tensorKFp32, tensorDkFp32, first_dkSize);
            PipeBarrier<PIPE_V>();

            // Add0.B = row_sum(dk_state * k) -> tensorGFp32[0..first_BT-1]
            Add(tensorKFp32, tensorKFp32, tensorKFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, first_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorGFp32, tensorKFp32, FP32_PER_REPEAT, first_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();

            inQue2.FreeTensor(tensorDkInFp16);
        }
        if (second_BT > 0) {
            // Second half: dk_inner, k
            {
                auto tensorDkIn = inQue2.AllocTensor<DataType>();
                auto tensorKIn = tensorDkIn[half_dkSize * 2];
                DataCopy(tensorDkIn[half_dkSize], gmMm5[mm5Offset + half_dkSize], second_dkSize);
                DataCopy(tensorKIn[half_dkSize], gmK[kOffset + half_dkSize], second_dkSize);
                inQue2.EnQue(tensorDkIn);
            }
            {
                auto tensorDkInFp16 = inQue2.DeQue<DataType>();
                auto tensorDkFp32 = tensorDkInFp16.template ReinterpretCast<float>();
                auto tensorKInFp16 = tensorDkInFp16[half_dkSize * 2];
                auto tensorKFp32 = tensorKInFp16.template ReinterpretCast<float>();

                Cast(tensorDkFp32, tensorDkInFp16[half_dkSize], RoundMode::CAST_NONE, second_dkSize);
                Cast(tensorKFp32, tensorKInFp16[half_dkSize], RoundMode::CAST_NONE, second_dkSize);
                PipeBarrier<PIPE_V>();

                // dk_state = dk_inner * exp(-g + g_last)[:,None]
                Brcb(tensorDgFinal, tensorGFp32[half_BT], CEIL_DIV(second_BT, 8), {1, 8});
                PipeBarrier<PIPE_V>();
                AscendC::Mul(tensorDkFp32, tensorDkFp32, tensorDgFinal, CAL_NUM_FLOAT, second_BT, {1, 1, 0, 16, 16, 1});
                AscendC::Mul(tensorDkFp32[CAL_NUM_FLOAT], tensorDkFp32[CAL_NUM_FLOAT], tensorDgFinal, CAL_NUM_FLOAT,
                                second_BT, {1, 1, 0, 16, 16, 1});
                PipeBarrier<PIPE_V>();

                if (h % n_ratio == 0) {
                    DataCopy(tensorDkSum[half_dkSize], tensorDkFp32, second_dkSize);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(tensorDkSum[half_dkSize], tensorDkSum[half_dkSize], tensorDkFp32, second_dkSize);
                    PipeBarrier<PIPE_V>();
                }

                // mul8 = dk_state * k
                Mul(tensorKFp32, tensorKFp32, tensorDkFp32, second_dkSize);
                PipeBarrier<PIPE_V>();

                // Add0.B = row_sum(dk_state * k) -> tensorGFp32[half_BT..]
                Add(tensorKFp32, tensorKFp32, tensorKFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, second_BT, {1, 1, 1, 16, 16, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorGFp32[half_BT], tensorKFp32, FP32_PER_REPEAT, second_BT, 1, 1, 16);
                PipeBarrier<PIPE_V>();

                inQue2.FreeTensor(tensorDkInFp16);
            }
        }

        Sub(tensorDgTmp, tensorDgTmp, tensorGFp32, BT); // Add.0 最终结果 (dg_B+dg_C+dg_D)
        PipeBarrier<PIPE_V>();
        // Sum0: [real_BT] -> [1]
        if (real_BT > FP32_PER_REPEAT) {
            Add(tensorGFp32, tensorGFp32, tensorGFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorDgFinal, tensorGFp32, FP32_PER_REPEAT, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        } else {
            WholeReduceSum(tensorDgFinal, tensorGFp32, real_BT, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
        Brcb(tensorDgTmpFp32, tensorDgFinal, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        Exp(tensorGTmpFp32, tensorGTmpFp32, 8);
        PipeBarrier<PIPE_V>();
        Mul(tensorDgTmpFp32[16], tensorGTmpFp32[16], tensorGTmpFp32, 8);
        PipeBarrier<PIPE_V>();
        Add(tensorDgTmpFp32[16], tensorDgTmpFp32[16], tensorDgTmpFp32, 8); // add4 = dg_last_term
        PipeBarrier<PIPE_V>();
        Brcb(tensorDgFinal, tensorDgTmpFp32[16], 1, {1, 8});
        PipeBarrier<PIPE_V>();
        uint64_t offset = (real_BT - 1) / 8 * 8;
        uint64_t mask[1] = {0};
        mask[0] = 1ULL << (real_BT - 1 - offset); // 仅最后一个位置加 dg_last_term
        Add(tensorDgTmp[offset], tensorDgTmp[offset], tensorDgFinal, mask, 1, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<GType, float>::value) {
            DataCopy(tensorDgOut, tensorDgTmp, BT);
        } else {
            Cast(tensorDgOut, tensorDgTmp, RoundMode::CAST_RINT, BT);
        }
        PipeBarrier<PIPE_V>();

        outQue2.EnQue(tensorDgOut);
        inQue3.FreeTensor(tensorGIn);
        inQue4.FreeTensor(tensorDgIn);
        {
            auto tensorDgOutDeq = outQue2.DeQue<GType>();
            DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
            outQue2.FreeTensor(tensorDgOutDeq);
        }

        // dk += mm7 (从 wsMm7 读取, dk_state 仍在 UB)
        {
            auto tensorMm7In = inQue2.AllocTensor<DataType>();
            DataCopy(tensorMm7In[actual_dkSize], gmMm7[mm5Offset], actual_dkSize);
            inQue2.EnQue(tensorMm7In);
        }
        {
            auto tensorMm7In = inQue2.DeQue<DataType>();
            auto tensorMm7Fp32 = tensorMm7In.template ReinterpretCast<float>();
            Cast(tensorMm7Fp32, tensorMm7In[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
            PipeBarrier<PIPE_V>();
            Add(tensorDkSum, tensorDkSum, tensorMm7Fp32, actual_dkSize);
            PipeBarrier<PIPE_V>();
            inQue2.FreeTensor(tensorMm7In);
        }
        if (h % n_ratio == n_ratio - 1) {
            auto tensorDkOut = outQue1.AllocTensor<DataType>();
            Cast(tensorDkOut, tensorDkSum, RoundMode::CAST_RINT, actual_dkSize);
            outQue1.EnQue(tensorDkOut);
            inQue1.FreeTensor(tensorDkSum);

            tensorDkOut = outQue1.DeQue<DataType>();
            DataCopy(gmDk[dkOutOffset], tensorDkOut, actual_dkSize);
            outQue1.FreeTensor(tensorDkOut);
        }
    }
    SetVecClear();
}

#endif // CHUNK_BWD_DQKWG_VECTOR_H
