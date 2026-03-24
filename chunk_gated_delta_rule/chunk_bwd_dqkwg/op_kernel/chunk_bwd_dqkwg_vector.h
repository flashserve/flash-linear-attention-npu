/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dqkwg_vector.h
 */

#ifndef CHUNK_BWD_DQKWG_VECTOR_H
#define CHUNK_BWD_DQKWG_VECTOR_H

#include "chunk_bwd_dqkwg_common.h"
#include "kernel_operator.h"

using namespace AscendC;

template <typename DataType, typename GType>
class ChunkBwdDqkwgVectorProcess {
public:
    __aicore__ inline ChunkBwdDqkwgVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
        GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices, GM_ADDR mask_a,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
        GM_ADDR workspace
    );
    
    __aicore__ inline void Init(const ChunkBwdDqkwgTilingData &tiling, TPipe *pipe_);
    __aicore__ inline void Process();
    
private:
    // 7 个 Part 的处理函数
    __aicore__ inline void ProcessPart1();  // dw = -dv @ h^T, dg_last 计算
    __aicore__ inline void ProcessPart2();  // 等待 mm5 完成
    __aicore__ inline void ProcessPart3();  // ds 处理, dg 部分计算
    __aicore__ inline void ProcessPart4();  // dq 处理, dg 累加
    __aicore__ inline void ProcessPart5();  // dk 处理, dg 最终计算
    __aicore__ inline void ProcessPart6();  // dq 累加
    __aicore__ inline void ProcessPart7();  // dk 累加
    
    // 辅助函数
    __aicore__ inline void ComputeExpScalar(float input, float &output);
    __aicore__ inline void ApplyLowerTriangularMask(LocalTensor<float> &tensor, uint32_t size);
    __aicore__ inline void ReduceSumX(LocalTensor<float> &src, LocalTensor<float> &dst,
                                      uint32_t rows, uint32_t cols, int axis);
    
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
    GM_ADDR ptrMaskA;
    GM_ADDR ptrDq;
    GM_ADDR ptrDk;
    GM_ADDR ptrDw;
    GM_ADDR ptrDg;
    GM_ADDR ptrWorkspace;
    
    // Tiling 参数
    uint64_t B;
    uint64_t H;
    uint64_t T;
    uint64_t K;
    uint64_t V;
    uint64_t BT;
    uint64_t numChunks;
    float scale;
    int isVarLen;
    
    // Workspace 偏移
    uint64_t wsDwOffset;
    uint64_t wsDgLastOffset;
    uint64_t wsMm5Offset;
    uint64_t wsDsTempOffset;
    uint64_t wsMm6Offset;
    uint64_t wsMm7Offset;
    uint64_t wsMul1Offset;
    int BUFFER_NUM = 1;
    
    // Pipeline
    TPipe *pipe = nullptr;
    
    // Global Tensors
    GlobalTensor<DataType> gmQ, gmK, gmV, gmDo, gmH, gmDh, gmDv;
    GlobalTensor<DataType> gmDq, gmDk, gmDw;
    GlobalTensor<GType> gmG, gmDg;
    GlobalTensor<DataType> gmWorkspace;
    GlobalTensor<float> gmDgLast;
    GlobalTensor<DataType> gmMm5, gmDsTemp, gmMul1, gmMm6, gmMm7;
    
    // Queues (用于流水)
    TQue<TPosition::VECIN, 2> inQue1;
    TQue<TPosition::VECIN, 2> inQue2;
    TQue<TPosition::VECIN, 2> inQue3;
    TQue<TPosition::VECIN, 2> inQue4;  //用于Add0累加
    TQue<TPosition::VECOUT, 2> outQue1;
    TQue<TPosition::VECOUT, 2> outQue2;

    
    // Calc Buffers (UB 空间)
    TBuf<TPosition::VECCALC> calcBuf1;  // 主计算缓冲区 (fp32)
    TBuf<TPosition::VECCALC> calcBuf2;  // 辅助计算缓冲区 (fp32)
    TBuf<TPosition::VECCALC> calcBuf3;  // Exp 缓冲区
    TBuf<TPosition::VECCALC> calcBuf4;  // 中间结果
    TBuf<TPosition::VECCALC> gBuf;      // g 值缓冲区
    TBuf<TPosition::VECCALC> dgBuf;     // dg 值缓冲区
    
    // UB 空间常量
    static constexpr uint32_t UB_BLOCK_SIZE = 32;
    static constexpr uint32_t FP32_ELEMENTS_PER_BLOCK = 8;
    static constexpr uint32_t FP16_ELEMENTS_PER_BLOCK = 16;
};

// ============== 构造函数 ==============
template <typename DataType, typename GType>
__aicore__ inline ChunkBwdDqkwgVectorProcess<DataType, GType>::ChunkBwdDqkwgVectorProcess(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h,
    GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices, GM_ADDR mask_a,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg,
    GM_ADDR workspace
) : ptrQ(q), ptrK(k), ptrV(v), ptrG(g), ptrH(h),
    ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLen(cu_seqlen), ptrChunkIndices(chunk_indices), ptrMaskA(mask_a),
    ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrDg(dg),
    ptrWorkspace(workspace) {}

// ============== 初始化 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Init(const ChunkBwdDqkwgTilingData &tiling, TPipe *pipe_) {
    pipe = pipe_;

    scale = tiling.scale;
    B = tiling.B;
    H = tiling.H;
    T = tiling.T;
    K = tiling.K;
    V = tiling.V;
    BT = tiling.BT;
    numChunks = tiling.numChunks;
    wsDgLastOffset = tiling.wsDgLastOffset;
    wsMm5Offset = tiling.wsMm5Offset;
    wsDsTempOffset = tiling.wsDsTempOffset;
    wsMm6Offset = tiling.wsMm6Offset;
    wsMm7Offset = tiling.wsMm7Offset;
    wsMul1Offset = tiling.wsMul1Offset;
    uint64_t dgLastSize = tiling.dgLastSize;
    isVarLen = tiling.isVarLen;

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

    gmWorkspace.SetGlobalBuffer((__gm__ DataType *)ptrWorkspace);
    gmDgLast.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t*)ptrWorkspace + wsDgLastOffset));     //中间结果使用float

    gmMm5.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t*)ptrWorkspace + wsMm5Offset));
    gmDsTemp.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t*)ptrWorkspace + wsDsTempOffset));
    gmMm6.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t*)ptrWorkspace + wsMm6Offset));

    gmMm7.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t*)ptrWorkspace + wsMm7Offset));
    gmMul1.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t*)ptrWorkspace + wsMul1Offset));
}

// ============== 主处理函数 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Process() {
    // Part 1: dw 和 dg_last 计算
    ProcessPart1();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // Part 2: 等待 mm5 (q @ k^T) 完成
    ProcessPart2();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // Part 3: ds 处理和 dg 部分计算
    ProcessPart3();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // Part 4: dq 处理
    ProcessPart4();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // Part 5: dk 处理和 dg 最终计算
    ProcessPart5();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // Part 6: dq 累加
    ProcessPart6();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // Part 7: dk 累加
    ProcessPart7();
}

// ============== Part 1: dw 和 dg_last 计算 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessPart1() {
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNum = GetBlockNum();
    // printf("coreIdx %d, coreNum %d\n", coreIdx, coreNum);
    uint32_t coreLoops = B * numChunks;
    
    // 分配 UB 空间
    // Part 1 的 Vector 部分主要处理:
    // 1. h * dh 的逐元素乘法和求和 -> dg_last
    // 2. dw 的负号处理
    
    const uint32_t hDhSize = K * V;  // h 和 dh 的大小
    const uint32_t dwSize = BT * K;
    uint32_t hDhSize_sub = hDhSize;
    uint32_t BT_sub = BT;
    uint32_t BT_sub_offset = 0;

    uint32_t dwSize_sub = BT_sub * K;
    uint32_t vecTaskIdx = 0;

    // 初始化 buffers
    pipe->InitBuffer(inQue1, 1, hDhSize_sub * sizeof(float));
    pipe->InitBuffer(inQue2, 1, hDhSize_sub * sizeof(float));   // hDhSize >= dwSize
    pipe->InitBuffer(outQue1, BUFFER_NUM, sizeof(float) * 8);  // dg_last (对齐到 32 字节)
    pipe->InitBuffer(outQue2, BUFFER_NUM, dwSize * sizeof(DataType));

    // 发送同步信号给 Cube
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    uint32_t bos = 0;
    uint32_t eos = 0;
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;
        GetChunkOffset(ptrCuSeqLen, ptrChunkIndices, B, H, T, BT, loopIdx, bos, eos);
        BT_sub = eos-bos;
        dwSize_sub = BT_sub * K;
        
        for (uint32_t h = 0; h < H; h++) {
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
                continue;
            }
            // 计算偏移
            // h, dh: [B, H, num_chunks, K, V]
            uint64_t hOffset = ((bIdx * H + h) * numChunks + chunkIdx) * K * V;
            // dw (workspace): 按 [B, H, T, K] 布局
            uint64_t dwOffset = (h * T + bos) * K;
            // dg_last: [B, H, num_chunks]
            uint64_t dgLastOffset = (bIdx * H + h) * numChunks + chunkIdx;

            // ========== 计算 dg_last = sum(h * dh) ==========
            float dg_last_sum = 0.0f;
            // 等待 Cube 信号 (dw Cube 计算)
            

            // CopyIn: h 和 dh
            {
                auto tensorHIn = inQue1.AllocTensor<DataType>();
                auto tensorDhIn = inQue2.AllocTensor<DataType>();
                DataCopy(tensorHIn[hDhSize_sub], gmH[hOffset], hDhSize_sub);
                DataCopy(tensorDhIn[hDhSize_sub], gmDh[hOffset], hDhSize_sub);
                inQue1.EnQue(tensorHIn);
                inQue2.EnQue(tensorDhIn);
            }

            // Compute: h * dh -> reduceSum
            {
                auto tensorHIn = inQue1.DeQue<DataType>();
                auto tensorCalcH = tensorHIn.template ReinterpretCast<float>();
                auto tensorDhIn = inQue2.DeQue<DataType>();
                auto tensorCalcDh = tensorDhIn.template ReinterpretCast<float>();
                auto tensorDgLastOut = outQue1.AllocTensor<float>();

                // Cast to fp32 (bf16 不支持直接 Mul)
                Cast(tensorCalcH, tensorHIn[hDhSize_sub], RoundMode::CAST_NONE, hDhSize_sub);
                Cast(tensorCalcDh, tensorDhIn[hDhSize_sub], RoundMode::CAST_NONE, hDhSize_sub);
                PipeBarrier<PIPE_V>();
                
                // 逐元素乘法
                Mul(tensorCalcH, tensorCalcH, tensorCalcDh, hDhSize_sub);
                PipeBarrier<PIPE_V>();  
                // 求和 (使用 ReduceSum)
                // 注: 需要将结果累加到一个标量
                
                // 简化处理: 分块求和
                LocalTensor<float> tensorSum = tensorDgLastOut;
                
                // 使用向量求和
                uint32_t reduceLen = hDhSize_sub;
                WholeReduceSum(tensorCalcDh, tensorCalcH,      // tensorCalcH: [K,V] = [128,128]
                                64, 256-8, 1, 1, 8);
                WholeReduceSum(tensorCalcDh[128*2-8], tensorCalcH[128*128-8*64],      // tensorCalcH: [K,V] = [128,128]
                                64, 8, 1, 1, 8);
                // tensorCalcDh: [128,2]
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorCalcH, tensorCalcDh,      // tensorCalcH: [K,V] = [128,128]
                                64, 128 * 128 / 64 / 64, 1, 1, 8);
                // tensorCalcH: [2,2]
                PipeBarrier<PIPE_V>();  
                WholeReduceSum(tensorSum, tensorCalcH,      // tensorCalcH: [K,V] = [128,128]
                                4, 1, 1, 1, 8);

                inQue1.FreeTensor(tensorHIn);
                inQue2.FreeTensor(tensorDhIn);
                outQue1.EnQue(tensorDgLastOut);
            }
            
            {
                auto tensorDgLastOut = outQue1.DeQue<float>();

                DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = 1*sizeof(float);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                DataCopyPad(gmDgLast[dgLastOffset], tensorDgLastOut,dataCopyParams);

                outQue1.FreeTensor(tensorDgLastOut);
            }

            // ========== 处理 dw: 取负号 ==========
            // Cube 计算的是 dv @ h^T, 需要乘以 -1
            // 从 workspace 读取 dw, 乘以 -1, 写回最终输出
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            // CopyIn: dw from workspace
            {
                auto tensorDwIn = inQue2.AllocTensor<DataType>();
                DataCopy(tensorDwIn[dwSize_sub], gmDw[dwOffset + BT_sub_offset * K], dwSize_sub);
                inQue2.EnQue(tensorDwIn);
            }

            // Compute: -dw
            {
                auto tensorDwIn = inQue2.DeQue<DataType>();
                auto tensorCalcDw = tensorDwIn.template ReinterpretCast<float>();
                auto tensorDwOut = outQue2.AllocTensor<DataType>();
                
                // Cast to fp32, 乘以 -1, cast back
                Cast(tensorCalcDw, tensorDwIn[dwSize_sub], RoundMode::CAST_NONE, dwSize_sub);
                PipeBarrier<PIPE_V>();
                Muls(tensorCalcDw, tensorCalcDw, -1.0f, dwSize_sub);
                PipeBarrier<PIPE_V>();
                Cast(tensorDwOut, tensorCalcDw, RoundMode::CAST_RINT, dwSize_sub);
                inQue2.FreeTensor(tensorDwIn);
                outQue2.EnQue(tensorDwOut);
            }
            
            // CopyOut: dw to final output
            {
                auto tensorDwOut = outQue2.DeQue<DataType>();
                DataCopy(gmDw[dwOffset + BT_sub_offset * K], tensorDwOut, dwSize_sub);

                outQue2.FreeTensor(tensorDwOut);
            }

            // 通知 Cube 继续
            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
        }
    }

}

// ============== Part 2: 等待 mm5 完成 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessPart2() {
    // Part 2 MUL1 + and(m_A)
    constexpr int32_t CAL_NUM_FLOAT = 64; // API一次能处理256B，能计算64个float元素
    constexpr int32_t BLOCK_SIZE = 32; // API一次能处理256B，能计算64个float元素
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNum = GetBlockNum();
    uint32_t coreLoops = B * numChunks;
    uint32_t real_BT = BT;

    uint32_t BT_sub = real_BT;
    uint32_t BT_sub_start = 0;
    uint32_t BT_sub_end = BT_sub;

    uint32_t dsSize_sub = BT_sub * BT;
    uint32_t dsSize_sub_offset = 0;
    const uint32_t gSize = BT;
    
    // 初始化 buffers
    
    pipe->InitBuffer(inQue3, 2, gSize * sizeof(float));        // g values
    pipe->InitBuffer(outQue1, 2, dsSize_sub * sizeof(float) / 2);   // ds_temp output   32K/8K

    pipe->InitBuffer(calcBuf1, BT * 8 * sizeof(float));             // g in fp32
    pipe->InitBuffer(calcBuf2, BT * sizeof(float));            // g temp [BT,1]
    pipe->InitBuffer(calcBuf3, BT * sizeof(float));            // g temp [1,BT]
    pipe->InitBuffer(calcBuf4, BLOCK_SIZE);

    auto tensorBrcbTemp = calcBuf1.Get<float>();
    auto tensorGFp32Left = calcBuf2.Get<float>();
    auto tensorGFp32Right = calcBuf3.Get<float>();
    auto tensorZeroFp32 = calcBuf4.template Get<float>();
    
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t bos_orig = 0;
    uint32_t eos_orig = 0;
    // 发送同步信号

    //初始化zero
    AscendC::Duplicate<float>(tensorZeroFp32, float(0.0), BLOCK_SIZE / sizeof(float));
    PipeBarrier<PIPE_V>();
    //搬入m_A

    const uint32_t maskASize = 64*64*sizeof(float);//BT * BT / 8 * sizeof(uint8_t);
    pipe->InitBuffer(inQue1, 1, maskASize);    // m_A from input
    auto tensorMaskA = inQue1.AllocTensor<float>();
    for(int i=0;i<64;i++){
        Duplicate(tensorMaskA[i*64],static_cast<float>(0),64);
        PipeBarrier<PIPE_V>();
        Duplicate(tensorMaskA[i*64],static_cast<float>(1),i+1);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GetChunkOffset(ptrCuSeqLen, ptrChunkIndices, B, H, T, BT, loopIdx, bos, eos);
        bos_orig = bos;
        eos_orig = eos;
        uint32_t vec_core_0_length = (eos_orig - bos_orig) >= (BT / 2) ? (BT / 2) : (eos_orig - bos_orig);
        uint32_t vec_core_1_length = (eos_orig - bos_orig) >= (BT / 2) ? (eos_orig - bos_orig - (BT / 2)) : 0;
        if (GetSubBlockIdx() == 0) {        // BT: 0..BT_sub_end
            BT_sub_start = 0;
            BT_sub_end = vec_core_0_length;         // [0, vec_core_0_length)
            eos = bos + vec_core_0_length;
        } else {       // BT: BT_sub_end+1..eos
            BT_sub_start = vec_core_0_length;
            BT_sub_end = eos_orig - bos_orig;
            bos = bos + vec_core_0_length;
        }
        
        uint32_t real_BT= eos-bos;
        uint32_t real_BT_align = ALIGN_UP((eos-bos), 16);
        dsSize_sub = (eos-bos) * BT;
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;

        for (uint32_t h = 0; h < H; h++) {
            if (real_BT == 0) {       //sub core get 0 token
                continue;
            }

            // 偏移计算
            // g: [B, H, T]
            uint64_t gOffset = (h * T + bos_orig);
            // ds, mm5, ds_temp: [B, H, T, BT]
            uint64_t dsOffset = (h * T + bos) * BT;
            // CopyIn: g
            {
                auto tensorGIn = inQue3.AllocTensor<GType>();
                DataCopy(tensorGIn, gmG[gOffset], gSize);
                inQue3.EnQue(tensorGIn);
            }

            // Compute MUL1
            {
                auto tensorGIn = inQue3.DeQue<GType>();
                auto tensorDsTempOut = outQue1.AllocTensor<float>();

                // g 可能已经是 fp32
                if constexpr (std::is_same<GType, float>::value) {
                    DataCopy(tensorGFp32Left, tensorGIn, gSize);
                    DataCopy(tensorGFp32Right, tensorGIn, gSize);
                } else {
                    Cast(tensorGFp32Left, tensorGIn, RoundMode::CAST_NONE, gSize);
                    Cast(tensorGFp32Right, tensorGIn, RoundMode::CAST_NONE, gSize);
                }
                PipeBarrier<PIPE_V>();

                Muls(tensorGFp32Right, tensorGFp32Right, static_cast<float>(-1), gSize);
                PipeBarrier<PIPE_V>();

                Brcb(tensorBrcbTemp, tensorGFp32Left[BT_sub_start], CEIL_DIV(real_BT, 8), {1, 8}); // Brcb处理数据个数需要8对齐 [BT,8]
                PipeBarrier<PIPE_V>();

                // copy tensorGFp32Right  chunkLen / 2行
                if (BT == 64) {
                    AscendC::Copy(tensorDsTempOut, tensorGFp32Right, CAL_NUM_FLOAT, real_BT, {1, 1, 8, 0});

                    PipeBarrier<PIPE_V>();
                    AscendC::Add(tensorDsTempOut, tensorDsTempOut, tensorBrcbTemp, CAL_NUM_FLOAT, real_BT,
                                {1, 1, 0, 8, 8, 1});

                    PipeBarrier<PIPE_V>();
                    Mins(tensorDsTempOut, tensorDsTempOut, static_cast<float>(0.0), real_BT * BT);

                    PipeBarrier<PIPE_V>();
                    Exp(tensorDsTempOut, tensorDsTempOut, real_BT * BT);
                    PipeBarrier<PIPE_V>();

                } else {
                    AscendC::Copy(tensorDsTempOut, tensorGFp32Right, CAL_NUM_FLOAT, real_BT, {1, 1, 16, 0});
                    PipeBarrier<PIPE_V>();
                    AscendC::Copy(tensorDsTempOut[CAL_NUM_FLOAT], tensorGFp32Right[CAL_NUM_FLOAT], CAL_NUM_FLOAT,
                                real_BT, {1, 1, 16, 0});
                    PipeBarrier<PIPE_V>();
                    AscendC::Add(tensorDsTempOut, tensorDsTempOut, tensorBrcbTemp, CAL_NUM_FLOAT, real_BT,
                                {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                    AscendC::Add(tensorDsTempOut[CAL_NUM_FLOAT], tensorDsTempOut[CAL_NUM_FLOAT], tensorBrcbTemp,
                                CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                    Mins(tensorDsTempOut, tensorDsTempOut, static_cast<float>(0.0), real_BT * BT);
                    PipeBarrier<PIPE_V>();
                    Exp(tensorDsTempOut, tensorDsTempOut, real_BT * BT);
                }

                PipeBarrier<PIPE_V>();

                if(BT==64) {    //TODO： MASK
                    Mul(tensorDsTempOut,tensorDsTempOut,tensorMaskA[BT_sub_start*64],32*64);
                    PipeBarrier<PIPE_V>();
                } else {
                    BinaryRepeatParams binaryRepeatParams{1,1,1,16,16,8};
                    UnaryRepeatParams unaryRepeatParams{1,1,16,8};
                    PipeBarrier<PIPE_V>();
                    if (BT_sub_start == 0) {    // vec 0 : 0..64
                        Mul(tensorDsTempOut,tensorDsTempOut,tensorMaskA,64,64,binaryRepeatParams);
                        PipeBarrier<PIPE_V>();
                        Muls(tensorDsTempOut[64],tensorDsTempOut[64],static_cast<float>(0),64,64,unaryRepeatParams);
                        PipeBarrier<PIPE_V>();
                    }
                    else {      // vec 0 : 64..128
                        Mul(tensorDsTempOut[64],tensorDsTempOut[64],tensorMaskA,64,64,binaryRepeatParams);
                        PipeBarrier<PIPE_V>();
                    }

                }

                AscendC::Muls(tensorDsTempOut, tensorDsTempOut, static_cast<float>(scale), real_BT * BT);
                PipeBarrier<PIPE_V>();

                //Ds是 fp16/bf16
                Cast(tensorDsTempOut.template ReinterpretCast<DataType>(), tensorDsTempOut, RoundMode::CAST_RINT, real_BT * BT);

                inQue3.FreeTensor(tensorGIn);
                outQue1.EnQue(tensorDsTempOut);
            }

            // CopyOut
            {
                auto tensorDsTempOut = outQue1.DeQue<float>();
                DataCopy(gmMul1[dsOffset], tensorDsTempOut.template ReinterpretCast<DataType>(), real_BT * BT);

                outQue1.FreeTensor(tensorDsTempOut);
            }
        }
    }
    inQue1.FreeTensor<float>(tensorMaskA);
}

// ============== Part 3: ds 处理和 dg 部分计算 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessPart3() {
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNum = GetBlockNum();
    uint32_t coreLoops = B * numChunks;
    uint32_t real_BT = BT; // TODO：如果BT不对齐

    uint32_t BT_sub = real_BT;
    uint32_t BT_sub_start = 0;
    uint32_t BT_sub_end = BT_sub;


    uint32_t dsSize_sub = BT_sub * BT;
    uint32_t dsSize_sub_offset = 0;
    const uint32_t gSize = BT;

    // 初始化 buffers
    pipe->InitBuffer(inQue1, BUFFER_NUM, dsSize_sub * sizeof(float));    // ds from Cube
    pipe->InitBuffer(inQue2, BUFFER_NUM, dsSize_sub * sizeof(float));    // mm5/mul1 from workspace
    pipe->InitBuffer(outQue1, BUFFER_NUM, dsSize_sub * sizeof(DataType));   // ds_temp output   32K/8K
    pipe->InitBuffer(outQue2, BUFFER_NUM, gSize * sizeof(float));       // dg output

    pipe->InitBuffer(calcBuf4, gSize * sizeof(float));        // m_A
    pipe->InitBuffer(gBuf, gSize * sizeof(float));             // g in fp32
    pipe->InitBuffer(dgBuf, gSize * sizeof(float));            // dg temp

    auto tensorMaskA = calcBuf4.Get<float>();
    auto tensorGFp32 = gBuf.Get<float>();
    auto tensorDgTemp = dgBuf.Get<float>();

    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t vecTaskIdx = 0;
    // 发送同步信号
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GetChunkOffset(ptrCuSeqLen, ptrChunkIndices, B, H, T, BT, loopIdx, bos, eos);
        BT_sub_end = eos-bos;
        uint32_t real_BT= eos-bos;
        dsSize_sub = (eos-bos) * BT;
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;

        for (uint32_t h = 0; h < H; h++) {
            // if (GetSubBlockIdx() == 1) {
            //     CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            //     CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
            //     continue;
            // }
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
                continue;
            }

            // 偏移计算
            // g: [B, H, T]
            uint64_t gOffset = (h * T + bos);
            // ds, mm5, ds_temp: [B, H, T, BT]
            uint64_t dsOffset = (h * T + bos) * BT;

            // dg: [B, H, T]
            uint64_t dgOffset = gOffset;

            // 等待 Cube 完成 ds 计算
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);

            // CopyIn: ds (Cube output), g
            {
                auto tensorDsIn = inQue1.AllocTensor<DataType>();
                auto tensorMul1In = inQue2.AllocTensor<DataType>();

                DataCopy(tensorDsIn[BT * BT], gmDsTemp[dsOffset + dsSize_sub_offset], dsSize_sub);
                DataCopy(tensorMul1In[BT * BT], gmMul1[dsOffset + dsSize_sub_offset], dsSize_sub);

                inQue1.EnQue(tensorDsIn);
                inQue2.EnQue(tensorMul1In);
            }

            // Compute MUL1
            {
                auto tensorDsInFp16 = inQue1.DeQue<DataType>();
                auto tensorDsInFp32 = tensorDsInFp16.template ReinterpretCast<float>();

                auto tensorDsTempOut = outQue1.AllocTensor<DataType>();

                auto tensorMul1InFp16 = inQue2.DeQue<DataType>();
                auto tensorMul1InFp32 = tensorMul1InFp16.template ReinterpretCast<float>();
                auto tensorDgOut = outQue2.AllocTensor<float>();

                // Cast to fp32
                Cast(tensorDsInFp32, tensorDsInFp16[BT * BT], RoundMode::CAST_NONE, dsSize_sub);
                PipeBarrier<PIPE_V>();

                Cast(tensorMul1InFp32, tensorMul1InFp16[BT * BT], RoundMode::CAST_NONE, dsSize_sub);
                PipeBarrier<PIPE_V>();
                // b_ds_temp = b_ds * mul1 (已经应用了掩码)
                Mul(tensorDsInFp32, tensorDsInFp32, tensorMul1InFp32, dsSize_sub);
                inQue2.FreeTensor(tensorMul1InFp32);

                //搬入MM5,复用Mul1空间
                auto tensorMm5InFp16Tmp = inQue2.AllocTensor<DataType>();
                DataCopy(tensorMm5InFp16Tmp[BT * BT], gmMm5[dsOffset + dsSize_sub_offset], dsSize_sub);
                inQue2.EnQue(tensorMm5InFp16Tmp);
                auto tensorMm5InFp16 = inQue2.DeQue<DataType>();

                auto tensorMm5InFp32 = tensorMm5InFp16.template ReinterpretCast<float>();
                Cast(tensorMm5InFp32, tensorMm5InFp16[BT * BT], RoundMode::CAST_NONE, dsSize_sub);
                PipeBarrier<PIPE_V>();
                // Calcute ds_temp * mm5 and after

                Mul(tensorMm5InFp32, tensorDsInFp32, tensorMm5InFp32, dsSize_sub); // b_ds2 = b_ds_temp * mm5

                PipeBarrier<PIPE_V>();
                Cast(tensorDsTempOut, tensorDsInFp32, RoundMode::CAST_RINT, dsSize_sub);  //ds_tmp -> fp16, tensorDsFp32已经空闲
                PipeBarrier<PIPE_V>();

                // axis=1: 对每行求和 -> [BT] +Add0.C
                Duplicate(tensorDgOut, static_cast<float>(0.0), BT);
                PipeBarrier<PIPE_V>();

                // reducesum
                uint64_t wholeReduceSumCnt = CeilDiv(real_BT, FP32_PER_REPEAT);
                uint32_t remainCnt = real_BT % FP32_PER_REPEAT;
                if(remainCnt > 0) {
                    uint32_t DuplicateOffset = wholeReduceSumCnt * FP32_PER_REPEAT - FP32_PER_REPEAT;
                    uint64_t mask[1] = {0xffffffffffffffff};
                    mask[0] <<= remainCnt;
                    for (uint32_t row = BT_sub_start; row < BT_sub_end; row++) {
                        Duplicate(tensorMm5InFp32[row * BT + DuplicateOffset], 0.0f, mask, 1, 1, 8);
                    }
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t i = BT_sub_start; i < BT_sub_end; i++) {
                    WholeReduceSum(tensorDsInFp32[i * 8], tensorMm5InFp32[i * BT],
                                   FP32_PER_REPEAT, wholeReduceSumCnt, 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut, tensorDsInFp32, wholeReduceSumCnt, real_BT, 1, 1, 1);

                // axis=0: 对每列求和 -> [BT] -Add0.D
                Duplicate(tensorDgTemp, static_cast<float>(0.0), BT);
                PipeBarrier<PIPE_V>();

                for (uint32_t i = BT_sub_start; i < BT_sub_end; i++) {
                    Muls(tensorDgTemp, tensorMm5InFp32[(i - BT_sub_start) * BT], static_cast<float>(-1), BT);
                    PipeBarrier<PIPE_V>();

                    Add(tensorDgOut, tensorDgOut, tensorDgTemp, BT);
                    PipeBarrier<PIPE_V>();
                }

                PipeBarrier<PIPE_V>();

                // 保存 tensorDgOut 供后续 Part 使用
                if constexpr (std::is_same<GType, float>::value) {
                    // pass
                } else {
                    Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, gSize);
                }

                inQue1.FreeTensor(tensorDsInFp16);
                inQue2.FreeTensor(tensorMm5InFp16);

                outQue1.EnQue(tensorDsTempOut);
                outQue2.EnQue(tensorDgOut);
            }
            
            // CopyOut
            {
                auto tensorDsTempOut = outQue1.DeQue<DataType>();
                auto tensorDgOut = outQue2.DeQue<float>();

                DataCopy(gmDsTemp[dsOffset + dsSize_sub_offset], tensorDsTempOut, dsSize_sub);

                DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = real_BT * sizeof(GType);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                // dg 写入最终输出
                if constexpr (std::is_same<GType, float>::value) {
                    DataCopyPad(gmDg[dgOffset], tensorDgOut,dataCopyParams);
                } else {
                    // 需要 cast
                    DataCopyPad(gmDg[dgOffset], tensorDgOut.template ReinterpretCast<GType>(), dataCopyParams);
                }

                outQue1.FreeTensor(tensorDsTempOut);
                outQue2.FreeTensor(tensorDgOut);
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
        }

    }

}

// ============== Part 4: dq 处理 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessPart4() {
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNum = GetBlockNum();
    uint32_t coreLoops = B * numChunks;
    
    uint32_t dqSize = BT * K;
    uint32_t real_BT = BT;


    uint32_t BT_sub = real_BT;
    uint32_t BT_sub_start = 0;
    uint32_t BT_sub_end = BT_sub;
    
    uint32_t dqSize_sub = BT_sub * K;
    uint32_t dqSize_sub_offset = BT_sub_start * K;
    const uint32_t gSize = BT;
    // 初始化 buffers
    pipe->InitBuffer(inQue1, BUFFER_NUM, dqSize_sub * sizeof(float));    // dq from Cube   //64K
    pipe->InitBuffer(inQue2, BUFFER_NUM, dqSize_sub * sizeof(float));    // q
    pipe->InitBuffer(inQue3, BUFFER_NUM, gSize * sizeof(GType));        // g
    pipe->InitBuffer(inQue4, BUFFER_NUM, gSize * sizeof(GType));        // g             
    pipe->InitBuffer(outQue1, BUFFER_NUM, dqSize_sub * sizeof(DataType));   // dq output
    pipe->InitBuffer(outQue2, BUFFER_NUM, gSize * sizeof(float));       // dg partial

    pipe->InitBuffer(calcBuf3, gSize * (8) * sizeof(float));        //第一次reducesum结果：[BT, 8]
    pipe->InitBuffer(gBuf, gSize * sizeof(float));
    pipe->InitBuffer(dgBuf, gSize * sizeof(float));
    
    auto tensorShareTmpFp32 = calcBuf3.Get<float>();
    auto tensorGFp32 = gBuf.Get<float>();
    auto tensorDgAdd = dgBuf.Get<float>();

    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GetChunkOffset(ptrCuSeqLen, ptrChunkIndices, B, H, T,
                        BT, loopIdx, bos, eos);
        uint32_t actual_chunk_len = eos-bos;
        dqSize_sub = actual_chunk_len * K;
        BT_sub_end = actual_chunk_len;
        BT_sub = actual_chunk_len;
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;
        
        for (uint32_t h = 0; h < H; h++) {
            // if (GetSubBlockIdx() == 1) {
            //     CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            //     CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
            //     continue;
            // }
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
                continue;
            }
            uint64_t qkOffset = (h * T + bos) * K;
            uint64_t gOffset = (h * T + bos);
            
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            // CopyIn
            {
                auto tensorDqIn = inQue1.AllocTensor<DataType>();
                auto tensorQIn = inQue2.AllocTensor<DataType>();
                auto tensorGIn = inQue3.AllocTensor<GType>();
                auto tensorDgIn = inQue4.AllocTensor<GType>();
                
                DataCopy(tensorDqIn[dqSize_sub], gmDq[qkOffset + dqSize_sub_offset], dqSize_sub);

                DataCopy(tensorQIn[dqSize_sub], gmQ[qkOffset + dqSize_sub_offset], dqSize_sub);
                DataCopy(tensorGIn, gmG[gOffset], gSize);
                DataCopy(tensorDgIn, gmDg[gOffset], gSize);

                inQue1.EnQue(tensorDqIn);
                inQue2.EnQue(tensorQIn);
                inQue3.EnQue(tensorGIn);
                inQue4.EnQue(tensorDgIn);
            }
            
            // Compute
            {
                auto tensorDqInFp16 = inQue1.DeQue<DataType>();
                auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
                auto tensorQInFp16 = inQue2.DeQue<DataType>();
                auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();
                auto tensorGIn = inQue3.DeQue<GType>();
                auto tensorDgIn = inQue4.DeQue<GType>();
                auto tensorDqOut = outQue1.AllocTensor<DataType>();
                auto tensorDgOut = outQue2.AllocTensor<float>();

                // Cast
                Cast(tensorDqInFp32, tensorDqInFp16[dqSize_sub], RoundMode::CAST_NONE, dqSize_sub);
                PipeBarrier<PIPE_V>();

                Cast(tensorQInFp32, tensorQInFp16[dqSize_sub], RoundMode::CAST_NONE, dqSize_sub);
                PipeBarrier<PIPE_V>();
                Duplicate(tensorGFp32, static_cast<float>(0), gSize);
                PipeBarrier<PIPE_V>();
                if constexpr (std::is_same<GType, float>::value) {
                    DataCopy(tensorGFp32, tensorGIn, gSize);
                } else {
                    Cast(tensorGFp32, tensorGIn, RoundMode::CAST_NONE, gSize);
                }
                PipeBarrier<PIPE_V>();

                // mul3 = exp(g) * scale
                // b_dq_temp = b_dq * mul3 (broadcast along K dimension)
                Exp(tensorGFp32, tensorGFp32, gSize);
                PipeBarrier<PIPE_V>();

                Muls(tensorGFp32, tensorGFp32, static_cast<float>(scale), gSize);       //mul3 = exp(g) * scale
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0x1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0x1);

                for (uint32_t t = BT_sub_start; t < BT_sub_end; t++) {
                    Muls(tensorDqInFp32[(t - BT_sub_start)* K], tensorDqInFp32[(t - BT_sub_start)* K], tensorGFp32.GetValue(t), K);
                    PipeBarrier<PIPE_V>();
                }

                PipeBarrier<PIPE_V>();
                Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, dqSize_sub);
                PipeBarrier<PIPE_V>();
                // Add0.A = reduceSum((q * dq), axis=1)
                // axis=1: 对每行求和 -> [BT] +Add0.A
                Duplicate(tensorDgOut, static_cast<float>(0.0), BT);
                PipeBarrier<PIPE_V>();

                // reducesum
                uint64_t wholeReduceSumCnt = CeilDiv(K, FP32_PER_REPEAT);
                for (uint32_t i = BT_sub_start; i < BT_sub_end; i++) {
                    WholeReduceSum(tensorShareTmpFp32[i * 8], tensorQInFp32[i * K],
                                   FP32_PER_REPEAT, wholeReduceSumCnt, 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut, tensorShareTmpFp32, wholeReduceSumCnt, actual_chunk_len, 1, 1, 1);

                //累加tensorDgIn += + tensorDgOut
                PipeBarrier<PIPE_V>();

                if constexpr (std::is_same<GType, float>::value) {
                    DataCopy(tensorDgAdd, tensorDgIn, real_BT);
                } else {
                    Cast(tensorDgAdd, tensorDgIn, RoundMode::CAST_NONE, real_BT);
                }

                PipeBarrier<PIPE_V>();
                Add(tensorDgOut, tensorDgAdd, tensorDgOut, real_BT);

                PipeBarrier<PIPE_V>();

                if constexpr (std::is_same<GType, float>::value) {

                } else {
                    Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, gSize);
                }
                PipeBarrier<PIPE_V>();
                Cast(tensorDqOut, tensorDqInFp32, RoundMode::CAST_RINT, dqSize_sub);

                inQue1.FreeTensor(tensorDqInFp16);
                inQue2.FreeTensor(tensorQInFp16);
                inQue3.FreeTensor(tensorGIn);
                inQue4.FreeTensor(tensorDgIn);
                outQue1.EnQue(tensorDqOut);
                outQue2.EnQue(tensorDgOut);
            }
            // CopyOut: dq to final output, dg accumulate
            {
                auto tensorDqOut = outQue1.DeQue<DataType>();
                auto tensorDgOut = outQue2.DeQue<GType>();

                DataCopy(gmDq[qkOffset + dqSize_sub_offset], tensorDqOut, dqSize_sub);

                // 累加到 dg (读取现有值, 加上新值, 写回)
                // 简化: 直接累加 (需要在 Part 5 中处理最终值)
                // dg 写入最终输出
                DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = BT_sub * sizeof(GType);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                DataCopyPad(gmDg[gOffset + BT_sub_start], tensorDgOut,dataCopyParams);

                outQue1.FreeTensor(tensorDqOut);
                outQue2.FreeTensor(tensorDgOut);
            }
            
            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
        }
    }
}

// ============== Part 5: dk 处理和 dg 最终计算 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessPart5() {
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNum = GetBlockNum();
    uint32_t coreLoops = B * numChunks;
    
    uint32_t dkSize = BT * K;
    uint32_t gSize = BT;
    uint32_t real_BT = BT;
    
    // 初始化 buffers
    pipe->InitBuffer(inQue1, BUFFER_NUM, dkSize * sizeof(float));
    pipe->InitBuffer(inQue2, BUFFER_NUM, dkSize * sizeof(float));
    pipe->InitBuffer(inQue3, BUFFER_NUM, gSize * sizeof(GType));
    pipe->InitBuffer(inQue4, BUFFER_NUM, gSize * sizeof(GType));
    pipe->InitBuffer(outQue1, BUFFER_NUM, dkSize * sizeof(DataType));
    pipe->InitBuffer(outQue2, BUFFER_NUM, gSize * sizeof(GType));

    pipe->InitBuffer(calcBuf4, gSize * sizeof(float));
    pipe->InitBuffer(gBuf, gSize * sizeof(float));
    pipe->InitBuffer(dgBuf, gSize * 8 * sizeof(float));

    auto tensorGFp32 = gBuf.Get<float>();
    auto tensorDgFinal = dgBuf.Get<float>();
    auto tensorDgTmp = calcBuf4.Get<float>();
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t vecTaskIdx = 0;
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GetChunkOffset(ptrCuSeqLen, ptrChunkIndices, B, H, T, BT, loopIdx, bos, eos);
        uint32_t actual_chunk_len = eos - bos;
        real_BT = actual_chunk_len;
        dkSize = actual_chunk_len * K;
        uint32_t real_BT_aligned = (real_BT + 15) / 16 * 16;
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;
        
        for (uint32_t h = 0; h < H; h++) {
            // if (GetSubBlockIdx() == 1) {
            //     CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            //     CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
            //     continue;
            // }
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
                continue;
            }
            uint64_t kOffset = (h * T + bos) * K;
            uint64_t gOffset = (h * T + bos);
            uint64_t dgLastOffset = (bIdx * H + h) * numChunks + chunkIdx;
            
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);

            // CopyIn
            {
                auto tensorDkIn = inQue1.AllocTensor<DataType>();
                auto tensorKIn = inQue2.AllocTensor<DataType>();
                auto tensorGIn = inQue3.AllocTensor<GType>();
                auto tensorDgIn = inQue4.AllocTensor<GType>();
                
                DataCopy(tensorDkIn[dkSize], gmDk[kOffset], dkSize);
                DataCopy(tensorKIn[dkSize], gmK[kOffset], dkSize);
                DataCopy(tensorGIn, gmG[gOffset], gSize);
                DataCopy(tensorDgIn, gmDg[gOffset], gSize);

                inQue1.EnQue(tensorDkIn);
                inQue2.EnQue(tensorKIn);
                inQue3.EnQue(tensorGIn);
                inQue4.EnQue(tensorDgIn);
            }
            
            // Compute
            {
                auto tensorDkIn = inQue1.DeQue<DataType>();
                auto tensorDkFp32 = tensorDkIn.template ReinterpretCast<float>();
                auto tensorKIn = inQue2.DeQue<DataType>();
                auto tensorKFp32 = tensorKIn.template ReinterpretCast<float>();
                auto tensorGIn = inQue3.DeQue<GType>();
                auto tensorDgIn = inQue4.DeQue<GType>();
                auto tensorDkOut = outQue1.AllocTensor<DataType>();
                auto tensorDgOut = outQue2.AllocTensor<GType>();

                // Cast
                Cast(tensorDkFp32, tensorDkIn[dkSize], RoundMode::CAST_NONE, dkSize);
                PipeBarrier<PIPE_V>();

                Cast(tensorKFp32, tensorKIn[dkSize], RoundMode::CAST_NONE, dkSize);
                PipeBarrier<PIPE_V>();
                Duplicate(tensorGFp32, static_cast<float>(0), gSize);
                Duplicate(tensorDgTmp, static_cast<float>(0), gSize);
                PipeBarrier<PIPE_V>();
                if constexpr (std::is_same<GType, float>::value) {
                    DataCopy(tensorGFp32, tensorGIn, BT);
                    DataCopy(tensorDgTmp, tensorDgIn, BT);
                } else {
                    Cast(tensorGFp32, tensorGIn, RoundMode::CAST_NONE, real_BT_aligned);
                    Cast(tensorDgTmp, tensorDgIn, RoundMode::CAST_NONE, real_BT_aligned);
                }
                PipeBarrier<PIPE_V>();

                // 获取 g_last: chunk 内最后一个位置的 g 值
                // g_last = g[min(BT-1, T - chunk*BT - 1)]
                uint32_t lastIdx = actual_chunk_len - 1;

                AscendC::SetFlag<AscendC::HardEvent::V_S>(0x1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0x1);
                float gLast = tensorGFp32.GetValue(lastIdx);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0x2);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0x2);

                Muls(tensorGFp32, tensorGFp32, static_cast<float>(-1), real_BT_aligned);
                PipeBarrier<PIPE_V>();

                Adds(tensorGFp32, tensorGFp32, static_cast<float>(gLast), real_BT_aligned);
                PipeBarrier<PIPE_V>();

                Exp(tensorGFp32, tensorGFp32, real_BT_aligned);
                PipeBarrier<PIPE_V>();

                // 获取 dg_last
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0x1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0x1);
                float dgLast = gmDgLast.GetValue(dgLastOffset);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0x2);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0x2);

                // mul2 = exp(-g + g_last), 应用 m_t 掩码 (有效位置)
                // b_dk_temp = b_dk * mul2
                for (uint32_t t = 0; t < BT; t++) {
                    float mul5_value = tensorGFp32.GetValue(t);
                    Muls(tensorDkFp32[t * K], tensorDkFp32[t * K], mul5_value, K);
                    PipeBarrier<PIPE_V>();
                }

                Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, dkSize);      //dk -> fp16 tensorDkFp32
                PipeBarrier<PIPE_V>();

                Mul(tensorKFp32, tensorKFp32, tensorDkFp32, dkSize);    // mul8 = dk * k
                PipeBarrier<PIPE_V>();

                // Add0.B = (dk_temp * k).reduceSum(axis=1)
                // axis=1: 对每行求和 -> [BT] +Add0.B
                Duplicate(tensorGFp32, static_cast<float>(0.0), BT);
                PipeBarrier<PIPE_V>();

                // reducesum
                uint64_t wholeReduceSumCnt = CeilDiv(K, FP32_PER_REPEAT);
                for (uint32_t i = 0; i < actual_chunk_len; i++) {
                    WholeReduceSum(tensorDgFinal[i * 8], tensorKFp32[i * K],
                                   FP32_PER_REPEAT, wholeReduceSumCnt, 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorGFp32, tensorDgFinal, wholeReduceSumCnt, actual_chunk_len, 1, 1, 1);

                float totalSum = 0.0f;
                PipeBarrier<PIPE_V>();

                //Sum0: [actual_chunk_len] -> [1]
                uint64_t sum0SumCnt = CeilDiv(actual_chunk_len, FP32_PER_REPEAT);
                uint32_t remainCnt = actual_chunk_len % FP32_PER_REPEAT;
                if(remainCnt > 0) {
                    uint32_t DuplicateOffset = sum0SumCnt * FP32_PER_REPEAT - FP32_PER_REPEAT;
                    uint64_t mask[1] = {0xffffffffffffffff};
                    mask[0] <<= remainCnt;

                    Duplicate(tensorGFp32[(sum0SumCnt-1)*FP32_PER_REPEAT], 0.0f, mask, 1, 1, 8);

                    PipeBarrier<PIPE_V>();
                }
                WholeReduceSum(tensorDkFp32[0], tensorGFp32[0], FP32_PER_REPEAT, sum0SumCnt, 1, 1, 8);
                // PipeBarrier<PIPE_ALL>();
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgFinal[0], tensorDkFp32[0], sum0SumCnt, 1, 1, 1, 8);

                AscendC::SetFlag<AscendC::HardEvent::V_S>(0x1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0x1);
                totalSum = tensorDgFinal.GetValue(0);              //Add4.A = sum(reduceSum(Mul8))
                tensorDgFinal.SetValue(0, gLast);

                AscendC::SetFlag<AscendC::HardEvent::S_V>(0x2);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0x2);
                Exp(tensorDgFinal, tensorDgFinal, 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0x1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0x1);
                float add4 = tensorDgFinal.GetValue(0) * dgLast + totalSum;

                AscendC::SetFlag<AscendC::HardEvent::S_V>(0x2);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0x2);

                PipeBarrier<PIPE_V>();
                Muls(tensorGFp32, tensorGFp32, static_cast<float>(-1), BT);     //Add0.b = -1 * reduceSum
                PipeBarrier<PIPE_V>();

                Add(tensorGFp32, tensorGFp32, tensorDgTmp, BT); //Add.0 最终结果
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0x1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0x1);
                tensorGFp32.SetValue(real_BT - 1, tensorGFp32.GetValue(real_BT - 1) + add4);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0x2);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0x2);
                PipeBarrier<PIPE_V>();
                if constexpr (std::is_same<GType, float>::value) {
                    DataCopy(tensorDgOut, tensorGFp32, BT);
                } else {
                    Cast(tensorDgOut, tensorGFp32, RoundMode::CAST_RINT, BT);
                }
                PipeBarrier<PIPE_V>();

                
                // 处理最后一个位置的 dg
                // b_dg_last *= exp(g_last)
                // is_last_mask: 只有最后一个位置添加 dgLastTerm

                // 读取之前 Part 3/4 计算的 dg, 累加
                // (简化处理, 实际应该从 GM 读取并累加)

                inQue1.FreeTensor(tensorDkIn);
                inQue2.FreeTensor(tensorKIn);
                inQue3.FreeTensor(tensorGIn);
                inQue4.FreeTensor(tensorDgIn);
                outQue1.EnQue(tensorDkOut);
                outQue2.EnQue(tensorDgOut);
            }
            
            // CopyOut
            {
                auto tensorDkOut = outQue1.DeQue<DataType>();
                auto tensorDgOut = outQue2.DeQue<GType>();

                DataCopy(gmDk[kOffset], tensorDkOut, dkSize);

                // dg 需要与之前的累加
                // 累加到 dg (读取现有值, 加上新值, 写回)
                // 简化: 直接累加 (需要在 Part 5 中处理最终值)
                // dg 写入最终输出
                DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = real_BT * sizeof(GType);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                DataCopyPad(gmDg[gOffset], tensorDgOut, dataCopyParams);
                outQue1.FreeTensor(tensorDkOut);
                outQue2.FreeTensor(tensorDgOut);
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
        }
    }
}

// ============== Part 6: dq 累加 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessPart6() {
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNum = GetBlockNum();
    uint32_t coreLoops = B * numChunks;
    
    uint32_t dqSize = BT * K;
    
    // 初始化 buffers
    pipe->InitBuffer(inQue1, BUFFER_NUM, dqSize * sizeof(float));    // dq current
    pipe->InitBuffer(inQue2, BUFFER_NUM, dqSize * sizeof(float));    // mm6 from Cube
    pipe->InitBuffer(outQue1, BUFFER_NUM, dqSize * sizeof(DataType));

    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t vecTaskIdx = 0;
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GetChunkOffset(ptrCuSeqLen, ptrChunkIndices, B, H, T,
                        BT, loopIdx, bos, eos);
        uint32_t actual_chunk_len = eos - bos;
        dqSize = actual_chunk_len * K;
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;
        
        for (uint32_t h = 0; h < H; h++) {
            // if (GetSubBlockIdx() == 1) {
            //     CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            //     CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
            //     continue;
            // }
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
                continue;
            }
            uint64_t dqOffset = (h * T + bos) * K;
            
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            
            // Part 6 的 Vector 工作是将 Cube 计算的 mm6 累加到 dq
            // CopyIn
            {
                auto tensorDqIn = inQue1.AllocTensor<DataType>();
                auto tensorMM6In = inQue2.AllocTensor<DataType>();
                
                DataCopy(tensorDqIn[dqSize], gmDq[dqOffset], dqSize);
                DataCopy(tensorMM6In[dqSize], gmMm6[dqOffset], dqSize);

                inQue1.EnQue(tensorDqIn);
                inQue2.EnQue(tensorMM6In);
            }

            //Compute
            {
                auto tensorDqIn = inQue1.DeQue<DataType>();
                auto tensorDqFp32 = tensorDqIn.template ReinterpretCast<float>();
                auto tensorMM6In = inQue2.DeQue<DataType>();
                auto tensorMM6Fp32 = tensorMM6In.template ReinterpretCast<float>();
                auto tensorDqOut = outQue1.AllocTensor<DataType>();
                // Cast

                Cast(tensorDqFp32, tensorDqIn[dqSize], RoundMode::CAST_NONE, dqSize);
                PipeBarrier<PIPE_V>();
                Cast(tensorMM6Fp32, tensorMM6In[dqSize], RoundMode::CAST_NONE, dqSize);
                PipeBarrier<PIPE_V>();

                Add(tensorDqFp32, tensorDqFp32, tensorMM6Fp32, dqSize);

                PipeBarrier<PIPE_V>();
                Cast(tensorDqOut, tensorDqFp32, RoundMode::CAST_RINT, dqSize);

                PipeBarrier<PIPE_V>();
                inQue1.FreeTensor(tensorDqIn);
                inQue2.FreeTensor(tensorMM6In);
                outQue1.EnQue<DataType>(tensorDqOut);
            }

            //CopyOut
            {
                auto tensorDqOut = outQue1.DeQue<DataType>();
                DataCopy(gmDq[dqOffset], tensorDqOut, dqSize);
                outQue1.FreeTensor(tensorDqOut);
            }
            
            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
        }
    }
}

// ============== Part 7: dk 累加 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessPart7() {
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNum = GetBlockNum();
    uint32_t coreLoops = B * numChunks;
    
    uint32_t dkSize = BT * K;
    
    // 初始化 buffers
    pipe->InitBuffer(inQue1, BUFFER_NUM, dkSize * sizeof(float));    // dk current
    pipe->InitBuffer(inQue2, BUFFER_NUM, dkSize * sizeof(float));    // mm6 from Cube
    pipe->InitBuffer(outQue1, BUFFER_NUM, dkSize * sizeof(DataType));

    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t vecTaskIdx = 0;
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GetChunkOffset(ptrCuSeqLen, ptrChunkIndices, B, H, T,
                        BT, loopIdx, bos, eos);
        uint32_t actual_chunk_len = eos - bos;
        dkSize = actual_chunk_len * K;
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;
        
        for (uint32_t h = 0; h < H; h++) {
            // if (GetSubBlockIdx() == 1) {
            //     CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            //     CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
            //     continue;
            // }
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
                continue;
            }
            uint64_t dkOffset = (h * T + bos) * K;
            
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
            
            // Part 7 的 Vector 工作是将 Cube 计算的 mm7 累加到 dk
            // CopyIn
            {
                auto tensorDkIn = inQue1.AllocTensor<DataType>();
                auto tensorMm7In = inQue2.AllocTensor<DataType>();
                
                DataCopy(tensorDkIn[dkSize], gmDk[dkOffset], dkSize);
                DataCopy(tensorMm7In[dkSize], gmMm7[dkOffset], dkSize);

                inQue1.EnQue(tensorDkIn);
                inQue2.EnQue(tensorMm7In);
            }

            //Compute
            {
                auto tensorDkIn = inQue1.DeQue<DataType>();
                auto tensorDkFp32 = tensorDkIn.template ReinterpretCast<float>();
                auto tensorMm7In = inQue2.DeQue<DataType>();
                auto tensorMm7Fp32 = tensorMm7In.template ReinterpretCast<float>();
                auto tensorDkOut = outQue1.AllocTensor<DataType>();

                // Cast
                Cast(tensorDkFp32, tensorDkIn[dkSize], RoundMode::CAST_NONE, dkSize);
                PipeBarrier<PIPE_V>();
                Cast(tensorMm7Fp32, tensorMm7In[dkSize], RoundMode::CAST_NONE, dkSize);
                PipeBarrier<PIPE_V>();
                Add(tensorDkFp32, tensorDkFp32, tensorMm7Fp32, dkSize);
                PipeBarrier<PIPE_V>();
                Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, dkSize);

                PipeBarrier<PIPE_V>();
                inQue1.FreeTensor(tensorDkIn);
                inQue2.FreeTensor(tensorMm7In);
                outQue1.EnQue<DataType>(tensorDkOut);
            }

            //CopyOut
            {
                auto tensorDkOut = outQue1.DeQue<DataType>();
                DataCopy(gmDk[dkOffset], tensorDkOut, dkSize);

                outQue1.FreeTensor(tensorDkOut);
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
        }
    }
}

#endif  // CHUNK_BWD_DQKWG_VECTOR_H