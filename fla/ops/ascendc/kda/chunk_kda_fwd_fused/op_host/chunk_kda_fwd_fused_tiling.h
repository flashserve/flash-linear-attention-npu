#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

// 与 op_kernel/chunk_kda_fwd_fused.cpp 的 KdaFd32TilingData 布局保持一致
BEGIN_TILING_DATA_DEF(ChunkKdaFwdFusedTilingData)
TILING_DATA_FIELD_DEF(uint64_t, nChunks);       // T / 128
TILING_DATA_FIELD_DEF(uint64_t, scaleBf16);     // 16 位 scale 位模式:bf16=fp32bits>>16;fp16=正确舍入 fp16 bits
TILING_DATA_FIELD_DEF(uint64_t, gateScaleBits); // fp32(lowerBound) 位模式
TILING_DATA_FIELD_DEF(uint64_t, headNum);       // H(q/k 头数);blockDim = hvNum(v 头数,一核一 value head)
TILING_DATA_FIELD_DEF(uint64_t, seqLen);        // T
TILING_DATA_FIELD_DEF(uint64_t, hasInitialState);   // initial_state 输入是否提供(0/1)
TILING_DATA_FIELD_DEF(uint64_t, outputFinalState);  // final_state 输出是否请求(0/1)
TILING_DATA_FIELD_DEF(uint64_t, qkGroup);        // GVA:HV/H(q/k 头组大小),H==HV 时恒 1
TILING_DATA_FIELD_DEF(uint64_t, taskCount);     // 任务数 = B*HV(batch 拍平)
TILING_DATA_FIELD_DEF(uint64_t, hvNumReal);     // 真实 HV(dt_bias/a_log 寻址,不随 batch)
TILING_DATA_FIELD_DEF(uint64_t, blockNum);      // 实际 blockDim = min(taskCount, aicCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, varlenMode);    // 0 = dense;1 = TND varlen;2 = NTD varlen
TILING_DATA_FIELD_DEF(uint64_t, epsilonBits);   // fp32(q/k L2 归一化 epsilon)位模式
TILING_DATA_FIELD_DEF(uint64_t, gateFlags);    // bit0: g 为 fp32;bit1: beta 为 fp32(fp32 原生组合 1)
TILING_DATA_FIELD_DEF_ARR(uint64_t, 1025, cuData);  // varlen cu_seqlens 打包(nSeq<=1024)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwdFused, ChunkKdaFwdFusedTilingData)

struct ChunkKdaFwdFusedCompileInfo {};

} // namespace optiling
