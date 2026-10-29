// -----------------------------------------------------------------------------------------------------------
// chunk_kda_fwd_fused.cpp — KDA forward chunk32 大融合 kernel 入口(issue #829)
// 双源 mix kernel:asc_opc 的 mix 流程把本源文件分别按 dav-c310-cube / dav-c310-vec
// 编译两次再链接,__DAV_C310_CUBE__ / __DAV_C310_VEC__ 由编译器按 pass 自动定义,
// 各 pass 只见自己一侧的实现:
//   AIC:kda32_fd32_aic.h — 裸指令(mmad/fixpipe)
//   AIV:kda32_fd32_aiv.h — MicroAPI
// 两侧 Process() 使用对应的软件流水线。
// -----------------------------------------------------------------------------------------------------------
#include <stdint.h>

// q/k/v 仅支持 bf16。

// tiling 布局须与 op_host/chunk_kda_fwd_fused_tiling.h 完全一致(两侧各读各的)
struct KdaFd32TilingData {
    uint64_t nChunks;        // T / 128(kernel 每组处理 128 token)
    uint64_t scaleBf16;      // 16 位 scale 位模式:bf16 = fp32bits>>16;fp16 = 正确舍入的 fp16 bits
    uint64_t gateScaleBits;  // fp32(lowerBound) 位模式(如 -5.0)
    uint64_t headNum;        // B*H(batch 拍平后的 q/k 头数)
    uint64_t seqLen;         // 序列长度
    uint64_t hasInitialState;    // initial_state 输入是否提供(0/1)
    uint64_t outputFinalState;   // final_state 输出是否请求(0/1)
    uint64_t qkGroup;        // GVA:HV/H(q/k 头组大小);H==HV 时恒 1
    uint64_t taskCount;      // 任务数:dense = B*HV,varlen = nSeq*HV
    uint64_t hvNumReal;      // 真实 HV:dt_bias/a_log 按它寻址,所有 batch/序列共享
    uint64_t blockNum;       // 实际 blockDim = min(taskCount, aicCoreNum),任务步进用
    uint64_t varlenMode;     // 1 = varlen:任务 = seq*HV + head,cu_seqlens 见 cuData
    uint64_t epsilonBits;    // fp32(q/k L2 归一化 epsilon)位模式
    uint64_t gateFlags;      // bit0: g 为 fp32;bit1: beta 为 fp32
    uint64_t cuData[1025];   // varlen 的 cu_seqlens(attr 打包进 tiling,免 H2D;nSeq<=1024)
};

#if defined(__DAV_C310_CUBE__)

#include "kda32_fd32_aic.h"

extern "C" __global__ __aicore__ void chunk_kda_fwd_fused(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v, __gm__ uint8_t *g,
    __gm__ uint8_t *beta, __gm__ uint8_t *a_log, __gm__ uint8_t *dt_bias,
    __gm__ uint8_t *initial_state,
    __gm__ uint8_t *o, __gm__ uint8_t *final_state,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const __gm__ KdaFd32TilingData *td = reinterpret_cast<const __gm__ KdaFd32TilingData *>(tiling);
    (void)q; (void)k; (void)v; (void)g; (void)beta; (void)a_log; (void)dt_bias; (void)o;
    (void)initial_state; (void)final_state;  // state IO 全在 AIV 侧
    {
        Kernel<bfloat16_t> kda;
        kda.kdaImp.tmp_gm = workspace;   // tmp 仅 debug 用,生产路径不写
        kda.task_count = td->taskCount; kda.block_num = td->blockNum;
        kda.is_varlen = td->varlenMode; kda.hv_real = td->hvNumReal;
        kda.cu_base = (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData));
        kda.kdaImp.Init();               // 只配 SPR(fixpipe/mmad),不碰 GM
        kda.Process((int)td->nChunks);
    }
}

#elif defined(__DAV_C310_VEC__)

#include "kda32_fd32_aiv.h"           // 内部已含 kernel_operator.h / MicroAPI

extern "C" __global__ __aicore__ void chunk_kda_fwd_fused(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v, __gm__ uint8_t *g,
    __gm__ uint8_t *beta, __gm__ uint8_t *a_log, __gm__ uint8_t *dt_bias,
    __gm__ uint8_t *initial_state,
    __gm__ uint8_t *o, __gm__ uint8_t *final_state,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const __gm__ KdaFd32TilingData *td = reinterpret_cast<const __gm__ KdaFd32TilingData *>(tiling);
    // Init(N, scale, gate_scale, a, g, q, k, v, beta, dt, tmp, o)
    // state 扩展:initial_state/final_state 指针 + tiling flag 传入,Init 内登记
    // 尾块:seqLen 传入作头间 GM 步长与有效行钳制(nChunks = ceil(T/128))
    // 不传 resource:AIV 侧 Resource 成员会增加热路径开销。
    if (td->varlenMode == 0 && td->seqLen % tokens_per_group == 0) {
        Kernel<bfloat16_t, false, false> kda;
        kda.kdaImp.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        a_log, g, q, k, v, beta, dt_bias, workspace, o,
                        initial_state, final_state, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    } else if (td->varlenMode == 0) {
        Kernel<bfloat16_t, false, true> kda;
        kda.kdaImp.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        a_log, g, q, k, v, beta, dt_bias, workspace, o,
                        initial_state, final_state, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    } else {
        Kernel<bfloat16_t, true, true> kda;
        kda.kdaImp.Init(td->nChunks, td->scaleBf16, td->gateScaleBits,
                        a_log, g, q, k, v, beta, dt_bias, workspace, o,
                        initial_state, final_state, td->hasInitialState, td->outputFinalState,
                        td->seqLen, td->qkGroup, td->taskCount, td->hvNumReal, td->blockNum,
                        td->varlenMode,
                        (uint64_t)((__gm__ uint8_t *)tiling + offsetof(KdaFd32TilingData, cuData)),
                        td->epsilonBits, td->gateFlags);
        kda.Process((int)td->nChunks);
    }
}

#endif
