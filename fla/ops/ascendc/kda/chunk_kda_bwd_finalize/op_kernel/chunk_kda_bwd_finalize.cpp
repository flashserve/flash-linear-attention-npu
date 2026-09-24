/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
#error "chunk_kda_bwd_finalize is an Ascend 950 / arch35-only kernel"
#endif

#include "chunk_kda_bwd_finalize_struct.h"
#include "arch35/chunk_kda_bwd_finalize_cube.h"
#include "arch35/chunk_kda_bwd_finalize_vector.h"

// 阶段导航（沿用主线编号；同一编号下的 AIC/AIV 按数据依赖并行推进）：
// Stage0  AIC：四项基础矩阵乘；AIV：E/kE、g_last、r_h，并补齐 dW 残差。
// Stage1  AIC：Akkᵀ @ dW、dW @ kEᵀ；Stage2 AIV：构造严格下三角 Zb。
// Stage3  AIC：Tza=Zb @ Akkᵀ；AIV：状态梯度、dv、dq_base 与 gate_state。
// Stage4  AIC：dAkk_raw=Akkᵀ @ Tza；AIV：dk_base、db_base、dg_base。
// Stage5  AIV：32 行分带、gate 平移及高低位 NZ 操作数。
// Stage6  AIC：dq_local 与 left；Stage7 AIV：合并 dq、更新 dg，可选 Q 归一化反向。
// Stage8  AIC：right；Stage9/10 AIV：合并 dk/db、更新 dg，可选 K 归一化反向。
// Stage11 AIV：完整 chunk 逆序 gate 扫描与参数部分和；Stage12：跨 chunk 归约。
// owner 是窗口内 head，aiv 是目标向量核，slot 是轮转交接槽，三者不可混用。
namespace KDA {

// 执行主体：同一个 MIX launch 中分派 Cube 与 Vector 两条流水。
__aicore__ inline void ChunkKdaBwdFinalizeImpl(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR akk,
    GM_ADDR vNew, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
    GM_ADDR dAqk, GM_ADDR dqRaw, GM_ADDR qRstd, GM_ADDR dq, GM_ADDR dv,
    GM_ADDR kRstd, GM_ADDR dk, GM_ADDR dBeta,
    GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR dG,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
    const ChunkKdaBwdFinalizeTilingData *tiling)
{
    // 1. AIC 负责矩阵乘与 L0C→GM/L1/UB 交接。
    if ASCEND_IS_AIC {
        ChunkKdaBwdFinalizeCubeStage10 cube;
        cube.Init(v, akk, vNew, h, dh, dvScan, cuSeqlens, chunkIndices,
                  workspace, tiling);
        cube.Process();
    }

    // 2. AIV 负责输入变换、非矩阵计算、输出与参数部分和。
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        ChunkKdaBwdFinalizeVectorStage12 vector;
        vector.Init(q, k, v, gk, beta, h, dh, dAqk, dqRaw, qRstd, dq, dv,
                    kRstd, dk, dBeta,
                    rawG, aLog, dtBias, dG,
                    cuSeqlens, chunkIndices,
                    workspace, tiling, &pipe);
        vector.Process();
    }
}

} // namespace KDA

extern "C" __global__ __aicore__ void chunk_kda_bwd_finalize(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk,
    GM_ADDR rawG, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
    GM_ADDR dvScan, GM_ADDR dAqk, GM_ADDR dqRaw,
    GM_ADDR qRstd, GM_ADDR kRstd,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dBeta,
    GM_ADDR dG, GM_ADDR dALog, GM_ADDR dDtBias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // 1. 读取 host tiling；key 1/2 为定长/变长，3/4 为对应的 Q/K 归一化版本。
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdFinalizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdFinalizeTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(1) || TILING_KEY_IS(2) ||
        TILING_KEY_IS(3) || TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

        // 2. 完成逐 chunk 的 Stage0–11，参数部分和留在 workspace。
        KDA::ChunkKdaBwdFinalizeImpl(
            q, k, v, gk, beta, akk, vNew, h, dh, dvScan,
            dAqk, dqRaw, qRstd, dq, dv, kRstd, dk, dBeta,
            rawG, aLog, dtBias, dG,
            cuSeqlens, chunkIndices, userWorkspace, &tilingData);
        // 仅 AIV 生成参数梯度部分和；AIV-only SyncAll 使用 flag 14。
        // 此方向的 flag 14 未用于前序交接；不能换成占用 12/13 的 MIX SyncAll，
        // 否则会消费尚未用尽的 ZB_FREE 通知。
        AscendC::SyncAll<true>();

        // 3. 所有 AIV 的部分和就绪后，Stage12 按 head 归约 d_a_log 与 d_dt_bias。
        if ASCEND_IS_AIV {
            KDA::FinalizeStage12(userWorkspace, dALog, dDtBias, tilingData);
        }
    }
}
