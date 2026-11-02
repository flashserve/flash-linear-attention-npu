/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_struct.h
 * \brief 主机/设备共用的 tiling 结构（字段顺序必须与 host 侧一致）。
 */

#ifndef PRE_PROCESS_FWD_KERNEL_MERGED_STRUCT_H
#define PRE_PROCESS_FWD_KERNEL_MERGED_STRUCT_H

#include <cstdint>

// Cube(Matmul) 自己的 tiling。host/device 两侧统一用 AscendC::tiling::TCubeTiling：
// 实测 CANN 9.1.0 上 `MatmulApiTiling::GetTiling(optiling::TCubeTiling&)` 重载会段错误，
// 而仓内其它算子（chunk_gated_delta_rule_fwd arch35、chunk_scaled_dot_kkt）用的
// `GetTiling(AscendC::tiling::TCubeTiling&)` 正常。
#include "kernel_tiling/kernel_tiling.h"
#if !defined(__CCE_AICORE__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV_C220_VEC__)
#include "tiling/tiling_api.h"   // host 侧：matmul_tiling::MatmulApiTiling
#endif
using PpFwdCubeTiling = AscendC::tiling::TCubeTiling;

namespace GDN {

// gate 模式（与 TilingKey / kernel 分派一致）
constexpr int64_t PPFM_GATE_USE_G = 0;
constexpr int64_t PPFM_GATE_USE_GK = 1;
constexpr int64_t PPFM_GATE_USE_BG = 2;
// gate dtype（0=fp16, 1=bf16, 2=fp32），与 kernel 模板分派一致
constexpr int64_t PPFM_DTYPE_FP16 = 0;
constexpr int64_t PPFM_DTYPE_BF16 = 1;
constexpr int64_t PPFM_DTYPE_FP32 = 2;

struct PreProcessFwdKernelMergedTilingData {
    int64_t B;            // 恒 1（varlen 打包）
    int64_t Hk;           // key head 数
    int64_t Hv;           // value head 数
    int64_t hvPerHk;      // Hv / Hk（GVA 复用比）
    int64_t T;            // 张量 T 轴全长（子区间窗口时 T_win < T）
    int64_t K;            // key 维（固定 128）
    int64_t V;            // value 维（固定 128）
    int64_t chunkSize;    // 固定 64
    int64_t nSeq;         // 本次调用段数 = len(cu_seqlens) - 1
    int64_t gateMode;     // PPFM_GATE_*
    int64_t gateDtype;    // PPFM_DTYPE_*
    int64_t isVariedLen;  // 恒 1
    int64_t usedAicNum;   // 实际使用的 Cube 核数
    int64_t taskNum;      // nSeq * Hv
    // P5 列块切分因子：1 = 不切（列宽 V/K = 128）；2 = 每条链按列切两半（列宽 64）。
    // 切分只影响"列"维度：h 链切 V 列、m 链切 K 列，两侧互相独立。
    int64_t colSplit;     // ∈ {1, 2}
    // —— Cube(Matmul) tiling：v2 的四个矩阵乘只有两种形状 ——
    //   cubeNoTrans：① vTmp = W_c @ bf16(h)、③ T1 = W_c @ bf16(m)（M=BT,N=128,K=128，A 不转置）
    //   cubeTransA ：② dH   = k^T @ bf16(v_new)、④ T2 = left^T @ bf16(T1)（M=128,N=128,K=BT，A 转置）
    PpFwdCubeTiling cubeNoTrans;
    PpFwdCubeTiling cubeTransA;
    // ---- R21 混合调度 ----
    // 链数 > 核数 且余数较少时：前 hybridBase 条链整宽，余数链按 hybridS 切列；
    // 切出来的 hybridS 片作为「第二个任务（task += usedAicNum）」交给队首若干核 ⇒
    // 把原来「余数核跑 2 条整宽链」的 2 波尾巴换成「1 条整宽 + 1 片」。
    // hybridS <= 1 表示关闭（走原有 colSplit 路径）。
    int64_t hybridS;
    int64_t hybridBase;
};

// v1 每个工作项（AIC/AIV 对）在 user workspace 里的分段布局（单位：float 元素）。
// m 常驻 GM（UB 侧只留小行缓冲，实测这是唯一稳定的写法），另加两段矩阵乘暂存：
//   m  [K,K]     64 KiB   —— 仿射链的状态，跨 chunk 常驻
//   T1 [BT,K]    32 KiB   —— W_c @ m
//   T2 [K,K]     64 KiB   —— L_c^T @ T1
constexpr int64_t PPFM_CORE_M_OFFSET = 0;
constexpr int64_t PPFM_CORE_T1_OFFSET = PPFM_CORE_M_OFFSET + 128 * 128;
constexpr int64_t PPFM_CORE_T2_OFFSET = PPFM_CORE_T1_OFFSET + 64 * 128;
constexpr int64_t PPFM_CORE_MNEXT_OFFSET = PPFM_CORE_T2_OFFSET + 128 * 128;

// 每个工作项（AIC + 2×AIV 对）在 user workspace 里占用的字节数。
// v2 的 GM 暂存区布局（固定 K = V = 128、BT = 64，见 op_kernel 里的 WS_* 常量）：
//   状态 h/m（fp32 + bf16 各一份）192K | 本 chunk 输入 W/k/left/v（bf16）64K
//   | vTmp 32K | bf16(v_new) 16K | dH 64K | T1 32K + bf16(T1) 16K | T2 64K | gate 1K
//   ≈ 481 KiB，留余量取 512 KiB。
constexpr int64_t PPFM_CORE_WS_BYTES = 768 * 1024;

} // namespace GDN

#endif // PRE_PROCESS_FWD_KERNEL_MERGED_STRUCT_H
