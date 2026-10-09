/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_bwd_kernel_merged_struct.h
 * \brief Tiling data struct and split/parallel constants for pre_process_bwd_kernel_merged.
 */

#ifndef PRE_PROCESS_BWD_KERNEL_MERGED_STRUCT_H
#define PRE_PROCESS_BWD_KERNEL_MERGED_STRUCT_H

#include <cstdint>

// Tiling key（ASCENDC_TPL 模板参数）声明单独成文件，与仓内其它算子一致；这里转包含，
// 保证 op_kernel 与 op_host 两侧只要包含本文件就能同时看到模板参数与 tiling 数据结构。
#include "pre_process_bwd_kernel_merged_tiling_key.h"

namespace CP {

// 分核模式
constexpr uint64_t PRE_PROCESS_BWD_KERNEL_MERGED_SPLIT_BY_HEAD = 0;  // 默认：仅按 head 连续分核
constexpr uint64_t PRE_PROCESS_BWD_KERNEL_MERGED_SPLIT_BY_TILE = 1;  // Hv < 核数：按 (hv, 列 tile) 展平
// v19：head 内并行（两个 AIV 合干同一条链）。aivPerBlock == 2 时可选：
//   两个 AIV 各承包**同一个 head**的一半——E 链（dH，[K,V]）按 V 列切半、P 链（P，[K,K]）按 K 行切半，
//   V0 的操作数按行切半；操作数/中间量平面按 block 共享（slice 不再按 sub 分），
//   跨核 flag 用集合语义（同一 id 两侧共享 + 需要时 AIC 连续 wait 两次）。
//   动机见 design.md §24/§27：A2（910B，mode 0x2）的集合同步语义只允许"两 AIV 合干同一条链"，
//   而 A2 又只有靠 1:2 才能把 32 条链摊到 40 个 AIV 上。
constexpr uint64_t PRE_PROCESS_BWD_KERNEL_MERGED_SPLIT_BY_HEAD_HALF = 2;

constexpr uint32_t PRE_PROCESS_BWD_KERNEL_MERGED_K_GROUP_ROWS = 64;
constexpr uint32_t PRE_PROCESS_BWD_KERNEL_MERGED_CHUNK_SIZE = 64;
// 本版只支持的状态行/列维：K = V = 128。
// 依据：状态行按 64 行分组、列按 16 个元素（64B）成组，Vector 侧逐行向量/寄存器读写也要求行首
// 对齐；实测 K/V 取其他值（如 64、96、72、256）会出现设备报错或结果错误，因此 host 直接拦截，
// 只放开目标场景 K = V = 128、chunk_size = 64。放宽该约束需要同步改 tiling、workspace 账本与用例。
constexpr uint32_t PRE_PROCESS_BWD_KERNEL_MERGED_K_DIM = 128;
constexpr uint32_t PRE_PROCESS_BWD_KERNEL_MERGED_V_DIM = 128;

struct PreProcessBwdKernelMergedTilingData {
    // shape
    uint64_t B;
    uint64_t T;
    uint64_t Hk;
    uint64_t Hv;
    uint64_t K;
    uint64_t V;
    // chunk 与 tile
    uint64_t chunkSize;
    uint64_t chunkNum;
    uint64_t seqNum;
    uint64_t blockSize;   // 列 tile 宽度：K <= 64 时为 32，否则 64
    uint64_t tileV;
    uint64_t tileK;
    uint64_t tileNum;     // tileV + tileK
    uint64_t kGroupNum;   // ceil(K / 64)，最多 4
    // segment 与模式
    uint64_t bos;
    uint64_t eos;
    uint64_t isVarLen;
    uint64_t isScale;
    uint64_t useGateG;
    uint64_t useGateGk;
    // 分核
    uint64_t usedCoreNum;
    uint64_t blockDim;
    uint64_t splitMode;
    uint64_t groupHeads;  // splitMode == BY_HEAD 时每核连续 head 数
    uint64_t halfSplit;   // 1：启用 head 内并行（两个 AIV 合干同一条链）；0：一个 AIV 一条链
    // v20：把 ZP = (-T1)@P_bf 并进"AB + Z"的同一次 GEMM（仅 A5/arch35；host 按 SoC 下发）。
    // 打开时 B 操作数变成 [do; -dv; dH_bf | P_bf]（行主序 [2M+K, V+K]），C 平面变成 [K, V+K]（inc | ZP），
    // 于是每 chunk 链上的 AIC MMAD 从 2 次（ABZ + ZP）降到 1 次。
    uint64_t mergeZp;
    uint64_t aivPerBlock;  // 1：A2/A3（MIX_AIC_1_1）；2：A5（MIX_AIC_1_2，一个 block 的两个 AIV 各承包一个 head）
    // 用户 workspace 规划（相对 user workspace 起始的字节偏移）
    uint64_t slotNum;
    uint64_t slotBytes;
    uint64_t slotWsOffset;
    uint64_t dhWsOffset;
    uint64_t dhBfWsOffset;
    uint64_t dvPreWsOffset;
    uint64_t dvHatWsOffset;
    uint64_t qtermWsOffset;
    uint64_t wtermWsOffset;
    uint64_t t1WsOffset;
    // v13：AB 与 Z 合并成一次 GEMM 用的两份拼接操作数（arch35 专用；arch22 仍用 slot + 旧平面）
    //   aOperWs → [Q̄s(M,K) | W(M,K) | (-T1)ᵀ(K,K)]，模型 dtype，Cube 的 A 操作数（列主序）
    //   bOperWs → [do(M,V) | -dv(M,V) | dH_bf(K,V)]，模型 dtype，Cube 的 B 操作数（行主序）
    uint64_t aOperWsOffset;
    uint64_t aOperWsBytes;
    uint64_t bOperWsOffset;
    uint64_t bOperWsBytes;
    uint64_t pcWsOffset;
    uint64_t pWsOffset;
    uint64_t pBfWsOffset;
    uint64_t metaWsOffset;
    uint64_t dhWsBytes;
    uint64_t dhBfWsBytes;
    uint64_t dvPreWsBytes;
    uint64_t dvHatWsBytes;
    uint64_t qtermWsBytes;
    uint64_t wtermWsBytes;
    uint64_t t1WsBytes;
    uint64_t pcWsBytes;
    uint64_t pWsBytes;
    uint64_t pBfWsBytes;
    uint64_t metaWsBytes;
    uint64_t totalWsBytes;
    float scale;
};

} // namespace CP

#endif // PRE_PROCESS_BWD_KERNEL_MERGED_STRUCT_H
