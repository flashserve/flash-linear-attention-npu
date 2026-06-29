/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRIL_H
#define SOLVE_TRIL_H

#include "kernel_operator.h"
#include "solve_tril_common.h"
#include "mem.h"

using namespace AscendC;

// ============================================================================
// SolveTril —— 完整下三角逆 (I+A)^{-1} 的 ascend950 实现（MCH + MBH 合一）
//
// 两段算法合并在同一 SolveTril 类（source 风格：OnChipBuffer + SyncAll 协同）：
//   - MCH（块对角逆）：源仓移植。AIV 生成 NZ 辅助矩阵 + gather 对角块，AIC 牛顿迭代
//     求每个 16×16 对角块逆，结果(NZ)暂存 ub_Res(UB)。
//   - MBH（递归合并）：基于【本仓已验证 RecursiveMerge(UB-opt) 算法】移植（非源仓）。
//     X 常驻 ub_Res(UB,NZ)；逐层 blockSize=16->cur：AIV 从 ub_Res 提取 drv/oth 对角块
//     到 L1，AIC 做 B/C/E/G 四步矩乘合并，层间结果 Fixpipe 回 ub_Res，末层写 gm_out。
//
// MCH 移植中的接口/“非泛化参数”适配（不改 MCH 整体算法逻辑）：
//   [接口] kernel 形参收敛为 x / cu_seqlens / chunk_indices / x_out / workspace /
//          tiling；输出沿用本仓 x_out 命名；索引为 INT64；支持 FP16 + BF16。
//   [tiling 字段] 沿用本仓字段名、按 MCH 语义读取：
//          batchSize / seqLen / numHeads / chunkSize / numChunks(=chunkNumInSeq) /
//          totalChunks(=chunkNumTotal, 主循环上界) / layoutMode(=mode) / isLower。
//   [尾块辅助矩阵修复] 源实现把单位/全零矩阵按整 chunk_size 在循环外生成一次；尾块
//          chunk_size_actual 变化时 NZ 单位阵的分形偏移随之改变，会算错。改为仅在
//          chunk 尺寸“发生变化”时（即 seqlen 尾块）按 chunk_size_actual 重建 I/Zero
//          并重拷 l1_I；非尾块尺寸不变则复用，避免额外开销。
//   [GM 偏移泛化] 源 x_gm_offset 仅在“单 chunk / 单 head”下成立。改为与本仓已验证 MBH
//          的 GetTileGMOffset 对齐的完整公式（[B,T,H,BT] 布局，行跨度 = num_head*chunk_size）。
//   [搬运步长泛化] 源对角块 gather 的 srcStride 硬编码为 1（仅 BT=32、H=1 成立）。
//          改为按物理行跨度推导 srcStride = num_head*chunk_size/16 - 1。
//
// MBH 移植要点（对齐本仓已验证 UB-opt 路径）：
//   - 跨核同步用 SyncAll<false>（本仓实测：CrossCoreFlag 在 1:2 MIX 上死锁）。多核“固定调度”：
//     每 (AIC,其2 AIV subcore) 组处理连续 tile 段 [groupIdx*tilesPerCore, +tilesPerCore)，
//     所有组都走 tilesPerCore 个 slot、每 slot 固定 (2 + 2*maxLevels) 次 SyncAll（maxLevels=
//     log2(chunk_size/16)，各核一致）—— 真实层做活、其余层空屏障补齐、越界 slot 全空屏障，
//     故组间 SyncAll 计数恒等，不死锁。单核(usedCoreNum=1)退化为逐 tile 行为。
//   - -A 由 AIV 备数：完整 A GM(ND)->ub_FullA(NZ) 后 Muls(-1) -> l1_MNEG（替代源 -I 矩乘，
//     代数等价；省一次 cube 矩乘与 -I 常量）。
//   - MBH 各步矩乘用本仓已验证 RecursiveMerge 的 V1 约定（MbhMatmulToL0C，对 A 做块转置预交换），
//     不复用 MCH 的 V2 MatmulToL0C：MBH 的 -A / Y 含非对角分形，两种约定在非对角分形上不等价。
//
// 已知限制 / 假设（待硬件验证）：
//   * 多核固定调度假设：MIX 1:2 下 GetBlockIdx() 对 AIC 与其配对 AIV 返回同一组索引；若目标
//     SDK 语义不同（如 AIV 返回 AIV-global 索引），仅需改 Process() 中 groupIdx 推导一行。
//   * 尾块在 ChunkAlign 上取整后会按对齐尺寸读取 GM，可能越过有效行（源既有行为）。
//   * MCH 用 V2、MBH 用 V1：MCH 操作数为块对角（非对角分形恒 0），两约定等价，故 MCH 验证
//     无法覆盖非对角分形；MBH 改用本仓已验证 V1 约定，BT>16 正确性由 V1 自身保证。
// ============================================================================

constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};

template <typename InDtype, typename OutDtype>
class SolveTril {
public:
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                GM_ADDR workspace, const SolveTrilTilingData *tilingData)
    {
        // Tiling（沿用本仓字段名，按 MCH 语义解释）
        batch_size = tilingData->batchSize;
        seq_length = tilingData->seqLen;          // <- seqLength
        num_head = tilingData->numHeads;          // <- numHead
        chunk_size = tilingData->chunkSize;
        chunk_num_in_seq = tilingData->numChunks; // <- chunkNumInSeq
        chunk_num_total = tilingData->totalChunks;// <- chunkNumTotal（主循环上界 = 全部 tile 数）
        mode = tilingData->layoutMode;            // <- mode（1=定长 bsnd, 2=变长 tnd）
        is_lower = tilingData->isLower;           // MBH 驱动块选择（下三角=1）
        tiles_per_core = tilingData->tilesPerCore;// 每个 (AIC,其AIV) 组处理的 tile 数（多核划分）

        // GM
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aGm));
        gm_cu_seqlens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens));
        gm_chunk_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunk_indices));
        gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(outGm));

        OnChipBuffer buf;

        // UB（每块 chunk_size*chunk_size 个 InDtype）：
        //   ub_I / ub_Zero / ub_A / ub_I_A / ub_Res 为 MCH 所用；
        //   ub_FullA 为 MBH 阶段暂存“完整 -A”（GM->UB nd2nz 后取负）。
        ub_I = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        ub_Zero = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        ub_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);
        ub_I_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 3);
        ub_Res = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 4);
        ub_FullA = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 5);
        // L1（NZ 槽，每槽 chunk_size*chunk_size 个 InDtype）：
        //   MCH 用 l1_I / l1_X / l1_Y；MBH 复用三者并额外用 l1_MNEG(-A) / l1_INPUT(提取的 oth 块)。
        l1_I = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_X = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l1_Y = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);
        l1_MNEG = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 3);
        l1_INPUT = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 4);

        // L0
        l0a_X = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0a_Y = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0b_X = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
        l0b_Y = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0c_X = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0c_Y = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(chunk_size * chunk_size * sizeof(float));

        // Core
        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        sub_block_idx = AscendC::GetSubBlockIdx();

        // 辅助矩阵缓存标记：0 为无效初值（chunk_size_actual 恒 >=16），首个 tile 必触发生成。
        last_chunk_size = 0;
    }

    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void ub_to_l1(AscendC::LocalTensor<InDtype> l1Tensor,
                                    AscendC::LocalTensor<InDtype> ubTensor, uint32_t chunkSize)
    {
        AscendC::DataCopy(l1Tensor,                               // dst
                          ubTensor,                               // src
                          AscendC::DataCopyParams(1,              // nBurst
                                                  chunkSize * chunkSize / 16,  // lenBurst
                                                  0,              // srcGap
                                                  0));            // dstGap
    }

    __aicore__ inline void FixpipeL0cToL1(AscendC::LocalTensor<InDtype> l1Tensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t chunkSize)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
        fixPipeParams.nSize = chunkSize;
        fixPipeParams.mSize = chunkSize;
        fixPipeParams.srcStride = chunkSize;
        fixPipeParams.dstStride = chunkSize * 16;

        if constexpr (std::is_same_v<InDtype, half>) {
            fixPipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixPipeParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<InDtype, float, CFG_NZ_L1>(l1Tensor, l0CTensor, fixPipeParams);
    }

    __aicore__ inline void FixpipeL0cToUB(AscendC::LocalTensor<InDtype> ubTensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t chunkSize)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
        fixPipeParams.nSize = chunkSize;
        fixPipeParams.mSize = chunkSize;
        fixPipeParams.srcStride = chunkSize;
        fixPipeParams.dstStride = chunkSize * 16;
        fixPipeParams.dualDstCtl = 0;
        fixPipeParams.subBlockId = 0;

        if constexpr (std::is_same_v<InDtype, half>) {
            fixPipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixPipeParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<InDtype, float, CFG_NZ_UB>(ubTensor, l0CTensor, fixPipeParams);
    }

    __aicore__ inline int64_t ChunkAlign(int64_t cur_chunk)
    {
        if (cur_chunk <= 16)
            return 16;
        if (cur_chunk <= 32)
            return 32;
        if (cur_chunk <= 64)
            return 64;
        return 128;
    }

    // 结果写回：L0C(FP32, NZ) -> gm_out(ND, 行优先)。dstStride = GM 物理行跨度。
    // 用于 cur==16（纯 MCH，无 MBH）及 cur>16 时 MBH 末层的最终结果写出。
    __aicore__ inline void FixpipeL0cToGM(AscendC::GlobalTensor<OutDtype> gmTensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t curSize, uint32_t dstStride)
    {
        auto intriParams = AscendC::FixpipeParamsV220(curSize, curSize, curSize, dstStride, false);
        if constexpr (std::is_same_v<OutDtype, half>) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else {
            intriParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<OutDtype, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0CTensor, intriParams);
    }

    __aicore__ inline void MatmulToL0C(AscendC::LocalTensor<InDtype> l1A, AscendC::LocalTensor<InDtype> l1B,
                                       AscendC::LocalTensor<InDtype> l0A, AscendC::LocalTensor<InDtype> l0B,
                                       AscendC::LocalTensor<float> l0C, int64_t chunkSize, bool initC)
    {
        int64_t numFracs = chunkSize / 16;

        AscendC::LoadData2DParamsV2 loadDataParamsA;
        loadDataParamsA.mStartPosition = 0;
        loadDataParamsA.kStartPosition = 0;
        loadDataParamsA.mStep = numFracs;
        loadDataParamsA.kStep = numFracs;
        loadDataParamsA.srcStride = numFracs;
        loadDataParamsA.dstStride = numFracs;
        loadDataParamsA.ifTranspose = false;
        AscendC::LoadData(l0A, l1A, loadDataParamsA);

        AscendC::LoadData2DParamsV2 loadDataParamsB;
        loadDataParamsB.mStartPosition = 0;
        loadDataParamsB.kStartPosition = 0;
        loadDataParamsB.mStep = numFracs;
        loadDataParamsB.kStep = numFracs;
        loadDataParamsB.srcStride = numFracs;
        loadDataParamsB.dstStride = numFracs;
        loadDataParamsB.ifTranspose = true;
        AscendC::LoadData(l0B, l1B, loadDataParamsB);

        PipeBarrier<PIPE_ALL>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunkSize;
        mmadParams.n = chunkSize;
        mmadParams.k = chunkSize;
        mmadParams.cmatrixInitVal = initC;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmadParams);
    }

    // 按给定 cur_size 生成 NZ 块对角辅助矩阵（尾块安全）：单位阵 I、全零阵 Zero，并清零 ub_A。
    __aicore__ inline void AuxMatrixGen(int64_t cur_size)
    {
        uint64_t NUM_FRACS = cur_size / 16; // 对角分形个数
        uint64_t NUM_ITER = NUM_FRACS * 2;  // 8x16 条带数
        int32_t chunkElems = static_cast<int32_t>(cur_size * cur_size);

        // 清零 A / 生成单位阵 I（NZ：对角分形落在 (i*NUM_FRACS+i)*256 = i*(cur_size*16+256)）
        Duplicate(ub_A, (InDtype)0, chunkElems);
        Duplicate(ub_I, (InDtype)0, chunkElems);
        for (uint64_t stripIdx = 0; stripIdx < NUM_ITER; stripIdx++) {
            uint64_t fracsIdx = stripIdx / 2;
            uint64_t oldEvenIdx = stripIdx % 2;
            uint64_t diagMask[2] = {
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][0],
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][1]
            };
            uint64_t UB_DIAG_I_OFF = fracsIdx * (cur_size + 16) * 16 + oldEvenIdx * 8 * 16;
            Duplicate(ub_I[UB_DIAG_I_OFF], (InDtype)1.0f, diagMask, 1, 1, 1);
        }
        // 全零矩阵
        Duplicate(ub_Zero, (InDtype)0, chunkElems);
    }

    // 由 loop_idx 计算该 tile 的 GM 偏移与对齐后的实际 chunk 尺寸。
    // 与本仓已验证 MBH 的 GetTileGMOffset / GetTileValidSize 公式对齐（[B,T,H,BT] 布局）。
    __aicore__ inline void ComputeTile(int64_t loop_idx, int64_t &x_gm_offset, int64_t &cur_size)
    {
        int64_t seq_idx = 0;
        int64_t chunk_in_seq_idx = 0;
        int64_t head_idx = 0;
        int64_t chunk_idx = 0;
        int64_t local_seq_length = seq_length;
        int64_t local_chunk_num_in_seq = chunk_num_in_seq;

        if (mode == 1) { // 定长 bsnd
            seq_idx = loop_idx / (chunk_num_in_seq * num_head);
            chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
            head_idx = loop_idx % (chunk_num_in_seq * num_head) % num_head;
            // [B,T,H,BT]: base = b*T*H*BT + chunk*BT*H*BT + h*BT
            x_gm_offset = seq_idx * seq_length * num_head * chunk_size +
                          chunk_in_seq_idx * chunk_size * num_head * chunk_size +
                          head_idx * chunk_size;
        } else if (mode == 2) { // 变长 tnd
            chunk_idx = loop_idx / num_head;
            head_idx = loop_idx % num_head;
            seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
            chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
            local_seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
            local_chunk_num_in_seq = CeilDiv(local_seq_length, chunk_size);
            int64_t bos = gm_cu_seqlens.GetValue(seq_idx);
            // (bos + chunk*BT) * H * BT + h * BT
            x_gm_offset = (bos + chunk_in_seq_idx * chunk_size) * num_head * chunk_size +
                          head_idx * chunk_size;
        }

        cur_size = (chunk_in_seq_idx == (local_chunk_num_in_seq - 1)) ?
                       ChunkAlign(local_seq_length - chunk_in_seq_idx * chunk_size) :
                       chunk_size;
    }

    // ---- AIV：MCH 单 tile 备数（含 MBH 所需的完整 -A 暂存）----
    __aicore__ inline void AivMchPrep(int64_t cur, int64_t x_gm_offset, int64_t row_stride)
    {
        // [尾块辅助矩阵] 仅 chunk 尺寸变化时重建 NZ 单位/全零阵并重拷 l1_I（避免每 tile 开销）。
        if (cur != last_chunk_size) {
            AuxMatrixGen(cur);
            AscendC::PipeBarrier<PIPE_ALL>();
            ub_to_l1(l1_I, ub_I, static_cast<uint32_t>(cur));
            AscendC::PipeBarrier<PIPE_ALL>();
            last_chunk_size = cur;
        }

        // 对角 16x16 块 GM(ND) -> ub_A(NZ 块对角)；srcStride 按物理行跨度泛化。
        uint16_t src_blk_stride = static_cast<uint16_t>(row_stride / 16 - 1);
        for (uint64_t i = 0; i < (uint64_t)(cur / 16); i++) {
            uint64_t srcOffset = i * (16 * (uint64_t)row_stride + 16);
            uint64_t dstOffset = i * ((uint64_t)cur * 16 + 16 * 16);
            AscendC::DataCopy(ub_A[dstOffset], gm_a[x_gm_offset + srcOffset],
                              AscendC::DataCopyParams(16, 1, src_blk_stride, 0));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        ub_to_l1(l1_Y, ub_A, static_cast<uint32_t>(cur));        // l1_Y = A(块对角)
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sub(ub_I_A, ub_I, ub_A, (int32_t)(cur * cur));  // I - A
        AscendC::PipeBarrier<PIPE_ALL>();
        ub_to_l1(l1_X, ub_I_A, static_cast<uint32_t>(cur));      // l1_X = I - A
        AscendC::PipeBarrier<PIPE_ALL>();

        // MBH 预备（cur>16）：完整 A GM(ND)->ub_FullA(NZ)，取负 -> l1_MNEG(NZ)。
        if (cur > 16) {
            AscendC::Nd2NzParams p;
            p.ndNum = 1;
            p.nValue = static_cast<uint32_t>(cur);
            p.dValue = static_cast<uint32_t>(cur);
            p.srcDValue = static_cast<uint32_t>(row_stride);
            p.srcNdMatrixStride = 0;
            p.dstNzNStride = 1;
            p.dstNzC0Stride = static_cast<uint16_t>(cur);
            p.dstNzMatrixStride = 0;
            AscendC::DataCopy(ub_FullA, gm_a[x_gm_offset], p);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(ub_FullA, ub_FullA, (InDtype)(-1.0f), (int32_t)(cur * cur));
            AscendC::PipeBarrier<PIPE_ALL>();
            ub_to_l1(l1_MNEG, ub_FullA, static_cast<uint32_t>(cur));  // l1_MNEG = -A
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // ---- AIC：MCH 牛顿迭代，求 16x16 对角块逆（块对角逆）----
    // cur>16：结果 Fixpipe 暂存 ub_Res(UB,NZ) 供 MBH 消费；cur==16：纯 MCH，直接写 gm_out。
    __aicore__ inline void AicMchNewton(int64_t cur, int64_t x_gm_offset, int64_t row_stride)
    {
        MatmulToL0C(l1_Y, l1_Y, l0a_Y, l0b_Y, l0c_Y, cur, true);
        AscendC::PipeBarrier<PIPE_ALL>();
        FixpipeL0cToL1(l1_Y, l0c_Y, cur);
        AscendC::PipeBarrier<PIPE_ALL>();

        MatmulToL0C(l1_I, l1_X, l0a_X, l0b_X, l0c_X, cur, true);
        AscendC::PipeBarrier<PIPE_ALL>();
        FixpipeL0cToL1(l1_X, l0c_X, cur);
        AscendC::PipeBarrier<PIPE_ALL>();

        MatmulToL0C(l1_X, l1_Y, l0a_X, l0b_X, l0c_X, cur, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        FixpipeL0cToL1(l1_X, l0c_X, cur);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint64_t iter = 0; iter < 2; iter++) {
            MatmulToL0C(l1_Y, l1_Y, l0a_Y, l0b_Y, l0c_Y, cur, true);
            AscendC::PipeBarrier<PIPE_ALL>();
            FixpipeL0cToL1(l1_Y, l0c_Y, cur);
            AscendC::PipeBarrier<PIPE_ALL>();
            MatmulToL0C(l1_X, l1_Y, l0a_X, l0b_X, l0c_X, cur, false);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (iter == 1) {
                if (cur > 16) {
                    // MCH 块对角逆 -> ub_Res(UB,NZ)，供 MBH 从 UB 提取消费。
                    FixpipeL0cToUB(ub_Res, l0c_X, cur);
                } else {
                    // cur==16：无 MBH，直接写最终结果到 gm_out(ND)。
                    FixpipeL0cToGM(gm_out[x_gm_offset], l0c_X,
                                   static_cast<uint32_t>(cur), static_cast<uint32_t>(row_stride));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                FixpipeL0cToL1(l1_X, l0c_X, cur);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // ---- AIV：清 L1 槽（zeroUB->L1，整槽置零）。zeroUB 复用 ub_Zero（已按 cur 清零）----
    __aicore__ inline void ClearSlotUB(AscendC::LocalTensor<InDtype> l1Slot, int64_t cur)
    {
        AscendC::DataCopy(l1Slot, ub_Zero,
                          AscendC::DataCopyParams(1, (uint16_t)(cur * cur / 16), 0, 0));
    }

    // ---- AIV：从 ub_Res(NZ) 按块 raw UB->L1 提取选中对角块到 l1Slot（非选中分形由 ClearSlotUB 置零）----
    // ub_Res 与 L1 槽 NZ 布局一致：分形 (fr,fc) 偏移 = (fc*NUM_FRACS+fr)*256。
    __aicore__ inline void ExtractFromUB(AscendC::LocalTensor<InDtype> l1Slot,
                                         int64_t cur, int32_t blockSize, int32_t startBlock)
    {
        int32_t numFracsTotal = static_cast<int32_t>(cur / 16);
        int32_t numBlocks = static_cast<int32_t>(cur) / blockSize;
        int32_t fracsPerBlock = blockSize / 16;
        constexpr int32_t FRAC_LEN = 16 * 16;

        for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
            for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
                for (int32_t fj = 0; fj < fracsPerBlock; fj++) {
                    int32_t fr = blk * fracsPerBlock + fi;
                    int32_t fc = blk * fracsPerBlock + fj;
                    int32_t off = (fc * numFracsTotal + fr) * FRAC_LEN;
                    AscendC::DataCopy(l1Slot[off], ub_Res[off],
                                      AscendC::DataCopyParams(1, (uint16_t)(FRAC_LEN / 16), 0, 0));
                }
            }
        }
    }

    // ---- AIC：MBH 矩乘（本仓已验证 RecursiveMerge 的 V1 约定，运行期 cur 参数化）----
    // 关键：cube 实际算 blockT(OpA)@B，左算子非对角分形会被块转置；故对 A 按 (i,k)<-src(k,i)
    // 预交换源分形（srcOffsetA = i*numFracs*FRAC_LEN），使 cube 还原出 plain A@B。
    // 不能复用 MCH 的 V2 MatmulToL0C：MCH 操作数是块对角（非对角分形恒 0），两种约定等价，
    // 故 MCH 永远无法验证非对角分形行为；而 MBH 的 -A / Y 含非对角分形，必须用本 V1 约定。
    __aicore__ inline void MbhMatmulToL0C(AscendC::LocalTensor<InDtype> l1A,
                                          AscendC::LocalTensor<InDtype> l1B,
                                          int64_t cur, bool initC)
    {
        constexpr int32_t FRAC_LEN = 16 * 16;
        int32_t numFracs = static_cast<int32_t>(cur / 16);

        AscendC::LoadData2DParams loadParamsA;
        loadParamsA.startIndex = 0;
        loadParamsA.repeatTimes = numFracs;
        loadParamsA.srcStride = 1;
        loadParamsA.dstGap = 0;
        loadParamsA.ifTranspose = false;
        for (int32_t i = 0; i < numFracs; ++i) {
            AscendC::LoadData(l0a_X[i * numFracs * FRAC_LEN], l1A[i * numFracs * FRAC_LEN], loadParamsA);
        }
        AscendC::LoadData2DParams loadParamsB;
        loadParamsB.startIndex = 0;
        loadParamsB.repeatTimes = numFracs;
        loadParamsB.srcStride = numFracs;
        loadParamsB.dstGap = 0;
        loadParamsB.ifTranspose = true;
        for (int32_t i = 0; i < numFracs; ++i) {
            AscendC::LoadData(l0b_X[i * numFracs * FRAC_LEN], l1B[i * FRAC_LEN], loadParamsB);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = cur;
        mmadParams.n = cur;
        mmadParams.k = cur;
        mmadParams.cmatrixInitVal = initC;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0c_X, l0a_X, l0b_X, mmadParams);
    }

    // ---- AIC：MBH 一层的 matmul + Fixpipe 落 L1 ----
    __aicore__ inline void MbhMatmulToSlot(AscendC::LocalTensor<InDtype> l1A,
                                           AscendC::LocalTensor<InDtype> l1B,
                                           AscendC::LocalTensor<InDtype> l1Dst,
                                           int64_t cur, bool initC)
    {
        MbhMatmulToL0C(l1A, l1B, cur, initC);
        AscendC::PipeBarrier<PIPE_ALL>();
        FixpipeL0cToL1(l1Dst, l0c_X, cur);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        int32_t drvStart = is_lower ? 1 : 0;
        int32_t othStart = is_lower ? 0 : 1;
        int64_t row_stride = num_head * chunk_size;

        // ===== 多核固定调度协同 =====
        // SyncAll<false> 是全局屏障（要求所有 launched 核调用次数完全一致，否则死锁；
        // CrossCoreFlag 在本 1:2 MIX 上死锁，故沿用本仓已验证的 SyncAll<false>）。
        // 为使各 (AIC,其2个AIV subcore) 组的 SyncAll 计数恒等，采用“固定调度”：
        //   - 每组处理连续 tile 段 [groupIdx*tiles_per_core, +tiles_per_core)；所有组都走
        //     tiles_per_core 个 slot；每 slot 固定 (2 + 2*maxLevels) 次 SyncAll，与该 tile 的
        //     实际 cur 无关 —— 真实层做活、其余层用空屏障补齐；越界 slot 全空屏障。
        //   - maxLevels = log2(chunk_size/16)，由 chunk_size 推导，各核一致。
        // 这样每核总 SyncAll 次数 = tiles_per_core*(2 + 2*maxLevels)，组间恒等，不死锁。
        // 单核(usedCoreNum=1)时 tiles_per_core=totalTiles、groupIdx=0，退化为原逐 tile 行为。
        //
        // 【关键假设】MIX 1:2 下 GetBlockIdx() 对 AIC 与其配对 AIV 返回同一组索引（core_idx）。
        //   若目标 SDK 语义不同（如 AIV 返回 AIV-global 索引），仅需改此处 groupIdx 推导一行。
        int64_t groupIdx = core_idx;
        int32_t maxLevels = 0;
        for (int32_t bs = 16; bs < chunk_size; bs *= 2) maxLevels++;

        for (int64_t slot = 0; slot < tiles_per_core; slot++) {
            int64_t loop_idx = groupIdx * tiles_per_core + slot;
            bool valid = (loop_idx < chunk_num_total);

            int64_t x_gm_offset = 0;
            int64_t cur = 0;
            if (valid) {
                ComputeTile(loop_idx, x_gm_offset, cur);
            }

            // ===== MCH 阶段 =====
            if ASCEND_IS_AIV {
                if (valid && sub_block_idx == 0) {
                    AivMchPrep(cur, x_gm_offset, row_stride);
                }
            }
            AscendC::SyncAll<false>();   // SP_a: AIV 备数完成 -> AIC 可做 MCH 牛顿
            if ASCEND_IS_AIC {
                if (valid) {
                    AicMchNewton(cur, x_gm_offset, row_stride);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            AscendC::SyncAll<false>();   // SP_b: AIC 牛顿完成(ub_Res 就绪 / cur==16 已写 GM)

            // ===== MBH 阶段：固定 maxLevels 层；真实层(blockSize<cur)做活，其余空屏障补齐 =====
            for (int32_t lvl = 0; lvl < maxLevels; lvl++) {
                int32_t blockSize = 16 << lvl;                       // 16,32,64,...
                bool realLevel = valid && (blockSize < cur);
                bool lastLevel = realLevel && !(blockSize < cur / 2);

                AscendC::SyncAll<false>();   // SP1: ub_Res 就绪 -> AIV 提取
                if ASCEND_IS_AIV {
                    if (realLevel && sub_block_idx == 0) {
                        ClearSlotUB(l1_X, cur);
                        ClearSlotUB(l1_INPUT, cur);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        ExtractFromUB(l1_X, cur, blockSize, drvStart);     // drv -> l1_X
                        ExtractFromUB(l1_INPUT, cur, blockSize, othStart); // oth -> l1_INPUT
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }
                AscendC::SyncAll<false>();   // SP2: AIV 提取完成 -> AIC 矩乘
                if ASCEND_IS_AIC {
                    if (realLevel) {
                        // step B: L0C = I × I（完整单位阵）
                        MbhMatmulToL0C(l1_I, l1_I, cur, true);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        // step C: Y = drv(l1_X) × (-A) + I -> l1_Y
                        MbhMatmulToSlot(l1_X, l1_MNEG, l1_Y, cur, false);
                        // step E: L0C = Y × oth(l1_INPUT)
                        MbhMatmulToL0C(l1_Y, l1_INPUT, cur, true);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        // step G: L0C += I × drv(l1_X)
                        MbhMatmulToL0C(l1_I, l1_X, cur, false);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        if (!lastLevel) {
                            FixpipeL0cToUB(ub_Res, l0c_X, cur);   // 层间结果 -> ub_Res，下层提取
                        } else {
                            FixpipeL0cToGM(gm_out[x_gm_offset], l0c_X,
                                        static_cast<uint32_t>(cur), static_cast<uint32_t>(row_stride));
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }
                
            }
        }
    }

private:
    // Gm
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<int64_t> gm_cu_seqlens;
    AscendC::GlobalTensor<int64_t> gm_chunk_indices;
    AscendC::GlobalTensor<OutDtype> gm_out;

    // UB
    AscendC::LocalTensor<InDtype> ub_A;
    AscendC::LocalTensor<InDtype> ub_I_A;
    AscendC::LocalTensor<InDtype> ub_I;
    AscendC::LocalTensor<InDtype> ub_Zero;
    AscendC::LocalTensor<InDtype> ub_Res;
    AscendC::LocalTensor<InDtype> ub_FullA;   // MBH: 完整 -A 的 UB 暂存

    // L1
    AscendC::LocalTensor<InDtype> l1_X;
    AscendC::LocalTensor<InDtype> l1_Y;
    AscendC::LocalTensor<InDtype> l1_I;
    AscendC::LocalTensor<InDtype> l1_MNEG;    // MBH: -A（NZ）
    AscendC::LocalTensor<InDtype> l1_INPUT;   // MBH: 提取的 oth 对角块（NZ）

    // L0
    AscendC::LocalTensor<InDtype> l0a_X;
    AscendC::LocalTensor<InDtype> l0a_Y;
    AscendC::LocalTensor<InDtype> l0b_X;
    AscendC::LocalTensor<InDtype> l0b_Y;
    AscendC::LocalTensor<float> l0c_X;
    AscendC::LocalTensor<float> l0c_Y;

    // Tiling
    int64_t batch_size;
    int64_t seq_length;
    int64_t num_head;
    int64_t chunk_size;
    int64_t chunk_num_in_seq;
    int64_t chunk_num_total;
    int64_t mode;
    int64_t is_lower;
    int64_t tiles_per_core;

    // Core
    int64_t num_core;
    int64_t core_idx;
    int64_t sub_block_idx;

    // 辅助矩阵当前缓存对应的 chunk 尺寸（仅尺寸变化时重建 I/Zero/l1_I）
    int64_t last_chunk_size;
};


#endif  // SOLVE_TRIL_H
