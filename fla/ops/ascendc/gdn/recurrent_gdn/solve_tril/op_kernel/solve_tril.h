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
// SolveTril —— 下三角逆 (I+A)^{-1} 的 ascend950 MCH 实现
//
// MCH（块对角逆）：每个 chunk 独立求逆，AIV/AIC 协同流水。
//   - AIV(sub_block_idx=0): AuxMatrixGen → l1_I；逐 tile 提取对角块 GM->UB->l1_Y、
//     计算 I-A → l1_X；CrossCoreSetFlag(0x2) 通知 AIC 数据就绪。
//   - AIC: 逐 tile 牛顿迭代求逆 → gm_out；CrossCoreSetFlag(0x0) 通知 AIV 计算完成。
//   - 跨核采用 CrossCoreSetFlag/WaitFlag 每 tile 握手机制，无死锁。
//   - 多核调度：round-robin (loop_idx = core_idx; loop_idx < total; loop_idx += num_core)。
//
// 移植适配要点（对比源仓参考）：
//   [索引] INT64（与本仓 def / op_api / tiling 一致）
//   [同步] CrossCoreSetFlag/WaitFlag 替代 SyncAll（源仓 SyncAll<false> 在 1:2 MIX 上死锁）
//   [GM 偏移] bsnd / tnd 完整公式，泛化 batch > 1 / seq / head 任意组合
//   [srcStride] 对角块 gather 的 srcStride 按 row_stride 推导（源仓硬编码 1 仅 H=1 成立）
//   [尾块] 辅助矩阵按 chunk_size_actual 生成并缓存（仅尺寸变化时重建），避免每 tile 开销
//   [写出] MCH 结果 Fixpipe 写 gm_out（源仓仅 DumpTensor ub_Res 用于联调）
// ============================================================================

constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};

template <typename InDtype, typename OutDtype>
class SolveTril {
public:
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                GM_ADDR workspace, const SolveTrilTilingData *tilingData)
    {
        // Tiling（沿用本仓字段名）
        batch_size       = tilingData->batchSize;
        seq_length       = tilingData->seqLen;
        num_head         = tilingData->numHeads;
        chunk_size       = tilingData->chunkSize;
        chunk_num_in_seq = tilingData->numChunks;
        chunk_num_total  = tilingData->totalChunks;
        mode             = tilingData->layoutMode;
        
        // GM（INT64 索引，与本仓 infra 一致）
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aGm));
        gm_cu_seqlens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens));
        gm_chunk_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunk_indices));
        gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(outGm));

        OnChipBuffer buf;

        // UB（5 槽，每槽 chunk_size*chunk_size）
        ub_I    = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        ub_Zero = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        ub_A    = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);
        ub_I_A  = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 3);
        ub_Res  = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 4);

        // L1（3 槽）
        l1_I = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_X = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l1_Y = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);

        // L0
        l0a_X = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0a_Y = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0b_X = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
        l0b_Y = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0c_X = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0c_Y = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(chunk_size * chunk_size * sizeof(float));

        // Core
        num_core      = AscendC::GetBlockNum();
        core_idx      = AscendC::GetBlockIdx();
        sub_block_idx = AscendC::GetSubBlockIdx();

        // 辅助矩阵缓存：0 为初始值（chunk_size_actual 恒 >=16），首个 tile 必触发生成
        last_chunk_size = 0;
        last_chunk_actual_size = 0;
    }

    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void ub_to_l1(AscendC::LocalTensor<InDtype> l1Tensor,
                                    AscendC::LocalTensor<InDtype> ubTensor, uint32_t chunkSize)
    {
        AscendC::DataCopy(l1Tensor, ubTensor,
                          AscendC::DataCopyParams(1, chunkSize * chunkSize / 16, 0, 0));
    }

    __aicore__ inline void FixpipeL0cToL1(AscendC::LocalTensor<InDtype> l1Tensor,
                                          AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
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
                                          AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
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
        if (cur_chunk <= 16)  return 16;
        if (cur_chunk <= 32)  return 32;
        if (cur_chunk <= 64)  return 64;
        return 128;
    }

    // MCH V2 MatmulToL0C（块对角操作数专用；非对角分形恒 0，V1/V2 等价）
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

    // 结果写回：L0C(FP32, NZ) -> gm_out(ND, 行优先)
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

    // 按给定 cur_size 生成 NZ 块对角辅助矩阵（尾块安全）
    __aicore__ inline void AuxMatrixGen(int64_t cur_size)
    {
        uint64_t NUM_FRACS = cur_size / 16;
        uint64_t NUM_ITER  = NUM_FRACS * 2;
        int32_t chunkElems = static_cast<int32_t>(cur_size * cur_size);

        // 清零 A / 生成单位阵 I
        Duplicate(ub_A, (InDtype)0, chunkElems);
        Duplicate(ub_I, (InDtype)0, chunkElems);
        for (uint64_t stripIdx = 0; stripIdx < NUM_ITER; stripIdx++) {
            uint64_t fracsIdx   = stripIdx / 2;
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

    // 由 loop_idx 计算该 tile 的 GM 偏移与对齐后的实际 chunk 尺寸
    __aicore__ inline void ComputeTile(int64_t loop_idx, int64_t &x_gm_offset,
                                       int64_t &cur_size, int64_t &row_stride)
    {
        int64_t seq_idx            = 0;
        int64_t chunk_in_seq_idx   = 0;
        int64_t head_idx           = 0;
        int64_t chunk_idx          = 0;
        int64_t local_seq_length   = seq_length;
        int64_t local_chunk_num_in_seq = chunk_num_in_seq;

        row_stride = num_head * chunk_size;

        if (mode == 0) {
            // BHTD: [B, H, T, BT]
            seq_idx          = loop_idx / (chunk_num_in_seq * num_head);
            chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
            head_idx         = loop_idx % (chunk_num_in_seq * num_head) % num_head;
            x_gm_offset = seq_idx * num_head * seq_length * chunk_size +
                          chunk_in_seq_idx * num_head * chunk_size +
                          head_idx * chunk_size;
        } else if (mode == 1) {
            // BSND: [B, T, H, BT]
            seq_idx          = loop_idx / (chunk_num_in_seq * num_head);
            chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
            head_idx         = loop_idx % (chunk_num_in_seq * num_head) % num_head;
            x_gm_offset = seq_idx * seq_length * num_head * chunk_size +
                          chunk_in_seq_idx * chunk_size * num_head * chunk_size +
                          head_idx * chunk_size;
        } else {
            // TND varlen: [total_T, H, BT]; B = 1
            chunk_idx = loop_idx / num_head;
            head_idx  = loop_idx % num_head;
            seq_idx          = gm_chunk_indices.GetValue(chunk_idx * 2);
            chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
            local_seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
            local_chunk_num_in_seq = CeilDiv(local_seq_length, chunk_size);
            int64_t bos = gm_cu_seqlens.GetValue(seq_idx);
            x_gm_offset = (bos + chunk_in_seq_idx * chunk_size) * num_head * chunk_size +
                          head_idx * chunk_size;
        }
        last_chunk_actual_size = local_seq_length - (local_chunk_num_in_seq - 1) * chunk_size;
        last_chunk_size = ChunkAlign(last_chunk_actual_size);


        cur_size = (chunk_in_seq_idx == (local_chunk_num_in_seq - 1))
                       ? ChunkAlign(local_seq_length - chunk_in_seq_idx * chunk_size)
                       : chunk_size;
    }

    // ---- AIV：MCH 单 tile 备数 ----
    __aicore__ inline void AivMchPrep(int64_t loop_idx, int64_t cur, int64_t x_gm_offset, int64_t row_stride)
    {
        // [尾块辅助矩阵] 仅 chunk 尺寸变化时重建 NZ 单位/全零阵并重拷 l1_I
        if ((loop_idx == core_idx / 2) || (cur != last_chunk_size)) {
            AuxMatrixGen(cur);
            AscendC::PipeBarrier<PIPE_ALL>();
            ub_to_l1(l1_I, ub_I, static_cast<uint32_t>(cur));
            AscendC::PipeBarrier<PIPE_ALL>();
            last_chunk_size = cur;
        }

        // 对角 16x16 块 GM(ND) -> ub_A(NZ 块对角)
        uint64_t src_blk_stride = static_cast<uint64_t>(row_stride / 16 - 1);
        
        bool last_chunk_unalign = (cur != chunk_size) && (last_chunk_actual_size != last_chunk_size);
        // uint64_t cur_chunk = last_chunk_actual_size == cur ? cur : last_chunk_size;
        uint64_t repeat = CeilDiv(cur, 16);


        if(last_chunk_unalign){
            for (uint64_t i = 0; i < repeat; i++) {
                uint64_t srcOffset = i * (16 * static_cast<uint64_t>(row_stride) + 16);
                uint64_t dstOffset = i * (static_cast<uint64_t>(cur) * 16 + 16 * 16);
                AscendC::DataCopy(ub_A[dstOffset], gm_a[x_gm_offset + srcOffset],
                              AscendC::DataCopyParams(i == (repeat - 1) ? last_chunk_actual_size % 16 : 16, 
                                1, 
                                src_blk_stride, 
                                0));
            }
        } else {
            for (uint64_t i = 0; i < repeat; i++) {
                uint64_t srcOffset = i * (16 * static_cast<uint64_t>(row_stride) + 16);
                uint64_t dstOffset = i * (static_cast<uint64_t>(cur) * 16 + 16 * 16);
                AscendC::DataCopy(ub_A[dstOffset], gm_a[x_gm_offset + srcOffset],
                              AscendC::DataCopyParams(16, 
                                1, 
                                src_blk_stride, 
                                0));
            }
        }
        
        for (uint64_t i = 0; i < repeat; i++) {
            uint64_t srcOffset = i * (16 * static_cast<uint64_t>(row_stride) + 16);
            uint64_t dstOffset = i * (static_cast<uint64_t>(cur) * 16 + 16 * 16);
            AscendC::DataCopy(ub_A[dstOffset], gm_a[x_gm_offset + srcOffset],
                              AscendC::DataCopyParams(i == (repeat-1) ? last_chunk_actual_size % 16 : 16, 
                                1, 
                                src_blk_stride, 
                                0));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        ub_to_l1(l1_Y, ub_A, static_cast<uint32_t>(cur));        // l1_Y = A（块对角）
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sub(ub_I_A, ub_I, ub_A, (int32_t)(cur * cur));  // I - A
        AscendC::PipeBarrier<PIPE_ALL>();
        ub_to_l1(l1_X, ub_I_A, static_cast<uint32_t>(cur));      // l1_X = I - A
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // ---- AIC：MCH 牛顿迭代，求 16x16 对角块逆，Fixpipe 写 gm_out ----
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
                // MCH 块对角逆 -> gm_out(ND)
                // FixpipeL0cToGM(gm_out[x_gm_offset], l0c_X,
                //                static_cast<uint32_t>(cur), static_cast<uint32_t>(row_stride));
                FixpipeL0cToUB(ub_Res, l0c_X, cur);
            } else {
                FixpipeL0cToL1(l1_X, l0c_X, cur);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // ---- Process：AIV/AIC 每 tile 握手流水 ----
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            
            if (sub_block_idx == 0) {

                for (int64_t loop_idx = core_idx / 2; loop_idx < chunk_num_total; loop_idx += num_core) {
                    int64_t x_gm_offset = 0;
                    int64_t cur         = 0;
                    int64_t row_stride  = 0;
                    ComputeTile(loop_idx, x_gm_offset, cur, row_stride);


                    AivMchPrep(loop_idx, cur, x_gm_offset, row_stride);


                    // 通知同组 AIC：数据就绪
                    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(0x2);
                    // // 等待同组 AIC：计算完成（unlock L1 供下个 tile 使用）
                    AscendC::CrossCoreWaitFlag<0x4>(0x0);

                }
            }
        }

        if ASCEND_IS_AIC {
            for (int64_t loop_idx = core_idx; loop_idx < chunk_num_total; loop_idx += num_core) {
                int64_t x_gm_offset = 0;
                int64_t cur         = 0;
                int64_t row_stride  = 0;
                ComputeTile(loop_idx, x_gm_offset, cur, row_stride);

                // 等待同组 AIV：数据就绪
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(0x2);
                
                
                AicMchNewton(cur, x_gm_offset, row_stride);
                // // 通知同组 AIV：计算完成
                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(0x0);
            }
        }
    }

private:
    // GM
    AscendC::GlobalTensor<InDtype>  gm_a;
    AscendC::GlobalTensor<int64_t>  gm_cu_seqlens;
    AscendC::GlobalTensor<int64_t>  gm_chunk_indices;
    AscendC::GlobalTensor<OutDtype> gm_out;

    // UB
    AscendC::LocalTensor<InDtype> ub_A;
    AscendC::LocalTensor<InDtype> ub_I_A;
    AscendC::LocalTensor<InDtype> ub_I;
    AscendC::LocalTensor<InDtype> ub_Zero;
    AscendC::LocalTensor<InDtype> ub_Res;

    // L1
    AscendC::LocalTensor<InDtype> l1_X;
    AscendC::LocalTensor<InDtype> l1_Y;
    AscendC::LocalTensor<InDtype> l1_I;

    // L0
    AscendC::LocalTensor<InDtype> l0a_X;
    AscendC::LocalTensor<InDtype> l0a_Y;
    AscendC::LocalTensor<InDtype> l0b_X;
    AscendC::LocalTensor<InDtype> l0b_Y;
    AscendC::LocalTensor<float>   l0c_X;
    AscendC::LocalTensor<float>   l0c_Y;

    // Tiling
    int64_t batch_size;
    int64_t seq_length;
    int64_t num_head;
    int64_t chunk_size;
    int64_t chunk_num_in_seq;
    int64_t chunk_num_total;
    int64_t mode;

    // Core
    int64_t num_core;
    int64_t core_idx;
    int64_t sub_block_idx;

    // 辅助矩阵当前缓存尺寸（仅尺寸变化时重建）
    int64_t last_chunk_size;
    int64_t last_chunk_actual_size;
};

#endif  // SOLVE_TRIL_H
