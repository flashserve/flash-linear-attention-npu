#ifndef SOLVE_TRI_ASCEND950_16_H
#define SOLVE_TRI_ASCEND950_16_H

#include "kernel_operator.h"
#include "solve_tri_ascend950_common.h"
#include "mem.h"

using namespace AscendC;

// ============================================================================
// SolveTri16 —— chunk=16，ascend950：纯 Vector FP32 VCS，无 Cube
//
// 一轮 AIC 逻辑核处理 8 个互相独立的 16×16 chunk：
//   Vector0：tile [8b+0, 8b+3]
//   Vector1：tile [8b+4, 8b+7]
// 每个 Vector 把 4 个叶子打成 16×64，走与 chunk64 相同的
// MulReduceScatterVF + TransposeB32，再 MTE3 写出。
//
// Double buffer（搬运 ∥ 计算）：
//   ping/pong 各一套 A / A_fp32 / res / out
//   事件（同一 id 用在不同 HardEvent 上互不冲突）：
//     MTE2_V(slot)  GM→UB 完成，可以开始该 slot 的 Vector 计算
//     V_MTE2(slot)  该 slot 的 InDtype A 已 Cast 完，可被下一轮 MTE2 覆盖
//     V_MTE3(slot)  NZ 结果已 Cast 到 out，可以 MTE3 写 GM
//     MTE3_V(slot)  写 GM 完成，out[slot] 可被下一轮 Cast 覆盖
// ============================================================================

constexpr uint32_t kChunk16 = 16;
constexpr int64_t kTilesPerVec16 = static_cast<int64_t>(kLeavesPerVec);     // 4
constexpr int64_t kTilesPerAicBatch16 = kTilesPerVec16 * 2;                  // 8
constexpr uint32_t kDbStage16 = 2;

template <typename InDtype, typename OutDtype>
class SolveTri16 {
public:
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                GM_ADDR workspace, const SolveTriTilingData *tilingData)
    {
        (void)workspace;
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aGm));
        gm_cu_seqlens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens));
        gm_chunk_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunk_indices));
        gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(outGm));

        seq_length = tilingData->seqLen;
        num_head = tilingData->numHeads;
        chunk_size = tilingData->chunkSize;
        chunk_num_in_seq = tilingData->numChunks;
        chunk_num_total = tilingData->totalTiles;
        mode = tilingData->layoutMode;
        total_tokens = tilingData->totalTokens;

        OnChipBuffer buf;
        // 常驻：VCS 单位阵 + scatter idx
        ub_vcs_I = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        ub_vcs_I_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(2 * 1024);
        ub_idx_b32 = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(6 * 1024);
        ub_gen = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(8 * 1024);

        // ping/pong：InDtype 16×64 = 2KB，fp32 16×64 = 4KB
        constexpr uint32_t kABytes = kVcsPackedElems * static_cast<uint32_t>(sizeof(InDtype));
        constexpr uint32_t kFp32Bytes = kVcsPackedElems * static_cast<uint32_t>(sizeof(float));
        constexpr uint32_t kOutBytes = kVcsPackedElems * static_cast<uint32_t>(sizeof(OutDtype));
        uint32_t off = 12 * 1024;
        for (uint32_t s = 0; s < kDbStage16; s++) {
            ub_A[s] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(off);
            off += kABytes;
            ub_A_fp32[s] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(off);
            off += kFp32Bytes;
            ub_res_fp32[s] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(off);
            off += kFp32Bytes;
            ub_nz_fp32[s] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(off);
            off += kFp32Bytes;
            ub_out[s] = buf.template GetBuffer<BufferType::ASCEND_UB, OutDtype>(off);
            off += kOutBytes;
        }

        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        sub_block_idx = AscendC::GetSubBlockIdx();
        aux_ready = 0;
    }

    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void ComputeTile(int64_t loop_idx, int64_t &x_gm_offset, int64_t &actual_size)
    {
        int64_t seq_idx = 0;
        int64_t chunk_in_seq_idx = 0;
        int64_t head_idx = 0;
        int64_t chunk_idx = 0;
        int64_t local_seq_length = seq_length;
        int64_t local_chunk_num_in_seq = chunk_num_in_seq;

        if (mode == 0) {
            seq_idx = loop_idx / (chunk_num_in_seq * num_head);
            head_idx = (loop_idx / chunk_num_in_seq) % num_head;
            chunk_in_seq_idx = loop_idx % chunk_num_in_seq;
            x_gm_offset = seq_idx * num_head * seq_length * chunk_size +
                          head_idx * seq_length * chunk_size +
                          chunk_in_seq_idx * chunk_size * chunk_size;
        } else if (mode == 1) {
            seq_idx = loop_idx / (chunk_num_in_seq * num_head);
            chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
            head_idx = loop_idx % (chunk_num_in_seq * num_head) % num_head;
            x_gm_offset = seq_idx * seq_length * num_head * chunk_size +
                          chunk_in_seq_idx * chunk_size * num_head * chunk_size +
                          head_idx * chunk_size;
        } else if (mode == 2) {
            chunk_idx = loop_idx / num_head;
            head_idx = loop_idx % num_head;
            seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
            chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
            local_seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
            local_chunk_num_in_seq = CeilDiv(local_seq_length, chunk_size);
            int64_t bos = gm_cu_seqlens.GetValue(seq_idx);
            x_gm_offset = (bos + chunk_in_seq_idx * chunk_size) * num_head * chunk_size +
                          head_idx * chunk_size;
        } else {
            chunk_idx = loop_idx / num_head;
            head_idx = loop_idx % num_head;
            seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
            chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
            local_seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
            local_chunk_num_in_seq = CeilDiv(local_seq_length, chunk_size);
            int64_t bos = gm_cu_seqlens.GetValue(seq_idx);
            x_gm_offset = head_idx * total_tokens * chunk_size +
                          (bos + chunk_in_seq_idx * chunk_size) * chunk_size;
        }

        bool is_last = (chunk_in_seq_idx == (local_chunk_num_in_seq - 1));
        actual_size = is_last ? (local_seq_length - chunk_in_seq_idx * chunk_size) : chunk_size;
    }

    __aicore__ inline void GenLocalVcsAux()
    {
        AscendC::Duplicate(ub_gen, (InDtype)0, kFracLen);
        for (uint64_t stripIdx = 0; stripIdx < 2; stripIdx++) {
            uint64_t oldEvenIdx = stripIdx % 2;
            uint64_t diagMask[2] = {
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][0],
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][1]
            };
            uint64_t off = oldEvenIdx * 8 * 16;
            Duplicate(ub_gen[off], (InDtype)1.0f, diagMask, 1, 1, 1);
        }
        AscendC::DataCopy(ub_vcs_I, ub_gen, AscendC::DataCopyParams(16, 1, 0, 3));
        for (uint64_t i = 1; i < kLeavesPerVec; i++) {
            AscendC::DataCopy(ub_vcs_I[i * 16], ub_vcs_I, AscendC::DataCopyParams(16, 1, 3, 3));
        }
        AscendC::Cast(ub_vcs_I_fp32, ub_vcs_I, AscendC::RoundMode::CAST_NONE, kVcsPackedElems);

        AscendC::Duplicate(ub_idx_b32, (uint32_t)0, 4);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
        for (uint32_t i = 0; i < 4; i++) {
            ub_idx_b32.SetValue(i, (uint32_t)(16 * i));
        }
        SetFlag<AscendC::HardEvent::S_V>(0);
        WaitFlag<AscendC::HardEvent::S_V>(0);
    }

    __aicore__ inline int64_t TileBaseOf(int64_t aicIdx, int64_t localIter)
    {
        int64_t batch = aicIdx + localIter * num_core;
        return batch * kTilesPerAicBatch16 + sub_block_idx * kTilesPerVec16;
    }

    // 把 [tileBase, tileBase+4) 的 16×16 叶子打进 ub_A[slot]（16×64 打包）
    __aicore__ inline void CopyInPackedLeaves(uint32_t slot, int64_t tileBase, int64_t row_stride)
    {
        const uint16_t srcBlkStride = static_cast<uint16_t>(row_stride / 16 - 1);
        const uint16_t packDstStride = static_cast<uint16_t>(kVcsPack / 16 - 1);

        batch_valid[slot] = 0;
        AscendC::Duplicate(ub_A[slot], (InDtype)0, static_cast<int32_t>(kVcsPackedElems));
        SetFlag<AscendC::HardEvent::V_MTE2>(2);
        WaitFlag<AscendC::HardEvent::V_MTE2>(2);
        for (uint32_t li = 0; li < kLeavesPerVec; li++) {
            int64_t tileIdx = tileBase + static_cast<int64_t>(li);
            batch_offset[slot][li] = 0;
            batch_actual[slot][li] = 0;
            if (tileIdx >= chunk_num_total) {
                continue;
            }
            ComputeTile(tileIdx, batch_offset[slot][li], batch_actual[slot][li]);
            if (batch_actual[slot][li] <= 0) {
                continue;
            }
            batch_valid[slot]++;
            AscendC::DataCopy(ub_A[slot][li * 16], gm_a[batch_offset[slot][li]],
                              AscendC::DataCopyParams(static_cast<uint16_t>(batch_actual[slot][li]), 1,
                                                      srcBlkStride, packDstStride));
        }
    }

    __aicore__ inline void ComputePackedLeaves(uint32_t slot)
    {
        AscendC::DataCopy(ub_res_fp32[slot], ub_vcs_I_fp32, AscendC::DataCopyParams(1, 128, 0, 0));
        AscendC::Muls(ub_A[slot], ub_A[slot], (InDtype)(-1.0f), kVcsPackedElems);
        AscendC::Cast(ub_A_fp32[slot], ub_A[slot], AscendC::RoundMode::CAST_NONE, kVcsPackedElems);
        // InDtype A 此后不再使用，允许下一轮 MTE2 覆盖
        SetFlag<AscendC::HardEvent::V_MTE2>(slot);

        __ubuf__ float *src0Addr = reinterpret_cast<__ubuf__ float *>(ub_A_fp32[slot].GetPhyAddr());
        __ubuf__ float *src1Addr = reinterpret_cast<__ubuf__ float *>(ub_res_fp32[slot].GetPhyAddr());
        __ubuf__ float *dstAddr = reinterpret_cast<__ubuf__ float *>(ub_res_fp32[slot].GetPhyAddr());
        __ubuf__ uint32_t *idxAddr = reinterpret_cast<__ubuf__ uint32_t *>(ub_idx_b32.GetPhyAddr());
        MulReduceScatterVF(dstAddr, src0Addr, src1Addr, idxAddr, 4, 64);
        TransposeB32(ub_nz_fp32[slot], ub_res_fp32[slot], kVcsPack);
        AscendC::Cast(ub_out[slot], ub_nz_fp32[slot], AscendC::RoundMode::CAST_RINT, kVcsPackedElems);
    }

    __aicore__ inline void CopyOutPackedLeaves(uint32_t slot, int64_t row_stride)
    {
        for (uint32_t li = 0; li < kLeavesPerVec; li++) {
            if (batch_actual[slot][li] <= 0) {
                continue;
            }
            WriteVcsNzLeafMte3(gm_out, ub_out[slot], li,
                               static_cast<uint32_t>(batch_actual[slot][li]),
                               static_cast<uint32_t>(row_stride), batch_offset[slot][li]);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t row_stride = (mode == 0 || mode == 3) ? chunk_size : (num_head * chunk_size);
        int64_t totalBatches = CeilDiv(chunk_num_total, kTilesPerAicBatch16);

        if ASCEND_IS_AIV {
            int64_t aicIdx = core_idx / 2;
            if (aux_ready == 0) {
                GenLocalVcsAux();
                aux_ready = 1;
            }

            // 本 Vector 的 batch 序列：tileBase = 8*batch + 4*sub
            int64_t myBatches = 0;
            for (int64_t batch = aicIdx; batch < totalBatches; batch += num_core) {
                myBatches++;
            }
            if (myBatches == 0) {
                return;
            }

            // prologue：先搬第 0 轮
            CopyInPackedLeaves(0, TileBaseOf(aicIdx, 0), row_stride);
            SetFlag<AscendC::HardEvent::MTE2_V>(0);

            for (int64_t i = 0; i < myBatches; i++) {
                uint32_t slot = static_cast<uint32_t>(i % 2);
                uint32_t nxt = 1U - slot;

                WaitFlag<AscendC::HardEvent::MTE2_V>(slot);

                // 预取下一轮 GM→UB，与本轮 VF 并行
                if (i + 1 < myBatches) {
                    if (i >= 1) {
                        WaitFlag<AscendC::HardEvent::V_MTE2>(nxt);
                    }
                    CopyInPackedLeaves(nxt, TileBaseOf(aicIdx, i + 1), row_stride);
                    SetFlag<AscendC::HardEvent::MTE2_V>(nxt);
                }

                if (i >= 2) {
                    WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
                }

                if (batch_valid[slot] == 0) {
                    SetFlag<AscendC::HardEvent::V_MTE2>(slot);
                    SetFlag<AscendC::HardEvent::MTE3_V>(slot);
                    continue;
                }

                ComputePackedLeaves(slot);
                SetFlag<AscendC::HardEvent::V_MTE3>(slot);
                WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
                CopyOutPackedLeaves(slot, row_stride);
                SetFlag<AscendC::HardEvent::MTE3_V>(slot);
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint32_t>((myBatches - 1) % 2));
            if (myBatches >= 2) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<uint32_t>((myBatches - 2) % 2));
            }
        }
    }

private:
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<int64_t> gm_cu_seqlens;
    AscendC::GlobalTensor<int64_t> gm_chunk_indices;
    AscendC::GlobalTensor<OutDtype> gm_out;

    AscendC::LocalTensor<InDtype> ub_vcs_I;
    AscendC::LocalTensor<float> ub_vcs_I_fp32;
    AscendC::LocalTensor<uint32_t> ub_idx_b32;
    AscendC::LocalTensor<InDtype> ub_gen;

    AscendC::LocalTensor<InDtype> ub_A[kDbStage16];
    AscendC::LocalTensor<float> ub_A_fp32[kDbStage16];
    AscendC::LocalTensor<float> ub_res_fp32[kDbStage16];
    AscendC::LocalTensor<float> ub_nz_fp32[kDbStage16];
    AscendC::LocalTensor<OutDtype> ub_out[kDbStage16];

    int64_t batch_offset[kDbStage16][kLeavesPerVec];
    int64_t batch_actual[kDbStage16][kLeavesPerVec];
    uint32_t batch_valid[kDbStage16];

    int64_t seq_length;
    int64_t num_head;
    int64_t chunk_size;
    int64_t chunk_num_in_seq;
    int64_t chunk_num_total;
    int64_t mode;
    int64_t total_tokens;

    int64_t num_core;
    int64_t core_idx;
    int64_t sub_block_idx;
    int64_t aux_ready;
};

#endif  // SOLVE_TRI_ASCEND950_16_H
