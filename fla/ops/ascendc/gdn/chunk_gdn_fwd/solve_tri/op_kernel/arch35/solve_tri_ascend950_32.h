#ifndef SOLVE_TRI_ASCEND950_32_H
#define SOLVE_TRI_ASCEND950_32_H

#include "kernel_operator.h"
#include "solve_tri_ascend950_common.h"
#include "mem.h"

using namespace AscendC;

// ============================================================================
// SolveTri32 —— chunk=32，ascend950：单 Vector VCS + 一层 MBH，全程 FP32
//
// 与 chunk64 相同：只开 Vector0，每个 tile 独立做 16×16 VCS +（actual>16 时）一层
// 16→32 MBH。一轮 AIC 仍调度 2 个 tile，但必须串行完成，不能把两个 32×32 打进
// 一块 64×64 做 MBH：L0C 上 BR 32×32 的 NZ 偏移与 ChannelSplit 布局不一致，
// 大 shape 下会把 tile1 写成垃圾，ATK 双标杆失败。
//
// 工作区仍是 64×64；MBH 的 cur = ChunkAlign(actual) ∈ {16,32}
// L1 清零走 L0C→L1 直写（与 chunk64 FixpipeZeroToL1 相同），不经 GM 中转
// ============================================================================

constexpr uint32_t kChunk32 = 32;
constexpr uint32_t kWsSide32 = 64; // 两个 32×32 打进一块 64×64 工作区
constexpr uint32_t kPacked32 = kVcsPackedElems;
constexpr int32_t kNumFracs32Ws = static_cast<int32_t>(kWsSide32 / 16); // 4
constexpr int32_t kNumMFracs32 = kNumFracs32Ws;
constexpr int32_t kNumNFracs32 = static_cast<int32_t>(kWsSide32 / 8); // 8
constexpr int32_t kBrNzOffL0C32 = (2 * kNumFracs32Ws + 2) * kFracLen;    // 2560
constexpr int32_t kBrNzOffL1_32 = (4 * kNumMFracs32 + 2) * kFracLen8;    // 2304
constexpr uint32_t kWsElems32 = kWsSide32 * kWsSide32;
constexpr int64_t kTilesPerAicBatch32 = 2;

template <typename InDtype, typename OutDtype>
class SolveTri32 {
public:
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                GM_ADDR workspace, const SolveTriTilingData *tilingData)
    {
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
        is_lower = tilingData->isLower;
        total_tokens = tilingData->totalTokens;

        OnChipBuffer buf;
        uint32_t slot = kWsSide32 * kWsSide32 * static_cast<uint32_t>(sizeof(InDtype));

        ub_mbh_I = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        ub_mbh_I_fp32_blk16 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot);
        ub_mbh_I_fp32_blk8 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 3);
        ub_Zero_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 5);
        ub_vcs_I = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(slot * 7);
        ub_vcs_I_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 7 + 2 * 1024);
        ub_idx_b32 = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(slot * 7 + 6 * 1024);

        ub_FullA = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(slot * 7 + 8 * 1024);
        ub_FullA_fp32_blk16 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 8 + 8 * 1024);
        ub_FullA_fp32_blk8 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 10 + 8 * 1024);
        ub_vcs_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(slot * 12 + 8 * 1024);
        ub_vcs_A_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 12 + 10 * 1024);
        ub_vcs_res_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 12 + 14 * 1024);
        ub_vcs_res_fp32_T = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 12 + 18 * 1024);
        ub_leafInvNz16 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 12 + 22 * 1024);
        ub_leafInvNz8 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(slot * 12 + 22 * 1024 +
                                                                           kWsElems32 * sizeof(float));

        l1_I = buf.template GetBuffer<BufferType::ASCEND_CB, float>(0);
        l1_X = buf.template GetBuffer<BufferType::ASCEND_CB, float>(slot * 2);
        l1_Y = buf.template GetBuffer<BufferType::ASCEND_CB, float>(slot * 4);
        l1_MNEG = buf.template GetBuffer<BufferType::ASCEND_CB, float>(slot * 6);
        l1_INPUT = buf.template GetBuffer<BufferType::ASCEND_CB, float>(slot * 8);
        l1_Zero = buf.template GetBuffer<BufferType::ASCEND_CB, float>(slot * 10);

        l0a_X = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(0);
        l0a_Y = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(slot * 2);
        l0b_X = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(0);
        l0b_Y = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(slot * 2);
        l0c_X = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0c_Y = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(slot * 2);
        l0c_Zero = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(slot * 4);

        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        sub_block_idx = AscendC::GetSubBlockIdx();

        int64_t wsCore = core_idx;
        if ASCEND_IS_AIV {
            wsCore = core_idx / 2;
        }
        GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
        gm_ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs) +
                              static_cast<uint64_t>(wsCore) * kWsElems32);
        aux_ready = 0;
    }

    __aicore__ inline void ub_to_l1(AscendC::LocalTensor<float> l1Tensor,
                                    AscendC::LocalTensor<float> ubTensor, uint32_t n)
    {
        AscendC::DataCopy(l1Tensor, ubTensor, AscendC::DataCopyParams(1, n * n / 8, 0, 0));
    }

    __aicore__ inline void CopyGmNzToL1(AscendC::LocalTensor<float> l1Tensor,
                                        AscendC::GlobalTensor<float> gmTensor, uint32_t n)
    {
        AscendC::DataCopy(l1Tensor, gmTensor, AscendC::DataCopyParams(1, n * n / 8, 0, 0));
    }

    __aicore__ inline void FixpipeL0cToGmNzCs(AscendC::GlobalTensor<float> gmTensor,
                                              AscendC::LocalTensor<float> l0CTensor,
                                              uint32_t nSize, uint32_t mSize,
                                              uint32_t srcStride, uint32_t dstStride)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
        p.nSize = nSize;
        p.mSize = mSize;
        p.srcStride = srcStride;
        p.dstStride = dstStride;
        p.quantPre = QuantMode_t::NoQuant;
        p.isChannelSplit = true;
        AscendC::Fixpipe<float, float, CFG_NZ_L1>(gmTensor, l0CTensor, p);
    }

    __aicore__ inline void CopyGmNz64RectToL1(AscendC::LocalTensor<float> l1Tensor,
                                              uint32_t nSize, uint32_t mSize)
    {
        const uint16_t nFracs = static_cast<uint16_t>(nSize / 8);
        const uint16_t mFracs = static_cast<uint16_t>(mSize / 16);
        const uint16_t blkLen = static_cast<uint16_t>(mFracs * (kFracLen8 / 8));
        const uint16_t gap = static_cast<uint16_t>((kNumMFracs32 - mFracs) * (kFracLen8 / 8));
        if (nSize == kWsSide32 && mSize == kWsSide32) {
            CopyGmNzToL1(l1Tensor, gm_ws, kWsSide32);
            return;
        }
        AscendC::DataCopy(l1Tensor, gm_ws, AscendC::DataCopyParams(nFracs, blkLen, gap, gap));
    }

    __aicore__ inline void FixpipeL0cToL1(AscendC::LocalTensor<float> l1Tensor,
                                          AscendC::LocalTensor<float> l0CTensor, uint32_t cur)
    {
        FixpipeL0cToGmNzCs(gm_ws, l0CTensor, cur, cur, cur, kWsSide32 * 8);
        SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
        WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
        CopyGmNz64RectToL1(l1Tensor, cur, cur);
    }

    __aicore__ inline void FixpipeL0cToL1MBH(AscendC::LocalTensor<float> l1Tensor,
                                             AscendC::LocalTensor<float> l0CTensor,
                                             uint32_t n, uint32_t blockSize)
    {
        FixpipeL0cToGmNzCs(gm_ws, l0CTensor, blockSize, blockSize, n, kWsSide32 * 8);
        SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
        WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
        CopyGmNz64RectToL1(l1Tensor, blockSize, blockSize);
    }

    // 全零：L0C→L1 直写，铺满 64×64 工作区（与 chunk64 相同，避免 GM 中转残留）
    __aicore__ inline void FixpipeZeroToL1(AscendC::LocalTensor<float> l1Tensor)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
        p.nSize = kWsSide32;
        p.mSize = kWsSide32;
        p.srcStride = kWsSide32;
        p.dstStride = kWsSide32 * 16;
        p.quantPre = QuantMode_t::NoQuant;
        p.isChannelSplit = false;
        AscendC::Fixpipe<float, float, CFG_NZ_L1>(l1Tensor, l0c_Zero, p);
    }

    __aicore__ inline void FixpipeL0cToGM(AscendC::GlobalTensor<OutDtype> gmTensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t validRows, uint32_t nCols,
                                          uint32_t srcStride, uint32_t dstStride)
    {
        auto p = AscendC::FixpipeParamsV220(nCols, validRows, srcStride, dstStride, false);
        if constexpr (std::is_same_v<OutDtype, half>) {
            p.quantPre = QuantMode_t::F322F16;
        } else {
            p.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<OutDtype, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0CTensor, p);
    }

    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline int64_t ChunkAlign(int64_t cur_chunk)
    {
        if (cur_chunk <= 16) {
            return 16;
        }
        return 32;
    }

    __aicore__ inline void ComputeTile(int64_t loop_idx, int64_t &x_gm_offset,
                                       int64_t &cur_size, int64_t &actual_size)
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
        cur_size = is_last ? ChunkAlign(actual_size) : chunk_size;
    }

    __aicore__ inline void AuxMatrixGen()
    {
        constexpr uint64_t numFracs = kWsSide32 / 16;
        constexpr int32_t chunkElems = static_cast<int32_t>(kWsElems32);
        Duplicate(ub_mbh_I, (InDtype)0, chunkElems);
        for (uint64_t stripIdx = 0; stripIdx < numFracs * 2; stripIdx++) {
            uint64_t fracsIdx = stripIdx / 2;
            uint64_t oldEvenIdx = stripIdx % 2;
            uint64_t diagMask[2] = {
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][0],
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][1]
            };
            uint64_t off = fracsIdx * (kWsSide32 + 16) * 16 + oldEvenIdx * 8 * 16;
            Duplicate(ub_mbh_I[off], (InDtype)1.0f, diagMask, 1, 1, 1);
        }
        AscendC::Cast(ub_mbh_I_fp32_blk16, ub_mbh_I, AscendC::RoundMode::CAST_NONE, chunkElems);
        NzFp32Blk16ToBlk8(ub_mbh_I_fp32_blk8, ub_mbh_I_fp32_blk16, kWsSide32);

        AscendC::DataCopy(ub_vcs_I, ub_mbh_I, AscendC::DataCopyParams(16, 1, 0, 3));
        for (uint64_t i = 1; i < numFracs; i++) {
            AscendC::DataCopy(ub_vcs_I[i * 16], ub_vcs_I, AscendC::DataCopyParams(16, 1, 3, 3));
        }
        AscendC::Cast(ub_vcs_I_fp32, ub_vcs_I, AscendC::RoundMode::CAST_NONE, kPacked32);
        AscendC::Duplicate(ub_idx_b32, (uint32_t)0, 4);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
        for (uint32_t i = 0; i < 4; i++) {
            ub_idx_b32.SetValue(i, (uint32_t)(16 * i));
        }
        SetFlag<AscendC::HardEvent::S_V>(0);
        WaitFlag<AscendC::HardEvent::S_V>(0);
        Duplicate(ub_Zero_fp32, (float)0, chunkElems);
    }

    __aicore__ inline void EnsureAux()
    {
        if (aux_ready != 0) {
            return;
        }
        AuxMatrixGen();
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        ub_to_l1(l1_Zero, ub_Zero_fp32, kWsSide32);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(0x3);
        ub_to_l1(l1_I, ub_mbh_I_fp32_blk8, kWsSide32);
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        aux_ready = 1;
    }

    __aicore__ inline void RunVcsPacked(uint64_t numValidLeaves)
    {
        AscendC::DataCopy(ub_vcs_res_fp32_T, ub_vcs_I_fp32, AscendC::DataCopyParams(1, 128, 0, 0));
        __ubuf__ float *src0Addr = reinterpret_cast<__ubuf__ float *>(ub_vcs_A_fp32.GetPhyAddr());
        __ubuf__ float *src1Addr = reinterpret_cast<__ubuf__ float *>(ub_vcs_res_fp32_T.GetPhyAddr());
        __ubuf__ float *dstAddr = reinterpret_cast<__ubuf__ float *>(ub_vcs_res_fp32_T.GetPhyAddr());
        __ubuf__ uint32_t *idxAddr = reinterpret_cast<__ubuf__ uint32_t *>(ub_idx_b32.GetPhyAddr());
        (void)numValidLeaves;
        MulReduceScatterVF(dstAddr, src0Addr, src1Addr, idxAddr, 4, 64);
        TransposeB32(ub_vcs_res_fp32, ub_vcs_res_fp32_T, kWsSide32);
    }

    __aicore__ inline void ExpandLeavesToWsNz(uint64_t numCurFracs)
    {
        AscendC::Duplicate(ub_leafInvNz16, (float)0, static_cast<int32_t>(kWsElems32));
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (uint64_t i = 0; i < numCurFracs; i++) {
            uint64_t dstOffset = i * ((uint64_t)kWsSide32 * 16 + kFracLen);
            AscendC::DataCopy(ub_leafInvNz16[dstOffset], ub_vcs_res_fp32[i * kFracLen],
                              AscendC::DataCopyParams(1, kFracLen / 8, 0, 0));
        }
        NzFp32Blk16ToBlk8(ub_leafInvNz8, ub_leafInvNz16, kWsSide32);
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void LoadNegAToWs(int64_t x_gm_offset, int64_t row_stride, int64_t actual_size)
    {
        AscendC::Duplicate(ub_FullA, (InDtype)0, static_cast<int32_t>(kWsElems32));
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint32_t>(actual_size);
        p.dValue = kChunk32;
        p.srcDValue = static_cast<uint32_t>(row_stride);
        p.srcNdMatrixStride = 0;
        p.dstNzNStride = 1;
        p.dstNzC0Stride = kWsSide32;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(ub_FullA, gm_a[x_gm_offset], p);
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(ub_FullA, ub_FullA, (InDtype)(-1.0f), static_cast<int32_t>(kWsElems32));
        AscendC::Cast(ub_FullA_fp32_blk16, ub_FullA, AscendC::RoundMode::CAST_NONE,
                      static_cast<int32_t>(kWsElems32));
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        NzFp32Blk16ToBlk8(ub_FullA_fp32_blk8, ub_FullA_fp32_blk16, kWsSide32);
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    // 单 tile：最多 2 个 16×16 叶子，放在 16×64 打包的前半
    __aicore__ inline void AivVcsOneTile(int64_t x_gm_offset, int64_t row_stride,
                                         int64_t cur, int64_t actual_size)
    {
        EnsureAux();
        AscendC::Duplicate(ub_vcs_A, (InDtype)0, static_cast<int32_t>(kPacked32));
        uint16_t src_blk_stride = static_cast<uint16_t>(row_stride / 16 - 1);
        uint16_t des_blk_stride = static_cast<uint16_t>(kWsSide32 / 16 - 1);
        uint64_t num_valid_fracs = static_cast<uint64_t>(CeilDiv(actual_size, 16));
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint64_t i = 0; i < num_valid_fracs; i++) {
            int64_t rows64 = actual_size - static_cast<int64_t>(i) * 16;
            uint16_t rows = static_cast<uint16_t>(rows64 >= 16 ? 16 : rows64);
            uint64_t srcOffset = i * (16 * (uint64_t)row_stride + 16);
            AscendC::DataCopy(ub_vcs_A[i * 16], gm_a[x_gm_offset + srcOffset],
                              AscendC::DataCopyParams(rows, 1, src_blk_stride, des_blk_stride));
        }
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(ub_vcs_A, ub_vcs_A, (InDtype)(-1.0f), kPacked32);
        AscendC::Cast(ub_vcs_A_fp32, ub_vcs_A, AscendC::RoundMode::CAST_NONE, kPacked32);
        RunVcsPacked(num_valid_fracs);
        ExpandLeavesToWsNz(static_cast<uint64_t>(cur / 16));
        if (cur > 16) {
            LoadNegAToWs(x_gm_offset, row_stride, actual_size);
        }
    }

    // 双 tile：4 个 16×16 叶子占满 16×64；FullA 块对角两个 -A_32
    __aicore__ inline void AivVcsTwoTiles(int64_t off0, int64_t actual0,
                                          int64_t off1, int64_t actual1, int64_t row_stride)
    {
        EnsureAux();
        AscendC::Duplicate(ub_vcs_A, (InDtype)0, static_cast<int32_t>(kPacked32));
        uint16_t src_blk_stride = static_cast<uint16_t>(row_stride / 16 - 1);
        uint16_t des_blk_stride = static_cast<uint16_t>(kWsSide32 / 16 - 1);
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        int64_t offs[2] = {off0, off1};
        int64_t actuals[2] = {actual0, actual1};
        for (uint32_t t = 0; t < 2; t++) {
            uint64_t nLeaves = static_cast<uint64_t>(CeilDiv(actuals[t], 16));
            for (uint64_t j = 0; j < nLeaves; j++) {
                uint64_t leaf = static_cast<uint64_t>(t) * 2 + j;
                int64_t rows64 = actuals[t] - static_cast<int64_t>(j) * 16;
                uint16_t rows = static_cast<uint16_t>(rows64 >= 16 ? 16 : rows64);
                uint64_t srcOffset = j * (16 * (uint64_t)row_stride + 16);
                AscendC::DataCopy(ub_vcs_A[leaf * 16], gm_a[offs[t] + srcOffset],
                                  AscendC::DataCopyParams(rows, 1, src_blk_stride, des_blk_stride));
            }
        }
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(ub_vcs_A, ub_vcs_A, (InDtype)(-1.0f), kPacked32);
        AscendC::Cast(ub_vcs_A_fp32, ub_vcs_A, AscendC::RoundMode::CAST_NONE, kPacked32);
        RunVcsPacked(4);
        ExpandLeavesToWsNz(4);

        AscendC::Duplicate(ub_FullA, (InDtype)0, static_cast<int32_t>(kWsElems32));
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint32_t t = 0; t < 2; t++) {
            AscendC::Nd2NzParams p;
            p.ndNum = 1;
            p.nValue = static_cast<uint32_t>(actuals[t]);
            p.dValue = kChunk32;
            p.srcDValue = static_cast<uint32_t>(row_stride);
            p.srcNdMatrixStride = 0;
            p.dstNzNStride = 1;
            p.dstNzC0Stride = kWsSide32;
            p.dstNzMatrixStride = 0;
            uint64_t dstOff = (t == 0) ? 0 : static_cast<uint64_t>(kBrNzOffL0C32);
            AscendC::DataCopy(ub_FullA[dstOff], gm_a[offs[t]], p);
        }
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(ub_FullA, ub_FullA, (InDtype)(-1.0f), static_cast<int32_t>(kWsElems32));
        AscendC::Cast(ub_FullA_fp32_blk16, ub_FullA, AscendC::RoundMode::CAST_NONE,
                      static_cast<int32_t>(kWsElems32));
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        NzFp32Blk16ToBlk8(ub_FullA_fp32_blk8, ub_FullA_fp32_blk16, kWsSide32);
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void ExtractFromUB(AscendC::LocalTensor<float> l1Slot,
                                         int64_t cur, int32_t blockSize, int32_t startBlock)
    {
        int32_t numBlocks = static_cast<int32_t>(cur) / blockSize;
        int32_t nFracsPerBlock = blockSize / 8;
        int32_t mFracsPerBlock = blockSize / 16;
        for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
            for (int32_t fi = 0; fi < mFracsPerBlock; fi++) {
                for (int32_t fj = 0; fj < nFracsPerBlock; fj++) {
                    int32_t fr = blk * mFracsPerBlock + fi;
                    int32_t fc = blk * nFracsPerBlock + fj;
                    int32_t off = (fc * kNumMFracs32 + fr) * kFracLen8;
                    AscendC::DataCopy(l1Slot[off], ub_leafInvNz8[off],
                                      AscendC::DataCopyParams(1, (uint16_t)(kFracLen8 / 8), 0, 0));
                }
            }
        }
    }

    __aicore__ inline void MbhMatmulToL0C(AscendC::LocalTensor<float> l1A, AscendC::LocalTensor<float> l1B,
                                          AscendC::LocalTensor<float> l0A, AscendC::LocalTensor<float> l0B,
                                          AscendC::LocalTensor<float> l0C, int64_t cur, bool initC)
    {
        const uint16_t mFracs = static_cast<uint16_t>(cur / 16);
        const uint16_t nFracs = static_cast<uint16_t>(cur / 8);
        const int32_t n = static_cast<int32_t>(cur);

        AscendC::LoadData2DParamsV2 loadA;
        loadA.mStartPosition = 0;
        loadA.kStartPosition = 0;
        loadA.mStep = mFracs;
        loadA.kStep = nFracs;
        loadA.srcStride = kNumMFracs32;
        loadA.dstStride = mFracs;
        loadA.ifTranspose = false;
        loadA.sid = 0;
        AscendC::LoadData(l0A, l1A, loadA);

        AscendC::LoadData2DParamsV2 loadB;
        loadB.mStartPosition = 0;
        loadB.kStartPosition = 0;
        loadB.mStep = mFracs;
        loadB.kStep = nFracs;
        loadB.srcStride = kNumMFracs32;
        loadB.dstStride = mFracs;
        loadB.ifTranspose = true;
        loadB.sid = 0;
        AscendC::LoadData(l0B, l1B, loadB);

        SetFlag<AscendC::HardEvent::MTE1_M>(0);
        WaitFlag<AscendC::HardEvent::MTE1_M>(0);

        AscendC::MmadParams mmad;
        mmad.m = n;
        mmad.n = n;
        mmad.k = n;
        mmad.cmatrixInitVal = initC;
        mmad.cmatrixSource = false;
        mmad.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmad);
    }

    __aicore__ inline void MbhLevelComputeY(int64_t cur)
    {
        MbhMatmulToL0C(l1_I, l1_I, l0a_X, l0b_X, l0c_X, cur, true);
        MbhMatmulToL0C(l1_X, l1_MNEG, l0a_Y, l0b_Y, l0c_X, cur, false);
        SetFlag<AscendC::HardEvent::M_FIX>(0);
        WaitFlag<AscendC::HardEvent::M_FIX>(0);
        FixpipeL0cToL1(l1_Y, l0c_X, static_cast<uint32_t>(cur));
        AscendC::PipeBarrier<PIPE_ALL>();
        MbhMatmulToL0C(l1_I, l1_X, l0a_X, l0b_X, l0c_Y, cur, true);
        MbhMatmulToL0C(l1_Y, l1_INPUT, l0a_Y, l0b_Y, l0c_Y, cur, false);
        SetFlag<AscendC::HardEvent::M_FIX>(1);
        WaitFlag<AscendC::HardEvent::M_FIX>(1);
    }

    __aicore__ inline void WriteVcsLeafMte3(int64_t actual_size, int64_t x_gm_offset, int64_t row_stride)
    {
        AscendC::Cast(ub_vcs_A, ub_vcs_res_fp32, AscendC::RoundMode::CAST_RINT, kFracLen);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        WriteVcsNzLeafMte3(gm_out, ub_vcs_A, 0,
                           static_cast<uint32_t>(actual_size),
                           static_cast<uint32_t>(row_stride), x_gm_offset);
    }

    __aicore__ inline void AivFinishOneTile(int64_t cur, int64_t actual_size,
                                            int64_t x_gm_offset, int64_t row_stride,
                                            int32_t drvStart, int32_t othStart)
    {
        AscendC::CrossCoreWaitFlag<0x4>(0x1);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if (cur > 16) {
            ub_to_l1(l1_MNEG, ub_FullA_fp32_blk8, kWsSide32);
            ExtractFromUB(l1_X, cur, 16, drvStart);
            ExtractFromUB(l1_INPUT, cur, 16, othStart);
        } else {
            WriteVcsLeafMte3(actual_size, x_gm_offset, row_stride);
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(0x2);
    }

    __aicore__ inline void AicClearAndHandshake()
    {
        FixpipeZeroToL1(l1_X);
        FixpipeZeroToL1(l1_INPUT);
        SetFlag<AscendC::HardEvent::FIX_MTE1>(1);
        WaitFlag<AscendC::HardEvent::FIX_MTE1>(1);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(0x1);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(0x2);
    }

    __aicore__ inline void AicFinishOneTile(int64_t cur, int64_t actual_size,
                                            int64_t x_gm_offset, int64_t row_stride)
    {
        AicClearAndHandshake();
        if (cur == 16) {
            return;
        }
        MbhLevelComputeY(cur);
        FixpipeL0cToGM(gm_out[x_gm_offset], l0c_Y,
                       static_cast<uint32_t>(actual_size), static_cast<uint32_t>(cur),
                       static_cast<uint32_t>(cur), static_cast<uint32_t>(row_stride));
    }

    __aicore__ inline void Process()
    {
        int32_t drvStart = is_lower ? 1 : 0;
        int32_t othStart = is_lower ? 0 : 1;
        int64_t row_stride = (mode == 0 || mode == 3) ? chunk_size : (num_head * chunk_size);
        int64_t totalBatches = CeilDiv(chunk_num_total, kTilesPerAicBatch32);

        if ASCEND_IS_AIV {
            if (sub_block_idx == 0) {
                int64_t aicIdx = core_idx / 2;
                for (int64_t batch = aicIdx; batch < totalBatches; batch += num_core) {
                    int64_t tile0 = batch * kTilesPerAicBatch32;
                    int64_t tile1 = tile0 + 1;
                    int64_t off0 = 0;
                    int64_t cur0 = 0;
                    int64_t actual0 = 0;
                    int64_t off1 = 0;
                    int64_t cur1 = 0;
                    int64_t actual1 = 0;
                    ComputeTile(tile0, off0, cur0, actual0);
                    bool has1 = (tile1 < chunk_num_total);
                    if (has1) {
                        ComputeTile(tile1, off1, cur1, actual1);
                    }

                    AivVcsOneTile(off0, row_stride, cur0, actual0);
                    AivFinishOneTile(cur0, actual0, off0, row_stride, drvStart, othStart);
                    if (has1 && actual1 > 0) {
                        AivVcsOneTile(off1, row_stride, cur1, actual1);
                        AivFinishOneTile(cur1, actual1, off1, row_stride, drvStart, othStart);
                    }
                }
            }
        }

        if ASCEND_IS_AIC {
            for (int64_t batch = core_idx; batch < totalBatches; batch += num_core) {
                int64_t tile0 = batch * kTilesPerAicBatch32;
                int64_t tile1 = tile0 + 1;
                int64_t off0 = 0;
                int64_t cur0 = 0;
                int64_t actual0 = 0;
                int64_t off1 = 0;
                int64_t cur1 = 0;
                int64_t actual1 = 0;
                ComputeTile(tile0, off0, cur0, actual0);
                bool has1 = (tile1 < chunk_num_total);
                if (has1) {
                    ComputeTile(tile1, off1, cur1, actual1);
                }
                if (batch == core_idx) {
                    AscendC::CrossCoreWaitFlag<0x4>(0x3);
                    MbhMatmulToL0C(l1_Zero, l1_Zero, l0a_X, l0b_X, l0c_Zero, kWsSide32, true);
                    SetFlag<AscendC::HardEvent::M_FIX>(0);
                    WaitFlag<AscendC::HardEvent::M_FIX>(0);
                }

                AicFinishOneTile(cur0, actual0, off0, row_stride);
                if (has1 && actual1 > 0) {
                    AicFinishOneTile(cur1, actual1, off1, row_stride);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<int64_t> gm_cu_seqlens;
    AscendC::GlobalTensor<int64_t> gm_chunk_indices;
    AscendC::GlobalTensor<OutDtype> gm_out;
    AscendC::GlobalTensor<float> gm_ws;

    AscendC::LocalTensor<InDtype> ub_mbh_I;
    AscendC::LocalTensor<float> ub_mbh_I_fp32_blk16;
    AscendC::LocalTensor<float> ub_mbh_I_fp32_blk8;
    AscendC::LocalTensor<float> ub_Zero_fp32;
    AscendC::LocalTensor<InDtype> ub_vcs_I;
    AscendC::LocalTensor<float> ub_vcs_I_fp32;
    AscendC::LocalTensor<uint32_t> ub_idx_b32;

    AscendC::LocalTensor<InDtype> ub_FullA;
    AscendC::LocalTensor<float> ub_FullA_fp32_blk16;
    AscendC::LocalTensor<float> ub_FullA_fp32_blk8;
    AscendC::LocalTensor<InDtype> ub_vcs_A;
    AscendC::LocalTensor<float> ub_vcs_A_fp32;
    AscendC::LocalTensor<float> ub_vcs_res_fp32;
    AscendC::LocalTensor<float> ub_vcs_res_fp32_T;
    AscendC::LocalTensor<float> ub_leafInvNz16;
    AscendC::LocalTensor<float> ub_leafInvNz8;

    AscendC::LocalTensor<float> l1_X;
    AscendC::LocalTensor<float> l1_Y;
    AscendC::LocalTensor<float> l1_I;
    AscendC::LocalTensor<float> l1_MNEG;
    AscendC::LocalTensor<float> l1_INPUT;
    AscendC::LocalTensor<float> l1_Zero;

    AscendC::LocalTensor<float> l0a_X;
    AscendC::LocalTensor<float> l0a_Y;
    AscendC::LocalTensor<float> l0b_X;
    AscendC::LocalTensor<float> l0b_Y;
    AscendC::LocalTensor<float> l0c_X;
    AscendC::LocalTensor<float> l0c_Y;
    AscendC::LocalTensor<float> l0c_Zero;

    int64_t seq_length;
    int64_t num_head;
    int64_t chunk_size;
    int64_t chunk_num_in_seq;
    int64_t chunk_num_total;
    int64_t mode;
    int64_t is_lower;
    int64_t total_tokens;

    int64_t num_core;
    int64_t core_idx;
    int64_t sub_block_idx;
    int64_t aux_ready;
};

#endif  // SOLVE_TRI_ASCEND950_32_H
