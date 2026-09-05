#ifndef SOLVE_TRI_ASCEND950_64_H
#define SOLVE_TRI_ASCEND950_64_H

#include "kernel_operator.h"
#include "solve_tri_ascend950_common.h"
#include "mem.h"

using namespace AscendC;

// ============================================================================
// SolveTri64 —— chunk=64，ascend950：单 Vector VCS(32×32) + 一层 MBH，全程 FP32
//
// 备份：solve_tri_ascend950_64_16leaf.h（4×16×16 VCS + 两层 MBH 16→32→64）
//
// 数据流（每个 GM tile 一次）：
//   1) 2 个对角 32×32 叶子按行拼成 ND 32×64（一行 64 个 FP32）
//      一次 MulReduceScatterVF32（scatterCount=2, oneRepeatSize=64）
//      → TransposeB32(width=64) 两次（各 16 行）再按叶收成 2×32×32 NZ → 转 C0=8
//   2) cur<=32：叶子逆 Cast 后 MTE3 写 GM，Cube 不参与
//   3) cur==64：叶子逆散到 L1 对角（偶→INPUT，奇→X），再搬整块 -A 到 l1_MNEG
//   4) Cube 只做一层 MBH：32→64，Fixpipe 量化写出
//
// MBH 一层：
//   Y   = I + X * MNEG
//   Out = X + Y * INPUT
//
// 片上约定：
//   工作区固定 64×64 FP32；尾块 cur=ChunkAlign(actual)∈{16,32,64}
//   Cube 分型 16×8；非零 L0C→L1 仍 ChannelSplit 绕 GM；全零 Fixpipe 直写 L1
//   只开 Vector0（CrossCore mode=0x4）；Kernel 类型 MIX_AIC_1_2
// ============================================================================

constexpr uint32_t kChunk64 = 64;
constexpr int32_t kNumFracs64 = static_cast<int32_t>(kChunk64 / 16); // 4，L0C/UB 16×16
constexpr int32_t kNumMFracs64 = kNumFracs64;                        // 4，FP32 NZ M 向
constexpr int32_t kNumNFracs64 = static_cast<int32_t>(kChunk64 / 8); // 8，FP32 NZ N 向
constexpr uint32_t kWsElems64 = kChunk64 * kChunk64;
constexpr uint32_t kSlotFp16_64 = kChunk64 * kChunk64 * static_cast<uint32_t>(sizeof(half)); // 8KB
constexpr uint32_t kSlotFp32_64 = kWsElems64 * static_cast<uint32_t>(sizeof(float));          // 16KB

template <typename InDtype, typename OutDtype>
class SolveTri64 {
public:
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                GM_ADDR workspace, const TilingData *tilingData,
                                bool workspaceIsUser = false, bool contiguousSchedule = false)
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
        tiles_per_core = tilingData->tilesPerCore;
        mode = tilingData->layoutMode;
        is_lower = tilingData->isLower;
        total_tokens = tilingData->totalTokens;

        OnChipBuffer buf;
        // UB：
        //   [0, 8KB)      InDtype 64×64：aux I / FullA / VCS A
        //   [8KB, 24KB)   fp32 64×64：blk16 / Zero / Transpose tmp
        //   [24KB, 40KB)  fp32 64×64：blk8（I / FullA / 叶子逆）
        //   [40KB, ...)   VCS 常驻 2×32×32 fp32 与 idx
        ub_inDtypeScratch = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        ub_fp32Nz16 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kSlotFp16_64);
        ub_fp32Nz8 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kSlotFp16_64 + kSlotFp32_64);
        ub_vcs_I_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kSlotFp16_64 + 2 * kSlotFp32_64);
        ub_vcs_A_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kSlotFp16_64 + 2 * kSlotFp32_64 +
                                                                           kVcsPackedElems32 * sizeof(float));
        ub_vcs_res_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kSlotFp16_64 + 2 * kSlotFp32_64 +
                                                                             2 * kVcsPackedElems32 * sizeof(float));
        ub_vcs_res_nz = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kSlotFp16_64 + 2 * kSlotFp32_64 +
                                                                           3 * kVcsPackedElems32 * sizeof(float));
        ub_idx_b32 = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(kSlotFp16_64 + 2 * kSlotFp32_64 +
                                                                            4 * kVcsPackedElems32 * sizeof(float));
        ub_vcs_A = ub_inDtypeScratch;
        ub_FullA = ub_inDtypeScratch;
        ub_FullA_fp32_blk16 = ub_fp32Nz16;
        ub_FullA_fp32_blk8 = ub_fp32Nz8;

        uint32_t slot = kSlotFp16_64;
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
        GM_ADDR userWs = workspaceIsUser ? workspace : AscendC::GetUserWorkspace(workspace);
        if (workspaceIsUser) {
            wsCore = 0;
        }
        gm_ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs) +
                              static_cast<uint64_t>(wsCore) * kWsElems64);
        contiguous_schedule = contiguousSchedule;
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
        const uint16_t gap = static_cast<uint16_t>((kNumMFracs64 - mFracs) * (kFracLen8 / 8));
        if (nSize == kChunk64 && mSize == kChunk64) {
            CopyGmNzToL1(l1Tensor, gm_ws, kChunk64);
            return;
        }
        AscendC::DataCopy(l1Tensor, gm_ws, AscendC::DataCopyParams(nFracs, blkLen, gap, gap));
    }

    __aicore__ inline void FixpipeL0cToL1(AscendC::LocalTensor<float> l1Tensor,
                                          AscendC::LocalTensor<float> l0CTensor, uint32_t cur)
    {
        FixpipeL0cToGmNzCs(gm_ws, l0CTensor, cur, cur, cur, kChunk64 * 8);
        SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
        WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
        CopyGmNz64RectToL1(l1Tensor, cur, cur);
    }

    // 全零：L0C→L1 直写，铺满 64×64 工作区
    __aicore__ inline void FixpipeZeroToL1(AscendC::LocalTensor<float> l1Tensor)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
        p.nSize = kChunk64;
        p.mSize = kChunk64;
        p.srcStride = kChunk64;
        p.dstStride = kChunk64 * 16;
        p.quantPre = QuantMode_t::NoQuant;
        p.isChannelSplit = false;
        AscendC::Fixpipe<float, float, CFG_NZ_L1>(l1Tensor, l0c_Zero, p);
    }

    __aicore__ inline void FixpipeL0cToGM(AscendC::GlobalTensor<OutDtype> gmTensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t validRows, uint32_t curSize, uint32_t dstStride)
    {
        auto p = AscendC::FixpipeParamsV220(curSize, validRows, curSize, dstStride, false);
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
        if (cur_chunk <= 16)
            return 16;
        if (cur_chunk <= 32)
            return 32;
        if (cur_chunk <= 64)
            return 64;
        return 128;
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

    // 2 个 32×32 I 按行拼成 32×64；scatter idx = [0, 32]，VF 每轮 Adds iterIdx
    __aicore__ inline void GenLocalVcsAux()
    {
        AscendC::Duplicate(ub_vcs_I_fp32, (float)0, static_cast<int32_t>(kVcsPackedElems32));
        AscendC::Duplicate(ub_idx_b32, (uint32_t)0, 8);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
        for (uint32_t i = 0; i < kVcs32; i++) {
            ub_vcs_I_fp32.SetValue(i * kVcsPack32 + i, 1.0f);
            ub_vcs_I_fp32.SetValue(i * kVcsPack32 + kVcs32 + i, 1.0f);
        }
        ub_idx_b32.SetValue(0, (uint32_t)0);
        ub_idx_b32.SetValue(1, (uint32_t)kVcs32);
        SetFlag<AscendC::HardEvent::S_V>(0);
        WaitFlag<AscendC::HardEvent::S_V>(0);
    }

    // 64×64 NZ I / Zero → FP32 16×8 后上 L1
    __aicore__ inline void AuxMatrixGenFullAndUpload()
    {
        constexpr int32_t chunkElems = static_cast<int32_t>(kWsElems64);
        Duplicate(ub_inDtypeScratch, (InDtype)0, chunkElems);
        for (uint64_t stripIdx = 0; stripIdx < static_cast<uint64_t>(kNumFracs64) * 2; stripIdx++) {
            uint64_t fracsIdx = stripIdx / 2;
            uint64_t oldEvenIdx = stripIdx % 2;
            uint64_t diagMask[2] = {
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][0],
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][1]
            };
            uint64_t off = fracsIdx * (kChunk64 + 16) * 16 + oldEvenIdx * 8 * 16;
            Duplicate(ub_inDtypeScratch[off], (InDtype)1.0f, diagMask, 1, 1, 1);
        }
        AscendC::Cast(ub_fp32Nz16, ub_inDtypeScratch, AscendC::RoundMode::CAST_NONE, chunkElems);
        NzFp32Blk16ToBlk8(ub_fp32Nz8, ub_fp32Nz16, kChunk64);

        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        ub_to_l1(l1_I, ub_fp32Nz8, kChunk64);
        Duplicate(ub_fp32Nz16, (float)0, chunkElems);
        SetFlag<AscendC::HardEvent::V_MTE3>(1);
        WaitFlag<AscendC::HardEvent::V_MTE3>(1);
        ub_to_l1(l1_Zero, ub_fp32Nz16, kChunk64);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(0x3);
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    // 本 Vector：2 个 32×32 对角叶子，打包 32×64 VCS，结果 2×32×32 NZ
    __aicore__ inline void AivVcsLocalLeaves(int64_t x_gm_offset, int64_t row_stride,
                                             int64_t cur, int64_t actual_size)
    {
        const uint64_t numCurLeaves = static_cast<uint64_t>(CeilDiv(cur, static_cast<int64_t>(kVcs32)));
        const uint64_t numValidLeaves = static_cast<uint64_t>(CeilDiv(actual_size, kVcs32));

        AscendC::DataCopy(ub_vcs_res_fp32, ub_vcs_I_fp32,
                          AscendC::DataCopyParams(1, kVcsPackedElems32 / 8, 0, 0));
        if (actual_size < chunk_size) {
            AscendC::Duplicate(ub_vcs_A, (InDtype)0, static_cast<int32_t>(kVcsPackedElems32));
        }

        uint16_t src_blk_stride = static_cast<uint16_t>(row_stride / 16 - 1);
        uint16_t des_blk_stride = static_cast<uint16_t>(kVcsPack32 / 16 - 1);
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint64_t li = 0; li < kLeavesPerVec32; li++) {
            if (li >= numCurLeaves) {
                break;
            }
            if (li >= numValidLeaves) {
                continue;
            }
            int64_t rows64 = actual_size - static_cast<int64_t>(li) * kVcs32;
            uint16_t rows = static_cast<uint16_t>(rows64 >= static_cast<int64_t>(kVcs32) ?
                                                 kVcs32 : rows64);
            uint64_t srcBase = li * (static_cast<uint64_t>(kVcs32) * (uint64_t)row_stride + kVcs32);
            uint64_t dstBase = li * kVcs32;
            for (uint64_t c16 = 0; c16 < 2; c16++) {
                AscendC::DataCopy(ub_vcs_A[dstBase + c16 * 16],
                                  gm_a[x_gm_offset + srcBase + c16 * 16],
                                  AscendC::DataCopyParams(rows, 1, src_blk_stride, des_blk_stride));
            }
        }
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(ub_vcs_A, ub_vcs_A, (InDtype)(-1.0f), kVcsPackedElems32);
        AscendC::Cast(ub_vcs_A_fp32, ub_vcs_A, AscendC::RoundMode::CAST_NONE, kVcsPackedElems32);

        __ubuf__ float *src0Addr = reinterpret_cast<__ubuf__ float *>(ub_vcs_A_fp32.GetPhyAddr());
        __ubuf__ float *src1Addr = reinterpret_cast<__ubuf__ float *>(ub_vcs_res_fp32.GetPhyAddr());
        __ubuf__ float *dstAddr = reinterpret_cast<__ubuf__ float *>(ub_vcs_res_fp32.GetPhyAddr());
        __ubuf__ uint32_t *idxAddr = reinterpret_cast<__ubuf__ uint32_t *>(ub_idx_b32.GetPhyAddr());
        MulReduceScatterVF32(dstAddr, src0Addr, src1Addr, idxAddr, kLeavesPerVec32, kVcsPack32);

        TransposeB32Vcs32(ub_vcs_res_nz, ub_vcs_res_fp32, ub_fp32Nz16);
        for (uint64_t li = 0; li < kLeavesPerVec32; li++) {
            if (li >= numCurLeaves) {
                break;
            }
            uint32_t off = static_cast<uint32_t>(li * kVcs32NzElems);
            NzFp32Blk16ToBlk8(ub_fp32Nz8[off], ub_vcs_res_nz[off], kVcs32);
        }
    }

    __aicore__ inline void AivPrepFullA(int64_t x_gm_offset, int64_t row_stride, int64_t actual_size)
    {
        constexpr int32_t chunkElems = static_cast<int32_t>(kWsElems64);
        if (actual_size < chunk_size) {
            AscendC::Duplicate(ub_FullA, (InDtype)0, chunkElems);
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint32_t>(actual_size);
        p.dValue = kChunk64;
        p.srcDValue = static_cast<uint32_t>(row_stride);
        p.srcNdMatrixStride = 0;
        p.dstNzNStride = 1;
        p.dstNzC0Stride = kChunk64;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(ub_FullA, gm_a[x_gm_offset], p);
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(ub_FullA, ub_FullA, (InDtype)(-1.0f), chunkElems);
        AscendC::Cast(ub_FullA_fp32_blk16, ub_FullA, AscendC::RoundMode::CAST_NONE, chunkElems);
        NzFp32Blk16ToBlk8(ub_FullA_fp32_blk8, ub_FullA_fp32_blk16, kChunk64);
    }

    // cur<=32：Transpose 后的 32×32 NZ 16×16 按 16×16 瓦片写 GM
    __aicore__ inline void WriteVcsLeafMte3(int64_t actual_size, int64_t cur,
                                            int64_t x_gm_offset, int64_t row_stride)
    {
        AscendC::Cast(ub_inDtypeScratch, ub_vcs_res_nz, AscendC::RoundMode::CAST_RINT,
                      static_cast<int32_t>(kVcs32NzElems));
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const uint32_t nWrite = static_cast<uint32_t>(cur);
        const uint32_t rows = static_cast<uint32_t>(actual_size);
        for (uint32_t tr = 0; tr < nWrite; tr += 16) {
            for (uint32_t tc = 0; tc < nWrite; tc += 16) {
                if (tr >= rows) {
                    continue;
                }
                uint32_t nRows = rows - tr;
                if (nRows > 16) {
                    nRows = 16;
                }
                uint32_t fr = tr / 16;
                uint32_t fc = tc / 16;
                uint32_t leafIdx = fc * 2 + fr;
                int64_t gmOff = x_gm_offset + static_cast<int64_t>(tr) * row_stride +
                                static_cast<int64_t>(tc);
                WriteVcsNzLeafMte3(gm_out, ub_inDtypeScratch, leafIdx, nRows,
                                   static_cast<uint32_t>(row_stride), gmOff);
            }
        }
    }

    // 2 个 32×32 叶子逆（16×8）写入 L1 对角：偶数 → INPUT，奇数 → X
    __aicore__ inline void AivScatterLeavesToL1(int64_t cur, int32_t drvStart, int32_t othStart)
    {
        const uint64_t numCurLeaves = static_cast<uint64_t>(CeilDiv(cur, static_cast<int64_t>(kVcs32)));
        const int32_t mFracsLeaf = static_cast<int32_t>(kVcs32 / 16);
        const int32_t nFracsLeaf = static_cast<int32_t>(kVcs32 / 8);

        for (uint64_t li = 0; li < kLeavesPerVec32; li++) {
            if (li >= numCurLeaves) {
                break;
            }
            int32_t startParity = static_cast<int32_t>(li % 2);
            AscendC::LocalTensor<float> l1Slot;
            if (startParity == drvStart) {
                l1Slot = l1_X;
            } else if (startParity == othStart) {
                l1Slot = l1_INPUT;
            } else {
                continue;
            }
            for (int32_t fi = 0; fi < mFracsLeaf; fi++) {
                for (int32_t fj = 0; fj < nFracsLeaf; fj++) {
                    int32_t fr = static_cast<int32_t>(li) * mFracsLeaf + fi;
                    int32_t fc = static_cast<int32_t>(li) * nFracsLeaf + fj;
                    int32_t l1Off = (fc * kNumMFracs64 + fr) * kFracLen8;
                    int32_t srcOff = static_cast<int32_t>(li) * kVcs32NzElems +
                                     (fj * mFracsLeaf + fi) * kFracLen8;
                    AscendC::DataCopy(l1Slot[l1Off], ub_fp32Nz8[srcOff],
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
        loadA.srcStride = kNumMFracs64;
        loadA.dstStride = mFracs;
        loadA.ifTranspose = false;
        loadA.sid = 0;
        AscendC::LoadData(l0A, l1A, loadA);

        AscendC::LoadData2DParamsV2 loadB;
        loadB.mStartPosition = 0;
        loadB.kStartPosition = 0;
        loadB.mStep = mFracs;
        loadB.kStep = nFracs;
        loadB.srcStride = kNumMFracs64;
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

    // 仅一层 32→64：Y = I + X*(-A)，Out = X + Y*INPUT，Fixpipe 写 GM
    __aicore__ inline void MbhLevelAic(int64_t cur, int64_t actual_size, int64_t x_gm_offset,
                                       int64_t row_stride)
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
        FixpipeL0cToGM(gm_out[x_gm_offset], l0c_Y,
                       static_cast<uint32_t>(actual_size), static_cast<uint32_t>(cur),
                       static_cast<uint32_t>(row_stride));
    }

    __aicore__ inline void Process()
    {
        int32_t drvStart = is_lower ? 1 : 0;
        int32_t othStart = is_lower ? 0 : 1;
        int64_t row_stride = (mode == 0 || mode == 3) ? chunk_size : (num_head * chunk_size);

        if ASCEND_IS_AIV {
            if (sub_block_idx == 0) {
                const int64_t begin = contiguous_schedule ? (core_idx / 2) * tiles_per_core : core_idx / 2;
                const int64_t end = contiguous_schedule
                    ? (begin + tiles_per_core < chunk_num_total ? begin + tiles_per_core : chunk_num_total)
                    : chunk_num_total;
                const int64_t step = contiguous_schedule ? 1 : num_core;
                for (int64_t loop_idx = begin; loop_idx < end; loop_idx += step) {
                    int64_t x_gm_offset = 0;
                    int64_t cur = 0;
                    int64_t actual_size = 0;
                    ComputeTile(loop_idx, x_gm_offset, cur, actual_size);

                    if (aux_ready == 0) {
                        GenLocalVcsAux();
                        AuxMatrixGenFullAndUpload();
                        aux_ready = 1;
                    }

                    AivVcsLocalLeaves(x_gm_offset, row_stride, cur, actual_size);

                    AscendC::CrossCoreWaitFlag<0x4>(0x1);
                    SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    if (cur <= 32) {
                        WriteVcsLeafMte3(actual_size, cur, x_gm_offset, row_stride);
                    } else {
                        AivScatterLeavesToL1(cur, drvStart, othStart);
                        SetFlag<AscendC::HardEvent::MTE3_V>(1);
                        WaitFlag<AscendC::HardEvent::MTE3_V>(1);
                        AivPrepFullA(x_gm_offset, row_stride, actual_size);
                        SetFlag<AscendC::HardEvent::V_MTE3>(1);
                        WaitFlag<AscendC::HardEvent::V_MTE3>(1);
                        ub_to_l1(l1_MNEG, ub_FullA_fp32_blk8, kChunk64);
                    }
                    SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(0x2);
                }
            }
        }

        if ASCEND_IS_AIC {
            const int64_t begin = contiguous_schedule ? core_idx * tiles_per_core : core_idx;
            const int64_t end = contiguous_schedule
                ? (begin + tiles_per_core < chunk_num_total ? begin + tiles_per_core : chunk_num_total)
                : chunk_num_total;
            const int64_t step = contiguous_schedule ? 1 : num_core;
            for (int64_t loop_idx = begin; loop_idx < end; loop_idx += step) {
                int64_t x_gm_offset = 0;
                int64_t cur = 0;
                int64_t actual_size = 0;
                ComputeTile(loop_idx, x_gm_offset, cur, actual_size);
                if (loop_idx == begin) {
                    AscendC::CrossCoreWaitFlag<0x4>(0x3);
                    MbhMatmulToL0C(l1_Zero, l1_Zero, l0a_X, l0b_X, l0c_Zero, kChunk64, true);
                    SetFlag<AscendC::HardEvent::M_FIX>(0);
                    WaitFlag<AscendC::HardEvent::M_FIX>(0);
                }

                FixpipeZeroToL1(l1_X);
                FixpipeZeroToL1(l1_INPUT);
                SetFlag<AscendC::HardEvent::FIX_MTE1>(1);
                WaitFlag<AscendC::HardEvent::FIX_MTE1>(1);

                AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(0x1);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(0x2);

                if (cur <= 32) {
                    // 写回已由 AIV MTE3 完成
                } else {
                    MbhLevelAic(cur, actual_size, x_gm_offset, row_stride);
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

    AscendC::LocalTensor<InDtype> ub_inDtypeScratch;
    AscendC::LocalTensor<float> ub_fp32Nz16;
    AscendC::LocalTensor<float> ub_fp32Nz8;
    AscendC::LocalTensor<float> ub_vcs_I_fp32;
    AscendC::LocalTensor<float> ub_vcs_A_fp32;
    AscendC::LocalTensor<float> ub_vcs_res_fp32;
    AscendC::LocalTensor<float> ub_vcs_res_nz;
    AscendC::LocalTensor<uint32_t> ub_idx_b32;
    AscendC::LocalTensor<InDtype> ub_vcs_A;
    AscendC::LocalTensor<InDtype> ub_FullA;
    AscendC::LocalTensor<float> ub_FullA_fp32_blk16;
    AscendC::LocalTensor<float> ub_FullA_fp32_blk8;

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
    int64_t tiles_per_core;
    int64_t mode;
    int64_t is_lower;
    int64_t total_tokens;

    int64_t num_core;
    int64_t core_idx;
    int64_t sub_block_idx;
    int64_t aux_ready;
    bool contiguous_schedule;
};

#endif  // SOLVE_TRI_ASCEND950_64_H
