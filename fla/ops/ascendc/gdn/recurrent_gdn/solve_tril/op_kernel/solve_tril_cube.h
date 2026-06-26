/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRIL_CUBE_H
#define SOLVE_TRIL_CUBE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

// 内联 l0c_to_gm：L0C -> GM (NZ→ND, FP32→FP16)
namespace NsSolveTril {

__aicore__ inline void L0CToGM(AscendC::GlobalTensor<half> gmTensor,
                                AscendC::LocalTensor<float> l0cTensor,
                                uint32_t mTileActual,
                                uint32_t nTileActual,
                                uint32_t srcStride,
                                uint32_t dstStride)
{
    auto intriParams = AscendC::FixpipeParamsV220(nTileActual, // nSize
                                                  mTileActual, // mSize
                                                  srcStride,   // srcStride
                                                  dstStride,   // dstStride
                                                  false);      // enRelu
    intriParams.quantPre = QuantMode_t::F322F16;
    AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, intriParams);
}

}  // namespace NsSolveTril


#include "solve_tril_common.h"

namespace NsSolveTril {

using namespace AscendC;

constexpr int32_t FRAC = 16;
constexpr int32_t FRAC_LEN = FRAC * FRAC;

#if SOLVE_TRIL_MBH_UB_OPT
// L0C->UB 且输出 NZ：预定义 CFG_NZ 是 isToUB=false（L0C->L1），此处定义 isToUB=true 用于 L0C->UB。
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};
#endif

template <int MATRIX_SIZE>
class SolveTrilCube {
    static constexpr int32_t TILE_LEN = MATRIX_SIZE * MATRIX_SIZE;
    static constexpr int32_t NUM_FRACS = MATRIX_SIZE / FRAC;
    static constexpr int32_t L1_SLOT_ELEMS = TILE_LEN;

#if SOLVE_TRIL_PLATFORM_ASCEND950
    // Ascend950: I 和 -I 常驻 UB，按需通过 MTE3 加载到 L1
    static constexpr int32_t SLOT_MNEG  = 0;
    static constexpr int32_t SLOT_X     = 1;
    static constexpr int32_t SLOT_Y     = 2;
    static constexpr int32_t SLOT_INPUT = 3;
    static constexpr int32_t L1_SLOT_COUNT = 4;

    static constexpr int32_t UB_AUX_I_OFF    = 0;
    static constexpr int32_t UB_AUX_INEG_OFF = TILE_LEN;
    static constexpr int32_t UB_AUX_ZERO_OFF = 2 * TILE_LEN;
    static constexpr int32_t UB_WORK_X_OFF   = 3 * TILE_LEN;
    static constexpr int32_t UB_WORK_Y_OFF   = 4 * TILE_LEN;
    static constexpr int32_t UB_WORK_IN_OFF  = 5 * TILE_LEN;
    static constexpr int32_t UB_TOTAL_ELEMS  = 6 * TILE_LEN;
    static constexpr int32_t L1_TOTAL_ELEMS = L1_SLOT_COUNT * L1_SLOT_ELEMS;
#else
    static constexpr int32_t SLOT_INEG = 0;
    static constexpr int32_t SLOT_I = 1;
    static constexpr int32_t SLOT_MNEG = 2;
    static constexpr int32_t SLOT_X = 3;
    static constexpr int32_t SLOT_Y = 4;
    static constexpr int32_t SLOT_INPUT = 5;
    static constexpr int32_t L1_SLOT_COUNT = 6;
    static constexpr int32_t L1_TOTAL_ELEMS = L1_SLOT_COUNT * L1_SLOT_ELEMS;
#endif

    static constexpr int32_t EVT_MTE2_MTE1 = 0;
    static constexpr int32_t EVT_MTE1_M = 0;
    static constexpr int32_t EVT_M_MTE1 = 0;
    static constexpr int32_t EVT_M_FIX = 0;
    static constexpr int32_t EVT_FIX_MTE2 = 0;
    static constexpr int32_t EVT_MTE3_MTE2 = 0;
    static constexpr int32_t EVT_MTE2_MTE3 = 0;

public:
    __aicore__ inline SolveTrilCube() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                GM_ADDR mch_out, GM_ADDR zero_mat, GM_ADDR eye_mat,
                                GM_ADDR x_out, GM_ADDR workspace,
                                const SolveTrilTilingData* tilingData
#if SOLVE_TRIL_MBH_UB_OPT
                                , Catlass::Arch::Resource<Catlass::Arch::Ascend950>* res
#endif
                                );
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneTile(int64_t tileIdx);
    __aicore__ inline int64_t GetTileGMOffset(int64_t tileIdx);
    __aicore__ inline int64_t GetTileValidSize(int64_t tileIdx);
#if !SOLVE_TRIL_PLATFORM_ASCEND950
    __aicore__ inline void PrepareConstants();
#endif
    __aicore__ inline void LoadInputTile(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
    __aicore__ inline void LoadFullInputForMBH(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
    __aicore__ inline void StoreFinalResult(int64_t gmOffset, int64_t validSize = MATRIX_SIZE);
    __aicore__ inline void MCHInvertDiagonal();
    __aicore__ inline void RecursiveMerge();

    __aicore__ inline void ProcessPartialTile(int64_t gmOffset, int64_t validSize);

    __aicore__ inline void MatmulToSlot(int32_t slotA, int32_t slotB, int32_t slotDst, bool initC);
    __aicore__ inline void MatmulToL0C(int32_t slotA, int32_t slotB, bool initC);
    __aicore__ inline void MatmulToL0CTest(int32_t slotA, int32_t slotB, bool initC);
    __aicore__ inline void L0CToSlot(int32_t slotDst);
    __aicore__ inline void ExtractBlocksToSlot(int32_t srcSlot, int32_t dstSlot,
                                                int32_t blockSize, int32_t startBlock);
#if SOLVE_TRIL_PLATFORM_ASCEND950
    __aicore__ inline void GenerateAuxMatricesOnUB();
    __aicore__ inline void LoadAuxToL1(LocalTensor<half> ubSrc, int32_t l1Slot);
    __aicore__ inline void L0CToUB_X();
    __aicore__ inline void LoadUBXToL1(int32_t slotDst);
#endif
#if SOLVE_TRIL_MBH_DEBUG_ONLY && SOLVE_TRIL_PLATFORM_ASCEND950
    // MBH 调试（旧 UB 方案，已禁用）：从 GM 入参把 单位矩阵/全 0 矩阵 加载进 UB
    __aicore__ inline void LoadAuxMatricesFromGM();
#endif
#if SOLVE_TRIL_MBH_DEBUG_ONLY
    // MBH 调试：从 GM 入参把 MCH 输出（块对角逆）加载到 SLOT_X（NZ 格式），平台无关
    __aicore__ inline void LoadMchOutToSlotX(int64_t validSize = MATRIX_SIZE);
#endif
#if SOLVE_TRIL_MBH_DEBUG_ONLY && !SOLVE_TRIL_PLATFORM_ASCEND950
    // MBH 调试（GM-only 数据流）：按块从 GM 源提取到 L1，及 L0C->xGM 写回
    __aicore__ inline void ExtractBlocksFromGM(GlobalTensor<half> srcGM, int32_t dstSlot,
                                                int32_t blockSize, int32_t startBlock);
    __aicore__ inline void SpillL0CToXGM();
#endif
#if SOLVE_TRIL_MBH_UB_OPT
    // arch3510 UB 优化（AIC 侧）：递归结果 Fixpipe L0C->UB(NZ)。
    // mch_out 暂存(GM->UB) 与每层块提取(UB->L1) 由 AIV 负责（见 vector.h）。
    __aicore__ inline void SpillL0CToUB();
#endif
    __aicore__ inline void ClearSlot(int32_t slot);

private:
#if !SOLVE_TRIL_MBH_UB_OPT
    TPipe pipe_;
#endif
    GlobalTensor<half> inputGM_;
    GlobalTensor<half> outputGM_;
    GlobalTensor<half> workspaceGM_;
#if SOLVE_TRIL_MBH_DEBUG_ONLY
    // MBH 调试入参：MCH 输出（块对角逆）、全 0 矩阵、单位矩阵
    GlobalTensor<half> mchOutGM_;
    GlobalTensor<half> zeroMatGM_;
    GlobalTensor<half> eyeMatGM_;
#endif
#if SOLVE_TRIL_PLATFORM_ASCEND950
    // Ascend950: UB 替代 GM scratch，不需要 scratchGM_

    TBuf<TPosition::A1> l1Buf_;
    LocalTensor<half> l1_;
    TBuf<TPosition::A2> l0aBuf_;
    LocalTensor<half> l0a_;
    TBuf<TPosition::B2> l0bBuf_;
    LocalTensor<half> l0b_;
    TBuf<TPosition::CO1> l0cBuf_;
    LocalTensor<float> l0c_;

    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<half> ub_;
    LocalTensor<half> ubI_;
    LocalTensor<half> ubINeg_;
    LocalTensor<half> ubZero_;
    LocalTensor<half> ubWorkX_;
    LocalTensor<half> ubWorkY_;
    LocalTensor<half> ubWorkIn_;
#else
    GlobalTensor<half> scratchGM_;
#if SOLVE_TRIL_MBH_DEBUG_ONLY
    // MBH 调试：X 的 GM 常驻副本。ExtractBlocksToSlot 从这里按块 GM->L1 提取
    // （避免该 arch 上失效的 L1->GM 原始 DataCopy）。
    GlobalTensor<half> xGM_;
#endif
#if SOLVE_TRIL_MBH_UB_OPT
    // arch3510 UB 优化：所有 L1/L0/UB 缓冲来自单个共享 Catlass::Arch::Resource
    //（在 solve_tril.cpp 构造一次、其 ctor 经 pipe.Destroy() 去除 TPipe 隐式同步），
    // AIC/AIV 用相同 GetBufferByByte 偏移 -> L1/UB 物理共享。这里只持有 LocalTensor 句柄。
    LocalTensor<half> l1_;
    LocalTensor<half> l0a_;
    LocalTensor<half> l0b_;
    LocalTensor<float> l0c_;
    LocalTensor<half> xUB_;   // X 常驻 UB(NZ)，AIC Fixpipe 写、AIV 读
#else
    TBuf<TPosition::A1> l1Buf_;
    LocalTensor<half> l1_;
    TBuf<TPosition::A2> l0aBuf_;
    LocalTensor<half> l0a_;
    TBuf<TPosition::B2> l0bBuf_;
    LocalTensor<half> l0b_;
    TBuf<TPosition::CO1> l0cBuf_;
    LocalTensor<float> l0c_;
#endif
#endif

    int64_t totalTiles_;
    int64_t matrixSize_;
    int64_t numHeads_;
    int64_t seqLen_;
    int64_t batchSize_;
    int64_t isLower_;
    int64_t hasCuSeqlens_;
    int64_t tilesPerCore_;
    int64_t aicIdx_;
    int64_t numChunks_;
    int64_t lastChunkValidSize_;
    int64_t isVarlen_;
    int64_t totalChunks_;
    int64_t rowStride_;
    int64_t layoutMode_;
    GlobalTensor<int64_t> cuSeqlensGM_;
    GlobalTensor<int64_t> chunkIndicesGM_;

#if !SOLVE_TRIL_PLATFORM_ASCEND950
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinish_{SYNC_AIC_AIV_FLAG_SOLVE, SYNC_AIV_AIC_FLAG_SOLVE};
#endif
};


// ============ Implementation ============

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::Init(
    GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR mch_out, GM_ADDR zero_mat, GM_ADDR eye_mat,
    GM_ADDR x_out, GM_ADDR workspace,
    const SolveTrilTilingData* tilingData
#if SOLVE_TRIL_MBH_UB_OPT
    , Catlass::Arch::Resource<Catlass::Arch::Ascend950>* res
#endif
    )
{
    totalTiles_ = tilingData->totalTiles;
    matrixSize_ = tilingData->matrixSize;
    numHeads_ = tilingData->numHeads;
    seqLen_ = tilingData->seqLen;
    batchSize_ = tilingData->batchSize;
    isLower_ = tilingData->isLower;
    hasCuSeqlens_ = tilingData->hasCuSeqlens;
    tilesPerCore_ = tilingData->tilesPerCore;
    numChunks_ = tilingData->numChunks;
    lastChunkValidSize_ = tilingData->lastChunkValidSize;
    isVarlen_ = tilingData->isVarlen;
    totalChunks_ = tilingData->totalChunks;
    layoutMode_ = tilingData->layoutMode;

    if (layoutMode_ == 0) {
        rowStride_ = matrixSize_;  // BHTD
    } else {
        rowStride_ = numHeads_ * matrixSize_;  // BSND or THD
    }

    aicIdx_ = GetBlockIdx();

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x_out));
    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace));

#if SOLVE_TRIL_MBH_DEBUG_ONLY
    // MBH 调试入参绑定：mch_out/zero_mat/eye_mat 均为 BT×BT 的 ND 半精度矩阵
    mchOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(mch_out));
    zeroMatGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(zero_mat));
    eyeMatGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(eye_mat));
#endif

    if (isVarlen_) {
        cuSeqlensGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cu_seqlens));
        chunkIndicesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunk_indices));
    }
#if SOLVE_TRIL_PLATFORM_ASCEND950
    // Ascend950: UB 替代 GM scratch，不需要 scratchGM_

    pipe_.InitBuffer(l1Buf_, L1_TOTAL_ELEMS * sizeof(half));
    l1_ = l1Buf_.Get<half>();
    pipe_.InitBuffer(l0aBuf_, TILE_LEN * sizeof(half));
    l0a_ = l0aBuf_.Get<half>();
    pipe_.InitBuffer(l0bBuf_, TILE_LEN * sizeof(half));
    l0b_ = l0bBuf_.Get<half>();
    pipe_.InitBuffer(l0cBuf_, TILE_LEN * sizeof(float));
    l0c_ = l0cBuf_.Get<float>();

    pipe_.InitBuffer(ubBuf_, UB_TOTAL_ELEMS * sizeof(half));
    ub_ = ubBuf_.Get<half>();
    ubI_     = ub_[UB_AUX_I_OFF];
    ubINeg_  = ub_[UB_AUX_INEG_OFF];
    ubZero_  = ub_[UB_AUX_ZERO_OFF];
    ubWorkX_ = ub_[UB_WORK_X_OFF];
    ubWorkY_ = ub_[UB_WORK_Y_OFF];
    ubWorkIn_ = ub_[UB_WORK_IN_OFF];
#else
    int64_t scratchOffset = GM_NUM_SHARED_SLOTS * TILE_LEN + aicIdx_ * TILE_LEN;
    scratchGM_ = workspaceGM_[scratchOffset];
#if SOLVE_TRIL_MBH_DEBUG_ONLY
    // X 的 GM 常驻副本，放在所有 per-core scratch 之后，每核独占 TILE_LEN。
    // 容量由 tiling 的 userWorkspaceSize 预留（见 solve_tril_tiling.cpp）。
    int64_t numCores = static_cast<int64_t>(GetBlockNum());
    int64_t xGmOffset = GM_NUM_SHARED_SLOTS * TILE_LEN
                      + numCores * TILE_LEN
                      + aicIdx_ * TILE_LEN;
    xGM_ = workspaceGM_[xGmOffset];
#endif

#if SOLVE_TRIL_MBH_UB_OPT
    // 来自共享 Resource：各缓冲取所属池偏移 0；xUB_ 取 UB 池偏移 0（AIC/AIV 一致 -> 物理共享）。
    // 不再使用私有 TPipe（已移除 pipe_ 成员）——避免两核 TPipe 隐式同步不匹配导致的 setup 死锁。
    l1_  = res->l1Buf.template GetBufferByByte<half>(0);
    l0a_ = res->l0ABuf.template GetBufferByByte<half>(0);
    l0b_ = res->l0BBuf.template GetBufferByByte<half>(0);
    l0c_ = res->l0CBuf.template GetBufferByByte<float>(0);
    xUB_ = res->ubBuf.template GetBufferByByte<half>(0);
#else
    pipe_.InitBuffer(l1Buf_, L1_TOTAL_ELEMS * sizeof(half));
    l1_ = l1Buf_.Get<half>();
    pipe_.InitBuffer(l0aBuf_, TILE_LEN * sizeof(half));
    l0a_ = l0aBuf_.Get<half>();
    pipe_.InitBuffer(l0bBuf_, TILE_LEN * sizeof(half));
    l0b_ = l0bBuf_.Get<half>();
    pipe_.InitBuffer(l0cBuf_, TILE_LEN * sizeof(float));
    l0c_ = l0cBuf_.Get<float>();
#endif
#endif
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::Process()
{
    int64_t startTile = aicIdx_ * tilesPerCore_;
    int64_t endTile = startTile + tilesPerCore_;

    if (endTile > totalTiles_) endTile = totalTiles_;
    if (startTile >= totalTiles_) {
        return;
    }

#if SOLVE_TRIL_PLATFORM_ASCEND950
#if SOLVE_TRIL_MBH_DEBUG_ONLY
    // MBH 调试：单位矩阵 / 全 0 矩阵 来自接口入参，直接加载进 UB（替代片上生成）
    LoadAuxMatricesFromGM();
#else
    GenerateAuxMatricesOnUB();
#endif
#else
    SyncAll<false>();
    PrepareConstants();
#endif

    for (int64_t t = startTile; t < endTile; t++) {
        ProcessOneTile(t);
    }
}

template <int MATRIX_SIZE>
__aicore__ inline int64_t SolveTrilCube<MATRIX_SIZE>::GetTileGMOffset(int64_t tileIdx)
{
    int64_t H = numHeads_;
    int64_t BT = matrixSize_;

    if (layoutMode_ == 2) {
        int64_t chunk_global_idx = tileIdx / H;
        int64_t h = tileIdx % H;

        int64_t seq_idx = chunkIndicesGM_.GetValue(chunk_global_idx * 2);
        int64_t chunk_in_seq = chunkIndicesGM_.GetValue(chunk_global_idx * 2 + 1);

        int64_t bos = cuSeqlensGM_.GetValue(seq_idx);

        return (bos + chunk_in_seq * BT) * H * BT + h * BT;

    } else if (layoutMode_ == 1) {
        int64_t T = seqLen_;

        int64_t h = tileIdx % H;
        int64_t chunk = (tileIdx / H) % numChunks_;
        int64_t b = tileIdx / (H * numChunks_);

        return b * T * H * BT + chunk * BT * H * BT + h * BT;

    } else {
        int64_t T = seqLen_;
        int64_t chunk = tileIdx % numChunks_;
        int64_t h = (tileIdx / numChunks_) % H;
        int64_t b = tileIdx / (numChunks_ * H);
        return b * H * T * BT + h * T * BT + chunk * BT * BT;
    }
}

template <int MATRIX_SIZE>
__aicore__ inline int64_t SolveTrilCube<MATRIX_SIZE>::GetTileValidSize(int64_t tileIdx)
{
    if (layoutMode_ == 2) {
        int64_t H = numHeads_;
        int64_t BT = matrixSize_;
        int64_t chunk_global_idx = tileIdx / H;

        int64_t seq_idx = chunkIndicesGM_.GetValue(chunk_global_idx * 2);
        int64_t chunk_in_seq = chunkIndicesGM_.GetValue(chunk_global_idx * 2 + 1);

        int64_t bos = cuSeqlensGM_.GetValue(seq_idx);
        int64_t eos = cuSeqlensGM_.GetValue(seq_idx + 1);
        int64_t seq_len = eos - bos;

        int64_t chunk_start = chunk_in_seq * BT;
        int64_t remaining = seq_len - chunk_start;
        return (remaining >= BT) ? BT : remaining;
    } else {
        int64_t chunk;
        if (layoutMode_ == 1) {
            chunk = (tileIdx / numHeads_) % numChunks_;
        } else {
            chunk = tileIdx % numChunks_;
        }
        if (chunk == numChunks_ - 1) {
            return lastChunkValidSize_;
        }
        return matrixSize_;
    }
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ProcessOneTile(int64_t tileIdx)
{
    int64_t gmOffset = GetTileGMOffset(tileIdx);
    int64_t validSize = GetTileValidSize(tileIdx);

    if (validSize < matrixSize_) {
        ProcessPartialTile(gmOffset, validSize);
    } else {
#if SOLVE_TRIL_MBH_DEBUG_ONLY
        // ===== MBH 调试模式：屏蔽 MCH，X 直接来自接口入参 mch_out =====
        // mch_out 已是 BT×BT 的块对角逆矩阵（含 BT/16 个 16×16 对角逆块），
        // 等价于原 MCHInvertDiagonal() 的输出，加载进 SLOT_X 后直接进入 MBH。
        // 其余（-A、I、Zero、块中转）沿用经验证可用的 910b GM 通路。
        // LoadInputTile(gmOffset);   // MCH 输入加载（已屏蔽）
        // MCHInvertDiagonal();       // MCH 求对角块逆（已屏蔽）
#if SOLVE_TRIL_MBH_UB_OPT
        // UB 优化：BT>16 时 mch_out 由 AIV 暂存到 UB(GM->UB nd2nz)，AIC 不再 GM->L1；
        // BT==16 无 MBH 递归，仍需 SLOT_X=mch_out 以做 X×I 写回。
        if constexpr (MATRIX_SIZE == FRAC) {
            LoadMchOutToSlotX();
        }
#else
        LoadMchOutToSlotX();
#endif
#else
        LoadInputTile(gmOffset);
        MCHInvertDiagonal();
#endif

        if constexpr (MATRIX_SIZE > FRAC) {
            LoadFullInputForMBH(gmOffset);
            RecursiveMerge();
        } else {
#if SOLVE_TRIL_PLATFORM_ASCEND950
            LoadAuxToL1(ubI_, SLOT_Y);
            MatmulToL0C(SLOT_X, SLOT_Y, true);
#else
            MatmulToL0C(SLOT_X, SLOT_I, true);
#endif
            SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
            WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
        }
        StoreFinalResult(gmOffset);
    }
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MatmulToL0C(int32_t slotA, int32_t slotB, bool initC)
{
    LoadData2DParams loadParamsA;
    loadParamsA.startIndex = 0;
    loadParamsA.repeatTimes = NUM_FRACS;
    loadParamsA.srcStride = 1;
    loadParamsA.dstGap = 0;
    loadParamsA.ifTranspose = false;
    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        // A(左乘)算子需"块级转置"预交换：经探测确认 cube 实际计算 blockT(OpA)@B，
        // 即左算子的非对角分形会被块转置。故按 (i,k)<-src(k,i) 预交换源分形块，
        // 使 cube 还原出 plain A。NUM_FRACS=1 时 i*NF==i 无影响（故 BT=16/MCH 一直正确）。
        int32_t srcOffsetA = slotA * L1_SLOT_ELEMS + i * NUM_FRACS * FRAC_LEN;
        int32_t dstOffsetA = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0a_[dstOffsetA], l1_[srcOffsetA], loadParamsA);
    }
    LoadData2DParams loadParamsB;
    loadParamsB.startIndex = 0;
    loadParamsB.repeatTimes = NUM_FRACS;
    loadParamsB.srcStride = NUM_FRACS;
    loadParamsB.dstGap = 0;
    loadParamsB.ifTranspose = true;

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t srcOffsetB = slotB * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dstOffsetB = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0b_[dstOffsetB], l1_[srcOffsetB], loadParamsB);
    }

    SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    MmadParams mmadParams;
    mmadParams.m = MATRIX_SIZE;
    mmadParams.n = MATRIX_SIZE;
    mmadParams.k = MATRIX_SIZE;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    Mmad(l0c_, l0a_, l0b_, mmadParams);
}


template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MatmulToL0CTest(int32_t slotA, int32_t slotB, bool initC)
{
    LoadData2DParams loadParamsA;
    loadParamsA.startIndex = 0;
    loadParamsA.repeatTimes = NUM_FRACS;
    loadParamsA.srcStride = NUM_FRACS;
    loadParamsA.dstGap = 0;
    loadParamsA.ifTranspose = false;

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t srcOffsetA = slotA * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dstOffsetA = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0a_[dstOffsetA], l1_[srcOffsetA], loadParamsA);
    }

    LoadData2DParams loadParamsB;
    loadParamsB.startIndex = 0;
    loadParamsB.repeatTimes = NUM_FRACS;
    loadParamsB.srcStride = NUM_FRACS;
    loadParamsB.dstGap = 0;
    loadParamsB.ifTranspose = true;

    for (int32_t i = 0; i < NUM_FRACS; ++i) {
        int32_t srcOffsetB = slotB * L1_SLOT_ELEMS + i * FRAC_LEN;
        int32_t dstOffsetB = i * NUM_FRACS * FRAC_LEN;
        LoadData(l0b_[dstOffsetB], l1_[srcOffsetB], loadParamsB);
    }

    SetFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    WaitFlag<HardEvent::MTE1_M>(EVT_MTE1_M);
    MmadParams mmadParams;
    mmadParams.m = MATRIX_SIZE;
    mmadParams.n = MATRIX_SIZE;
    mmadParams.k = MATRIX_SIZE;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    Mmad(l0c_, l0a_, l0b_, mmadParams);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::L0CToSlot(int32_t slotDst)
{
    int32_t rowStride = MATRIX_SIZE;
#if SOLVE_TRIL_PLATFORM_ASCEND950
    auto intriParams = AscendC::FixpipeParamsV220(
        MATRIX_SIZE, MATRIX_SIZE,
        MATRIX_SIZE, rowStride, false);
    intriParams.quantPre = QuantMode_t::F322F16;
    AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(ubWorkY_, l0c_, intriParams);
    SetFlag<HardEvent::FIX_MTE3>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE3>(EVT_FIX_MTE2);

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = MATRIX_SIZE;
    nd2nzParams.dValue = MATRIX_SIZE;
    nd2nzParams.srcDValue = rowStride;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(l1_[slotDst * L1_SLOT_ELEMS], ubWorkY_, nd2nzParams);
    SetFlag<HardEvent::MTE3_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE3_MTE1>(EVT_MTE2_MTE1);
#else
    NsSolveTril::L0CToGM(
        scratchGM_,
        l0c_,
        MATRIX_SIZE,
        MATRIX_SIZE,
        MATRIX_SIZE,
        rowStride
    );
    SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = MATRIX_SIZE;
    nd2nzParams.dValue = MATRIX_SIZE;
    nd2nzParams.srcDValue = rowStride;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(l1_[slotDst * L1_SLOT_ELEMS], scratchGM_, nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
#endif
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MatmulToSlot(
    int32_t slotA, int32_t slotB, int32_t slotDst, bool initC)
{
    MatmulToL0C(slotA, slotB, initC);
    SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
    WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
    L0CToSlot(slotDst);
}

#if SOLVE_TRIL_PLATFORM_ASCEND950
// ========== Ascend950 新增函数 ==========

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::GenerateAuxMatricesOnUB()
{
    Duplicate(ubZero_, half(0.0f), TILE_LEN);
    Duplicate(ubI_, half(0.0f), TILE_LEN);
    Duplicate(ubINeg_, half(0.0f), TILE_LEN);

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    PipeBarrier<PIPE_V>();

    constexpr int32_t NUM_FRACS_AUX = MATRIX_SIZE / 16;
    for (int32_t fi = 0; fi < NUM_FRACS_AUX; fi++) {
        int32_t baseOff = fi * 16 * MATRIX_SIZE + fi * 16;
        for (int32_t r = 0; r < 16; r++) {
            int32_t diagOff = baseOff + r * MATRIX_SIZE + r;
            ubI_.SetValue(diagOff, half(1.0f));
            ubINeg_.SetValue(diagOff, half(-1.0f));
        }
    }

    SetFlag<HardEvent::V_MTE3>(1);
    WaitFlag<HardEvent::V_MTE3>(1);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadAuxToL1(
    LocalTensor<half> ubSrc, int32_t l1Slot)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = MATRIX_SIZE;
    nd2nzParams.dValue = MATRIX_SIZE;
    nd2nzParams.srcDValue = MATRIX_SIZE;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(l1_[l1Slot * L1_SLOT_ELEMS], ubSrc, nd2nzParams);
    SetFlag<HardEvent::MTE3_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE3_MTE1>(EVT_MTE2_MTE1);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::L0CToUB_X()
{
    auto intriParams = AscendC::FixpipeParamsV220(
        MATRIX_SIZE, MATRIX_SIZE,
        MATRIX_SIZE, MATRIX_SIZE, false);
    intriParams.quantPre = QuantMode_t::F322F16;
    AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(ubWorkX_, l0c_, intriParams);
    SetFlag<HardEvent::FIX_MTE3>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE3>(EVT_FIX_MTE2);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadUBXToL1(int32_t slotDst)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = MATRIX_SIZE;
    nd2nzParams.dValue = MATRIX_SIZE;
    nd2nzParams.srcDValue = MATRIX_SIZE;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(l1_[slotDst * L1_SLOT_ELEMS], ubWorkX_, nd2nzParams);
    SetFlag<HardEvent::MTE3_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE3_MTE1>(EVT_MTE2_MTE1);
}
#endif  // SOLVE_TRIL_PLATFORM_ASCEND950

#if SOLVE_TRIL_MBH_DEBUG_ONLY && SOLVE_TRIL_PLATFORM_ASCEND950
// ========== MBH 调试模式新增函数 ==========

// 从接口入参把 单位矩阵(eye_mat) / 全 0 矩阵(zero_mat) 加载进 UB，
// 并由 单位矩阵 取负得到 -I，替代原 GenerateAuxMatricesOnUB() 的片上生成。
// eye_mat / zero_mat 均为 BT×BT 的 ND 半精度矩阵。
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadAuxMatricesFromGM()
{
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = TILE_LEN * sizeof(half) / 32;  // 32B 对齐的块数
    params.srcStride = 0;
    params.dstStride = 0;

    // GM(ND) -> UB(ND)：单位矩阵 I、全 0 矩阵 Zero
    DataCopy(ubI_, eyeMatGM_, params);
    DataCopy(ubZero_, zeroMatGM_, params);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    // -I = (-1) * I（在 UB 上用 vector 取负，避免再引入一个 GM 入参）
    Muls(ubINeg_, ubI_, half(-1.0f), TILE_LEN);

    // 确保 ubI_/ubINeg_/ubZero_ 全部就绪后，再由后续 MTE3（UB->L1）读取
    PipeBarrier<PIPE_ALL>();
}
#endif  // SOLVE_TRIL_MBH_DEBUG_ONLY && SOLVE_TRIL_PLATFORM_ASCEND950

#if SOLVE_TRIL_MBH_DEBUG_ONLY
// 将接口入参 mch_out（BT×BT 块对角逆矩阵，ND 格式）加载进 L1 SLOT_X（NZ 格式），
// 作为 MBH 递归的初始 X，等价于原 MCHInvertDiagonal() 的输出。
// 仅依赖 ClearSlot + GM->L1 的 nd2nz 搬运，平台无关（910b/950 均可用）。
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadMchOutToSlotX(int64_t validSize)
{
    ClearSlot(SLOT_X);
    PipeBarrier<PIPE_MTE2>();

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = static_cast<uint32_t>(validSize);
    nd2nzParams.dValue = static_cast<uint32_t>(validSize);
    nd2nzParams.srcDValue = MATRIX_SIZE;   // mch_out 为独立的 BT×BT ND 矩阵，行步长=BT
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(l1_[SLOT_X * L1_SLOT_ELEMS], mchOutGM_, nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}
#endif  // SOLVE_TRIL_MBH_DEBUG_ONLY

#if SOLVE_TRIL_MBH_DEBUG_ONLY && !SOLVE_TRIL_PLATFORM_ASCEND950
// ---- MBH(GM-only 数据流，规避该 arch 上失效的 L1->GM 直拷) ----
// 本 arch 实测：raw L1->GM DataCopy 为静默 no-op，导致 ExtractBlocksToSlot 失效。
// 解决：X 始终以 GM 副本（mch_out 或 xGM_）为权威源，按块经 GM->L1 nd2nz 提取
//（GM->L1 已验证可用），层间结果用 Fixpipe L0C->GM（已验证）写回 xGM_。

// 把 srcGM(ND, BT×BT) 中选定的对角/非对角块，按 NZ 分形布局提取到 L1 dstSlot。
// 与原 ExtractBlocksToSlot 选块逻辑一致：从 startBlock 起步长 2 取块；只是源在 GM。
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ExtractBlocksFromGM(
    GlobalTensor<half> srcGM, int32_t dstSlot, int32_t blockSize, int32_t startBlock)
{
    int32_t numBlocks = MATRIX_SIZE / blockSize;
    int32_t fracsPerBlock = blockSize / FRAC;

    ClearSlot(dstSlot);
    PipeBarrier<PIPE_MTE2>();

    // 逐 16×16 分形：从 GM(ND) 的 (row,col) 分形位置 -> L1(NZ) 同一分形偏移。
    // GM ND 中分形 (fr,fc) 的元素 (r,c) 偏移 = (fr*FRAC+r)*MATRIX_SIZE + (fc*FRAC+c)。
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = FRAC;
    p.dValue = FRAC;
    p.srcDValue = MATRIX_SIZE;        // GM ND 行步长
    p.srcNdMatrixStride = 0;
    p.dstNzNStride = 1;
    p.dstNzC0Stride = MATRIX_SIZE;    // 与整槽 NZ 布局一致（dstNz 的 C0 跨度=BT）
    p.dstNzMatrixStride = 0;

    for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
        for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
            for (int32_t fj = 0; fj < fracsPerBlock; fj++) {
                int32_t fr = blk * fracsPerBlock + fi;   // 分形行号
                int32_t fc = blk * fracsPerBlock + fj;   // 分形列号
                int32_t srcOff = (fr * FRAC) * MATRIX_SIZE + (fc * FRAC);  // GM ND 偏移
                int32_t dstOff = (fc * NUM_FRACS + fr) * FRAC_LEN;          // L1 NZ 偏移
                DataCopy(l1_[dstSlot * L1_SLOT_ELEMS + dstOff], srcGM[srcOff], p);
            }
        }
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

// L0C(FP32) -> xGM_(FP16, ND)，Fixpipe 直写（已验证 L0C->GM 可用）。供下一层提取。
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::SpillL0CToXGM()
{
    NsSolveTril::L0CToGM(xGM_, l0c_, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
    WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
}
#endif  // SOLVE_TRIL_MBH_DEBUG_ONLY && !SOLVE_TRIL_PLATFORM_ASCEND950

#if SOLVE_TRIL_MBH_UB_OPT
// arch3510 UB 优化（AIC 侧）：递归结果 L0C(FP32) -> xUB_(FP16, NZ)，Fixpipe(F322F16, CFG_NZ_UB)。
// Fixpipe 是 cube 指令，但目标 UB（AIV 的）——这是 arch3510 的 L0C->UB 通路（task1）。
// 用 3 参重载（无 cbufWorkspace）：quantPre=F322F16 是均匀下转(pre-quant)，不需要 deq 标量张量
// （cbufWorkspace 仅在按通道 deq-tensor 量化时承载标量张量；与既有 L0CToGM 的 3 参 F322F16 一致）。
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::SpillL0CToUB()
{
    // L0C(FP32) -> xUB_(FP16, 稠密 zN/NZ)。dstStride 单位=元素，为相邻 N-fractal-column 的跨度。
    // 稠密 zN：每个 N-column-fractal-strip 跨 mSize*16 = NUM_FRACS*FRAC_LEN 元素，故 dstStride 必须
    // = NUM_FRACS*FRAC_LEN，才能让分形 (fr,fc) 落在 (fc*NUM_FRACS+fr)*FRAC_LEN —— 与 nd2nz 暂存布局
    // 及 ExtractFromUB/matmul 的 zN 偏移一致。（之前误用 MATRIX_SIZE，相差 16 倍 -> 仅暂存(nd2nz)的
    // BT=32 通过、含 Fixpipe spill 的 BT>=64 失败。SDK GetGMLen 推导见 kernel_operator_fixpipe_impl.h。）
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> intriParams(
        MATRIX_SIZE,             // nSize
        MATRIX_SIZE,             // mSize
        MATRIX_SIZE,             // srcStride (L0C，与已验证的 L0CToGM 一致)
        NUM_FRACS * FRAC_LEN);   // dstStride (UB zN 稠密列跨度 = NUM_FRACS*256 元素)
    intriParams.quantPre = QuantMode_t::F322F16;
    AscendC::Fixpipe<half, float, CFG_NZ_UB>(xUB_, l0c_, intriParams);
    PipeBarrier<PIPE_ALL>();
}
#endif  // SOLVE_TRIL_MBH_UB_OPT

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ClearSlot(int32_t slot)
{
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = TILE_LEN * sizeof(half) / 32;
    params.srcStride = 0;
    params.dstStride = 0;
#if SOLVE_TRIL_PLATFORM_ASCEND950
    // Ascend950: 清零源是 UB(ubZero_)，走 MTE3 (UB->L1)。
    // 自带 PipeBarrier<PIPE_ALL>，确保清零完成后调用方再向该 slot 写入
    // （后续可能是 MTE2 的 GM->L1 加载或 MTE3 的分形块拷贝），避免 WAW 竞争。
    DataCopy(l1_[slot * L1_SLOT_ELEMS], ubZero_, params);
    PipeBarrier<PIPE_ALL>();
#else
    DataCopy(l1_[slot * L1_SLOT_ELEMS], workspaceGM_[GM_WS_ZERO * TILE_LEN], params);
#endif
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ExtractBlocksToSlot(
    int32_t srcSlot, int32_t dstSlot, int32_t blockSize, int32_t startBlock)
{
    int32_t numBlocks = MATRIX_SIZE / blockSize;
    int32_t fracsPerBlock = blockSize / FRAC;

    ClearSlot(dstSlot);
#if !SOLVE_TRIL_PLATFORM_ASCEND950
    SetFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
#endif
    // 注：950 上 ClearSlot 内部已 PipeBarrier<PIPE_ALL>，此处无需额外同步。

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    copyParams.blockLen = FRAC_LEN * sizeof(half) / 32;
    for (int32_t blk = startBlock; blk < numBlocks; blk += 2) {
        for (int32_t fi = 0; fi < fracsPerBlock; fi++) {
            for (int32_t fj = 0; fj < fracsPerBlock; fj++) {
                int32_t row = blk * fracsPerBlock + fi;
                int32_t col = blk * fracsPerBlock + fj;
                int32_t off = (col * NUM_FRACS + row) * FRAC_LEN;
#if SOLVE_TRIL_PLATFORM_ASCEND950
                // Ascend950: UB<->L1 经 MTE3 直通（无 GM 中转）。
                // 两次 DataCopy（L1->UB 再 UB->L1）对同一 ubWorkY_ 先写后读，
                // 必须严格串行；原 MTE3_V/V_MTE3 跨 V 流水的 flag 并不能约束两条
                // MTE 搬运指令，存在数据竞争。此处用 PipeBarrier<PIPE_ALL> 保证次序。
                DataCopy(ubWorkY_, l1_[srcSlot * L1_SLOT_ELEMS + off], copyParams);
                PipeBarrier<PIPE_ALL>();
                DataCopy(l1_[dstSlot * L1_SLOT_ELEMS + off], ubWorkY_, copyParams);
                PipeBarrier<PIPE_ALL>();
#else
                DataCopy(scratchGM_, l1_[srcSlot * L1_SLOT_ELEMS + off], copyParams);
                SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                DataCopy(l1_[dstSlot * L1_SLOT_ELEMS + off], scratchGM_, copyParams);
                SetFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
                WaitFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
#endif
            }
        }
    }
}


#if !SOLVE_TRIL_PLATFORM_ASCEND950
template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::PrepareConstants()
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = MATRIX_SIZE;
    nd2nzParams.dValue = MATRIX_SIZE;
    nd2nzParams.srcDValue = MATRIX_SIZE;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(l1_[SLOT_I * L1_SLOT_ELEMS], workspaceGM_[GM_WS_I * TILE_LEN], nd2nzParams);
    DataCopy(l1_[SLOT_INEG * L1_SLOT_ELEMS], workspaceGM_[GM_WS_INEG * TILE_LEN], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}
#endif

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadInputTile(int64_t gmOffset, int64_t validSize)
{
    ClearSlot(SLOT_INPUT);
    PipeBarrier<PIPE_MTE2>();

    int32_t numDiagFracs = (static_cast<int32_t>(validSize) + FRAC - 1) / FRAC;

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = numDiagFracs;
    nd2nzParams.nValue = FRAC;
    nd2nzParams.dValue = FRAC;
    nd2nzParams.srcDValue = static_cast<uint32_t>(rowStride_);
    nd2nzParams.srcNdMatrixStride = FRAC * static_cast<int32_t>(rowStride_) + FRAC;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = FRAC;
    nd2nzParams.dstNzMatrixStride = (NUM_FRACS + 1) * FRAC_LEN;

    DataCopy(l1_[SLOT_INPUT * L1_SLOT_ELEMS], inputGM_[gmOffset], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::StoreFinalResult(int64_t gmOffset, int64_t validSize)
{
    NsSolveTril::L0CToGM(
        outputGM_[gmOffset],
        l0c_,
        static_cast<uint32_t>(validSize),
        static_cast<uint32_t>(validSize),
        MATRIX_SIZE,
        static_cast<uint32_t>(rowStride_)
    );
    PipeBarrier<PIPE_FIX>();
}


template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::MCHInvertDiagonal()
{
    MatmulToSlot(SLOT_INPUT, SLOT_INPUT, SLOT_Y, true);
    PipeBarrier<PIPE_ALL>();

#if SOLVE_TRIL_PLATFORM_ASCEND950
    LoadAuxToL1(ubI_, SLOT_MNEG);
    MatmulToL0C(SLOT_MNEG, SLOT_MNEG, true);
#else
    MatmulToL0C(SLOT_I, SLOT_I, true);
#endif
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
#if SOLVE_TRIL_PLATFORM_ASCEND950
    LoadAuxToL1(ubINeg_, SLOT_MNEG);
    MatmulToSlot(SLOT_MNEG, SLOT_INPUT, SLOT_X, false);
#else
    MatmulToSlot(SLOT_INEG, SLOT_INPUT, SLOT_X, false);
#endif

    constexpr int32_t NUM_ITERS = 3;
    for (int32_t iter = 0; iter < NUM_ITERS - 1; iter++) {
        MatmulToL0C(SLOT_X, SLOT_Y, true);
        SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
        WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
        SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
        WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
#if SOLVE_TRIL_PLATFORM_ASCEND950
        LoadAuxToL1(ubI_, SLOT_MNEG);
        MatmulToSlot(SLOT_X, SLOT_MNEG, SLOT_X, false);
#else
        MatmulToSlot(SLOT_X, SLOT_I, SLOT_X, false);
#endif
        MatmulToSlot(SLOT_Y, SLOT_Y, SLOT_Y, true);
    }
    MatmulToL0C(SLOT_X, SLOT_Y, true);
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    SetFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
    WaitFlag<HardEvent::M_MTE1>(EVT_MTE1_M);
#if SOLVE_TRIL_PLATFORM_ASCEND950
    LoadAuxToL1(ubI_, SLOT_MNEG);
    MatmulToSlot(SLOT_X, SLOT_MNEG, SLOT_X, false);
#else
    MatmulToSlot(SLOT_X, SLOT_I, SLOT_X, false);
#endif
}


template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::LoadFullInputForMBH(int64_t gmOffset, int64_t validSize)
{
    ClearSlot(SLOT_INPUT);
    PipeBarrier<PIPE_MTE2>();

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = static_cast<uint32_t>(validSize);
    nd2nzParams.dValue = static_cast<uint32_t>(validSize);
    nd2nzParams.srcDValue = static_cast<uint32_t>(rowStride_);
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = MATRIX_SIZE;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(l1_[SLOT_INPUT * L1_SLOT_ELEMS], inputGM_[gmOffset], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);

#if SOLVE_TRIL_PLATFORM_ASCEND950
    LoadAuxToL1(ubINeg_, SLOT_Y);
    MatmulToSlot(SLOT_Y, SLOT_INPUT, SLOT_MNEG, true);
#else
    MatmulToSlot(SLOT_INEG, SLOT_INPUT, SLOT_MNEG, true);
#endif
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::RecursiveMerge()
{
#if SOLVE_TRIL_MBH_UB_OPT
    // ===== arch3510 UB 优化（AIC 侧）：cube 做 Mmad + Fixpipe(L0C->UB)，UB<->L1 搬运在 AIV =====
    // 调用前置：LoadFullInputForMBH 已算出 SLOT_MNEG=-A（SLOT_INPUT 用完即释放）。
    // X 常驻 xUB_(UB,NZ)：初值由 AIV 把 mch_out GM->UB(nd2nz) 暂存；每层结果由本核 SpillL0CToUB 写回。
    // 每层 drv/oth 两组块由 AIV 经 raw UB->L1 提取到 SLOT_X / SLOT_INPUT（清零也在 AIV）。
    // A、F 步同取 drv，故复用 SLOT_X（X 在 UB，L1 SLOT_X 空闲）。与 GM 通路代数等价；每 step 全 PipeBarrier 串行。
    // 握手（每层各 1 次，双核计数 = 层数，匹配）：
    //   AIC SetFlag(AicReady)：常量就绪(L0)/本层结果已写回 xUB_(L>=1) -> AIV 可提取；
    //   AIC WaitFlag(AivReady)：AIV 已把 drv->SLOT_X、oth->SLOT_INPUT 提取到 L1 -> 可 Mmad。
    // 握手（1:2 MIX FFTS 计数，见 common.h）：
    //   【跨核同步改用 SyncAll<false> 屏障】CrossCoreFlag 在本 1:2 MIX 上多种拓扑均死锁（实测），
    //   改用本算子已验证可用的 SyncAll<false>（GM 通路靠它做 AIV->GM->AIC 的 aux 交接，11/11 通过）。
    //   每层 2 个屏障：SP1 = xUB_ 就绪(level0 AIV已stage / level>=1 上层AIC已Spill) -> AIV 可提取；
    //                 SP2 = AIV 提取完成(SLOT_X/INPUT 已写 L1) -> AIC 可 Mmad。
    //   AIC 与 AIV(两 subcore) 每 tile 各调用 2L 次 SyncAll，计数恒等 -> 不死锁（单 tile 测试成立）。
    for (int32_t blockSize = FRAC; blockSize < MATRIX_SIZE; blockSize *= 2) {
        bool lastLevel = !(blockSize < MATRIX_SIZE / 2);

        AscendC::SyncAll<false>();   // SP1: 等 xUB_ 就绪后 AIV 提取
        AscendC::SyncAll<false>();   // SP2: 等 AIV 提取完成后 AIC matmul
        PipeBarrier<PIPE_ALL>();

        // step B: L0C = I × I = 完整单位阵
        MatmulToL0C(SLOT_I, SLOT_I, true);
        PipeBarrier<PIPE_ALL>();
        // step C: Y = drv(SLOT_X) × (-A) + I -> SLOT_Y（经 L0C->GM scratch->L1，AIC 内部）
        MatmulToSlot(SLOT_X, SLOT_MNEG, SLOT_Y, false);
        PipeBarrier<PIPE_ALL>();
        // step E: L0C = Y × oth(SLOT_INPUT)
        MatmulToL0C(SLOT_Y, SLOT_INPUT, true);
        PipeBarrier<PIPE_ALL>();
        // step G: L0C += I × drv(SLOT_X)
        MatmulToL0C(SLOT_I, SLOT_X, false);
        PipeBarrier<PIPE_ALL>();

        if (!lastLevel) {
            // 层间结果 L0C -> xUB_（Fixpipe，NZ）；下一层 SP1 屏障后 AIV 从 xUB_ 提取
            SpillL0CToUB();
            PipeBarrier<PIPE_ALL>();
        }
        // lastLevel：结果留在 L0C，由 ProcessOneTile 的 StoreFinalResult 写出。
    }
#elif SOLVE_TRIL_MBH_DEBUG_ONLY && !SOLVE_TRIL_PLATFORM_ASCEND950
    // ===== MBH 调试：GM-only 数据流（规避失效的 L1->GM 直拷）=====
    // X 的权威副本在 GM：第 0 层为 mch_out，之后为 xGM_（由 SpillL0CToXGM 写回）。
    // 所有块提取改为 ExtractBlocksFromGM（GM->L1，已验证）。
    GlobalTensor<half> xSrc = mchOutGM_;   // 当前 X 的 GM 源
    for (int32_t blockSize = FRAC; blockSize < MATRIX_SIZE; blockSize *= 2) {
        int32_t drvStart = isLower_ ? 1 : 0;
        int32_t othStart = isLower_ ? 0 : 1;

        // step A: 提取驱动对角块 -> SLOT_Y
        // 注：本调试路径不追求性能，每个 step 之间全量 PipeBarrier<PIPE_ALL> 强制串行，
        //   规避本 arch 上 --cce-auto-sync=off 下多分形跨 step 的残留同步竞争
        //   （症状：BT=32 欠计、BT=64 重复累加、BT=128 偶然正确等非确定性错误）。
        PipeBarrier<PIPE_ALL>();
        ExtractBlocksFromGM(xSrc, SLOT_Y, blockSize, drvStart);
        PipeBarrier<PIPE_ALL>();
        // step B: L0C = I × I = 完整单位阵
        MatmulToL0C(SLOT_I, SLOT_I, true);
        PipeBarrier<PIPE_ALL>();
        // step C: Y = Y × (-A) + I（累加到 step B 的 L0C=I）
        MatmulToSlot(SLOT_Y, SLOT_MNEG, SLOT_Y, false);
        PipeBarrier<PIPE_ALL>();

        // step D: 提取非驱动对角块(L11_inv 等) -> SLOT_INPUT
        ExtractBlocksFromGM(xSrc, SLOT_INPUT, blockSize, othStart);
        PipeBarrier<PIPE_ALL>();
        // step E: L0C = Y × INPUT
        MatmulToL0C(SLOT_Y, SLOT_INPUT, true);
        PipeBarrier<PIPE_ALL>();
        // step F: 提取驱动对角块 -> SLOT_INPUT（准备累加对角逆）
        ExtractBlocksFromGM(xSrc, SLOT_INPUT, blockSize, drvStart);
        PipeBarrier<PIPE_ALL>();
        // step G: L0C += I × INPUT
        MatmulToL0C(SLOT_I, SLOT_INPUT, false);
        PipeBarrier<PIPE_ALL>();

        if (blockSize < MATRIX_SIZE / 2) {
            // 层间结果 L0C -> xGM_（Fixpipe，已验证），下一层从 xGM_ 提取
            SpillL0CToXGM();
            PipeBarrier<PIPE_ALL>();
            xSrc = xGM_;
        }
    }
#else
    for (int32_t blockSize = FRAC; blockSize < MATRIX_SIZE; blockSize *= 2) {
        int32_t drvStart = isLower_ ? 1 : 0;
        int32_t othStart = isLower_ ? 0 : 1;

        ExtractBlocksToSlot(SLOT_X, SLOT_Y, blockSize, drvStart);

#if SOLVE_TRIL_PLATFORM_ASCEND950
        // 注意：SLOT_MNEG 此时保存着 -A，MBH 整个递归过程都需要它，
        // 不能用作 I 的临时落点。此处用空闲的 SLOT_INPUT 暂存 I（step 4 才会被覆盖）。
        LoadAuxToL1(ubI_, SLOT_INPUT);
        MatmulToL0C(SLOT_INPUT, SLOT_INPUT, true);
#else
        MatmulToL0C(SLOT_I, SLOT_I, true);
#endif
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        MatmulToSlot(SLOT_Y, SLOT_MNEG, SLOT_Y, false);

        ExtractBlocksToSlot(SLOT_X, SLOT_INPUT, blockSize, othStart);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);

        MatmulToL0C(SLOT_Y, SLOT_INPUT, true);
        SetFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        WaitFlag<HardEvent::M_MTE1>(EVT_M_MTE1);
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
        SetFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
        WaitFlag<HardEvent::FIX_MTE2>(EVT_FIX_MTE2);
        ExtractBlocksToSlot(SLOT_X, SLOT_INPUT, blockSize, drvStart);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_MTE1);
#if SOLVE_TRIL_PLATFORM_ASCEND950
        LoadAuxToL1(ubI_, SLOT_Y);
        MatmulToL0C(SLOT_Y, SLOT_INPUT, false);
#else
        MatmulToL0C(SLOT_I, SLOT_INPUT, false);
#endif
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);

        if (blockSize < MATRIX_SIZE / 2) {
#if SOLVE_TRIL_PLATFORM_ASCEND950
            L0CToUB_X();
            LoadUBXToL1(SLOT_X);
#else
            L0CToSlot(SLOT_X);
#endif
            SetFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(EVT_MTE2_MTE3);
        }
    }
#endif
}

template <int MATRIX_SIZE>
__aicore__ inline void SolveTrilCube<MATRIX_SIZE>::ProcessPartialTile(int64_t gmOffset, int64_t validSize)
{
#if SOLVE_TRIL_MBH_DEBUG_ONLY
    // MBH 调试模式：屏蔽 MCH，X 直接来自接口入参 mch_out（只取 validSize 部分）
    // LoadInputTile(gmOffset, validSize);   // MCH 输入加载（已屏蔽）
    // MCHInvertDiagonal();                   // MCH 求对角块逆（已屏蔽）
#if SOLVE_TRIL_MBH_UB_OPT
    // UB 优化：BT>16 时 mch_out 由 AIV 暂存到 UB；BT==16 仍需 SLOT_X=mch_out。
    if constexpr (MATRIX_SIZE == FRAC) {
        LoadMchOutToSlotX(validSize);
    }
#else
    LoadMchOutToSlotX(validSize);
#endif
#else
    LoadInputTile(gmOffset, validSize);
    MCHInvertDiagonal();
#endif

    if constexpr (MATRIX_SIZE > FRAC) {
        LoadFullInputForMBH(gmOffset, validSize);
        RecursiveMerge();
    } else {
#if SOLVE_TRIL_PLATFORM_ASCEND950
        LoadAuxToL1(ubI_, SLOT_Y);
        MatmulToL0C(SLOT_X, SLOT_Y, true);
#else
        MatmulToL0C(SLOT_X, SLOT_I, true);
#endif
        SetFlag<HardEvent::M_FIX>(EVT_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_M_FIX);
    }

    StoreFinalResult(gmOffset, validSize);
}

}  // namespace NsSolveTril

#endif  // SOLVE_TRIL_CUBE_H
