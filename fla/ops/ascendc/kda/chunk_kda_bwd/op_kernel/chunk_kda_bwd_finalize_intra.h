#ifndef CHUNK_KDA_BWD_FINALIZE_INTRA_H
#define CHUNK_KDA_BWD_FINALIZE_INTRA_H

#include "chunk_kda_bwd_finalize_wy.h"

#ifndef CHUNK_KDA_BWD_C_INTRA_COMMON_H
#define CHUNK_KDA_BWD_C_INTRA_COMMON_H
namespace KDA {

constexpr uint32_t kCIntraChunkSize = 64;
constexpr uint32_t kCIntraHeadsPerWindow = 2;
constexpr uint32_t kCIntraWorkspaceSlots = 4;
constexpr uint32_t kCIntraVecReadyFlag = 2;
constexpr uint32_t kCIntraCubeReadyFlag = 4;
// Keep the mature Intra implementation's local names inside this private
// header; they are not exported outside Kernel C.
constexpr uint32_t kRowBlock = 16;
constexpr uint32_t kChunkSize = kCIntraChunkSize;
constexpr uint32_t kHeadsPerWindow = kCIntraHeadsPerWindow;
constexpr uint32_t kWorkspaceSlots = kCIntraWorkspaceSlots;
constexpr uint32_t kVecToCubeReadyFlag = kCIntraVecReadyFlag;
constexpr uint32_t kCubeToVecReadyFlag = kCIntraCubeReadyFlag;
constexpr float kLn2 = 0.69314718055994530942f;

template <uint32_t K_DIM, bool VARLEN_TND>
struct ProcessRowBlock {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr uint32_t value = kRowBlock;
#else
    static constexpr uint32_t value = kRowBlock;
#endif
};

struct CIntraTask {
    uint32_t batchIdx;
    uint32_t chunkIdx;
    uint32_t begin;
    uint32_t end;
};

__aicore__ inline uint32_t CIntraWorkspaceSlot(
    uint64_t windowIdx, uint32_t headInWindow)
{
    return static_cast<uint32_t>(
        ((windowIdx & 1U) << 1U) + headInWindow);
}

__aicore__ inline CIntraTask GetCIntraTask(
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdCTilingData &tiling, uint32_t taskIdx)
{
    const WyChunkTask task = GetWyChunkTask(
        cuSeqlens, chunkIndices, tiling, taskIdx);
    return {task.batchIdx, task.chunkIdx, task.begin, task.end};
}

__aicore__ inline uint64_t CIntraTensorOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx, uint32_t col = 0)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx) *
                   tiling.keyDim +
               col;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
                tiling.seqlen +
            tokenIdx) *
               tiling.keyDim +
           col;
}

__aicore__ inline uint64_t CIntraMatrixOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx, uint32_t col = 0)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx) *
                   tiling.chunkSize +
               col;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
                tiling.seqlen +
            tokenIdx) *
               tiling.chunkSize +
           col;
}

__aicore__ inline uint64_t CIntraScalarOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx)
{
    if (tiling.isVarLen != 0) {
        return static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx;
    }
    return (static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
               tiling.seqlen +
           tokenIdx;
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_INTRA_COMMON_H


#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_finalize_intra.h"
#endif

#ifndef CHUNK_KDA_BWD_C_INTRA_CUBE_H
#define CHUNK_KDA_BWD_C_INTRA_CUBE_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
namespace KDA {

template <typename LayoutA, uint32_t L1_M, uint32_t L1_K>
class CIntraSingleTileMmad {
private:
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Catlass::Arch::Ascend950;
#else
    using ArchTag = Catlass::Arch::AtlasA2;
#endif
    using RowMajor = Catlass::layout::RowMajor;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, float, LayoutA, float, RowMajor, float, RowMajor>;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, float, typename Copy::LayoutTagL1A>;
    static constexpr int32_t kEventL1A = 0;
    static constexpr int32_t kEventL1B = 1;
    static constexpr int32_t kEventL0A = 0;
    static constexpr int32_t kEventL0B = 1;
    static constexpr int32_t kEventL0C = 0;

public:
    __aicore__ inline explicit CIntraSingleTileMmad(
        Catlass::Arch::Resource<ArchTag> &resource)
        : resource_(resource)
    {
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA &a, TensorB &b, TensorC &c,
        const Catlass::GemmCoord &shape)
    {
        constexpr uint32_t kL1PlaneBytes = 128 * 128 * sizeof(float);
        auto l1AStorage =
            resource_.l1Buf.template GetBufferByByte<float>(0);
        auto l1BStorage =
            resource_.l1Buf.template GetBufferByByte<float>(kL1PlaneBytes);
        auto l0AStorage =
            resource_.l0ABuf.template GetBufferByByte<float>(0);
        auto l0BStorage =
            resource_.l0BBuf.template GetBufferByByte<float>(0);
        auto l0CStorage =
            resource_.l0CBuf.template GetBufferByByte<float>(0);

        const uint32_t m = shape.m() == 1 ? 16 : shape.m();
        // A5 FP32 packed-copy layouts require compile-time physical extents.
        // Build the proven maximum lower/upper base and select the runtime
        // tile from it.
        auto l1ABase = tla::MakeTensor(
            l1AStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL1A>(
                tla::Int<L1_M>{}, tla::Int<L1_K>{}),
            Catlass::Arch::PositionL1{});
        auto l1BBase = tla::MakeTensor(
            l1BStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL1B>(
                tla::Int<L1_K>{}, tla::Int<128>{}),
            Catlass::Arch::PositionL1{});
        auto l1A = GetTile(
            l1ABase, tla::MakeCoord(0, 0),
            tla::MakeShape(m, shape.k()));
        auto l1B = GetTile(
            l1BBase, tla::MakeCoord(0, 0),
            tla::MakeShape(shape.k(), shape.n()));
        typename Copy::template CopyGmToL1A<TensorA> copyGmA;
        typename Copy::template CopyGmToL1B<TensorB> copyGmB;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        copyGmA(l1A, a);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
        copyGmB(l1B, b);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);

        auto l0A = tla::MakeTensor(
            l0AStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL0A>(
                m, shape.k()),
            Catlass::Arch::PositionL0A{});
        auto l0B = tla::MakeTensor(
            l0BStorage,
            tla::MakeLayout<float, typename Copy::LayoutTagL0B>(
                shape.k(), shape.n()),
            Catlass::Arch::PositionL0B{});
        typename Copy::CopyL1ToL0A copyA;
        typename Copy::CopyL1ToL0B copyB;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        copyA(l0A, l1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        copyB(l0B, l1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);

        auto l0C = tla::MakeTensor(
            l0CStorage, tla::MakeLayoutL0C(m, shape.n()),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
        TileMmad mm;
        // The A5 dynamic tile view does not carry sufficient static shape
        // information for the short overload to recover the physical N
        // mapping.  Pass M/N/K explicitly, as in the proven PR294 pipeline.
        mm(l0C, l0A, l0B,
           m, shape.n(), shape.k(), true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C);

        KdaBwdCopyL0CToDst<Copy, TensorC> fix;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C);
        fix(c, l0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
    }

private:
    Catlass::Arch::Resource<ArchTag> &resource_;
};

class ChunkKdaBwdCIntraCubeProcess {
private:
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;

public:
    __aicore__ ChunkKdaBwdCIntraCubeProcess(
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace)
        : cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkKdaBwdCTilingData &tiling)
    {
        tiling_ = tiling;
        AscendC::SetHF32Mode(false);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // BlockMmad used to manage the A5 packed L0C/FixPipe mapping
        // internally.  The direct TileMmadTla path owns that lifecycle.
        AscendC::SetMMLayoutTransform(true);
#endif
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    }

    __aicore__ inline void Process()
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
#else
        using ArchTag = Catlass::Arch::AtlasA2;
#endif
        using RowMajor = Catlass::layout::RowMajor;
        using ColumnMajor = Catlass::layout::ColumnMajor;
        Catlass::Arch::Resource<ArchTag> resource;
        using LowerCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, float, RowMajor, float, RowMajor, float, RowMajor>;
        using UpperCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, float, ColumnMajor, float, RowMajor, float, RowMajor>;
        // Direct packed TileMmad path.  BlockMmad is intentionally not used.
        using LowerMmad = WyTileGemmDirect<ArchTag, float, LowerCopy, 128>;
        using UpperMmad = WyTileGemmDirect<ArchTag, float, UpperCopy, 64>;
        LowerMmad lower(resource);
        UpperMmad upper(resource);

        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t headWindows =
            (static_cast<uint32_t>(tiling_.headNum) + 1U) / 2U;
        const uint64_t groups =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindows;
        uint64_t window = 0;
        for (uint64_t group = core; group < groups;
             group += tiling_.usedCoreNum) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(group / headWindows);
            const uint32_t headBase =
                static_cast<uint32_t>(group % headWindows) * 2U;
            const uint32_t headCount =
                headBase + 1U < static_cast<uint32_t>(tiling_.headNum) ? 2U : 1U;
            const CIntraTask task = GetCIntraTask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx);
            const uint32_t validC = task.end - task.begin;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            const uint32_t processRowBlock = kRowBlock;
#else
            const uint32_t processRowBlock = kRowBlock;
#endif
            for (uint32_t rowStart = 0; rowStart < validC;
                 rowStart += processRowBlock, ++window) {
                const uint32_t validRows =
                    rowStart + processRowBlock <= validC ?
                        processRowBlock : validC - rowStart;
                const uint32_t prefix = rowStart + validRows;
                const uint32_t lowerK = (prefix + 15U) & ~15U;
                const uint32_t future = validC - rowStart;
                for (uint32_t lane = 0; lane < headCount; ++lane) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeReadyFlag_);
#else
                    AscendC::CrossCoreWaitFlag(kCIntraVecReadyFlag);
#endif
                    const uint32_t slot =
                        CIntraWorkspaceSlot(window, lane);
                    const uint64_t slotBase =
                        static_cast<uint64_t>(core) *
                            tiling_.workspaceCoreSize +
                        static_cast<uint64_t>(slot) *
                            tiling_.workspaceSlotSize;
                    uint64_t bSlotBase = slotBase;
                    bool useSharedB = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    useSharedB = tiling_.isVarLen == 0 &&
                                 tiling_.keyDim == 128 &&
                                 tiling_.useGateInKernel != 0 &&
                                 validC == kChunkSize &&
                                 processRowBlock == kRowBlock;
                    if (useSharedB) {
                        const uint64_t groupFirstWindow =
                            window - rowStart / processRowBlock;
                        const uint32_t bSlot =
                            CIntraWorkspaceSlot(groupFirstWindow, lane);
                        bSlotBase =
                            static_cast<uint64_t>(core) *
                                tiling_.workspaceCoreSize +
                            static_cast<uint64_t>(bSlot) *
                                tiling_.workspaceSlotSize;
                    }
#endif
                    // Upper-A/B are padded to the physical row tile.  Keep M
                    // at that tile size even for a varlen tail; Vector-Post
                    // consumes only validRows.  A non-aligned M (for example
                    // six rows) is not a valid Cube tile and corrupts the
                    // final tail chunk.
                    RunLower(
                        lower, slotBase, bSlotBase,
                        lowerK, processRowBlock);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    if (useSharedB) {
                        RunUpperA5Shared(
                            upper, slotBase, bSlotBase,
                            rowStart, future, processRowBlock);
                    } else
#endif
                    {
                        RunUpper(
                            upper, slotBase, future,
                            processRowBlock);
                    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        cubeToVecReadyFlag_);
#else
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        kCIntraCubeReadyFlag);
#endif
                }
            }
        }
        // Drain the reusable single-tile event credits before changing the
        // matrix-layout mode or handing the AIC back to the outer kernel.  A2
        // initializes the same credits in Init(), so it must drain them too;
        // otherwise the next operator launch blocks on the first SetFlag.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        AscendC::SetMMLayoutTransform(false);
#endif
    }

private:
    template <typename Mmad>
    __aicore__ inline void RunLower(
        Mmad &mm, uint64_t slotBase, uint64_t bSlotBase,
        uint32_t lowerK,
        uint32_t processRowBlock)
    {
        Run<RowMajor>(
            mm, slotBase + tiling_.intraALowerOffset,
            bSlotBase + tiling_.intraBLowerOffset,
            slotBase + tiling_.intraResultRegionOffset +
                tiling_.intraResultDqOffset,
            2 * processRowBlock, tiling_.keyDim, lowerK,
            lowerK, tiling_.keyDim, tiling_.keyDim);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    template <typename Mmad>
    __aicore__ inline void RunUpperA5Shared(
        Mmad &mm, uint64_t slotBase, uint64_t bSlotBase,
        uint32_t rowStart, uint32_t future,
        uint32_t processRowBlock)
    {
        const uint32_t reduction = 2 * future;
        const uint32_t reductionOffset = 2 * rowStart;
        Run<ColumnMajor>(
            mm, slotBase + tiling_.intraAUpperOffset,
            bSlotBase + tiling_.intraBUpperOffset,
            slotBase + tiling_.intraResultRegionOffset +
                tiling_.intraResultDkUpperOffset,
            processRowBlock, tiling_.keyDim, reduction,
            2 * kChunkSize, tiling_.keyDim, tiling_.keyDim,
            reductionOffset, reductionOffset,
            processRowBlock, 0, 2 * kChunkSize);
    }
#endif

    template <typename Mmad>
    __aicore__ inline void RunUpper(
        Mmad &mm, uint64_t slotBase, uint32_t future,
        uint32_t processRowBlock)
    {
        const uint32_t reduction = 2 * future;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (processRowBlock == 32) {
            // A5's proven Intra Upper tile is M=16.  The packed A operand is
            // physically [32, 2*future] ColumnMajor, so expose each 16-row
            // view with the original leading dimension.  WyTileGemmDirect
            // then reduces K in 64-wide TileMmadTla slices; no BlockMmad is
            // used and L0B never reaches the unstable 64-KiB full-capacity
            // tile.
            constexpr uint32_t kReductionTile = 64;
            for (uint32_t kOffset = 0; kOffset < reduction;
                 kOffset += kReductionTile) {
                const uint32_t curK =
                    reduction - kOffset < kReductionTile ?
                        reduction - kOffset : kReductionTile;
                const uint64_t partialOffset =
                    static_cast<uint64_t>(kOffset / kReductionTile) *
                    processRowBlock * tiling_.keyDim * sizeof(float);
                for (uint32_t rowOffset = 0; rowOffset < 32;
                     rowOffset += 16) {
                    const uint64_t resultOffset =
                        partialOffset +
                        static_cast<uint64_t>(rowOffset) *
                            tiling_.keyDim * sizeof(float);
                    Run<ColumnMajor>(
                        mm, slotBase + tiling_.intraAUpperOffset,
                        slotBase + tiling_.intraBUpperOffset,
                        slotBase + tiling_.intraResultRegionOffset +
                            tiling_.intraResultDkUpperOffset + resultOffset,
                        16, tiling_.keyDim, curK,
                        reduction, tiling_.keyDim, tiling_.keyDim,
                        kOffset, kOffset, processRowBlock, rowOffset,
                        reduction);
                }
            }
        } else {
            Run<ColumnMajor>(
                mm, slotBase + tiling_.intraAUpperOffset,
                slotBase + tiling_.intraBUpperOffset,
                slotBase + tiling_.intraResultRegionOffset +
                    tiling_.intraResultDkUpperOffset,
                processRowBlock, tiling_.keyDim, reduction,
                reduction, tiling_.keyDim, tiling_.keyDim,
                0, 0, processRowBlock, 0);
        }
#else
        Run<ColumnMajor>(
            mm, slotBase + tiling_.intraAUpperOffset,
            slotBase + tiling_.intraBUpperOffset,
            slotBase + tiling_.intraResultRegionOffset +
                tiling_.intraResultDkUpperOffset,
            processRowBlock, tiling_.keyDim, reduction,
            reduction, tiling_.keyDim, tiling_.keyDim);
#endif
    }

    template <typename LayoutA, typename Mmad>
    __aicore__ inline void Run(
        Mmad &mm, uint64_t aByte, uint64_t bByte, uint64_t cByte,
        uint32_t m, uint32_t n, uint32_t k,
        uint32_t aPhysicalCols, uint32_t bPhysicalCols,
        uint32_t cPhysicalCols,
        uint32_t aColOffset = 0, uint32_t bRowOffset = 0,
        uint32_t aPhysicalRows = 0, uint32_t aRowOffset = 0,
        uint32_t bPhysicalRows = 0)
    {
        using RowMajor = Catlass::layout::RowMajor;
        AscendC::GlobalTensor<float> a;
        AscendC::GlobalTensor<float> b;
        AscendC::GlobalTensor<float> c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + aByte));
        b.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + bByte));
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + cByte));
        const uint32_t physicalRows = aPhysicalRows == 0 ? m : aPhysicalRows;
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<float, LayoutA>(physicalRows, aPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<float, RowMajor>(
                   bPhysicalRows == 0 ? k : bPhysicalRows,
                   bPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<float, RowMajor>(m, cPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto ba = GetTile(
            ta, tla::MakeCoord(aRowOffset, aColOffset), tla::MakeShape(m, k));
        auto bb = GetTile(
            tb, tla::MakeCoord(bRowOffset, 0), tla::MakeShape(k, n));
        auto bc = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        mm(ba, bb, bc, Catlass::GemmCoord{m, n, k});
    }

    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR workspace_;
    ChunkKdaBwdCTilingData tiling_{};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    Catlass::Arch::CrossCoreFlag vecToCubeReadyFlag_{kCIntraVecReadyFlag};
    Catlass::Arch::CrossCoreFlag cubeToVecReadyFlag_{kCIntraCubeReadyFlag};
#endif
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_INTRA_CUBE_H


/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_C_INTRA_VECTOR_H
#define CHUNK_KDA_BWD_C_INTRA_VECTOR_H

#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "catlass/arch/cross_core_sync.hpp"
#endif

namespace KDA {

template <
    uint32_t K_DIM, uint32_t CHUNK_SIZE, bool SAFE_GATE, bool VARLEN_TND,
    bool PUBLIC_VARLEN, typename DataT, typename BetaT, typename RawGateT>
class ChunkKdaBwdCIntraVectorProcess {
public:
    __aicore__ ChunkKdaBwdCIntraVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR gk, GM_ADDR beta, GM_ADDR dAqk, GM_ADDR dAkk,
        GM_ADDR dqBaseRaw, GM_ADDR dq, GM_ADDR dk, GM_ADDR db, GM_ADDR dg,
        GM_ADDR dqOut, GM_ADDR dkOut, GM_ADDR dbOut, GM_ADDR dgOut, GM_ADDR cuSeqlens,
        GM_ADDR chunkMetadata, GM_ADDR rawG, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR dA, GM_ADDR dBias, GM_ADDR workspace)
        : q_(q), k_(k), gk_(gk), beta_(beta), dAqk_(dAqk), dAkk_(dAkk),
          dqBaseRaw_(dqBaseRaw), dq_(dq), dk_(dk), db_(db), dg_(dg),
          dqOut_(dqOut), dkOut_(dkOut), dbOut_(dbOut), dgOut_(dgOut),
          cuSeqlens_(cuSeqlens),
          chunkMetadata_(chunkMetadata),
          rawG_(rawG), aLog_(aLog), dtBias_(dtBias),
          dA_(dA), dBias_(dBias),
          workspace_(workspace)
    {
        static_assert(K_DIM == 128, "Kernel C requires K=128.");
        static_assert(CHUNK_SIZE == 64, "Kernel C requires C=64.");
    }

    __aicore__ inline void Init(const ChunkKdaBwdCTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(q_));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(k_));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk_));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta_));
        dAqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk_));
        dAkkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk_));
        dqBaseRawGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dqBaseRaw_));
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq_));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk_));
        dbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db_));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg_));
        dqOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqOut_));
        dkOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dkOut_));
        dbOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dbOut_));
        dgOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgOut_));
        if (tiling_.useGateInKernel != 0) {
            rawGGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawGateT *>(rawG_));
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog_));
            if (tiling_.hasDtBias != 0) {
                dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias_));
            }
            dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA_));
            if (tiling_.hasDtBias != 0) {
                dBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dBias_));
            }
        }
        if constexpr (VARLEN_TND) {
            chunkMetadataGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t *>(chunkMetadata_));
        }
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_));

        pipe_->InitBuffer(inputQueue_, 2, kIoBufferBytes);
        pipe_->InitBuffer(outputQueue_, 2, kIoBufferBytes);
        pipe_->InitBuffer(matrixInputPing_, kIoBufferBytes);
        pipe_->InitBuffer(matrixInputPong_, kIoBufferBytes);
        pipe_->InitBuffer(arena_, kArenaBytes);
        pipe_->InitBuffer(reduceTmp_, kReduceTmpBytes);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // Dense accumulated-gate fusion keeps row0's complete 32x128 dg
        // contribution until row32 arrives.  Each AIV owns one head, so one
        // 16-KiB resident plane is sufficient per sub-block.
        pipe_->InitBuffer(pendingDg_, kPendingDgBytes);
#endif
        InitMatrixInputEvents();
        if constexpr (VARLEN_TND) {
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
            auto offsets = ScalarGatherOffsets();
            for (uint32_t row = 0; row < kProcessRowBlock; ++row) {
                offsets.SetValue(
                    row, row * (32 / sizeof(float)) * sizeof(float));
                offsets.SetValue(
                    kProcessRowBlock + row,
                    row * (32 / sizeof(DataT)) * sizeof(float));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
#endif
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t coreNum = static_cast<uint32_t>(tiling_.usedCoreNum);
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kHeadsPerWindow - 1) / kHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!PUBLIC_VARLEN && K_DIM == 128 &&
                      kProcessRowBlock == kRowBlock) {
            if (tiling_.isVarLen == 0 &&
                tiling_.useGateInKernel != 0 &&
                tiling_.seqlen % CHUNK_SIZE == 0) {
                ProcessA5DenseRow16Pipeline(
                    coreIdx, coreNum, headNum, headWindowCount,
                    taskGroupCount);
                ReleaseMatrixInputEvents();
                return;
            }
        }
        if constexpr (!PUBLIC_VARLEN && K_DIM == 128 &&
                      kProcessRowBlock == 32) {
            if (tiling_.isVarLen == 0 &&
                tiling_.seqlen % CHUNK_SIZE == 0) {
                ProcessA5DenseFourSlot(
                    coreIdx, coreNum, headNum, headWindowCount,
                    taskGroupCount);
                ReleaseMatrixInputEvents();
                return;
            }
        }
#endif
        uint64_t windowIdx = 0;

        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum) {
            // Decode directly from the logical group exactly as the paired
            // AIC path does.  This is required when varlen tiling selects a
            // usedCoreNum that is not a multiple of headWindowCount.
            const uint32_t taskIdx =
                static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindowIdx =
                static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase = headWindowIdx * kHeadsPerWindow;
            const uint32_t headCount = headBase + 1 < headNum ? 2 : 1;
            const CIntraTask task = GetCIntraTask(
                cuSeqlens_, chunkMetadata_, tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            const uint32_t rowBlockCount =
                (validLen + kProcessRowBlock - 1) / kProcessRowBlock;
            for (uint32_t rowBlock = 0; rowBlock < rowBlockCount; ++rowBlock) {
                const uint32_t rowStart = rowBlock * kProcessRowBlock;
                const uint32_t validRows =
                    rowStart + kProcessRowBlock <= validLen ?
                        kProcessRowBlock : validLen - rowStart;

                // PR190-style two-head stage ordering: finish Vector-Pre for
                // head0 then head1 before entering Vector-Post.  This lets
                // Cube(head0) overlap Vector-Pre(head1).
                for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
                    const uint32_t slot = CIntraWorkspaceSlot(windowIdx, headInWindow);
                    PrepareHead(task, headBase + headInWindow, rowStart, validRows, slot);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                        vecToCubeReadyFlag_);
#else
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kVecToCubeReadyFlag);
#endif
                }
                for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecReadyFlag_);
#else
                    AscendC::CrossCoreWaitFlag(kCubeToVecReadyFlag);
#endif
                    const uint32_t slot = CIntraWorkspaceSlot(windowIdx, headInWindow);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    if constexpr (!PUBLIC_VARLEN && K_DIM == 128 &&
                                  kProcessRowBlock == kRowBlock) {
                        if (tiling_.isVarLen == 0 &&
                            tiling_.useGateInKernel != 0) {
                            // Both AIVs consume every ready generation, but
                            // each completes one head so their Vector-Post
                            // work can overlap across the two-head window.
                            if (headInWindow == AscendC::GetSubBlockIdx()) {
                                FinishHeadA5Dense16(
                                    task, headBase + headInWindow,
                                    rowStart, validRows, slot, 0);
                            }
                            continue;
                        }
                    }
#endif
                    FinishHead(task, headBase + headInWindow, rowStart, validRows, slot);
                }
                ++windowIdx;
            }
        }
        ReleaseMatrixInputEvents();
    }

private:
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void PrepareA5DenseRow16Window(
        const CIntraTask &task, uint32_t headBase, uint32_t headCount,
        uint32_t rowStart, uint64_t windowIdx)
    {
        for (uint32_t headInWindow = 0;
             headInWindow < headCount; ++headInWindow) {
            const uint32_t slot =
                CIntraWorkspaceSlot(windowIdx, headInWindow);
            PrepareHead(
                task, headBase + headInWindow, rowStart,
                kProcessRowBlock, slot);
            // Both AIV sub-blocks write one half of the packed operands.
            // Publish the generation only after both halves reach GM.
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                vecToCubeReadyFlag_);
        }
    }

    __aicore__ inline void FinishA5DenseRow16Window(
        const CIntraTask &task, uint32_t headBase, uint32_t headCount,
        uint32_t rowStart, uint64_t windowIdx)
    {
        const uint32_t ownedHead = AscendC::GetSubBlockIdx();
        for (uint32_t headInWindow = 0;
             headInWindow < headCount; ++headInWindow) {
            Catlass::Arch::CrossCoreWaitFlag(cubeToVecReadyFlag_);
            if (headInWindow == ownedHead) {
                const uint32_t slot =
                    CIntraWorkspaceSlot(windowIdx, headInWindow);
                FinishHeadA5Dense16(
                    task, headBase + headInWindow, rowStart,
                    kProcessRowBlock, slot, 0);
            }
        }
        // A parity slot contains both heads.  Do not let either AIV refill
        // it until the other AIV has completed its owned head.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
    }

    __aicore__ inline void ProcessA5DenseRow16Pipeline(
        uint32_t coreIdx, uint32_t coreNum, uint32_t headNum,
        uint32_t headWindowCount, uint64_t taskGroupCount)
    {
        static_assert(kProcessRowBlock == kRowBlock,
                      "The A5 row16 pipeline requires row16 tiles.");
        uint64_t windowIdx = 0;
        for (uint64_t taskGroupIdx = coreIdx;
             taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum, windowIdx += 4U) {
            const uint32_t taskIdx =
                static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindowIdx =
                static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase =
                headWindowIdx * kHeadsPerWindow;
            const uint32_t headCount =
                headBase + 1U < headNum ? 2U : 1U;
            const CIntraTask task = GetCIntraTask(
                cuSeqlens_, chunkMetadata_, tiling_, taskIdx);

            // Prime two parity slots.  Refill each slot only after its Post
            // is complete.  Drain all four rows at the chunk boundary so
            // the next chunk cannot overwrite row0's shared B while row48
            // still consumes it.
            PrepareA5DenseRow16Window(
                task, headBase, headCount, 0, windowIdx);
            PrepareA5DenseRow16Window(
                task, headBase, headCount, 16, windowIdx + 1U);
            FinishA5DenseRow16Window(
                task, headBase, headCount, 0, windowIdx);
            PrepareA5DenseRow16Window(
                task, headBase, headCount, 32, windowIdx + 2U);
            FinishA5DenseRow16Window(
                task, headBase, headCount, 16, windowIdx + 1U);
            PrepareA5DenseRow16Window(
                task, headBase, headCount, 48, windowIdx + 3U);
            FinishA5DenseRow16Window(
                task, headBase, headCount, 32, windowIdx + 2U);
            FinishA5DenseRow16Window(
                task, headBase, headCount, 48, windowIdx + 3U);
        }
    }

    __aicore__ inline void PrepareA5DenseWindow(
        uint64_t windowIdx, uint32_t coreIdx, uint32_t coreNum,
        uint32_t headNum, uint32_t headWindowCount)
    {
        const uint64_t taskGroupIdx =
            static_cast<uint64_t>(coreIdx) +
            (windowIdx >> 1U) * coreNum;
        const uint32_t taskIdx =
            static_cast<uint32_t>(taskGroupIdx / headWindowCount);
        const uint32_t headBase =
            static_cast<uint32_t>(taskGroupIdx % headWindowCount) *
            kHeadsPerWindow;
        const uint32_t headCount =
            headBase + kHeadsPerWindow <= headNum ?
                kHeadsPerWindow : headNum - headBase;
        const CIntraTask task = GetCIntraTask(
            cuSeqlens_, chunkMetadata_, tiling_, taskIdx);
        const uint32_t rowStart =
            static_cast<uint32_t>(windowIdx & 1U) * kProcessRowBlock;

        for (uint32_t headInWindow = 0;
             headInWindow < headCount; ++headInWindow) {
            const uint32_t slot =
                CIntraWorkspaceSlot(windowIdx, headInWindow);
            PrepareHead(
                task, headBase + headInWindow, rowStart,
                kProcessRowBlock, slot);
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                vecToCubeReadyFlag_);
        }
    }

    __aicore__ inline void FinishA5DenseWindow(
        uint64_t windowIdx, uint32_t coreIdx, uint32_t coreNum,
        uint32_t headNum, uint32_t headWindowCount)
    {
        const uint64_t taskGroupIdx =
            static_cast<uint64_t>(coreIdx) +
            (windowIdx >> 1U) * coreNum;
        const uint32_t taskIdx =
            static_cast<uint32_t>(taskGroupIdx / headWindowCount);
        const uint32_t headBase =
            static_cast<uint32_t>(taskGroupIdx % headWindowCount) *
            kHeadsPerWindow;
        const uint32_t headCount =
            headBase + kHeadsPerWindow <= headNum ?
                kHeadsPerWindow : headNum - headBase;
        const CIntraTask task = GetCIntraTask(
            cuSeqlens_, chunkMetadata_, tiling_, taskIdx);
        const uint32_t rowStart =
            static_cast<uint32_t>(windowIdx & 1U) * kProcessRowBlock;

        {
            const uint32_t ownedHead = AscendC::GetSubBlockIdx();
            // Every AIV consumes the complete ready stream so the paired
            // AIC/AIV generation counters stay aligned.  Only the matching
            // sub-block performs Vector-Post, owning all 32 rows of one head.
            for (uint32_t headInWindow = 0;
                 headInWindow < headCount; ++headInWindow) {
                Catlass::Arch::CrossCoreWaitFlag(cubeToVecReadyFlag_);
                if (headInWindow != ownedHead) {
                    continue;
                }
                const uint32_t slot =
                    CIntraWorkspaceSlot(windowIdx, headInWindow);
                const uint32_t head = headBase + headInWindow;
                if (rowStart == 0) {
                    FinishHeadA5Dense32(
                        task, head, rowStart, kProcessRowBlock, slot);
                } else {
                    if (tiling_.deferGatePost == 0) {
                        auto carry =
                            reduceTmp_.Get<float>()[7 * kPlaneElements];
                        KdaRegbaseFill(
                            (__ubuf__ float *)carry.GetPhyAddr(),
                            0.0f, K_DIM);
                    }
                    FinishHeadA5Dense32(
                        task, head, rowStart, kProcessRowBlock, slot);
                    if (tiling_.deferGatePost == 0 &&
                        tiling_.useGateInKernel == 0) {
                        FinishPendingDgA5(task, head);
                    }
                }
                CompleteDirectOutputGeneration();
            }
        }
    }

    __aicore__ inline void ProcessA5DenseFourSlot(
        uint32_t coreIdx, uint32_t coreNum, uint32_t headNum,
        uint32_t headWindowCount, uint64_t taskGroupCount)
    {
        static_assert(kProcessRowBlock == 32,
                      "The A5 four-slot path requires row32 tiles.");
        if (coreIdx >= taskGroupCount) {
            return;
        }
        const uint64_t ownedGroups =
            (taskGroupCount - 1U - coreIdx) / coreNum + 1U;
        const uint64_t windowCount = ownedGroups * 2U;
        const uint64_t primeCount = windowCount < 2U ? windowCount : 2U;

        // Prime both row generations.  Post(i) releases parity slot i before
        // Pre(i+2) reuses it, while AIC advances through generation i+1.
        for (uint64_t windowIdx = 0; windowIdx < primeCount; ++windowIdx) {
            PrepareA5DenseWindow(
                windowIdx, coreIdx, coreNum, headNum, headWindowCount);
        }
        for (uint64_t windowIdx = 0; windowIdx < windowCount; ++windowIdx) {
            FinishA5DenseWindow(
                windowIdx, coreIdx, coreNum, headNum, headWindowCount);
            const uint64_t nextWindow = windowIdx + 2U;
            if (nextWindow < windowCount) {
                PrepareA5DenseWindow(
                    nextWindow, coreIdx, coreNum,
                    headNum, headWindowCount);
            }
        }
    }
#endif

    static constexpr uint32_t kProcessRowBlock =
        ProcessRowBlock<K_DIM, PUBLIC_VARLEN>::value;
    static constexpr uint32_t kPlaneElements = 8 * 128;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr uint32_t kLowerPackRows = K_DIM == 128 ? 16 : 8;
    static constexpr uint32_t kUpperPackRows = K_DIM == 128 ? 16 : 8;
    static constexpr uint32_t kIoBufferBytes = 16 * 1024;
    static constexpr uint32_t kUbBudgetBytes = 248 * 1024;
#else
    static constexpr uint32_t kLowerPackRows = K_DIM == 128 ? 16 : 8;
    static constexpr uint32_t kUpperPackRows = K_DIM == 128 ? 16 : 8;
    static constexpr uint32_t kIoBufferBytes = 8 * 1024;
    static constexpr uint32_t kUbBudgetBytes = 192 * 1024;
#endif
    static constexpr uint32_t kLowerMatrixPlanes =
        (kLowerPackRows * (K_DIM / 2) + kPlaneElements - 1) /
        kPlaneElements;
    static constexpr uint32_t kUpperMatrixPlanes =
        (kUpperPackRows * 128 + kPlaneElements - 1) / kPlaneElements;
    static constexpr uint32_t kArenaBytes = 96 * 1024;
    static constexpr uint32_t kReduceTmpBytes = 32 * 1024;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr uint32_t kPendingDgBytes = 32 * 128 * sizeof(float);
#else
    static constexpr uint32_t kPendingDgBytes = 0;
#endif
    static constexpr uint32_t kFp32BlockElements = 32 / sizeof(float);
    static constexpr uint32_t kMatrixInputBufferCount = 2;
    static_assert(
        6 * kIoBufferBytes + kArenaBytes + kReduceTmpBytes +
                kPendingDgBytes <= kUbBudgetBytes,
        "Vector UB buffers exceed the architecture-specific budget.");
    static_assert(kProcessRowBlock * CHUNK_SIZE <= 2 * kPlaneElements,
                  "Packed A-matrix tile exceeds its two-plane arena group.");
    static_assert(kProcessRowBlock * CHUNK_SIZE * sizeof(float) <= kIoBufferBytes,
                  "Packed A-matrix FP32 tile exceeds the IO queue buffer.");
    static_assert(kLowerPackRows * (K_DIM / 2) <=
                      kLowerMatrixPlanes * kPlaneElements,
                  "PackLowerB row tile exceeds its arena plane group.");
    static_assert(kLowerPackRows * (K_DIM / 2) * sizeof(float) <= kIoBufferBytes,
                  "PackLowerB FP32 row tile exceeds the IO queue buffer.");
    static_assert(kUpperPackRows * 128 <= kUpperMatrixPlanes * kPlaneElements,
                  "PackUpperB row tile exceeds its arena plane group.");
    static_assert(kUpperPackRows * 128 * sizeof(float) <= kIoBufferBytes,
                  "PackUpperB FP32 row tile exceeds the IO queue buffer.");
    static_assert(24 * kPlaneElements * sizeof(float) <= kArenaBytes,
                  "Vector scratch layout exceeds the arena buffer.");

    __aicore__ inline AscendC::LocalTensor<float> Plane(uint32_t index)
    {
        return arena_.Get<float>()[index * kPlaneElements];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> ScalarGatherOffsets()
    {
        return arena_.Get<uint32_t>()[22 * kPlaneElements];
    }

    __aicore__ inline AscendC::LocalTensor<float> ScalarStage()
    {
        return Plane(23);
    }

    __aicore__ inline void InitMatrixInputEvents()
    {
        for (uint32_t slot = 0; slot < kMatrixInputBufferCount; ++slot) {
            matrixMte2ToVEvent_[slot] =
                static_cast<event_t>(
                    pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
            matrixVToMte2Event_[slot] =
                static_cast<event_t>(
                    pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        directMte2ToVEvent_ = static_cast<event_t>(
            pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        directVToMte2Event_ = static_cast<event_t>(
            pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(directVToMte2Event_);
        directVToMte3Event_ = static_cast<event_t>(
            pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>());
        directMte3ToMte2Event_ = static_cast<event_t>(
            pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            directMte3ToMte2Event_);
#endif
    }

    __aicore__ inline void ReleaseMatrixInputEvents()
    {
        for (uint32_t slot = 0; slot < kMatrixInputBufferCount; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(
                matrixMte2ToVEvent_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(directVToMte2Event_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(directMte2ToVEvent_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(directVToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            directMte3ToMte2Event_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(
            directVToMte3Event_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(
            directMte3ToMte2Event_);
#endif
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> MatrixInput(uint32_t slot)
    {
        return slot == 0 ? matrixInputPing_.Get<T>() : matrixInputPong_.Get<T>();
    }

    template <typename SrcT>
    __aicore__ inline void ConvertToFp32(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<SrcT> src, uint32_t count)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (IsSameType<SrcT, half>::value) {
            // The former CastHalf2Float RegBase helper stored its two result
            // registers as contiguous halves, but A5 returns a blocked lane
            // distribution.  That silently permuted K/N columns.  Use the
            // proven vector Cast until a correctly interleaved RegBase
            // implementation has its own impulse test.
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            KdaRegbaseCastBf16ToFp32(
                (__ubuf__ float *)dst.GetPhyAddr(),
                (__ubuf__ SrcT *)src.GetPhyAddr(), count);
        }
#else
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
#endif
    }

    template <typename T>
    __aicore__ inline uint32_t CopyInMatrixRows(
        AscendC::GlobalTensor<T> src, uint32_t rows, uint32_t cols,
        uint32_t srcRowElements)
    {
        const uint32_t slot = currentMatrixInputSlot_;
        currentMatrixInputSlot_ ^= 1U;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((srcRowElements - cols) * sizeof(T)),
            0,
            0
        };
        AscendC::DataCopyPad(
            MatrixInput<T>(slot), src, copyParams, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
        return slot;
    }

    __aicore__ inline void ConsumeMatrixRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t slot, uint32_t count)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            count);
#else
        AscendC::Adds(dst, src, 0.0f, count);
#endif
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
    }

    __aicore__ inline void ConsumeMatrixRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<DataT> src,
        uint32_t slot, uint32_t count)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
        ConvertToFp32(dst, src, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
    }

    template <typename T0, typename T1>
    __aicore__ inline void LoadMatrixRowsPair(
        AscendC::LocalTensor<float> dst0, AscendC::GlobalTensor<T0> src0,
        uint32_t rows0, uint32_t cols0, uint32_t srcRowElements0,
        AscendC::LocalTensor<float> dst1, AscendC::GlobalTensor<T1> src1,
        uint32_t rows1, uint32_t cols1, uint32_t srcRowElements1)
    {
        // Issue both GM reads before consuming either input.  The second
        // MTE2 transfer can overlap the first input's Vector conversion.
        const uint32_t slot0 =
            CopyInMatrixRows(src0, rows0, cols0, srcRowElements0);
        const uint32_t slot1 =
            CopyInMatrixRows(src1, rows1, cols1, srcRowElements1);
        ConsumeMatrixRows(
            dst0, MatrixInput<T0>(slot0), slot0, rows0 * cols0);
        ConsumeMatrixRows(
            dst1, MatrixInput<T1>(slot1), slot1, rows1 * cols1);
        AscendC::PipeBarrier<PIPE_V>();
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    template <typename T>
    __aicore__ inline void CopyInMatrixRowsDirect(
        AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
        uint32_t rows, uint32_t cols, uint32_t srcRowElements)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((srcRowElements - cols) * sizeof(T)),
            0,
            0
        };
        AscendC::DataCopyPad(dst, src, copyParams, {false, 0, 0, 0});
    }

    __aicore__ inline void StoreRowsDirect(
        AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows, uint32_t cols, uint32_t dstRowElements)
    {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(directVToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(directVToMte3Event_);
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            0,
            static_cast<uint32_t>((dstRowElements - cols) * sizeof(float)),
            0
        };
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void StorePreparedRows(
        AscendC::GlobalTensor<float> dst,
        AscendC::LocalTensor<float> output,
        uint32_t rows, uint32_t cols, uint32_t dstRowElements)
    {
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            0,
            static_cast<uint32_t>((dstRowElements - cols) * sizeof(float)),
            0
        };
        AscendC::DataCopyPad(dst, ready, copyParams);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void CompleteDirectOutputGeneration()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            directMte3ToMte2Event_);
    }
#endif

    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, uint32_t count)
    {
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyPad(
            input, src, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)ready.GetPhyAddr(),
            count);
#else
        AscendC::Adds(dst, ready, 0.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
#endif
        inputQueue_.FreeTensor(ready);
    }

    template <typename SrcT>
    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<SrcT> src,
        uint32_t count)
    {
        auto input = inputQueue_.AllocTensor<SrcT>();
        AscendC::DataCopyPad(
            input, src, {1, static_cast<uint32_t>(count * sizeof(SrcT)), 0, 0, 0},
            {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<SrcT>();
        ConvertToFp32(dst, ready, count);
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Store(
        AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t count)
    {
        AscendC::PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            count);
#else
        AscendC::Adds(output, src, 0.0f, count);
#endif
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<float>();
        AscendC::DataCopyPad(
            dst, ready, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
        uint32_t rows, uint32_t cols, uint32_t srcRowElements)
    {
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            static_cast<uint32_t>((srcRowElements - cols) * sizeof(float)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)ready.GetPhyAddr(),
            rows * cols);
#else
        AscendC::Adds(dst, ready, 0.0f, rows * cols);
        AscendC::PipeBarrier<PIPE_V>();
#endif
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<DataT> src,
        uint32_t rows, uint32_t cols, uint32_t srcRowElements)
    {
        auto input = inputQueue_.AllocTensor<DataT>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(DataT)),
            static_cast<uint32_t>((srcRowElements - cols) * sizeof(DataT)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<DataT>();
        ConvertToFp32(dst, ready, rows * cols);
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void StoreRows(
        AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows, uint32_t cols, uint32_t dstRowElements)
    {
        AscendC::PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            rows * cols);
#else
        AscendC::Adds(output, src, 0.0f, rows * cols);
#endif
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            0,
            static_cast<uint32_t>((dstRowElements - cols) * sizeof(float)),
            0
        };
        AscendC::DataCopyPad(dst, ready, copyParams);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline uint32_t TensorRowElements() const
    {
        if constexpr (VARLEN_TND) {
            return static_cast<uint32_t>(tiling_.headNum) * K_DIM;
        }
        return K_DIM;
    }

    __aicore__ inline uint32_t MatrixRowElements() const
    {
        if constexpr (VARLEN_TND) {
            return static_cast<uint32_t>(tiling_.headNum) * CHUNK_SIZE;
        }
        return CHUNK_SIZE;
    }

    __aicore__ inline uint32_t ScalarRowElements() const
    {
        if constexpr (VARLEN_TND) {
            return static_cast<uint32_t>(tiling_.headNum);
        }
        return 1;
    }

    __aicore__ inline void LoadStridedScalars(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src,
        uint32_t rows, uint32_t srcRowElements)
    {
        constexpr uint32_t kLocalRowElements = 32 / sizeof(float);
        const uint32_t stagedElements = rows * kLocalRowElements;
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(float)),
            static_cast<uint32_t>((srcRowElements - 1) * sizeof(float)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<float>();
        auto stage = ScalarStage();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseCopy(
            (__ubuf__ float *)stage.GetPhyAddr(),
            (__ubuf__ float *)ready.GetPhyAddr(),
            stagedElements);
        inputQueue_.FreeTensor(ready);
        KdaRegbaseGatherScalars(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)stage.GetPhyAddr(),
            rows, kLocalRowElements);
#else
        AscendC::Adds(stage, ready, 0.0f, stagedElements);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
        AscendC::Gather(
            dst, stage, ScalarGatherOffsets(), static_cast<uint32_t>(0), rows);
        AscendC::PipeBarrier<PIPE_V>();
#endif
    }

    template <typename SrcT>
    __aicore__ inline void LoadStridedScalars(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<SrcT> src,
        uint32_t rows, uint32_t srcRowElements)
    {
        constexpr uint32_t kLocalRowElements = 32 / sizeof(SrcT);
        const uint32_t stagedElements = rows * kLocalRowElements;
        auto input = inputQueue_.AllocTensor<SrcT>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(SrcT)),
            static_cast<uint32_t>((srcRowElements - 1) * sizeof(SrcT)),
            0,
            0
        };
        AscendC::DataCopyPad(input, src, copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(input);
        auto ready = inputQueue_.DeQue<SrcT>();
        auto stage = ScalarStage();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ConvertToFp32(stage, ready, stagedElements);
        inputQueue_.FreeTensor(ready);
        KdaRegbaseGatherScalars(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)stage.GetPhyAddr(),
            rows, kLocalRowElements);
#else
        AscendC::Cast(stage, ready, AscendC::RoundMode::CAST_NONE, stagedElements);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
        AscendC::Gather(
            dst, stage, ScalarGatherOffsets()[kProcessRowBlock],
            static_cast<uint32_t>(0), rows);
        AscendC::PipeBarrier<PIPE_V>();
#endif
    }

    __aicore__ inline void LoadScalarRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src, uint32_t rows)
    {
        if constexpr (VARLEN_TND) {
            LoadStridedScalars(dst, src, rows, ScalarRowElements());
        } else {
            Load(dst, src, rows);
        }
    }

    template <typename SrcT>
    __aicore__ inline void LoadScalarRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<SrcT> src, uint32_t rows)
    {
        if constexpr (VARLEN_TND) {
            LoadStridedScalars(dst, src, rows, ScalarRowElements());
        } else {
            Load(dst, src, rows);
        }
    }

    __aicore__ inline void StoreScalarRows(
        AscendC::GlobalTensor<float> dst,
        AscendC::LocalTensor<float> src, uint32_t rows)
    {
        if constexpr (!VARLEN_TND) {
            Store(dst, src, rows);
            return;
        }
        constexpr uint32_t kLocalRowElements = 32 / sizeof(float);
        AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        auto output = outputQueue_.AllocTensor<float>();
        KdaRegbaseScatterScalars(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            rows, kLocalRowElements);
#else
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        auto output = outputQueue_.AllocTensor<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Duplicate(
                output[row * kLocalRowElements], src.GetValue(row), 1);
        }
#endif
        outputQueue_.EnQue(output);
        auto ready = outputQueue_.DeQue<float>();
        // For VECOUT->GM DataCopyPad, a non-aligned local block is rounded to
        // one 32-byte data block.  Adjacent staged rows therefore use
        // srcStride=0; GM dstStride remains byte-based.
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(float)),
            0,
            static_cast<uint32_t>((ScalarRowElements() - 1) * sizeof(float)),
            0
        };
        AscendC::DataCopyPad(dst, ready, copyParams);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline uint64_t SlotBase(uint32_t slot) const
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        return static_cast<uint64_t>(coreIdx) * tiling_.workspaceCoreSize +
               static_cast<uint64_t>(slot) * tiling_.workspaceSlotSize;
    }

    __aicore__ inline void Exp2(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t count)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseExp2(
            (__ubuf__ float *)dst.GetPhyAddr(),
            (__ubuf__ float *)src.GetPhyAddr(),
            count);
#else
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(dst, src, kLn2, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dst, dst, count);
#endif
    }

    __aicore__ inline void PrepareHead(
        const CIntraTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t slot)
    {
        const uint64_t slotBase = SlotBase(slot);
        const uint32_t prefix = rowStart + validRows;
        const uint32_t future = task.end - task.begin - rowStart;
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        PackLowerA(task, head, rowStart, validRows, prefix, subBlock, slotBase);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!PUBLIC_VARLEN && K_DIM == 128 &&
                      kProcessRowBlock == kRowBlock) {
            if (tiling_.useGateInKernel != 0 &&
                validRows == kProcessRowBlock &&
                task.end - task.begin == CHUNK_SIZE) {
                PackUpperA5Shared(
                    task, head, rowStart, validRows, future,
                    subBlock, slotBase);
                if (rowStart == 0) {
                    PackDenseA5B<true>(
                        task, head, rowStart, prefix, future,
                        subBlock, slotBase);
                }
                return;
            }
        }
#endif
        PackUpperA(task, head, rowStart, validRows, future, subBlock, slotBase);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!PUBLIC_VARLEN && K_DIM == 128 &&
                      kProcessRowBlock == kRowBlock) {
            if (validRows == kProcessRowBlock &&
                task.end - task.begin == CHUNK_SIZE) {
                PackDenseA5B<false>(
                    task, head, rowStart, prefix, future,
                    subBlock, slotBase);
                return;
            }
        }
#endif
        PackLowerB(task, head, rowStart, validRows, prefix, subBlock, slotBase);
        PackUpperB(task, head, rowStart, future, subBlock, slotBase);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void PackUpperA5Shared(
        const CIntraTask &task, uint32_t head, uint32_t rowStart,
        uint32_t validRows, uint32_t future, uint32_t subBlock,
        uint64_t slotBase)
    {
        auto work = Plane(0);
        auto masked = Plane(2);
        auto &source = subBlock == 0 ? dAqkGm_ : dAkkGm_;
        const uint64_t srcOffset = CIntraMatrixOffset(
            tiling_, task.batchIdx, head,
            task.begin + rowStart, rowStart);
        LoadRows(
            work, source[srcOffset], future, kProcessRowBlock,
            MatrixRowElements());

        KdaRegbaseMaskUpperA(
            (__ubuf__ float *)masked.GetPhyAddr(),
            (__ubuf__ float *)work.GetPhyAddr(),
            future, validRows, kProcessRowBlock);

        // Interleave Aq/Akk reduction rows so every row block can consume a
        // compact suffix from the one shared [q,k] B matrix.
        const uint32_t physicalRowBase = 2 * rowStart + subBlock;
        const uint64_t dstOffset =
            slotBase / sizeof(float) +
            tiling_.intraAUpperOffset / sizeof(float) +
            static_cast<uint64_t>(physicalRowBase) * kProcessRowBlock;
        StoreRows(
            workspaceGm_[dstOffset], masked,
            future, kProcessRowBlock, 2 * kProcessRowBlock);
    }
#endif

    __aicore__ inline void PackLowerA(
        const CIntraTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t prefix, uint32_t subBlock, uint64_t slotBase)
    {
        const uint32_t lowerK = (prefix + 15U) & ~15U;
        auto work = Plane(0);
        // The A5 row32 tile occupies two consecutive 4 KiB planes.  Keep the
        // masked destination in a disjoint two-plane group.
        auto masked = Plane(2);
        auto &source = subBlock == 0 ? dAqkGm_ : dAkkGm_;
        const float inputScale = subBlock == 0 ? tiling_.scale : 1.0f;
        const uint32_t rowBase = subBlock * kProcessRowBlock;

        // DataCopyPad lays out each UB row at a 32-byte boundary.  Full chunks
        // therefore use one 16-row transfer for the aligned 16/32/48/64
        // prefixes.  Preserve the row-wise path for non-aligned tail chunks.
        if (prefix % kFp32BlockElements == 0) {
            const uint64_t srcOffset =
                CIntraMatrixOffset(
                    tiling_, task.batchIdx, head, task.begin + rowStart);
            LoadRows(work, source[srcOffset], validRows, prefix, MatrixRowElements());
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
            if (subBlock == 0) {
                AscendC::Muls(
                    work, work, tiling_.scale, validRows * prefix);
                AscendC::PipeBarrier<PIPE_V>();
            }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseMaskLowerA(
                (__ubuf__ float *)masked.GetPhyAddr(),
                (__ubuf__ float *)work.GetPhyAddr(),
                validRows, rowStart, prefix, kProcessRowBlock);
#else
            AscendC::Duplicate(masked, 0.0f, kProcessRowBlock * prefix);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < validRows; ++row) {
                const uint32_t validCols = rowStart + row + 1;
                AscendC::Adds(
                    masked[row * prefix], work[row * prefix], 0.0f, validCols);
            }
#endif
            const uint64_t dstOffset =
                slotBase / sizeof(float) + tiling_.intraALowerOffset / sizeof(float) +
                static_cast<uint64_t>(rowBase) * lowerK;
            StoreRows(
                workspaceGm_[dstOffset], masked, kProcessRowBlock, prefix, lowerK);
            return;
        }

        for (uint32_t row = 0; row < kProcessRowBlock; ++row) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseFill(
                (__ubuf__ float *)work.GetPhyAddr(), 0.0f, lowerK);
#else
            AscendC::Duplicate(work, 0.0f, lowerK);
#endif
            if (row < validRows) {
                const uint32_t validCols = rowStart + row + 1;
                const uint64_t srcOffset =
                    CIntraMatrixOffset(
                        tiling_, task.batchIdx, head, task.begin + rowStart + row);
                Load(work, source[srcOffset], validCols);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
                AscendC::Muls(work, work, inputScale, validCols);
                AscendC::PipeBarrier<PIPE_V>();
#else
                if (subBlock != 0) {
                    AscendC::Muls(work, work, inputScale, validCols);
                    AscendC::PipeBarrier<PIPE_V>();
                }
#endif
            }
            const uint64_t dstOffset =
                slotBase / sizeof(float) + tiling_.intraALowerOffset / sizeof(float) +
                static_cast<uint64_t>(rowBase + row) * lowerK;
            Store(workspaceGm_[dstOffset], work, lowerK);
        }
    }

    __aicore__ inline void PackUpperA(
        const CIntraTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t future, uint32_t subBlock, uint64_t slotBase)
    {
        auto work = Plane(0);
        auto masked = Plane(2);
        auto &source = subBlock == 0 ? dAqkGm_ : dAkkGm_;
        const uint32_t physicalRowBase = subBlock * future;

        // The largest Upper-A tile is 64 rows x 16 FP32 values, exactly one
        // 4 KiB IO buffer.  Move the complete future range in one MTE2/MTE3
        // batch instead of issuing one pair of transfers per 8-row tile.
        const uint64_t srcOffset =
            CIntraMatrixOffset(
                tiling_, task.batchIdx, head, task.begin + rowStart, rowStart);
        LoadRows(
            work, source[srcOffset], future, kProcessRowBlock,
            MatrixRowElements());
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        if (subBlock == 0) {
            AscendC::Muls(
                work, work, tiling_.scale, future * kProcessRowBlock);
            AscendC::PipeBarrier<PIPE_V>();
        }
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseMaskUpperA(
            (__ubuf__ float *)masked.GetPhyAddr(),
            (__ubuf__ float *)work.GetPhyAddr(),
            future, validRows, kProcessRowBlock);
#else
        AscendC::Duplicate(masked, 0.0f, future * kProcessRowBlock);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t maskedRows = future < validRows ? future : validRows;
        for (uint32_t row = 0; row < maskedRows; ++row) {
            AscendC::Adds(
                masked[row * kProcessRowBlock], work[row * kProcessRowBlock],
                0.0f, row + 1);
        }
        if (future > maskedRows) {
            const uint32_t tailElements =
                (future - maskedRows) * kProcessRowBlock;
            AscendC::Adds(
                masked[maskedRows * kProcessRowBlock],
                work[maskedRows * kProcessRowBlock], 0.0f, tailElements);
        }
#endif
        const uint64_t dstOffset =
            slotBase / sizeof(float) + tiling_.intraAUpperOffset / sizeof(float) +
            static_cast<uint64_t>(physicalRowBase) * kProcessRowBlock;
        StoreRows(
            workspaceGm_[dstOffset], masked, future, kProcessRowBlock,
            kProcessRowBlock);
    }

    __aicore__ inline void PackLowerB(
        const CIntraTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t prefix, uint32_t subBlock, uint64_t slotBase)
    {
        auto data = Plane(0);
        auto gate = Plane(kLowerMatrixPlanes);
        auto anchor = Plane(2 * kLowerMatrixPlanes);
        auto exponent = Plane(3 * kLowerMatrixPlanes);
        const uint32_t cols = K_DIM / 2;
        const uint32_t col = subBlock * cols;
        const uint32_t anchorRow =
            task.begin + rowStart + (validRows > 8 ? 8 : validRows - 1);
        Load(anchor,
             gkGm_[CIntraTensorOffset(
                 tiling_, task.batchIdx, head, anchorRow, col)],
             cols);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        for (uint32_t row = 1; row < kLowerPackRows; ++row) {
            AscendC::Adds(anchor[row * cols], anchor, 0.0f, cols);
        }
        AscendC::PipeBarrier<PIPE_V>();
#endif
        for (uint32_t sourceRow = 0; sourceRow < prefix; sourceRow += kLowerPackRows) {
            const uint32_t rows =
                sourceRow + kLowerPackRows <= prefix ? kLowerPackRows : prefix - sourceRow;
            const uint32_t count = rows * cols;
            const uint32_t token = task.begin + sourceRow;
            LoadMatrixRowsPair(
                data,
                kGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, token, col)],
                rows, cols, TensorRowElements(),
                gate,
                gkGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, token, col)],
                rows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseGateScale<true, false>(
                (__ubuf__ float *)data.GetPhyAddr(),
                (__ubuf__ float *)gate.GetPhyAddr(),
                (__ubuf__ float *)anchor.GetPhyAddr(),
                (__ubuf__ float *)0, rows, cols);
#else
            AscendC::Sub(exponent, anchor, gate, count);
            Exp2(exponent, exponent, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(data, data, exponent, count);
#endif
            const uint64_t dstOffset =
                slotBase / sizeof(float) + tiling_.intraBLowerOffset / sizeof(float) +
                static_cast<uint64_t>(sourceRow) * K_DIM + col;
            StoreRows(workspaceGm_[dstOffset], data, rows, cols, K_DIM);
        }
        // Cube reduces over lowerK=align16(prefix).  A varlen tail may have
        // fewer valid prefix rows, so explicitly zero the physical padding
        // instead of reusing stale rows from the previous slot generation.
        const uint32_t lowerK = (prefix + 15U) & ~15U;
        if (prefix < lowerK) {
            const uint32_t rows = lowerK - prefix;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseFill(
                (__ubuf__ float *)data.GetPhyAddr(), 0.0f, rows * cols);
#else
            AscendC::Duplicate(data, 0.0f, rows * cols);
#endif
            const uint64_t dstOffset =
                slotBase / sizeof(float) + tiling_.intraBLowerOffset / sizeof(float) +
                static_cast<uint64_t>(prefix) * K_DIM + col;
            StoreRows(workspaceGm_[dstOffset], data, rows, cols, K_DIM);
        }
    }

    __aicore__ inline void PackUpperB(
        const CIntraTask &task, uint32_t head, uint32_t rowStart, uint32_t future,
        uint32_t subBlock, uint64_t slotBase)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // Partition q and k by columns across the two AIVs.  Both operands
        // use the same exp2(gk-anchor) factor, so each AIV loads its gate
        // half and evaluates Exp only once for both q and k.
        auto qData = Plane(0);   // planes 0-1
        auto kData = Plane(2);   // planes 2-3
        auto gate = Plane(4);    // planes 4-5
        auto anchor = Plane(6);
        auto beta = Plane(8);
        const uint32_t cols = K_DIM / 2;
        const uint32_t col = subBlock * cols;
        const uint32_t anchorLocal =
            rowStart + 8 < task.end - task.begin ?
                rowStart + 8 : task.end - task.begin - 1;
        const uint32_t anchorRow = task.begin + anchorLocal;
        Load(
            anchor,
            gkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, anchorRow, col)],
            cols);
        for (uint32_t sourceRow = 0; sourceRow < future;
             sourceRow += kUpperPackRows) {
            const uint32_t rows =
                sourceRow + kUpperPackRows <= future ?
                    kUpperPackRows : future - sourceRow;
            const uint32_t token = task.begin + rowStart + sourceRow;
            LoadMatrixRowsPair(
                qData,
                qGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, token, col)],
                rows, cols, TensorRowElements(),
                kData,
                kGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, token, col)],
                rows, cols, TensorRowElements());
            LoadRows(
                gate,
                gkGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, token, col)],
                rows, cols, TensorRowElements());
            LoadScalarRows(
                beta,
                betaGm_[CIntraScalarOffset(
                    tiling_, task.batchIdx, head, token)],
                rows);
            KdaRegbaseGateScalePair(
                (__ubuf__ float *)qData.GetPhyAddr(),
                (__ubuf__ float *)kData.GetPhyAddr(),
                (__ubuf__ float *)gate.GetPhyAddr(),
                (__ubuf__ float *)anchor.GetPhyAddr(),
                (__ubuf__ float *)beta.GetPhyAddr(),
                rows, cols);

            const uint64_t qDstOffset =
                slotBase / sizeof(float) +
                tiling_.intraBUpperOffset / sizeof(float) +
                static_cast<uint64_t>(sourceRow) * K_DIM + col;
            const uint64_t kDstOffset =
                slotBase / sizeof(float) +
                tiling_.intraBUpperOffset / sizeof(float) +
                static_cast<uint64_t>(future + sourceRow) * K_DIM + col;
            StoreRows(
                workspaceGm_[qDstOffset], qData,
                rows, cols, K_DIM);
            StoreRows(
                workspaceGm_[kDstOffset], kData,
                rows, cols, K_DIM);
        }
#else
        // data/gate/anchor/exponent each own one complete pack batch.  Compute
        // the plane stride from kUpperPackRows so the A5 32-row matrices and
        // the A2/A3 16-row matrices use the same code without overlap.
        auto data = Plane(0);
        auto gate = Plane(kUpperMatrixPlanes);
        auto anchor = Plane(2 * kUpperMatrixPlanes);
        auto exponent = Plane(3 * kUpperMatrixPlanes);
        auto beta = Plane(4 * kUpperMatrixPlanes);
        auto betaBroadcast = Plane(4 * kUpperMatrixPlanes + 1);
        const uint32_t anchorLocal = rowStart + 8 < task.end - task.begin ?
                                     rowStart + 8 : task.end - task.begin - 1;
        const uint32_t anchorRow = task.begin + anchorLocal;
        for (uint32_t col = 0; col < K_DIM; col += 128) {
            const uint32_t cols = col + 128 <= K_DIM ? 128 : K_DIM - col;
            Load(anchor,
                 gkGm_[CIntraTensorOffset(
                     tiling_, task.batchIdx, head, anchorRow, col)],
                 cols);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
            for (uint32_t row = 1; row < kUpperPackRows; ++row) {
                AscendC::Adds(anchor[row * cols], anchor, 0.0f, cols);
            }
            AscendC::PipeBarrier<PIPE_V>();
#endif
            for (uint32_t sourceRow = 0; sourceRow < future; sourceRow += kUpperPackRows) {
                const uint32_t rows =
                    sourceRow + kUpperPackRows <= future ? kUpperPackRows : future - sourceRow;
                const uint32_t count = rows * cols;
                const uint32_t token = task.begin + rowStart + sourceRow;
                auto &source = subBlock == 0 ? qGm_ : kGm_;
                LoadMatrixRowsPair(
                    data,
                    source[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements(),
                    gate,
                    gkGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if (subBlock != 0) {
                    LoadScalarRows(
                        beta,
                        betaGm_[CIntraScalarOffset(
                            tiling_, task.batchIdx, head, token)],
                        rows);
                    KdaRegbaseGateScale<false, true>(
                        (__ubuf__ float *)data.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)anchor.GetPhyAddr(),
                        (__ubuf__ float *)beta.GetPhyAddr(),
                        rows, cols);
                } else {
                    KdaRegbaseGateScale<false, false>(
                        (__ubuf__ float *)data.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)anchor.GetPhyAddr(),
                        (__ubuf__ float *)0, rows, cols);
                }
#else
                AscendC::Sub(exponent, gate, anchor, count);
                Exp2(exponent, exponent, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(data, data, exponent, count);
                if (subBlock != 0) {
                    LoadScalarRows(
                        beta,
                        betaGm_[CIntraScalarOffset(
                            tiling_, task.batchIdx, head, token)],
                        rows);
                    AscendC::Brcb(
                        betaBroadcast, beta, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint8_t rowStride =
                        static_cast<uint8_t>(cols * sizeof(float) / 32);
                    for (uint32_t offset = 0; offset < cols; offset += 64) {
                        const uint32_t mask = offset + 64 <= cols ? 64 : cols - offset;
                        AscendC::Mul(data[offset], data[offset], betaBroadcast, mask, rows,
                                   {1, 1, 0, rowStride, rowStride, 1});
                    }
                }
#endif
                const uint32_t physicalRow = subBlock * future + sourceRow;
                const uint64_t dstOffset =
                    slotBase / sizeof(float) + tiling_.intraBUpperOffset / sizeof(float) +
                    static_cast<uint64_t>(physicalRow) * K_DIM + col;
                StoreRows(workspaceGm_[dstOffset], data, rows, cols, K_DIM);
            }
        }
#endif
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    template <bool SHARED>
    __aicore__ inline void PackDenseA5B(
        const CIntraTask &task, uint32_t head, uint32_t rowStart,
        uint32_t prefix, uint32_t future, uint32_t subBlock,
        uint64_t slotBase)
    {
        static_assert(kProcessRowBlock == kRowBlock,
                      "A5 fused B-pack requires stable row16 tiles.");
        auto qData = Plane(0);
        auto kData = Plane(2);
        auto gate = Plane(4);
        auto lowerData = Plane(6);
        auto lowerAnchor = Plane(8);
        auto upperAnchor = Plane(9);
        auto beta = Plane(SHARED ? 10 : 9);
        constexpr uint32_t rows = kProcessRowBlock;
        constexpr uint32_t cols = K_DIM / 2;
        const uint32_t col = subBlock * cols;
        const uint32_t lowerAnchorRow =
            task.begin + (SHARED ? CHUNK_SIZE - 8 : rowStart + 8);
        const uint32_t upperAnchorRow =
            task.begin + (SHARED ? 8 : rowStart + 8);
        Load(
            lowerAnchor,
            gkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, lowerAnchorRow, col)],
            cols);
        if constexpr (SHARED) {
            Load(
                upperAnchor,
                gkGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, upperAnchorRow, col)],
                cols);
        }

        for (uint32_t sourceRow = 0; sourceRow < CHUNK_SIZE;
             sourceRow += kProcessRowBlock) {
            const bool needLower = SHARED || sourceRow < prefix;
            const bool needUpper = SHARED || sourceRow >= rowStart;
            const uint32_t token = task.begin + sourceRow;

            if (needLower && needUpper) {
                // The shared row block supplies both contractions.  Keep one
                // raw k copy for the lower scale before the upper pair
                // updates kData in place.
                LoadMatrixRowsPair(
                    qData,
                    qGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements(),
                    kData,
                    kGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements());
                LoadRows(
                    gate,
                    gkGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements());
                LoadScalarRows(
                    beta,
                    betaGm_[CIntraScalarOffset(
                        tiling_, task.batchIdx, head, token)],
                    rows);
                if constexpr (SHARED) {
                    KdaRegbaseGateScaleLowerPair<true>(
                        (__ubuf__ float *)qData.GetPhyAddr(),
                        (__ubuf__ float *)kData.GetPhyAddr(),
                        (__ubuf__ float *)lowerData.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                        (__ubuf__ float *)upperAnchor.GetPhyAddr(),
                        (__ubuf__ float *)beta.GetPhyAddr(), rows, cols);
                } else {
                    KdaRegbaseGateScaleLowerPair(
                        (__ubuf__ float *)qData.GetPhyAddr(),
                        (__ubuf__ float *)kData.GetPhyAddr(),
                        (__ubuf__ float *)lowerData.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                        (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                        (__ubuf__ float *)beta.GetPhyAddr(), rows, cols);
                }
            } else if (needLower) {
                LoadMatrixRowsPair(
                    lowerData,
                    kGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements(),
                    gate,
                    gkGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements());
                KdaRegbaseGateScale<true, false>(
                    (__ubuf__ float *)lowerData.GetPhyAddr(),
                    (__ubuf__ float *)gate.GetPhyAddr(),
                    (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                    (__ubuf__ float *)0, rows, cols);
            } else {
                LoadMatrixRowsPair(
                    qData,
                    qGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements(),
                    kData,
                    kGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements());
                LoadRows(
                    gate,
                    gkGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, token, col)],
                    rows, cols, TensorRowElements());
                LoadScalarRows(
                    beta,
                    betaGm_[CIntraScalarOffset(
                        tiling_, task.batchIdx, head, token)],
                    rows);
                if constexpr (SHARED) {
                    KdaRegbaseGateScalePair(
                        (__ubuf__ float *)qData.GetPhyAddr(),
                        (__ubuf__ float *)kData.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)upperAnchor.GetPhyAddr(),
                        (__ubuf__ float *)beta.GetPhyAddr(), rows, cols);
                } else {
                    KdaRegbaseGateScalePair(
                        (__ubuf__ float *)qData.GetPhyAddr(),
                        (__ubuf__ float *)kData.GetPhyAddr(),
                        (__ubuf__ float *)gate.GetPhyAddr(),
                        (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                        (__ubuf__ float *)beta.GetPhyAddr(), rows, cols);
                }
            }

            if (needLower) {
                const uint64_t lowerOffset =
                    slotBase / sizeof(float) +
                    tiling_.intraBLowerOffset / sizeof(float) +
                    static_cast<uint64_t>(sourceRow) * K_DIM + col;
                StoreRows(
                    workspaceGm_[lowerOffset], lowerData,
                    rows, cols, K_DIM);
            }
            if (needUpper) {
                const uint32_t upperRow = SHARED ? 2 * sourceRow :
                                                  sourceRow - rowStart;
                const uint64_t qOffset =
                    slotBase / sizeof(float) +
                    tiling_.intraBUpperOffset / sizeof(float) +
                    static_cast<uint64_t>(upperRow) * K_DIM + col;
                const uint64_t kOffset =
                    slotBase / sizeof(float) +
                    tiling_.intraBUpperOffset / sizeof(float) +
                    static_cast<uint64_t>(
                        SHARED ? upperRow + 1 : future + upperRow) *
                        K_DIM + col;
                StoreRows(
                    workspaceGm_[qOffset], qData,
                    rows, cols, SHARED ? 2 * K_DIM : K_DIM);
                StoreRows(
                    workspaceGm_[kOffset], kData,
                    rows, cols, SHARED ? 2 * K_DIM : K_DIM);
            }
        }
    }
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline float ReadGateScalarA5(
        AscendC::LocalTensor<float> value)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return value.GetValue(0);
    }

    __aicore__ inline void ApplyFusedSafeGateA5(
        const CIntraTask &task, uint32_t head,
        AscendC::LocalTensor<float> laterDg, uint32_t rows)
    {
        auto pending = pendingDg_.Get<float>();
        auto reduce = reduceTmp_.Get<float>();
        auto carry = reduce[7 * kPlaneElements];
        auto dbAcc = reduce;
        auto dAAcc = reduce[128];
        auto bias = reduce[2 * kPlaneElements];
        auto scalar = bias[128];
        auto raw = Plane(0);

        KdaRegbaseFill(
            (__ubuf__ float *)dbAcc.GetPhyAddr(), 0.0f, 136);

        // First materialize dg_hv for both resident halves.  The second
        // reverse scan below is the raw-gate upstream chain and therefore
        // must not replace this Intra reconstruction scan.
        KdaRegbaseReverseScanCarry(
            (__ubuf__ float *)laterDg.GetPhyAddr(),
            (__ubuf__ float *)carry.GetPhyAddr(), rows);
        KdaRegbaseReverseScanCarry(
            (__ubuf__ float *)pending.GetPhyAddr(),
            (__ubuf__ float *)carry.GetPhyAddr(), rows);

        Load(scalar, aLogGm_[head], 1);
        AscendC::Exp(scalar, scalar, 1);
        AscendC::PipeBarrier<PIPE_V>();
        const float expA = ReadGateScalarA5(scalar);
        if (tiling_.hasDtBias != 0) {
            Load(
                bias,
                dtBiasGm_[static_cast<uint64_t>(head) * K_DIM], K_DIM);
        }

        KdaRegbaseFill(
            (__ubuf__ float *)carry.GetPhyAddr(), 0.0f, K_DIM);
        const uint64_t laterOffset = CIntraTensorOffset(
            tiling_, task.batchIdx, head,
            task.begin + kProcessRowBlock, 0);
        Load(raw, rawGGm_[laterOffset], rows * K_DIM);
        if (tiling_.hasDtBias != 0) {
            KdaBwdCSafeGateBackwardA5<true>(
                (__ubuf__ float *)laterDg.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dAAcc.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)laterDg.GetPhyAddr(),
                (__ubuf__ float *)bias.GetPhyAddr(),
                (__ubuf__ float *)carry.GetPhyAddr(),
                static_cast<uint16_t>(rows), expA, tiling_.lowerBound);
        } else {
            KdaBwdCSafeGateBackwardA5<false>(
                (__ubuf__ float *)laterDg.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dAAcc.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)laterDg.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)carry.GetPhyAddr(),
                static_cast<uint16_t>(rows), expA, tiling_.lowerBound);
        }

        const uint64_t firstOffset = CIntraTensorOffset(
            tiling_, task.batchIdx, head, task.begin, 0);
        Load(raw, rawGGm_[firstOffset], rows * K_DIM);
        if (tiling_.hasDtBias != 0) {
            KdaBwdCSafeGateBackwardA5<true>(
                (__ubuf__ float *)pending.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dAAcc.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)pending.GetPhyAddr(),
                (__ubuf__ float *)bias.GetPhyAddr(),
                (__ubuf__ float *)carry.GetPhyAddr(),
                static_cast<uint16_t>(rows), expA, tiling_.lowerBound);
        } else {
            KdaBwdCSafeGateBackwardA5<false>(
                (__ubuf__ float *)pending.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dAAcc.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)pending.GetPhyAddr(),
                (__ubuf__ float *)raw.GetPhyAddr(),
                (__ubuf__ float *)carry.GetPhyAddr(),
                static_cast<uint16_t>(rows), expA, tiling_.lowerBound);
        }

        StorePreparedRows(
            dgOutGm_[laterOffset], laterDg,
            rows, K_DIM, TensorRowElements());
        StoreRowsDirect(
            dgOutGm_[firstOffset], pending,
            rows, K_DIM, TensorRowElements());
        AscendC::SetAtomicAdd<float>();
        Store(dAGm_[head], dAAcc, 1U);
        if (tiling_.hasDtBias != 0) {
            Store(
                dBiasGm_[static_cast<uint64_t>(head) * K_DIM],
                dbAcc, K_DIM);
        }
        // Atomic mode is a DMA state.  Drain both writes before restoring
        // normal stores for the next C stage/generation.
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::SetAtomicNone();
    }

    __aicore__ inline void FinishPendingDgA5(
        const CIntraTask &task, uint32_t head)
    {
        auto pending = pendingDg_.Get<float>();
        auto carry = reduceTmp_.Get<float>()[7 * kPlaneElements];
        KdaRegbaseReverseScanCarry(
            (__ubuf__ float *)pending.GetPhyAddr(),
            (__ubuf__ float *)carry.GetPhyAddr(), 32);
        StoreRowsDirect(
            dgOutGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, task.begin, 0)],
            pending, 32, K_DIM, TensorRowElements());
    }

    __aicore__ inline void FinishHeadA5Dense32(
        const CIntraTask &task, uint32_t head, uint32_t rowStart,
        uint32_t validRows, uint32_t slot)
    {
        static_assert(K_DIM == 128,
                      "The A5 32-row Vector-Post path is specialized for K=128.");
        static_assert(!PUBLIC_VARLEN,
                      "The A5 32-row Vector-Post path is dense-only.");
        static_assert(kProcessRowBlock == 32,
                      "The A5 32-row Vector-Post path requires row32 tiles.");

        const uint32_t ownedRows = validRows;
        const uint32_t tokenBegin = task.begin + rowStart;
        const uint64_t slotBase = SlotBase(slot);

        // Six 32x128 FP32 matrices exactly fill the 96-KiB arena.  The raw
        // dq-base tile uses reduceTmp planes 0-3.  FinishScale consumes gate
        // before writing the matching dq-final element, so dqFinal may alias
        // gate and releases a complete 16-KiB tile from the live set.
        auto rawDq = Plane(0);       // planes 0-3
        auto rawDkLower = Plane(4);  // planes 4-7
        auto rawDkUpper = Plane(8);  // planes 8-11
        auto k = Plane(12);          // planes 12-15
        auto gate = Plane(16);       // planes 16-19; becomes dqFinal/output
        auto q = Plane(20);          // planes 20-23
        auto reduce = reduceTmp_.Get<float>();
        auto dqBaseRaw = reduce;                         // planes 0-3
        auto anchor = reduce[4 * kPlaneElements];        // plane 4
        auto beta = reduce[5 * kPlaneElements];          // plane 5
        auto anchorExp = reduce[6 * kPlaneElements];     // plane 6
        auto dbAcc = beta[128];

        KdaRegbaseFill(
            (__ubuf__ float *)dbAcc.GetPhyAddr(), 0.0f, ownedRows);

        const uint32_t anchorLocal =
            rowStart + 8 < task.end - task.begin ?
                rowStart + 8 : task.end - task.begin - 1;
        const uint32_t anchorRow = task.begin + anchorLocal;
        LoadScalarRows(
            beta,
            betaGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);

        constexpr uint32_t cols = 128;
        const uint32_t count = ownedRows * cols;
        // These FP32 operands already have permanent destinations in the
        // dense32 resident layout.  Copy them there directly instead of
        // staging through matrixInputPing/Pong and executing five 16-KiB
        // UB-to-UB copies.  One group event protects the shared resident
        // layout across consecutive heads/generations.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(directVToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            directMte3ToMte2Event_);
        CopyInMatrixRowsDirect(
            anchor,
            gkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, anchorRow, 0)],
            1, cols, TensorRowElements());
        CopyInMatrixRowsDirect(
            dqBaseRaw,
            dqBaseRawGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        const uint64_t resultBase =
            slotBase / sizeof(float) +
            tiling_.intraResultRegionOffset / sizeof(float);
        CopyInMatrixRowsDirect(
            rawDq,
            workspaceGm_[
                resultBase + tiling_.intraResultDqOffset / sizeof(float)],
            ownedRows, cols, K_DIM);
        CopyInMatrixRowsDirect(
            rawDkLower,
            workspaceGm_[
                resultBase +
                tiling_.intraResultDkLowerOffset / sizeof(float)],
            ownedRows, cols, K_DIM);
        CopyInMatrixRowsDirect(
            rawDkUpper,
            workspaceGm_[
                resultBase +
                tiling_.intraResultDkUpperOffset / sizeof(float)],
            ownedRows, cols, K_DIM);
        if (rowStart == 0) {
            const uint64_t secondUpperOffset =
                tiling_.intraResultDkUpperOffset / sizeof(float) +
                static_cast<uint64_t>(kProcessRowBlock) * K_DIM;
            CopyInMatrixRowsDirect(
                q,
                workspaceGm_[resultBase + secondUpperOffset],
                ownedRows, cols, K_DIM);
        }
        CopyInMatrixRowsDirect(
            gate,
            gkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(directMte2ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(directMte2ToVEvent_);
        if (rowStart == 0) {
            KdaRegbaseAdd2(
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                (__ubuf__ float *)q.GetPhyAddr(), count);
        }

        LoadMatrixRowsPair(
            q,
            qGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements(),
            k,
            kGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());

        KdaRegbaseExp2(
            (__ubuf__ float *)anchorExp.GetPhyAddr(),
            (__ubuf__ float *)anchor.GetPhyAddr(), cols);
        KdaRegbaseFinishScale<true>(
            (__ubuf__ float *)rawDq.GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
            (__ubuf__ float *)k.GetPhyAddr(),
            (__ubuf__ float *)gate.GetPhyAddr(),
            (__ubuf__ float *)anchor.GetPhyAddr(),
            (__ubuf__ float *)anchor.GetPhyAddr(),
            (__ubuf__ float *)beta.GetPhyAddr(),
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)dqBaseRaw.GetPhyAddr(),
            (__ubuf__ float *)anchorExp.GetPhyAddr(),
            (__ubuf__ float *)gate.GetPhyAddr(), tiling_.scale,
            ownedRows, cols);

        const uint32_t dkSlot = CopyInMatrixRows(
            dkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        const uint32_t dgSlot = CopyInMatrixRows(
            dgGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        StoreRowsDirect(
            dqOutGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            gate, ownedRows, cols, TensorRowElements());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[dkSlot]);
        KdaRegbaseAdd3(
            (__ubuf__ float *)dqBaseRaw.GetPhyAddr(),
            (__ubuf__ float *)MatrixInput<float>(dkSlot).GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(), count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[dkSlot]);
        StoreRowsDirect(
            dkOutGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            dqBaseRaw, ownedRows, cols, TensorRowElements());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[dgSlot]);
        auto dgTile = rowStart == 0 ?
            pendingDg_.Get<float>() : outputQueue_.AllocTensor<float>();
        KdaRegbaseDg(
            (__ubuf__ float *)dgTile.GetPhyAddr(),
            (__ubuf__ float *)MatrixInput<float>(dgSlot).GetPhyAddr(),
            (__ubuf__ float *)q.GetPhyAddr(),
            (__ubuf__ float *)rawDq.GetPhyAddr(),
            (__ubuf__ float *)k.GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(), count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(directVToMte2Event_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[dgSlot]);
        if (tiling_.useGateInKernel != 0 && rowStart != 0) {
            ApplyFusedSafeGateA5(task, head, dgTile, ownedRows);
        } else if (tiling_.deferGatePost != 0) {
            if (rowStart == 0) {
                StoreRowsDirect(
                    dgOutGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, tokenBegin, 0)],
                    dgTile, ownedRows, cols, TensorRowElements());
            } else {
                StorePreparedRows(
                    dgOutGm_[CIntraTensorOffset(
                        tiling_, task.batchIdx, head, tokenBegin, 0)],
                    dgTile, ownedRows, cols, TensorRowElements());
            }
        } else if (rowStart != 0) {
            auto carry = reduce[7 * kPlaneElements];
            KdaRegbaseReverseScanCarry(
                (__ubuf__ float *)dgTile.GetPhyAddr(),
                (__ubuf__ float *)carry.GetPhyAddr(), ownedRows);
            StorePreparedRows(
                dgOutGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, tokenBegin, 0)],
                dgTile, ownedRows, cols, TensorRowElements());
        }

        LoadScalarRows(
            beta,
            dbGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);
        KdaRegbaseAdd2(
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)beta.GetPhyAddr(), ownedRows);
        StoreScalarRows(
            dbOutGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            dbAcc, ownedRows);
    }

    __aicore__ inline void FinishHeadA5Dense16(
        const CIntraTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t slot, uint32_t ownedBegin)
    {
        static_assert(K_DIM == 128,
                      "The A5 16-row Vector-Post path is specialized for K=128.");
        static_assert(!PUBLIC_VARLEN,
                      "The A5 16-row Vector-Post path is dense-only.");
        static_assert(kProcessRowBlock == 16 || kProcessRowBlock == 32,
                      "The A5 Vector-Post path requires a 16- or 32-row outer tile.");

        if (ownedBegin >= validRows) {
            return;
        }
        const uint32_t ownedRows =
            ownedBegin + 16 <= validRows ? 16 : validRows - ownedBegin;
        const uint32_t tokenBegin = task.begin + rowStart + ownedBegin;
        const uint64_t slotBase = SlotBase(slot);

        // One 16 x 128 FP32 matrix occupies two consecutive 4 KiB planes.
        // Keep only the values that remain live across the scale and output
        // phases.  Once scale finishes, gate/anchor/beta are dead and their
        // planes are reused by inputGrad/output, so the existing 96 KiB arena
        // is sufficient and no additional UB is reserved.
        auto rawDq = Plane(0);       // planes 0-1
        auto rawDkLower = Plane(2);  // planes 2-3
        auto rawDkUpper = Plane(4);  // planes 4-5
        auto k = Plane(6);           // planes 6-7
        auto gate = Plane(8);        // planes 8-9; reused by inputGrad
        auto q = Plane(10);          // planes 10-11
        auto lowerAnchor = Plane(12);  // plane 12; reused by output
        auto upperAnchor = Plane(13);  // plane 13; reused by output
        auto scalars = Plane(14);
        auto beta = scalars;
        auto dbAcc = scalars[128];
        auto dqBaseRaw = Plane(15);  // planes 15-16
        auto anchorExp = Plane(17);  // first 128 elements
        auto dqFinal = Plane(18);    // planes 18-19

        KdaRegbaseFill(
            (__ubuf__ float *)dbAcc.GetPhyAddr(), 0.0f, ownedRows);

        const bool useSharedB =
            task.end - task.begin == CHUNK_SIZE &&
            validRows == kProcessRowBlock;
        const uint32_t localAnchor =
            rowStart + 8 < task.end - task.begin ?
                rowStart + 8 : task.end - task.begin - 1;
        const uint32_t lowerAnchorRow =
            task.begin + (useSharedB ? CHUNK_SIZE - 8 : localAnchor);
        const uint32_t upperAnchorRow =
            task.begin + (useSharedB ? 8 : localAnchor);
        LoadScalarRows(
            beta,
            betaGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);

        const uint32_t cols = 128;
        const uint32_t count = ownedRows * cols;
        const uint32_t qSlot = CopyInMatrixRows(
            qGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        const uint32_t kSlot = CopyInMatrixRows(
            kGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());

        // The remaining inputs are already FP32 and have permanent arena
        // destinations.  Move them there directly so Vector does not spend
        // seven extra copies shuttling them through matrixInputPing/Pong.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(directVToMte2Event_);
        CopyInMatrixRowsDirect(
            lowerAnchor,
            gkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, lowerAnchorRow, 0)],
            1, cols, TensorRowElements());
        CopyInMatrixRowsDirect(
            upperAnchor,
            gkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, upperAnchorRow, 0)],
            1, cols, TensorRowElements());
        CopyInMatrixRowsDirect(
            dqBaseRaw,
            dqBaseRawGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        const uint64_t resultBase =
            slotBase / sizeof(float) + tiling_.intraResultRegionOffset / sizeof(float);
        CopyInMatrixRowsDirect(
            gate,
            gkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        CopyInMatrixRowsDirect(
            rawDq,
            workspaceGm_[resultBase + tiling_.intraResultDqOffset / sizeof(float) +
                         static_cast<uint64_t>(ownedBegin) * K_DIM],
            ownedRows, cols, K_DIM);
        CopyInMatrixRowsDirect(
            rawDkLower,
            workspaceGm_[resultBase + tiling_.intraResultDkLowerOffset / sizeof(float) +
                         static_cast<uint64_t>(ownedBegin) * K_DIM],
            ownedRows, cols, K_DIM);
        CopyInMatrixRowsDirect(
            rawDkUpper,
            workspaceGm_[resultBase + tiling_.intraResultDkUpperOffset / sizeof(float) +
                         static_cast<uint64_t>(ownedBegin) * K_DIM],
            ownedRows, cols, K_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(directMte2ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(directMte2ToVEvent_);
        ConsumeMatrixRows(q, MatrixInput<DataT>(qSlot), qSlot, count);
        ConsumeMatrixRows(k, MatrixInput<DataT>(kSlot), kSlot, count);
        AscendC::PipeBarrier<PIPE_V>();

        KdaRegbaseExp2(
            (__ubuf__ float *)anchorExp.GetPhyAddr(),
            (__ubuf__ float *)lowerAnchor.GetPhyAddr(), cols);
        if (useSharedB) {
            KdaRegbaseFinishScale<true, true>(
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                (__ubuf__ float *)k.GetPhyAddr(),
                (__ubuf__ float *)gate.GetPhyAddr(),
                (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                (__ubuf__ float *)upperAnchor.GetPhyAddr(),
                (__ubuf__ float *)beta.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dqBaseRaw.GetPhyAddr(),
                (__ubuf__ float *)anchorExp.GetPhyAddr(),
                (__ubuf__ float *)dqFinal.GetPhyAddr(), tiling_.scale,
                ownedRows, cols);
        } else {
            KdaRegbaseFinishScale<true>(
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                (__ubuf__ float *)k.GetPhyAddr(),
                (__ubuf__ float *)gate.GetPhyAddr(),
                (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                (__ubuf__ float *)lowerAnchor.GetPhyAddr(),
                (__ubuf__ float *)beta.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)dqBaseRaw.GetPhyAddr(),
                (__ubuf__ float *)anchorExp.GetPhyAddr(),
                (__ubuf__ float *)dqFinal.GetPhyAddr(), tiling_.scale,
                ownedRows, cols);
        }

        auto output = Plane(12);  // planes 12-13, after anchor/beta's last use

        // dq_base is completed above from Kernel A's raw dq tile and written
        // directly with the Intra correction, so only the mature dk/dg matrix
        // input ping/pong remains here.  Issue both MTE2 reads before storing
        // dq_final to overlap input traffic with the independent MTE3 write.
        const uint32_t dkSlot = CopyInMatrixRows(
            dkGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        const uint32_t dgSlot = CopyInMatrixRows(
            dgGm_[CIntraTensorOffset(
                tiling_, task.batchIdx, head, tokenBegin, 0)],
            ownedRows, cols, TensorRowElements());
        StoreRows(dqOutGm_[CIntraTensorOffset(
                      tiling_, task.batchIdx, head, tokenBegin, 0)],
                  dqFinal, ownedRows, cols, TensorRowElements());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[dkSlot]);
        KdaRegbaseAdd3(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)MatrixInput<float>(dkSlot).GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
            count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[dkSlot]);
        StoreRows(dkOutGm_[CIntraTensorOffset(
                      tiling_, task.batchIdx, head, tokenBegin, 0)],
                  output, ownedRows, cols, TensorRowElements());

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[dgSlot]);
        KdaRegbaseDg(
            (__ubuf__ float *)output.GetPhyAddr(),
            (__ubuf__ float *)MatrixInput<float>(dgSlot).GetPhyAddr(),
            (__ubuf__ float *)q.GetPhyAddr(),
            (__ubuf__ float *)rawDq.GetPhyAddr(),
            (__ubuf__ float *)k.GetPhyAddr(),
            (__ubuf__ float *)rawDkLower.GetPhyAddr(),
            (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
            count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[dgSlot]);
        if (tiling_.useGateInKernel == 0) {
            if (rowStart == 0) {
                auto pending = pendingDg_.Get<float>()[ownedBegin * K_DIM];
                KdaRegbaseCopy(
                    (__ubuf__ float *)pending.GetPhyAddr(),
                    (__ubuf__ float *)output.GetPhyAddr(),
                    static_cast<uint16_t>(count));
            } else {
                auto carry = reduceTmp_.Get<float>();
                KdaRegbaseReverseScanCarry(
                    (__ubuf__ float *)output.GetPhyAddr(),
                    (__ubuf__ float *)carry.GetPhyAddr(),
                    static_cast<uint16_t>(ownedRows));
                StoreRows(dgOutGm_[CIntraTensorOffset(
                              tiling_, task.batchIdx, head, tokenBegin, 0)],
                          output, ownedRows, cols, TensorRowElements());
            }
        } else {
            StoreRows(dgOutGm_[CIntraTensorOffset(
                          tiling_, task.batchIdx, head, tokenBegin, 0)],
                      output, ownedRows, cols, TensorRowElements());
        }

        // output is no longer live; plane 13 can hold the final db input.
        LoadScalarRows(
            beta,
            dbGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);
        KdaRegbaseAdd2(
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)beta.GetPhyAddr(),
            ownedRows);
        StoreScalarRows(
            dbOutGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            dbAcc, ownedRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(directVToMte2Event_);
    }
#endif

    __aicore__ inline void FinishHead(
        const CIntraTask &task, uint32_t head, uint32_t rowStart, uint32_t validRows,
        uint32_t slot)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!PUBLIC_VARLEN && K_DIM == 128 &&
                      kProcessRowBlock == 32) {
            FinishHeadA5Dense16(
                task, head, rowStart, validRows, slot,
                AscendC::GetSubBlockIdx() * 16U);
            return;
        }
#endif
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t rowsPerSubBlock = kProcessRowBlock / 2;
        // Preserve the proven eight-row path for A2/A3, A5 varlen and all
        // non-K128 specializations.
        for (uint32_t ownedOffset = 0; ownedOffset < rowsPerSubBlock;
             ownedOffset += 8) {
            const uint32_t ownedBegin =
                subBlock * rowsPerSubBlock + ownedOffset;
            if (ownedBegin >= validRows) {
                continue;
            }
            const uint32_t ownedRows =
                ownedBegin + 8 <= validRows ? 8 : validRows - ownedBegin;
            const uint32_t tokenBegin = task.begin + rowStart + ownedBegin;
            const uint64_t slotBase = SlotBase(slot);

        auto rawDq = Plane(0);
        auto rawDkLower = Plane(1);
        auto rawDkUpper = Plane(2);
        auto q = Plane(3);
        auto k = Plane(4);
        auto gate = Plane(5);
        auto anchor = Plane(6);
        auto posScale = Plane(7);
        auto negScale = Plane(8);
        auto beta = Plane(9);
        auto betaBroadcast = Plane(10);
        auto inputGrad = Plane(11);
        auto temp = Plane(12);
        auto output = Plane(13);
        auto product = Plane(14);
        auto rowReduce = Plane(15);
        auto dbAcc = Plane(16);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseFill(
            (__ubuf__ float *)dbAcc.GetPhyAddr(), 0.0f, ownedRows);
#else
        AscendC::Duplicate(dbAcc, 0.0f, ownedRows);
#endif

        const uint32_t anchorLocal = rowStart + 8 < task.end - task.begin ?
                                     rowStart + 8 : task.end - task.begin - 1;
        const uint32_t anchorRow = task.begin + anchorLocal;
        LoadScalarRows(
            beta,
            betaGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        AscendC::Brcb(betaBroadcast, beta, static_cast<uint8_t>((ownedRows + 7) / 8), {1, 8});
#endif

        for (uint32_t col = 0; col < K_DIM; col += 128) {
            const uint32_t cols = col + 128 <= K_DIM ? 128 : K_DIM - col;
            const uint32_t count = ownedRows * cols;
            Load(anchor,
                 gkGm_[CIntraTensorOffset(
                     tiling_, task.batchIdx, head, anchorRow, col)],
                 cols);
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
            for (uint32_t row = 1; row < ownedRows; ++row) {
                AscendC::Adds(anchor[row * cols], anchor, 0.0f, cols);
            }
#endif
            const uint64_t resultBase =
                slotBase / sizeof(float) + tiling_.intraResultRegionOffset / sizeof(float);
            LoadMatrixRowsPair(
                q,
                qGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, tokenBegin, col)],
                ownedRows, cols, TensorRowElements(),
                k,
                kGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, tokenBegin, col)],
                ownedRows, cols, TensorRowElements());
            LoadMatrixRowsPair(
                gate,
                gkGm_[CIntraTensorOffset(
                    tiling_, task.batchIdx, head, tokenBegin, col)],
                ownedRows, cols, TensorRowElements(),
                rawDq,
                workspaceGm_[resultBase + tiling_.intraResultDqOffset / sizeof(float) +
                             static_cast<uint64_t>(ownedBegin) * K_DIM + col],
                ownedRows, cols, K_DIM);
            LoadMatrixRowsPair(
                rawDkLower,
                workspaceGm_[resultBase + tiling_.intraResultDkLowerOffset / sizeof(float) +
                             static_cast<uint64_t>(ownedBegin) * K_DIM + col],
                ownedRows, cols, K_DIM,
                rawDkUpper,
                workspaceGm_[resultBase + tiling_.intraResultDkUpperOffset / sizeof(float) +
                             static_cast<uint64_t>(ownedBegin) * K_DIM + col],
                ownedRows, cols, K_DIM);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseFinishScale<false>(
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                (__ubuf__ float *)k.GetPhyAddr(),
                (__ubuf__ float *)gate.GetPhyAddr(),
                (__ubuf__ float *)anchor.GetPhyAddr(),
                (__ubuf__ float *)anchor.GetPhyAddr(),
                (__ubuf__ float *)beta.GetPhyAddr(),
                (__ubuf__ float *)dbAcc.GetPhyAddr(),
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                (__ubuf__ float *)anchor.GetPhyAddr(),
                (__ubuf__ float *)rawDq.GetPhyAddr(), 0.0f,
                ownedRows, cols);
#else
            AscendC::Sub(posScale, gate, anchor, count);
            AscendC::Sub(negScale, anchor, gate, count);
            Exp2(posScale, posScale, count);
            Exp2(negScale, negScale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(rawDq, rawDq, posScale, count);
            AscendC::Mul(rawDkLower, rawDkLower, posScale, count);
            AscendC::Mul(rawDkUpper, rawDkUpper, negScale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, rawDkLower, k, count);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t reduceShape[2] = {ownedRows, cols};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                rowReduce, product, reduceTmp_.Get<uint8_t>(), reduceShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(dbAcc, dbAcc, rowReduce, ownedRows);

            const uint8_t rowStride = static_cast<uint8_t>(cols * sizeof(float) / 32);
            for (uint32_t offset = 0; offset < cols; offset += 64) {
                const uint32_t mask = offset + 64 <= cols ? 64 : cols - offset;
                AscendC::Mul(rawDkLower[offset], rawDkLower[offset], betaBroadcast,
                             mask, ownedRows, {1, 1, 0, rowStride, rowStride, 1});
            }
            AscendC::PipeBarrier<PIPE_V>();
#endif

            LoadRows(inputGrad,
                     dqGm_[CIntraTensorOffset(
                         tiling_, task.batchIdx, head, tokenBegin, col)],
                     ownedRows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseAdd2(
                (__ubuf__ float *)output.GetPhyAddr(),
                (__ubuf__ float *)inputGrad.GetPhyAddr(),
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                count);
#else
            AscendC::Add(output, inputGrad, rawDq, count);
#endif
            StoreRows(dqOutGm_[CIntraTensorOffset(
                          tiling_, task.batchIdx, head, tokenBegin, col)],
                      output, ownedRows, cols, TensorRowElements());

            LoadRows(inputGrad,
                     dkGm_[CIntraTensorOffset(
                         tiling_, task.batchIdx, head, tokenBegin, col)],
                     ownedRows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseAdd3(
                (__ubuf__ float *)output.GetPhyAddr(),
                (__ubuf__ float *)inputGrad.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                count);
#else
            AscendC::Add(temp, rawDkLower, rawDkUpper, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(output, inputGrad, temp, count);
#endif
            StoreRows(dkOutGm_[CIntraTensorOffset(
                          tiling_, task.batchIdx, head, tokenBegin, col)],
                      output, ownedRows, cols, TensorRowElements());

            LoadRows(inputGrad,
                     dgGm_[CIntraTensorOffset(
                         tiling_, task.batchIdx, head, tokenBegin, col)],
                     ownedRows, cols, TensorRowElements());
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaRegbaseDg(
                (__ubuf__ float *)output.GetPhyAddr(),
                (__ubuf__ float *)inputGrad.GetPhyAddr(),
                (__ubuf__ float *)q.GetPhyAddr(),
                (__ubuf__ float *)rawDq.GetPhyAddr(),
                (__ubuf__ float *)k.GetPhyAddr(),
                (__ubuf__ float *)rawDkLower.GetPhyAddr(),
                (__ubuf__ float *)rawDkUpper.GetPhyAddr(),
                count);
#else
            AscendC::Mul(temp, q, rawDq, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(output, inputGrad, temp, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(temp, rawDkLower, rawDkUpper, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(temp, k, temp, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(output, output, temp, count);
#endif
            StoreRows(dgOutGm_[CIntraTensorOffset(
                          tiling_, task.batchIdx, head, tokenBegin, col)],
                      output, ownedRows, cols, TensorRowElements());
        }

        LoadScalarRows(
            beta,
            dbGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            ownedRows);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaRegbaseAdd2(
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)dbAcc.GetPhyAddr(),
            (__ubuf__ float *)beta.GetPhyAddr(),
            ownedRows);
#else
        AscendC::Add(dbAcc, dbAcc, beta, ownedRows);
#endif
        StoreScalarRows(
            dbOutGm_[CIntraScalarOffset(
                tiling_, task.batchIdx, head, tokenBegin)],
            dbAcc, ownedRows);
        }
    }

    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR gk_;
    GM_ADDR beta_;
    GM_ADDR dAqk_;
    GM_ADDR dAkk_;
    GM_ADDR dqBaseRaw_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR db_;
    GM_ADDR dg_;
    GM_ADDR dqOut_;
    GM_ADDR dkOut_;
    GM_ADDR dbOut_;
    GM_ADDR dgOut_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkMetadata_;
    GM_ADDR rawG_;
    GM_ADDR aLog_;
    GM_ADDR dtBias_;
    GM_ADDR dA_;
    GM_ADDR dBias_;
    GM_ADDR workspace_;

    ChunkKdaBwdCTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outputQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> matrixInputPing_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> matrixInputPong_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> arena_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceTmp_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    AscendC::TBuf<AscendC::QuePosition::VECCALC> pendingDg_;
#endif
    uint32_t currentMatrixInputSlot_ = 0;
    event_t matrixMte2ToVEvent_[kMatrixInputBufferCount]{};
    event_t matrixVToMte2Event_[kMatrixInputBufferCount]{};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    event_t directMte2ToVEvent_{};
    event_t directVToMte2Event_{};
    event_t directVToMte3Event_{};
    event_t directMte3ToMte2Event_{};
    Catlass::Arch::CrossCoreFlag vecToCubeReadyFlag_{kVecToCubeReadyFlag};
    Catlass::Arch::CrossCoreFlag cubeToVecReadyFlag_{kCubeToVecReadyFlag};
#endif
    AscendC::GlobalTensor<DataT> qGm_;
    AscendC::GlobalTensor<DataT> kGm_;
    AscendC::GlobalTensor<float> gkGm_;
    AscendC::GlobalTensor<BetaT> betaGm_;
    AscendC::GlobalTensor<float> dAqkGm_;
    AscendC::GlobalTensor<float> dAkkGm_;
    AscendC::GlobalTensor<float> dqBaseRawGm_;
    AscendC::GlobalTensor<float> dqGm_;
    AscendC::GlobalTensor<float> dkGm_;
    AscendC::GlobalTensor<float> dbGm_;
    AscendC::GlobalTensor<float> dgGm_;
    AscendC::GlobalTensor<float> dqOutGm_;
    AscendC::GlobalTensor<float> dkOutGm_;
    AscendC::GlobalTensor<float> dbOutGm_;
    AscendC::GlobalTensor<float> dgOutGm_;
    AscendC::GlobalTensor<RawGateT> rawGGm_;
    AscendC::GlobalTensor<float> aLogGm_;
    AscendC::GlobalTensor<float> dtBiasGm_;
    AscendC::GlobalTensor<float> dAGm_;
    AscendC::GlobalTensor<float> dBiasGm_;
    AscendC::GlobalTensor<int64_t> chunkMetadataGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_INTRA_VECTOR_H


#endif // CHUNK_KDA_BWD_FINALIZE_INTRA_H
