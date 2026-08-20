#ifndef CHUNK_KDA_BWD_FINALIZE_WY_H
#define CHUNK_KDA_BWD_FINALIZE_WY_H

#include "chunk_kda_bwd_common.h"

#ifndef CHUNK_KDA_BWD_C_COMMON_H
#define CHUNK_KDA_BWD_C_COMMON_H

#include "kernel_operator.h"
namespace KDA {

constexpr uint32_t kWyChunkSize = 64;
constexpr uint32_t kWyKeyDim = 128;
constexpr uint32_t kWyValueDim = 128;
constexpr uint32_t kWyHeadsPerWindow = 2;
// Keep WY and Intra on the same two-head owner grid.  Besides simplifying
// the MIX handshake, this is a correctness requirement: the kernel has no
// grid-wide barrier between WY, Intra and Gate, so a later phase may only
// consume data produced by the same physical core.
constexpr uint32_t kWyFusedHeadsPerWindow = 2;
constexpr uint32_t kWyWorkspaceSlotCount = 4;
// Match the proven two-slot flag arrays used by the fused GDN forward
// schedulers: each head lane owns a distinct synchronization channel.  The
// dependent Cube result can reuse that lane's Cube->Vector flag after the
// base-ready event has been consumed.
__aicore__ inline uint32_t WyVectorToCubeFlag(uint32_t headInWindow)
{
    return 2U + headInWindow;
}

__aicore__ inline uint32_t WyCubeToVectorFlag(uint32_t headInWindow)
{
    return 4U + headInWindow;
}

struct WyChunkTask {
    uint32_t sequence;
    uint32_t batchIdx;
    uint32_t chunkIdx;
    uint32_t begin;
    uint32_t end;
};

__aicore__ inline WyChunkTask GetWyChunkTask(
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdCTilingData &tiling, uint32_t taskIdx)
{
    WyChunkTask task{};
    if (tiling.isVarLen != 0) {
        AscendC::GlobalTensor<int64_t> cu;
        AscendC::GlobalTensor<int64_t> chunks;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        chunks.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        task.sequence = static_cast<uint32_t>(chunks.GetValue(2 * taskIdx));
        task.batchIdx = 0;
        const uint32_t localChunk =
            static_cast<uint32_t>(chunks.GetValue(2 * taskIdx + 1));
        task.chunkIdx = taskIdx;
        const uint32_t seqBegin =
            static_cast<uint32_t>(cu.GetValue(task.sequence));
        const uint32_t seqEnd =
            static_cast<uint32_t>(cu.GetValue(task.sequence + 1));
        task.begin = localChunk * static_cast<uint32_t>(tiling.chunkSize) +
                     seqBegin;
        task.end = task.begin + static_cast<uint32_t>(tiling.chunkSize);
        if (task.end > seqEnd) {
            task.end = seqEnd;
        }
        return task;
    }
    task.batchIdx = taskIdx / static_cast<uint32_t>(tiling.chunkNumPerBatch);
    task.sequence = task.batchIdx;
    task.chunkIdx = taskIdx % static_cast<uint32_t>(tiling.chunkNumPerBatch);
    task.begin = task.chunkIdx * static_cast<uint32_t>(tiling.chunkSize);
    task.end = task.begin + static_cast<uint32_t>(tiling.chunkSize);
    if (task.end > static_cast<uint32_t>(tiling.seqlen)) {
        task.end = static_cast<uint32_t>(tiling.seqlen);
    }
    return task;
}

__aicore__ inline uint64_t WyTokenOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx, uint32_t width)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx) *
               width;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
            tiling.seqlen + tokenIdx) * width;
}

// Stage 3 stores BF16 Zb and stage 6 stores BF16 T_za in the two halves of
// this task's final FP32 dAkk allocation.  Both temporaries therefore remain
// inside the task's own byte range until stage 7 writes the final FP32 tile.
__aicore__ inline uint64_t WyDAkkBf16TaskBase(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx)
{
    return 2U * WyTokenOffset(tiling, batchIdx, headIdx, tokenIdx, 64);
}

// Saved h keeps the forward sequence/chunk-major layout. Kernel B's dh can
// use PR291's head-major layout; C selects the matching offset directly, so
// no transpose kernel or GM copy is introduced.
__aicore__ inline uint64_t WySavedHOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t chunkIdx)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(chunkIdx) * tiling.headNum + headIdx) *
               tiling.keyDim * tiling.valueDim;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.chunkNumPerBatch +
             chunkIdx) * tiling.headNum + headIdx) *
           tiling.keyDim * tiling.valueDim;
}

__aicore__ inline uint64_t WyDhOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t chunkIdx)
{
    if (tiling.dhHeadMajor != 0) {
        if (tiling.isVarLen != 0) {
            return (static_cast<uint64_t>(headIdx) * tiling.chunkNum +
                    chunkIdx) * tiling.keyDim * tiling.valueDim;
        }
        return ((static_cast<uint64_t>(batchIdx) * tiling.headNum +
                 headIdx) * tiling.chunkNumPerBatch + chunkIdx) *
               tiling.keyDim * tiling.valueDim;
    }
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(chunkIdx) * tiling.headNum + headIdx) *
               tiling.keyDim * tiling.valueDim;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.chunkNumPerBatch +
             chunkIdx) * tiling.headNum + headIdx) *
           tiling.keyDim * tiling.valueDim;
}

__aicore__ inline uint64_t WyWorkspaceSlotBase(
    const ChunkKdaBwdCTilingData &tiling, uint32_t logicalCore,
    uint32_t generation, uint32_t headInWindow)
{
    // Adjacent generations use disjoint two-head windows.  A5 returns a
    // per-parity completion credit before a later generation reuses the slot;
    // A2/A3 retain the single-generation fallback.
    const uint32_t slot = ((generation & 1U) * kWyFusedHeadsPerWindow) +
                          headInWindow;
    return (static_cast<uint64_t>(logicalCore) * tiling.workspaceSlotCount + slot) *
           tiling.workspaceSlotSize;
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_COMMON_H


#ifndef CHUNK_KDA_BWD_C_CUBE_H
#define CHUNK_KDA_BWD_C_CUBE_H

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif
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

struct WyTileGemmDirectEvent {
    static constexpr int32_t kL1A = 0;
    static constexpr int32_t kL1B = 1;
    static constexpr int32_t kL0A = 0;
    static constexpr int32_t kL0B = 1;
    static constexpr int32_t kL0C = 0;
};

// The fused path owns the MM layout and L1/L0 event lifecycle at phase scope.
// Individual GEMMs only bind the shared resource buffers.  BF16/FP32 phase
// boundaries are still fully drained, preserving the proven type-transition
// safety while avoiding constructor/destructor synchronization for every
// contraction.
template <class ArchTag_, class ElementC_, class TileCopy_,
          uint32_t ReductionTile_ = 256>
struct WyTileGemmDirect {
    using ArchTag = ArchTag_;
    using TileCopy = TileCopy_;
    using ElementA = typename TileCopy::ElementA;
    using ElementB = typename TileCopy::ElementB;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, ElementA, LayoutTagL1A>;
    static constexpr int32_t kEventL1A = WyTileGemmDirectEvent::kL1A;
    static constexpr int32_t kEventL1B = WyTileGemmDirectEvent::kL1B;
    static constexpr int32_t kEventL0A = WyTileGemmDirectEvent::kL0A;
    static constexpr int32_t kEventL0B = WyTileGemmDirectEvent::kL0B;
    static constexpr int32_t kEventL0C = WyTileGemmDirectEvent::kL0C;

    // P0's largest reduction is V=256.  Its worst L0A/L0B products are
    // 64x256 and 256x128 respectively, both 32/64 KiB in FP16/BF16, so keep
    // the whole reduction in one MMAD.  Besides matching the proven PR190
    // direct-tile pattern, this avoids an unnecessary partial-sum lifecycle.
    static constexpr uint32_t kReductionTile = ReductionTile_;
    static constexpr uint32_t kL1PlaneBytes =
        128 * 256 * sizeof(ElementA);
    static_assert(2 * kL1PlaneBytes <= 512 * 1024,
                  "Kernel C direct MMAD L1 planes exceed A2/A3 L1");
    static_assert(64 * 256 * sizeof(ElementA) <= ArchTag::L0A_SIZE,
                  "Kernel C direct MMAD L0A exceeds capacity");
    static_assert(128 * kReductionTile * sizeof(ElementB) <= ArchTag::L0B_SIZE,
                  "Kernel C direct MMAD L0B tile exceeds capacity");
    static_assert(64 * 256 * sizeof(ElementAccumulator) <= ArchTag::L0C_SIZE,
                  "Kernel C V256 FP32 L0C exceeds capacity");

    CATLASS_DEVICE
    explicit WyTileGemmDirect(Catlass::Arch::Resource<ArchTag> &resource)
    {
        if ASCEND_IS_AIC {
            l1A_ = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1B_ = resource.l1Buf.template GetBufferByByte<ElementB>(
                kL1PlaneBytes);
            l0A_ = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0B_ = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0C_ =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        }
    }

    CATLASS_DEVICE ~WyTileGemmDirect() = default;

    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE
    void operator()(TensorA &a, TensorB &b, TensorC &c,
                    Catlass::GemmCoord const &shape)
    {
        using CopyGmToL1A =
            typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B =
            typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyL0CToDst = KdaBwdCopyL0CToDst<TileCopy, TensorC>;
        CopyGmToL1A copyGmA;
        CopyGmToL1B copyGmB;
        CopyL1ToL0A copyL0A;
        CopyL1ToL0B copyL0B;
        CopyL0CToDst copyC;
        TileMmad mm;

        const uint32_t m = shape.m() == 1 ? 16 : shape.m();
        const uint32_t n = shape.n();
        const uint32_t k = shape.k();
        auto l1A = tla::MakeTensor(
            l1A_, tla::MakeLayout<ElementA, LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        auto l1B = tla::MakeTensor(
            l1B_, tla::MakeLayout<ElementB, LayoutTagL1B>(k, n),
            Catlass::Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        copyGmA(l1A, a);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
        copyGmB(l1B, b);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);

        auto l0C = tla::MakeTensor(
            l0C_, tla::MakeLayoutL0C(m, n),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
        // GM->L1 completes once for the complete logical operands.  The
        // following reduction tiles only read disjoint views from the same
        // resident L1 tensors, so consume the ready credits once rather than
        // waiting for a new MTE2 producer on every k0 iteration.
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B);
        for (uint32_t k0 = 0; k0 < k; k0 += kReductionTile) {
            const uint32_t curK =
                k - k0 < kReductionTile ? k - k0 : kReductionTile;
            auto l0A = tla::MakeTensor(
                l0A_,
                tla::MakeLayout<ElementA, LayoutTagL0A>(m, curK),
                Catlass::Arch::PositionL0A{});
            auto l0B = tla::MakeTensor(
                l0B_,
                tla::MakeLayout<ElementB, LayoutTagL0B>(curK, n),
                Catlass::Arch::PositionL0B{});
            auto tileA = GetTile(
                l1A, tla::MakeCoord(0, k0), tla::MakeShape(m, curK));
            auto tileB = GetTile(
                l1B, tla::MakeCoord(k0, 0), tla::MakeShape(curK, n));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
            copyL0A(l0A, tileA);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
            copyL0B(l0B, tileB);
            const bool lastK = k0 + curK == k;
            if (lastK) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C);
            const uint8_t unitFlag = lastK ? 0b11 : 0b10;
            mm(l0C, l0A, l0B, m, n, curK, k0 == 0, unitFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C);
        copyC(c, l0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C);
    }

private:
    AscendC::LocalTensor<ElementA> l1A_;
    AscendC::LocalTensor<ElementB> l1B_;
    AscendC::LocalTensor<ElementA> l0A_;
    AscendC::LocalTensor<ElementB> l0B_;
    AscendC::LocalTensor<ElementAccumulator> l0C_;
};

// A5-local two-output path for contractions that share the complete left
// operand.  The left tile is copied to L1/L0A once.  The two right tiles use
// independent L1B buffers, so MTE2 can prefetch the second right operand while
// MTE1/MMAD consumes the first one.  On A5 V128, two 32-KiB L0B planes
// additionally let MTE1 stage the second right tile while M computes the
// first contraction.  V256 keeps the single-buffer fallback because two
// worst-case L0B planes do not fit the 64-KiB hardware capacity.
// Two compact L0C result planes let FIX copy the first output while M computes
// the second.
template <class ArchTag_, class ElementC_, class TileCopy_>
struct WyTileGemmSharedLeftDualRightDirect {
    using ArchTag = ArchTag_;
    using TileCopy = TileCopy_;
    using ElementA = typename TileCopy::ElementA;
    using ElementB = typename TileCopy::ElementB;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, ElementA, LayoutTagL1A>;

    static constexpr int32_t kEventL1A = WyTileGemmDirectEvent::kL1A;
    static constexpr int32_t kEventL1B0 = WyTileGemmDirectEvent::kL1B;
    static constexpr int32_t kEventL1B1 = 2;
    static constexpr int32_t kEventL0A = WyTileGemmDirectEvent::kL0A;
    static constexpr int32_t kEventL0B0 = WyTileGemmDirectEvent::kL0B;
    static constexpr int32_t kEventL0B1 = 2;
    static constexpr int32_t kEventL0C0 = WyTileGemmDirectEvent::kL0C;
    static constexpr int32_t kEventL0C1 = 1;
    static constexpr uint32_t kL1PlaneBytes =
        128 * 256 * sizeof(ElementA);
    static constexpr uint32_t kL0BPlaneBytes =
        128 * 128 * sizeof(ElementB);
    static constexpr uint32_t kL0CPlaneBytes =
        64 * 128 * sizeof(ElementAccumulator);
    static_assert(3 * kL1PlaneBytes <= 512 * 1024,
                  "Kernel C shared-left L1 buffers exceed capacity");
    static_assert(2 * kL0BPlaneBytes <= ArchTag::L0B_SIZE,
                  "Kernel C shared-left L0B double buffer exceeds capacity");
    static_assert(2 * kL0CPlaneBytes <= ArchTag::L0C_SIZE,
                  "Kernel C dual-output L0C buffers exceed capacity");

    CATLASS_DEVICE
    explicit WyTileGemmSharedLeftDualRightDirect(
        Catlass::Arch::Resource<ArchTag> &resource)
    {
        if ASCEND_IS_AIC {
            l1A_ = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1B0_ = resource.l1Buf.template GetBufferByByte<ElementB>(
                kL1PlaneBytes);
            l1B1_ = resource.l1Buf.template GetBufferByByte<ElementB>(
                2 * kL1PlaneBytes);
            l0A_ = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0B0_ = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0B1_ = resource.l0BBuf.template GetBufferByByte<ElementB>(
                kL0BPlaneBytes);
            l0C0_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            l0C1_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                kL0CPlaneBytes);
        }
    }

    template <class TensorA, class TensorB0, class TensorC0,
              class TensorB1, class TensorC1>
    CATLASS_DEVICE
    void operator()(TensorA &a, TensorB0 &b0, TensorC0 &c0,
                    Catlass::GemmCoord const &shape0,
                    TensorB1 &b1, TensorC1 &c1,
                    Catlass::GemmCoord const &shape1)
    {
        using CopyGmToL1A =
            typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B0 =
            typename TileCopy::template CopyGmToL1B<TensorB0>;
        using CopyGmToL1B1 =
            typename TileCopy::template CopyGmToL1B<TensorB1>;
        using CopyL0CToDst0 = KdaBwdCopyL0CToDst<TileCopy, TensorC0>;
        using CopyL0CToDst1 = KdaBwdCopyL0CToDst<TileCopy, TensorC1>;
        CopyGmToL1A copyGmA;
        CopyGmToL1B0 copyGmB0;
        CopyGmToL1B1 copyGmB1;
        CopyL1ToL0A copyL0A;
        CopyL1ToL0B copyL0B;
        CopyL0CToDst0 copyC0;
        CopyL0CToDst1 copyC1;
        TileMmad mm;

        const uint32_t m = shape0.m() == 1 ? 16 : shape0.m();
        const uint32_t k = shape0.k();
        const uint32_t n0 = shape0.n();
        const uint32_t n1 = shape1.n();
        auto l1A = tla::MakeTensor(
            l1A_, tla::MakeLayout<ElementA, LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        auto l1B0 = tla::MakeTensor(
            l1B0_, tla::MakeLayout<ElementB, LayoutTagL1B>(k, n0),
            Catlass::Arch::PositionL1{});
        auto l1B1 = tla::MakeTensor(
            l1B1_, tla::MakeLayout<ElementB, LayoutTagL1B>(k, n1),
            Catlass::Arch::PositionL1{});
        auto tileL1A = GetTile(
            l1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B0 = GetTile(
            l1B0, tla::MakeCoord(0, 0), tla::MakeShape(k, n0));
        auto tileL1B1 = GetTile(
            l1B1, tla::MakeCoord(0, 0), tla::MakeShape(k, n1));

        // All three L1 events are owned by the surrounding S3 MMAD phase.
        // Hoisting event-2 initialization/drain out of the per-head call is
        // essential: otherwise the synchronization cost erases the saved
        // left-operand copy.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);
        copyGmA(l1A, a);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B0);
        copyGmB0(l1B0, b0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B1);
        copyGmB1(l1B1, b1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B1);

        auto l0A = tla::MakeTensor(
            l0A_, tla::MakeLayout<ElementA, LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        copyL0A(l0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1A);

        auto l0B0 = tla::MakeTensor(
            l0B0_, tla::MakeLayout<ElementB, LayoutTagL0B>(k, n0),
            Catlass::Arch::PositionL0B{});
        auto l0C0 = tla::MakeTensor(
            l0C0_, tla::MakeLayoutL0C(m, n0),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B0);
        copyL0B(l0B0, tileL1B0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C0);
        mm(l0C0, l0A, l0B0, m, n0, k, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C0);

        // The first MMAD is now live on PIPE_M.  Stage the second right tile
        // through the independent L0B plane so PIPE_MTE1 can overlap it.
        auto l0B1 = tla::MakeTensor(
            l0B1_, tla::MakeLayout<ElementB, LayoutTagL0B>(k, n1),
            Catlass::Arch::PositionL0B{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1B1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B1);
        copyL0B(l0B1, tileL1B1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1B1);

        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C0);
        copyC0(c0, l0C0, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);

        auto l0C1 = tla::MakeTensor(
            l0C1_, tla::MakeLayoutL0C(m, n1),
            Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventL0C1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventL0C1);
        mm(l0C1, l0A, l0B1, m, n1, k, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B1);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C1);
        copyC1(c1, l0C1, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
    }

private:
    AscendC::LocalTensor<ElementA> l1A_;
    AscendC::LocalTensor<ElementB> l1B0_;
    AscendC::LocalTensor<ElementB> l1B1_;
    AscendC::LocalTensor<ElementA> l0A_;
    AscendC::LocalTensor<ElementB> l0B0_;
    AscendC::LocalTensor<ElementB> l0B1_;
    AscendC::LocalTensor<ElementAccumulator> l0C0_;
    AscendC::LocalTensor<ElementAccumulator> l0C1_;
};

template <typename DataT, uint32_t V_DIM>
class ChunkKdaBwdCCubeProcess {
public:
    __aicore__ ChunkKdaBwdCCubeProcess(
        GM_ADDR v, GM_ADDR vNew, GM_ADDR a, GM_ADDR h, GM_ADDR dh,
        GM_ADDR dvScan, GM_ADDR dq, GM_ADDR dk, GM_ADDR dg,
        GM_ADDR dAkk, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR workspace)
        : v_(v), vNew_(vNew), a_(a), h_(h), dh_(dh),
          dvScan_(dvScan), dq_(dq), dk_(dk), dg_(dg), dAkk_(dAkk),
          cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices),
          workspace_(workspace) {}

    __aicore__ inline void Init(const ChunkKdaBwdCTilingData &tiling)
    {
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        AscendC::SetHF32Mode(false);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ArchTag = Catlass::Arch::Ascend950;
#else
        using ArchTag = Catlass::Arch::AtlasA2;
#endif
        using RowMajor = Catlass::layout::RowMajor;
        using ColumnMajor = Catlass::layout::ColumnMajor;
        using Element = DataT;

        using Fp32C64Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, RowMajor, Element, ColumnMajor, float, RowMajor>;
        using ElementC64Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, RowMajor, Element, ColumnMajor, Element, RowMajor>;
        using Fp32AT64x128Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, ColumnMajor, Element, RowMajor, float, RowMajor>;
        using ElementSquareRightTransposeCopy =
            Catlass::Gemm::Tile::PackedTileCopyTla<
                ArchTag, Element, RowMajor, Element, ColumnMajor, Element, RowMajor>;
        using Fp32SquareLeftTransposeCopy =
            Catlass::Gemm::Tile::PackedTileCopyTla<
                ArchTag, Element, ColumnMajor, Element, RowMajor, float, RowMajor>;

        using Fp32C64Mmad =
            WyTileGemmDirect<ArchTag, float, Fp32C64Copy>;
        using ElementC64Mmad =
            WyTileGemmDirect<ArchTag, Element, ElementC64Copy>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using ElementC64DualRightMmad =
            WyTileGemmSharedLeftDualRightDirect<
                ArchTag, Element, ElementC64Copy>;
#else
        using ElementC64DualRightMmad = ElementC64Mmad;
#endif
        using Fp32AT64x128Mmad =
            WyTileGemmDirect<ArchTag, float, Fp32AT64x128Copy>;
        using ElementSquareRightTransposeMmad =
            WyTileGemmDirect<ArchTag, Element,
                             ElementSquareRightTransposeCopy>;
        using Fp32SquareLeftTransposeMmad =
            WyTileGemmDirect<ArchTag, float,
                             Fp32SquareLeftTransposeCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        ProcessFused<
            Fp32C64Mmad, ElementC64Mmad, ElementC64DualRightMmad,
            Fp32AT64x128Mmad,
            ElementSquareRightTransposeMmad,
            Fp32SquareLeftTransposeMmad,
            RowMajor, ColumnMajor>(resource);
    }

    template <typename Fp32C64Mmad, typename Bf16C64Mmad,
              typename Bf16C64DualRightMmad,
              typename Fp32AT64x128Mmad,
              typename Bf16SquareRightTransposeMmad,
              typename Fp32SquareLeftTransposeMmad,
              typename RowMajor, typename ColumnMajor>
    __aicore__ inline void ProcessFused(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource)
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource)
#endif
    {
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(true);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr uint32_t kS0ReadyBegin = 0;
        constexpr uint32_t kS1ReadyBegin = 2;
        constexpr uint32_t kS2ReadyBegin = 4;
        constexpr uint32_t kS3aReadyBegin = 6;
        constexpr uint32_t kStageConsumedBegin = 8;
        constexpr uint32_t kZbReadyBegin = 10;
        constexpr uint32_t kTaskDoneBegin = 12;
#else
        constexpr uint32_t kS0Ready = 0;
        constexpr uint32_t kS1Ready = 1;
        constexpr uint32_t kS2Ready = 4;
        constexpr uint32_t kS3aReady = 5;
        constexpr uint32_t kS0Consumed = 2;
        constexpr uint32_t kZbReady = 3;
        constexpr uint32_t kS1Consumed = 6;
        constexpr uint32_t kTaskDone = 7;
#endif

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = static_cast<uint32_t>(tiling_.usedCoreNum);
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kWyFusedHeadsPerWindow - 1U) /
            kWyFusedHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
        uint32_t localGeneration = 0;

        // Every direct MMAD call returns its L1/L0 event credits to the
        // reusable state before it returns.  Initialize the complete event
        // set once for the WY phase instead of draining and recreating it at
        // every formula boundary of every owner.  The final drain still
        // protects the following Intra phase, which reuses the same local
        // storage and event ids with a different layout.
        BeginSharedLeftMmadPhase();
        for (uint64_t taskGroupIdx = coreIdx;
             taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum, ++localGeneration) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            const uint32_t parity = localGeneration & 1U;
            const uint32_t s0Ready = kS0ReadyBegin + parity;
            const uint32_t s1Ready = kS1ReadyBegin + parity;
            const uint32_t s2Ready = kS2ReadyBegin + parity;
            const uint32_t s3aReady = kS3aReadyBegin + parity;
            const uint32_t s0Consumed = kStageConsumedBegin + parity;
            const uint32_t s1Consumed = s0Consumed;
            const uint32_t s3aConsumed = s0Consumed;
            const uint32_t zbReady = kZbReadyBegin + parity;
            const uint32_t taskDone = kTaskDoneBegin + parity;
            if (localGeneration >= 2U) {
                WaitVectorStage(taskDone);
            }
#else
            constexpr uint32_t s0Ready = kS0Ready;
            constexpr uint32_t s1Ready = kS1Ready;
            constexpr uint32_t s2Ready = kS2Ready;
            constexpr uint32_t s3aReady = kS3aReady;
            constexpr uint32_t s0Consumed = kS0Consumed;
            constexpr uint32_t s1Consumed = kS1Consumed;
            constexpr uint32_t s3aConsumed = kTaskDone;
            constexpr uint32_t zbReady = kZbReady;
            constexpr uint32_t taskDone = kTaskDone;
            if (localGeneration >= 1U) {
                WaitVectorStage(taskDone);
            }
#endif
            const uint32_t taskIdx =
                static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindow =
                static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase =
                headWindow * kWyFusedHeadsPerWindow;
            const uint32_t headCount =
                headBase + kWyFusedHeadsPerWindow <= headNum ?
                kWyFusedHeadsPerWindow : headNum - headBase;
            const WyChunkTask task = GetWyChunkTask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            // S0: dq_raw is produced by Kernel A.  Publish the dependency
            // credit immediately; AIV applies exp2(gk) and scale while AIC
            // starts the independent dk_raw contraction below.
            PublishVectorStage(s0Ready);
            // S1: v_new @ dh^T -> dk_base, followed by gate postprocess.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t tokenV = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, V_DIM);
                const uint64_t tokenK = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 128);
                const uint64_t dhOffset = WyDhOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                // dk_base is one [C,128] result.  A5 can retain the complete
                // right operand and FP32 accumulator, so issue one N=128 MMAD
                // instead of two adjacent N=64 contractions.
                RunTransposeBToOutput<Fp32C64Mmad, RowMajor, ColumnMajor>(
                    resource, vNew_, tokenV, dh_, dhOffset,
                    dk_, tokenK, validLen, 128, 128, V_DIM);
            }
            PublishVectorStage(s1Ready);

            // S2: A^T @ dv_scan -> dVb; AIV emits dv and db_base.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t tokenV = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, V_DIM);
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                    resource, a_, token64, dvScan_, tokenV,
                    slot + tiling_.dVbOffset, validLen, validLen, V_DIM,
                    64, V_DIM, V_DIM);
            }
            PublishVectorStage(s2Ready);

            // S3a: dW_raw/zV.  Keep the shared dW operand unnegated so S3b
            // and S5 can consume it immediately.  Their downstream Vector
            // epilogues fold in the mathematical minus sign, eliminating an
            // AIV GM read/write and the corresponding AIC wait.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t tokenV = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, V_DIM);
                const uint64_t hOffset = WySavedHOffset(
                    tiling_, task.batchIdx, head, task.chunkIdx);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                // A5 L0B can hold the complete Vx128 right operand and L0C
                // can hold the complete 64x128 FP32 accumulator.  Compute
                // dW in one direct tile instead of launching two adjacent
                // N=64 tiles; this also avoids reloading dvScan for the
                // second half.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if constexpr (V_DIM == 128) {
                    if (headCount == kWyFusedHeadsPerWindow) {
                        RunSharedLeftDualTransposeB<
                            Bf16C64DualRightMmad, RowMajor, ColumnMajor>(
                            resource, dvScan_, tokenV,
                            h_, hOffset, slot + tiling_.dWOffset,
                            validLen, 128, 128, V_DIM,
                            v_, tokenV, slot + tiling_.zVOffset,
                            validLen, 64);
                    } else {
                        // The one-head tail has no second owner to hide the
                        // dual-output event lifecycle.  Use the proven single
                        // output direct tiles; the H=96 fast path is unchanged.
                        RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                            resource, dvScan_, tokenV, h_, hOffset,
                            slot + tiling_.dWOffset,
                            validLen, 128, 128, V_DIM);
                        RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                            resource, dvScan_, tokenV, v_, tokenV,
                            slot + tiling_.zVOffset,
                            validLen, validLen, 64, V_DIM);
                    }
                } else {
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, dvScan_, tokenV, h_, hOffset,
                        slot + tiling_.dWOffset, validLen, 128, 128, V_DIM);
                    RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                        resource, dvScan_, tokenV, v_, tokenV,
                        slot + tiling_.zVOffset, validLen, validLen, 64, V_DIM);
                }
#else
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, tokenV, h_, hOffset,
                    slot + tiling_.dWOffset, validLen, 128, 128, V_DIM);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, dvScan_, tokenV, v_, tokenV,
                    slot + tiling_.zVOffset, validLen, validLen, 64, V_DIM);
#endif
            }
            PublishVectorStage(s3aReady);

            // S3b produces zW_raw = dW_raw @ kE^T; AIV forms
            // Zb = tril(zV - zW_raw) * beta.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunTransposeB<Bf16C64Mmad, RowMajor, ColumnMajor>(
                    resource, workspace_,
                    (slot + tiling_.dWOffset) / sizeof(DataT),
                    workspace_,
                    (slot + tiling_.kEOffset) / sizeof(DataT),
                    slot + tiling_.zWOffset, validLen, validLen, 64, 128);
            }
            // S0 is reused for zW only after both AIVs consumed the initial
            // dq-ready generation.
            WaitVectorStage(s0Consumed);
            PublishVectorStage(s0Ready);

            // S5 produces dKgb_raw = A^T @ dW_raw.  The gradient/gate Vector
            // stage consumes it with a negative sign, avoiding a materialized
            // negated dW while preserving the original formulas.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32AT64x128Mmad, ColumnMajor, RowMajor>(
                    resource, a_, token64, workspace_,
                    (slot + tiling_.dWOffset) / sizeof(DataT),
                    slot + tiling_.dKgbOffset, validLen, validLen, 128,
                    64, 128, 128);
            }
            // Likewise, do not overwrite the first dk-base notification with
            // the dKgb notification until AIV has consumed it.
            WaitVectorStage(s1Consumed);
            PublishVectorStage(s1Ready);
            // S6 consumes Zb, while the S5 AIV gradient path is independent.
            WaitVectorStage(zbReady);

            // S6: Zb @ A^T -> Tza.  Keep Tza in the current slot; S7 is an
            // AIC consumer, so an AIC->GM->AIV->GM round trip is unnecessary.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                const uint64_t zBase = WyDAkkBf16TaskBase(
                    tiling_, task.batchIdx, head, task.begin);
                RunLayouts<Bf16SquareRightTransposeMmad,
                           RowMajor, ColumnMajor>(
                    resource, dAkk_, zBase, a_, token64,
                    slot + tiling_.zaOutputOffset,
                    validLen, validLen, validLen, 64, 64, 64);
            }
            // S7: A^T @ Tza and final causal/sign postprocess for dAkk.
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint32_t head = headBase + lane;
                const uint64_t token64 = WyTokenOffset(
                    tiling_, task.batchIdx, head, task.begin, 64);
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                RunLayouts<Fp32SquareLeftTransposeMmad,
                           ColumnMajor, RowMajor>(
                    resource, a_, token64, workspace_,
                    (slot + tiling_.zaOutputOffset) / sizeof(DataT),
                    slot + tiling_.zaInputOffset,
                    validLen, validLen, validLen, 64, 64, 64);
            }
            // Do not reuse the S3a channel until both AIVs consumed its first
            // publication.  A5 tracks this per parity; A2/A3 keep the mature
            // ordered kTaskDone stream.
            WaitVectorStage(s3aConsumed);
            PublishVectorStage(s3aReady);
        }
        // Drain the last generation that still owns workspace before the
        // following Intra phase reuses local resources and event ids.
        const uint32_t outstanding =
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            localGeneration < 2U ? localGeneration : 2U;
#else
            localGeneration == 0U ? 0U : 1U;
#endif
        for (uint32_t i = 0; i < outstanding; ++i) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            const uint32_t generation = localGeneration - outstanding + i;
            WaitVectorStage(kTaskDoneBegin + (generation & 1U));
#else
            WaitVectorStage(kTaskDone);
#endif
        }
        EndSharedLeftMmadPhase();
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void PublishVectorStage(uint32_t flag)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr uint32_t kSubBlockFlagStride = 16;
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(flag);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
            flag + kSubBlockFlagStride);
#else
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flag);
#endif
    }

    __aicore__ inline void WaitVectorStage(uint32_t flag)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr uint32_t kSubBlockFlagStride = 16;
        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(flag);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
            flag + kSubBlockFlagStride);
#else
        AscendC::CrossCoreWaitFlag(flag);
#endif
    }

    __aicore__ inline void BeginFusedMmadPhase()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1A);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1B);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0B);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(
                WyTileGemmDirectEvent::kL0C);
        }
    }

    __aicore__ inline void EndFusedMmadPhase()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1A);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                WyTileGemmDirectEvent::kL1B);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0A);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                WyTileGemmDirectEvent::kL0B);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(
                WyTileGemmDirectEvent::kL0C);
        }
    }

    __aicore__ inline void BeginSharedLeftMmadPhase()
    {
        BeginFusedMmadPhase();
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
        }
    }

    __aicore__ inline void EndSharedLeftMmadPhase()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        }
        EndFusedMmadPhase();
    }

    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunTransposeB(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        uint64_t cByteOffset, uint32_t m, uint32_t n,
        uint32_t physicalN, uint32_t k)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b;
        a.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(
            reinterpret_cast<__gm__ CType *>(workspace_) +
            cByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, RowMajor>(64, k),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, RowMajor>(64, physicalN),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(
            ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(
            tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(
            tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunSharedLeftDualTransposeB(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset,
        GM_ADDR b0Addr, uint64_t b0Offset, uint64_t c0ByteOffset,
        uint32_t m, uint32_t n0, uint32_t physicalN0, uint32_t k,
        GM_ADDR b1Addr, uint64_t b1Offset, uint64_t c1ByteOffset,
        uint32_t n1, uint32_t physicalN1)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b0;
        AscendC::GlobalTensor<DataT> b1;
        a.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b0.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(b0Addr) + b0Offset);
        b1.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataT *>(b1Addr) + b1Offset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c0;
        AscendC::GlobalTensor<CType> c1;
        c0.SetGlobalBuffer(
            reinterpret_cast<__gm__ CType *>(workspace_) +
            c0ByteOffset / sizeof(CType));
        c1.SetGlobalBuffer(
            reinterpret_cast<__gm__ CType *>(workspace_) +
            c1ByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, RowMajor>(64, k),
            Catlass::Arch::PositionGM{});
        auto tb0 = tla::MakeTensor(
            b0, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN0),
            Catlass::Arch::PositionGM{});
        auto tb1 = tla::MakeTensor(
            b1, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN1),
            Catlass::Arch::PositionGM{});
        auto tc0 = tla::MakeTensor(
            c0, tla::MakeLayout<CType, RowMajor>(64, physicalN0),
            Catlass::Arch::PositionGM{});
        auto tc1 = tla::MakeTensor(
            c1, tla::MakeLayout<CType, RowMajor>(64, physicalN1),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(
            ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB0 = GetTile(
            tb0, tla::MakeCoord(0, 0), tla::MakeShape(k, n0));
        auto tileB1 = GetTile(
            tb1, tla::MakeCoord(0, 0), tla::MakeShape(k, n1));
        auto tileC0 = GetTile(
            tc0, tla::MakeCoord(0, 0), tla::MakeShape(m, n0));
        auto tileC1 = GetTile(
            tc1, tla::MakeCoord(0, 0), tla::MakeShape(m, n1));
        Catlass::GemmCoord shape0{m, n0, k};
        Catlass::GemmCoord shape1{m, n1, k};
        Mmad mm(resource);
        mm(tileA, tileB0, tileC0, shape0,
           tileB1, tileC1, shape1);
    }


    template <typename Mmad, typename LayoutA, typename LayoutB>
    __aicore__ inline void RunLayouts(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        uint64_t cByteOffset, uint32_t m, uint32_t k, uint32_t n,
        uint32_t aPhysicalCols, uint32_t bPhysicalCols,
        uint32_t cPhysicalCols)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(workspace_) + cByteOffset / sizeof(CType));
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, LayoutA>(64, aPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<DataT, LayoutB>(64, bPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, Catlass::layout::RowMajor>(64, cPhysicalCols),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    template <typename Mmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunTransposeBToOutput(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> &resource,
#else
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
#endif
        GM_ADDR aAddr, uint64_t aOffset, GM_ADDR bAddr, uint64_t bOffset,
        GM_ADDR cAddr, uint64_t cOffset, uint32_t m, uint32_t n,
        uint32_t physicalN, uint32_t k)
    {
        AscendC::GlobalTensor<DataT> a;
        AscendC::GlobalTensor<DataT> b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(aAddr) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(bAddr) + bOffset);
        using CType = typename Mmad::ElementC;
        AscendC::GlobalTensor<CType> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(cAddr) + cOffset);
        auto ta = tla::MakeTensor(
            a, tla::MakeLayout<DataT, RowMajor>(64, k),
            Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(
            b, tla::MakeLayout<DataT, ColumnMajor>(k, physicalN),
            Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            c, tla::MakeLayout<CType, RowMajor>(64, physicalN),
            Catlass::Arch::PositionGM{});
        auto tileA = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        Catlass::GemmCoord actualShape{m, n, k};
        Mmad mm(resource);
        mm(tileA, tileB, tileC, actualShape);
    }

    GM_ADDR v_;
    GM_ADDR vNew_;
    GM_ADDR a_;
    GM_ADDR h_;
    GM_ADDR dh_;
    GM_ADDR dvScan_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR dg_;
    GM_ADDR dAkk_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR workspace_;
    ChunkKdaBwdCTilingData tiling_{};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_CUBE_H


#ifndef CHUNK_KDA_BWD_C_VECTOR_H
#define CHUNK_KDA_BWD_C_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_utils/vector/regbase.hpp"
#endif

namespace KDA {

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
static __simd_vf__ inline void KdaBwdCRowDotAccA5(
    __ubuf__ float *dst, __ubuf__ float *lhs, __ubuf__ float *rhs,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> acc;
        Duplicate(acc, 0.0f, fullMask);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            RegTensor<float> lhsReg;
            RegTensor<float> rhsReg;
            RegTensor<float> product;
            DataCopy(lhsReg, lhs + row * cols + col);
            DataCopy(rhsReg, rhs + row * cols + col);
            Mul(product, lhsReg, rhsReg, fullMask);
            Add(acc, acc, product, fullMask);
        }
        RegTensor<float> sum;
        RegTensor<float> current;
        ReduceSum(sum, acc, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(current, dst + row);
        Add(sum, sum, current, fullMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dst + row, sum, fullMask);
    }
}

static __simd_vf__ inline void KdaBwdCDvDbA5(
    __ubuf__ float *dvDst, __ubuf__ float *rowAcc,
    __ubuf__ float *dvb, __ubuf__ float *v, __ubuf__ float *beta,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> acc;
        RegTensor<float> betaReg;
        Duplicate(acc, 0.0f, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            const uint32_t offset = row * cols + col;
            RegTensor<float> dvbReg;
            RegTensor<float> vReg;
            RegTensor<float> product;
            DataCopy(dvbReg, dvb + offset);
            DataCopy(vReg, v + offset);
            Mul(product, dvbReg, vReg, fullMask);
            Add(acc, acc, product, fullMask);
            Mul(product, dvbReg, betaReg, fullMask);
            DataCopy(dvDst + offset, product, fullMask);
        }
        RegTensor<float> sum;
        RegTensor<float> current;
        ReduceSum(sum, acc, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(current, rowAcc + row);
        Add(sum, sum, current, fullMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            rowAcc + row, sum, fullMask);
    }
}

static __simd_vf__ inline void KdaBwdCMulRowDotSubA5(
    __ubuf__ float *productDst, __ubuf__ float *rowAcc,
    __ubuf__ float *lhs, __ubuf__ float *rhs,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> acc;
        Duplicate(acc, 0.0f, fullMask);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            const uint32_t offset = row * cols + col;
            RegTensor<float> lhsReg;
            RegTensor<float> rhsReg;
            RegTensor<float> product;
            DataCopy(lhsReg, lhs + offset);
            DataCopy(rhsReg, rhs + offset);
            Mul(product, lhsReg, rhsReg, fullMask);
            Add(acc, acc, product, fullMask);
            DataCopy(productDst + offset, product, fullMask);
        }
        RegTensor<float> sum;
        RegTensor<float> current;
        ReduceSum(sum, acc, fullMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(current, rowAcc + row);
        Sub(current, current, sum, fullMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            rowAcc + row, current, fullMask);
    }
}

static __simd_vf__ inline void KdaBwdCFinishDkDgA5(
    __ubuf__ float *dkDst, __ubuf__ float *dgDst,
    __ubuf__ float *dkg, __ubuf__ float *expG,
    __ubuf__ float *beta, __ubuf__ float *dkState,
    __ubuf__ float *q, __ubuf__ float *dq,
    __ubuf__ float *k, __ubuf__ float *gateW,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> betaReg;
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        for (uint32_t col = 0; col < cols; col += kRegElements) {
            const uint32_t offset = row * cols + col;
            RegTensor<float> dkgReg;
            RegTensor<float> expReg;
            RegTensor<float> stateReg;
            RegTensor<float> qReg;
            RegTensor<float> dqReg;
            RegTensor<float> kReg;
            RegTensor<float> gateWReg;
            RegTensor<float> tmp0;
            RegTensor<float> tmp1;
            DataCopy(dkgReg, dkg + offset);
            DataCopy(expReg, expG + offset);
            DataCopy(stateReg, dkState + offset);
            DataCopy(qReg, q + offset);
            DataCopy(dqReg, dq + offset);
            DataCopy(kReg, k + offset);
            DataCopy(gateWReg, gateW + offset);

            Mul(tmp0, dkgReg, expReg, fullMask);
            Mul(tmp0, tmp0, betaReg, fullMask);
            Sub(tmp0, stateReg, tmp0, fullMask);
            DataCopy(dkDst + offset, tmp0, fullMask);

            Mul(tmp0, qReg, dqReg, fullMask);
            Mul(tmp1, kReg, stateReg, fullMask);
            Sub(tmp0, tmp0, tmp1, fullMask);
            Mul(tmp1, gateWReg, betaReg, fullMask);
            Sub(tmp0, tmp0, tmp1, fullMask);
            DataCopy(dgDst + offset, tmp0, fullMask);
        }
    }
}

static __simd_vf__ inline void KdaBwdCFinishDAA5(
    __ubuf__ float *dst, __ubuf__ float *src,
    uint16_t rows, uint16_t rowStart, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> value;
    RegTensor<float> result;
    RegTensor<float> zero;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, 0.0f, fullMask);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t validCols = rowStart + row;
        uint32_t activeCount = validCols;
        MaskReg lowerMask = UpdateMask<float>(activeCount);
        LoadAlign(value, src + row * cols);
        Muls(value, value, -1.0f, fullMask);
        Select(result, value, zero, lowerMask);
        StoreAlign(dst + row * cols, result, fullMask);
    }
}

static __simd_vf__ inline void KdaBwdCBuildZbA5(
    __ubuf__ float *dst, __ubuf__ float *zv, __ubuf__ float *zw,
    __ubuf__ float *beta, uint16_t rows, uint16_t rowStart, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> zvReg;
    RegTensor<float> zwReg;
    RegTensor<float> betaReg;
    RegTensor<float> value;
    RegTensor<float> result;
    RegTensor<float> zero;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, 0.0f, fullMask);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t validCols = rowStart + row;
        uint32_t activeCount = validCols;
        MaskReg lowerMask = UpdateMask<float>(activeCount);
        LoadAlign(zvReg, zv + row * cols);
        LoadAlign(zwReg, zw + row * cols);
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
        Sub(value, zvReg, zwReg, fullMask);
        Mul(value, value, betaReg, fullMask);
        Select(result, value, zero, lowerMask);
        StoreAlign(dst + row * cols, result, fullMask);
    }
}

// kE and dq share exp2(gk).  Keep that value in a vector register and form
// both consumers before advancing to the next 64-element slice, avoiding the
// five full-tile UB passes and intervening PIPE_V barriers of the generic
// elementwise sequence.
static __simd_vf__ inline void KdaBwdCBuildKEDqA5(
    __ubuf__ float *keDst, __ubuf__ float *dqDst,
    __ubuf__ float *k, __ubuf__ float *gk, __ubuf__ float *dqRaw,
    float ln2, float scale, uint16_t elements)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t kRegElements =
        AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    const uint16_t loops = static_cast<uint16_t>(elements / kRegElements);
    for (uint16_t loop = 0; loop < loops; ++loop) {
        const uint32_t offset = loop * kRegElements;
        RegTensor<float> kReg;
        RegTensor<float> gkReg;
        RegTensor<float> dqReg;
        RegTensor<float> expReg;
        RegTensor<float> outReg;
        LoadAlign(kReg, k + offset);
        LoadAlign(gkReg, gk + offset);
        LoadAlign(dqReg, dqRaw + offset);
        Muls(expReg, gkReg, ln2, fullMask);
        Exp(expReg, expReg, fullMask);
        Mul(outReg, kReg, expReg, fullMask);
        StoreAlign(keDst + offset, outReg, fullMask);
        Mul(outReg, dqReg, expReg, fullMask);
        Muls(outReg, outReg, scale, fullMask);
        StoreAlign(dqDst + offset, outReg, fullMask);
    }
}
#endif

template <typename DataT, uint32_t V_DIM, typename BetaT>
class ChunkKdaBwdCVectorProcess {
public:
    __aicore__ ChunkKdaBwdCVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
        GM_ADDR h, GM_ADDR dh, GM_ADDR dqRaw,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
        GM_ADDR db, GM_ADDR dg, GM_ADDR dAkk,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace)
        : q_(q), k_(k), v_(v), gk_(gk), beta_(beta), h_(h), dh_(dh),
          dqRaw_(dqRaw), dq_(dq), dk_(dk), dv_(dv),
          db_(db), dg_(dg), dAkk_(dAkk), cuSeqlens_(cuSeqlens),
          chunkIndices_(chunkIndices),
          workspace_(workspace) {}

    __aicore__ inline void Init(
        const ChunkKdaBwdCTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(q_));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(k_));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(v_));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk_));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta_));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(h_));
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(dh_));
        dqRawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw_));
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq_));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk_));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(dv_));
        dbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db_));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg_));
        dAkkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAkk_));
        dAkkBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(dAkk_));
        wsFp32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_));
        wsBf16_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT *>(workspace_));

        // Two entries let MTE2/MTE3 alternate buffers while Vector consumes
        // the preceding tile.  RowReduce uses the unused tail of its output
        // plane, so the full 96 KiB UB budget remains available to IO ping-pong.
        // A5 keeps one generic queue entry and spends the released 16 KiB on
        // two explicit MTE2 ping/pong tiles.  Hot independent input pairs can
        // then issue both GM reads before Vector consumes the first tile,
        // matching the proven PR190/PR291 overlap pattern.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        pipe_->InitBuffer(inputQueue_, 1, kIoBytes);
#else
        pipe_->InitBuffer(inputQueue_, 2, kIoBytes);
#endif
        pipe_->InitBuffer(outputQueue_, 2, kIoBytes);
        pipe_->InitBuffer(arena_, kArenaBytes);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        pipe_->InitBuffer(matrixInputPing_, kIoBytes);
        pipe_->InitBuffer(matrixInputPong_, kIoBytes);
        InitMatrixInputEvents();
        // Keep the two heads' owned dk_state rows in A5's larger UB until
        // FinishGradientRows forms final dk.  This removes one full FP32
        // write/read round trip through GM for every owner.
        pipe_->InitBuffer(dkStateBuffer_, kDkStateBytes);
#endif
    }

    __aicore__ inline void Process()
    {
        ProcessFused();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ReleaseMatrixInputEvents();
#endif
    }


private:
    __aicore__ inline void SignalVectorDependency(uint32_t flag)
    {
        // Both AIV sub-blocks own disjoint rows.  Release AIC only after all
        // MTE3 writes required by the actual dependency are globally visible.
        // Match the mature Kernel-A/PR190 protocol: both AIV sub-blocks
        // participate in the collective notification consumed by one AIC.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);
    }

    __aicore__ inline void WaitCubeStage(uint32_t flag)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(flag);
#else
        AscendC::CrossCoreWaitFlag(flag);
#endif
    }

    __aicore__ inline void SignalCubeStage(uint32_t flag)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // Each AIV publishes its own 0x4 credit.  The paired AIC waits for
        // both physical sub-block flags before advancing the parity slot.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(flag);
#else
        SignalVectorDependency(flag);
#endif
    }

    __aicore__ inline void ProcessFused()
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr uint32_t kS0ReadyBegin = 0;
        constexpr uint32_t kS1ReadyBegin = 2;
        constexpr uint32_t kS2ReadyBegin = 4;
        constexpr uint32_t kS3aReadyBegin = 6;
        constexpr uint32_t kStageConsumedBegin = 8;
        constexpr uint32_t kZbReadyBegin = 10;
        constexpr uint32_t kTaskDoneBegin = 12;
#else
        constexpr uint32_t kS0Ready = 0;
        constexpr uint32_t kS1Ready = 1;
        constexpr uint32_t kS2Ready = 4;
        constexpr uint32_t kS3aReady = 5;
        constexpr uint32_t kS0Consumed = 2;
        constexpr uint32_t kZbReady = 3;
        constexpr uint32_t kS1Consumed = 6;
        constexpr uint32_t kTaskDone = 7;
#endif

        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = static_cast<uint32_t>(tiling_.usedCoreNum);
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kWyFusedHeadsPerWindow - 1U) /
            kWyFusedHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;

        uint32_t localGeneration = 0;
        for (uint64_t taskGroupIdx = coreIdx;
             taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum, ++localGeneration) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            const uint32_t parity = localGeneration & 1U;
            const uint32_t s0Ready = kS0ReadyBegin + parity;
            const uint32_t s1Ready = kS1ReadyBegin + parity;
            const uint32_t s2Ready = kS2ReadyBegin + parity;
            const uint32_t s3aReady = kS3aReadyBegin + parity;
            const uint32_t s0Consumed = kStageConsumedBegin + parity;
            const uint32_t s1Consumed = s0Consumed;
            const uint32_t s3aConsumed = s0Consumed;
            const uint32_t zbReady = kZbReadyBegin + parity;
            const uint32_t taskDone = kTaskDoneBegin + parity;
#else
            constexpr uint32_t s0Ready = kS0Ready;
            constexpr uint32_t s1Ready = kS1Ready;
            constexpr uint32_t s2Ready = kS2Ready;
            constexpr uint32_t s3aReady = kS3aReady;
            constexpr uint32_t s0Consumed = kS0Consumed;
            constexpr uint32_t s1Consumed = kS1Consumed;
            constexpr uint32_t s3aConsumed = kTaskDone;
            constexpr uint32_t zbReady = kZbReady;
            constexpr uint32_t taskDone = kTaskDone;
#endif
            const uint32_t taskIdx =
                static_cast<uint32_t>(taskGroupIdx / headWindowCount);
            const uint32_t headWindow =
                static_cast<uint32_t>(taskGroupIdx % headWindowCount);
            const uint32_t headBase =
                headWindow * kWyFusedHeadsPerWindow;
            const uint32_t headCount =
                headBase + kWyFusedHeadsPerWindow <= headNum ?
                kWyFusedHeadsPerWindow : headNum - headBase;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // A5 has two AIV sub-blocks and the steady-state window owns two
            // heads.  Give one complete head to each AIV instead of making
            // both AIVs split every head and then serially visit the second
            // head.  Each AIV still processes 64 rows in total, so the UB
            // footprint is unchanged.  A one-head tail keeps the proven
            // half-row split so both AIVs remain useful.
            const bool headParallel =
                headCount == kWyFusedHeadsPerWindow &&
                subBlockNum == kWyFusedHeadsPerWindow;
            // A split single-head tail is intermittently unsafe for FP16 on
            // A5 because both AIVs update the same logical owner through
            // independent half-row state paths.  Let AIV0 own the complete
            // tail head; AIV1 still participates in every collective event.
            const bool singleHeadOwner =
                headCount == 1U && subBlockNum == 2U;
#else
            const bool headParallel = false;
            const bool singleHeadOwner = false;
#endif
            const bool wholeHeadRows = headParallel || singleHeadOwner;
            const uint32_t rowSubBlockIdx =
                wholeHeadRows ? 0U : subBlockIdx;
            const uint32_t rowSubBlockNum =
                wholeHeadRows ? 1U : subBlockNum;
            const uint32_t rowLaneBegin =
                headParallel ? subBlockIdx :
                (singleHeadOwner && subBlockIdx != 0U ? headCount : 0U);
            const uint32_t rowLaneEnd =
                headParallel ? subBlockIdx + 1U : headCount;
            const WyChunkTask task = GetWyChunkTask(
                cuSeqlens_, chunkIndices_, tiling_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            // kE has no AIC dependency.  Build it while AIC produces the
            // independent base GEMMs instead of serializing it behind S2.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // Dense A5 completes dq_base in Intra Post, where the existing
            // exp2(gk-gk_last) tile can be reused.  Build kE while AIC is
            // still producing S0, then consume the S0 generation before its
            // flag is reused.  Varlen keeps the mature standalone path.
            if (tiling_.isVarLen != 0) {
                WaitCubeStage(s0Ready);
            }
#endif
            for (uint32_t lane = rowLaneBegin; lane < rowLaneEnd; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if (tiling_.isVarLen == 0) {
                    // The generic 16-row Intra post path consumes dq_base
                    // from GM.  The retired dense row32 post used to form it
                    // locally, so dense WY skipped the write.  Complete dq
                    // here before Intra reads it.
                    BuildKE<true>(task, headBase + lane, validLen, slot,
                                  rowSubBlockIdx, rowSubBlockNum);
                } else {
                    BuildKE<true>(task, headBase + lane, validLen, slot,
                                  rowSubBlockIdx, rowSubBlockNum);
                }
#else
                BuildKE<false>(task, headBase + lane, validLen, slot,
                               rowSubBlockIdx, rowSubBlockNum);
#endif
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (tiling_.isVarLen == 0) {
                WaitCubeStage(s0Ready);
            }
#endif
            SignalCubeStage(s0Consumed);

            // S0: consume dq_raw and finish dq_base.
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            WaitCubeStage(s0Ready);
            for (uint32_t lane = 0; lane < headCount; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                subBlockIdx, subBlockNum, lane, 0);
            }
#endif
            // S1: consume dk_raw and finish dk_state.
            WaitCubeStage(s1Ready);
            for (uint32_t lane = rowLaneBegin; lane < rowLaneEnd; ++lane) {
                const uint32_t stateLane = headParallel ? 0U : lane;
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                rowSubBlockIdx, rowSubBlockNum, stateLane, 1);
            }
            SignalCubeStage(s1Consumed);
            // S2 is ready, but dv_scan is also the final dv storage in the
            // fused contract.  Do not overwrite it until S3a has completed
            // every final read of the original dv_scan values.
            WaitCubeStage(s2Ready);
            WaitCubeStage(s3aReady);
            SignalCubeStage(s3aConsumed);
            for (uint32_t lane = rowLaneBegin; lane < rowLaneEnd; ++lane) {
                const uint32_t stateLane = headParallel ? 0U : lane;
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishBaseStage(task, headBase + lane, validLen, slot,
                                rowSubBlockIdx, rowSubBlockNum, stateLane, 2);
            }

            // The S1 state partials are now complete on both AIV sub-blocks.
            // Build the expensive h*dh state-gate term while AIC executes the
            // dependent zW GEMMs, then keep the 128-element result in the
            // otherwise-unused fused dk_raw workspace until S5 writes dg.
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            for (uint32_t lane = subBlockIdx;
                 lane < headCount; lane += subBlockNum) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                PrepareStateGate(
                    task, headBase + lane, validLen, slot, rowSubBlockNum);
            }

            // S3b: consume zW and form the saved BF16 Zb tile.
            WaitCubeStage(s0Ready);
            for (uint32_t lane = rowLaneBegin; lane < rowLaneEnd; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                BuildZbStage(task, headBase + lane, validLen, slot,
                             rowSubBlockIdx, rowSubBlockNum);
            }
            SignalCubeStage(zbReady);

            // Gradient rows remain evenly split across both AIV sub-blocks.
            // Only the lightweight final state add remains on the S5 path;
            // its h*dh reduction was overlapped with AIC above.
            WaitCubeStage(s1Ready);
            for (uint32_t lane = rowLaneBegin; lane < rowLaneEnd; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                uint32_t begin = 0;
                uint32_t end = 0;
                NormalRows(validLen, rowSubBlockIdx, rowSubBlockNum, begin, end);
                const uint32_t stateLane = headParallel ? 0U : lane;
                FinishGradientRows(task, headBase + lane, validLen,
                                   slot, begin, end, stateLane);
            }
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            for (uint32_t lane = subBlockIdx;
                 lane < headCount; lane += subBlockNum) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                AddPreparedStateGate(
                    task, headBase + lane, validLen, slot);
            }
            // S7: final dAkk mask/sign/writeback.
            WaitCubeStage(s3aReady);
            for (uint32_t lane = rowLaneBegin; lane < rowLaneEnd; ++lane) {
                const uint64_t slot = WyWorkspaceSlotBase(
                    tiling_, coreIdx, localGeneration, lane);
                FinishDA(task, headBase + lane, validLen, slot,
                         rowSubBlockIdx, rowSubBlockNum);
            }
            SignalCubeStage(taskDone);

        }
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // Ascend950's larger per-AIV UB allows a 32-row WY tile.  This halves
    // row-loop, queue and event overhead on the scalar-heavy A5 path while
    // retaining the proven 16-row footprint on A2/A3.
    static constexpr uint32_t kRows = 32;
    static constexpr uint32_t kUbBudgetBytes = 248 * 1024;
    static constexpr uint32_t kDkStateRowsPerHead = 32;
    static constexpr uint32_t kDkStateBytes =
        kWyFusedHeadsPerWindow * kDkStateRowsPerHead * kWyKeyDim *
        sizeof(float);
#else
    static constexpr uint32_t kRows = 16;
    static constexpr uint32_t kUbBudgetBytes = 96 * 1024;
#endif
    static constexpr uint32_t kPlaneElements = kRows * kWyKeyDim;
    static constexpr uint32_t kIoBytes = kPlaneElements * sizeof(float);
    // FinishGradients has the largest live set and uses Plane(0)..Plane(7).
    // Reserving 24 planes exceeded the conservative per-AIV UB budget of the
    // 1AIC:2AIV MIX kernel without providing any reusable live storage.
    static constexpr uint32_t kArenaBytes = 8 * kPlaneElements * sizeof(float);
    static constexpr float kLn2 = 0.69314718055994530942f;
    static_assert(
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                      5 * kIoBytes + kArenaBytes
#else
                      4 * kIoBytes + kArenaBytes
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                      + kDkStateBytes
#endif
                      <= kUbBudgetBytes,
                   "ChunkKdaBwdC AIV buffers exceed the architecture UB budget");

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline AscendC::LocalTensor<float> DkStateTile(
        uint32_t storageLane, uint32_t ownedRow)
    {
        const uint32_t offset =
            (storageLane * kDkStateRowsPerHead + ownedRow) * kWyKeyDim;
        return dkStateBuffer_.Get<float>()[offset];
    }

#endif

    __aicore__ inline AscendC::LocalTensor<float> Plane(uint32_t idx)
    {
        return arena_.Get<float>()[idx * kPlaneElements];
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void InitMatrixInputEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            matrixMte2ToVEvent_[slot] = static_cast<event_t>(
                pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
            matrixVToMte2Event_[slot] = static_cast<event_t>(
                pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
        }
    }

    __aicore__ inline void ReleaseMatrixInputEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(
                matrixMte2ToVEvent_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(
                matrixVToMte2Event_[slot]);
        }
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> MatrixInput(uint32_t slot)
    {
        return slot == 0 ? matrixInputPing_.Get<T>() :
                           matrixInputPong_.Get<T>();
    }

    template <typename T>
    __aicore__ inline uint32_t CopyInMatrixRows(
        AscendC::GlobalTensor<T> src, uint32_t rows, uint32_t cols,
        uint32_t physicalCols)
    {
        const uint32_t slot = currentMatrixInputSlot_;
        currentMatrixInputSlot_ ^= 1U;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((physicalCols - cols) * sizeof(T)),
            0, 0};
        AscendC::DataCopyPad(
            MatrixInput<T>(slot), src, params, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
        return slot;
    }

    __aicore__ inline void ConsumeMatrixRows(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src, uint32_t slot, uint32_t count)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
        AscendC::Adds(dst, src, 0.0f, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
    }

    template <typename SrcT>
    __aicore__ inline void ConsumeMatrixRows(
        AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<SrcT> src, uint32_t slot, uint32_t count)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            matrixMte2ToVEvent_[slot]);
        AscendC::Cast(
            dst, src, AscendC::RoundMode::CAST_NONE, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            matrixVToMte2Event_[slot]);
    }

    template <typename T0, typename T1>
    __aicore__ inline void LoadRowsPair(
        AscendC::LocalTensor<float> dst0, AscendC::GlobalTensor<T0> src0,
        uint32_t rows0, uint32_t cols0, uint32_t physicalCols0,
        AscendC::LocalTensor<float> dst1, AscendC::GlobalTensor<T1> src1,
        uint32_t rows1, uint32_t cols1, uint32_t physicalCols1)
    {
        const uint32_t slot0 =
            CopyInMatrixRows(src0, rows0, cols0, physicalCols0);
        const uint32_t slot1 =
            CopyInMatrixRows(src1, rows1, cols1, physicalCols1);
        ConsumeMatrixRows(
            dst0, MatrixInput<T0>(slot0), slot0, rows0 * cols0);
        ConsumeMatrixRows(
            dst1, MatrixInput<T1>(slot1), slot1, rows1 * cols1);
        AscendC::PipeBarrier<PIPE_V>();
    }
#endif

    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
        uint32_t count)
    {
        auto in = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyPad(
            in, src, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
            {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<float>();
        AscendC::Adds(dst, ready, 0.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    template <typename SrcT>
    __aicore__ inline void Load(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<SrcT> src,
        uint32_t count)
    {
        auto in = inputQueue_.AllocTensor<SrcT>();
        AscendC::DataCopyPad(
            in, src,
            {1, static_cast<uint32_t>(count * sizeof(SrcT)), 0, 0, 0},
            {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<SrcT>();
        AscendC::Cast(dst, ready, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Store(
        AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t count)
    {
        auto out = outputQueue_.AllocTensor<float>();
        AscendC::Adds(out, src, 0.0f, count);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<float>();
        AscendC::DataCopyPad(
            dst, ready, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<float> src, uint32_t rows,
        uint32_t cols, uint32_t physicalCols)
    {
        auto in = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(float)),
            static_cast<uint32_t>((physicalCols - cols) * sizeof(float)),
            0, 0};
        AscendC::DataCopyPad(in, src, params, {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<float>();
        AscendC::Adds(dst, ready, 0.0f, rows * cols);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void LoadRows(
        AscendC::LocalTensor<float> dst,
        AscendC::GlobalTensor<DataT> src, uint32_t rows,
        uint32_t cols, uint32_t physicalCols)
    {
        auto in = inputQueue_.AllocTensor<DataT>();
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(DataT)),
            static_cast<uint32_t>((physicalCols - cols) * sizeof(DataT)),
            0, 0};
        AscendC::DataCopyPad(in, src, params, {false, 0, 0, 0});
        inputQueue_.EnQue(in);
        auto ready = inputQueue_.DeQue<DataT>();
        AscendC::Cast(
            dst, ready, AscendC::RoundMode::CAST_NONE, rows * cols);
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Store(
        AscendC::GlobalTensor<DataT> dst, AscendC::LocalTensor<float> src,
        uint32_t count)
    {
        auto out = outputQueue_.AllocTensor<DataT>();
        AscendC::Cast(out, src, AscendC::RoundMode::CAST_RINT, count);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<DataT>();
        AscendC::DataCopyPad(
            dst, ready,
            {1, static_cast<uint32_t>(count * sizeof(DataT)), 0, 0, 0});
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void StoreStrided(
        AscendC::GlobalTensor<DataT> dst,
        AscendC::LocalTensor<float> src, uint32_t rows, uint32_t cols,
        uint32_t physicalCols)
    {
        auto out = outputQueue_.AllocTensor<DataT>();
        AscendC::Cast(
            out, src, AscendC::RoundMode::CAST_RINT, rows * cols);
        outputQueue_.EnQue(out);
        auto ready = outputQueue_.DeQue<DataT>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(DataT)),
            0,
            static_cast<uint32_t>((physicalCols - cols) * sizeof(DataT)),
            0
        };
        AscendC::DataCopyPad(dst, ready, copyParams);
        outputQueue_.FreeTensor(ready);
    }

    __aicore__ inline void Exp2(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t count)
    {
        AscendC::Muls(dst, src, kLn2, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dst, dst, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadBeta(
        AscendC::LocalTensor<float> dst, uint64_t offset, uint32_t count)
    {
        Load(dst, betaGm_[offset], count);
    }

    __aicore__ inline void BroadcastRows(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> scalar,
        uint32_t rows)
    {
        AscendC::Brcb(dst, scalar, static_cast<uint8_t>((rows + 7U) / 8U), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulRowsByScalar(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        AscendC::LocalTensor<float> broadcast, uint32_t rows, uint32_t cols)
    {
        const uint8_t stride = static_cast<uint8_t>(cols * sizeof(float) / 32);
        for (uint32_t col = 0; col < cols; col += 64) {
            const uint32_t mask = cols - col < 64 ? cols - col : 64;
            AscendC::Mul(dst[col], src[col], broadcast, mask, rows,
                         {1, 1, 0, stride, stride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowReduce128(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
        uint32_t rows)
    {
        // dst contains at most kRows scalar outputs.  Its remaining plane is
        // dead during the reduction and provides aligned 8-float partial
        // slots for every row, avoiding a dedicated 512-byte TBuf.
        auto partial = dst[kRows];
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::WholeReduceSum(
                partial[row * 8], src[row * 128], 64, 2, 1, 1, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(dst, partial, 2, rows, 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalRows(
        uint32_t validLen, uint32_t subBlockIdx, uint32_t subBlockNum,
        uint32_t &begin, uint32_t &end)
    {
        begin = validLen * subBlockIdx / subBlockNum;
        end = validLen * (subBlockIdx + 1U) / subBlockNum;
    }

    template <bool FINISH_DQ>
    __aicore__ inline void BuildKE(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t tokenBaseV = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, V_DIM);
        const uint64_t ws = (slot + tiling_.kEOffset) / sizeof(DataT);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto k = Plane(0);
            auto e = Plane(1);
            auto out = Plane(2);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            LoadRowsPair(
                k, kGm_[tokenBase + row * 128], rows, 128, 128,
                e, gkGm_[tokenBase + row * 128], rows, 128, 128);
#else
            Load(k, kGm_[tokenBase + row * 128], rows * 128);
            Load(e, gkGm_[tokenBase + row * 128], rows * 128);
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (FINISH_DQ) {
                auto dqRaw = Plane(2);
                auto keOut = Plane(3);
                auto dqOut = Plane(4);
                Load(dqRaw, dqRawGm_[tokenBase + row * 128], rows * 128);
                KdaBwdCBuildKEDqA5(
                    reinterpret_cast<__ubuf__ float *>(keOut.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(dqOut.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(k.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(e.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(dqRaw.GetPhyAddr()),
                    kLn2, tiling_.scale,
                    static_cast<uint16_t>(rows * 128));
                Store(wsBf16_[ws + row * 128], keOut, rows * 128);
                Store(dqGm_[tokenBase + row * 128], dqOut, rows * 128);
                continue;
            }
#endif
            Exp2(e, e, rows * 128);
            AscendC::Mul(out, k, e, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(wsBf16_[ws + row * 128], out, rows * 128);
            if constexpr (FINISH_DQ) {
                // e still contains exp2(gk).  Finish dq before this tile is
                // reused, eliminating one gk GM load and one Exp2 pass.
                Load(k, dqRawGm_[tokenBase + row * 128], rows * 128);
                AscendC::Mul(out, k, e, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(out, out, tiling_.scale, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                Store(dqGm_[tokenBase + row * 128], out, rows * 128);
            }
        }
    }

    __aicore__ inline void FinishBaseStage(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum, uint32_t stateLane,
        uint32_t stage)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t tokenBaseV = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, V_DIM);
        const uint64_t scalarBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 1);
        const uint64_t dqRaw = (slot + tiling_.dqRawOffset) / sizeof(float);
        const uint64_t dkRaw = (slot + tiling_.dkRawOffset) / sizeof(float);
        const uint64_t dVb = (slot + tiling_.dVbOffset) / sizeof(float);
        auto statePartial = Plane(6);
        auto gkLast = Plane(7);
        if (begin == end) {
            if (stage == 1) {
                AscendC::Duplicate(statePartial, 0.0f, 128);
                AscendC::PipeBarrier<PIPE_V>();
                const uint64_t stateBase =
                    (slot + tiling_.dkRawOffset) / sizeof(float) +
                    subBlockIdx * 128U;
                Store(wsFp32_[stateBase], statePartial, 128);
            }
            return;
        }
        if (stage == 1) {
            AscendC::Duplicate(statePartial, 0.0f, 128);
            // The chunk anchor is shared by every row tile.  Keep one copy in
            // UB instead of issuing the same 512-byte GM load per tile.
            Load(gkLast, gkGm_[tokenBase + (validLen - 1U) * 128], 128);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (stage == 2) {
            // beta and db are chunk-row scalars.  Keep the complete rows owned
            // by this sub-block in otherwise-dead planes and perform one GM
            // transaction instead of one transaction per matrix tile.
            LoadBeta(Plane(6), scalarBase + begin, end - begin);
            AscendC::Duplicate(Plane(7), 0.0f, end - begin);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto x = Plane(0);
            auto y = Plane(1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            auto z = stage == 1 ? DkStateTile(stateLane, row - begin) : Plane(2);
#else
            auto z = Plane(2);
#endif
            auto aux = Plane(3);
            auto scalar = Plane(4);
            auto brcb = Plane(5);

            if (stage == 0) {
                Load(x, dqRawGm_[tokenBase + row * 128], rows * 128);
                Load(y, gkGm_[tokenBase + row * 128], rows * 128);
                Exp2(y, y, rows * 128);
                AscendC::Mul(z, x, y, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(z, z, tiling_.scale, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                Store(dqGm_[tokenBase + row * 128], z, rows * 128);
            } else if (stage == 1) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                LoadRowsPair(
                    x, dkGm_[tokenBase + row * 128], rows, 128, 128,
                    y, gkGm_[tokenBase + row * 128], rows, 128, 128);
#else
                Load(x, dkGm_[tokenBase + row * 128], rows * 128);
#endif
                for (uint32_t r = 0; r < rows; ++r) {
                    AscendC::Adds(z[r * 128], gkLast, 0.0f, 128);
                }
                AscendC::PipeBarrier<PIPE_V>();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                Load(y, gkGm_[tokenBase + row * 128], rows * 128);
#endif
                AscendC::Sub(z, z, y, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                Exp2(z, z, rows * 128);
                AscendC::Mul(z, x, z, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                Store(dkGm_[tokenBase + row * 128], z, rows * 128);
#endif

                // GPU keeps dk_state live and immediately accumulates
                // sum_t(k * dk_state).  Do the same here instead of reading
                // final dk back from GM in S5 and reconstructing dk_state.
                Load(x, kGm_[tokenBase + row * 128], rows * 128);
                AscendC::Mul(y, x, z, rows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t stride = 1; stride < rows; stride <<= 1U) {
                    for (uint32_t r = 0; r + stride < rows;
                         r += stride << 1U) {
                        AscendC::Add(
                            y[r * 128], y[r * 128],
                            y[(r + stride) * 128], 128);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(statePartial, statePartial, y, 128);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                BroadcastRows(brcb, Plane(6)[row - begin], rows);
#endif
                for (uint32_t v0 = 0; v0 < V_DIM; v0 += 128) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    LoadRowsPair(
                        x, wsFp32_[dVb + row * V_DIM + v0],
                        rows, 128, V_DIM,
                        y, vGm_[tokenBaseV + row * V_DIM + v0],
                        rows, 128, V_DIM);
#else
                    LoadRows(
                        x, wsFp32_[dVb + row * V_DIM + v0],
                        rows, 128, V_DIM);
                    LoadRows(
                        y, vGm_[tokenBaseV + row * V_DIM + v0],
                        rows, 128, V_DIM);
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    KdaBwdCDvDbA5(
                        (__ubuf__ float *)z.GetPhyAddr(),
                        (__ubuf__ float *)Plane(7)[row - begin].GetPhyAddr(),
                        (__ubuf__ float *)x.GetPhyAddr(),
                        (__ubuf__ float *)y.GetPhyAddr(),
                        (__ubuf__ float *)Plane(6)[row - begin].GetPhyAddr(),
                        static_cast<uint16_t>(rows), 128);
                    AscendC::PipeBarrier<PIPE_V>();
#else
                    AscendC::Mul(aux, x, y, rows * 128);
                    AscendC::PipeBarrier<PIPE_V>();
                    RowReduce128(z, aux, rows);
                    AscendC::Add(
                        Plane(7)[row - begin],
                        Plane(7)[row - begin], z, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    MulRowsByScalar(z, x, brcb, rows, 128);
#endif
                    StoreStrided(
                        dvGm_[tokenBaseV + row * V_DIM + v0],
                        z, rows, 128, V_DIM);
                }
            }
        }
        if (stage == 1) {
            const uint64_t stateBase =
                (slot + tiling_.dkRawOffset) / sizeof(float) +
                subBlockIdx * 128U;
            Store(wsFp32_[stateBase], statePartial, 128);
        } else if (stage == 2) {
            Store(dbGm_[scalarBase + begin], Plane(7), end - begin);
        }
    }

    __aicore__ inline void FinishGradientRows(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t begin, uint32_t end, uint32_t stateLane)
    {
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t scalarBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 1);
        const uint64_t kE = (slot + tiling_.kEOffset) / sizeof(DataT);
        const uint64_t dKgb = (slot + tiling_.dKgbOffset) / sizeof(float);
        const uint32_t ownedRows = end - begin;
        if (ownedRows == 0) {
            return;
        }
        // Plane(5)'s first 16 scalars and aligned reduction scratch occupy at
        // most 144 elements.  Its tail safely holds the sub-block's beta/db
        // vectors across all row tiles.
        auto scalarStorage = Plane(5);
        auto betaRows = scalarStorage[256];
        auto dbRows = scalarStorage[256 + 64];
        LoadBeta(betaRows, scalarBase + begin, ownedRows);
        Load(dbRows, dbGm_[scalarBase + begin], ownedRows);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto dkg = Plane(0);
            auto ke = Plane(1);
            auto e = Plane(2);
            auto tmp = Plane(3);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            auto acc = DkStateTile(stateLane, row - begin);
#else
            auto acc = Plane(4);
#endif
            auto scalar = Plane(5);
            auto brcb = Plane(6);
            auto qk = Plane(7);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            LoadRowsPair(
                dkg, wsFp32_[dKgb + row * 128], rows, 128, 128,
                ke, wsBf16_[kE + row * 128], rows, 128, 128);
#else
            Load(dkg, wsFp32_[dKgb + row * 128], rows * 128);
            Load(ke, wsBf16_[kE + row * 128], rows * 128);
#endif

            // dKgb_raw has the opposite sign of mathematical dKgb, so fold
            // the minus sign into each consumer instead of materializing a
            // second full matrix.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaBwdCMulRowDotSubA5(
                (__ubuf__ float *)tmp.GetPhyAddr(),
                (__ubuf__ float *)dbRows[row - begin].GetPhyAddr(),
                (__ubuf__ float *)dkg.GetPhyAddr(),
                (__ubuf__ float *)ke.GetPhyAddr(),
                static_cast<uint16_t>(rows), 128);
#else
            AscendC::Mul(tmp, dkg, ke, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            RowReduce128(scalar, tmp, rows);
            AscendC::Sub(
                dbRows[row - begin], dbRows[row - begin], scalar, rows);
            AscendC::PipeBarrier<PIPE_V>();
#endif

            // dk_base = dk_state - dKgb_raw * beta * exp2(gk).
            Load(e, gkGm_[tokenBase + row * 128], rows * 128);
            Exp2(e, e, rows * 128);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // Keep the complete dk/dg elementwise chain in registers.  The
            // two outputs share beta and dk_state, while tmp already holds
            // dKgb_raw*kE from the db reduction above.
            LoadRowsPair(
                qk, qGm_[tokenBase + row * 128], rows, 128, 128,
                brcb, dqGm_[tokenBase + row * 128], rows, 128, 128);
            Load(ke, kGm_[tokenBase + row * 128], rows * 128);
            KdaBwdCFinishDkDgA5(
                (__ubuf__ float *)e.GetPhyAddr(),
                (__ubuf__ float *)qk.GetPhyAddr(),
                (__ubuf__ float *)dkg.GetPhyAddr(),
                (__ubuf__ float *)e.GetPhyAddr(),
                (__ubuf__ float *)betaRows[row - begin].GetPhyAddr(),
                (__ubuf__ float *)acc.GetPhyAddr(),
                (__ubuf__ float *)qk.GetPhyAddr(),
                (__ubuf__ float *)brcb.GetPhyAddr(),
                (__ubuf__ float *)ke.GetPhyAddr(),
                (__ubuf__ float *)tmp.GetPhyAddr(),
                static_cast<uint16_t>(rows), 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dkGm_[tokenBase + row * 128], e, rows * 128);
            Store(dgGm_[tokenBase + row * 128], qk, rows * 128);
#else
            BroadcastRows(brcb, betaRows[row - begin], rows);
            // qk is dead until the gate expression below.  Use it here so
            // tmp keeps kE*dKgb resident for the gate_w contribution.
            AscendC::Mul(qk, dkg, e, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            MulRowsByScalar(qk, qk, brcb, rows, 128);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Load(acc, dkGm_[tokenBase + row * 128], rows * 128);
#endif
            AscendC::Sub(e, acc, qk, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dkGm_[tokenBase + row * 128], e, rows * 128);

            // gate_qk + gate_w algebraically; dKgb_raw carries the opposite
            // sign, so subtract its contribution.  Keep dk_state resident in acc while e is
            // used for the final-dk writeback; no subtractive reconstruction.
            Load(qk, qGm_[tokenBase + row * 128], rows * 128);
            Load(e, dqGm_[tokenBase + row * 128], rows * 128);
            AscendC::Mul(qk, qk, e, rows * 128);
            Load(e, kGm_[tokenBase + row * 128], rows * 128);
            AscendC::Mul(acc, e, acc, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(qk, qk, acc, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            // tmp still contains kE*dKgb from the db reduction above.
            MulRowsByScalar(tmp, tmp, brcb, rows, 128);
            AscendC::Sub(qk, qk, tmp, rows * 128);
            AscendC::PipeBarrier<PIPE_V>();
            Store(dgGm_[tokenBase + row * 128], qk, rows * 128);
#endif
        }
        Store(dbGm_[scalarBase + begin], dbRows, ownedRows);
    }

    __aicore__ inline void BuildZb(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t begin, uint32_t end)
    {
        const uint64_t scalarBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 1);
        const uint64_t zV = (slot + tiling_.zVOffset) / sizeof(DataT);
        const uint64_t zW = (slot + tiling_.zWOffset) / sizeof(DataT);
        const uint64_t zB = WyDAkkBf16TaskBase(
            tiling_, task.batchIdx, head, task.begin);
        auto beta = Plane(0);
        LoadBeta(beta, scalarBase, validLen);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto zv = Plane(1);
            auto zw = Plane(2);
            auto out = Plane(3);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            LoadRowsPair(
                zv, wsBf16_[zV + row * 64], rows, 64, 64,
                zw, wsBf16_[zW + row * 64], rows, 64, 64);
#else
            Load(zv, wsBf16_[zV + row * 64], rows * 64);
            Load(zw, wsBf16_[zW + row * 64], rows * 64);
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaBwdCBuildZbA5(
                reinterpret_cast<__ubuf__ float *>(out.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(zv.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(zw.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(beta[row].GetPhyAddr()),
                static_cast<uint16_t>(rows), static_cast<uint16_t>(row), 64);
#else
            // zW was formed from dW_raw; subtracting it is equivalent to
            // adding the original zW formed from -dW_raw.
            AscendC::Sub(out, zv, zw, rows * 64);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(
                out, out, beta, 64, static_cast<uint8_t>(rows),
                {1, 1, 1, 8, 8, 0});
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t logicalRow = row + r;
                if (logicalRow == 0) {
                    AscendC::Duplicate(out[r * 64], 0.0f, 64);
                } else if (logicalRow < 64) {
                    uint64_t upperMask[1] = {0xffffffffffffffffULL};
                    upperMask[0] <<= logicalRow;
                    AscendC::Duplicate(
                        out[r * 64], 0.0f, upperMask, 1, 1, 8);
                }
            }
            // Store starts with an AIV Adds into the output queue.  Preserve
            // the Duplicate -> Adds dependency explicitly, as in mature
            // vector post-processing paths.
            AscendC::PipeBarrier<PIPE_V>();
#endif
            Store(dAkkBf16Gm_[zB + row * 64], out, rows * 64);
        }
    }

    __aicore__ inline void PrepareStateGate(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockNum)
    {
        const uint64_t tokenBase = WyTokenOffset(tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t hBase = WySavedHOffset(tiling_, task.batchIdx, head, task.chunkIdx);
        const uint64_t dhBase = WyDhOffset(tiling_, task.batchIdx, head, task.chunkIdx);
        const uint32_t last = validLen - 1U;
        const uint64_t stateBase =
            (slot + tiling_.dkRawOffset) / sizeof(float);
        auto state = Plane(0);
        auto partial = Plane(1);
        auto gateAnchor = Plane(7);
        Load(state, wsFp32_[stateBase], 128);
        for (uint32_t part = 1; part < subBlockNum; ++part) {
            Load(partial, wsFp32_[stateBase + part * 128U], 128);
            AscendC::Add(state, state, partial, 128);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // gk at the final token is invariant across all eight 16-row state
        // tiles.  Load and exponentiate the complete 128-column anchor once.
        Load(gateAnchor, gkGm_[tokenBase + last * 128], 128);
        Exp2(gateAnchor, gateAnchor, 128);

        for (uint32_t col = 0; col < 128; col += kRows) {
            auto x = Plane(2);
            auto y = Plane(3);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            auto product = Plane(4);
#endif
            auto reduced = Plane(5);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            auto tileReduced = Plane(6);
#endif
            AscendC::Duplicate(reduced, 0.0f, kRows);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t v0 = 0; v0 < V_DIM; v0 += 128) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                LoadRowsPair(
                    x, hGm_[hBase + col * V_DIM + v0],
                    kRows, 128, V_DIM,
                    y, dhGm_[dhBase + col * V_DIM + v0],
                    kRows, 128, V_DIM);
#else
                LoadRows(
                    x, hGm_[hBase + col * V_DIM + v0],
                    kRows, 128, V_DIM);
                LoadRows(
                    y, dhGm_[dhBase + col * V_DIM + v0],
                    kRows, 128, V_DIM);
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                KdaBwdCRowDotAccA5(
                    (__ubuf__ float *)reduced.GetPhyAddr(),
                    (__ubuf__ float *)x.GetPhyAddr(),
                    (__ubuf__ float *)y.GetPhyAddr(),
                    static_cast<uint16_t>(kRows), 128);
#else
                AscendC::Mul(product, x, y, kRows * 128);
                AscendC::PipeBarrier<PIPE_V>();
                RowReduce128(tileReduced, product, kRows);
                AscendC::Add(reduced, reduced, tileReduced, kRows);
                AscendC::PipeBarrier<PIPE_V>();
#endif
            }
            AscendC::Mul(reduced, reduced, gateAnchor[col], kRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(state[col], state[col], reduced, kRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Store(wsFp32_[stateBase], state, 128);
    }

    __aicore__ inline void AddPreparedStateGate(
        const WyChunkTask &task, uint32_t head, uint32_t validLen,
        uint64_t slot)
    {
        const uint64_t tokenBase = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 128);
        const uint64_t stateBase =
            (slot + tiling_.dkRawOffset) / sizeof(float);
        const uint32_t last = validLen - 1U;
        auto state = Plane(0);
        auto gradient = Plane(1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        LoadRowsPair(
            state, wsFp32_[stateBase], 1, 128, 128,
            gradient, dgGm_[tokenBase + last * 128], 1, 128, 128);
#else
        Load(state, wsFp32_[stateBase], 128);
        Load(gradient, dgGm_[tokenBase + last * 128], 128);
#endif
        AscendC::Add(gradient, gradient, state, 128);
        AscendC::PipeBarrier<PIPE_V>();
        Store(dgGm_[tokenBase + last * 128], gradient, 128);
    }

    __aicore__ inline void BuildZbStage(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        BuildZb(task, head, validLen, slot, begin, end);
    }

    __aicore__ inline void FinishDA(
        const WyChunkTask &task, uint32_t head, uint32_t validLen, uint64_t slot,
        uint32_t subBlockIdx, uint32_t subBlockNum)
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        NormalRows(validLen, subBlockIdx, subBlockNum, begin, end);
        const uint64_t raw = (slot + tiling_.zaInputOffset) / sizeof(float);
        // dAkk is [B,H,T,64] without chunk padding.  Use the actual token
        // stride so a tail chunk does not shift the following head.
        const uint64_t out = WyTokenOffset(
            tiling_, task.batchIdx, head, task.begin, 64);
        for (uint32_t row = begin; row < end; row += kRows) {
            const uint32_t rows = row + kRows <= end ? kRows : end - row;
            auto value = Plane(0);
            Load(value, wsFp32_[raw + row * 64], rows * 64);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaBwdCFinishDAA5(
                reinterpret_cast<__ubuf__ float *>(value.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(value.GetPhyAddr()),
                static_cast<uint16_t>(rows), static_cast<uint16_t>(row), 64);
#else
            AscendC::Muls(value, value, -1.0f, rows * 64);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t logicalRow = row + r;
                if (logicalRow == 0) {
                    AscendC::Duplicate(value[r * 64], 0.0f, 64);
                } else {
                    uint64_t upperMask[1] = {0xffffffffffffffffULL};
                    upperMask[0] <<= logicalRow;
                    AscendC::Duplicate(
                        value[r * 64], 0.0f, upperMask, 1, 1, 8);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
#endif
            Store(dAkkGm_[out + row * 64], value, rows * 64);
        }
    }

    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR gk_;
    GM_ADDR beta_;
    GM_ADDR h_;
    GM_ADDR dh_;
    GM_ADDR dqRaw_;
    GM_ADDR dq_;
    GM_ADDR dk_;
    GM_ADDR dv_;
    GM_ADDR db_;
    GM_ADDR dg_;
    GM_ADDR dAkk_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR workspace_;
    ChunkKdaBwdCTilingData tiling_{};
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<DataT> qGm_;
    AscendC::GlobalTensor<DataT> kGm_;
    AscendC::GlobalTensor<DataT> vGm_;
    AscendC::GlobalTensor<float> gkGm_;
    AscendC::GlobalTensor<BetaT> betaGm_;
    AscendC::GlobalTensor<DataT> hGm_;
    AscendC::GlobalTensor<DataT> dhGm_;
    AscendC::GlobalTensor<float> dqRawGm_;
    AscendC::GlobalTensor<float> dqGm_;
    AscendC::GlobalTensor<float> dkGm_;
    AscendC::GlobalTensor<DataT> dvGm_;
    AscendC::GlobalTensor<float> dbGm_;
    AscendC::GlobalTensor<float> dgGm_;
    AscendC::GlobalTensor<float> dAkkGm_;
    AscendC::GlobalTensor<DataT> dAkkBf16Gm_;
    AscendC::GlobalTensor<float> wsFp32_;
    AscendC::GlobalTensor<DataT> wsBf16_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixInputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixInputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dkStateBuffer_;
    event_t matrixMte2ToVEvent_[2]{};
    event_t matrixVToMte2Event_[2]{};
    uint32_t currentMatrixInputSlot_ = 0;
#endif
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_VECTOR_H


#endif // CHUNK_KDA_BWD_FINALIZE_WY_H
