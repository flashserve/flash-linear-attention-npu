#ifndef CHUNK_KKT_CUBE_H
#define CHUNK_KKT_CUBE_H

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace NsChunkKktCube {

template <typename T>
class ChunkKktCube {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Catlass::Arch::Ascend950;
#else
    using ArchTag = Catlass::Arch::AtlasA2;
#endif
    using LayoutK = Catlass::layout::RowMajor;
    using LayoutKt = Catlass::layout::ColumnMajor;
    using LayoutScore = Catlass::layout::RowMajor;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, LayoutK, T, LayoutKt, float, LayoutScore>;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, T, LayoutTagL1A>;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr uint32_t MAX_BT = 128;
    static constexpr uint32_t K_DIM = 128;
    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<T, LayoutTagL1A>(tla::Int<MAX_BT>{}, tla::Int<K_DIM>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<T, LayoutTagL1B>(tla::Int<K_DIM>{}, tla::Int<MAX_BT>{});

public:
    template <typename TilingData>
    __aicore__ inline void Process(GM_ADDR k, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                   GM_ADDR score, const TilingData *tiling)
    {
        Catlass::Arch::Resource<ArchTag> resource;
        auto l1A = resource.l1Buf.template GetBufferByByte<T>(0);
        auto l1B = resource.l1Buf.template GetBufferByByte<T>(MAX_BT * K_DIM * sizeof(T));
        auto l0A = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto l0B = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto l0C = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;
        constexpr int32_t EVENT_L1A = 0;
        constexpr int32_t EVENT_L1B = 1;
        constexpr int32_t EVENT_L0A = 0;
        constexpr int32_t EVENT_L0B = 1;
        constexpr int32_t EVENT_L0C = 0;
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);

        const int64_t begin = static_cast<int64_t>(AscendC::GetBlockIdx()) * tiling->tilesPerCore;
        int64_t end = begin + tiling->tilesPerCore;
        end = end < tiling->taskNum ? end : static_cast<int64_t>(tiling->taskNum);
        for (int64_t task = begin; task < end; ++task) {
            int64_t inputOffset = 0;
            int64_t valid = 0;
            DecodeTask(task, cuSeqlens, chunkIndices, tiling, inputOffset, valid);
            if (valid <= 0) {
                continue;
            }
            RunTask(k, score, task, inputOffset, valid, tiling, l1A, l1B, l0A, l0B, l0C,
                    copyL1ToL0A, copyL1ToL0B, tileMmad,
                    EVENT_L1A, EVENT_L1B, EVENT_L0A, EVENT_L0B, EVENT_L0C);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_L1B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0C);
    }

private:
    template <typename TilingData>
    __aicore__ inline void DecodeTask(int64_t task, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                      const TilingData *tiling, int64_t &inputOffset, int64_t &valid)
    {
        const int64_t chunk = task % static_cast<int64_t>(tiling->NT);
        const int64_t taskHeads = static_cast<int64_t>(tiling->B * tiling->NT) > 0
                                      ? static_cast<int64_t>(tiling->taskNum / (tiling->B * tiling->NT))
                                      : static_cast<int64_t>(tiling->Hk);
        const int64_t taskHead = (task / static_cast<int64_t>(tiling->NT)) % taskHeads;
        const int64_t b = task / static_cast<int64_t>(taskHeads * tiling->NT);
        // Phase6 schedules one ABC task per value head. Map that value head
        // back to the shared logical key head; standalone KKT retains its
        // legacy Hk task axis because taskHeads == Hk there.
        const int64_t h = (taskHeads == static_cast<int64_t>(tiling->Hv) && tiling->hvPerHk > 0)
                              ? taskHead / static_cast<int64_t>(tiling->hvPerHk)
                              : taskHead;
        int64_t rowStart = chunk * static_cast<int64_t>(tiling->BT);
        valid = static_cast<int64_t>(tiling->BT);
        if (tiling->isVarlen != 0) {
            AscendC::GlobalTensor<int64_t> cu;
            AscendC::GlobalTensor<int64_t> indices;
            cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
            const int64_t sequence = indices.GetValue(chunk * 2);
            const int64_t localChunk = indices.GetValue(chunk * 2 + 1);
            const int64_t bos = cu.GetValue(sequence);
            const int64_t eos = cu.GetValue(sequence + 1);
            rowStart = bos + localChunk * static_cast<int64_t>(tiling->BT);
            valid = eos - rowStart;
            valid = valid < static_cast<int64_t>(tiling->BT) ? valid : static_cast<int64_t>(tiling->BT);
        } else {
            const int64_t remaining = static_cast<int64_t>(tiling->T) - rowStart;
            valid = valid < remaining ? valid : remaining;
        }
        inputOffset = ((b * static_cast<int64_t>(tiling->Hk) + h) * static_cast<int64_t>(tiling->T) + rowStart) *
                      static_cast<int64_t>(tiling->K);
    }

    template <typename TilingData, typename L1Tensor, typename L0Tensor, typename L0CTensor>
    __aicore__ inline void RunTask(
        GM_ADDR k, GM_ADDR score, int64_t task, int64_t inputOffset, int64_t valid,
        const TilingData *tiling, L1Tensor l1A, L1Tensor l1B, L0Tensor l0A, L0Tensor l0B, L0CTensor l0C,
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
        int32_t eventL1A, int32_t eventL1B, int32_t eventL0A, int32_t eventL0B, int32_t eventL0C)
    {
        AscendC::GlobalTensor<T> kGm;
        AscendC::GlobalTensor<float> scoreGm;
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k) + inputOffset);
        scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(score) +
                                task * static_cast<int64_t>(tiling->BT * tiling->BT));

        LayoutK tagK = LayoutK::MakeLayout<T>(valid, tiling->K);
        LayoutKt tagKt = LayoutKt::MakeLayout<T>(tiling->K, valid);
        LayoutScore tagScore = LayoutScore::MakeLayout<float>(tiling->BT, tiling->BT);
        auto tensorK = tla::MakeTensor(kGm, tla::MakeLayoutFromTag(tagK), Catlass::Arch::PositionGM{});
        auto tensorKt = tla::MakeTensor(kGm, tla::MakeLayoutFromTag(tagKt), Catlass::Arch::PositionGM{});
        auto tensorScore = tla::MakeTensor(scoreGm, tla::MakeLayoutFromTag(tagScore), Catlass::Arch::PositionGM{});
        auto blockK = tla::GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(valid, tiling->K));
        auto blockKt = tla::GetTile(tensorKt, tla::MakeCoord(0, 0), tla::MakeShape(tiling->K, valid));
        auto blockScore = tla::GetTile(tensorScore, tla::MakeCoord(0, 0), tla::MakeShape(valid, valid));
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockK)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockKt)>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(blockScore)>;
#else
        using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<decltype(blockScore)>;
#endif
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL0CToGm copyL0CToGm;
        auto tensorL1A = tla::MakeTensor(l1A, L1A_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, L1B_LAYOUT, Catlass::Arch::PositionL1{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
        copyGmToL1A(tensorL1A, blockK);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
        copyGmToL1B(tensorL1B, blockKt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);

        uint32_t mActual = static_cast<uint32_t>(valid == 1 ? 16 : valid);
        auto tensorL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<T, LayoutTagL0A>(mActual, tiling->K), Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<T, LayoutTagL0B>(tiling->K, valid), Catlass::Arch::PositionL0B{});
        auto tileL1A = tla::GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, tiling->K));
        auto tileL1B = tla::GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(tiling->K, valid));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventL0A);
        copyL1ToL0A(tensorL0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventL0B);
        copyL1ToL0B(tensorL0B, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);

        auto tensorL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(mActual, valid), Catlass::Arch::PositionL0C{});
        auto tileL0C = tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, valid));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);
        tileMmad(tileL0C, tensorL0A, tensorL0B, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(eventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(eventL0C);
        copyL0CToGm(blockScore, tensorL0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
    }
};

}  // namespace NsChunkKktCube

#endif  // CHUNK_KKT_CUBE_H
