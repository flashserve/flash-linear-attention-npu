/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef GDN_ARCH22_BLOCK_MMAD_RECOMPUTE_WU_KEEPA_HPP
#define GDN_ARCH22_BLOCK_MMAD_RECOMPUTE_WU_KEEPA_HPP

#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
namespace Catlass::Gemm::Block {

// WU 私有相邻 U/W 调用保留 A；资源、初始化与退休沿用公共父类。
template <class Base_>
class RecomputeWUKeepAMmad : public Base_ {
public:
    using Base = Base_;
    using ArchTag = typename Base::ArchTag;
    using TileCopy = typename Base::TileCopy;
    using L1TileShape = typename Base::L1TileShape;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementBias = typename Base::ElementBias;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using LayoutTagL0A = typename Base::LayoutTagL0A;
    using LayoutTagL0B = typename Base::LayoutTagL0B;
    using TileCopy_ = typename Base::TileCopy;
    using Base::L1A_STAGES;
    using Base::L1B_STAGES;
    using Base::L0A_STAGES;
    using Base::L0B_STAGES;
    using Base::L0C_STAGES;
    using Base::ENABLE_UNIT_FLAG;
    using Base::ENABLE_L1_RESIDENT;
    using Base::HAS_BIAS;
    using Base::L1_TILE_N;
    using Base::L1_TILE_K;
    using Base::L0_TILE_M;
    using Base::L0_TILE_N;
    using Base::L0_TILE_K;
    using Base::L1A_TILE_SIZE;
    using Base::L1B_TILE_SIZE;
    using Base::L1A_LAYOUT;
    using Base::L1B_LAYOUT;
    using Base::L1BIAS_LAYOUT;
    using Base::L0BIAS_LAYOUT;

    CATLASS_DEVICE
    explicit RecomputeWUKeepAMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
        : Base(resource, l1BufAddrStart) {}

    /// Perform a block-scoped matrix multiply-accumulate
    // KeepA/ReuseA pair adjacent calls on one L1 K tile without external-A configuration.
    // KeepA loads and retains the current A slot; ReuseA reads it, releases it and advances.
    template <bool KeepA = false, bool ReuseA = false,
              class TensorA, class TensorB, class TensorC, class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, GemmCoord const &actualShape,
        TensorBias const &tensorBias = {}, bool clearL1Padding = false)
    {
        static_assert(!(KeepA && ReuseA), "KeepA and ReuseA must be separate calls");
        // Check L1TileShape
        if constexpr (HAS_BIAS) {
            static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
            static constexpr uint32_t L1BIAS_SIZE = L1_TILE_N * sizeof(ElementBias);
            static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE,
                "BIAS_BUF_SIZE exceeding the BT space! Reduce L0_TILE_N");
            static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES + L1BIAS_SIZE <= ArchTag::L1_SIZE,
                "L1TileShape exceeding the L1 space!");
        }

        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 2201)
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<TensorC>;
        CopyL0CToGm copyL0CToDst;
#endif
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
#endif

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();
        bool useExternalL1AThisCall = useExternalL1ATensor && kBlockActual <= L1_TILE_K;
        if (useExternalL1ATensor && !useExternalL1AThisCall) {
            useExternalL1ATensor = false;
            skipNextADataCopy = false;
        }

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            // Avoid using the gemv mode in mmad
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);
        // load first matrix A tile from GM to L1
        if constexpr (!ReuseA) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
            if (clearL1Padding && !useExternalL1AThisCall) {
                AscendC::InitConstValueParams<ElementA> clearParams(
                    1, static_cast<uint16_t>(L1A_TILE_SIZE / 32), 0,
                    static_cast<ElementA>(0));
                AscendC::InitConstValue(l1ATensorList[l1AListId], clearParams);
            }
        }
        auto tensorL1A = tla::MakeTensor(
            useExternalL1AThisCall ? externalL1ATensor : l1ATensorList[l1AListId],
            L1A_LAYOUT, Arch::PositionL1{});
        if constexpr (!ReuseA) {
            auto tensorTileA = GetTileA(tensorA, 0, 0, mBlockActual, kL1Actual);
            if (skipNextADataCopy) {
                skipNextADataCopy = false;
            } else if constexpr (ENABLE_L1_RESIDENT) {
                // If the currently loaded GM pointer and block coordinates are the same as the last loaded ones,
                // skip this loading.
                if (lastAddrA[l1AListId] != tensorTileA.data().GetPhyAddr()
                    || tla::get<0>(tensorTileA.coord()) != lastCoordA[l1AListId].row()
                    || tla::get<1>(tensorTileA.coord()) != lastCoordA[l1AListId].column()) {
                    copyGmToL1A(tensorL1A, tensorTileA);
                    lastCoordA[l1AListId] = MatrixCoord{
                        static_cast<uint32_t>(tla::get<0>(tensorTileA.coord())),
                        static_cast<uint32_t>(tla::get<1>(tensorTileA.coord()))};
                    lastAddrA[l1AListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                        tensorTileA.data().GetPhyAddr()
                    );
                }
            } else {
                copyGmToL1A(tensorL1A, tensorTileA);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
        }

        // load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        if (clearL1Padding) {
            AscendC::InitConstValueParams<ElementB> clearParams(
                1, static_cast<uint16_t>(L1B_TILE_SIZE / 32), 0,
                static_cast<ElementB>(0));
            AscendC::InitConstValue(l1BTensorList[l1BListId], clearParams);
        }
        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(kL1Actual, nBlockActual));
        if constexpr (ENABLE_L1_RESIDENT) {
            if (lastAddrB[l1BListId] != tensorTileB.data().GetPhyAddr()
                || tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListId].row()
                || tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListId].column()) {
                copyGmToL1B(tensorL1B, tensorTileB);
                lastCoordB[l1BListId] = MatrixCoord{
                    static_cast<uint32_t>(tla::get<0>(tensorTileB.coord())),
                    static_cast<uint32_t>(tla::get<1>(tensorTileB.coord()))};
                lastAddrB[l1BListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                    tensorTileB.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1B(tensorL1B, tensorTileB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
            using CopyGmToL1Bias = typename TileCopy::template CopyGmToL1Bias<TensorBias>;
            CopyGmToL1Bias copyGmToL1Bias;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
            auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
            copyGmToL1Bias(tensorL1Bias, tensorBias);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        }

        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mL1Actual);
        uint32_t nL0Loop = CeilDiv<L0_TILE_N>(nL1Actual);

        // main loop
        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
            uint32_t l1AListIdNext = useExternalL1AThisCall ? l1AListId :
                ((l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0);
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
            uint32_t kL1ActualNext{0};
            // preload next tile from GM to L1
            if (kL1Idx < kL1Loop - 1) {
                uint32_t kL1IdxNext = kL1Idx + 1;
                kL1ActualNext = (kL1IdxNext < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);

                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1AListIdNext];
                auto l1BTensor = l1BTensorList[l1BListIdNext];
                auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
                auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
                // Get GM tile for next stage
                auto tensorTileA = GetTileA(tensorA, 0, kL1IdxNext * L1_TILE_K, mBlockActual, kL1ActualNext);
                auto tensorTileB = GetTile(tensorB, tla::MakeCoord(kL1IdxNext * L1_TILE_K, 0),
                    tla::MakeShape(kL1ActualNext, nBlockActual));

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                if (clearL1Padding) {
                    AscendC::InitConstValueParams<ElementA> clearParams(
                        1, static_cast<uint16_t>(L1A_TILE_SIZE / 32), 0,
                        static_cast<ElementA>(0));
                    AscendC::InitConstValue(l1ATensorList[l1AListIdNext], clearParams);
                }
                if constexpr (ENABLE_L1_RESIDENT) {
                    if (lastAddrA[l1AListIdNext] != tensorTileA.data().GetPhyAddr()
                        || tla::get<0>(tensorTileA.coord()) != lastCoordA[l1AListIdNext].row()
                        || tla::get<1>(tensorTileA.coord()) != lastCoordA[l1AListIdNext].column()) {
                        copyGmToL1A(tensorL1A, tensorTileA);
                        lastCoordA[l1AListIdNext] = MatrixCoord{
                            static_cast<uint32_t>(tla::get<0>(tensorTileA.coord())),
                            static_cast<uint32_t>(tla::get<1>(tensorTileA.coord()))};
                        lastAddrA[l1AListIdNext] =
                            const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                                tensorTileA.data().GetPhyAddr()
                            );
                    }
                } else {
                    copyGmToL1A(tensorL1A, tensorTileA);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                if (clearL1Padding) {
                    AscendC::InitConstValueParams<ElementB> clearParams(
                        1, static_cast<uint16_t>(L1B_TILE_SIZE / 32), 0,
                        static_cast<ElementB>(0));
                    AscendC::InitConstValue(l1BTensorList[l1BListIdNext], clearParams);
                }
                if constexpr (ENABLE_L1_RESIDENT) {
                    if (lastAddrB[l1BListIdNext] != tensorTileB.data().GetPhyAddr()
                        || tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListIdNext].row()
                        || tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListIdNext].column()) {
                        copyGmToL1B(tensorL1B, tensorTileB);
                        lastCoordB[l1BListIdNext] = MatrixCoord{
                            static_cast<uint32_t>(tla::get<0>(tensorTileB.coord())),
                            static_cast<uint32_t>(tla::get<1>(tensorTileB.coord()))};
                        lastAddrB[l1BListIdNext] =
                            const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                                tensorTileB.data().GetPhyAddr()
                            );
                    }
                } else {
                    copyGmToL1B(tensorL1B, tensorTileB);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = useExternalL1AThisCall ? externalL1ATensor : l1ATensorList[l1AListId];
            auto l1BTensor = l1BTensorList[l1BListId];
            tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
            tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
            // Get the loop nums on L0
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

            for (int mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mL1Actual - mL0Idx * L0_TILE_M);

                for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mL0Actual, kL0Actual);
                    auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
                    // Locate the current tile of matrix A on L1
                    auto tensorTileL1A = GetTileA(tensorL1A, mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K, mL0Actual, kL0Actual);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if constexpr (!ReuseA) {
                        if ((mL0Idx == 0) && (kL0Idx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                        }
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    if constexpr (!KeepA) {
                        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                        }
                    }

                    bool initC = ((kL1Idx == 0) && (kL0Idx == 0));
                    for (int nL0Idx = 0; nL0Idx < nL0Loop; nL0Idx++) {
                        uint32_t nL0Actual = (nL0Idx < nL0Loop - 1) ? L0_TILE_N : (nL1Actual - nL0Idx * L0_TILE_N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL0Actual);
                        auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
                        // Locate the current tile of matrix B on L1
                        auto tensorTileL1B = GetTile(tensorL1B,
                                                     tla::MakeCoord(kL0Idx * L0_TILE_K, nL0Idx * L0_TILE_N),
                                                     tla::MakeShape(kL0Actual, nL0Actual));

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((mL0Idx == 0) && (kL0Idx == 0) && (nL0Idx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(tensorL0B, tensorTileL1B);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                        }

                        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                            if (initC) {
                                if (nL0Idx == 0) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                                auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
                                auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
                                auto tensorTileL1Bias = GetTile(tensorL1Bias,
                                                                tla::MakeCoord(nL0Idx * L0_TILE_N),
                                                                tla::MakeShape(nL0Actual));
                                // Load bias to l0 biasTable
                                copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                                if (nL0Idx == nL0Loop - 1) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                                }
                            }
                        }

                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                        // Locate the current tile on L0C
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                     tla::MakeCoord(mL0Idx * L0_TILE_M, nL0Idx * L0_TILE_N),
                                                     tla::MakeShape(mL0Actual, nL0Actual));

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kL1Idx == kL1Loop - 1) && (mL0Idx == mL0Loop - 1) &&
                                (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }

                        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                            if (initC) {
                                tileMmad(tensorTileL0C, tensorL0A, tensorL0B, tensorL0Bias,
                                    mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                            } else {
                                tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                                    mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                            }
                        } else {
                            tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                                mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                        }

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
                }
            }
            if constexpr (!KeepA) {
                if (!useExternalL1AThisCall) {
                    l1AListId = l1AListIdNext;
                }
            }
            l1BListId = l1BListIdNext;
            kL1Actual = kL1ActualNext;
        }
        useExternalL1ATensor = false;

        // copy block out
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            copyL0CToDst(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        } else {
            copyL0CToDst(tensorC, tensorL0C, 0b11);
        }
    }

protected:
    using Base::GetTileA;
    using Base::l1ATensorList;
    using Base::l1BTensorList;
    using Base::l0ATensorList;
    using Base::l0BTensorList;
    using Base::l0CTensorList;
    using Base::l1BiasTensor;
    using Base::l0BiasTensor;
    using Base::l1AEventList;
    using Base::l1BEventList;
    using Base::l0AEventList;
    using Base::l0BEventList;
    using Base::l0CEventList;
    using Base::lastAddrA;
    using Base::lastAddrB;
    using Base::lastCoordA;
    using Base::lastCoordB;
    using Base::l1AListId;
    using Base::l1BListId;
    using Base::l0AListId;
    using Base::l0BListId;
    using Base::l0CListId;
    using Base::skipNextADataCopy;
    using Base::useExternalL1ATensor;
    using Base::externalL1ATensor;
    using Base::tileMmad;
    using Base::copyL1ToL0A;
    using Base::copyL1ToL0B;
    using Base::copyL1ToBT;
};

} // namespace Catlass::Gemm::Block
#endif // __CCE_AICORE__ == 220

#endif // GDN_ARCH22_BLOCK_MMAD_RECOMPUTE_WU_KEEPA_HPP
