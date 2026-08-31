/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_TLA_PIPELINED_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_TLA_PIPELINED_HPP

#include "block_mmad_pingpong_tla_gdn_fwd_o.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

/// Pipelined version of BlockMmadTla that separates GM->L1 copy from
/// L1->L0 + Mmad + L0C->GM.  This allows overlapping the GM->L1 phase of
/// one instance with the Mmad phase of the previous instance, enabling
/// cross-Gemm pipeline overlap on the same AIC core.
///
/// Usage pattern (overlapping instance A's Mmad with instance B's GM->L1):
///
///   // Instance A (e.g. Cube1)
///   A.preSetFlags();                  // init all L1+L0 flags + HW mode
///   A(tensorA, tensorB, tensorC, shape);  // full operator() internally
///   A.waitL1Drained();                // wait only L1 flags (GM->L1 done)
///
///   // Instance B (e.g. Cube2) -- GM->L1 overlaps with A's Mmad
///   B.preSetL1Flags();                // init L1 flags + HW mode only
///   B.copyGmToL1(tensorA, tensorB, shape);  // GM->L1A/L1B (L1 flags only)
///
///   A.finalWaitFlags();               // wait A's L0 flags (Mmad done)
///
///   B.preSetL0Flags();                // init L0 flags (A's L0 now free)
///   B.executeCompute(tensorC, shape); // L1->L0 + Mmad + L0C->GM
///   B.finalWaitFlags();
///
/// Constraint: only valid when kL1Loop == 1 (kBlockActual <= L1_TILE_K),
/// i.e. the K dimension fits in a single L1 tile.  When kL1Loop > 1 the
/// original operator() interleaves GM->L1 preload with Mmad and cannot be
/// split this way.
template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_ = void,
    class TileCopy_ = Catlass::Gemm::Tile::PackedTileCopyTla<
        typename DispatchPolicy::ArchTag, ElementA_, Catlass::layout::RowMajor,
        ElementB_, Catlass::layout::RowMajor, ElementC_, Catlass::layout::RowMajor, ElementBias_>,
    class TileMmad_ = Catlass::Gemm::Tile::TileMmadTla<
        typename DispatchPolicy::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>
>
struct BlockMmadTlaPipelined : public BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape,
    ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_
> {
    using Base = BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape,
        ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_
    >;

    // Type aliases from Base
    using typename Base::ElementA;
    using typename Base::ElementB;
    using typename Base::ElementC;
    using typename Base::TileCopy;
    using typename Base::ArchTag;
    using typename Base::LayoutC;

    // Compile-time constants from Base
    using Base::L1_TILE_M;
    using Base::L1_TILE_N;
    using Base::L1_TILE_K;
    using Base::L0_TILE_M;
    using Base::L0_TILE_N;
    using Base::L0_TILE_K;
    using Base::L1A_STAGES;
    using Base::L1B_STAGES;
    using Base::L0A_STAGES;
    using Base::L0B_STAGES;
    using Base::L0C_STAGES;
    using Base::ENABLE_UNIT_FLAG;
    using Base::ENABLE_L1_RESIDENT;
    using Base::SHARE_L1A;
    using Base::HAS_BIAS;
    using Base::L1A_TILE_SIZE;
    using Base::L1B_TILE_SIZE;

    // Layouts from Base
    using Base::L1A_LAYOUT;
    using Base::L1B_LAYOUT;

    CATLASS_DEVICE
    BlockMmadTlaPipelined(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0,
                          uint32_t l1AEventIdOffset = 0, uint32_t l1BEventIdOffset = 0,
                          uint32_t l1ABufOffset = 0)
        : Base(resource, l1BufAddrStart) {
        if ASCEND_IS_AIC {
            // Override L1A buffer offset (for SHARE_L1A: L1A at 0, L1B at l1BufAddrStart)
            if (l1ABufOffset > 0 || (SHARE_L1A && l1BufAddrStart > 0)) {
                uint32_t l1AOff = SHARE_L1A ? 0 : l1ABufOffset;
                for (uint32_t i = 0; i < L1A_STAGES; i++) {
                    this->l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOff + L1A_TILE_SIZE * i);
                }
            }
            // Override L1A eventIDs
            if (l1AEventIdOffset > 0) {
                for (uint32_t i = 0; i < L1A_STAGES; i++) {
                    this->l1AEventList[i] = i + l1AEventIdOffset;
                }
            }
            // Override L1B eventIDs
            if (l1BEventIdOffset > 0) {
                for (uint32_t i = 0; i < L1B_STAGES; i++) {
                    this->l1BEventList[i] = i + l1BEventIdOffset;
                }
            }
        }
    }

    /// Wait only for L1 flags to drain (GM->L1A/L1B complete).
    /// L0A/L0B flags are NOT waited -- the caller may overlap the next
    /// instance's GM->L1 with this instance's Mmad phase.
    CATLASS_DEVICE
    void waitL1Drained() {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[i]);
            }
        }
    }

    /// Wait only for L0A/L0B flags to drain (Mmad complete).
    /// Use after waitL1Drained() to avoid double-waiting L1 flags.
    CATLASS_DEVICE
    void waitL0Drained() {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[i]);
            }
            if constexpr (HAS_BIAS) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }
        }
    }

    /// Initialize only L1 flags and hardware mode (HF32, LayoutTransform).
    /// Does NOT touch L0A/L0B flags -- safe to call while another instance's
    /// operator() or executeCompute() is still running.
    CATLASS_DEVICE
    void preSetL1Flags() {
        if ASCEND_IS_AIC {
            if constexpr (Base::USE_HF32_MODE) {
                AscendC::SetHF32Mode(true);
            } else {
                AscendC::SetHF32Mode(false);
            }
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(true);
            }
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[i]);
            }
            if constexpr (ENABLE_L1_RESIDENT) {
                Base::RestoreStatus();
            }
        }
    }

    /// Initialize only L0A/L0B flags.  Must be called after the previous
    /// instance's finalWaitFlags() has drained L0 flags.
    CATLASS_DEVICE
    void preSetL0Flags() {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[i]);
            }
        }
    }

    /// Phase 1: Copy GM->L1A/L1B.  Only uses L1 flags.
    /// Requires preSetL1Flags() to have been called.
    template <class TensorA, class TensorB>
    CATLASS_DEVICE void copyGmToL1(TensorA &tensorA, TensorB &tensorB, GemmCoord const &actualShape)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);

        // load matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[this->l1AListId]);
        auto tensorL1A = tla::MakeTensor(this->l1ATensorList[this->l1AListId], L1A_LAYOUT, Arch::PositionL1{});
        auto tensorTileA = this->GetTileA(tensorA, 0, 0, mBlockActual, kL1Actual);
        if constexpr (SHARE_L1A) {
            // L1A is shared from another BlockMmad instance; skip GM->L1A copy.
        } else if constexpr (ENABLE_L1_RESIDENT) {
            if (this->lastAddrA[this->l1AListId] != tensorTileA.data().GetPhyAddr()
                || tla::get<0>(tensorTileA.coord()) != this->lastCoordA[this->l1AListId].row()
                || tla::get<1>(tensorTileA.coord()) != this->lastCoordA[this->l1AListId].column()) {
                copyGmToL1A(tensorL1A, tensorTileA);
                this->lastCoordA[this->l1AListId] = MatrixCoord{tla::get<0>(tensorTileA.coord()), tla::get<1>(tensorTileA.coord())};
                this->lastAddrA[this->l1AListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                    tensorTileA.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1A(tensorL1A, tensorTileA);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[this->l1AListId]);

        // load matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[this->l1BListId]);
        auto tensorL1B = tla::MakeTensor(this->l1BTensorList[this->l1BListId], L1B_LAYOUT, Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(kL1Actual, nBlockActual));
        if constexpr (ENABLE_L1_RESIDENT) {
            if (this->lastAddrB[this->l1BListId] != tensorTileB.data().GetPhyAddr()
                || tla::get<0>(tensorTileB.coord()) != this->lastCoordB[this->l1BListId].row()
                || tla::get<1>(tensorTileB.coord()) != this->lastCoordB[this->l1BListId].column()) {
                copyGmToL1B(tensorL1B, tensorTileB);
                this->lastCoordB[this->l1BListId] = MatrixCoord{tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                this->lastAddrB[this->l1BListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                    tensorTileB.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1B(tensorL1B, tensorTileB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[this->l1BListId]);
    }

    /// Phase 1a: Copy GM->L1A only.  Only uses L1A flags.
    /// Requires preSetL1Flags() to have been called.
    /// Useful when L1A and L1B have different cross-core dependencies and
    /// should be loaded separately to maximize MTE2 overlap with M pipeline.
    template <class TensorA>
    CATLASS_DEVICE void copyGmToL1AOnly(TensorA &tensorA, GemmCoord const &actualShape)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[this->l1AListId]);
        auto tensorL1A = tla::MakeTensor(this->l1ATensorList[this->l1AListId], L1A_LAYOUT, Arch::PositionL1{});
        auto tensorTileA = this->GetTileA(tensorA, 0, 0, mBlockActual, kL1Actual);
        if constexpr (SHARE_L1A) {
            // L1A is shared from another BlockMmad instance; skip GM->L1A copy.
        } else if constexpr (ENABLE_L1_RESIDENT) {
            if (this->lastAddrA[this->l1AListId] != tensorTileA.data().GetPhyAddr()
                || tla::get<0>(tensorTileA.coord()) != this->lastCoordA[this->l1AListId].row()
                || tla::get<1>(tensorTileA.coord()) != this->lastCoordA[this->l1AListId].column()) {
                copyGmToL1A(tensorL1A, tensorTileA);
                this->lastCoordA[this->l1AListId] = MatrixCoord{tla::get<0>(tensorTileA.coord()), tla::get<1>(tensorTileA.coord())};
                this->lastAddrA[this->l1AListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                    tensorTileA.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1A(tensorL1A, tensorTileA);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[this->l1AListId]);
    }

    /// Phase 1b: Copy GM->L1B only.  Only uses L1B flags.
    /// Requires preSetL1Flags() to have been called.
    /// Useful when L1A and L1B have different cross-core dependencies and
    /// should be loaded separately to maximize MTE2 overlap with M pipeline.
    template <class TensorB>
    CATLASS_DEVICE void copyGmToL1BOnly(TensorB &tensorB, GemmCoord const &actualShape)
    {
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;

        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();
        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[this->l1BListId]);
        auto tensorL1B = tla::MakeTensor(this->l1BTensorList[this->l1BListId], L1B_LAYOUT, Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(kL1Actual, nBlockActual));
        if constexpr (ENABLE_L1_RESIDENT) {
            if (this->lastAddrB[this->l1BListId] != tensorTileB.data().GetPhyAddr()
                || tla::get<0>(tensorTileB.coord()) != this->lastCoordB[this->l1BListId].row()
                || tla::get<1>(tensorTileB.coord()) != this->lastCoordB[this->l1BListId].column()) {
                copyGmToL1B(tensorL1B, tensorTileB);
                this->lastCoordB[this->l1BListId] = MatrixCoord{tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                this->lastAddrB[this->l1BListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                    tensorTileB.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1B(tensorL1B, tensorTileB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[this->l1BListId]);
    }

    /// Phase 2: L1->L0 + Mmad + L0C->GM.  Requires preSetL0Flags() and
    /// copyGmToL1() to have been called.
    template <class TensorC>
    CATLASS_DEVICE void executeCompute(TensorC &tensorC, GemmCoord const &actualShape)
    {
        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(this->l0CTensorList[this->l0CListId], layoutInL0C, Arch::PositionL0C{});

        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(this->l0CEventList[this->l0CListId]);
        }

        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mL1Actual);
        uint32_t nL0Loop = CeilDiv<L0_TILE_N>(nL1Actual);
        uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

        auto l1ATensor = this->l1ATensorList[this->l1AListId];
        auto l1BTensor = this->l1BTensorList[this->l1BListId];
        auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});

        for (int mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
            uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mL1Actual - mL0Idx * L0_TILE_M);

            for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);

                auto l0ATile = this->l0ATensorList[this->l0AListId];
                auto layoutAInL0 = tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(mL0Actual, kL0Actual);
                auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
                auto tensorTileL1A = this->GetTileA(tensorL1A, mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K, mL0Actual, kL0Actual);

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[this->l0AListId]);
                if ((mL0Idx == 0) && (kL0Idx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[this->l1AListId]);
                }

                this->copyL1ToL0A(tensorL0A, tensorTileL1A);

                if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[this->l1AListId]);
                }

                bool initC = (kL0Idx == 0);
                for (int nL0Idx = 0; nL0Idx < nL0Loop; nL0Idx++) {
                    uint32_t nL0Actual = (nL0Idx < nL0Loop - 1) ? L0_TILE_N : (nL1Actual - nL0Idx * L0_TILE_N);

                    auto l0BTile = this->l0BTensorList[this->l0BListId];
                    auto layoutBInL0 = tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(kL0Actual, nL0Actual);
                    auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(tensorL1B,
                                                 tla::MakeCoord(kL0Idx * L0_TILE_K, nL0Idx * L0_TILE_N),
                                                 tla::MakeShape(kL0Actual, nL0Actual));

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[this->l0BListId]);
                    if ((mL0Idx == 0) && (kL0Idx == 0) && (nL0Idx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[this->l1BListId]);
                    }

                    this->copyL1ToL0B(tensorL0B, tensorTileL1B);

                    if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[this->l1BListId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(this->l0CEventList[this->l0CListId]);

                    auto tensorTileL0C = GetTile(tensorL0C,
                                                 tla::MakeCoord(mL0Idx * L0_TILE_M, nL0Idx * L0_TILE_N),
                                                 tla::MakeShape(mL0Actual, nL0Actual));

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(this->l0CEventList[this->l0CListId]);

                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }

                    this->tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                        mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[this->l0BListId]);
                    this->l0BListId = (this->l0BListId + 1 < L0B_STAGES) ? (this->l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[this->l0AListId]);
                this->l0AListId = (this->l0AListId + 1 < L0A_STAGES) ? (this->l0AListId + 1) : 0;
            }
        }

        // copy block out
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(this->l0CEventList[this->l0CListId]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(this->l0CEventList[this->l0CListId]);
            using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(this->l0CEventList[this->l0CListId]);
            this->l0CListId = (this->l0CListId + 1 < L0C_STAGES) ? (this->l0CListId + 1) : 0;
        } else {
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
            using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(tensorC, tensorL0C, 0b11);
#else
            using CopyL0CToDst = typename TileCopy::template CopyL0CToGm<TensorC>;
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(tensorC, tensorL0C);
#endif
        }

        // Advance L1 ping-pong slot, matching the original operator() behavior
        // where l1AListId/l1BListId are advanced at the end of the kL1Loop.
        // This is critical for SHARE_L1A: without it, Cube2 would always read
        // from slot 0 while Cube1 alternates between slot 0 and 1.
        this->l1AListId = (this->l1AListId + 1 < L1A_STAGES) ? (this->l1AListId + 1) : 0;
        this->l1BListId = (this->l1BListId + 1 < L1B_STAGES) ? (this->l1BListId + 1) : 0;
    }
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_TLA_PIPELINED_HPP
