/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef GDN_FWD_O_BLOCK_MMAD_QKH_CONCAT_HPP
#define GDN_FWD_O_BLOCK_MMAD_QKH_CONCAT_HPP

#include <type_traits>
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"

namespace Catlass::Gemm::Block {

template <class Original>
using GDNFwdOQKHConcatBase = BlockMmadTla<
    MmadPingpongTlaMulti<typename Original::ArchTag, true, false, 1, false, 2, 2, 2, 1>,
    tla::Shape<tla::_128, tla::Int<192>, tla::_128>,
    tla::Shape<tla::_128, tla::Int<192>, tla::_128>,
    typename Original::ElementA, typename Original::ElementB, typename Original::ElementC,
    void, typename Original::TileCopy>;

// K^T保持nZ、H保持zN；分别装入同一个nZ L0B，不将两种L1布局互相重解释。
template <class Original, class HTileCopy>
class GDNFwdOQKHConcatMmad : public GDNFwdOQKHConcatBase<Original> {
public:
    using Base = GDNFwdOQKHConcatBase<Original>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using KTileCopy = typename Base::TileCopy;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t MAX_TOKENS = 64;
    static constexpr uint32_t K_L1_ELEMENTS = HEAD_DIM * MAX_TOKENS;
    static constexpr uint32_t H_COLUMNS = 128;
    static_assert(std::is_same_v<typename KTileCopy::LayoutTagB, layout::ColumnMajor> &&
                      std::is_same_v<typename HTileCopy::LayoutTagB, layout::RowMajor> &&
                      std::is_same_v<typename HTileCopy::ElementB, ElementB>,
                  "K and H retain their distinct native B copy layouts");
    static_assert(std::is_same_v<typename Base::ArchTag, Arch::AtlasA2> &&
                      sizeof(ElementA) == 2 && sizeof(ElementB) == 2 &&
                      std::is_same_v<typename Base::ElementC, float> &&
                      std::is_same_v<typename Base::ElementAccumulator, float>,
                  "O QKH concat preserves the A2 two-byte input and FP32 output path");
    static_assert(Base::L0B_STAGES == 1 && Base::L0A_STAGES == 2 && Base::L0C_STAGES == 1 &&
                      Base::L1A_STAGES == 2 && Base::L1B_STAGES == 2 &&
                      Base::L0_TILE_K == HEAD_DIM && Base::L1_TILE_K == HEAD_DIM &&
                      Base::L0_TILE_N == MAX_TOKENS + H_COLUMNS && Base::ENABLE_UNIT_FLAG &&
                      !Base::USE_HF32_MODE && !Base::HAS_BIAS,
                  "O QKH concat uses one K128 step and one 48KiB L0B bank");

    CATLASS_DEVICE
    explicit GDNFwdOQKHConcatMmad(Arch::Resource<typename Base::ArchTag> &resource) : Base(resource) {}

    template <class TensorQ, class TensorK, class TensorH, class TensorRaw>
    CATLASS_DEVICE void RunRaw(TensorQ &tensorQ, TensorK &tensorK, TensorH &tensorH,
                              TensorRaw &tensorRaw, uint32_t valid)
    {
        typename KTileCopy::template CopyGmToL1A<TensorQ> copyQ;
        typename KTileCopy::template CopyGmToL1B<TensorK> copyK;
        typename HTileCopy::template CopyGmToL1B<TensorH> copyH;
        typename HTileCopy::CopyL1ToL0B copyHToL0;
        typename KTileCopy::template CopyL0CToGm<TensorRaw> copyRaw;
        const uint32_t a1 = this->l1AListId;
        const uint32_t b1 = this->l1BListId;
        const uint32_t a0 = this->l0AListId;
        const uint32_t mActual = valid == 1 ? 16 : valid; // 保留原GEMV规避。
        const uint32_t hColumn = (valid + 15u) / 16u * 16u;
        const uint32_t nActual = hColumn + H_COLUMNS;

        auto tensorL1A = tla::MakeTensor(this->l1ATensorList[a1], Base::L1A_LAYOUT, Arch::PositionL1{});
        auto layoutL1K = tla::MakeLayout<ElementB, typename KTileCopy::LayoutTagL1B>(
            tla::Int<HEAD_DIM>{}, tla::Int<MAX_TOKENS>{});
        auto layoutL1H = tla::MakeLayout<ElementB, typename HTileCopy::LayoutTagL1B>(
            tla::Int<HEAD_DIM>{}, tla::Int<H_COLUMNS>{});
        auto tensorL1K = tla::MakeTensor(this->l1BTensorList[b1], layoutL1K, Arch::PositionL1{});
        auto tensorL1H = tla::MakeTensor(this->l1BTensorList[b1][K_L1_ELEMENTS],
                                       layoutL1H, Arch::PositionL1{});
        auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(valid, HEAD_DIM));
        auto blockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(HEAD_DIM, valid));
        auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(HEAD_DIM, H_COLUMNS));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[a1]);
        copyQ(tensorL1A, blockQ);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[a1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[b1]);
        copyK(tensorL1K, blockK);
        copyH(tensorL1H, blockH);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[b1]);

        auto layoutL0A = tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(mActual, HEAD_DIM);
        auto layoutL0B = tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(HEAD_DIM, nActual);
        auto tensorL0A = tla::MakeTensor(this->l0ATensorList[a0], layoutL0A, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(this->l0BTensorList[0], layoutL0B, Arch::PositionL0B{});
        auto l1Q = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, HEAD_DIM));
        auto l1K = GetTile(tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(HEAD_DIM, valid));
        auto l1H = GetTile(tensorL1H, tla::MakeCoord(0, 0), tla::MakeShape(HEAD_DIM, H_COLUMNS));
        auto l0K = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(HEAD_DIM, valid));
        auto l0H = GetTile(tensorL0B, tla::MakeCoord(0, hColumn), tla::MakeShape(HEAD_DIM, H_COLUMNS));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[a0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[a1]);
        this->copyL1ToL0A(tensorL0A, l1Q);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[a1]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[b1]);
        this->copyL1ToL0B(l0K, l1K);
        copyHToL0(l0H, l1H);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[b1]);

        auto layoutL0C = tla::MakeLayoutL0C(mActual, nActual);
        auto tensorL0C = tla::MakeTensor(this->l0CTensorList[0], layoutL0C, Arch::PositionL0C{});
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(this->l0CEventList[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(this->l0CEventList[0]);
        this->tileMmad(tensorL0C, tensorL0A, tensorL0B, mActual, nActual, HEAD_DIM, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[0]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[a0]);
        this->l1AListId = a1 ^ 1u;
        this->l1BListId = b1 ^ 1u;
        this->l0AListId = a0 ^ 1u;
        // L0B只有一个物理bank，索引始终为0。

        auto rawDst = GetTile(tensorRaw, tla::MakeCoord(0, 0), tla::MakeShape(valid, valid));
        auto rawSrc = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(valid, valid));
        copyRaw(rawDst, rawSrc, 0b11);
    }

    // RunRaw后不得finalWait/再种票/调用其它MMAD；等旧HWork读者退休后再写H半区。
    template <class TensorHWork>
    CATLASS_DEVICE void StoreH(TensorHWork &tensorHWork, uint32_t valid)
    {
        typename KTileCopy::template CopyL0CToGm<TensorHWork> copyHWork;
        const uint32_t mActual = valid == 1 ? 16 : valid;
        const uint32_t hColumn = (valid + 15u) / 16u * 16u;
        auto layoutL0C = tla::MakeLayoutL0C(mActual, hColumn + H_COLUMNS);
        auto tensorL0C = tla::MakeTensor(this->l0CTensorList[0], layoutL0C, Arch::PositionL0C{});
        auto hDst = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(valid, H_COLUMNS));
        auto hSrc = GetTile(tensorL0C, tla::MakeCoord(0, hColumn), tla::MakeShape(valid, H_COLUMNS));
        copyHWork(hDst, hSrc, 0b11);
    }
};

} // namespace Catlass::Gemm::Block

#endif // GDN_FWD_O_BLOCK_MMAD_QKH_CONCAT_HPP
