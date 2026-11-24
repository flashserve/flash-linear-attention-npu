/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef RECOMPUTE_W_U_FWD_CONCAT_CUBE_H
#define RECOMPUTE_W_U_FWD_CONCAT_CUBE_H

#include <type_traits>
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"

namespace Catlass::Gemm::Block {

// BT64/V128的单K步：复用原宽形状资源，只扩展为两个GM B输入和两个GM输出。
template <class Base_>
class RecomputeWUConcatMmad : public Base_ {
public:
    using Base = Base_;
    using ArchTag = typename Base::ArchTag;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using TileCopy = typename Base::TileCopy;
    static constexpr uint32_t HALF_N = 128;
    static constexpr uint32_t CONCAT_N = 2 * HALF_N;
    static_assert(std::is_same_v<ArchTag, Arch::AtlasA2> && sizeof(ElementA) == 2 && sizeof(ElementB) == 2,
                  "WU concat preserves the A2 two-byte input path");
    static_assert(Base::L1_TILE_M == 128 && Base::L1_TILE_N == CONCAT_N && Base::L1_TILE_K == 256 &&
                      Base::L0_TILE_M == 128 && Base::L0_TILE_N == CONCAT_N && Base::L0_TILE_K == 64,
                  "WU concat uses the existing V256 physical tile shapes");
    static_assert(Base::L1A_STAGES == 2 && Base::L1B_STAGES == 2 && Base::L0A_STAGES == 2 &&
                      Base::L0B_STAGES == 2 && Base::L0C_STAGES == 1 && Base::ENABLE_UNIT_FLAG &&
                      !Base::USE_HF32_MODE && !Base::HAS_BIAS &&
                      std::is_same_v<typename Base::ElementAccumulator, float>,
                  "WU concat preserves two input slots and one UnitFlag accumulator");

    CATLASS_DEVICE
    explicit RecomputeWUConcatMmad(Arch::Resource<ArchTag> &resource) : Base(resource) {}

    template <class TensorA, class TensorVb, class TensorKbg, class TensorU, class TensorW>
    CATLASS_DEVICE void RunPair(TensorA &tensorA, TensorVb &tensorVb, TensorKbg &tensorKbg,
                               TensorU &tensorU, TensorW &tensorW, uint32_t valid)
    {
        using CopyA = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyVb = typename TileCopy::template CopyGmToL1B<TensorVb>;
        using CopyKbg = typename TileCopy::template CopyGmToL1B<TensorKbg>;
        using CopyU = typename TileCopy::template CopyL0CToGm<TensorU>;
        using CopyW = typename TileCopy::template CopyL0CToGm<TensorW>;
        CopyA copyA;
        CopyVb copyVb;
        CopyKbg copyKbg;
        CopyU copyU;
        CopyW copyW;
        const uint32_t a1 = this->l1AListId;
        const uint32_t b1 = this->l1BListId;
        const uint32_t a0 = this->l0AListId;
        const uint32_t b0 = this->l0BListId;
        uint32_t mActual = valid;
        if (mActual == 1) {
            mActual = 16; // 与原MMAD相同，避免GEMV；GM读和写出仍只用valid行。
        }

        auto tensorL1A = tla::MakeTensor(this->l1ATensorList[a1], Base::L1A_LAYOUT, Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(this->l1BTensorList[b1], Base::L1B_LAYOUT, Arch::PositionL1{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(valid, valid));
        auto blockVb = GetTile(tensorVb, tla::MakeCoord(0, 0), tla::MakeShape(valid, HALF_N));
        auto blockKbg = GetTile(tensorKbg, tla::MakeCoord(0, 0), tla::MakeShape(valid, HALF_N));
        // 继承父布局的K256 pitch；第二半区起点是64KiB，不随valid变化。
        auto l1Vb = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(Base::L1_TILE_K, HALF_N));
        auto l1Kbg = GetTile(tensorL1B, tla::MakeCoord(0, HALF_N), tla::MakeShape(Base::L1_TILE_K, HALF_N));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[a1]);
        copyA(tensorL1A, blockA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[a1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[b1]);
        copyVb(l1Vb, blockVb);
        copyKbg(l1Kbg, blockKbg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[b1]);

        auto layoutL0A = tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(mActual, valid);
        auto layoutL0B = tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(valid, CONCAT_N);
        auto tensorL0A = tla::MakeTensor(this->l0ATensorList[a0], layoutL0A, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(this->l0BTensorList[b0], layoutL0B, Arch::PositionL0B{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, valid));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(valid, CONCAT_N));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[a0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[a1]);
        this->copyL1ToL0A(tensorL0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[a1]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[b0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[b1]);
        this->copyL1ToL0B(tensorL0B, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[b1]);

        // actual M决定L0C pitch，不能使用物理分配上限128。
        auto layoutL0C = tla::MakeLayoutL0C(mActual, CONCAT_N);
        auto tensorL0C = tla::MakeTensor(this->l0CTensorList[0], layoutL0C, Arch::PositionL0C{});
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(this->l0CEventList[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(this->l0CEventList[0]);
        this->tileMmad(tensorL0C, tensorL0A, tensorL0B, mActual, CONCAT_N, valid, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[b0]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[a0]);
        this->l1AListId = a1 ^ 1u;
        this->l1BListId = b1 ^ 1u;
        this->l0AListId = a0 ^ 1u;
        this->l0BListId = b0 ^ 1u;

        auto blockU = GetTile(tensorU, tla::MakeCoord(0, 0), tla::MakeShape(valid, HALF_N));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(valid, HALF_N));
        auto l0CU = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(valid, HALF_N));
        auto l0CW = GetTile(tensorL0C, tla::MakeCoord(0, HALF_N), tla::MakeShape(valid, HALF_N));
        // 两半是不同512B状态块；各自等待MMAD发布并以3释放，不留左半标记。
        copyU(blockU, l0CU, 0b11);
        copyW(blockW, l0CW, 0b11);
    }
};

// 延迟构造宽形状；非适用类型只实例化原BlockMmad，不触发宽形状预算断言。
template <class Original, bool Enabled>
struct RecomputeWUConcatSelector {
    using Type = Original;
};

template <class Original>
struct RecomputeWUConcatSelector<Original, true> {
    using WideBase = BlockMmadTla<typename Original::DispatchPolicy,
        tla::Shape<tla::_128, tla::_256, tla::_256>, tla::Shape<tla::_128, tla::_256, tla::_64>,
        typename Original::ElementA, typename Original::ElementB, typename Original::ElementC,
        void, typename Original::TileCopy>;
    using Type = RecomputeWUConcatMmad<WideBase>;
};

} // namespace Catlass::Gemm::Block

#endif // RECOMPUTE_W_U_FWD_CONCAT_CUBE_H
