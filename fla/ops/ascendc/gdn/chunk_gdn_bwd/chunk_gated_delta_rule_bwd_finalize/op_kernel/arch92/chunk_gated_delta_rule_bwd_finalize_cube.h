#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_CUBE_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_CUBE_H

#include "chunk_gated_delta_rule_bwd_finalize_common.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/copy_l0c_to_l1.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"

namespace GDN::FinalizeArch92 {

template <typename DT>
class ChunkGatedDeltaRuleBwdFinalizeCube {
    using ArchTag = Catlass::Arch::Ascend960DT;
    using RM = Catlass::layout::RowMajor;
    using CM = Catlass::layout::ColumnMajor;
    using ZN = Catlass::layout::zN;
    using NZ = Catlass::layout::nZ;
    using Resource = Catlass::Arch::Resource<ArchTag>;
    template <typename A, typename B>
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, DT, A, DT, B, DT, RM>;
    using TileCopyDau = Copy<RM, CM>;
    using TileCopyStateVFirst = Copy<RM, RM>;
    using TileCopyDw0 = Copy<RM, RM>;
    using TileCopyDvb = Copy<CM, RM>;
    using TileCopyStage9DkbT = Copy<CM, RM>;
    enum class Destination { UB, L1, GM };

public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR dO, GM_ADDR du, GM_ADDR h, GM_ADDR dh, GM_ADDR a,
        GM_ADDR kbg, GM_ADDR vb, GM_ADDR dA0, GM_ADDR ds0Out,
        GM_ADDR dkbOut, GM_ADDR dkbTOut, GM_ADDR dsIn,
        GM_ADDR doGIn, GM_ADDR vDecayIn, GM_ADDR workspace,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling)
    {
        q_ = q;
        k_ = k;
        vNew_ = vNew;
        do_ = dO;
        du_ = du;
        h_ = h;
        dh_ = dh;
        a_ = a;
        kbg_ = kbg;
        vb_ = vb;
        dA_ = dA0;
        dkb_ = dkbOut;
        dkbT_ = dkbTOut;
        ds_ = dsIn;
        doG_ = doGIn;
        vDecay_ = vDecayIn;
        kb_ = workspace;
        cu_ = cuSeqlens;
        indices_ = chunkIndices;
        tiling_ = tiling;
        core_ = AscendC::GetBlockIdx();
        cores_ = AscendC::GetBlockNum();
        stateChunks_ = tiling_->isVariable ? tiling_->totalChunkNum : tiling_->chunkNumForT;
    }

    __aicore__ inline void Process()
    {
        Resource resource;
        resource_ = &resource;
        // Resource drains its own TPipe before the event allocator is used.
        AscendC::TPipe events;
        pipe_ = &events;
        InitEvents();
        for (int64_t task = core_; task < tiling_->taskNum; task += cores_) {
            ChunkInfo chunk;
            GetChunkInfo(task / tiling_->headGroupNum, cu_, indices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            const int64_t group = task % tiling_->headGroupNum;
            const int64_t windows = tiling_->headRatio > HEAD_COUNT ?
                (tiling_->headRatio + HEAD_COUNT - 1) / HEAD_COUNT : 1;
            for (int64_t window = 0; window < windows; ++window) {
                const int64_t hvBase = tiling_->headRatio > HEAD_COUNT ?
                    group * tiling_->headRatio + window * HEAD_COUNT : group * tiling_->taskGroupSize;
                const int64_t count = Min(static_cast<int64_t>(HEAD_COUNT),
                    tiling_->headRatio > HEAD_COUNT ?
                        tiling_->headRatio - window * HEAD_COUNT : tiling_->taskGroupSize);
                // Stage-major loops and matching head order on the single AIV.
                for (uint32_t r = 0; r < count; ++r) {
                    Stage1(chunk, hvBase + r, r);
                }
                for (uint32_t r = 0; r < count; ++r) {
                    Stage3(chunk, hvBase + r, r);
                    NotifyVector();
                }
                for (uint32_t r = 0; r < count; ++r) {
                    Stage5(chunk, hvBase + r, r);
                }
                for (uint32_t r = 0; r < count; ++r) {
                    Stage7(chunk, r);
                    NotifyVector();
                }
                for (uint32_t r = 0; r < count; ++r) {
                    Stage9(chunk, r);
                }
                for (uint32_t r = 0; r < count; ++r) {
                    Stage11(chunk, hvBase + r, r);
                    NotifyVector();
                }
                // dkBase slots overlap adjacent ds slots. Finish ALL S12 MTE3
                // readers before any S13 Fixpipe can write those UB addresses.
                for (uint32_t r = 0; r < count; ++r) {
                    WaitVector();
                }
                for (uint32_t r = 0; r < count; ++r) {
                    Stage13(chunk, hvBase + r, r);
                    NotifyVector();
                }
                WaitVector<PIPE_FIX>(); // S14 free bounds both ordered counters.
                ++windowSerial_;
            }
        }
        ReleaseEvents();
    }

private:
    template <AscendC::HardEvent E>
    __aicore__ inline void Allocate(AscendC::TEventID *ids, uint32_t count, bool free)
    {
        for (uint32_t i = 0; i < count; ++i) {
            ids[i] = pipe_->AllocEventID<E>();
            if (free) {
                AscendC::SetFlag<E>(ids[i]);
            }
        }
    }

    template <AscendC::HardEvent E>
    __aicore__ inline void Release(AscendC::TEventID *ids, uint32_t count, bool free)
    {
        for (uint32_t i = 0; i < count; ++i) {
            if (free) {
                AscendC::WaitFlag<E>(ids[i]);
            }
            pipe_->ReleaseEventID<E>(ids[i]);
        }
    }

    __aicore__ inline void InitEvents()
    {
        Allocate<AscendC::HardEvent::MTE1_MTE2>(residentFree_, 2 * HEAD_COUNT, true);
        Allocate<AscendC::HardEvent::MTE2_MTE1>(residentReady_, 2 * HEAD_COUNT, false);
        Allocate<AscendC::HardEvent::FIX_MTE1>(l1ResultReady_, HEAD_COUNT, false);
        Allocate<AscendC::HardEvent::M_MTE1>(aFree_, SLOT_COUNT, true);
        Allocate<AscendC::HardEvent::M_MTE1>(bFree_, SLOT_COUNT, true);
        Allocate<AscendC::HardEvent::MTE1_M>(operandReady_, SLOT_COUNT, false);
        Allocate<AscendC::HardEvent::FIX_M>(cFree_, SLOT_COUNT, true);
        Allocate<AscendC::HardEvent::M_FIX>(resultReady_, SLOT_COUNT, false);
    }

    __aicore__ inline void ReleaseEvents()
    {
        Release<AscendC::HardEvent::MTE1_MTE2>(residentFree_, 2 * HEAD_COUNT, true);
        Release<AscendC::HardEvent::MTE2_MTE1>(residentReady_, 2 * HEAD_COUNT, false);
        Release<AscendC::HardEvent::FIX_MTE1>(l1ResultReady_, HEAD_COUNT, false);
        Release<AscendC::HardEvent::M_MTE1>(aFree_, SLOT_COUNT, true);
        Release<AscendC::HardEvent::M_MTE1>(bFree_, SLOT_COUNT, true);
        Release<AscendC::HardEvent::MTE1_M>(operandReady_, SLOT_COUNT, false);
        Release<AscendC::HardEvent::FIX_M>(cFree_, SLOT_COUNT, true);
        Release<AscendC::HardEvent::M_FIX>(resultReady_, SLOT_COUNT, false);
    }

    __aicore__ inline void BeginInput(uint32_t event)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(residentFree_[event]);
    }

    __aicore__ inline void InputReady(uint32_t event)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(residentReady_[event]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(residentReady_[event]);
    }

    __aicore__ inline void EndInput(uint32_t event)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(residentFree_[event]);
    }

    template <typename Layout>
    __aicore__ inline auto L1View(uint32_t offset, uint32_t head, uint32_t stride,
                                 uint32_t rows, uint32_t cols)
    {
        return tla::MakeTensor(resource_->l1Buf.template GetBufferByByte<DT>(offset + head * stride),
            tla::MakeLayout<DT, Layout>(rows, cols), Catlass::Arch::PositionL1{});
    }

    template <typename Layout, typename TensorDst>
    __aicore__ inline void Load(const TensorDst &dst, GM_ADDR src, int64_t offset,
                               uint32_t rows, uint32_t cols, int64_t leading)
    {
        AscendC::GlobalTensor<DT> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src) + offset);
        auto tensor = tla::MakeTensor(gm, tla::MakeLayoutFromTag(Layout(rows, cols, leading)),
                                     Catlass::Arch::PositionGM{});
        auto tile = tla::GetTile(tensor, tla::MakeCoord(0, 0), tla::MakeShape(rows, cols));
        Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(tile), TensorDst>{}(dst, tile);
    }

    __aicore__ inline auto UbOutput(uint32_t offset, uint32_t head, uint32_t stride, uint32_t cols)
    {
        return tla::MakeTensor(resource_->ubBuf.template GetBufferByByte<DT>(offset + head * stride),
            tla::MakeLayout<DT, RM>(CHUNK_SIZE_64, cols), Catlass::Arch::PositionUB{});
    }

    __aicore__ inline auto GmOutput(GM_ADDR addr, uint32_t head)
    {
        AscendC::GlobalTensor<DT> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(addr) + WorkspaceOffset(head));
        return tla::MakeTensor(gm, tla::MakeLayout<DT, RM>(CHUNK_SIZE_64, K_SIZE_128),
                               Catlass::Arch::PositionGM{});
    }

    // Every emitted N tile owns one 16-KiB L0C slot through Fixpipe.
    // A/B independently rotate on every MMAD; no 128-column B occupies L0B.
    template <typename TileCopy, typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void Mmad(const TensorA &a, const TensorB &b, const TensorC &c,
                               uint32_t cSlot, uint32_t m, uint32_t n, uint32_t k,
                               uint32_t column, bool init)
    {
        const uint32_t aSlot = aSerial_++ % SLOT_COUNT;
        const uint32_t bSlot = bSerial_++ % SLOT_COUNT;
        auto l0a = tla::MakeTensor(
            resource_->l0ABuf.template GetBufferByByte<DT>(aSlot * L0_SLOT_BYTES),
            tla::MakeLayout<DT, typename TileCopy::LayoutTagL0A>(m, k), Catlass::Arch::PositionL0A{});
        auto l0b = tla::MakeTensor(
            resource_->l0BBuf.template GetBufferByByte<DT>(bSlot * L0_SLOT_BYTES),
            tla::MakeLayout<DT, typename TileCopy::LayoutTagL0B>(k, n), Catlass::Arch::PositionL0B{});
        auto tileA = tla::GetTile(a, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = tla::GetTile(b, tla::MakeCoord(0, column), tla::MakeShape(k, n));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(aFree_[aSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bFree_[bSlot]);
        typename TileCopy::CopyL1ToL0A{}(l0a, tileA);
        typename TileCopy::CopyL1ToL0B{}(l0b, tileB);
        // Follow the A-slot credit, including the second MMAD of fused dq.
        // Reusing this event cannot overtake the M consumer of the same A slot.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(operandReady_[aSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(operandReady_[aSlot]);
        if (init) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cFree_[cSlot]);
        }
        Catlass::Gemm::Tile::TileMmadTla<ArchTag, DT, typename TileCopy::LayoutTagL1A>{}(
            c, l0a, l0b, m, n, k, init, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(aFree_[aSlot]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bFree_[bSlot]);
    }

    template <Destination DEST, typename TensorDst, typename TensorC>
    __aicore__ inline void Emit(const TensorDst &dst, const TensorC &c, uint32_t slot,
                               uint32_t rows, uint32_t cols, uint32_t column)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
        // NZ intermediates must initialize every aligned row subsequently read by
        // Cube, including the padding of a one-token tail.
        const uint32_t outputRows = DEST == Destination::L1 ? (rows + 15U) / 16U * 16U : rows;
        auto out = tla::GetTile(dst, tla::MakeCoord(0, column), tla::MakeShape(outputRows, cols));
        if constexpr (DEST == Destination::UB) {
            Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorC, decltype(out)>{}(out, c, false, 0);
        } else if constexpr (DEST == Destination::L1) {
            Catlass::Gemm::Tile::CopyL0CToL1Tla<ArchTag, TensorC, decltype(out)>{}(out, c, 0);
        } else {
            Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorC, decltype(out)>{}(out, c, 0);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cFree_[slot]);
    }

    template <typename TileCopy, Destination DEST, typename TensorA, typename TensorB, typename TensorDst>
    __aicore__ inline void Gemm(const TensorA &a, const TensorB &b, const TensorDst &dst,
                               uint32_t rows, uint32_t cols, uint32_t k)
    {
        const uint32_t m = rows == 1 ? 16 : rows;
        for (uint32_t column = 0; column < cols; column += CHUNK_SIZE_64) {
            const uint32_t n = Min(CHUNK_SIZE_64, cols - column);
            const uint32_t slot = cSerial_++ % SLOT_COUNT;
            auto c = tla::MakeTensor(
                resource_->l0CBuf.template GetBufferByByte<float>(slot * L0_SLOT_BYTES),
                tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
            Mmad<TileCopy>(a, b, c, slot, m, n, k, column, true);
            Emit<DEST>(dst, c, slot, rows, n, column);
        }
    }

    __aicore__ inline int64_t WorkspaceOffset(uint32_t r) const
    {
        return GetWorkspaceHeadOffset(core_, windowSerial_, r);
    }

    __aicore__ inline int64_t TokenOffset(const ChunkInfo &c, int64_t hv, uint32_t width) const
    {
        return ((c.bIdx * tiling_->HV + hv) * tiling_->T + c.tokenStart) * width;
    }

    __aicore__ inline int64_t QkOffset(const ChunkInfo &c, int64_t hk) const
    {
        return tiling_->layout == FinalizeLayout::BSND ?
            ((c.bIdx * tiling_->T + c.tokenStart) * tiling_->HK + hk) * K_SIZE_128 :
            ((c.bIdx * tiling_->HK + hk) * tiling_->T + c.tokenStart) * K_SIZE_128;
    }

    __aicore__ inline int64_t StateOffset(const ChunkInfo &c, int64_t hv) const
    {
        return ((c.bIdx * stateChunks_ + c.stateChunkIdx) * tiling_->HV + hv) * STATE_ELEMS;
    }

    __aicore__ inline void Stage1(const ChunkInfo &c, int64_t hv, uint32_t r)
    {
        auto du = L1View<ZN>(L1::DU, r, VECTOR_BYTES, CHUNK_SIZE_64, V_SIZE_128);
        auto vb = L1View<NZ>(L1::VB, r, VECTOR_BYTES, V_SIZE_128, CHUNK_SIZE_64);
        BeginInput(r);
        Load<RM>(du, du_, TokenOffset(c, hv, V_SIZE_128), c.chunkLen, V_SIZE_128, V_SIZE_128);
        if (tiling_->stateVFirst) {
            auto h = L1View<ZN>(L1::H, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Load<RM>(h, h_, StateOffset(c, hv), V_SIZE_128, K_SIZE_128, K_SIZE_128);
        } else {
            auto h = L1View<NZ>(L1::H, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Load<CM>(h, h_, StateOffset(c, hv), V_SIZE_128, K_SIZE_128, V_SIZE_128);
        }
        WaitVector();
        Load<CM>(vb, vb_, WorkspaceOffset(r), V_SIZE_128, c.chunkLen, V_SIZE_128);
        InputReady(r);
        Gemm<TileCopyDau, Destination::UB>(
            du, vb, UbOutput(UB::DAU, r, MATRIX_BYTES, CHUNK_SIZE_64), c.chunkLen, c.chunkLen, V_SIZE_128);
        NotifyVector();
        auto dw0 = L1View<ZN>(L1::DW0, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        if (tiling_->stateVFirst) {
            auto h = L1View<ZN>(L1::H, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Gemm<TileCopyStateVFirst, Destination::L1>(du, h, dw0, c.chunkLen, K_SIZE_128, V_SIZE_128);
        } else {
            auto h = L1View<NZ>(L1::H, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Gemm<TileCopyDau, Destination::L1>(du, h, dw0, c.chunkLen, K_SIZE_128, V_SIZE_128);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(l1ResultReady_[r]);
        EndInput(r);
    }

    __aicore__ inline void Stage3(const ChunkInfo &c, int64_t hv, uint32_t r)
    {
        auto a = L1View<ZN>(L1::A, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        auto kbg = L1View<NZ>(L1::KBG, r, VECTOR_BYTES, K_SIZE_128, CHUNK_SIZE_64);
        auto dw0 = L1View<ZN>(L1::DW0, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        BeginInput(r);
        Load<RM>(a, a_, TokenOffset(c, hv, CHUNK_SIZE_64), c.chunkLen, c.chunkLen, CHUNK_SIZE_64);
        Load<CM>(kbg, kbg_, WorkspaceOffset(r), K_SIZE_128, c.chunkLen, K_SIZE_128);
        InputReady(r);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(l1ResultReady_[r]);
        Gemm<TileCopyDau, Destination::UB>(dw0, kbg,
            UbOutput(UB::KBG, r, VECTOR_BYTES, CHUNK_SIZE_64), c.chunkLen, c.chunkLen, K_SIZE_128);
        auto at = L1View<NZ>(L1::A, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        auto du = L1View<ZN>(L1::DU, r, VECTOR_BYTES, CHUNK_SIZE_64, V_SIZE_128);
        Gemm<TileCopyDvb, Destination::UB>(at, du,
            UbOutput(UB::VB, r, VECTOR_BYTES, V_SIZE_128), c.chunkLen, V_SIZE_128, c.chunkLen);
        EndInput(r);
    }

    __aicore__ inline void Stage5(const ChunkInfo &c, int64_t hv, uint32_t r)
    {
        auto da = L1View<ZN>(L1::DA, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        auto k = L1View<ZN>(L1::DU, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        BeginInput(r);
        const int64_t leading = tiling_->layout == FinalizeLayout::BSND ? tiling_->HK * K_SIZE_128 : K_SIZE_128;
        Load<RM>(k, k_, QkOffset(c, hv / tiling_->headRatio), c.chunkLen, K_SIZE_128, leading);
        WaitVector();
        Load<RM>(da, dA_, WorkspaceOffset(r), c.chunkLen, c.chunkLen, CHUNK_SIZE_64);
        InputReady(r);
        auto at = L1View<NZ>(L1::A, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        Gemm<TileCopyDau, Destination::L1>(da, at, da, c.chunkLen, c.chunkLen, c.chunkLen);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(l1ResultReady_[r]);
        auto kt = L1View<NZ>(L1::DU, r, VECTOR_BYTES, K_SIZE_128, CHUNK_SIZE_64);
        Gemm<TileCopyDau, Destination::UB>(k, kt,
            UbOutput(UB::KBG, r, MATRIX_BYTES, CHUNK_SIZE_64), c.chunkLen, c.chunkLen, K_SIZE_128);
        EndInput(r);
    }

    __aicore__ inline void Stage7(const ChunkInfo &c, uint32_t r)
    {
        auto at = L1View<NZ>(L1::A, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        auto da1 = L1View<ZN>(L1::DA, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        auto dw0 = L1View<ZN>(L1::DW0, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(l1ResultReady_[r]);
        Gemm<TileCopyDvb, Destination::UB>(at, da1,
            UbOutput(UB::DA2, r, MATRIX_BYTES, CHUNK_SIZE_64), c.chunkLen, c.chunkLen, c.chunkLen);
        Gemm<TileCopyDvb, Destination::UB>(at, dw0,
            UbOutput(UB::DKBG0, r, VECTOR_BYTES, K_SIZE_128), c.chunkLen, K_SIZE_128, c.chunkLen);
    }

    __aicore__ inline void Stage9(const ChunkInfo &c, uint32_t r)
    {
        auto da = L1View<ZN>(L1::DA, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        auto kb = L1View<ZN>(L1::VB, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        BeginInput(r);
        WaitVector();
        Load<RM>(da, dA_, WorkspaceOffset(r), c.chunkLen, c.chunkLen, CHUNK_SIZE_64);
        Load<RM>(kb, kb_, WorkspaceOffset(r), c.chunkLen, K_SIZE_128, K_SIZE_128);
        InputReady(r);
        auto k = L1View<ZN>(L1::DU, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        Gemm<TileCopyDw0, Destination::GM>(da, k, GmOutput(dkb_, r), c.chunkLen, K_SIZE_128, c.chunkLen);
        auto dat = L1View<NZ>(L1::DA, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        Gemm<TileCopyStage9DkbT, Destination::GM>(dat, kb, GmOutput(dkbT_, r), c.chunkLen, K_SIZE_128, c.chunkLen);
        EndInput(r);
    }

    __aicore__ inline void Stage11(const ChunkInfo &c, int64_t hv, uint32_t r)
    {
        auto dO = L1View<ZN>(L1::A, r, VECTOR_BYTES, CHUNK_SIZE_64, V_SIZE_128);
        auto vn = L1View<NZ>(L1::KBG, r, VECTOR_BYTES, V_SIZE_128, CHUNK_SIZE_64);
        BeginInput(r);
        Load<RM>(dO, do_, TokenOffset(c, hv, V_SIZE_128), c.chunkLen, V_SIZE_128, V_SIZE_128);
        Load<CM>(vn, vNew_, TokenOffset(c, hv, V_SIZE_128), V_SIZE_128, c.chunkLen, V_SIZE_128);
        InputReady(r);
        Gemm<TileCopyDau, Destination::UB>(dO, vn,
            UbOutput(UB::DS, r, MATRIX_BYTES, CHUNK_SIZE_64), c.chunkLen, c.chunkLen, V_SIZE_128);
        // Keep this resident credit held until the last S13 MTE1 read. The next
        // window cannot overwrite h/k or any of their stage aliases early.
    }

    __aicore__ inline void Stage13(const ChunkInfo &c, int64_t hv, uint32_t r)
    {
        auto q = L1View<ZN>(L1::A, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        auto ds = L1View<ZN>(L1::DA, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        auto dog = L1View<ZN>(L1::KBG, r, VECTOR_BYTES, CHUNK_SIZE_64, V_SIZE_128);
        auto vd = L1View<ZN>(L1::VB, r, VECTOR_BYTES, CHUNK_SIZE_64, V_SIZE_128);
        BeginInput(HEAD_COUNT + r);
        const int64_t leading = tiling_->layout == FinalizeLayout::BSND ? tiling_->HK * K_SIZE_128 : K_SIZE_128;
        Load<RM>(q, q_, QkOffset(c, hv / tiling_->headRatio), c.chunkLen, K_SIZE_128, leading);
        Load<RM>(ds, ds_, WorkspaceOffset(r), c.chunkLen, c.chunkLen, CHUNK_SIZE_64);
        Load<RM>(dog, doG_, WorkspaceOffset(r), c.chunkLen, V_SIZE_128, V_SIZE_128);
        Load<RM>(vd, vDecay_, WorkspaceOffset(r), c.chunkLen, V_SIZE_128, V_SIZE_128);
        if (tiling_->stateVFirst) {
            auto dh = L1View<ZN>(L1::DH, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Load<RM>(dh, dh_, StateOffset(c, hv), V_SIZE_128, K_SIZE_128, K_SIZE_128);
        } else {
            auto dh = L1View<NZ>(L1::DH, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Load<CM>(dh, dh_, StateOffset(c, hv), V_SIZE_128, K_SIZE_128, V_SIZE_128);
        }
        InputReady(HEAD_COUNT + r);
        if (tiling_->stateVFirst) {
            auto h = L1View<ZN>(L1::H, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Dq<TileCopyStateVFirst>(c, r, dog, h, ds);
        } else {
            auto h = L1View<NZ>(L1::H, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Dq<TileCopyDau>(c, r, dog, h, ds);
        }
        auto dst = L1View<NZ>(L1::DA, r, MATRIX_BYTES, CHUNK_SIZE_64, CHUNK_SIZE_64);
        Gemm<TileCopyDvb, Destination::UB>(dst, q,
            UbOutput(UB::DK_INTRA, r, VECTOR_BYTES, K_SIZE_128), c.chunkLen, K_SIZE_128, c.chunkLen);
        if (tiling_->stateVFirst) {
            auto dh = L1View<ZN>(L1::DH, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Gemm<TileCopyStateVFirst, Destination::UB>(vd, dh,
                UbOutput(UB::DK_BASE, r, VECTOR_BYTES, K_SIZE_128), c.chunkLen, K_SIZE_128, V_SIZE_128);
        } else {
            auto dh = L1View<NZ>(L1::DH, r, STATE_BYTES, V_SIZE_128, K_SIZE_128);
            Gemm<TileCopyDau, Destination::UB>(vd, dh,
                UbOutput(UB::DK_BASE, r, VECTOR_BYTES, K_SIZE_128), c.chunkLen, K_SIZE_128, V_SIZE_128);
        }
        EndInput(HEAD_COUNT + r);
        EndInput(r);
    }

    template <typename StateCopy, typename TensorDoG, typename TensorH, typename TensorDs>
    __aicore__ inline void Dq(const ChunkInfo &c, uint32_t r, const TensorDoG &dog,
                             const TensorH &h, const TensorDs &ds)
    {
        auto k = L1View<ZN>(L1::DU, r, VECTOR_BYTES, CHUNK_SIZE_64, K_SIZE_128);
        auto dq = UbOutput(UB::DQ_HV, r, VECTOR_BYTES, K_SIZE_128);
        const uint32_t m = c.chunkLen == 1 ? 16 : c.chunkLen;
        for (uint32_t column = 0; column < K_SIZE_128; column += CHUNK_SIZE_64) {
            const uint32_t slot = cSerial_++ % SLOT_COUNT;
            auto l0c = tla::MakeTensor(
                resource_->l0CBuf.template GetBufferByByte<float>(slot * L0_SLOT_BYTES),
                tla::MakeLayoutL0C(m, CHUNK_SIZE_64), Catlass::Arch::PositionL0C{});
            Mmad<StateCopy>(dog, h, l0c, slot, m, CHUNK_SIZE_64, V_SIZE_128, column, true);
            // Accumulate both terms in the same FP32 L0C tile; round only once.
            Mmad<TileCopyDw0>(ds, k, l0c, slot, m, CHUNK_SIZE_64, c.chunkLen, column, false);
            Emit<Destination::UB>(dq, l0c, slot, c.chunkLen, CHUNK_SIZE_64, column);
        }
    }

    static_assert(SLOT_COUNT * L0_SLOT_BYTES <= ArchTag::L0A_SIZE);
    static_assert(SLOT_COUNT * L0_SLOT_BYTES <= ArchTag::L0B_SIZE);
    static_assert(SLOT_COUNT * L0_SLOT_BYTES <= ArchTag::L0C_SIZE);
    static_assert(UB_TOTAL_BYTES <= ArchTag::UB_SIZE);
    static_assert(L1_TOTAL_BYTES <= ArchTag::L1_SIZE);
    Resource *resource_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TEventID residentFree_[2 * HEAD_COUNT];
    AscendC::TEventID residentReady_[2 * HEAD_COUNT];
    AscendC::TEventID l1ResultReady_[HEAD_COUNT];
    AscendC::TEventID aFree_[SLOT_COUNT], bFree_[SLOT_COUNT];
    AscendC::TEventID cFree_[SLOT_COUNT], operandReady_[SLOT_COUNT], resultReady_[SLOT_COUNT];
    uint64_t aSerial_ = 0, bSerial_ = 0, cSerial_ = 0;
    int64_t core_ = 0, cores_ = 1, stateChunks_ = 0, windowSerial_ = 0;
    GM_ADDR q_, k_, vNew_, do_, du_, h_, dh_, a_;
    GM_ADDR kbg_, vb_, dA_, dkb_, dkbT_, ds_, doG_, vDecay_, kb_, cu_, indices_;
    const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling_ = nullptr;
};

} // namespace GDN::FinalizeArch92

#endif
