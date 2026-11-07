#ifndef CHUNK_KDA_BWD_PREPARE_ARCH22_CUBE_H
#define CHUNK_KDA_BWD_PREPARE_ARCH22_CUBE_H
#include "chunk_kda_bwd_prepare_common.h"
namespace KDA {
template <uint32_t INPUT_LAYOUT, bool STATE_V_FIRST>
class ChunkKdaBwdPrepareCube {
public:
    __aicore__ inline void Init(GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
        GM_ADDR cu, GM_ADDR indices, GM_ADDR dv, GM_ADDR dq, GM_ADDR scratch,
        const ChunkKdaBwdPrepareTilingData *tiling, AscendC::TPipe *pipe)
    {
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aqk));
        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(vNew));
        dO_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dO));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq));
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch));
        // Match arch35's streaming policy: each input tile is loaded once
        // from GM and reused in L1. Keep the small GM handoff cached.
        aqk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        vNew_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dO_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        h_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dv_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dq_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        cu_ = cu; indices_ = indices; t_ = tiling; pipe_ = pipe;
        pipe_->InitBuffer(l1_, 4 * HEAD_OWNER_BYTES);
        pipe_->InitBuffer(l0a_, 2 * L0_SLOT_BYTES);
        pipe_->InitBuffer(l0b_, 2 * L0_SLOT_BYTES);
        pipe_->InitBuffer(l0c_, 2 * L0_SLOT_BYTES);
        AllocateEvents<AscendC::HardEvent::MTE1_MTE2>(pipe_, ownerFree_, 4, true);
        AllocateEvents<AscendC::HardEvent::MTE2_MTE1>(pipe_, pairReady_, 6, false);
        AllocateEvents<AscendC::HardEvent::M_MTE1>(pipe_, aFree_, 2, true);
        AllocateEvents<AscendC::HardEvent::M_MTE1>(pipe_, bFree_, 2, true);
        AllocateEvents<AscendC::HardEvent::MTE1_M>(pipe_, operandReady_, 2, false);
        AllocateEvents<AscendC::HardEvent::FIX_M>(pipe_, cFree_, 2, true);
        AllocateEvents<AscendC::HardEvent::M_FIX>(pipe_, resultReady_, 2, false);
    }
    __aicore__ inline void Process()
    {
        const int64_t block = AscendC::GetBlockIdx(), blocks = AscendC::GetBlockNum();
        uint64_t generation = 0, formula = 0;
        for (int64_t work = block; work < t_->workTaskNum; work += blocks) {
            const int64_t window = work / t_->chunkTaskNum;
            const int64_t headBegin = window * 4, headEnd = KdaMin(headBegin + 4, t_->NV);
            ChunkInfo chunk;
            ResolveChunk(work % t_->chunkTaskNum, cu_, indices_, *t_, chunk);
            if (!chunk.valid) continue;
            const uint32_t rows = chunk.validRows;
            Preload(chunk, headBegin, headEnd, rows);
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t owner = head - headBegin, pair = owner / 2;
                const bool leader = (owner % 2) == 0;
                const uint32_t record = 2 * (generation & 1) + ((generation >> 1) & 1);
                const uint32_t aSlot = formula++ & 1;
                Compute<0>(owner, leader, pair * 3, rows, rows, 128, aSlot, aSlot);
                AscendC::CrossCoreWaitFlag<CROSS_MODE, PIPE_FIX>(FREE_BASE + record);
                CopyOut(raw_[(block * 4 + record) * RECORD_ELEMENTS], rows, rows, 64, aSlot);
                AscendC::CrossCoreSetFlag<CROSS_MODE, PIPE_FIX>(READY_BASE + record);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cFree_[aSlot]);
                const uint32_t qSlot = formula++ & 1;
                Compute<1>(owner, leader, pair * 3 + 1, rows, 128, 128, qSlot, aSlot);
                CopyOut(dq_[TokenOutputOffset(*t_, chunk, head, 128)], rows, 128, 128, qSlot);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cFree_[qSlot]);
                const uint32_t dSlot = formula++ & 1;
                Compute<2>(owner, leader, pair * 3 + 2, rows, 128, rows, dSlot, dSlot);
                CopyOut(dv_[TokenOutputOffset(*t_, chunk, head, 128)], rows, 128, 128, dSlot);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cFree_[dSlot]);
            }
        }
        for (uint16_t record = 0; record < 4; ++record)
            AscendC::CrossCoreWaitFlag<CROSS_MODE, PIPE_FIX>(FREE_BASE + record);
        ReleaseEvents<AscendC::HardEvent::MTE1_MTE2>(pipe_, ownerFree_, 4, true);
        ReleaseEvents<AscendC::HardEvent::MTE2_MTE1>(pipe_, pairReady_, 6, false);
        ReleaseEvents<AscendC::HardEvent::M_MTE1>(pipe_, aFree_, 2, true);
        ReleaseEvents<AscendC::HardEvent::M_MTE1>(pipe_, bFree_, 2, true);
        ReleaseEvents<AscendC::HardEvent::MTE1_M>(pipe_, operandReady_, 2, false);
        ReleaseEvents<AscendC::HardEvent::FIX_M>(pipe_, cFree_, 2, true);
        ReleaseEvents<AscendC::HardEvent::M_FIX>(pipe_, resultReady_, 2, false);
    }
private:
    static constexpr bool HEAD_CONTIGUOUS = INPUT_LAYOUT == 1 || INPUT_LAYOUT == 3;
    __aicore__ inline AscendC::LocalTensor<bfloat16_t> Owner(uint32_t owner, uint32_t offset)
    {
        if constexpr (HEAD_CONTIGUOUS) {
            // Pair-local SoA: dO[2][16 KiB], v[2][16 KiB], H[2][32 KiB],
            // Aqk[2][8 KiB]. Adjacent token heads form one wide ND matrix;
            // its NZ column blocks remain independent per-head views.
            const uint32_t headBytes = offset == 32 * 1024 ? 32 * 1024
                : (offset == 64 * 1024 ? 8 * 1024 : 16 * 1024);
            return l1_.Get<bfloat16_t>()[((owner / 2) * 2 * HEAD_OWNER_BYTES
                + 2 * offset + (owner % 2) * headBytes) / sizeof(bfloat16_t)];
        }
        return l1_.Get<bfloat16_t>()[(owner * HEAD_OWNER_BYTES + offset) / sizeof(bfloat16_t)];
    }
    __aicore__ inline void Load(AscendC::LocalTensor<bfloat16_t> dst,
        AscendC::GlobalTensor<bfloat16_t> src, uint32_t rows, uint32_t cols,
        int64_t stride, uint32_t pitch, uint32_t headCount, int64_t headStride,
        uint32_t dstHeadStride = HEAD_OWNER_BYTES / sizeof(bfloat16_t))
    {
        // ND2NZ writes logical rows, but LoadData2D reads whole 16x16
        // fractals. Initialize the missing rows in every NZ column block;
        // consecutive blocks are separated by the fixed L1 row pitch.
        const uint32_t paddingRows = Align16(rows) - rows;
        if (paddingRows != 0) {
            AscendC::InitConstValueParams<uint16_t> padding(
                static_cast<uint16_t>(Align16(cols) / 16),
                static_cast<uint16_t>(paddingRows),
                static_cast<uint16_t>(pitch - paddingRows),
                0);
            auto paddingView = dst.template ReinterpretCast<uint16_t>();
            for (uint32_t head = 0; head < headCount; ++head)
                AscendC::InitConstValue(paddingView[head * dstHeadStride + rows * 16], padding);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::Nd2NzParams p;
        p.ndNum = 1; p.nValue = rows; p.dValue = cols;
        p.srcNdMatrixStride = 0; p.dstNzMatrixStride = 0;
        p.dstNzC0Stride = pitch; p.dstNzNStride = 1;
        if (stride <= 65535) {
            p.srcDValue = static_cast<uint16_t>(stride);
            // A batched copy keeps independent NZ matrices. Token-major
            // full-width inputs use a wider single matrix in LoadToken.
            if (headCount == 2 && headStride <= 65535) {
                p.ndNum = 2;
                p.srcNdMatrixStride = static_cast<uint16_t>(headStride);
                p.dstNzMatrixStride = dstHeadStride;
                AscendC::DataCopy(dst, src, p);
            } else {
                for (uint32_t head = 0; head < headCount; ++head)
                    AscendC::DataCopy(dst[head * dstHeadStride],
                                     src[head * headStride], p);
            }
        } else {
            // Large BSND/TND head counts cannot encode the token row stride.
            // Retain the row copy fallback without truncating either stride.
            p.nValue = 1; p.srcDValue = 0;
            for (uint32_t head = 0; head < headCount; ++head)
                for (uint32_t row = 0; row < rows; ++row)
                    AscendC::DataCopy(dst[head * dstHeadStride + row * 16],
                                     src[head * headStride + row * stride], p);
        }
    }
    template <uint32_t WIDTH>
    __aicore__ inline void LoadToken(AscendC::LocalTensor<bfloat16_t> dst,
        AscendC::GlobalTensor<bfloat16_t> src, const ChunkInfo &chunk, int64_t head,
        uint32_t rows, uint32_t cols, uint32_t count)
    {
        const int64_t off = TokenInputOffset<INPUT_LAYOUT>(*t_, chunk, head, WIDTH);
        const int64_t stride = TokenInputRowStride<INPUT_LAYOUT>(*t_, WIDTH);
        if constexpr (HEAD_CONTIGUOUS) {
            if (cols == WIDTH) {
                // No padding or extra token rows are read. Tail Aqk keeps
                // its original valid square via the per-head path below.
                Load(dst, src[off], rows, WIDTH * count, stride, 64, 1, 0);
                return;
            }
        }
        const int64_t headStride = count == 2
            ? TokenInputOffset<INPUT_LAYOUT>(*t_, chunk, head + 1, WIDTH) - off : 0;
        constexpr uint32_t dstStride = HEAD_CONTIGUOUS ? 64 * WIDTH
            : HEAD_OWNER_BYTES / sizeof(bfloat16_t);
        Load(dst, src[off], rows, cols, stride, 64, count, headStride, dstStride);
    }
    __aicore__ inline void Preload(const ChunkInfo &chunk, int64_t begin, int64_t end, uint32_t rows)
    {
        for (uint32_t pair = 0; begin + 2 * pair < end; ++pair) {
            const uint32_t first = pair * 2, count = KdaMin(2, end - begin - first);
            for (uint32_t i = first; i < first + count; ++i)
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ownerFree_[i]);
            const int64_t head = begin + first;
            LoadToken<128>(Owner(first, 0), dO_, chunk, head, rows, 128, count);
            LoadToken<128>(Owner(first, 16 * 1024), vNew_, chunk, head, rows, 128, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pairReady_[pair * 3]);
            Load(Owner(first, 32 * 1024), h_[StateOffset(*t_, chunk, head)],
                 128, 128, 128, 128, count, 128 * 128,
                 HEAD_CONTIGUOUS ? 128 * 128 : HEAD_OWNER_BYTES / sizeof(bfloat16_t));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pairReady_[pair * 3 + 1]);
            LoadToken<64>(Owner(first, 64 * 1024), aqk_, chunk, head, rows, rows, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pairReady_[pair * 3 + 2]);
        }
    }
    // L1 NZ has a fixed physical row pitch. L0A is ZZ; L0B is ZN.
    template <bool TRANSPOSE>
    __aicore__ inline void LoadA(AscendC::LocalTensor<bfloat16_t> dst,
        AscendC::LocalTensor<bfloat16_t> src, uint32_t m, uint32_t k, uint32_t pitch)
    {
        AscendC::LoadData2DParams p;
        p.ifTranspose = TRANSPOSE;
        const uint32_t mb = Align16(m) / 16, kb = Align16(k) / 16;
        p.repeatTimes = static_cast<uint8_t>(kb);
        p.srcStride = TRANSPOSE ? 1 : pitch / 16;
        // One instruction emits the complete K strip into consecutive L0A
        // fractals, rather than issuing one instruction per 16x16 fractal.
        for (uint32_t row = 0; row < mb; ++row) {
            const uint32_t off = TRANSPOSE ? row * pitch * 16 : row * 256;
            AscendC::LoadData(dst[row * kb * 256], src[off], p);
        }
    }
    template <bool TRANSPOSE>
    __aicore__ inline void LoadB(AscendC::LocalTensor<bfloat16_t> dst,
        AscendC::LocalTensor<bfloat16_t> src, uint32_t k, uint32_t n, uint32_t pitch)
    {
        AscendC::LoadData2DParams p;
        p.ifTranspose = !TRANSPOSE;
        const uint32_t nb = Align16(n) / 16, kb = Align16(k) / 16;
        if constexpr (TRANSPOSE) {
            if (pitch / 16 == nb) {
                // The complete NZ tile already has the required ZN order.
                // Flatten its K/N fractals into one contiguous instruction.
                p.repeatTimes = static_cast<uint8_t>(kb * nb);
                p.srcStride = 1;
                AscendC::LoadData(dst, src, p);
                return;
            }
        }
        p.repeatTimes = static_cast<uint8_t>(kb);
        p.srcStride = TRANSPOSE ? pitch / 16 : 1;
        p.dstGap = static_cast<uint16_t>(nb - 1);
        // Interleave each K strip with dstGap so the resulting ZN layout is
        // identical to the former nested loop, including transposed H.
        for (uint32_t col = 0; col < nb; ++col) {
            const uint32_t off = TRANSPOSE ? col * 256 : col * pitch * 16;
            AscendC::LoadData(dst[col * 256], src[off], p);
        }
    }
    template <uint32_t FORMULA>
    __aicore__ inline void Compute(uint32_t owner, bool leader, uint32_t ready,
        uint32_t m, uint32_t n, uint32_t k, uint32_t slot, uint32_t aSlot)
    {
        if (leader) AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pairReady_[ready]);
        auto a = l0a_.Get<bfloat16_t>()[aSlot * L0_SLOT_BYTES / 2];
        auto b = l0b_.Get<bfloat16_t>()[slot * L0_SLOT_BYTES / 2];
        auto c = l0c_.Get<float>()[slot * L0_SLOT_BYTES / 4];
        if constexpr (FORMULA != 1) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(aFree_[aSlot]);
            LoadA<FORMULA == 2>(a, Owner(owner, FORMULA == 2 ? 64 * 1024 : 0), m, k, 64);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bFree_[slot]);
        if constexpr (FORMULA == 0) LoadB<true>(b, Owner(owner, 16 * 1024), k, n, 64);
        if constexpr (FORMULA == 1) LoadB<!STATE_V_FIRST>(b, Owner(owner, 32 * 1024), k, n, 128);
        if constexpr (FORMULA == 2) LoadB<false>(b, Owner(owner, 0), k, n, 64);
        if constexpr (FORMULA == 2) AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ownerFree_[owner]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cFree_[slot]);
        AscendC::MmadParams p;
        p.m = Align16(m); p.n = n; p.k = k; p.cmatrixInitVal = true; p.unitFlag = 0;
        AscendC::Mmad(c, a, b, p);
        if constexpr (FORMULA != 0) AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(aFree_[aSlot]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bFree_[slot]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
    }
    template <typename T>
    __aicore__ inline void CopyOut(AscendC::GlobalTensor<T> dst,
        uint32_t m, uint32_t n, uint32_t stride, uint32_t slot)
    {
        auto c = l0c_.Get<float>()[slot * L0_SLOT_BYTES / 4];
        AscendC::FixpipeParamsV220 p(n, m, Align16(m), stride, false);
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) p.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(dst, c, p);
    }
    AscendC::TPipe *pipe_;
    const ChunkKdaBwdPrepareTilingData *t_;
    GM_ADDR cu_;
    GM_ADDR indices_;
    AscendC::GlobalTensor<bfloat16_t> aqk_, vNew_, dO_, h_, dv_;
    AscendC::GlobalTensor<float> dq_, raw_;
    AscendC::TBuf<AscendC::TPosition::A1> l1_;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_;
    AscendC::TEventID ownerFree_[4], pairReady_[6], aFree_[2], bFree_[2];
    AscendC::TEventID operandReady_[2], cFree_[2], resultReady_[2];
};
}
#endif
