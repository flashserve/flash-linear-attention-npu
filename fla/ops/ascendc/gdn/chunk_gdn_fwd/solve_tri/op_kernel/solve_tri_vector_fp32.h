#ifndef SOLVE_TRI_VECTOR_FP32_H
#define SOLVE_TRI_VECTOR_FP32_H
#include <type_traits>
#include "kernel_operator.h"

namespace NsSolveTri {
using namespace AscendC;

// Correctness-first forward substitution. Tiles are independent across AIVs.
template <typename T, int N>
class SolveTriVectorFullFp32 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cu, GM_ADDR indices, GM_ADDR out,
                               const SolveTriTilingData* tiling)
    {
        td_ = tiling;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        cu_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cu));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));
        pipe_.InitBuffer(aBuf_, N * N * sizeof(float));
        pipe_.InitBuffer(invBuf_, N * N * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, N * sizeof(float));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(ioBuf_, N * N * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        auto a = aBuf_.Get<float>();
        auto inv = invBuf_.Get<float>();
        auto tmp = tmpBuf_.Get<float>();
        LocalTensor<T> io;
        if constexpr (std::is_same_v<T, float>) {
            io = a;
        } else {
            io = ioBuf_.Get<T>();
        }
        const int64_t workers = GetBlockNum() * GetSubBlockNum();
        const int64_t worker = GetBlockIdx();
        for (int64_t tile = worker; tile < td_->totalTiles; tile += workers) {
            int64_t offset, valid, stride;
            Locate(tile, offset, valid, stride);
            Duplicate(io, T(0), N * N);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            const uint32_t bytes = static_cast<uint32_t>(valid * sizeof(T));
            const uint32_t aligned = (bytes + 31U) & ~31U;
            constexpr uint32_t localRowBytes = static_cast<uint32_t>(N * sizeof(T));
            const uint32_t localRowGap = (localRowBytes - aligned) / 32U;
            DataCopyExtParams load{static_cast<uint16_t>(valid), bytes,
                static_cast<uint32_t>((stride - valid) * sizeof(T)),
                localRowGap, 0};
            DataCopyPad(io, x_[offset], load, DataCopyPadExtParams<T>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if constexpr (!std::is_same_v<T, float>) {
                Cast(a, io, RoundMode::CAST_NONE, N * N);
            }
            Duplicate(inv, 0.0f, N * N);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            for (int32_t row = 0; row < valid; ++row) {
                inv.SetValue(row * N + row, 1.0f);
            }
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            // X_i = e_i - sum_{k<i} A_ik X_k; diagonal(A) is zero.
            for (int32_t row = 0; row < valid; ++row) {
                for (int32_t k = 0; k < row; ++k) {
                    const float coefficient = a.GetValue(row * N + k);
                    Muls(tmp, inv[k * N], -coefficient, N);
                    PipeBarrier<PIPE_V>();
                    Add(inv[row * N], inv[row * N], tmp, N);
                    PipeBarrier<PIPE_V>();
                }
            }
            if constexpr (std::is_same_v<T, float>) {
                io = inv;
            } else {
                Cast(io, inv, RoundMode::CAST_RINT, N * N);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams store{static_cast<uint16_t>(valid), bytes,
                localRowGap,
                static_cast<uint32_t>((stride - valid) * sizeof(T)), 0};
            DataCopyPad(out_[offset], io, store);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            if constexpr (std::is_same_v<T, float>) { io = a; }
        }
    }
private:
    __aicore__ inline void Locate(int64_t tile, int64_t& offset, int64_t& valid, int64_t& stride)
    {
        const int64_t hCount = td_->numHeads;
        const int64_t mode = td_->layoutMode;
        stride = (mode == 0 || mode == 3) ? N : hCount * N;
        if (mode == 2 || mode == 3) {
            const int64_t chunk = tile / hCount, h = tile % hCount;
            const int64_t seq = indices_.GetValue(2 * chunk);
            const int64_t start = cu_.GetValue(seq) + indices_.GetValue(2 * chunk + 1) * N;
            const int64_t remaining = cu_.GetValue(seq + 1) - start;
            valid = remaining < N ? remaining : N;
            offset = mode == 2 ? start * hCount * N + h * N : (h * td_->seqLen + start) * N;
        } else {
            const int64_t chunk = mode == 0 ? tile % td_->numChunks : (tile / hCount) % td_->numChunks;
            const int64_t h = mode == 0 ? (tile / td_->numChunks) % hCount : tile % hCount;
            const int64_t b = tile / (hCount * td_->numChunks);
            valid = chunk == td_->numChunks - 1 ? td_->lastChunkValidSize : N;
            offset = mode == 0 ? ((b * hCount + h) * td_->seqLen + chunk * N) * N
                               : ((b * td_->seqLen + chunk * N) * hCount + h) * N;
        }
    }
    const SolveTriTilingData* td_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> aBuf_, invBuf_, tmpBuf_, ioBuf_;
    GlobalTensor<T> x_, out_;
    GlobalTensor<int64_t> cu_, indices_;
};
}
#endif
