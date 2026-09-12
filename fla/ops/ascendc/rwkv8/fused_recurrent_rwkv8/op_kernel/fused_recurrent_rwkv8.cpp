// fused_recurrent_rwkv8 AscendC kernel（纯向量核，AIV only）
//
// 语义锚点：RWKV-LM/RWKV-v8/cuda/wkv7_cuda.cu forward_kernel (lines 10-52)
//   sa    = state @ z_t
//   state = state * decay_t[None,:] + sa[:,None] * b_t[None,:] + v_t[:,None] * k_t[None,:]
//   o_t   = state @ (q_t * scale)        decay = exp(-exp(w))
//
// 实现要点：state 以转置 S^T 驻留 UB（K 行 × V 列，行 j 连续 = k/z 通道 j 的完整 V 段），
// 使递推全部化为 标量×向量 的 Axpy/Muls：
//   sa = Σ_j z_j · S^T_row_j                    （长度 V）
//   S^T_row_j = S^T_row_j·decay_j + sa·b_j + v·k_j
//   o  = Σ_j q'_j · S^T_row_j（更新后）          （长度 V）
// K（q/w/k/z/b/decay 侧）与 V（v/o/sa 侧）独立，K==V 为特例。
// io GM 布局定档 BHTC = (B,H,T,C)（2026-08-17，H 在 T 前，每核 (b,h) 段连续、
// token 步长 = C；与 fla DPLR 的 BTHC 口径不同，对接 fla 时需 transpose(1,2)）。
// grid = B*H，每核一个 (b,h) 账本的完整 (K,V) state；内部递推全程 fp32。
// K,V 均不切分（sa/o 对 K 求和要求每核拿全 K；V 定档 ≤128 后 K×V 整体可入单核 UB）。
// initial_state 为 (K,V) 朝向（= 内部账本 Sᵀ 原样，与 s 快照 / fla 一致）且恒 fp32，
// 有初态时逐行载入 stateBuf_（行距 V，逐行避免 DataCopyPad blockLen 超 uint16），零转置。
//
// 多 dtype（混合精度，对齐 fla fused_recurrent 口径）：
//   io（q/w/k/v/z/b/o）支持 fp16/bf16/fp32，由 OPP build 注入 -DDTYPE_Q=half/bfloat16_t/float
//   编译期确定（一 dtype 一 variant）；输入读入 UB 后 Cast 到 fp32 再递推，o 算出后
//   Cast 回 io dtype 写回；initial_state 与递推主体永远 fp32。
//
// 训练预埋（对齐官方 wkv7_cuda.cu 三输出）：
//   sa（可选，flag 门控）：每 token 的 state@z（更新前），逐 token DataCopyPad 写出；
//   s（可选，flag 门控）：每满 chunkLen 个 token 把 UB 的 Sᵀ (K,V) 整块写盘——Sᵀ 布局即官方
//     s_ 的转置布局，零转置成本；槽位 t/chunkLen，T 非 chunkLen 倍数时尾部不满的一段不拍
//     （chunkLen 默认 16 对齐官方 backward 重建粒度，host attr 下发，kernel 不存常量）；
//   reverse（flag）：T 维倒序递推（对齐 fla reverse），initial_state 种子在
//     t=T-1 侧。

#include "kernel_operator.h"
#include "fused_recurrent_rwkv8_tiling_data.h"

// 仓外 KernelLaunch 直调工程无 OPP 宏注入，退化为 fp32
#ifndef DTYPE_Q
#define DTYPE_Q float
#endif

using namespace AscendC;
using namespace FusedRecurrentRwkv8;

namespace {
// device 侧避免 STL 的极简 is_same
template <typename A, typename B>
struct IsSame {
    static constexpr bool value = false;
};
template <typename A>
struct IsSame<A, A> {
    static constexpr bool value = true;
};
} // namespace

template <typename T>
class KernelFusedRecurrentRwkv8 {
public:
    __aicore__ inline KernelFusedRecurrentRwkv8(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR z, GM_ADDR b,
                                GM_ADDR initialState, GM_ADDR o, GM_ADDR s, GM_ADDR sa,
                                const FusedRecurrentRwkv8TilingData* tiling)
    {
        B_ = tiling->B;
        T_ = tiling->T;
        H_ = tiling->H;
        K_ = tiling->K;
        V_ = tiling->V;
        scale_ = tiling->scale;
        hasInit_ = tiling->hasInitialState;
        reverse_ = tiling->reverse;
        outS_ = tiling->outputChunkState;
        outSa_ = tiling->outputSa;
        chunkLen_ = tiling->chunkLen;   // s 快照间隔（host attr 下发，kernel 不存常量）

        // 防御：tiling 字段合理性校验。tiling 的 GM 读若被扰动（脏数据/竞争），
        // 异常字段会把循环边界/地址计算带飞——此时本核静默退出，宁可不写也不越界
        sane_ = (T_ > 0 && T_ <= (1u << 20) && K_ > 0 && K_ <= 256 && V_ > 0 && V_ <= 256 &&
                 (uint64_t)K_ * V_ <= 16384) ? 1 : 0;
        if (!sane_) {
            return;
        }
        // 本核 s 快照槽位数（chunkLen_ == 0 时归 0，顺带免除 Process 里的除零风险）
        sSlots_ = (chunkLen_ > 0) ? (T_ / chunkLen_) : 0;

        uint32_t bh = GetBlockIdx();           // 一个核负责一个 (b, h)，扁平下标 = b*H + h

        // (B,H,T,K)/(B,H,T,V) 连续布局（BHTC）：每核 (b,h) 段连续，token t 步长 = dim
        uint64_t baseK = (uint64_t)bh * T_ * K_;
        uint64_t baseV = (uint64_t)bh * T_ * V_;
        seqLenK_ = (uint64_t)T_ * K_;              // K 侧（q/w/k/z/b）本核 GM 跨度
        seqLenV_ = (uint64_t)T_ * V_;              // V 侧（v/o/sa）本核 GM 跨度
        qGm_.SetGlobalBuffer((__gm__ T*)q + baseK, seqLenK_);
        wGm_.SetGlobalBuffer((__gm__ T*)w + baseK, seqLenK_);
        kGm_.SetGlobalBuffer((__gm__ T*)k + baseK, seqLenK_);
        zGm_.SetGlobalBuffer((__gm__ T*)z + baseK, seqLenK_);
        bGm_.SetGlobalBuffer((__gm__ T*)b + baseK, seqLenK_);
        vGm_.SetGlobalBuffer((__gm__ T*)v + baseV, seqLenV_);
        oGm_.SetGlobalBuffer((__gm__ T*)o + baseV, seqLenV_);
        if (hasInit_) {
            // init: (B,H,K,V) 布局，该 (b,h) 的完整 (K,V) 账本，逐行读入（行距 V）
            initGm_.SetGlobalBuffer((__gm__ float*)initialState + (uint64_t)bh * V_ * K_,
                                    V_ * K_);
        }
        if (outS_) {
            // s: (B,H,T//chunkLen,K,V)，该 (b,h) 的基址，逐行写出（行距 V）
            sGm_.SetGlobalBuffer((__gm__ float*)s + (uint64_t)bh * (T_ / chunkLen_) * K_ * V_,
                                 (T_ / chunkLen_) * K_ * V_);
        }
        if (outSa_) {
            saGm_.SetGlobalBuffer((__gm__ float*)sa + baseV, seqLenV_);   // 布局同 v/o
        }

        pipe_->InitBuffer(stateBuf_, K_ * V_ * sizeof(float));   // S^T (K 行 × V 列)
        // K 侧向量（q/w/k/z/b/decay/e/qs）
        pipe_->InitBuffer(qBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(wBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(kBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(zBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(bBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(decayBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(eBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(qsBuf_, K_ * sizeof(float));
        // V 侧向量（v/sa/o）
        pipe_->InitBuffer(vBuf_, V_ * sizeof(float));
        pipe_->InitBuffer(saBuf_, V_ * sizeof(float));
        pipe_->InitBuffer(oBuf_, V_ * sizeof(float));
        if constexpr (!IsSame<T, float>::value) {
            // 低精度 staging：GM→UB 的原生 dtype 落点，再 Cast 成 fp32 进递推
            pipe_->InitBuffer(qStBuf_, K_ * sizeof(T));
            pipe_->InitBuffer(wStBuf_, K_ * sizeof(T));
            pipe_->InitBuffer(kStBuf_, K_ * sizeof(T));
            pipe_->InitBuffer(zStBuf_, K_ * sizeof(T));
            pipe_->InitBuffer(bStBuf_, K_ * sizeof(T));
            pipe_->InitBuffer(vStBuf_, V_ * sizeof(T));
            pipe_->InitBuffer(oStBuf_, V_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (!sane_) {
            return;   // tiling 异常：本核零读写（UB/GM 均未触碰）
        }
        LocalTensor<float> state = stateBuf_.Get<float>();   // S^T (K,V)

        // 初始状态：GM (K,V) 布局，逐行读入（行距 V；逐行而非整块是为避开
        // DataCopyPad blockLen uint16 上限：K=V=128 时整块 65536B 会截断）
        if (hasInit_) {
            for (uint32_t j = 0; j < K_; j++) {
                DataCopyPad(state[j * V_], initGm_[(uint64_t)j * V_],
                    {1, static_cast<uint16_t>(V_ * sizeof(float)), 0, 0},
                    {false, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            Duplicate(state, 0.0f, K_ * V_);
        }

        LocalTensor<float> qL = qBuf_.Get<float>();
        LocalTensor<float> wL = wBuf_.Get<float>();
        LocalTensor<float> kL = kBuf_.Get<float>();
        LocalTensor<float> vL = vBuf_.Get<float>();
        LocalTensor<float> zL = zBuf_.Get<float>();
        LocalTensor<float> bL = bBuf_.Get<float>();
        LocalTensor<float> decayL = decayBuf_.Get<float>();
        LocalTensor<float> eL = eBuf_.Get<float>();
        LocalTensor<float> qsL = qsBuf_.Get<float>();
        LocalTensor<float> saL = saBuf_.Get<float>();
        LocalTensor<float> oL = oBuf_.Get<float>();

        const uint32_t copyBytesK = K_ * sizeof(T);
        const uint32_t copyBytesV = V_ * sizeof(T);
        for (uint32_t i = 0; i < T_; i++) {
            uint32_t t = reverse_ ? (T_ - 1 - i) : i;   // reverse：倒序递推
            uint64_t offK = (uint64_t)t * K_;
            uint64_t offV = (uint64_t)t * V_;
            if constexpr (IsSame<T, float>::value) {
                DataCopyPad(qL, qGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(wL, wGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(kL, kGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(zL, zGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(bL, bGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(vL, vGm_[offV], {1, static_cast<uint16_t>(copyBytesV), 0, 0}, {false, 0, 0, 0});
            } else {
                LocalTensor<T> qSt = qStBuf_.Get<T>();
                LocalTensor<T> wSt = wStBuf_.Get<T>();
                LocalTensor<T> kSt = kStBuf_.Get<T>();
                LocalTensor<T> zSt = zStBuf_.Get<T>();
                LocalTensor<T> bSt = bStBuf_.Get<T>();
                LocalTensor<T> vSt = vStBuf_.Get<T>();
                DataCopyPad(qSt, qGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(wSt, wGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(kSt, kGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(zSt, zGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(bSt, bGm_[offK], {1, static_cast<uint16_t>(copyBytesK), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(vSt, vGm_[offV], {1, static_cast<uint16_t>(copyBytesV), 0, 0}, {false, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();   // MTE2 → V：等搬入完成再 Cast
                Cast(qL, qSt, RoundMode::CAST_NONE, K_);
                Cast(wL, wSt, RoundMode::CAST_NONE, K_);
                Cast(kL, kSt, RoundMode::CAST_NONE, K_);
                Cast(zL, zSt, RoundMode::CAST_NONE, K_);
                Cast(bL, bSt, RoundMode::CAST_NONE, K_);
                Cast(vL, vSt, RoundMode::CAST_NONE, V_);
            }
            PipeBarrier<PIPE_ALL>();

            // decay = exp(-exp(w))；q 预乘 scale
            Exp(eL, wL, K_);
            Muls(eL, eL, -1.0f, K_);
            Exp(decayL, eL, K_);
            Muls(qsL, qL, scale_, K_);
            PipeBarrier<PIPE_ALL>();

            // sa = Σ_j z_j · S^T_row_j（长度 V，行宽 V）
            Duplicate(saL, 0.0f, V_);
            for (uint32_t j = 0; j < K_; j++) {
                Axpy(saL, state[j * V_], zL.GetValue(j), V_);
            }

            // 逐行：S^T_row_j = row·decay_j + sa·b_j + v·k_j；o += q'_j·row(更新后)
            Duplicate(oL, 0.0f, V_);
            for (uint32_t j = 0; j < K_; j++) {
                LocalTensor<float> row = state[j * V_];
                float dj = decayL.GetValue(j);
                float bj = bL.GetValue(j);
                float kj = kL.GetValue(j);
                float qj = qsL.GetValue(j);
                Muls(row, row, dj, V_);
                Axpy(row, saL, bj, V_);
                Axpy(row, vL, kj, V_);
                Axpy(oL, row, qj, V_);
            }
            PipeBarrier<PIPE_ALL>();

            // 训练预埋写出（flag 门控，默认全跳过硬零开销）
            if (outSa_) {
                DataCopyPad(saGm_[offV], saL, {1, static_cast<uint16_t>(V_ * sizeof(float)), 0, 0});
            }
            if (outS_ && sSlots_ > 0 && (t + 1) % chunkLen_ == 0) {
                // s 槽位内 GM 是 (K,V) 布局，逐行写出（行距 V，同 init 读的 uint16 考虑）
                // 防御：逐行核对本核 s 区段边界（sExtent = sSlots_·K·V），rowOff 单调
                // 递增，越段即 break——slotBase 被异常数据带飞时也不会扫写出 s 区段
                const uint64_t sExtent = (uint64_t)sSlots_ * K_ * V_;
                uint64_t slotBase = (uint64_t)(t / chunkLen_) * K_ * V_;
                for (uint32_t j = 0; j < K_; j++) {
                    uint64_t rowOff = slotBase + (uint64_t)j * V_;
                    if (rowOff + V_ > sExtent) {
                        break;
                    }
                    DataCopyPad(sGm_[rowOff], state[j * V_],
                            {1, static_cast<uint16_t>(V_ * sizeof(float)), 0, 0});
                }
            }

            if constexpr (IsSame<T, float>::value) {
                DataCopyPad(oGm_[offV], oL, {1, static_cast<uint16_t>(copyBytesV), 0, 0});
            } else {
                LocalTensor<T> oSt = oStBuf_.Get<T>();
                Cast(oSt, oL, RoundMode::CAST_RINT, V_);
                PipeBarrier<PIPE_ALL>();   // V → MTE3：等 Cast 完成再写回
                DataCopyPad(oGm_[offV], oSt, {1, static_cast<uint16_t>(copyBytesV), 0, 0});
            }
            PipeBarrier<PIPE_ALL>();   // oL 复用前等待 MTE3 完成
        }
    }

private:
    TPipe* pipe_;
    uint32_t B_, T_, H_, K_, V_;
    float scale_;
    uint32_t hasInit_;
    uint32_t reverse_, outS_, outSa_;
    uint32_t chunkLen_;          // s 快照间隔（tiling 下发；默认 16 = 官方 backward 重建粒度）
    uint32_t sSlots_;            // 本核 s 快照槽位数（chunkLen_==0 时归 0，免除零）
    uint32_t sane_;              // tiling 字段合理性校验结果（0 = 本核静默退出）
    uint64_t seqLenK_, seqLenV_;

    GlobalTensor<T> qGm_, wGm_, kGm_, vGm_, zGm_, bGm_, oGm_;
    GlobalTensor<float> initGm_;   // state 张量恒 fp32
    GlobalTensor<float> sGm_, saGm_;         // 训练预埋输出，恒 fp32

    TBuf<TPosition::VECCALC> stateBuf_;
    TBuf<TPosition::VECCALC> qBuf_, wBuf_, kBuf_, vBuf_, zBuf_, bBuf_;
    TBuf<TPosition::VECCALC> decayBuf_, eBuf_, qsBuf_, saBuf_, oBuf_;
    TBuf<TPosition::VECCALC> qStBuf_, wStBuf_, kStBuf_, vStBuf_, zStBuf_, bStBuf_, oStBuf_;
};

extern "C" __global__ __aicore__ void
fused_recurrent_rwkv8(GM_ADDR q, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR z, GM_ADDR b,
                      GM_ADDR initialState, GM_ADDR o, GM_ADDR s, GM_ADDR sa,
                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(FusedRecurrentRwkv8TilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    KernelFusedRecurrentRwkv8<DTYPE_Q> op(&pipe);
    op.Init(q, w, k, v, z, b, initialState, o, s, sa, &tilingData);
    op.Process();
}
