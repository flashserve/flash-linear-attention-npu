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
// 有初态时逐行载入 stateBuf_（GM 行距 V，逐行避免 DataCopyPad blockLen 超 uint16），零转置。
//
// 已否决实验记录：行 padding 消 bank 冲突（行距 V+16，2026-09-11 实测）——msprof 复测
// bankgroup_cflt 未降（fp32_main 18.6→24.3%、kv128 31.1→30.0%）且全 shape 变慢 ~11%
// （行尾 pad 使 UB 搬运量 +25%、访问从连续流变逐行跳行），已回退。详见 profiling 文档 §八。
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
        sane_ = (T_ > 0 && T_ <= (1u << 20) && K_ > 0 && K_ <= 128 && V_ > 0 && V_ <= 128 &&
                 (K_ % 8 == 0) && (V_ % 8 == 0) && (uint64_t)K_ * V_ <= 16384) ? 1 : 0;
        if (!sane_) {
            return;
        }
        // K 维折叠要求 2 幂行数：Kp = ≥K 的最小 2 幂（K≥8 故 Kp≥8）；
        // state/mat 的 pad 行（K..Kp-1）恒 0，折叠天然忽略
        Kp_ = 8;
        while (Kp_ < K_) {
            Kp_ <<= 1;
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

        pipe_->InitBuffer(stateBuf_, Kp_ * V_ * sizeof(float));   // S^T (Kp 行 × V)
        // 整块改写的工作矩阵：系数广播 (K,V) 落点 + 折叠求和原地工作区（二者复用同一块）
        pipe_->InitBuffer(matBuf_, Kp_ * V_ * sizeof(float));
        pipe_->InitBuffer(tmp8Buf_, Kp_ * 8 * sizeof(float));     // Brcb 第一级 (K,8) 落点
        // K 侧向量（q/w/k/z/b/decay/e/qs）
        pipe_->InitBuffer(qBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(wBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(kBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(zBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(bBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(decayBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(eBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(qsBuf_, K_ * sizeof(float));
        // V 侧向量（v/sa）
        pipe_->InitBuffer(vBuf_, V_ * sizeof(float));
        pipe_->InitBuffer(saBuf_, V_ * sizeof(float));
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

        // 初始状态：GM (K,V) 布局（行距 V）逐行读入 UB；逐行而非
        // 整块是为避开 DataCopyPad blockLen uint16 上限：K=V=128 时整块 65536B 会截断
        if (hasInit_) {
            for (uint32_t j = 0; j < K_; j++) {
                DataCopyPad(state[j * V_], initGm_[(uint64_t)j * V_],
                    {1, static_cast<uint16_t>(V_ * sizeof(float)), 0, 0},
                    {false, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            Duplicate(state, 0.0f, Kp_ * V_);   // 整块清零（含 pad 行）
        }
        // pad 行钉 0：hasInit 分支只读了前 K 行；mat 的 pad 行在折叠求和中被当作
        // "state_pad(0) × mat_pad" 恒 0 忽略，前提是初值不是 NaN/Inf 位模式
        if (Kp_ > K_) {
            LocalTensor<float> mat0 = matBuf_.Get<float>();
            Duplicate(state[K_ * V_], 0.0f, (Kp_ - K_) * V_);
            Duplicate(mat0[K_ * V_], 0.0f, (Kp_ - K_) * V_);
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
        LocalTensor<float> mat = matBuf_.Get<float>();     // (Kp,V) 工作矩阵
        LocalTensor<float> tmp8 = tmp8Buf_.Get<float>();   // (Kp,8) Brcb 第一级落点

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

            // ===== 整块向量改写（B+C）：系数 (K,) 经 Brcb 广播成 (K,V) 列常数矩阵，
            // sa/o 用 state⊙coefM 后 K 维成对折叠求和，state 更新用整块 Mul/Add ——
            // 消除逐行 GetValue（320 次/token）和小向量调用（320 次/token）。
            // mat 在五种系数矩阵与折叠求和工作区之间按时序复用；pad 行恒 0 不参与结果。

            // sa = Σ_j z_j·S^T_row_j：state⊙zM 后成对折叠，结果落在 mat[0:V]
            ColBroadcast(mat, zL, tmp8);
            StridedMul(mat, state, mat, Kp_);
            for (uint32_t h = Kp_ / 2; h > 0; h >>= 1) {
                StridedAdd(mat, mat, mat[h * V_], h);
            }
            Adds(saL, mat, 0.0f, V_);   // sa 落 (V,)：mat 下一步要被 decayM 复用

            // state = state ⊙ decayM（只写前 K 行，pad 行保持 0）
            ColBroadcast(mat, decayL, tmp8);
            StridedMul(state, state, mat, K_);
            // state += bM ⊙ sa / kM ⊙ v（mat 就地逐行乘连续的 sa/v：src1 行内
            // 逐 block 推进、每 repeat 回到行起点，免行广播矩阵）
            ColBroadcast(mat, bL, tmp8);
            OuterMulAdd(state, mat, saL);
            ColBroadcast(mat, kL, tmp8);
            OuterMulAdd(state, mat, vL);

            // o = Σ_j qs_j·S^T_row_j(更新后)：同 sa 的折叠，结果落在 mat[0:V]
            ColBroadcast(mat, qsL, tmp8);
            StridedMul(mat, state, mat, Kp_);
            for (uint32_t h = Kp_ / 2; h > 0; h >>= 1) {
                StridedAdd(mat, mat, mat[h * V_], h);
            }
            PipeBarrier<PIPE_ALL>();

            // 训练预埋写出（flag 门控，默认全跳过硬零开销）
            if (outSa_) {
                DataCopyPad(saGm_[offV], saL, {1, static_cast<uint16_t>(V_ * sizeof(float)), 0, 0});
            }
            if (outS_ && sSlots_ > 0 && (t + 1) % chunkLen_ == 0) {
                // s 槽位内 GM 是 (K,V) 布局（行距 V，同 init 读的 uint16 考虑），
                // 逐行写出
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
                DataCopyPad(oGm_[offV], mat, {1, static_cast<uint16_t>(copyBytesV), 0, 0});
            } else {
                LocalTensor<T> oSt = oStBuf_.Get<T>();
                Cast(oSt, mat, RoundMode::CAST_RINT, V_);
                PipeBarrier<PIPE_ALL>();   // V → MTE3：等 Cast 完成再写回
                DataCopyPad(oGm_[offV], oSt, {1, static_cast<uint16_t>(copyBytesV), 0, 0});
            }
            PipeBarrier<PIPE_ALL>();   // mat 复用前等待 MTE3 完成
        }
    }

private:
    // c (K,) → mat (K,V)：mat[j*V+m] = c[j]（列常数广播；只写前 K 行，pad 行不动）。
    // 第一级 Brcb 把每个元素扩成连续 8 份 → tmp8 (K,8)；
    // 第二级按 V 展开：V==8 直接 strided 拷；V%64==0 再 Brcb（src 每 block 8 元素相同，
    // dstRepStride=V/8 把每 repeat 的 8 个 block 落在同一行内）；其余 V（8 的倍数且
    // <64）用 strided Adds 逐 8 列组复制。
    __aicore__ inline void ColBroadcast(const LocalTensor<float>& mat, const LocalTensor<float>& c,
                                        const LocalTensor<float>& tmp8)
    {
        const uint8_t rs = static_cast<uint8_t>(V_ / 8);
        Brcb(tmp8, c, static_cast<uint8_t>(K_ / 8), {1, 8});
        if (V_ == 8) {
            Adds(mat, tmp8, 0.0f, 8, static_cast<uint8_t>(K_), {1, 1, rs, 1});
        } else if (V_ % 64 == 0) {
            for (uint32_t g = 0; g < V_ / 64; g++) {
                Brcb(mat[g * 64], tmp8, static_cast<uint8_t>(K_), {1, rs});
            }
        } else {
            for (uint32_t g = 0; g < V_ / 8; g++) {
                Adds(mat[g * 8], tmp8, 0.0f, 8, static_cast<uint8_t>(K_),
                     {1, 1, rs, 1});
            }
        }
    }

    // state += cM ⊙ x（mat 已是 c 的列广播阵）：mat 逐行乘 x 再累加进 state。
    // mask > 64（V=128）时一个 repeat 内第二组 64 元素的 src1 寻址会绕回
    // repeat 基址（910B 实测：右半读成 x[0:64]，ATK case177-191 右半全错），
    // 故按 64 列组拆成多次 Mul，组内 src1 行内逐 block 推进、每 repeat 回起点。
    __aicore__ inline void OuterMulAdd(const LocalTensor<float>& state, const LocalTensor<float>& mat,
                                       const LocalTensor<float>& x)
    {
        const uint8_t rs = static_cast<uint8_t>(V_ / 8);
        for (uint32_t g = 0; g * 64 < V_; g++) {
            const uint32_t off = g * 64;
            const uint64_t cols = (V_ - off < 64) ? (V_ - off) : 64;
            Mul(mat[off], mat[off], x[off], cols, static_cast<uint8_t>(K_),
                {1, 1, 1, rs, rs, 0});
        }
        StridedAdd(state, state, mat, K_);
    }

    // (rows, V_) 行主序矩阵的整块二元运算。
    // 带自定义 strides 的二元运算 mask≤64（同 OuterMulAdd 注释的绕回 bug），
    // V=128 按 64 列组拆成两次。
    __aicore__ inline void StridedMul(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, uint32_t rows)
    {
        const uint8_t rs = static_cast<uint8_t>(V_ / 8);
        for (uint32_t g = 0; g * 64 < V_; g++) {
            const uint32_t off = g * 64;
            const uint32_t cols = (V_ - off < 64) ? (V_ - off) : 64;
            Mul(dst[off], src0[off], src1[off], cols, static_cast<uint8_t>(rows),
                {1, 1, 1, rs, rs, rs});
        }
    }

    __aicore__ inline void StridedAdd(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, uint32_t rows)
    {
        const uint8_t rs = static_cast<uint8_t>(V_ / 8);
        for (uint32_t g = 0; g * 64 < V_; g++) {
            const uint32_t off = g * 64;
            const uint32_t cols = (V_ - off < 64) ? (V_ - off) : 64;
            Add(dst[off], src0[off], src1[off], cols, static_cast<uint8_t>(rows),
                {1, 1, 1, rs, rs, rs});
        }
    }

    TPipe* pipe_;
    uint32_t B_, T_, H_, K_, V_;
    uint32_t Kp_;                // ≥K 的最小 2 幂（≥8）：sa/o 的 K 维成对折叠行数
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

    TBuf<TPosition::VECCALC> stateBuf_, matBuf_, tmp8Buf_;
    TBuf<TPosition::VECCALC> qBuf_, wBuf_, kBuf_, vBuf_, zBuf_, bBuf_;
    TBuf<TPosition::VECCALC> decayBuf_, eBuf_, qsBuf_, saBuf_;
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
