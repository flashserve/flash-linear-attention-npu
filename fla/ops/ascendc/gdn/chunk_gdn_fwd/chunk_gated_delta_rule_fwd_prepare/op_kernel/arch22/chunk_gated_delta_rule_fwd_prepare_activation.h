// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. 独立gate/beta激活和log2坐标；只借当前尚未产叶的逻辑slot。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_ACTIVATION_H
#define GDN_ARCH22_UNIFIED_PREPARE_ACTIVATION_H

#include "chunk_gated_delta_rule_fwd_prepare_resources.h"

namespace GdnUnifiedPrepare {

// 与固定A5 Prepare common.h使用同一FP32常量，不将Ln2移到KKT的clamp之前。
constexpr float kPrepareRcpLn2 = 1.4426950216f;
constexpr float kPrepareLn2 = 0.6931471825f;

template <int BT>
__aicore__ inline void ActivatePrepareGate(Resources &r, const Args &a,
                                           LocalTensor<float> input, uint64_t headBegin,
                                           uint32_t heads, uint32_t rows, uint32_t slot)
{
    if (a.useGate == 0) return;
    constexpr uint32_t MatrixBytes = BT * 8 * sizeof(float);
    constexpr uint32_t SlotBytes = 128 * BT;
    static_assert(3 * MatrixBytes <= SlotBytes, "gate workspace exceeds its dead logical leaf slot");
    const uint32_t scratch = Local::kLeaf + slot * SlotBytes;
    auto softplus = r.template UB<float>(scratch);
    auto positive = r.template UB<float>(scratch + MatrixBytes);
    auto negA = r.template UB<float>(scratch + 2 * MatrixBytes);
    auto bias = negA[8];
    // 前一beta激活可刚使用同槽；关闭V读者后才让MTE2覆盖参数区。
    r.template Fence<HardEvent::V_MTE2>();
    DataCopyExtParams cp{1, heads * 4, 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(8 - heads), 0};
    DataCopyPad(negA, Gm<float>(a.aLog, a.hv)[headBegin], cp, pad);
    if (a.dtBias != nullptr) DataCopyPad(bias, Gm<float>(a.dtBias, a.hv)[headBegin], cp, pad);
    r.template Fence<HardEvent::MTE2_V>();
    Exp(negA, negA, heads);
    if (a.dtBias == nullptr) Duplicate(bias, 0.0f, 8);
    PipeBarrier<PIPE_V>();
    Muls(negA, negA, -1.0f, heads);
    for (uint32_t row = 0; row < rows; ++row) Add(input[row * 8], input[row * 8], bias, heads);
    PipeBarrier<PIPE_V>();
    Abs(softplus, input, rows * 8);
    PipeBarrier<PIPE_V>();
    Muls(softplus, softplus, -1.0f, rows * 8);
    PipeBarrier<PIPE_V>();
    Exp(softplus, softplus, rows * 8);
    PipeBarrier<PIPE_V>();
    Adds(softplus, softplus, 1.0f, rows * 8);
    PipeBarrier<PIPE_V>();
    Ln(softplus, softplus, rows * 8);
    Maxs(positive, input, 0.0f, rows * 8);
    PipeBarrier<PIPE_V>();
    Add(softplus, softplus, positive, rows * 8);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < rows; ++row) Mul(input[row * 8], softplus[row * 8], negA, heads);
    PipeBarrier<PIPE_V>();
}

template <int BT>
__aicore__ inline void ActivatePrepareBeta(Resources &r, const Args &a,
                                           LocalTensor<float> beta, uint32_t heads, uint32_t slot)
{
    if (a.useBetaSigmoid == 0) return;
    constexpr uint32_t SlotBytes = 128 * BT;
    auto one = r.template UB<float>(Local::kLeaf + slot * SlotBytes);
    const uint32_t elements = heads * BT;
    Duplicate(one, 1.0f, elements);
    Muls(beta, beta, -1.0f, elements);
    PipeBarrier<PIPE_V>();
    Exp(beta, beta, elements);
    PipeBarrier<PIPE_V>();
    Adds(beta, beta, 1.0f, elements);
    PipeBarrier<PIPE_V>();
    Div(beta, one, beta, elements);
    PipeBarrier<PIPE_V>();
    if (a.allowNegEigval != 0) {
        Muls(beta, beta, 2.0f, elements);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ScalePrepareCumsum(const Args &a, LocalTensor<float> prefix,
                                          LocalTensor<float> acc, uint32_t rows, uint32_t heads)
{
    if (a.useExp2 == 0) return;
    Muls(prefix, prefix, kPrepareRcpLn2, rows * 8);
    Muls(acc, acc, kPrepareRcpLn2, heads);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void PrepareExp(const Args &a, LocalTensor<float> dst,
                                  LocalTensor<float> src, uint32_t elements)
{
    if (a.useExp2 != 0) {
        Muls(dst, src, kPrepareLn2, elements);
        PipeBarrier<PIPE_V>();
        Exp(dst, dst, elements);
    } else {
        Exp(dst, src, elements);
    }
}

} // namespace GdnUnifiedPrepare
#endif
