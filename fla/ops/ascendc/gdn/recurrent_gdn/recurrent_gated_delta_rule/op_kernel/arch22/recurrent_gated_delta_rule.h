/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef RECURRENT_GATED_DELTA_RULE_ARCH22_H
#define RECURRENT_GATED_DELTA_RULE_ARCH22_H

#include "kernel_operator.h"
#include "../recurrent_gated_delta_rule_struct.h"

namespace RecurrentGatedDeltaRule {
using namespace AscendC;

struct RGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR state;
    GM_ADDR lengths;
    GM_ADDR indices;
    GM_ADDR gate;
    GM_ADDR accepted;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR out;
};

// Each physical region has one owner. In particular a BF16 snapshot lives in
// the consumed source bank, never in the FP32 bank used by the next recurrence.
template <typename State, typename Beta, typename Gate>
class RGDR {
    static constexpr uint32_t kHeadDim = 128;
    static constexpr uint32_t kWindowTokens = 8;
    static constexpr uint32_t kBufferCount = 2;
    static constexpr uint32_t kQkvInputCount = 3;
    static constexpr uint32_t kQueryInput = 0;
    static constexpr uint32_t kKeyInput = 1;
    static constexpr uint32_t kValueInput = 2;
    static constexpr uint32_t kBlockBytes = 32;
    static constexpr uint32_t kVectorBytes = 256;
    static constexpr uint32_t kFloatBlockElements = kBlockBytes / sizeof(float);
    static constexpr uint32_t kFloatVectorElements = kVectorBytes / sizeof(float);
    static constexpr uint32_t kFloatRowBytes = kHeadDim * sizeof(float);
    static constexpr uint32_t kBf16RowBytes = kHeadDim * sizeof(bfloat16_t);
    static constexpr uint32_t kRowBlocks = kFloatRowBytes / kBlockBytes;
    static constexpr uint32_t kStateElements = kHeadDim * kHeadDim;
    static constexpr uint32_t kStateBankBytes = kStateElements * sizeof(float);
    static constexpr uint32_t kSnapshotOffset = kStateBankBytes - kStateElements * sizeof(bfloat16_t);
    static constexpr uint32_t kWindowElements = kWindowTokens * kHeadDim;
    static constexpr uint32_t kRawInputBytes = kWindowElements * sizeof(bfloat16_t);
    static constexpr uint32_t kRawBetaOffset = kQkvInputCount * kRawInputBytes;
    static constexpr uint32_t kRawGateOffset = kRawBetaOffset + kBlockBytes;
    static constexpr uint32_t kRawBankBytes = kRawGateOffset + kBlockBytes;
    static constexpr uint32_t kRawBaseOffset = kBufferCount * kStateBankBytes;
    static constexpr uint32_t kFloatWindowBytes = kWindowElements * sizeof(float);
    static constexpr uint32_t kQueryOffset = kRawBaseOffset + kBufferCount * kRawBankBytes;
    static constexpr uint32_t kKeyOffset = kQueryOffset + kFloatWindowBytes;
    static constexpr uint32_t kValueOffset = kKeyOffset + kFloatWindowBytes;
    static constexpr uint32_t kDeltaOffset = kValueOffset + kFloatWindowBytes;
    static constexpr uint32_t kOutputOffset = kDeltaOffset + kFloatRowBytes;
    static constexpr uint32_t kReductionOffset = kOutputOffset + kFloatRowBytes;
    static constexpr uint32_t kBroadcastOffset = kReductionOffset + kFloatRowBytes;
    static constexpr uint32_t kBetaOffset = kBroadcastOffset + kHeadDim * kBlockBytes;
    static constexpr uint32_t kGateOffset = kBetaOffset + kBlockBytes;
    static constexpr uint32_t kTemp0Offset = kGateOffset + kBlockBytes;
    static constexpr uint32_t kTemp1Offset = kTemp0Offset + kBlockBytes;
    static constexpr uint32_t kTemp2Offset = kTemp1Offset + kBlockBytes;
    static constexpr uint32_t kTemp3Offset = kTemp2Offset + kBlockBytes;
    static constexpr uint32_t kOutputBf16Offset = kTemp3Offset + kBlockBytes;
    static constexpr uint32_t kHeadParamsOffset = kOutputBf16Offset + kBufferCount * kBf16RowBytes;
    static constexpr uint32_t kWriteSlotsOffset = kHeadParamsOffset + kBlockBytes;
    static constexpr uint32_t kMetadataScratchOffset = kWriteSlotsOffset + kBlockBytes;
    static constexpr uint32_t kControlOffset = kMetadataScratchOffset + kBlockBytes;
    static constexpr uint32_t kUbBytes = kControlOffset + kBlockBytes;

    static constexpr uint32_t kNormNewtonIterations = 3;
    static constexpr float kNormEpsilon = 1.0e-6f;
    static_assert(kWindowTokens == kFloatBlockElements, "Metadata window must fill one FP32 block");

public:
    __aicore__ inline RGDR(const RecurrentGatedDeltaRuleTilingData *td) : td_(*td) {}

    __aicore__ inline void Init(const RGDRInitParams &p, TPipe *pipe)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ bfloat16_t *)p.query);
        k_.SetGlobalBuffer((__gm__ bfloat16_t *)p.key);
        v_.SetGlobalBuffer((__gm__ bfloat16_t *)p.value);
        beta_.SetGlobalBuffer((__gm__ Beta *)p.beta);
        gate_.SetGlobalBuffer((__gm__ Gate *)p.gate);
        state_.SetGlobalBuffer((__gm__ State *)p.state);
        lengths_.SetGlobalBuffer((__gm__ int32_t *)p.lengths);
        indices_.SetGlobalBuffer((__gm__ int32_t *)p.indices);
        accepted_.SetGlobalBuffer((__gm__ int32_t *)p.accepted);
        aLog_.SetGlobalBuffer((__gm__ float *)p.aLog);
        bias_.SetGlobalBuffer((__gm__ float *)p.dtBias);
        out_.SetGlobalBuffer((__gm__ bfloat16_t *)p.out);
        pipe_->InitBuffer(ub_, kUbBytes);
        for (uint32_t bank = 0; bank < kBufferCount; ++bank) {
            for (uint32_t input = 0; input < kQkvInputCount; ++input) {
                ready_[bank][input] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            }
            rawFree_[bank] = pipe_->AllocEventID<HardEvent::V_MTE2>();
            stateReady_[bank] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            stateFree_[bank] = pipe_->AllocEventID<HardEvent::MTE3_V>();
            outReady_[bank] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            outFree_[bank] = pipe_->AllocEventID<HardEvent::MTE3_V>();
        }
        scalarToVector_ = pipe_->AllocEventID<HardEvent::S_V>();
        taskUbFree_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        taskGmFree_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        const uint64_t blocks = GetBlockNum();
        const uint64_t block = GetBlockIdx();
        const uint64_t quotient = td_.taskCount / blocks;
        const uint64_t remainder = td_.taskCount % blocks;
        const uint64_t begin = block * quotient + (block < remainder ? block : remainder);
        const uint64_t end = begin + quotient + (block < remainder ? 1 : 0);
        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t batch = task / td_.valueHeads;
            const uint64_t head = task % td_.valueHeads;
            const int32_t length = lengths_.GetValue(batch + 1);
            if (length == 0) continue;
            uint64_t start = lengths_.GetValue(0);
            for (uint64_t b = 0; b < batch; ++b) start += lengths_.GetValue(b + 1);
            RunHead(batch, head, start, static_cast<uint32_t>(length));
        }
        ReleaseEvents();
    }

private:
    __aicore__ inline void CopyInput(uint32_t bank, uint32_t input, uint64_t start,
                                     uint64_t head, uint32_t count)
    {
        const uint64_t *strides = input == kQueryInput ? td_.queryStrides :
                                  (input == kKeyInput ? td_.keyStrides : td_.valueStrides);
        const uint64_t offset = start * strides[0] + head * strides[1];
        DataCopyExtParams params{static_cast<uint16_t>(count), kBf16RowBytes,
                                 static_cast<uint32_t>((strides[0] - kHeadDim) * sizeof(bfloat16_t)), 0, 0};
        DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, 0};
        auto dst = ub_.GetWithOffset<bfloat16_t>(
            kWindowElements, kRawBaseOffset + bank * kRawBankBytes + input * kRawInputBytes);
        if (input == kQueryInput) DataCopyPad(dst, q_[offset], params, padding);
        else if (input == kKeyInput) DataCopyPad(dst, k_[offset], params, padding);
        else DataCopyPad(dst, v_[offset], params, padding);
        SetFlag<HardEvent::MTE2_V>(ready_[bank][input]);
    }

    __aicore__ inline void CopyWindowMeta(uint32_t bank, uint64_t batch, uint64_t head,
                                          uint64_t start, uint32_t position, uint32_t count)
    {
        auto betaRaw = ub_.GetWithOffset<Beta>(kWindowTokens, kRawBaseOffset + bank * kRawBankBytes + kRawBetaOffset);
        auto gateRaw = ub_.GetWithOffset<Gate>(kWindowTokens, kRawBaseOffset + bank * kRawBankBytes + kRawGateOffset);
        auto slots = ub_.GetWithOffset<int32_t>(kWindowTokens, kWriteSlotsOffset);
        for (uint32_t i = 0; i < kWindowTokens; ++i) {
            betaRaw.SetValue(i, static_cast<Beta>(0.0f));
            if (td_.hasGama) gateRaw.SetValue(i, static_cast<Gate>(0.0f));
            slots.SetValue(i, 0);
        }
        // `start` is already the absolute GM token of this window.  Only the
        // checkpoint layout uses the request-local `position` axis.
        const uint64_t indexBase = td_.checkpointWidth ? batch * td_.checkpointWidth + position : start;
        for (uint32_t i = 0; i < count; ++i) {
            betaRaw.SetValue(i, static_cast<Beta>(beta_.GetValue((start + i) * td_.valueHeads + head)));
            if (td_.hasGama) {
                gateRaw.SetValue(i, static_cast<Gate>(gate_.GetValue((start + i) * td_.valueHeads + head)));
            }
            slots.SetValue(i, indices_.GetValue(indexBase + i));
        }
        // GM metadata is gathered once per window; all gate/beta arithmetic
        // below is vectorized over the padded window.
        SetFlag<HardEvent::S_V>(scalarToVector_);
    }

    __aicore__ inline void PrepareWindow(uint32_t bank, uint64_t batch, uint64_t head,
                                         uint64_t start, uint32_t position, uint32_t count,
                                         float aLog, float bias)
    {
        auto betaRaw = ub_.GetWithOffset<Beta>(kWindowTokens, kRawBaseOffset + bank * kRawBankBytes + kRawBetaOffset);
        auto gateRaw = ub_.GetWithOffset<Gate>(kWindowTokens, kRawBaseOffset + bank * kRawBankBytes + kRawGateOffset);
        const uint64_t keyHead = head / (td_.valueHeads / td_.keyHeads);
        CopyWindowMeta(bank, batch, head, start, position, count);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][kQueryInput]);
        CopyInput(bank, kKeyInput, start, keyHead, count);
        Cast(ub_.GetWithOffset<float>(kWindowElements, kQueryOffset),
             ub_.GetWithOffset<bfloat16_t>(kWindowElements, kRawBaseOffset + bank * kRawBankBytes),
             RoundMode::CAST_NONE, count * kHeadDim);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][kKeyInput]);
        CopyInput(bank, kValueInput, start, head, count);
        Cast(ub_.GetWithOffset<float>(kWindowElements, kKeyOffset),
             ub_.GetWithOffset<bfloat16_t>(kWindowElements, kRawBaseOffset + bank * kRawBankBytes + kRawInputBytes),
             RoundMode::CAST_NONE, count * kHeadDim);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][kValueInput]);
        Cast(ub_.GetWithOffset<float>(kWindowElements, kValueOffset),
             ub_.GetWithOffset<bfloat16_t>(
                 kWindowElements, kRawBaseOffset + bank * kRawBankBytes + kValueInput * kRawInputBytes),
             RoundMode::CAST_NONE, count * kHeadDim);
        WaitFlag<HardEvent::S_V>(scalarToVector_);
        auto beta = ub_.GetWithOffset<float>(kWindowTokens, kBetaOffset);
        auto gate = ub_.GetWithOffset<float>(kWindowTokens, kGateOffset);
        auto tmp0 = ub_.GetWithOffset<float>(kWindowTokens, kTemp0Offset);
        auto tmp1 = ub_.GetWithOffset<float>(kWindowTokens, kTemp1Offset);
        auto tmp2 = ub_.GetWithOffset<float>(kWindowTokens, kTemp2Offset);
        // Pad short metadata windows to the A2 FP32 vector granularity.  The
        // token loop consumes only [0, count), so the padded tail is ignored.
        const uint32_t vectorCount = kWindowTokens;
        if constexpr (std::is_same<Beta, float>::value) DataCopy(beta, betaRaw, vectorCount);
        else Cast(beta, betaRaw, RoundMode::CAST_NONE, vectorCount);
        if (td_.useBetaSigmoidInKernel) {
            // Both exponent arguments are nonpositive. This preserves sigmoid
            // tails without a packed-mask select for the short metadata window.
            Abs(tmp0, beta, vectorCount);
            Mins(tmp1, beta, 0.0f, vectorCount);
            PipeBarrier<PIPE_V>();
            Muls(tmp0, tmp0, -1.0f, vectorCount);
            Exp(tmp1, tmp1, vectorCount);
            PipeBarrier<PIPE_V>();
            Exp(tmp0, tmp0, vectorCount);
            PipeBarrier<PIPE_V>();
            Adds(tmp0, tmp0, 1.0f, vectorCount);
            PipeBarrier<PIPE_V>();
            Div(beta, tmp1, tmp0, vectorCount);
            if (td_.allowNegEigval) {
                PipeBarrier<PIPE_V>();
                Muls(beta, beta, 2.0f, vectorCount);
            }
        }
        if (td_.hasGama) {
            if constexpr (std::is_same<Gate, float>::value) DataCopy(gate, gateRaw, vectorCount);
            else Cast(gate, gateRaw, RoundMode::CAST_NONE, vectorCount);
            if (td_.useGateInKernel) {
                Adds(gate, gate, bias, vectorCount);
                PipeBarrier<PIPE_V>();
                Maxs(tmp0, gate, 0.0f, vectorCount);
                Abs(tmp1, gate, vectorCount);
                Muls(tmp1, tmp1, -1.0f, vectorCount);
                Exp(tmp1, tmp1, vectorCount);
                PipeBarrier<PIPE_V>();
                Adds(tmp2, tmp1, 1.0f, vectorCount);
                PipeBarrier<PIPE_V>();
                // u=round(1+x), x=exp(-abs(z)). Recover the rounding loss
                // before Ln so small softplus tails survive multiplication
                // by exp(A_log). No scalar branch or packed mask is needed.
                Adds(gate, tmp2, -1.0f, vectorCount);
                PipeBarrier<PIPE_V>();
                Sub(gate, tmp1, gate, vectorCount);
                PipeBarrier<PIPE_V>();
                Div(gate, gate, tmp2, vectorCount);
                Ln(tmp1, tmp2, vectorCount);
                PipeBarrier<PIPE_V>();
                Add(gate, gate, tmp1, vectorCount);
                PipeBarrier<PIPE_V>();
                Add(gate, tmp0, gate, vectorCount);
                Duplicate(tmp1, aLog, vectorCount);
                Exp(tmp1, tmp1, vectorCount);
                PipeBarrier<PIPE_V>();
                Mul(gate, gate, tmp1, vectorCount);
                Muls(gate, gate, -1.0f, vectorCount);
                PipeBarrier<PIPE_V>();
                Exp(gate, gate, vectorCount);
            } else {
                Exp(gate, gate, vectorCount);
            }
        }
        // The raw bank is only reusable after all MTE2-produced values have
        // been consumed by V.  The vectorized casts above are the last raw
        // readers; the remaining stages operate on typed UB vectors.
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(rawFree_[bank]);
        rawPending_[bank] = true;
        auto q = ub_.GetWithOffset<float>(kWindowElements, kQueryOffset);
        auto k = ub_.GetWithOffset<float>(kWindowElements, kKeyOffset);
        if (td_.useQkL2NormInKernel) {
            NormalizeWindow(q, count);
            NormalizeWindow(k, count);
        }
        Muls(q, q, td_.scale, count * kHeadDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeWindow(LocalTensor<float> vector, uint32_t rows)
    {
        auto product = ub_.GetWithOffset<float>(kWindowElements, kBroadcastOffset);
        auto sum = ub_.GetWithOffset<float>(kHeadDim, kDeltaOffset);
        auto inverse = ub_.GetWithOffset<float>(kHeadDim, kReductionOffset);
        auto correction = ub_.GetWithOffset<float>(kWindowTokens, kTemp0Offset);
        Mul(product, vector, vector, rows * kHeadDim);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; ++row) {
            Add(product[row * kHeadDim + kFloatVectorElements], product[row * kHeadDim],
                product[row * kHeadDim + kFloatVectorElements], kFloatVectorElements);
            WholeReduceSum(sum[row], product[row * kHeadDim + kFloatVectorElements],
                           kFloatVectorElements, 1, 1, 1, kVectorBytes / kBlockBytes);
        }
        PipeBarrier<PIPE_V>();
        const uint32_t vectorRows = kWindowTokens;
        Adds(sum, sum, kNormEpsilon, vectorRows);
        PipeBarrier<PIPE_V>();
        Adds(inverse, sum, 0.0f, vectorRows);
        Rsqrt(inverse, inverse, vectorRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t iteration = 0; iteration < kNormNewtonIterations; ++iteration) {
            Mul(correction, inverse, inverse, vectorRows);
            PipeBarrier<PIPE_V>();
            Mul(correction, correction, sum, vectorRows);
            PipeBarrier<PIPE_V>();
            Muls(correction, correction, -0.5f, vectorRows);
            Adds(correction, correction, 1.5f, vectorRows);
            PipeBarrier<PIPE_V>();
            Mul(inverse, inverse, correction, vectorRows);
            PipeBarrier<PIPE_V>();
        }
        // Broadcast one inverse norm to each kHeadDim-element token row and scale
        // the whole window in one VEC pass.  This removes the former scalar
        // read/scale loop over tokens while retaining an extra Newton step for
        // the A2 rsqrt approximation.
        auto broadcast = ub_.GetWithOffset<float>(kWindowElements, kBroadcastOffset);
        Brcb(broadcast, inverse, static_cast<uint8_t>((rows + kFloatBlockElements - 1) / kFloatBlockElements),
             {1, kFloatBlockElements});
        PipeBarrier<PIPE_V>();
        // Brcb stores one 8-scalar block per repeat.  The repeat form below
        // reuses those row scalars across all kHeadDim features without reading
        // beyond the compact broadcast block.
        for (uint32_t half = 0; half < kHeadDim; half += kFloatVectorElements) {
            Mul(vector[half], vector[half], broadcast, kFloatVectorElements, rows,
                {1, 1, 0, kRowBlocks, kRowBlocks, 1});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadState(uint64_t slot, uint64_t head)
    {
        SetFlag<HardEvent::V_MTE2>(taskUbFree_);
        WaitFlag<HardEvent::V_MTE2>(taskUbFree_);
        SetFlag<HardEvent::MTE3_MTE2>(taskGmFree_);
        WaitFlag<HardEvent::MTE3_MTE2>(taskGmFree_);
        DataCopyExtParams params{kHeadDim, static_cast<uint32_t>(kHeadDim * sizeof(State)),
            static_cast<uint32_t>((td_.stateStrides[2] - kHeadDim) * sizeof(State)), 0, 0};
        DataCopyPadExtParams<State> padding{false, 0, 0, 0};
        const uint64_t stateOffset = slot * td_.stateStrides[0] + head * td_.stateStrides[1];
        if constexpr (std::is_same<State, float>::value) {
            DataCopyPad(ub_.GetWithOffset<float>(kStateElements, 0), state_[stateOffset], params, padding);
        } else {
            DataCopyPad(ub_.GetWithOffset<State>(kStateElements, kSnapshotOffset), state_[stateOffset], params, padding);
        }
        SetFlag<HardEvent::MTE2_V>(ready_[0][kQueryInput]);
        WaitFlag<HardEvent::MTE2_V>(ready_[0][kQueryInput]);
        if constexpr (!std::is_same<State, float>::value) {
            Cast(ub_.GetWithOffset<float>(kStateElements, 0), ub_.GetWithOffset<State>(kStateElements, kSnapshotOffset),
                 RoundMode::CAST_NONE, kStateElements);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DotRows(LocalTensor<float> dst, LocalTensor<float> matrix,
                                    LocalTensor<float> vector, LocalTensor<float> product)
    {
        // The reclaimed source State bank is disjoint from matrix and dst.
        for (uint32_t half = 0; half < kHeadDim; half += kFloatVectorElements) {
            Mul(product[half], matrix[half], vector[half], kFloatVectorElements, kHeadDim,
                {1, 1, 1, kRowBlocks, kRowBlocks, 0});
        }
        PipeBarrier<PIPE_V>();
        Add(product[kFloatVectorElements], product, product[kFloatVectorElements],
            kFloatVectorElements, kHeadDim, {1, 1, 1, kRowBlocks, kRowBlocks, kRowBlocks});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dst, product[kFloatVectorElements], kFloatVectorElements, kHeadDim, 1, 1, kRowBlocks);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void EmitState(uint32_t bank, uint64_t slot, uint64_t head)
    {
        SetFlag<HardEvent::V_MTE3>(stateReady_[bank]);
        WaitFlag<HardEvent::V_MTE3>(stateReady_[bank]);
        DataCopyExtParams params{kHeadDim, static_cast<uint32_t>(kHeadDim * sizeof(State)), 0,
            static_cast<uint32_t>((td_.stateStrides[2] - kHeadDim) * sizeof(State)), 0};
        const uint64_t stateOffset = slot * td_.stateStrides[0] + head * td_.stateStrides[1];
        if constexpr (std::is_same<State, float>::value) {
            DataCopyPad(state_[stateOffset], ub_.GetWithOffset<float>(kStateElements, bank * kStateBankBytes), params);
        } else {
            DataCopyPad(state_[stateOffset],
                        ub_.GetWithOffset<State>(kStateElements, bank * kStateBankBytes + kSnapshotOffset), params);
        }
        SetFlag<HardEvent::MTE3_V>(stateFree_[bank]);
        statePending_[bank] = true;
    }

    __aicore__ inline void Token(uint32_t token, uint64_t head,
                                  uint64_t globalToken, uint32_t src, uint32_t dst)
    {
        auto query = ub_.GetWithOffset<float>(kHeadDim, kQueryOffset + token * kFloatRowBytes);
        auto key = ub_.GetWithOffset<float>(kHeadDim, kKeyOffset + token * kFloatRowBytes);
        auto value = ub_.GetWithOffset<float>(kHeadDim, kValueOffset + token * kFloatRowBytes);
        const float beta = ub_.GetWithOffset<float>(kWindowTokens, kBetaOffset).GetValue(token);
        const float alpha = td_.hasGama ? ub_.GetWithOffset<float>(kWindowTokens, kGateOffset).GetValue(token) : 1.0f;
        auto source = ub_.GetWithOffset<float>(kStateElements, src * kStateBankBytes);
        auto target = ub_.GetWithOffset<float>(kStateElements, dst * kStateBankBytes);
        auto delta = ub_.GetWithOffset<float>(kHeadDim, kDeltaOffset);
        if (statePending_[dst]) {
            WaitFlag<HardEvent::MTE3_V>(stateFree_[dst]);
            statePending_[dst] = false;
        }
        Muls(target, source, alpha, kStateElements);
        PipeBarrier<PIPE_V>();
        if (statePending_[src]) {
            WaitFlag<HardEvent::MTE3_V>(stateFree_[src]);
            statePending_[src] = false;
        }
        DotRows(delta, target, key, source);
        Sub(delta, value, delta, kHeadDim);
        PipeBarrier<PIPE_V>();
        Muls(delta, delta, beta, kHeadDim);
        PipeBarrier<PIPE_V>();
        auto broadcast = ub_.GetWithOffset<float>(kWindowElements, kBroadcastOffset);
        Brcb(broadcast, delta, kHeadDim / kFloatBlockElements, {1, kFloatBlockElements});
        PipeBarrier<PIPE_V>();
        for (uint32_t half = 0; half < kHeadDim; half += kFloatVectorElements) {
            MulAddDst(target[half], key[half], broadcast, kFloatVectorElements, kHeadDim,
                      {1, 1, 0, kRowBlocks, 0, 1});
        }
        PipeBarrier<PIPE_V>();
        auto output = ub_.GetWithOffset<float>(kHeadDim, kOutputOffset);
        DotRows(output, target, query, source);
        const uint32_t outBank = globalToken % kBufferCount;
        if (outPending_[outBank]) {
            WaitFlag<HardEvent::MTE3_V>(outFree_[outBank]);
            outPending_[outBank] = false;
        }
        auto outputBf16 = ub_.GetWithOffset<bfloat16_t>(kHeadDim, kOutputBf16Offset + outBank * kBf16RowBytes);
        Cast(outputBf16, output, RoundMode::CAST_RINT, kHeadDim);
        const int32_t writeSlot = ub_.GetWithOffset<int32_t>(kWindowTokens, kWriteSlotsOffset).GetValue(token);
        if constexpr (std::is_same<State, float>::value) {
            EmitState(dst, static_cast<uint64_t>(writeSlot), head);
        } else {
            // Source was reclaimed on its preceding turn as a destination.
            // All V reads above must finish before its upper half is reused.
            PipeBarrier<PIPE_V>();
            Cast(ub_.GetWithOffset<State>(kStateElements, src * kStateBankBytes + kSnapshotOffset), target,
                 RoundMode::CAST_RINT, kStateElements);
            EmitState(src, static_cast<uint64_t>(writeSlot), head);
        }
        SetFlag<HardEvent::V_MTE3>(outReady_[outBank]);
        WaitFlag<HardEvent::V_MTE3>(outReady_[outBank]);
        DataCopy(out_[(globalToken * td_.valueHeads + head) * kHeadDim], outputBf16, kHeadDim);
        SetFlag<HardEvent::MTE3_V>(outFree_[outBank]);
        outPending_[outBank] = true;
    }

    __aicore__ inline void RunHead(uint64_t batch, uint64_t head, uint64_t start, uint32_t length)
    {
        const int32_t accepted = td_.hasAcceptedTokens ? accepted_.GetValue(batch) : 1;
        const uint64_t indexBase = td_.checkpointWidth ? batch * td_.checkpointWidth : start;
        const int32_t initial = indices_.GetValue(indexBase + static_cast<uint64_t>(accepted - 1));
        LoadState(static_cast<uint64_t>(initial), head);
        const float aLog = td_.hasALog ? aLog_.GetValue(head) : 0.0f;
        const float bias = td_.hasDtBias ? bias_.GetValue(head) : 0.0f;
        const uint64_t keyHead = head / (td_.valueHeads / td_.keyHeads);
        uint32_t src = 0;
        if (rawPending_[0]) {
            WaitFlag<HardEvent::V_MTE2>(rawFree_[0]);
            rawPending_[0] = false;
        }
        CopyInput(0, kQueryInput, start, keyHead, length < kWindowTokens ? length : kWindowTokens);
        for (uint32_t position = 0; position < length; position += kWindowTokens) {
            const uint32_t bank = (position / kWindowTokens) % kBufferCount;
            const uint32_t count = length - position < kWindowTokens ? length - position : kWindowTokens;
            PrepareWindow(bank, batch, head, start + position, position, count, aLog, bias);
            if (position + count < length) {
                const uint32_t nextCount =
                    length - position - count < kWindowTokens ? length - position - count : kWindowTokens;
                const uint32_t nextBank = 1 - bank;
                if (rawPending_[nextBank]) {
                    WaitFlag<HardEvent::V_MTE2>(rawFree_[nextBank]);
                    rawPending_[nextBank] = false;
                }
                CopyInput(nextBank, kQueryInput, start + position + count, keyHead, nextCount);
            }
            for (uint32_t token = 0; token < count; ++token) {
                Token(token, head, start + position + token, src, 1 - src);
                src = 1 - src;
            }
        }
        for (uint32_t bank = 0; bank < kBufferCount; ++bank) {
            if (statePending_[bank]) {
                WaitFlag<HardEvent::MTE3_V>(stateFree_[bank]);
                statePending_[bank] = false;
            }
            if (outPending_[bank]) {
                WaitFlag<HardEvent::MTE3_V>(outFree_[bank]);
                outPending_[bank] = false;
            }
            if (rawPending_[bank]) {
                WaitFlag<HardEvent::V_MTE2>(rawFree_[bank]);
                rawPending_[bank] = false;
            }
        }
    }

    __aicore__ inline void ReleaseEvents()
    {
        for (uint32_t bank = 0; bank < kBufferCount; ++bank) {
            for (uint32_t i = 0; i < kQkvInputCount; ++i) pipe_->ReleaseEventID<HardEvent::MTE2_V>(ready_[bank][i]);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(rawFree_[bank]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(stateReady_[bank]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(stateFree_[bank]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(outReady_[bank]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outFree_[bank]);
        }
        pipe_->ReleaseEventID<HardEvent::S_V>(scalarToVector_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(taskUbFree_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(taskGmFree_);
    }

    RecurrentGatedDeltaRuleTilingData td_;
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<bfloat16_t> q_, k_, v_, out_;
    GlobalTensor<Beta> beta_;
    GlobalTensor<Gate> gate_;
    GlobalTensor<State> state_;
    GlobalTensor<int32_t> lengths_, indices_, accepted_;
    GlobalTensor<float> aLog_, bias_;
    TEventID ready_[kBufferCount][kQkvInputCount], rawFree_[kBufferCount];
    TEventID stateReady_[kBufferCount], stateFree_[kBufferCount], outReady_[kBufferCount], outFree_[kBufferCount];
    TEventID scalarToVector_, taskUbFree_, taskGmFree_;
    bool rawPending_[kBufferCount] = {false, false};
    bool statePending_[kBufferCount] = {false, false};
    bool outPending_[kBufferCount] = {false, false};
};
} // namespace RecurrentGatedDeltaRule
#endif
