// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "recurrent_kda_vec.h"

/*
Stage | 生产者/消费者 | 同步
  0   | Scalar/MTE2 将输入、初态、参数搬到UB | MTE2_V EVENT0/1
  1   | VF 消费UB并递推State、收集整tile out | S_V/V_S EVENT0，token发布/完成计数
  2   | MTE3 搬出整tile out和需要的State   | S_MTE3 EVENT0，MTE3_S EVENT0/1

UB布局（B=StateLayout::BYTES，X<=8；偏移由Stage0初始化一次）
区域     | 字节数                    | 生命周期
State    | 3*B                       | 2个bank按状态交替，第三个可用bank预取
Q/K/V    | 各512*X                   | 双槽，输入搬入到VF读完
Gate     | 1024*X                    | 双槽，原dtype保存
Beta     | 2*max(128,32*X)           | 双槽，短行按32B间隔
Params   | 2*544                     | 双槽，各任务参数
Out      | 512*X                     | 双槽，整tile完成后批量搬出
Guard    | 2048                      | 前36B为事件字，+256..+768为scratch

协议 | 发布/消费 | 复用与背压
输入 | MTE2_V0  | 覆盖slot前等待旧VF/out完成
初态 | MTE2_V1  | 预取bank接管前wait；不得覆盖MTE3源bank
快照 | events[0]/events[8] | VF发布token，Scalar在DMA完成后发布done；覆盖前poll
out  | MTE3_S1（indexed）/MTE3_S0（final-only） | tile批量搬出完成后复用slot
末态 | MTE3_S0  | DMA期间保留源bank，下一VF期间回收，最终排空
*/
namespace RecurrentKda {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// 输入：GM地址和host tiling
// 输出：context与UB视图。
// 复用：本核全生命周期
// 同步：只接线，不提交DMA。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void InitRecurrentKda(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, const RecurrentKdaTilingData *tiling, const RKDAInitParams &args, TPipe *pipe)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    ctx.td_ = tiling;
    ctx.args_ = args;
    ctx.x_ = ctx.td_->windowSize;
    ctx.betaSlot_ = 32 * ctx.x_;
    if (ctx.betaSlot_ < 128) ctx.betaSlot_ = 128;
    // Keep V-first offsets unchanged; K-first FP32 banks include row padding.
    ctx.qOffset_ = 3 * BANK;
    ctx.kOffset_ = ctx.qOffset_ + RKDA_ROW_FP32_BYTES_512 * ctx.x_;
    ctx.vOffset_ = ctx.kOffset_ + RKDA_ROW_FP32_BYTES_512 * ctx.x_;
    ctx.gateOffset_ = ctx.vOffset_ + RKDA_ROW_FP32_BYTES_512 * ctx.x_;
    ctx.betaOffset_ = ctx.gateOffset_ + RKDA_GATE_DOUBLE_BYTES_1024 * ctx.x_;
    ctx.paramOffset_ = ctx.betaOffset_ + 2 * ctx.betaSlot_;
    ctx.outOffset_ = ctx.paramOffset_ + RKDA_PARAM_DOUBLE_BYTES_1088;
    ctx.eventOffset_ = ctx.outOffset_ + RKDA_ROW_FP32_BYTES_512 * ctx.x_;
    pipe->InitBuffer(ctx.storage_, ctx.eventOffset_ + RKDA_GUARD_BYTES_2048);
    ctx.ub_ = ctx.storage_.template Get<uint8_t>();
    ctx.q_.SetGlobalBuffer((__gm__ inType *)args.query);
    ctx.k_.SetGlobalBuffer((__gm__ inType *)args.key);
    ctx.v_.SetGlobalBuffer((__gm__ inType *)args.value);
    ctx.stateIn_.SetGlobalBuffer((__gm__ stateType *)args.initState);
    ctx.stateOut_.SetGlobalBuffer((__gm__ stateType *)args.finalState);
    ctx.out_.SetGlobalBuffer((__gm__ outType *)args.attnOut);
    ctx.aLog_.SetGlobalBuffer((__gm__ float *)args.aLog);
    ctx.bias_.SetGlobalBuffer((__gm__ float *)args.dtBias);

}

#ifdef RKDA_EXECUTION_TRACE
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaInitExecutionTrace(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, GM_ADDR records)
{
    ctx.trace_.SetGlobalBuffer((__gm__ uint64_t *)records + GetBlockIdx() * 32768);
    ctx.traceCount_ = 0;
    ctx.trace_.SetValue(0, 0);

}

#endif
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline uint32_t RecurrentKdaFreeBank(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx)
{
    for (uint32_t bank = 0; bank < 3; ++bank)
        if (ctx.bankUse_[bank] == RKDA_BANK_FREE) return bank;
    return RKDA_NO_BANK;

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaBeginBanks(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint32_t initial)
{
    // A prefetched initial State stays reserved until its task takes ownership.
    ctx.initialBank_ = initial;
    ctx.bankUse_[ctx.initialBank_] = RKDA_BANK_RECURRENCE;
    ctx.alternateBank_ = RecurrentKdaFreeBank(ctx);
    ctx.bankUse_[ctx.alternateBank_] = RKDA_BANK_RECURRENCE;

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaReleaseBanks(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx)
{
    ctx.bankUse_[ctx.initialBank_] &= ~RKDA_BANK_RECURRENCE;
    ctx.bankUse_[ctx.alternateBank_] &= ~RKDA_BANK_RECURRENCE;

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaReleaseStateDma(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx)
{
    if (ctx.pendingStateBank_ != RKDA_NO_BANK) {
        ctx.bankUse_[ctx.pendingStateBank_] &= ~RKDA_BANK_MTE3_SOURCE;
        ctx.pendingStateBank_ = RKDA_NO_BANK;
    }

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline int64_t RecurrentKdaMetadata(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, GM_ADDR address, uint32_t dtype, uint64_t index)
{
    if (dtype == 0) {
        GlobalTensor<int32_t> values;
        values.SetGlobalBuffer((__gm__ int32_t *)address);
        return values.GetValue(index);
    }
    GlobalTensor<int64_t> values;
    values.SetGlobalBuffer((__gm__ int64_t *)address);
    return values.GetValue(index);

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline int64_t RecurrentKdaSequence(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint64_t request)
{
    return ctx.td_->hasCuSeqlens ? RecurrentKdaMetadata(ctx, ctx.args_.cuSeqlens, ctx.td_->cuSeqlensDtype, request) :
                               int64_t(request * ctx.td_->seqLen);

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline uint64_t RecurrentKdaSlot(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint64_t request, int64_t begin, int64_t token)
{
    if (!ctx.td_->hasSsmStateIndices) return request;
    const uint64_t offset = ctx.td_->ssmStateStride ?
        request * ctx.td_->ssmStateStride + uint64_t(token - begin) : uint64_t(token);
    const int64_t slot = RecurrentKdaMetadata(ctx, ctx.args_.ssmStateIndices, ctx.td_->ssmStateIndicesDtype, offset);
    return slot >= 0 && uint64_t(slot) < ctx.td_->sBlockNum ? uint64_t(slot) : RKDA_INVALID_SLOT;

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline bool RecurrentKdaValidateSlots(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint64_t request, int64_t begin, int64_t end)
{
    if (ctx.td_->hasAcceptedTokens) {
        int64_t accepted = RecurrentKdaMetadata(ctx, ctx.args_.numAcceptedTokens, ctx.td_->acceptedTokensDtype, request);
        if (accepted <= 0 || accepted > end - begin) return false;
    }
    if (!ctx.td_->hasSsmStateIndices) return request < ctx.td_->sBlockNum;
    for (int64_t token = begin; token < end; ++token)
        if (RecurrentKdaSlot(ctx, request, begin, token) == RKDA_INVALID_SLOT) return false;
    return true;

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline bool RecurrentKdaPrepareTask(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint64_t &cursor, RKDATask &task, bool &valid)
{
    const uint64_t count = uint64_t(ctx.td_->b) * ctx.td_->nv;
    while (cursor < count) {
        const uint64_t id = cursor;
        cursor += GetBlockNum();
        const uint64_t request = id / ctx.td_->nv;
        const int64_t begin = RecurrentKdaSequence(ctx, request), end = RecurrentKdaSequence(ctx, request + 1);
        if (begin == end) continue;
        if (!RecurrentKdaValidateSlots(ctx, request, begin, end)) { valid = false; return false; }
        task = {id, request, id % ctx.td_->nv, begin, end};
        return true;
    }
    return false;

}

template <class T,
          typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaCopyRows(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, GlobalTensor<T> &gm, uint64_t offset, uint64_t stride,
                                    uint32_t destination, uint32_t width, uint32_t rows,
                                    uint32_t rowGap = 0)
{
    auto local = ctx.ub_[destination].template ReinterpretCast<T>();
    DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(T)),
                           static_cast<uint32_t>((stride - width) * sizeof(T)), rowGap, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(local, gm[offset], copy, pad);

}

template <class T,
          typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaCopyAux(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, GM_ADDR address, uint64_t head, int64_t token,
                                   uint32_t slot, uint32_t rows, bool beta)
{
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer((__gm__ T *)address);
    if (beta) {
        // MTE2's short rows land at 32-byte intervals, inside ctx.betaSlot_.
        RecurrentKdaCopyRows(ctx, gm, uint64_t(token) * ctx.td_->nv + head, ctx.td_->nv,
                 ctx.betaOffset_ + slot * ctx.betaSlot_, 1, rows);
    } else {
        RecurrentKdaCopyRows(ctx, gm, (uint64_t(token) * ctx.td_->nv + head) * RKDA_DIM_128, uint64_t(ctx.td_->nv) * RKDA_DIM_128,
                 ctx.gateOffset_ + slot * RKDA_ROW_FP32_BYTES_512 * ctx.x_, RKDA_DIM_128, rows);
    }

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline uint64_t RecurrentKdaInputOffset(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint64_t token, uint64_t tokenStride,
                                         uint64_t batchStride)
{
    if (ctx.td_->hasCuSeqlens) return token * tokenStride;
    return (token / ctx.td_->seqLen) * batchStride + (token % ctx.td_->seqLen) * tokenStride;

}

// 输入：当前tile的GM输入
// 输出：指定UB输入槽。
// 复用：覆盖前由编排层回收slot
// 同步：消费者使用前等待MTE2_V EVENT0。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage0PrefetchInputs(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, int64_t token, uint32_t rows, uint64_t head, uint32_t slot)
{
    const uint64_t hk = head / (ctx.td_->nv / ctx.td_->nk);
    RecurrentKdaCopyRows(ctx, ctx.q_, RecurrentKdaInputOffset(ctx, uint64_t(token), ctx.td_->queryTokenStride, ctx.td_->queryBatchStride) + hk * ctx.td_->queryHeadStride,
             ctx.td_->queryTokenStride, ctx.qOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_, RKDA_DIM_128, rows);
    RecurrentKdaCopyRows(ctx, ctx.k_, RecurrentKdaInputOffset(ctx, uint64_t(token), ctx.td_->keyTokenStride, ctx.td_->keyBatchStride) + hk * ctx.td_->keyHeadStride,
             ctx.td_->keyTokenStride, ctx.kOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_, RKDA_DIM_128, rows);
    RecurrentKdaCopyRows(ctx, ctx.v_, RecurrentKdaInputOffset(ctx, uint64_t(token), ctx.td_->valueTokenStride, ctx.td_->valueBatchStride) + head * ctx.td_->valueHeadStride,
             ctx.td_->valueTokenStride, ctx.vOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_, RKDA_DIM_128, rows);
#if defined(DTYPE_GATE) && defined(DTYPE_BETA)
    // The OPP binary's input types also determine the DMA width and strides.
    RecurrentKdaCopyAux<DTYPE_GATE>(ctx, ctx.args_.gate, head, token, slot, rows, false);
    RecurrentKdaCopyAux<DTYPE_BETA>(ctx, ctx.args_.beta, head, token, slot, rows, true);
#else
    if (ctx.td_->gateDtype == 0) RecurrentKdaCopyAux<float>(ctx, ctx.args_.gate, head, token, slot, rows, false);
    else if (ctx.td_->gateDtype == 1) RecurrentKdaCopyAux<bfloat16_t>(ctx, ctx.args_.gate, head, token, slot, rows, false);
    else RecurrentKdaCopyAux<half>(ctx, ctx.args_.gate, head, token, slot, rows, false);
    if (ctx.td_->betaDtype == 0) RecurrentKdaCopyAux<float>(ctx, ctx.args_.beta, head, token, slot, rows, true);
    else if (ctx.td_->betaDtype == 1) RecurrentKdaCopyAux<bfloat16_t>(ctx, ctx.args_.beta, head, token, slot, rows, true);
    else RecurrentKdaCopyAux<half>(ctx, ctx.args_.beta, head, token, slot, rows, true);
#endif

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline uint32_t RecurrentKdaValidTokens(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, int64_t begin, int64_t end)
{
    return uint64_t(end - begin) < ctx.x_ ? uint32_t(end - begin) : ctx.x_;

}

// 输入：完整tile的UB out
// 输出：GM out。
// 复用：完成前保持源slot
// 同步：上游V_S/S_MTE3，完成后MTE3_S。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage2WriteOutputTile(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, int64_t token, uint32_t rows, uint64_t head, uint32_t slot)
{
    // UB rows are contiguous; GM rows for one value head are Nv rows apart.
    DataCopyExtParams copy{static_cast<uint16_t>(rows), RKDA_DIM_128 * sizeof(outType), 0,
        static_cast<uint32_t>(uint64_t(ctx.td_->nv - 1) * RKDA_DIM_128 * sizeof(outType)), 0};
    DataCopyPad(ctx.out_[(uint64_t(token) * ctx.td_->nv + head) * RKDA_DIM_128],
                ctx.ub_[ctx.outOffset_ + slot * ctx.x_ * RKDA_ROW_BF16_BYTES_256].template ReinterpretCast<outType>(), copy);

}

// 输入：任务初态和参数
// 输出：指定bank/参数槽。
// 复用：只写可用bank
// 同步：消费者通过MTE2_V EVENT1等待。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage0PrefetchState(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, const RKDATask &task, uint32_t bank, uint32_t parameterSlot)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    int64_t initialToken = task.begin;
    if (ctx.td_->hasAcceptedTokens)
        initialToken += RecurrentKdaMetadata(ctx, ctx.args_.numAcceptedTokens, ctx.td_->acceptedTokensDtype, task.request) - 1;
    const uint64_t initial = RecurrentKdaSlot(ctx, task.request, task.begin, initialToken);
    RecurrentKdaCopyRows(ctx, ctx.stateIn_, initial * ctx.td_->stateInStride0 + task.head * ctx.td_->stateInStride1,
             ctx.td_->stateInStride2, bank * BANK, RKDA_DIM_128, RKDA_DIM_128, StateLayout::ROW_GAP);
    const uint32_t parameter = ctx.paramOffset_ + parameterSlot * RKDA_PARAM_SLOT_BYTES_544;
    if (ctx.td_->hasALog) {
        RecurrentKdaCopyRows(ctx, ctx.aLog_, task.head, 1, parameter, 1, 1);
        if (ctx.td_->hasDtBias) RecurrentKdaCopyRows(ctx, ctx.bias_, task.head * RKDA_DIM_128, RKDA_DIM_128, parameter + 32, RKDA_DIM_128, 1);
    }

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaTrace(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint64_t event, uint64_t task, uint64_t token, uint64_t slot)
{
#ifdef RKDA_EXECUTION_TRACE
    const uint64_t row = 1 + ctx.traceCount_ * 8;
    ctx.trace_.SetValue(row, event); ctx.trace_.SetValue(row + 1, task);
    ctx.trace_.SetValue(row + 2, token); ctx.trace_.SetValue(row + 3, slot);
    ctx.trace_.SetValue(row + 4, GetBlockNum());
    ctx.trace_.SetValue(row + 5, event == 7 || event == 8 ? slot : ((token & 1) ? ctx.alternateBank_ : ctx.initialBank_));
    auto events = (__ubuf__ volatile uint32_t *)ctx.ub_[ctx.eventOffset_].GetPhyAddr();
    ctx.trace_.SetValue(row + 6, events[0]);
    ctx.trace_.SetValue(row + 7, event == 1 ? ctx.alternateBank_ : events[8]);
    ctx.trace_.SetValue(0, ++ctx.traceCount_);
#endif

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage0PrefetchNextState(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, const RKDATask &next, uint32_t parameterSlot, uint32_t bank)
{
    ctx.bankUse_[bank] = RKDA_BANK_PREFETCHED;
    ctx.prefetchedBank_ = bank;
    Stage0PrefetchState(ctx, next, bank, parameterSlot ^ 1);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    RecurrentKdaTrace(ctx, 7, next.id, 0, bank);

}

template <class Gate, class Beta,
          typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaDispatchLayout(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint32_t base, uint32_t valid, uint32_t slot,
                                         uint32_t parameter, bool last)
{
    if constexpr (stateVFirst) {
        if (ctx.x_ == 1) {
            if (ctx.td_->useQkL2norm && ctx.td_->hasALog && ctx.td_->hasDtBias && ctx.td_->useBetaSigmoid &&
                !ctx.td_->allowNegEigval && !ctx.td_->safeGate && !ctx.td_->hasSsmStateIndices) {
                Stage1LaunchRecurrentKdaVF<Gate, Beta, true, true, 0, true>(ctx, base, valid, slot, parameter, last);
            } else Stage1LaunchRecurrentKdaVF<Gate, Beta, true, false, 0, true>(ctx, base, valid, slot, parameter, last);
            return;
        }
        // Specialize the common fused recurrence to retire runtime flags
        // before the VF's State/out row loops. All other flags use the generic VF.
        if (ctx.td_->useQkL2norm && ctx.td_->hasALog && ctx.td_->hasDtBias && ctx.td_->useBetaSigmoid &&
            !ctx.td_->allowNegEigval && !ctx.td_->safeGate && !ctx.td_->hasSsmStateIndices) {
            if (ctx.x_ == 4 && valid == 4) Stage1LaunchRecurrentKdaVF<Gate, Beta, true, true, 4>(ctx, base, valid, slot, parameter, last);
            else if (ctx.x_ == 8 && valid == 8) Stage1LaunchRecurrentKdaVF<Gate, Beta, true, true, 8>(ctx, base, valid, slot, parameter, last);
            else Stage1LaunchRecurrentKdaVF<Gate, Beta, true, true>(ctx, base, valid, slot, parameter, last);
        } else if (ctx.x_ == 8 && valid == 8) {
            Stage1LaunchRecurrentKdaVF<Gate, Beta, true, false, 8>(ctx, base, valid, slot, parameter, last);
        } else Stage1LaunchRecurrentKdaVF<Gate, Beta, true>(ctx, base, valid, slot, parameter, last);
    } else Stage1LaunchRecurrentKdaVF<Gate, Beta, false>(ctx, base, valid, slot, parameter, last);

}

template <class Gate,
          typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaDispatchBeta(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint32_t base, uint32_t valid, uint32_t slot,
                                       uint32_t parameter, bool last)
{
    if (ctx.td_->betaDtype == 0) RecurrentKdaDispatchLayout<Gate, float>(ctx, base, valid, slot, parameter, last);
    else if (ctx.td_->betaDtype == 1) RecurrentKdaDispatchLayout<Gate, bfloat16_t>(ctx, base, valid, slot, parameter, last);
    else RecurrentKdaDispatchLayout<Gate, half>(ctx, base, valid, slot, parameter, last);

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaDispatch(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint32_t base, uint32_t valid, uint32_t slot,
                                    uint32_t parameter, bool last)
{
#if defined(DTYPE_GATE) && defined(DTYPE_BETA)
    // OPP already selects a binary for each input dtype combination.
    RecurrentKdaDispatchLayout<DTYPE_GATE, DTYPE_BETA>(ctx, base, valid, slot, parameter, last);
#else
    // Direct kernel callers may supply dtype codes only through tiling.
    if (ctx.td_->gateDtype == 0) RecurrentKdaDispatchBeta<float>(ctx, base, valid, slot, parameter, last);
    else if (ctx.td_->gateDtype == 1) RecurrentKdaDispatchBeta<bfloat16_t>(ctx, base, valid, slot, parameter, last);
    else RecurrentKdaDispatchBeta<half>(ctx, base, valid, slot, parameter, last);
#endif

}

// 输入：当前任务及lookahead
// 输出：完整tile out/末态。
// 复用：交替输入/out槽和动态bank
// 同步：保留MTE2_V、S_V、V_S、S_MTE3和MTE3_S。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage1RunFinalStateTask(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, const RKDATask &current, const RKDATask *nextTask,
                                           bool prefetched, uint32_t parameterSlot,
                                           uint32_t &slot, bool &pending, uint64_t &cursor,
                                           RKDATask &following, bool &hasFollowing, bool &validTask,
                                           const DataCopyExtParams &stateCopy)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    const int64_t begin = current.begin, end = current.end;
    const uint64_t head = current.head, task = current.id;
    const uint32_t finalBank = ((end - begin) & 1) ? ctx.alternateBank_ : ctx.initialBank_;
    const bool writeState = ctx.td_->inplaceFinalState || ctx.td_->outputFinalState;
    const uint64_t stateOffset = current.request * ctx.td_->stateOutStride0 + head * ctx.td_->stateOutStride1;
    const uint32_t parameter = ctx.paramOffset_ + parameterSlot * RKDA_PARAM_SLOT_BYTES_544;
    auto events = (__ubuf__ volatile uint32_t *)ctx.ub_[ctx.eventOffset_].GetPhyAddr();
    events[0] = 0;
    hasFollowing = false;
    RecurrentKdaTrace(ctx, 0, task, 0, slot);
    if (prefetched) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        RecurrentKdaTrace(ctx, 8, task, 0, ctx.initialBank_);
    } else {
        Stage0PrefetchState(ctx, current, ctx.initialBank_, parameterSlot);
        Stage0PrefetchInputs(ctx, begin, RecurrentKdaValidTokens(ctx, begin, end), head, slot);
    }
    // The spare bank can still be read by the previous task's final DMA.
    bool deferStatePrefetch = nextTask && RecurrentKdaFreeBank(ctx) == RKDA_NO_BANK;
    for (int64_t tile = begin; tile < end; tile += ctx.x_, slot ^= 1) {
        const uint32_t valid = RecurrentKdaValidTokens(ctx, tile, end), base = uint32_t(tile - begin);
        const bool lastTile = tile + valid == end;
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        RecurrentKdaTrace(ctx, 1, task, base, slot);
        RecurrentKdaDispatch(ctx, base, valid, slot, parameter, lastTile);
        // Enqueue future inputs while the current slot is consumed by VF.
        if (!lastTile) {
            Stage0PrefetchInputs(ctx, tile + valid, RecurrentKdaValidTokens(ctx, tile + valid, end), head, slot ^ 1);
        } else if (nextTask) {
            Stage0PrefetchInputs(ctx, nextTask->begin, RecurrentKdaValidTokens(ctx, nextTask->begin, nextTask->end), nextTask->head, slot ^ 1);
            RecurrentKdaTrace(ctx, 9, nextTask->id, 0, slot ^ 1);
        }
        if (nextTask && base == 0 && !deferStatePrefetch)
            Stage0PrefetchNextState(ctx, *nextTask, parameterSlot, RecurrentKdaFreeBank(ctx));
        if (base == 0 && nextTask)
            hasFollowing = RecurrentKdaPrepareTask(ctx, cursor, following, validTask);
        // The old batch belongs to the opposite out slot. Reclaim it while
        // this VF runs, also releasing any old final-State DMA source bank.
        if (pending) {
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            pending = false;
            RecurrentKdaReleaseStateDma(ctx);
            RecurrentKdaTrace(ctx, 10, task, base, slot ^ 1);
        }
        if (deferStatePrefetch) {
            Stage0PrefetchNextState(ctx, *nextTask, parameterSlot, RecurrentKdaFreeBank(ctx));
            deferStatePrefetch = false;
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        RecurrentKdaTrace(ctx, 5, task, base + valid, slot);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        Stage2WriteOutputTile(ctx, tile, valid, head, slot);
        if (lastTile && writeState) {
            DataCopyPad(ctx.stateOut_[stateOffset], ctx.ub_[finalBank * BANK].template ReinterpretCast<stateType>(), stateCopy);
            ctx.bankUse_[finalBank] |= RKDA_BANK_MTE3_SOURCE;
            ctx.pendingStateBank_ = finalBank;
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        pending = true;
        RecurrentKdaTrace(ctx, 11, task, base + valid, slot);
    }
    RecurrentKdaTrace(ctx, 6, task, uint32_t(end - begin), slot);

}

// 输入：无indices任务
// 输出：tile out及最终State。
// 复用：旧末态DMA可跨任务，源bank完成前不得覆盖
// 同步：MTE3_S回收及尾部排空。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void RecurrentKdaProcessFinalState(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    uint64_t cursor = GetBlockIdx();
    bool valid = true, prefetched = false, pending = false;
    RKDATask current{}, next{}, following{};
    bool active = RecurrentKdaPrepareTask(ctx, cursor, current, valid);
    if (!active || !valid) return;
    bool hasNext = RecurrentKdaPrepareTask(ctx, cursor, next, valid), hasFollowing = false;
    if (!valid) return;
    uint32_t initial = 0, parameterSlot = 0, inputSlot = 0;
    // The output State row shape is invariant across all tasks on this core.
    const DataCopyExtParams stateCopy{RKDA_DIM_128, RKDA_DIM_128 * sizeof(stateType), StateLayout::ROW_GAP,
        static_cast<uint32_t>((ctx.td_->stateOutStride2 - RKDA_DIM_128) * sizeof(stateType)), 0};
#ifdef RKDA_EXECUTION_TRACE
    auto events = (__ubuf__ volatile uint32_t *)ctx.ub_[ctx.eventOffset_].GetPhyAddr();
    events[8] = 0; // No bank-done consumer on this path.
#endif
    while (active) {
        if (!valid) {
            if (prefetched) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if (pending) WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            return;
        }
        RecurrentKdaBeginBanks(ctx, initial);
        Stage1RunFinalStateTask(ctx, current, hasNext ? &next : nullptr, prefetched,
                          parameterSlot, inputSlot, pending, cursor, following, hasFollowing, valid, stateCopy);
        if (!valid) {
            if (hasNext) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if (pending) WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            return;
        }
        RecurrentKdaReleaseBanks(ctx);
        if (!hasNext) break;
        initial = ctx.prefetchedBank_;
        current = next;
        next = following;
        hasNext = hasFollowing;
        parameterSlot ^= 1;
        prefetched = true;
    }
    // The final DMA has no subsequent VF consumer; do not publish unused done.
    if (pending) {
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        RecurrentKdaTrace(ctx, 12, current.id, uint32_t(current.end - current.begin), inputSlot ^ 1);
    }

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline bool RecurrentKdaValidateSequences(const RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx)
{
    if (!ctx.td_->hasCuSeqlens) return true;
    if (RecurrentKdaSequence(ctx, 0) != 0) return false;
    int64_t previous = 0;
    for (uint64_t request = 0; request < ctx.td_->b; ++request) {
        int64_t end = RecurrentKdaSequence(ctx, request + 1);
        if (end < previous || end > ctx.td_->t ||
            (ctx.td_->hasSsmStateIndices && ctx.td_->ssmStateStride &&
             uint64_t(end - previous) > ctx.td_->ssmStateStride)) return false;
        previous = end;
    }
    return true;

}

template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage2CompleteStateDma(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint32_t count, uint64_t task)
{
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    RecurrentKdaReleaseStateDma(ctx);
    auto events = (__ubuf__ volatile uint32_t *)ctx.ub_[ctx.eventOffset_].GetPhyAddr();
    events[8] = count;
    RecurrentKdaTrace(ctx, 4, task, count, 0);

}

// 输入：有indices任务
// 输出：逐token State快照及tile out。
// 复用：两个递推bank、可用第三bank预取
// 同步：token发布/完成计数及独立out完成事件。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage1RunIndexedTask(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, const RKDATask &current, const RKDATask *nextTask, bool prefetched,
                                  uint32_t parameterSlot, uint32_t &nextInitialBank, uint32_t &slot)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    const uint64_t task = current.id, request = current.request, head = current.head;
    const int64_t begin = current.begin, end = current.end;
    auto events = (__ubuf__ volatile uint32_t *)ctx.ub_[ctx.eventOffset_].GetPhyAddr();
    events[0] = 0; // monotonically increasing VF publication count
    events[8] = 0; // ordered MTE3 completion count, in a separate 32B block
    RecurrentKdaTrace(ctx, 0, task, 0, 0);
    if (prefetched) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        RecurrentKdaTrace(ctx, 8, task, 0, ctx.initialBank_);
    } else {
        Stage0PrefetchState(ctx, current, ctx.initialBank_, parameterSlot);
    }
    const uint32_t parameter = ctx.paramOffset_ + parameterSlot * RKDA_PARAM_SLOT_BYTES_544;
    if (!prefetched) Stage0PrefetchInputs(ctx, begin, RecurrentKdaValidTokens(ctx, begin, end), head, slot);
    uint32_t pending = 0;
    bool pendingOut = false;
    const bool writeState = ctx.td_->inplaceFinalState || ctx.td_->outputFinalState;
    const DataCopyExtParams stateCopy{RKDA_DIM_128, RKDA_DIM_128 * sizeof(stateType), StateLayout::ROW_GAP,
        static_cast<uint32_t>((ctx.td_->stateOutStride2 - RKDA_DIM_128) * sizeof(stateType)), 0};
    for (int64_t tile = begin; tile < end; tile += ctx.x_, slot ^= 1) {
        const uint32_t valid = RecurrentKdaValidTokens(ctx, tile, end), base = uint32_t(tile - begin);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        const bool lastTile = tile + valid == end;
        RecurrentKdaTrace(ctx, 1, task, base, slot);
        RecurrentKdaDispatch(ctx, base, valid, slot, parameter, lastTile);
        // Only the opposite input slot and a free State bank are prefetched.
        if (!lastTile) {
            Stage0PrefetchInputs(ctx, tile + valid, RecurrentKdaValidTokens(ctx, tile + valid, end), head, slot ^ 1);
        } else if (nextTask) {
            Stage0PrefetchInputs(ctx, nextTask->begin, RecurrentKdaValidTokens(ctx, nextTask->begin, nextTask->end), nextTask->head, slot ^ 1);
            RecurrentKdaTrace(ctx, 9, nextTask->id, 0, slot ^ 1);
        }
        // Two banks alternate throughout this task; the third belongs to the next task.
        if (nextTask && base == 0) {
            nextInitialBank = RecurrentKdaFreeBank(ctx);
            Stage0PrefetchNextState(ctx, *nextTask, parameterSlot, nextInitialBank);
        }
        // State snapshots remain token-granular: the VF can need a bank
        // released before it can finish the tile. Out has its own event.
        for (uint32_t t = 0; writeState && t < valid; ++t) {
            const uint32_t token = base + t, count = token + 1;
            // Release an old snapshot before waiting for the next publication.
            // The whole tile VF is already running asynchronously.
            if (pending) { Stage2CompleteStateDma(ctx, pending, task); pending = 0; }
            while (events[0] < count) {}
            RecurrentKdaTrace(ctx, 2, task, count, slot);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            {
                uint32_t bank = ((token + 1) & 1) ? ctx.alternateBank_ : ctx.initialBank_;
                const uint64_t destination = RecurrentKdaSlot(ctx, request, begin, tile + t) * ctx.td_->stateOutStride0 +
                                             head * ctx.td_->stateOutStride1;
                DataCopyPad(ctx.stateOut_[destination], ctx.ub_[bank * BANK].template ReinterpretCast<stateType>(), stateCopy);
                ctx.bankUse_[bank] |= RKDA_BANK_MTE3_SOURCE;
                ctx.pendingStateBank_ = bank;
            }
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            pending = count;
            RecurrentKdaTrace(ctx, 3, task, count, slot);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        RecurrentKdaTrace(ctx, 5, task, base + valid, slot);
        // Reclaim the previous tile's out, never treating State-done as
        // proof that an out slot is free. EVENT1 is distinct from State's 0.
        if (pendingOut) {
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
            RecurrentKdaTrace(ctx, 12, task, base, slot ^ 1);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        Stage2WriteOutputTile(ctx, tile, valid, head, slot);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        pendingOut = true;
        RecurrentKdaTrace(ctx, 11, task, base + valid, slot);
    }
    if (pending) Stage2CompleteStateDma(ctx, pending, task);
    if (pendingOut) {
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        RecurrentKdaTrace(ctx, 12, task, uint32_t(end - begin), slot ^ 1);
    }
    RecurrentKdaTrace(ctx, 6, task, uint32_t(end - begin), slot);

}

// 输入：已初始化context
// 输出：本核全部任务。
// 复用：输入/out双槽、两递推bank及一个可用预取bank
// 同步：各Stage内部成对事件与尾部排空。
template <typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void ProcessRecurrentKda(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx)
{
    if (ctx.x_ == 0 || 32 * ctx.x_ > ctx.betaSlot_ || !RecurrentKdaValidateSequences(ctx)) return;
    if (!ctx.td_->hasSsmStateIndices) { RecurrentKdaProcessFinalState(ctx); return; }
    uint64_t cursor = GetBlockIdx();
    bool valid = true;
    RKDATask current{}, next{};
    bool active = RecurrentKdaPrepareTask(ctx, cursor, current, valid);
    uint32_t initialBank = 0, parameterSlot = 0, inputSlot = 0;
    bool prefetched = false;
    while (active) {
        const bool hasNext = RecurrentKdaPrepareTask(ctx, cursor, next, valid);
        if (!valid) {
            if (prefetched) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            return;
        }
        RecurrentKdaBeginBanks(ctx, initialBank);
        Stage1RunIndexedTask(ctx, current, hasNext ? &next : nullptr, prefetched, parameterSlot, initialBank, inputSlot);
        RecurrentKdaReleaseBanks(ctx);
        if (!hasNext) break;
        current = next;
        parameterSlot ^= 1;
        prefetched = true;
    }

}

} // namespace RecurrentKda
