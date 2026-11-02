// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "recurrent_kda_vec.h"

/*
Stage | 生产者/消费者 | 同步
  0   | MTE2 将输入、初态、参数搬到UB | 对应资源的Mutex Lock/Unlock
  1   | VF 消费UB并递推State、收集out | 输入/参数/out按槽；State区分访问与快照
  2   | MTE3 搬出整tile out和State   | out槽或State快照Mutex

UB：3个State bank，三个bank轮换递推，末token期间向未参与的bank预取下一初态；固定X=8。
输入、参数和out双缓冲，沿用原偏移与峰值；guard尾部保留512B数值scratch。

资源 | Mutex份数 | 生命周期与反向依赖
State访问 | 3 | MTE2初始化、VF源/目标；保护旧VF读完后才覆盖
State快照 | 3 | MTE2初始化、VF目标、MTE3搬出；VF源读取不持此锁
输入 | 2 | MTE2搬入、V消费完整tile；下次MTE2覆盖自动等待
参数 | 2 | MTE2搬入、V消费完整任务
out | 2 | V生成完整tile、MTE3批量搬出
共12个ID由AllocMutexID申请，核尾部通过V获取所有资源后ReleaseMutexID。
tile边界不等待Scalar；各消费流水通过资源Mutex等待生产完成。
同流水的访存依赖仍使用LocalMemBar/PipeBarrier，不用Mutex替代。
*/
namespace RecurrentKda {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// 输入：GM地址和host tiling；只接线、预留容量，不提交DMA。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void InitRecurrentKda(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RecurrentKdaTilingData *tiling, const RKDAInitParams &args)
{
    ctx.td_ = tiling;
    ctx.args_ = args;
    ctx.vfFlags_ = ctx.td_->a5VfFlags;
    for (uint32_t i = 0; i < RKDA_MUTEX_COUNT; ++i) ctx.mutex_[i] = AllocMutexID();
    // Fixed per-specialization UB ownership, without TPipe buffer allocation.
    ctx.ub_ = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ctx.eventOffset_ + RKDA_GUARD_BYTES_2048);
    ctx.q_.SetGlobalBuffer((__gm__ inType *)args.query);
    ctx.k_.SetGlobalBuffer((__gm__ inType *)args.key);
    ctx.v_.SetGlobalBuffer((__gm__ inType *)args.value);
    ctx.stateIn_.SetGlobalBuffer((__gm__ stateType *)args.initState);
    ctx.stateOut_.SetGlobalBuffer((__gm__ stateType *)args.finalState);
    ctx.out_.SetGlobalBuffer((__gm__ outType *)args.attnOut);
    ctx.aLog_.SetGlobalBuffer((__gm__ float *)args.aLog);
    ctx.bias_.SetGlobalBuffer((__gm__ float *)args.dtBias);
    // Row gaps and widths are invariant for this invocation; only rows change.
    ctx.queryCopy_ = {0, RKDA_DIM_128 * sizeof(inType),
        static_cast<uint32_t>((ctx.td_->queryTokenStride - RKDA_DIM_128) * sizeof(inType)), 0, 0};
    ctx.keyCopy_ = {0, RKDA_DIM_128 * sizeof(inType),
        static_cast<uint32_t>((ctx.td_->keyTokenStride - RKDA_DIM_128) * sizeof(inType)), 0, 0};
    ctx.valueCopy_ = {0, RKDA_DIM_128 * sizeof(inType),
        static_cast<uint32_t>((ctx.td_->valueTokenStride - RKDA_DIM_128) * sizeof(inType)), 0, 0};
#if defined(DTYPE_GATE) && defined(DTYPE_BETA)
    constexpr uint32_t gateBytes = sizeof(DTYPE_GATE), betaBytes = sizeof(DTYPE_BETA);
#else
    const uint32_t gateBytes = ctx.td_->gateDtype == 0 ? sizeof(float) : sizeof(half);
    const uint32_t betaBytes = ctx.td_->betaDtype == 0 ? sizeof(float) : sizeof(half);
#endif
    ctx.gateCopy_ = {0, RKDA_DIM_128 * gateBytes,
        (ctx.td_->nv - 1) * RKDA_DIM_128 * gateBytes, 0, 0};
    ctx.betaCopy_ = {0, betaBytes, (ctx.td_->nv - 1) * betaBytes, 0, 0};
    ctx.outputCopy_ = {0, RKDA_DIM_128 * sizeof(outType), 0,
        static_cast<uint32_t>((ctx.td_->nv - 1) * RKDA_DIM_128 * sizeof(outType)), 0};


}

#ifdef RKDA_EXECUTION_TRACE
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaInitExecutionTrace(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, GM_ADDR records)
{
    ctx.trace_.SetGlobalBuffer((__gm__ uint64_t *)records + GetBlockIdx() * 32768);
    ctx.traceCount_ = 0;
    ctx.trace_.SetValue(0, 0);

}

#endif
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaBeginBanks(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint32_t initial)
{
    ctx.initialBank_ = initial;
    ctx.currentBank_ = initial;
    ctx.nextBank_ = initial == 2 ? 0 : initial + 1;
}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaAdvanceBank(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx)
{
    ctx.currentBank_ = ctx.nextBank_;
    ctx.nextBank_ = ctx.nextBank_ == 2 ? 0 : ctx.nextBank_ + 1;
}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline int64_t RecurrentKdaMetadata(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, GM_ADDR address, uint32_t dtype, uint64_t index)
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

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline int64_t RecurrentKdaSequence(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint64_t request)
{
    return ctx.td_->hasCuSeqlens ? RecurrentKdaMetadata(ctx, ctx.args_.cuSeqlens, ctx.td_->cuSeqlensDtype, request) :
                               int64_t(request * ctx.td_->seqLen);

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline uint64_t RecurrentKdaSlot(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint64_t request, int64_t begin, int64_t token)
{
    if (!ctx.td_->hasSsmStateIndices) return request;
    const uint64_t offset = ctx.td_->ssmStateStride ?
        request * ctx.td_->ssmStateStride + uint64_t(token - begin) : uint64_t(token);
    const int64_t slot = RecurrentKdaMetadata(ctx, ctx.args_.ssmStateIndices, ctx.td_->ssmStateIndicesDtype, offset);
    return slot >= 0 && uint64_t(slot) < ctx.td_->sBlockNum ? uint64_t(slot) : RKDA_INVALID_SLOT;

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline bool RecurrentKdaValidateSlots(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint64_t request, int64_t begin, int64_t end)
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

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline bool RecurrentKdaPrepareTask(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint64_t &cursor, RKDATask &task, bool &valid)
{
    const uint64_t count = uint64_t(ctx.td_->b) * ctx.td_->nv;
    while (cursor < count) {
        const uint64_t id = cursor;
        cursor += GetBlockNum();
        const uint64_t request = id / ctx.td_->nv;
        const int64_t begin = RecurrentKdaSequence(ctx, request);
        const int64_t end = RecurrentKdaSequence(ctx, request + 1);
        if (begin == end) continue;
        if (!RecurrentKdaValidateSlots(ctx, request, begin, end)) {
            valid = false; return false;
        }
        int64_t initialToken = begin;
        if (ctx.td_->hasAcceptedTokens)
            initialToken += RecurrentKdaMetadata(ctx, ctx.args_.numAcceptedTokens,
                ctx.td_->acceptedTokensDtype, request) - 1;
        const uint64_t initial = RecurrentKdaSlot(ctx, request, begin, initialToken);
        if (initial == RKDA_INVALID_SLOT) { valid = false; return false; }
        task = {id, request, id % ctx.td_->nv, begin, end, initial};
        return true;
    }
    return false;

}

template <class T,
          typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaCopyRows(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, GlobalTensor<T> &gm, uint64_t offset, uint64_t stride,
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
          typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaCopyInputRows(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx,
    GlobalTensor<T> &gm, uint64_t offset, uint32_t destination, uint32_t rows, DataCopyExtParams &copy)
{
    copy.blockCount = static_cast<uint16_t>(rows);
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(ctx.ub_[destination].template ReinterpretCast<T>(), gm[offset], copy, pad);
}

template <class T,
          typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaCopyAux(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, GM_ADDR address, uint64_t head, int64_t token,
                                   uint32_t slot, uint32_t rows, bool beta)
{
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer((__gm__ T *)address);
    if (beta) {
        // MTE2's short rows land at 32-byte intervals, inside ctx.betaSlot_.
        RecurrentKdaCopyInputRows(ctx, gm, uint64_t(token) * ctx.td_->nv + head,
                 ctx.betaOffset_ + slot * ctx.betaSlot_, rows, ctx.betaCopy_);
    } else {
        RecurrentKdaCopyInputRows(ctx, gm, (uint64_t(token) * ctx.td_->nv + head) * RKDA_DIM_128,
                 ctx.gateOffset_ + slot * RKDA_ROW_FP32_BYTES_512 * ctx.x_, rows, ctx.gateCopy_);
    }

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline uint64_t RecurrentKdaInputOffset(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RKDATask &task,
                                         uint64_t token, uint64_t tokenStride, uint64_t batchStride)
{
    if (ctx.td_->hasCuSeqlens || ctx.td_->layout == 1) return token * tokenStride;
    // Task preparation already resolved the request and its first token.
    // Preserve independent batch strides without dividing the flat token again.
    return task.request * batchStride + (token - uint64_t(task.begin)) * tokenStride;

}

// 输入：当前tile的GM输入
// 输出：指定UB输入槽。
// 复用：覆盖前由编排层回收slot
// 同步：输入槽Mutex由MTE2交给Vector。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage0PrefetchInputs(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RKDATask &task,
                                         int64_t token, uint32_t rows, uint32_t slot)
{
    Mutex::Lock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_INPUT + slot]);
    const uint64_t head = task.head;
    const uint64_t hk = head / ctx.td_->a5HeadRatio;
    RecurrentKdaCopyInputRows(ctx, ctx.q_, RecurrentKdaInputOffset(ctx, task, uint64_t(token), ctx.td_->queryTokenStride, ctx.td_->queryBatchStride) + hk * ctx.td_->queryHeadStride,
             ctx.qOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_, rows, ctx.queryCopy_);
    RecurrentKdaCopyInputRows(ctx, ctx.k_, RecurrentKdaInputOffset(ctx, task, uint64_t(token), ctx.td_->keyTokenStride, ctx.td_->keyBatchStride) + hk * ctx.td_->keyHeadStride,
             ctx.kOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_, rows, ctx.keyCopy_);
    RecurrentKdaCopyInputRows(ctx, ctx.v_, RecurrentKdaInputOffset(ctx, task, uint64_t(token), ctx.td_->valueTokenStride, ctx.td_->valueBatchStride) + head * ctx.td_->valueHeadStride,
             ctx.vOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_, rows, ctx.valueCopy_);
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
    Mutex::Unlock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_INPUT + slot]);

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline uint32_t RecurrentKdaValidTokens(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, int64_t begin, int64_t end)
{
    return uint64_t(end - begin) < ctx.x_ ? uint32_t(end - begin) : ctx.x_;

}

// 输入：完整tile的UB out
// 输出：GM out。
// 复用：完成前保持源slot
// 同步：out槽Mutex保护Vector生产、MTE3消费和再次覆盖。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage2WriteOutputTile(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, int64_t token, uint32_t rows, uint64_t head, uint32_t slot)
{
    Mutex::Lock<PIPE_MTE3>(ctx.mutex_[RKDA_MUTEX_OUT + slot]);
    // UB rows are contiguous; GM rows for one value head are Nv rows apart.
    ctx.outputCopy_.blockCount = static_cast<uint16_t>(rows);
    DataCopyPad(ctx.out_[(uint64_t(token) * ctx.td_->nv + head) * RKDA_DIM_128],
                ctx.ub_[ctx.outOffset_ + slot * ctx.x_ * RKDA_ROW_BF16_BYTES_256].template ReinterpretCast<outType>(), ctx.outputCopy_);
    Mutex::Unlock<PIPE_MTE3>(ctx.mutex_[RKDA_MUTEX_OUT + slot]);

}

// 输入：任务初态
// 输出：指定bank。
// 复用：只写可用bank
// 同步：State访问/快照Mutex保护初始化。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage0PrefetchState(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RKDATask &task, uint32_t bank)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    Mutex::Lock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_STATE_USE + bank]);
    Mutex::Lock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + bank]);
    const uint64_t initial = task.initialSlot;
    RecurrentKdaCopyRows(ctx, ctx.stateIn_, initial * ctx.td_->stateInStride0 + task.head * ctx.td_->stateInStride1,
             ctx.td_->stateInStride2, bank * BANK, RKDA_DIM_128, RKDA_DIM_128, StateLayout::ROW_GAP);
    Mutex::Unlock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + bank]);
    Mutex::Unlock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_STATE_USE + bank]);

}

// 参数槽独立于State bank，在当前任务首token提交下一任务参数，
// 避免参数搬入排在末token的State bank等待之后。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage0PrefetchParameters(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RKDATask &task, uint32_t parameterSlot)
{
    Mutex::Lock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_PARAM + parameterSlot]);
    const uint32_t parameter = ctx.paramOffset_ + parameterSlot * RKDA_PARAM_SLOT_BYTES_544;
    if (ctx.td_->useGateInKernel) {
        RecurrentKdaCopyRows(ctx, ctx.aLog_, task.head, 1, parameter, 1, 1);
        if (ctx.td_->hasDtBias) RecurrentKdaCopyRows(ctx, ctx.bias_, task.head * RKDA_DIM_128, RKDA_DIM_128, parameter + 32, RKDA_DIM_128, 1);
    }
    Mutex::Unlock<PIPE_MTE2>(ctx.mutex_[RKDA_MUTEX_PARAM + parameterSlot]);

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaTrace(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint64_t event, uint64_t task, uint64_t token, uint64_t slot)
{
#ifdef RKDA_EXECUTION_TRACE
    const uint64_t row = 1 + ctx.traceCount_ * 8;
    ctx.trace_.SetValue(row, event); ctx.trace_.SetValue(row + 1, task);
    ctx.trace_.SetValue(row + 2, token); ctx.trace_.SetValue(row + 3, slot);
    ctx.trace_.SetValue(row + 4, GetBlockNum());
    ctx.trace_.SetValue(row + 5, event == 7 || event == 8 ? slot : ctx.currentBank_);
    ctx.trace_.SetValue(row + 6, ctx.currentBank_ | (ctx.nextBank_ << 2) | (ctx.prefetchedBank_ << 4));
    ctx.trace_.SetValue(row + 7, event == 1 ? ctx.nextBank_ : ctx.prefetchedBank_);
    ctx.trace_.SetValue(0, ++ctx.traceCount_);
#endif

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage0PrefetchNextState(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RKDATask &next, uint32_t bank)
{
    ctx.prefetchedBank_ = bank;
    Stage0PrefetchState(ctx, next, bank);
    RecurrentKdaTrace(ctx, 7, next.id, 0, bank);

}

template <class Gate, class Beta,
          typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaDispatchLayout(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint32_t base, uint32_t slot,
                                         uint32_t parameter)
{
    Stage1LaunchRecurrentKdaVF<Gate, Beta, stateVFirst, fusedMode>(ctx, base, slot, parameter);

}

template <class Gate,
          typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaDispatchBeta(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint32_t base, uint32_t slot,
                                       uint32_t parameter)
{
    if (ctx.td_->betaDtype == 0) RecurrentKdaDispatchLayout<Gate, float>(ctx, base, slot, parameter);
    else if (ctx.td_->betaDtype == 1) RecurrentKdaDispatchLayout<Gate, bfloat16_t>(ctx, base, slot, parameter);
    else RecurrentKdaDispatchLayout<Gate, half>(ctx, base, slot, parameter);

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaDispatch(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint32_t base, uint32_t slot,
                                    uint32_t parameter)
{
    const uint32_t source = ctx.currentBank_;
    const uint32_t target = ctx.nextBank_;
    // VF reads may overlap an MTE3 read of the same source bank. Only its
    // destination takes the snapshot lock; MTE2 takes both bank locks.
    Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_STATE_USE + source]);
    Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_STATE_USE + target]);
    Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + target]);
#if defined(DTYPE_GATE) && defined(DTYPE_BETA)
    // OPP already selects a binary for each input dtype combination.
    RecurrentKdaDispatchLayout<DTYPE_GATE, DTYPE_BETA>(ctx, base, slot, parameter);
#else
    // Direct kernel callers may supply dtype codes only through tiling.
    if (ctx.td_->gateDtype == 0) RecurrentKdaDispatchBeta<float>(ctx, base, slot, parameter);
    else if (ctx.td_->gateDtype == 1) RecurrentKdaDispatchBeta<bfloat16_t>(ctx, base, slot, parameter);
    else RecurrentKdaDispatchBeta<half>(ctx, base, slot, parameter);
#endif
    Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + target]);
    Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_STATE_USE + target]);
    Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_STATE_USE + source]);

}

// 输入：当前任务及lookahead
// 输出：完整tile out/末态。
// 复用：交替输入/out槽和动态bank
// 同步：输入/参数/out槽及State两类Mutex，Scalar可连续提交后续tile。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage1RunFinalStateTask(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RKDATask &current, const RKDATask *nextTask,
                                           bool prefetched, uint32_t parameterSlot,
                                           uint32_t &slot, bool &pending, uint64_t &cursor,
                                           RKDATask &following, bool &hasFollowing, bool &validTask,
                                           const DataCopyExtParams &stateCopy)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    const int64_t begin = current.begin, end = current.end;
    const uint64_t head = current.head, task = current.id;
    const bool writeState = ctx.td_->inplaceFinalState || ctx.td_->outputFinalState;
    const uint64_t stateOffset = current.request * ctx.td_->stateOutStride0 + head * ctx.td_->stateOutStride1;
    const uint32_t parameter = ctx.paramOffset_ + parameterSlot * RKDA_PARAM_SLOT_BYTES_544;
    hasFollowing = false;
    RecurrentKdaTrace(ctx, 0, task, 0, slot);
    if (prefetched) {
        RecurrentKdaTrace(ctx, 8, task, 0, ctx.currentBank_);
    } else {
        Stage0PrefetchParameters(ctx, current, parameterSlot);
        Stage0PrefetchState(ctx, current, ctx.currentBank_);
        Stage0PrefetchInputs(ctx, current, begin, RecurrentKdaValidTokens(ctx, begin, end), slot);
    }
    Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_PARAM + parameterSlot]);
    for (int64_t tile = begin; tile < end; tile += ctx.x_, slot ^= 1) {
        const uint32_t valid = RecurrentKdaValidTokens(ctx, tile, end), base = uint32_t(tile - begin);
        const bool lastTile = tile + valid == end;
        Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_INPUT + slot]);
        Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_OUT + slot]);
        for (uint32_t t = 0; t < valid; ++t) {
            const uint32_t token = base + t;
            RecurrentKdaTrace(ctx, 1, task, token, slot);
            RecurrentKdaDispatch(ctx, token, slot, parameter);
            if (t == 0) {
                // Submit tile input before late State prefetch on the MTE2 queue.
                if (!lastTile) {
                    Stage0PrefetchInputs(ctx, current, tile + valid,
                        RecurrentKdaValidTokens(ctx, tile + valid, end), slot ^ 1);
                } else if (nextTask) {
                    Stage0PrefetchInputs(ctx, *nextTask, nextTask->begin,
                        RecurrentKdaValidTokens(ctx, nextTask->begin, nextTask->end), slot ^ 1);
                    RecurrentKdaTrace(ctx, 9, nextTask->id, 0, slot ^ 1);
                }
            }
            if (base == 0 && t == 0 && nextTask)
                Stage0PrefetchParameters(ctx, *nextTask, parameterSlot ^ 1);
            if (nextTask && lastTile && t + 1 == valid) {
                // Neither the final VF source nor its destination is overwritten.
                // The MTE2 resource locks wait for the spare bank's old VF/DMA.
                Stage0PrefetchNextState(ctx, *nextTask,
                    3 - ctx.currentBank_ - ctx.nextBank_);
            }
            RecurrentKdaAdvanceBank(ctx);
        }
        if (base == 0 && nextTask)
            hasFollowing = RecurrentKdaPrepareTask(ctx, cursor, following, validTask);
        pending = false;
        Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_INPUT + slot]);
        Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_OUT + slot]);
        RecurrentKdaTrace(ctx, 5, task, base + valid, slot);
        Stage2WriteOutputTile(ctx, tile, valid, head, slot);
        if (lastTile && writeState) {
            const uint32_t finalBank = ctx.currentBank_;
            Mutex::Lock<PIPE_MTE3>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + finalBank]);
            DataCopyPad(ctx.stateOut_[stateOffset], ctx.ub_[finalBank * BANK].template ReinterpretCast<stateType>(), stateCopy);
            Mutex::Unlock<PIPE_MTE3>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + finalBank]);
        }
        pending = true;
        RecurrentKdaTrace(ctx, 11, task, base + valid, slot);
    }
    Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_PARAM + parameterSlot]);
    RecurrentKdaTrace(ctx, 6, task, uint32_t(end - begin), slot);

}

// 输入：无indices任务
// 输出：tile out及最终State。
// 复用：旧末态DMA可跨任务，源bank完成前不得覆盖
// 同步：资源Mutex保护实际覆盖，外层统一排空并释放ID。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaProcessFinalState(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx)
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
    while (active) {
        if (!valid) {
            return;
        }
        RecurrentKdaBeginBanks(ctx, initial);
        Stage1RunFinalStateTask(ctx, current, hasNext ? &next : nullptr, prefetched,
                          parameterSlot, inputSlot, pending, cursor, following, hasFollowing, valid, stateCopy);
        if (!valid) {
            return;
        }
        if (!hasNext) break;
        initial = ctx.prefetchedBank_;
        current = next;
        next = following;
        hasNext = hasFollowing;
        parameterSlot ^= 1;
        prefetched = true;
    }
    // The outer Process drains all resource locks, including cold-start error exits.
    if (pending) {
        RecurrentKdaTrace(ctx, 12, current.id, uint32_t(current.end - current.begin), inputSlot ^ 1);
    }

}

template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline bool RecurrentKdaValidateSequences(const RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx)
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

// 输入：有indices任务；每token由VF更新State，MTE3写一份快照。
// 复用：State快照Mutex保护递推目标，访问/快照两类Mutex保护第三bank预取。
// out仍在整个tile计算完成后批量搬出，双槽使用独立完成事件。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage1RunIndexedTask(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, const RKDATask &current, const RKDATask *nextTask, bool prefetched,
                                  uint32_t parameterSlot, uint32_t &nextInitialBank, uint32_t &slot,
                                  uint32_t &pendingOut, const DataCopyExtParams &stateCopy)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    const uint64_t task = current.id, request = current.request, head = current.head;
    const int64_t begin = current.begin, end = current.end;
    RecurrentKdaTrace(ctx, 0, task, 0, slot);
    if (prefetched) {
        RecurrentKdaTrace(ctx, 8, task, 0, ctx.currentBank_);
    }
    // Source readiness is covered by the first VF's State-use lock.
    Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_PARAM + parameterSlot]);
    const uint32_t parameter = ctx.paramOffset_ + parameterSlot * RKDA_PARAM_SLOT_BYTES_544;
    const bool writeState = ctx.td_->inplaceFinalState || ctx.td_->outputFinalState;
    uint64_t previousSlot = RKDA_INVALID_SLOT, olderSlot = RKDA_INVALID_SLOT;
    for (int64_t tile = begin; tile < end; tile += ctx.x_, slot ^= 1) {
        const uint32_t valid = RecurrentKdaValidTokens(ctx, tile, end), base = uint32_t(tile - begin);
        const bool lastTile = tile + valid == end;
        Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_INPUT + slot]);
        Mutex::Lock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_OUT + slot]);
        if (pendingOut & (1U << slot)) {
            pendingOut &= ~(1U << slot);
            RecurrentKdaTrace(ctx, 12, task, base, slot);
        }
        for (uint32_t t = 0; t < valid; ++t) {
            const uint32_t token = base + t, count = token + 1;
            const uint32_t bank = ctx.nextBank_;
            RecurrentKdaTrace(ctx, 1, task, token, slot);
            RecurrentKdaDispatch(ctx, token, slot, parameter);
            if (t == 0) {
                // Prefetch into the opposite input slot while the first VF runs.
                if (!lastTile) {
                    Stage0PrefetchInputs(ctx, current, tile + valid,
                        RecurrentKdaValidTokens(ctx, tile + valid, end), slot ^ 1);
                } else if (nextTask) {
                    Stage0PrefetchInputs(ctx, *nextTask, nextTask->begin,
                        RecurrentKdaValidTokens(ctx, nextTask->begin, nextTask->end), slot ^ 1);
                    RecurrentKdaTrace(ctx, 9, nextTask->id, 0, slot ^ 1);
                }
            }
            if (base == 0 && t == 0 && nextTask)
                Stage0PrefetchParameters(ctx, *nextTask, parameterSlot ^ 1);
            if (writeState) {
                // The snapshot mutex waits for this token's stores; Scalar continues
                // issuing the next token instead of polling a done word in UB.
                Mutex::Lock<PIPE_MTE3>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + bank]);
                const uint64_t destinationSlot = RecurrentKdaSlot(ctx, request, begin, tile + t);
                // Three banks permit two older snapshots to remain in flight.
                // Serialize repeated GM destinations even if their UB banks differ.
                if (destinationSlot == previousSlot || destinationSlot == olderSlot) PipeBarrier<PIPE_MTE3>();
                olderSlot = previousSlot;
                previousSlot = destinationSlot;
                const uint64_t destination = destinationSlot * ctx.td_->stateOutStride0 +
                                             head * ctx.td_->stateOutStride1;
                DataCopyPad(ctx.stateOut_[destination], ctx.ub_[bank * BANK].template ReinterpretCast<stateType>(), stateCopy);
                Mutex::Unlock<PIPE_MTE3>(ctx.mutex_[RKDA_MUTEX_STATE_DMA + bank]);
                RecurrentKdaTrace(ctx, 3, task, count, slot);
            }
            if (nextTask && lastTile && t + 1 == valid) {
                nextInitialBank = 3 - ctx.currentBank_ - ctx.nextBank_;
                Stage0PrefetchNextState(ctx, *nextTask, nextInitialBank);
            }
            RecurrentKdaAdvanceBank(ctx);
        }
        // Resource locks keep input/output slots alive while Scalar submits
        // the next tile; MTE3 waits for all tile stores through the out lock.
        Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_INPUT + slot]);
        Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_OUT + slot]);
        Stage2WriteOutputTile(ctx, tile, valid, head, slot);
        pendingOut |= 1U << slot;
        RecurrentKdaTrace(ctx, 11, task, base + valid, slot);
    }
    Mutex::Unlock<PIPE_V>(ctx.mutex_[RKDA_MUTEX_PARAM + parameterSlot]);
    RecurrentKdaTrace(ctx, 6, task, uint32_t(end - begin), slot);
}

// 输入：已初始化context
// 输出：本核全部任务。
// 复用：输入/out双槽、三个轮换递推bank，末token向空闲bank预取
// 同步：各资源成对Lock/Unlock，外层统一排空。
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void RecurrentKdaProcessIndexed(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx)
{
    uint64_t cursor = GetBlockIdx();
    bool valid = true;
    RKDATask current{}, next{};
    bool active = RecurrentKdaPrepareTask(ctx, cursor, current, valid);
    uint32_t initialBank = 0, parameterSlot = 0, inputSlot = 0;
    uint32_t pendingOut = 0;
    bool prefetched = false;
    // Every task on this core uses the same State row descriptor.
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    const DataCopyExtParams stateCopy{RKDA_DIM_128, RKDA_DIM_128 * sizeof(stateType), StateLayout::ROW_GAP,
        static_cast<uint32_t>((ctx.td_->stateOutStride2 - RKDA_DIM_128) * sizeof(stateType)), 0};
    if (active) {
        // Current metadata is valid. Fill the first task's bank/input slot
        // while Scalar prepares and validates the next task.
        Stage0PrefetchParameters(ctx, current, parameterSlot);
        Stage0PrefetchState(ctx, current, initialBank);
        Stage0PrefetchInputs(ctx, current, current.begin,
            RecurrentKdaValidTokens(ctx, current.begin, current.end), inputSlot);
    }
    while (active) {
        const bool hasNext = RecurrentKdaPrepareTask(ctx, cursor, next, valid);
        if (!valid) {
            break;
        }
        RecurrentKdaBeginBanks(ctx, initialBank);
        Stage1RunIndexedTask(ctx, current, hasNext ? &next : nullptr, prefetched, parameterSlot, initialBank, inputSlot,
            pendingOut, stateCopy);
        if (!hasNext) break;
        current = next;
        parameterSlot ^= 1;
        prefetched = true;
    }


}

// All error/empty/tail paths return here. Drain on Vector, without a Scalar
// wait. ReleaseMutexID only frees allocator bookkeeping; no ID is reused in
// this invocation. Kernel completion retains the queued Vector/DMA lifetime.
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void ProcessRecurrentKda(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx)
{
    if (RecurrentKdaValidateSequences(ctx)) {
        if (ctx.td_->hasSsmStateIndices) RecurrentKdaProcessIndexed(ctx);
        else RecurrentKdaProcessFinalState(ctx);
    }
    for (uint32_t i = 0; i < RKDA_MUTEX_COUNT; ++i) {
        Mutex::Lock<PIPE_V>(ctx.mutex_[i]);
        Mutex::Unlock<PIPE_V>(ctx.mutex_[i]);
        ReleaseMutexID(ctx.mutex_[i]);
    }
}

} // namespace RecurrentKda
