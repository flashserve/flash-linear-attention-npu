#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && \
    (!defined(TILING_KEY_VAR) || TILING_KEY_VAR == 2UL)
#define KDA_COMPILE_ARCH35_FAST_PATH 1
#include "arch35/chunk_kda_fwd_impl.h"
#else
#define KDA_COMPILE_ARCH35_FAST_PATH 0
#endif

namespace KdaForward {

constexpr int64_t KDA_PARAM_DTYPE_BF16 = 1;

template <bool SAFE_GATE, typename G_T, typename TilingData>
__aicore__ inline void DispatchGateMode(
    GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens,
    GM_ADDR gk, const TilingData &tiling, AscendC::TPipe &pipe)
{
    if (tiling.computeGateInPrepare) {
        return;
    }
    if (!tiling.useGateInKernel) {
        RunGateCumsum<false, false, G_T, float, float>(
            g, aLog, dtBias, cuSeqlens, gk, tiling, pipe);
        return;
    }

    const bool aLogIsBf16 = tiling.aLogDataType == KDA_PARAM_DTYPE_BF16;
    const bool dtBiasIsBf16 = tiling.dtBiasDataType == KDA_PARAM_DTYPE_BF16;
    if (aLogIsBf16 && dtBiasIsBf16) {
        RunGateCumsum<true, SAFE_GATE, G_T, bfloat16_t, bfloat16_t>(
            g, aLog, dtBias, cuSeqlens, gk, tiling, pipe);
    } else if (aLogIsBf16) {
        RunGateCumsum<true, SAFE_GATE, G_T, bfloat16_t, float>(
            g, aLog, dtBias, cuSeqlens, gk, tiling, pipe);
    } else if (dtBiasIsBf16) {
        RunGateCumsum<true, SAFE_GATE, G_T, float, bfloat16_t>(
            g, aLog, dtBias, cuSeqlens, gk, tiling, pipe);
    } else {
        RunGateCumsum<true, SAFE_GATE, G_T, float, float>(
            g, aLog, dtBias, cuSeqlens, gk, tiling, pipe);
    }
}

template <typename G_T, typename TilingData>
__aicore__ inline void RunGateStage(
    GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens,
    GM_ADDR gk, const TilingData &tiling, AscendC::TPipe &pipe)
{
    if (tiling.safeGate) {
        DispatchGateMode<true, G_T>(
            g, aLog, dtBias, cuSeqlens, gk, tiling, pipe);
    } else {
        DispatchGateMode<false, G_T>(
            g, aLog, dtBias, cuSeqlens, gk, tiling, pipe);
    }
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void RunPrepareVariant(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, AscendC::TPipe &pipe)
{
    RunPrepareStage<SAFE_GATE, T, float, BETA_T, float, float,
        TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, aqk, akk, addresses, userWorkspace, compactPlan,
        tiling, pipe);
}

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchPrepareSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, AscendC::TPipe &pipe)
{
    if (tiling.safeGate) {
        RunPrepareVariant<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses, userWorkspace, compactPlan,
            tiling, pipe);
    } else {
        RunPrepareVariant<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses, userWorkspace, compactPlan,
            tiling, pipe);
    }
}

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void RunPostGateFrontEndStages(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, AscendC::TPipe &pipe)
{
    DispatchPrepareSafeGate<T, BETA_T, TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, aqk, akk, addresses, userWorkspace, compactPlan,
        tiling, pipe);
    RunPostWuStage<T, float, BETA_T, TilingData>(
        q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk, akk,
        addresses, userWorkspace, compactPlan, tiling, pipe);
}

template <typename T, typename G_T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void RunFrontEndStages(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk,
    const ChunkKdaFwdAddresses &addresses, GM_ADDR userWorkspace,
    GM_ADDR compactPlan, const TilingData &tiling, AscendC::TPipe &pipe)
{
    if (tiling.safeGate) {
        DispatchGateMode<true, G_T>(
            g, aLog, dtBias, cuSeqlens, addresses.gk, tiling, pipe);
    } else {
        DispatchGateMode<false, G_T>(
            g, aLog, dtBias, cuSeqlens, addresses.gk, tiling, pipe);
    }
    if (!tiling.computeGateInPrepare) {
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        SyncAll<false>();
    }
    DispatchPrepareSafeGate<T, BETA_T, TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, aqk, akk, addresses, userWorkspace, compactPlan,
        tiling, pipe);
    RunPostWuStage<T, float, BETA_T, TilingData>(
        q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk, akk,
        addresses, userWorkspace, compactPlan, tiling, pipe);
}

template <bool USE_ARCH35, typename T, typename G_T, typename BETA_T,
          typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void RunDispatched(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, GM_ADDR compactPlan, const TilingData &tiling)
{
    static_assert(IsSameType<G_T, float>::value ||
                  IsSameType<G_T, bfloat16_t>::value,
                  "g dtype must be FP32 or BF16");
    const auto addresses = ResolveAddresses(
        finalState, gk, w, u, qg, kg, vNew, h, userWorkspace, tiling);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (!tiling.computeGateInPrepare) {
        if ASCEND_IS_AIV {
            {
                AscendC::TPipe gatePipe;
                RunGateStage<G_T>(
                    g, aLog, dtBias, cuSeqlens, addresses.gk, tiling,
                    gatePipe);
            }
        }
        SyncAll<false>();
    }
    AscendC::TPipe pipe;
    ReleaseAicPipeReservedMmadEvents(pipe);
    RunPostGateFrontEndStages<T, BETA_T, TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, aqk, akk, addresses, userWorkspace, compactPlan,
        tiling, pipe);
#if KDA_COMPILE_ARCH35_FAST_PATH
    if constexpr (USE_ARCH35) {
        arch35::RunBackEnd<T, BETA_T, TilingData>(
            q, k, v, beta, initialState, cuSeqlens, chunkIndices, attnOut,
            aqk, addresses, userWorkspace, compactPlan, tiling, pipe);
    } else {
#else
    static_assert(!USE_ARCH35, "arch35 backend requires tiling key 2");
    {
#endif
        if (!tiling.isVarLen && tiling.seqlen % tiling.chunkSize == 0) {
            if ASCEND_IS_AIV {
                pipe.Destroy();
            }
            RunGenericBackEnd<T, BETA_T, TilingData>(
                q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
                attnOut, addresses, userWorkspace, compactPlan, tiling);
        } else {
            RunGenericBackEnd<T, BETA_T, TilingData>(
                q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
                attnOut, addresses, userWorkspace, compactPlan, tiling, pipe);
        }
    }
#else
    static_assert(!USE_ARCH35, "arch35 backend is unavailable on this architecture");
    {
        AscendC::TPipe pipe;
        ReleaseAicPipeReservedMmadEvents(pipe);
        RunFrontEndStages<T, G_T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, aqk, akk, addresses, userWorkspace, compactPlan,
            tiling, pipe);
    }
    RunGenericBackEnd<T, BETA_T, TilingData>(
        q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
        attnOut, addresses, userWorkspace, compactPlan, tiling);
#endif
}

} // namespace KdaForward

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdTilingData, tilingData, tiling);
#if defined(TILING_KEY_VAR) && TILING_KEY_VAR == 1UL
    KdaForward::RunDispatched<false, DTYPE_Q, DTYPE_G, DTYPE_BETA,
        ChunkKdaFwdTilingData, 0, 0, 0>(
        q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
        chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
        kg, v_new, h, userWorkspace,
        tiling + tilingData.compactPlanOffset, tilingData);
#elif defined(TILING_KEY_VAR) && TILING_KEY_VAR == 2UL
#if KDA_COMPILE_ARCH35_FAST_PATH
    KdaForward::RunDispatched<true, DTYPE_Q, DTYPE_G, DTYPE_BETA,
        ChunkKdaFwdTilingData, 64, 128, 128>(
        q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
        chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
        kg, v_new, h, userWorkspace,
        tiling + tilingData.compactPlanOffset, tilingData);
#else
    KdaForward::RunDispatched<false, DTYPE_Q, DTYPE_G, DTYPE_BETA,
        ChunkKdaFwdTilingData, 64, 128, 128>(
        q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
        chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
        kg, v_new, h, userWorkspace,
        tiling + tilingData.compactPlanOffset, tilingData);
#endif
#else
    if (TILING_KEY_IS(1)) {
        KdaForward::RunDispatched<false, DTYPE_Q, DTYPE_G, DTYPE_BETA,
            ChunkKdaFwdTilingData, 0, 0, 0>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace,
            tiling + tilingData.compactPlanOffset, tilingData);
    } else if (TILING_KEY_IS(2)) {
#if KDA_COMPILE_ARCH35_FAST_PATH
        KdaForward::RunDispatched<true, DTYPE_Q, DTYPE_G, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace,
            tiling + tilingData.compactPlanOffset, tilingData);
#else
        KdaForward::RunDispatched<false, DTYPE_Q, DTYPE_G, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace,
            tiling + tilingData.compactPlanOffset, tilingData);
#endif
    }
#endif
}

#undef KDA_COMPILE_ARCH35_FAST_PATH
