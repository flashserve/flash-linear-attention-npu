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

constexpr int64_t KDA_STAGE_FULL = -1;
constexpr int64_t KDA_STAGE_GATE_PREPARE = 0;
constexpr int64_t KDA_STAGE_POST_WU = 1;
constexpr int64_t KDA_STAGE_FWD_H = 2;
constexpr int64_t KDA_STAGE_FINALIZE = 3;

__aicore__ inline GM_ADDR ResolveStageInput(
    GM_ADDR stageInput, GM_ADDR fallback)
{
    return stageInput == nullptr ? fallback : stageInput;
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchStage(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR stageGk, GM_ADDR stageAqk, GM_ADDR stageAkk, GM_ADDR stageW,
    GM_ADDR stageU, GM_ADDR stageKg, GM_ADDR stageVNew, GM_ADDR stageH,
    GM_ADDR stageQgScaled, GM_ADDR stageUSeed, GM_ADDR stageVNewFp32,
    GM_ADDR stageHFp32, GM_ADDR stageAkkFp32, GM_ADDR stageWFp32,
    GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR vNewFp32, GM_ADDR hFp32,
    GM_ADDR akkFp32, GM_ADDR wFp32,
    GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    auto addresses = ResolveAddresses(
        finalState, gk, w, u, qg, kg, vNew, vNewFp32, h, hFp32,
        akkFp32, wFp32, userWorkspace, tiling);
    addresses.qgScaled = qgScaled;
    addresses.uSeed = uSeed;
    if (tiling.stage == KDA_STAGE_GATE_PREPARE) {
        RunGateCumsum(g, aLog, dtBias, cuSeqlens, addresses.gk, tiling);
        if (!tiling.computeGateInPrepare) {
            SyncAll<false>();
        }
        TPipe pipe;
        KdaPrepare::RunChunkKdaPrepare<SAFE_GATE, T, float, BETA_T,
            TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, addresses.gk, g, aLog, dtBias, beta, initialState,
            cuSeqlens, chunkIndices, aqk, akk, addresses.qg,
            addresses.qgScaled, addresses.w, uSeed, addresses.kg,
            addresses.akkFp32, userWorkspace, tiling, pipe,
            tiling.storeQG);
    } else if (tiling.stage == KDA_STAGE_POST_WU) {
        TPipe pipe;
        RunPostWu<SAFE_GATE, T, float, BETA_T, TilingData, COMPILE_BT,
                  COMPILE_K, COMPILE_V>(
            q, k, v, ResolveStageInput(stageGk, addresses.gk), beta,
            initialState, cuSeqlens, chunkIndices,
            ResolveStageInput(stageW, addresses.w),
            ResolveStageInput(stageAkk, akk),
            ResolveStageInput(stageUSeed, uSeed), addresses.w, addresses.u,
            addresses.kg, addresses.vNew,
            ResolveStageInput(stageAkkFp32, addresses.akkFp32),
            addresses.wFp32, addresses.vNewFp32, userWorkspace, tiling,
            pipe);
    } else if (tiling.stage == KDA_STAGE_FWD_H) {
        auto fwdInputs = addresses;
        fwdInputs.gk = ResolveStageInput(stageGk, addresses.gk);
        fwdInputs.w = ResolveStageInput(stageW, addresses.w);
        fwdInputs.u = ResolveStageInput(stageU, addresses.u);
        fwdInputs.kg = ResolveStageInput(stageKg, addresses.kg);
        fwdInputs.uSeed = ResolveStageInput(stageUSeed, addresses.uSeed);
        fwdInputs.wFp32 = ResolveStageInput(stageWFp32, addresses.wFp32);
        fwdInputs.vNewFp32 = ResolveStageInput(
            stageVNewFp32, addresses.vNewFp32);
        if (tiling.vHeadDim > 128) {
            RunFwdH<SAFE_GATE, T,
                    Catlass::Gemm::Kernel::KDAFwdHTileShapes256, TilingData,
                    COMPILE_BT, COMPILE_K, COMPILE_V>(
                initialState, cuSeqlens, chunkIndices, fwdInputs,
                userWorkspace, tiling);
        } else {
            RunFwdH<SAFE_GATE, T,
                    Catlass::Gemm::Kernel::KDAFwdHTileShapes128, TilingData,
                    COMPILE_BT, COMPILE_K, COMPILE_V>(
                initialState, cuSeqlens, chunkIndices, fwdInputs,
                userWorkspace, tiling);
        }
    } else if (tiling.stage == KDA_STAGE_FINALIZE) {
        TPipe pipe;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using PropagatedVType =
            std::conditional_t<IsSameType<T, half>::value, float, T>;
        using PropagatedHType =
            std::conditional_t<IsSameType<T, half>::value, float, T>;
        GM_ADDR propagatedVNew = IsSameType<T, half>::value
            ? ResolveStageInput(stageVNewFp32, addresses.vNewFp32)
            : ResolveStageInput(stageVNew, addresses.vNew);
        GM_ADDR propagatedH = IsSameType<T, half>::value
            ? ResolveStageInput(stageHFp32, addresses.hFp32)
            : ResolveStageInput(stageH, addresses.h);
        KdaFinalize::RunChunkKdaOutput<
            T, float, BETA_T, PropagatedVType, PropagatedHType>(
#else
        GM_ADDR propagatedVNew = ResolveStageInput(stageVNew, addresses.vNew);
        GM_ADDR propagatedH = ResolveStageInput(stageH, addresses.h);
        KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
#endif
            q, k, v, ResolveStageInput(stageGk, addresses.gk), beta,
            initialState, cuSeqlens, chunkIndices,
            ResolveStageInput(stageQgScaled, addresses.qgScaled),
            ResolveStageInput(stageAqk, aqk),
            propagatedVNew,
            propagatedH, attnOut, userWorkspace, tiling, pipe);
    }
}

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchStageSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR stageGk, GM_ADDR stageAqk, GM_ADDR stageAkk, GM_ADDR stageW,
    GM_ADDR stageU, GM_ADDR stageKg, GM_ADDR stageVNew, GM_ADDR stageH,
    GM_ADDR stageQgScaled, GM_ADDR stageUSeed, GM_ADDR stageVNewFp32,
    GM_ADDR stageHFp32, GM_ADDR stageAkkFp32, GM_ADDR stageWFp32,
    GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR vNewFp32, GM_ADDR hFp32,
    GM_ADDR akkFp32, GM_ADDR wFp32,
    GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchStage<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, stageGk, stageAqk, stageAkk, stageW, stageU,
            stageKg, stageVNew, stageH, stageQgScaled, stageUSeed,
            stageVNewFp32, stageHFp32, stageAkkFp32, stageWFp32, attnOut,
            finalState, gk, aqk, akk, w, u, qg, kg, vNew, h, qgScaled,
            uSeed, vNewFp32, hFp32, akkFp32, wFp32, userWorkspace, tiling);
    } else {
        DispatchStage<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, stageGk, stageAqk, stageAkk, stageW, stageU,
            stageKg, stageVNew, stageH, stageQgScaled, stageUSeed,
            stageVNewFp32, stageHFp32, stageAkkFp32, stageWFp32, attnOut,
            finalState, gk, aqk, akk, w, u, qg, kg, vNew, h, qgScaled,
            uSeed, vNewFp32, hFp32, akkFp32, wFp32, userWorkspace, tiling);
    }
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGeneric(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew,
    GM_ADDR vNewFp32, GM_ADDR h, GM_ADDR hFp32, GM_ADDR akkFp32,
    GM_ADDR wFp32, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    RunGeneric<SAFE_GATE, T, BETA_T, TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg, kg,
        vNew, vNewFp32, h, hFp32, akkFp32, wFp32, userWorkspace, tiling);
}

#if KDA_COMPILE_ARCH35_FAST_PATH
template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew,
    GM_ADDR vNewFp32, GM_ADDR h, GM_ADDR hFp32, GM_ADDR akkFp32,
    GM_ADDR wFp32, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    AscendC::TPipe pipe;
    if (tiling.safeGate) {
        arch35::Run<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, vNewFp32, h, hFp32, akkFp32, wFp32,
            userWorkspace, tiling, pipe);
    } else {
        arch35::Run<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, vNewFp32, h, hFp32, akkFp32, wFp32,
            userWorkspace, tiling, pipe);
    }
}
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR,
    GM_ADDR,
    const TilingData &)
{
}
#endif

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGenericSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew,
    GM_ADDR vNewFp32, GM_ADDR h, GM_ADDR hFp32, GM_ADDR akkFp32,
    GM_ADDR wFp32, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchGeneric<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, vNewFp32, h, hFp32, akkFp32, wFp32,
            userWorkspace, tiling);
    } else {
        DispatchGeneric<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, vNewFp32, h, hFp32, akkFp32, wFp32,
            userWorkspace, tiling);
    }
}

} // namespace KdaForward

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR stage_gk, GM_ADDR stage_aqk, GM_ADDR stage_akk, GM_ADDR stage_w,
    GM_ADDR stage_u, GM_ADDR stage_kg, GM_ADDR stage_v_new, GM_ADDR stage_h,
    GM_ADDR stage_qg_scaled, GM_ADDR stage_u_seed, GM_ADDR stage_v_new_fp32,
    GM_ADDR stage_h_fp32, GM_ADDR stage_akk_fp32, GM_ADDR stage_w_fp32,
    GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR qg_scaled, GM_ADDR u_seed, GM_ADDR v_new_fp32, GM_ADDR h_fp32,
    GM_ADDR akk_fp32, GM_ADDR w_fp32,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
            KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
                ChunkKdaFwdTilingData, 0, 0, 0>(
                q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
                chunk_indices, stage_gk, stage_aqk, stage_akk, stage_w,
                stage_u, stage_kg, stage_v_new, stage_h, stage_qg_scaled,
                stage_u_seed, stage_v_new_fp32, stage_h_fp32,
                stage_akk_fp32, stage_w_fp32, attn_out,
                final_state, gk, aqk, akk, w, u, qg, kg, v_new, h,
                qg_scaled, u_seed, v_new_fp32, h_fp32, akk_fp32, w_fp32,
                userWorkspace, tilingData);
            return;
        }
        KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 0, 0, 0>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, v_new_fp32, h, h_fp32, akk_fp32, w_fp32,
            userWorkspace, tilingData);
    } else if (TILING_KEY_IS(2)) {
        if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
            KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
                ChunkKdaFwdTilingData, 64, 128, 128>(
                q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
                chunk_indices, stage_gk, stage_aqk, stage_akk, stage_w,
                stage_u, stage_kg, stage_v_new, stage_h, stage_qg_scaled,
                stage_u_seed, stage_v_new_fp32, stage_h_fp32,
                stage_akk_fp32, stage_w_fp32, attn_out,
                final_state, gk, aqk, akk, w, u, qg, kg, v_new, h,
                qg_scaled, u_seed, v_new_fp32, h_fp32, akk_fp32, w_fp32,
                userWorkspace, tilingData);
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaForward::DispatchArch35SafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
#else
        KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
#endif
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, v_new_fp32, h, h_fp32, akk_fp32, w_fp32,
            userWorkspace, tilingData);
    }
}

#undef KDA_COMPILE_ARCH35_FAST_PATH
