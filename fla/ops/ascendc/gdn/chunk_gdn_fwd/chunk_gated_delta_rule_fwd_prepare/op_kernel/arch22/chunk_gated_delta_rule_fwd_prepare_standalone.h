// Copyright (c) 2026 Tianjin University, Ltd.
// BSD-3-Clause. 独立 Prepare 的 DAV_2201 入口，复用完整三段式 Prepare 核心。
#ifndef GDN_ARCH22_PREPARE_STANDALONE_H
#define GDN_ARCH22_PREPARE_STANDALONE_H
#define CATLASS_ARCH 2201
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_fwd_prepare.h"
#include "chunk_gated_delta_rule_fwd_prepare_standalone_tiling.h"
#undef CATLASS_ARCH

namespace GdnUnifiedPrepare {
template <typename InputT, int V, int BT>
__aicore__ inline void RunStandalone(const Args &args, bool norm)
{
    if (norm) Run<InputT, false, V, BT, 0, true>(args);
    else Run<InputT, false, V, BT, 0, false>(args);
}
} // namespace GdnUnifiedPrepare

REGISTER_TILING_DEFAULT(GdnUnifiedPrepare::StandaloneTiling);
extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_prepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR g_out, GM_ADDR w_out, GM_ADDR u_out, GM_ADDR a_out,
    GM_ADDR q_hat, GM_ADDR k_hat, GM_ADDR q_rstd, GM_ADDR k_rstd, GM_ADDR beta_eff,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(22)) {
        KERNEL_TASK_TYPE(22, KERNEL_TYPE_MIX_AIC_1_2);
        const auto *t = reinterpret_cast<const __gm__ GdnUnifiedPrepare::StandaloneTiling *>(tiling);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        GdnUnifiedPrepare::Args args{
            k, v, beta, g, cu_seqlens, chunk_indices,
            g_out, nullptr, nullptr, nullptr,
            t->outputA != 0 ? a_out : userWorkspace + t->internalAOffset,
            w_out, u_out, userWorkspace,
            t->batch, t->hk, t->hv, t->tokens, t->nt, t->isVarlen, 0};
        args.useGate = t->useGate;
        args.useBetaSigmoid = t->useBetaSigmoid;
        args.allowNegEigval = t->allowNegEigval;
        args.useExp2 = t->useExp2;
        args.aLog = t->useGate != 0 ? a_log : nullptr;
        args.dtBias = t->useDtBias != 0 ? dt_bias : nullptr;
        args.betaEff = t->useBetaSigmoid != 0 ? beta_eff : nullptr;
        const bool norm = t->useQkNorm != 0;
        if (norm) {
            args.normQ = q; args.normK = k; args.qHat = q_hat; args.k = k_hat;
            args.qRstd = q_rstd; args.kRstd = k_rstd;
        }
        if (t->bt == 64) {
            if (t->valueDim == 128) GdnUnifiedPrepare::RunStandalone<DTYPE_Q, 128, 64>(args, norm);
            else GdnUnifiedPrepare::RunStandalone<DTYPE_Q, 256, 64>(args, norm);
        } else {
            if (t->valueDim == 128) GdnUnifiedPrepare::RunStandalone<DTYPE_Q, 128, 128>(args, norm);
            else GdnUnifiedPrepare::RunStandalone<DTYPE_Q, 256, 128>(args, norm);
        }
    }
}
#endif
