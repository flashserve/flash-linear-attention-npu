#include "kernel_operator.h"
#include "mega_kernel.h"
#include "mega_recurrent_kda_tiling_data.h"

extern "C" __global__ __aicore__ void mega_recurrent_kda(
    GM_ADDR x, GM_ADDR qkv, GM_ADDR wa, GM_ADDR wb, GM_ADDR beta, GM_ADDR gate, GM_ADDR gamma,
    GM_ADDR wo, GM_ADDR convW, GM_ADDR conv, GM_ADDR checkpoint, GM_ADDR u, GM_ADDR key, GM_ADDR decay,
    GM_ADDR indices, GM_ADDR accepted, GM_ADDR alphaLog, GM_ADDR bias, GM_ADDR prefix, GM_ADDR out,
    GM_ADDR convOut, GM_ADDR uOut, GM_ADDR keyOut, GM_ADDR decayOut, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MegaRecurrentKdaTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaRecurrentKdaTilingData,t,tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    kda::MegaParams p{};
    p.x=x; p.qkv=qkv; p.wa=wa; p.wb=wb; p.beta=beta; p.gate=gate;
    p.rows=t.tokens; p.heads=t.heads; p.decayRing=AscendC::GetUserWorkspace(workspace);
    auto &a=p.recurrence;
    a.convW=convW; a.conv=convOut; a.checkpoint=checkpoint; a.u=uOut; a.key=keyOut; a.decay=decayOut;
    a.gamma=gamma; a.outWeight=wo; a.output=out; a.indices=indices; a.accepted=accepted; a.prefix=prefix;
    a.alphaLog=alphaLog; a.bias=bias; a.batch=t.batch; a.sequence=t.sequence; a.capacity=t.capacity;
    a.prefix64=t.prefix64; a.convLayout=t.convLayout; a.crosses=t.crosses; a.staged=t.crosses?0:3;
    a.convStride=t.convStride; a.checkpointStride=t.checkpointStride; a.uStride=t.uStride;
    a.keyStride=t.keyStride; a.decayStride=t.decayStride;
    a.scale=t.scale; a.lower=t.lowerBound; a.eps=t.rmsNormEps;

    if(TILING_KEY_IS(1)) {
        kda::MegaKernel<false> op;
        op(p);
    } else if(TILING_KEY_IS(2)) {
        kda::MegaKernel<true> op;
        op(p);
    }
}
