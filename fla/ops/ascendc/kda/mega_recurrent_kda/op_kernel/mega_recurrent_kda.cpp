#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define CATLASS_ARCH 3510
#include "mega_kernel.h"
#else
#define CATLASS_ARCH 2201
#include "lib/matmul_intf.h"
#include "arch22/entry.h"
#endif
#include "mega_recurrent_kda_tiling_data.h"

extern "C" __global__ __aicore__ void mega_recurrent_kda(
    GM_ADDR x, GM_ADDR qkv, GM_ADDR wa, GM_ADDR wb, GM_ADDR beta, GM_ADDR gate, GM_ADDR gamma,
    GM_ADDR wo, GM_ADDR convW, GM_ADDR conv, GM_ADDR checkpoint, GM_ADDR u, GM_ADDR key, GM_ADDR decay,
    GM_ADDR indices, GM_ADDR accepted, GM_ADDR alphaLog, GM_ADDR bias, GM_ADDR prefix, GM_ADDR out,
    GM_ADDR convOut, GM_ADDR uOut, GM_ADDR keyOut, GM_ADDR decayOut, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
    REGISTER_TILING_DEFAULT(MegaRecurrentKdaArch22TilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaRecurrentKdaArch22TilingData,t,tiling);
    using namespace MegaRecurrentKdaV2;
    V2Params p{};
    p.hiddenStates=x; p.qkvWeight=qkv; p.decayAWeight=wa; p.decayBWeight=wb;
    p.betaWeight=beta; p.gateWeight=gate; p.wOut=wo; p.convWeight=convW;
    p.convState=convOut; p.checkpoint=checkpoint; p.replayU=uOut; p.replayK=keyOut; p.replayDecay=decayOut;
    p.stateIndices=indices; p.accepted=accepted; p.aLog=alphaLog; p.dtBias=bias; p.gamma=gamma; p.out=out;
    runArch22(t,p,prefix,AscendC::GetUserWorkspace(workspace));
#else
    REGISTER_TILING_DEFAULT(MegaRecurrentKdaTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaRecurrentKdaTilingData,t,tiling);
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
#endif
}
