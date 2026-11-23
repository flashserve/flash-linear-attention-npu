#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define CATLASS_ARCH 3510
#include "arch35/mega_kernel.h"
#else
#define CATLASS_ARCH 2201
#include "lib/matmul_intf.h"
#include "arch22/entry.h"
#endif
#include "mega_recurrent_kda_tiling_data.h"

extern "C" __global__ __aicore__ void
mega_recurrent_kda(GM_ADDR x, GM_ADDR qkv, GM_ADDR wa, GM_ADDR wb, GM_ADDR beta, GM_ADDR gate, GM_ADDR gamma,
                   GM_ADDR wo, GM_ADDR convW, GM_ADDR conv, GM_ADDR checkpoint, GM_ADDR u, GM_ADDR key, GM_ADDR decay,
                   GM_ADDR indices, GM_ADDR accepted, GM_ADDR alphaLog, GM_ADDR bias, GM_ADDR prefix, GM_ADDR out,
                   GM_ADDR convOut, GM_ADDR uOut, GM_ADDR keyOut, GM_ADDR decayOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
    REGISTER_TILING_DEFAULT(MegaRecurrentKdaArch22TilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaRecurrentKdaArch22TilingData, tilingData, tiling);
    using namespace MegaRecurrentKdaV2;
    V2Params params{};
    params.hiddenStates = x;
    params.qkvWeight = qkv;
    params.decayAWeight = wa;
    params.decayBWeight = wb;
    params.betaWeight = beta;
    params.gateWeight = gate;
    params.wOut = wo;
    params.convWeight = convW;
    params.convState = convOut;
    params.checkpoint = checkpoint;
    params.replayU = uOut;
    params.replayK = keyOut;
    params.replayDecay = decayOut;
    params.stateIndices = indices;
    params.accepted = accepted;
    params.aLog = alphaLog;
    params.dtBias = bias;
    params.gamma = gamma;
    params.out = out;
    RunArch22(tilingData, params, prefix, AscendC::GetUserWorkspace(workspace));
#else
    REGISTER_TILING_DEFAULT(MegaRecurrentKdaTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaRecurrentKdaTilingData, tilingData, tiling);
    kda::MegaParams params{};
    params.hiddenStates = x;
    params.qkvWeight = qkv;
    params.decayAWeight = wa;
    params.decayBWeight = wb;
    params.betaWeight = beta;
    params.gateWeight = gate;
    params.rows = tilingData.tokens;
    params.heads = tilingData.heads;
    params.decayRing = AscendC::GetUserWorkspace(workspace);
    auto &recurrence = params.recurrence;
    recurrence.convW = convW;
    recurrence.conv = convOut;
    recurrence.checkpoint = checkpoint;
    recurrence.u = uOut;
    recurrence.key = keyOut;
    recurrence.decay = decayOut;
    recurrence.gamma = gamma;
    recurrence.outWeight = wo;
    recurrence.output = out;
    recurrence.indices = indices;
    recurrence.accepted = accepted;
    recurrence.prefix = prefix;
    recurrence.alphaLog = alphaLog;
    recurrence.bias = bias;
    recurrence.batch = tilingData.batch;
    recurrence.sequence = tilingData.sequence;
    recurrence.capacity = tilingData.capacity;
    recurrence.prefix64 = tilingData.prefix64;
    recurrence.convLayout = tilingData.convLayout;
    recurrence.crosses = tilingData.crosses;
    recurrence.staged = tilingData.crosses ? 0 : 3;
    recurrence.convStride = tilingData.convStride;
    recurrence.checkpointStride = tilingData.checkpointStride;
    recurrence.uStride = tilingData.uStride;
    recurrence.keyStride = tilingData.keyStride;
    recurrence.decayStride = tilingData.decayStride;
    recurrence.scale = tilingData.scale;
    recurrence.lower = tilingData.lowerBound;
    recurrence.eps = tilingData.rmsNormEps;

    if (TILING_KEY_IS(1)) {
        kda::MegaKernel<false> op;
        op(params);
    } else if (TILING_KEY_IS(2)) {
        kda::MegaKernel<true> op;
        op(params);
    }
#endif
}
