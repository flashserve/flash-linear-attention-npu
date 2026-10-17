#ifndef KDA_BWD_FINALIZE_ARCH22_H
#define KDA_BWD_FINALIZE_ARCH22_H
#include "kernel_operator.h"
#ifndef TORCH_MODE
// 与整体 Bwd 一致，为真正的 MIX 内核生成包装提供 workspace 初始化声明。
#include "lib/matmul_intf.h"
#endif
#include "tiling.h"
#include "../../../chunk_kda_bwd/op_kernel/chunk_kda_bwd_finalize.h"
#include "../../../chunk_kda_bwd/op_kernel/kda_gate_bwd_post.h"

namespace KDA {
// 独立接口已经缩放 dAqk；共享 A2 Phase C 会再次乘 scale，先还原有效下三角。
__aicore__ inline void RestoreArch22DAqk(GM_ADDR src, GM_ADDR dst, uint64_t elements, float scale)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 4096 * sizeof(float));
    auto local = buffer.Get<float>();
    AscendC::GlobalTensor<float> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
    const uint64_t stride = AscendC::GetBlockNum() * AscendC::GetSubBlockNum() * 4096ULL;
    for (uint64_t offset = AscendC::GetBlockIdx() * 4096ULL; offset < elements; offset += stride) {
        const uint32_t count = elements - offset < 4096 ? elements - offset : 4096;
        AscendC::DataCopy(local, input[offset], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(local, local, scale == 0.0f ? 0.0f : 1.0f / scale, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(output[offset], local, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
}

// Phase C 内部以 FP32 累加 dq/dk；公开独立接口仍返回 BF16。
__aicore__ inline void StoreArch22Gradients(
    GM_ADDR src, GM_ADDR dst, uint64_t elements)
{
    constexpr uint32_t tile = 4096;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input, output;
    pipe.InitBuffer(input, tile * sizeof(float));
    pipe.InitBuffer(output, tile * sizeof(bfloat16_t));
    AscendC::GlobalTensor<float> in;
    AscendC::GlobalTensor<bfloat16_t> out;
    in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dst));
    auto a = input.Get<float>();
    auto b = output.Get<bfloat16_t>();
    // 对齐 dense 输入保证尾块也是 16 元素倍数，所有 DMA 写互不重叠。
    const uint64_t stride = static_cast<uint64_t>(AscendC::GetBlockNum()) *
                            AscendC::GetSubBlockNum() * tile;
    for (uint64_t offset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tile;
         offset < elements; offset += stride) {
        const uint32_t count = static_cast<uint32_t>(
            elements - offset < tile ? elements - offset : tile);
        AscendC::DataCopy(a, in[offset], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(b, a, AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(out[offset], b, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
}
}

extern "C" __global__ __aicore__ void chunk_kda_bwd_finalize(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk,
    GM_ADDR rawG, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
    GM_ADDR dvScan, GM_ADDR dAqk, GM_ADDR dqRaw,
    GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dBeta,
    GM_ADDR dG, GM_ADDR dALog, GM_ADDR dDtBias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdFinalizeArch22TilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdFinalizeArch22TilingData, t, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(101, KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(101)) {
        GM_ADDR ws = AscendC::GetUserWorkspace(workspace);
        if (ws == nullptr) { return; }
        if ASCEND_IS_AIV {
            KDA::RestoreArch22DAqk(dAqk, ws + t.dAqkOffset,
                static_cast<uint64_t>(t.phaseC.batch) * t.phaseC.headNum * t.phaseC.seqlen * 64, t.phaseC.scale);
        }
        AscendC::SyncAll<false>();
        GM_ADDR dqTarget = t.fp32Output ? dq : ws + t.dqOffset;
        GM_ADDR dkTarget = t.fp32Output ? dk : ws + t.dkOffset;
#define RUN_FINALIZE(SAFE, VARLEN) KDA::RunChunkKdaBwdC<DTYPE_Q, 128, DTYPE_BETA, SAFE, VARLEN>( \
q, k, v, vNew, gk, beta, akk, h, dh, dvScan, dqRaw, ws + t.dAqkOffset, \
cuSeqlens, chunkIndices, rawG, aLog, dtBias, \
dqTarget, dkTarget, dv, dBeta, dG, \
ws + t.dAkkOffset, dALog, dDtBias, \
ws + t.phaseWorkspaceOffset, &t.phaseC, true);
        if (t.safeGate) {
            if (t.phaseC.isVarLen) { RUN_FINALIZE(true, true); }
            else { RUN_FINALIZE(true, false); }
        } else {
            if (t.phaseC.isVarLen) { RUN_FINALIZE(false, true); }
            else { RUN_FINALIZE(false, false); }
        }
#undef RUN_FINALIZE
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            // A2 沿用整体 Bwd 的 GatePost 实现，但在本次 launch 内执行。
            if (t.phaseC.useGateInKernel) {
                KDA::KdaGateBwdPostTilingData gate{};
                gate.batch = t.phaseC.batch;
                gate.seqlen = t.phaseC.seqlen;
                gate.headNum = t.phaseC.headNum;
                gate.keyDim = 128;
                gate.chunkSize = 64;
                gate.chunkNum = t.phaseC.chunkNum;
                gate.chunkNumPerBatch = t.phaseC.chunkNumPerBatch;
                gate.usedCoreNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
                gate.isVarLen = t.phaseC.isVarLen;
                gate.hasDtBias = t.phaseC.hasDtBias;
                gate.lowerBound = t.phaseC.lowerBound;
                AscendC::TPipe pipe;
                if (t.safeGate) {
                    KDA::KdaGateBwdPostKernel<true, float, false> process;
                    process.Init(dG, rawG, aLog, dtBias, cuSeqlens, chunkIndices,
                        dG, dALog, dDtBias, gate, &pipe);
                    process.Process();
                } else {
                    KDA::KdaGateBwdPostKernel<false, float, false> process;
                    process.Init(dG, rawG, aLog, dtBias, cuSeqlens, chunkIndices,
                        dG, dALog, dDtBias, gate, &pipe);
                    process.Process();
                }
            }
            const uint64_t elements = static_cast<uint64_t>(t.phaseC.batch) *
                t.phaseC.headNum * t.phaseC.seqlen * 128;
            if (!t.fp32Output) {
                KDA::StoreArch22Gradients(ws + t.dqOffset, dq, elements);
                KDA::StoreArch22Gradients(ws + t.dkOffset, dk, elements);
            }
        }
    }
}
#endif
