#ifndef KDA_BWD_PREPARE_ARCH22_H
#define KDA_BWD_PREPARE_ARCH22_H
#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

// 复用整体 Bwd 的 Phase A；不复制矩阵计算和片上缓冲区实现。
#include "../../../chunk_kda_bwd/op_kernel/chunk_kda_bwd_prepare.h"
#include "../chunk_kda_bwd_prepare_struct.h"

extern "C" __global__ __aicore__ void chunk_kda_bwd_prepare(
    GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdPrepareTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdPrepareTilingData, t, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(101, KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(101)) {
        KDA::ChunkKdaBwdATilingData a{
            static_cast<int32_t>(t.NV), static_cast<int32_t>(t.T), 64,
            static_cast<int32_t>(t.totalChunkNum),
            static_cast<int32_t>(t.denseChunkNum), static_cast<int32_t>(t.isVariable),
            static_cast<int32_t>(AscendC::GetBlockNum())};
        KDA::ChunkKdaBwdCTilingData c{};
        c.scale = t.scale;
        KDA::RunChunkKdaBwdA<DTYPE_AQK, 128>(
            aqk, vNew, h, dO, cuSeqlens, chunkIndices,
            dv, dqRaw, dAqk, AscendC::GetUserWorkspace(workspace), a, c);
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            // 整体 A2 Phase A 输出原始矩阵；独立接口要求 scale + tril。
            AscendC::TPipe pipe;
            AscendC::TBuf<AscendC::TPosition::VECCALC> data, mask;
            pipe.InitBuffer(data, 4096 * sizeof(float));
            pipe.InitBuffer(mask, 4096 * sizeof(float));
            auto local = data.Get<float>();
            auto weights = mask.Get<float>();
            for (uint32_t i = 0; i < 4096; ++i) {
                weights.SetValue(i, i % 64 <= i / 64 ? t.scale : 0.0f);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::GlobalTensor<float> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
            AscendC::GlobalTensor<int64_t> cu, indices;
            if (t.isVariable) {
                cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
                indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
            }
            const uint64_t tasks = static_cast<uint64_t>(t.totalChunkNum) * t.NV;
            const uint64_t stride = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
            for (uint64_t task = AscendC::GetBlockIdx(); task < tasks; task += stride) {
                const uint64_t chunk = task / t.NV;
                const uint64_t head = task % t.NV;
                uint64_t batch = 0, begin = 0, end = t.T;
                if (t.isVariable) {
                    const auto sequence = indices.GetValue(2 * chunk);
                    begin = cu.GetValue(sequence) + indices.GetValue(2 * chunk + 1) * 64;
                    end = cu.GetValue(sequence + 1);
                } else {
                    batch = chunk / t.denseChunkNum;
                    begin = (chunk % t.denseChunkNum) * 64;
                }
                const uint32_t rows = end - begin < 64 ? end - begin : 64;
                const uint32_t count = rows * 64;
                const uint64_t offset = ((batch * t.NV + head) * t.T + begin) * 64;
                AscendC::DataCopy(local, gm[offset], count);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::Mul(local, local, weights, count);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::DataCopy(gm[offset], local, count);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            }
        }
    }
}
#endif
