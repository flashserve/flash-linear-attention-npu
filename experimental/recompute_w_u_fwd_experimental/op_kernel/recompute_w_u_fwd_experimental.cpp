#include "kernel_operator.h"
#include "build_probe.hpp"
#include "recompute_w_u_fwd_experimental_tiling.h"

extern "C" __global__ __aicore__ void recompute_w_u_fwd_experimental(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    static_assert(FLA_EXPERIMENTAL_PROBE_ELEMENTS == 8);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(Experimental::RecomputeWUFwdExperimentalTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        Experimental::RecomputeWUFwdExperimentalTilingData, tilingData, tiling);

    AscendC::GlobalTensor<float> input;
    AscendC::GlobalTensor<float> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y));
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 32);
    auto tile = buffer.Get<float>();
    AscendC::DataCopy(tile, input, tilingData.elements);
    AscendC::DataCopy(output, tile, tilingData.elements);
}
