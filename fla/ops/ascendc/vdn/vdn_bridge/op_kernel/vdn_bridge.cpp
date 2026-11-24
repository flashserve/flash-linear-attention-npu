// SPDX-License-Identifier: BSD-3-Clause
#include "kernel_operator.h"
using namespace AscendC;
#include "vdn_bridge_tiling_key.h"
#include "arch22/vdn_bridge_vec.h"

template <bool OUT_BF16>
__global__ __aicore__ void vdn_bridge(GM_ADDR prefix, GM_ADDR suffix, GM_ADDR bounds, GM_ADDR logs, GM_ADDR out,
                                      GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(VdnBridgeKernel::VdnBridgeTilingData);
    GET_TILING_DATA_WITH_STRUCT(VdnBridgeKernel::VdnBridgeTilingData, tile, tiling);
    if constexpr (OUT_BF16) {
        VdnBridgeKernel::Gather<bfloat16_t>(prefix, suffix, bounds, logs, out, tile);
    } else {
        VdnBridgeKernel::Gather<float>(prefix, suffix, bounds, logs, out, tile);
    }
}
