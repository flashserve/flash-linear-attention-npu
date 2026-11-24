// SPDX-License-Identifier: BSD-3-Clause
#include "kernel_operator.h"
using namespace AscendC;

#include "vdn_scan_struct.h"
#include "vdn_scan_tiling_key.h"
#include "arch22/vdn_scan_cube.h"

template <uint32_t HEAD_DIM>
__global__ __aicore__ void vdn_scan(
    GM_ADDR transition, GM_ADDR prefix, GM_ADDR suffix, GM_ADDR prefixOut, GM_ADDR suffixOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    REGISTER_TILING_DEFAULT(VdnScanTilingData);
    GET_TILING_DATA_WITH_STRUCT(VdnScanTilingData, data, tiling);
#if defined(__DAV_C220_CUBE__)
    // L0 binds both outputs to their corresponding prefilled input banks.
    Vdn::Scan<HEAD_DIM>(reinterpret_cast<__gm__ float *>(transition), reinterpret_cast<__gm__ float *>(prefixOut),
                       reinterpret_cast<__gm__ float *>(suffixOut), data);
#endif
}
