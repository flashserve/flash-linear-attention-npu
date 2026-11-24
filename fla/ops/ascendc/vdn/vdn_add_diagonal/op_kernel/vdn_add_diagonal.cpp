// SPDX-License-Identifier: BSD-3-Clause
#include "kernel_operator.h"
using namespace AscendC;

#include "vdn_add_diagonal_struct.h"
#include "vdn_add_diagonal_tiling_key.h"
#include "arch22/vdn_add_diagonal_vec.h"

template <uint32_t SECTOR_WIDTH>
__global__ __aicore__ void vdn_add_diagonal(
    GM_ADDR correction, GM_ADDR twoEye, GM_ADDR correctionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(VdnAddDiagonalTilingData);
    GET_TILING_DATA_WITH_STRUCT(VdnAddDiagonalTilingData, data, tiling);
    Vdn::AddDiagonal<SECTOR_WIDTH>(reinterpret_cast<__gm__ float *>(correctionOut),
                                   reinterpret_cast<__gm__ float *>(twoEye), data);
}
