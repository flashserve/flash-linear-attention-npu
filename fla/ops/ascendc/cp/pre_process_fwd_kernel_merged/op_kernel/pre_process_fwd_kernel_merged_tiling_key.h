/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License. Please refer to the License for details.
 */

/*!
 * \file pre_process_fwd_kernel_merged_tiling_key.h
 * \brief 模板化 TilingKey（§4.2）：唯一位是 GATE_MODE，取值与 host 侧 SetTilingKey(gateMode+1) 一一对应。
 *        禁运行期 TILING_KEY_IS 分派；实例化由本文件的 ASCENDC_TPL_SEL 决定。
 */

#ifndef PRE_PROCESS_FWD_KERNEL_MERGED_TILING_KEY_H
#define PRE_PROCESS_FWD_KERNEL_MERGED_TILING_KEY_H

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

// 与 op_host 的 TilingKey 取值一致：1=USE_G（GDN 标量门控）/ 2=USE_GK（KDA 逐 k 门控）。
// 3=USE_BG（DPLR）只保留模板槽位、host 永不产生该 key：本版本不支持 DPLR，
// host 校验对非空 bg/v 直接报错（见 docs/api.md §3.4），详见 tiling_processor.h。
#define PPFM_TPL_GATE_G  1
#define PPFM_TPL_GATE_GK 2
#define PPFM_TPL_GATE_BG 3

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(PreProcessFwdKernelMerged,
    ASCENDC_TPL_UINT_DECL(GATE_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                          PPFM_TPL_GATE_G, PPFM_TPL_GATE_GK, PPFM_TPL_GATE_BG),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(GATE_MODE, ASCENDC_TPL_UI_LIST,
                             PPFM_TPL_GATE_G, PPFM_TPL_GATE_GK, PPFM_TPL_GATE_BG),
    ),
);
#endif  // TORCH_MODE

}  // namespace GDN

#endif  // PRE_PROCESS_FWD_KERNEL_MERGED_TILING_KEY_H
