// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_BRIDGE_TILING_KEY_H
#define VDN_BRIDGE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace VdnBridgeKernel {
ASCENDC_TPL_ARGS_DECL(VdnBridge, ASCENDC_TPL_BOOL_DECL(OUT_BF16, 0, 1));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(OUT_BF16, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(OUT_BF16, 1)));
} // namespace VdnBridgeKernel

#endif
