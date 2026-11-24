/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * chunk_gated_delta_rule_fwd_h TPL 模板化 tiling key（替代 TILING_KEY_IS，检视意见第 6 条）。
 * V_DIM = value head 维度（128/256），对应 TileShapes128/256 两个 kernel 变体。
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_TILING_KEY_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    ChunkGatedDeltaRuleFwdH,
    ASCENDC_TPL_UINT_DECL(V_DIM, 9, ASCENDC_TPL_UI_LIST, 128, 256));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(V_DIM, ASCENDC_TPL_UI_LIST, 128)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(V_DIM, ASCENDC_TPL_UI_LIST, 256)));

#endif
