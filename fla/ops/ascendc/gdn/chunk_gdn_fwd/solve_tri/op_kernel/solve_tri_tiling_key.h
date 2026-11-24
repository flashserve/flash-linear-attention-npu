/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * SolveTri TPL 模板化 tiling key（替代 TILING_KEY_IS 运行时分支，检视要求）。
 * tilingKey 语义 = chunkSize ∈ {16, 32, 64, 128}；
 * 仅 960dt（9201）经此模板；950（3510）与 arch22（910b/910_93）保持 main 既有运行时 key 路径，行为不变。
 */

#ifndef SOLVE_TRI_TILING_KEY_H
#define SOLVE_TRI_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    SolveTri,
    ASCENDC_TPL_UINT_DECL(CHUNK_SIZE, 8, ASCENDC_TPL_UI_LIST, 16, 32, 64, 128));

#define SOLVE_TRI_SEL_ONE(CHUNK) \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CHUNK_SIZE, ASCENDC_TPL_UI_LIST, CHUNK))

ASCENDC_TPL_SEL(
    SOLVE_TRI_SEL_ONE(16),
    SOLVE_TRI_SEL_ONE(32),
    SOLVE_TRI_SEL_ONE(64),
    SOLVE_TRI_SEL_ONE(128));

#endif
