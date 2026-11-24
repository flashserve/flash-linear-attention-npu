/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0
 *
 * chunk_local_cumsum TPL 模板化 tiling key（替代 TILING_KEY_IS，检视要求）。
 * SEG128 = 1 表示 chunkSize=128 的 headFirst 优化路径，0 为通用路径；
 * arch22（910b/910_93）走 CHUNK_LOCAL_CUMSUM_RUNTIME_KEYS 旧路径，不经此模板。
 */

#ifndef CHUNK_LOCAL_CUMSUM_TILING_KEY_H
#define CHUNK_LOCAL_CUMSUM_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    ChunkLocalCumsum,
    ASCENDC_TPL_UINT_DECL(SEG128, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SEG128, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SEG128, ASCENDC_TPL_UI_LIST, 1)));

#endif
