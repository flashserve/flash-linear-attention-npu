/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_FWD_H_TILING_KEY_H
#define CHUNK_FWD_H_TILING_KEY_H

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(ChunkFwdH,
    ASCENDC_TPL_UINT_DECL(V_DIM, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 128),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(V_DIM, ASCENDC_TPL_UI_LIST, 128),
    ),
);
#endif

} // namespace GDN

#endif // CHUNK_FWD_H_TILING_KEY_H
