/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef RECOMPUTE_W_U_FWD_TILING_KEY_H
#define RECOMPUTE_W_U_FWD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace GDN {
ASCENDC_TPL_ARGS_DECL(RecomputeWUFwd,
    ASCENDC_TPL_UINT_DECL(V, 1, ASCENDC_TPL_UI_LIST, 128, 256),
    ASCENDC_TPL_BOOL_DECL(QKV_SEQUENCE_MAJOR, 0, 1),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(V, ASCENDC_TPL_UI_LIST, 128, 256),
        ASCENDC_TPL_BOOL_SEL(QKV_SEQUENCE_MAJOR, 0, 1),
    ),
);
} // namespace GDN
#endif
