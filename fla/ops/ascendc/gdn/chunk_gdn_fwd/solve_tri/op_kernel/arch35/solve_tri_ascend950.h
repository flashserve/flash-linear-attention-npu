/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 *
 * A5 SolveTri dispatch headers. PR #340 split the implementation by
 * chunk-size; keep this stable include as the shared entry point for the
 * standalone and fused kernels.
 */
#ifndef SOLVE_TRI_ASCEND950_H
#define SOLVE_TRI_ASCEND950_H

#include "solve_tri_ascend950_common.h"
#include "solve_tri_ascend950_16.h"
#include "solve_tri_ascend950_32.h"
#include "solve_tri_ascend950_64.h"
#include "solve_tri_ascend950_128.h"

#endif  // SOLVE_TRI_ASCEND950_H
