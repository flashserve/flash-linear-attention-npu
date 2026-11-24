/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
#include "internal_a96/arch35/chunk_gated_delta_rule_fwd_arch35.cpp"
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp"
#else
#include "internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp"
#endif
