#pragma once

#if defined(FLA_NPU_OP_TRACE) && FLA_NPU_OP_TRACE
#include "kernel_operator.h"
#define FLA_KERNEL_TRACE(name) \
    do { \
        if (AscendC::GetBlockIdx() == 0) { \
            AscendC::printf("[FLA][KERNEL] " name " file=" __FILE__ "\n"); \
        } \
    } while (0)
#else
#define FLA_KERNEL_TRACE(name) do {} while (0)
#endif
