#ifndef MERGE_FWD_BWD_KERNEL_VECTOR_H
#define MERGE_FWD_BWD_KERNEL_VECTOR_H
// 910b / 910_93 shares the Vector. C arrives in GM scratch, not UB;
// PrepareGemmTile reloads it when __NPU_ARCH__ is not 3510.
#include "arch35/merge_fwd_bwd_kernel_vector.h"
#endif
