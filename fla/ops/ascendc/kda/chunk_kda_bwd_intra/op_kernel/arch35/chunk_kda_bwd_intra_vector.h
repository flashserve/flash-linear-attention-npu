/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_INTRA_ARCH35_VECTOR_H

// The orchestration and GM/UB pipeline are intentionally shared with A2/A3.
// All A5 Vector arithmetic and cross-core synchronization are selected inside
// the common class at compile time and implemented by the RegBase helpers.
#include "../chunk_kda_bwd_intra_vector.h"

#endif // CHUNK_KDA_BWD_INTRA_ARCH35_VECTOR_H
