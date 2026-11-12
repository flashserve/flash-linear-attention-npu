/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */
#ifndef KDA_CHUNK_INDICES_H
#define KDA_CHUNK_INDICES_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

namespace Kda {

// Call after the public entry has validated cu_seqlens and chunk_size. The
// executor copies the values and owns the array through asynchronous execution.
inline aclnnStatus EnsureChunkIndices(const aclIntArray *cuSeqlens, int64_t chunkSize,
                                      const aclIntArray *&chunkIndices, aclOpExecutor *executor)
{
    if (cuSeqlens == nullptr || chunkIndices != nullptr) {
        return ACLNN_SUCCESS;
    }

    std::vector<int64_t> indices;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        const int64_t chunks = length / chunkSize + (length % chunkSize != 0);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            indices.push_back(static_cast<int64_t>(seq));
            indices.push_back(chunk);
        }
    }
    chunkIndices = executor->AllocIntArray(indices.data(), indices.size());
    if (chunkIndices == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to allocate canonical KDA chunk indices.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

} // namespace Kda

#endif // KDA_CHUNK_INDICES_H
