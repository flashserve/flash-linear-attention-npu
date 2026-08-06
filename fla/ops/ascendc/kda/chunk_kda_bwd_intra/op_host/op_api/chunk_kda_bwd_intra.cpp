#include "chunk_kda_bwd_intra.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#include <algorithm>
#include <vector>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdIntra);

namespace {
const aclIntArray *BuildPackedChunkMetadata(
    const aclIntArray *cuSeqlens, const aclIntArray *chunkIndices,
    int64_t chunkSize, int64_t totalChunks, aclOpExecutor *executor)
{
    if (cuSeqlens == nullptr || cuSeqlens->Size() < 2 ||
        chunkSize <= 0 || totalChunks <= 0) {
        return nullptr;
    }
    const aclIntArray &cu = *cuSeqlens;
    std::vector<int64_t> packed;
    packed.reserve(static_cast<size_t>(totalChunks) * 4);
    auto appendChunk = [&](int64_t seq, int64_t localChunk) -> bool {
        if (seq < 0 || static_cast<size_t>(seq + 1) >= cu.Size() ||
            localChunk < 0) {
            return false;
        }
        const int64_t seqStart = cu[static_cast<size_t>(seq)];
        const int64_t seqEnd = cu[static_cast<size_t>(seq + 1)];
        const int64_t begin = seqStart + localChunk * chunkSize;
        if (begin < seqStart || begin >= seqEnd) {
            return false;
        }
        const int64_t end = std::min(begin + chunkSize, seqEnd);
        packed.insert(packed.end(), {seq, begin, end, localChunk});
        return true;
    };

    if (chunkIndices != nullptr) {
        if (chunkIndices->Size() != static_cast<size_t>(totalChunks) * 2) {
            return nullptr;
        }
        for (size_t idx = 0; idx < chunkIndices->Size(); idx += 2) {
            if (!appendChunk((*chunkIndices)[idx], (*chunkIndices)[idx + 1])) {
                return nullptr;
            }
        }
    } else {
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            const int64_t seqLength = cu[seq + 1] - cu[seq];
            const int64_t chunkCount =
                (seqLength + chunkSize - 1) / chunkSize;
            for (int64_t localChunk = 0; localChunk < chunkCount; ++localChunk) {
                if (!appendChunk(static_cast<int64_t>(seq), localChunk)) {
                    return nullptr;
                }
            }
        }
    }
    if (packed.size() != static_cast<size_t>(totalChunks) * 4) {
        return nullptr;
    }
    return executor->AllocIntArray(packed.data(), packed.size());
}

void SetNdFormat(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
}
} // namespace

const std::array<const aclTensor *, 4> ChunkKdaBwdIntra(
    const aclTensor *q, const aclTensor *k, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *dAqk, const aclTensor *dAkk, const aclTensor *dq, const aclTensor *dk,
    const aclTensor *db, const aclTensor *dg,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, bool safeGate, int64_t layoutMode, int64_t totalChunks,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dbOut,
    const aclTensor *dgOut, aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdIntra, q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
           cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate,
           layoutMode, totalChunks, dqOut, dkOut, dbOut, dgOut);

    const aclTensor *actualCuSeqlens = nullptr;
    const aclTensor *actualChunkMetadata = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens =
            executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const aclIntArray *packed = BuildPackedChunkMetadata(
            cuSeqlensOptional, chunkIndicesOptional, chunkSize,
            totalChunks, executor);
        if (actualCuSeqlens == nullptr || packed == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "failed to build varlen metadata for ChunkKdaBwdIntra.");
            return {nullptr, nullptr, nullptr, nullptr};
        }
        actualChunkMetadata =
            executor->ConvertToTensor(packed, DataType::DT_INT64);
        if (actualChunkMetadata == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                    "failed to convert packed chunk metadata.");
            return {nullptr, nullptr, nullptr, nullptr};
        }
        SetNdFormat(actualCuSeqlens);
        SetNdFormat(actualChunkMetadata);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdIntra,
        OP_INPUT(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
                 actualCuSeqlens, actualChunkMetadata),
        OP_OUTPUT(dqOut, dkOut, dbOut, dgOut),
        OP_ATTR(chunkSize, safeGate, layoutMode, totalChunks));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdIntra failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dbOut, dgOut};
}

} // namespace l0op
