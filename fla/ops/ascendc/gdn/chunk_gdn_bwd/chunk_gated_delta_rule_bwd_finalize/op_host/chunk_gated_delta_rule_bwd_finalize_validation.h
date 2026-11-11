#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VALIDATION_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VALIDATION_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace GDN::FinalizeHost {

constexpr int64_t kA5PublishedMaxG = 16;

struct MetadataError {
    std::string parameter;
    std::string actual;
    std::string expected;
};

inline bool InvalidMetadata(MetadataError &error, const std::string &parameter,
                     const std::string &actual, const std::string &expected)
{
    error = {parameter, actual, expected};
    return false;
}

// A5 调用方已校验 chunkSize。先验证边界，再使用 pair 中的 seq 索引。
// 不强加 pair 排序；h/dh 的全局 chunk 顺序仍由调用者的 pair 顺序定义。
template <typename IntArray>
bool ValidateVarlenMetadata(const IntArray &cuSeqlens, const IntArray &chunkIndices,
                            int64_t tokens, int64_t chunkSize, MetadataError &error)
{
    const size_t cuSize = cuSeqlens.Size();
    const size_t pairValueSize = chunkIndices.Size();
    if (tokens <= 0) {
        return InvalidMetadata(error, "q.T", std::to_string(tokens), "T > 0");
    }
    if (cuSize < 2 || cuSize - 1 > std::numeric_limits<int64_t>::max()) {
        return InvalidMetadata(error, "cuSeqlensOptional.size", std::to_string(cuSize),
                               "at least two entries with int64 sequence indexing");
    }
    if (pairValueSize == 0 || pairValueSize % 2 != 0 ||
        pairValueSize > std::numeric_limits<int64_t>::max()) {
        return InvalidMetadata(error, "chunkIndicesOptional.size", std::to_string(pairValueSize),
                               "a positive even size with int64 pair indexing");
    }
    if (cuSeqlens[0] != 0) {
        return InvalidMetadata(error, "cuSeqlensOptional[0]", std::to_string(cuSeqlens[0]), "0");
    }
    if (cuSeqlens[cuSize - 1] != tokens) {
        return InvalidMetadata(error, "cuSeqlensOptional[last]", std::to_string(cuSeqlens[cuSize - 1]),
                               "T=" + std::to_string(tokens));
    }

    // prefix[seq] 给该序列的 chunk 分配唯一的 bitmap 区间；空序列占 0 位。
    const size_t seqNum = cuSize - 1;
    std::vector<size_t> prefix(cuSize, 0);
    for (size_t seq = 0; seq < seqNum; ++seq) {
        const int64_t begin = cuSeqlens[seq];
        const int64_t end = cuSeqlens[seq + 1];
        if (begin < 0 || end < begin || end > tokens) {
            return InvalidMetadata(error, "cuSeqlensOptional[" + std::to_string(seq) + ":" +
                                   std::to_string(seq + 1) + "]",
                                   "[" + std::to_string(begin) + ", " + std::to_string(end) + "]",
                                   "0 <= begin <= end <= T=" + std::to_string(tokens));
        }
        const int64_t length = end - begin;
        const size_t chunks = static_cast<size_t>(length / chunkSize + (length % chunkSize != 0));
        prefix[seq + 1] = prefix[seq] + chunks;
    }
    const size_t pairCount = pairValueSize / 2;
    if (prefix.back() != pairCount) {
        return InvalidMetadata(error, "chunkIndicesOptional.size / 2", std::to_string(pairCount),
                               "complete chunk count=" + std::to_string(prefix.back()));
    }
    std::vector<bool> visited(pairCount, false);
    for (size_t task = 0; task < pairCount; ++task) {
        const int64_t seq = chunkIndices[2 * task];
        const int64_t localChunk = chunkIndices[2 * task + 1];
        if (seq < 0 || static_cast<uint64_t>(seq) >= seqNum) {
            return InvalidMetadata(error, "chunkIndicesOptional pair " + std::to_string(task) + ".seq",
                                   std::to_string(seq),
                                   "0 <= seq < " + std::to_string(seqNum));
        }
        const size_t seqIndex = static_cast<size_t>(seq);
        const size_t chunks = prefix[seqIndex + 1] - prefix[seqIndex];
        if (localChunk < 0 || static_cast<uint64_t>(localChunk) >= chunks) {
            return InvalidMetadata(error, "chunkIndicesOptional pair " + std::to_string(task) + ".localChunk",
                                   std::to_string(localChunk),
                                   "0 <= localChunk < " + std::to_string(chunks) +
                                   " for seq=" + std::to_string(seq));
        }
        const size_t index = prefix[seqIndex] + static_cast<size_t>(localChunk);
        if (visited[index]) {
            return InvalidMetadata(error, "chunkIndicesOptional pair " + std::to_string(task),
                                   "[" + std::to_string(seq) + ", " + std::to_string(localChunk) + "]",
                                   "each nonempty sequence chunk must appear exactly once");
        }
        visited[index] = true;
    }
    // 有效范围 + 总数相等 + 无重复，等价于覆盖全部非空序列的 chunk。
    return true;
}

} // namespace GDN::FinalizeHost

#endif
