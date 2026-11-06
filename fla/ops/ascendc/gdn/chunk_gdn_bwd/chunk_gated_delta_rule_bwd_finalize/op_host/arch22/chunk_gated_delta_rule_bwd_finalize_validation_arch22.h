#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VALIDATION_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VALIDATION_ARCH22_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

namespace GDN {
namespace FinalizeHostArch22 {

template <typename Shape>
std::string ShapeString(const Shape &shape)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << shape.GetDim(index);
    }
    stream << "]";
    return stream.str();
}

inline std::string DimensionsString(std::initializer_list<int64_t> dimensions)
{
    std::ostringstream stream;
    stream << "[";
    size_t index = 0;
    for (const int64_t dimension : dimensions) {
        stream << (index++ == 0 ? "" : ", ") << dimension;
    }
    stream << "]";
    return stream.str();
}

template <typename Shape>
bool HasShape(const Shape &shape, std::initializer_list<int64_t> expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dimension : expected) {
        if (shape.GetDim(index++) != dimension) {
            return false;
        }
    }
    return true;
}

// WY 使用 uint32 timeBos + BT；在 min(end, ...) 之前该加法也必须可表示。
constexpr bool ChunkEndFitsUint32(uint64_t start, uint64_t chunkSize)
{
    return chunkSize <= std::numeric_limits<uint32_t>::max() &&
           start <= std::numeric_limits<uint32_t>::max() - chunkSize;
}

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

// 只读 host 数组；先验证边界，再使用 pair 中的 seq 索引。
// 不强加 pair 排序；h/dh 的全局 chunk 顺序仍由调用者的 pair 顺序定义。
template <typename IntArray>
bool ValidateVarlenMetadata(const IntArray &cuSeqlens, const IntArray &chunkIndices,
                            int64_t tokens, int64_t chunkSize, MetadataError &error)
{
    const size_t cuSize = cuSeqlens.Size();
    const size_t pairValueSize = chunkIndices.Size();
    constexpr uint64_t MAX_INDEX = std::numeric_limits<uint32_t>::max();
    if (tokens <= 0 || static_cast<uint64_t>(tokens) > MAX_INDEX) {
        return InvalidMetadata(error, "q.T", std::to_string(tokens), "1 <= T <= UINT32_MAX");
    }
    if (chunkSize != 64 && chunkSize != 128) {
        return InvalidMetadata(error, "chunkSize", std::to_string(chunkSize), "64 or 128 on A2/A3");
    }
    if (cuSize < 2 || cuSize - 1 > MAX_INDEX) {
        return InvalidMetadata(error, "cuSeqlensOptional.size", std::to_string(cuSize),
                               "2 <= size <= UINT32_MAX + 1");
    }
    // kernel 以 uint32 taskIdx 计算 2 * taskIdx + 1，也要保证 pair 元素下标不回绕。
    constexpr uint64_t MAX_PAIR_COUNT = (MAX_INDEX + 1) / 2;
    if (pairValueSize == 0 || pairValueSize % 2 != 0 || pairValueSize / 2 > MAX_PAIR_COUNT) {
        return InvalidMetadata(error, "chunkIndicesOptional.size", std::to_string(pairValueSize),
                               "a positive even size with size / 2 <= (UINT32_MAX + 1) / 2 for uint32 pair indexing");
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
        if (chunks != 0) {
            const uint64_t lastStart = static_cast<uint64_t>(begin) +
                static_cast<uint64_t>(chunks - 1) * static_cast<uint64_t>(chunkSize);
            if (!ChunkEndFitsUint32(lastStart, static_cast<uint64_t>(chunkSize))) {
                return InvalidMetadata(error, "cuSeqlensOptional sequence " + std::to_string(seq),
                                       "last chunk start=" + std::to_string(lastStart) +
                                       ", chunkSize=" + std::to_string(chunkSize),
                                       "last chunk start + chunkSize <= UINT32_MAX");
            }
        }
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

} // namespace FinalizeHostArch22
} // namespace GDN

#endif
