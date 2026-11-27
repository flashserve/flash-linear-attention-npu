#pragma once

#include "kernel_operator.h"

namespace KdaVarlen {

// One cursor per kernel instance. cu_seqlens is immutable for the launch.
// Ascend C address-space/function decorations are supplied by the caller.
class ChunkCursor {
public:
    __aicore__ inline bool Resolve(
        const __gm__ int64_t *cu, uint64_t seqNum, uint64_t totalTokens,
        uint64_t chunkSize, uint64_t flatChunk,
        uint64_t &seq, uint64_t &begin, uint64_t &end)
    {
        if (cu == nullptr || chunkSize == 0) {
            return false;
        }
        // Backward queries within the cached sequence need no reset. This also
        // handles prefetch/revisit and a valid query after an exhausted search.
        if (flatChunk < chunkPrefix_) {
            sequence_ = 0;
            chunkPrefix_ = 0;
            valid_ = false;
        }
        while (sequence_ < seqNum) {
            if (!valid_) {
                const int64_t start = cu[sequence_];
                const int64_t finish = cu[sequence_ + 1];
                if (start < 0 || finish < start ||
                    static_cast<uint64_t>(finish) > totalTokens) {
                    return false;
                }
                sequenceBegin_ = static_cast<uint64_t>(start);
                sequenceEnd_ = static_cast<uint64_t>(finish);
                const uint64_t length = sequenceEnd_ - sequenceBegin_;
                sequenceChunks_ = length / chunkSize + (length % chunkSize != 0);
                valid_ = true;
            }
            const uint64_t localChunk = flatChunk - chunkPrefix_;
            if (localChunk < sequenceChunks_) {
                seq = sequence_;
                begin = sequenceBegin_ + localChunk * chunkSize;
                end = sequenceEnd_ - begin > chunkSize ? begin + chunkSize : sequenceEnd_;
                return begin < end;
            }
            chunkPrefix_ += sequenceChunks_;
            ++sequence_;
            valid_ = false;
        }
        return false;
    }

private:
    uint64_t sequence_ = 0;
    uint64_t chunkPrefix_ = 0;
    uint64_t sequenceBegin_ = 0;
    uint64_t sequenceEnd_ = 0;
    uint64_t sequenceChunks_ = 0;
    bool valid_ = false;
};

__aicore__ inline bool ResolveChunkRange(
    const __gm__ int64_t *cuSeqlens, const __gm__ int64_t *chunkIndices,
    uint64_t seqNum, uint64_t totalTokens, uint64_t chunkSize,
    uint64_t flatChunk, uint64_t &seq, uint64_t &start, uint64_t &end, ChunkCursor &cursor)
{
    if (cuSeqlens == nullptr || chunkSize == 0) {
        return false;
    }

    int64_t seqValue = -1;
    int64_t localChunkValue = -1;
    if (chunkIndices != nullptr) {
        const uint64_t metadataOffset = flatChunk * 2;
        seqValue = chunkIndices[metadataOffset];
        localChunkValue = chunkIndices[metadataOffset + 1];
    } else {
        return cursor.Resolve(cuSeqlens, seqNum, totalTokens, chunkSize,
                              flatChunk, seq, start, end);
    }

    if (seqValue < 0 || localChunkValue < 0 ||
        static_cast<uint64_t>(seqValue) >= seqNum) {
        return false;
    }
    const int64_t seqStartValue = cuSeqlens[seqValue];
    const int64_t seqEndValue = cuSeqlens[seqValue + 1];
    if (seqStartValue < 0 || seqEndValue < seqStartValue) {
        return false;
    }
    const uint64_t seqStart = static_cast<uint64_t>(seqStartValue);
    const uint64_t seqEnd = static_cast<uint64_t>(seqEndValue);
    const uint64_t localChunk = static_cast<uint64_t>(localChunkValue);
    start = seqStart + localChunk * chunkSize;
    if (start >= seqEnd || start >= totalTokens) {
        return false;
    }
    end = start + chunkSize;
    if (end > seqEnd) {
        end = seqEnd;
    }
    if (end > totalTokens) {
        return false;
    }
    seq = static_cast<uint64_t>(seqValue);
    return start < end;
}

} // namespace KdaVarlen
