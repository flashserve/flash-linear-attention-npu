#pragma once
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace kda {
enum class TemplateId { T01, T02 };

struct Workspace {
    TemplateId kernel;
    uint64_t gatedOffset;
    uint64_t carryOffset;
    uint64_t bytes;
};

struct Shape {
    int64_t batch;
    int64_t sequence;  // Zero for TH.
    int64_t tokens;
    int64_t heads;
    int64_t blocks;
    int64_t convLength;
    int64_t replayCapacity;
};

inline uint64_t checkedProduct(uint64_t a, uint64_t b)
{
    if (b && a > UINT64_MAX / b) { throw std::overflow_error("tensor dimension product"); }
    return a * b;
}

template<class PrefixInt>
Workspace makeTiling(const Shape &s, const PrefixInt *prefix, const int32_t *indices, const int32_t *accepted,
                     uint32_t projectionSlots=0)
{
    static_assert(std::is_same_v<PrefixInt, int32_t> || std::is_same_v<PrefixInt, int64_t>);
    if (s.batch < 1 || s.heads < 1 || s.tokens < 1 || s.blocks < s.batch || s.convLength < 3 ||
        s.replayCapacity < 1 || !indices || !accepted) { throw std::invalid_argument("invalid KDA shape or metadata"); }
    if (prefix) {
        if (s.sequence != 0 || prefix[0] != 0 || prefix[s.batch] != s.tokens) {
            throw std::invalid_argument("invalid TH prefix endpoints");
        }
    } else if (s.sequence < 1 || checkedProduct(s.batch, s.sequence) != static_cast<uint64_t>(s.tokens)) {
        throw std::invalid_argument("invalid BSH shape");
    }
    std::unordered_set<int32_t> slots;
    bool crosses = false;
    int64_t start = 0;
    for (int64_t b = 0; b < s.batch; ++b) {
        int64_t end = prefix ? static_cast<int64_t>(prefix[b + 1]) : start + s.sequence;
        if (end <= start || end > s.tokens || end - start > s.convLength - 2 || end - start > s.replayCapacity) {
            throw std::invalid_argument("invalid sequence extent or state capacity");
        }
        if (indices[b] < 0 || indices[b] >= s.blocks || !slots.insert(indices[b]).second ||
            accepted[b] < 0 || accepted[b] > s.convLength - 2) {
            throw std::invalid_argument("invalid state slot or accepted-token count");
        }
        crosses |= start / 64 != (end - 1) / 64;
        start = end;
    }
    // BF16 decay-A and gated each have two slots of 64 rows. All offsets are multiples of 512.
    constexpr uint64_t decayBytes = 2 * 64 * 128 * sizeof(uint16_t);
    uint64_t perHead = decayBytes + (crosses ? 128 * 128 * sizeof(float) : 0);
    uint64_t variableBytes = checkedProduct(s.heads, perHead);
    if (variableBytes > UINT64_MAX - decayBytes) { throw std::overflow_error("workspace size"); }
    uint64_t carryOffset = decayBytes + checkedProduct(s.heads, decayBytes);
    uint64_t bytes=decayBytes+variableBytes;
    if(projectionSlots>3) throw std::invalid_argument("invalid projection buffer count");
    if(projectionSlots==3 && !crosses) {
        const uint64_t rows=s.tokens<256?(uint64_t(s.tokens)+15)/16*16:256;
        const uint64_t betaStride=(uint64_t(s.heads)+15)/16*16;
        uint64_t dynamic=checkedProduct(rows*2,6*uint64_t(s.heads)*128+betaStride);
        const uint64_t decayBudget=(s.tokens>128 && s.tokens<=256)?rows*256:32768;
        if(dynamic>UINT64_MAX-decayBudget) throw std::overflow_error("projection workspace size");
        bytes=decayBudget+dynamic;
        carryOffset=decayBudget+checkedProduct(rows*2,checkedProduct(s.heads,128));
    } else if(projectionSlots && !crosses) {
        // 64 rows of Q/K/V, decay, gate and one BF16 beta per head.
        uint64_t payload=checkedProduct(s.heads,64*2*(5*128+1));
        if(payload>UINT64_MAX-511) throw std::overflow_error("projection scratch alignment");
        uint64_t aligned=checkedProduct((payload+511)&~uint64_t(511),projectionSlots);
        if(bytes>UINT64_MAX-aligned) throw std::overflow_error("projection workspace size");
        bytes+=aligned;
    }
    return {crosses ? TemplateId::T02 : TemplateId::T01,
            projectionSlots==3 && !crosses && s.tokens>128 && s.tokens<=256?((uint64_t(s.tokens)+15)/16*16)*256:decayBytes,
            carryOffset, bytes};
}

// The size-query caller can print without allocating device memory.
inline void printTiling(const Workspace &w)
{
    std::fprintf(stderr, "template=%s user_workspace_bytes=%llu\n", w.kernel == TemplateId::T01 ? "T01" : "T02",
                 static_cast<unsigned long long>(w.bytes));
}
} // namespace kda
