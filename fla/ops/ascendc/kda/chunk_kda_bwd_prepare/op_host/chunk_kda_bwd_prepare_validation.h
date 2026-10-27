#ifndef CHUNK_KDA_BWD_PREPARE_VALIDATION_H
#define CHUNK_KDA_BWD_PREPARE_VALIDATION_H
#include <cstring>
#include <limits>
#include "../op_kernel/chunk_kda_bwd_prepare_layout.h"
namespace KDA {
struct TokenShape { int64_t B, H, T, D; };
inline bool ParseLayout(const char *name, uint32_t &layout)
{
    if (name == nullptr) return false;
    if (std::strcmp(name, "BNSD") == 0) layout = BNSD;
    else if (std::strcmp(name, "BSND") == 0) layout = BSND;
    else if (std::strcmp(name, "NTD") == 0) layout = NTD;
    else if (std::strcmp(name, "TND") == 0) layout = TND;
    else return false;
    return true;
}
template <typename Shape>
inline bool ParseTokenShape(const Shape &s, uint32_t layout, TokenShape &out)
{
    const bool packed = Rank3Layout(layout), seq = SequenceMajorLayout(layout);
    if (s.GetDimNum() != (packed ? 3 : 4)) return false;
    const size_t base = packed ? 0 : 1;
    out = {packed ? 1 : s.GetDim(0), s.GetDim(base + (seq ? 1 : 0)),
           s.GetDim(base + (seq ? 0 : 1)), s.GetDim(base + 2)};
    int64_t elements = 1;
    for (int64_t dim : {out.B, out.H, out.T, out.D}) {
        if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim / 4) return false;
        elements *= dim;
    }
    return out.T <= std::numeric_limits<int64_t>::max() - 63;
}
template <typename Shape>
inline bool HeadOutputShape(const Shape &s, const TokenShape &input, int64_t width, bool packed)
{
    if (s.GetDimNum() != (packed ? 3 : 4)) return false;
    const size_t base = packed ? 0 : 1;
    return (packed || s.GetDim(0) == input.B) && s.GetDim(base) == input.H &&
           s.GetDim(base + 1) == input.T && s.GetDim(base + 2) == width;
}
}
#endif
