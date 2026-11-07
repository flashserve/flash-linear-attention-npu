#ifndef CHUNK_KDA_BWD_PREPARE_LAYOUT_H
#define CHUNK_KDA_BWD_PREPARE_LAYOUT_H
#include <cstdint>
namespace KDA {
enum InputLayout : uint32_t { BNSD = 0, BSND = 1, NTD = 2, TND = 3 };
inline constexpr bool Rank3Layout(uint32_t layout) { return layout == NTD || layout == TND; }
inline constexpr bool SequenceMajorLayout(uint32_t layout) { return layout == BSND || layout == TND; }
}
#endif
