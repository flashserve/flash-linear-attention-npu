// Layout arithmetic for the chunked operators.
//
// aclnn takes the layout as a `char*`, which the stable value conversions
// cannot carry, so a wrapper sends an int code and the adapter maps it back
// here.  Every layout argument uses the same order -- BSND, BNSD, TND, NTD --
// and the two questions the adapters actually ask are whether this is the
// packed (rank-3) spelling and whether the token axis comes before the head
// axis.
//
// The sizes themselves stay in the adapter's own `SIZE_OF`: these helpers only
// answer "which axis holds the tokens / heads / channels under this layout",
// which is the part that is easy to get subtly wrong and worth having once.
// Reading the dimension here instead would cost the message its tensor name --
// inside a function the only name available is the helper's own parameter (see
// acl_meta.h on `SIZE_OF`) -- and the head axis is the same number for q and v,
// so one helper serves both callers and no literal name is right for all of
// them.
#pragma once

#include "stable/acl_meta.h"

#include <cstdint>
#include <vector>

namespace fla_npu_stable {
namespace stable {
namespace layout_math {

// Must match the name tables the adapters pass to aclnn and `_stable.py`'s
// `_LAYOUT_CODES`; tools/op_abi_parity.py checks those two against each other.
enum Code : int64_t { kBsnd = 0, kBnsd = 1, kTnd = 2, kNtd = 3 };

inline bool packed(int64_t code) {
  return code == kTnd || code == kNtd;
}

inline bool sequence_major(int64_t code) {
  return code == kBsnd || code == kTnd;
}

// Call sites read the dimension themselves: `SIZE_OF(q_meta, token_axis(code))`
// reports the line they wrote and their own tensor name.
//
// [B, T, H, D] / [T, H, D] versus [B, H, T, D] / [H, T, D].
inline int64_t token_axis(int64_t code) {
  if (packed(code)) {
    return sequence_major(code) ? 0 : 1;
  }
  return sequence_major(code) ? 1 : 2;
}

// The head axis is the same number for q and v.
inline int64_t head_axis(int64_t code) {
  if (packed(code)) {
    return sequence_major(code) ? 1 : 0;
  }
  return sequence_major(code) ? 2 : 1;
}

// D is the last axis in both spellings.
inline int64_t dim_axis(int64_t code) {
  return packed(code) ? 2 : 3;
}

// The composite operators accept the TND/NTD names but still read a rank-4
// tensor, so for them only the token axis differs.  (Feeding the rank-3 axes
// here produced o/A/g_cumsum/final_state with the wrong shapes and the
// Ascend950 tiling rejected the call with 161002.)
inline int64_t token_axis4(int64_t code) {
  return sequence_major(code) ? 1 : 2;
}

inline int64_t head_axis4(int64_t code) {
  return sequence_major(code) ? 2 : 1;
}

// One state per segment: `cu_seqlens` describes them, the batch dimension
// otherwise.
inline int64_t sequences(const std::vector<int64_t>& cu_seqlens,
                         int64_t batch_size) {
  return cu_seqlens.empty() ? batch_size
                            : static_cast<int64_t>(cu_seqlens.size()) - 1;
}

// chunk_indices wins when present: it is the canonical sequence-major list the
// kernel expects, and counting it is exact even when a segment is empty.
inline int64_t chunks(const std::vector<int64_t>& cu_seqlens,
                      const std::vector<int64_t>& chunk_indices,
                      int64_t chunk_size, int64_t tokens) {
  if (!chunk_indices.empty()) {
    return static_cast<int64_t>(chunk_indices.size()) / 2;
  }
  if (!cu_seqlens.empty()) {
    int64_t total = 0;
    for (size_t i = 0; i + 1 < cu_seqlens.size(); ++i) {
      total += (cu_seqlens[i + 1] - cu_seqlens[i] + chunk_size - 1) /
               chunk_size;
    }
    return total;
  }
  return (tokens + chunk_size - 1) / chunk_size;
}

}  // namespace layout_math
}  // namespace stable
}  // namespace fla_npu_stable
