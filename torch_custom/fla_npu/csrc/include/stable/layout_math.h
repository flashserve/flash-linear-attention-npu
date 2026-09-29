// Layout arithmetic for the chunked operators.
//
// aclnn takes the layout as a `char*`, which the stable value conversions
// cannot carry, so a wrapper sends an int code and the adapter maps it back
// here.  Every layout argument uses the same order -- BSND, BNSD, TND, NTD --
// and the two questions the adapters actually ask are whether this is the
// packed (rank-3) spelling and whether the token axis comes before the head
// axis.
//
// The sizes themselves stay in the adapter: these helpers only answer "how
// many tokens / heads / channels does this tensor have under this layout",
// which is the part that is easy to get subtly wrong and worth having once.
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

// [B, T, H, D] / [T, H, D] versus [B, H, T, D] / [H, T, D].
inline int64_t tokens(const TensorMeta& q, int64_t code) {
  if (packed(code)) {
    return size_of(q, sequence_major(code) ? 0 : 1);
  }
  return size_of(q, sequence_major(code) ? 1 : 2);
}

inline int64_t key_heads(const TensorMeta& q, int64_t code) {
  if (packed(code)) {
    return size_of(q, sequence_major(code) ? 1 : 0);
  }
  return size_of(q, sequence_major(code) ? 2 : 1);
}

inline int64_t value_heads(const TensorMeta& v, int64_t code) {
  if (packed(code)) {
    return size_of(v, sequence_major(code) ? 1 : 0);
  }
  return size_of(v, sequence_major(code) ? 2 : 1);
}

inline int64_t batch(const TensorMeta& q, int64_t code) {
  return packed(code) ? 1 : size_of(q, 0);
}

inline int64_t key_dim(const TensorMeta& q, int64_t code) {
  return size_of(q, packed(code) ? 2 : 3);
}

inline int64_t value_dim(const TensorMeta& v, int64_t code) {
  return size_of(v, packed(code) ? 2 : 3);
}

// The composite operators accept the TND/NTD names but still read a rank-4
// tensor, so for them only the token axis differs.
inline int64_t tokens4(const TensorMeta& q, int64_t code) {
  return size_of(q, sequence_major(code) ? 1 : 2);
}

inline int64_t value_heads4(const TensorMeta& v, int64_t code) {
  return size_of(v, sequence_major(code) ? 2 : 1);
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
