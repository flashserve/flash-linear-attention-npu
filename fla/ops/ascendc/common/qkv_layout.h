/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef FLA_GDN_QKV_LAYOUT_H
#define FLA_GDN_QKV_LAYOUT_H

#include <cstdint>

namespace GDN {
enum QkvLayout : int64_t {
    QKV_LAYOUT_BNSD = 0,
    QKV_LAYOUT_BSND = 1,
    QKV_LAYOUT_NTD = 2,
    QKV_LAYOUT_TND = 3,
};
} // namespace GDN

#endif // FLA_GDN_QKV_LAYOUT_H
