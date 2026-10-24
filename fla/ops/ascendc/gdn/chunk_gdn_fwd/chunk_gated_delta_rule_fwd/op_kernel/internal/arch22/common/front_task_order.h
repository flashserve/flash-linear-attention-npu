#ifndef GDN_ARCH22_FRONT_TASK_ORDER_H
#define GDN_ARCH22_FRONT_TASK_ORDER_H
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. 私有 Front parent 遍历；GM tensor/score 布局保持原编号。
#include <cstdint>

namespace GdnFrontTaskOrder {
struct Parent {
    int64_t batch;
    int64_t head;
    int64_t chunk;  // dense: batch 内；varlen: canonical chunk_indices 全局下标。
};

// 只重排逻辑 parent；原 ceil(parent/core) 分核和批次边界保持不变。
// dense 为 [batch, key_head, chunk, sibling]；varlen 的 chunk 保留输入索引顺序。
__aicore__ inline Parent Decode(int64_t parent, int64_t heads, int64_t headsPerKey,
                                int64_t chunks)
{
    const int64_t sibling = parent % headsPerKey;
    const int64_t group = parent / headsPerKey;
    const int64_t keyHeads = heads / headsPerKey;
    const int64_t keyHead = (group / chunks) % keyHeads;
    return {group / (keyHeads * chunks), keyHead * headsPerKey + sibling, group % chunks};
}

__aicore__ inline int64_t LegacyParent(Parent p, int64_t heads, int64_t chunks)
{
    return (p.batch * heads + p.head) * chunks + p.chunk;
}
} // namespace GdnFrontTaskOrder
#endif
