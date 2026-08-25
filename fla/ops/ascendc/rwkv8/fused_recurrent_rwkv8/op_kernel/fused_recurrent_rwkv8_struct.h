/*!
 * \file fused_recurrent_rwkv8_struct.h
 * \brief Plain tiling struct shared by aclnn tiling and kernel launch.
 */

#ifndef FUSED_RECURRENT_RWKV8_STRUCT_H
#define FUSED_RECURRENT_RWKV8_STRUCT_H

#include <cstdint>

namespace FusedRecurrentRwkv8 {

#pragma pack(push, 8)
struct alignas(8) FusedRecurrentRwkv8TilingData {
    uint32_t B;
    uint32_t T;
    uint32_t H;
    uint32_t K;                  // k/q/w/z/b 侧 head dim（q 的末维）
    uint32_t V;                  // v/o/sa 侧 head dim（v 的末维）；K==V 为特例
    float scale;
    uint32_t hasInitialState;
    uint32_t reverse;            // 1 = T 维倒序递推（对齐 fla reverse）
    uint32_t outputChunkState;   // 1 = 写出 chunk 快照 s（官方转置布局 (K,V)）
    uint32_t outputSa;           // 1 = 写出每 token 的 sa
    uint32_t chunkLen;           // s 快照间隔（attr chunk_len，默认 16 对齐官方 backward；kernel 快照条件以此为唯一依据）
};
#pragma pack(pop)

} // namespace FusedRecurrentRwkv8

#endif // FUSED_RECURRENT_RWKV8_STRUCT_H
