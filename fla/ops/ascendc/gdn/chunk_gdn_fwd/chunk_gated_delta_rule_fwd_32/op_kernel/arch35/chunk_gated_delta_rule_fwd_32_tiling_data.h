#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_TILING_DATA_H

#include <cstdint>

struct ChunkGatedDeltaRuleFwd32TilingData {
    uint32_t B;
    uint32_t HV;
    uint32_t T;
    uint32_t hasInitial;
    float scale;
    uint32_t workspaceOffset;
    uint32_t coreNum;
};

constexpr uint32_t kGdn32WsK = 0;
constexpr uint32_t kGdn32WsQ = 8192;
constexpr uint32_t kGdn32WsVb = 16384;
constexpr uint32_t kGdn32WsKbg = 24576;
constexpr uint32_t kGdn32WsA = 32768;
constexpr uint32_t kGdn32WsH = 34816;
constexpr uint32_t kGdn32WsW = 67584;
constexpr uint32_t kGdn32WsKt = 75776;
constexpr uint32_t kGdn32WsVg = 83968;
constexpr uint32_t kGdn32WsQe = 92160;
constexpr uint32_t kGdn32WsAttn = 100352;
constexpr uint32_t kGdn32WsVn = 102400;
constexpr uint32_t kGdn32WsKkt = 110592;
constexpr uint32_t kGdn32WsQk = 114688;
constexpr uint32_t kGdn32WsU = 118784;
constexpr uint32_t kGdn32WsWfp = 135168;
constexpr uint32_t kGdn32WsP = 151552;
constexpr uint32_t kGdn32WsD = 167936;
constexpr uint32_t kGdn32WsQh = 233472;
constexpr uint32_t kGdn32WsAv = 249856;
constexpr uint32_t kGdn32WsKtK = 266240;
constexpr uint32_t kGdn32WsStride = 274432;

#endif
