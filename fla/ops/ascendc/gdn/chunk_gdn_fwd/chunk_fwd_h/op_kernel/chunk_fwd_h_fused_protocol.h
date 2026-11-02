#ifndef CHUNK_FWD_H_FUSED_PROTOCOL_H
#define CHUNK_FWD_H_FUSED_PROTOCOL_H

#include <cstdint>

namespace GDN {

// Workspace record shared by K0, K1 and K2. The layout is fixed at 64 bytes so
// all stages can use the same descriptor without packing or Python metadata.
struct FwdHFusedSegmentDesc {
    int64_t ownerSequence;
    int64_t physicalBatch;
    int64_t tokenBegin;
    int64_t validTokens;
    int64_t outputChunkBegin;
    int64_t ordinal;
    int64_t affineIndex;
    int64_t flags;
};
static_assert(sizeof(FwdHFusedSegmentDesc) == 64, "segment descriptor must be 64B");

constexpr int64_t FWD_H_FUSED_FIRST = 1;
constexpr int64_t FWD_H_FUSED_LAST = 2;
constexpr int64_t FWD_H_FUSED_SPLIT = 4;

__aicore__ inline __gm__ FwdHFusedSegmentDesc *FwdHFusedSegmentDescPtr(
    const FwdHKernelArgs &args, uint32_t segment) {
    return reinterpret_cast<__gm__ FwdHFusedSegmentDesc *>(
        args.workspace + args.tiling.segmentDescWorkspaceOffset) + segment;
}

} // namespace GDN

#endif
