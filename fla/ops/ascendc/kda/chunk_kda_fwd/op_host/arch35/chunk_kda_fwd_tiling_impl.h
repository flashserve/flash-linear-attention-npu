#pragma once

namespace optiling::arch35 {

struct ChunkKdaFwdArch35Options {
    bool computeGateInPrepare = false;
    bool fusePostWu = false;
    bool fusePostWuIntoFwdH = false;
    bool useDenseFwdH = false;
};

inline ChunkKdaFwdArch35Options ConfigureChunkKdaFwdArch35(
    bool isAscend950, bool qIsBf16, bool rawGIsFp32, bool hasALog,
    bool gateParamsAreFp32, bool useGateInKernel, bool safeGate, bool isVarLen,
    bool hasVarlenTail, int64_t seqNum, int64_t seqlen,
    int64_t vHeads, int64_t chunkSize, int64_t kDim, int64_t vDim,
    bool storeQG, bool storeVNew, bool storeH)
{
    ChunkKdaFwdArch35Options options;
    const bool shapeSupported =
        isAscend950 && chunkSize == 64 && kDim == 128 && vDim == 128;
    if (!shapeSupported) {
        return options;
    }

    // Tiling keys describe shape families independently of the SoC. These
    // options only enable arch35 sub-pipelines within the selected family.
    options.computeGateInPrepare =
        qIsBf16 && rawGIsFp32 && hasALog &&
        gateParamsAreFp32 && useGateInKernel && safeGate;
    const bool denseScheduled = !isVarLen;
    const bool sequenceAwareVarlen = isVarLen && seqNum > 0;
    options.useDenseFwdH =
        (denseScheduled || sequenceAwareVarlen) && qIsBf16;
    const bool canFusePreparePostWu =
        (denseScheduled || sequenceAwareVarlen) &&
        qIsBf16 && safeGate && vHeads % 2 == 0;
    options.fusePostWuIntoFwdH = false;
    options.fusePostWu = canFusePreparePostWu;
    return options;
}

} // namespace optiling::arch35
