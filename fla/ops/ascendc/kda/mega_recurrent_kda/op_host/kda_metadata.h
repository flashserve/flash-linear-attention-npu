#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace kda {
// Input order registered by MegaRecurrentKda and used by GE inference and tiling.
enum class Input : std::size_t {
    HiddenStates,
    QkvWeight,
    DecayAWeight,
    DecayBWeight,
    BetaWeight,
    GateWeight,
    NormWeight,
    OutputWeight,
    ConvWeight,
    ConvState,
    Checkpoint,
    ReplayU,
    ReplayK,
    ReplayDecay,
    StateIndices,
    AcceptedTokens,
    ALog,
    DtBias,
    CuSeqlens,
    SequencePlan,
    WeightScale, VerifyConv, VerifyStarts, VerifyFlag
};

// The ACLNN executor sends state_geometry to GE as a list of int64_t.
namespace geometry {
enum Field : std::size_t {
    PoolSlots,
    Heads,
    ConvLength,
    ReplayCapacity,
    ConvStride,
    CheckpointStride,
    UStride,
    KStride,
    DecayStride,
    WorkspaceBytes,
    CrossesTile,
    VerifyConvStride,
    FieldCount
};
using Values = std::array<int64_t, FieldCount>;
} // namespace geometry
} // namespace kda
