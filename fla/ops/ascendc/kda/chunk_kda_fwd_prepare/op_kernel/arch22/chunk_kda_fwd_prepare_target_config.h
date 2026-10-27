/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */
#ifndef ARCH22_CHUNK_KDA_FWD_PREPARE_TARGET_CONFIG_H
#define ARCH22_CHUNK_KDA_FWD_PREPARE_TARGET_CONFIG_H

#include <cstdint>

namespace KdaPrepare::Arch22TargetConfig {
constexpr uint64_t kSlotBytes = 0x32400;
constexpr uint64_t kSlotCount = 4;

struct Contract {
    uint64_t batch;
    uint64_t seqLen;
    uint64_t qkHeads;
    uint64_t valueHeads;
    uint64_t totalChunks;
    bool arch35;
    bool varLen;
    bool sequenceMajor;
    bool betaBf16;
    bool normL2;
    bool betaSigmoid;
    bool precomputedGate;
    bool exp2;
    bool saveOutputs;
};

constexpr bool Matches(const Contract &contract)
{
    return !contract.arch35 && !contract.varLen && !contract.sequenceMajor &&
        contract.batch == 1 && contract.seqLen == 8192 &&
        contract.qkHeads == 96 && contract.valueHeads == 96 &&
        contract.totalChunks == 128 && contract.betaBf16 && contract.normL2 &&
        contract.betaSigmoid && contract.precomputedGate && contract.exp2 &&
        contract.saveOutputs;
}
} // namespace KdaPrepare::Arch22TargetConfig

#endif // ARCH22_CHUNK_KDA_FWD_PREPARE_TARGET_CONFIG_H
