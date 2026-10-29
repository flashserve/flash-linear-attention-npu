/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */
#ifndef ARCH22_CHUNK_KDA_FWD_PREPARE_RELAY_CONFIG_H
#define ARCH22_CHUNK_KDA_FWD_PREPARE_RELAY_CONFIG_H

#include <cstdint>

namespace KdaPrepare::Arch22RelayConfig {

// Keep the established allocation for both relay variants. Host scheduling
// and device capacity assertions consume this same definition.
constexpr uint64_t kSlotBytes = 0x33400;
constexpr uint32_t kSlotsPerWorkgroup = 4;

// Shared host/device relay capability. Scale is a finite, host-validated
// runtime operand; it does not change the inverse or its FP32 T relay.
// No model id, exact sequence length or boundary distribution is selected.
struct Contract {
    uint32_t qkHeads;
    uint32_t valueHeads;
    bool sequenceMajor;
    float epsilon;
    float lowerBound;
    float scale;
};

#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr bool Matches(const Contract &c)
{
    return !c.sequenceMajor && c.qkHeads != 0 && c.valueHeads >= c.qkHeads &&
           c.valueHeads % c.qkHeads == 0 &&
           c.epsilon == 1.0e-6F && c.lowerBound == -5.0F;
}

} // namespace KdaPrepare::Arch22RelayConfig

#endif // ARCH22_CHUNK_KDA_FWD_PREPARE_RELAY_CONFIG_H
