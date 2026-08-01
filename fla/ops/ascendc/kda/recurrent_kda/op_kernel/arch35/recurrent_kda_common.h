/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef RECURRENT_KDA_ARCH35_COMMON_H
#define RECURRENT_KDA_ARCH35_COMMON_H

#include <cstdint>

namespace RecurrentKda {

constexpr uint64_t RKDA_STATE_READY_FLAG = 4;
constexpr uint64_t RKDA_STATE_READY_REVERSE_FLAG = 5;
constexpr uint64_t RKDA_STATE_FREE_FLAG = 3;
constexpr uint64_t RKDA_L0C_FREE_FLAG = 1;
constexpr uint64_t RKDA_L0C_READY_FLAG = 6;
constexpr uint64_t RKDA_FLAG_ID_MAX = 16;
constexpr uint32_t RKDA_CV_BUFFER_NUM = 1;
constexpr uint32_t RKDA_CUBE_M = 16;

} // namespace RecurrentKda

#endif // RECURRENT_KDA_ARCH35_COMMON_H
