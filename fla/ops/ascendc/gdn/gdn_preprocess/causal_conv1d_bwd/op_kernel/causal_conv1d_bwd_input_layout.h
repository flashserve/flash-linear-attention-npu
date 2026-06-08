/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ASCENDC_CAUSAL_CONV1D_BWD_INPUT_LAYOUT_H_
#define ASCENDC_CAUSAL_CONV1D_BWD_INPUT_LAYOUT_H_

#include "causal_conv1d_bwd.h"

template <typename inputT, typename calT>
class CausalConv1dBwdInputLayoutKernel : public CausalConv1dBwdKernel<inputT, calT> {
public:
    __aicore__ inline CausalConv1dBwdInputLayoutKernel() {}
};

#endif  // ASCENDC_CAUSAL_CONV1D_BWD_INPUT_LAYOUT_H_
