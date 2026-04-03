/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file l1_to_fb_iterator.h
 * \brief
 */

#ifndef L1_TO_FB_ITERATOR_H
#define L1_TO_FB_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l1_to_fb
/////////////////////////////////////////////////////

// Partial specialization for V220
template <ArchType ArchTag, typename DataType>
struct l1_to_fb {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::fbBlockSize / sizeof(DataType);

    __aicore__
    l1_to_fb(AscendC::LocalTensor<DataType> fbTensor, AscendC::LocalTensor<DataType> l1Tensor, uint32_t ntileActual)
    {
        copy_cbuf_to_fbuf((__fbuf__ DataType *)fbTensor.GetPhyAddr(),
                          (__cbuf__ DataType *)l1Tensor.GetPhyAddr(),
                          1,
                          CeilDiv<BLOCK_SIZE>(ntileActual),
                          0,
                          0);
    };
};

#endif // L1_TO_FB_ITERATOR_H