/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file op_resource.h
 * \brief
 */
#ifndef COMMON_NN_OP_RESOURCE_H
#define COMMON_NN_OP_RESOURCE_H

#define EXTERN_OP_RESOURCE(kernelName)                        \
namespace l0op {                                                \
    extern void * kernelName##TilingRegisterResource();                 \
    extern void * kernelName##InferShapeRegisterResource();             \
    extern void * kernelName##TuningRegisterResource();             \
    extern const OP_BINARY_RES& kernelName##KernelResource();                 \
    extern const OP_RUNTIME_KB_RES& kernelName##TuningResource();             \
    [[maybe_unused]] uint32_t kernelName##_kernelName_Be_Defined_Multi_Times___;  \
}

#define AUTO_GEN_OP_RESOURCE(kernelName)    {{ #kernelName,                     \
    {{l0op::kernelName##TilingRegisterResource(), l0op::kernelName##InferShapeRegisterResource(), l0op::kernelName##TuningRegisterResource()},  \
      l0op::kernelName##KernelResource(), l0op::kernelName##TuningResource()}}} \

#endif  // COMMON_NN_OP_RESOURCE_H