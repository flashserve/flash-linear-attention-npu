/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file add_example_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_AddEXAMPLE_H_
#define OPS_OP_PROTO_INC_AddEXAMPLE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns x1 + x2.
*@par Inputs:
*Two inputs, including:
* @li x1: A NCHW or NHWC Tensor. Must be one of the following types: float32.
* @li x2: A NCHW or NHWC Tensor. Must be one of the following types: float32. \n

*@par Outputs:
*y: A NCHW or NHWC Tensor. Must be one of the following types: float32.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Add.
*/
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(AddExample)

} // namespace ge

#endif // OPS_OP_PROTO_INC_AddEXAMPLE_H_

