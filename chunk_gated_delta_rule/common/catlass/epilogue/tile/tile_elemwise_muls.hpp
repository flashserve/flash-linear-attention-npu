/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

 #ifndef CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
 #define CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP

 #include "catlass/gemm/helper.hpp"

 namespace Catlass::Epilogue::Tile{
 template<
     class ArchTag_,
     class ComputeType_,
     uint32_t COMPUTE_LENGTH_
 >
 struct TileElemWiseMuls{
     using ArchTag = ArchTag_;
     using ElementCompute = typename ComputeType_::Element;

     static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

     CATLASS_DEVICE
     TileElemWiseMuls(){}

     CATLASS_DEVICE
     void operator()(
         AscendC::LocalTensor<ElementCompute> dstLocal,
         AscendC::LocalTensor<ElementCompute> srcTensor,
         ElementCompute scalar
     ){
         AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
     }
 };
 }

 #endif // CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP