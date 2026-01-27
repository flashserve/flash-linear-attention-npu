/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_vec.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_VEC_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_VEC_H
#endif

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_bwd_dhu_base.h"

using namespace AscendC;
namespace ChunkGDRBwdDhu {

template <typename DT>
class GDRVec : public GDRBase<DT>
{
public:
    __aicore__ inline GDRVec(){};
    __aicore__ inline void Process();
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                GM_ADDR chunk_indices, const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData);

}; // class GDRVec

template <typename DT>
__aicore__ inline void GDRVec<DT>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                GM_ADDR chunk_indices, const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData)
{
    GDRBase<DT>::InitTilingData(tilingData);
}

template <typename DT>
__aicore__ inline void GDRVec<DT>::Process( )
{
   
}

} // namespace ChunkGDRBwdDhu