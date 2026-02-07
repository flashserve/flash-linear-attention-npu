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
 * \file chunk_bwd_dv_local_common.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_COMMON_H
#define CHUNK_BWD_DV_LOCAL_COMMON_H

namespace GDN {
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SIZE_FLOAT = 4;
constexpr uint64_t SYNC_AIV_AIC_FLAG_1 = 1;
constexpr uint64_t SYNC_AIV_AIC_FLAG_2 = 2;
constexpr uint64_t SYNC_AIC_AIV_FLAG_3 = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_4 = 4;

__aicore__ inline int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    if (unlikely(divisor == 0)) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

__aicore__ inline void MTE2ToVSync()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
}

__aicore__ inline void MTE3ToVSync()
{
    event_t eventIMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIMTE3ToV);
}

__aicore__ inline void VToMTE3Sync()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
}

} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_COMMON_H