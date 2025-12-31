/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_full_load_tiling_arch35.cpp
 * \brief
 */
#include <cmath>
#include "moe_init_routing_v2_grad_tiling.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
#define TILINGKEY_FULL_LOAD_DROPLESS 300001
#define TILINGKEY_FULL_LOAD_DROP_PAD 300002
#define TILINGKEY_FULL_LOAD_ACTIVE 300003
const static int64_t MAX_FULL_LOAD_CNT = 10;
const static int64_t DOUBLE_BUFFER = 2;
const static int64_t FP16_BF16_SIZE = 2;
const static int64_t FP32_SIZE = 4;
const static int64_t UINT16_MAX_VALUE = 65535;

class MoeInitRoutingV2GradRegbaseFullLoad : public MoeInitRoutingV2GradTilingBaseClass
{
public:
    explicit MoeInitRoutingV2GradRegbaseFullLoad(gert::TilingContext* context)
        : MoeInitRoutingV2GradTilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRoutingV2GradRegbaseFullLoad() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeInitRoutingV2GradTilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        if (socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }
        return true;
    }

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    void SetTilingShapeInfo();
    void SetTilingSplitCore();
    ge::graphStatus SetTilingFactor();

    int64_t blockDim = 0;
    MoeInitRoutingV2GradRegbaseFullLoadTilingData moeInitRoutingV2GradTilingData;
};

void MoeInitRoutingV2GradRegbaseFullLoad::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeInitRoutingV2GradRegbaseFullLoad::DoOpTiling()
{
    SetTilingShapeInfo();
    SetTilingSplitCore();
    return SetTilingFactor();
}

void MoeInitRoutingV2GradRegbaseFullLoad::SetTilingShapeInfo()
{
    moeInitRoutingV2GradTilingData.set_h(hiddenSize);
    moeInitRoutingV2GradTilingData.set_n(N);
    moeInitRoutingV2GradTilingData.set_k(topK);
    moeInitRoutingV2GradTilingData.set_activeNum(activeNum);
}

void MoeInitRoutingV2GradRegbaseFullLoad::SetTilingSplitCore()
{
    int64_t nBlockFactor = Ops::Base::CeilDiv(N, aivNum); // 单核处理最大token数
    blockDim = Ops::Base::CeilDiv(N, nBlockFactor);       // 实际使用核数

    moeInitRoutingV2GradTilingData.set_blockDim(blockDim);
    moeInitRoutingV2GradTilingData.set_nBlockFactor(nBlockFactor);
}

ge::graphStatus MoeInitRoutingV2GradRegbaseFullLoad::SetTilingFactor()
{
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t regSize = Ops::Base::GetVRegSize(context_) / sizeof(float);

    bool notFloat = (inDtype != ge::DT_FLOAT) ? true : false;
    int64_t typeSize = notFloat ? FP16_BF16_SIZE : FP32_SIZE;
    int64_t ubSizeCanUse = aicoreParams_.ubSize;
    int64_t hiddenSizeAlign = Ops::Base::CeilAlign(hiddenSize * typeSize, blockSize) / typeSize;
    // 每个输入topK行，输出1行，
    // 只对输入开DB
    int64_t rowsSizeForOneToken = (topK * DOUBLE_BUFFER + 1) * typeSize;
    bool canHiddenSizeFullLoad = (rowsSizeForOneToken * hiddenSizeAlign) <= ubSizeCanUse;
    if (canHiddenSizeFullLoad) {
        // h全部放入，ub内一共可以放nUbFactor个token
        int64_t tokensCanInUb = ubSizeCanUse / (rowsSizeForOneToken * hiddenSizeAlign);
        hiddenSizeAlign = ClipMax(hiddenSizeAlign, UINT16_MAX_VALUE * regSize);
        tokensCanInUb = ClipMax(tokensCanInUb, UINT16_MAX_VALUE);
        moeInitRoutingV2GradTilingData.set_nUbFactor(tokensCanInUb);
        moeInitRoutingV2GradTilingData.set_hUbFactor(hiddenSizeAlign);
    } else {
        // k小于阈值，拷贝的H量足够打满带宽，否则切到非全载模版，保证H拷入带宽
        if (topK > MAX_FULL_LOAD_CNT) {
            return ge::GRAPH_PARAM_INVALID;
        }
        // h无法全部放入，一次只能放入hUbFactor大小
        int64_t tokenHiddenSizeCanInUb = ubSizeCanUse / rowsSizeForOneToken;
        tokenHiddenSizeCanInUb = Ops::Base::FloorAlign(tokenHiddenSizeCanInUb * typeSize, blockSize) / typeSize;
        tokenHiddenSizeCanInUb = ClipMax(tokenHiddenSizeCanInUb, UINT16_MAX_VALUE * regSize);
        moeInitRoutingV2GradTilingData.set_nUbFactor(1);
        moeInitRoutingV2GradTilingData.set_hUbFactor(tokenHiddenSizeCanInUb);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingV2GradRegbaseFullLoad::GetTilingKey() const
{
    uint64_t tilingKey = TILINGKEY_FULL_LOAD_DROPLESS;
    if (dropPadMode == 1) {
        tilingKey = TILINGKEY_FULL_LOAD_DROP_PAD;
    } else if (activeNum > 0) {
        tilingKey = TILINGKEY_FULL_LOAD_ACTIVE;
    }

    return tilingKey;
}

ge::graphStatus MoeInitRoutingV2GradRegbaseFullLoad::PostTiling()
{
    context_->SetBlockDim(blockDim);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto tilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    moeInitRoutingV2GradTilingData.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(moeInitRoutingV2GradTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV2Grad, MoeInitRoutingV2GradRegbaseFullLoad, 30000);
} // namespace optiling