/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule_tiling.h
 * \brief
 */
#ifndef __OP_HOST_RECURRENT_GETED_DELTA_RULE_TILING_H__
#define __OP_HOST_RECURRENT_GETED_DELTA_RULE_TILING_H__
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/recurrent_gated_delta_rule_tiling_data.h"
#include "recurrent_gated_delta_rule_tiling_processor.h"

namespace optiling {
using namespace RecurrentGatedDeltaRule;

struct RecurrentGatedDeltaRuleCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct RecurrentGatedDeltaRuleInfo {
public:
    int64_t usedCoreNum = 0;
    const char *opName = "RecurrentGatedDeltaRule";
};

class RecurrentGatedDeltaRuleTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RecurrentGatedDeltaRuleTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~RecurrentGatedDeltaRuleTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();
    RecurrentGatedDeltaRuleTilingContext BuildProcessorContext() const;

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus CalUbSize();
    ge::graphStatus GetScale();
    ge::graphStatus GetOptionalInput();
    ge::graphStatus AnalyzeFormat();
    bool CheckFormat(ge::Format format, const std::string &Desc);

    RecurrentGatedDeltaRuleCompileInfo compileInfo_;
    RecurrentGatedDeltaRuleTilingData tilingData_;
    RecurrentGatedDeltaRuleInfo inputParams_;
};

} // namespace optiling
#endif // __OP_HOST_RECURRENT_GETED_DELTA_RULE_TILING_H__
