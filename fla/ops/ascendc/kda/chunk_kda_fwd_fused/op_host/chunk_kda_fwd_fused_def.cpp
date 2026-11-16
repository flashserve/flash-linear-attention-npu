/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaFwdFused : public OpDef {
public:
    explicit ChunkKdaFwdFused(const char *name) : OpDef(name)
    {
        // q/k/v 与 attn_out 恒为 BF16；g/beta 由路由侧统一为 FP32 或 BF16；
        // a_log/dt_bias/initial_state/final_state 恒为 FP32。
        const std::initializer_list<ge::DataType> dataTypes = {ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> gateTypes = {ge::DT_FLOAT, ge::DT_BF16};
        const std::initializer_list<ge::DataType> stateTypes = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> cuTypes = {ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("k").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("v").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("g").ParamType(REQUIRED).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("beta").ParamType(REQUIRED).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("a_log").ParamType(REQUIRED).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("dt_bias").ParamType(REQUIRED).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("initial_state").ParamType(OPTIONAL).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(cuTypes).Format(formats).UnknownShapeFormat(formats);

        this->Output("attn_out").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("final_state").ParamType(OPTIONAL).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("layout").AttrType(OPTIONAL).String("TND");
        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaFwdFused);
} // namespace ops
