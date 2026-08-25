/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_def.cpp
 * \brief Op definition for chunk_gated_delta_rule_bwd_dhu.
 */

#include "register/op_def_registry.h"

namespace ops {

class ChunkGatedDeltaRuleBwdDhu : public OpDef {
public:
    explicit ChunkGatedDeltaRuleBwdDhu(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
        };
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16,
        };
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        };

        this->Input("q")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("d_o")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("dv")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("g")
            .ParamType(OPTIONAL)
            .DataType(gateTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("gk")
            .ParamType(OPTIONAL)
            .DataType(gateTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("h0")
            .ParamType(OPTIONAL)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("dht")
            .ParamType(OPTIONAL)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType(indexTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("chunk_indices")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType(indexTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Output("dh")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("dh0")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("dv2")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);

        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
        this->Attr("use_exp2").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(ChunkGatedDeltaRuleBwdDhu);

} // namespace ops
