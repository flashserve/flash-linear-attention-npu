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
class ChunkKdaFwdFinalize : public OpDef {
public:
    explicit ChunkKdaFwdFinalize(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16,
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> stateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("qg_scaled").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("Aqk").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("v_new").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("h").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format(formats).UnknownShapeFormat(formats);

        this->Output("attn_out").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("logical_batch").AttrType(REQUIRED).Int(1);
        this->Attr("logical_seqlen").AttrType(REQUIRED).Int(1);
        this->Attr("logical_q_heads").AttrType(REQUIRED).Int(1);
        this->Attr("logical_v_heads").AttrType(REQUIRED).Int(1);
        this->Attr("logical_k_dim").AttrType(REQUIRED).Int(1);
        this->Attr("logical_v_dim").AttrType(REQUIRED).Int(1);
        this->Attr("logical_total_chunks").AttrType(REQUIRED).Int(1);

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

OP_ADD(ChunkKdaFwdFinalize);
} // namespace ops
