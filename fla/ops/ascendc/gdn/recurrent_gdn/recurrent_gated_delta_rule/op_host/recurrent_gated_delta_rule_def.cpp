/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule.h.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include <initializer_list>

namespace ops {
class RecurrentGatedDeltaRule : public OpDef {
public:
    explicit RecurrentGatedDeltaRule(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32 = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> int32 = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        const std::initializer_list<ge::DataType> stateTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT,
            ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::Format> nd = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .IgnoreContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .IgnoreContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .IgnoreContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType(betaTypes)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("state")
            .ParamType(REQUIRED)
            .DataType(stateTypes)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .IgnoreContiguous();
        this->Input("actual_seq_lengths")
            .ParamType(REQUIRED)
            .DataType(int32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("ssm_state_indices")
            .ParamType(REQUIRED)
            .DataType(int32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("g")
            .ParamType(OPTIONAL)
            .DataType(gateTypes)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("gk")
            .ParamType(OPTIONAL)
            .DataType(fp32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("num_accepted_tokens")
            .ParamType(OPTIONAL)
            .DataType(int32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("A_log")
            .ParamType(OPTIONAL).DataType(fp32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("dt_bias")
            .ParamType(OPTIONAL).DataType(fp32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Output("state")
            .ParamType(REQUIRED)
            .DataType(stateTypes)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .IgnoreContiguous();
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
        this->Attr("use_qk_l2norm_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_gate_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_beta_sigmoid_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("allow_neg_eigval").AttrType(OPTIONAL).Bool(false);
        // Internal dispatch bit set by the V2 L0 wrapper.  It is deliberately
        // separate from the legacy optional gk input.
        this->Attr("use_v2").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
        this->AICore().AddConfig("ascend950", aicConfig);
    }
};

OP_ADD(RecurrentGatedDeltaRule);

} // namespace ops
