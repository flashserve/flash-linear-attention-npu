/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_def.cpp
 * \brief Op def for pre_process_fwd_kernel_merged (CP pre-process: window -> (h | m) affine chain).
 *
 * 输入输出语义见算子目录 docs/api.md。
 * dtype 组合只有一条：k/w/u BF16、g/gk **FP32**（BF16 gate 由 aclnn 层先 Cast 成 FP32）、
 * cu_seqlens INT64、hm FP32；gate 路径由 TilingKey（1=USE_G / 2=USE_GK）区分。
 *
 * v / bg 是 DPLR 专用的可选输入位：本版本不支持 DPLR（bg，DPLR 的 K 侧项），
 * 两者必须为空——host tiling 与 aclnn 都会对非空值直接报错，不会下发 TilingKey 3。
 */

#include "register/op_def_registry.h"

namespace ops {
class PreProcessFwdKernelMerged : public OpDef {
public:
    explicit PreProcessFwdKernelMerged(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> bf16 = {ge::DT_BF16};
        const std::vector<ge::DataType> fp32 = {ge::DT_FLOAT};
        const std::vector<ge::DataType> i64 = {ge::DT_INT64};
        const std::vector<ge::Format> nd = {ge::FORMAT_ND};

        this->Input("k")
            .ParamType(REQUIRED)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("u")
            .ParamType(REQUIRED)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd);
        // g：GDN 的标量 gate（[1, HV, T]）；与 gk 二选一
        this->Input("g")
            .ParamType(OPTIONAL)
            .DataType(fp32)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .AutoContiguous();
        // gk：KDA/DPLR 的逐 K gate（[1, HV, T, K]）；与 g 二选一
        this->Input("gk")
            .ParamType(OPTIONAL)
            .DataType(fp32)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .AutoContiguous();
        // bg：DPLR 专用的 K 侧项（[1, HK, T, K]）——本版本不支持 DPLR，必须为空
        this->Input("bg")
            .ParamType(OPTIONAL)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .AutoContiguous();
        // v：DPLR 专用（独立于 u）——本版本不支持 DPLR，必须为空；GDN/KDA 的取值来自 u
        this->Input("v")
            .ParamType(OPTIONAL)
            .DataType(bf16)
            .Format(nd)
            .UnknownShapeFormat(nd)
            .AutoContiguous();
        // cu_seqlens：host int 数组，经 aclnn 转成 INT64 tensor；必给
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType(i64)
            .Format(nd)
            .UnknownShapeFormat(nd);

        this->Output("hm")
            .ParamType(REQUIRED)
            .DataType(fp32)
            .Format(nd)
            .UnknownShapeFormat(nd);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);

        OpAICoreConfig opFileConfig;
        opFileConfig.ExtendCfgInfo("opFile.value", "pre_process_fwd_kernel_merged")
            .ExtendCfgInfo("opInterface.value", "pre_process_fwd_kernel_merged");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "pre_process_fwd_kernel_merged")
            .ExtendCfgInfo("opInterface.value", "pre_process_fwd_kernel_merged")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", opFileConfig);
        this->AICore().AddConfig("ascend910_93", opFileConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(PreProcessFwdKernelMerged);
} // namespace ops
