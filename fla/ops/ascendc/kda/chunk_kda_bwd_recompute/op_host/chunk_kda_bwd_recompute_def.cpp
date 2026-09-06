/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaBwdRecompute : public OpDef {
public:
    explicit ChunkKdaBwdRecompute(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> fp32 = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> int64 = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> nd = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("k").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("v").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("g").ParamType(REQUIRED).DataType(gateTypes).Format(nd).UnknownShapeFormat(nd)
            .AutoContiguous();
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(nd).UnknownShapeFormat(nd)
            .AutoContiguous();
        this->Input("A").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("a_log").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);

        this->Output("w").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("u").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("qg").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("kg").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("gk").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(true);
        this->Attr("use_exp2").AttrType(REQUIRED).Bool(true);
        this->Attr("lower_bound").AttrType(REQUIRED).Float(-5.0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_recompute")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwdRecompute);
} // namespace ops
