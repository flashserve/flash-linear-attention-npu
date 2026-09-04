/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "register/op_def_registry.h"

namespace ops {

class ChunkGdnFwdPrepare : public OpDef {
public:
    explicit ChunkGdnFwdPrepare(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        };
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16,
        };
        const std::initializer_list<ge::DataType> fp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
        };
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        };

        this->Input("q").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("k").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("v").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("g").ParamType(REQUIRED).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("beta").ParamType(REQUIRED).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("a_log").ParamType(OPTIONAL).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();

        this->Output("g_out").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("w_out").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("u_out").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("a_out").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("q_hat").ParamType(OPTIONAL).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("k_hat").ParamType(OPTIONAL).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("q_rstd").ParamType(OPTIONAL).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("k_rstd").ParamType(OPTIONAL).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("beta_eff").ParamType(OPTIONAL).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);

        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
        this->Attr("allow_neg_eigval").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_exp2").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_qk_l2norm").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_gate_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_beta_sigmoid").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "chunk_gdn_fwd_prepare")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(ChunkGdnFwdPrepare);

} // namespace ops
