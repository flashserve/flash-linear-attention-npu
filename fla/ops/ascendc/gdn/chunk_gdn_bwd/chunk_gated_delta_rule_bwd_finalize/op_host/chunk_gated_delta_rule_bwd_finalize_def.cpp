#include "register/op_def_registry.h"

namespace ops {

class ChunkGatedDeltaRuleBwdFinalize : public OpDef {
public:
    explicit ChunkGatedDeltaRuleBwdFinalize(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> qTypes = {
            ge::DT_BF16, ge::DT_BF16,
        };
        const std::initializer_list<ge::DataType> scalarTypes = {
            ge::DT_BF16, ge::DT_FLOAT,
        };
        const std::initializer_list<ge::DataType> floatTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT,
        };
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT64, ge::DT_INT64,
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND,
        };

        this->Input("q")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("v")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("v_new")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("do")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("du")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("g")
            .ParamType(REQUIRED).DataType(scalarTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED).DataType(scalarTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("h")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("dh")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("A")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("q_rstd")
            .ParamType(OPTIONAL).DataType(floatTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("k_rstd")
            .ParamType(OPTIONAL).DataType(floatTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("beta_raw")
            .ParamType(OPTIONAL).DataType(scalarTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL).ValueDepend(OPTIONAL).DataType(indexTypes)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("chunk_indices")
            .ParamType(OPTIONAL).ValueDepend(OPTIONAL).DataType(indexTypes)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();

        this->Output("dq")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("dk")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("dv")
            .ParamType(REQUIRED).DataType(qTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("dbeta")
            .ParamType(REQUIRED).DataType(scalarTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("dg")
            .ParamType(REQUIRED).DataType(scalarTypes).Format(formats)
            .UnknownShapeFormat(formats);

        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
        this->Attr("use_qk_l2norm_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_beta_sigmoid_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("allow_neg_eigval").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_gate_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_exp2").AttrType(OPTIONAL).Bool(true);

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

OP_ADD(ChunkGatedDeltaRuleBwdFinalize);

} // namespace ops
