#include "register/op_def_registry.h"

namespace ops {
class RecurrentGatedDeltaRuleV2 : public OpDef {
public:
    explicit RecurrentGatedDeltaRuleV2(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> qkv = {ge::DT_BF16};
        const std::initializer_list<ge::DataType> floatOrBf16 = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> state = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND};

        this->Input("query").ParamType(REQUIRED).DataTypeList(qkv).FormatList(nd).IgnoreContiguous();
        this->Input("key").ParamType(REQUIRED).DataTypeList(qkv).FormatList(nd).IgnoreContiguous();
        this->Input("value").ParamType(REQUIRED).DataTypeList(qkv).FormatList(nd).IgnoreContiguous();
        this->Input("beta").ParamType(REQUIRED).DataTypeList(floatOrBf16).FormatList(nd).IgnoreContiguous();
        this->Input("state").ParamType(REQUIRED).DataTypeList(state).FormatList(nd).IgnoreContiguous();
        this->Input("actual_seq_lengths").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList(nd);
        this->Input("ssm_state_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList(nd);
        this->Input("g").ParamType(OPTIONAL).DataTypeList(floatOrBf16).FormatList(nd).IgnoreContiguous();
        this->Input("gk").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList(nd).IgnoreContiguous();
        this->Input("num_accepted_tokens").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList(nd);
        this->Input("A_log").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList(nd);
        this->Input("dt_bias").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList(nd);
        this->Output("out").ParamType(REQUIRED).DataTypeList(qkv).FormatList(nd).IgnoreContiguous();
        this->Output("state").ParamType(REQUIRED).DataTypeList(state).FormatList(nd).IgnoreContiguous();

        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("use_qk_l2norm_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_gate_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_beta_sigmoid_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("allow_neg_eigval").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(RecurrentGatedDeltaRuleV2);
} // namespace ops
