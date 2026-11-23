#include "register/op_def_registry.h"

namespace ops {
class MegaRecurrentKda : public OpDef {
public:
    explicit MegaRecurrentKda(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND};
        this->Input("hidden_states").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("qkv_projection_weight").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("decay_projection_a_weight").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("decay_projection_b_weight").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("beta_projection_weight").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("output_gate_projection_weight").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("output_norm_weight").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList(nd);
        this->Input("output_projection_weight").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("conv1d_weight").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Input("conv_state").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd).IgnoreContiguous();
        this->Input("checkpoint_state")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList(nd)
            .IgnoreContiguous();
        this->Input("replay_u").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList(nd).IgnoreContiguous();
        this->Input("replay_k").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT16}).FormatList(nd).IgnoreContiguous();
        this->Input("replay_decay").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList(nd).IgnoreContiguous();
        this->Input("state_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList(nd);
        this->Input("conv_num_accepted_tokens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList(nd);
        this->Input("a_log").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList(nd);
        this->Input("dt_bias").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList(nd);
        this->Input("cu_seqlens").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32, ge::DT_INT64}).FormatList(nd);
        this->Output("out").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList(nd);
        this->Output("conv_state_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16})
            .FormatList(nd)
            .IgnoreContiguous();
        this->Output("replay_u_out").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList(nd).IgnoreContiguous();
        this->Output("replay_k_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList(nd)
            .IgnoreContiguous();
        this->Output("replay_decay_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList(nd)
            .IgnoreContiguous();
        this->Attr("scale").AttrType(REQUIRED).Float();
        this->Attr("lower_bound").AttrType(REQUIRED).Float();
        this->Attr("rms_norm_eps").AttrType(REQUIRED).Float();
        this->Attr("state_geometry").AttrType(REQUIRED).ListInt();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(MegaRecurrentKda);
} // namespace ops
