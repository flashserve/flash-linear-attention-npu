/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_def_registry.h"

namespace ops {
class GdnRecoverssmVerify : public OpDef {
public:
    explicit GdnRecoverssmVerify(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> state = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> metadata = {ge::DT_INT32, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("A_log").ParamType(REQUIRED).DataTypeList(fp32).FormatList(formats);
        this->Input("a").ParamType(REQUIRED).DataTypeList(bf16).FormatList(formats);
        this->Input("b").ParamType(REQUIRED).DataTypeList(bf16).FormatList(formats);
        this->Input("dt_bias").ParamType(REQUIRED).DataTypeList(fp32).FormatList(formats);
        this->Input("q").ParamType(REQUIRED).DataTypeList(bf16).FormatList(formats).IgnoreContiguous();
        this->Input("k").ParamType(REQUIRED).DataTypeList(bf16).FormatList(formats).IgnoreContiguous();
        this->Input("v").ParamType(REQUIRED).DataTypeList(bf16).FormatList(formats).IgnoreContiguous();
        this->Input("checkpoint_state").ParamType(REQUIRED).DataTypeList(state).FormatList(formats).IgnoreContiguous();
        this->Input("replay_cache").ParamType(REQUIRED).DataTypeList(fp32).FormatList(formats).IgnoreContiguous();
        this->Input("query_start_loc").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats).IgnoreContiguous();
        this->Input("state_indices").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats).IgnoreContiguous();

        this->Output("out").ParamType(REQUIRED).DataTypeList(bf16).FormatList(formats).IgnoreContiguous();
        this->Output("replay_cache").ParamType(REQUIRED).DataTypeList(fp32).FormatList(formats).IgnoreContiguous();

        this->Attr("scale").AttrType(OPTIONAL).Float(0.0);
        this->Attr("softplus_beta").AttrType(OPTIONAL).Float(1.0);
        this->Attr("softplus_threshold").AttrType(OPTIONAL).Float(20.0);
        this->Attr("use_qk_l2norm_in_kernel").AttrType(OPTIONAL).Bool(true);

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

OP_ADD(GdnRecoverssmVerify);

} // namespace ops
