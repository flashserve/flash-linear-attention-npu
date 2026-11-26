/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_def_registry.h"

namespace ops {
class CompactConvState : public OpDef {
public:
    explicit CompactConvState(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> state = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> int64 = {ge::DT_INT64};
        const std::initializer_list<ge::DataType> metadata = {ge::DT_INT32, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("conv_state").ParamType(REQUIRED).DataTypeList(state).FormatList(formats).IgnoreContiguous();
        this->Input("conv_state_base_addrs").ParamType(REQUIRED).DataTypeList(int64).FormatList(formats);
        this->Input("conv_state_block_strides").ParamType(REQUIRED).DataTypeList(int64).FormatList(formats);
        this->Input("conv_state_dim_strides").ParamType(REQUIRED).DataTypeList(int64).FormatList(formats);
        this->Input("conv_state_token_strides").ParamType(REQUIRED).DataTypeList(int64).FormatList(formats);
        this->Input("state_indices").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats).IgnoreContiguous();
        this->Input("commit_lens").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);
        this->Input("final_state_indices").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);
        this->Input("boundary_state_indices").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);
        this->Input("boundary_recovery_lens").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);

        this->Output("conv_state_out").ParamType(REQUIRED).DataTypeList(state).FormatList(formats).IgnoreContiguous();

        this->Attr("conv_history_len").AttrType(REQUIRED).Int(1);
        this->Attr("align_mode").AttrType(REQUIRED).Bool(false);
        this->Attr("dim_stride").AttrType(OPTIONAL).Int(-1);
        this->Attr("null_block_id").AttrType(OPTIONAL).Int(0);

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

OP_ADD(CompactConvState);

} // namespace ops
