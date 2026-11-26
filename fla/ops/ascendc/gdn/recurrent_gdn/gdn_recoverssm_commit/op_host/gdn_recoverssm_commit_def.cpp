/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_def_registry.h"

namespace ops {
class GdnRecoverssmCommit : public OpDef {
public:
    explicit GdnRecoverssmCommit(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> state = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> int64 = {ge::DT_INT64};
        const std::initializer_list<ge::DataType> metadata = {ge::DT_INT32, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("state").ParamType(REQUIRED).DataTypeList(state).FormatList(formats).IgnoreContiguous();
        this->Input("replay").ParamType(REQUIRED).DataTypeList(fp32).FormatList(formats).IgnoreContiguous();
        this->Input("state_commit_cache_address_stride_table")
            .ParamType(REQUIRED)
            .DataTypeList(int64)
            .FormatList(formats);
        this->Input("state_indices").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);
        this->Input("commit_lens").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);
        this->Input("final_state_indices").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);
        this->Input("boundary_state_indices").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);
        this->Input("boundary_recovery_lens").ParamType(REQUIRED).DataTypeList(metadata).FormatList(formats);

        this->Output("state_out")
            .ParamType(REQUIRED)
            .DataTypeList(state)
            .FormatList(formats)
            .IgnoreContiguous();

        this->Attr("spec_query_len").AttrType(REQUIRED).Int(0);
        this->Attr("align_mode").AttrType(REQUIRED).Bool(false);

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

OP_ADD(GdnRecoverssmCommit);

} // namespace ops
