/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "register/op_def_registry.h"

namespace ops {
class MergeFwdBwd : public OpDef {
public:
    explicit MergeFwdBwd(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_BF16};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("ag_hm").ParamType(REQUIRED).DataType(dtypes).Format(nd).UnknownShapeFormat(nd)
            .AutoContiguous();
        this->Output("h").ParamType(REQUIRED).DataType(dtypes).Format(nd).UnknownShapeFormat(nd);

        this->Attr("forward").AttrType(REQUIRED).Bool(true);
        this->Attr("rank").AttrType(REQUIRED).Int(0);
        this->Attr("preOrPostNumRanks").AttrType(REQUIRED).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "merge_fwd_bwd")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(MergeFwdBwd);
} // namespace ops
