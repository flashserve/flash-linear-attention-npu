// SPDX-License-Identifier: BSD-3-Clause
#include "register/op_def_registry.h"

namespace ops {
class VdnBridge : public OpDef {
public:
    explicit VdnBridge(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("prefix").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("suffix").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("bounds").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64}).Format(nd).UnknownShapeFormat(nd);
        this->Input("logs").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16}).Format(nd).UnknownShapeFormat(nd);
        this->Attr("start").AttrType(REQUIRED).Int();
        this->Attr("count").AttrType(REQUIRED).Int();
        this->Attr("out_bf16").AttrType(REQUIRED).Bool();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
    }
};
OP_ADD(VdnBridge);
} // namespace ops
