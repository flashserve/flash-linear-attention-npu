/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "register/op_def_registry.h"

namespace ops {

class WMmadDemo : public OpDef {
public:
    explicit WMmadDemo(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> fp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        };

        this->Input("a").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("b").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("pre").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Output("c").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "w_mmad_demo")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(WMmadDemo);

} // namespace ops
