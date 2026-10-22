#include "register/op_def_registry.h"

namespace ops {
class RecomputeWUFwdExperimental : public OpDef {
public:
    explicit RecomputeWUFwdExperimental(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "recompute_w_u_fwd_experimental")
            .ExtendCfgInfo("opInterface.value", "recompute_w_u_fwd_experimental")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(RecomputeWUFwdExperimental);
} // namespace ops
