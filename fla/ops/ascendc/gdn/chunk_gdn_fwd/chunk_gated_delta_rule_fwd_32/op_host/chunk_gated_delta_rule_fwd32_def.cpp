#include "register/op_def_registry.h"

namespace ops {

class ChunkGatedDeltaRuleFwd32 : public OpDef {
public:
    explicit ChunkGatedDeltaRuleFwd32(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16Types = {ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32Types = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> stateTypes = {ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> indexTypes = {ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q")
            .ParamType(REQUIRED)
            .DataType(bf16Types)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType(bf16Types)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType(bf16Types)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("g")
            .ParamType(REQUIRED)
            .DataType(fp32Types)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType(bf16Types)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("initial_state")
            .ParamType(OPTIONAL)
            .DataType(stateTypes)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType(indexTypes)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("chunk_indices")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType(indexTypes)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Output("o")
            .ParamType(REQUIRED)
            .DataType(bf16Types)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Output("final_state")
            .ParamType(REQUIRED)
            .DataType(stateTypes)
            .FormatList(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Attr("scale").AttrType(REQUIRED).Float(0.0883883476f);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "chunk_gated_delta_rule_fwd_32")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkGatedDeltaRuleFwd32);

}  // namespace ops
