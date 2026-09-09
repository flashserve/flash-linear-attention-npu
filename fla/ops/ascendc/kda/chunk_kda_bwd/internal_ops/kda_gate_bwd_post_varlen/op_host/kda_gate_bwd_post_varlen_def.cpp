#include "register/op_def_registry.h"

namespace ops {

class KdaGateBwdPostVarlen : public OpDef {
public:
    explicit KdaGateBwdPostVarlen(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> fp32 = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> gate = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> i64 = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> nd = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("dg_act").ParamType(REQUIRED)
            .DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("raw_g").ParamType(REQUIRED)
            .DataType(gate).Format(nd).UnknownShapeFormat(nd);
        this->Input("a_log").ParamType(REQUIRED)
            .DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dt_bias").ParamType(OPTIONAL)
            .DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("cu_seqlens").ParamType(REQUIRED).ValueDepend(REQUIRED)
            .DataType(i64).Format(nd).UnknownShapeFormat(nd);
        this->Input("chunk_indices").ParamType(REQUIRED).ValueDepend(REQUIRED)
            .DataType(i64).Format(nd).UnknownShapeFormat(nd);

        this->Output("dg").ParamType(REQUIRED)
            .DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dA").ParamType(REQUIRED)
            .DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dbias").ParamType(REQUIRED)
            .DataType(fp32).Format(nd).UnknownShapeFormat(nd);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(true);
        this->Attr("lower_bound").AttrType(REQUIRED).Float(-5.0f);
        this->Attr("input_scanned").AttrType(REQUIRED).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(KdaGateBwdPostVarlen);
} // namespace ops
