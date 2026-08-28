/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops {

class ChunkGdnCoreFwd : public OpDef {
public:
    explicit ChunkGdnCoreFwd(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> inputTypes = {
            ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16};
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> stateTypes = {
            ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q").ParamType(REQUIRED).DataType(inputTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("k").ParamType(REQUIRED).DataType(inputTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("v").ParamType(REQUIRED).DataType(inputTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("beta").ParamType(REQUIRED).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("a_storage").ParamType(REQUIRED).DataType(inputTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("raw_g").ParamType(REQUIRED).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("gk").ParamType(OPTIONAL).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("initial_state").ParamType(OPTIONAL).DataType(stateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL).DataType(indexTypes)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL).DataType(indexTypes)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();

        this->Output("o").ParamType(REQUIRED).DataType(inputTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("final_state").ParamType(REQUIRED).DataType(stateTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("g_cumsum_bth").ParamType(REQUIRED).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("A").ParamType(REQUIRED).DataType(inputTypes).Format(formats)
            .UnknownShapeFormat(formats);

        this->Attr("output_final_state").AttrType(REQUIRED).Bool(false);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("scale").AttrType(REQUIRED).Float(1.0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkGdnCoreFwd);

} // namespace ops
