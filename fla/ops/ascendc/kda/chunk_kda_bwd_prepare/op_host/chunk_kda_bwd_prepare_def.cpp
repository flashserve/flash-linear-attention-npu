/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {
class ChunkKdaBwdPrepare : public OpDef {
public:
    explicit ChunkKdaBwdPrepare(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> int64 = {ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("aqk").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("v_new").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("d_o").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("h").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);

        this->Output("d_aqk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dv").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Output("dq_raw").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);

        this->Attr("scale").AttrType(REQUIRED).Float();
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_prepare")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwdPrepare);
} // namespace ops
