/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {
class ChunkKdaBwdIntra : public OpDef {
public:
    explicit ChunkKdaBwdIntra(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {
            ge::DT_BF16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_BF16, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> fp32 = {
            ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> int64 = {
            ge::DT_INT64, ge::DT_INT64
        };
        const std::initializer_list<ge::Format> nd = {
            ge::FORMAT_ND, ge::FORMAT_ND
        };
        this->Input("q").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("k").ParamType(REQUIRED).DataType(bf16).Format(nd).UnknownShapeFormat(nd);
        this->Input("gk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("dAqk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dAkk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dq").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("db").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dg").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);
        this->Input("chunk_metadata").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64).Format(nd).UnknownShapeFormat(nd);

        this->Output("dq_out").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dk_out").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("db_out").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dg_out").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(true);
        this->Attr("layout_mode").AttrType(REQUIRED).Int(0);
        this->Attr("total_chunks").AttrType(REQUIRED).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_intra")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwdIntra);
} // namespace ops
